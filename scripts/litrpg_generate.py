"""Generate model-written LitRPG chapters from a manifest, and validate them against it.

    uv run python scripts/litrpg_generate.py --manuscripts 6 --chapters 12

`findings-litrpg.md` §4 says the detector's 99%/88% and A2d's 84% are upper bounds because
the prose is templated and the status block has a fixed schema. This replaces the templates
with model-written chapters conditioned on the same manifest, so the whole pipeline can be
re-measured on prose that varies the way real prose does.

**Validation is the point of this script, not a nicety.** A model asked to write a chapter
will invent state — a skill it likes the sound of, a level it thought was next. Those
inventions are contradictions the manifest cannot adjudicate, and they would land in the
corpus as *detector false positives* when they are really corpus errors. A single number
would then be doing double duty and neither reading would be right.

So every generated chapter is parsed and checked against the manifest before it is kept:
level, every stat, the skill list and the inventory must all match. Chapters that fail are
regenerated up to ``--attempts`` times; chapters that never pass are reported, and the run
says how many there were rather than quietly substituting a template.

The headline output is therefore **generation fidelity** — the share of chapters a model can
write while holding a state table it was handed. That is a result in its own right: it is
the same discipline the harness asks of a repair, applied to composition.

Output is cached per chapter under ``--out``, so a run resumes rather than regenerating.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.litrpg import (  # noqa: E402
    STAT_NAMES,
    Manifest,
    build_manifest,
    prompt_for_chapter,
)
from revisionbench.litrpg_detect import parse_chapters  # noqa: E402
from revisionbench.ollama import GenerationOptions, OllamaClient  # noqa: E402
from revisionbench.records import write_json  # noqa: E402


def load_corpus(folder: Path, manuscript_id: str) -> dict[int, str] | None:
    """Generated chapters for one manuscript, or None if it was never generated.

    Returns whatever chapters exist. A manuscript missing chapters is usable but its
    cross-chapter checks then compare non-adjacent chapters, so callers must report the
    count rather than assume it is complete.
    """
    directory = folder / manuscript_id
    if not directory.is_dir():
        return None
    chapters = {
        int(path.stem.split("_")[1]): path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("chapter_*.txt"))
    }
    return chapters or None


def assemble(chapter: int, body: str) -> str:
    """Prepend the chapter heading rather than trusting the model to emit it.

    The heading is structure, not prose, and a missing one silently merges two chapters
    into one — which makes every cross-chapter check compare the wrong pair.
    """
    text = body.strip()
    if text.lower().startswith(f"chapter {chapter}"):
        text = text.split("\n", 1)[1].strip() if "\n" in text else ""
    return f"Chapter {chapter}\n\n{text}\n"


def validate(manifest: Manifest, chapter: int, text: str) -> list[str]:
    """Every way this chapter disagrees with the manifest. Empty means faithful."""
    state = manifest.state_at(chapter)
    readings = parse_chapters(text)
    if not readings:
        return ["no chapter heading or status block found"]
    reading = readings[0]

    problems: list[str] = []
    if reading.level != state.level:
        problems.append(f"level {reading.level} != {state.level}")
    for stat in STAT_NAMES:
        if reading.stats.get(stat) != state.stats[stat]:
            problems.append(f"{stat} {reading.stats.get(stat)} != {state.stats[stat]}")
    if set(reading.skills) != set(state.skills):
        problems.append(f"skills {sorted(reading.skills)} != {sorted(state.skills)}")
    if set(reading.inventory) != set(state.inventory):
        problems.append(f"inventory {sorted(reading.inventory)} != {sorted(state.inventory)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manuscripts", type=int, default=6)
    parser.add_argument("--chapters", type=int, default=12)
    parser.add_argument("--model", default="phi4:latest")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "litrpg")
    args = parser.parse_args(argv)

    client = OllamaClient()
    identity = client.identity(args.model)
    print(f"model {identity.tag} digest {identity.digest[:12]}")

    attempts_used: Counter[int] = Counter()
    failures: list[dict] = []
    kept = total = 0
    started = time.time()

    for index in range(args.manuscripts):
        manuscript_id = f"ms-{index:03d}"
        manifest = build_manifest(manuscript_id, chapters=args.chapters, seed=index)
        folder = args.out / manuscript_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "manifest.json", manifest.as_dict())

        for state in manifest.chapters:
            total += 1
            path = folder / f"chapter_{state.chapter:03d}.txt"
            if path.is_file():  # resume
                kept += 1
                continue

            problems: list[str] = ["not attempted"]
            text = ""
            for attempt in range(args.attempts):
                options = GenerationOptions(
                    # Vary the seed per attempt: retrying at the same seed and temperature
                    # reproduces the same failure, so the retry would be free and useless.
                    seed=attempt,
                    temperature=args.temperature,
                    top_k=40,
                    top_p=0.9,
                    num_ctx=4096,
                    num_predict=700,
                    repeat_penalty=1.05,
                )
                if attempt == 0 and state.chapter == 1 and index == 0:
                    client.warm_up(args.model, options, think=False)
                generation = client.generate(
                    args.model, prompt_for_chapter(manifest, state.chapter), options, think=False
                )
                text = assemble(state.chapter, generation.text)
                problems = validate(manifest, state.chapter, text)
                if not problems:
                    attempts_used[attempt + 1] += 1
                    break

            if problems:
                failures.append(
                    {"manuscript_id": manuscript_id, "chapter": state.chapter, "problems": problems}
                )
                continue
            path.write_text(text, encoding="utf-8")
            kept += 1

        done = len(list(folder.glob("chapter_*.txt")))
        print(f"  {manuscript_id}: {done}/{args.chapters} chapters kept")

    elapsed = time.time() - started
    print(
        f"\n{kept}/{total} chapters faithful to the manifest ({kept / total:.0%}), {elapsed:.0f}s"
    )
    if attempts_used:
        print("  attempts needed:", dict(sorted(attempts_used.items())))
    if failures:
        print(f"\n  {len(failures)} chapters never passed validation:")
        for failure in failures[:8]:
            print(
                f"    {failure['manuscript_id']} ch{failure['chapter']}: "
                f"{'; '.join(failure['problems'][:2])}"
            )
        print(
            "\n  These are NOT in the corpus. A manuscript missing chapters still evaluates,\n"
            "  but its cross-chapter checks compare non-adjacent chapters — read any number\n"
            "  from it with that in mind."
        )

    write_json(
        args.out / "generation.json",
        {
            "model": identity.tag,
            "model_digest": identity.digest,
            "manuscripts": args.manuscripts,
            "chapters": args.chapters,
            "temperature": args.temperature,
            "attempts": args.attempts,
            "kept": kept,
            "total": total,
            "fidelity": kept / total if total else 0.0,
            "attempts_used": dict(attempts_used),
            "failures": failures,
            "elapsed_seconds": round(elapsed, 1),
        },
    )
    print(f"\nwrote {args.out / 'generation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
