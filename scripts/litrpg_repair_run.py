"""Run A2d — complaint-gated repair — over the synthetic LitRPG stratum.

    uv run python scripts/litrpg_repair_run.py --manuscripts 5 --model phi4:latest

`litrpg_eval.py` showed the detector finds 99% of planted contradictions at 88% precision.
This is the other half: given those complaints, can a local model *repair* them without
touching anything else?

Three things get measured, and the third is the one no earlier stratum could compute.

1. **Resolution** — did the cited complaint go away?
2. **Rejection reasons** — of proposals refused, how many were refused for collateral damage
   (``new_complaint``) rather than for failing their own target? That number is the value of
   the verifier, because collateral damage is precisely what a per-edit reviewer cannot see.
3. **Exact restoration** — did the repair put back the text the manifest says belongs there?
   Stratum A can only ask whether a defect "looks fixed"; here the clean text is known
   byte-for-byte, so a repair that resolves the complaint by deleting the sentence is
   distinguishable from one that actually restores the fact. Those score identically under
   every metric this project had before.

The headline number is **chapters edited that had no defect in them**. Not "how much of the
book was left alone" — a densely corrupted manuscript should have many chapters edited, so
that figure measures the corruption density rather than the arm. The guarantee is narrower
and stronger: nothing without a complaint against it is eligible to be edited at all, so any
chapter modified without a planted defect is the overreach this design was built to make
structurally impossible.
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

from revisionbench.litrpg import build_manifest, render_manuscript  # noqa: E402
from revisionbench.litrpg_inject import inject_manuscript  # noqa: E402
from revisionbench.litrpg_repair import repair_manuscript  # noqa: E402
from revisionbench.ollama import GenerationOptions, OllamaClient  # noqa: E402
from revisionbench.records import write_json  # noqa: E402


def _chapters(text: str) -> dict[int, str]:
    """Split on chapter headings so comparisons survive length-changing repairs."""
    out: dict[int, str] = {}
    current = 0
    for block in text.split("\nChapter "):
        head, _, rest = block.partition("\n")
        number = head.strip().split()[-1] if head.strip() else ""
        if number.isdigit():
            current = int(number)
            out[current] = rest
        elif current:
            out[current] += block
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manuscripts", type=int, default=5)
    parser.add_argument("--chapters", type=int, default=12)
    parser.add_argument("--per-type", type=int, default=1)
    parser.add_argument("--model", default="phi4:latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "litrpg" / "repair.json"
    )
    args = parser.parse_args(argv)

    options = GenerationOptions(
        seed=args.seed,
        temperature=0.0,
        top_k=40,
        top_p=0.9,
        num_ctx=8192,
        num_predict=256,
        repeat_penalty=1.0,
    )
    client = OllamaClient()
    identity = client.identity(args.model)
    print(f"model {identity.tag} digest {identity.digest[:12]}")
    client.warm_up(args.model, options, think=False)

    reasons: Counter[str] = Counter()
    resolved = restored = planted_total = 0
    untouched_chapters = total_chapters = 0
    collateral_chapters: list[tuple[str, int]] = []
    complaints_before = complaints_after = 0
    runs: list[dict] = []
    started = time.time()

    for index in range(args.manuscripts):
        manuscript_id = f"ms-{index:03d}"
        manifest = build_manifest(manuscript_id, chapters=args.chapters, seed=index)
        clean = render_manuscript(manifest)
        corrupt, defects = inject_manuscript(manifest, per_type=args.per_type, seed=index)
        planted_total += len(defects)

        report = repair_manuscript(
            client,
            args.model,
            corrupt,
            options,
            manuscript_id=manuscript_id,
            max_rounds=args.max_rounds,
        )
        complaints_before += report.complaints_before
        complaints_after += report.complaints_after
        for outcome in report.outcomes:
            reasons[outcome.reason] += 1

        clean_chapters = _chapters(clean)
        corrupt_chapters = _chapters(corrupt)
        fixed_chapters = _chapters(report.text)

        # Exact restoration, per planted defect: the corruption gone AND the manifest's
        # text back. A repair that resolves the complaint by deleting the line scores on
        # the first half and fails the second, which is the distinction that matters.
        for defect in defects:
            chapter_text = fixed_chapters.get(defect.chapter, "")
            gone = defect.corrupt_fragment.strip() not in chapter_text
            back = defect.original_fragment.strip() in chapter_text
            resolved += int(gone)
            restored += int(gone and back)

        # The guarantee this arm exists to provide is not "most of the book is untouched" —
        # a densely corrupted manuscript should have many chapters edited. It is that NO
        # chapter without a complaint against it was edited at all. Anything else is the
        # overreach the inversion was designed to make structurally impossible.
        planted_chapters = {d.chapter for d in defects}
        for number, body in corrupt_chapters.items():
            total_chapters += 1
            changed = fixed_chapters.get(number, "") != body
            untouched_chapters += int(not changed)
            if changed and number not in planted_chapters:
                collateral_chapters.append((manuscript_id, number))

        runs.append(
            {
                "manuscript_id": manuscript_id,
                "planted": [d.as_dict() for d in defects],
                **report.as_dict(),
            }
        )
        print(
            f"  {manuscript_id}: {report.complaints_before} complaints -> "
            f"{report.complaints_after}, {sum(1 for o in report.outcomes if o.applied)} applied"
        )

    elapsed = time.time() - started
    proposals = sum(reasons.values())
    accepted = reasons["accepted"]
    corrupted_chapters = total_chapters - untouched_chapters

    print(f"\n{args.manuscripts} manuscripts, {planted_total} planted defects, {elapsed:.0f}s\n")
    print(f"  complaints            {complaints_before} -> {complaints_after}")
    print(f"  proposals             {proposals}")
    print(f"  {'accepted':<22}{accepted:>5}  ({accepted / proposals:.0%})" if proposals else "")
    for reason, count in reasons.most_common():
        if reason != "accepted":
            print(f"    rejected: {reason:<20}{count:>5}")
    print(f"\n  defect resolved       {resolved}/{planted_total} = {resolved / planted_total:.0%}")
    print(
        f"  EXACTLY RESTORED      {restored}/{planted_total} = {restored / planted_total:.0%}"
        "   (corruption gone AND manifest text back)"
    )
    print(
        f"\n  chapters left untouched  {untouched_chapters}/{total_chapters} = "
        f"{untouched_chapters / total_chapters:.0%}"
    )
    print(f"  chapters modified        {corrupted_chapters}  (defects planted in {planted_total})")
    print(
        f"  CHAPTERS EDITED WITH NO DEFECT IN THEM   {len(collateral_chapters)}"
        "   <- the overreach number"
    )
    for manuscript_id, number in collateral_chapters[:5]:
        print(f"      {manuscript_id} chapter {number}")

    write_json(
        args.out,
        {
            "model": identity.tag,
            "model_digest": identity.digest,
            "manuscripts": args.manuscripts,
            "chapters": args.chapters,
            "planted": planted_total,
            "resolved": resolved,
            "restored": restored,
            "reasons": dict(reasons),
            "complaints_before": complaints_before,
            "complaints_after": complaints_after,
            "untouched_chapters": untouched_chapters,
            "collateral_chapters": [list(c) for c in collateral_chapters],
            "total_chapters": total_chapters,
            "elapsed_seconds": round(elapsed, 1),
            "runs": runs,
        },
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
