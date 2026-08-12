"""Phase 2, first experiment: can the panel tell broken prose from clean prose?

    uv run python scripts/judge_sanity.py

**A positive control, and nothing more.** Every pair here has an objectively correct
answer: one version is a sentence from the corpus, the other is that same sentence with a
planted defect in it (Stratum B). Choosing the clean one is the floor of competence. A
judge that cannot clear this floor cannot be trusted on aesthetic comparisons, and its
agreement with the rest of the panel would be agreement about nothing.

Its counterpart matters just as much and is why this runs before anything else:
**a judge that clears it has not thereby been shown to judge prose quality.** Detecting a
misspelled name is not the same skill as preferring better writing. This experiment can
only disqualify, never qualify.

Reported first, before accuracy: **position bias**. Version order is randomised per
(pair, judge) and recorded, so a judge that simply picks the first option is visible as
~1.0 rather than hidden inside a 50%-ish accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.judge import (  # noqa: E402
    DEFAULT_PANEL,
    JudgeVerdict,
    Pair,
    agreement_matrix,
    judge_pair,
    position_bias,
)
from revisionbench.ollama import GenerationOptions, OllamaClient, OllamaError  # noqa: E402
from revisionbench.provenance import RunProvenance, utc_now  # noqa: E402
from revisionbench.records import JsonlWriter, write_json  # noqa: E402
from revisionbench.text import sentences  # noqa: E402


def build_pairs(clean_dir: Path, defects_path: Path) -> list[Pair]:
    """One pair per planted defect: the clean sentence against the corrupted one.

    Built from the defect manifest, which stores both forms exactly, so no diffing or
    guessing is involved and the ground truth is the injector's own record.
    """
    clean = {
        p.stem: json.loads(p.read_text("utf-8"))["text"] for p in sorted(clean_dir.glob("*.json"))
    }
    pairs: list[Pair] = []
    for line in defects_path.read_text("utf-8").splitlines():
        defect = json.loads(line)
        original = defect["clean_sentence"].strip()
        corrupt = defect["corrupt_fragment"].strip()
        # Only defects that replace a whole sentence give a clean sentence-level pair. A
        # name drift swaps one word, so the corrupted *sentence* has to be reconstructed.
        if defect["type"] in ("name_drift", "number_drift"):
            corrupt_sentence = original.replace(defect["original_fragment"], corrupt, 1)
            if corrupt_sentence == original:
                continue
        elif defect["detect_kind"] == "minimal_pair":
            corrupt_sentence = corrupt
        elif defect["type"] == "clunker":
            # The clunker is an inserted sentence, not a replacement, so the honest pair is
            # the passage's own neighbouring sentence against the bloated intruder.
            corrupt_sentence = corrupt
        else:
            continue
        if not original or not corrupt_sentence or original == corrupt_sentence:
            continue
        context = clean.get(defect["passage_id"], "")
        pairs.append(
            Pair(
                pair_id=defect["defect_id"],
                context=_context_around(context, original),
                version_1=original,
                version_2=corrupt_sentence,
                better="version_1",
                kind=defect["type"],
                meta={"passage_id": defect["passage_id"]},
            )
        )
    return pairs


def _context_around(passage: str, sentence: str, window: int = 3) -> str:
    """A few sentences around the one under judgement, so 'in context' means something."""
    parts = sentences(passage)
    for index, part in enumerate(parts):
        if part.strip() == sentence.strip():
            lo = max(0, index - window)
            return " ".join(parts[lo : index + window + 1])
    return " ".join(parts[: 2 * window + 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages")
    parser.add_argument(
        "--defects", type=Path, default=REPO_ROOT / "data" / "corpus" / "defects.jsonl"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase2")
    parser.add_argument("--panel", nargs="+", default=list(DEFAULT_PANEL))
    parser.add_argument(
        "--self-judge",
        default="phi4:latest",
        help="the reviser, judged separately to measure self-preference; never in the panel",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args(argv)

    pairs = build_pairs(args.clean, args.defects)
    if not pairs:
        print("error: no pairs could be built", file=sys.stderr)
        return 2

    kinds: dict[str, int] = defaultdict(int)
    for pair in pairs:
        kinds[pair.kind] += 1
    print(f"{len(pairs)} objective pairs (clean vs corrupted): {dict(kinds)}")

    options = GenerationOptions(
        seed=args.seed,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        num_ctx=8192,
        num_predict=64,
        repeat_penalty=1.0,
    )
    client = OllamaClient(args.host, keep_alive="60m")

    judges = list(args.panel)
    if args.self_judge and args.self_judge not in judges:
        judges.append(args.self_judge)

    identities = {}
    for tag in judges:
        try:
            identities[tag] = client.identity(tag)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    verdicts: list[JudgeVerdict] = []
    failures: dict[str, int] = defaultdict(int)
    args.out.mkdir(parents=True, exist_ok=True)
    # The artifact is rewritten, not appended: JsonlWriter opens in append mode so that a
    # resumed sweep continues, but this script always judges every pair, and two runs'
    # verdicts in one file would silently double every denominator.
    (args.out / "judge_sanity.jsonl").unlink(missing_ok=True)
    with JsonlWriter(args.out / "judge_sanity.jsonl", fsync=False) as out:
        for tag in judges:
            # Warmed here rather than up front. Five judges total ~22 GB against a 24 GB
            # card, so Ollama evicts and reloads between blocks; a warm-up done before the
            # first block is stale by the fifth, and the first call after a reload is the
            # irreproducible one (docs/findings-phase0.md §5.1). Two runs of this script
            # disagreed by two verdicts before this line existed.
            client.warm_up(tag, options)
            for pair in pairs:
                try:
                    verdict = judge_pair(client, identities[tag], pair, options, seed=args.seed)
                except OllamaError as exc:
                    # Recorded, never defaulted: an invented verdict enters the agreement
                    # statistics as if it were data.
                    failures[tag] += 1
                    print(f"  ! {tag} failed on {pair.pair_id}: {str(exc)[:90]}")
                    continue
                verdicts.append(verdict)
                out.write({**verdict.as_dict(), "kind": pair.kind})
            done = [v for v in verdicts if v.judge == tag]
            correct = sum(1 for v in done if v.correct)
            print(
                f"  {tag:<20} {correct}/{len(done)} correct   "
                f"position-A rate {position_bias(done):.2f}"
            )

    print("\n=== position bias (0.50 = unbiased; report before accuracy) ===")
    for tag in judges:
        subset = [v for v in verdicts if v.judge == tag]
        bias = position_bias(subset)
        flag = "" if bias is None or 0.35 <= bias <= 0.65 else "   <-- ANSWERING WITH POSITION"
        print(f"  {tag:<20} {bias if bias is None else f'{bias:.2f}'}{flag}")

    print("\n=== accuracy on a task with an objectively correct answer ===")
    print(f"{'judge':<20}{'n':>5}{'correct':>9}{'accuracy':>10}   by defect type")
    summary = {}
    for tag in judges:
        subset = [v for v in verdicts if v.judge == tag]
        if not subset:
            continue
        correct = sum(1 for v in subset if v.correct)
        by_kind: dict[str, list[int]] = defaultdict(list)
        for verdict in subset:
            kind = next(p.kind for p in pairs if p.pair_id == verdict.pair_id)
            by_kind[kind].append(int(bool(verdict.correct)))
        detail = "  ".join(f"{k}={sum(v)}/{len(v)}" for k, v in sorted(by_kind.items()))
        role = "SELF" if tag == args.self_judge else "panel"
        print(f"{tag:<20}{len(subset):>5}{correct:>9}{correct / len(subset):>10.2f}   {detail}")
        summary[tag] = {
            "role": role,
            "n": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset),
            "position_a_rate": position_bias(subset),
            "by_kind": {k: {"correct": sum(v), "n": len(v)} for k, v in by_kind.items()},
            "failures": failures[tag],
        }

    panel_only = [v for v in verdicts if v.judge != args.self_judge]
    print("\n=== inter-judge agreement (panel only, shared pairs) ===")
    for (a, b), value in sorted(agreement_matrix(panel_only).items()):
        print(f"  {a:<20} vs {b:<20} {value:.2f}")

    # --- what a gate would actually use: majority vote, with its coverage ---------------
    #
    # Reported as a pair, always. Majority voting turns disagreement into *abstention*, so
    # accuracy on the pairs where the panel speaks can look excellent while it declines to
    # answer a third of the time. Quoting the accuracy alone would sell a gate that mostly
    # says nothing — the same shape as quoting defect recall without the removal rate.
    by_pair: dict[str, dict[str, str]] = defaultdict(dict)
    for verdict in verdicts:
        by_pair[verdict.pair_id][verdict.judge] = verdict.chose

    def vote(judges: Sequence[str]) -> tuple[int, int, int]:
        correct = answered = 0
        for votes_by_judge in by_pair.values():
            votes = [votes_by_judge[j] for j in judges if j in votes_by_judge]
            if not votes:
                continue
            tally = Counter(votes)
            top, count = tally.most_common(1)[0]
            if len(tally) > 1 and count * 2 <= len(votes):
                continue  # a tie is not a verdict
            answered += 1
            correct += top == "version_1"
        return correct, answered, len(by_pair)

    unbiased = [
        t
        for t in args.panel
        if (position_bias([v for v in verdicts if v.judge == t]) or 0.5) >= 0.35
    ]
    configurations: list[tuple[str, list[str]]] = [
        ("full panel", list(args.panel)),
        ("panel minus position-biased", unbiased),
        *[(f"{t} alone", [t]) for t in judges],
    ]
    print("\n=== candidate gate signals: majority vote, with coverage ===")
    print(f"{'signal':<38}{'accuracy':>10}{'coverage':>10}   (accuracy is over answered pairs)")
    signals = {}
    for label, members in configurations:
        if not members:
            continue
        correct, answered, total = vote(members)
        accuracy = correct / answered if answered else None
        print(
            f"{label:<38}{(f'{accuracy:.2f}' if accuracy is not None else '   -'):>10}"
            f"{answered / total:>10.2f}   {correct}/{answered} of {total}"
        )
        signals[label] = {
            "members": members,
            "correct": correct,
            "answered": answered,
            "total": total,
            "accuracy": accuracy,
            "coverage": answered / total,
        }

    print(
        "\nChance is 0.50. This experiment can only DISQUALIFY a judge: clearing a floor of\n"
        "'spot the planted defect' is not evidence of taste in prose."
    )

    write_json(
        args.out / "judge_sanity.json",
        {
            "provenance": RunProvenance(
                run_id="judge-sanity", started_at=utc_now(), config_hash="n/a"
            )
            .with_artifacts(
                panel=list(args.panel),
                self_judge=args.self_judge,
                seed=args.seed,
                judges={t: identities[t].as_dict() for t in identities},
                n_pairs=len(pairs),
            )
            .as_dict(),
            "judges": summary,
            "agreement": {f"{a} vs {b}": v for (a, b), v in agreement_matrix(panel_only).items()},
            "gate_signals": signals,
        },
    )
    print(f"\nwritten: {args.out / 'judge_sanity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
