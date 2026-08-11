"""The recall-vs-preservation frontier (plan.md §6, §7 M5).

    uv run python scripts/frontier.py

Reads a Stratum-B round artifact and the defect manifest, and reports each arm on **both**
axes at once. Until Stratum B existed the project could only measure one of them, and one
axis is not a frontier: an architecture that blocks every edit preserves voice perfectly
and fixes nothing, which plan.md §6 says must be rejected.

Three conventions, each guarding a way this table could mislead:

**Recall is reported beside the removal rate, never alone.** A defect can vanish because
the region containing it was cut. Those are censored rather than counted as repairs (see
:mod:`revisionbench.metrics.defects`), so a high recall on a passage that lost half its
content is a small denominator, not a good result — and the removal-rate column is what
says so.

**Edits applied is reported beside both.** An arm can look flawless on preservation by
doing nothing at all, and the apply count is the cheapest way to see it.

**The unrevised corrupted text is included as a row.** It is the floor: whatever it scores
is what "changed nothing" scores, and any arm near it is not working.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.metrics.defects import recall_report  # noqa: E402
from revisionbench.metrics.slop import load_lexicon, slop_index  # noqa: E402
from revisionbench.metrics.stats import mean  # noqa: E402
from revisionbench.metrics.stylometry import MetricError, StyleModel  # noqa: E402
from revisionbench.records import read_jsonl, write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rounds",
        type=Path,
        default=REPO_ROOT / "results" / "phase1" / "frontier" / "rounds.jsonl",
    )
    parser.add_argument(
        "--defects", type=Path, default=REPO_ROOT / "data" / "corpus" / "defects.jsonl"
    )
    parser.add_argument("--clean", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages")
    parser.add_argument(
        "--corrupted", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages_b"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase1")
    args = parser.parse_args(argv)

    rows = list(read_jsonl(args.rounds))
    if not rows:
        print(f"error: no rows in {args.rounds}", file=sys.stderr)
        return 2

    defects_by_passage: dict[str, list[dict]] = defaultdict(list)
    for line in args.defects.read_text("utf-8").splitlines():
        defect = json.loads(line)
        defects_by_passage[defect["passage_id"]].append(defect)

    clean = {
        p.stem: json.loads(p.read_text("utf-8"))["text"] for p in sorted(args.clean.glob("*.json"))
    }
    corrupted = {
        p.stem: json.loads(p.read_text("utf-8"))["text"]
        for p in sorted(args.corrupted.glob("*.json"))
    }

    # Fitted on the CLEAN passages: round 0 of a Stratum-B run is already corrupted, and
    # standardising against corrupted text would fold the defects into the baseline the
    # voice drift is measured from.
    model = StyleModel.fit(list(clean.values()), n_function_words=100)
    lexicon = load_lexicon()

    # Final round per (arm, passage).
    final: dict[tuple[str, str], dict] = {}
    for row in rows:
        if row["round"] == 0:
            continue
        key = (row["arm"], row["passage_id"])
        if key not in final or row["round"] > final[key]["round"]:
            final[key] = row

    arms = sorted({arm for arm, _ in final})
    results: list[dict[str, Any]] = []

    baseline = score_arm(
        "(unrevised)",
        [(pid, text) for pid, text in sorted(corrupted.items())],
        defects_by_passage,
        clean,
        model,
        lexicon,
        applied=0,
        rounds=0.0,
    )
    results.append(baseline)

    for arm in arms:
        items = [(pid, row["text"]) for (a, pid), row in final.items() if a == arm]
        applied = sum(
            (r.get("proposal") or {}).get("applied", 0)
            for r in rows
            if r["arm"] == arm and r["round"] > 0
        )
        rounds = mean([row["round"] for (a, _), row in final.items() if a == arm])
        results.append(
            score_arm(arm, items, defects_by_passage, clean, model, lexicon, applied, rounds)
        )

    header = (
        f"{'arm':<14}{'length':>8}{'drift':>8}{'slop':>7}"
        f"{'recall':>8}{'removed':>9}{'fixed':>7}{'applied':>9}{'rounds':>8}"
    )
    print("\n=== recall vs preservation ===")
    print(header)
    for r in results:
        rec = f"{r['recall']:.2f}" if r["recall"] is not None else "   -"
        print(
            f"{r['arm']:<14}{r['length_ratio']:>8.2f}{r['drift']:>8.2f}{r['slop']:>7.2f}"
            f"{rec:>8}{r['removal_rate']:>9.2f}{r['fixed']:>7}{r['applied']:>9}{r['rounds']:>8.1f}"
        )
    print(
        "\nrecall is over SURVIVING defects; `removed` is the fraction whose region was cut\n"
        "rather than repaired, and is not counted as a fix. Read the two together."
    )

    write_json(args.out / "frontier.json", {"arms": results})
    print(f"\nwritten: {args.out / 'frontier.json'}")
    return 0


def score_arm(
    arm: str,
    items: list[tuple[str, str]],
    defects_by_passage: dict[str, list[dict]],
    clean: dict[str, str],
    model: StyleModel,
    lexicon: Any,
    applied: int,
    rounds: float,
) -> dict[str, Any]:
    lengths, drifts, slops = [], [], []
    fixed = present = removed = planted = 0
    for passage_id, text in items:
        base = clean[passage_id]
        lengths.append(len(text.split()) / max(len(base.split()), 1))
        try:
            drifts.append(model.delta(base, text))
            slops.append(slop_index(text, lexicon).per_1000_words)
        except (MetricError, ValueError):
            pass
        report = recall_report(defects_by_passage[passage_id], text)
        fixed += report.fixed
        present += report.present
        removed += report.removed
        planted += report.planted
    surviving = fixed + present
    return {
        "arm": arm,
        "passages": len(items),
        "length_ratio": mean(lengths) if lengths else 0.0,
        "drift": mean(drifts) if drifts else 0.0,
        "slop": mean(slops) if slops else 0.0,
        "planted": planted,
        "fixed": fixed,
        "present": present,
        "removed": removed,
        "surviving": surviving,
        "recall": (fixed / surviving) if surviving else None,
        "removal_rate": (removed / planted) if planted else 0.0,
        "applied": applied,
        "rounds": rounds,
    }


if __name__ == "__main__":
    raise SystemExit(main())
