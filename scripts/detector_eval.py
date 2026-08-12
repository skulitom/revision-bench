"""Detector precision and recall against Stratum B (docs/harness-gap.md §3).

    uv run python scripts/detector_eval.py

No model, no network. Answers the question that bounds a detect-then-repair harness: if
only flagged spans are eligible to be edited, **how many spans get flagged, and how many of
them deserved it?**

Two numbers, and they are not equally trustworthy:

**Precision, measured on the CLEAN Stratum-A passages.** Nothing was planted there, so
every complaint is a false positive. This is the honest number — the detectors have never
been shown the corruption of this text — and it is the harness's overreach ceiling: it
bounds how many spans a detect-then-repair loop would put in front of a model per passage.

**Recall against the planted defects: an UPPER BOUND, not a measurement.** The detectors
and the injector live in the same repo. They were written from properties of text rather
than from the injector's parameters (enforced by a test), but the corpus is still one
whose flaws are known, and no synthetic flaw is a substitute for a real manuscript's.

The caveat on precision too: Woolf and Richardson genuinely repeat phrases and shift
register, so some "false positives" are the detector noticing something real. Treat the
false-positive rate as an upper bound on spurious complaints.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.detect import DETECTORS, detect_all  # noqa: E402
from revisionbench.records import write_json  # noqa: E402
from revisionbench.text import word_count  # noqa: E402


#: A complaint counts as catching a defect if their spans overlap at all. Generous on
#: purpose: a detector that flags the containing sentence of a one-word name drift has
#: done its job, since the repair step gets the span and its context either way.
def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages")
    parser.add_argument(
        "--corrupted", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages_b"
    )
    parser.add_argument(
        "--defects", type=Path, default=REPO_ROOT / "data" / "corpus" / "defects.jsonl"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase1")
    args = parser.parse_args(argv)

    clean = {p.stem: json.loads(p.read_text("utf-8")) for p in sorted(args.clean.glob("*.json"))}
    corrupt = {
        p.stem: json.loads(p.read_text("utf-8")) for p in sorted(args.corrupted.glob("*.json"))
    }
    defects: dict[str, list[dict]] = defaultdict(list)
    for line in args.defects.read_text("utf-8").splitlines():
        row = json.loads(line)
        defects[row["passage_id"]].append(row)

    # ---- precision: complaints on clean prose are false positives by construction ----
    clean_complaints: Counter[str] = Counter()
    clean_words = 0
    per_passage_clean: dict[str, int] = {}
    for pid, record in clean.items():
        found = detect_all(record["text"])
        per_passage_clean[pid] = len(found)
        clean_words += word_count(record["text"])
        for complaint in found:
            clean_complaints[complaint.type] += 1

    # ---- recall: do complaints on the corrupted text land on the planted defects? ----
    caught: Counter[str] = Counter()
    planted: Counter[str] = Counter()
    corrupt_complaints: Counter[str] = Counter()
    corrupt_total = 0
    for pid, record in corrupt.items():
        found = detect_all(record["text"])
        corrupt_total += len(found)
        for complaint in found:
            corrupt_complaints[complaint.type] += 1
        for defect in defects[pid]:
            planted[defect["type"]] += 1
            span = tuple(defect["char_span"])
            if any(_overlaps(span, c.span) for c in found):
                caught[defect["type"]] += 1

    print("=== precision: complaints on CLEAN passages (all are false positives) ===")
    print(f"{'detector':<20}{'complaints':>12}{'per 1000 words':>17}")
    total_clean = sum(clean_complaints.values())
    for name in sorted(DETECTORS):
        n = clean_complaints[name]
        print(f"{name:<20}{n:>12}{n / clean_words * 1000:>17.2f}")
    print(f"{'TOTAL':<20}{total_clean:>12}{total_clean / clean_words * 1000:>17.2f}")
    print(
        f"\n{len(clean)} clean passages, {clean_words} words: "
        f"{total_clean / len(clean):.1f} spurious complaints per passage"
    )

    print("\n=== recall against planted defects (UPPER BOUND — see the module docstring) ===")
    print(f"{'defect type':<20}{'planted':>9}{'caught':>8}{'recall':>9}")
    mapping = {
        "name_drift": "name_variant",
        "echo": "echo",
        "tense_slip": "tense_outlier",
        "clunker": "overlong_sentence",
        "number_drift": "(no detector)",
    }
    for defect_type in sorted(planted):
        n, c = planted[defect_type], caught[defect_type]
        print(
            f"{defect_type:<20}{n:>9}{c:>8}{c / n if n else 0:>9.2f}"
            f"   <- {mapping.get(defect_type, '?')}"
        )
    total_planted, total_caught = sum(planted.values()), sum(caught.values())
    print(
        f"{'TOTAL':<20}{total_planted:>9}{total_caught:>8}"
        f"{total_caught / max(total_planted, 1):>9.2f}"
    )

    print(
        f"\ncomplaints raised on the corrupted text: {corrupt_total} "
        f"({corrupt_total / len(corrupt):.1f} per passage)"
    )
    print(
        f"of which land on a planted defect: {total_caught} "
        f"({total_caught / max(corrupt_total, 1):.0%})"
    )
    print(
        "\nThat last figure is the one that matters for the harness: it is the fraction of\n"
        "spans a detect-then-repair loop would put in front of a model that are actually\n"
        "defective. Compare it with A2i's ~4% of applied edits landing on a defect."
    )

    write_json(
        args.out / "detector_eval.json",
        {
            "clean": {
                "passages": len(clean),
                "words": clean_words,
                "complaints_by_type": dict(clean_complaints),
                "total": total_clean,
                "per_passage": total_clean / len(clean),
                "per_1000_words": total_clean / clean_words * 1000,
                "per_passage_detail": per_passage_clean,
            },
            "corrupted": {
                "complaints_by_type": dict(corrupt_complaints),
                "total": corrupt_total,
                "landing_on_a_defect": total_caught,
                "precision_vs_planted": total_caught / max(corrupt_total, 1),
            },
            "recall_upper_bound": {
                "planted": dict(planted),
                "caught": dict(caught),
                "overall": total_caught / max(total_planted, 1),
            },
        },
    )
    print(f"\nwritten: {args.out / 'detector_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
