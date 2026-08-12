"""Detector precision and recall on the synthetic LitRPG stratum.

    uv run python scripts/litrpg_eval.py --manuscripts 20

`harness-gap.md` §1.1 identified the product blocker: A2i applied 195 edits to repair 8
planted defects — roughly 24 edits per fix, with ~96% of what it changed not broken. §2.1
argued the way out is to make eligibility depend on a *located, checkable complaint*, and
that most defect classes that matter to a book need no judge to produce one. §3 called for
exactly this measurement and never got it, because the corpus could not express a
cross-chapter contradiction.

This measures it. Two numbers, and the second is the one that decides the architecture:

- **recall** — of the contradictions planted, how many does the detector locate?
- **precision** — of the complaints it raises, how many correspond to a planted defect?

Precision is the harness's overreach ceiling. Under detect-then-repair only a complained-of
span may be edited, so a false complaint is a licensed unnecessary edit — precision *is* the
24:1 ratio, restated in a form that can be improved deliberately rather than hoped for.

Reported separately for cross-chapter defects, because within-chapter and cross-chapter
detection are different problems and a pooled figure hides which one is actually solved.

**What this cannot support.** The prose is templated and the status blocks have a fixed
schema, so extraction is near-perfect here and both numbers are upper bounds. A real serial
varies its formatting constantly. Treat these as "what is achievable when parsing is easy",
not as a forecast — and see `litrpg.py` for the rest of the caveats.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.litrpg import build_manifest, render_manuscript  # noqa: E402
from revisionbench.litrpg_detect import detect_all_litrpg  # noqa: E402
from revisionbench.litrpg_inject import inject_manuscript  # noqa: E402
from revisionbench.records import write_json  # noqa: E402
from scripts.litrpg_generate import load_corpus  # noqa: E402


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manuscripts", type=int, default=20)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="directory of model-written chapters from litrpg_generate.py. Without it, "
        "the templated prose is used and the numbers are the upper bound reported in "
        "findings-litrpg.md section 4 rather than a measurement on realistic text.",
    )
    parser.add_argument("--chapters", type=int, default=16)
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "litrpg" / "eval.json")
    args = parser.parse_args(argv)

    planted_by_type: dict[str, list[bool]] = defaultdict(list)
    cross_hits: list[bool] = []
    total_complaints = 0
    matched_complaints = 0
    clean_false_positives = 0
    planted = 0
    corpus: list[dict] = []

    for index in range(args.manuscripts):
        manuscript_id = f"ms-{index:03d}"
        manifest = build_manifest(manuscript_id, chapters=args.chapters, seed=index)
        generated = load_corpus(args.corpus, manuscript_id) if args.corpus else None
        if args.corpus and not generated:
            print(f"  {manuscript_id}: not generated, skipped", file=sys.stderr)
            continue
        if generated:
            # Restrict the manifest to the chapters that survived validation, so a missing
            # chapter is an absent comparison rather than a silently wrong one.
            manifest = replace(
                manifest, chapters=tuple(c for c in manifest.chapters if c.chapter in generated)
            )

        # The precision floor: a detector that fires on an uncorrupted manuscript makes
        # every recall number below meaningless, because nothing downstream could tell a
        # real complaint from a reflex.
        clean_text = (
            "\n".join(generated[c.chapter] for c in manifest.chapters)
            if generated
            else render_manuscript(manifest)
        )
        clean_false_positives += len(detect_all_litrpg(clean_text))

        text, defects = inject_manuscript(
            manifest, per_type=args.per_type, seed=index, chapters_text=generated
        )
        complaints = detect_all_litrpg(text)
        planted += len(defects)

        hit_defects: set[int] = set()
        for complaint in complaints:
            total_complaints += 1
            for position, defect in enumerate(defects):
                if _overlaps(complaint.span, defect.char_span):
                    matched_complaints += 1
                    hit_defects.add(position)
                    break

        for position, defect in enumerate(defects):
            found = position in hit_defects
            planted_by_type[defect.type].append(found)
            if defect.cross_chapter:
                cross_hits.append(found)

        corpus.append(
            {
                "manuscript_id": manuscript_id,
                "chapters": args.chapters,
                "planted": [d.as_dict() for d in defects],
                "complaints": [c.as_dict() for c in complaints],
            }
        )

    hits = sum(sum(v) for v in planted_by_type.values())
    recall = hits / planted if planted else 0.0
    precision = matched_complaints / total_complaints if total_complaints else 0.0

    print(f"{args.manuscripts} manuscripts x {args.chapters} chapters, {planted} defects planted\n")
    print(f"  {'defect type':<28}{'recall':>10}{'planted':>10}")
    for defect_type in sorted(planted_by_type):
        found = planted_by_type[defect_type]
        print(f"  {defect_type:<28}{sum(found) / len(found):>9.0%}{len(found):>10}")
    print(f"\n  {'OVERALL RECALL':<28}{recall:>9.0%}{planted:>10}")
    if cross_hits:
        print(
            f"  {'  of which cross-chapter':<28}"
            f"{sum(cross_hits) / len(cross_hits):>9.0%}{len(cross_hits):>10}"
        )
    print(f"\n  complaints raised          {total_complaints}")
    print(f"  matching a planted defect  {matched_complaints}")
    print(f"  PRECISION                  {precision:.0%}")
    print(f"  false positives on clean text  {clean_false_positives}")

    # A2i's number, restated: edits applied per defect repaired. Under detect-then-repair
    # every complaint licenses at most one edit, so 1/precision is the ceiling.
    if precision:
        print(
            f"\n  edits licensed per real defect: {1 / precision:.1f}"
            f"   (A2i's revise-then-gate loop: ~24, harness-gap.md §1.1)"
        )

    write_json(
        args.out,
        {
            "manuscripts": args.manuscripts,
            "chapters": args.chapters,
            "planted": planted,
            "recall": recall,
            "precision": precision,
            "clean_false_positives": clean_false_positives,
            "recall_by_type": {k: sum(v) / len(v) for k, v in planted_by_type.items()},
            "cross_chapter_recall": (sum(cross_hits) / len(cross_hits)) if cross_hits else None,
            "corpus": corpus,
        },
    )
    print(f"\nwrote {args.out}")
    print(
        "\nUpper bounds, not forecasts: templated prose and a fixed status-block schema make\n"
        "extraction near-perfect here. See litrpg.py for what this stratum cannot support."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
