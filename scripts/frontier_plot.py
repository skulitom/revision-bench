"""The recall-vs-preservation frontier, plotted (plan.md §6; NEXT.md standing rules).

    uv run python scripts/frontier_plot.py

Reads ``results/phase1/frontier.json`` and writes ``results/phase1/frontier.png``.

**Edit volume is encoded in the mark, not left to the caption.** A2e's lesson —
preservation by inaction — is a class of failure, not an instance: any arm that changes
little scores well on every preservation axis, so a plot showing only preservation and
recall would put a do-nothing arm in the best corner and say nothing about why. Marker area
is proportional to edits applied, so an arm that did nothing is visibly a dot.

The second panel carries the removal rate, for the same reason the table prints it next to
recall: a defect that vanished because its region was cut is censored rather than repaired,
so recall on a gutted passage is a small denominator rather than a good score.

The unrevised corrupted text is plotted as the floor. Any arm near it is not working.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Marker area per applied edit. Tuned so a zero-edit arm is still visible as a small dot
#: rather than vanishing, which would hide the very failure the encoding exists to show.
AREA_PER_EDIT = 3.0
MIN_AREA = 40.0

LABELS = {
    "(unrevised)": "unrevised (floor)",
    "A0": "A0 whole passage",
    "A2p": "A2p paragraph",
    "A2e": "A2e edit-list",
    "A2i": "A2i indexed + vetoes",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--frontier", type=Path, default=REPO_ROOT / "results" / "phase1" / "frontier.json"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "phase1" / "frontier.png"
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("error: matplotlib is not installed; run `uv sync`", file=sys.stderr)
        return 2

    if not args.frontier.is_file():
        print(f"error: {args.frontier} not found; run scripts/frontier.py first", file=sys.stderr)
        return 2
    arms = json.loads(args.frontier.read_text("utf-8"))["arms"]

    palette = ["#7f7f7f", "#d62728", "#2ca02c", "#9467bd", "#1f77b4", "#ff7f0e"]
    colours = {a["arm"]: palette[i % len(palette)] for i, a in enumerate(arms)}

    figure, (axis, bars) = plt.subplots(1, 2, figsize=(15, 6.5), width_ratios=[1.6, 1])

    for entry in arms:
        recall = entry["recall"]
        if recall is None:
            # Nothing survived to be judged. Not a zero — plotting it as one would put a
            # passage that was deleted wholesale on the same spot as one the arm failed to
            # repair, which are opposite outcomes.
            continue
        area = max(MIN_AREA, entry["applied"] * AREA_PER_EDIT)
        axis.scatter(
            recall,
            entry["drift"],
            s=area,
            color=colours[entry["arm"]],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
        )
        axis.annotate(
            entry["arm"],
            (recall, entry["drift"]),
            textcoords="offset points",
            xytext=(0, -4),
            ha="center",
            va="top",
            fontsize=9,
        )

    axis.set_xlabel("defect-fix recall  (over surviving defects)  →  better")
    axis.set_ylabel("voice drift from the clean original  (mean |Δz|)  ←  better")
    axis.invert_yaxis()
    axis.set_title(
        "Recall vs preservation\nmarker area ∝ edits applied — a do-nothing arm is a dot",
        fontsize=11,
    )
    axis.grid(alpha=0.25, zorder=0)
    # Fixed-size proxy handles. A scatter legend inherits each series' marker area, and the
    # busiest arm applies 719 edits — so the legend swatch grew large enough to cover the
    # plot it was labelling. The sizes carry meaning on the axes, not in the key.
    from matplotlib.lines import Line2D

    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=colours[e["arm"]],
                markeredgecolor="black",
                label=f"{LABELS.get(e['arm'], e['arm'])} — {e['applied']} edits applied",
            )
            for e in arms
            if e["recall"] is not None
        ],
        fontsize=8,
        loc="lower left",
        framealpha=0.9,
    )

    names = [LABELS.get(a["arm"], a["arm"]) for a in arms]
    bars.barh(
        names,
        [a["removal_rate"] for a in arms],
        color=[colours[a["arm"]] for a in arms],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
    )
    bars.set_xlabel("fraction of planted defects whose region was CUT (censored, not fixed)")
    bars.set_title("The caveat on recall", fontsize=11)
    bars.grid(axis="x", alpha=0.25)
    bars.set_xlim(0, 1)

    figure.text(
        0.5,
        0.01,
        "Recall counts only defects whose region survived; a defect deleted along with its "
        "surroundings is censored, not repaired. Read the left panel against the right: high "
        "recall beside a high removal rate is a small denominator, not a good result.",
        ha="center",
        fontsize=8.5,
        style="italic",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=140)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
