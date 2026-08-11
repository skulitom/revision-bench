"""Phase 0 figures: degradation curves from the metrics artifact (plan.md §14 M0-c).

    uv run python scripts/phase0_plots.py

Reads ``results/phase0/metrics.jsonl`` and writes ``results/phase0/phase0_curves.png``.
No model, no network.

Every panel plots the two prompt variants as separate lines, and the **word-count panel
comes first on purpose**. A single probe round showed this reviser cutting a 901-word
passage to 389 words, and a shorter text has mechanically fewer semicolons and a narrower
sentence-length spread. Any voice-drift curve read without the length curve beside it is
uninterpretable, so the figure refuses to present one without the other.

Intervals are a percentile bootstrap over **passages** (plan.md §11): rounds within a
passage are a dependent series by construction, so resampling rounds would treat one
trajectory as many independent observations and produce an interval far too narrow.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.metrics.stats import bootstrap_ci, mean  # noqa: E402
from revisionbench.records import read_jsonl  # noqa: E402

PANELS = [
    ("length_ratio", "Length vs round 0", "word count / round-0 word count", 1.0),
    ("delta_from_round0", "M1 voice drift from round 0", "mean |Δz| (all features)", None),
    ("delta_punct_from_round0", "M1 punctuation drift", "mean |Δz| (punctuation)", None),
    ("slop_per_1000w", "M3 slop index", "lexicon hits / 1000 words", None),
    (
        "m2_cross_author_delta",
        "M2 homogenization (H1) — axis from 0",
        "mean cross-author distance",
        None,
    ),
    ("edit_fraction", "M6 edit volume and thrash", "fraction", None),
]


def collect(records: list[dict]) -> dict[tuple[str, str, int], list[float]]:
    """``(prompt, field, round) -> values across passages``, skipping unscored rounds."""
    out: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in records:
        prompt, round_index = row["run"], row["round"]
        out[(prompt, "length_ratio", round_index)].append(row["length_ratio"])
        for field in ("delta_from_round0", "delta_punct_from_round0", "slop_per_1000w"):
            value = row.get(field)
            if value is not None:
                out[(prompt, field, round_index)].append(value)
        if row.get("edits"):
            out[(prompt, "edit_fraction", round_index)].append(row["edits"]["edit_fraction"])
        thrash = row.get("thrash")
        if thrash and thrash["thrash_fraction"] is not None:
            out[(prompt, "thrash_fraction", round_index)].append(thrash["thrash_fraction"])
    return out


def series(values: dict, prompt: str, field: str, rounds: list[int], seed: int = 0):
    """Mean and 95% bootstrap interval per round; ``None`` where a cell has no data."""
    means, lows, highs = [], [], []
    for round_index in rounds:
        cell = values.get((prompt, field, round_index), [])
        if not cell:
            means.append(None)
            lows.append(None)
            highs.append(None)
            continue
        means.append(mean(cell))
        if len(cell) > 1:
            low, high = bootstrap_ci(cell, mean, n_resamples=1000, seed=seed)
        else:
            low = high = cell[0]
        lows.append(low)
        highs.append(high)
    return means, lows, highs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metrics", type=Path, default=REPO_ROOT / "results" / "phase0" / "metrics.jsonl"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "results" / "phase0" / "metrics_summary.json",
        help="M2 is a per-round cross-passage statistic, so it lives in the summary rather "
        "than on the per-passage metric rows",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "phase0" / "phase0_curves.png"
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("error: matplotlib is not installed; run `uv sync`", file=sys.stderr)
        return 2

    records = list(read_jsonl(args.metrics))
    if not records:
        print(
            f"error: no rows in {args.metrics}; run scripts/phase0_metrics.py first",
            file=sys.stderr,
        )
        return 2

    # One line per run, a run being one model crossed with one prompt.
    prompts = sorted({r["run"] for r in records})
    rounds = sorted({r["round"] for r in records})
    values = collect(records)

    # M2 compares passages to each other, so it is one number per (prompt, round) rather
    # than a per-passage value, and phase0_metrics.py computes it into the summary. The
    # first version of this script read it off the per-passage rows, where the key simply
    # does not exist, and drew an empty panel -- the good failure mode, but only by luck.
    if not args.summary.is_file():
        print(
            f"error: {args.summary} not found; run scripts/phase0_metrics.py first", file=sys.stderr
        )
        return 2
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    m2 = {
        (prompt, entry["round"]): entry.get("m2_cross_author_delta")
        for prompt, block in summary["by_run"].items()
        for entry in block["per_round"]
    }

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    if len(prompts) > len(palette):
        print(
            f"error: {len(prompts)} runs but only {len(palette)} distinct colours; add "
            f"more rather than letting two runs share one",
            file=sys.stderr,
        )
        return 2
    colours = dict(zip(prompts, palette, strict=False))
    # Derived, not hardcoded: the title named a single model while the figure was already
    # plotting two, which is the sort of caption error that outlives the run it describes.
    models = sorted({r["model_tag"] for r in records})
    passages = len({r["passage_id"] for r in records})
    authors = len({r["author_id"] for r in records})
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    figure.suptitle(
        f"Phase 0 — arm A0 (unconstrained revision loop) — {', '.join(models)} — "
        f"{passages} passages x {authors} authors",
        fontsize=13,
    )

    for axis, (field, title, ylabel, reference) in zip(axes.flat, PANELS, strict=True):
        for prompt in prompts:
            colour = colours[prompt]
            if field == "m2_cross_author_delta":
                means = [m2.get((prompt, r)) for r in rounds]
                xs = [r for r, v in zip(rounds, means, strict=True) if v is not None]
                ys = [v for v in means if v is not None]
                axis.plot(xs, ys, marker="o", color=colour, label=prompt)
                # Anchored at zero deliberately. Autoscaled, this panel spans 1.125-1.19
                # and a 3% wobble reads as a cliff -- on the one panel that carries a
                # primary endpoint (plan.md §11: H1 is the M2 slope). H1 predicts this
                # line falls monotonically. It does not, and the honest way to show a null
                # is at a scale where "flat" looks flat.
                axis.set_ylim(bottom=0.0, top=max(ys) * 1.25 if ys else 1.0)
                continue

            means, lows, highs = series(values, prompt, field, rounds)
            xs = [r for r, v in zip(rounds, means, strict=True) if v is not None]
            ys = [v for v in means if v is not None]
            lo = [v for v in lows if v is not None]
            hi = [v for v in highs if v is not None]
            axis.plot(xs, ys, marker="o", color=colour, label=prompt)
            axis.fill_between(xs, lo, hi, color=colour, alpha=0.15)

            if field == "edit_fraction":
                tm, _, _ = series(values, prompt, "thrash_fraction", rounds)
                txs = [r for r, v in zip(rounds, tm, strict=True) if v is not None]
                tys = [v for v in tm if v is not None]
                axis.plot(
                    txs, tys, marker="s", linestyle="--", color=colour, label=f"{prompt} (thrash)"
                )

        if reference is not None:
            axis.axhline(reference, color="grey", linestyle=":", linewidth=1)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("round")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.005,
        "Bands are 95% percentile bootstrap intervals over passages (1000 resamples). Read "
        "every panel against the length panel: both revisers compress under the neutral "
        "prompt, and a shorter text has mechanically fewer semicolons and a narrower "
        "sentence-length spread. Note that the length instruction holds for gemma3:4b "
        "(0.97x) and largely fails for phi4 (0.69x).",
        ha="center",
        fontsize=8.5,
        style="italic",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=140)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
