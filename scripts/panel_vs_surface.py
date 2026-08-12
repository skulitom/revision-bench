"""Does the panel follow surface features into error? The decisive version of §12.

    uv run python scripts/panel_vs_surface.py

`surface_predictability.py` established that a bag of counts identifies the model's edit
66.1% of the time (95% CI [60.9%, 68.0%], null 50%). That is suggestive but it does not
settle anything, because agreement between the panel and a surface classifier has two
readings: the panel may be keying on surface features, or both may be independently right —
the edits really are shorter and flatter, *and* the originals really are better prose.

Those two readings come apart on the pairs where the surface classifier is **wrong**.

Take the pairs where the classifier's out-of-fold margin points at the original as if it
were the edit. On those pairs:

- If the panel is surface-driven, it should follow the classifier into the error, and its
  preference for the original should collapse — below chance, since the surface evidence
  now actively points the wrong way.
- If the panel is reading the prose, its preference for the original should hold at roughly
  the rate it shows everywhere else. The classifier's mistake is not its mistake.

This is a strictly stronger test than measuring agreement, and it needs no human labels, no
new generations and no GPU. It cannot prove the panel reads well — a panel keying on some
*other* surface regularity this feature set does not capture would pass. It can only rule
out the specific alternative that the measured 66% explains the measured 80%, which is the
alternative currently standing between the project and its one quality claim.

Verdicts are filtered to order-consistent ones first (`findings-phase2.md` §10). Verdicts
that flip when the options are swapped are about position, and asking whether a positional
artifact tracks surface features is not a question with an answer.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.judge import JudgeVerdict, order_consistent  # noqa: E402
from revisionbench.records import read_jsonl  # noqa: E402
from scripts.self_preference import extract_edits  # noqa: E402
from scripts.surface_predictability import features, informative, out_of_fold_margins  # noqa: E402


def bootstrap_rate(
    hits: list[bool], *, draws: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Rate with a percentile CI. Returns (rate, low, high); NaNs if the subset is empty."""
    if not hits:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    values = np.array(hits, dtype=float)
    resampled = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    return float(values.mean()), *np.percentile(resampled, [2.5, 97.5])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verdicts", type=Path, default=REPO_ROOT / "results" / "phase2" / "self_preference.jsonl"
    )
    parser.add_argument(
        "--rounds", type=Path, default=REPO_ROOT / "results" / "phase0" / "rounds.jsonl"
    )
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    # Rebuild the exact sample self_preference.py judged, so pair ids line up. Same seed,
    # same order, same call sequence — any divergence silently misaligns every verdict with
    # a different pair's margin, and the result would still look like a number.
    pools = extract_edits(args.rounds)
    rng = random.Random(args.seed)
    edits: dict[str, dict[str, Any]] = {}
    for reviser in ("phi4:latest", "gemma3:4b"):
        if reviser not in pools:
            continue
        for index, edit in enumerate(rng.sample(pools[reviser], 60)):
            edits[f"{reviser.replace(':', '_')}-{index:03d}"] = edit

    pair_ids = list(edits)
    names = sorted(features(edits[pair_ids[0]]["original"]))
    rows = []
    for pair_id in pair_ids:
        edit = features(edits[pair_id]["replacement"])
        original = features(edits[pair_id]["original"])
        rows.append([edit[k] - original[k] for k in names])
    deltas, kept = informative(np.array(rows), names)

    passages = sorted({edits[p]["passage_id"] for p in pair_ids})
    groups = np.array([passages.index(edits[p]["passage_id"]) for p in pair_ids])
    margins = out_of_fold_margins(deltas, folds=args.folds, seed=args.seed, groups=groups)

    # version_2 is the model's edit for every pair by construction, so a positive margin is
    # the classifier getting it right.
    surface_right = {pair_id: bool(margins[i] > 0) for i, pair_id in enumerate(pair_ids)}
    accuracy = sum(surface_right.values()) / len(pair_ids)

    fields = set(JudgeVerdict.__slots__)
    forward: dict[str, list[JudgeVerdict]] = defaultdict(list)
    backward: dict[str, list[JudgeVerdict]] = defaultdict(list)
    chose: dict[tuple[str, str], str | None] = {}
    for row in read_jsonl(args.verdicts):
        verdict = JudgeVerdict(**{k: v for k, v in row.items() if k in fields})
        (backward if row.get("order") == "reversed" else forward)[verdict.judge].append(verdict)
        if row.get("order") != "reversed":
            chose[(verdict.judge, row["pair_id"])] = row.get("chose")

    # Guard the reconstruction. If the sampling above drifted from self_preference.py, every
    # verdict would be paired with a different pair's margin and the output would still be a
    # plausible-looking table. An id in the verdicts that this run cannot produce is the
    # cheapest possible detector for that.
    judged_ids = {pair_id for _, pair_id in chose}
    missing = judged_ids - set(pair_ids)
    if missing:
        print(
            f"error: {len(missing)} judged pair ids are not reproducible from {args.rounds}, "
            f"e.g. {sorted(missing)[:3]}. The sample in this script has drifted from "
            "scripts/self_preference.py; margins would be matched to the wrong pairs.",
            file=sys.stderr,
        )
        return 2

    print(f"{len(pair_ids)} pairs judged by the panel; {len(kept)} surface features")
    print(f"surface classifier identifies the edit on {accuracy:.1%} (null 50%)\n")
    print(
        f"  {'judge':<16}{'consistent':>11}{'prefers original':>18}"
        f"{'| where surface RIGHT':>23}{'where surface WRONG':>22}"
    )

    pooled: dict[str, list[bool]] = {"right": [], "wrong": []}
    for judge in sorted(forward):
        consistent = set(order_consistent(forward[judge], backward.get(judge, [])))
        subsets: dict[str, list[bool]] = {"right": [], "wrong": [], "all": []}
        for pair_id in consistent:
            picked = chose.get((judge, pair_id))
            if picked is None:  # abstention carries no direction
                continue
            prefers_original = picked == "version_1"
            subsets["all"].append(prefers_original)
            bucket = "right" if surface_right.get(pair_id, False) else "wrong"
            subsets[bucket].append(prefers_original)
            pooled[bucket].append(prefers_original)

        overall, lo, hi = bootstrap_rate(subsets["all"], seed=args.seed)
        right, r_lo, r_hi = bootstrap_rate(subsets["right"], seed=args.seed)
        wrong, w_lo, w_hi = bootstrap_rate(subsets["wrong"], seed=args.seed)
        print(
            f"  {judge:<16}{len(consistent):>11}"
            f"{overall:>13.0%} [{lo:.0%},{hi:.0%}]"
            f"{right:>13.0%} [{r_lo:.0%},{r_hi:.0%}] n={len(subsets['right']):<3}"
            f"{wrong:>7.0%} [{w_lo:.0%},{w_hi:.0%}] n={len(subsets['wrong'])}"
        )

    right, r_lo, r_hi = bootstrap_rate(pooled["right"], seed=args.seed)
    wrong, w_lo, w_hi = bootstrap_rate(pooled["wrong"], seed=args.seed)
    n_right, n_wrong = len(pooled["right"]), len(pooled["wrong"])
    print(
        f"\n  pooled across judges:"
        f"\n    where surface is RIGHT   {right:.0%}  [{r_lo:.0%}, {r_hi:.0%}]  n={n_right}"
        f"\n    where surface is WRONG   {wrong:.0%}  [{w_lo:.0%}, {w_hi:.0%}]  n={n_wrong}"
        f"\n    gap                      {right - wrong:+.0%}"
    )
    print(
        "\nReading it. A gap near zero means the panel's preference does not move with the\n"
        "surface evidence, and the 66% classifier does not explain the 80% preference. A\n"
        "large gap — especially 'wrong' falling below 50% — means the panel follows surface\n"
        "features into error, and the quality signal in findings-phase2 §10 is that reflex."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
