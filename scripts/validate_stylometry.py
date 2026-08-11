"""Does the stylometry tell these authors apart? (plan.md §12, assumption 2)

    uv run python scripts/validate_stylometry.py

plan.md §12 lists as an *unverified assumption* that "writeprint-feature stylometry is
sensitive enough at 500-1500 words to serve as a veto", and says to validate it on
held-out human passages by same-author vs cross-author separation. This script is that
validation, and it must run before the A4 voice veto (plan.md §8) is designed, because
A4's whole premise is that a stylometric distance can tell "this edit changed the voice"
from "this edit did not".

METHOD

For every pair of Phase-0 passages, compute a distance and label the pair same-author or
different-author. The readout is

    AUC = P(a different-author pair is more distant than a same-author pair)

with ties counted as a half. 1.0 is perfect separation and 0.5 is chance. Significance
comes from a **permutation test on the author labels**: shuffle which passage belongs to
which author, recompute the AUC, and ask how often chance beats the observed value. A
permutation test is used rather than a bootstrap because with 10 passages a resample
draws the same passage twice routinely, and a passage paired with itself is a
same-author distance of exactly 0 that inflates the AUC for free.

WHAT THIS CANNOT TELL YOU

Ten passages and three authors. The point estimates below are wide and the corpus is a
convenience sample of what is on Project Gutenberg, so read the *ordering* of the feature
families, not the third decimal place of any one number. It also measures separation
between different authors, which is a proxy for what A4 actually needs — separation
between a passage and a revised version of itself. Those are related but not the same
question, and the second one cannot be answered until there are revision rounds to
measure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.metrics.stats import mean  # noqa: E402
from revisionbench.metrics.stylometry import StyleModel  # noqa: E402
from revisionbench.provenance import RunProvenance, utc_now  # noqa: E402
from revisionbench.records import write_json  # noqa: E402

FAMILIES = ("fw", "punct", "sent", "all", "balanced")


def auc(within: list[float], cross: list[float]) -> float:
    """P(cross-author distance > same-author distance), ties at 0.5."""
    if not within or not cross:
        raise ValueError("AUC needs at least one pair of each kind")
    wins = sum((c > w) + 0.5 * (c == w) for w in within for c in cross)
    return wins / (len(within) * len(cross))


def split_by_label(distances: list[tuple[int, int, float]], authors: list[str]):
    """Partition precomputed pair distances using a (possibly permuted) author labelling."""
    within: list[float] = []
    cross: list[float] = []
    for i, j, distance in distances:
        (within if authors[i] == authors[j] else cross).append(distance)
    return within, cross


def pair_distances(
    model: StyleModel, texts: list[str], family: str
) -> list[tuple[int, int, float]]:
    out = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if family == "balanced":
                distance = model.family_balanced_delta(texts[i], texts[j])
            elif family == "all":
                distance = model.delta(texts[i], texts[j])
            else:
                distance = model.delta(texts[i], texts[j], family=family)
            out.append((i, j, distance))
    return out


def permutation_p(
    distances: list[tuple[int, int, float]],
    authors: list[str],
    observed: float,
    *,
    n_permutations: int,
    seed: int,
) -> float:
    """One-sided p: how often a shuffled labelling reaches the observed AUC.

    The +1 in both numerator and denominator is the standard correction that keeps a
    p-value of exactly 0 — which would be a claim no permutation test can support — off
    the table.
    """
    rng = random.Random(seed)
    shuffled = list(authors)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(shuffled)
        within, cross = split_by_label(distances, shuffled)
        if not within or not cross:
            continue
        if auc(within, cross) >= observed:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_permutations + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase0")
    parser.add_argument("--n-function-words", type=int, nargs="+", default=[20, 50, 100, 150])
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    records = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.corpus.glob("*.json"))
    ]
    if len(records) < 4:
        print(
            f"error: need at least 4 passages, found {len(records)} in {args.corpus}",
            file=sys.stderr,
        )
        return 2
    texts = [r["text"] for r in records]
    authors = [r["author_id"] for r in records]

    n_within = sum(
        1
        for i in range(len(authors))
        for j in range(i + 1, len(authors))
        if authors[i] == authors[j]
    )
    n_cross = len(authors) * (len(authors) - 1) // 2 - n_within

    print(
        f"passages {len(records)}  authors {len(set(authors))}  "
        f"pairs: {n_within} same-author, {n_cross} different-author"
    )
    print(f"permutations {args.permutations}  seed {args.seed}\n")
    print(f"{'n_fw':>5}  {'family':<10} {'same':>7} {'diff':>7} {'AUC':>6} {'p':>7}")

    rows = []
    for n_fw in args.n_function_words:
        model = StyleModel.fit(texts, n_function_words=n_fw)
        for family in FAMILIES:
            distances = pair_distances(model, texts, family)
            within, cross = split_by_label(distances, authors)
            observed = auc(within, cross)
            p_value = permutation_p(
                distances,
                authors,
                observed,
                n_permutations=args.permutations,
                seed=args.seed,
            )
            rows.append(
                {
                    "n_function_words": n_fw,
                    "family": family,
                    "mean_same_author": mean(within),
                    "mean_diff_author": mean(cross),
                    "auc": observed,
                    "permutation_p": p_value,
                    "n_features": len(model.features),
                    "n_features_in_family": (
                        len(model.features)
                        if family in ("all", "balanced")
                        else sum(1 for f in model.features if f.startswith(f"{family}:"))
                    ),
                }
            )
            print(
                f"{n_fw:>5}  {family:<10} {mean(within):>7.3f} {mean(cross):>7.3f} "
                f"{observed:>6.3f} {p_value:>7.4f}"
            )
        print()

    best = max(rows, key=lambda r: r["auc"])
    print(
        f"best: {best['family']} at n_fw={best['n_function_words']} "
        f"-> AUC {best['auc']:.3f} (p={best['permutation_p']:.4f})"
    )
    print("\nRead the ordering of families, not the third decimal of any one cell:")
    print("10 passages and 3 authors is a small sample, and every cell above is wide.")

    # Deterministic half: committed, and byte-reproducible from the same corpus and seed.
    # The volatile half (wall-clock, hostname, absolute paths) goes to results/provenance/,
    # which is gitignored -- same split as fetch_corpus.py and phase0.py. Embedding a
    # timestamp and a machine name in a published result file makes it undiffable and
    # publishes the machine's name for no measurement benefit.
    payload = {
        "inputs": {
            "n_passages": len(records),
            "n_authors": len(set(authors)),
            "permutations": args.permutations,
            "seed": args.seed,
            "n_function_words": args.n_function_words,
        },
        "pairs": {"same_author": n_within, "different_author": n_cross},
        "rows": rows,
    }
    out_path = args.out / "stylometry_validation.json"
    write_json(out_path, payload)

    provenance = (
        RunProvenance(run_id="validate-stylometry", started_at=utc_now(), config_hash="n/a")
        .with_artifacts(
            corpus_dir=str(args.corpus.as_posix()),
            n_passages=len(records),
            permutations=args.permutations,
            seed=args.seed,
        )
        .as_dict()
    )
    write_json(REPO_ROOT / "results" / "provenance" / "validate-stylometry.json", provenance)
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
