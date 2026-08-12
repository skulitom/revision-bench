"""How much of "the judge prefers the original" needs no aesthetic explanation?

    uv run python scripts/surface_predictability.py

`findings-phase2.md` §10 reports the project's first quality signal: on order-consistent
verdicts, all three judges preferred the human original about 80% of the time. That number
has two competing explanations, and they imply opposite things about the panel.

1. **The judges read the prose and the original is better.** The signal is real.
2. **The model's edit carries a surface regularity, and the judges key on it.** Then 80%
   measures a typographic reflex, and the panel has told us nothing about quality.

These are distinguishable without asking anyone. Fit a classifier that sees *only* surface
features — counts of words, commas, dashes, quotes, subordinators; no semantics, no model —
and ask how well it separates the model's edit from the original on the same 92 pairs. If a
feature-count model reaches ~80%, explanation 2 accounts for the whole signal on its own and
the panel's agreement is not evidence of anything. If it sits near chance, explanation 2 is
ruled out and the panel is keying on something a bag of counts cannot see.

This is a lower bound on explanation 2, not a proof of explanation 1: a classifier that
fails does not establish that the judges were reading well, only that they were not reading
*these* features. It rules a hypothesis out, which is the most a null result can do.

**Design notes, because the naive version of this is a lie.**

*Symmetric pairs, no intercept.* Each pair contributes two rows — ``f(edit) - f(original)``
labelled 1, and its negation labelled 0. Feeding only the first direction makes every label
identical and any classifier scores 100% by predicting the constant. The symmetric encoding
also forces the decision boundary through the origin, which is correct here: with no
evidence, "which of these two is the edit" must be a coin flip.

*Cross-validated, folded by pair.* 92 pairs and 13 features overfit comfortably. Scores are
out-of-fold, and both rows of a pair go into the same fold — split them and the model sees
the negation of a test row during training, which is leakage that reads as skill.

*Raw text, not normalised.* RevisionJudge normalises punctuation before showing a person
(``findings-phase2.md`` §11), but the panel judged the raw strings. Raw is therefore the
number that speaks to the panel's 80%. Both are reported, and the gap between them is a
direct measure of how much of the separability was typographic.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.text import punctuation_counts, sentences, words  # noqa: E402

#: Subordinators and connectives. A rewrite that flattens subordination is a plausible and
#: purely structural tell, independent of whether the flattening reads better.
SUBORDINATORS = frozenset(
    [
        "which",
        "that",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "while",
        "because",
        "although",
        "though",
        "since",
        "unless",
        "until",
        "whether",
        "if",
        "as",
        "after",
        "before",
    ]
)


def features(text: str) -> dict[str, float]:
    """Surface counts only. Nothing here knows what the sentence means."""
    toks = words(text)
    sents = sentences(text)
    punct = punctuation_counts(text)
    n = max(len(toks), 1)
    return {
        "words": float(len(toks)),
        "chars_per_word": sum(len(w) for w in toks) / n,
        "sentences": float(len(sents)),
        "words_per_sentence": len(toks) / max(len(sents), 1),
        "type_token_ratio": len(set(toks)) / n,
        "long_word_rate": sum(1 for w in toks if len(w) >= 7) / n,
        "subordinator_rate": sum(1 for w in toks if w in SUBORDINATORS) / n,
        **{f"punct_{k}": v / n for k, v in punct.items()},
    }


#: A feature that separates the two sides on fewer than this many pairs cannot support a
#: rule, and standardising it divides by a near-zero spread — which is where the first run
#: of this script produced weights of ±1800 on parenthesis counts that differ on two pairs.
MIN_VARYING_PAIRS = 5


def informative(deltas: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop features that barely vary, and collapse exact duplicates.

    Open and close parentheses are perfectly collinear in this corpus — every open has a
    close — so the pair is unidentifiable and the fit splits an arbitrary weight between
    them. Keeping one is not a loss of information.
    """
    keep = [
        i for i in range(deltas.shape[1]) if int((deltas[:, i] != 0).sum()) >= MIN_VARYING_PAIRS
    ]
    seen: dict[bytes, int] = {}
    unique = []
    for i in keep:
        signature = deltas[:, i].tobytes()
        if signature not in seen:
            seen[signature] = i
            unique.append(i)
    return deltas[:, unique], [names[i] for i in unique]


def _fit(x: np.ndarray, y: np.ndarray, *, steps: int = 600, lr: float = 1.0) -> np.ndarray:
    """Logistic regression through the origin, L2-penalised, plain gradient descent.

    No intercept by design — see the module docstring. The penalty matters at this n; the
    strength is fixed rather than tuned, because tuning it on 92 pairs and then reporting
    the best score is the same overfitting one loop further out.
    """
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-x @ w))
        w -= lr * ((x.T @ (p - y)) / len(y) + 0.05 * w)
    return w


def cross_validated_accuracy(
    deltas: np.ndarray, *, folds: int = 8, seed: int = 0, groups: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Out-of-fold accuracy, with both rows of a pair kept in the same fold.

    ``groups`` identifies which original pair each row came from. It matters only under
    bootstrap: resampling pairs with replacement puts copies of the same pair in the array,
    and if fold assignment ignores that, a pair trains and tests in different folds. That is
    leakage, and it shows up as a CI skewed upwards — the first run of this script reported
    63.0% with a 95% interval of [57.6%, 78.3%], where the whole upper tail was the model
    scoring itself on rows it had already seen. Folding by group removes it.
    """
    if groups is None:
        groups = np.arange(len(deltas))
    rng = random.Random(seed)
    unique = sorted({int(g) for g in groups})
    rng.shuffle(unique)
    rows_of = {g: np.flatnonzero(groups == g) for g in unique}

    correct = 0
    weights = []
    for k in range(folds):
        test_groups = unique[k::folds]
        held_out = set(test_groups)
        test_pairs = np.concatenate([rows_of[g] for g in test_groups]) if test_groups else []
        train_groups = [g for g in unique if g not in held_out]
        train_pairs = np.concatenate([rows_of[g] for g in train_groups]) if train_groups else []
        if not len(train_pairs) or not len(test_pairs):
            continue

        # Symmetric expansion: every pair appears as +delta (label 1) and -delta (label 0).
        train_x = np.vstack([deltas[train_pairs], -deltas[train_pairs]])
        train_y = np.concatenate([np.ones(len(train_pairs)), np.zeros(len(train_pairs))])

        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale == 0] = 1.0
        w = _fit((train_x - mean) / scale, train_y)
        weights.append(w / scale)

        # Score the +delta direction only: predicting >0.5 means "the edit is side A",
        # which is the right answer for every test row by construction.
        test_x = (deltas[test_pairs] - mean) / scale
        correct += int(((test_x @ w) > 0).sum())

    return correct / len(deltas), np.mean(weights, axis=0)


def single_feature_rules(deltas: np.ndarray, names: list[str]) -> list[tuple[str, float, int]]:
    """Accuracy of "the side with more X is the edit", among pairs where X differs.

    Two things this must not do, both of which the first version of this script did. It
    must not score a feature over pairs where the two sides are equal — a feature that never
    varies then reads as 100% by unanimously predicting nothing. And it must report the
    coverage alongside the rate, because a rule that is right every time on 3 pairs is not a
    rule. The *sign* is chosen post hoc, so these are "how learnable is this feature in
    principle", not out-of-sample estimates; the cross-validated number is that.
    """
    out = []
    for i, name in enumerate(names):
        varying = deltas[:, i][deltas[:, i] != 0]
        if len(varying) < MIN_VARYING_PAIRS:
            continue
        agree = float((varying > 0).mean())
        out.append((name, max(agree, 1.0 - agree), len(varying)))
    return sorted(out, key=lambda t: -t[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, default=Path(r"C:\DEV\RevisionJudge\data\pairs.json"))
    parser.add_argument(
        "--all-edits",
        action="store_true",
        help="fit on every edit in results/phase0/rounds.jsonl (~1385) instead of the 92 "
        "sampled for human judging. The 92 are the pairs the panel actually judged, so they "
        "are the right set for explaining the panel's 80%%; the full pool is 15x the data "
        "and is the right set for asking whether the separation exists at all.",
    )
    parser.add_argument(
        "--rounds", type=Path, default=REPO_ROOT / "results" / "phase0" / "rounds.jsonl"
    )
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)

    if not args.pairs.is_file():
        print(f"error: {args.pairs} not found; run scripts/export_human_pairs.py", file=sys.stderr)
        return 2

    if args.all_edits:
        from scripts.self_preference import extract_edits

        pools = extract_edits(args.rounds)
        subjective = [
            {
                "version_1": edit["original"],
                "version_2": edit["replacement"],
                "passage_id": edit["passage_id"],
                "_key": {"version_1": "original"},
            }
            for pool in pools.values()
            for edit in pool
        ]
        source = (
            f"{args.rounds} (full pool, {', '.join(f'{k}={len(v)}' for k, v in pools.items())})"
        )
    else:
        pairs = json.loads(args.pairs.read_text("utf-8"))["pairs"]
        subjective = [p for p in pairs if p["_key"]["version_1"] == "original"]
        source = f"{args.pairs} (the pairs the panel judged)"
    if not subjective:
        print("error: no original-vs-edit pairs", file=sys.stderr)
        return 2

    # Fold by passage, not by edit. Edits from one passage share an author, a register, and
    # a source vocabulary; splitting them across folds lets the model learn a passage in
    # training and be scored on the same passage's other edits, which is leakage of exactly
    # the kind the grouped bootstrap above exists to prevent.
    def passage_of(pair: dict) -> str:
        return pair.get("passage_id") or pair.get("_key", {}).get("passage_id", "")

    passages = sorted({passage_of(p) for p in subjective})
    group_ids = np.array([passages.index(passage_of(p)) for p in subjective])
    if len(passages) < 2:
        group_ids = np.arange(len(subjective))

    # The app's render-time normalisation, imported rather than reimplemented so the two
    # cannot drift apart and quietly make this comparison meaningless.
    judge_app = args.pairs.resolve().parent.parent
    sys.path.insert(0, str(judge_app))
    try:
        from serve import normalise  # type: ignore[import-not-found]
    except ImportError:
        print(f"warning: no serve.py under {judge_app}; raw only", file=sys.stderr)
        normalise = None  # type: ignore[assignment]

    names = sorted(features(subjective[0]["version_1"]))
    print(f"{len(subjective)} original-vs-edit pairs from {source}")
    print(f"{len(names)} surface features, folded by passage ({len(passages)} passages)")
    print("null = 50.0% (which of two sides is the model's edit)\n")

    rng = np.random.default_rng(args.seed)
    for label, transform in (
        ("raw (what the panel judged)", None),
        ("normalised (what a person is shown)", normalise),
    ):
        if transform is None and label.startswith("normalised"):
            continue
        fn = transform or (lambda t: t)
        # Extract once per side, not once per feature — the first version called features()
        # inside the inner loop and paid for it 23 times over.
        rows = []
        for pair in subjective:
            edit = features(fn(pair["version_2"]))
            original = features(fn(pair["version_1"]))
            rows.append([edit[k] - original[k] for k in names])
        deltas, kept = informative(np.array(rows), names)
        accuracy, weights = cross_validated_accuracy(
            deltas, folds=args.folds, seed=args.seed, groups=group_ids
        )

        # Bootstrap over pairs, refitting each time so the interval includes uncertainty in
        # the fit itself — and carrying the original pair index as the grouping key, so a
        # resampled duplicate cannot train and test in different folds.
        boot = np.array(
            [
                cross_validated_accuracy(
                    deltas[(pick := rng.integers(0, len(deltas), len(deltas)))],
                    folds=args.folds,
                    seed=args.seed,
                    groups=group_ids[pick],
                )[0]
                for _ in range(min(args.bootstrap, 200))
            ]
        )
        low, high = np.percentile(boot, [2.5, 97.5])

        print(f"  {label}  ({len(kept)} of {len(names)} features vary enough to use)")
        print(f"    cross-validated accuracy  {accuracy:>6.1%}   95% CI [{low:.1%}, {high:.1%}]")
        top = sorted(zip(kept, weights, strict=True), key=lambda t: -abs(t[1]))[:5]
        print("    strongest weights:", ", ".join(f"{n}({w:+.2f})" for n, w in top))
        print(f"    {'single-feature rule':<28}{'accuracy':>10}{'decisive on':>13}")
        for name, rule, coverage in single_feature_rules(deltas, kept)[:5]:
            print(f"      {name:<26}{rule:>10.1%}{coverage:>10} pairs")
        print()

    print(
        "Read against findings-phase2.md §10: the panel preferred the original on ~80% of\n"
        "order-consistent verdicts. If the number above approaches 80%, that agreement needs\n"
        "no aesthetic explanation. If it sits near 50%, surface features are ruled out as the\n"
        "explanation — which does not prove the judges read well, only that whatever they\n"
        "keyed on is not a bag of counts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
