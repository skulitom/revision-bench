"""Tests for the surface-predictability classifier.

Every bug this file guards against produced a *confident wrong number* rather than a crash,
which is the failure mode that matters for an analysis script. The first run of
`surface_predictability.py` reported single-feature rules at 100% accuracy (they were
features that never varied, scored as ``max(0, 1-0)``), weights of ±1800 (near-constant
features divided by a near-zero spread), and a 95% CI of [57.6%, 78.3%] whose upper tail was
entirely bootstrap leakage. None of those would have failed a smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.surface_predictability import (
    cross_validated_accuracy,
    features,
    informative,
    single_feature_rules,
)


def test_null_holds_on_noise() -> None:
    """A classifier with nothing to learn must land at chance, not above it.

    This is the test that makes every other number in the script interpretable. If the
    encoding were asymmetric — only ``f(edit) - f(original)`` rows, all labelled 1 — this
    would score 100% by predicting the constant, and the reported 66% would mean nothing.
    """
    rng = np.random.default_rng(0)
    deltas = rng.normal(size=(400, 8))
    accuracy, _ = cross_validated_accuracy(deltas, folds=8, seed=0)
    assert 0.40 < accuracy < 0.60, accuracy


def test_a_real_signal_is_found() -> None:
    """The converse: a perfectly aligned feature must be picked up."""
    rng = np.random.default_rng(1)
    deltas = rng.normal(size=(200, 4))
    deltas[:, 2] = np.abs(deltas[:, 2]) + 0.5  # always positive on the +delta direction
    accuracy, weights = cross_validated_accuracy(deltas, folds=5, seed=0)
    assert accuracy > 0.95, accuracy
    assert np.argmax(np.abs(weights)) == 2


def test_direction_is_symmetric() -> None:
    """Negating every pair must not change the score.

    "Which of these two is the edit" has no privileged side, so the fit is through the
    origin. A stray intercept would break this and would show up as accuracy drifting with
    whichever direction happened to be encoded first.
    """
    rng = np.random.default_rng(2)
    deltas = rng.normal(size=(200, 5))
    deltas[:, 0] += 0.4
    forward, _ = cross_validated_accuracy(deltas, folds=5, seed=0)
    backward, _ = cross_validated_accuracy(-deltas, folds=5, seed=0)
    assert forward == pytest.approx(backward, abs=0.02)


def test_grouping_keeps_duplicates_out_of_the_test_fold() -> None:
    """Rows sharing a group must never be split across the train/test boundary.

    This is the bootstrap-leakage fix. Constructed so leakage is unmistakable: each group
    is a distinct random direction repeated four times, and the label is learnable only by
    memorising the direction. Ungrouped, copies of a test row sit in training and the score
    inflates; grouped, it cannot.
    """
    rng = np.random.default_rng(3)
    directions = rng.normal(size=(12, 6)) * 3.0
    deltas = np.repeat(directions, 4, axis=0)
    groups = np.repeat(np.arange(12), 4)

    grouped, _ = cross_validated_accuracy(deltas, folds=6, seed=0, groups=groups)
    ungrouped, _ = cross_validated_accuracy(deltas, folds=6, seed=0)
    assert ungrouped >= grouped, (ungrouped, grouped)


def test_informative_drops_near_constant_and_duplicate_features() -> None:
    rng = np.random.default_rng(4)
    deltas = np.column_stack(
        [
            rng.normal(size=50),  # 0: varies
            np.zeros(50),  # 1: never varies -> dropped
            np.concatenate([[1.0, -1.0], np.zeros(48)]),  # 2: varies twice -> dropped
            rng.normal(size=50),  # 3: varies
        ]
    )
    deltas = np.column_stack([deltas, deltas[:, 0]])  # 4: exact duplicate of 0 -> dropped
    kept, names = informative(deltas, ["a", "b", "c", "d", "a_copy"])
    assert names == ["a", "d"]
    assert kept.shape == (50, 2)


def test_single_feature_rules_ignore_features_that_never_vary() -> None:
    """The 100% bug: a feature equal on both sides is not a perfect rule, it is no rule."""
    deltas = np.column_stack(
        [
            np.zeros(40),  # identical on both sides in every pair
            np.concatenate([np.ones(30), -np.ones(10)]),  # a real 75% rule
        ]
    )
    rules = single_feature_rules(deltas, ["never_varies", "real_rule"])
    assert [name for name, _, _ in rules] == ["real_rule"]
    _, accuracy, coverage = rules[0]
    assert accuracy == pytest.approx(0.75)
    assert coverage == 40


def test_features_are_length_normalised_where_they_should_be() -> None:
    """Punctuation is per-word, so a longer sentence is not automatically punctuation-rich.

    Without this the classifier would rediscover length through every punctuation channel,
    and "the edit is shorter" would be smeared across a dozen features instead of sitting
    in `words` where it can be seen and reasoned about.
    """
    short = "He went home, tired."
    long = short + " " + " ".join(["and walked on"] * 20)
    assert features(short)["punct_comma"] > features(long)["punct_comma"]
    assert features(long)["words"] > features(short)["words"]
