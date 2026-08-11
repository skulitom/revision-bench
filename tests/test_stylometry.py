"""M1/M2 stylometry.

The headline test is :class:`TestAuthorSeparation`, which is milestone M0-b's stated
acceptance criterion ("verify Delta separates two known authors"). It runs against the
committed Phase-0 corpus rather than synthetic text, because the question is specifically
whether the measurement works at real passage length on real prose — plan.md §12.2 lists
that as an *unverified assumption*, and a synthetic fixture would answer a different and
easier question.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from revisionbench.metrics.stylometry import (
    MIN_WORDS_FOR_MTLD,
    MetricError,
    StyleModel,
    burrows_delta,
    cluster_purity,
    describe,
    feature_vector,
    load_function_words,
    mean_pairwise_delta,
    mtld,
    punctuation_profile,
    sentence_length_stats,
    wasserstein1,
)

PROSE = (
    "The morning was bright and the wind came off the water in short gusts. "
    "She walked down the lane past the low wall, counting the gates as she went. "
    "There were seven of them, and the seventh was open, which surprised her. "
    "Beyond it the field ran away downhill toward a line of poplars, grey in the haze. "
    "Someone had been burning stubble; the smell hung about the hedges and would not lift. "
    "She thought of the letter in her pocket and did not take it out. "
    "It could wait until she reached the bridge, or until the afternoon, or until never. "
    "The gulls went over in twos and threes, calling, and the light moved on the grass. "
    "A dog barked somewhere behind the barns and then stopped, as if thinking better of it. "
    "She had come this way every day for a month and had noticed none of it before now. "
    "The gate swung a little in the wind and knocked against its post, twice, then again. "
    "She put her hand on the top bar, which was wet, and stood there for a long while. "
)


@pytest.fixture(scope="module")
def model(passage_texts: list[str]) -> StyleModel:
    return StyleModel.fit(passage_texts, n_function_words=100)


class TestFunctionWordList:
    def test_loads(self) -> None:
        version, wordlist = load_function_words()
        assert version >= 1
        assert len(wordlist) > 150
        assert all(isinstance(w, str) for w in wordlist)

    def test_yaml_boolean_trap_words_survived(self) -> None:
        """`no`, `yes`, `on` and `off` are YAML 1.1 booleans.

        Unquoted, they silently leave the list and are replaced by the tokens "true" and
        "false", which occur in no passage -- so Delta would be computed over a feature
        set missing four of the commonest words in English, with no error raised.
        """
        _, wordlist = load_function_words()
        for word in ("no", "yes", "on", "off"):
            assert word in wordlist
        assert "true" not in wordlist and "false" not in wordlist

    def test_non_string_entry_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "fw.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "words": ["the", True]}), encoding="utf-8")
        with pytest.raises(MetricError, match=r"YAML 1\.1 boolean trap"):
            load_function_words(path)

    def test_duplicates_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "fw.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "words": ["the", "the"]}), encoding="utf-8")
        with pytest.raises(MetricError, match="duplicate"):
            load_function_words(path)


class TestMtld:
    def test_repetitive_text_scores_below_varied_text(self) -> None:
        repetitive = ("the cat sat on the mat and the cat sat on the mat " * 12).strip()
        assert mtld(repetitive) < mtld(PROSE)

    def test_short_text_raises(self) -> None:
        with pytest.raises(MetricError, match=f"at least {MIN_WORDS_FOR_MTLD} tokens"):
            mtld("Too short to be stable.")

    def test_all_distinct_tokens_raises_rather_than_returning_inf(self) -> None:
        """The naive formula divides by zero here; `inf` in a mean is unrecoverable.

        Tokens must be alphabetic and genuinely distinct: "word0"/"word1" would tokenise
        as "word"+"0"/"word"+"1", repeating "word" and completing a factor after all.
        """
        import string

        alphabet = string.ascii_lowercase
        distinct = [a + b for a in alphabet for b in alphabet][: MIN_WORDS_FOR_MTLD + 10]
        assert len(set(distinct)) == len(distinct)
        with pytest.raises(MetricError, match="undefined"):
            mtld(" ".join(distinct))

    def test_threshold_must_be_a_proportion(self) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            mtld(PROSE, threshold=1.0)

    def test_is_not_merely_a_length_proxy(self, passage_texts: list[str]) -> None:
        """MTLD's whole point is length-independence; a raw TTR would fail this."""
        text = passage_texts[0]
        half = text[: len(text) // 2]
        assert mtld(half) == pytest.approx(mtld(text), rel=0.45)


class TestSentenceStats:
    def test_basic_shape(self) -> None:
        stats = sentence_length_stats("One two three. Four five. Six seven eight nine ten.")
        assert stats["sent:mean"] == pytest.approx(10 / 3)
        assert stats["sent:median"] == 3.0
        assert stats["sent:count_per_1000w"] == pytest.approx(300.0)

    def test_single_sentence_raises(self) -> None:
        with pytest.raises(MetricError, match="at least 2 sentences"):
            sentence_length_stats("Only one sentence here.")

    def test_splitting_sentences_moves_the_density_metric(self) -> None:
        """The commonest LLM 'improvement' to literary prose, and it must be visible."""
        long_form = "She walked to the door and opened it, and the wind came in cold."
        split_form = "She walked to the door. She opened it. The wind came in cold."
        before = sentence_length_stats(long_form + " " + long_form)
        after = sentence_length_stats(split_form + " " + split_form)
        assert after["sent:count_per_1000w"] > before["sent:count_per_1000w"] * 2
        assert after["sent:mean"] < before["sent:mean"]


class TestPunctuationProfile:
    def test_is_a_rate_not_a_count(self) -> None:
        once = "Yes, and so, and then, we went."
        twice = once + " " + once
        assert punctuation_profile(once) == pytest.approx(punctuation_profile(twice))

    def test_empty_text_raises(self) -> None:
        with pytest.raises(MetricError, match="no words"):
            punctuation_profile("...")


class TestWasserstein:
    def test_known_values(self) -> None:
        assert wasserstein1([0.0], [1.0]) == pytest.approx(1.0)
        assert wasserstein1([0.0, 2.0], [1.0, 1.0]) == pytest.approx(1.0)
        assert wasserstein1([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_sees_a_flattening_the_mean_cannot(self) -> None:
        """Same mean, very different rhythm -- the case that motivates this metric."""
        varied = [5.0, 60.0, 5.0, 60.0]
        uniform = [32.5, 32.5, 32.5, 32.5]
        assert sum(varied) / 4 == sum(uniform) / 4
        assert wasserstein1(varied, uniform) > 20.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            wasserstein1([], [1.0])


class TestStyleModel:
    def test_fit_requires_two_texts(self) -> None:
        with pytest.raises(MetricError, match="at least 2 reference texts"):
            StyleModel.fit([PROSE])

    def test_zero_variance_features_are_dropped_not_divided_by(self, model: StyleModel) -> None:
        assert model.dropped_low_variance, "the Phase-0 corpus has unused punctuation classes"
        for name in model.dropped_low_variance:
            assert name not in model.features
        assert all(sd > model.sd_floor for sd in model.sds.values())

    def test_identical_texts_have_zero_delta(self, model: StyleModel, passage_texts) -> None:
        assert model.delta(passage_texts[0], passage_texts[0]) == pytest.approx(0.0)

    def test_delta_is_symmetric(self, model: StyleModel, passage_texts) -> None:
        a, b = passage_texts[0], passage_texts[5]
        assert model.delta(a, b) == pytest.approx(model.delta(b, a))

    def test_feature_families_are_all_present(self, model: StyleModel) -> None:
        prefixes = {name.split(":")[0] for name in model.features}
        assert prefixes == {"fw", "punct", "sent", "lex"}

    def test_selection_is_order_independent(self, passage_texts: list[str]) -> None:
        """Feature selection must not depend on how texts happen to be passed in."""
        a = StyleModel.fit(passage_texts, n_function_words=40)
        b = StyleModel.fit(list(reversed(passage_texts)), n_function_words=40)
        assert a.function_words == b.function_words
        assert a.features == b.features

    def test_absent_function_word_is_a_measured_zero(self) -> None:
        vector = feature_vector(PROSE, ["the", "zzzznotaword"])
        assert vector["fw:zzzznotaword"] == 0.0

    def test_serialisation_roundtrip(self, model: StyleModel, passage_texts) -> None:
        """A result file must be able to record the exact scaler that produced it."""
        restored = StyleModel.from_dict(model.to_dict())
        assert restored.features == model.features
        assert restored.delta(passage_texts[0], passage_texts[1]) == pytest.approx(
            model.delta(passage_texts[0], passage_texts[1])
        )

    def test_unknown_family_rejected(self, model: StyleModel, passage_texts) -> None:
        with pytest.raises(ValueError, match="unknown feature family"):
            model.delta(passage_texts[0], passage_texts[1], family="nope")

    def test_explicit_wordlist_needs_a_version(self, passage_texts: list[str]) -> None:
        with pytest.raises(MetricError, match="unversioned feature set"):
            StyleModel.fit(passage_texts, function_words=["the", "and"])


class TestAuthorSeparation:
    """M0-b acceptance: does the stylometry tell these authors apart? (plan.md §12.2)"""

    @staticmethod
    def _split(model: StyleModel, labelled, scorer):
        within, cross = [], []
        for i in range(len(labelled)):
            for j in range(i + 1, len(labelled)):
                distance = scorer(model, labelled[i][1], labelled[j][1])
                (within if labelled[i][0] == labelled[j][0] else cross).append(distance)
        return within, cross

    @staticmethod
    def _auc(within, cross) -> float:
        """P(a cross-author pair is more distant than a within-author pair)."""
        wins = sum((c > w) + 0.5 * (c == w) for w in within for c in cross)
        return wins / (len(within) * len(cross))

    def test_mean_cross_author_delta_exceeds_within_author(self, model, labelled_passages) -> None:
        within, cross = self._split(
            model, labelled_passages, lambda m, a, b: burrows_delta(a, b, m)
        )
        assert sum(cross) / len(cross) > sum(within) / len(within)

    def test_function_word_delta_is_above_chance(self, model, labelled_passages) -> None:
        within, cross = self._split(
            model, labelled_passages, lambda m, a, b: burrows_delta(a, b, m)
        )
        assert self._auc(within, cross) > 0.70

    def test_punctuation_separates_authors_best(self, model, labelled_passages) -> None:
        """Measured, not assumed -- and it is *not* the family Burrows' Delta uses.

        On the Phase-0 corpus, punctuation rate discriminates author markedly better than
        function-word frequency, and sentence shape is near chance. This test pins that
        ordering because plan.md §8's A4 voice veto has to be built on whichever family
        actually carries the signal. If a future corpus change flips it, that is a finding
        and this test should fail loudly rather than be quietly relaxed.
        """
        punct = self._auc(
            *self._split(model, labelled_passages, lambda m, a, b: m.delta(a, b, family="punct"))
        )
        fw = self._auc(
            *self._split(model, labelled_passages, lambda m, a, b: m.delta(a, b, family="fw"))
        )
        sent = self._auc(
            *self._split(model, labelled_passages, lambda m, a, b: m.delta(a, b, family="sent"))
        )
        assert punct > fw > sent
        assert punct > 0.80
        assert sent < 0.70, "sentence shape alone is close to chance at this passage length"

    def test_the_two_stylistic_extremes_are_the_farthest_apart(self, model, passages) -> None:
        """Hemingway and Woolf are the poles plan.md §5 picked them to be."""
        pairs: dict[tuple[str, str], list[float]] = {}
        for i in range(len(passages)):
            for j in range(i + 1, len(passages)):
                key = tuple(sorted((passages[i]["author_id"], passages[j]["author_id"])))
                pairs.setdefault(key, []).append(
                    burrows_delta(passages[i]["text"], passages[j]["text"], model)
                )
        means = {k: sum(v) / len(v) for k, v in pairs.items()}
        assert max(means, key=means.get) == ("hemingway", "woolf")


class TestMeanPairwiseDelta:
    def test_cross_author_only_by_default(self, model, labelled_passages) -> None:
        both = mean_pairwise_delta(labelled_passages, model, cross_author_only=False)
        cross = mean_pairwise_delta(labelled_passages, model, cross_author_only=True)
        assert cross > both, "including within-author pairs must pull the mean down"

    def test_needs_two_texts(self, model) -> None:
        with pytest.raises(MetricError, match="at least 2 texts"):
            mean_pairwise_delta([("woolf", PROSE)], model)

    def test_no_qualifying_pairs_raises_rather_than_returning_zero(self, model) -> None:
        with pytest.raises(MetricError, match="no qualifying pairs"):
            mean_pairwise_delta([("w", PROSE), ("w", PROSE)], model)


class TestClusterPurity:
    def test_perfect_and_worst_cases(self) -> None:
        assert cluster_purity(["a", "a", "b", "b"], [0, 0, 1, 1]) == 1.0
        assert cluster_purity(["a", "b"], [0, 0]) == 0.5

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="one assignment per label"):
            cluster_purity(["a"], [0, 1])


def test_describe_covers_every_model_free_readout(passage_texts: list[str]) -> None:
    out = describe(passage_texts[0])
    assert out["words"] > 500
    assert {"sent:mean", "sent:sd", "lex:mtld", "punct:comma"} <= set(out)
