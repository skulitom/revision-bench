"""M3 slop index and lexicon validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from revisionbench.metrics.slop import SlopError, SlopLexicon, load_lexicon, slop_index


def make_lexicon(tmp_path: Path, groups: list[dict], version: int = 1) -> SlopLexicon:
    path = tmp_path / "lex.yaml"
    path.write_text(yaml.safe_dump({"version": version, "groups": groups}), encoding="utf-8")
    return load_lexicon(path)


@pytest.fixture
def lexicon(tmp_path: Path) -> SlopLexicon:
    return make_lexicon(
        tmp_path,
        [
            {"name": "g1", "source": "test", "terms": ["delve", "sense of unease", "sense of"]},
            {
                "name": "g2",
                "source": "test",
                "terms": ["pit of * stomach", "it's important to note"],
            },
        ],
    )


class TestMatching:
    def test_single_word_hit(self, lexicon: SlopLexicon) -> None:
        report = slop_index("We delve into the matter now, and then we go home again.", lexicon)
        assert report.hits == 1
        assert report.by_term == {"delve": 1}
        assert report.by_group == {"g1": 1, "g2": 0}

    def test_case_and_punctuation_do_not_matter(self, lexicon: SlopLexicon) -> None:
        a = slop_index("Delve! Delve, delve. " + "filler " * 20, lexicon)
        assert a.hits == 3

    def test_leftmost_longest_prevents_double_counting(self, lexicon: SlopLexicon) -> None:
        """'sense of unease' and 'sense of' both listed; the longer one wins, once."""
        report = slop_index("There was a sense of unease in the room that evening.", lexicon)
        assert report.hits == 1
        assert report.by_term == {"sense of unease": 1}

    def test_shorter_term_still_fires_on_its_own(self, lexicon: SlopLexicon) -> None:
        report = slop_index("There was a sense of purpose in the room that evening.", lexicon)
        assert report.by_term == {"sense of": 1}

    def test_wildcard_matches_exactly_one_token(self, lexicon: SlopLexicon) -> None:
        assert slop_index("a cold weight in the pit of her stomach today", lexicon).hits == 1
        assert slop_index("a cold weight in the pit of my stomach today", lexicon).hits == 1
        # Two tokens in the gap: the single-token wildcard must not stretch.
        assert slop_index("in the pit of her own stomach today", lexicon).hits == 0

    def test_contraction_term_matches_the_tokeniser(self, lexicon: SlopLexicon) -> None:
        """The term is written as prose and tokenised by the same code the text is."""
        report = slop_index("Well, it's important to note that the door was open.", lexicon)
        assert report.by_term == {"it's important to note": 1}
        assert slop_index("Well, it’s important to note that the door was open.", lexicon).hits == 1

    def test_no_hits_is_a_measured_zero(self, lexicon: SlopLexicon) -> None:
        report = slop_index("The wind came off the water in short gusts today.", lexicon)
        assert report.hits == 0
        assert report.per_1000_words == 0.0
        assert report.by_group == {"g1": 0, "g2": 0}

    def test_matches_do_not_overlap(self, lexicon: SlopLexicon) -> None:
        report = slop_index("sense of sense of sense of " + "x " * 10, lexicon)
        assert report.hits == 3

    def test_rate_is_per_1000_words(self, lexicon: SlopLexicon) -> None:
        text = "delve " + "filler " * 99
        report = slop_index(text, lexicon)
        assert report.word_count == 100
        assert report.per_1000_words == pytest.approx(10.0)

    def test_empty_text_raises_rather_than_scoring_zero(self, lexicon: SlopLexicon) -> None:
        with pytest.raises(ValueError, match="no word tokens"):
            slop_index("!!! ...", lexicon)


class TestLexiconValidation:
    def test_missing_source_is_rejected(self, tmp_path: Path) -> None:
        """An uncited group reports a rate that looks as authoritative as a cited one."""
        with pytest.raises(SlopError, match="missing source"):
            make_lexicon(tmp_path, [{"name": "g", "terms": ["delve"]}])

    def test_duplicate_term_across_groups_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SlopError, match="appears in both"):
            make_lexicon(
                tmp_path,
                [
                    {"name": "a", "source": "s", "terms": ["delve"]},
                    {"name": "b", "source": "s", "terms": ["delve"]},
                ],
            )

    def test_duplicate_group_name_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SlopError, match="duplicate group name"):
            make_lexicon(
                tmp_path,
                [
                    {"name": "a", "source": "s", "terms": ["delve"]},
                    {"name": "a", "source": "s", "terms": ["tapestry"]},
                ],
            )

    def test_multi_wildcard_term_is_rejected(self, tmp_path: Path) -> None:
        """It would never fire, and would look like a term that simply does not occur."""
        with pytest.raises(SlopError, match="wildcards"):
            make_lexicon(tmp_path, [{"name": "a", "source": "s", "terms": ["a * b * c"]}])

    def test_all_wildcard_term_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(SlopError, match="nothing but wildcards"):
            make_lexicon(tmp_path, [{"name": "a", "source": "s", "terms": ["*"]}])

    def test_non_string_term_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "lex.yaml"
        path.write_text(
            "version: 1\ngroups:\n  - name: a\n    source: s\n    terms:\n      - no\n",
            encoding="utf-8",
        )
        with pytest.raises(SlopError, match=r"YAML 1\.1"):
            load_lexicon(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SlopError, match="not found"):
            load_lexicon(tmp_path / "absent.yaml")


class TestShippedLexicon:
    def test_loads_and_every_group_is_attributed(self) -> None:
        lexicon = load_lexicon()
        assert lexicon.version >= 1
        assert len(lexicon.terms) > 50
        assert all(source for source in lexicon.groups.values())

    def test_curated_groups_are_marked_as_such(self) -> None:
        """plan.md §12: project-curated terms are hypotheses, not citations."""
        lexicon = load_lexicon()
        curated = [n for n, s in lexicon.groups.items() if s.startswith("curated:project")]
        cited = [n for n, s in lexicon.groups.items() if not s.startswith("curated:project")]
        assert curated and cited, "the lexicon should carry both cited and curated groups"

    def test_scores_real_prose_without_error(self, passage_texts: list[str]) -> None:
        lexicon = load_lexicon()
        for text in passage_texts:
            report = slop_index(text, lexicon)
            assert report.word_count > 500
            assert report.per_1000_words >= 0.0

    def test_human_prose_baseline_is_not_saturated(self, passage_texts: list[str]) -> None:
        """A lexicon that fires constantly on 1920s fiction cannot show a rise from a loop.

        This is a sanity bound on the lexicon, not a claim about the authors. If the
        baseline were high, the M3 slope would be measuring ordinary English.
        """
        lexicon = load_lexicon()
        rates = [slop_index(t, lexicon).per_1000_words for t in passage_texts]
        assert max(rates) < 12.0, f"human baseline too high: {rates}"
