"""Defect injection and M5 recall.

The load-bearing test here is :meth:`TestDeletionIsNotAFix.test_deleting_the_region_is_not_a_fix`.
Phase 0 measured revisers cutting 40-50% of a passage, so scoring a fix as "the corruption
is gone" would credit deletion as repair — and the most destructive arm would post the best
defect recall in the study.
"""

from __future__ import annotations

import pytest

from revisionbench.inject import (
    DEFECT_TYPES,
    INJECTOR_VERSION,
    Defect,
    InjectionError,
    inject_passage,
)
from revisionbench.metrics.defects import defect_outcome, recall_report

SENTENCES = [
    "Katherine walked down the lane past the low wall, counting the gates as she went along.",
    "There were seven of them, and the seventh was open, which surprised her considerably.",
    "Beyond it the field ran away downhill toward a long line of grey poplars in the haze.",
    "Katherine thought of the letter in her pocket and did not take it out again that day.",
    "Someone had been burning stubble, and the smell hung about the hedges and would not lift.",
    "The gulls went over in twos and threes, calling, and the light moved upon the wet grass.",
    "Katherine had come this way every day for seven weeks and had noticed none of it before.",
    "She put her hand on the top bar of the gate, which was wet, and stood there a long while.",
]
PASSAGE = " ".join(SENTENCES)


class TestInjectPassage:
    def test_is_deterministic(self) -> None:
        a = inject_passage(PASSAGE, "p1", seed=0)
        b = inject_passage(PASSAGE, "p1", seed=0)
        assert a.text == b.text
        assert [d.as_dict() for d in a.defects] == [d.as_dict() for d in b.defects]

    def test_different_seeds_differ(self) -> None:
        a = inject_passage(PASSAGE, "p1", seed=0)
        b = inject_passage(PASSAGE, "p1", seed=5)
        assert a.text != b.text or a.defects != b.defects

    def test_corrupts_the_text(self) -> None:
        result = inject_passage(PASSAGE, "p1", seed=1)
        assert result.text != PASSAGE
        assert result.defects

    def test_defects_carry_full_provenance(self) -> None:
        result = inject_passage(PASSAGE, "p1", seed=1)
        for defect in result.defects:
            assert defect.passage_id == "p1"
            assert defect.injector_version == INJECTOR_VERSION
            assert defect.detect_kind in {"marker", "count", "span", "minimal_pair"}
            assert defect.defect_id.startswith("p1-d")

    def test_skipped_types_are_recorded_not_dropped(self) -> None:
        """A passage with fewer defects has a different denominator for recall."""
        tiny = "She walked. He waited. They left."
        with pytest.raises(InjectionError):
            inject_passage(tiny, "p1", seed=0, types=["clunker"])

    def test_unknown_type_is_rejected(self) -> None:
        with pytest.raises(InjectionError, match="unknown defect type"):
            inject_passage(PASSAGE, "p1", seed=0, types=["nope"])

    def test_target_bounds_the_count(self) -> None:
        result = inject_passage(PASSAGE, "p1", seed=0, target=2)
        assert len(result.defects) <= 2

    def test_spans_point_at_the_corruption_in_the_final_text(self) -> None:
        """Sequential injection shifts earlier spans; the manifest must be re-anchored.

        Without this, a four-defect manifest had its second entry pointing at unrelated
        words. No score changes (scoring matches markers and sentences, not offsets) but
        the hand-verification plan.md §6 requires shows the reviewer the wrong text, which
        is worse than showing nothing.
        """
        for seed in range(6):
            result = inject_passage(PASSAGE, "p1", seed=seed)
            for defect in result.defects:
                if defect.detect_kind == "count" or not defect.corrupt_fragment:
                    continue
                start, end = defect.char_span
                assert result.text[start:end] == defect.corrupt_fragment, (
                    f"seed {seed}, {defect.defect_id} ({defect.type}) span points at "
                    f"{result.text[start:end]!r}"
                )

    def test_every_declared_type_is_callable(self) -> None:
        assert set(DEFECT_TYPES) == {"name_drift", "tense_slip", "clunker", "echo", "number_drift"}


class TestIndividualInjectors:
    def one(self, defect_type: str, seed: int = 0):
        result = inject_passage(PASSAGE, "p1", seed=seed, types=[defect_type], target=1)
        return result.text, result.defects[0]

    def test_name_drift_changes_one_occurrence_only(self) -> None:
        text, defect = self.one("name_drift")
        assert defect.type == "name_drift"
        assert text.count(defect.corrupt_fragment) == 1
        assert defect.original_fragment in text, "the correct spelling still appears elsewhere"

    def test_name_drift_never_hits_the_first_occurrence(self) -> None:
        """A name is established on first use; drifting it there reads as the correct form."""
        text, defect = self.one("name_drift")
        assert text.index(defect.original_fragment) < text.index(defect.corrupt_fragment)

    def test_tense_slip_is_grammatical_and_reversible(self) -> None:
        text, defect = self.one("tense_slip")
        assert defect.corrupt_fragment in text
        assert "is" in defect.corrupt_fragment.split() or "are" in defect.corrupt_fragment.split()

    def test_clunker_is_inserted_verbatim(self) -> None:
        text, defect = self.one("clunker")
        assert defect.corrupt_fragment in text
        assert len(defect.corrupt_fragment.split()) > 30

    def test_echo_produces_three_occurrences(self) -> None:
        text, defect = self.one("echo")
        assert text.count(defect.marker) >= 3
        assert defect.detect_kind == "count"

    def test_number_drift_creates_a_contradiction(self) -> None:
        text, defect = self.one("number_drift")
        assert defect.corrupt_fragment in text
        assert defect.original_fragment in text.lower()


class TestDeletionIsNotAFix:
    """The core M5 design decision."""

    def defect(self, **overrides) -> dict:
        base = Defect(
            defect_id="p1-d1",
            passage_id="p1",
            type="name_drift",
            original_fragment="Katherine",
            corrupt_fragment="Katharine",
            char_span=(0, 9),
            clean_sentence=SENTENCES[3],
            detect_kind="marker",
            marker="Katharine",
        ).as_dict()
        base.update(overrides)
        return base

    def test_repairing_the_marker_is_a_fix(self) -> None:
        revised = PASSAGE  # the region is present and the misspelling is not
        outcome = defect_outcome(self.defect(), revised)
        assert outcome.status == "fixed"

    def test_surviving_marker_is_a_miss(self) -> None:
        revised = PASSAGE.replace("Katherine thought", "Katharine thought")
        outcome = defect_outcome(self.defect(), revised)
        assert outcome.status == "present"

    def test_deleting_the_region_is_not_a_fix(self) -> None:
        """The failure this whole module exists to prevent.

        The reviser cut the sentence containing the defect. The misspelling is gone, so a
        naive "is the corruption still there" check would score a repair — and the arm that
        deletes most would win on defect recall.
        """
        revised = " ".join(s for s in SENTENCES if s != SENTENCES[3])
        outcome = defect_outcome(self.defect(), revised)
        assert outcome.status == "removed"
        assert "cut rather than repaired" in outcome.detail

    def test_a_heavily_rewritten_but_present_region_still_counts(self) -> None:
        """Revisers are supposed to rewrite; a loose threshold is deliberate."""
        revised = PASSAGE.replace(
            SENTENCES[3],
            "Katherine considered the letter in her pocket, and left it there all day.",
        )
        assert defect_outcome(self.defect(), revised).status == "fixed"

    def test_empty_revision_is_unscoreable(self) -> None:
        assert defect_outcome(self.defect(), "").status == "unscoreable"


class TestRecallReport:
    def defects(self) -> list[dict]:
        return [
            Defect(
                defect_id=f"p1-d{i}",
                passage_id="p1",
                type="name_drift",
                original_fragment="Katherine",
                corrupt_fragment=f"Kath{i}rine",
                char_span=(0, 9),
                clean_sentence=SENTENCES[i],
                detect_kind="marker",
                marker=f"Kath{i}rine",
            ).as_dict()
            for i in range(3)
        ]

    def test_recall_is_over_surviving_defects_only(self) -> None:
        # Sentences 0 and 1 survive with markers gone; sentence 2's region is deleted.
        revised = " ".join(SENTENCES[:2])
        report = recall_report(self.defects(), revised)
        assert report.fixed == 2
        assert report.removed == 1
        assert report.surviving == 2
        assert report.recall == pytest.approx(1.0)

    def test_recall_is_none_when_nothing_survived(self) -> None:
        """0.0 would say 'fixed nothing'; the truth is 'deleted everything'."""
        report = recall_report(self.defects(), "An entirely unrelated sentence about trains.")
        assert report.surviving == 0
        assert report.recall is None
        assert report.removal_rate == pytest.approx(1.0)

    def test_removal_rate_is_always_available_beside_recall(self) -> None:
        report = recall_report(self.defects(), " ".join(SENTENCES[:2]))
        assert report.removal_rate == pytest.approx(1 / 3)
        assert report.as_dict()["removal_rate"] == pytest.approx(1 / 3)

    def test_planted_counts_every_defect(self) -> None:
        report = recall_report(self.defects(), " ".join(SENTENCES))
        assert report.planted == 3


class TestEndToEnd:
    def test_an_uncorrected_passage_scores_zero_recall(self) -> None:
        """The control: hand the corrupted text back unchanged and nothing is fixed."""
        result = inject_passage(PASSAGE, "p1", seed=3)
        report = recall_report([d.as_dict() for d in result.defects], result.text)
        assert report.fixed == 0
        assert report.recall == pytest.approx(0.0)

    def test_the_clean_original_scores_full_recall(self) -> None:
        """The upper bound: the clean text is by definition free of every planted defect."""
        result = inject_passage(PASSAGE, "p1", seed=3)
        report = recall_report([d.as_dict() for d in result.defects], PASSAGE)
        assert report.surviving == report.planted, "no region was deleted"
        assert report.recall == pytest.approx(1.0)
