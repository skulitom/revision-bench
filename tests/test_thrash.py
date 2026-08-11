"""M6 thrash and convergence.

:class:`TestHandBuiltRevertSequence` is milestone M0-b's stated acceptance criterion —
"verify thrash detector on a hand-built revert sequence".
"""

from __future__ import annotations

import pytest

from revisionbench.metrics.thrash import (
    align_sentences,
    edit_report,
    rounds_to_fixed_point,
    sentence_similarity,
    thrash_report,
)

# A four-sentence passage. Sentence 2 is the one that will be thrashed.
BASE = (
    "The wind came off the water in short gusts. "
    "She walked down the lane past the low wall, counting the gates. "
    "There were seven of them, and the seventh was open. "
    "Beyond it the field ran away downhill toward a line of poplars."
)
# Round 1: sentence 2 rewritten.
EDITED = (
    "The wind came off the water in short gusts. "
    "She made her way along the lane beside the low wall, tallying the gates. "
    "There were seven of them, and the seventh was open. "
    "Beyond it the field ran away downhill toward a line of poplars."
)
# Round 2 (revert): sentence 2 restored to its round-0 wording.
REVERTED = BASE
# Round 2 (further rewrite): sentence 2 changed again, into something new.
REWRITTEN = (
    "The wind came off the water in short gusts. "
    "She went along the track by the fence, and did not count anything at all. "
    "There were seven of them, and the seventh was open. "
    "Beyond it the field ran away downhill toward a line of poplars."
)


class TestSentenceSimilarity:
    def test_identical_is_one(self) -> None:
        assert sentence_similarity("A cat sat.", "A cat sat.") == 1.0

    def test_typographic_churn_is_invisible(self) -> None:
        """A reviser re-typesetting its input must not register as an edit."""
        assert sentence_similarity("He said--yes, it's fine.", "He said—yes, it’s fine.") == 1.0

    def test_unrelated_sentences_are_low(self) -> None:
        assert sentence_similarity("The wind came off the water.", "Bank rates fell again.") < 0.4

    def test_empty_handling(self) -> None:
        assert sentence_similarity("", "") == 1.0
        assert sentence_similarity("", "words here") == 0.0


class TestAlignment:
    def test_identical_sequences_align_one_to_one(self) -> None:
        s = ["One two three.", "Four five six.", "Seven eight nine."]
        pairs = align_sentences(s, s)
        assert [p.op for p in pairs] == ["unchanged"] * 3
        assert [(p.index_a, p.index_b) for p in pairs] == [(0, 0), (1, 1), (2, 2)]

    def test_insertion_is_detected(self) -> None:
        a = ["The wind came off the water.", "Beyond it the field ran downhill."]
        b = [
            "The wind came off the water.",
            "A gull went over, calling twice.",
            "Beyond it the field ran downhill.",
        ]
        assert [p.op for p in align_sentences(a, b)] == ["unchanged", "inserted", "unchanged"]

    def test_deletion_is_detected(self) -> None:
        a = [
            "The wind came off the water.",
            "A gull went over, calling twice.",
            "Beyond it the field ran downhill.",
        ]
        b = ["The wind came off the water.", "Beyond it the field ran downhill."]
        assert [p.op for p in align_sentences(a, b)] == ["unchanged", "deleted", "unchanged"]

    def test_light_edit_keeps_the_slot(self) -> None:
        a = ["She walked down the lane past the low wall, counting the gates."]
        b = ["She walked down the lane past the low wall, counting the gateposts."]
        pairs = align_sentences(a, b)
        assert [p.op for p in pairs] == ["modified"]
        assert pairs[0].similarity > 0.8

    def test_wholly_different_sentence_does_not_capture_the_slot(self) -> None:
        a = ["The wind came off the water in short gusts."]
        b = ["Interest rates were raised again on Thursday morning."]
        assert sorted(p.op for p in align_sentences(a, b)) == ["deleted", "inserted"]

    def test_alignment_is_monotonic(self) -> None:
        """A crossing alignment inflates 'modified' with merely-generic pairs."""
        a = ["Alpha one here.", "Beta two here.", "Gamma three here.", "Delta four here."]
        b = ["Gamma three here.", "Alpha one here.", "Delta four here.", "Beta two here."]
        pairs = align_sentences(a, b)
        matched = [
            (p.index_a, p.index_b) for p in pairs if p.index_a is not None and p.index_b is not None
        ]
        assert all(matched[i][0] < matched[i + 1][0] for i in range(len(matched) - 1))
        assert all(matched[i][1] < matched[i + 1][1] for i in range(len(matched) - 1))

    def test_empty_sides(self) -> None:
        assert align_sentences([], []) == []
        assert [p.op for p in align_sentences([], ["A sentence here."])] == ["inserted"]
        assert [p.op for p in align_sentences(["A sentence here."], [])] == ["deleted"]


class TestEditReport:
    def test_no_change(self) -> None:
        report = edit_report(BASE, BASE)
        assert report.unchanged == 4
        assert report.edits == 0
        assert report.edit_fraction == 0.0
        assert report.mean_modified_similarity is None

    def test_one_sentence_rewritten(self) -> None:
        report = edit_report(BASE, EDITED)
        assert (report.unchanged, report.modified, report.inserted, report.deleted) == (3, 1, 0, 0)
        assert report.edit_fraction == pytest.approx(0.25)
        assert 0.5 < report.mean_modified_similarity < 1.0

    def test_edit_depth_is_reported_separately_from_edit_volume(self) -> None:
        """Same edit volume, different depth -- the two must not be conflated."""
        shallow_text = BASE.replace("counting the gates", "counting the gateposts")
        shallow = edit_report(BASE, shallow_text)
        deep = edit_report(BASE, EDITED)
        assert shallow.modified == deep.modified == 1
        assert shallow.mean_modified_similarity > deep.mean_modified_similarity

    def test_serialises_derived_fields(self) -> None:
        data = edit_report(BASE, EDITED).as_dict()
        assert data["edits"] == 1
        assert data["edit_fraction"] == pytest.approx(0.25)


class TestHandBuiltRevertSequence:
    """M0-b acceptance criterion (plan.md §14)."""

    def test_a_revert_is_counted_as_a_revert(self) -> None:
        [report] = thrash_report([BASE, EDITED, REVERTED])
        assert report.edits_at_k == 1
        assert report.reverted == 1
        assert report.rewritten_again == 0
        assert report.thrash_fraction == pytest.approx(1.0)
        assert report.revert_fraction == pytest.approx(1.0)

    def test_a_further_rewrite_is_not_counted_as_a_revert(self) -> None:
        """Sentence 2 is replaced wholesale at k+2. Still thrash, but not a revert.

        A wholesale replacement is too dissimilar to hold its slot, so it reaches the
        alignment as a deletion beside an insertion. Counting that as a plain deletion
        would drop it out of the thrash fraction entirely and make a loop that rewrites
        the same sentence every round look like one that had settled.
        """
        [report] = thrash_report([BASE, EDITED, REWRITTEN])
        assert report.edits_at_k == 1
        assert report.reverted == 0
        assert report.replaced == 1
        assert report.deleted_after_edit == 0
        assert report.thrash_fraction == pytest.approx(1.0)
        assert report.revert_fraction == pytest.approx(0.0)

    def test_an_outright_cut_is_not_thrash(self) -> None:
        """The loop made a decision and kept it; that is convergence, not churn."""
        cut = EDITED.replace(
            "She made her way along the lane beside the low wall, tallying the gates. ", ""
        )
        [report] = thrash_report([BASE, EDITED, cut])
        assert report.edits_at_k == 1
        assert report.deleted_after_edit == 1
        assert report.replaced == 0
        assert report.thrash_fraction == pytest.approx(0.0)

    def test_a_settled_edit_is_not_thrash(self) -> None:
        [report] = thrash_report([BASE, EDITED, EDITED])
        assert report.edits_at_k == 1
        assert report.settled == 1
        assert report.thrash_fraction == pytest.approx(0.0)

    def test_no_edits_reports_none_not_zero(self) -> None:
        """0.0 reads as 'stable'; the truth is 'already stopped', the opposite finding."""
        [report] = thrash_report([BASE, BASE, BASE])
        assert report.edits_at_k == 0
        assert report.thrash_fraction is None
        assert report.revert_fraction is None

    def test_windows_cover_the_whole_trajectory(self) -> None:
        reports = thrash_report([BASE, EDITED, REVERTED, EDITED, REVERTED])
        assert [r.round_index for r in reports] == [0, 1, 2]
        assert all(r.reverted == 1 for r in reports), "an oscillating loop thrashes every window"

    def test_short_trajectories_have_no_windows(self) -> None:
        assert thrash_report([BASE]) == []
        assert thrash_report([BASE, EDITED]) == []

    def test_serialises_derived_fields(self) -> None:
        data = thrash_report([BASE, EDITED, REVERTED])[0].as_dict()
        assert data["thrash_fraction"] == pytest.approx(1.0)
        assert data["revert_fraction"] == pytest.approx(1.0)


class TestFixedPoint:
    def test_detected_at_the_first_repeat(self) -> None:
        assert rounds_to_fixed_point([BASE, EDITED, EDITED, EDITED]) == 2

    def test_none_when_never_settles(self) -> None:
        assert rounds_to_fixed_point([BASE, EDITED, REVERTED, EDITED]) is None

    def test_retypesetting_alone_counts_as_a_fixed_point(self) -> None:
        """plan.md §8 A5's idempotence check: re-typesetting is not a proposal."""
        assert rounds_to_fixed_point([BASE, BASE.replace("--", "—")]) == 1

    def test_single_version_has_no_fixed_point(self) -> None:
        assert rounds_to_fixed_point([BASE]) is None
