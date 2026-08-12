"""Mechanical defect detection.

The most important test here is :func:`test_detectors_do_not_import_the_injector`. These
detectors are scored against defects planted by code in the same repo, so a detector
written by inverting the injector would score beautifully and generalise to nothing.
"""

from __future__ import annotations

import pytest

from revisionbench.detect import (
    DETECTORS,
    Complaint,
    detect_all,
    detect_echo,
    detect_name_variant,
    detect_overlong_sentence,
    detect_tense_outlier,
)

PAST_PASSAGE = " ".join(
    [
        "Katherine walked down the lane past the low wall, counting the gates as she went.",
        "There were seven of them, and the seventh was open, which surprised her greatly.",
        "Beyond it the field ran away downhill toward a long line of poplars in the haze.",
        "Katherine thought of the letter in her pocket and did not take it out that day.",
        "Someone had been burning stubble, and the smell hung about the hedges all morning.",
        "Katherine had come this way for weeks and had noticed none of it before now.",
        "Katherine put her hand on the wet top bar and stood there a long while.",
    ]
)


def test_detectors_do_not_import_the_injector() -> None:
    """Anti-circularity, enforced rather than promised.

    A detector that knows how the corruption was made detects this corpus and nothing
    else. `revisionbench.detect` must be writable against a manuscript whose flaws nobody
    planted.
    """
    import ast
    import pathlib

    # Parsed, not grepped: the module docstring discusses the injector at length, and a
    # substring check would flag that discussion while missing `import x as y`.
    tree = ast.parse(pathlib.Path("revisionbench/detect.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("inject" in name for name in imported), imported


class TestComplaint:
    def test_carries_a_locatable_span_and_evidence(self) -> None:
        """An unverifiable complaint is just another invitation to rewrite."""
        complaint = Complaint(type="x", span=(3, 9), message="m", evidence="e")
        assert complaint.as_dict()["span"] == [3, 9]
        assert complaint.as_dict()["evidence"] == "e"


class TestNameVariant:
    def test_flags_a_rare_near_spelling_of_a_frequent_name(self) -> None:
        text = PAST_PASSAGE.replace("Katherine had come", "Katharine had come")
        found = detect_name_variant(text)
        assert [c.type for c in found] == ["name_variant"]
        assert text[found[0].span[0] : found[0].span[1]] == "Katharine"

    def test_silent_on_consistent_spelling(self) -> None:
        assert detect_name_variant(PAST_PASSAGE) == []

    def test_needs_no_dictionary_or_cast_list(self) -> None:
        """The property is statistical, so it works on a book it has never seen."""
        text = "Zorblax spoke first. Zorblax waited. Zorblax left the room. Zorblex returned."
        assert [c.evidence for c in detect_name_variant(text)] == ["Zorblex / Zorblax"]

    def test_two_genuinely_different_names_are_not_flagged(self) -> None:
        text = "Anna spoke. Anna waited. Anna left. Peter returned and Peter sat down."
        assert detect_name_variant(text) == []


class TestEcho:
    def test_flags_a_phrase_repeated_three_times_nearby(self) -> None:
        text = PAST_PASSAGE.replace(
            "There were seven of them", "counting the gates, there were seven of them"
        ).replace("Someone had been burning", "counting the gates, someone had been burning")
        found = detect_echo(text)
        assert found and found[0].type == "echo"
        assert "counting the gates" in found[0].evidence

    def test_silent_on_ordinary_prose(self) -> None:
        assert detect_echo(PAST_PASSAGE) == []

    def test_a_distant_recurrence_is_a_motif_not_an_echo(self) -> None:
        """Window-limited on purpose: the distinction between motif and echo is distance."""
        phrase = "the light on the water"
        # Every n-gram in the filler must be unique, or the filler plants the very defect
        # the test is asserting is absent. Two earlier attempts failed exactly that way.
        filler = " ".join(
            f"Alpha{i} beta{i} gamma{i} delta{i} epsilon{i} zeta{i}." for i in range(60)
        )
        text = f"{phrase} {filler} {phrase} {filler} {phrase}"
        assert detect_echo(text, window=200) == []


class TestTenseOutlier:
    def test_flags_a_present_tense_sentence_in_past_narration(self) -> None:
        text = PAST_PASSAGE.replace(
            "There were seven of them, and the seventh was open, which surprised her greatly.",
            "There are seven of them, and the seventh is open, which is surprising to her.",
        )
        found = detect_tense_outlier(text)
        assert found and found[0].type == "tense_outlier"
        assert "are seven" in text[found[0].span[0] : found[0].span[1]]

    def test_silent_on_consistent_tense(self) -> None:
        assert detect_tense_outlier(PAST_PASSAGE) == []

    def test_majority_is_taken_from_the_passage_not_assumed(self) -> None:
        """A present-tense manuscript should have its past-tense stragglers flagged."""
        present = " ".join(
            [
                "She walks down the lane and counts the gates as she goes along today.",
                "There are seven of them and the seventh is open, which surprises her.",
                "The field runs away downhill and the smell of stubble is everywhere.",
                "She thinks of the letter in her pocket and it is still there unopened.",
                "She was walking this way for weeks and had noticed none of it before.",
            ]
        )
        found = detect_tense_outlier(present)
        assert found and "past-tense sentence" in found[0].message


class TestOverlongSentence:
    def test_requires_both_length_and_hedging(self) -> None:
        """Length alone is not a defect — Woolf's sentences run to 100 words."""
        long_but_concrete = (
            "She walked past the wall and the gate and the field and the poplars and the "
            "hedges and the stubble and the smoke and the gulls and the grass and the "
            "water and the bridge and the lane and the letter and the morning and the sun."
        )
        assert detect_overlong_sentence(PAST_PASSAGE + " " + long_but_concrete) == []

    def test_flags_a_long_hedging_sentence(self) -> None:
        clunker = (
            "It was the case that the situation, which had been developing for some "
            "considerable time and in a manner that few could have been expected to "
            "anticipate with any degree of confidence, was one which now seemed, on "
            "balance and all things considered, to be somewhat difficult in certain "
            "respects and various other matters."
        )
        found = detect_overlong_sentence(PAST_PASSAGE + " " + clunker)
        assert found and found[0].type == "overlong_sentence"


class TestDetectAll:
    def test_orders_by_position(self) -> None:
        text = PAST_PASSAGE.replace("Katherine had come", "Katharine had come")
        found = detect_all(text)
        assert found == sorted(found, key=lambda c: c.span)

    def test_unknown_detector_is_fatal(self) -> None:
        """A detector that did not run looks exactly like one that found nothing."""
        with pytest.raises(ValueError, match="unknown detector"):
            detect_all(PAST_PASSAGE, types=["spellcheck"])

    def test_can_be_restricted_to_a_subset(self) -> None:
        text = PAST_PASSAGE.replace("Katherine had come", "Katharine had come")
        assert detect_all(text, types=["echo"]) == []
        assert len(detect_all(text, types=["name_variant"])) == 1

    def test_registry_matches_the_exported_functions(self) -> None:
        assert set(DETECTORS) == {
            "name_variant",
            "echo",
            "tense_outlier",
            "overlong_sentence",
        }


class TestOnRealProse:
    """The number that is not circular: complaints on clean literary prose."""

    def test_false_positive_rate_stays_low_on_the_clean_corpus(self, passage_texts) -> None:
        complaints = [c for text in passage_texts for c in detect_all(text)]
        per_passage = len(complaints) / len(passage_texts)
        assert per_passage < 4.0, (
            f"{per_passage:.1f} complaints per clean passage. A detect-then-repair harness "
            f"puts every complaint in front of a model, so this rate IS the overreach "
            f"ceiling: {[c.type for c in complaints]}"
        )
