"""Tokenisation, sentence splitting and punctuation classes.

Every case in :data:`SPLIT_CASES` is a construction that breaks a naive
``re.split(r'[.!?]')`` splitter, and several of them come from the Phase-0 corpus itself.
"""

from __future__ import annotations

import pytest

from revisionbench.text import (
    fold_typography,
    normalise_newlines,
    paragraph_spans,
    paragraphs,
    punctuation_counts,
    sentence_lengths,
    sentence_spans,
    sentences,
    word_count,
    words,
)

SPLIT_CASES = [
    # (text, expected sentence count, why this case exists)
    ("Mrs. Dalloway said she would buy the flowers herself.", 1, "honorific abbreviation"),
    ('"Go away!" he shouted. She went.', 2, "terminator inside quotes, lowercase after"),
    ('"Stop!" — he said.', 1, "em-dash dialogue attribution must not split"),
    ("It was I. She knew it.", 2, "'I' is a pronoun, not an initial"),
    ("Oh, no. She's gone.", 2, "'no' is an ordinary word, not the abbreviation No."),
    ("See No. 5 for details.", 1, "'No.' before a number IS the abbreviation"),
    ("E. M. Forster wrote it. Then he stopped.", 2, "initials"),
    ("The value was 3.14 exactly.", 1, "decimal point"),
    ("He paused... then went on.", 1, "ellipsis followed by lowercase"),
    ("He paused… Then he went on.", 2, "ellipsis followed by capital"),
    ("Vol. II was missing.", 1, "numeric abbreviation with a Roman numeral"),
    ("Dr. Smith and Mr. Jones met on St. Paul's.", 1, "several honorifics in one sentence"),
    ("First para line one.\n\nSecond para line two.", 2, "blank line is a hard boundary"),
    ("No terminator here", 1, "paragraph ending without punctuation"),
    ('She said, "I don’t know," and left. He nodded.', 2, "comma inside quotes"),
    ("Was it? Yes! Certainly.", 3, "consecutive short sentences"),
    ("", 0, "empty text has no sentences"),
    ("   \n\n  ", 0, "whitespace-only text has no sentences"),
]


@pytest.mark.parametrize(("text", "expected", "why"), SPLIT_CASES)
def test_sentence_split_counts(text: str, expected: int, why: str) -> None:
    got = sentences(text)
    assert len(got) == expected, f"{why}: got {got}"


def test_sentence_spans_are_exact_slices() -> None:
    text = normalise_newlines("First one. Second here!\n\nNew para begins.")
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == sentences(text)
    # Spans carry no surrounding whitespace, and they are ordered and disjoint.
    for a, b in spans:
        assert text[a:b] == text[a:b].strip()
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_paragraph_spans_are_exact_slices() -> None:
    text = normalise_newlines("Para one.\n\n\nPara two, line one.\nStill two.\n\nPara three.")
    spans = paragraph_spans(text)
    assert [text[a:b] for a, b in spans] == paragraphs(text)
    assert len(spans) == 3


def test_crlf_normalises_to_lf() -> None:
    assert normalise_newlines("a\r\nb\rc") == "a\nb\nc"
    # A BOM would otherwise become part of the first word token.
    assert normalise_newlines("﻿Hello") == "Hello"


class TestWords:
    def test_contractions_and_hyphens_survive(self) -> None:
        assert words("Don’t go, mother-in-law! O'clock 1914.") == [
            "don't",
            "go",
            "mother-in-law",
            "o'clock",
            "1914",
        ]

    def test_leading_and_trailing_apostrophes_are_dropped(self) -> None:
        assert words("'Tis the dogs' bone") == ["tis", "the", "dogs", "bone"]

    def test_case_folding_is_optional(self) -> None:
        assert words("The The", lowercase=False) == ["The", "The"]
        assert words("The The") == ["the", "the"]

    def test_em_dash_separates_words(self) -> None:
        assert words("one—two") == ["one", "two"]
        assert words("one--two") == ["one", "two"]

    def test_word_count_matches_words(self) -> None:
        text = "A short sentence, with punctuation; and 3 numbers: 4, 5."
        assert word_count(text) == len(words(text))


class TestTypographicFolding:
    def test_double_hyphen_is_an_em_dash(self) -> None:
        """The Phase-0 corpus depends on this.

        Mrs. Dalloway contains 545 "--" and 2 real em dashes; The Sun Also Rises contains
        0 and 39. Without the fold, Woolf's characteristic mark is counted as a hyphen and
        her em-dash rate reads as zero -- and any reviser that re-typesets "--" as an em
        dash manufactures an enormous punctuation-profile shift out of nothing.
        """
        assert fold_typography("a--b") == "a—b"
        assert fold_typography("a----b") == "a—b"
        assert punctuation_counts("a--b")["em_dash"] == 1
        assert punctuation_counts("a--b")["hyphen"] == 0
        assert punctuation_counts("a—b")["em_dash"] == 1

    def test_single_hyphen_stays_a_hyphen(self) -> None:
        assert punctuation_counts("mother-in-law")["hyphen"] == 2
        assert punctuation_counts("mother-in-law")["em_dash"] == 0

    def test_quote_styles_fold_together(self) -> None:
        curly = punctuation_counts("“Hello,” she said. It’s fine.")
        straight = punctuation_counts('"Hello," she said. It\'s fine.')
        assert curly == straight

    def test_ellipsis_spellings_fold_together(self) -> None:
        assert punctuation_counts("a...b")["ellipsis"] == 1
        assert punctuation_counts("a… b")["ellipsis"] == 1
        assert punctuation_counts("a. . . b")["ellipsis"] == 1

    def test_unfolded_counts_are_still_available(self) -> None:
        """Whether a reviser re-typesets its input is a real fact, just not a voice fact."""
        assert punctuation_counts("a--b", fold=False)["hyphen"] == 2
        assert punctuation_counts("a--b", fold=False)["em_dash"] == 0

    def test_en_dash_is_not_folded_to_hyphen(self) -> None:
        counts = punctuation_counts("1914–1918")
        assert counts["en_dash"] == 1
        assert counts["hyphen"] == 0


def test_punctuation_profile_always_has_every_class() -> None:
    """A missing key would let a comparison silently skip whatever a reviser removed."""
    empty = punctuation_counts("")
    rich = punctuation_counts("Hi, there; yes: no! Why? (a) — ...")
    assert set(empty) == set(rich)
    assert all(v == 0 for v in empty.values())


def test_sentence_lengths_keep_wordless_sentences() -> None:
    """A scene break is exactly the kind of thing a reviser deletes; it must be counted."""
    lengths = sentence_lengths("A real sentence here.\n\n* * *\n\nAnother real one.")
    assert lengths == [4, 0, 3]


def test_sentence_lengths_on_real_prose(passage_texts: list[str]) -> None:
    for text in passage_texts:
        lengths = sentence_lengths(text)
        assert len(lengths) >= 10
        assert sum(lengths) == word_count(text), "every word belongs to exactly one sentence"
