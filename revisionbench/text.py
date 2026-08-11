"""Tokenisation, sentence splitting and punctuation classification.

Everything downstream reads through this module: the function-word frequencies Burrows'
Delta runs on, the sentence-length distribution, the punctuation profile, MTLD, the slop
index's phrase matching, and the thrash detector's sentence alignment. A bug here does
not raise — it moves every metric at once, in a direction that will look like a finding.

Three decisions are worth arguing with before changing:

**Typographic folding is on by default.** An LLM reviser rewrites ``don't`` as ``don’t``
and ``"quote"`` as ``“quote”`` constantly, because that is what its training data looks
like. Counted naively, that single formatting habit shows up as a large punctuation-profile
shift and a collapse in the apostrophe-bearing function words — a big, clean, entirely
spurious "voice drift" signal in exactly the direction plan.md §4 predicts, on round 1.
So the tokeniser folds typographic variants to their ASCII class. The raw, unfolded counts
remain available (``punctuation_counts(fold=False)``) because *whether* a model
re-typesets its input is a genuine and separately interesting fact — it is just not
evidence about voice.

**The sentence splitter is a heuristic, and its job is consistency, not truth.** No
regex splitter is exactly right on literary prose: dialogue, ellipses, initials and
19th-century abbreviation habits all break simple rules. What plan.md §7 M1 actually needs
is a sentence-length *distribution* compared across rounds of the same passage, so a
splitter that is stably wrong in the same way on round 0 and round 7 is fine, while one
that is right on average but varies with formatting is not. Concretely: absolute sentence
counts here carry a few percent error; differences across rounds are what gets reported.

**Words are lowercased for frequency work but the raw text is never mutated.** Corpus
files stay byte-identical to what was extracted (see :mod:`revisionbench.corpus`);
normalisation happens on the way into a metric, not on the way into storage.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

__all__ = [
    "ABBREVIATIONS",
    "NUMERIC_ABBREVIATIONS",
    "PUNCTUATION_CLASSES",
    "PUNCTUATION_CLASS_NAMES",
    "TYPOGRAPHIC_FOLD",
    "fold_typography",
    "normalise_newlines",
    "paragraph_spans",
    "paragraphs",
    "punctuation_counts",
    "sentence_lengths",
    "sentence_spans",
    "sentences",
    "word_count",
    "words",
]

# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------

#: Typographic variants folded to a canonical ASCII form before word/punctuation counting.
#:
#: Note what is *absent*: the en dash (U+2013) and em dash (U+2014) are not folded to
#: "-". A hyphen joins a compound word, an em dash breaks a clause, and telling
#: Dickens from Hemingway partly means telling those apart. They are separate classes in
#: PUNCTUATION_CLASSES instead.
TYPOGRAPHIC_FOLD = {
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark / typographic apostrophe
    "‚": "'",  # single low-9 quotation mark
    "‛": "'",  # single high-reversed-9 quotation mark
    "ʼ": "'",  # modifier letter apostrophe
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "„": '"',  # double low-9 quotation mark
    "″": '"',  # double prime (occurs in OCR'd Gutenberg texts)
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    " ": " ",  # no-break space
    " ": " ",  # en space
    " ": " ",  # em space
    " ": " ",  # thin space
    " ": " ",  # hair space
    "﻿": "",  # BOM, which Gutenberg's UTF-8 files carry
}

_FOLD_TABLE = str.maketrans(TYPOGRAPHIC_FOLD)

#: Multi-character ellipsis spellings collapse to U+2026 so that "..." and "…" are one
#: punctuation class and one sentence-terminator shape. Applied after the table above.
_ELLIPSIS_RE = re.compile(r"\.\s?\.\s?\.(\s?\.)*")

#: A run of two or more hyphens is an em dash. This is not a nicety — it is measured.
#:
#: Project Gutenberg's plain-text files render em dashes as "--" for most titles. In this
#: corpus, Mrs. Dalloway contains 545 "--" and 2 real em dashes; Pointed Roofs has 749 and
#: 2; The Sun Also Rises has 0 and 39. Without this fold, Woolf's and Richardson's most
#: characteristic mark counts as a *hyphen*, their em-dash rate reads as zero, and the
#: punctuation profile that plan.md §7 M1 uses as a voice fingerprint is wrong for two of
#: the three Phase-0 authors.
#:
#: It gets worse under revision. An LLM re-typesets "--" as "—" almost every time, so
#: round 1 would show a collapse in hyphens and an explosion in em dashes: a large, clean,
#: entirely artefactual "voice drift" pointing the way plan.md §4 predicts drift should
#: point. That is the exact shape of wrong number this repo exists to not produce.
#:
#: Four-hyphen name redactions ("Mr. ----") also fold to one em dash. That is a rare and
#: acceptable loss next to the above.
_DASH_RUN_RE = re.compile(r"-{2,}")


def normalise_newlines(text: str) -> str:
    """Convert CRLF and CR line endings to LF, and strip a leading BOM.

    Called on the way in from every external source. Two texts differing only in line
    endings must hash identically and tokenise identically, or a Windows-produced artifact
    and a Linux-produced one will disagree about a passage that nobody edited.
    """
    if text.startswith("﻿"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def fold_typography(text: str) -> str:
    """Fold typographic variants to canonical ASCII (see :data:`TYPOGRAPHIC_FOLD`).

    Also applies Unicode NFC composition first, so that a letter written as "e" +
    combining acute compares equal to the precomposed character. Gutenberg files mix both.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_FOLD_TABLE)
    text = _ELLIPSIS_RE.sub("…", text)
    return _DASH_RUN_RE.sub("—", text)


# --------------------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------------------

# A word is a run of letters, optionally continued by an apostrophe or hyphen followed by
# more letters, so "don't", "mother-in-law" and "o'clock" survive as single tokens. This
# matters more than it looks: a naive \w+ splits "don't" into "don" and "t", and "don't"
# is a high-frequency function word that Burrows' Delta leans on, while the phantom token
# "t" lands in the lexical-diversity count as a distinct type.
#
# [^\W\d_] is "a Unicode letter" -- \w minus digits and underscore -- so accented
# characters in the corpus are letters rather than boundaries.
#
# Numbers are matched separately and are kept: "1914" is a word for length and diversity
# purposes. They are never function words, so Delta is unaffected either way.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")


def words(text: str, *, lowercase: bool = True, fold: bool = True) -> list[str]:
    """Tokenise ``text`` into words.

    Args:
        text: Raw passage text.
        lowercase: Lowercase every token. On for frequency work (a sentence-initial "The"
            and a mid-sentence "the" are the same function word); off when case is the
            thing being measured.
        fold: Apply :func:`fold_typography` first. Leave on unless measuring typography.

    Returns:
        Tokens in order. Punctuation, whitespace and standalone symbols are dropped;
        leading and trailing apostrophes are not part of a token, so ``'Tis`` yields
        ``tis`` and ``dogs'`` yields ``dogs``.
    """
    prepared = normalise_newlines(text)
    if fold:
        prepared = fold_typography(prepared)
    found = _WORD_RE.findall(prepared)
    return [w.lower() for w in found] if lowercase else found


def word_count(text: str) -> int:
    """Number of word tokens in ``text``, by the same rule :func:`words` uses.

    This is the denominator for every "per 1000 words" rate in the repo. Because it is a
    *definition* rather than a measurement, it must not drift: a metric that quietly
    switched to ``len(text.split())`` would rescale the slop index by roughly the
    punctuation rate without any other symptom.
    """
    return len(words(text))


# --------------------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------------------

#: Tokens that end in a period without ending a sentence, in *any* context. Stored
#: lowercase and without the period.
#:
#: The entries here are all words that are not also ordinary English words, and that
#: restriction is the whole design. A generous abbreviation list is the obvious way to
#: build this and it is wrong: put "no", "in", "art", "ed" or "sat" in here and
#: ``"Oh, no. She's gone."`` stops being two sentences. There is no error and no warning
#: — the passage simply reports a longer mean sentence length, which is one of the two
#: headline M1 numbers, and dialogue-heavy prose (the stratum most at risk of voice loss)
#: is hit hardest. Ambiguous entries go in :data:`NUMERIC_ABBREVIATIONS` instead.
ABBREVIATIONS = frozenset(
    {
        # honorifics and titles
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "rev",
        "hon",
        "st",
        "sr",
        "jr",
        "esq",
        "capt",
        "col",
        "gen",
        "lt",
        "sgt",
        "maj",
        "adm",
        "cmdr",
        "messrs",
        "mme",
        "mlle",
        "mons",
        "supt",
        "insp",
        "pres",
        "gov",
        # latin and editorial
        "etc",
        "vs",
        "viz",
        "cf",
        "ibid",
        "seq",
        "inst",
        # calendar. Only spellings that are not also common words: "mar", "may", "sun",
        # "sat" and "mon" are deliberately absent for the reason given above.
        "jan",
        "feb",
        "apr",
        "jun",
        "jul",
        "aug",
        "sept",
        "sep",
        "oct",
        "nov",
        "dec",
        "tues",
        "wed",
        "thurs",
        "thur",
        "fri",
    }
)

#: Abbreviations that suppress a sentence boundary *only when a number follows*.
#:
#: Every one of these is also an ordinary English word ("No. 5" vs "Oh, no."; "p. 12" vs
#: a sentence ending in "up"), so the disambiguator is the next token. "Vol. II" works
#: too: Roman numerals are covered by the uppercase-run check in :func:`_is_numeric_ref`.
NUMERIC_ABBREVIATIONS = frozenset(
    {
        "no",
        "nos",
        "p",
        "pp",
        "vol",
        "vols",
        "ch",
        "chap",
        "fig",
        "figs",
        "sec",
        "art",
        "pt",
        "ed",
        "eds",
        "op",
        "l",
        "ll",
        "in",
        "ft",
        "yd",
        "lb",
        "lbs",
        "oz",
        "hr",
        "hrs",
        "min",
        "al",
    }
)

#: Terminating punctuation, optionally followed by closing quotes/brackets.
#: The trailing group is what makes `He shouted, "Go!"  She went.` split correctly.
_TERMINATOR_RE = re.compile(r"[.!?…]+[\"')\]’”]*")

#: Characters a new sentence is allowed to start with, beyond a capital letter or digit:
#: an opening quote or a bracket.
#:
#: Dashes are deliberately *not* openers. Including them splits the extremely common
#: dialogue attribution `"Stop!" — he said.` into two "sentences", which both inflates
#: the sentence count and halves the mean length on exactly the dialogue-heavy passages
#: where that number matters most.
_OPENERS = "\"'([‘“"

#: A single letter before a period is an initial ("J. R. R. Tolkien", "E. M. Forster"),
#: not a sentence end. One letter only: requiring more would swallow real two-letter
#: sentence endings.
_INITIAL_RE = re.compile(r"(?:^|[\s\"'(‘“—-])([^\W\d_])$")

#: The one single letter excluded from the initial rule. "I" as an initial is vanishingly
#: rare in prose; "It was I." is not, and it appears in exactly the Victorian dialogue
#: this corpus is built from.
_NOT_AN_INITIAL = frozenset({"I"})

_WORD_BEFORE_RE = re.compile(r"([^\W\d_]+)$")

#: What counts as "a number follows" for NUMERIC_ABBREVIATIONS: an Arabic number, or a
#: short run of Roman-numeral letters ("Vol. II", "Ch. XIV").
_NUMERIC_REF_RE = re.compile(r"^\s*(?:\d|[IVXLCDM]{1,7}\b)")


_PARAGRAPH_SEP_RE = re.compile(r"\n[ \t]*\n+")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Shrink ``[start, end)`` past surrounding whitespace; ``None`` if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each paragraph, as ``[start, end)`` half-open pairs.

    **Spans index the text you passed in, unnormalised.** :func:`normalise_newlines`
    changes length (CRLF is two characters, LF is one), so a caller mixing raw and
    normalised text would get spans that are silently off by the number of preceding line
    breaks. Normalise once, at the boundary where text enters the system
    (:mod:`revisionbench.corpus` does this on fetch), and pass normalised text everywhere
    after that. This function does not normalise for you, precisely so that it cannot
    hand back offsets into a string that no longer exists.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sep in _PARAGRAPH_SEP_RE.finditer(text):
        trimmed = _trim_span(text, cursor, sep.start())
        if trimmed:
            spans.append(trimmed)
        cursor = sep.end()
    trimmed = _trim_span(text, cursor, len(text))
    if trimmed:
        spans.append(trimmed)
    return spans


def paragraphs(text: str) -> list[str]:
    """Split ``text`` into paragraphs on blank lines.

    A blank line is treated as an unconditional sentence boundary by :func:`sentences`,
    which removes a whole class of splitter error: a paragraph that ends without
    punctuation (common in dialogue and in verse quoted inside prose) cannot bleed into
    the next one.
    """
    normalised = normalise_newlines(text)
    return [normalised[a:b] for a, b in paragraph_spans(normalised)]


def sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    Heuristic, in the specific sense set out in the module docstring: tuned for stability
    across revision rounds of the same passage rather than for linguistic correctness.
    The rules, in the order they are applied at each candidate terminator:

    1. A blank line always ends a sentence (handled by :func:`paragraphs`).
    2. The terminator must be followed by whitespace or end-of-paragraph. This keeps
       ``3.14`` and ``e.g.something`` intact.
    3. A period directly after a known abbreviation (:data:`ABBREVIATIONS`) or after a
       single letter (an initial) does not end a sentence. ``Mrs. Dalloway`` is one
       sentence, not two. Abbreviations that are also ordinary words live in
       :data:`NUMERIC_ABBREVIATIONS` and suppress the boundary only before a number.
    4. The next non-space character must be a capital letter, a digit, or an opening
       quote or bracket. This is what keeps dialogue together: in
       ``"Go away!" he shouted.`` the ``!`` is followed by lowercase ``he``, so the
       sentence continues, which is correct and is also what a human counts.

    Returns:
        Sentences with surrounding whitespace stripped, in order, including their
        terminating punctuation. Interior newlines are preserved: the text is not
        otherwise rewritten, so ``"".join`` of the results is not the input.
    """
    normalised = normalise_newlines(text)
    return [normalised[a:b] for a, b in sentence_spans(normalised)]


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each sentence, as ``[start, end)`` half-open pairs.

    Same offset contract as :func:`paragraph_spans` — the spans index ``text`` exactly as
    given, and it is the caller's job to have normalised newlines already.

    This is the form :mod:`revisionbench.corpus` extracts passages with: cutting on a
    span means the stored passage is a byte-exact slice of the source book, so "does this
    passage really appear in that Gutenberg file" stays a checkable question rather than
    a matter of trusting the extractor's rejoining.
    """
    spans: list[tuple[int, int]] = []
    for block_start, block_end in paragraph_spans(text):
        block = text[block_start:block_end]
        for rel_start, rel_end in _split_block(block):
            spans.append((block_start + rel_start, block_start + rel_end))
    return spans


def _is_bare_period(terminator: str) -> bool:
    """True for "." and "..." — i.e. no closing quote and no ! or ?.

    The abbreviation and initial rules apply only here. ``Mr!`` is not a title, and a
    closing quote after the period (``said "Dr."``) means the sentence really has ended.
    """
    return set(terminator) == {"."}


def _is_numeric_ref(rest: str) -> bool:
    """True if ``rest`` begins with a number or a Roman numeral (see NUMERIC_ABBREVIATIONS)."""
    return _NUMERIC_REF_RE.match(rest) is not None


def _starts_a_sentence(rest: str) -> bool:
    """True if the text after a terminator looks like the start of a new sentence."""
    stripped = rest.lstrip()
    if not stripped:
        return True  # end of paragraph
    first = stripped[0]
    return first.isupper() or first.isdigit() or first in _OPENERS


def _split_block(block: str) -> list[tuple[int, int]]:
    """Sentence-split one paragraph, returning spans relative to ``block``."""
    result: list[tuple[int, int]] = []
    start = 0
    for match in _TERMINATOR_RE.finditer(block):
        end = match.end()
        # Rule 2: must be followed by whitespace or the end of the paragraph, so that
        # "3.14" and "e.g.something" stay intact.
        if end < len(block) and not block[end].isspace():
            continue

        rest = block[end:]
        preceding = block[: match.start()]

        # Rule 3: abbreviations and initials.
        if _is_bare_period(match.group()):
            initial = _INITIAL_RE.search(preceding)
            if initial and initial.group(1) not in _NOT_AN_INITIAL:
                continue
            word_match = _WORD_BEFORE_RE.search(preceding)
            if word_match:
                previous = word_match.group(1).lower()
                if previous in ABBREVIATIONS:
                    continue
                if previous in NUMERIC_ABBREVIATIONS and _is_numeric_ref(rest):
                    continue

        # Rule 4: what follows must look like the start of a sentence.
        if not _starts_a_sentence(rest):
            continue

        trimmed = _trim_span(block, start, end)
        if trimmed:
            result.append(trimmed)
        start = end

    tail = _trim_span(block, start, len(block))
    if tail:
        # A paragraph ending without terminating punctuation still ends a sentence --
        # this is the case rule 1 exists for.
        result.append(tail)
    return result


def sentence_lengths(text: str) -> list[int]:
    """Word count of each sentence, in order (plan.md §7 M1, sentence-length distribution).

    Sentences that contain no word tokens (a lone ``"—"``, a row of asterisks used as a
    scene break) contribute a 0 and are *not* dropped. Dropping them would make a
    passage's sentence count depend on its typographic furniture, and scene breaks are
    precisely the kind of thing a reviser deletes.
    """
    return [len(words(s)) for s in sentences(text)]


# --------------------------------------------------------------------------------------
# Punctuation
# --------------------------------------------------------------------------------------

#: Punctuation marks tracked in the profile, mapped to the class name reported.
#:
#: Classes rather than raw characters because the profile is compared across rounds, and
#: the question "does this author use dashes" is answerable while "does this author use
#: U+2014 specifically" is mostly a question about the typesetter. Folding happens first
#: (see TYPOGRAPHIC_FOLD), so only canonical forms need entries here.
PUNCTUATION_CLASSES = {
    ",": "comma",
    ";": "semicolon",
    ":": "colon",
    ".": "period",
    "!": "exclamation",
    "?": "question",
    "…": "ellipsis",
    "—": "em_dash",
    "–": "en_dash",
    "-": "hyphen",
    "'": "apostrophe_or_single_quote",
    '"': "double_quote",
    "(": "paren_open",
    ")": "paren_close",
    "¡": "inverted_exclamation",
    "¿": "inverted_question",
}

#: Every class name, so a profile always has the same keys and a zero is a measured zero
#: rather than a missing entry. A metric that compares dicts of different shapes across
#: rounds is a metric that silently ignores whatever the reviser removed entirely.
PUNCTUATION_CLASS_NAMES = tuple(sorted(set(PUNCTUATION_CLASSES.values())))


def punctuation_counts(text: str, *, fold: bool = True) -> dict[str, int]:
    """Count punctuation by class.

    Args:
        text: Raw passage text.
        fold: Fold typographic variants first (default). Pass ``False`` to count the
            characters as they literally appear, which is the right call when the
            question is whether a reviser re-typeset its input.

    Returns:
        A dict with one entry per name in :data:`PUNCTUATION_CLASS_NAMES`, always the
        same keys. Counts are absolute; :func:`revisionbench.metrics.stylometry.punctuation_profile`
        turns them into a per-1000-word rate.
    """
    prepared = normalise_newlines(text)
    if fold:
        prepared = fold_typography(prepared)
    tally: Counter[str] = Counter()
    for char in prepared:
        name = PUNCTUATION_CLASSES.get(char)
        if name is not None:
            tally[name] += 1
    return {name: tally.get(name, 0) for name in PUNCTUATION_CLASS_NAMES}
