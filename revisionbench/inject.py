"""Planted-defect injection — Stratum B (plan.md §6).

Injecting known flaws into known text is what converts the fitness question from pure
aesthetics into partial ground truth. Without it, "this loop preserves voice" is
unfalsifiable: an architecture that blocks every edit preserves voice perfectly and fixes
nothing, and plan.md §6 is explicit that such a gate must be rejected on those grounds. The
number this module exists to make measurable is the *other* axis — defect-fix recall.

Design constraints, in order of how much damage getting them wrong would do:

**Ground truth must be exact, not approximate.** Every defect records the precise fragment
it replaced, the precise fragment it inserted, and the span. Nothing about scoring should
require re-reading the passage and forming a judgement.

**Injection must be deterministic.** A seeded RNG per passage, and an ``injector_version``
stamped on every defect, so the corpus regenerates byte-identically and a change to the
injector is detectable rather than merely confusing.

**A defect must be locatable after the text has been rewritten.** The reviser will not leave
the surrounding sentence intact, so a defect that could only be found by exact string search
would be unscoreable the moment anything nearby changed. Each defect therefore also stores
its *clean* containing sentence, which :mod:`revisionbench.metrics.defects` matches fuzzily
against the revised text to decide whether the content survived at all.

**The corruption must not be trivially detectable.** plan.md §12.4 flags this as an
unverified assumption: if a model can spot the injection by pattern rather than by
understanding, recall measures pattern-matching. The injectors here mutate existing material
rather than appending flagged text, and ``scripts/inject_defects.py`` supports a blind
locate probe for checking it.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from revisionbench.text import normalise_newlines, sentence_spans, words

__all__ = [
    "DEFECT_TYPES",
    "INJECTOR_VERSION",
    "Defect",
    "InjectionError",
    "inject_passage",
]

#: Bumped whenever a change here could alter what gets injected or where. Stamped on every
#: defect, so a manifest built by two different versions is detectable.
INJECTOR_VERSION = 1


class InjectionError(RuntimeError):
    """A passage does not admit the requested defect (too short, no candidates, ...)."""


@dataclass(frozen=True, slots=True)
class Defect:
    """One planted flaw, with everything scoring needs.

    Attributes:
        detect_kind: How a fix is recognised.

            - ``marker`` — ``corrupt_fragment`` must no longer appear.
            - ``count`` — ``marker`` must appear at most ``target_count`` times.
            - ``span`` — the inserted sentence must be gone or substantially rewritten.
            - ``minimal_pair`` — the surviving text must resemble the clean version more
              than the corrupted one (a tense slip differs by one word).

        clean_sentence: The containing sentence *before* corruption. Scoring matches this
            fuzzily against the revised text; if nothing resembles it, the region was cut
            and the defect is censored rather than counted as fixed. This field is the one
            that stops a reviser's deletions from reading as defect repairs.
    """

    defect_id: str
    passage_id: str
    type: str
    original_fragment: str
    corrupt_fragment: str
    char_span: tuple[int, int]
    clean_sentence: str
    detect_kind: str
    marker: str = ""
    target_count: int = 1
    injector_version: int = INJECTOR_VERSION
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["char_span"] = list(self.char_span)
        return data


def _overlaps(span: tuple[int, int], forbidden: Sequence[tuple[int, int]]) -> bool:
    """True if ``span`` intersects any already-corrupted region."""
    start, end = span
    return any(start < f_end and f_start < end for f_start, f_end in forbidden)


def _forbidden_spans(text: str, fragments: Sequence[str]) -> list[tuple[int, int]]:
    """Locate previously injected fragments in the current text.

    Protection is re-derived by content rather than tracked as offsets, because every
    injection shifts the offsets of everything after it and the bookkeeping is exactly the
    kind of thing that silently goes stale.
    """
    spans: list[tuple[int, int]] = []
    for fragment in fragments:
        if not fragment:
            continue
        at = text.find(fragment)
        while at != -1:
            spans.append((at, at + len(fragment)))
            at = text.find(fragment, at + 1)
    return spans


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

#: Words that look like proper nouns to a capitalisation heuristic but are not.
_NOT_NAMES = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "but",
        "or",
        "for",
        "he",
        "she",
        "it",
        "they",
        "we",
        "you",
        "i",
        "his",
        "her",
        "their",
        "there",
        "then",
        "this",
        "that",
        "these",
        "those",
        "what",
        "when",
        "where",
        "why",
        "how",
        "who",
        "yes",
        "no",
        "oh",
        "ah",
        "well",
        "mr",
        "mrs",
        "miss",
        "dr",
        "sir",
        "lady",
        "lord",
        "god",
        "one",
        "two",
        "so",
        "if",
        "in",
        "on",
        "at",
        "of",
        "to",
        "was",
        "were",
        "had",
        "has",
        "did",
        "not",
        "all",
        "some",
        "such",
        "here",
        "now",
        "up",
        "down",
        "out",
        "over",
        "very",
        "as",
        "by",
        "from",
        "with",
        "into",
        "like",
        "after",
        "before",
        "while",
        "still",
        "yet",
        "even",
    }
)

_WORD_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

#: Deterministic near-miss spellings, tried in order. The first that changes the name and
#: does not collide with a word already in the passage wins.
_MISSPELLINGS: tuple[tuple[str, str], ...] = (
    ("K", "C"),
    ("C", "K"),
    ("ph", "f"),
    ("ie", "ei"),
    ("ei", "ie"),
    ("y", "ey"),
    ("ll", "l"),
    ("tt", "t"),
    ("nn", "n"),
    ("ss", "s"),
)

#: Bloated sentences for the `clunker` defect. Deliberately tangled rather than merely long,
#: and written to be era-neutral so they do not stand out as modern.
_CLUNKERS: tuple[str, ...] = (
    "It was the case that the situation, which had been developing for some considerable "
    "time and in a manner that few if any of those present could have been expected to "
    "anticipate with any degree of confidence, was one which now seemed, on balance and "
    "all things considered, to be somewhat difficult.",
    "There was a feeling, not exactly of unease and not exactly of anything else that "
    "could readily be named or described in terms that would satisfy a person of ordinary "
    "sensibility, which had settled upon the whole of the proceedings in a way that was "
    "difficult to account for.",
    "The thing about the matter, and it was a thing that had been remarked upon before by "
    "others who had occasion to consider it at length and from a variety of angles, was "
    "that it admitted of no very simple explanation of the kind that might have been "
    "hoped for.",
)


def _distinct_names(text: str, minimum: int = 3) -> list[str]:
    """Capitalised words occurring at least ``minimum`` times that look like names."""
    counts: dict[str, int] = {}
    for match in _WORD_TOKEN_RE.finditer(text):
        token = match.group()
        if token.lower() in _NOT_NAMES:
            continue
        counts[token] = counts.get(token, 0) + 1
    return sorted([name for name, n in counts.items() if n >= minimum])


def _containing_sentence(text: str, index: int) -> str:
    for start, end in sentence_spans(text):
        if start <= index < end:
            return text[start:end]
    return ""


# --------------------------------------------------------------------------------------
# Injectors. Each returns (corrupted_text, Defect) or raises InjectionError.
# --------------------------------------------------------------------------------------


def _inject_name_drift(
    text: str,
    rng: random.Random,
    passage_id: str,
    index: int,
    forbidden: Sequence[tuple[int, int]] = (),
) -> tuple[str, Defect]:
    """Misspell exactly one occurrence of a recurring proper name.

    The commonest real continuity error in a manuscript, and the cleanest to score: the
    misspelling either survives or it does not.
    """
    names = _distinct_names(text)
    if not names:
        raise InjectionError("no recurring proper name to drift")
    name = rng.choice(names)

    variant = ""
    for old, new in _MISSPELLINGS:
        if old in name:
            candidate = name.replace(old, new, 1)
            if candidate != name and candidate not in text:
                variant = candidate
                break
    if not variant:
        raise InjectionError(f"no near-miss spelling found for {name!r}")

    occurrences = [
        m.start()
        for m in re.finditer(re.escape(name), text)
        if not _overlaps((m.start(), m.end()), forbidden)
    ]
    if len(occurrences) < 2:
        raise InjectionError(f"no free later occurrence of {name!r}")
    # Never the first occurrence: a name is established on first use, and drifting it there
    # reads as the correct spelling rather than as an error.
    at = rng.choice(occurrences[1:])
    corrupted = text[:at] + variant + text[at + len(name) :]
    return corrupted, Defect(
        defect_id=f"{passage_id}-d{index}",
        passage_id=passage_id,
        type="name_drift",
        original_fragment=name,
        corrupt_fragment=variant,
        char_span=(at, at + len(variant)),
        clean_sentence=_containing_sentence(text, at),
        detect_kind="marker",
        marker=variant,
        notes=f"{name} -> {variant} at occurrence {occurrences.index(at) + 1}",
    )


def _inject_tense_slip(
    text: str,
    rng: random.Random,
    passage_id: str,
    index: int,
    forbidden: Sequence[tuple[int, int]] = (),
) -> tuple[str, Defect]:
    """Shift one sentence from past to present by swapping was/were to is/are.

    Restricted to the copula on purpose: it is exactly reversible, needs no conjugator, and
    the resulting sentence is grammatical — so the defect is a *consistency* error rather
    than a broken sentence a model could flag without reading anything around it.
    """
    spans = [(s, e) for s, e in sentence_spans(text) if len(words(text[s:e])) >= 8]
    candidates = [
        (s, e)
        for s, e in spans
        if re.search(r"\b(was|were)\b", text[s:e]) and not _overlaps((s, e), forbidden)
    ]
    if not candidates:
        raise InjectionError("no free sentence with a past-tense copula")
    start, end = rng.choice(candidates)
    sentence = text[start:end]

    def swap(match: re.Match[str]) -> str:
        return {"was": "is", "were": "are"}[match.group().lower()]

    corrupted_sentence = re.sub(r"\b(was|were)\b", swap, sentence)
    if corrupted_sentence == sentence:
        raise InjectionError("tense swap produced no change")
    corrupted = text[:start] + corrupted_sentence + text[end:]
    return corrupted, Defect(
        defect_id=f"{passage_id}-d{index}",
        passage_id=passage_id,
        type="tense_slip",
        original_fragment=sentence,
        corrupt_fragment=corrupted_sentence,
        char_span=(start, start + len(corrupted_sentence)),
        clean_sentence=sentence,
        # A minimal pair, not a span: the corrupted and clean sentences differ by one or
        # two words and are ~0.97 similar, so a span check cannot tell them apart and the
        # swapped word is far too common to serve as a marker. Scoring is comparative.
        detect_kind="minimal_pair",
        notes="past -> present copula for one sentence",
    )


def _inject_clunker(
    text: str,
    rng: random.Random,
    passage_id: str,
    index: int,
    forbidden: Sequence[tuple[int, int]] = (),
) -> tuple[str, Defect]:
    """Insert one bloated, tangled sentence at a sentence boundary."""
    spans = sentence_spans(text)
    if len(spans) < 4:
        raise InjectionError("passage too short to place a clunker")
    clunker = rng.choice(_CLUNKERS)
    # Not the first or last boundary: an inserted sentence at the very edge of a passage
    # reads as a framing device rather than as a flaw.
    free = [(s, e) for s, e in spans[1:-1] if not _overlaps((s, e), forbidden)]
    if not free:
        raise InjectionError("no free sentence boundary for a clunker")
    host_start, boundary = rng.choice(free)
    corrupted = text[:boundary] + " " + clunker + text[boundary:]
    return corrupted, Defect(
        defect_id=f"{passage_id}-d{index}",
        passage_id=passage_id,
        type="clunker",
        original_fragment="",
        corrupt_fragment=clunker,
        char_span=(boundary + 1, boundary + 1 + len(clunker)),
        # The sentence the clunker was inserted *after*, not the clunker itself.
        #
        # This is an insertion, so there is no original region that could go missing — and
        # setting clean_sentence to the clunker would make the survival check ask "does the
        # clean text contain this bloated sentence", which it never does. Every clunker
        # would then score `removed` even against a perfect repair. Pointing at the
        # untouched neighbour restores the intended question: did the surrounding material
        # survive, so that losing the clunker is a repair rather than collateral damage?
        clean_sentence=text[host_start:boundary],
        detect_kind="span",
        notes="inserted bloated sentence after an existing sentence",
    )


def _inject_echo(
    text: str,
    rng: random.Random,
    passage_id: str,
    index: int,
    forbidden: Sequence[tuple[int, int]] = (),
) -> tuple[str, Defect]:
    """Repeat a distinctive phrase so it appears three times instead of once."""
    spans = sentence_spans(text)
    long_enough = [
        (s, e) for s, e in spans if len(words(text[s:e])) >= 12 and not _overlaps((s, e), forbidden)
    ]
    if len(long_enough) < 3:
        raise InjectionError("passage too short to plant an echo")

    source = rng.choice(long_enough)
    tokens = text[source[0] : source[1]].split()
    if len(tokens) < 8:
        raise InjectionError("source sentence too short")
    at = rng.randrange(1, max(2, len(tokens) - 4))
    phrase = " ".join(tokens[at : at + 3]).strip(",;:—-\"'")
    if len(phrase) < 10 or text.count(phrase) != 1:
        raise InjectionError("no distinctive one-off phrase available")

    targets = [span for span in long_enough if span != source]
    rng.shuffle(targets)
    chosen = sorted(targets[:2])
    if len(chosen) < 2:
        raise InjectionError("not enough target sentences for an echo")

    # Insert back-to-front so earlier offsets stay valid.
    corrupted = text
    for start, end in reversed(chosen):
        sentence = corrupted[start:end]
        pieces = sentence.split(" ", 3)
        if len(pieces) < 4:
            continue
        rebuilt = " ".join(pieces[:2]) + f", {phrase}, " + " ".join(pieces[2:])
        corrupted = corrupted[:start] + rebuilt + corrupted[end:]

    if corrupted.count(phrase) < 3:
        raise InjectionError("echo insertion did not take")
    return corrupted, Defect(
        defect_id=f"{passage_id}-d{index}",
        passage_id=passage_id,
        type="echo",
        original_fragment=phrase,
        corrupt_fragment=phrase,
        char_span=(chosen[0][0], chosen[-1][1]),
        clean_sentence=text[source[0] : source[1]],
        detect_kind="count",
        marker=phrase,
        target_count=1,
        notes=f"phrase {phrase!r} repeated to 3 occurrences",
    )


def _inject_number_drift(
    text: str,
    rng: random.Random,
    passage_id: str,
    index: int,
    forbidden: Sequence[tuple[int, int]] = (),
) -> tuple[str, Defect]:
    """Change one occurrence of a repeated number word, creating a contradiction."""
    number_words = (
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "twenty",
        "thirty",
        "fifty",
        "hundred",
    )
    replacement = {
        "one": "two",
        "two": "three",
        "three": "four",
        "four": "five",
        "five": "six",
        "six": "seven",
        "seven": "eight",
        "eight": "nine",
        "nine": "ten",
        "ten": "twelve",
        "twenty": "thirty",
        "thirty": "forty",
        "fifty": "sixty",
        "hundred": "thousand",
    }
    lowered = text.lower()
    candidates = [w for w in number_words if len(re.findall(rf"\b{w}\b", lowered)) >= 2]
    if not candidates:
        raise InjectionError("no repeated number word")
    word = rng.choice(candidates)
    occurrences = [
        m.start()
        for m in re.finditer(rf"\b{word}\b", lowered)
        if not _overlaps((m.start(), m.end()), forbidden)
    ]
    if len(occurrences) < 2:
        raise InjectionError(f"no free later occurrence of {word!r}")
    at = rng.choice(occurrences[1:])
    new = replacement[word]
    corrupted = text[:at] + new + text[at + len(word) :]
    return corrupted, Defect(
        defect_id=f"{passage_id}-d{index}",
        passage_id=passage_id,
        type="number_drift",
        original_fragment=word,
        corrupt_fragment=new,
        char_span=(at, at + len(new)),
        clean_sentence=_containing_sentence(text, at),
        detect_kind="marker",
        marker=new,
        notes=f"{word} -> {new}, contradicting an earlier mention",
    )


DEFECT_TYPES: dict[str, Any] = {
    "name_drift": _inject_name_drift,
    "tense_slip": _inject_tense_slip,
    "clunker": _inject_clunker,
    "echo": _inject_echo,
    "number_drift": _inject_number_drift,
}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class InjectionResult:
    text: str
    defects: list[Defect] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def inject_passage(
    text: str,
    passage_id: str,
    *,
    seed: int,
    types: list[str] | None = None,
    target: int = 4,
) -> InjectionResult:
    """Plant up to ``target`` defects in one passage, deterministically.

    Types are attempted in a seeded shuffle. A type that cannot apply to this passage — no
    recurring name, no repeated number — is **skipped and recorded**, never silently
    dropped: a passage that ended up with two defects instead of four has a different
    denominator, and a recall figure computed against the intended count would be wrong.

    Args:
        text: Clean Stratum-A passage text.
        passage_id: Used to build defect ids.
        seed: Per-passage RNG seed.
        types: Which injectors to consider. Defaults to all.
        target: How many defects to plant (plan.md §6 says 3–6).

    Returns:
        The corrupted text, the defects actually planted, and the types skipped.

    Raises:
        InjectionError: Nothing could be planted at all.
    """
    chosen = list(types or DEFECT_TYPES)
    unknown = [t for t in chosen if t not in DEFECT_TYPES]
    if unknown:
        raise InjectionError(f"unknown defect type(s): {unknown}; known: {sorted(DEFECT_TYPES)}")

    rng = random.Random(seed)
    order = list(chosen)
    rng.shuffle(order)

    current = normalise_newlines(text)
    result = InjectionResult(text=current)
    index = 1
    # Fragments already injected. Later injectors must not corrupt them: a tense slip
    # applied to an inserted clunker is a defect inside a defect, and the ground truth for
    # both stops being exact.
    protected: list[str] = []
    for defect_type in order:
        if len(result.defects) >= target:
            break
        try:
            current, defect = DEFECT_TYPES[defect_type](
                current, rng, passage_id, index, _forbidden_spans(current, protected)
            )
        except InjectionError as exc:
            result.skipped.append(f"{defect_type}: {exc}")
            continue
        result.defects.append(defect)
        if defect.corrupt_fragment:
            protected.append(defect.corrupt_fragment)
        index += 1

    if not result.defects:
        raise InjectionError(
            f"{passage_id}: no defect could be planted. Skipped: {'; '.join(result.skipped)}"
        )
    result.text = current
    result.defects = _reanchor_spans(current, result.defects)
    return result


def _reanchor_spans(text: str, defects: list[Defect]) -> list[Defect]:
    """Re-point every span at the final text.

    Defects are planted one after another, so each injection shifts the offsets of
    everything after it and leaves earlier spans stale. Left alone, a manifest built with
    four defects had its second one pointing at unrelated text — which does not change any
    score (scoring matches on markers and sentences, not offsets) but does defeat the
    hand-verification plan.md §6 asks for, since the reviewer is shown the wrong words.

    Each span is re-anchored to the occurrence of its corrupt fragment nearest the recorded
    position; nearest rather than first, because a fragment like a number word occurs many
    times and the first one is usually not ours. ``count`` defects (echo) span a range
    rather than a fragment and are left as recorded, which the notes field says.
    """
    from dataclasses import replace as replace_field

    out: list[Defect] = []
    for defect in defects:
        fragment = defect.corrupt_fragment
        if defect.detect_kind == "count" or not fragment:
            out.append(defect)
            continue
        positions = [m for m in range(len(text)) if text.startswith(fragment, m)] or []
        if not positions:
            out.append(replace_field(defect, notes=f"{defect.notes} [span unresolved]"))
            continue
        nearest = min(positions, key=lambda p: abs(p - defect.char_span[0]))
        out.append(replace_field(defect, char_span=(nearest, nearest + len(fragment))))
    return out
