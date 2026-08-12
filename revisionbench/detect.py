"""Mechanical defect detection — the linter half of a detect-then-repair harness.

``docs/harness-gap.md`` argues that the product blocker is no longer safety of application
but aim: A2i applied ~24 edits per defect repaired, because a model invited to improve a
paragraph will always find something. The proposed inversion is that **only a span with a
located, checkable complaint is eligible to be edited**. This module emits those complaints.

Every detector here is mechanical — no model, no network. That matters for three reasons:
it is free, so it can run over a whole manuscript on every pass; it is deterministic, so a
complaint is reproducible and a repair can be *verified* by re-running it; and it is
checkable, so "the defect is fixed" becomes a re-run rather than a judgement.

---

**The circularity hazard, and what is done about it.**

These detectors are being written in a repo that also contains the injector that planted
the defects they are scored against. A detector written by inverting the injector would
score beautifully and mean nothing: it would detect *this* corpus's synthetic flaws and
nothing a real manuscript contains.

Two guards, and the second is the one that matters:

1. Each detector is written from a property of the *text* — a rare spelling variant of a
   frequent name, an n-gram repeated in a short window, a sentence whose tense disagrees
   with its neighbours — not from the injector's parameters. None of them import from
   :mod:`revisionbench.inject`, and that is enforced by a test.

2. **Recall against planted defects is an upper bound and is labelled as one.** The number
   that is not circular is *precision on clean prose*: run these detectors over the
   untouched Stratum-A passages, where by construction there is nothing planted, and every
   complaint is a false positive. That false-positive rate is the harness's overreach
   ceiling — it bounds how many spans a detect-then-repair loop would put in front of a
   model — and it is measured on text this module has never seen the corruption of.

A caveat on even that: Woolf and Richardson genuinely repeat phrases and genuinely shift
register, so a "false positive" on clean literary prose is sometimes the detector noticing
something real. The rate is therefore an upper bound on spurious complaints, not an exact
count of them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from revisionbench.text import normalise_newlines, sentence_spans, words

__all__ = [
    "DETECTORS",
    "Complaint",
    "detect_all",
    "detect_echo",
    "detect_name_variant",
    "detect_overlong_sentence",
    "detect_tense_outlier",
]


@dataclass(frozen=True, slots=True)
class Complaint:
    """One located, checkable objection to a span of text.

    ``span`` is what makes a repair scopeable and ``evidence`` is what makes the complaint
    arguable: a detector that says "this is wrong" without saying why cannot be verified
    as resolved, and an unverifiable complaint is just another invitation to rewrite.
    """

    type: str
    span: tuple[int, int]
    message: str
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["span"] = list(self.span)
        return data


# --------------------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------------------

_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

#: Capitalised words that are not names. Deliberately short: over-filtering here costs
#: recall, and a false positive is visible in the output while a false negative is not.
_NOT_NAMES = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Then",
        "They",
        "Their",
        "What",
        "When",
        "Where",
        "Why",
        "How",
        "Who",
        "Yes",
        "But",
        "And",
        "For",
        "Not",
        "All",
        "She",
        "Her",
        "His",
        "Him",
        "Its",
        "Was",
        "Were",
        "Had",
        "Has",
        "One",
        "Two",
        "Mrs",
        "Miss",
        "Sir",
        "God",
        "Oh",
        "Ah",
        "Well",
        "Now",
        "Here",
        "Some",
        "Such",
        "Even",
        "Only",
        "After",
        "Before",
        "While",
        "Still",
        "Yet",
        "Like",
        "From",
        "With",
        "Into",
        "Over",
        "Under",
        "Very",
        "Just",
        "Never",
        "Always",
        "Perhaps",
        "Indeed",
    }
)


def _edit_distance_at_most(a: str, b: str, limit: int) -> bool:
    """True if ``a`` and ``b`` are within ``limit`` single-character edits."""
    if abs(len(a) - len(b)) > limit:
        return False
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return False
        previous = current
    return previous[-1] <= limit


def detect_name_variant(text: str, *, min_frequent: int = 3, max_rare: int = 1) -> list[Complaint]:
    """A rare near-spelling of a frequently used proper name.

    The property exploited is statistical, not lexical: a manuscript that uses a name 12
    times and something within one edit of it once has almost certainly misspelled it. No
    dictionary and no knowledge of what the name *should* be is required, which is what
    lets this work on a book whose cast it has never seen.
    """
    counts: Counter[str] = Counter()
    positions: dict[str, list[int]] = defaultdict(list)
    for match in _NAME_RE.finditer(text):
        token = match.group()
        if token in _NOT_NAMES:
            continue
        counts[token] += 1
        positions[token].append(match.start())

    frequent = [name for name, n in counts.items() if n >= min_frequent]
    out: list[Complaint] = []
    for rare, n_rare in counts.items():
        if n_rare > max_rare:
            continue
        for common in frequent:
            if rare == common or not _edit_distance_at_most(rare.lower(), common.lower(), 1):
                continue
            at = positions[rare][0]
            out.append(
                Complaint(
                    type="name_variant",
                    span=(at, at + len(rare)),
                    message=f"{rare!r} appears {n_rare}x but {common!r} appears {counts[common]}x",
                    evidence=f"{rare} / {common}",
                )
            )
            break
    return out


def detect_echo(
    text: str, *, n: int = 3, min_count: int = 3, window: int = 1200
) -> list[Complaint]:
    """A distinctive n-gram repeated within a short span of text.

    Window-limited on purpose. A phrase recurring across a whole novel is a motif; the same
    phrase three times in two paragraphs is an echo, and the distinction is distance.
    """
    tokens: list[tuple[str, int]] = []
    for match in re.finditer(r"\b[\w'-]+\b", text):
        tokens.append((match.group().lower(), match.start()))
    if len(tokens) < n:
        return []

    grams: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for i in range(len(tokens) - n + 1):
        gram = tuple(t for t, _ in tokens[i : i + n])
        grams[gram].append(tokens[i][1])

    out: list[Complaint] = []
    for gram, offsets in grams.items():
        if len(offsets) < min_count or all(len(g) <= 3 for g in gram):
            continue
        for start in range(len(offsets) - min_count + 1):
            cluster = offsets[start : start + min_count]
            if cluster[-1] - cluster[0] <= window:
                phrase = " ".join(gram)
                out.append(
                    Complaint(
                        type="echo",
                        span=(cluster[0], cluster[0] + len(phrase)),
                        message=f"{phrase!r} occurs {len(offsets)}x, {min_count}x within "
                        f"{cluster[-1] - cluster[0]} characters",
                        evidence=phrase,
                    )
                )
                break
    return out


_PAST = re.compile(r"\b(was|were|had|did|could|would|should)\b", re.IGNORECASE)
_PRESENT = re.compile(r"\b(is|are|has|does|can|will|shall)\b", re.IGNORECASE)


def detect_tense_outlier(text: str, *, min_words: int = 8) -> list[Complaint]:
    """A sentence in the present tense inside a passage that is otherwise past tense.

    Decided by the passage's own majority rather than by an assumption about narration, so
    it works on a present-tense manuscript too — there it flags the past-tense stragglers.
    """
    spans = [(s, e) for s, e in sentence_spans(text) if len(words(text[s:e])) >= min_words]
    if len(spans) < 4:
        return []

    scored = []
    past_total = present_total = 0
    for start, end in spans:
        sentence = text[start:end]
        past = len(_PAST.findall(sentence))
        present = len(_PRESENT.findall(sentence))
        scored.append((start, end, past, present))
        past_total += past
        present_total += present

    if past_total + present_total == 0:
        return []
    passage_is_past = past_total >= present_total

    out: list[Complaint] = []
    for start, end, past, present in scored:
        # Only an unambiguous outlier: the sentence must lean the wrong way with no support
        # for the majority tense at all. Sentences mixing both are ordinary English.
        if passage_is_past and present > 0 and past == 0:
            out.append(
                Complaint(
                    type="tense_outlier",
                    span=(start, end),
                    message=f"present-tense sentence ({present} marker(s)) in a past-tense "
                    f"passage ({past_total} past vs {present_total} present overall)",
                    evidence=text[start : min(end, start + 90)],
                )
            )
        elif not passage_is_past and past > 0 and present == 0:
            out.append(
                Complaint(
                    type="tense_outlier",
                    span=(start, end),
                    message="past-tense sentence in a present-tense passage",
                    evidence=text[start : min(end, start + 90)],
                )
            )
    return out


_HEDGE = re.compile(
    r"\b(somewhat|rather|quite|perhaps|possibly|considerable|certain|various|things"
    r"|matter|situation|case|manner|degree|respect|regard)\b",
    re.IGNORECASE,
)


def detect_overlong_sentence(
    text: str, *, length_multiple: float = 2.2, min_hedges: int = 4
) -> list[Complaint]:
    """A sentence far longer than its neighbours *and* dense in hedging abstraction.

    Length alone is not a defect — Woolf's sentences run to 100 words and are the point.
    The conjunction is what distinguishes a long sentence from a bloated one: the flagged
    sentence must be both an outlier against *this passage's* own distribution and packed
    with the vague nouns and qualifiers that mark padding rather than syntax.
    """
    spans = sentence_spans(text)
    if len(spans) < 4:
        return []
    lengths = [len(words(text[s:e])) for s, e in spans]
    typical = sorted(lengths)[len(lengths) // 2]
    if typical == 0:
        return []

    out: list[Complaint] = []
    for (start, end), length in zip(spans, lengths, strict=True):
        if length < typical * length_multiple:
            continue
        hedges = len(_HEDGE.findall(text[start:end]))
        if hedges < min_hedges:
            continue
        out.append(
            Complaint(
                type="overlong_sentence",
                span=(start, end),
                message=f"{length} words against a passage median of {typical}, with "
                f"{hedges} hedging/abstract terms",
                evidence=text[start : min(end, start + 90)],
            )
        )
    return out


DETECTORS = {
    "name_variant": detect_name_variant,
    "echo": detect_echo,
    "tense_outlier": detect_tense_outlier,
    "overlong_sentence": detect_overlong_sentence,
}


def detect_all(text: str, *, types: list[str] | None = None) -> list[Complaint]:
    """Run every detector and return complaints ordered by position.

    Raises:
        ValueError: An unknown detector name. A typo must not silently reduce coverage —
            a detector that did not run looks exactly like a detector that found nothing.
    """
    chosen = list(types or DETECTORS)
    unknown = [t for t in chosen if t not in DETECTORS]
    if unknown:
        raise ValueError(f"unknown detector(s): {unknown}; known: {sorted(DETECTORS)}")
    normalised = normalise_newlines(text)
    out: list[Complaint] = []
    for name in chosen:
        out.extend(DETECTORS[name](normalised))
    return sorted(out, key=lambda c: c.span)
