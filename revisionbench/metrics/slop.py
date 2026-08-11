"""M3 — the slop index (plan.md §7).

Frequency per 1000 words of a versioned LLM-tell lexicon, with a per-term breakdown.

The per-term breakdown is not a debugging nicety. An aggregate rate that rises across
rounds is uninterpretable on its own: it could be a reviser adding "a wave of" to every
paragraph, or it could be one term in the lexicon that happens to be ordinary English in
this corpus drifting by chance. plan.md §11's rule about reporting the whole surface
rather than the summary applies here, so :func:`slop_index` always returns the
attribution alongside the number, and callers are expected to persist both.

Matching runs over the tokenised word stream from :mod:`revisionbench.text`, which means
it is immune to the typographic churn a reviser introduces — ``"Delve"``, ``delve`` and
``delve,`` are one hit — and it means a term's tokens must be written the way that module
tokenises them (``"it's"`` is one token, not two).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from revisionbench.text import words

__all__ = [
    "DEFAULT_LEXICON_PATH",
    "WILDCARD",
    "SlopLexicon",
    "SlopReport",
    "load_lexicon",
    "slop_index",
]

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "slop_lexicon.yaml"

#: Matches exactly one token of any kind. Kept to a single token deliberately: a
#: variable-length wildcard turns the matcher into a regex engine over token streams and
#: makes "leftmost-longest" ambiguous, which is how double counting gets in.
WILDCARD = "*"


class SlopError(RuntimeError):
    """The lexicon file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class SlopLexicon:
    """A loaded, versioned lexicon.

    Attributes:
        version: The file's version field, stamped onto every result that uses it. A slop
            rate is only comparable to another slop rate from the same lexicon version.
        terms: ``term -> group name``, where a term is a tuple of tokens.
        groups: Group name -> its ``source`` string, so a report can say where each
            contribution came from without re-reading the file.
    """

    version: int
    terms: dict[tuple[str, ...], str]
    groups: dict[str, str]

    @property
    def max_term_tokens(self) -> int:
        return max((len(t) for t in self.terms), default=0)


@dataclass(frozen=True, slots=True)
class SlopReport:
    """The result of scoring one text."""

    #: Hits per 1000 word tokens — the headline M3 number.
    per_1000_words: float
    #: Raw hit count.
    hits: int
    #: Denominator, recorded so a rate can be un-normalised without re-tokenising.
    word_count: int
    #: Hit count per term that fired at least once, keyed by the term as written.
    by_term: dict[str, int]
    #: Hit count per group.
    by_group: dict[str, int]
    #: The lexicon version these numbers came from.
    lexicon_version: int

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def load_lexicon(path: str | Path | None = None) -> SlopLexicon:
    """Load and validate the slop lexicon.

    Raises:
        SlopError: The file is missing or malformed, a group lacks a ``source``, or a term
            appears in two groups. The last one matters: a duplicated term would be
            attributed to whichever group happened to load second, and its group total
            would be wrong while the overall total stayed right — a discrepancy nobody
            would notice.
    """
    import yaml

    target = Path(path) if path is not None else DEFAULT_LEXICON_PATH
    if not target.is_file():
        raise SlopError(f"slop lexicon not found: {target}")
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data or "groups" not in data:
        raise SlopError(f"{target}: expected a mapping with 'version' and 'groups' keys")

    terms: dict[tuple[str, ...], str] = {}
    groups: dict[str, str] = {}
    for index, group in enumerate(data["groups"]):
        if not isinstance(group, dict):
            raise SlopError(f"{target}: groups[{index}] is not a mapping")
        missing = [k for k in ("name", "source", "terms") if k not in group]
        if missing:
            raise SlopError(
                f"{target}: groups[{index}] is missing {', '.join(missing)}. Every group "
                f"needs a source; an uncited lexicon group reports a rate that looks as "
                f"authoritative as a cited one."
            )
        name = str(group["name"])
        if name in groups:
            raise SlopError(f"{target}: duplicate group name {name!r}")
        groups[name] = str(group["source"])
        for raw in group["terms"]:
            if not isinstance(raw, str):
                raise SlopError(
                    f"{target}: group {name!r} has a non-string term {raw!r} (YAML 1.1 "
                    f"parses bare no/yes/on/off as booleans; quote them)"
                )
            tokens = _tokenise_term(raw)
            if not tokens:
                raise SlopError(f"{target}: group {name!r} has an empty term {raw!r}")
            if tokens.count(WILDCARD) > 1:
                raise SlopError(
                    f"{target}: term {raw!r} has {tokens.count(WILDCARD)} wildcards; the "
                    f"matcher substitutes one position at a time, so a multi-wildcard "
                    f"term would never fire and would look like a term that simply does "
                    f"not occur"
                )
            if all(t == WILDCARD for t in tokens):
                raise SlopError(f"{target}: term {raw!r} is nothing but wildcards")
            if tokens in terms:
                raise SlopError(
                    f"{target}: term {raw!r} appears in both {terms[tokens]!r} and "
                    f"{name!r}; group totals would silently disagree with the overall total"
                )
            terms[tokens] = name
    if not terms:
        raise SlopError(f"{target}: lexicon contains no terms")
    return SlopLexicon(version=int(data["version"]), terms=terms, groups=groups)


def _tokenise_term(term: str) -> tuple[str, ...]:
    """Tokenise a lexicon term the same way a passage is tokenised, keeping wildcards."""
    out: list[str] = []
    for piece in term.split():
        if piece == WILDCARD:
            out.append(WILDCARD)
            continue
        # A term is written as prose ("it's important to note"), so run it through the
        # real tokeniser rather than splitting on spaces. This is what guarantees a term
        # can actually match: if the tokeniser turns "it's" into one token, the term must
        # too, and deriving it here means the two can never drift apart.
        out.extend(words(piece))
    return tuple(out)


def slop_index(text: str, lexicon: SlopLexicon) -> SlopReport:
    """Score one text against the lexicon.

    Matching is **leftmost-longest and non-overlapping**: at each token position the
    longest matching term wins, and its tokens are consumed before scanning resumes. That
    rule is what makes the lexicon safe to extend — "sense of unease" and "sense of" can
    both be listed without the first counting twice.

    Raises:
        ValueError: The text has no word tokens, so there is no denominator. A rate of 0.0
            would be indistinguishable from a genuinely clean text.
    """
    tokens = words(text)
    total = len(tokens)
    if total == 0:
        raise ValueError(
            "cannot compute a slop rate for a text with no word tokens; record the round "
            "as a metric failure rather than as a rate of zero"
        )

    by_term: dict[str, int] = {}
    by_group: dict[str, int] = dict.fromkeys(lexicon.groups, 0)
    hits = 0
    longest = lexicon.max_term_tokens

    position = 0
    while position < total:
        matched = _longest_match_at(tokens, position, lexicon, longest)
        if matched is None:
            position += 1
            continue
        term_tokens, group = matched
        label = " ".join(term_tokens)
        by_term[label] = by_term.get(label, 0) + 1
        by_group[group] += 1
        hits += 1
        position += len(term_tokens)

    return SlopReport(
        per_1000_words=hits * 1000.0 / total,
        hits=hits,
        word_count=total,
        by_term=dict(sorted(by_term.items())),
        by_group=by_group,
        lexicon_version=lexicon.version,
    )


def _longest_match_at(
    tokens: Sequence[str],
    position: int,
    lexicon: SlopLexicon,
    longest: int,
) -> tuple[tuple[str, ...], str] | None:
    """Longest lexicon term matching at ``position``, or ``None``."""
    limit = min(longest, len(tokens) - position)
    for length in range(limit, 0, -1):
        window = tuple(tokens[position : position + length])
        group = lexicon.terms.get(window)
        if group is not None:
            return window, group
        # Wildcard terms cannot be looked up directly, so try them by substitution: for a
        # window of length n there are at most n single-token wildcard variants.
        for i in range(length):
            candidate = (*window[:i], WILDCARD, *window[i + 1 :])
            group = lexicon.terms.get(candidate)
            if group is not None:
                return candidate, group
    return None
