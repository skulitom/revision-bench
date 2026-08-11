"""M6 — thrash and convergence (plan.md §7).

Three questions about a revision trajectory, all answered from the saved round texts:

1. **How much is each round changing?** Per-round edit volume, which arm A1 needs: the
   random-accept control accepts at a rate matched to A0's *realised* edit volume, so
   that number has to be measured rather than assumed.
2. **Is the loop undoing itself?** The fraction of round-k edits that are reverted or
   re-rewritten by round k+2. A loop with a fixed point stops editing; a loop without one
   keeps trading the same sentence back and forth, and that is visible here long before it
   is visible in a quality score.
3. **Does it stop?** Rounds to a fixed point, if one is reached.

The measurement rests on **sentence alignment**, and the alignment is the part worth
scrutinising. Revisers rewrite sentences rather than replacing them wholesale, so exact
matching finds almost nothing and would report every round as a total rewrite. Alignment
here is a similarity-scored monotonic alignment (Needleman–Wunsch over sentences), which
means a sentence that was lightly edited stays the *same slot* across rounds and can be
observed changing back.

Monotonic is doing real work: a greedy best-match pairing can cross, aligning sentence 3
of one version to sentence 9 of the next and vice versa. Prose does not reorder like that
under revision, and a crossing alignment inflates the "modified" count with pairs that are
merely both generic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from revisionbench.text import sentences, words

__all__ = [
    "MATCH_THRESHOLD",
    "REVERT_THRESHOLD",
    "AlignedPair",
    "EditReport",
    "ThrashReport",
    "align_sentences",
    "edit_report",
    "rounds_to_fixed_point",
    "sentence_similarity",
    "thrash_report",
]

#: Two sentences below this similarity are not the same slot; they are counted as one
#: deletion plus one insertion rather than as a modification. Chosen so that a lightly
#: edited sentence stays aligned while a wholly new sentence does not capture the slot of
#: whatever it replaced.
MATCH_THRESHOLD = 0.5

#: How similar a round-(k+2) sentence must be to its round-k form to count as a *revert*
#: rather than a further rewrite. Deliberately strict: "the loop put it back" is a strong
#: claim, and a loose threshold would let any two rewrites of the same sentence — which
#: necessarily share their subject matter and much of their vocabulary — read as one.
REVERT_THRESHOLD = 0.9


def sentence_similarity(a: str, b: str) -> float:
    """Similarity of two sentences in ``[0, 1]``, over word tokens.

    Token-level rather than character-level on purpose. At character level, "She walked to
    the door." and "She walked to the doors." are ~0.98 similar, and so are two unrelated
    sentences that happen to share function words and length; token level separates them
    better and is not moved at all by the typographic churn (curly quotes, ``--`` to em
    dash) that a reviser introduces for free, because :func:`revisionbench.text.words`
    folds that away.
    """
    ta, tb = words(a), words(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb, autojunk=False).ratio()


@dataclass(frozen=True, slots=True)
class AlignedPair:
    """One slot in an alignment between two versions.

    ``index_a`` is ``None`` for an insertion, ``index_b`` is ``None`` for a deletion, and
    both are set for a match (``similarity == 1.0``) or a modification.
    """

    index_a: int | None
    index_b: int | None
    similarity: float

    @property
    def op(self) -> str:
        if self.index_a is None:
            return "inserted"
        if self.index_b is None:
            return "deleted"
        return "unchanged" if self.similarity >= 1.0 else "modified"


def align_sentences(
    a: Sequence[str],
    b: Sequence[str],
    *,
    threshold: float = MATCH_THRESHOLD,
) -> list[AlignedPair]:
    """Monotonic similarity alignment between two sentence sequences.

    Needleman–Wunsch with a match score of ``similarity - threshold`` and a zero gap
    penalty. That scoring is what makes ``threshold`` meaningful: pairing two sentences
    that are less similar than the threshold scores negative, so the optimiser prefers a
    deletion plus an insertion, while any pairing above it is preferred to a gap.

    Returns the alignment in order, one entry per slot.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return []
    if n == 0:
        return [AlignedPair(None, j, 0.0) for j in range(m)]
    if m == 0:
        return [AlignedPair(i, None, 0.0) for i in range(n)]

    similarity = [[sentence_similarity(a[i], b[j]) for j in range(m)] for i in range(n)]

    # score[i][j] = best score aligning a[:i] with b[:j]. Gaps score 0, so the table is
    # non-decreasing and the traceback below cannot loop.
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score[i - 1][j - 1] + (similarity[i - 1][j - 1] - threshold)
            score[i][j] = max(diagonal, score[i - 1][j], score[i][j - 1])

    out: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diagonal = score[i - 1][j - 1] + (similarity[i - 1][j - 1] - threshold)
            if diagonal >= score[i - 1][j] and diagonal >= score[i][j - 1]:
                out.append(AlignedPair(i - 1, j - 1, similarity[i - 1][j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and (j == 0 or score[i - 1][j] >= score[i][j - 1]):
            out.append(AlignedPair(i - 1, None, 0.0))
            i -= 1
        else:
            out.append(AlignedPair(None, j - 1, 0.0))
            j -= 1
    out.reverse()
    return out


@dataclass(frozen=True, slots=True)
class EditReport:
    """Edit volume between two consecutive versions."""

    unchanged: int
    modified: int
    inserted: int
    deleted: int
    #: Sentences in the *earlier* version, the denominator for :attr:`edit_fraction`.
    sentences_before: int
    sentences_after: int
    #: Mean similarity over modified slots — how deep the edits were, as opposed to how
    #: many. A round that touches every sentence at similarity 0.97 and one that rewrites
    #: a third of them from scratch have very different edit fractions and can have the
    #: same effect on voice, so both numbers are reported.
    mean_modified_similarity: float | None

    @property
    def edits(self) -> int:
        return self.modified + self.inserted + self.deleted

    @property
    def edit_fraction(self) -> float:
        """Edits per sentence of the earlier version. 0.0 when that version was empty."""
        denominator = max(self.sentences_before, 1)
        return self.edits / denominator

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        data["edits"] = self.edits
        data["edit_fraction"] = self.edit_fraction
        return data


def edit_report(before: str, after: str, *, threshold: float = MATCH_THRESHOLD) -> EditReport:
    """Edit volume between two versions of a passage."""
    sa, sb = sentences(before), sentences(after)
    pairs = align_sentences(sa, sb, threshold=threshold)
    modified_similarities = [p.similarity for p in pairs if p.op == "modified"]
    return EditReport(
        unchanged=sum(1 for p in pairs if p.op == "unchanged"),
        modified=len(modified_similarities),
        inserted=sum(1 for p in pairs if p.op == "inserted"),
        deleted=sum(1 for p in pairs if p.op == "deleted"),
        sentences_before=len(sa),
        sentences_after=len(sb),
        mean_modified_similarity=(
            sum(modified_similarities) / len(modified_similarities)
            if modified_similarities
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class ThrashReport:
    """Thrash over one ``(k, k+1, k+2)`` window."""

    round_index: int
    #: Sentences modified at k -> k+1 that were touched again at k+1 -> k+2 and came back
    #: to within REVERT_THRESHOLD of their round-k form.
    reverted: int
    #: Modified again at k+1 -> k+2, into something new, while keeping its slot.
    rewritten_again: int
    #: Dropped at k+1 -> k+2 and a new sentence put in its place. This is a re-rewrite
    #: too, and it must be counted as one: an edited sentence that the next round replaces
    #: wholesale is the *most* churning thing a loop can do, and bucketing it with
    #: deletions would make an unstable loop read as a stable one.
    replaced: int
    #: Modified at k -> k+1 and left alone at k+1 -> k+2.
    settled: int
    #: Modified at k -> k+1 and cut at k+1 -> k+2 with nothing put in its place. Not
    #: thrash: the loop made a decision and kept it.
    deleted_after_edit: int
    #: Denominator: sentences modified at k -> k+1.
    edits_at_k: int

    @property
    def thrash_fraction(self) -> float | None:
        """Fraction of round-k edits reverted or re-rewritten by k+2.

        ``None`` when round k made no edits — there is nothing to thrash, and 0.0 would
        read as "the loop was stable" when the truth is "the loop had already stopped",
        which is the *opposite* finding.
        """
        if self.edits_at_k == 0:
            return None
        return (self.reverted + self.rewritten_again + self.replaced) / self.edits_at_k

    @property
    def revert_fraction(self) -> float | None:
        if self.edits_at_k == 0:
            return None
        return self.reverted / self.edits_at_k

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        data = asdict(self)
        data["thrash_fraction"] = self.thrash_fraction
        data["revert_fraction"] = self.revert_fraction
        return data


def thrash_report(
    versions: Sequence[str],
    *,
    threshold: float = MATCH_THRESHOLD,
    revert_threshold: float = REVERT_THRESHOLD,
) -> list[ThrashReport]:
    """Thrash over every ``(k, k+1, k+2)`` window of a revision trajectory.

    Args:
        versions: Round texts in order, ``versions[0]`` being round 0.
        threshold: Alignment threshold (see :func:`align_sentences`).
        revert_threshold: See :data:`REVERT_THRESHOLD`.

    Returns:
        One report per window, so ``len(versions) - 2`` entries. An empty list for a
        trajectory shorter than three rounds — with fewer than three versions there is no
        window in which a revert could be observed, which is different from observing no
        reverts.
    """
    if len(versions) < 3:
        return []
    reports: list[ThrashReport] = []
    for k in range(len(versions) - 2):
        reports.append(
            _thrash_window(
                versions[k],
                versions[k + 1],
                versions[k + 2],
                round_index=k,
                threshold=threshold,
                revert_threshold=revert_threshold,
            )
        )
    return reports


def _thrash_window(
    text_a: str,
    text_b: str,
    text_c: str,
    *,
    round_index: int,
    threshold: float,
    revert_threshold: float,
) -> ThrashReport:
    sa, sb, sc = sentences(text_a), sentences(text_b), sentences(text_c)
    first = align_sentences(sa, sb, threshold=threshold)
    second = align_sentences(sb, sc, threshold=threshold)

    # Position in `second` of the entry carrying each round-(k+1) sentence. Every index of
    # sb appears exactly once as an `index_a`, so this is total.
    onward_at: dict[int, int] = {
        p.index_a: i for i, p in enumerate(second) if p.index_a is not None
    }

    reverted = rewritten_again = replaced = settled = deleted_after_edit = 0
    edits_at_k = 0
    for pair in first:
        if pair.op != "modified":
            continue
        edits_at_k += 1
        assert pair.index_a is not None and pair.index_b is not None
        position = onward_at[pair.index_b]
        follow = second[position]

        if follow.index_b is None:
            # Dropped. Whether that is churn or a decision depends on whether something
            # took its place; an unrelated replacement is below the alignment threshold
            # and so shows up as a deletion beside an insertion rather than as a match.
            substitute = _adjacent_insertion(second, position)
            if substitute is None:
                deleted_after_edit += 1
                continue
            if sentence_similarity(sa[pair.index_a], sc[substitute]) >= revert_threshold:
                reverted += 1
            else:
                replaced += 1
            continue

        if follow.op == "unchanged":
            settled += 1
            continue

        # Edited again in place. Did it come back to where it started?
        if sentence_similarity(sa[pair.index_a], sc[follow.index_b]) >= revert_threshold:
            reverted += 1
        else:
            rewritten_again += 1

    return ThrashReport(
        round_index=round_index,
        reverted=reverted,
        rewritten_again=rewritten_again,
        replaced=replaced,
        settled=settled,
        deleted_after_edit=deleted_after_edit,
        edits_at_k=edits_at_k,
    )


def _adjacent_insertion(alignment: Sequence[AlignedPair], position: int) -> int | None:
    """Index in the later version of an insertion neighbouring ``position``, if any.

    A wholesale replacement appears in a monotonic alignment as a deletion sitting next to
    an insertion, because the new sentence is too dissimilar to capture the old one's slot.
    Looking only at immediate neighbours keeps this from pairing a deletion with an
    insertion elsewhere in the passage.
    """
    for neighbour in (position + 1, position - 1):
        if 0 <= neighbour < len(alignment):
            candidate = alignment[neighbour]
            if candidate.index_a is None and candidate.index_b is not None:
                return candidate.index_b
    return None


def rounds_to_fixed_point(versions: Sequence[str]) -> int | None:
    """Index of the first round whose text equals the previous round's, else ``None``.

    "Equal" is compared over the word-token stream, not the raw string, so a round that
    changes nothing but whitespace or quote style counts as a fixed point. That is the
    right call for plan.md §8's idempotence requirement — A5 must produce *no new
    proposals* on unchanged text, and re-typesetting is not a proposal — but it does mean
    a text can be at a fixed point here while differing byte-for-byte. Callers that need
    byte identity should compare the artifacts directly.

    Returns the index ``k`` such that ``versions[k]`` first equals ``versions[k-1]``;
    ``None`` if no round repeats its predecessor.
    """
    for index in range(1, len(versions)):
        if words(versions[index]) == words(versions[index - 1]):
            return index
    return None
