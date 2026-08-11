"""M5 — defect-fix recall and overreach (plan.md §7).

This is the second axis of the frontier. Without it, "architecture X preserves voice" is
unfalsifiable, because an architecture that blocks every edit preserves voice perfectly;
plan.md §6 says such a gate scores zero recall and is correctly rejected, and this module
is what makes that sentence operational.

**The trap this module is built around.** The obvious way to score a fix is "the corruption
is no longer present". On this corpus that would be catastrophic. Phase 0 measured revisers
cutting 40–50% of a passage, so a defect can vanish simply because the paragraph containing
it was deleted — and naive scoring would credit that as a repair. The unconstrained arm,
which is the *most* destructive, would post the best defect recall in the study.

So every defect resolves to one of four states, and deletion is not one of the good ones:

``fixed``
    The region survived — a sentence resembling the defect's clean containing sentence is
    still present — and the corruption is gone. This is a repair.
``present``
    The region survived and the corruption is still there. A genuine miss.
``removed``
    Nothing in the revised text resembles the region. The content was cut. **Censored, not
    counted as either a fix or a miss**, and reported separately so a reader can see how
    much of the passage was scoreable at all.
``unscoreable``
    The revised text has no sentences to match against.

Recall is therefore computed over surviving defects only, and always reported alongside the
removal rate. A high recall on a passage where most content was removed is not a good
result; it is a small denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from revisionbench.metrics.thrash import sentence_similarity
from revisionbench.text import sentences

__all__ = [
    "SURVIVAL_THRESHOLD",
    "DefectOutcome",
    "RecallReport",
    "defect_outcome",
    "recall_report",
]

#: How similar a revised sentence must be to a defect's clean containing sentence for the
#: region to count as having survived. Deliberately loose: the reviser is *supposed* to
#: rewrite, so demanding a close match would classify successful repairs as deletions. The
#: question this threshold answers is "is this material still here in some form", not "is it
#: unchanged".
SURVIVAL_THRESHOLD = 0.35


@dataclass(frozen=True, slots=True)
class DefectOutcome:
    """What happened to one planted defect."""

    defect_id: str
    type: str
    status: str
    #: Best similarity between the defect's clean sentence and any revised sentence. The
    #: evidence behind a `removed` verdict, so it can be argued with.
    survival_similarity: float
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecallReport:
    """Aggregate over one passage's defects."""

    fixed: int
    present: int
    removed: int
    unscoreable: int
    outcomes: list[DefectOutcome]

    @property
    def planted(self) -> int:
        return self.fixed + self.present + self.removed + self.unscoreable

    @property
    def surviving(self) -> int:
        """Defects whose region was still there to be judged."""
        return self.fixed + self.present

    @property
    def recall(self) -> float | None:
        """Fixed as a fraction of *surviving* defects.

        ``None`` when nothing survived. Returning 0.0 there would say "the loop fixed
        nothing", when the truth is "the loop deleted everything and there was nothing left
        to fix" — a different and much worse outcome that must not be averaged in as if it
        were a miss.
        """
        if self.surviving == 0:
            return None
        return self.fixed / self.surviving

    @property
    def removal_rate(self) -> float | None:
        """Fraction of planted defects whose region was cut. Always report beside recall."""
        if self.planted == 0:
            return None
        return self.removed / self.planted

    def as_dict(self) -> dict[str, Any]:
        return {
            "planted": self.planted,
            "fixed": self.fixed,
            "present": self.present,
            "removed": self.removed,
            "unscoreable": self.unscoreable,
            "surviving": self.surviving,
            "recall": self.recall,
            "removal_rate": self.removal_rate,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def _best_survival(clean_sentence: str, revised_sentences: Sequence[str]) -> float:
    if not clean_sentence.strip() or not revised_sentences:
        return 0.0
    return max(sentence_similarity(clean_sentence, s) for s in revised_sentences)


def defect_outcome(
    defect: dict[str, Any],
    revised_text: str,
    *,
    survival_threshold: float = SURVIVAL_THRESHOLD,
) -> DefectOutcome:
    """Classify one defect against a revised passage.

    Args:
        defect: A row from the defect manifest (see :class:`revisionbench.inject.Defect`).
        revised_text: The passage after revision.
        survival_threshold: See :data:`SURVIVAL_THRESHOLD`.
    """
    revised_sentences = sentences(revised_text)
    if not revised_sentences:
        return DefectOutcome(
            defect_id=defect["defect_id"],
            type=defect["type"],
            status="unscoreable",
            survival_similarity=0.0,
            detail="revised text has no sentences",
        )

    similarity = _best_survival(defect.get("clean_sentence", ""), revised_sentences)
    kind = defect["detect_kind"]

    # `count` defects are the exception to the survival check: an echo is scored over the
    # whole passage rather than at one site, so the relevant question is how many copies
    # remain, and a deleted copy is a legitimate way to fix a repetition.
    if kind == "count":
        marker = defect["marker"]
        remaining = revised_text.count(marker)
        if remaining <= int(defect.get("target_count", 1)):
            status = "fixed"
        elif similarity < survival_threshold and remaining == 0:
            status = "removed"
        else:
            status = "present"
        return DefectOutcome(
            defect_id=defect["defect_id"],
            type=defect["type"],
            status=status,
            survival_similarity=similarity,
            detail=f"{remaining} occurrence(s) of {marker!r} remain",
        )

    if similarity < survival_threshold:
        return DefectOutcome(
            defect_id=defect["defect_id"],
            type=defect["type"],
            status="removed",
            survival_similarity=similarity,
            detail=(
                f"no revised sentence resembles the defect's region "
                f"(best similarity {similarity:.2f} < {survival_threshold}); the content "
                f"was cut rather than repaired"
            ),
        )

    if kind == "marker":
        marker = defect["marker"]
        present = marker in revised_text
        return DefectOutcome(
            defect_id=defect["defect_id"],
            type=defect["type"],
            status="present" if present else "fixed",
            survival_similarity=similarity,
            detail=f"marker {marker!r} {'still present' if present else 'gone'}",
        )

    if kind == "minimal_pair":
        # For a defect that differs from the clean text by a word or two — a tense slip is
        # the canonical case — neither an absolute similarity threshold nor a marker works.
        # The corrupted and clean sentences are ~0.97 similar to each other, so any span
        # check calls them the same sentence, and the swapped word ("is") is far too common
        # to use as a marker. Scoring the clean original this way produced 0.75 instead of
        # the 1.00 it is by definition.
        #
        # The right question for a minimal pair is comparative: of the two versions, which
        # does the surviving text now look more like?
        best_sentence = max(
            revised_sentences, key=lambda s: sentence_similarity(defect["clean_sentence"], s)
        )
        to_clean = sentence_similarity(defect["original_fragment"], best_sentence)
        to_corrupt = sentence_similarity(defect["corrupt_fragment"], best_sentence)
        return DefectOutcome(
            defect_id=defect["defect_id"],
            type=defect["type"],
            status="fixed" if to_clean > to_corrupt else "present",
            survival_similarity=similarity,
            detail=f"resembles clean {to_clean:.2f} vs corrupt {to_corrupt:.2f}",
        )

    if kind == "span":
        # The corrupted span is "fixed" when nothing in the revised text still closely
        # resembles it. Stricter than the survival threshold on purpose: a clunker that has
        # merely been trimmed is not a fix.
        corrupt = defect["corrupt_fragment"]
        best = _best_survival(corrupt, revised_sentences)
        return DefectOutcome(
            defect_id=defect["defect_id"],
            type=defect["type"],
            status="present" if best >= 0.6 else "fixed",
            survival_similarity=similarity,
            detail=f"closest match to the corrupted span: {best:.2f}",
        )

    raise ValueError(
        f"unknown detect_kind {kind!r} for {defect['defect_id']}; "
        f"known: marker, count, span, minimal_pair"
    )


def recall_report(
    defects: Sequence[dict[str, Any]],
    revised_text: str,
    *,
    survival_threshold: float = SURVIVAL_THRESHOLD,
) -> RecallReport:
    """Classify every defect in a passage and aggregate."""
    outcomes = [
        defect_outcome(d, revised_text, survival_threshold=survival_threshold) for d in defects
    ]
    counts = {"fixed": 0, "present": 0, "removed": 0, "unscoreable": 0}
    for outcome in outcomes:
        counts[outcome.status] += 1
    return RecallReport(outcomes=outcomes, **counts)
