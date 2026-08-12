"""A2d — complaint-gated repair, with a mechanical acceptance rule.

The arm `harness-gap.md` §2 proposed and §5 put third in the queue. Every arm before it has
the same shape — **revise, then gate**: the model is invited to improve a unit, and something
downstream decides how much to keep. Bounded diffs made the keeping safe (A2i: 74% apply
rate, zero protocol failures) and did nothing about the inviting, which is where the 24:1
overreach comes from. A model asked to improve a paragraph will always find something to
change, because that is what it was asked for.

The inversion:

> Only a span with a located, checkable complaint against it is eligible to be edited.
> Everything else is frozen.

Two things follow, and they are the whole point:

1. **Eligibility is structural, not prompted.** The model never sees the manuscript. It sees
   one complaint and one span, and it returns replacement text for that span. There is no
   channel through which it could alter anything else, so "leave the rest alone" is not an
   instruction that can be ignored — it is a property of the interface. This is the same
   move that fixed length in Phase 1: length stopped being a problem when changing unnamed
   text became impossible rather than discouraged.

2. **Acceptance is mechanical.** A repair is applied only if the complaint it cites
   disappears *and* no new complaint appears anywhere in the manuscript. No judge, no model
   call, no threshold. A repair that fixes its own target while breaking something else is
   rejected automatically, which is the failure mode a per-edit reviewer cannot see.

Detection is re-run from scratch after every accepted repair rather than incrementally.
Incremental invalidation is a real optimisation for a book-length manuscript and it is also
exactly the kind of bookkeeping that goes stale silently, so it is deliberately not done
here — correctness first, and the detector costs no model calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from revisionbench.detect import Complaint
from revisionbench.litrpg_detect import detect_all_litrpg
from revisionbench.ollama import GenerationOptions, OllamaClient

__all__ = [
    "REPAIR_SCHEMA",
    "RepairOutcome",
    "RepairReport",
    "repair_manuscript",
    "repair_prompt",
]

#: Schema-constrained so a refusal or a chatty preamble cannot masquerade as a repair.
REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"replacement": {"type": "string"}},
    "required": ["replacement"],
}

#: A repair may not rewrite the world. The span is a stat line or a phrase, so anything
#: much longer than what it replaced is the model expanding scope, and expansion is the
#: behaviour this arm exists to make impossible.
MAX_GROWTH = 3.0


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """One complaint, and what became of it."""

    complaint_type: str
    span: tuple[int, int]
    original: str
    proposed: str
    applied: bool
    #: Why it was rejected, or ``accepted``. The distribution over these is the result:
    #: ``new_complaint`` means the verifier caught collateral damage a reviewer would not.
    reason: str
    new_complaints: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["span"] = list(self.span)
        return data


@dataclass(frozen=True, slots=True)
class RepairReport:
    manuscript_id: str
    text: str
    outcomes: tuple[RepairOutcome, ...]
    rounds: int
    complaints_before: int
    complaints_after: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "manuscript_id": self.manuscript_id,
            "outcomes": [o.as_dict() for o in self.outcomes],
            "rounds": self.rounds,
            "complaints_before": self.complaints_before,
            "complaints_after": self.complaints_after,
        }


def repair_prompt(text: str, complaint: Complaint, *, window: int = 400) -> str:
    """One complaint, its span, and just enough context to repair it.

    The manuscript is deliberately not supplied. A model given the whole thing would have to
    be *told* not to touch the rest; a model given one span cannot.
    """
    start, end = complaint.span
    before = text[max(0, start - window) : start]
    after = text[end : end + window]
    return (
        "You are fixing one specific continuity error in a LitRPG web serial. "
        "Return replacement text for the marked span ONLY.\n\n"
        f"THE PROBLEM: {complaint.message}\n"
        f"EVIDENCE: {complaint.evidence}\n\n"
        f"CONTEXT BEFORE:\n...{before}\n\n"
        f">>> SPAN TO REPLACE: {text[start:end]}\n\n"
        f"CONTEXT AFTER:\n{after}...\n\n"
        'Return JSON: {"replacement": "..."}\n'
        "Rules: fix ONLY the stated problem. Keep the same form as the span you are "
        "replacing — if it is a status line like '  Level: 4', return a status line; if it "
        "is a phrase, return a phrase. Do not add commentary. Do not rewrite the prose "
        "around it. Do not change anything the problem statement did not mention."
    )


_FIELD = re.compile(r"^\s*(\w+):")


def _same_form(original: str, replacement: str) -> bool:
    """A status field must come back as the same status field.

    The loophole this closes: the cheapest way to resolve "Strength changes 10 -> 14 with no
    level-up" is to stop mentioning Strength. The complaint disappears, the count falls, and
    every acceptance check passes — while the manuscript has silently lost a fact. Any
    verifier that only asks "did the complaint go away" is vulnerable to its evidence being
    deleted, and this is the general shape of that bug, not a quirk of one detector.

    Symmetric, and it has to be. Protecting only status fields leaves the mirror-image hole
    open: a prose span replaced by ``  Level: 999`` also resolves its complaint, because the
    offending phrase is gone. That was accepted by an earlier version of this rule. Prose
    must come back as prose.
    """
    field = _FIELD.match(original)
    candidate = _FIELD.match(replacement)
    if not field:
        return not candidate  # prose must not be replaced by a status line
    return bool(candidate) and candidate.group(1) == field.group(1)


def _apply(text: str, span: tuple[int, int], replacement: str) -> str:
    return text[: span[0]] + replacement + text[span[1] :]


def _signature(complaints: list[Complaint]) -> set[tuple[str, str]]:
    """Identify complaints by (type, message) rather than span.

    Spans shift as soon as a repair changes length, so comparing them across an edit would
    report every downstream complaint as "new" and reject every repair. The message carries
    the chapter numbers and values, so it identifies the grievance rather than its position.
    """
    return {(c.type, c.message) for c in complaints}


def repair_manuscript(
    client: OllamaClient,
    tag: str,
    text: str,
    options: GenerationOptions,
    *,
    manuscript_id: str = "",
    max_rounds: int = 3,
    think: bool | None = False,
) -> RepairReport:
    """Detect, repair one complaint at a time, verify, and keep only what survives.

    Loops because repairing one contradiction can reveal another that the first was masking.
    Stops when a round accepts nothing, so a manuscript the model cannot improve costs one
    wasted round rather than ``max_rounds`` of them.
    """
    before = detect_all_litrpg(text)
    outcomes: list[RepairOutcome] = []
    baseline = _signature(before)
    rounds = 0

    for _ in range(max_rounds):
        rounds += 1
        complaints = detect_all_litrpg(text)
        if not complaints:
            break
        accepted_this_round = 0

        for complaint in complaints:
            # Re-detect per complaint: an earlier acceptance in this round may have shifted
            # spans or resolved this grievance outright.
            live = detect_all_litrpg(text)
            match = next(
                (c for c in live if c.type == complaint.type and c.message == complaint.message),
                None,
            )
            if match is None:
                continue

            original = text[match.span[0] : match.span[1]]
            generation = client.generate(
                tag, repair_prompt(text, match), options, schema=REPAIR_SCHEMA, think=think
            )
            try:
                replacement = json.loads(generation.text)["replacement"]
            except (json.JSONDecodeError, KeyError, TypeError):
                outcomes.append(
                    RepairOutcome(
                        match.type,
                        match.span,
                        original,
                        generation.text[:200],
                        False,
                        "no_valid_proposal",
                    )
                )
                continue

            if not replacement.strip():
                outcomes.append(
                    RepairOutcome(match.type, match.span, original, replacement, False, "empty")
                )
                continue
            if len(replacement) > max(40, len(original) * MAX_GROWTH):
                outcomes.append(
                    RepairOutcome(
                        match.type, match.span, original, replacement, False, "out_of_scope"
                    )
                )
                continue

            if not _same_form(original, replacement):
                # A status field replaced by a different field is not a repair, it is a
                # deletion — and deleting the evidence resolves the complaint. Enforced
                # structurally rather than asked for in the prompt, for the same reason
                # eligibility is: a rule the model could ignore is not a rule.
                outcomes.append(
                    RepairOutcome(
                        match.type, match.span, original, replacement, False, "changed_form"
                    )
                )
                continue

            candidate = _apply(text, match.span, replacement)
            before_count = len(detect_all_litrpg(text))
            after = detect_all_litrpg(candidate)

            # Net improvement, counted. Identity-based comparison was the first design and
            # it is wrong: complaint messages carry the offending values, so a repair that
            # changed 3 to 2 "resolved" one complaint and "introduced" another when it had
            # merely failed. Counting cannot be fooled that way — fix one and break one and
            # the total is unchanged, which is a rejection.
            if len(after) >= before_count:
                still_present = any(c.type == match.type for c in after)
                introduced = _signature(after) - baseline
                outcomes.append(
                    RepairOutcome(
                        match.type,
                        match.span,
                        original,
                        replacement,
                        False,
                        "complaint_persists" if still_present else "new_complaint",
                        len(introduced),
                    )
                )
                continue

            text = candidate
            accepted_this_round += 1
            outcomes.append(
                RepairOutcome(match.type, match.span, original, replacement, True, "accepted")
            )

        if accepted_this_round == 0:
            break

    return RepairReport(
        manuscript_id=manuscript_id,
        text=text,
        outcomes=tuple(outcomes),
        rounds=rounds,
        complaints_before=len(before),
        complaints_after=len(detect_all_litrpg(text)),
    )
