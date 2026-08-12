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
from dataclasses import asdict, dataclass, replace
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
    #: How many distinct candidates the model produced for this complaint, and which rank
    #: (0-based, ascending edit distance) was accepted. ``chosen_rank > 0`` is a repair
    #: best-of-N found that a single proposal would have missed — the measurement that says
    #: whether the extra generations bought anything.
    candidates_seen: int = 1
    chosen_rank: int = 0

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


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Used to rank candidate repairs by how little they change.

    Not "shortest replacement": a candidate that deletes the span is shortest by length and
    maximal by damage. Distance from the span being replaced is the quantity that means
    *minimal intervention*, which is the property this arm is for.
    """
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _screen(original: str, replacement: str) -> str:
    """Cheap structural checks. Returns a rejection reason, or "" to proceed.

    Extracted when best-of-N arrived: every candidate must face the identical gate, and a
    screen that drifted between the first candidate and the rest would silently make later
    candidates easier to accept.
    """
    if not replacement.strip():
        return "empty"
    if len(replacement) > max(40, len(original) * MAX_GROWTH):
        return "out_of_scope"
    if not _same_form(original, replacement):
        return "changed_form"
    return ""


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
    candidates: int = 3,
    candidate_temperature: float = 0.7,
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
            prompt = repair_prompt(text, match)

            # Best-of-N, ranked by minimal intervention. Edit-level best-of-N is live where
            # round-level re-rolling was dead (findings-phase1): candidates for a *named*
            # fix genuinely vary, whereas whole-round lengths did not, because the model
            # held a target length across seeds. Distinct seeds, deduplicated, then tried
            # in ascending edit distance from the span being replaced — so the least
            # invasive repair that survives verification is the one that lands.
            proposals: list[str] = []
            raw_first = ""
            for index in range(max(1, candidates)):
                # The first candidate is generated exactly as configured — normally greedy,
                # so the primary proposal stays reproducible. The rest are *sampled*, and
                # that is not optional: at temperature 0 decoding is deterministic and the
                # seed changes nothing, so N generations return N byte-identical strings.
                # Measured: candidates_seen was 1 for all 67 complaints, at 2.5x the GPU
                # time for zero variation. That is the same trap findings-phase1 recorded
                # for round-level resampling, arriving one level down.
                attempt = (
                    options
                    if index == 0
                    else replace(
                        options,
                        seed=options.seed + index,
                        temperature=max(options.temperature, candidate_temperature),
                    )
                )
                generation = client.generate(
                    tag,
                    prompt,
                    attempt,
                    schema=REPAIR_SCHEMA,
                    think=think,
                )
                raw_first = raw_first or generation.text
                try:
                    proposal = json.loads(generation.text)["replacement"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if proposal not in proposals:
                    proposals.append(proposal)

            if not proposals:
                outcomes.append(
                    RepairOutcome(
                        match.type,
                        match.span,
                        original,
                        raw_first[:200],
                        False,
                        "no_valid_proposal",
                        candidates_seen=candidates,
                    )
                )
                continue

            ranked = sorted(proposals, key=lambda p: edit_distance(original, p))
            before_count = len(detect_all_litrpg(text))
            rejected: RepairOutcome | None = None
            accepted: RepairOutcome | None = None

            for rank, replacement in enumerate(ranked):
                verdict = _screen(original, replacement)
                if verdict:
                    rejected = rejected or RepairOutcome(
                        match.type,
                        match.span,
                        original,
                        replacement,
                        False,
                        verdict,
                        candidates_seen=len(ranked),
                    )
                    continue

                candidate = _apply(text, match.span, replacement)
                after = detect_all_litrpg(candidate)
                if len(after) >= before_count:
                    still_present = any(c.type == match.type for c in after)
                    rejected = rejected or RepairOutcome(
                        match.type,
                        match.span,
                        original,
                        replacement,
                        False,
                        "complaint_persists" if still_present else "new_complaint",
                        len(_signature(after) - baseline),
                        len(ranked),
                    )
                    continue

                accepted = RepairOutcome(
                    match.type,
                    match.span,
                    original,
                    replacement,
                    True,
                    "accepted",
                    candidates_seen=len(ranked),
                    chosen_rank=rank,
                )
                break

            if accepted is None:
                outcomes.append(
                    rejected
                    or RepairOutcome(
                        match.type,
                        match.span,
                        original,
                        ranked[0],
                        False,
                        "empty",
                        candidates_seen=len(ranked),
                    )
                )
                continue

            text = _apply(text, match.span, accepted.proposed)
            accepted_this_round += 1
            outcomes.append(accepted)

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
