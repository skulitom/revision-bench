"""Revision strategies — what unit gets revised, and what gets applied (plan.md §8).

Phase 0 and M1-a between them killed the two easy levers on the length collapse: asking
the model to preserve length is model-dependent, and re-rolling it does nothing because the
model has a target length it holds across seeds (docs/findings-phase1.md §1). What is left
is architectural, which is what plan.md §8's arms were always about.

A strategy owns one question: **given the current text, produce the next text.** Everything
else — the feed-forward, the guards, the artifact — belongs to :mod:`revisionbench.loop`.

Implemented here:

- :class:`WholePassage` — arm A0. One call, output replaces input. The control, and the
  thing every other arm is trying to beat.
- :class:`ParagraphScoped` — arm A2p. Each paragraph revised independently and reassembled.
  Compression is bounded *per paragraph*, so a model that shortens each one by a third
  cannot compound that across the passage the way a whole-passage rewrite does.
- :class:`EditList` — arm A2e. The model returns a list of find/replace operations and only
  unambiguous ones are applied. The strongest bounded-diff form available without a judge,
  and the one the downstream harness (plan.md §10) actually needs.

**The trap this module is written around.** An arm that fails to parse, or proposes
nothing, returns the text unchanged — and unchanged text scores as *perfect* voice
preservation and zero slop. A broken arm therefore looks like the best arm in the study.
Every strategy reports telemetry distinguishing "proposed nothing", "proposed and all were
rejected", and "could not be parsed", and :class:`Proposal` refuses to hide the difference.
plan.md §6 makes the same point from the other side: a gate that blocks all edits scores
zero defect recall and must be rejected on those grounds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from revisionbench.ollama import Generation, GenerationOptions, ModelIdentity, OllamaClient
from revisionbench.text import normalise_newlines, paragraph_spans, word_count

__all__ = [
    "ARMS",
    "EditList",
    "ParagraphScoped",
    "Proposal",
    "ReviseContext",
    "Strategy",
    "WholePassage",
    "build_strategy",
]


@dataclass(frozen=True, slots=True)
class ReviseContext:
    """Everything a strategy needs to talk to a model."""

    client: OllamaClient
    model: ModelIdentity
    options: GenerationOptions
    #: Template with ``{passage}`` and optionally ``{word_count}``.
    template: str

    def render(self, text: str) -> str:
        rendered = self.template.replace("{word_count}", str(word_count(text)))
        return rendered.replace("{passage}", text)

    def generate(self, text: str) -> Generation:
        return self.client.generate(self.model.tag, self.render(text), self.options)


@dataclass(slots=True)
class Proposal:
    """One round's output plus enough telemetry to tell a working arm from a broken one."""

    text: str
    generations: list[Generation] = field(default_factory=list)
    #: Units the strategy split the passage into (1 for whole-passage).
    units: int = 1
    #: Edits proposed, applied, and rejected. For unit-based arms, "proposed" is the unit
    #: count and "applied" counts units whose text actually changed.
    proposed: int = 0
    applied: int = 0
    rejected: int = 0
    #: Non-fatal problems, e.g. unparseable output or an edit whose anchor was ambiguous.
    #: Never empty-and-silent: a strategy that produced nothing must say why.
    problems: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.applied > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "units": self.units,
            "proposed": self.proposed,
            "applied": self.applied,
            "rejected": self.rejected,
            "problems": self.problems,
            "generations": len(self.generations),
            "wall_seconds": round(sum(g.wall_seconds for g in self.generations), 2),
            "prompt_tokens": sum(g.prompt_tokens for g in self.generations),
            "output_tokens": sum(g.output_tokens for g in self.generations),
            "truncated": any(g.truncated for g in self.generations),
        }


class Strategy(Protocol):
    """Produce the next version of a text."""

    name: str

    def revise(self, ctx: ReviseContext, text: str) -> Proposal: ...


# --------------------------------------------------------------------------------------
# A0 — whole passage
# --------------------------------------------------------------------------------------


class WholePassage:
    """Arm A0: one call, the output replaces the input entirely."""

    name = "whole"

    def revise(self, ctx: ReviseContext, text: str) -> Proposal:
        generation = ctx.generate(text)
        revised = generation.text.strip()
        return Proposal(
            text=revised,
            generations=[generation],
            units=1,
            proposed=1,
            applied=int(revised != text.strip()),
            rejected=0,
        )


# --------------------------------------------------------------------------------------
# A2p — paragraph scoped
# --------------------------------------------------------------------------------------


class ParagraphScoped:
    """Arm A2p: revise each paragraph independently, then reassemble.

    The passage is split on blank lines, each paragraph is sent through the same prompt on
    its own, and the results are rejoined with the original blank-line separator. A model
    that cuts a third of each paragraph still cuts a third overall — this does not make
    compression impossible — but it removes the *structural* opportunity to drop whole
    paragraphs, which is what a whole-passage rewrite mostly does.

    Cost is one generation per paragraph rather than one per passage. Total tokens are
    similar (the same text goes through, split up); per-call overhead is not, so wall clock
    rises. That is recorded per round rather than estimated.

    Paragraphs with no word tokens (scene-break asterisks, stray rules) are passed through
    untouched and counted as rejected, because asking a model to "improve" ``* * *`` invites
    it to write a paragraph there.

    **Unit count is held invariant, and that is load-bearing.** A revision that comes back
    as two paragraphs is rejoined into one before reassembly. The first version of this arm
    did not do that, and the result was a compounding duplication cascade: the model split a
    paragraph in two, the next round saw two units and revised each separately, and on
    woolf-01 the passage went 7 → 10 → 12 → 16 → 20 paragraphs and 901 → 1569 words over
    five rounds, with visibly near-duplicate blocks. Averaged across passages that showed up
    as a reassuring mean length ratio of 1.03 — a number that looked like the arm had solved
    the length problem while one passage had inflated 74%.

    A bounded-diff architecture whose unit boundaries move is not bounded. If the k-th
    paragraph in goes to the k-th paragraph out, the passage cannot grow structurally.
    """

    name = "paragraph"

    #: Paragraphs shorter than this are passed through rather than revised. A two-word
    #: fragment gives a model nothing to work with and a lot of room to invent.
    min_words = 5

    def revise(self, ctx: ReviseContext, text: str) -> Proposal:
        normalised = normalise_newlines(text)
        spans = paragraph_spans(normalised)
        if not spans:
            return Proposal(
                text=text,
                units=0,
                problems=["passage contains no paragraphs"],
            )

        pieces: list[str] = []
        generations: list[Generation] = []
        applied = rejected = 0
        problems: list[str] = []

        for start, end in spans:
            paragraph = normalised[start:end]
            if word_count(paragraph) < self.min_words:
                pieces.append(paragraph)
                rejected += 1
                continue
            generation = ctx.generate(paragraph)
            generations.append(generation)
            revised = generation.text.strip()
            if not revised:
                pieces.append(paragraph)
                rejected += 1
                problems.append("empty revision for one paragraph")
                continue
            collapsed = _collapse_to_one_paragraph(revised)
            if collapsed != revised:
                problems.append("revision returned multiple paragraphs; rejoined into one")
            pieces.append(collapsed)
            applied += int(collapsed != paragraph)

        result = Proposal(
            text="\n\n".join(pieces),
            generations=generations,
            units=len(spans),
            proposed=len(spans),
            applied=applied,
            rejected=rejected,
            problems=problems,
        )
        # The invariant the arm depends on. Asserted rather than assumed: it was violated
        # by the first implementation and the symptom was a plausible average.
        out_units = len(paragraph_spans(normalise_newlines(result.text)))
        if out_units != len(spans):
            result.problems.append(
                f"paragraph count changed {len(spans)} -> {out_units}; the bounded-diff "
                f"guarantee is broken and this round's length is not structurally bounded"
            )
        return result


# --------------------------------------------------------------------------------------
# A2e — edit list
# --------------------------------------------------------------------------------------

#: Instruction appended to the configured template for the edit-list arm. Kept here rather
#: than in config because it is a *protocol*, not a wording choice: the parser below and
#: this text are one unit, and separating them invites them to drift apart.
EDIT_LIST_INSTRUCTION = """
Respond with a JSON array of edits and nothing else. Each edit is an object with two keys:
"find" — text copied exactly from the passage, long enough to occur only once — and
"replace" — what it should become. Use an empty "replace" to delete. Propose only edits
that improve the passage; an empty array is a valid answer. Do not rewrite the passage.

Example: [{"find": "the very tall man", "replace": "the tall man"}]
"""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")


def _collapse_to_one_paragraph(text: str) -> str:
    """Join a multi-paragraph revision back into a single paragraph.

    Blank lines become a single space. Interior single newlines are left alone: Gutenberg
    prose is hard-wrapped, so they are line breaks rather than structure, and rewrapping
    would change nothing a metric reads but would make diffs unreadable.
    """
    return _BLANK_LINE_RE.sub(" ", normalise_newlines(text).strip())


class EditList:
    """Arm A2e: apply only the find/replace operations the model names.

    Each edit is applied only if its ``find`` string occurs **exactly once** in the current
    text. Zero occurrences means the model hallucinated the anchor; more than one means the
    edit is ambiguous and applying it to the first match would be arbitrary. Both are
    rejected and counted.

    This is the strongest length control available without a judge: text the model does not
    name cannot change. It is also the arm most likely to fail silently, because a model
    that returns prose instead of JSON proposes zero edits and leaves the passage pristine —
    which scores as flawless preservation. Parse failures are therefore recorded in
    ``problems`` and surface as a distinct outcome from "proposed nothing".
    """

    name = "editlist"

    #: Reject an edit whose `find` is shorter than this. Short anchors are usually ambiguous
    #: and, when they are not, they tend to be function words whose replacement is a
    #: typographic tweak rather than a revision.
    min_anchor_chars = 12

    def revise(self, ctx: ReviseContext, text: str) -> Proposal:
        instruction_ctx = ReviseContext(
            client=ctx.client,
            model=ctx.model,
            options=ctx.options,
            template=ctx.template.rstrip() + "\n" + EDIT_LIST_INSTRUCTION,
        )
        generation = instruction_ctx.generate(text)
        edits, problems = self._parse(generation.text)

        current = text
        applied = rejected = 0
        for edit in edits:
            find, replace = edit
            if len(find) < self.min_anchor_chars:
                rejected += 1
                problems.append(f"anchor too short: {find[:40]!r}")
                continue
            occurrences = current.count(find)
            if occurrences == 0:
                rejected += 1
                problems.append(f"anchor not found: {find[:40]!r}")
                continue
            if occurrences > 1:
                rejected += 1
                problems.append(f"anchor ambiguous ({occurrences}x): {find[:40]!r}")
                continue
            current = current.replace(find, replace, 1)
            applied += 1

        return Proposal(
            text=current,
            generations=[generation],
            units=1,
            proposed=len(edits),
            applied=applied,
            rejected=rejected,
            problems=problems,
        )

    def _parse(self, raw: str) -> tuple[list[tuple[str, str]], list[str]]:
        """Extract ``(find, replace)`` pairs, tolerating fences and surrounding chatter."""
        problems: list[str] = []
        candidate = raw.strip()

        fenced = _FENCE_RE.search(candidate)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            start, end = candidate.find("["), candidate.rfind("]")
            if start != -1 and end > start:
                candidate = candidate[start : end + 1]

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return [], [f"unparseable edit list ({exc.msg}); no edits applied"]

        if not isinstance(parsed, list):
            return [], [f"edit list is a {type(parsed).__name__}, not an array"]

        edits: list[tuple[str, str]] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict) or "find" not in item:
                problems.append(f"edit {index} is malformed")
                continue
            find = item.get("find")
            replace = item.get("replace", "")
            if not isinstance(find, str) or not isinstance(replace, str):
                problems.append(f"edit {index} has non-string find/replace")
                continue
            edits.append((find, replace))
        return edits, problems


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

ARMS: dict[str, type] = {
    WholePassage.name: WholePassage,
    ParagraphScoped.name: ParagraphScoped,
    EditList.name: EditList,
}


def build_strategy(name: str) -> Strategy:
    """Instantiate a strategy by name.

    Raises:
        ValueError: Unknown name, listing the known ones. A typo must not fall back to the
            control, which would report A0's numbers under another arm's label.
    """
    try:
        return ARMS[name]()
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}; known strategies are {sorted(ARMS)}"
        ) from None
