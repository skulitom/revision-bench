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
- :class:`IndexedEditList` — arm A2i. Edits addressed by sentence number under an enforced
  schema, with per-edit mechanical vetoes. Removes A2e's two measured failure classes.
- :class:`FeedbackIndexedEditList` — arm A2f. A2i with the vetoes stated in the protocol,
  a punctuation veto, and one per-edit feedback retry for vetoed edits.

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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from revisionbench.ollama import Generation, GenerationOptions, ModelIdentity, OllamaClient
from revisionbench.text import (
    normalise_newlines,
    paragraph_spans,
    punctuation_counts,
    sentence_spans,
    word_count,
)

__all__ = [
    "ARMS",
    "INDEXED_EDIT_SCHEMA",
    "EditList",
    "FeedbackIndexedEditList",
    "IndexedEditList",
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
# A2i — indexed edits, enforced schema, per-edit mechanical vetoes
# --------------------------------------------------------------------------------------

#: JSON schema for the indexed edit protocol. Passed to Ollama's `format`, which constrains
#: decoding rather than merely requesting a shape.
INDEXED_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_index": {"type": "integer"},
                    "replacement": {"type": "string"},
                },
                "required": ["sentence_index", "replacement"],
            },
        }
    },
    "required": ["edits"],
}

INDEXED_EDIT_INSTRUCTION = """
The text above is numbered by sentence. Return the sentences you want to change, each as
its number and its full replacement text. Change only what needs changing; returning no
edits is a valid answer. Do not renumber, merge, or reorder sentences.
"""


class IndexedEditList:
    """Arm A2i: edits addressed by sentence number, under an enforced schema.

    Two changes from :class:`EditList`, each aimed at a measured failure:

    **Symbolic addressing instead of verbatim quotation.** A2e lost 223 of its 236 rejected
    edits (94.5%) to a single cause — the model paraphrased the ``find`` anchor instead of
    copying it. Quoting asks a model to do the one thing transformers are unreliable at, for
    no benefit, since the span is already on screen and already addressable. An integer
    cannot be misquoted, so this failure class disappears rather than shrinking.

    **Enforced schema instead of requested JSON.** A2e lost 3 of 13 rounds to unparseable
    output. Under Ollama's ``format`` the sampler cannot emit a token that breaks the schema.

    Indices refer to the *input* snapshot and every edit is applied against it, so edits
    cannot interfere with one another and the k-th sentence in maps to the k-th slot out.
    Whitespace and paragraph structure come from the original spans, so the arm cannot
    reflow the passage as a side effect.

    Optional per-edit mechanical vetoes (all free — no judge):

    - ``length`` — the replacement must stay within a band of the sentence it replaces.
      Applied per edit, so it cannot be defeated by a model that has decided to summarise.
    - ``slop`` — reject a replacement that introduces a term from the versioned lexicon
      which was not already in the sentence. M3 measured slop rising 0.54 -> 3.73 under A0;
      this makes that rise unrepresentable rather than merely discouraged.
    """

    name = "indexed"

    #: Replacement word count must fall within these multiples of the original sentence's.
    length_band = (0.5, 2.0)
    #: Vetoes this arm knows how to apply. Subclasses extend rather than restate.
    known_vetoes: tuple[str, ...] = ("length", "slop")

    def __init__(self, *, vetoes: Sequence[str] = ("length", "slop")) -> None:
        unknown = [v for v in vetoes if v not in self.known_vetoes]
        if unknown:
            raise ValueError(f"unknown veto(es): {unknown}; known: {', '.join(self.known_vetoes)}")
        self.vetoes = tuple(vetoes)
        self._lexicon = None

    def _slop_terms(self, text: str) -> set[str]:
        from revisionbench.metrics.slop import load_lexicon, slop_index

        if self._lexicon is None:
            self._lexicon = load_lexicon()
        return set(slop_index(text, self._lexicon).by_term)

    def revise(self, ctx: ReviseContext, text: str) -> Proposal:
        normalised = normalise_newlines(text)
        spans = sentence_spans(normalised)
        if not spans:
            return Proposal(text=text, units=0, problems=["passage has no sentences"])

        numbered = "\n".join(f"[{i}] {normalised[a:b]}" for i, (a, b) in enumerate(spans))
        prompt = (
            ctx.template.replace("{word_count}", str(word_count(normalised))).replace(
                "{passage}", numbered
            )
            + "\n"
            + INDEXED_EDIT_INSTRUCTION
        )
        generation = ctx.client.generate(
            ctx.model.tag, prompt, ctx.options, schema=INDEXED_EDIT_SCHEMA
        )

        problems: list[str] = []
        try:
            payload = json.loads(generation.text)
            raw_edits = payload.get("edits", [])
        except json.JSONDecodeError as exc:
            # Should be unreachable under constrained decoding. Recorded rather than
            # assumed away, because "the schema guarantees it" is exactly the kind of
            # assumption that turns into a silent no-op arm.
            return Proposal(
                text=text,
                generations=[generation],
                units=len(spans),
                problems=[f"schema-constrained output still did not parse ({exc.msg})"],
            )

        chosen: dict[int, str] = {}
        rejected, _ = self._screen(raw_edits, spans, normalised, chosen, problems)

        rebuilt = self._rebuild(normalised, spans, chosen)
        return Proposal(
            text=rebuilt,
            generations=[generation],
            units=len(spans),
            proposed=len(raw_edits),
            applied=len(chosen),
            rejected=rejected,
            problems=problems,
        )

    def _screen(
        self,
        raw_edits: Sequence[dict[str, Any]],
        spans: Sequence[tuple[int, int]],
        normalised: str,
        chosen: dict[int, str],
        problems: list[str],
        *,
        eligible: set[int] | None = None,
        phase: str = "",
    ) -> tuple[int, list[tuple[int, str, str, str]]]:
        """Apply index checks and vetoes, adding survivors to ``chosen``.

        Returns ``(rejected_count, vetoed)`` where ``vetoed`` lists
        ``(index, original, replacement, reason)`` for edits that failed a *veto* — the
        recoverable class, where a corrected replacement could exist — as opposed to index
        errors and no-op edits, which have no coherent correction to ask for.

        ``eligible`` restricts which indices may be edited (used by the feedback pass so a
        retry cannot smuggle in edits to sentences that were never vetoed). ``phase``
        prefixes problem strings so the artifact shows which pass produced them.
        """
        rejected = 0
        vetoed: list[tuple[int, str, str, str]] = []
        for edit in raw_edits:
            index = edit.get("sentence_index")
            replacement = edit.get("replacement", "")
            if not isinstance(index, int) or not 0 <= index < len(spans):
                rejected += 1
                problems.append(f"{phase}index {index!r} out of range 0..{len(spans) - 1}")
                continue
            if eligible is not None and index not in eligible:
                rejected += 1
                problems.append(f"{phase}unsolicited index {index}")
                continue
            if index in chosen:
                rejected += 1
                problems.append(f"{phase}duplicate edit for sentence {index}")
                continue
            original = normalised[spans[index][0] : spans[index][1]]
            veto = self._veto(original, replacement)
            if veto:
                rejected += 1
                problems.append(f"{phase}sentence {index}: {veto}")
                vetoed.append((index, original, replacement, veto))
                continue
            if replacement.strip() == original.strip():
                rejected += 1
                continue
            chosen[index] = replacement
        return rejected, vetoed

    def _veto(self, original: str, replacement: str) -> str:
        if not replacement.strip():
            return "empty replacement"
        if "length" in self.vetoes:
            before, after = word_count(original), word_count(replacement)
            low, high = self.length_band
            if before and not (low * before <= after <= high * before):
                return f"length {after}w outside {low}-{high}x of {before}w"
        if "slop" in self.vetoes:
            introduced = self._slop_terms(replacement) - self._slop_terms(original)
            if introduced:
                return f"introduces slop {sorted(introduced)}"
        return ""

    @staticmethod
    def _rebuild(text: str, spans: Sequence[tuple[int, int]], edits: dict[int, str]) -> str:
        """Substitute replacements into their slots, preserving everything between them.

        Rebuilt from the original spans rather than by joining sentences, so inter-sentence
        whitespace and paragraph breaks survive exactly. An arm that silently reflowed the
        passage would move the punctuation and sentence-density metrics on its own.
        """
        out: list[str] = []
        cursor = 0
        for index, (start, end) in enumerate(spans):
            out.append(text[cursor:start])
            out.append(
                edits.get(index, text[start:end]).strip() if index in edits else text[start:end]
            )
            cursor = end
        out.append(text[cursor:])
        return "".join(out)


# --------------------------------------------------------------------------------------
# A2f — A2i with a constraint-visible protocol, punctuation veto, and feedback retry
# --------------------------------------------------------------------------------------


class FeedbackIndexedEditList(IndexedEditList):
    """Arm A2f: :class:`IndexedEditList` plus three changes, each aimed at measured headroom.

    **The protocol states its own rules.** A2i's rejections were the intended gates firing
    (6 length, 1 slop of 27 proposals on its first probe) — but the model was never told
    the rules, so every one of those proposals was doomed before it was made. The rule text
    is *generated from the enforcing attributes* (``length_band``, ``punctuation_cap``), so
    the stated protocol and the applied veto cannot drift apart.

    **A punctuation veto.** Designed in docs/design-space.md (axis 4) and unimplemented
    until now. Targets the one attractor effect Phase 0 found surviving length control:
    punctuation restyling (Woolf's em dashes lost at 0.96x length; Hemingway *gaining*
    dashes). Per edit, the count of each punctuation class may move by at most
    ``punctuation_cap``. A per-edit cap cannot stop a many-round walk — that is the drift
    budget's job (design-space axis 5, still unimplemented) — but it blocks the one-shot
    restyling actually observed.

    **One feedback retry, per edit.** Vetoed edits are re-requested once, with the veto
    reason stated. This is not the round-level re-roll that failed in findings-phase1.md §1
    — that resampled an unchanged prompt and found the model holds its target length across
    seeds. This retry differs in kind: it is per edit, and the prompt *changes* — it now
    contains the constraint and the measured violation. Only previously-vetoed indices are
    eligible; a retry cannot smuggle in edits to sentences that were never vetoed.
    """

    name = "indexed_fb"

    known_vetoes = ("length", "slop", "punctuation")
    #: Per edit, the count of each punctuation class may change by at most this much.
    punctuation_cap = 1
    #: Feedback generations per round (0 disables the retry pass).
    feedback_attempts = 1

    def __init__(self, *, vetoes: Sequence[str] = ("length", "slop", "punctuation")) -> None:
        super().__init__(vetoes=vetoes)

    def _veto(self, original: str, replacement: str) -> str:
        veto = super()._veto(original, replacement)
        if veto:
            return veto
        if "punctuation" in self.vetoes:
            before = punctuation_counts(original)
            after = punctuation_counts(replacement)
            shifted = {
                name: after[name] - before[name]
                for name in before
                if abs(after[name] - before[name]) > self.punctuation_cap
            }
            if shifted:
                detail = ", ".join(f"{name} {d:+d}" for name, d in sorted(shifted.items()))
                return f"punctuation shift beyond +/-{self.punctuation_cap}: {detail}"
        return ""

    def _rules(self) -> str:
        lines = []
        if "length" in self.vetoes:
            low, high = self.length_band
            lines.append(
                f"- Keep each replacement between {low}x and {high}x of the original "
                f"sentence's word count."
            )
        if "punctuation" in self.vetoes:
            lines.append(
                f"- Keep the sentence's punctuation style: do not change the count of any "
                f"punctuation mark by more than {self.punctuation_cap}."
            )
        if "slop" in self.vetoes:
            lines.append(
                "- Do not introduce stock phrases or cliches the sentence did not already contain."
            )
        if not lines:
            return ""
        return (
            "\nEvery edit is checked mechanically against these rules; a violating edit "
            "is discarded:\n" + "\n".join(lines) + "\n"
        )

    def revise(self, ctx: ReviseContext, text: str) -> Proposal:
        normalised = normalise_newlines(text)
        spans = sentence_spans(normalised)
        if not spans:
            return Proposal(text=text, units=0, problems=["passage has no sentences"])

        numbered = "\n".join(f"[{i}] {normalised[a:b]}" for i, (a, b) in enumerate(spans))
        prompt = (
            ctx.template.replace("{word_count}", str(word_count(normalised))).replace(
                "{passage}", numbered
            )
            + "\n"
            + INDEXED_EDIT_INSTRUCTION
            + self._rules()
        )
        generations = [
            ctx.client.generate(ctx.model.tag, prompt, ctx.options, schema=INDEXED_EDIT_SCHEMA)
        ]

        problems: list[str] = []
        raw_edits, parse_problem = self._payload(generations[-1].text)
        if parse_problem:
            return Proposal(
                text=text, generations=generations, units=len(spans), problems=[parse_problem]
            )

        chosen: dict[int, str] = {}
        rejected, vetoed = self._screen(raw_edits, spans, normalised, chosen, problems)
        proposed = len(raw_edits)

        if vetoed and self.feedback_attempts >= 1:
            recovered_before = len(chosen)
            feedback = self._feedback_prompt(vetoed)
            generations.append(
                ctx.client.generate(
                    ctx.model.tag, feedback, ctx.options, schema=INDEXED_EDIT_SCHEMA
                )
            )
            raw_retry, parse_problem = self._payload(generations[-1].text)
            if parse_problem:
                problems.append(f"feedback: {parse_problem}")
            else:
                proposed += len(raw_retry)
                retry_rejected, _ = self._screen(
                    raw_retry,
                    spans,
                    normalised,
                    chosen,
                    problems,
                    eligible={index for index, *_ in vetoed},
                    phase="feedback: ",
                )
                rejected += retry_rejected
            # Informational, in the A2p "rejoined into one" tradition: the artifact must
            # show the retry ran and what it bought, or a recovered edit is
            # indistinguishable from a first-pass acceptance.
            problems.append(
                f"feedback: {len(vetoed)} vetoed, {len(chosen) - recovered_before} recovered"
            )

        rebuilt = self._rebuild(normalised, spans, chosen)
        return Proposal(
            text=rebuilt,
            generations=generations,
            units=len(spans),
            proposed=proposed,
            applied=len(chosen),
            rejected=rejected,
            problems=problems,
        )

    @staticmethod
    def _payload(raw: str) -> tuple[list[dict[str, Any]], str]:
        try:
            return json.loads(raw).get("edits", []), ""
        except json.JSONDecodeError as exc:
            return [], f"schema-constrained output still did not parse ({exc.msg})"

    def _feedback_prompt(self, vetoed: Sequence[tuple[int, str, str, str]]) -> str:
        blocks = []
        for index, original, replacement, reason in vetoed:
            blocks.append(
                f"[{index}] original: {original}\n"
                f"    your replacement: {replacement}\n"
                f"    rejected: {reason}"
            )
        return (
            "Some of your edits were rejected by mechanical checks:\n\n"
            + "\n".join(blocks)
            + "\n\nReturn corrected replacements for these sentence numbers only, or omit "
            "any you no longer wish to change. Returning no edits is a valid answer.\n"
            + self._rules()
        )


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

ARMS: dict[str, type] = {
    WholePassage.name: WholePassage,
    ParagraphScoped.name: ParagraphScoped,
    EditList.name: EditList,
    IndexedEditList.name: IndexedEditList,
    FeedbackIndexedEditList.name: FeedbackIndexedEditList,
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
