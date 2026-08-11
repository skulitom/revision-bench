"""Revision strategies (plan.md §8 arms).

The tests that matter most are the silent-failure ones. An arm that parses nothing, or
proposes nothing, returns the passage unchanged — and unchanged text scores as perfect
voice preservation with zero slop, so a broken arm looks like the winning arm. Every such
path must be distinguishable in the telemetry.
"""

from __future__ import annotations

import json

import pytest

from revisionbench.arms import (
    ARMS,
    EditList,
    ParagraphScoped,
    Proposal,
    ReviseContext,
    WholePassage,
    build_strategy,
)
from revisionbench.ollama import Generation, GenerationOptions

OPTIONS = GenerationOptions(
    seed=0,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    num_ctx=8192,
    num_predict=3072,
    repeat_penalty=1.0,
)

PARA_A = "The wind came off the water in short gusts and the gulls went over calling."
PARA_B = "She walked down the lane past the low wall, counting the gates as she went along."
PASSAGE = f"{PARA_A}\n\n{PARA_B}"


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate(self, tag, prompt, options):
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("FakeClient ran out of scripted outputs")
        return Generation(
            text=self.outputs.pop(0),
            prompt_tokens=10,
            output_tokens=5,
            done_reason="stop",
            wall_seconds=0.1,
        )


def ctx(outputs, template="Revise:\n{passage}"):
    from revisionbench.ollama import ModelIdentity

    client = FakeClient(outputs)
    model = ModelIdentity("fake:1b", "d" * 64, "fake", "1B", "Q4", "0.0.0")
    return ReviseContext(client=client, model=model, options=OPTIONS, template=template), client


class TestRegistry:
    def test_known_names(self) -> None:
        assert set(ARMS) == {"whole", "paragraph", "editlist"}
        assert isinstance(build_strategy("whole"), WholePassage)

    def test_unknown_name_is_fatal(self) -> None:
        """A typo must not fall back to the control and report A0 under another label."""
        with pytest.raises(ValueError, match="unknown strategy"):
            build_strategy("paragrpah")


class TestWholePassage:
    def test_replaces_the_whole_text(self) -> None:
        context, client = ctx(["A completely new passage."])
        result = WholePassage().revise(context, PASSAGE)
        assert result.text == "A completely new passage."
        assert (result.units, result.proposed, result.applied) == (1, 1, 1)
        assert len(client.prompts) == 1

    def test_unchanged_output_is_reported_as_not_applied(self) -> None:
        context, _ = ctx([PASSAGE])
        result = WholePassage().revise(context, PASSAGE)
        assert result.applied == 0
        assert result.changed is False


class TestParagraphScoped:
    def test_revises_each_paragraph_separately(self) -> None:
        context, client = ctx(["First revised.", "Second revised."])
        result = ParagraphScoped().revise(context, PASSAGE)
        assert result.text == "First revised.\n\nSecond revised."
        assert result.units == 2
        assert result.applied == 2
        assert len(client.prompts) == 2, "one call per paragraph"
        assert PARA_A in client.prompts[0] and PARA_B not in client.prompts[0]

    def test_paragraph_structure_is_preserved(self) -> None:
        context, _ = ctx(["One.", "Two."])
        assert ParagraphScoped().revise(context, PASSAGE).text.count("\n\n") == 1

    def test_short_paragraphs_are_passed_through(self) -> None:
        """Asking a model to improve `* * *` invites it to write a paragraph there."""
        passage = f"{PARA_A}\n\n* * *\n\n{PARA_B}"
        context, client = ctx(["First revised.", "Second revised."])
        result = ParagraphScoped().revise(context, passage)
        assert "* * *" in result.text
        assert result.rejected == 1
        assert len(client.prompts) == 2, "the scene break is never sent to the model"

    def test_empty_revision_keeps_the_original_paragraph(self) -> None:
        context, _ = ctx(["   ", "Second revised."])
        result = ParagraphScoped().revise(context, PASSAGE)
        assert PARA_A in result.text
        assert result.problems, "an empty revision must be recorded, not silently accepted"

    def test_bounds_compression_relative_to_whole_passage(self) -> None:
        """The point of the arm: a dropped paragraph cannot take the passage with it."""
        context, _ = ctx(["Short.", "Also short."])
        scoped = ParagraphScoped().revise(context, PASSAGE)
        assert scoped.text.count("\n\n") == 1, "both paragraph slots still exist"

    def test_a_split_revision_is_rejoined_into_one_paragraph(self) -> None:
        """The duplication cascade this arm shipped with, pinned.

        A model that returns a paragraph as two makes the next round see two units and
        revise each separately. Compounded over five rounds on woolf-01 that took the
        passage from 7 paragraphs and 901 words to 20 paragraphs and 1569 words, while the
        cross-passage mean length ratio read a reassuring 1.03.
        """
        context, _ = ctx(["First half.\n\nSecond half.", "Second revised."])
        result = ParagraphScoped().revise(context, PASSAGE)
        assert result.text == "First half. Second half.\n\nSecond revised."
        assert any("rejoined" in p for p in result.problems)

    def test_paragraph_count_is_invariant(self) -> None:
        """A bounded-diff architecture whose unit boundaries move is not bounded."""
        from revisionbench.text import normalise_newlines, paragraph_spans

        context, _ = ctx(["A.\n\nB.\n\nC.", "D.\n\nE."])
        result = ParagraphScoped().revise(context, PASSAGE)
        assert len(paragraph_spans(normalise_newlines(result.text))) == 2
        assert not any("count changed" in p for p in result.problems)


class TestEditList:
    def edits(self, pairs):
        return json.dumps([{"find": f, "replace": r} for f, r in pairs])

    def test_applies_unambiguous_edits(self) -> None:
        context, _ = ctx([self.edits([("in short gusts and the gulls", "in gusts and the gulls")])])
        result = EditList().revise(context, PASSAGE)
        assert "in gusts and the gulls" in result.text
        assert (result.proposed, result.applied, result.rejected) == (1, 1, 0)

    def test_text_outside_the_named_span_is_untouched(self) -> None:
        """The property the arm exists for: unnamed text structurally cannot change."""
        context, _ = ctx([self.edits([("counting the gates as she went", "counting gates")])])
        result = EditList().revise(context, PASSAGE)
        assert PARA_A in result.text

    def test_missing_anchor_is_rejected_and_named(self) -> None:
        context, _ = ctx([self.edits([("text that is nowhere in the passage", "x")])])
        result = EditList().revise(context, PASSAGE)
        assert (result.applied, result.rejected) == (0, 1)
        assert any("not found" in p for p in result.problems)

    def test_ambiguous_anchor_is_rejected(self) -> None:
        """Applying to the first of several matches would be arbitrary."""
        doubled = f"{PARA_A}\n\n{PARA_A}"
        context, _ = ctx([self.edits([("the water in short gusts", "the sea in short gusts")])])
        result = EditList().revise(context, doubled)
        assert result.rejected == 1
        assert any("ambiguous" in p for p in result.problems)

    def test_short_anchor_is_rejected(self) -> None:
        context, _ = ctx([self.edits([("the", "a")])])
        result = EditList().revise(context, PASSAGE)
        assert result.rejected == 1
        assert any("too short" in p for p in result.problems)

    def test_deletion_via_empty_replace(self) -> None:
        context, _ = ctx([self.edits([(" in short gusts", "")])])
        result = EditList().revise(context, PASSAGE)
        assert "in short gusts" not in result.text

    def test_fenced_json_is_parsed(self) -> None:
        payload = (
            "Here you go:\n```json\n"
            + self.edits([("in short gusts and", "in gusts and")])
            + "\n```"
        )
        context, _ = ctx([payload])
        assert EditList().revise(context, PASSAGE).applied == 1

    def test_prose_reply_is_recorded_not_silently_ignored(self) -> None:
        """THE trap: unparseable output leaves the passage pristine, which scores as ideal.

        Without a recorded problem, a model that never learned the protocol would top the
        voice-preservation table while doing nothing at all.
        """
        context, _ = ctx(["I think the passage is quite good as it stands."])
        result = EditList().revise(context, PASSAGE)
        assert result.text == PASSAGE
        assert result.proposed == 0
        assert result.problems, "a parse failure must be visible in the telemetry"
        assert any("unparseable" in p for p in result.problems)

    def test_empty_array_is_a_valid_no_op_distinct_from_a_parse_failure(self) -> None:
        context, _ = ctx(["[]"])
        result = EditList().revise(context, PASSAGE)
        assert result.text == PASSAGE
        assert result.proposed == 0
        assert result.problems == [], "proposing nothing is not an error"

    def test_non_array_json_is_recorded(self) -> None:
        context, _ = ctx(['{"find": "x", "replace": "y"}'])
        result = EditList().revise(context, PASSAGE)
        assert any("not an array" in p for p in result.problems)

    def test_malformed_entries_are_counted_individually(self) -> None:
        payload = json.dumps(
            [{"nope": 1}, {"find": "in short gusts and the", "replace": "and the"}]
        )
        context, _ = ctx([payload])
        result = EditList().revise(context, PASSAGE)
        assert result.applied == 1
        assert any("malformed" in p for p in result.problems)

    def test_protocol_instruction_reaches_the_model(self) -> None:
        context, client = ctx(["[]"])
        EditList().revise(context, PASSAGE)
        assert "JSON array of edits" in client.prompts[0]


class TestProposalTelemetry:
    def test_distinguishes_the_three_ways_of_changing_nothing(self) -> None:
        proposed_none = Proposal(text="x", proposed=0, applied=0, rejected=0)
        all_rejected = Proposal(text="x", proposed=3, applied=0, rejected=3)
        unparseable = Proposal(text="x", proposed=0, applied=0, problems=["unparseable"])
        assert not any(p.changed for p in (proposed_none, all_rejected, unparseable))
        assert proposed_none.as_dict()["proposed"] == 0 and not proposed_none.problems
        assert all_rejected.as_dict()["rejected"] == 3
        assert unparseable.as_dict()["problems"] == ["unparseable"]

    def test_aggregates_generation_cost(self) -> None:
        gens = [Generation("a", 10, 5, "stop", 1.0), Generation("b", 20, 7, "length", 2.0)]
        data = Proposal(text="x", generations=gens).as_dict()
        assert data["generations"] == 2
        assert data["prompt_tokens"] == 30 and data["output_tokens"] == 12
        assert data["truncated"] is True, "one truncated call makes the round suspect"
