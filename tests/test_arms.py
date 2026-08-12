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

    def generate(self, tag, prompt, options, schema=None):
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
        assert set(ARMS) == {"whole", "paragraph", "editlist", "indexed", "indexed_fb"}
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


class TestIndexedEditList:
    """A2i: symbolic addressing + enforced schema + per-edit mechanical vetoes."""

    def payload(self, edits):
        return json.dumps({"edits": [{"sentence_index": i, "replacement": r} for i, r in edits]})

    def arm(self, **kwargs):
        from revisionbench.arms import IndexedEditList

        return IndexedEditList(**kwargs)

    def test_applies_an_indexed_edit_to_the_right_sentence(self) -> None:
        context, _ = ctx([self.payload([(1, "She strolled down the lane, counting gates.")])])
        result = self.arm(vetoes=()).revise(context, PASSAGE)
        assert "She strolled down the lane" in result.text
        assert PARA_A in result.text, "the untouched sentence is byte-identical"
        assert (result.proposed, result.applied, result.rejected) == (1, 1, 0)

    def test_an_integer_cannot_be_misquoted(self) -> None:
        """The whole point of the arm.

        A2e lost 223 of 236 rejected edits to the model paraphrasing its `find` anchor.
        Indices remove that failure class by construction rather than reducing it.
        """
        context, _ = ctx([self.payload([(0, "Rewritten first sentence.")])])
        result = self.arm(vetoes=()).revise(context, PASSAGE)
        assert result.applied == 1
        assert not any("not found" in p for p in result.problems)

    def test_the_model_is_shown_numbered_sentences(self) -> None:
        context, client = ctx([self.payload([])])
        self.arm().revise(context, PASSAGE)
        assert "[0] " in client.prompts[0] and "[1] " in client.prompts[0]

    def test_out_of_range_index_is_rejected_not_clamped(self) -> None:
        context, _ = ctx([self.payload([(99, "Nope.")])])
        result = self.arm(vetoes=()).revise(context, PASSAGE)
        assert (result.applied, result.rejected) == (0, 1)
        assert any("out of range" in p for p in result.problems)

    def test_duplicate_index_keeps_the_first_only(self) -> None:
        context, _ = ctx([self.payload([(0, "First try here now."), (0, "Second try here.")])])
        result = self.arm(vetoes=()).revise(context, PASSAGE)
        assert result.applied == 1
        assert any("duplicate" in p for p in result.problems)

    def test_whitespace_and_paragraph_structure_survive(self) -> None:
        """An arm that reflowed the passage would move the punctuation metrics itself."""
        context, _ = ctx([self.payload([(0, "A short new first sentence.")])])
        result = self.arm(vetoes=()).revise(context, PASSAGE)
        assert result.text.count("\n\n") == PASSAGE.count("\n\n")

    def test_empty_edit_list_is_a_clean_no_op(self) -> None:
        context, _ = ctx([self.payload([])])
        result = self.arm().revise(context, PASSAGE)
        assert result.text == PASSAGE
        assert result.problems == []

    def test_length_veto_rejects_a_collapsing_replacement(self) -> None:
        """Applied per edit, so a model that decided to summarise cannot defeat it."""
        context, _ = ctx([self.payload([(0, "Wind.")])])
        result = self.arm(vetoes=("length",)).revise(context, PASSAGE)
        assert result.applied == 0
        assert any("length" in p for p in result.problems)

    def test_length_veto_allows_a_proportionate_rewrite(self) -> None:
        context, _ = ctx(
            [self.payload([(0, "The wind blew off the water in gusts and gulls called overhead.")])]
        )
        assert self.arm(vetoes=("length",)).revise(context, PASSAGE).applied == 1

    def test_slop_veto_rejects_an_introduced_lexicon_term(self) -> None:
        """M3 measured slop rising 0.54 -> 3.73 under A0; this makes the rise impossible."""
        context, _ = ctx(
            [self.payload([(0, "A wave of unease hung in the air above the restless water.")])]
        )
        result = self.arm(vetoes=("slop",)).revise(context, PASSAGE)
        assert result.applied == 0
        assert any("slop" in p for p in result.problems)

    def test_slop_veto_permits_a_term_already_present(self) -> None:
        """Only *introduced* slop is vetoed, or the arm could never touch such a sentence."""
        text = "There was a sense of unease about the whole business that morning.\n\nSecond one."
        context, _ = ctx(
            [
                self.payload(
                    [(0, "There was a sense of unease about the affair that whole morning.")]
                )
            ]
        )
        assert self.arm(vetoes=("slop",)).revise(context, text).applied == 1

    def test_unknown_veto_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown veto"):
            self.arm(vetoes=("vibes",))

    def test_schema_is_sent_to_the_client(self) -> None:
        from revisionbench.arms import INDEXED_EDIT_SCHEMA

        seen = {}

        class SchemaRecordingClient(FakeClient):
            def generate(self, tag, prompt, options, schema=None):
                seen["schema"] = schema
                return super().generate(tag, prompt, options)

        from revisionbench.ollama import ModelIdentity

        client = SchemaRecordingClient([self.payload([])])
        context = ReviseContext(
            client=client,
            model=ModelIdentity("fake:1b", "d" * 64, "fake", "1B", "Q4", "0.0.0"),
            options=OPTIONS,
            template="Revise:\n{passage}",
        )
        self.arm().revise(context, PASSAGE)
        assert seen["schema"] == INDEXED_EDIT_SCHEMA

    def test_unparseable_output_is_still_recorded(self) -> None:
        """Constrained decoding should make this unreachable; assuming so is how no-op
        arms get shipped."""
        context, _ = ctx(["not json at all"])
        result = self.arm().revise(context, PASSAGE)
        assert result.text == PASSAGE
        assert any("did not parse" in p for p in result.problems)


class TestFeedbackIndexedEditList:
    """A2f: constraint-visible protocol + punctuation veto + one per-edit feedback retry."""

    def payload(self, edits):
        return json.dumps({"edits": [{"sentence_index": i, "replacement": r} for i, r in edits]})

    def arm(self, **kwargs):
        from revisionbench.arms import FeedbackIndexedEditList

        return FeedbackIndexedEditList(**kwargs)

    def test_protocol_states_the_enforced_band(self) -> None:
        """The rule text is generated from the enforcing attributes, so the stated protocol
        and the applied veto cannot drift apart — asserted, not assumed."""
        context, client = ctx([self.payload([])])
        arm = self.arm()
        arm.revise(context, PASSAGE)
        low, high = arm.length_band
        assert f"{low}x" in client.prompts[0] and f"{high}x" in client.prompts[0]
        assert "punctuation" in client.prompts[0]

    def test_punctuation_veto_rejects_a_restyling_edit(self) -> None:
        """Targets the one attractor effect Phase 0 found surviving length control."""
        context, _ = ctx(
            [
                self.payload(
                    [(0, "The wind, in short gusts, came off the water, and gulls went over.")]
                ),
                self.payload([]),  # the veto triggers a feedback pass; the model declines
            ]
        )
        result = self.arm(vetoes=("punctuation",)).revise(context, PASSAGE)
        assert result.applied == 0
        assert any("punctuation shift" in p for p in result.problems)
        assert any("1 vetoed, 0 recovered" in p for p in result.problems)

    def test_punctuation_veto_allows_an_incremental_change(self) -> None:
        context, _ = ctx(
            [
                self.payload(
                    [(0, "The wind came off the water in short gusts, and the gulls went over.")]
                )
            ]
        )
        result = self.arm(vetoes=("punctuation",)).revise(context, PASSAGE)
        assert result.applied == 1
        assert not any("feedback" in p for p in result.problems), "nothing was vetoed"

    def test_vetoed_edit_is_recovered_by_feedback(self) -> None:
        """The retry differs in kind from the dead round-level re-roll: the prompt changes —
        it now carries the constraint and the measured violation."""
        context, client = ctx(
            [
                self.payload([(0, "Wind.")]),  # collapses: length-vetoed
                self.payload([(0, "The wind blew off the water in gusts and gulls called.")]),
            ]
        )
        result = self.arm(vetoes=("length",)).revise(context, PASSAGE)
        assert result.applied == 1
        assert "The wind blew off the water" in result.text
        assert len(client.prompts) == 2
        assert "rejected: length" in client.prompts[1]
        assert PARA_A in client.prompts[1], "the feedback prompt shows the original sentence"
        assert any("1 vetoed, 1 recovered" in p for p in result.problems)

    def test_no_feedback_call_when_nothing_was_vetoed(self) -> None:
        """FakeClient raises if a second output is requested, so this also asserts cost."""
        context, client = ctx(
            [self.payload([(0, "The wind blew off the water in gusts and gulls called.")])]
        )
        result = self.arm().revise(context, PASSAGE)
        assert result.applied == 1
        assert len(client.prompts) == 1

    def test_feedback_cannot_smuggle_in_an_unvetoed_sentence(self) -> None:
        context, _ = ctx(
            [
                self.payload([(0, "Wind.")]),  # only sentence 0 is vetoed
                self.payload([(1, "A brand new second sentence smuggled in by the retry.")]),
            ]
        )
        result = self.arm(vetoes=("length",)).revise(context, PASSAGE)
        assert result.applied == 0
        assert PARA_B in result.text
        assert any("unsolicited index 1" in p for p in result.problems)

    def test_still_violating_retry_is_rejected_and_not_retried_again(self) -> None:
        context, client = ctx(
            [
                self.payload([(0, "Wind.")]),
                self.payload([(0, "Gusts.")]),  # still collapses
            ]
        )
        result = self.arm(vetoes=("length",)).revise(context, PASSAGE)
        assert result.applied == 0
        assert len(client.prompts) == 2, "one retry, never a second"
        assert any("1 vetoed, 0 recovered" in p for p in result.problems)

    def test_unknown_veto_lists_the_extended_set(self) -> None:
        with pytest.raises(ValueError, match="punctuation"):
            self.arm(vetoes=("vibes",))


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
