"""Tests for A2d, the complaint-gated repair arm.

Uses a stub client rather than Ollama, so the acceptance *rule* is tested rather than a
model's behaviour. The rule is the contribution — a model that proposes nonsense and a model
that proposes brilliance must both be handled by the same mechanical check, and only the
check can be pinned by a test.

The property under test throughout: **a repair is applied only if the complaint it cites
disappears and no new complaint appears.** Everything else about this arm follows from that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from revisionbench.litrpg import build_manifest, render_manuscript
from revisionbench.litrpg_detect import detect_all_litrpg
from revisionbench.litrpg_inject import inject_manuscript
from revisionbench.litrpg_repair import repair_manuscript, repair_prompt
from revisionbench.ollama import GenerationOptions

OPTIONS = GenerationOptions(
    seed=0,
    temperature=0.0,
    top_k=40,
    top_p=0.9,
    num_ctx=4096,
    num_predict=128,
    repeat_penalty=1.0,
)


@dataclass
class _Generation:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    done_reason: str = "stop"
    wall_seconds: float = 0.0


class StubClient:
    """Returns scripted replacements. Records every prompt it was given."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, tag, prompt, options, schema=None, think=None):
        self.prompts.append(prompt)
        # Repeat the last scripted reply rather than falling back to a hardcoded empty.
        # With best-of-N the runner asks several times per complaint, and a fallback that
        # happens to *parse* would be ranked and screened alongside the reply under test —
        # which silently changed two tests' rejection reasons to "empty".
        if self.replies:
            self.last = self.replies.pop(0)
        return _Generation(text=getattr(self, "last", '{"replacement": ""}'))


def _corrupted() -> tuple[str, list]:
    manifest = build_manifest("ms-01", chapters=12, seed=0)
    return inject_manuscript(manifest, per_type=1, seed=0)


class TestAcceptanceRule:
    def test_a_repair_that_resolves_its_complaint_is_applied(self) -> None:
        text = "Chapter 1\n  Level: 5\n\nprose one.\n\nChapter 2\n  Level: 3\n\nprose two.\n"
        assert detect_all_litrpg(text)
        client = StubClient(['{"replacement": "  Level: 5"}'])
        report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=1)
        assert [o.reason for o in report.outcomes] == ["accepted"]
        assert report.complaints_after == 0

    def test_a_repair_that_does_not_resolve_its_complaint_is_rejected(self) -> None:
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        client = StubClient(['{"replacement": "  Level: 2"}'])  # still a regression
        report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=1)
        assert [o.reason for o in report.outcomes] == ["complaint_persists"]
        assert report.complaints_after == report.complaints_before

    def test_a_repair_that_introduces_a_new_complaint_is_rejected(self) -> None:
        """The failure a per-edit reviewer cannot see.

        The proposal fixes the level regression it was asked about and, in doing so, makes
        the stats move without a level-up. Judged on its own span it is a correct repair.
        """
        # Chapter 2 has a level SKIP (3 -> 5), which is one complaint; because the level
        # rises, its stat change is legal. "Repairing" the level down to 3 removes the skip
        # and thereby makes the stat change illegal — a new complaint, produced by a
        # proposal that is correct when judged on its own span alone.
        text = (
            "Chapter 1\n  Level: 3\n  Strength: 10\n\nprose.\n\n"
            "Chapter 2\n  Level: 5\n  Strength: 14\n\nprose.\n"
        )
        assert len(detect_all_litrpg(text)) == 1
        client = StubClient(['{"replacement": "  Level: 3"}'])
        report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=1)
        assert [o.reason for o in report.outcomes] == ["new_complaint"]
        assert report.outcomes[0].new_complaints >= 1

    def test_an_oversized_replacement_is_refused(self) -> None:
        """Scope expansion is the behaviour the whole arm exists to prevent."""
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        client = StubClient(['{"replacement": "%s"}' % ("  Level: 5 and then " * 40)])
        report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=1)
        assert [o.reason for o in report.outcomes] == ["out_of_scope"]

    def test_unparseable_output_is_recorded_not_crashed_on(self) -> None:
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        report = repair_manuscript(
            StubClient(["I'm sorry, I can't help with that."]), "stub", text, OPTIONS, max_rounds=1
        )
        assert [o.reason for o in report.outcomes] == ["no_valid_proposal"]

    def test_empty_replacement_is_refused(self) -> None:
        """Deleting the span resolves most complaints and repairs nothing."""
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        report = repair_manuscript(
            StubClient(['{"replacement": "   "}']), "stub", text, OPTIONS, max_rounds=1
        )
        assert [o.reason for o in report.outcomes] == ["empty"]


class TestScoping:
    def test_the_model_never_sees_the_whole_manuscript(self) -> None:
        """Eligibility is structural. A model that cannot see a span cannot edit it."""
        text, _ = _corrupted()
        client = StubClient(['{"replacement": ""}'] * 40)
        repair_manuscript(client, "stub", text, OPTIONS, max_rounds=1)
        assert client.prompts
        for prompt in client.prompts:
            assert len(prompt) < len(text), "the whole manuscript reached the model"

    def test_a_clean_manuscript_produces_no_model_calls_at_all(self) -> None:
        """No complaint, no edit — and no spend."""
        clean = render_manuscript(build_manifest("ms-01", chapters=10, seed=0))
        client = StubClient(['{"replacement": "x"}'])
        report = repair_manuscript(client, "stub", clean, OPTIONS, max_rounds=3)
        assert client.prompts == []
        assert report.text == clean
        assert report.outcomes == ()

    def test_rejected_repairs_leave_the_text_byte_identical(self) -> None:
        text, _ = _corrupted()
        # Level 999 resolves nothing: it turns every regression into a skip. It must also
        # be refused where the span is a different field, rather than deleting that field.
        client = StubClient(['{"replacement": "  Level: 999"}'] * 40)
        report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=1)
        assert all(not o.applied for o in report.outcomes), [
            (o.reason, o.original, o.proposed) for o in report.outcomes if o.applied
        ]
        assert report.text == text

    def test_prompt_states_the_complaint_and_the_span(self) -> None:
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        complaint = detect_all_litrpg(text)[0]
        prompt = repair_prompt(text, complaint)
        assert complaint.message in prompt
        assert "SPAN TO REPLACE" in prompt


class TestLooping:
    def test_stops_as_soon_as_a_round_accepts_nothing(self) -> None:
        """A manuscript the model cannot improve costs one wasted round, not max_rounds."""
        text = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"
        client = StubClient(['{"replacement": "  Level: 2"}'] * 20)
        report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=5)
        assert report.rounds == 1

    def test_complaints_never_increase(self) -> None:
        """The arm's safety property: repair cannot make the manuscript worse.

        Guaranteed by the acceptance rule rather than by the model, so it must hold for any
        proposal at all — including deliberately destructive ones.
        """
        text, _ = _corrupted()
        for reply in ('{"replacement": "  Level: 1"}', '{"replacement": "Ember Whip"}'):
            client = StubClient([reply] * 60)
            report = repair_manuscript(client, "stub", text, OPTIONS, max_rounds=2)
            assert report.complaints_after <= report.complaints_before


class TestBestOfN:
    """Ranked by minimal intervention, not by order of generation or by length."""

    LEVEL_TEXT = "Chapter 1\n  Level: 5\n\nprose.\n\nChapter 2\n  Level: 3\n\nprose.\n"

    def test_edit_distance_ranks_the_least_invasive_repair_first(self) -> None:
        from revisionbench.litrpg_repair import edit_distance

        original = "  Level: 3"
        assert edit_distance(original, "  Level: 5") == 1
        assert edit_distance(original, "  Level: 5 (after the trial)") > 1
        assert edit_distance(original, original) == 0

    def test_the_smallest_clearing_candidate_wins_not_the_first(self) -> None:
        """The point of best-of-N: a later, smaller candidate beats an earlier valid one.

        Both proposals clear the complaint. The first generated rewrites more than it needs
        to; the second changes one character. Without ranking, the first would land.
        """
        client = StubClient(
            ['{"replacement": "  Level: 5 at last"}', '{"replacement": "  Level: 5"}']
        )
        report = repair_manuscript(
            client, "stub", self.LEVEL_TEXT, OPTIONS, max_rounds=1, candidates=2
        )
        assert [o.reason for o in report.outcomes] == ["accepted"]
        assert report.outcomes[0].proposed == "  Level: 5"
        assert report.outcomes[0].candidates_seen == 2

    def test_a_failing_smallest_candidate_falls_through_to_the_next(self) -> None:
        """Ranking picks the order to *try*, not the winner. Verification still decides."""
        client = StubClient(['{"replacement": "  Level: 4"}', '{"replacement": "  Level: 5"}'])
        report = repair_manuscript(
            client, "stub", self.LEVEL_TEXT, OPTIONS, max_rounds=1, candidates=2
        )
        # "Level: 4" is distance 1 and still a regression from 5; "Level: 5" is also
        # distance 1 but clears it. Whichever is tried first, only the clearing one lands.
        assert [o.reason for o in report.outcomes] == ["accepted"]
        assert report.outcomes[0].proposed == "  Level: 5"

    def test_identical_candidates_are_deduplicated(self) -> None:
        """Three seeds agreeing is one candidate, and the record should say so."""
        client = StubClient(['{"replacement": "  Level: 5"}'])
        report = repair_manuscript(
            client, "stub", self.LEVEL_TEXT, OPTIONS, max_rounds=1, candidates=3
        )
        assert len(client.prompts) == 3  # still paid for three generations
        assert report.outcomes[0].candidates_seen == 1  # but there was only one proposal

    def test_later_candidates_are_sampled_not_just_reseeded(self) -> None:
        """The bug this pins cost 2.5x GPU time for zero variation.

        At temperature 0 decoding is deterministic, so changing only the seed returns N
        byte-identical strings — measured as candidates_seen == 1 on all 67 complaints of
        a real run. The first candidate stays greedy so the primary proposal is
        reproducible; every later one must actually sample.
        """
        seen: list[tuple[int, float]] = []

        class SeedRecorder(StubClient):
            def generate(self, tag, prompt, options, schema=None, think=None):
                seen.append((options.seed, options.temperature))
                return super().generate(tag, prompt, options, schema, think)

        repair_manuscript(
            SeedRecorder(['{"replacement": "  Level: 5"}']),
            "stub",
            self.LEVEL_TEXT,
            OPTIONS,
            max_rounds=1,
            candidates=3,
        )
        assert len({s for s, _ in seen[:3]}) == 3, seen
        assert seen[0][1] == OPTIONS.temperature, "the first candidate must stay greedy"
        assert all(t > 0 for _, t in seen[1:3]), f"later candidates were not sampled: {seen}"

    def test_candidates_one_is_the_old_single_proposal_behaviour(self) -> None:
        client = StubClient(['{"replacement": "  Level: 5"}'])
        report = repair_manuscript(
            client, "stub", self.LEVEL_TEXT, OPTIONS, max_rounds=1, candidates=1
        )
        assert len(client.prompts) == 1
        assert [o.reason for o in report.outcomes] == ["accepted"]


@pytest.mark.parametrize("seed", range(3))
def test_end_to_end_with_a_perfect_model_restores_the_manuscript(seed: int) -> None:
    """An oracle model that always proposes the original text must reach zero complaints.

    Pins the ceiling: if this fails, a shortfall in the real run is the harness's fault
    rather than the model's, and that distinction is not otherwise visible.
    """
    manifest = build_manifest(f"ms-{seed:02d}", chapters=12, seed=seed)
    text, defects = inject_manuscript(manifest, per_type=1, seed=seed)

    class Oracle(StubClient):
        def generate(self, tag, prompt, options, schema=None, think=None):
            self.prompts.append(prompt)
            marker = ">>> SPAN TO REPLACE: "
            span = prompt.split(marker)[1].split("\n")[0]
            import json as _json

            # A complaint's span is often a sub-span of the planted fragment: the
            # inventory_ghost detector points at "cracked whetstone" while the injector
            # replaced "the cracked whetstone". Map the span back through the same
            # surrounding context so the oracle answers the question it was asked.
            for defect in defects:
                corrupt, clean_text = defect.corrupt_fragment, defect.original_fragment
                if span.strip() == corrupt.strip():
                    return _Generation(text=_json.dumps({"replacement": clean_text}))
                if span in corrupt:
                    lead = corrupt.index(span)
                    trail = len(corrupt) - lead - len(span)
                    core = clean_text[lead : len(clean_text) - trail if trail else None]
                    return _Generation(text=_json.dumps({"replacement": core}))
            return _Generation(text=_json.dumps({"replacement": span}))

    report = repair_manuscript(Oracle([]), "stub", text, OPTIONS, max_rounds=3)
    assert report.complaints_after == 0, [o.reason for o in report.outcomes]
