"""The A0 revision loop: feed-forward, guards, stopping rules, and resume.

Offline. A fake reviser stands in for Ollama so the loop's control flow is tested without
a GPU, which is also the only way to exercise the branches that matter — collapse,
fixed point, and a resumed trajectory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from revisionbench.loop import (
    RESUME_KEY_FIELDS,
    LoopSpec,
    PromptSpec,
    recover_texts,
    run_passage,
    summarise_generation,
)
from revisionbench.ollama import Generation, GenerationOptions, ModelIdentity
from revisionbench.records import JsonlWriter, read_jsonl, resume_index

OPTIONS = GenerationOptions(
    seed=0,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    num_ctx=8192,
    num_predict=3072,
    repeat_penalty=1.0,
)
MODEL = ModelIdentity(
    tag="fake:1b",
    digest="d" * 64,
    family="fake",
    parameter_size="1B",
    quantization_level="Q4",
    ollama_version="0.0.0",
)
SPEC = LoopSpec(arm="A0", rounds=4, length_guard=(0.7, 1.4), min_words_to_continue=10)
PROMPT = PromptSpec(name="neutral", template="Revise this.\n\n{passage}")

SENTENCE = "The wind came off the water in short gusts and the gulls went over calling."
PASSAGE = {
    "passage_id": "test-01",
    "author_id": "tester",
    "fame": "famous",
    "stratum": "A",
    "text": " ".join([SENTENCE] * 6),
}
CONFIG_HASH = "cafef00d"


class FakeClient:
    """Returns a scripted sequence of texts and records the prompts it was given."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, tag: str, prompt: str, options: GenerationOptions) -> Generation:
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("FakeClient ran out of scripted outputs")
        return Generation(
            text=self.outputs.pop(0),
            prompt_tokens=100,
            output_tokens=50,
            done_reason="stop",
            wall_seconds=0.1,
        )


def drive(client, tmp_path: Path, *, spec=SPEC, done=None, texts=None, passage=PASSAGE):
    path = tmp_path / "rounds.jsonl"
    with JsonlWriter(path, fsync=False) as writer:
        rows = list(
            run_passage(
                client,
                passage,
                prompt=PROMPT,
                model=MODEL,
                options=OPTIONS,
                spec=spec,
                writer=writer,
                config_hash=CONFIG_HASH,
                already_done=done,
                resume_texts=texts,
            )
        )
    return rows, path


class TestPromptSpec:
    def test_requires_a_passage_placeholder(self) -> None:
        with pytest.raises(ValueError, match=r"no \{passage\} placeholder"):
            PromptSpec(name="bad", template="Revise something.")

    def test_renders_passage_and_word_count(self) -> None:
        prompt = PromptSpec(name="p", template="{word_count} words:\n{passage}")
        assert prompt.render("one two three") == "3 words:\none two three"

    def test_braces_in_the_passage_do_not_break_rendering(self) -> None:
        """Literary text can contain braces; str.format would raise or misinterpret."""
        prompt = PromptSpec(name="p", template="Revise:\n{passage}")
        assert prompt.render("a {weird} brace") == "Revise:\na {weird} brace"

    def test_hash_tracks_the_template(self) -> None:
        assert PromptSpec("a", "x {passage}").sha256 != PromptSpec("a", "y {passage}").sha256
        assert PromptSpec("a", "x {passage}").sha256 == PromptSpec("b", "x {passage}").sha256


class TestLoopSpec:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"rounds": 0}, "rounds must be"),
            ({"length_guard": (1.2, 1.4)}, "length_guard must be"),
            ({"length_guard": (0.7, 0.9)}, "length_guard must be"),
            ({"min_words_to_continue": 0}, "min_words_to_continue"),
        ],
    )
    def test_rejects_incoherent_settings(self, kwargs, match) -> None:
        base = {"arm": "A0", "rounds": 4, "length_guard": (0.7, 1.4), "min_words_to_continue": 10}
        with pytest.raises(ValueError, match=match):
            LoopSpec(**{**base, **kwargs})


class TestFeedForward:
    def test_round_zero_is_the_original_and_makes_no_model_call(self, tmp_path: Path) -> None:
        client = FakeClient([f"{SENTENCE} v{i}" * 6 for i in range(4)])
        rows, _ = drive(client, tmp_path)
        assert rows[0]["round"] == 0
        assert rows[0]["text"] == PASSAGE["text"]
        assert rows[0]["generation"] is None
        assert len(client.prompts) == 4, "one call per revision round, none for round 0"

    def test_each_round_revises_the_previous_output(self, tmp_path: Path) -> None:
        """The feed-forward is the loop. Without it these are ten independent rewrites."""
        outputs = [" ".join([f"Version {i} text here."] * 12) for i in range(1, 5)]
        client = FakeClient(outputs)
        drive(client, tmp_path)
        assert PASSAGE["text"] in client.prompts[0]
        for index in range(1, 4):
            assert outputs[index - 1] in client.prompts[index]

    def test_rows_carry_the_config_hash(self, tmp_path: Path) -> None:
        client = FakeClient([" ".join([SENTENCE] * 6)] * 4)
        rows, _ = drive(client, tmp_path)
        assert all(r["config_hash"] == CONFIG_HASH for r in rows)


class TestLengthGuard:
    def test_flags_a_collapse_without_stopping_the_loop(self, tmp_path: Path) -> None:
        """A0 is the unconstrained control; intervening would make it a different arm.

        The four outputs must differ from each other, or the fixed-point rule stops the
        loop first and this would be testing that instead.
        """
        short = [" ".join([f"Short version {i} of the text here."] * 3) for i in range(4)]
        client = FakeClient(short)
        rows, _ = drive(client, tmp_path)
        revisions = [r for r in rows if r["round"] > 0]
        assert revisions[0]["length_guard_tripped"] is True
        assert revisions[0]["length_ratio"] < 0.7
        assert len(revisions) == 4, "the loop must keep running"

    def test_does_not_flag_a_faithful_revision(self, tmp_path: Path) -> None:
        client = FakeClient([" ".join([SENTENCE] * 6)] * 4)
        rows, _ = drive(client, tmp_path)
        assert not any(r["length_guard_tripped"] for r in rows)


class TestStopping:
    def test_stops_when_the_text_collapses(self, tmp_path: Path) -> None:
        client = FakeClient(["Too short now.", "never reached"])
        rows, _ = drive(client, tmp_path)
        assert rows[-1]["stop_reason"] == "collapsed"
        assert len(client.prompts) == 1

    def test_stops_at_a_fixed_point(self, tmp_path: Path) -> None:
        """Under a reproducible sampler a repeat is absorbing, so continuing is pointless."""
        settled = " ".join(["The loop has settled on this wording now."] * 8)
        client = FakeClient([settled, settled, "never reached"])
        rows, _ = drive(client, tmp_path)
        assert rows[-1]["stop_reason"] == "fixed_point"
        assert rows[-1]["round"] == 2, "round 1 differs from round 0; round 2 repeats round 1"
        assert len(client.prompts) == 2

    def test_round_one_repeating_the_original_is_a_fixed_point(self, tmp_path: Path) -> None:
        """A reviser that returns its input unchanged has already converged."""
        client = FakeClient([PASSAGE["text"], "never reached"])
        rows, _ = drive(client, tmp_path)
        assert rows[-1]["stop_reason"] == "fixed_point"
        assert rows[-1]["round"] == 1

    def test_fixed_point_ignores_typographic_churn(self, tmp_path: Path) -> None:
        """plan.md §8 A5 idempotence: re-typesetting is not a proposal."""
        base = " ".join(["He said--yes, it's fine, and then he left again."] * 8)
        client = FakeClient([base, base.replace("--", "—").replace("'", "’"), "never reached"])
        rows, _ = drive(client, tmp_path)
        assert rows[-1]["stop_reason"] == "fixed_point"
        assert rows[-1]["round"] == 2

    def test_runs_to_the_cap_when_nothing_settles(self, tmp_path: Path) -> None:
        client = FakeClient([" ".join([f"Round {i} sentence here."] * 15) for i in range(1, 5)])
        rows, _ = drive(client, tmp_path)
        assert [r["round"] for r in rows] == [0, 1, 2, 3, 4]
        assert all(r["stop_reason"] is None for r in rows)


class TestResume:
    def test_resumed_run_skips_done_rounds_and_continues_the_chain(self, tmp_path: Path) -> None:
        first = [" ".join([f"Round {i} sentence here."] * 15) for i in range(1, 5)]
        client = FakeClient(first)
        spec2 = LoopSpec(arm="A0", rounds=2, length_guard=(0.7, 1.4), min_words_to_continue=10)
        rows, path = drive(client, tmp_path, spec=spec2)
        assert [r["round"] for r in rows] == [0, 1, 2]

        done = resume_index(path, RESUME_KEY_FIELDS)
        prior = list(read_jsonl(path))
        client2 = FakeClient(first[2:])
        rows2, _ = drive(client2, tmp_path, done=done, texts=recover_texts(prior))
        assert [r["round"] for r in rows2] == [3, 4], "only the missing rounds re-run"
        # The chain must continue from round 2's text, not from round 0's.
        assert first[1] in client2.prompts[0]

    def test_resume_without_recovered_text_is_fatal(self, tmp_path: Path) -> None:
        """Splicing two trajectories would be invisible in the artifact."""
        client = FakeClient(["x"])
        done = {("cafef00d", "test-01", "A0", "neutral", "d" * 64, r) for r in (0, 1)}
        with pytest.raises(ValueError, match="feed-forward chain cannot be rebuilt"):
            drive(client, tmp_path, done=done, texts={0: PASSAGE["text"]})

    def test_resume_key_includes_config_hash_and_digest(self) -> None:
        """Both omissions would silently splice runs that are not comparable."""
        assert "config_hash" in RESUME_KEY_FIELDS
        assert "model_digest" in RESUME_KEY_FIELDS


class TestCostSummary:
    def test_aggregates_generations_only(self, tmp_path: Path) -> None:
        client = FakeClient([" ".join([f"Round {i} here."] * 20) for i in range(1, 5)])
        rows, _ = drive(client, tmp_path)
        stats = summarise_generation(rows)
        assert stats["generations"] == 4
        assert stats["prompt_tokens_total"] == 400
        assert stats["truncated_rounds"] == 0

    def test_empty_when_nothing_was_generated(self) -> None:
        assert summarise_generation([{"generation": None}]) == {"generations": 0}
