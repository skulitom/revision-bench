"""The revision loop runner (plan.md §8). Arm A0 — unconstrained — only, for now.

A0 is the control that characterises the trap: every round, ask the model to improve the
passage, and always accept. The one thing that makes it a *loop* rather than ten
independent rewrites is that round k+1's input is round k's **output**. That feed-forward
is the whole mechanism plan.md §4 predicts an attractor from, so it is stated here rather
than left implicit in a script.

This module writes **text and generation facts, and no metric values**. Metrics are a
separate pass over the artifact (``scripts/phase0_metrics.py``). Keeping them apart is what
makes plan.md §9's acceptance criterion — "metrics reproducible from artifacts alone" —
something that can actually be checked: if the numbers lived in the same file the model
produced, "recomputed" and "recorded" could drift apart and nothing would notice. It also
means a change to a metric costs a re-read of a JSONL file rather than 100 GPU generations.

Later arms (A1–A5) need an acceptance gate between proposal and state, which is a different
control flow and a Phase-3 concern. There is deliberately no gate hook here yet.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from revisionbench.ollama import Generation, GenerationOptions, ModelIdentity, OllamaClient
from revisionbench.provenance import sha256_text
from revisionbench.records import JsonlWriter
from revisionbench.text import word_count, words

__all__ = [
    "ARM_A0",
    "RESUME_KEY_FIELDS",
    "LoopSpec",
    "PromptSpec",
    "RoundOutcome",
    "run_passage",
]

ARM_A0 = "A0"

#: Fields that jointly identify one unit of work, for :func:`revisionbench.records.resume_index`.
#:
#: Two of these are here because leaving them out is a silent-splice hazard rather than a
#: mere inconvenience:
#:
#: - ``model_digest`` — re-running after ``ollama pull`` moved a tag must produce new rows,
#:   not resume someone else's trajectory with different weights.
#: - ``config_hash`` — this one was missing in the first version, and the omission bit
#:   immediately: changing the sampler from greedy to seeded-sampling changes every
#:   generation but changes neither the prompt nor the digest, so a resume would have
#:   stitched the tail of a temperature-0.8 run onto the head of a greedy one and nothing
#:   in the artifact would have shown the join.
#:
#: The cost is that any config edit invalidates resume for the whole sweep. In a
#: measurement repo that is the right trade: redoing 200 generations is cheap, and a
#: spliced trajectory is undetectable.
RESUME_KEY_FIELDS = ("config_hash", "passage_id", "arm", "prompt_name", "model_digest", "round")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """A revise instruction, identified by the hash of its template.

    The prompt is this experiment's main free variable, so it lives in config and its hash
    goes on every row. A prompt living in code is invisible to ``config_hash``, which means
    two runs could differ in the one input that matters most and claim the same provenance.
    """

    name: str
    template: str

    def __post_init__(self) -> None:
        if "{passage}" not in self.template:
            raise ValueError(
                f"prompt {self.name!r} has no {{passage}} placeholder, so the passage would "
                f"never reach the model"
            )

    @property
    def sha256(self) -> str:
        return sha256_text(self.template)

    def render(self, passage: str) -> str:
        """Substitute the passage and its word count into the template.

        Uses ``str.replace`` rather than ``str.format`` deliberately: literary prose may
        contain braces, and ``format`` would raise or, worse, interpret them.
        """
        rendered = self.template.replace("{word_count}", str(word_count(passage)))
        return rendered.replace("{passage}", passage)


@dataclass(frozen=True, slots=True)
class LoopSpec:
    """How a trajectory runs and when it stops early."""

    arm: str
    rounds: int
    #: Word count outside ``round0 * (low, high)`` flags the round. It does **not** stop
    #: the loop: A0 is the unconstrained control, and intervening would make it a different
    #: arm. The flag exists so the length confound (docs/findings-phase0.md §5.2) can be
    #: separated from voice change at analysis time rather than silently averaged into it.
    length_guard: tuple[float, float]
    #: Below this many words the trajectory stops. Revising a stub for seven more rounds
    #: burns GPU time and yields nothing but MetricErrors; the collapse is itself the
    #: result, and it is recorded as one.
    min_words_to_continue: int

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        low, high = self.length_guard
        if not 0.0 < low < 1.0 < high:
            raise ValueError(
                f"length_guard must be (low, high) with 0 < low < 1 < high, got {self.length_guard}"
            )
        if self.min_words_to_continue < 1:
            raise ValueError("min_words_to_continue must be positive")


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """One round's row, before provenance stamping."""

    row: dict[str, Any]
    text: str
    stop: bool
    stop_reason: str | None


def _base_row(
    passage: dict[str, Any],
    prompt: PromptSpec,
    model: ModelIdentity,
    spec: LoopSpec,
    config_hash: str,
):
    return {
        "config_hash": config_hash,
        "passage_id": passage["passage_id"],
        "author_id": passage["author_id"],
        "fame": passage["fame"],
        "stratum": passage["stratum"],
        "arm": spec.arm,
        "prompt_name": prompt.name,
        "prompt_sha256": prompt.sha256,
        "model_tag": model.tag,
        "model_digest": model.digest,
    }


def _text_fields(text: str) -> dict[str, Any]:
    return {
        "text": text,
        "text_sha256": sha256_text(text),
        "word_count": word_count(text),
    }


def run_passage(
    client: OllamaClient,
    passage: dict[str, Any],
    *,
    prompt: PromptSpec,
    model: ModelIdentity,
    options: GenerationOptions,
    spec: LoopSpec,
    writer: JsonlWriter,
    config_hash: str,
    already_done: set[tuple[Any, ...]] | None = None,
    resume_texts: dict[int, str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one passage's trajectory, yielding each row as it is written.

    Round 0 is the original passage and involves no model call. Rounds 1..``spec.rounds``
    each feed the previous round's output back in.

    Args:
        client: A live :class:`~revisionbench.ollama.OllamaClient`.
        passage: A corpus passage record (from ``data/corpus/passages/``).
        prompt: The revise instruction.
        model: Resolved model identity, including digest.
        options: Pinned sampling options.
        spec: Loop shape and stopping rules.
        writer: Destination for rows.
        already_done: Keys from :func:`revisionbench.records.resume_index`; matching rounds
            are skipped.
        resume_texts: ``round -> text`` recovered from a previous run, used to rebuild the
            feed-forward chain without re-generating. A resumed trajectory that could not
            recover its predecessor's text would have to start over, so this is required
            whenever ``already_done`` is non-empty for this passage.

    Raises:
        ValueError: Resume state is incoherent — a later round is present without the round
            it was derived from. Continuing would splice two different trajectories
            together and the join would be invisible in the artifact.
    """
    done = already_done or set()
    texts = dict(resume_texts or {})
    base = _base_row(passage, prompt, model, spec, config_hash)

    def key(round_index: int) -> tuple[Any, ...]:
        row = {**base, "round": round_index}
        return tuple(row[f] for f in RESUME_KEY_FIELDS)

    # --- round 0: the original passage, no model call -------------------------------
    if key(0) not in done:
        row = {
            **base,
            "round": 0,
            **_text_fields(passage["text"]),
            "length_ratio": 1.0,
            "length_guard_tripped": False,
            "generation": None,
            "stop_reason": None,
        }
        writer.write(row)
        yield row
    texts.setdefault(0, passage["text"])
    round_zero_words = word_count(texts[0])

    low, high = spec.length_guard
    previous_text = texts[0]

    for round_index in range(1, spec.rounds + 1):
        if key(round_index) in done:
            if round_index not in texts:
                raise ValueError(
                    f"{passage['passage_id']}: round {round_index} is recorded as done but "
                    f"its text was not recovered, so the feed-forward chain cannot be "
                    f"rebuilt. Delete the artifact and re-run rather than splicing."
                )
            previous_text = texts[round_index]
            continue

        if round_index - 1 not in texts:
            raise ValueError(
                f"{passage['passage_id']}: cannot run round {round_index} because round "
                f"{round_index - 1} is missing from the artifact"
            )

        generation = client.generate(model.tag, prompt.render(previous_text), options)
        text = generation.text.strip()
        current_words = word_count(text)
        ratio = current_words / round_zero_words if round_zero_words else 0.0

        stop_reason = None
        # A fixed point. Under deterministic decoding this is provably absorbing: the next
        # round's input would equal this one's, so its output would equal this one's too.
        # Stopping is therefore free of information loss, and the claim is only made when
        # the sampler actually is deterministic.
        if options.is_deterministic and words(text) == words(previous_text):
            stop_reason = "fixed_point"
        elif current_words < spec.min_words_to_continue:
            stop_reason = "collapsed"

        row = {
            **base,
            "round": round_index,
            **_text_fields(text),
            "length_ratio": round(ratio, 4),
            "length_guard_tripped": not (low <= ratio <= high),
            "generation": generation.as_dict(),
            "stop_reason": stop_reason,
        }
        writer.write(row)
        yield row

        texts[round_index] = text
        previous_text = text
        if stop_reason is not None:
            return


def recover_texts(rows: Sequence[dict[str, Any]]) -> dict[int, str]:
    """Rebuild ``round -> text`` for one passage from previously written rows."""
    return {int(r["round"]): r["text"] for r in rows}


def summarise_generation(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate wall-clock and token counts, for the cost calibration plan.md §9 asks for."""
    gens = [r["generation"] for r in rows if r.get("generation")]
    if not gens:
        return {"generations": 0}
    return {
        "generations": len(gens),
        "wall_seconds_total": round(sum(g["wall_seconds"] for g in gens), 1),
        "wall_seconds_mean": round(sum(g["wall_seconds"] for g in gens) / len(gens), 2),
        "prompt_tokens_total": sum(g["prompt_tokens"] for g in gens),
        "output_tokens_total": sum(g["output_tokens"] for g in gens),
        "truncated_rounds": sum(1 for g in gens if g["truncated"]),
    }


def as_generation(row: dict[str, Any]) -> Generation | None:  # pragma: no cover - helper
    """Reconstruct a :class:`Generation` from a row, for callers that want the typed form."""
    data = row.get("generation")
    if not data:
        return None
    return Generation(
        text=row["text"],
        prompt_tokens=data["prompt_tokens"],
        output_tokens=data["output_tokens"],
        done_reason=data["done_reason"],
        wall_seconds=data["wall_seconds"],
    )
