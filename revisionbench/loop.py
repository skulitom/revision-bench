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
from dataclasses import dataclass, replace
from typing import Any

from revisionbench.arms import ReviseContext, Strategy
from revisionbench.ollama import Generation, GenerationOptions, ModelIdentity, OllamaClient
from revisionbench.provenance import sha256_text
from revisionbench.records import JsonlWriter
from revisionbench.text import word_count, words

__all__ = [
    "ARM_A0",
    "LENGTH_POLICIES",
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


#: Length-guard policies. **These are different experimental arms, not settings.**
#:
#: ``observe`` flags an out-of-band round and does nothing else. That is arm A0 as
#: plan.md §8 defines it: unconstrained, always accept.
#:
#: ``retry`` re-generates an out-of-band round with a fresh seed and keeps the attempt
#: closest to the original length. That is *already an acceptance architecture* — a
#: one-dimensional gate — so a run using it is emphatically **not** A0 and must not be
#: labelled as such. It exists because Phase 0 showed prompt-level length control is
#: model-dependent (docs/findings-phase0.md §6.7: the length clause holds gemma3:4b at
#: 0.97x and phi4 at only 0.69x), so a cross-model comparison needs length controlled by
#: the runner rather than by asking politely.
#:
#: Keeping both is the point. ``observe`` remains the honest control; ``retry`` is the
#: condition in which H1-H3 can be tested without compression swamping every metric.
LENGTH_POLICIES = ("observe", "retry")


@dataclass(frozen=True, slots=True)
class LoopSpec:
    """How a trajectory runs, when it retries, and when it stops early."""

    arm: str
    rounds: int
    #: Word count outside ``round0 * (low, high)`` flags the round, and — under the
    #: ``retry`` policy — triggers re-generation.
    length_guard: tuple[float, float]
    #: Below this many words the trajectory stops. Revising a stub for seven more rounds
    #: burns GPU time and yields nothing but MetricErrors; the collapse is itself the
    #: result, and it is recorded as one.
    min_words_to_continue: int
    length_policy: str = "observe"
    #: Total generations allowed per round under ``retry`` (1 = no retry). Bounded because
    #: a model that simply will not produce in-band text would otherwise spin forever; when
    #: the budget runs out the round is accepted anyway and flagged.
    max_attempts: int = 1

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
        if self.length_policy not in LENGTH_POLICIES:
            raise ValueError(
                f"length_policy must be one of {LENGTH_POLICIES}, got {self.length_policy!r}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.length_policy == "observe" and self.max_attempts != 1:
            raise ValueError(
                "length_policy 'observe' cannot retry; max_attempts must be 1. Set "
                "length_policy: retry if you meant to gate on length — and rename the arm, "
                "because a length gate is an acceptance architecture and no longer A0."
            )
        if self.length_policy == "retry" and self.arm == ARM_A0:
            raise ValueError(
                f"arm {ARM_A0!r} is the unconstrained control and cannot use the retry "
                f"policy; a retried run is a gated run and needs its own arm label, or the "
                f"artifact will claim a control it does not contain"
            )


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
    strategy: Strategy | None = None,
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

        if strategy is None:
            rendered = prompt.render(previous_text)
            generation, text, attempts = _generate_within_guard(
                client, model, rendered, options, spec, round_zero_words
            )
            proposal_stats = None
        else:
            context = ReviseContext(
                client=client, model=model, options=options, template=prompt.template
            )
            proposal = strategy.revise(context, previous_text)
            text = proposal.text.strip()
            proposal_stats = proposal.as_dict()
            attempts = [word_count(text)]
            # A strategy may make many calls; the row's `generation` block summarises them
            # so downstream code keeps one shape. The per-call detail is in `proposal`.
            generation = _merge_generations(proposal.generations, text)
        current_words = word_count(text)
        ratio = current_words / round_zero_words if round_zero_words else 0.0

        stop_reason = None
        unchanged = words(text) == words(previous_text)
        failed_to_propose = (
            proposal_stats is not None
            and proposal_stats["applied"] == 0
            and bool(proposal_stats["problems"])
        )
        # Both of the next two branches stop the trajectory, and both are absorbing for the
        # same reason: under a reproducible sampler the next round's input would equal this
        # one's, so its output would too. They are separated because they mean opposite
        # things about the arm. `fixed_point` says the model looked at the text and chose to
        # leave it alone — convergence. `no_valid_proposal` says the model tried and its
        # output could not be used: unparseable JSON, or every edit rejected. Reporting the
        # second as the first would credit a protocol failure as voluntary halting, which is
        # exactly the direction that flatters a bounded-diff arm, since an arm that changes
        # nothing also scores as perfectly voice-preserving.
        if failed_to_propose and unchanged:
            stop_reason = "no_valid_proposal"
        elif options.is_deterministic and unchanged:
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
            # Every attempt's word count, so a best-of-N selection is auditable rather
            # than merely asserted. Length 1 under the `observe` policy.
            "attempt_word_counts": attempts,
            "attempts": len(attempts),
            "strategy": getattr(strategy, "name", "whole"),
            "proposal": proposal_stats,
            "stop_reason": stop_reason,
        }
        writer.write(row)
        yield row

        texts[round_index] = text
        previous_text = text
        if stop_reason is not None:
            return


def _merge_generations(generations: Sequence[Generation], text: str) -> Generation:
    """Collapse a strategy's model calls into one summary row entry.

    Sums tokens and wall clock, and reports ``done_reason`` as ``"length"`` if *any* call
    was truncated — the pessimistic direction on purpose, because one cut paragraph is
    enough to make a reassembled passage suspect, and a summary that hid it would let the
    round be scored as clean.
    """
    if not generations:
        return Generation(
            text=text, prompt_tokens=0, output_tokens=0, done_reason="none", wall_seconds=0.0
        )
    return Generation(
        text=text,
        prompt_tokens=sum(g.prompt_tokens for g in generations),
        output_tokens=sum(g.output_tokens for g in generations),
        done_reason="length" if any(g.truncated for g in generations) else "stop",
        wall_seconds=round(sum(g.wall_seconds for g in generations), 2),
    )


def _generate_within_guard(
    client: OllamaClient,
    model: ModelIdentity,
    rendered: str,
    options: GenerationOptions,
    spec: LoopSpec,
    round_zero_words: int,
) -> tuple[Generation, str, list[int]]:
    """Generate one round, retrying out-of-band lengths when the policy asks for it.

    Returns ``(generation, text, attempt_word_counts)``.

    Under ``retry``, each attempt re-seeds deterministically (``seed + attempt``) so the
    round stays reproducible: the same config re-run makes the same attempts in the same
    order. If no attempt lands in band, the one **closest to the round-0 length** is kept.

    That last rule is a best-of-N selection and is therefore a mild optimisation pressure
    on length — recorded rather than hidden, via ``attempt_word_counts`` on every row. It
    is preferred to keeping the last attempt because the alternative is worse in a specific
    way: "keep the last" would make the accepted text depend on how many attempts the
    budget happened to allow, which is a knob nobody would think to report.
    """
    low, high = spec.length_guard
    attempts: list[int] = []
    best: tuple[float, Generation, str] | None = None

    for attempt in range(spec.max_attempts):
        step_options = options if attempt == 0 else replace(options, seed=options.seed + attempt)
        generation = client.generate(model.tag, rendered, step_options)
        text = generation.text.strip()
        words_here = word_count(text)
        attempts.append(words_here)
        ratio = words_here / round_zero_words if round_zero_words else 0.0

        if low <= ratio <= high:
            return generation, text, attempts

        distance = abs(ratio - 1.0)
        if best is None or distance < best[0]:
            best = (distance, generation, text)

    assert best is not None
    return best[1], best[2], attempts


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
