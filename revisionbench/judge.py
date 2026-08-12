"""Blinded pairwise judging (plan.md §7 M4, §11 judge hygiene) — Phase 2.

Everything measured before this point is *movement*: length, drift, slop, recall. None of
it says whether prose got better, and the project's original question — whether an in-loop
judge's "improved" tracks anything real (H4) — lives here.

Design, and each point is a rule from plan.md §11 rather than a preference:

**Pairwise, not absolute.** A judge asked to score a passage out of ten produces a number
with no anchor. Asked which of two versions is better, it produces a comparison, and
comparisons compose into a ranking.

**Position randomised on every call, and the position recorded.** Which version appears
first is decided by a hash of (pair, judge, seed) — deterministic, so a run reproduces, and
independent of which version is the "real" one. The recorded position is what makes
:func:`position_bias` computable, and that measurement is not optional: a judge that picks
the first option 80% of the time is reporting position, not quality, and its agreement with
anything else is coincidence.

**Judges are never from the reviser's family.** phi4 (Microsoft) produced the edits, so the
panel is drawn from Google, Meta, Alibaba and DeepSeek. Self-judging is run as a separate,
explicitly labelled condition to measure self-preference, never mixed into the panel.

**Reasoning models must be told not to reason.** qwen3 and deepseek-r1 put their chain of
thought in a separate field; left on, it exhausts the token budget and the answer comes
back as an empty string — which for a judge reads as an abstention rather than a failure.
:class:`OllamaClient` raises on that specific shape, and this module passes ``think=False``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from revisionbench.ollama import GenerationOptions, ModelIdentity, OllamaClient, OllamaError

__all__ = [
    "CHOICE_SCHEMA",
    "DEFAULT_PANEL",
    "JudgeVerdict",
    "Pair",
    "agreement_matrix",
    "judge_pair",
    "position_bias",
]

#: Constrained decoding: the judge cannot emit anything but one of the two labels. Without
#: it, a judge that answers "Both are good in their own way" has to be parsed, and every
#: parser here would be inventing a verdict the model did not give.
CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
    "required": ["choice"],
}

#: Cross-family panel. The reviser under study is phi4 (Microsoft); none of these share a
#: lineage with it or with each other.
DEFAULT_PANEL = ("gemma3:4b", "llama3.2:latest", "qwen3:4b", "deepseek-r1:8b")

PROMPT = """You are comparing two versions of one sentence from a work of literary fiction.

Context (the surrounding passage):
{context}

Version A: {a}

Version B: {b}

Which version is better as prose in this context? Answer with A or B."""


@dataclass(frozen=True, slots=True)
class Pair:
    """Two versions of a sentence, with whatever ground truth is known about them.

    ``better`` is the objectively correct answer where one exists — for a planted defect,
    the clean version — and ``None`` where the comparison is a matter of taste. Keeping the
    two tiers in one type is deliberate: the subjective pairs are the ones worth spending a
    human subsample on, and they are only identifiable if the objective ones are labelled.
    """

    pair_id: str
    context: str
    version_1: str
    version_2: str
    #: "version_1", "version_2", or None when there is no ground truth.
    better: str | None = None
    kind: str = ""
    meta: dict[str, Any] = None  # type: ignore[assignment]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One judge's answer, plus the bookkeeping needed to distrust it."""

    pair_id: str
    judge: str
    judge_digest: str
    #: Which version the judge chose: "version_1" or "version_2".
    chose: str
    #: Which *slot* that was — "A" or "B". The raw answer, before de-randomisation.
    chose_slot: str
    #: Which version was shown in slot A. Recorded so position bias is computable.
    slot_a_was: str
    correct: bool | None
    wall_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slot_a_holds(pair_id: str, judge: str, seed: int) -> str:
    """Deterministically decide which version occupies slot A.

    Hashed rather than randomised from a shared RNG so that adding a judge or reordering
    the pair list cannot change the layout of any other call — a resumed or partial run
    stays comparable with a complete one.
    """
    key = f"{pair_id}|{judge}|{seed}".encode()
    return "version_1" if hashlib.sha256(key).digest()[0] % 2 == 0 else "version_2"


def judge_pair(
    client: OllamaClient,
    model: ModelIdentity,
    pair: Pair,
    options: GenerationOptions,
    *,
    seed: int = 0,
    context_chars: int = 700,
) -> JudgeVerdict:
    """Ask one judge which version is better, with position randomised.

    Raises:
        OllamaError: The judge failed or returned something unusable. Never silently
            defaulted to a choice — an invented verdict is worse than a missing one, since
            it enters the agreement statistics as if it were data.
    """
    slot_a = _slot_a_holds(pair.pair_id, model.tag, seed)
    a_text = pair.version_1 if slot_a == "version_1" else pair.version_2
    b_text = pair.version_2 if slot_a == "version_1" else pair.version_1

    prompt = PROMPT.format(
        context=pair.context[:context_chars].strip(),
        a=a_text.strip(),
        b=b_text.strip(),
    )
    generation = client.generate(model.tag, prompt, options, schema=CHOICE_SCHEMA, think=False)
    try:
        choice = json.loads(generation.text)["choice"].strip().upper()
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        raise OllamaError(
            f"{model.tag} gave an unparseable verdict on {pair.pair_id}: "
            f"{generation.text[:120]!r} ({exc})"
        ) from exc
    if choice not in ("A", "B"):
        raise OllamaError(f"{model.tag} answered {choice!r}, expected A or B")

    chose = slot_a if choice == "A" else ("version_2" if slot_a == "version_1" else "version_1")
    return JudgeVerdict(
        pair_id=pair.pair_id,
        judge=model.tag,
        judge_digest=model.digest,
        chose=chose,
        chose_slot=choice,
        slot_a_was=slot_a,
        correct=None if pair.better is None else (chose == pair.better),
        wall_seconds=generation.wall_seconds,
    )


def position_bias(verdicts: Sequence[JudgeVerdict]) -> float | None:
    """Fraction of verdicts that picked slot A, regardless of content.

    0.5 is unbiased. Anything far from it means the judge is answering with its attention
    rather than its opinion, and every other statistic about that judge — accuracy,
    agreement, self-preference — is contaminated by it. Report this *first*.

    ``None`` for an empty sequence: a bias of 0.5 on no data would look like a clean bill
    of health.
    """
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v.chose_slot == "A") / len(verdicts)


def agreement_matrix(verdicts: Sequence[JudgeVerdict]) -> dict[tuple[str, str], float]:
    """Pairwise agreement between judges, over the pairs they both answered.

    Only shared pairs count. Comparing a judge's verdicts on one set against another's on a
    different set produces a number that describes the sampling, not the judges.
    """
    by_judge: dict[str, dict[str, str]] = {}
    for verdict in verdicts:
        by_judge.setdefault(verdict.judge, {})[verdict.pair_id] = verdict.chose

    out: dict[tuple[str, str], float] = {}
    judges = sorted(by_judge)
    for i, first in enumerate(judges):
        for second in judges[i + 1 :]:
            shared = set(by_judge[first]) & set(by_judge[second])
            if not shared:
                continue
            same = sum(1 for p in shared if by_judge[first][p] == by_judge[second][p])
            out[(first, second)] = same / len(shared)
    return out
