# revision-bench

**An open, reproducible study of revision-loop dynamics in LLMs.**

When you put an LLM in a loop and ask it to keep improving a piece of prose, the text does
not asymptotically approach perfection. This repo measures what it does instead — and
which loop architectures make revision non-degrading while still fixing real defects.

Status: **Phase 0 and Phase 1 complete; Phase 2 (judge validity) in progress; a
detect-then-repair harness built and measured.** Licence: Apache-2.0. See
[`plan.md`](plan.md) for the full research plan and [`AGENTS.md`](AGENTS.md) for the rules
this code is written under.

---

## The question

An unbounded revision loop with an LLM in the "is this better?" seat has no trustworthy
signal of *better*. LLM judges have self-preference bias and drift, so a loop can report
improvement every round while the text flattens. The predicted failure modes are
homogenization, voice loss, slop accumulation, thrash, and — the one that ties this to
[mirror-bench](https://github.com/skulitom/mirror-bench) — **report–state divergence**,
where the in-loop judge keeps saying "improved" while blinded external judgment does not
agree.

So the research question is not "how do we write a better improve-prompt". It is:

> Under what acceptance architecture is iterative revision provably non-degrading while
> retaining genuine fixes?

The disease is documented in the literature ([plan.md §3](plan.md)); the cure is not. This
repo occupies that gap with a controlled multi-round comparison of loop architectures,
with planted defects giving objective fix-recall, stylometric voice-preservation as a hard
constraint, and the judge's in-loop reports validated against blinded cross-family panels.

## What has been measured so far

Each links to the findings document that states the caveats; none of these numbers should be
quoted without them.

**The loop degrades text, and compression is the mechanism.** Unconstrained whole-passage
revision compresses roughly 50%, and most measured voice loss follows from the compression
rather than from style drift. ([phase 0](docs/findings-phase0.md))

**Prompt-level control fails; structural control works.** Asking for a length is
model-dependent and resampling does nothing — the model holds a target length across seeds.
Bounded diffs control length structurally because changing unnamed text becomes impossible
rather than discouraged. ([phase 1](docs/findings-phase1.md))

**The binding constraint is aim, not application.** The best bounded arm applies edits
reliably and still lands ~24 edits per defect repaired, ~96% of them on text that was not
broken. Voice drift also has to be read against a calibrated scale: the entire
between-author signal in this corpus is 0.194, and the safest arm perturbs 1.6× that.
([harness gap](docs/harness-gap.md))

**LLM judges are mostly measuring position.** 43–65% of the panel's raw verdicts flip when
the two options are swapped. On the verdicts that survive an order swap, all three judges
prefer the human original ~80% of the time — the project's first quality signal. A
surface-feature classifier identifies the model's edit 66% of the time, but the panel does
*not* follow it into its errors, which rules that explanation out. ([phase 2](docs/findings-phase2.md))

**Consistency defects need no judge at all.** On a synthetic genre stratum with exact ground
truth, mechanical detectors find 99–100% of planted cross-chapter contradictions at 88–90%
precision with zero false positives on clean text, and a complaint-gated repair arm restores
97% of them while editing **no chapter that had no complaint against it**.
([litrpg](docs/findings-litrpg.md))

## What this is, and is not

**Is:** measurement infrastructure. Controlled revision loops over a fixed prose corpus,
instrumented with stylometric, lexical and blinded-preference readouts, scored per round,
with seeds, confidence intervals and full provenance.

**Is not:** a writing product, a claim that AI editing is good or bad, or a
prompt-engineering demo. All language here stays at the level of *measured change per
revision round*. That discipline is load-bearing for credibility, and it is enforced in
review.

## Everything runs locally

No paid API is used anywhere in this project. Revisers and judge panels are local
open-weight models served by [Ollama](https://ollama.com) on a single consumer GPU.

That is partly a cost decision and partly a methodological one. [plan.md
§12](plan.md) lists "API model versions stay stable across a phase" as an unverified
assumption, because a mid-phase model bump invalidates cross-round comparison. Ollama pins
an exact model digest per tag, so a run's model identity is recorded exactly and that
assumption stops being load-bearing.

Cross-family judging (required by [plan.md §11](plan.md) — judges are never from the
reviser's family) is served by keeping several distinct families available locally.

## Repo layout

```
plan.md                  research plan; milestones and acceptance criteria
AGENTS.md                rules for anyone (human or agent) changing this code
NEXT*.md                 owner direction notes; absorbed, then retired to legacy/
legacy/                  retired notes, kept for provenance with an absorption header
configs/                 every constant lives here, never in code
data/
  corpus/passages/       extracted passages + per-passage provenance (committed)
  corpus/defects.jsonl   Stratum B planted defects with exact spans
  litrpg/                generated genre manuscripts + their ground-truth manifests
  slop_lexicon.yaml      versioned, per-entry citations
revisionbench/
  config.py              YAML loading, strict key checking, config hashing
  provenance.py          run stamps: git sha, package versions, model digests
  records.py             crash-safe JSONL artifacts + resume-by-key
  text.py                tokenisation, sentence splitting, punctuation classes
  corpus.py              Gutenberg fetch, licence records, passage extraction
  ollama.py              local inference; model digests are load-bearing
  loop.py                the revision runner: rounds, resume, stop reasons
  arms.py                loop architectures (whole / paragraph / edit-list / indexed)
  inject.py              Stratum B defect injection
  detect.py              mechanical detectors — never imports the injector
  judge.py               blinded pairwise panel, position bias, order consistency
  litrpg.py              synthetic genre world model; the manifest IS the ground truth
  litrpg_inject.py       cross-chapter contradictions
  litrpg_detect.py       consistency checks from the manuscript alone
  litrpg_repair.py       A2d: complaint-gated repair with a mechanical acceptance rule
  metrics/               stylometry, slop, thrash, defects, statistics
scripts/                 one command per phase
docs/                    findings, design space, harness gap, and external evidence
results/                 JSONL artifacts + figures, committed per phase
tests/                   offline, hermetic — no network, no GPU
```

A companion app, [RevisionJudge](../RevisionJudge), serves the blinded human-judging
subsample. It is a separate repo because it is a UI rather than measurement code.

## Quickstart

```bash
uv sync
```

```bash
uv run pytest -q
```

```bash
uv run ruff check && uv run ruff format --check
```

Fetch the Phase-0 corpus (the only step that touches the public internet):

```bash
uv run python scripts/fetch_corpus.py --config configs/corpus/phase0.yaml
```

Run the Phase-0 revision loop against a local model, then score and plot it. Only the
first command needs Ollama; the other two work from the artifact alone, which is what
plan.md §9 asks Phase 0 to demonstrate.

```bash
uv run python scripts/phase0.py --config configs/phase0.yaml
```

```bash
uv run python scripts/phase0_metrics.py && uv run python scripts/phase0_plots.py
```

### The detect-then-repair harness

This is the part closest to a usable tool, and the only pipeline here that needs no judge.
It runs on a synthetic genre corpus whose ground truth is exact by construction — the world
is generated as a state machine first and the prose written from it, so a contradiction is a
provable disagreement with a table rather than a matter of opinion.

Score the detectors. No model calls, no GPU, a few seconds:

```bash
uv run python scripts/litrpg_eval.py --manuscripts 20
```

Run the repair arm. Only complained-of spans are eligible to be edited, and a repair is kept
only if the manuscript's total complaint count strictly falls:

```bash
uv run python scripts/litrpg_repair_run.py --manuscripts 10 --model phi4:latest
```

Regenerate the corpus with model-written chapters instead of templated ones. Every chapter
is validated against the manifest before it is kept, because a model that invents state
produces contradictions the ground truth cannot adjudicate:

```bash
uv run python scripts/litrpg_generate.py --manuscripts 8 --chapters 14
```

Then pass `--corpus data/litrpg` to either script above to score against it.

### Judge validity

```bash
uv run python scripts/self_preference.py
```

```bash
uv run python scripts/surface_predictability.py --all-edits
```

```bash
uv run python scripts/panel_vs_surface.py
```

Findings live in [`docs/`](docs): [phase 0](docs/findings-phase0.md),
[phase 1](docs/findings-phase1.md), [phase 2](docs/findings-phase2.md),
[litrpg](docs/findings-litrpg.md), plus [the design space](docs/design-space.md) and
[what stands between this and a usable harness](docs/harness-gap.md).

## Related work in this portfolio

[SchemeStressProject](https://github.com/skulitom/SchemeStressProject) (checkable
compilation) → [mirror-bench](https://github.com/skulitom/mirror-bench) (checkable
introspection) → revision-bench (checkable revision).
