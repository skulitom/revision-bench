# revision-bench

**An open, reproducible study of revision-loop dynamics in LLMs.**

When you put an LLM in a loop and ask it to keep improving a piece of prose, the text does
not asymptotically approach perfection. This repo measures what it does instead — and
which loop architectures make revision non-degrading while still fixing real defects.

Status: **planning / Phase 0 in progress.** Licence: Apache-2.0. See
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
configs/                 every constant lives here, never in code
data/
  corpus/passages/       extracted passages + per-passage provenance (committed)
  slop_lexicon.yaml      versioned, per-entry citations
revisionbench/
  config.py              YAML loading, strict key checking, config hashing
  provenance.py          run stamps: git sha, package versions, model digests
  records.py             crash-safe JSONL artifacts + resume-by-key
  text.py                tokenisation, sentence splitting, punctuation classes
  corpus.py              Gutenberg fetch, licence records, passage extraction
  metrics/               stylometry, slop, thrash, statistics
scripts/                 one command per phase
results/                 JSONL artifacts + figures, committed per phase
tests/                   offline, hermetic
```

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

Findings so far live in [`docs/findings-phase0.md`](docs/findings-phase0.md).

## Related work in this portfolio

[SchemeStressProject](https://github.com/skulitom/SchemeStressProject) (checkable
compilation) → [mirror-bench](https://github.com/skulitom/mirror-bench) (checkable
introspection) → revision-bench (checkable revision).
