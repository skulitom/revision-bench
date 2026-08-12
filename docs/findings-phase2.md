# Phase 2 findings — judge validity

Status: **M2-a complete.** The panel has been put through a positive control before being
asked anything subtle. The result disqualifies half of it.

Phases 0 and 1: [`findings-phase0.md`](findings-phase0.md), [`findings-phase1.md`](findings-phase1.md).

Reproduce: `uv run python scripts/judge_sanity.py`

---

## 1. The positive control, and why it comes first

Everything measured in Phases 0 and 1 is *movement* — length, drift, slop, recall. Whether
any of it is an improvement is Phase 2's question, and the instrument for answering it is a
judge. So the judge is the thing that has to be validated first.

Stratum B supplies 32 sentence pairs with an **objectively correct answer**: one version is
a corpus sentence, the other is that sentence with a planted defect in it. Choosing the
clean one is the floor of competence.

The control is asymmetric on purpose, and this is the sentence to keep:

> **It can only disqualify, never qualify.** A judge that cannot spot a misspelled name is
> useless for aesthetics. A judge that *can* has not thereby been shown to have taste.

Order is randomised per (pair, judge) and recorded, so position bias is measured rather
than assumed away.

## 2. Position bias, reported before accuracy

| judge | family | position-A rate |
|---|---|---|
| gemma3:4b | Google | **0.28** |
| llama3.2:latest | Meta | **0.22** |
| qwen3:4b | Alibaba | 0.50 |
| deepseek-r1:8b | DeepSeek | 0.62 |
| phi4:latest | Microsoft | 0.44 |

0.50 is unbiased. **Two of the four panel members answer substantially with position** —
llama3.2 picks the second option 78% of the time regardless of what is in it. Any agreement
those two show with anything else is contaminated, and reporting their accuracy without
this column would hide it.

## 3. Accuracy on a question that has a right answer

| judge | n | correct | accuracy | by defect type |
|---|---|---|---|---|
| phi4:latest | 32 | 31 | **0.97** | clunker 8/9, name 3/3, number 10/10, tense 10/10 |
| qwen3:4b | 32 | 29 | **0.91** | clunker 9/9, name 3/3, number 9/10, tense 8/10 |
| gemma3:4b | 32 | 25 | 0.78 | clunker 8/9, name 2/3, number 8/10, tense 7/10 |
| llama3.2:latest | 32 | 21 | 0.66 | — |
| deepseek-r1:8b | 32 | 21 | 0.66 | — |

Chance is 0.50. Two judges sit within 16 points of chance on a task with an unambiguous
answer.

**phi4's 0.97 is not self-preference.** It is the reviser, but here it is judging corpus
sentences against planted corruptions — none of this text is its own output. This number
measures capability, not bias. Self-preference requires phi4 judging phi4's *edits*, which
is a separate experiment and has not been run.

## 4. Inter-judge agreement is close to chance

| pair | agreement |
|---|---|
| deepseek-r1 vs gemma3 | 0.69 |
| gemma3 vs qwen3 | 0.69 |
| gemma3 vs llama3.2 | 0.66 |
| llama3.2 vs qwen3 | 0.66 |
| deepseek-r1 vs qwen3 | 0.56 |
| deepseek-r1 vs llama3.2 | 0.47 |

Two binary judges agree 50% of the time by chance. **deepseek-r1 and llama3.2 agree 47% —
worse than coin-flipping** — on questions with a correct answer.

## 5. What a gate would actually use

Majority vote, with coverage, because a panel converts disagreement into *abstention*:

| signal | accuracy | coverage |
|---|---|---|
| full 4-judge panel | 0.92 | 0.78 |
| panel minus the two position-biased | **1.00** | **0.56** |
| phi4 alone | 0.97 | 1.00 |
| qwen3 alone | 0.91 | 1.00 |
| gemma3 alone | 0.78 | 1.00 |
| deepseek-r1 alone | 0.66 | 1.00 |
| llama3.2 alone | 0.66 | 1.00 |

The two-judge panel is perfect where it speaks and declines to answer 44% of the time.
Quoting its 1.00 without its coverage would sell a gate that mostly says nothing — the same
shape as quoting defect recall without the removal rate.

### The finding

**Cross-family panel diversity bought nothing here; judge competence did.** A single
competent judge (qwen3, 0.91 at full coverage) matches the full four-family panel (0.92 at
0.78 coverage) on cost, coverage and simplicity, and the two weak members contribute ties
rather than accuracy.

plan.md §12.3 lists as unverified: *"Cross-family panels are meaningfully less biased than
self-judges. (Phase 2 measures this; do not assume.)"* On this evidence, at this scale, the
axis that matters is not family diversity but per-judge competence — and competence is
cheaply measurable with exactly this control.

**Provisional recommendation for the Phase-3 gate:** screen candidate judges on this
control, drop anything with a position-A rate outside 0.35–0.65 or accuracy below ~0.85,
and prefer two competent judges over four diverse ones. Do not fix a threshold on 32 pairs.

## 6. A reproducibility bug worth recording

The first two runs of this script disagreed by two verdicts, at temperature 0 with a pinned
seed. Cause: five judges total ~22 GB against a 24 GB card, so Ollama evicts and reloads
between blocks, and the warm-up was done for all judges *up front* — stale by the fifth
block, leaving later judges answering from a cold load. The first generation after a load
is the irreproducible one (`findings-phase0.md` §5.1).

Warming each judge immediately before its own block fixed it; two consecutive runs are now
identical. **The lesson generalises: a warm-up is only valid until the next eviction, and
adding models to a sweep can silently invalidate one done earlier.**

## 7. What this does not establish

- **32 pairs.** Every number here has a wide interval and none should set a threshold.
- **This is the floor task, not the real one.** Ranking judges on "spot the planted defect"
  need not predict ranking on aesthetic comparison, and the whole point of §1 is that
  clearing this bar proves only the absence of a disqualifying flaw.
- **No human anchor yet.** plan.md §11 requires the panel to be calibrated against a
  blinded human subsample before it is trusted; that is the next step and it needs the
  project owner, not a model.
- **Self-preference is unmeasured.** §3 explains why phi4's score here is not it.
