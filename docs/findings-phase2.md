# Phase 2 findings — judge validity

Status: **M2-a and M2-b complete.** The panel was put through a positive control before
being asked anything subtle (§1–§7); it disqualified half of it. M2-b then asked the panel
about real edits, and found that **most of its verdicts are positional artifacts** — but
that the verdicts which survive an order swap carry the project's first quality signal, and
it points the opposite way to "revision improves prose" (§8–§10).

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

~~**Provisional recommendation for the Phase-3 gate:** screen candidate judges on this
control, drop anything with a position-A rate outside 0.35–0.65 or accuracy below ~0.85,
and prefer two competent judges over four diverse ones.~~ **Retracted in §9:** position
bias measured on this control does not transfer to the real task — qwen3 scores 0.50 here
and 0.22 on actual edits. The competence half of the recommendation stands; the
impartiality half must be re-measured on the task of interest.

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
- **Self-preference is unmeasured** *at the time of M2-a*. §3 explains why phi4's score
  here is not it; §8 measures it properly and does not detect it.


---

# M2-b — self-preference, and what the panel says about real edits

`uv run python scripts/self_preference.py --both-orders`

## 8. Self-preference: not detected, by a design that could have detected it

Asking "does phi4 prefer phi4's edits more than the panel does" confounds bias with
competence — phi4 scored 0.97 on the §3 control against qwen3's 0.91, so it may simply
judge better. The separation comes from a **difference in differences**, which Phase 0's
artifacts happen to support: two revisers (phi4, gemma3:4b) ran the same passages under the
same prompts, so each is both an author and a judge, with a third judge holding no stake.

| judge \ edits from | phi4 | gemma3 | own − other |
|---|---|---|---|
| **phi4** | 0.35 *(self)* | 0.37 | −0.02 |
| **gemma3** | 0.32 | 0.50 *(self)* | +0.18 |
| qwen3 *(neutral)* | 0.33 | 0.32 | — |

Values are the rate at which the judge prefers the *edited* sentence. Subtracting the
neutral judge's view of the genuine quality gap between the two revisers:

| reviser | self-preference | 95% CI |
|---|---|---|
| phi4 | **−0.03** | [−0.27, +0.20] |
| gemma3:4b | **+0.20** | [−0.03, +0.43] |

**Neither is conclusive at 60 edits per cell.** phi4 shows nothing; gemma3 shows a
suggestive +0.20 whose interval still contains zero. The design absorbs a judge's general
leniency and, to first order, its position bias — but it cannot manufacture power, and the
honest statement is that this sample did not detect self-preference rather than that there
is none.

## 9. Position bias is task-dependent — which invalidates §5's recommendation

M2-a recommended screening judges on the objective control and keeping those with a
position-A rate inside 0.35–0.65. **That screen does not transfer.**

| judge | position-A on the objective control (§2) | position-A on real edits |
|---|---|---|
| qwen3:4b | 0.50 | **0.22** |
| gemma3:4b | 0.28 | 0.19 |
| phi4:latest | 0.44 | 0.31 |

qwen3 was the one judge that looked perfectly unbiased on the control and is heavily
positional here. Slot assignment was verified balanced (version 1 occupied slot A on
0.44–0.52 of pairs), so this is the judges, not the randomisation.

**Retract the §5 screening rule.** A judge must be characterised on the task it will
actually be used for; competence on a floor task predicts neither competence nor
impartiality on the real one.

## 10. The order-consistency filter, and the first quality signal in this project

Asking each pair twice with the order swapped, and keeping only verdicts that name the same
*text* both times:

| judge | order-consistent | retained | prefers the edit |
|---|---|---|---|
| phi4:latest | 68 / 120 | 0.57 | **0.19** |
| gemma3:4b | 43 / 120 | 0.36 | **0.21** |
| qwen3:4b | 42 / 120 | 0.35 | **0.19** |

Two things, and the second is the important one.

**43–65% of these judges' verdicts are positional artifacts.** They vanish when the order
changes. Any panel statistic computed without this filter — including everything in §8 — is
a mixture of opinion and seating position.

**On the verdicts that do survive, all three judges independently prefer the ORIGINAL
sentence about 80% of the time.** Blind, order-controlled, two of the three from different
families than either reviser.

That is the first evidence in this project that bears on *quality* rather than movement.
Phases 0 and 1 established that revision loops change prose in measurable ways and could
not say whether the change was an improvement. This says the panel thinks it is not.

### The contamination check, which the corpus was built for

The originals are famous published prose, and a judge may prefer them because it has read
them. plan.md §5's obscure-author control speaks to this directly:

| judge | prefers the edit — famous originals | — obscure (Richardson) |
|---|---|---|
| phi4 | 6/45 = 0.13 | 7/23 = 0.30 |
| gemma3 | 5/32 = 0.16 | 4/11 = 0.36 |
| qwen3 | 5/30 = 0.17 | 3/12 = 0.25 |

Every judge defends the famous originals about twice as hard as the obscure ones, which is
the direction memorisation predicts — **but the effect survives the control**: even on
Richardson, judges prefer the original 64–75% of the time. Read the famous/obscure gap
(~0.15) as an upper bound on how much familiarity contributes, and the remainder as a
preference for the writing.

The alternative reading is not excluded: Woolf and Hemingway may simply be harder to
improve than Richardson, which predicts the same pattern. The obscure cells hold 11–23
verdicts, so this check narrows the space rather than settling it.

## 11. The human subsample is not blind by default — a result about the corpus

The export in `scripts/export_human_pairs.py` is a *blinded* comparison only if the two
versions are indistinguishable except as prose. On this corpus they were not. Measured
across the 92 subjective pairs in the export:

| feature | decisive on | points to | precision |
| --- | ---: | --- | ---: |
| spaced en dash | 7 pairs | the model's edit | 100% |
| curly double quote | 8 pairs | the model's edit | 88% |
| curly single quote | 7 pairs | the model's edit | 86% |
| straight double quote | 7 pairs | the original | 86% |

("Decisive" = present on one side and absent on the other; a feature on both sides is
invisible to a judge, and one pointing both ways equally is noise.)

Together these give a rule covering roughly 17% of pairs at near-perfect precision. The
cause is mundane and was already on record: §5.1 of the Phase 0 findings noted that models
re-typeset Gutenberg's `--` as an em dash, which is why `text.py` folds dash runs before
computing punctuation profiles. The same re-typesetting that would have manufactured fake
voice drift in M2 turns out to de-blind a human A/B test.

**This does not affect the model panel.** Each judge call is independent with no memory
across pairs, so there is nothing to learn a rule *from*. It is specific to a person doing
104 pairs in a sitting, and it is the worst kind of failure for this dataset: a judge who
notices on pair 30 has contaminated every verdict after it, and nothing in the artifact
says which one that was.

RevisionJudge normalises punctuation to one convention on both versions and on the context,
at render time only — the pair file keeps the text as generated. That takes the exploitable
count to zero, with no feature decisive on ≥5 pairs above 75%. Length, worth checking
because Phase 0's headline finding is ~50% compression, carries no signal here: the edit is
shorter in 47% of pairs, because these are per-edit sentence pairs rather than whole-passage
revisions. **A future export at passage granularity would hand the judge "pick the longer
one" as a near-perfect rule**, and must re-run the check.

Standing caveat on everything that comes out of this: the panel judged un-normalised text
and the human judges normalised text, so panel-vs-human agreement is measured across
slightly different stimuli. Accepted deliberately — the alternative is an unrecoverable and
undetectable failure rather than a stated one — but it belongs beside the agreement number
whenever that number is reported.

## 12. A bag of counts identifies the model's edit 66% of the time

`scripts/surface_predictability.py`. §10's quality signal — all three judges prefer the
human original on ~80% of order-consistent verdicts — has two readings, and they imply
opposite things about the panel. Either the judges read the prose, or the model's edit
carries a surface regularity and the judges key on it. The second is testable with no human
input at all: fit a classifier that sees only counts (words, punctuation classes,
type-token ratio, subordinator rate — no semantics, no model) and ask how well it separates
edit from original.

| set | n | cross-validated accuracy | 95% CI |
|---|---:|---:|---|
| the 92 pairs the panel judged | 92 | 60.9% | [42.4%, 71.8%] |
| **the full edit pool** | **1385** | **66.1%** | **[60.9%, 68.0%]** |
| full pool, punctuation normalised | 1385 | 65.6% | [60.4%, 67.8%] |

Null is 50%. Folded by passage, so no passage trains and tests at once; symmetric pair
encoding so the fit has no intercept and chance really is 50%.

Two things worth separating. Normalising punctuation barely moves the number, so **this is
not the typographic tell from §11** — parenthesis, question-mark and quote rates carry it,
and those are structural choices rather than typesetting. And the 92-pair estimate spans
chance: **the human calibration set was underpowered for this question before anyone judged
anything**, which is worth knowing independently of whether a human ever judges it.

66% is not 80%, and "identify the edit" is not "prefer the original". But it is a floor
obtained from counts alone, so §10's 80% cannot be read as evidence of aesthetic judgement
until the surface explanation is ruled out. Which is the next section.

## 13. The panel does not follow surface features into error

`scripts/panel_vs_surface.py`. Agreement between the panel and the §12 classifier would not
settle anything, because both could be independently right — the edits really are flatter
*and* the originals really are better prose. The two readings come apart on the pairs where
the classifier is **wrong**: if the panel keys on surface features, it should follow the
classifier into its errors and its preference for the original should collapse.

Pooled over order-consistent verdicts from all three judges:

| | rate | 95% CI | n |
|---|---:|---|---:|
| where the surface classifier is **right** | 83% | [75%, 90%] | 92 |
| where the surface classifier is **wrong** | 77% | [66%, 87%] | 61 |
| gap | **+6%** | — | |

Per judge the gap is +12% (gemma3:4b), +4% (phi4), +2% (qwen3), and every interval overlaps
its partner heavily. On pairs where the surface evidence points at the original as if it
were the edit, the panel still prefers the original 77% of the time.

**So the surface explanation is ruled out, and §10's quality signal survives its first real
challenge.** State the ceiling precisely, because this is easy to overclaim:

- It rules out *this feature set*. A panel keying on a surface regularity these 23 features
  do not capture — a lexical preference, a syntactic template — would pass this test
  unchanged. The check is only as strong as the features it uses, and extending them is
  cheap.
- It does not show the judges read *well*, only that whatever they respond to is not a bag
  of counts.
- n=61 in the decisive cell. The gap could be anywhere from clearly zero to moderately
  positive.

What makes it worth having anyway: it is the first check in this project to *strengthen* a
claim rather than qualify one, it cost no GPU-hours and no human labels, and it is the
template for the rest — take the alternative explanation seriously enough to construct the
subset where it makes a different prediction, then look there.

## 14. Where this leaves Phase 2

- **A quality claim now exists**, with three caveats attached: small n, local 4B–14B
  judges, and no human calibration.
- **Blinding is a property to be tested, not assumed.** §11 found the pairs de-blindable by
  punctuation alone. The check — sweep every surface feature, flag any that is decisive on
  5+ pairs at 85%+ precision — is cheap and now automated; it should run on every export.
- **The quality signal survived its first challenge without a human** (§12–§13). The
  surface-feature explanation for §10's 80% is ruled out for the features tested. This does
  not remove the need for calibration, but it does mean the panel is responding to something
  a count-based model cannot see, which was not established before.
- **Adversarial subsetting is the cheapest tool here.** Both §12 and §13 cost zero GPU-hours
  and zero labels. The pattern — name the alternative explanation, find the subset where it
  predicts something different, look there — should be tried before any experiment that
  needs new generations.
- **The human subsample is now the critical path**, not a formality. plan.md §11 says the
  panel is trusted only where it agrees with a human, and §9–§10 have just shown the panel
  needs that anchor more than assumed: most of its raw verdicts are positional, and the
  filter that fixes this costs two-thirds of its coverage.
- **Order-consistency filtering should be standard** for any judge used as a gate here, and
  its retention rate reported — the same discipline as recall-with-removal-rate and
  panel-accuracy-with-coverage. A gate that discards 65% of its own opinions is viable; one
  that hides that it should have is not.
