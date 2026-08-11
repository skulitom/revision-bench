# Phase 1 findings

Status: **in progress.** M1-a is complete: the runner-level length gate failed (§1), and the
architecture comparison that followed found the first mechanism that does control length
(§2). Its verdict is that the **next milestone should be Stratum B and defect-fix recall,
not the three-family sweep** — the most promising architecture is currently the least
evaluable, and more model families would multiply an unfalsifiable result.

Phase 0 is written up separately in [`findings-phase0.md`](findings-phase0.md).

---

## 1. Length cannot be controlled by re-rolling (M1-a)

Phase 0 ended with a clear instruction to Phase 1: handle length in the runner, because
prompt-level control is model-dependent (§6.7 — the length clause holds gemma3:4b at 0.97×
and phi4 at only 0.69×). The obvious runner-level mechanism is a **retry gate**: if a
round's word count leaves the guard band, generate it again with a fresh seed, up to a
budget, and keep the attempt closest to the original length.

It was implemented (`length_policy: retry` in `revisionbench/loop.py`), tested, and run
against phi4 on the passages that had actually tripped the guard in Phase 0.

**It does not work.**

| passage | round | attempt word counts | kept | ratio | in band? |
|---|---|---|---|---|---|
| woolf-03 | 1 | 553, 525, 528 | 553 | 0.61 | no |
| woolf-03 | 5 | 557, 559, 557 | 559 | 0.62 | no |
| woolf-04 | 2 | 617, 611, 609 | 617 | 0.65 | no |
| woolf-04 | 5 | 619, 619, 618 | 619 | 0.65 | no |
| hemingway-02 | 1 | 513, 583, 532 | 583 | 0.62 | no |

18 of 32 rounds were still out of band after three attempts, at **2.12× the generation
cost**. The reason is visible in the attempt columns: re-seeding moves the output length by
roughly ±10 words, and the gap to the band is ~300. Given an input, this model has a target
length and holds it across seeds. Best-of-N cannot select what the distribution does not
produce.

### Why this is worth more than the mechanism it kills

Three consequences, in increasing order of importance.

**The audit trail did its job.** `attempt_word_counts` was added so a best-of-N selection
would be inspectable rather than asserted. Without it this run would have reported "gated
arm, 18/32 rounds flagged" and the natural reading would have been "the gate helped a bit";
with it, the flatness of each triple shows the gate was never going to help at all. Keep
recording rejected attempts, not just accepted ones.

**A near-miss on a selection artifact.** The first validation of this gate ran on
`woolf-01` and `hemingway-01`, both came out in band with zero retries, and it looked like
the fix worked. Those were the two passages that had *never* tripped in Phase 0. Validating
an intervention on cases that do not exhibit the problem returns a clean pass and means
nothing — check which units actually show the defect before selecting a test set.

**Length is not a nuisance parameter — it is the finding.** The working assumption through
Phase 0 was that compression is a confound to be engineered out so that voice drift could
be measured cleanly. Two attempts have now failed: asking the model (works on one family,
not another) and re-rolling the model (works on neither). What survives is simpler and more
interesting: **asked to improve a passage of literary prose, these models rewrite it
shorter, and that is not a parameter they expose.** Compression is part of what
unconstrained whole-passage revision *is* on current small open-weight models.

That reframes plan.md §8's A2 arm. "Scoped: every proposal must cite a specific finding and
stay within a bounded diff size" was motivated as a way to stop global rewrites; this is
direct evidence that a bounded-diff architecture is the *only* remaining lever on length,
because neither the prompt nor the sampler is one. Paragraph- or sentence-scoped revision
structurally cannot halve a passage.

### What Phase 1 does instead

- **Report length as an outcome, not a covariate to be removed.** Every M1/M3 number is
  reported alongside the length ratio, as Phase 0 already does.
- **Control statistically rather than at generation time.** Compare voice drift across
  reviser families at matched length ratios, and treat any cross-family difference in drift
  that vanishes under length matching as a length effect rather than a voice effect.
- **Keep `length_policy: retry` in the code, off by default.** It is tested, it is cheap to
  re-enable, and a future reviser may have enough length variance for it to bite. Its
  Phase-1 setting is `observe`.
- **Promote scoped revision from Phase 3 to a Phase-1 exploratory arm** if budget allows,
  since it is now the only untested lever on the dominant effect.

---

## 2. Architecture comparison: three revision units, one model (M1-a continued)

With the prompt and the sampler both eliminated as levers, what remains is architectural.
Three strategies were implemented in `revisionbench/arms.py` and run head-to-head on phi4 —
the reviser where every non-architectural control has failed — over 5 passages × 5 rounds,
holding corpus, sampling and prompt wording fixed so the arms differ only in **what unit
gets revised and what gets applied**.

| arm | unit | length ratio | voice drift | punct drift | slop /1k | rounds run | gens |
|---|---|---|---|---|---|---|---|
| *(round 0)* | — | 1.00 | 0 | 0 | 0.54 | — | — |
| **A0** `whole` | whole passage | **0.53** | 0.99 | 0.67 | 1.69 | 5.0 | 5 |
| **A2p** `paragraph` | one paragraph | 0.83 | 0.85 | 0.67 | **1.93** | 5.0 | 36 |
| **A2e** `editlist` | a named span | **1.00** | **0.08** | **0.08** | **0.64** | **2.6** | 5 |

Per-passage length ratios (the mean hides less than usual here — see §2.3):

| passage | whole | paragraph | edit-list |
|---|---|---|---|
| hemingway-02 | 0.51 | 0.81 | 1.00 |
| richardson-02 | 0.56 | 0.87 | 1.00 |
| woolf-01 | 0.59 | 0.84 | 1.00 |
| woolf-03 | 0.51 | 0.77 | 1.00 |
| woolf-04 | 0.48 | 0.83 | 0.98 |

### 2.1 Bounded diffs work, and they work structurally

The edit-list arm holds length at 1.00 on four of five passages and 0.98 on the fifth, and
it does so **by construction rather than by persuasion**: the model names spans, only
unambiguous spans are replaced, and text it does not name cannot change. That is the first
mechanism in this project that has controlled length at all, after a prompt clause and a
resampling gate both failed.

Its preservation numbers are correspondingly dramatic — voice drift 0.08 against A0's 0.99,
and a slop rate of 0.64 against a round-0 baseline of 0.54.

### 2.2 …but the arm that preserves best is the arm that barely edits

**Do not read the table as "A2e wins".** Across 13 rounds it proposed 316 edits and applied
**80 — a 25% apply rate.** 236 were rejected, overwhelmingly because the `find` anchor did
not occur in the text: the model paraphrases the span it means instead of copying it. Three
of 13 rounds returned output that was not parseable JSON at all. And its trajectories ran
2.6 rounds on average against 5.0 for the other arms, because a round that applies nothing
leaves the text unchanged and halts the loop.

plan.md §6 anticipated exactly this: *"A gate that prevents degradation by blocking all
edits scores zero recall and is correctly rejected."* A2e's near-perfect preservation is
partly a real property of bounded diffs and partly the trivial consequence of changing very
little, and **the instruments in this repo cannot yet separate those two.** Stratum B does
not exist, so defect-fix recall is unmeasurable, so the recall-vs-preservation frontier that
is the whole point of the project has exactly one axis so far.

That makes the next milestone clear, and it is not the three-family sweep: **build Stratum B
and measure recall.** A2e is the most promising architecture found so far and the least
evaluable, and running it across more model families would multiply an unfalsifiable result.

### 2.3 Paragraph-scoping half-works, and adds slop

A2p moves length from 0.53 to 0.83 — a real improvement, and unlike A2e it keeps editing
(5.0 rounds, 36 generations). But voice drift barely moves (0.99 → 0.85), punctuation drift
does not move at all (0.67 → 0.67), and **the slop rate gets worse than the unconstrained
control** (1.69 → 1.93, against a 0.54 baseline).

The slop result is the interesting one and it has a plausible mechanism: revising each
paragraph in isolation gives the model N separate opportunities to reach for its stock
phrases, where a whole-passage rewrite gives it one. Bounding the diff bounds the *length*,
not the register.

### 2.4 A defect this arm shipped with, and how the mean hid it

The first implementation of A2p produced a cross-passage mean length ratio of **1.03**,
which reads as "solved the length problem". It had not. woolf-01 had inflated to **1.74**
while the others sat at 0.82–0.87, and the mean of a compression and an inflation is a
number that describes neither.

The cause was a **duplication cascade**. When the model returned one paragraph as two, the
next round saw two units and revised each separately; paragraph count went 7 → 10 → 12 → 16
→ 20 over five rounds and word count 901 → 1569, with visibly near-duplicate blocks
(`[185, 64, 195, 1, 186, 1, 183, 0, 197, …]`).

**A bounded-diff architecture whose unit boundaries move is not bounded.** Multi-paragraph
revisions are now rejoined into one, the k-th paragraph in maps to the k-th paragraph out,
and the invariant is asserted per round rather than assumed — a violation is recorded on the
row rather than averaged into a reassuring figure. The arm was re-run from scratch after the
fix; every number in §2 is post-fix.

### 2.5 Two instrumentation changes this comparison forced

- **`stop_reason` now distinguishes `no_valid_proposal` from `fixed_point`.** A2e rounds
  whose JSON failed to parse were being recorded as fixed points, i.e. as the model
  *choosing* to leave the text alone. That credits a protocol failure as voluntary halting,
  in the one direction that flatters bounded-diff arms — since an arm that changes nothing
  also scores as perfectly voice-preserving.
- **The run grouping key now includes the arm.** It already included the model after Phase
  0; three architectures under one model and one prompt name would have merged into single
  trajectories with duplicate round numbers and thrash windows spanning an architecture
  boundary. Stated generally so there is no third instance: *the grouping key must contain
  every field the run plan varies.*
