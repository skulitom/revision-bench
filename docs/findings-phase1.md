# Phase 1 findings

Status: **M1-a and M1-b complete.** The runner-level length gate failed (§1); the
architecture comparison that followed found the first mechanism that does control length
(§2); and with Stratum B built, §3 reports the recall-vs-preservation frontier — the
project's central deliverable, and the first result here that can falsify an architecture
rather than describe one.

**The headline is in §3.4, and it is not in the table.** The bounded arms now apply edits
reliably (74% apply rate, zero protocol failures) but apply roughly **24 edits per planted
defect repaired**. The bottleneck has moved from *how much* a loop changes to *what it
chooses to change*. Next, per `NEXT.md`: Phase 2 judge validity at per-edit granularity.

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

### 2.5 Two instrumentation changes this comparison forced (see also §3)

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

---

## 3. The recall-vs-preservation frontier (M1-b complete)

The project's central deliverable, and the first result here that can falsify an
architecture rather than describe one. Four arms × 10 Stratum-B passages × 5 rounds,
phi4, 148 generations, 35 minutes. Figure:
[`results/phase1/frontier.png`](../results/phase1/frontier.png).

| arm | length | voice drift | slop /1k | **recall** | **removed** | defects fixed | edits applied |
|---|---|---|---|---|---|---|---|
| *unrevised (floor)* | 1.05 | 0.12 | 0.52 | 0.00 | 0.00 | 0 | 0 |
| **A0** whole passage | 0.59 | 0.96 | 2.31 | **0.73** | **0.45** | 16 | 48 |
| **A2p** paragraph | 0.93 | 0.84 | 3.29 | 0.59 | 0.33 | 16 | 719 |
| **A2e** edit-list | 1.02 | 0.13 | 0.86 | 0.15 | 0.00 | 6 | 208 |
| **A2i** indexed + vetoes | 1.03 | 0.31 | **0.43** | 0.22 | 0.07 | 8 | 195 |

Recall is over *surviving* defects; `removed` is the fraction whose region was cut rather
than repaired, and is censored rather than counted as a fix.

### 3.1 The trade-off is real, and no arm escapes it

Nothing dominates. The two arms with high recall (A0 0.73, A2p 0.59) sit at voice drift
0.96 and 0.84 — near-total loss of the round-0 fingerprint. The two bounded arms sit at
drift 0.13–0.31, close to the unrevised floor of 0.12, and fix 0.15–0.22 of surviving
defects. **This is the frontier plan.md §6 predicted, measured for the first time.**

It is also steeper than hoped. There is no arm here that both preserves voice and fixes
most defects; the best bounded arm repairs about a fifth of what survives.

### 3.2 The high-recall arms are partly deleting, not repairing

A0 and A2p each "fixed" 16 defects — and each also lost 45% and 33% of planted defects to
deletion. Their recall is computed over a denominator that their own destructiveness
shrank. A naive metric would have scored A0 at 32/40 and called it the winner; the removal
column is the whole reason it does not.

A0 gets its recall by rewriting 41% of the passage away. That is not a defect-fixing
strategy, it is a defect-*removal* strategy, and it takes the surrounding prose with it.

### 3.3 A2i strictly dominates A2e, and the slop veto works

A2i beats A2e on recall (0.22 vs 0.15) at the same length, with **lower slop than the
unrevised input** (0.43 vs the floor's 0.52). That last number is worth pausing on: the
per-edit no-new-slop veto did not merely hold the line, it pulled the slop rate *below the
text it was given*, because edits that would have introduced a lexicon term were rejected
while edits that removed one were not. A free mechanical veto, doing exactly what it was
specified to do.

A2e's drift of 0.13 is barely above the floor — but §2.2's warning applies: that is largely
preservation by inaction, and its recall of 0.15 is the price.

### 3.4 The finding that is not in the table: the edits are not aimed at the defects

A2i applied **195 edits and fixed 8 planted defects**. A2e applied 208 and fixed 6. That is
roughly **24 applied edits per defect repaired**, and it is the most actionable number in
this section.

The bounded arms are not failing to *apply* edits any more — A2i's apply rate is 74% and
its protocol failures are zero. They are failing to apply edits **to the right places**.
Nearly everything they change is cosmetic: a word swapped, a clause reordered, in sentences
that had nothing wrong with them. Meanwhile a planted name drift three sentences away goes
untouched.

So the bottleneck has moved. Phase 0 and M1-a were about *how much* a loop changes; this is
about *what it chooses to change*. Two consequences:

- **plan.md §7 M5's overreach precision is now the interesting metric**, not recall.
  "Fraction of applied edits that touch a defective span" is measurable today from the
  artifacts already written, and 4% is a striking starting number.
- **The "diagnose, then fix" arm from `design-space.md` §7 is now the highest-value
  untested architecture**, because it separates the two failure modes the current arms
  conflate: not finding the defect, and finding it but editing elsewhere.

### 3.5 A2p is the worst of both worlds

719 applied edits — 3.7× the next busiest arm — for drift 0.84, the **highest slop of any
arm at 3.29** (6.3× the floor), and still 33% removal. Revising paragraph by paragraph
gives the model the most opportunities to reach for stock phrasing and buys the least. On
this evidence it should not be carried into Phase 2.

### 3.6 What this does not establish

- **One model.** phi4 only. Length compliance and halting have both already proven
  model-dependent, so no threshold from this table should travel until it is replicated on
  a larger model and a second family.
- **One prompt, five rounds, ten passages, 40 defects.** Cell counts are small; A2e and A2i
  fixed 6 and 8 defects respectively, so the recall gap between them rests on two events.
- **No quality judgment anywhere.** Every column is mechanical movement. Whether A2i's
  surviving prose actually *reads* better than A0's is exactly the question Phase 2 exists
  to answer, and nothing here bears on it.
