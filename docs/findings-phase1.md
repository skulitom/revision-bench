# Phase 1 findings

Status: **in progress.** M1-a (runner-level length control) is complete and its result is
negative. M1-b (corpus expansion) and M1-c (the three-family sweep) have not run.

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
