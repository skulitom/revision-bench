# The design space for a non-degrading revision loop

Working notes, written after the Phase-1 arm comparison. The purpose is to lay out the
axes an architecture can vary along, so that arms are chosen by argument rather than by
whichever one was easy to build next.

Evidence base: [`findings-phase0.md`](findings-phase0.md), [`findings-phase1.md`](findings-phase1.md).

---

## What is actually established

1. Asked to "improve" literary prose with no further constraint, both tested revisers
   rewrite it ~40–50% shorter, and that compression drives most measured voice loss.
2. Prompt-level length instruction is model-dependent (holds gemma3:4b, fails phi4).
3. Resampling is not a lever: given an input the model has a target length and holds it
   across seeds to within ±10 words.
4. Bounded diffs control length **structurally** — text the model does not name cannot
   change.
5. The bounded-diff arm as built applied only 25% of its proposals, and **223 of 236
   rejections (94.5%) were a single failure: the model paraphrased the anchor instead of
   copying it.** Zero were ambiguity failures.

Point 5 is the important one. It is a failure of the *interface*, not of the architecture.

---

## Seven axes

### 1. Addressing — how does the model point at what to change?

| form | burden on the model | observed |
|---|---|---|
| rewrite everything | none | A0: 0.53× length |
| rewrite a unit | none (position is implicit) | A2p: 0.83×, slop worse |
| **quote the span verbatim** | exact copying | A2e: 94.5% of failures |
| **index the span** (`sentence 7`) | none — it is a number | untested |

Verbatim quotation asks a model to do the one thing transformers are known to be
unreliable at, for no benefit: the span is already on screen and already addressable.
Symbolic addressing should eliminate this failure class by construction rather than
reduce it. **Highest-value untested change in the whole space.**

### 2. Grammar — is the output format enforced or requested?

Prompt-requested JSON failed to parse in 3 of 13 rounds here. Ollama supports
schema-constrained decoding via the `format` parameter (verified on this box, gemma3:4b);
under it, malformed output is not merely unlikely but unrepresentable. Free, and removes an
entire error mode. There is no argument for keeping prompt-requested JSON.

### 3. Granularity of acceptance — per round, or per edit?

plan.md §8's A3/A4 gate a whole proposal. With an edit list the natural unit is the
individual edit: one bad edit need not force rejecting the good ones in the same round.
Per-edit acceptance is strictly more informative (it yields an accept rate, not a binary),
strictly cheaper to satisfy, and it is the granularity the downstream harness needs.

### 4. What the gate consults

| gate | cost | notes |
|---|---|---|
| nothing | 0 | A0 |
| random at matched rate | 0 | A1 — the control that separates "editing" from "editing badly" |
| **mechanical vetoes** | **0** | see below |
| self-judge | 1 call | known self-preference bias (plan.md §3) |
| cross-family panel | ≥3 calls | A3; the expensive one |

The mechanical vetoes deserve their own line because this project already has the
instruments and they cost nothing:

- **no-new-slop** — reject any edit that introduces a term from the versioned lexicon.
  M3 measured a rise from 0.54 to 3.73 per 1k under A0; this makes that rise
  representationally impossible rather than merely discouraged.
- **punctuation-preserving** — reject edits that move the punctuation profile beyond δ.
  Grounded in this project's own data: punctuation was the *best* author discriminator
  (AUC 0.884), beating function-word Delta.
- **per-edit length band** — reject an edit whose replacement is far shorter than what it
  replaces. Applied per edit rather than per passage, so it cannot be defeated by a model
  that has decided to summarise.

A mechanical pre-filter also makes a judge affordable: panels only ever see edits that
already passed the free checks.

### 5. Budget — per-round test, or conservation law?

plan.md §8's ε is a per-round veto (reject a proposal that moves the fingerprint > ε from
round 0). The alternative is to treat ε as a **cumulative budget for the whole trajectory**:
each accepted edit spends drift, and the loop halts when the budget is exhausted.

That operationalises "non-degrading" as a conservation law rather than a repeated test, and
it fixes a hole in the per-round form — a sequence of individually-tiny edits can walk the
text arbitrarily far while never once tripping a per-round threshold. Phase 0 saw exactly
that shape: drift reached 0.99 by round 2 in small steps.

### 6. Memory

plan.md §8's A5 dismissal memory (a rejected proposal class is never re-proposed) has a
cheaper cousin worth testing first: a **do-not-touch set** of spans that have already been
reverted once. Both target the same thing — a loop that stops re-litigating.

### 7. What the model is asked for

- *Fix it* — current arms.
- **Diagnose, then fix** — one call finds problems, a second repairs only what was named.
  Worth its own arm because diagnosis is directly checkable against Stratum B, which turns
  "did it fix the defect" into "did it even find the defect", separating two failure modes
  the current design conflates.

---

## Recommended next arm

**A2i — indexed, schema-constrained, per-edit mechanically vetoed, drift-budgeted.**

- addressing: sentence index (axis 1)
- grammar: JSON schema, enforced (axis 2)
- granularity: per edit (axis 3)
- gate: mechanical only — no-new-slop, punctuation δ, per-edit length band (axis 4)
- halting: cumulative drift budget ε, or no proposals (axis 5)

It uses **zero judge calls**, so it is cheap enough to run across every model family, and it
is close to what the downstream harness would actually ship. Every component either removes
a measured failure mode or enforces a measured signal.

**It is still unfalsifiable without Stratum B.** An arm this constrained will preserve voice
almost by definition, and plan.md §6 is explicit that a gate which blocks everything scores
zero recall and must be rejected on those grounds. Build order therefore stays:

1. **Stratum B + M5 recall** — gives the second axis of the frontier.
2. **A2i** — the architecture above, measured on both axes.
3. Cross-family sweep, once there is a result worth generalising.
