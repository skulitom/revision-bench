# What stands between this research and a usable book harness

Written after the Phase-1 frontier. The question: the goal is a harness that can work over a
book-length manuscript without degrading it — what actually has to be true for that, and
what is missing?

Evidence: [`findings-phase0.md`](findings-phase0.md), [`findings-phase1.md`](findings-phase1.md),
[`design-space.md`](design-space.md).

---

## 1. Two numbers that decide the product question

### 1.1 The loop is not aimed

A2i applied **195 edits and repaired 8 planted defects** — about 24 applied edits per defect
fixed, and roughly 4% of edits landing on a defective span. Extrapolated to a 100,000-word
manuscript (~110 passages of the size used here), one pass at that rate is on the order of
**2,100 edits to repair ~90 real problems.**

That is the product blocker. Not safety of application — A2i applies edits reliably now
(74% apply rate, zero protocol failures) — but that ~96% of what it changes was not broken.

### 1.2 "Voice-preserving" does not yet survive calibration

Voice drift is easy to read as small because the numbers are small. Against the scale that
matters, it is not. Measured on this corpus, with the same all-feature metric the frontier
reports:

| comparison | distance |
|---|---|
| two passages by the **same** author | 0.967 |
| two passages by **different** authors | 1.161 |
| **the entire between-author signal** | **0.194** |

Against that width:

| arm | drift from clean | as a multiple of the author gap |
|---|---|---|
| A2e edit-list | 0.13 | 0.7× |
| **A2i indexed + vetoes** | **0.31** | **1.6×** |
| A2p paragraph | 0.84 | 4.3× |
| A0 whole passage | 0.96 | 5.0× |

Read this carefully, because it is easy to overstate. A revision at 0.31 is still far
*closer* to its original than any two distinct passages are to each other (~0.97+). What the
table says is narrower and still damning: **the perturbation these arms apply is larger than
the whole signal that separates one author from another.** Magnitude alone therefore cannot
support a voice-preservation claim — what matters is whether the perturbation is
*directional*, and whether it points toward a shared house style is precisely H1/H3, which
remains unmeasured for these arms.

The practical consequence is concrete: **a voice veto threshold ε has to be calibrated
against the author gap (0.194), not chosen by eye — and at ε meaningfully below that, no arm
built so far passes.** A2i at 0.31 would be vetoed.

---

## 2. The architectural consequence: detect, then repair

Every arm so far has the same shape: **revise, then gate.** The model is invited to improve
the whole unit, and something downstream decides how much of that to keep. Bounded diffs
made the *keeping* safe. They did nothing about the *inviting*, which is where the 24:1
overreach comes from — a model asked to improve a paragraph will always find something to
change, because that is what it was asked for.

The inversion:

> **Only a span with a located, checkable complaint against it is eligible to be edited.
> Everything else is frozen.**

This is the same move that fixed length. Length stopped being a problem when it became
structurally impossible to change unnamed text, rather than discouraged by a prompt.
Overreach stops being a problem when it becomes structurally impossible to edit an
unflagged span.

Note what this implies about plan.md §8. **A2 has two halves — "must cite a specific
finding" and "stay within a bounded diff size" — and only the second has been built.** The
citation half is the untested one, and it is the one that addresses the measured failure.

### 2.1 Most of the defect classes that matter need no judge at all

The defects a manuscript actually accumulates — name drift, timeline slips, continuity
contradictions, echo — are **mechanically detectable given a record of what is canonical**.
That is exactly the "compiler-like consistency checking" BookAgentZoo proposes (plan.md §10),
and Stratum B's taxonomy is already built from those categories.

For those classes the acceptance test is not aesthetic and needs no panel:

> the repair is accepted if the complaint it cites resolves, and no new complaint appears.

Checkable, free, and idempotent. An LLM judge is then needed only for the aesthetic residue
— the "clunker" class and anything a linter cannot state — which is a far smaller and far
more tractable judging problem than "is this revision better".

### 2.2 What the harness looks like under the inversion

1. **Detect.** A linter over the manuscript emits located, typed, checkable complaints.
   Mechanical where the class allows; LLM-assisted where it does not, but still required to
   emit a *located claim* rather than a rewrite.
2. **Repair, scoped.** The model sees one complaint plus its span and local context, and
   proposes a repair for that complaint only. A2i's machinery already does this part.
3. **Verify.** Accept only if the cited complaint resolves, no new complaint appears, and
   the drift budget is not exhausted.
4. **Freeze everything else.** No complaint, no edit.
5. **Remember dismissals.** A rejected repair class is not re-proposed (plan.md §8 A5).

Overreach under this design is bounded by detector precision, which is measurable — rather
than by a model's restraint, which is not.

---

## 3. What this makes measurable immediately

**Detector precision and recall against Stratum B, with no judge and no new model calls.**
40 planted defects with exact spans already exist. A detector can be scored directly:
does it find the planted defect, and how many complaints does it raise about spans that
were never corrupted?

That number is the harness's overreach ceiling, and it is the missing input to every
threshold. It is also, importantly, **inside Phase 2's remit as plan.md §9 already defines
it** — Phase 2 is explicitly about correlating *every candidate gate signal*, including
"M1–M3 mechanical signals", against the blinded panel. A linter is a mechanical gate signal.
So this is not mechanism-building queue-jumping the judge study; it is one of the signals the
judge study exists to rank.

---

## 4. Book-scale gaps that nothing here addresses yet

The corpus is ten isolated 900-word passages. A manuscript is not that, and four things
follow that the current code has no representation for:

- **A document model.** Chunking, ordering, reassembly, and stable identity for a span
  across edits. Today a "passage" is a standalone file.
- **Cross-passage consistency.** The defects that hurt a book most are non-local: a name
  established in chapter 1 drifting in chapter 12, a timeline that only contradicts itself
  200 pages apart. Every metric in this repo is within-passage. A canonical-facts store is
  the prerequisite, and BookAgentZoo already has a KB schema for it.
- **Incrementality.** Re-checking an entire book after every edit is not viable; the harness
  needs to know what a change could have invalidated.
- **A human surface with memory.** The author accepts and rejects, and rejections must
  stick. plan.md §10 already names AgentUI as a candidate.

---

## 5. Recommended order

1. **Phase 2 as scheduled** (`NEXT.md` priority 2), with one addition to its design: include
   a **mechanical detector as a candidate gate signal** alongside the self-judge and the
   cross-family panel. If the linter handles the consistency classes, the judge's job
   shrinks to the aesthetic residue — and that is a Phase-2 finding, not an assumption.
2. **Detector precision/recall against Stratum B.** No judge, no new generations, answers
   the overreach question directly.
3. **The detect-then-repair arm (A2d).** Built on A2i's application machinery; the only
   change is that eligibility comes from a complaint rather than from an invitation.
4. **Calibrate ε against the author gap** (§1.2) and re-run the frontier with the veto live.
   Report which arms survive it — on current numbers, none do.
5. **Model-dependence replication** before any threshold ships (`NEXT.md` standing rule).
6. **Then** the document model and cross-passage consistency, which is where the research
   becomes a product rather than a result.

The honest summary: the project has established what *not* to do with high confidence, and
has one architecture that fails safely. It does not yet have an architecture that fixes
things without also touching ~96% of what it should have left alone, and that — not voice
preservation — is now the gap to a usable harness.
