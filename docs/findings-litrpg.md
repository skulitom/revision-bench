# The LitRPG stratum — cross-chapter consistency without a judge

Built to answer the measurement `harness-gap.md` §3 asked for and could not run: **what is
the harness's overreach ceiling when eligibility comes from a mechanical complaint rather
than an invitation to improve?**

## 1. Why the corpus had to change

Every metric in this repo is within-passage, because the corpus is ten isolated 900-word
passages of 19th-century literary fiction. §4 of `harness-gap.md` names the gap that
actually matters for a book: the defects that hurt a manuscript most are *non-local* — a
name established in chapter 1 drifting in chapter 12, a number that only contradicts itself
two hundred pages later. **Nothing in the project could express such a defect, let alone
detect one.**

LitRPG is the genre where that problem is most tractable, because it carries explicit,
enumerable state: levels, stats, skill names, inventory. "Is this manuscript
self-consistent" becomes a diff against a table rather than a matter of opinion.

There is no licensed LitRPG corpus — the genre is roughly 2015-onward, so nothing is near
public domain, and web-serial chapters belong to their authors. So the world is generated
first, as a state machine, and the prose is rendered *from* it. The manifest is not an
annotation of the text but its source, which makes the ground truth exact.

## 2. Result

20 manuscripts × 16 chapters, 153 planted contradictions
(`scripts/litrpg_eval.py`, `results/litrpg/eval.json`):

| defect type | recall | planted |
|---|---:|---:|
| stat_drift | 100% | 32 |
| level_regression | 100% | 7 |
| skill_before_acquisition | 100% | 37 |
| inventory_ghost | 100% | 37 |
| entity_rename | 95% | 40 |
| **overall** | **99%** | **153** |
| of which genuinely cross-chapter | 100% | 76 |

| | |
|---|---:|
| complaints raised | 171 |
| matching a planted defect | 151 |
| **precision** | **88%** |
| false positives on a clean manuscript | **0** |

**No model was called.** No judge, no GPU, no canonical-facts database supplied from
outside — the detectors read the manuscript's own status blocks and check it against
itself.

## 3. Why precision is the number that matters

Under detect-then-repair only a complained-of span may be edited, so precision *is* the
overreach ratio. `harness-gap.md` §1.1 measured the revise-then-gate loop at **~24 applied
edits per defect repaired**, with ~96% of what it changed not broken, and called that the
product blocker. At 88% precision the same quantity is **1.1 edits licensed per real
defect**.

That is the architectural claim from §2.1, now with a number behind it: *overreach becomes
bounded by detector precision, which is measurable, rather than by a model's restraint,
which is not.*

## 4. What this does not establish

Stated plainly, because the numbers above are the highest in this project and the temptation
to quote them without the caveats will be strongest.

- **It says nothing about prose quality.** The clean text is not good writing and is not
  claimed to be. This measures whether contradictions are *findable*, not whether revisions
  are *better*. Every quality claim still rests on Stratum A.
- ~~**Both numbers are upper bounds**~~ — tested in §7 on model-written chapters, where
  recall and precision held (100%/90%) but only after two silent failures the templated
  corpus could not expose. The remaining untested assumption is a *fixed* status-block
  schema; phi4 varies whitespace but keeps the field set, and a real serial would not.
- **The detectors were built against this injector's five types.** They are not a general
  consistency checker, and recall on defect classes nobody thought to plant is unmeasured —
  the same caveat Stratum B carries.
- **The repair step is now built** (§6) but runs one model, phi4:latest, on templated prose.
  Model-dependence is unmeasured, and NEXT.md's standing rule requires it before any
  threshold ships.

## 5. Bugs worth recording

All three produced a plausible number rather than a failure, which is this project's
recurring failure mode.

1. **`detect_entity_rename` matched any token of a canonical name against the following
   word**, so `Frost Nail without thinking` read as a variant of `Frost Nail` — 20 false
   positives per clean manuscript. Anchoring on the name's leading tokens fixed it.
2. **The injector replaced the first occurrence of a skill name, which is inside the status
   block, not the prose.** The planted defect therefore agreed with the canonical record and
   was undetectable by construction. `skill_before_acquisition` and `entity_rename` scored
   0% recall and it looked like a detector problem.
3. **`inventory_ghost` chose items held in an adjacent chapter.** Prose legitimately names
   an item in the chapter it is lost, so those were not contradictions at all, and scoring
   them as misses understated the detector.

A fourth was a presentation error rather than a code one: every defect was flagged
`cross_chapter=True`, so the cross-chapter recall line read 153/153 and meant nothing. Two
of the five types are detectable within a single chapter; the corrected split is 76 of 153.

## 6. A2d — the repair half, with a mechanical acceptance rule

`revisionbench/litrpg_repair.py`, `scripts/litrpg_repair_run.py`. 10 manuscripts × 14
chapters, 63 planted defects, **phi4:latest, 21 seconds total**:

| | |
|---|---:|
| complaints | 68 → **5** |
| proposals | 67 |
| accepted | 57 (85%) |
| defect resolved | 57/63 = **90%** |
| **exactly restored** (corruption gone *and* manifest text back) | 53/63 = **84%** |
| **chapters edited that had no defect in them** | **0** |

Rejections: 4 `out_of_scope`, 4 `changed_form`, 2 `complaint_persists`.

Two properties do the work, and neither is a prompt instruction:

- **Eligibility is structural.** The model never sees the manuscript — it sees one complaint
  and one span, and returns replacement text for that span. "Leave everything else alone" is
  not an instruction it could ignore; there is no channel through which it could disobey.
  Same move that fixed length in Phase 1.
- **Acceptance is mechanical.** A repair is applied only if the manuscript's total complaint
  count strictly falls. No judge, no model call, no threshold.

The last row is the one that matters. `harness-gap.md` §1.1 measured revise-then-gate at
**~24 applied edits per defect repaired**, with ~96% of what it touched not broken, and
called that the product blocker. A2d applied 57 edits and repaired 57 defects, and **not one
chapter without a complaint against it was modified at all.**

The gap between 90% resolved and 84% exactly restored is the honest part: 6 defects were
made to *stop complaining* without the manifest's text coming back. Only a stratum with
byte-exact ground truth can see that difference, and every metric this project had before
would have scored those as clean repairs.

### 6.1 Two holes in the acceptance rule, found by testing it

Both let a repair pass by **destroying the evidence** rather than fixing the fact, and both
are general — any verifier that only asks "did the complaint go away" has them.

1. **Field deletion.** The cheapest way to resolve "Strength changes 10 → 14 with no
   level-up" is to stop mentioning Strength. Complaint gone, count down, every check passed,
   a fact silently lost from the manuscript.
2. **The mirror image.** Replacing a *prose* span with `  Level: 999` also resolves its
   complaint, because the offending phrase is gone. An earlier version of the guard
   protected status fields only and accepted this.

Both are closed by a symmetric form check: a status field must come back as the same status
field, and prose must come back as prose. Enforced structurally, for the same reason
eligibility is — a rule the model can ignore is not a rule.

A third issue was in the rule's identity function. Complaints were compared by message, but
messages carry the offending values, so a repair that changed `falls to 3` into `falls to 2`
registered as resolving one complaint and introducing another when it had simply failed.
Counting complaints instead cannot be fooled that way: fix one and break one and the total
is unchanged, which is a rejection.

## 7. Model-written chapters — the upper bounds tested

§4 warned that every number above was an upper bound, because the prose was templated and
the status block had a fixed schema. `scripts/litrpg_generate.py` replaces the templates
with phi4-written chapters conditioned on the same manifest, so the pipeline can be measured
on prose that varies the way real prose does.

**Generation fidelity: 112/112 chapters (100%), every one on the first attempt**, 600s.
Validation is not optional here — a model that invents state produces contradictions the
manifest cannot adjudicate, and those would land in the corpus as detector false positives
when they are really corpus errors. Every chapter is parsed and checked against the manifest
(level, every stat, skills, inventory) before it is kept.

That phi4 holds a handed-to-it state table across 14 chapters with no failures is itself a
result: **composition under a fact constraint is not the hard part.**

| | templated | model-written |
|---|---:|---:|
| recall | 99% | **100%** |
| precision | 88% | **90%** |
| false positives on clean text | 0 | **0** |
| defects resolved (A2d) | 90% | **97%** |
| exactly restored (A2d) | 84% | **97%** |
| chapters edited with no defect in them | 0 | **0** |

The numbers survived — but only after two failures that the templated corpus could never
have exposed, and both were silent.

### 7.1 The status block lost its indentation

phi4 reproduces the *fields* and quietly renormalises the whitespace: `Name: Bright` where
the template has `  Name: Bright`. The parser required exactly two spaces, so it found **no
status fields at all** — which raises no complaints, which reads as *perfect precision*.

Indentation is not part of the schema. The parser now tolerates arbitrary leading
whitespace and list markers, bounded by a known field-name set so a line of dialogue like
`Bex: not today` cannot become a canonical fact.

### 7.2 Precision collapsed to 23%, and all of it was one rule

First honest run on generated prose: recall 100%, **precision 23%, 118 false positives on
clean text.** Every one was `entity_rename`, and the cause was that it matched a skill's
leading tokens *case-insensitively* — so `Glass Song` fired on "glass shards" and "glass
with", `Silent Palm` on "silent enigma". Templated prose never used those words as common
nouns. Model prose does constantly.

Two fixes, neither requiring a dictionary — `detect.py` holds the line that a detector needs
no dictionary or cast list, and both rules keep that property:

1. **Case-sensitive matching plus capitalisation shape.** A variant of a proper noun is
   still a proper noun, so a lowercase match is ordinary prose. Recovered precision to 84%.
2. **A closed-class function word cannot end an item name.** Capitalisation does nothing for
   lowercase items, so `salt pouch` still fired on "salt from", "salt on", "salt and" — and
   A2d then dutifully repaired them, rewriting `salt and` to `sea`. Filtering on the same
   versioned function-word list Burrows' Delta uses removed exactly those three. A fixed
   grammatical category, not an open-ended word list.

Final: **precision 90% on generated prose, 88% on templated** — and the templated numbers
did not regress, which matters because a fix tuned on one corpus that degrades the other is
overfitting rather than correction.

### 7.3 What the collateral edits proved

Before the second fix, 2 chapters were edited that had no defect in them. Tracing them was
the point: a false complaint becomes a **licensed edit on clean prose**, and the mechanical
acceptance rule cannot catch it, because the complaint does genuinely go away.

That is the sharpest statement of why precision is the number that matters. Recall failures
leave a defect in the manuscript; precision failures *actively damage text that was fine*.
The safety guarantee is exactly as strong as detector precision and no stronger.

(One correction worth recording: `cracked grindstone` and `Thorn Vine` initially looked like
model-introduced drift. They are not — they are the injector's own variant swaps,
`whetstone→grindstone` and `Coil→Vine`, so they were planted defects found and repaired
correctly. Only the `salt` complaints were true false positives.)

## 8. What this answers from the literature sweeps

`NEXT_4.md` (retired to `legacy/`, absorbed into [`literature.md`](literature.md) §14–§15)
predicted this work before it was built, and two of its claims are now settled.

**Detection was the bottleneck, not repair.** The sweep argued A2i's 0.22 recall was
structurally capped because three defect classes are *relational* — the evidence spans
sentences, so a per-sentence editor never holds both halves of the contradiction in one
decision. It prescribed decomposing M5 into detection recall versus repair success, on the
grounds that the pooled number could not distinguish an architectural cap from a repair
failure. Decomposed here: detection recall 99–100%, repair clears 97% of what it is handed.
**The number that looked like a repair problem was a detection problem**, and explicit state
tracking is what moved it.

**The falsifiable prediction held.** The sweep predicted a detect–repair–verify arm would
raise recall specifically on the relational classes. All three cross-chapter defect types
are detected at 100%.

**The last specified element is now built — see §8.1.**

### 8.1 Best-of-N, ranked by minimal intervention

The sweep asked for best-of-N per finding, keeping the **smallest** candidate that clears
it, arguing that edit-level best-of-N is live where round-level re-rolling was dead: the
candidates for a *named* fix genuinely vary, whereas whole-round lengths did not. Built, and
the argument holds — but only after the first implementation of it did nothing at all.

Ranking is by **Levenshtein distance from the span being replaced**, not by length. A
candidate that deletes the span is shortest by length and maximal by damage; distance from
what is being replaced is what "minimal intervention" actually means.

Templated corpus, 63 planted defects:

| | N=1 | N=3 |
|---|---:|---:|
| complaints remaining | 5 | **1** |
| proposals accepted | 57 (85%) | **61 (97%)** |
| defect resolved | 90% | **97%** |
| exactly restored | 84% | **87%** |
| chapters edited with no defect | 0 | 0 |
| wall clock | 23s | 53s |

Sampling produced genuine variation on **43 of 63** complaints (2–3 distinct candidates
after deduplication). Of the 61 accepted repairs, 57 landed at rank 0 — the smallest
candidate was both available and correct — and **4 landed at rank 1 or 2**, which is exactly
the 57→61 gain. Those four are repairs a single proposal would have missed.

On the model-written corpus the arm was already at its ceiling (97%/97% at N=1), and N=3
holds it there while cutting residual complaints from 1 to 1 — no gain, and none available.

### 8.2 The first implementation was a no-op, and the cause is a repeat offender

The first version varied only the **seed** across candidates. The runner generates at
**temperature 0**, where decoding is deterministic — so three generations returned three
byte-identical strings, deduplication collapsed them to one, and the run cost 2.5× the GPU
time for exactly zero variation. Measured before the fix: `candidates_seen == 1` on all 67
complaints, and every headline number identical to N=1 to the digit.

This is `findings-phase1.md`'s resampling result arriving one level down. There, re-rolling a
whole round did nothing because the model held a target length across seeds; here,
re-rolling an edit did nothing because greedy decoding has no seed dependence at all. The
sweep's claim that "candidates for a named fix vary" was right, but it silently assumed
sampling.

The fix keeps candidate 0 exactly as configured — normally greedy, so the primary proposal
stays reproducible — and samples every later candidate at `candidate_temperature` (0.7).
Pinned by a test that asserts the temperature actually rises, not merely that the seeds
differ: the earlier test checked seeds and passed throughout the no-op.

## 9. Where this leaves the plan

The judge is not on the critical path for this class of defect, and this is the first
measurement in the project that shows it rather than argues it. Next, in order:

1. ~~**The repair half**~~ — built, §6. Next on this axis: the 6 defects that resolved
   without restoring, which is where a repair still degrades the manuscript undetectably.
2. ~~**Model-written chapters**~~ — done, §7. Numbers held; two silent failure modes found.
3. **Schema variation** — vary the status-block format *between* manuscripts (bullets,
   tables, inline prose stats). Now the least realistic remaining assumption, and §7.1
   showed how quietly a parsing failure reads as a perfect score.
4. **Model-dependence** — everything here is phi4. NEXT.md's standing rule requires
   gemma3:27b and Qwen before any threshold ships, and the §7.2 fixes were tuned against
   one model's habits.
