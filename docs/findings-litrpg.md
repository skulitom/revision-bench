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
- **Both numbers are upper bounds.** The prose is templated and the status block has a fixed
  schema, so extraction is near-perfect. Real serials vary formatting constantly. Read these
  as "what is achievable when parsing is easy", not as a forecast.
- **The detectors were built against this injector's five types.** They are not a general
  consistency checker, and recall on defect classes nobody thought to plant is unmeasured —
  the same caveat Stratum B carries.
- **No repair step exists yet.** Detection is half of detect-then-repair. The acceptance
  rule — *the repair is accepted if the complaint it cites resolves and no new complaint
  appears* — is implementable from here but unbuilt.

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

## 6. Where this leaves the plan

The judge is not on the critical path for this class of defect, and this is the first
measurement in the project that shows it rather than argues it. Next, in order:

1. **The repair half** — A2d, scoped to one complaint, accepted only if the cited complaint
   resolves and no new one appears.
2. **Model-written chapters** (`litrpg.prompt_for_chapter`) instead of templated ones, with
   the same manifest as ground truth. That measures how much of the 88% survives real prose,
   and it is the single most informative follow-up.
3. **Schema variation** — vary the status-block format across manuscripts, since fixed
   formatting is what makes extraction near-perfect and is the least realistic assumption
   here.
