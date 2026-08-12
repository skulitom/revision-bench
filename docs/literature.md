# External evidence, and where this repo stands against it

Absorbed from the owner literature sweeps `NEXT_2.md`, `NEXT_3.md` and `NEXT_4.md`
(2026-08-12), now retired to `legacy/`. **Every claim here is the literature's, not this
repo's, until reproduced here** — the status column says which.

The sweeps predate the LitRPG stratum and the Phase-2 judge work, so several entries have
since been answered by measurement. Those are marked, because a note that has already been
acted on is worse than useless if it still reads as pending.

| status | meaning |
|---|---|
| **answered** | this repo has since measured it; see the linked finding |
| **adopted** | absorbed into the plan, a metric, or a code constraint |
| **open** | still a live proposal, unbuilt |
| **blocked** | tried, and the obstacle is recorded |

---

## 1. Theory: a revision loop is a serial reproduction chain — **adopted**

Iterated-learning theory (Kalish & Griffiths) proves a chain of Bayesian learners, each
learning from the previous one's output, converges to the learners' **prior** regardless of
the starting data. An unconstrained revision loop is a single-agent chain, so plan.md §4's
attractor hypothesis is the LLM instance of a known result rather than a new conjecture.
Precedent for the method: *Probing BERT's priors with serial reproduction chains*
(arXiv 2202.12226); bridge: *Model Collapse as Cultural Evolution* (arXiv 2605.23054).

Two consequences, both now in plan.md §4:

- **H3 sharpened.** Fixed-point texts should resemble the reviser's *unconditional* prose,
  not merely cluster by family. Cheap unbuilt check: generate unconditional fiction from
  each reviser, and test whether A0 endpoints are stylometrically closer to it than to their
  own round 0.
- **A0 reframed** as a *prior probe* — a measurement of the house style, not only a control.
  The theory also predicts drift toward the prior cannot be prompted away, only structurally
  anchored, which is exactly what Phase 0/1 measured: prompt and sampler failed, A2i's
  untouched-span anchor worked.

## 2. Typicality bias: the human anchor is not neutral — **adopted, and it matters more than expected**

*Verbalized Sampling* (arXiv 2510.01171) traces post-training mode collapse to **typicality
bias in the preference data itself**: annotators systematically favour familiar, fluent,
predictable text. Together with §1 the picture closes — preference optimization concentrates
the prior, and the revision loop then walks text toward it.

The methodological consequence is sharp enough that it is now in **plan.md §2**: human
pairwise preference carries the same bias, so it is a valid anchor for *"is this fix an
improvement?"* and **not** for *"does this preserve what is distinctive?"* No pairwise
judge, human or model, can answer the second. **The preservation constraint is therefore
load-bearing even under a perfect human judge** — it is not a workaround for unreliable
ones.

This lands harder than the sweep anticipated. `findings-phase2.md` §11–§13 was built on the
assumption that the ~100-pair human subsample is the calibration anchor; §2 above says that
anchor was always going to be blind to the thing the project most wants to protect.

## 3. Stylometric fingerprints are fragile under paraphrase — **open, and a caution**

Adversarial stylometry (Brennan et al.) drops authorship attribution from ~90% to 20–30%
under paraphrase; neural paraphrasing can reach chance. The A4 voice veto's strongest
feature family on this corpus is punctuation (AUC 0.884) — exactly what rewriting destroys
first. Expect both false trips on honest edits and misses on register shifts.

Worth connecting to a result the sweep could not have known: `findings-phase2.md` §11 found
that punctuation *also* de-blinds a human A/B test, because a Gutenberg transcription and a
model's rewrite typeset differently. **Punctuation is simultaneously the strongest
attribution feature and the most fragile one** — it carries the most signal and survives
revision least. Any veto leaning on it inherits both properties.

## 4. Banalization: `lectio difficilior` operationalized — **open, high value, cheap**

Textual criticism codified copy-chain degradation centuries ago: *lectio difficilior potior*
— scribes smooth strange readings toward the expected. That is attractor convergence in
human transmission. (Inverse curiosity: scribes *added* glosses; these revisers compress.)

Proposal: per-edit **Δ log-perplexity under the reviser** — replacement minus replaced span.
An edit that lowers perplexity is making the text more expected under the model's own prior.
Flag, or veto beyond a band, as a banalization guard. Free, local, per-edit, and composes
with A2i's existing mechanical vetoes. **The most attractive unbuilt metric in these
sweeps**: it measures the attractor directly rather than by its stylometric shadow.

## 5. Judge-calibration norms — **partly answered**

From the 2026 LLM-as-judge literature (CalibraEval, judgment-distribution methods):

- Position bias: run both orderings and average. **Already done** — and
  `findings-phase2.md` §9 found 43–65% of raw verdicts are positional, worse than the
  literature's framing suggests.
- Verbosity bias: include a length-controlled calibration subset. **Partly answered** —
  §12 measured length as a discriminator at per-edit granularity and found it carries no
  signal (the edit is shorter in 47% of pairs, chance). At *passage* granularity it would,
  and §12 says so explicitly.
- Agreement thresholds in common use: human–human κ > 0.6 before trusting labels at all;
  judge–human κ < 0.5 means the rubric needs rework, not the judge. **Adopted as a
  reporting standard** for any future human calibration.

## 6. Over-editing is a named professional failure mode — **adopted (framing)**

Machine-translation post-editing (ISO 18587) treats **over-editing** — "refining sentences
toward the editor's personal style" — as a professional failure with a name, and finds the
heaviest editing level largely redundant against moderate editing. Human professionals
converged on "change only what is essential" for economic reasons; this project converged on
it for measurement reasons. Useful in the writeup: minimal intervention is the *professional
norm*, not a concession to model weakness.

## 7. Edit longevity as a judge-free quality signal — **open, fits the current direction**

Wikipedia research proxies edit quality by how long an edit survives subsequent revision.
For A5 and the downstream harness this is an implicit quality signal that **needs no judge
at all** — which is precisely the direction `findings-litrpg.md` argues for. Scoring applied
edits by survival under later rounds is buildable from artifacts this repo already writes.

Related: quality-diversity methods (QDAIF, MAP-Elites) note that optimizing one scalar
collapses diversity *by design*. Framing for the voice veto: quality subject to a
stay-in-the-author's-niche constraint, not global maximization. It explains why a scalar
"better" gate homogenizes even when the judge is unbiased.

## 8. The verbatim-anchor failure replicates in code agents — **answered (external replication)**

A2e's failure — 94.5% of rejections from paraphrased verbatim anchors — has a large-scale
precedent: aider found search/replace blocks capped at ~70–80% apply accuracy by
pattern-matching failures and moved to unified diffs for a 3× improvement; Diff-XYZ
benchmarks the formats; Copy-as-Decode formalizes grammar-constrained editing, the same move
as A2i's enforced schema.

One caveat worth carrying into the writeup: in multi-turn *code* editing, whole-file rewrite
is sometimes the stable option. For prose that option is exactly the degradation mode. Code
has a compiler to catch a bad rewrite; prose does not — which is the sentence that motivates
this whole project.

## 9. Form constraints as a fidelity mechanism: the 3,000-year A/B test — **open (hypothesis)**

Bartlett's serial-reproduction chains (unconstrained prose) degraded within a handful of
retellings; the oral-formulaic tradition (Parry/Lord: meter, formula systems,
composition-in-performance) preserved epic narrative and voice across generations without
writing. Form constraints were the fidelity mechanism. That is A0 vs A2i, run by history.

Testable hypothesis for `design-space.md`: constraints operating at the level of **form**
(meter, length bands, punctuation profile) may transmit voice with higher fidelity than
constraints at the level of **content approval** (judge gates).

## 10. LitBench — **blocked**

The sweep proposed pre-validating Phase-2 judge configurations against LitBench's human-
labelled test set before spending anything on this corpus, noting best off-the-shelf
commercial judges at ~73% and trained Bradley–Terry reward models at 78%, and flagged
"check the dataset/model licences before use".

**Checked, 2026-08-12: the released test set ships no text.** It is 2,381 rows of
`chosen_comment_id` / `rejected_comment_id`, 46.8 kB, requiring rehydration from the Reddit
API. Two further problems independent of that: the labels are upvote-derived (popularity,
not judgement — and per §2, typicality-biased in exactly the way the attractor exploits),
and each pair compares two *different stories on the same prompt* rather than two versions
of one text, which is the only comparison this project makes.

The sweep's own conclusion stands and is strengthened: use LitBench to validate *machinery*,
never as ground truth for *voice*. In practice the rehydration requirement makes even that
expensive.

**The usable substitute found instead: [IteraTeR](https://huggingface.co/datasets/wanyu/IteraTeR_human_sent)**
— Apache-2.0, 4,018 before/after sentence pairs, each a *human* revision annotated with the
reviser's intention (clarity, fluency, coherence, style, meaning-changed). It is the right
instrument for a test §2 makes urgent: our pairs always show a model's edit against a human
original, so a panel with mere status-quo bias scores identically to a panel with taste.
IteraTeR reverses the direction — the edit is the human's — so the two hypotheses finally
predict different things. Domain caveat: arXiv prose, not fiction.

## 11. Slop measurement has a literature, and full automation is open — **adopted (constraint on claims)**

*Measuring AI "Slop" in Text* (arXiv 2509.19163) built a taxonomy from 19 expert interviews
and found **automated methods did not reliably reproduce professional editors' slop
judgments**. Consequences: mine slop lexicon v2 from that taxonomy and the *Idiosyncrasies
in LLMs* catalogue rather than curating from intuition; keep M3 as what it is, a narrow and
reliable *lexicon-hit* proxy; and cite this paper in the writeup as the reason no broader
"slop score" is claimed.

## 12. A vocabulary for what each arm does: revision-type profiles — **open, cheap, sharp**

Faigley & Witte (1981) classify revisions as Surface (formal / meaning-preserving) vs
Meaning (microstructure / macrostructure); their famous finding is that experts revise
meaning while novices polish surface. A modern machine-codable successor exists (Conijn et
al. 2022 tagset).

Reporting a **revision-type profile per arm** answers a question no current metric can:
*what kind of editor is each architecture?* It also names precisely what the bounded arms
give up — sentence-indexed edits are structurally capped at surface and microstructure
changes, so **A2i is a novice-shaped editor by construction**, and A2d more so. That is the
right trade for a safety layer, but the downstream harness's ambitions eventually require
macrostructure operations, which will need their own bounded representation (scene- or
beat-level ops with their own vetoes) rather than an unbounding of A2i.

## 13. Reviser overconfidence is measurable — **adopted (one line for H4)**

LLMs exhibit significantly lower uncertainty in creative writing than professional writers
(arXiv 2602.16162): professionals entertain wide distributions over continuations; models
concentrate hard. The in-loop judge's confidence is not evidence, and there is now a
citation for why.

## 14. A2i's recall is structurally capped — **answered**

Three defect classes in `data/corpus/defects.jsonl` — continuity contradiction, timeline
slip, name drift — are **relational**: the evidence spans sentences. A per-sentence editor
never holds both halves of the contradiction in one decision, so its ceiling on those
classes is chance. External evidence agrees that detection, not repair, is the hard half:
*FlawedFictions* (arXiv 2504.11900) finds SOTA models struggle at plot-hole detection even
with full context, and long-form work attributes this to the absence of explicit state
tracking.

The sweep's prescription — **decompose M5 into detection recall (was the defect *named*?)
versus repair success (did the edit clear it?)** — is now plan.md §7, and is exactly what
the LitRPG stratum implements: `scripts/litrpg_eval.py` measures detection alone (no model
calls), `scripts/litrpg_repair_run.py` measures repair given detection.

Its diagnosis is also confirmed. Detection *was* the bottleneck: with explicit state
tracking, detection recall is 99–100% and repair clears 97% of what it is handed
([findings-litrpg.md](findings-litrpg.md)). The 0.22 that looked like a repair problem was
a detection problem.

## 15. Detect–repair–verify (A3d) — **answered in part, one gap remaining**

The sweep specified the successor architecture in three stages: detect globally by compiling
the passage into a state/fact ledger (converging with SCORE-style state tracking and
BookAgentZoo's KB design), repair locally per finding with the A2i machinery unchanged, and
verify per edit by re-running detection on the patched region.

**Built as A2d** (`revisionbench/litrpg_repair.py`) for the genre stratum, with the ledger
being the manifest and the verification being a strict fall in total complaint count. The
falsifiable prediction — that recall rises specifically on relational classes — held: the
three cross-chapter defect types are detected at 100%.

**Best-of-N is now built** (`findings-litrpg.md` §8.1), and the sweep's argument held:
resolution rose 90% → 97% and restoration 84% → 87% on the templated corpus, with 4 of 61
accepted repairs landing on a candidate past the smallest — repairs a single proposal would
have missed. Ranking is by edit distance from the replaced span, not by length, because the
shortest candidate is the one that deletes it.

One correction to the sweep's reasoning, recorded because it cost a wasted run: the claim
that "candidates for a named fix vary" silently assumed sampling. Varying only the seed at
temperature 0 returns byte-identical candidates, so the first implementation cost 2.5x the
GPU time for zero variation — `findings-phase1.md`'s dead-resampling result arriving one
level down.

## 16. A second defect corpus exists, contamination-robust — **open**

*FlawedFictions* controllably synthesizes plot holes in human-written stories — independently
the same move as Stratum B, with an algorithm designed to resist contamination. Worth
checking whether its licence, synthesizer or items can augment `data/corpus/defects.jsonl`
with plot-level holes rather than surface corruptions. It also supplies external validation
language for the planted-defect methodology in the writeup.

## 17. Farther out: trained editors, tighter representations — **open (Phase 4+)**

- **PEER** (arXiv 2208.11663) trained on Wikipedia/StackExchange *edit histories*: plan →
  edit → explain, i.e. cite-the-finding as trained behaviour. An editor whose prior is
  "human edit" rather than "rewrite" attacks the compression problem at its source. Open
  question: whether a small local model fine-tuned on edit traces beats prompting a 14B into
  edit-shaped behaviour.
- **The tagger lineage** — LaserTagger, Felix, EdiT5, G-SPEED — predicts keep/delete/insert
  per token: boundedness *below* the sentence, plus interpretability and speed. A reminder
  that the design space continues below A2i's unit if sentence-level repair proves too
  coarse.
