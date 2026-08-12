> **Retired 2026-08-12.** Absorbed into `docs/literature.md` (consolidated, with a
> status against each claim), plus targeted edits to `plan.md` sections 2, 4, 7 and 8,
> `docs/findings-phase2.md` section 14 and `docs/findings-litrpg.md` section 8.
> Kept for provenance per the note lifecycle in `NEXT.md`.

# NEXT_2 — external research notes (2026-08-12)

Reading notes from an owner-initiated literature sweep. Not instructions to redesign
anything: absorb what earns its place into `plan.md`, the findings docs, or code, then move
this file to `legacy/` per the note lifecycle in `NEXT.md`. Every claim below is the
literature's, not this repo's, until reproduced here.

---

## A. Theory: a revision loop is a serial reproduction chain

Iterated-learning theory ([Kalish & Griffiths](https://langev.com/pdf/kalish07iteratedLearning.pdf);
[OECS overview](https://oecs.mit.edu/pub/boh416wu/release/1)) proves that a chain of
Bayesian learners, each learning from the previous one's output, converges to the
learners' **prior**, regardless of the starting data. An unconstrained revision loop is a
single-agent chain, so plan.md §4's attractor hypothesis is the LLM instance of a known
result — cite it. Precedent for the method:
[Probing BERT's priors with serial reproduction chains](https://arxiv.org/pdf/2202.12226);
bridge paper: [Model Collapse as Cultural Evolution](https://arxiv.org/pdf/2605.23054).

Two consequences:
- **Sharpen H3's prediction:** fixed-point texts should resemble the reviser's
  *unconditional* prose. Cheap check: generate unconditional fiction from each reviser and
  measure whether A0 endpoints are stylometrically closer to it than to their round 0.
- **Reframe A0:** its fixed point is a *prior probe* — a measurement of the house style,
  not merely a control arm. The theory also predicts drift toward the prior cannot be
  prompted away, only structurally anchored — consistent with what Phase 0/1 measured
  (prompt and sampler failed; A2i's untouched-spans anchor worked).

## B. Instrument: model-attribution classifiers can serve H3

Stylometric attribution discriminates five model families at ~97%
([expert-systems study](https://www.sciencedirect.com/science/article/pii/S0957417425026181));
the tells are catalogued in [Idiosyncrasies in LLMs](https://arxiv.org/html/2502.12150).
- Candidate metric: **rounds-to-attribution-flip** — the round at which a passage stops
  attributing to its human author and starts attributing to the reviser. Local, free,
  per-passage, and more interpretable than cluster purity.
- Slop lexicon v2 should be mined from the idiosyncrasies literature rather than curated
  from intuition (the two project-curated groups have still never fired — see NEXT.md).

## C. Caution: stylometric fingerprints are fragile under paraphrase

Adversarial-stylometry results ([Brennan et al.](https://dl.acm.org/doi/10.1145/2382448.2382450);
[overview](https://www.emergentmind.com/topics/adversarial-stylometry)): paraphrase drops
authorship attribution from ~90% to 20–30%; neural paraphrasing can reach chance. The A4
veto's strongest feature family in this corpus (punctuation, AUC 0.884) is exactly what
rewriting destroys first. Expect both false trips on honest edits and misses on register
shifts. **The veto gates nothing until it is calibrated against the human subsample in
Phase 2.**

## D. Metric candidate: banalization, i.e. lectio difficilior operationalized

Textual criticism codified scribal copy-chain degradation centuries ago:
[*lectio difficilior potior*](https://en.wikipedia.org/wiki/Lectio_difficilior_potior) —
scribes smooth strange readings toward the expected. That is attractor convergence in
human transmission. (Inverse curiosity: scribes tended to *add* glosses
([lectio brevior](https://en.wikipedia.org/wiki/Lectio_brevior)); these revisers compress.)
- Proposal: per-edit **Δ log-perplexity under the reviser** — replacement minus replaced
  span. An edit that lowers perplexity is making the text more expected under the model's
  own prior. Flag (or veto beyond a band) as a "banalization guard". Free, local, and
  per-edit, so it composes with A2i's existing mechanical vetoes.

## E. Phase-2 design: current judge-calibration norms

From the 2026 LLM-as-judge literature
([bias roundup](https://futureagi.com/blog/evaluating-llm-judge-bias-mitigation-2026/),
[reliability data](https://www.adaline.ai/blog/llm-as-a-judge-reliability-bias),
[CalibraEval](https://arxiv.org/pdf/2410.15393),
[judgment-distribution method](https://arxiv.org/pdf/2503.03064)):
- Position bias: run both orderings on every pairwise call and average; doubles cost,
  near-eliminates the effect. (Already in plan.md §11 — keep.)
- Verbosity bias: include a length-controlled calibration subset (pairs within ±20% length
  at matched human-rated quality); if winrate collapses there, length was doing the work.
- Agreement thresholds in common use: human–human κ > 0.6 before trusting labels at all;
  judge–human κ < 0.5 means the rubric, not the judge, needs rework.
- All of this is per-edit compatible: the primary validated question is "apply this scoped
  edit or not", per NEXT.md priority 2.

## F. Convergent evidence: professional editors codified the same trap

Machine-translation post-editing (ISO 18587; [guidelines comparison](https://www.researchgate.net/publication/303564681_A_Comparative_Study_of_Post-editing_Guidelines);
[levels study](https://aclanthology.org/2020.eamt-1.33.pdf)) treats **over-editing** —
"refining sentences toward the editor's personal style" — as a named professional failure
mode, and finds the heaviest editing level largely redundant against moderate editing.
Human professionals converged on "change only what is essential" for economic reasons;
this project is converging on it for measurement reasons. Useful in the eventual writeup:
minimal intervention is the *professional* norm, not a concession to model weakness.

## G. Two systems ideas worth stealing

- **Edit longevity** (Wikipedia research: [edit longevity & contributor centrality](https://arxiv.org/pdf/1206.2517)):
  edit quality proxied by how long an edit survives subsequent revision. For A5 and the
  downstream harness: score applied edits by survival under later rounds / author
  retention — an implicit quality signal that needs no judge at all.
- **Quality-diversity** ([QDAIF](https://arxiv.org/pdf/2310.13032),
  [MAP-Elites overview](https://www.emergentmind.com/topics/map-elites-algorithm)):
  optimizing one scalar collapses diversity *by design*; QD maintains quality per niche.
  Frame for the voice veto: quality subject to a stay-in-the-author's-niche constraint,
  not global maximization. Explains why a scalar "better" gate homogenizes even when the
  judge is unbiased.
