# NEXT_3 — external research notes, second sweep (2026-08-12)

Continues `NEXT_2.md` (entries A–G; this file: H–N). Same rules: these are the
literature's claims until reproduced here; absorb what earns its place, then move this
file to `legacy/` per the note lifecycle in `NEXT.md`.

---

## H. The attractor has a named mechanism — and it implicates human judges too

[Verbalized Sampling](https://arxiv.org/pdf/2510.01171) traces post-training mode collapse
to **typicality bias in the preference data itself**: annotators systematically favor
familiar, fluent, predictable text, a well-established cognitive-psychology effect. Two
consequences, one theoretical and one methodological:

- The house-style attractor is not an accident of any model — it is what preference-based
  optimization *does* when the preference signal carries typicality bias. Together with
  NEXT_2 §A (chains converge to the prior), the picture closes: preference optimization
  concentrates the prior; the revision loop then walks text toward it.
- **The ~100-pair human subsample is not a neutral oracle for voice.** Human pairwise
  preference carries the same typicality bias — which is *why* Bartlett's human chains
  converged. Phase 2 must treat human preference as the anchor for "is this fix an
  improvement?" while treating "does this preserve what is distinctive?" as a separate
  question that pairwise preference — from any judge, human or model — cannot answer. This
  is the strongest argument yet that the preservation constraint (A4 veto / A2i bounds) is
  not a workaround for unreliable judges: it is load-bearing *even under a perfect human
  judge*. Consider promoting this to plan.md §2.

## I. The interface finding replicates independently in code agents

The A2e failure (94.5% of rejections from paraphrased verbatim anchors) has a large-scale
precedent: aider's edit-format work found search/replace blocks limited to ~70–80% apply
accuracy by pattern-matching failures, and moved to
[unified diffs, 3× improvement on GPT-4 Turbo](https://aider.chat/docs/unified-diffs.html)
([edit formats doc](https://aider.chat/docs/more/edit-formats.html));
[Diff-XYZ](https://arxiv.org/html/2510.12487v1) benchmarks the formats and
[Copy-as-Decode](https://arxiv.org/pdf/2604.18170) formalizes grammar-constrained editing —
the same move as A2i's enforced schema. Two uses: (a) cite as independent replication that
verbatim-quoting interfaces fail across domains; (b) one caveat worth carrying — in
multi-turn code editing, whole-file rewrite is sometimes the *stable* option. For prose
that option is exactly the degradation mode, which is a difference worth one sentence in
the writeup: code has a compiler to catch a bad rewrite; prose does not.

## J. Constraint-based transmission fidelity: the 3,000-year A/B test

Bartlett's serial-reproduction chains (unconstrained prose) degraded within a handful of
retellings; the [oral-formulaic tradition](https://en.wikipedia.org/wiki/Oral-formulaic_composition)
(Parry/Lord: meter + formula systems, "composition-in-performance") preserved epic
narrative and voice across generations without writing. Form constraints were the
fidelity mechanism. That is A0 vs A2i, run by history. Framing/epigraph material for the
writeup rather than a design input — but it does suggest a hypothesis worth a line in the
design-space doc: constraints that operate at the level of *form* (meter, length bands,
punctuation profile) may transmit voice with higher fidelity than constraints at the level
of *content approval* (judge gates).

## K. Phase-2 shortcut: LitBench, and judge-validity numbers for fiction

- Creative-writing judge agreement with humans is mediocre: ~58% in one line of work;
  [LitBench](https://arxiv.org/pdf/2507.00769) measures the best off-the-shelf commercial
  judge at **73%** — and **trained Bradley–Terry reward models at 78%, beating every
  off-the-shelf judge**. LitBench releases a 2,480-pair human-labeled test set, a
  43,827-pair training corpus (Reddit stories), and the trained reward models, via
  Hugging Face (paper CC-BY-4.0; check the dataset/model licences before use).
- Uses, in order of value: (1) **pre-validate Phase-2 judge configurations against
  LitBench's test set before spending anything on our corpus** — a judge setup that cannot
  clear ~73% there has no business gating here; (2) a trained local reward model fits the
  no-API rule and may outperform prompted local judges — candidate for the A3 gate signal;
  (3) their debiasing procedure is prior art for our pair construction.
- Two caveats: Reddit short-story preferences are a domain shift from 1920s literary
  prose, and crowd upvotes are a *typicality-biased* signal (§H) — a reward model trained
  on them likely inherits exactly the bias the attractor exploits. Use LitBench to
  validate the *machinery*, not as ground truth for *voice*.

## L. Slop measurement has a literature now, and full automation is an open problem

[Measuring AI "Slop" in Text](https://arxiv.org/abs/2509.19163) built a taxonomy from 19
expert interviews (Information Utility / Information Quality / Style Quality) and found
that **automated methods did not reliably reproduce professional editors' slop
judgments** — their dimensions shift with context. Implications: (a) mine slop lexicon v2
from their taxonomy and the [Idiosyncrasies](https://arxiv.org/html/2502.12150) catalog
rather than curating from intuition (NEXT_2 §B stands); (b) M3 stays what it is — a
narrow, reliable *lexicon-hit* proxy — and the writeup should cite this paper as the
reason no broader "slop score" is claimed; (c) their token-entropy density metric is
another free, local candidate signal for Phase 2's signal-validity table.

## M. A vocabulary for what each arm actually does: revision-type profiles

[Faigley & Witte (1981)](https://www.researchgate.net/figure/Taxonomy-of-Revision-Changes-Faigley-and-Witte-1981403-at-meaning-level-but-produced_fig1_258145720)
classify revisions as Surface (formal / meaning-preserving) vs Meaning
(microstructure / macrostructure); their famous finding is that experts revise meaning
while novices polish surface. A modern machine-codable successor:
[Conijn et al. 2022 revision tagset](https://journals.sagepub.com/doi/10.1177/07410883211052104).
- Cheap, informative addition to the frontier: report a **revision-type profile per arm**
  (fractions of formal / meaning-preserving / micro / macro changes, codable by a local
  model against the taxonomy). It answers a question the current metrics cannot: *what
  kind of editor is each architecture?*
- It also names precisely what A2i gives up: sentence-indexed edits are structurally
  capped at surface + microstructure changes. By the expert-novice finding, **A2i is a
  novice-shaped editor by construction.** That is the right trade for the safety layer,
  but the downstream harness's ambitions (§ plan.md, BookAgentZoo) eventually require
  macrostructure operations — which will need their own bounded representation (scene- or
  beat-level ops with their own vetoes), not an unbounding of A2i.

## N. Small but telling: reviser overconfidence is measurable

[LLMs exhibit significantly lower uncertainty in creative writing than professional
writers](https://arxiv.org/html/2602.16162v1) — professional writers entertain wide
distributions over continuations; models concentrate hard. A one-line connection for H4:
the in-loop judge's confidence is not evidence, and now there is a citation for why.
