# revision-bench — Research Plan

**Goal:** an open, reproducible study of *revision-loop dynamics* in LLMs — measuring when an iterative "improve this prose" loop degrades text (homogenization, voice loss, slop, thrash), and which loop architectures make revision **non-degrading** while still fixing real defects.

**Licence:** Apache-2.0 · **Status:** planning · **Last updated:** 2026-08-11

This document is the working plan for the repo. It is written to be executed incrementally with Claude Code: each milestone is a self-contained session-sized unit with acceptance criteria. Treat unverified assumptions (§12) as things to test, not facts.

---

## 1. What this is (and is not)

- **Is:** measurement infrastructure. Controlled revision loops over a fixed prose corpus, instrumented with stylometric, lexical, and blinded-preference readouts, scored per round, with seeds, confidence intervals, and full provenance.
- **Is not:** a writing product, a claim that AI editing is good or bad, or a prompt-engineering demo. All language in code, docs, and writeups stays at the level of *measured change per revision round*. That discipline is load-bearing for credibility (same rule as mirror-bench §1).

**Downstream consumer:** a continuous book-perfecting harness (see §10, `C:\DEV\BookAgentZoo`). The winning loop architecture from Phase 3 — with its measured thresholds — becomes that product's edit-acceptance layer. But revision-bench must stand alone as a research artifact.

---

## 2. The problem (the trap)

An unbounded revision loop with an LLM in the "is this better?" seat does not asymptotically approach perfection. Expected failure modes:

1. **Homogenization** — distinct authorial voices converge toward the reviser model's house style.
2. **Voice loss** — idiosyncrasy (rhythm, punctuation habits, function-word fingerprint) sanded off.
3. **Slop injection** — LLM-tell lexicon and formulaic constructions accumulate.
4. **Thrash** — later rounds revert earlier rounds' edits; no fixed point.
5. **Report–state divergence** — the in-loop judge keeps reporting "improved" while blinded external judgment flattens or falls. This is the mirror-bench problem in a new domain: the *report* ("this edit is better") and the *state* (is it?) are not the same thing, and the gap is measurable.

"Perfecting" only converges when there is a trustworthy signal of *better*. For prose aesthetics no such signal exists off the shelf: LLM judges have self-preference bias and drift. The research question is therefore not "how do we write a better improve-prompt" but **"under what acceptance architecture is iterative revision provably non-degrading while retaining genuine fixes?"**

**And the human anchor cannot close the gap either.** Preference data carries a *typicality bias* — annotators systematically favour familiar, fluent, predictable text (`docs/literature.md` §2). That is the mechanism behind post-training mode collapse, and it applies to any pairwise preference, human or model. So a preference judge is a valid anchor for **"is this fix an improvement?"** and is structurally blind to **"does this preserve what is distinctive?"** — the second question is about the *tails* the first question penalises.

The consequence is load-bearing for every arm in §8: **the preservation constraint (voice veto, bounded diffs, frozen spans) is not a workaround for unreliable judges. It remains necessary under a perfect human judge**, because no pairwise preference — at any level of reliability — measures distinctiveness. Plan accordingly: preservation is a separate mechanism from acceptance, and neither substitutes for the other.

---

## 3. Prior art and the gap (verified 2026-08-11 via web search)

The *disease* is documented; the *cure* is not.

| Work | What it shows | What it doesn't |
|---|---|---|
| [LLM as a Broken Telephone (ACL 2025)](https://aclanthology.org/2025.acl-long.371.pdf) | Iterative generation chains distort information; properties converge to equilibrium values | Not revision of a fixed text; no interventions |
| [Pride and Prejudice (self-bias)](https://arxiv.org/pdf/2402.11436) | Self-refinement amplifies the model's self-preference bias | Doesn't test external-gate architectures |
| [Voice Under Revision](https://arxiv.org/html/2604.22142v1) | Single-pass LLM rewriting measurably normalizes personal narrative voice; voice-preserving prompts only partially help | Single pass, not loop dynamics; prompts, not gates |
| [Homogenizing effect of LLMs on creative diversity](https://www.sciencedirect.com/science/article/pii/S294988212500091X) | Human-vs-LLM corpus homogenization at scale; prompt/parameter tweaks don't close the gap | Generation, not revision of existing prose |
| [Narrative Flattening](https://arxiv.org/pdf/2605.27878) | Post-training compresses thematic/stylistic variation in LLM fiction | Training-time, not inference-loop |
| [Mutation Without Variation](https://arxiv.org/pdf/2606.05408) | LLM-driven program evolution collapses in diversity | Code, not prose; no voice dimension |
| [Can AI writing be salvaged?](https://arxiv.org/html/2409.14509v2) | Catalogues LLM idiosyncrasies; expert edits improve alignment | Human editors, not automated gates |

**Gap we occupy:** a controlled, multi-round comparison of *loop architectures* on literary prose, with (a) planted defects giving objective fix-recall, (b) stylometric voice-preservation as a hard constraint, and (c) the judge's in-loop reports validated against blinded cross-family panels. Nobody has published the conditions under which the loop is *safe*.

---

## 4. Central hypothesis: the attractor model

A revision loop pulls text toward a **model-specific stylistic attractor** ("house style"). Degradation is not noise; it is convergence toward that attractor.

**This is the LLM instance of a known result, not a new conjecture.** Iterated-learning theory proves a chain of Bayesian learners, each learning from the previous one's output, converges to the learners' *prior* regardless of the starting data (`docs/literature.md` §1). An unconstrained revision loop is a single-agent chain. Two consequences: **A0 is a prior probe**, not only a control — its fixed point *measures* the house style; and the theory predicts drift toward the prior **cannot be prompted away, only structurally anchored**, which is what Phase 0/1 measured when prompt and sampler both failed and A2i's untouched-span anchor worked.

Falsifiable predictions:

- **H1 (homogenization):** under an unconstrained loop, mean pairwise stylometric distance between texts by *different* authors shrinks monotonically with round number.
- **H2 (position-dependence):** quality change depends on where the source starts relative to the attractor — deliberately weak drafts improve, distinctive strong prose degrades, and both end near the same endpoint.
- **H3 (model-specificity):** final-round texts cluster by *reviser model family*, not by source author (measurable: cluster purity of endpoint embeddings/stylometrics). **Sharpened by §1 of `docs/literature.md`:** the prediction is not merely that endpoints cluster by family but that they approach the reviser's *unconditional* prose. Cheap test — generate unconditional fiction from each reviser and check whether A0 endpoints are stylometrically closer to it than to their own round 0.
- **H4 (report–state divergence):** the in-loop judge's acceptance reports ("improved") diverge from blinded cross-family panel verdicts, and the divergence grows with round number.
- **H5 (the cure):** a loop with scoped edits + blinded cross-family accept gate with margin θ + stylometric voice veto ε + dismissal memory (arm A5, §8) reaches a fixed point voluntarily, fixes planted defects at near-control recall, and holds stylometric drift ≈ 0 with final blinded quality ≥ round 0.

---

## 5. Corpus design

~50 passages, 500–1500 words each, three strata:

- **Stratum A — distinctive human prose (public domain).** ~8–10 authors × 3–5 passages from Project Gutenberg. Mix famous, stylistically extreme voices (e.g. Hemingway-terse vs. Woolf-fluid vs. Dickens-ornate) with **obscure authors** of the same era as a contamination control (models may have memorized famous texts).
- **Stratum B — Stratum A with planted defects** (§6). Same passages, controlled flaws injected. This is the recall-measurement stratum.
- **Stratum C — deliberately mediocre drafts.** LLM-generated first-draft-quality fiction (generated with a weak model / low effort), for H2's "weak text improves" arm. Optionally add the author's own unpublished prose — the one text guaranteed absent from training data.

**Contamination controls:** (1) obscure-author inclusion; (2) memorization probe — prompt each reviser model to continue each source passage verbatim; passages with high continuation overlap get flagged and analysed separately; (3) Stratum C and own-prose are contamination-free by construction.

---

## 6. Planted-defect methodology

Each Stratum-B passage carries 3–6 injected defects from a fixed taxonomy, each recorded in a manifest (`defects.jsonl`: passage id, defect type, span, original text, corrupted text). Injection is scripted + hand-verified, so ground truth is exact.

Defect taxonomy (v0):

| Type | Example | Detection at scoring time |
|---|---|---|
| continuity contradiction | eye colour changes mid-passage | string/fact check against manifest |
| timeline slip | Monday → Tuesday → Monday | manifest check |
| name drift | "Katherine" → "Catherine" | exact match |
| tense slip | past → present for one sentence | manifest span check |
| POV slip | third-limited briefly head-hops | manifest span + judge assist |
| echo/repetition | same distinctive phrase 3× in 2 paragraphs | n-gram count |
| clunker | one bloated, tangled sentence inserted | manifest span; fixed = span substantially rewritten |

**Why this matters:** it converts the fitness question from pure aesthetics into partial ground truth. A gate that prevents degradation by blocking *all* edits scores zero recall and is correctly rejected. The tradeoff we actually care about — **defect-fix recall vs. voice preservation** — becomes a measurable frontier.

---

## 7. Metrics (per passage × per round)

All computed from saved round artifacts; M1–M3 and M6 are pure Python, zero API cost.

- **M1 Stylometric identity:** Burrows' Delta on function-word distributions; sentence-length distribution; punctuation profile; lexical diversity (MTLD). Plus an authorship classifier (logistic regression on writeprint features, trained on held-out passages of the same authors): does round-k text still attribute to its author?
- **M2 Homogenization:** mean pairwise inter-author stylometric + embedding distance per round (H1's direct measure).
- **M3 Slop index:** frequency per 1k words of a versioned LLM-tell lexicon (curated data file, cited sources) + formulaic-construction counts.
- **M4 Blinded quality:** position-randomized pairwise comparisons across rounds (round 0 vs k, k vs k+1), judged by a panel of ≥3 models from *different families than the reviser*; Bradley–Terry fit → quality-vs-round curve with CIs. Human spot-validation: ~100 blinded pairs judged by the project author; panel is trusted only where it agrees with the human subsample.
- **M5 Defect-fix recall/precision, decomposed:** report **detection recall** (was the defect *named*?) and **repair success** (did the edit clear it?) separately, never only their product. Three of the seven defect classes are *relational* — the evidence spans sentences — so a per-sentence editor cannot hold both halves of the contradiction in one decision and its ceiling on those classes is chance. A pooled number cannot distinguish that architectural cap from a repair failure, and A2i's 0.22 stayed uninterpretable for exactly that reason (`docs/literature.md` §14). Also report **spurious "fixes" to non-defective spans** — the precision proxy for overreach, and under a detect-then-repair arm the binding constraint outright, since a false complaint is a *licensed* unnecessary edit.
- **M6 Thrash:** sentence-level alignment across consecutive versions; fraction of round-k edits reverted or re-rewritten by round k+2; convergence = rounds-to-fixed-point (no accepted edits), if reached.
- **M7 Judge validity (H4):** per-round agreement between the in-loop judge's accept decisions and the blinded panel verdict on the same pairs; plotted against round number.

---

## 8. Intervention arms

Each arm runs to a 15-round cap or a fixed point (loop proposes no further accepted edits). Same reviser model and prompt scaffold across arms; only the acceptance architecture varies.

- **A0 — unconstrained (control):** "revise to improve" each round; always accept. The trap, characterized.
- **A1 — random-accept (control):** proposals as in A0, accepted at random with rate matched to A0's realized edit volume. Separates "editing at all" from "editing badly".
- **A2 — scoped:** every proposal must cite a specific finding (defect claim or named weakness) and stay within a bounded diff size. No global rewrites.
- **A3 — gated:** proposal accepted only if a blinded, position-swapped cross-family panel prefers it over the current version with margin > θ (majority + margin; θ swept in Phase 3).
- **A4 — gated + voice veto:** A3, plus reject any proposal moving the stylometric fingerprint > ε from the *round-0* baseline (ε swept).
- **A5 — full stack:** scoped + gated + voice veto + **dismissal memory** (a rejected proposal class is recorded and never re-proposed; re-running on unchanged text must yield zero new proposals — the idempotence requirement).
- **A2d — detect, then repair, then verify** *(added 2026-08-12; built for the genre stratum)*. Every arm above shares one shape: **revise, then gate**. The model is invited to improve a unit and something downstream decides how much to keep. Bounded diffs made the *keeping* safe and did nothing about the *inviting*, which is where the ~24:1 overreach comes from — a model asked to improve a paragraph will always find something to change, because that is what it was asked for. The inversion: **only a span with a located, checkable complaint against it is eligible to be edited; everything else is frozen.** Three stages — detect (compile the text into a state/fact ledger, find contradictions *in the ledger*, map back to spans), repair (one complaint, one span, A2i's machinery unchanged), verify (accept only if the cited complaint resolves and no new complaint appears). Two properties are structural rather than prompted: the model never sees the whole manuscript, so it *cannot* edit an unflagged span; and acceptance is mechanical, so no judge is on the critical path for any defect class a linter can state. Measured in [`docs/findings-litrpg.md`](docs/findings-litrpg.md). **Overreach under this design is bounded by detector precision, which is measurable, rather than by a model's restraint, which is not.**

**What the bounded arms give up, named:** sentence-indexed edits are structurally capped at *surface* and *microstructure* changes on the Faigley–Witte taxonomy, so A2i — and A2d more so — is a **novice-shaped editor by construction** (`docs/literature.md` §12). That is the right trade for a safety layer. It also means the downstream harness's ambitions eventually require macrostructure operations, which will need their own bounded representation (scene- or beat-level ops with their own vetoes), *not* an unbounding of A2i.

**Success criterion for H5 (A5):** defect-fix recall within ~10 points of A0; M1 drift and M3 slop flat across rounds; final M4 Bradley–Terry ≥ round 0; voluntary halting before the round cap on a majority of passages.

---

## 9. Phases, budgets, kill criteria

**Amended 2026-08-11 — this project runs entirely on local models. No paid API.** Revisers
and judge panels are open-weight models served by Ollama on the local RTX 4090, so the
dollar budgets below are superseded by GPU-hours and the API reviser families named in
Phases 0–1 are replaced by local ones. The local roster already spans five distinct
lineages — Google (gemma), Meta (llama), Microsoft (phi), OpenAI open-weight (gpt-oss) and
DeepSeek — which is what §11's "judges never from the reviser's family" rule actually
needs; newer tags can be pulled as required. A frontier model may be spot-checked through
the local Claude Code setup, which is subscription tooling already on the machine rather
than a metered endpoint. This is not only a cost decision: see the amendment to §12.5.

### Phase 0 — degradation replication (weekend-scale, a few GPU-hours)
10 Stratum-A passages × 10 unconstrained rounds × 1 cheap local reviser. Metrics M1–M3, M6 only (no judge costs). Deliverable: degradation curves + wall-clock and tokens-per-round calibration.
**Acceptance:** runner produces resumable JSONL provenance per round; metrics reproducible from artifacts alone.
**Kill criterion:** if A0 shows no M1/M3 degradation over 10 rounds, the trap needs re-characterization (longer horizons? stronger reviser?) before further work.

### Phase 1 — attractor characterization (~1 week of evenings)
Full Stratum A+C, 3 reviser families drawn from the local roster in the amendment above, chosen to be genuinely different lineages — H3 is a claim about *model family*, and two checkpoints of one family would not test it. Tests H1–H3. Deliverable: homogenization curves, endpoint clustering, the "watch Woolf and Hemingway collapse toward each other" figure. Standalone publishable/blog-worthy result.

### Phase 2 — judge validity
On Phase 0/1 artifacts: correlate every candidate gate signal (self-judge, cross-family judge, panel sizes, absolute vs pairwise, M1–M3 mechanical signals) against the blinded panel + human subsample. Tests H4. Deliverable: ranked signal-validity table → *empirically selects* the Phase 3 gate.

### Phase 3 — intervention tournament
Six arms × Stratum B (+A subset) × best-validated gate; sweep θ, ε coarsely. Tests H5. Deliverable: the recall-vs-preservation frontier per arm; validated (θ, ε); the winning architecture spec.

### Phase 4 — packaging
`revision-bench` public repo in the mirror-bench mould: pinned seeds, bootstrap CIs, one-command repro, CI badge, CITATION.cff, writeup. Thresholds and architecture doc handed to the book-harness project.

---

## 10. Related local projects (verified on disk 2026-08-11)

| Project | Path | Relationship |
|---|---|---|
| **MirrorBench** | `C:\DEV\MirrorBench` ([github](https://github.com/skulitom/mirror-bench)) | Methodological template: report–state agreement framing, provenance/seeds/CI discipline, plan.md structure, Apache-2.0. Reuse: repo skeleton, stats utilities (bootstrap CIs), local open-weight model setup (Gemma via uv/CUDA) for the third reviser family. |
| **BookAgentZoo** | `C:\DEV\BookAgentZoo` | The downstream product design this research de-risks: "compiler-like consistency checking" proposal, KB schema, consistency-agent spec. Its defect categories seed the §6 taxonomy. Phase 3's winning architecture becomes its edit-acceptance layer. |
| **ProseEvaluator** | `C:\DEV\ProseEvaluator` | Prior art of ours on cheap per-word prose scoring (Haiku) + RLAIF-style JSONL data collection. Reuse: scoring-prompt patterns, the JSONL logging shape; its word-level signal is a candidate M-metric / gate input in Phase 2. |
| **AgentUI** | `C:\DEV\AgentUI` ([github](https://github.com/skulitom/AgentUI)) | Human-eval surface: the ~100-pair blinded human judgments in M4 can run as an `ui_ask` A/B widget instead of a bespoke web app. |
| **SchemeStressProject (SX)** | `C:\DEV\SchemeStressProject` ([github](https://github.com/skulitom/SchemeStressProject)) | Philosophical ancestor: oracle-based differential testing, fixed-point convergence as a correctness signal. Revision-bench's "re-run on unchanged text ⇒ zero new proposals" idempotence check is the same fixed-point discipline. |
| **openclaw** (clone) | `C:\DEV\openclaw` | Reference for what to *avoid* operationally: always-on gateway daemons and wide integration surfaces. Revision-bench (and the eventual harness) stays batch/cron, local-first, one job. |

Portfolio through-line, worth one line in the eventual README: SX (checkable compilation) → mirror-bench (checkable introspection) → revision-bench (checkable revision).

---

## 11. Statistical discipline

- Pre-registered predictions: H1–H5 stated here, before data.
- Mixed-effects structure: rounds nested in passages nested in authors; report bootstrap CIs, not bare means.
- Judge hygiene: position randomization on every pairwise call; judges never from the reviser's family; panel calibrated against the human subsample before being trusted.
- Full provenance: every trial one JSONL line (passage id, round, arm, model+version, prompt hash, seed, raw output, metric values). Runs resumable; metrics recomputable from artifacts without API calls.
- Multiple comparisons: primary endpoints are H1 (M2 slope) and H5 (A5 frontier); everything else labelled exploratory.

---

## 12. Unverified assumptions (test, don't trust)

1. Ten rounds are enough to see degradation on strong prose. (Phase 0 kill criterion.)
2. Writeprint-feature stylometry is sensitive enough at 500–1500 words to serve as a veto. (Validate on held-out human passages: same-author vs cross-author separation.) **Partly answered 2026-08-11 — see `docs/findings-phase0.md` §2.** On the 10-passage Phase-0 corpus, separation is real but the feature families rank in an order the plan did not anticipate: punctuation rate AUC 0.88, function-word Burrows' Delta 0.72–0.82, sentence shape 0.56 (p = 0.22, indistinguishable from chance). So a veto is viable, but not on the family §7 M1 names first, and equal-weighting the families is worse than not weighting them at all. Still open: separation between a passage and a *revised version of itself*, which is what A4 actually needs and which nothing can measure until M0-c has run.
3. Cross-family panels are meaningfully less biased than self-judges. (Phase 2 measures this; do not assume.)
4. Scripted defect injection reads as natural enough that fixes require understanding, not pattern-matching the corruption. (Hand-verify a sample; have a model attempt to *locate* injections blind — if trivially detectable, soften the injector.)
5. ~~API model versions stay stable across a phase.~~ **Retired 2026-08-11 by the local-only amendment to §9, then partly reinstated the same day in a different form.** Ollama pins an exact digest per model tag and the weights sit on local disk, so a mid-phase *weight* bump cannot happen without someone doing it deliberately; record the digest alongside the tag on every row. **But the runtime is not pinned by that.** `gpt-oss:20b`, pulled twelve months ago and untouched since, no longer loads on Ollama 0.32.8 — it fails with `tensor "blk.0.ffn_down_exps.weight" size overflow`. The weights did not drift; the runtime's ability to read them did. So the assumption to carry forward is narrower and sharper: *the inference runtime stays able to load a given set of weights across a phase*. Mitigation: `ollama_version` is recorded in `ModelIdentity` on every row, and a model that stops loading mid-phase invalidates cross-round comparison exactly as an API bump would.
6. Gutenberg memorization doesn't dominate results. (§5 contamination controls; compare famous vs obscure strata.)

---

## 13. Proposed repo layout

Modules marked `(M0-c)` and later do not exist yet, deliberately: an absent module is
honest, a stub that returns a plausible zero is not. See `AGENTS.md` §2.5.

```
revision-bench/
  plan.md                  # this file
  AGENTS.md                # rules for anyone changing the code; the trap list
  pyproject.toml           # uv, python 3.13 (match MirrorBench toolchain)
  configs/
    corpus/phase0.yaml     # sources + anchors, source digests pinned
  data/
    corpus/passages/       # extracted passages + provenance (committed)
    function_words.yaml    # closed-class Delta feature pool, versioned
    slop_lexicon.yaml      # versioned, cited per group
    defects.jsonl          # planted-defect manifest             (Phase 3)
  revisionbench/
    config.py              # YAML load, strict key checks, config hashing
    provenance.py          # run stamps: git, packages, model digests
    records.py             # crash-safe JSONL + resume-by-key
    text.py                # tokenisation, sentence spans, punctuation classes
    corpus.py              # fetch, strip boilerplate, cut passages
    metrics/               # stats, stylometry, slop, thrash
    inject.py              # defect injector                      (Phase 3)
    loop.py                # revision loop runner (arms A0-A5)    (M0-c)
    gates.py               # acceptance architectures             (Phase 3)
    judge.py               # panel orchestration, position randomization (Phase 2)
  scripts/
    fetch_corpus.py        # the only script that touches the network
    validate_stylometry.py # answers §12.2
  docs/findings-phase0.md  # what has actually been measured
  results/                 # JSONL artifacts + figures, committed per phase
  tests/                   # offline, hermetic
```

## 14. Next actions (Phase 0, session-sized)

1. ~~**M0-a:** repo scaffold (uv project, config plumbing, JSONL provenance writer) + corpus fetcher for 10 passages (2 famous + 1 obscure author) with licence/provenance records.~~ **Done 2026-08-11.** Corpus: Woolf *Mrs. Dalloway* (4), Hemingway *The Sun Also Rises* (3), Richardson *Pointed Roofs* (3, obscure control); 901–1020 words each, byte-exact spans, source digests pinned, rebuilds offline.
2. ~~**M0-b:** stylometry + slop + thrash metrics with unit tests on synthetic fixtures (e.g. verify Delta separates two known authors; verify thrash detector on a hand-built revert sequence).~~ **Done 2026-08-11.** 216 tests. Both named acceptance checks pass, and the stylometry validation is written up in `docs/findings-phase0.md`. Deviation worth knowing: tests run against the *committed corpus*, not synthetic fixtures, because §12.2 asks whether the measurement works on real prose at real passage length and a synthetic fixture answers an easier question.
3. ~~**M0-c:** A0 loop runner against one cheap local reviser; run 10×10; plot M1/M2/M3/M6 curves; write up Phase 0 findings + measured wall-clock/tokens per round in `results/phase0/`.~~ **Done 2026-08-11.** 81 generations, 7.9 min, 5.9 s/round. Kill criterion **not tripped**: M1 and M3 both move immediately and substantially. Full write-up in `docs/findings-phase0.md` §6. Headline: under the literal A0 prompt this reviser is largely a *summariser* (0.53× length), and one clause about length cuts measured voice drift 5×; H1 (homogenization) is flat and not supported by a single-model run; 19/20 trajectories halt voluntarily by median round 3–5.

Settled and open items for M0-c:

- **Reviser: `gemma3:4b`** (Q4_K_M, pulled 2026-08-11), chosen over the 12–22-month-old
  cached tags so that a null result would be about revision loops rather than about model
  weakness. Google-family, which leaves Meta/Microsoft/OpenAI-open-weight/DeepSeek free for
  the §11 cross-family judge panels.
- **Reproducibility is about model warmth, not the sampler.** The first generation after a
  model load does not reproduce; every later one does, under both greedy and seeded
  sampling. `keep_alive` is pinned and the runner discards a warm-up generation. An earlier
  version of this bullet claimed `top_k: 1` was the fix, which was wrong — see
  `docs/findings-phase0.md` §5.1.
- **The length confound was the main design problem, and it was real.** Handled by running
  a length-preserving prompt variant alongside the neutral one, a runner-side length guard
  that flags without intervening, and reporting word count beside every metric. It
  accounted for most of the apparent voice degradation (§6.1).
- **The revise prompt** lives in `configs/phase0.yaml`, hashed, with its sha256 on every
  row.
- **Feed-forward confirmed:** round k+1's input is round k's output, asserted in
  `tests/test_loop.py::TestFeedForward`.

### Phase 1 in progress — and a re-ordering

`docs/findings-phase1.md` §2 compared three revision architectures on phi4. The bounded-diff
arm (**A2e**, model names find/replace spans; only unambiguous ones apply) is the first
mechanism in this project that controls length at all — ratio 1.00, voice drift 0.08 against
the unconstrained control's 0.99. But it applied only **25% of the edits it proposed** and
halted after 2.6 rounds, so its preservation is partly real and partly the trivial result of
changing very little. §6 already names that failure: a gate that blocks everything scores
zero recall and must be rejected on those grounds.

**Consequence: build Stratum B before the three-family sweep.** The recall-vs-preservation
frontier has one axis until planted defects exist, and running the most promising
architecture across more families would multiply a result nothing can falsify. M1-b and M1-c
are re-ordered accordingly.

### Carried into Phase 1 (from `docs/findings-phase0.md` §6.6–6.7)

- **Handle length in the runner, not the prompt.** Both revisers summarise under the
  literal A0 instruction (gemma3:4b to 0.53×, phi4 14.7B to 0.61×), so the confound is a
  property of the instruction rather than of small models. The length clause that fixed it
  on gemma3:4b (0.97×, 0/10 guard trips) largely fails on phi4 (0.69×, 6/10 trips), so
  prompt-level control is a per-model empirical claim and cannot be assumed. Phase 1 needs
  reject-and-retry on the guard band, or explicit statistical control for length.
- **Ten rounds is the wrong horizon** for a model that settles by round 5; spend the budget
  on passages and families instead, or perturb the loop to keep it moving.
- **Watch punctuation.** It carried the most author signal (§2) and produced the only
  non-length effect (§6.2): the dash-heavy author loses dashes and the dash-free author
  gains them, which is the §4 attractor in miniature.
- **Voluntary halting will not discriminate arms, and halting itself is model-dependent.**
  gemma3:4b's A0 halts on 9/10 passages; phi4's on 5/10, still thrashing at 0.4–0.6 through
  round 8. So §8's A5 halting criterion needs strengthening, and arms will differ by model
  as well as by architecture. The upside: M6 thrash *is* measurable on a model that keeps
  moving.
