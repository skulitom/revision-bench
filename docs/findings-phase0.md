# Phase 0 findings

Status: **M0-a and M0-b complete. M0-c (the A0 loop run) not yet started**, so there are no
degradation curves here yet — nothing in this document involves a language model. What
follows is the corpus baseline and the validation of the measuring instruments themselves.

Reproduce with:

```bash
uv run python scripts/validate_stylometry.py
```

---

## 1. The corpus

10 Stratum-A passages, 901–1020 words each, cut from three books that are public domain in
the United States.

| author | fame | source | passages | mean sentence length | semicolons /1k | em dashes /1k | MTLD |
|---|---|---|---|---|---|---|---|
| Ernest Hemingway | famous | *The Sun Also Rises* (PG 67138) | 3 | 8.7–22.9 | 0.0–1.1 | 0.0–1.1 | 44–67 |
| Virginia Woolf | famous | *Mrs. Dalloway* (PG 71865) | 4 | 18.9–31.3 | 18.7–42.2 | 4.2–14.3 | 52–117 |
| Dorothy M. Richardson | obscure | *Pointed Roofs* (PG 3019) | 3 | 14.5–21.6 | 0.0–2.0 | 5.5–24.5 | 90–97 |

The strata are doing three jobs (see `configs/corpus/phase0.yaml` for the full reasoning):
Woolf and Hemingway are the stylistic poles plan.md §5 names, Richardson is the
contamination control — same decade, same stream-of-consciousness technique as Woolf, but
658 Gutenberg downloads against Woolf's 30,616 — and Richardson doubles as a second
"fluid" voice so that H1 has both a near pair and a far pair to close.

The descriptive numbers above are already a sanity check on the instruments: Hemingway's
semicolon rate is ~0 and Woolf's is 18–42 per thousand words, which is the difference
anyone who has read both would predict.

## 2. Does the stylometry separate authors? (plan.md §12, assumption 2)

plan.md lists as *unverified* that "writeprint-feature stylometry is sensitive enough at
500–1500 words to serve as a veto". It is not a safe assumption, and the answer depends
sharply on which features you use.

Readout: AUC = P(a different-author pair is more distant than a same-author pair), over 12
same-author and 33 different-author pairs. Significance by permutation of author labels
(2000 permutations, seed 0); p = 0.0005 is the floor of that test.

| feature family | AUC | permutation p | what it is |
|---|---|---|---|
| **punctuation** | **0.884** | 0.0010 | rate per 1k words of 13 punctuation classes |
| all features, unweighted | 0.838–0.861 | 0.0010 | dominated by function words, ~100 of ~120 features |
| function words (Burrows' Delta) | 0.715–0.816 | 0.003–0.009 | rises with N: 0.715 at N=20, 0.816 at N=150 |
| family-balanced | 0.760–0.775 | 0.004 | equal weight per family |
| **sentence shape** | **0.561** | **0.223** | mean/median/sd/p10/p90/density |

### Three things follow

**Punctuation is the strongest author signal in this corpus, and it is not what Burrows'
Delta uses.** Function-word Delta — the standard instrument, and the one plan.md §7 M1
names first — is beaten by a 13-dimensional punctuation profile at every N tested. plan.md
§8's A4 voice veto has to be built on whichever family actually carries signal, so this is
a design input, not a curiosity.

**Sentence shape alone is indistinguishable from chance** (p = 0.22). It should not be
given a vote of its own. Note the tension with §1's table, where sentence length obviously
separates Hemingway from Woolf: it does, but *within*-author variation is just as large
(Hemingway's own passages run 8.7 to 22.9), so it cannot classify a single passage. Both
facts are true and only the second one matters for a veto.

**Equal weighting per family is worse than no weighting at all.** The family-balanced
distance (0.76) underperforms both the unweighted mean (0.85) and punctuation alone (0.88),
because it gives a full quarter of the weight to the family that carries no signal. If a
composite fingerprint is used for A4, the weights have to be fitted, not assumed.

### What this cannot tell you

Ten passages, three authors, and a convenience sample of what is on Project Gutenberg.
Every cell above is wide — read the ordering of the families, not the third decimal place.

More important: this measures separation **between different authors**, which is a proxy
for what A4 actually needs — separation between a passage and a *revised version of
itself*. Those are related but not the same question, and the second cannot be answered
until there are revision rounds to measure. Treat the ordering as a prior for M0-c, and
re-run this against round-k texts once they exist.

## 3. Slop-index baseline

Mean **0.54** lexicon hits per 1000 words across the 10 passages; range 0.00–2.20.

| group | source | hits on human prose |
|---|---|---|
| `chakrabarty2024_fiction_phrases` | Chakrabarty, Laban & Wu, CHI 2025 | 5 |
| `curated_project_fiction_cliche` | `curated:project` (unvalidated) | 0 |
| `curated_project_expository_register` | `curated:project` (unvalidated) | 0 |

Exactly one term fires on 1920s literary fiction at all: `sense of`, five times. Both
project-curated groups sit at zero.

That is the result you want from a baseline. The metric is not saturated, so a rise across
revision rounds has room to be visible and will be attributable to specific terms rather
than asserted in aggregate. It also means the curated groups are, so far, **untested** —
they have never fired on anything. plan.md §12 governs: if they never fire in M0-c either,
they should be removed in lexicon version 2 rather than quietly retained.

## 4. Instrument checks that are pinned by tests

- Burrows' Delta places the Hemingway–Woolf pair as the most distant author pair, as the
  corpus design predicts.
- The sentence splitter handles 16 constructions that break a naive `[.!?]` split,
  including `"Stop!" — he said.` (one sentence), `It was I. She knew it.` (two), and
  `See No. 5` vs `Oh, no.`
- MTLD raises rather than returning `inf` on degenerate input, and is stable under halving
  a passage.
- The thrash detector distinguishes revert / re-rewrite / replace / settled / cut on a
  hand-built four-round sequence, and reports `None` rather than 0.0 when a round made no
  edits — 0.0 would read as "stable" when the truth is "already stopped".

## 5. M0-c preflight: two measured facts about the reviser

Not results — a single probe call, one passage, one model (`gemma3:4b`, Q4_K_M, via Ollama
on the local 4090). Recorded here because both change how M0-c must be built.

### 5.1 The first generation after a model load is not reproducible. Every later one is.

**This section previously reported the wrong cause.** An earlier probe found two calls at
`seed=0, temperature=0` differing, and three calls with `top_k: 1` agreeing, and concluded
that greedy decoding was the fix. That was an artifact of call ordering: the three
"deterministic" calls all happened to be warm, and the two that disagreed straddled a model
load. The rule was written into `ollama.py` as a hard guard and into `configs/phase0.yaml`
as its sampler justification. Both have been corrected.

Controlled test — model unloaded from VRAM first, then five calls in one process:

| sampler | call 1 (cold) | calls 2–5 (warm) |
|---|---|---|
| greedy (`temp 0, top_k 1`) | 276 words | 291, 291, 291, 291 — byte-identical |
| sampled (`temp 0.8, top_k 40, seed 0`) | 286 words | 384, 384, 384, 384 — byte-identical |

The effect is the same size in both, so it is not a property of the sampler; it is the
load path. Floating-point work is batched differently on the first forward pass after a
load, the logits differ in the last bits, and that is occasionally enough to change a
token — after which the two trajectories diverge for good.

**Why it matters more than it looks.** Ollama unloads an idle model after five minutes by
default. A 200-generation sweep with any pause in it therefore acquires a scattering of
cold rounds, irreproducible and — in the artifact — indistinguishable from a genuine change
in what the loop is doing. Two mitigations are now in place: `keep_alive: 60m` is sent on
every request, and the runner spends one throwaway generation before the first scored round.

The correction generalises past this repo: **"I ran it twice and got the same answer" is
not evidence of determinism if both runs were warm.** The control has to include a load.

Second-order consequence, worth stating because it was nearly missed: this changed the
sampler, which changes every generation, but changes neither the prompt nor the model
digest. The resume key did not include the config hash, so a resumed run would have
stitched a temperature-0.8 tail onto a greedy head with nothing in the artifact showing the
join. `RESUME_KEY_FIELDS` now includes `config_hash`.

### 5.2 The reviser compresses hard, and that is a confound, not a finding

One unconstrained revision round on `woolf-01`:

| | round 0 | round 1 |
|---|---|---|
| words | 901 | 389 |
| semicolons /1k | 42.2 | 2.7 |
| em dashes /1k | 13.3 | 2.7 |
| sentence-length sd | 39.8 | 8.9 |
| slop /1k | 0.00 | 5.33 |

Every one of those moves in the direction plan.md §4 predicts. **Do not read it as
confirmation.** The model dropped 57% of the passage, and `done_reason` was `stop` with
`num_predict` at 2048 — so this is the model genuinely choosing to stop, not a client-side
truncation, but it does mean the round is a *summarisation* rather than a revision. A
metric computed across a 901-word text and a 389-word one is largely measuring the length
difference: fewer words means fewer chances to use a semicolon, and a shorter text has a
narrower sentence-length spread almost mechanically.

So M0-c has to separate "the loop flattens voice" from "the loop shortens text", and the
options are not equivalent:

- Instruct length preservation in the revise prompt, and record compliance rather than
  assume it. The prompt is the experiment's main free variable and belongs in `configs/`,
  hashed, with its wording on every trial row.
- Add a length guard to the runner, flagging any round whose word count moves beyond a
  configured band, so a collapsed round is recorded as a distinct event rather than
  averaged into a curve.
- Report length alongside every M1/M3 number, permanently. A degradation curve without a
  word-count curve beside it is not interpretable.

## 6. M0-c: the A0 unconstrained loop, 10 passages × 10 rounds

Run: `gemma3:4b` (Q4_K_M, digest `a2af6cc3eb7f`), seed 0, temperature 0.8, top_k 40,
`num_ctx` 8192. 81 generations, 7.9 minutes wall clock, 5.9 s/round, zero truncated.
Two prompt variants, differing by one clause about length and nothing else.

```bash
uv run python scripts/phase0.py --config configs/phase0.yaml
uv run python scripts/phase0_metrics.py && uv run python scripts/phase0_plots.py
```

Figure: [`results/phase0/phase0_curves.png`](../results/phase0/phase0_curves.png).

**A word about "degradation".** Everything below is *measured movement away from round 0
on mechanical readouts*. No blinded quality judgment exists yet (M4 is Phase 2), so
nothing here says the text got worse. A reviser that fixed ten real defects and one that
sanded off a voice would both show up as movement. That distinction is the entire point of
Phase 2, and it is not available here.

### 6.1 Headline: most of the "degradation" was compression

| | round 0 | neutral, settled | length_preserving, settled |
|---|---|---|---|
| word count | 933 | **492 (0.53×)** | 905 (0.97×) |
| voice drift from round 0 (mean \|Δz\|) | 0 | **0.99** | **0.20** |
| punctuation drift | 0 | 0.89 | 0.18 |
| slop / 1000 words | 0.54 | 3.73 | 1.26 |
| length guard tripped | 0/10 | 9/10 | **0/10** |

One clause — *"Keep the revision close to the original length of {word_count} words; this
is a revision, not a summary"* — held length at 0.97× and cut measured voice drift by
**5×**. It says nothing about voice, style, events or punctuation.

So the honest reading of the neutral arm is: **this reviser is mostly a summariser, and
most of what an unconstrained loop appears to do to voice is the second-order consequence
of it halving the text.** Had Phase 0 run only the neutral prompt — the literal reading of
plan.md §8's A0 — it would have produced a large, clean, confidently wrong result.

### 6.2 What survives length preservation

Not everything is compression. Per author, round 0 → settled:

| author | prompt | semicolons /1k | em dashes /1k | mean sentence length |
|---|---|---|---|---|
| Woolf | neutral | **26.5 → 1.4** | 9.8 → 9.8 | 24.6 → 20.4 |
| Woolf | length_preserving | 26.5 → 23.5 | **9.8 → 2.9** | 24.6 → 23.0 |
| Hemingway | neutral | 0.4 → 0.8 | **0.4 → 4.9** | 13.6 → 12.6 |
| Richardson | length_preserving | 0.7 → 0.9 | 14.8 → 11.7 | 17.4 → 18.0 |

Two things stand out. Woolf's semicolon rate — her single most characteristic mark —
falls **95%** under the neutral prompt and survives almost intact when length is held. But
the **em dash is lost either way**: Woolf 9.8 → 2.9 even at 0.96× length. And Hemingway,
who uses essentially no dashes, *gains* them (0.4 → 4.9) under the neutral prompt.

That pattern — the dash-heavy author losing dashes and the dash-free author gaining them —
is the attractor plan.md §4 describes, visible at the punctuation level. It is also the
one effect here that is not explained by length.

### 6.3 H1 (homogenization) is not supported

plan.md §4's H1 predicts mean pairwise cross-author distance shrinks monotonically with
round number. It does not:

| round | 0 | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| neutral | 1.161 | 1.190 | 1.138 | 1.139 | 1.139 | 1.141 |
| length_preserving | 1.161 | 1.130 | 1.126 | 1.126 | 1.125 | 1.125 |

A ~2–3% drop at round 1–2, then flat. Not monotonic, and small next to the round-0 spread.
The M2 panel is drawn with its axis anchored at zero for this reason: autoscaled it spans
1.125–1.19 and a 3% wobble reads as a collapse, on the panel carrying a primary endpoint.

Read this as *not yet supported* rather than *refuted*. Three authors and one reviser
cannot test a claim about convergence toward a **model-specific** attractor — that needs
the multiple reviser families of Phase 1 (H3), where the prediction is that endpoints
cluster by reviser rather than by author. What Phase 0 does establish is that H1 is not
going to fall out of a single-model run for free.

**This number was wrong twice before it was right**, both times in the direction of
appearing to support H1. See §6.5.

### 6.4 The loop stops, quickly

19 of 20 trajectories reached a fixed point — the model returning its input unchanged —
and stopped there: median round **3** under `length_preserving`, **5** under `neutral`.
Only `woolf-03`/neutral was still moving at round 10.

This matters more than it sounds for plan.md §8, which lists "voluntary halting before the
round cap" as a success criterion for the A5 full stack. On this model, **the unconstrained
control halts by itself on 95% of passages**, so halting alone will not discriminate
between arms and A5 will need a stronger criterion.

It also means M6 thrash is nearly unmeasurable here: edit volume drops to ~0 within three
rounds, and the thrash fractions in the figure after round 3 are computed over a handful
of live trajectories and swing between 0 and 1 on single-sentence changes. Do not read
them.

### 6.5 Three corrections made while producing this section

Kept because the *shapes* recur, and each was caught after producing a plausible number.

1. **Survivorship.** Settled trajectories stop emitting rows, so a round-10 mean was taken
   over only the passages still moving — the pathological ones. Uncorrected, round 10 read
   `slop = 10.87` from a single passage; corrected by carrying settled trajectories forward
   (valid, since a fixed point is absorbing under a reproducible sampler), it is `3.73`
   over all ten. The uncorrected version looked like a runaway degradation curve.
2. **The same bias in M2, separately.** Fixing the per-round means did not fix M2, which
   reads texts rather than metric rows. Uncorrected it showed cross-author distance rising
   to 1.36 at round 6 over four surviving passages; corrected it is flat at 1.14. H1
   predicts a *fall*, so this one would have been reported as a null either way — but the
   next arm to be run might not be so lucky.
3. **An axis that manufactured an effect.** See §6.3.

### 6.6 Verdict on the plan.md §9 kill criterion

> *If A0 shows no M1/M3 degradation over 10 rounds, the trap needs re-characterisation
> before further work.*

**Not tripped.** Both M1 and M3 move substantially and immediately: voice drift 0 → 0.99
and slop 0.54 → 3.73 under the neutral prompt, and 0 → 0.20 / 0.54 → 1.26 even when length
is held. Phase 1 is justified.

Three amendments it forces on the Phase-1 design:

- **Length must be controlled, not observed.** Run the length-preserving prompt as the
  primary arm and treat the neutral one as a second condition, or Phase 1 measures
  summarisation across three model families.
- **Ten rounds is the wrong horizon for this model.** Nearly everything happens at round 1
  and the loop is settled by round 5. Either accept a short horizon and spend the budget on
  passages and families instead, or perturb the loop to keep it moving.
- **Punctuation is the channel to watch.** It carried the most author signal in §2 and it
  is where the one non-length effect showed up in §6.2.

## 6.7 Reviser-capability check: is the compression a small-model artifact?

The obvious objection to §6.1 is that a 4B model summarises because it is small, in which
case the length confound is an artifact of a cheap Phase-0 reviser and not something later
phases need to design around. Checked by re-running the identical sweep — same corpus, same
passages, byte-identical prompts (same sha256 on every row), same sampling — against
**phi4 14.7B**, a 3.4× larger model from a different family. 237 generations total, 39
minutes, zero truncated.

`gpt-oss:20b` was the intended comparison and could not be used: the copy on disk no longer
loads on Ollama 0.32.8 (`tensor "blk.0.ffn_down_exps.weight" size overflow`). See §6.8.

Settled values (round 10, survivorship-corrected):

| | gemma3:4b | phi4 14.7B |
|---|---|---|
| **neutral** — length ratio | 0.53 | **0.61** |
| **neutral** — voice drift | 0.99 | 0.91 |
| **neutral** — slop /1k | 3.73 | 2.39 |
| **neutral** — guard tripped | 9/10 | 8/10 |
| **length_preserving** — length ratio | **0.97** | **0.69** |
| **length_preserving** — voice drift | **0.20** | **0.82** |
| **length_preserving** — guard tripped | **0/10** | **6/10** |
| fixed points reached (neutral) | 9/10 | 5/10 |

### Answer: no, and the follow-up is worse

**The compression is not a small-model artifact.** Under the neutral prompt the 14.7B
model compresses to 0.61× against the 4B model's 0.53× — same behaviour, marginally less of
it, across a 3.4× size gap and two different model families. Asked to "improve" literary
prose with no further constraint, both models summarise it. Treat that as a property of the
instruction, not of the reviser.

**But the mitigation from §6.1 does not transfer.** The length clause held gemma3:4b at
0.97× and 0/10 guard trips. On phi4 it reaches only 0.69× and trips the guard on 6 of 10
passages, and measured voice drift stays at 0.82 rather than falling to 0.20. The prompt
that looked like a clean fix for the confound is **model-dependent, and it largely fails on
the larger model.**

That is the more consequential result. §6.6 recommended running the length-preserving
prompt as Phase 1's primary arm; on this evidence that recommendation is not safe on its
own, because "the prompt controls length" is itself a per-model empirical claim. Phase 1
needs length handled at the *runner* level — reject-and-retry a round that leaves the guard
band, or record the violation and control for length statistically — with prompt-level
instruction as a supplement rather than the mechanism.

**Two smaller observations**, both exploratory:

- **phi4 does not settle.** Only 5/10 trajectories reached a fixed point under the neutral
  prompt against gemma3:4b's 9/10, and phi4's thrash fractions sit around 0.4–0.6 through
  round 8 rather than collapsing to noise by round 3. So M6 *is* measurable — on a model
  that keeps moving. §6.4's "the loop halts by itself" is a gemma3:4b property, not a
  general one, which strengthens rather than weakens the §6.4 warning about A5's halting
  criterion: the arms will differ by model as well as by architecture.
- **A hint of H1 on the larger model.** phi4/neutral's cross-author distance declines
  monotonically-ish from 1.16 to 1.07 (~8%) where gemma3:4b's was flat at 1.14. That is the
  direction H1 predicts, it is small, it is one model, and it is uncorrected for the fact
  that phi4 also kept editing for longer. Not a result — a reason to keep M2 as a Phase-1
  primary endpoint.

## 6.8 A qualification to retiring plan.md §12.5

This project retired §12.5 ("API model versions stay stable across a phase") on the grounds
that local weights cannot drift. That is true and the retirement stands for *weights*. It is
not the whole risk.

`gpt-oss:20b` was pulled twelve months ago and has sat untouched since. It no longer loads:

```
tensor "blk.0.ffn_down_exps.weight" size overflow
```

The weights are byte-identical to what was pulled; Ollama 0.32.8's ability to read that
MXFP4 MoE build is what changed. So the assumption that actually needs carrying forward is
narrower and sharper than the original: **the inference runtime stays able to load a given
set of weights across a phase.** A model that stops loading mid-phase invalidates
cross-round comparison exactly as an API version bump would. `ollama_version` is recorded in
`ModelIdentity` on every row for this reason, and plan.md §12.5 has been amended rather than
left retired.

## 7. What is still unmeasured

- **Whether any of this is a quality change.** No blinded panel exists yet (M4/M7,
  Phase 2). "Moved away from round 0" is all that has been measured.
- **Whether the effects are model-specific.** One reviser, one family (H3 needs Phase 1).
- **Whether stylometry can gate a *revision*.** §2 measured separation between different
  authors; A4 needs separation between a passage and its own revision. The round-0-to-round-k
  deltas in §6.1 are the first data on that and have not been analysed for it.
- **Defect-fix recall.** Stratum B does not exist yet, so nothing here says whether a loop
  that preserves voice also fixes anything. That trade-off is the whole point of the
  project, and Phase 0 does not touch it.
