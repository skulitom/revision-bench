# AGENTS.md — orientation for anyone changing this code

You are looking at a **measurement** repository. Its output is numbers that people may
cite. That changes what "working code" means here: a passing test suite is necessary and
nowhere near sufficient, because the defects that matter in this kind of project produce a
*plausible wrong number* rather than an error.

Read this file and [`plan.md`](plan.md) before changing anything.

---

## 1. What the project measures

What happens to prose when you put an LLM in a loop and ask it to keep improving the text.
Per revision round, per passage: stylometric identity, homogenization across authors, an
LLM-tell rate, thrash, and — later — blinded quality and defect-fix recall.

**Vocabulary discipline is load-bearing and enforced in review.** Every claim stays at the
level of *measured change per revision round*. Do not write, in code, docstrings, log
strings, commit messages or variable names, that AI editing is good or bad, that a text got
"better" or "worse", or that a model "understands" anything. The project's credibility
rests on the claim being exactly as strong as the measurement and no stronger. See
[`plan.md`](plan.md) §1.

## 2. Rules that are not negotiable

1. **Never invent an identifier, path, hash, URL or version.** If you need one and cannot
   verify it, leave a marked `TODO` and say so. Every Project Gutenberg ebook id in
   `configs/` was verified by fetching it. A fabricated constant here will not crash — it
   will produce a number.
2. **No paid API. Ever.** Revisers and judges are local models via Ollama on
   `http://localhost:11434`. This is a standing instruction from the project owner, and it
   also buys the research something: plan.md §12.5 worries about API model versions
   drifting mid-phase, and a pinned Ollama digest makes that worry go away. Record the
   model **digest**, not just the tag.
3. **Tests are offline and hermetic.** No network, no models, no API key. The one network
   path in the repo is `scripts/fetch_corpus.py`, and its output is committed so nothing
   else needs it. Anything that must hit the network goes behind the `network` marker and
   outside the default run.
4. **Config is the single source of truth.** Constants live in `configs/` and `data/`, not
   in code. If a script and its config disagree about anything, that is a bug.
5. **Scope discipline.** Work the current milestone. Do **not** create stub modules for
   future ones — an absent module is honest, a stub that returns a plausible zero is not.
   That is why there is no `gates.py`, `judge.py` or `inject.py` yet.

## 3. Verify before you write

```bash
uv run pytest -q
```

```bash
uv run ruff check && uv run ruff format --check
```

The corpus rebuilds offline from the committed cache, which is the cheapest check that the
"reproducible from artifacts alone" promise still holds:

```bash
uv run python scripts/fetch_corpus.py --config configs/corpus/phase0.yaml --offline
```

## 4. The traps — every one of these was live in this codebase

Each entry is a real defect caught while building M0-a/M0-b. They are listed because the
*shapes* recur, and because three of them would have produced a confident, publishable,
wrong figure.

**Bare `no` and `yes` in a YAML word list are booleans.** PyYAML implements YAML 1.1, where
`no`, `yes`, `on`, `off`, `y`, `n`, `true` and `false` resolve to booleans. Four of those
are function words. Unquoted, they left the Burrows' Delta feature set and were replaced by
the tokens `"true"` and `"false"`, which occur in no passage — so Delta would have been
computed over a feature set silently missing four of the commonest words in English.
Caught only because `load_function_words` rejects duplicates and both coerced to the same
string. It now rejects non-string entries by name.

**Project Gutenberg writes em dashes as `--`.** Mrs. Dalloway contains 545 `--` and 2 real
em dashes; Pointed Roofs, 749 and 2; The Sun Also Rises, 0 and 39. Counted literally, the
em dash — Woolf's most characteristic mark — reads as a *hyphen*, and her em-dash rate
reads as zero. It gets worse under revision: an LLM re-typesets `--` as `—` almost every
time, so round 1 shows a collapse in hyphens and an explosion in em dashes. That is a
large, clean, entirely artefactual "voice drift", pointing the direction plan.md §4
predicts drift should point. `revisionbench.text` folds dash runs for this reason.

**A generous abbreviation list merges sentences.** The first draft of the sentence splitter
listed `no`, `in`, `art`, `ed` and `sat` as abbreviations, so `"Oh, no. She's gone."`
became one sentence. No error — just a longer mean sentence length, which is a headline M1
number, with dialogue-heavy prose (the stratum most at risk of voice loss) hit hardest.
Ambiguous abbreviations now live in `NUMERIC_ABBREVIATIONS` and suppress a boundary only
before a number, so `"See No. 5"` and `"Oh, no."` both come out right.

**A wholesale sentence replacement is not a deletion.** The thrash detector originally
bucketed "edited at round k, replaced entirely at k+2" with plain deletions and excluded it
from the thrash fraction. That is the *most* churning thing a loop can do, and excluding it
made an unstable loop read as a settled one. It is now counted as `replaced`, separately
from a genuine cut, which is not thrash.

**Character names would have made Delta look brilliant.** The classic recipe for Burrows'
Delta is "the N most frequent words in the corpus". On a corpus of three novels, that list
contains *Miriam*, *Clarissa* and *Brett*. Delta would separate the authors almost
perfectly while measuring cast lists — and since a reviser rarely renames characters, those
features stay pinned across rounds, drift comes out reassuringly small, and the A4 voice
veto passes edits that had sanded the prose flat. The feature pool is a closed class, which
cannot contain a proper noun.

**Re-fitting the scaler each round would erase the finding.** `StyleModel` is fitted once,
on the round-0 corpus, and frozen. If the whole corpus drifts toward one house style,
re-standardising against the drifted corpus subtracts exactly the effect H1 predicts, and
homogenization comes out flat.

**"I ran it twice and got the same answer" is not evidence of determinism.** The first
generation after Ollama loads a model differs from every later one; warm calls are
byte-identical. It is the load path, not the sampler — the effect is the same size under
greedy decoding and under seeded sampling. The first probe of this missed it, concluded
`top_k: 1` was the fix, and that wrong rule was written into `ollama.py` as a hard guard
and into `configs/phase0.yaml` as its justification before a controlled test (unload, then
five calls) found the real cause. Ollama unloads an idle model after five minutes, so
untreated this sprinkles irreproducible rounds through any long sweep, indistinguishable in
the artifact from real loop dynamics. Hence `keep_alive` on every request and a discarded
warm-up generation. **A determinism control must include a cold start.**

**A resume key that omits the config hash splices incomparable runs.** `RESUME_KEY_FIELDS`
originally keyed on (passage, arm, prompt, model digest, round). Changing the sampler
changes every generation but changes neither the prompt nor the digest, so a resumed run
would have joined a temperature-0.8 tail to a greedy head with nothing in the artifact
marking the seam. The key now includes `config_hash`, which invalidates resume on any
config edit — the right trade, because 200 generations are cheap and a spliced trajectory
is undetectable.

**Local weights are immutable; the runtime that loads them is not.** `gpt-oss:20b` sat
untouched on disk for a year and now fails to load on Ollama 0.32.8
(`tensor "blk.0.ffn_down_exps.weight" size overflow`). Running locally removes *weight*
drift, which is why plan.md §12.5 was retired — but it does not remove runtime drift, so
§12.5 came back in a narrower form. `ollama_version` is on every row for this reason.

**A fabricated constant is still fabricated when it has a real prefix.** The `phi4` entry
in `configs/phase0_phi4.yaml` was first written by padding the 12-character short ID from
`ollama list` out to 64 characters. It looked entirely plausible. The digest check would
have caught it at run time, but the rule is rule 1 for a reason: get the value from the
API, never from a plausible-looking reconstruction.

**A CI step that cannot fail is worse than no CI step.** The first version of the corpus
check ran `fetch_corpus.py --dry-run` (which writes nothing) and then
`git diff --exit-code`, so it passed unconditionally while reading as a reproducibility
guarantee. If a check cannot fail, delete it or make it real.

### The shape they share

A wrong number that **looks like the result you expected**. Two of the above would have
manufactured evidence *for* the project's own hypothesis. That is the thing to be
suspicious of.

## 5. Invariants worth knowing

- **A passage is a byte-exact slice of its source book.** Extraction produces a character
  span and stores `body[start:end]`, never a rejoin of tokens. `source.sha256` plus the
  span reproduce it exactly.
- **Passages are located by anchor string, not offset.** Gutenberg re-issues files; an
  offset silently points elsewhere afterwards, an anchor fails loudly. Anchors must be
  unique in the book body, and `extract_passage` rejects them if not.
- **Spans index the text you passed in.** `normalise_newlines` changes length, so normalise
  once at the boundary and pass normalised text everywhere after.
- **Typographic folding is on by default and off on request.** Whether a reviser re-typesets
  its input is a real fact; it is just not evidence about voice.
- **Rates, not counts.** A reviser changes the length of what it revises, so a raw comma
  count moves for two reasons at once.
- **Degenerate text raises, it does not score.** A reviser that returns two sentences or an
  apology must be recorded as a metric failure, not as a passage with an interesting
  fingerprint. `MetricError` is the mechanism.
- **Resampling units are passages, not rounds.** Rounds within a passage are a dependent
  series by construction; bootstrapping over them returns an interval that is far too
  narrow.

## 6. Reporting results

- Report the **whole surface**, never a best cell. `scripts/validate_stylometry.py` prints
  every family at every N for this reason.
- Say what the sample was. Ten passages and three authors is a small sample and every
  interval on it is wide; the *ordering* of feature families is the finding, not the third
  decimal place.
- **Negative and inconvenient results ship with the same rigour.** The Phase-0 stylometry
  validation found that the feature family Burrows' Delta uses is *not* the one that
  separates these authors best. That is in `docs/findings-phase0.md` and pinned by a test.
- Never let a script print `PASS` on ambiguous evidence.
