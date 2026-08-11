# results/

Committed artifacts, one directory per phase. Everything here is regenerable from the
code, the configs and `data/`; it is committed anyway so that a reader can check a claim
without running anything, and so that a change to a number shows up in a diff.

Every file carries a `provenance` block: git sha and dirty flag, package versions, Python
version, platform, and the identities of whatever produced it. A number without that block
is not a result, it is a note.

## phase0/

| file | produced by | what it is |
|---|---|---|
| `stylometry_validation.json` | `scripts/validate_stylometry.py` | Same-author vs different-author separation per feature family, with permutation p-values. Answers plan.md §12 assumption 2. Written up in [`docs/findings-phase0.md`](../docs/findings-phase0.md) §2. |
| `rounds.jsonl` | `scripts/phase0.py` | One row per (model, prompt, passage, round): the round's **text** plus generation facts (tokens, wall clock, `done_reason`, length ratio, guard flag, stop reason). Deliberately carries **no metric values** — see below. Holds two revisers, `gemma3:4b` and `phi4:latest`; rows are separated by `config_hash` and `model_digest`, and the metrics pass groups on `run = "{model_tag} \| {prompt_name}"`. |
| `metrics.jsonl` | `scripts/phase0_metrics.py` | One row per (passage, prompt, round) of M1/M3/M6 readouts, recomputed from `rounds.jsonl` alone. Rows with `carried_forward: true` are settled trajectories extended to the round cap. |
| `metrics_summary.json` | `scripts/phase0_metrics.py` | Per-prompt, per-round aggregates including M2 (a cross-passage statistic, so it exists only here) and rounds-to-fixed-point. |
| `phase0_curves.png` | `scripts/phase0_plots.py` | The six degradation panels. |

**Why the text and the metrics are in separate files.** plan.md §9 makes "metrics
reproducible from artifacts alone" Phase 0's acceptance criterion. If the numbers lived in
the file the model produced, "recorded" and "recomputed" could drift apart with nothing to
notice. As it stands, `phase0_metrics.py` contacts no model and touches no network, and a
metric change costs a re-read rather than 81 GPU generations.

Not here yet: anything involving a judge (M4, M7 — Phase 2) or planted defects (M5 —
Phase 3).

## Reading a result

Two rules, both from plan.md §11:

1. **The whole surface, never a best cell.** `stylometry_validation.json` has 20 rows on
   purpose. A maximum picked over 20 cells after the fact is a multiple-comparisons
   artefact wearing a point estimate's clothes.
2. **Check the sample size before the third decimal place.** Phase 0 is 10 passages and 3
   authors. The orderings are informative; the exact values are not.
