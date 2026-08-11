# Owner direction note — 2026-08-12

Read after `AGENTS.md` and the two findings docs. This sets priority order and three
standing analytic rules; it does not change any rule in `AGENTS.md`.

## Priorities, in order

1. **The four-arm frontier over Stratum B** (A0, A2p, A2e, A2i) — the
   recall-vs-preservation plot is the project's central deliverable, and
   `findings-phase1.md` §2.2 already concluded it comes before the three-family sweep.
   Confirmed. A2i's recall of 1.00 is one passage with small denominators: nothing from a
   single passage feeds a threshold or a conclusion.

2. **Phase 2 (judge validity, M4/M7) immediately after — before any more
   mechanism-building.** Everything measured so far is movement, not quality; the project's
   original question lives in Phase 2 and is still untouched. Design it at **per-edit
   granularity**: under A2i the judge only ever answers "apply this scoped edit or not", so
   that is the question to validate the judge on — whole-passage pairs are a secondary
   condition, not the primary one. The ~100-pair blinded human subsample is the calibration
   anchor; the panel is trusted only where it agrees with it.

3. **H1/H3 (attractor, three-family sweep) waits** until 1 and 2 are done. It is the
   scientifically interesting claim and the least product-relevant one; it must not consume
   budget ahead of the frontier and the judge study.

## Standing analytic rules

- **Every preservation number is reported beside apply rate and edit volume.** A2e's
  lesson — preservation by inaction — is a class, not an instance. The frontier plot must
  encode edit volume (point size or annotation) so an arm that changes nothing cannot read
  as a winner.
- **Model-dependence hedge, within the no-API rule:** length compliance and halting have
  both already proven model-dependent, so before any (θ, ε) threshold is handed to the
  downstream harness, replicate the key arms on at least one larger local model and one
  additional family — e.g. `gemma3:27b` (size, within family) and a Qwen (third family),
  digests pinned as usual. The no-paid-API rule stands. If a frontier-API replication is
  ever judged necessary for product transfer, that is an owner decision and an `AGENTS.md`
  amendment — not something a session may decide.
- **Keep writing the corrections sections** (`findings-phase0.md` §6.5-style). They are the
  most valuable part of the documentation and the reason the numbers are believable.

## Note lifecycle and repo hygiene

Owner notes arrive as numbered files at root (`NEXT.md`, `NEXT_2.md`, …). They are input,
not documentation, and they must not accumulate:

- When a session has absorbed a note — acted on it, with its durable content moved into
  `plan.md`, a findings doc, or code — move the file to `legacy/` with a one-line header
  saying what absorbed it and when. Do not delete (provenance), do not leave at root
  (clutter), do not absorb silently (the header is the audit trail).
- Same standard for code and data: a mechanism retired by evidence is either kept
  deliberately — off by default and documented, like `length_policy: retry` — or deleted in
  the commit that retires it. Never left ambiguous. Current sweep candidates: the two
  curated slop-lexicon groups that have never fired on anything (remove in lexicon v2 if
  they are still silent after the Stratum-B frontier run).
