> **Retired 2026-08-12.** Absorbed into `docs/literature.md` (consolidated, with a
> status against each claim), plus targeted edits to `plan.md` sections 2, 4, 7 and 8,
> `docs/findings-phase2.md` section 14 and `docs/findings-litrpg.md` section 8.
> Kept for provenance per the note lifecycle in `NEXT.md`.

# NEXT_4 — external research notes: the architecture beyond A2i (2026-08-12)

Owner-initiated literature sweep, prompted by the four-arm frontier result (A2i recall
0.22 vs A0's 0.73 at drift 0.31 vs 0.96). Same rules as NEXT_2/NEXT_3: these are the
literature's claims until reproduced here; absorb, then move to `legacy/`.

---

## O. Why A2i's recall is structurally capped — and it is not a tuning problem

Three of the seven defect classes in `data/corpus/defects.jsonl` — continuity
contradiction, timeline slip, name drift — are **relational**: the evidence for the defect
spans sentences. A per-sentence editor never has both halves of the contradiction in one
decision, so its ceiling on those classes is chance. The external evidence says even
seeing everything is not enough:
[FlawedFictions](https://arxiv.org/pdf/2504.11900) finds SOTA LLMs struggle at plot-hole
*detection* with full context, degrading with story length, and long-form work attributes
this to the lack of explicit state tracking
([Lost in Stories](https://gist.science/paper/2603.05890)). Detection — not repair — is
the hard half, and the current arms ask for both in one breath and measure only their
product. **M5 should be decomposed: detection recall (was the defect *named*?) versus
repair success (did the edit clear it?), or the 0.22 stays uninterpretable.**

## P. The literature-backed successor: detect–repair–verify (A3d)

Design-space axis 7 ("diagnose, then fix") is independently supported from three
directions, and composes with everything already built:

1. **Detect, globally.** A dedicated pass over the whole passage per defect class,
   returning findings as `{sentence indices, claim, evidence}` under an enforced schema.
   The strong form, from [SCORE](https://arxiv.org/html/2608.08160)-style state tracking
   and BookAgentZoo's KB design (they converge): first *compile* the passage into a
   state/fact ledger (entities, attributes, times), detect contradictions in the ledger —
   where the two halves of a relational defect finally sit in one structure — then map
   back to sentences. Methods and metrics in
   [document inconsistency detection](https://arxiv.org/pdf/2601.02627).
2. **Repair, locally.** Per finding, an indexed edit restricted to the cited sentences —
   the A2i/A2f machinery survives unchanged as the repair stage, vetoes and feedback
   retry included.
3. **Verify, per edit.** Generate k candidate repairs per finding; keep the *smallest*
   candidate that makes the finding no longer trigger (re-run detection on the patched
   region as an outcome check) and passes the mechanical vetoes. Edit-level best-of-N is
   live where round-level re-rolling was dead: candidates for a named fix vary; whole-round
   lengths did not. The reasoning literature's process-vs-outcome result
   ([PRM > ORM](https://arxiv.org/pdf/2505.02686);
   [ReST-MCTS*](https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf))
   maps onto editing as: score steps (edits), not trajectories (rounds).

**Falsifiable prediction:** A3d raises recall specifically on the relational classes
(continuity, timeline, name drift) while echo/clunker recall stays near A2i's, and drift
stays in the A2i band because the repair representation is unchanged. If recall does not
move on the relational classes, the detection stage — not the architecture — is the
bottleneck, and that is measurable separately (§O).

Cost note: detection adds ~1 generation per class per round, so A3d is ~3–5x A2i's
generation count. Run it detection-classes-first (continuity+timeline+echo) before the
full taxonomy.

## Q. A second defect corpus exists, contamination-robust

[FlawedFictions](https://arxiv.org/pdf/2504.11900) *controllably synthesizes plot holes in
human-written stories* — independently, the same move as Stratum B, with an algorithm
designed to resist contamination. Worth checking its license and whether its synthesizer
or items can augment `data/corpus/defects.jsonl` with defect classes the current injector
lacks (plot-level holes rather than surface corruptions). It also supplies external
validation language for the planted-defect methodology in the eventual writeup.

## R. Farther out: editors trained on real edits, and tighter edit representations

- [PEER](https://arxiv.org/abs/2208.11663) (Meta, 2022) trained on Wikipedia/Wikinews/
  StackExchange *edit histories*: plan → edit → explain, i.e. cite-the-finding as a
  trained behavior. An editor whose prior is "human edit" rather than "rewrite" attacks
  the compression problem at its source. Open question: whether a small local model
  fine-tuned on edit traces (PEER's data recipe) beats prompting a 14B into edit-shaped
  behavior — a Phase-4+ question, not a current one.
- The tagger lineage — [LaserTagger](https://www.aclweb.org/anthology/D19-1510/),
  [Felix](https://arxiv.org/pdf/2003.10687),
  [EdiT5](https://www.semanticscholar.org/paper/8ba66cb690ff3a37c63ff0f67b595f03dd78dc75),
  [G-SPEED](https://arxiv.org/html/2310.10480) — predicts keep/delete/insert operations
  per token: boundedness below the sentence, plus interpretability and speed. A reminder
  that the design space continues below A2i's unit, if sentence-level repair ever proves
  too coarse.
