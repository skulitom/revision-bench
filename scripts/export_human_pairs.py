"""Export a blinded pair set for human judging (plan.md §7 M4, §11).

    uv run python scripts/export_human_pairs.py --out ../RevisionJudge/data/pairs.json

plan.md §11 makes a human subsample the calibration anchor: *"panel calibrated against the
human subsample before being trusted"*. `findings-phase2.md` §9–§10 raised the stakes —
43–65% of the panel's raw verdicts turned out to be positional artifacts, so the anchor is
now the critical path rather than a formality.

**What gets sampled, and why it is not a simple random draw.**

Three strata, because they answer different questions:

1. ``stable`` — pairs where a majority of judges gave the same verdict in both orders.
   These are the panel's actual opinions, and agreement with a human here is what
   calibration means.
2. ``unstable`` — pairs where the panel mostly flipped when the order flipped. Sampling these
   asks a question nothing else can: **are humans order-sensitive on the same pairs?** If
   they are, order-inconsistency is a property of the comparison rather than a failing of
   the models, and the filter in §10 is discarding hard cases rather than bad judges.
3. ``objective`` — clean-vs-corrupted pairs from Stratum B, where a correct answer exists.
   These are the human's own positive control: a session whose objective pairs come out at
   chance was not done carefully, and none of its subjective verdicts should be used.

Every pair carries ``version_1``/``version_2`` and a separate answer key. **The key is not
part of what the judging app displays** — see the app's server, which strips it — because a
blinded comparison that ships its answer in the page source is not blinded.

Order is *not* pre-assigned here. The app randomises presentation per session, so a
re-export does not lock in a layout, and pairs repeated for the order-consistency check get
opposite orders by construction.

Nor is punctuation normalised here. The text is written exactly as generated, and the app
normalises at render time — see `findings-phase2.md` §11, which found that dash and quote
conventions alone identify the model's edit on ~17% of pairs. Keeping the raw text in this
file means the normalisation can be revisited without re-running anything, and means the
verdicts can always be traced back to what the model actually produced.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.judge import JudgeVerdict, order_consistent  # noqa: E402
from revisionbench.provenance import RunProvenance, sha256_text, utc_now  # noqa: E402
from revisionbench.records import read_jsonl, write_json  # noqa: E402
from scripts.self_preference import extract_edits  # noqa: E402


def _verdicts(path: Path) -> tuple[list[JudgeVerdict], list[JudgeVerdict], dict[str, str]]:
    forward, backward, source = [], [], {}
    fields = set(JudgeVerdict.__slots__)
    for row in read_jsonl(path):
        verdict = JudgeVerdict(**{k: v for k, v in row.items() if k in fields})
        (backward if row.get("order") == "reversed" else forward).append(verdict)
        source[row["pair_id"]] = row.get("edits_from", "")
    return forward, backward, source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verdicts", type=Path, default=REPO_ROOT / "results" / "phase2" / "self_preference.jsonl"
    )
    parser.add_argument(
        "--rounds", type=Path, default=REPO_ROOT / "results" / "phase0" / "rounds.jsonl"
    )
    parser.add_argument(
        "--defects", type=Path, default=REPO_ROOT / "data" / "corpus" / "defects.jsonl"
    )
    parser.add_argument("--clean", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-consistent", type=int, default=45)
    parser.add_argument("--n-inconsistent", type=int, default=35)
    parser.add_argument("--n-objective", type=int, default=12)
    parser.add_argument(
        "--repeat-fraction",
        type=float,
        default=0.15,
        help="share of subjective pairs shown twice, to measure the human's own order "
        "consistency against the panel's",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    forward, backward, _source = _verdicts(args.verdicts)
    if not forward:
        print(f"error: no verdicts in {args.verdicts}", file=sys.stderr)
        return 2

    # Rebuild the exact sample self_preference.py judged, so pair ids line up.
    pools = extract_edits(args.rounds)
    rng = random.Random(args.seed)
    edits: dict[str, dict[str, Any]] = {}
    for reviser in ("phi4:latest", "gemma3:4b"):
        if reviser not in pools:
            continue
        for index, edit in enumerate(rng.sample(pools[reviser], 60)):
            edits[f"{reviser.replace(':', '_')}-{index:03d}"] = edit

    by_judge_fwd: dict[str, list[JudgeVerdict]] = defaultdict(list)
    by_judge_rev: dict[str, list[JudgeVerdict]] = defaultdict(list)
    for verdict in forward:
        by_judge_fwd[verdict.judge].append(verdict)
    for verdict in backward:
        by_judge_rev[verdict.judge].append(verdict)

    # "Stable" means a MAJORITY of judges were order-consistent on the pair, not all of
    # them. Requiring unanimity yields 14 pairs of 120 and would make the calibration
    # stratum too thin to calibrate anything; a majority yields 49 and still means what it
    # needs to mean — the panel has a view that survives swapping the options.
    consistent_per_judge = {
        judge: set(order_consistent(by_judge_fwd[judge], by_judge_rev.get(judge, [])))
        for judge in by_judge_fwd
    }
    needed = max(1, (len(consistent_per_judge) + 1) // 2)
    stable_ids = {
        pair_id
        for pair_id in edits
        if sum(1 for kept in consistent_per_judge.values() if pair_id in kept) >= needed
    }
    unstable_ids = set(edits) - stable_ids

    sampler = random.Random(args.seed + 1)
    chosen: list[dict[str, Any]] = []

    def add(pair_id: str, stratum: str) -> None:
        edit = edits[pair_id]
        chosen.append(
            {
                "pair_id": pair_id,
                "stratum": stratum,
                "context": edit["context"],
                "version_1": edit["original"],
                "version_2": edit["replacement"],
                # Answer key. The judging app's server must not send this to the browser.
                "_key": {
                    "version_1": "original",
                    "version_2": "edited",
                    "reviser": edit["reviser"],
                    "passage_id": edit["passage_id"],
                    "better": None,
                },
            }
        )

    for pair_id in sampler.sample(sorted(stable_ids), min(args.n_consistent, len(stable_ids))):
        add(pair_id, "stable")
    for pair_id in sampler.sample(
        sorted(unstable_ids), min(args.n_inconsistent, len(unstable_ids))
    ):
        add(pair_id, "unstable")

    # The human's own positive control: clean vs corrupted, where an answer exists.
    clean = {
        p.stem: json.loads(p.read_text("utf-8"))["text"] for p in sorted(args.clean.glob("*.json"))
    }
    objective: list[dict[str, Any]] = []
    for line in args.defects.read_text("utf-8").splitlines():
        defect = json.loads(line)
        original = defect["clean_sentence"].strip()
        corrupt = defect["corrupt_fragment"].strip()
        if defect["type"] in ("name_drift", "number_drift"):
            corrupted = original.replace(defect["original_fragment"], corrupt, 1)
        elif defect["detect_kind"] in ("minimal_pair",) or defect["type"] == "clunker":
            corrupted = corrupt
        else:
            continue
        if not original or corrupted == original:
            continue
        passage = clean.get(defect["passage_id"], "")
        objective.append(
            {
                "pair_id": f"obj-{defect['defect_id']}",
                "stratum": "objective",
                "context": passage[:600],
                "version_1": original,
                "version_2": corrupted,
                "_key": {
                    "version_1": "clean",
                    "version_2": "corrupted",
                    "reviser": None,
                    "passage_id": defect["passage_id"],
                    "better": "version_1",
                },
            }
        )
    chosen.extend(sampler.sample(objective, min(args.n_objective, len(objective))))

    # Repeats for the human's own order-consistency measurement. Marked with the id they
    # duplicate so the analysis can find them; the app shows them in the opposite order and
    # spaces them apart so the judge is unlikely to recall the first showing.
    subjective = [p for p in chosen if p["stratum"] in ("stable", "unstable")]
    repeats = sampler.sample(
        subjective, min(int(len(subjective) * args.repeat_fraction), len(subjective))
    )
    for pair in repeats:
        duplicate = dict(pair)
        duplicate["pair_id"] = f"{pair['pair_id']}#r"
        duplicate["repeat_of"] = pair["pair_id"]
        chosen.append(duplicate)

    sampler.shuffle(chosen)

    payload = {
        "provenance": RunProvenance(
            run_id="export-human-pairs", started_at=utc_now(), config_hash="n/a"
        )
        .with_artifacts(
            source_verdicts=str(args.verdicts.as_posix()),
            seed=args.seed,
            strata={
                "consistent": args.n_consistent,
                "inconsistent": args.n_inconsistent,
                "objective": args.n_objective,
                "repeats": len(repeats),
            },
        )
        .as_dict(),
        "pairs": chosen,
        "pairs_sha256": sha256_text(json.dumps([p["pair_id"] for p in chosen], sort_keys=True)),
    }
    write_json(args.out, payload)

    counts: dict[str, int] = defaultdict(int)
    for pair in chosen:
        counts[pair["stratum"]] += 1
    print(f"{len(chosen)} pairs -> {args.out}")
    for stratum, count in sorted(counts.items()):
        print(f"  {stratum:<14}{count:>4}")
    print(f"  repeats       {len(repeats):>4}  (shown twice, opposite orders)")
    print(
        "\nThe answer key travels in `_key` and MUST be stripped before the browser sees a\n"
        "pair. RevisionJudge's server does that; anything else reading this file must too."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
