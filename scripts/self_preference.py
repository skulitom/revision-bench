"""Phase 2: does a model prefer its own edits? (plan.md §3, §7 M7)

    uv run python scripts/self_preference.py

Self-preference is the bias that makes an in-loop judge untrustworthy — the loop keeps
reporting "improved" partly because the thing reporting is the thing that wrote it
(plan.md §3, "Pride and Prejudice"). This measures it.

**The confound, and the design that removes it.** Asking "does phi4 prefer phi4's edits
more than the panel does" does not isolate bias: phi4 scored 0.97 on the M2-a control
against qwen3's 0.91, so it may simply judge better, and a better judge legitimately
disagrees with worse ones. Preference for one's own work and competence at judging are not
separable from a single comparison.

They are separable from a **difference in differences**, and Phase 0 happens to supply the
matched sets for it — two revisers, gemma3:4b and phi4, run over the same passages under
the same prompts:

    judge \\ edits from      phi4        gemma3
    phi4                    SELF        other
    gemma3                  other       SELF
    qwen3                   neutral     neutral

For a judge J and an edit source S, let p(J, S) be the rate at which J prefers the edited
sentence over the original. Then

    self-preference(phi4) = [p(phi4, phi4) - p(phi4, gemma3)]
                          - [p(qwen3, phi4) - p(qwen3, gemma3)]

The first bracket is how much more phi4 likes phi4's edits than gemma3's; the second is how
much better phi4's edits actually are, as scored by a judge with no stake. What remains is
the part attributable to authorship. gemma3 gets the same treatment with the sign flipped,
which is a replication rather than a second data point on the same model.

The design also absorbs a judge's **general leniency** (a judge that prefers every edit
scores high on both sources, so the difference is unaffected) and, to first order, its
**position bias** (the bias applies equally to both sources). Position bias is measured on
this task anyway, because "to first order" is not "entirely".

Edits are drawn from the saved Phase-0 artifacts — no new revision generations — and are
presented blinded: the judge sees two versions of a sentence and is never told which is
original, which is edited, or who wrote either.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.judge import (  # noqa: E402
    JudgeVerdict,
    Pair,
    judge_pair,
    order_consistent,
    position_bias,
)
from revisionbench.metrics.stats import mean, percentile  # noqa: E402
from revisionbench.metrics.thrash import align_sentences  # noqa: E402
from revisionbench.ollama import GenerationOptions, OllamaClient, OllamaError  # noqa: E402
from revisionbench.provenance import RunProvenance, utc_now  # noqa: E402
from revisionbench.records import JsonlWriter, read_jsonl, write_json  # noqa: E402
from revisionbench.text import sentences, words  # noqa: E402


def extract_edits(rounds_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct (original, replacement) sentence pairs per reviser from an artifact.

    Filtered to substantive edits: the pair must be recognisably the same sentence
    (similarity above 0.35, or it is a replacement rather than an edit) but not a trivial
    one (below 0.97, or the judge is being asked about a comma).
    """
    series: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    for row in read_jsonl(rounds_path):
        series[(row["model_tag"], row["prompt_name"], row["passage_id"])][row["round"]] = row[
            "text"
        ]

    by_reviser: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (model, prompt, passage_id), rounds in sorted(series.items()):
        for index in sorted(rounds):
            if index + 1 not in rounds:
                continue
            before, after = sentences(rounds[index]), sentences(rounds[index + 1])
            for pair in align_sentences(before, after):
                if pair.op != "modified" or not 0.35 < pair.similarity < 0.97:
                    continue
                original = before[pair.index_a]
                if len(words(original)) < 8:
                    continue
                by_reviser[model].append(
                    {
                        "reviser": model,
                        "prompt": prompt,
                        "passage_id": passage_id,
                        "round": index + 1,
                        "original": original,
                        "replacement": after[pair.index_b],
                        "context": " ".join(before[max(0, pair.index_a - 2) : pair.index_a + 3]),
                        "similarity": pair.similarity,
                    }
                )
    return by_reviser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rounds", type=Path, default=REPO_ROOT / "results" / "phase0" / "rounds.jsonl"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase2")
    parser.add_argument("--revisers", nargs=2, default=["phi4:latest", "gemma3:4b"])
    parser.add_argument(
        "--neutral",
        default="qwen3:4b",
        help="a judge with no stake in either reviser; the M2-a control's best unbiased judge",
    )
    parser.add_argument("--per-reviser", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument(
        "--both-orders",
        action="store_true",
        help="judge every pair twice with the order swapped, and report the results "
        "restricted to verdicts that survive the swap",
    )
    args = parser.parse_args(argv)

    pools = extract_edits(args.rounds)
    missing = [r for r in args.revisers if len(pools.get(r, [])) < args.per_reviser]
    if missing:
        print(
            f"error: too few edits for {missing}; have "
            f"{ {r: len(pools.get(r, [])) for r in args.revisers} }",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    pairs: list[Pair] = []
    for reviser in args.revisers:
        sample = rng.sample(pools[reviser], args.per_reviser)
        for index, edit in enumerate(sample):
            pairs.append(
                Pair(
                    pair_id=f"{reviser.replace(':', '_')}-{index:03d}",
                    context=edit["context"],
                    version_1=edit["original"],
                    version_2=edit["replacement"],
                    better=None,  # no ground truth: this is the subjective tier
                    kind=reviser,
                    meta=edit,
                )
            )
    print(f"{len(pairs)} blinded pairs: {args.per_reviser} edits from each of {args.revisers}")

    judges = [*args.revisers, args.neutral]
    options = GenerationOptions(
        seed=args.seed,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        num_ctx=8192,
        num_predict=64,
        repeat_penalty=1.0,
    )
    client = OllamaClient(args.host, keep_alive="60m")
    identities = {}
    for tag in judges:
        try:
            identities[tag] = client.identity(tag)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    verdicts: list[JudgeVerdict] = []
    reversed_verdicts: list[JudgeVerdict] = []
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "self_preference.jsonl").unlink(missing_ok=True)
    with JsonlWriter(args.out / "self_preference.jsonl", fsync=False) as out:
        for tag in judges:
            # Warmed per block: models are evicted between blocks on a 24 GB card and the
            # first call after a reload does not reproduce (findings-phase2.md §6).
            client.warm_up(tag, options)
            failures = 0
            for pair in pairs:
                try:
                    verdict = judge_pair(client, identities[tag], pair, options, seed=args.seed)
                except OllamaError:
                    failures += 1
                    continue
                verdicts.append(verdict)
                out.write({**verdict.as_dict(), "edits_from": pair.kind, "order": "forward"})
                if args.both_orders:
                    flipped = "version_2" if verdict.slot_a_was == "version_1" else "version_1"
                    try:
                        swapped = judge_pair(
                            client,
                            identities[tag],
                            pair,
                            options,
                            seed=args.seed,
                            force_slot_a=flipped,
                        )
                    except OllamaError:
                        failures += 1
                        continue
                    reversed_verdicts.append(swapped)
                    out.write({**swapped.as_dict(), "edits_from": pair.kind, "order": "reversed"})
            done = [v for v in verdicts if v.judge == tag]
            print(
                f"  {tag:<18} {len(done)} verdicts, {failures} failures, "
                f"position-A {position_bias(done):.2f}"
            )

    # p(judge, source) = rate at which the judge prefers the EDITED version.
    prefer: dict[tuple[str, str], list[int]] = defaultdict(list)
    source_of = {p.pair_id: p.kind for p in pairs}
    for verdict in verdicts:
        prefer[(verdict.judge, source_of[verdict.pair_id])].append(
            int(verdict.chose == "version_2")
        )

    first, second = args.revisers
    print("\n=== p(judge prefers the edited version), by whose edit it is ===")
    print(
        f"{'judge':<18}{'edits from ' + first:>22}{'edits from ' + second:>22}{'own - other':>14}"
    )
    rates: dict[str, dict[str, float]] = {}
    for tag in judges:
        a = prefer[(tag, first)]
        b = prefer[(tag, second)]
        if not a or not b:
            continue
        pa, pb = mean(a), mean(b)
        rates[tag] = {first: pa, second: pb}
        if tag == first:
            gap = f"{pa - pb:+.2f}"
        elif tag == second:
            gap = f"{pb - pa:+.2f}"
        else:
            gap = "  (neutral)"
        marker = "  SELF" if tag in args.revisers else ""
        print(f"{tag:<18}{pa:>22.2f}{pb:>22.2f}{gap:>14}{marker}")

    neutral_gap = rates[args.neutral][first] - rates[args.neutral][second]
    print(
        f"\nneutral judge's view of the edit-quality difference "
        f"({first} minus {second}): {neutral_gap:+.2f}"
    )

    print("\n=== self-preference (difference in differences) ===")
    summary: dict[str, Any] = {}
    for reviser in args.revisers:
        other = second if reviser == first else first
        own_gap = rates[reviser][reviser] - rates[reviser][other]
        baseline = rates[args.neutral][reviser] - rates[args.neutral][other]
        effect = own_gap - baseline

        # The statistic combines four independent samples, so each is resampled
        # independently and at its own size. An earlier version drew one index list and
        # applied it to all four with a modulo — which both correlates pools that are not
        # correlated and indexes the shorter ones by wrap-around, narrowing the interval
        # for reasons that have nothing to do with the data.
        pools_for_ci = (
            prefer[(reviser, reviser)],
            prefer[(reviser, other)],
            prefer[(args.neutral, reviser)],
            prefer[(args.neutral, other)],
        )
        boot_rng = random.Random(args.seed)
        estimates = []
        for _ in range(2000):
            a, b, c, d = (boot_rng.choices(pool, k=len(pool)) for pool in pools_for_ci)
            estimates.append((mean(a) - mean(b)) - (mean(c) - mean(d)))
        estimates.sort()
        low, high = percentile(estimates, 0.025), percentile(estimates, 0.975)
        print(
            f"  {reviser:<18} own-minus-other {own_gap:+.2f}, neutral baseline "
            f"{baseline:+.2f}  ->  self-preference {effect:+.2f}  [95% CI {low:+.2f}, {high:+.2f}]"
        )
        summary[reviser] = {
            "own_gap": own_gap,
            "neutral_baseline": baseline,
            "self_preference": effect,
            "ci": [low, high],
            "n_own": len(pools_for_ci[0]),
            "n_other": len(pools_for_ci[1]),
        }

    consistent_rates: dict[str, Any] = {}
    if args.both_orders:
        print("\n=== order-consistency filter: verdicts that survive swapping the order ===")
        print(f"{'judge':<18}{'consistent':>12}{'retained':>10}{'prefers edit':>14}")
        for tag in judges:
            forward = [v for v in verdicts if v.judge == tag]
            backward = [v for v in reversed_verdicts if v.judge == tag]
            if not forward:
                continue
            kept = order_consistent(forward, backward)
            prefers = mean([int(c == "version_2") for c in kept.values()]) if kept else None
            print(
                f"{tag:<18}{len(kept):>12}{len(kept) / len(forward):>10.2f}"
                f"{(f'{prefers:.2f}' if prefers is not None else '   -'):>14}"
            )
            consistent_rates[tag] = {
                "consistent": len(kept),
                "asked": len(forward),
                "retained": len(kept) / len(forward),
                "prefers_edit": prefers,
                "by_source": {
                    src: (
                        mean([int(kept[p] == "version_2") for p in kept if source_of[p] == src])
                        if any(source_of[p] == src for p in kept)
                        else None
                    )
                    for src in args.revisers
                },
            }
        print(
            "\nA judge answering with position flips when the order flips, so its verdict is\n"
            "dropped here. Retention is the share of its opinions that were about the text."
        )

    print(
        "\nA positive figure means the judge favours its own edits beyond what a\n"
        "disinterested judge sees in them. Zero inside the interval means this design,\n"
        "at this sample size, did not detect self-preference — not that there is none."
    )

    write_json(
        args.out / "self_preference.json",
        {
            "provenance": RunProvenance(
                run_id="self-preference", started_at=utc_now(), config_hash="n/a"
            )
            .with_artifacts(
                revisers=args.revisers,
                neutral=args.neutral,
                per_reviser=args.per_reviser,
                seed=args.seed,
                judges={t: identities[t].as_dict() for t in identities},
            )
            .as_dict(),
            "prefer_rates": {j: rates[j] for j in rates},
            "position_bias": {
                t: position_bias([v for v in verdicts if v.judge == t]) for t in judges
            },
            "self_preference": summary,
            "order_consistency": consistent_rates,
            "n_verdicts": len(verdicts),
            "counts": dict(Counter(f"{v.judge}|{source_of[v.pair_id]}" for v in verdicts)),
        },
    )
    print(f"\nwritten: {args.out / 'self_preference.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
