"""Phase 0 metrics: compute M1/M2/M3/M6 from the round artifact alone (plan.md §7, §9).

    uv run python scripts/phase0_metrics.py

Reads ``results/phase0/rounds.jsonl`` and writes ``results/phase0/metrics.jsonl`` plus a
summary. **No model is contacted and no network is touched** — that is the acceptance
criterion plan.md §9 sets for Phase 0 ("metrics reproducible from artifacts alone"), and
keeping this script model-free is what makes it checkable rather than aspirational.

Two conventions that the numbers depend on:

**The style model is fitted on round 0 and never re-fitted.** Every round is standardised
against the same frozen scaler. Re-fitting per round would subtract exactly the effect H1
predicts (see ``revisionbench/metrics/stylometry.py``).

**A round that cannot be scored is recorded, not dropped.** A reviser that returns two
sentences raises :class:`MetricError`; the row gets ``metric_error`` and no metric values.
Dropping it would quietly remove the most degraded rounds from every mean, which biases
every curve toward "nothing much happened".

**Trajectories that reached a fixed point are carried forward.** A passage that stops
changing produces no more rows, so without this a round-9 mean would be taken over only the
passages that had *not* settled — which is a survivorship bias pointing in the worst
possible direction, since the passages still moving at round 9 are the pathological ones.
On the first partial run this made the round-10 cell an average over exactly one passage,
at four times the slop rate of the round-3 cell, and it looked like a degradation curve.

Carrying forward is not an assumption here. Under a reproducible sampler a fixed point is
absorbing: round k+1's input would equal round k's, so its output would equal round k's
too. The carried rows are what the loop *would* have produced, and they are flagged
``carried_forward`` so a reader can strip them.

Trajectories stopped for **collapse** are a different case and are *not* carried forward.
There the loop was halted by our own floor rule, so what it would have done next is
genuinely unknown. Those rounds are censored, and the per-round ``n_passages`` reports it.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.config import ConfigError, load_config  # noqa: E402
from revisionbench.metrics.slop import load_lexicon, slop_index  # noqa: E402
from revisionbench.metrics.stats import mean  # noqa: E402
from revisionbench.metrics.stylometry import (  # noqa: E402
    MetricError,
    StyleModel,
    describe,
    mean_pairwise_delta,
    wasserstein1,
)
from revisionbench.metrics.thrash import (  # noqa: E402
    edit_report,
    rounds_to_fixed_point,
    thrash_report,
)
from revisionbench.provenance import RunProvenance, utc_now  # noqa: E402
from revisionbench.records import JsonlWriter, read_jsonl, write_json  # noqa: E402
from revisionbench.text import sentence_lengths  # noqa: E402


def _carry_forward(last_row: dict, last_record: dict, max_round: int) -> list[dict]:
    """Extend a settled trajectory to ``max_round`` (see the module docstring).

    Only for ``fixed_point``. A settled passage's later rounds are byte-identical to its
    last one, so every metric is too; the edit report is a no-op and there is no thrash
    window, because nothing changed.
    """
    if last_row.get("stop_reason") != "fixed_point":
        return []
    out = []
    for round_index in range(last_row["round"] + 1, max_round + 1):
        clone = dict(last_record)
        clone["round"] = round_index
        clone["carried_forward"] = True
        clone["stop_reason"] = "fixed_point"
        clone["edits"] = {
            "modified": 0,
            "inserted": 0,
            "deleted": 0,
            "edits": 0,
            "edit_fraction": 0.0,
            "mean_modified_similarity": None,
        }
        clone["thrash"] = None
        out.append(clone)
    return out


def run_label(row: dict) -> str:
    """The series a row belongs to: one model crossed with one prompt.

    Keyed on the model **tag** for readability, but grouping must be model-aware at all:
    the first version of this keyed only on (prompt, passage), so running a second reviser
    into the same artifact would have interleaved two models' rounds into one trajectory —
    duplicate round numbers, a feed-forward chain that never existed, and thrash windows
    computed across a model boundary. Nothing would have raised.
    """
    return f"{row['model_tag']} | {row['prompt_name']}"


def trajectories(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rows into ``(run_label, passage_id) -> rows ordered by round``."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(run_label(row), row["passage_id"])].append(row)
    return {k: sorted(v, key=lambda r: r["round"]) for k, v in sorted(grouped.items())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rounds", type=Path, default=REPO_ROOT / "results" / "phase0" / "rounds.jsonl"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase0")
    parser.add_argument("--n-function-words", type=int, default=100)
    parser.add_argument(
        "--round-cap",
        type=int,
        default=None,
        help="round to carry settled trajectories forward to; defaults to loop.rounds in "
        "configs/phase0.yaml",
    )
    args = parser.parse_args(argv)

    rows = list(read_jsonl(args.rounds))
    if not rows:
        print(f"error: no rows in {args.rounds}; run scripts/phase0.py first", file=sys.stderr)
        return 2
    traj = trajectories(rows)
    prompts = sorted({r["prompt_name"] for r in rows})
    print(f"read {len(rows)} rows: {len(traj)} trajectories, prompts {prompts}")

    max_round = args.round_cap
    if max_round is None:
        try:
            max_round = int(load_config(REPO_ROOT / "configs" / "phase0.yaml")["loop"]["rounds"])
        except (OSError, KeyError, ValueError, ConfigError):
            max_round = max(r["round"] for r in rows)
            print(
                f"warning: could not read loop.rounds from config; carrying forward to the "
                f"highest round observed ({max_round}), which understates the cap if every "
                f"trajectory settled early"
            )
    print(f"carrying settled trajectories forward to round {max_round}")

    # --- the frozen scaler, fitted on round 0 only ----------------------------------
    round0 = {r["passage_id"]: r for r in rows if r["round"] == 0}
    baseline_texts = [round0[p]["text"] for p in sorted(round0)]
    model = StyleModel.fit(baseline_texts, n_function_words=args.n_function_words)
    print(
        f"style model fitted on {len(baseline_texts)} round-0 passages, "
        f"{len(model.features)} features (dropped {len(model.dropped_low_variance)})"
    )

    lexicon = load_lexicon()
    metrics_path = args.out / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    per_round_rows: list[dict[str, Any]] = []
    with JsonlWriter(metrics_path, fsync=False) as out:
        for (run, passage_id), series in traj.items():
            base_row = series[0]
            base_text = base_row["text"]
            base_lengths = sentence_lengths(base_text)
            texts = [r["text"] for r in series]

            thrash = {t.round_index: t for t in thrash_report(texts)}
            fixed_point = rounds_to_fixed_point(texts)

            for index, row in enumerate(series):
                record: dict[str, Any] = {
                    "run": run,
                    "model_tag": row["model_tag"],
                    "model_digest": row["model_digest"],
                    "prompt_name": row["prompt_name"],
                    "passage_id": passage_id,
                    "author_id": row["author_id"],
                    "fame": row["fame"],
                    "round": row["round"],
                    "word_count": row["word_count"],
                    "length_ratio": row["length_ratio"],
                    "length_guard_tripped": row["length_guard_tripped"],
                    "stop_reason": row.get("stop_reason"),
                    "rounds_to_fixed_point": fixed_point,
                }
                try:
                    stats = describe(row["text"])
                    record.update({k: v for k, v in stats.items() if k != "words"})
                    # M1: distance from this passage's own round 0, which is what the A4
                    # voice veto would threshold on.
                    record["delta_from_round0"] = model.delta(base_text, row["text"])
                    record["delta_fw_from_round0"] = model.delta(
                        base_text, row["text"], family="fw"
                    )
                    record["delta_punct_from_round0"] = model.delta(
                        base_text, row["text"], family="punct"
                    )
                    record["sent_len_w1_from_round0"] = wasserstein1(
                        base_lengths, sentence_lengths(row["text"])
                    )
                    # M3
                    slop = slop_index(row["text"], lexicon)
                    record["slop_per_1000w"] = slop.per_1000_words
                    record["slop_hits"] = slop.hits
                    record["slop_by_group"] = slop.by_group
                    record["metric_error"] = None
                except MetricError as exc:
                    # Recorded, never dropped: the unscoreable rounds are the most degraded
                    # ones, and silently omitting them biases every curve toward "nothing
                    # much happened".
                    record["metric_error"] = f"{type(exc).__name__}: {exc}"

                # M6: edit volume against the previous round, thrash over (k, k+1, k+2).
                if index > 0:
                    record["edits"] = edit_report(series[index - 1]["text"], row["text"]).as_dict()
                window = thrash.get(row["round"])
                record["thrash"] = window.as_dict() if window else None
                record["carried_forward"] = False

                out.write(record)
                per_round_rows.append(record)

            for extra in _carry_forward(series[-1], per_round_rows[-1], max_round):
                out.write(extra)
                per_round_rows.append(extra)

    summary = summarise(per_round_rows, model, traj)
    write_json(args.out / "metrics_summary.json", summary)
    report(summary, per_round_rows)

    provenance = (
        RunProvenance(run_id="phase0-metrics", started_at=utc_now(), config_hash="n/a")
        .with_artifacts(
            rounds_artifact=str(args.rounds.as_posix()),
            n_function_words=args.n_function_words,
            slop_lexicon_version=lexicon.version,
            style_model=model.to_dict(),
        )
        .as_dict()
    )
    write_json(REPO_ROOT / "results" / "provenance" / "phase0-metrics.json", provenance)
    print(f"\nwritten: {metrics_path}")
    print(f"         {args.out / 'metrics_summary.json'}")
    return 0


def summarise(records: list[dict], model: StyleModel, traj: dict) -> dict[str, Any]:
    """Per-prompt, per-round aggregates, plus M2 homogenization."""
    out: dict[str, Any] = {"by_run": {}}
    runs = sorted({r["run"] for r in records})

    for run in runs:
        subset = [r for r in records if r["run"] == run]
        rounds = sorted({r["round"] for r in subset})
        per_round = []
        for round_index in rounds:
            cell = [r for r in subset if r["round"] == round_index]
            scored = [r for r in cell if r["metric_error"] is None]
            entry: dict[str, Any] = {
                "round": round_index,
                "n_passages": len(cell),
                "n_scored": len(scored),
                # How many of this cell's passages had already settled and are being
                # carried forward. A cell that is mostly carried is reporting the loop's
                # resting state, not what round k did, and the two read very differently.
                "n_carried_forward": sum(1 for r in cell if r.get("carried_forward")),
                "n_live": sum(1 for r in cell if not r.get("carried_forward")),
                "n_metric_errors": len(cell) - len(scored),
                "n_length_guard_tripped": sum(1 for r in cell if r["length_guard_tripped"]),
                "mean_word_count": mean([r["word_count"] for r in cell]),
                "mean_length_ratio": mean([r["length_ratio"] for r in cell]),
            }
            for field in (
                "delta_from_round0",
                "delta_fw_from_round0",
                "delta_punct_from_round0",
                "sent_len_w1_from_round0",
                "slop_per_1000w",
                "sent:mean",
                "sent:sd",
                "lex:mtld",
                "punct:semicolon",
                "punct:em_dash",
                "punct:comma",
            ):
                values = [r[field] for r in scored if field in r]
                entry[field] = mean(values) if values else None

            # M2 (H1): mean pairwise cross-author distance at this round. Needs at least
            # two authors still scoreable, or the cell is genuinely missing rather than 0.
            labelled = [
                (r["author_id"], t)
                for r in scored
                for t in [_text_for(traj, run, r["passage_id"], round_index)]
                if t is not None
            ]
            entry["m2_n_texts"] = len(labelled)
            try:
                entry["m2_cross_author_delta"] = mean_pairwise_delta(labelled, model)
                entry["m2_within_author_delta"] = mean_pairwise_delta(
                    labelled, model, cross_author_only=False
                )
            except MetricError:
                entry["m2_cross_author_delta"] = None
                entry["m2_within_author_delta"] = None

            thrash_values = [
                r["thrash"]["thrash_fraction"]
                for r in cell
                if r.get("thrash") and r["thrash"]["thrash_fraction"] is not None
            ]
            entry["mean_thrash_fraction"] = mean(thrash_values) if thrash_values else None
            edit_values = [r["edits"]["edit_fraction"] for r in cell if r.get("edits")]
            entry["mean_edit_fraction"] = mean(edit_values) if edit_values else None
            per_round.append(entry)

        fixed_points = {r["passage_id"]: r["rounds_to_fixed_point"] for r in subset}
        out["by_run"][run] = {
            "per_round": per_round,
            "rounds_to_fixed_point": fixed_points,
            "n_reached_fixed_point": sum(1 for v in fixed_points.values() if v is not None),
            "n_passages": len(fixed_points),
        }
    return out


def _text_for(traj: dict, run: str, passage_id: str, round_index: int) -> str | None:
    """The text at ``round_index``, carrying a settled trajectory forward.

    M2 needs this as much as the per-round means do, and for the same reason: without the
    carry-forward the cross-author distance at round 6 would be computed over only the
    passages still moving at round 6. Since H1 is a primary endpoint (plan.md §11), a
    survivorship-biased M2 is the single most expensive wrong number this script could
    produce — and it would look like a homogenization signal.
    """
    series = traj.get((run, passage_id), [])
    if not series:
        return None
    for row in series:
        if row["round"] == round_index:
            return row["text"]
    last = series[-1]
    if round_index > last["round"] and last.get("stop_reason") == "fixed_point":
        return last["text"]
    return None


def report(summary: dict, records: list[dict]) -> None:
    for run, block in summary["by_run"].items():
        print(f"\n=== {run} ===")
        print(
            f"{'rd':>3}{'n':>4}{'live':>5}{'err':>4}{'guard':>6}{'words':>7}{'ratio':>7}"
            f"{'delta0':>8}{'d_punct':>8}{'slop':>7}{'sent_mn':>8}{'mtld':>7}"
            f"{'M2xA':>7}{'edit':>6}{'thrash':>7}"
        )
        for entry in block["per_round"]:

            def fmt(key, width, places=2, *, row=entry):
                value = row.get(key)
                return f"{value:>{width}.{places}f}" if value is not None else f"{'-':>{width}}"

            print(
                f"{entry['round']:>3}{entry['n_passages']:>4}{entry['n_live']:>5}"
                f"{entry['n_metric_errors']:>4}"
                f"{entry['n_length_guard_tripped']:>6}"
                f"{entry['mean_word_count']:>7.0f}{entry['mean_length_ratio']:>7.2f}"
                f"{fmt('delta_from_round0', 8)}{fmt('delta_punct_from_round0', 8)}"
                f"{fmt('slop_per_1000w', 7)}{fmt('sent:mean', 8, 1)}{fmt('lex:mtld', 7, 1)}"
                f"{fmt('m2_cross_author_delta', 7)}{fmt('mean_edit_fraction', 6)}"
                f"{fmt('mean_thrash_fraction', 7)}"
            )
        reached = block["n_reached_fixed_point"]
        print(
            f"  fixed point reached by {reached}/{block['n_passages']} passages: "
            f"{ {k: v for k, v in block['rounds_to_fixed_point'].items() if v is not None} }"
        )


if __name__ == "__main__":
    raise SystemExit(main())
