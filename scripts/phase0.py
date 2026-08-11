"""Phase 0: run the A0 unconstrained revision loop (plan.md §9, §14 M0-c).

    uv run python scripts/phase0.py --config configs/phase0.yaml --dry-run
    uv run python scripts/phase0.py --config configs/phase0.yaml

Writes one JSONL row per (passage, prompt, round) to ``results/phase0/rounds.jsonl``,
carrying the round's text and the facts needed to distrust it. **No metric values are
written here** — see ``revisionbench/loop.py`` for why, and ``scripts/phase0_metrics.py``
for the pass that computes them.

The run is resumable: re-running skips rounds already present, keyed on
(passage, arm, prompt, model digest, round). Interrupting it is safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.config import ConfigError, config_hash, load_config, require_keys  # noqa: E402
from revisionbench.loop import (  # noqa: E402
    RESUME_KEY_FIELDS,
    LoopSpec,
    PromptSpec,
    recover_texts,
    run_passage,
    summarise_generation,
)
from revisionbench.ollama import GenerationOptions, OllamaClient, OllamaError  # noqa: E402
from revisionbench.provenance import RunProvenance, utc_now  # noqa: E402
from revisionbench.records import JsonlWriter, read_jsonl, resume_index, write_json  # noqa: E402

TOP_LEVEL = {
    "required": ("version", "corpus", "model", "generation", "loop", "prompts"),
    "optional": ("description",),
}


def parse(cfg: dict) -> tuple[dict, GenerationOptions, LoopSpec, list[PromptSpec]]:
    require_keys(cfg, where="phase0 config", **TOP_LEVEL)
    require_keys(cfg["model"], required=("tag", "expected_digest", "keep_alive"), where="model")
    require_keys(
        cfg["generation"],
        required=(
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "num_ctx",
            "num_predict",
            "repeat_penalty",
        ),
        where="generation",
    )
    require_keys(
        cfg["loop"],
        required=("arm", "rounds", "length_guard", "min_words_to_continue"),
        optional=("length_policy", "max_attempts"),
        where="loop",
    )
    options = GenerationOptions(**cfg["generation"])
    spec = LoopSpec(
        arm=cfg["loop"]["arm"],
        rounds=int(cfg["loop"]["rounds"]),
        length_guard=tuple(cfg["loop"]["length_guard"]),
        min_words_to_continue=int(cfg["loop"]["min_words_to_continue"]),
        length_policy=str(cfg["loop"].get("length_policy", "observe")),
        max_attempts=int(cfg["loop"].get("max_attempts", 1)),
    )
    prompts = []
    seen = set()
    for index, raw in enumerate(cfg["prompts"]):
        require_keys(raw, required=("name", "template"), where=f"prompts[{index}]")
        prompt = PromptSpec(name=raw["name"], template=raw["template"])
        if prompt.name in seen:
            raise ConfigError(f"duplicate prompt name {prompt.name!r}")
        seen.add(prompt.name)
        prompts.append(prompt)
    return cfg["model"], options, spec, prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase0")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--dry-run", action="store_true", help="validate and plan; no model calls")
    parser.add_argument("--prompt", action="append", help="run only these prompt names")
    parser.add_argument("--passage", action="append", help="run only these passage ids")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        model_cfg, options, spec, prompts = parse(cfg)
    except (ConfigError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.prompt:
        prompts = [p for p in prompts if p.name in set(args.prompt)]
        if not prompts:
            print(f"error: no prompt matched {args.prompt}", file=sys.stderr)
            return 2

    corpus_dir = REPO_ROOT / "data" / "corpus" / "passages"
    passages = [json.loads(p.read_text("utf-8")) for p in sorted(corpus_dir.glob("*.json"))]
    if args.passage:
        wanted = set(args.passage)
        passages = [p for p in passages if p["passage_id"] in wanted]
    if not passages:
        print(f"error: no passages found under {corpus_dir}", file=sys.stderr)
        return 2

    cfg_hash = config_hash(cfg)
    total = len(passages) * len(prompts) * spec.rounds
    print(f"config     {args.config}  (hash {cfg_hash})")
    print(
        f"arm        {spec.arm}   rounds {spec.rounds}   "
        f"length guard {spec.length_guard}   floor {spec.min_words_to_continue}w"
    )
    gated = "   (GATED - this is not the A0 control)" if spec.length_policy != "observe" else ""
    print(f"length     policy={spec.length_policy} max_attempts={spec.max_attempts}{gated}")
    print(f"model      {model_cfg['tag']}")
    print(
        f"sampling   seed={options.seed} temp={options.temperature} top_k={options.top_k} "
        f"num_ctx={options.num_ctx} num_predict={options.num_predict}  "
        f"deterministic={options.is_deterministic}"
    )
    print(f"prompts    {', '.join(p.name for p in prompts)}")
    print(f"passages   {len(passages)}   -> up to {total} generations")

    if args.dry_run:
        print("\n-- plan (dry run; no model calls, nothing written) --")
        for prompt in prompts:
            print(f"  prompt {prompt.name}  sha256 {prompt.sha256[:16]}")
            print(f"    rendered head: {prompt.render(passages[0]['text'])[:110]!r}")
        for passage in passages:
            print(
                f"  {passage['passage_id']:<16} {passage['author_display']:<22} "
                f"{passage['word_count']:>5}w"
            )
        return 0

    client = OllamaClient(args.host, keep_alive=str(model_cfg["keep_alive"]))
    try:
        model = client.identity(model_cfg["tag"])
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if model.digest != model_cfg["expected_digest"]:
        print(
            f"error: model digest mismatch for {model.tag}\n"
            f"  expected {model_cfg['expected_digest']}\n"
            f"  got      {model.digest}\n"
            f"The tag has been re-pointed by a later `ollama pull`. These are different "
            f"weights, so rows produced now are not comparable with rows produced before. "
            f"Update expected_digest deliberately, and treat the earlier rounds as a "
            f"separate run.",
            file=sys.stderr,
        )
        return 2
    print(f"resolved   {model.summary()}")

    # Spend one throwaway generation so the first *scored* round is a warm one. The first
    # call after a model load does not reproduce (docs/findings-phase0.md §5.1), and a
    # cold round buried in a 200-generation sweep is indistinguishable in the artifact from
    # a genuine change in loop behaviour.
    warm = client.warm_up(model.tag, options)
    print(f"warm-up    discarded 1 generation ({warm.wall_seconds:.1f}s)\n")

    rounds_path = args.out / "rounds.jsonl"
    done = resume_index(rounds_path, RESUME_KEY_FIELDS) if rounds_path.exists() else set()
    existing: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(rounds_path):
        existing[(row["passage_id"], row["prompt_name"])].append(row)
    if done:
        print(f"resuming: {len(done)} rows already present in {rounds_path}\n")

    provenance = {"run_id": f"phase0-{cfg_hash}", "config_hash": cfg_hash}
    written = 0
    stopped: list[str] = []
    try:
        with JsonlWriter(rounds_path, provenance=provenance) as writer:
            for prompt in prompts:
                for passage in passages:
                    prior = existing[(passage["passage_id"], prompt.name)]
                    label = f"{prompt.name}/{passage['passage_id']}"
                    last = None
                    for row in run_passage(
                        client,
                        passage,
                        prompt=prompt,
                        model=model,
                        options=options,
                        spec=spec,
                        writer=writer,
                        config_hash=cfg_hash,
                        already_done=done,
                        resume_texts=recover_texts(prior),
                    ):
                        written += 1
                        last = row
                        if row["round"] > 0:
                            flag = "!" if row["length_guard_tripped"] else " "
                            gen = row["generation"]
                            print(
                                f"  {label:<34} r{row['round']:<3} "
                                f"{row['word_count']:>5}w {flag} "
                                f"ratio={row['length_ratio']:.2f} "
                                f"{gen['wall_seconds']:>6.1f}s "
                                f"{gen['done_reason']}"
                            )
                    if last and last.get("stop_reason"):
                        stopped.append(
                            f"{label} stopped at round {last['round']}: {last['stop_reason']}"
                        )
                        print(f"  {label:<34} STOPPED: {last['stop_reason']}")
    except (OllamaError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        print(f"{written} rows were written and are safe; re-run to resume.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\ninterrupted; {written} rows written and safe. Re-run to resume.")
        return 130

    all_rows = list(read_jsonl(rounds_path))
    stats = summarise_generation(all_rows)
    print(f"\nwrote {written} new rows ({len(all_rows)} total) -> {rounds_path}")
    print(
        f"generations {stats.get('generations', 0)}  "
        f"wall {stats.get('wall_seconds_total', 0) / 60:.1f} min  "
        f"mean {stats.get('wall_seconds_mean', 0):.1f} s/round  "
        f"truncated {stats.get('truncated_rounds', 0)}"
    )
    for line in stopped:
        print(f"  {line}")

    run_provenance = (
        RunProvenance(run_id=f"phase0-{cfg_hash}", started_at=utc_now(), config_hash=cfg_hash)
        .with_artifacts(
            phase0_config=str(args.config.as_posix()),
            model=model.as_dict(),
            generation=options.as_ollama_options(),
            prompts={p.name: p.sha256 for p in prompts},
            cost=stats,
        )
        .as_dict()
    )
    write_json(REPO_ROOT / "results" / "provenance" / f"phase0-{cfg_hash}.json", run_provenance)
    print("\nnext: uv run python scripts/phase0_metrics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
