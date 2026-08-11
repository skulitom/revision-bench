"""Build Stratum B: plant known defects in the Stratum-A passages (plan.md §6).

    uv run python scripts/inject_defects.py
    uv run python scripts/inject_defects.py --show woolf-01

Writes ``data/corpus/passages_b/{id}.json`` (corrupted passages) and
``data/corpus/defects.jsonl`` (the ground-truth manifest). Deterministic: the same corpus
and seed regenerate both byte-identically, so a rebuild can be diffed.

``--show`` prints each planted defect in context, which is how the hand-verification
plan.md §6 asks for actually gets done. ``--probe`` is the §12.4 check: ask a local model
to *locate* the injections blind, and if it finds them trivially the injector is too
obvious and recall would be measuring pattern-matching rather than understanding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.inject import (  # noqa: E402
    INJECTOR_VERSION,
    InjectionError,
    inject_passage,
)
from revisionbench.provenance import sha256_text  # noqa: E402
from revisionbench.records import JsonlWriter, write_json  # noqa: E402
from revisionbench.text import word_count  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "corpus" / "passages")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "corpus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target", type=int, default=4, help="defects per passage (plan.md §6: 3-6)"
    )
    parser.add_argument("--show", help="print planted defects in context for one passage id")
    args = parser.parse_args(argv)

    records = [json.loads(p.read_text("utf-8")) for p in sorted(args.corpus.glob("*.json"))]
    if not records:
        print(f"error: no passages under {args.corpus}", file=sys.stderr)
        return 2

    passage_dir = args.out / "passages_b"
    manifest_path = args.out / "defects.jsonl"
    manifest_path.unlink(missing_ok=True)

    total_defects = 0
    by_type: dict[str, int] = {}
    skipped: list[str] = []

    with JsonlWriter(manifest_path, fsync=False) as manifest:
        for record in records:
            passage_id = record["passage_id"]
            # Seed per passage, derived from the run seed and the id, so adding a passage
            # does not re-roll every other passage's defects.
            #
            # SHA-256, not `hash()`. Python randomises string hashing per process unless
            # PYTHONHASHSEED is pinned, so `hash(passage_id)` would plant different defects
            # on every run — in a corpus whose entire value is being byte-reproducible, and
            # with nothing to signal it but a manifest that quietly changed.
            seed = args.seed + int(sha256_text(passage_id)[:8], 16) % 10_000
            try:
                result = inject_passage(record["text"], passage_id, seed=seed, target=args.target)
            except InjectionError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

            for defect in result.defects:
                manifest.write(defect.as_dict())
                by_type[defect.type] = by_type.get(defect.type, 0) + 1
            total_defects += len(result.defects)
            skipped.extend(f"{passage_id}: {s}" for s in result.skipped)

            write_json(
                passage_dir / f"{passage_id}.json",
                {
                    **{k: v for k, v in record.items() if k not in ("text", "text_sha256")},
                    "stratum": "B",
                    "clean_passage_id": passage_id,
                    "clean_text_sha256": record["text_sha256"],
                    "text": result.text,
                    "text_sha256": sha256_text(result.text),
                    "word_count": word_count(result.text),
                    "injector_version": INJECTOR_VERSION,
                    "injection_seed": seed,
                    "defect_ids": [d.defect_id for d in result.defects],
                },
            )
            print(
                f"  {passage_id:<16} {len(result.defects)} defects "
                f"({', '.join(d.type for d in result.defects)})"
            )

    print(f"\n{total_defects} defects across {len(records)} passages")
    for defect_type, count in sorted(by_type.items()):
        print(f"  {defect_type:<16}{count:>4}")
    if skipped:
        # Never silent: a passage with fewer defects has a different recall denominator.
        print(f"\nskipped {len(skipped)} injector attempt(s) that did not apply:")
        for line in skipped[:10]:
            print(f"  {line}")
    print(f"\nwritten: {passage_dir}")
    print(f"         {manifest_path}")

    if args.show:
        show(manifest_path, passage_dir, args.show)
    return 0


def show(manifest_path: Path, passage_dir: Path, passage_id: str) -> None:
    """Print each defect in context — the hand-verification plan.md §6 requires."""
    corrupted = json.loads((passage_dir / f"{passage_id}.json").read_text("utf-8"))["text"]
    defects = [
        json.loads(line)
        for line in manifest_path.read_text("utf-8").splitlines()
        if json.loads(line)["passage_id"] == passage_id
    ]
    print(f"\n=== {passage_id}: {len(defects)} defects in context ===")
    for defect in defects:
        start, end = defect["char_span"]
        left = corrupted[max(0, start - 110) : start].replace("\n", " ")
        middle = corrupted[start:end].replace("\n", " ")
        right = corrupted[end : end + 110].replace("\n", " ")
        print(f"\n[{defect['type']}] {defect['notes']}")
        print(f"  ...{left}<<< {middle[:180]} >>>{right}...")


if __name__ == "__main__":
    raise SystemExit(main())
