"""Build a corpus from a config: fetch source books, cut passages, record provenance.

    uv run python scripts/fetch_corpus.py --config configs/corpus/phase0.yaml
    uv run python scripts/fetch_corpus.py --config configs/corpus/phase0.yaml --dry-run
    uv run python scripts/fetch_corpus.py --config configs/corpus/phase0.yaml --offline

``--dry-run`` validates the config and prints the plan without touching the network, and
``--offline`` rebuilds from the raw cache only. The second one is the interesting guard:
it proves the committed corpus can be regenerated without Project Gutenberg being
reachable, which is what "reproducible" has to mean for a corpus whose upstream re-issues
its files.

This is the only script in the repo that makes a network request.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revisionbench.config import config_hash, load_config  # noqa: E402
from revisionbench.corpus import (  # noqa: E402
    EXTRACTOR_VERSION,
    CorpusError,
    extract_passage,
    fetch_book,
    parse_config,
    passage_record,
    strip_boilerplate,
)
from revisionbench.provenance import RunProvenance, utc_now  # noqa: E402
from revisionbench.records import write_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="corpus config YAML")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "corpus",
        help="corpus directory (raw/ cache and passages/ output live under it)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and plan only")
    parser.add_argument("--offline", action="store_true", help="use the raw cache only")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
        extraction, sources, passages = parse_config(cfg)
    except (CorpusError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cfg_hash = config_hash(cfg)
    print(f"config      {args.config}  (hash {cfg_hash})")
    print(f"sources     {len(sources)}   passages {len(passages)}")
    print(
        f"extraction  {extraction.min_words}-{extraction.max_words} words, "
        f"target set per passage, extractor v{EXTRACTOR_VERSION}"
    )

    if args.dry_run:
        print("\n-- plan (dry run; no network, nothing written) --")
        for spec in passages:
            source = sources[spec.source]
            pin = "pinned" if source.expected_sha256 else "UNPINNED"
            print(
                f"  {spec.id:<16} {source.author_display:<22} target={spec.target_words:>4}w "
                f"src=pg{source.ebook_id} [{pin}]  anchor={spec.anchor[:44]!r}"
            )
        unpinned = [s.key for s in sources.values() if not s.expected_sha256]
        if unpinned:
            print(
                f"\nnote: {len(unpinned)} source(s) have no expected_sha256: {', '.join(unpinned)}"
            )
            print("      Run once, then paste the reported digests into the config.")
        return 0

    raw_dir = args.out / "raw"
    passage_dir = args.out / "passages"
    session = requests.Session()

    # Fetch each source once, then cut all of its passages.
    bodies = {}
    books = {}
    for key, source in sources.items():
        try:
            text, book = fetch_book(
                source, raw_dir, session=session, allow_network=not args.offline
            )
            bodies[key] = strip_boilerplate(text)
            books[key] = book
        except CorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        pinned = "ok" if source.expected_sha256 else "UNPINNED -> pin this"
        print(f"\nsource {key}")
        print(f"  title     {book.title!r}")
        print(f"  author    {book.author!r}")
        print(f"  published {book.original_publication!r}")
        print(f"  sha256    {book.sha256}  [{pinned}]")

    print()
    records = []
    for spec in passages:
        source = sources[spec.source]
        try:
            text, span = extract_passage(bodies[spec.source], spec, extraction)
        except CorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        record = passage_record(spec, source, books[spec.source], text, span)
        write_json(passage_dir / f"{spec.id}.json", record)
        records.append(record)
        print(
            f"  {spec.id:<16} {record['word_count']:>5}w  "
            f"chars[{span[0]}:{span[1]}]  sha={record['text_sha256'][:12]}"
        )

    # Two artifacts, split by whether they are deterministic.
    #
    # The manifest describes the corpus and nothing else, so rebuilding must reproduce it
    # byte for byte -- that is what makes `--offline` followed by `git diff --exit-code` a
    # real check on the extractor rather than a ceremony. The run provenance (when, on
    # which machine, at which git sha) is genuinely volatile, so it goes beside the run
    # instead of inside the corpus, and results/provenance/ is gitignored.
    manifest = {
        "corpus_config": str(args.config.as_posix()),
        "config_hash": cfg_hash,
        "extractor_version": EXTRACTOR_VERSION,
        "sources": {key: books[key].identity() for key in sorted(books)},
        "passages": [
            {
                "passage_id": r["passage_id"],
                "author_id": r["author_id"],
                "fame": r["fame"],
                "stratum": r["stratum"],
                "word_count": r["word_count"],
                "text_sha256": r["text_sha256"],
            }
            for r in records
        ],
    }
    write_json(args.out / "manifest.json", manifest)

    provenance = RunProvenance(
        run_id=f"corpus-{cfg_hash}",
        started_at=utc_now(),
        config_hash=cfg_hash,
    ).with_artifacts(
        corpus_config=str(args.config.as_posix()),
        extractor_version=EXTRACTOR_VERSION,
        offline=args.offline,
        sources={key: books[key].as_dict() for key in sorted(books)},
    )
    provenance_path = REPO_ROOT / "results" / "provenance" / f"corpus-{cfg_hash}.json"
    write_json(provenance_path, provenance.as_dict())

    by_author: dict[str, list[int]] = {}
    for r in records:
        by_author.setdefault(r["author_id"], []).append(r["word_count"])
    print("\nwritten:")
    print(f"  {len(records)} passages -> {passage_dir}")
    print(f"  manifest        -> {args.out / 'manifest.json'}")
    print(f"  run provenance  -> {provenance_path}  (gitignored)")
    for author in sorted(by_author):
        counts = by_author[author]
        print(
            f"  {author:<12} {len(counts)} passages, "
            f"{min(counts)}-{max(counts)} words (mean {sum(counts) // len(counts)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
