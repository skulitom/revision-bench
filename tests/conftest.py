"""Shared fixtures. Everything here is offline: no network, no models, no API key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PASSAGE_DIR = REPO_ROOT / "data" / "corpus" / "passages"


@pytest.fixture(scope="session")
def passages() -> list[dict]:
    """The committed Phase-0 corpus, as a list of passage records.

    These are committed to the repo (see .gitignore) precisely so that the metric tests
    can run against real literary prose without a network call. A synthetic fixture
    cannot answer the question M0-b actually has to answer — whether the stylometry
    separates two real authors at real passage length.
    """
    records = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(PASSAGE_DIR.glob("*.json"))
    ]
    if not records:
        pytest.skip(
            "no corpus passages found; run "
            "`uv run python scripts/fetch_corpus.py --config configs/corpus/phase0.yaml`"
        )
    return records


@pytest.fixture(scope="session")
def passage_texts(passages: list[dict]) -> list[str]:
    return [p["text"] for p in passages]


@pytest.fixture(scope="session")
def labelled_passages(passages: list[dict]) -> list[tuple[str, str]]:
    """``(author_id, text)`` pairs."""
    return [(p["author_id"], p["text"]) for p in passages]
