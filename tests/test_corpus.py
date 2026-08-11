"""Corpus config validation, boilerplate stripping, and passage extraction.

Offline: the fetching half is exercised against synthetic Gutenberg-shaped text, and the
committed passages stand in for a real fetch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from revisionbench.corpus import (
    EXTRACTOR_VERSION,
    CorpusError,
    ExtractionSpec,
    PassageSpec,
    book_url,
    extract_passage,
    parse_config,
    strip_boilerplate,
)
from revisionbench.provenance import sha256_text
from revisionbench.text import word_count

HEADER = """The Project Gutenberg eBook of A Test Book

This eBook is for the use of anyone anywhere in the United States and most
other parts of the world at no cost and with almost no restrictions whatsoever.

Title: A Test Book

Author: A Writer

Release date: January 1, 2020 [eBook #99999]

Language: English

*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***
"""

FOOTER = """
*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***

This and all associated files of various formats will be found in ...
"""


def body_of(paragraphs: list[str]) -> str:
    return "\n\n".join(paragraphs)


def gutenberg_file(paragraphs: list[str]) -> str:
    return HEADER + "\n" + body_of(paragraphs) + FOOTER


SENTENCE = "The wind came off the water in short gusts and the gulls went over calling. "


def paragraph(marker: str, sentences: int = 6) -> str:
    return marker + " " + (SENTENCE * sentences).strip()


class TestBookUrl:
    def test_shape(self) -> None:
        assert book_url(1400) == "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"


class TestStripBoilerplate:
    def test_removes_header_and_footer(self) -> None:
        text = gutenberg_file([paragraph("Alpha."), paragraph("Beta.")])
        body = strip_boilerplate(text)
        assert "Project Gutenberg" not in body
        assert "Alpha." in body and "Beta." in body

    def test_older_this_spelling_is_handled(self) -> None:
        text = gutenberg_file([paragraph("Alpha.")])
        text = text.replace("START OF THE PROJECT", "START OF THIS PROJECT")
        text = text.replace("END OF THE PROJECT", "END OF THIS PROJECT")
        assert "Alpha." in strip_boilerplate(text)

    def test_missing_start_marker_is_fatal(self) -> None:
        """Falling back to the whole file would measure the trademark licence as prose."""
        with pytest.raises(CorpusError, match="no Project Gutenberg START marker"):
            strip_boilerplate("Just some text with no markers at all.")

    def test_missing_end_marker_is_fatal(self) -> None:
        text = gutenberg_file([paragraph("Alpha.")]).replace("*** END OF", "### END OF")
        with pytest.raises(CorpusError, match="no END marker"):
            strip_boilerplate(text)


class TestExtraction:
    extraction = ExtractionSpec(min_words=50, max_words=400, complete_paragraph_slack_frac=0.2)

    def test_span_is_an_exact_slice(self) -> None:
        body = body_of([paragraph("Alpha."), paragraph("Beta."), paragraph("Gamma.")])
        spec = PassageSpec(id="p1", source="s", anchor="Beta.", target_words=60)
        text, (start, end) = extract_passage(body, spec, self.extraction)
        assert body[start:end] == text
        assert text.startswith("Beta.")

    def test_starts_at_the_paragraph_containing_the_anchor(self) -> None:
        # The anchor sits at the end of the first paragraph; extraction must back up to
        # that paragraph's start rather than beginning mid-sentence.
        first = paragraph("Alpha.") + " A quite unique tail sentence sits here."
        body = body_of([first, paragraph("Beta.")])
        spec = PassageSpec(
            id="p1", source="s", anchor="A quite unique tail sentence sits here.", target_words=60
        )
        text, _ = extract_passage(body, spec, self.extraction)
        assert text.startswith("Alpha.")

    def test_word_count_is_within_range(self) -> None:
        body = body_of([paragraph("Alpha.", 8) for _ in range(6)]).replace("Alpha.", "P.", 1)
        spec = PassageSpec(id="p1", source="s", anchor="P.", target_words=200)
        text, _ = extract_passage(body, spec, self.extraction)
        assert self.extraction.min_words <= word_count(text) <= self.extraction.max_words

    def test_absent_anchor_is_fatal(self) -> None:
        body = body_of([paragraph("Alpha.")])
        spec = PassageSpec(id="p1", source="s", anchor="Nowhere in the book", target_words=60)
        with pytest.raises(CorpusError, match="anchor not found"):
            extract_passage(body, spec, self.extraction)

    def test_ambiguous_anchor_is_fatal(self) -> None:
        """A non-unique anchor makes the cut irreproducible; it must not pick the first."""
        body = body_of([paragraph("Alpha."), paragraph("Alpha.")])
        spec = PassageSpec(id="p1", source="s", anchor="Alpha.", target_words=60)
        with pytest.raises(CorpusError, match="occurs 2 times"):
            extract_passage(body, spec, self.extraction)

    def test_empty_anchor_is_fatal(self) -> None:
        spec = PassageSpec(id="p1", source="s", anchor="", target_words=60)
        with pytest.raises(CorpusError, match="must not be empty"):
            extract_passage(body_of([paragraph("Alpha.")]), spec, self.extraction)

    def test_running_off_the_end_is_fatal(self) -> None:
        body = body_of([paragraph("Alpha.", 2)])
        spec = PassageSpec(id="p1", source="s", anchor="Alpha.", target_words=390)
        with pytest.raises(CorpusError, match="short of the"):
            extract_passage(body, spec, self.extraction)

    def test_overshooting_the_max_is_fatal(self) -> None:
        """One enormous sentence must not silently produce an out-of-range passage."""
        giant = "Alpha. " + " ".join(["word"] * 500) + "."
        spec = PassageSpec(id="p1", source="s", anchor="Alpha.", target_words=390)
        with pytest.raises(CorpusError, match="outside the configured range"):
            extract_passage(body_of([giant]), spec, self.extraction)


class TestParseConfig:
    def base(self) -> dict:
        return {
            "version": 1,
            "extraction": {"min_words": 500, "max_words": 1500},
            "sources": [
                {
                    "key": "s1",
                    "ebook_id": 1,
                    "author_id": "a",
                    "author_display": "A",
                    "fame": "famous",
                    "title_hint": "T",
                }
            ],
            "passages": [{"id": "p1", "source": "s1", "anchor": "x", "target_words": 900}],
        }

    def test_valid_config(self) -> None:
        extraction, sources, passages = parse_config(self.base())
        assert extraction.min_words == 500
        assert set(sources) == {"s1"}
        assert passages[0].stratum == "A"

    def test_unknown_key_is_rejected(self) -> None:
        cfg = self.base()
        cfg["passages"][0]["targt_words"] = 900
        with pytest.raises(CorpusError, match="unknown key"):
            parse_config(cfg)

    def test_undeclared_source_is_rejected(self) -> None:
        cfg = self.base()
        cfg["passages"][0]["source"] = "nope"
        with pytest.raises(CorpusError, match="undeclared source"):
            parse_config(cfg)

    def test_duplicate_passage_id_is_rejected(self) -> None:
        cfg = self.base()
        cfg["passages"].append(dict(cfg["passages"][0]))
        with pytest.raises(CorpusError, match="duplicate passage id"):
            parse_config(cfg)

    def test_fame_must_be_one_of_two_values(self) -> None:
        """It drives the plan.md §5 contamination comparison; it cannot be freeform."""
        cfg = self.base()
        cfg["sources"][0]["fame"] = "quite well known"
        with pytest.raises(CorpusError, match="must be 'famous' or 'obscure'"):
            parse_config(cfg)

    def test_target_outside_range_is_rejected(self) -> None:
        cfg = self.base()
        cfg["passages"][0]["target_words"] = 5000
        with pytest.raises(CorpusError, match="outside the configured range"):
            parse_config(cfg)

    def test_inverted_range_is_rejected(self) -> None:
        cfg = self.base()
        cfg["extraction"] = {"min_words": 1500, "max_words": 500}
        with pytest.raises(CorpusError, match="must be below max_words"):
            parse_config(cfg)


class TestCommittedCorpus:
    """The corpus in the repo must match what the config says it should be."""

    def test_config_and_corpus_agree(self, passages: list[dict]) -> None:
        from revisionbench.config import load_config

        repo = Path(__file__).resolve().parent.parent
        cfg = load_config(repo / "configs" / "corpus" / "phase0.yaml")
        extraction, _sources, specs = parse_config(cfg)
        assert {p["passage_id"] for p in passages} == {s.id for s in specs}
        for record in passages:
            assert extraction.min_words <= record["word_count"] <= extraction.max_words
            assert record["extractor_version"] == EXTRACTOR_VERSION

    def test_text_hashes_match_their_records(self, passages: list[dict]) -> None:
        """text_sha256 identifies round 0 for every experiment; it must be right."""
        for record in passages:
            assert sha256_text(record["text"]) == record["text_sha256"]

    def test_every_passage_carries_rights_provenance(self, passages: list[dict]) -> None:
        for record in passages:
            source = record["source"]
            assert source["sha256"] and len(source["sha256"]) == 64
            assert source["url"].startswith("https://www.gutenberg.org/")
            assert source["rights_statement"]

    def test_no_gutenberg_boilerplate_leaked_into_a_passage(self, passages: list[dict]) -> None:
        for record in passages:
            assert "Project Gutenberg" not in record["text"]
            assert "***" not in record["text"]

    def test_committed_artifacts_carry_no_timestamp(self, passages: list[dict]) -> None:
        """Determinism guard: a rebuild must reproduce these byte for byte.

        An embedded build time makes `--offline` followed by `git diff --exit-code`
        useless, and that diff is the cheapest check that a change to corpus.py or text.py
        did not silently move a passage boundary. The volatile half of a run's identity
        lives in results/provenance/ instead.
        """
        repo = Path(__file__).resolve().parent.parent
        manifest = json.loads((repo / "data" / "corpus" / "manifest.json").read_text("utf-8"))
        volatile = ("fetched_at", "started_at", "hostname", "run_id")
        for record in passages:
            assert not (set(record["source"]) & set(volatile)), record["passage_id"]
        assert not (set(manifest) & set(volatile))
        assert "provenance" not in manifest

    def test_manifest_matches_the_passage_files(self, passages: list[dict]) -> None:
        repo = Path(__file__).resolve().parent.parent
        manifest = json.loads((repo / "data" / "corpus" / "manifest.json").read_text("utf-8"))
        listed = {p["passage_id"]: p["text_sha256"] for p in manifest["passages"]}
        assert listed == {p["passage_id"]: p["text_sha256"] for p in passages}

    def test_authors_and_strata_are_as_designed(self, passages: list[dict]) -> None:
        """plan.md §5: two famous authors plus an obscure contamination control."""
        fame = {p["author_id"]: p["fame"] for p in passages}
        assert sorted(fame) == ["hemingway", "richardson", "woolf"]
        assert sorted(a for a, f in fame.items() if f == "famous") == ["hemingway", "woolf"]
        assert [a for a, f in fame.items() if f == "obscure"] == ["richardson"]
