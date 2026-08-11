"""Corpus acquisition: fetch source books, strip boilerplate, cut passages, record rights.

plan.md §5 wants ~50 passages of 500–1500 words across three strata, with contamination
controls. This module builds Stratum A (distinctive human prose from Project Gutenberg)
and the provenance record that makes it citable.

Two properties are worth defending against future convenience-minded edits:

**A passage is a byte-exact slice of its source book.** Extraction produces a character
span, and the stored text is ``body[start:end]`` — never a rejoin of tokens or a
"cleaned-up" version. That keeps "is this really what the book says" a checkable question
(the stored ``source.sha256`` plus the span reproduce the text exactly), and it keeps the
round-0 text of every experiment free of any normalisation this repo invented.

**Passages are located by anchor string, not by offset.** Project Gutenberg re-issues
files; an offset silently points somewhere else afterwards, while an anchor either still
matches or fails loudly. The expected SHA-256 is pinned in the config as the second half
of that guard: if the book file changes at all, the fetch stops rather than quietly
re-cutting every passage from a different edition.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

import requests

from revisionbench.provenance import sha256_bytes, sha256_text, utc_now
from revisionbench.text import normalise_newlines, paragraph_spans, sentence_spans, word_count

__all__ = [
    "EXTRACTOR_VERSION",
    "CorpusError",
    "PassageSpec",
    "SourceBook",
    "SourceSpec",
    "book_url",
    "extract_passage",
    "fetch_book",
    "parse_config",
    "strip_boilerplate",
]

#: Bumped whenever a change to :func:`extract_passage` or :func:`strip_boilerplate` could
#: move a span. Stamped onto every passage file, so a corpus cut by two different versions
#: of this code is detectable instead of merely inconsistent.
EXTRACTOR_VERSION = 1

#: Project Gutenberg asks automated clients to identify themselves and to go easy. Both
#: are cheap here: a Phase-0 corpus is a handful of files, fetched once, then cached.
USER_AGENT = "revision-bench/0.0 (research corpus fetch; contact via repository)"

#: Seconds to wait between two network fetches in one run.
FETCH_DELAY_SECONDS = 2.0


class CorpusError(RuntimeError):
    """A source could not be fetched, verified, parsed, or cut as the config asks."""


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One source book, as declared in a corpus config."""

    key: str
    ebook_id: int
    author_id: str
    author_display: str
    #: "famous" or "obscure" — plan.md §5's contamination control. Models plausibly
    #: memorised the famous stratum; comparing the two strata is how that gets tested
    #: rather than assumed.
    fame: str
    title_hint: str
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PassageSpec:
    """One passage to cut, as declared in a corpus config."""

    id: str
    source: str
    anchor: str
    target_words: int
    stratum: str = "A"
    note: str = ""


@dataclass(frozen=True, slots=True)
class ExtractionSpec:
    """Corpus-wide extraction bounds (plan.md §5: 500–1500 words)."""

    min_words: int
    max_words: int
    #: After reaching ``target_words`` at a sentence boundary, extend to the end of the
    #: current paragraph if doing so costs no more than this fraction of extra words.
    #: Passages that end mid-paragraph read as truncated, and a reviser asked to improve a
    #: truncated passage spends edits on the seam rather than on the prose.
    complete_paragraph_slack_frac: float = 0.2


_SOURCE_KEYS = {
    "required": ("key", "ebook_id", "author_id", "author_display", "fame", "title_hint"),
    "optional": ("expected_sha256",),
}
_PASSAGE_KEYS = {
    "required": ("id", "source", "anchor", "target_words"),
    "optional": ("stratum", "note"),
}
_EXTRACTION_KEYS = {
    "required": ("min_words", "max_words"),
    "optional": ("complete_paragraph_slack_frac",),
}


def parse_config(
    cfg: dict[str, Any],
) -> tuple[ExtractionSpec, dict[str, SourceSpec], list[PassageSpec]]:
    """Validate a corpus config and return its three parts.

    Every key is checked by name (see :func:`revisionbench.config.require_keys`). A
    misspelled key in a corpus config is especially cheap to make and expensive to catch:
    the corpus would still build, and the mistake would surface later as an author label
    that quietly defaulted.

    Raises:
        CorpusError: The config is structurally wrong, has duplicate ids, or references a
            source that is not declared.
    """
    from revisionbench.config import ConfigError, require_keys

    try:
        require_keys(
            cfg,
            required=("version", "extraction", "sources", "passages"),
            optional=("description",),
            where="corpus config",
        )
        raw_extraction = cfg["extraction"]
        require_keys(raw_extraction, where="extraction", **_EXTRACTION_KEYS)
        extraction = ExtractionSpec(**raw_extraction)

        sources: dict[str, SourceSpec] = {}
        for index, raw in enumerate(cfg["sources"]):
            require_keys(raw, where=f"sources[{index}]", **_SOURCE_KEYS)
            spec = SourceSpec(**raw)
            if spec.key in sources:
                raise CorpusError(f"duplicate source key: {spec.key!r}")
            if spec.fame not in ("famous", "obscure"):
                raise CorpusError(
                    f"sources[{index}].fame must be 'famous' or 'obscure', got "
                    f"{spec.fame!r}; this label drives the plan.md §5 contamination "
                    f"comparison, so it cannot be freeform"
                )
            sources[spec.key] = spec

        passages: list[PassageSpec] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(cfg["passages"]):
            require_keys(raw, where=f"passages[{index}]", **_PASSAGE_KEYS)
            spec = PassageSpec(**raw)
            if spec.id in seen_ids:
                raise CorpusError(f"duplicate passage id: {spec.id!r}")
            if spec.source not in sources:
                raise CorpusError(
                    f"passages[{index}] ({spec.id}) references undeclared source "
                    f"{spec.source!r}; declared sources are {sorted(sources)}"
                )
            seen_ids.add(spec.id)
            passages.append(spec)
    except ConfigError as exc:
        raise CorpusError(str(exc)) from exc
    except TypeError as exc:
        raise CorpusError(f"corpus config has a malformed entry: {exc}") from exc

    if extraction.min_words >= extraction.max_words:
        raise CorpusError(
            f"extraction.min_words ({extraction.min_words}) must be below max_words "
            f"({extraction.max_words})"
        )
    for spec in passages:
        if not extraction.min_words <= spec.target_words <= extraction.max_words:
            raise CorpusError(
                f"passage {spec.id}: target_words={spec.target_words} is outside the "
                f"configured range {extraction.min_words}–{extraction.max_words}"
            )
    return extraction, sources, passages


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceBook:
    """A fetched book plus everything needed to prove where its text came from."""

    ebook_id: int
    url: str
    sha256: str
    byte_length: int
    fetched_at: str
    title: str | None
    author: str | None
    language: str | None
    release_date: str | None
    #: The original book's publication line, e.g. "New York: Harcourt, Brace & World,
    #: Inc., 1925". This is the date that matters for a corpus stratified by era; the
    #: Gutenberg release date is when the transcription was posted, which is not the same
    #: thing and is a decade-scale error if confused.
    original_publication: str | None
    rights_statement: str | None

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def identity(self) -> dict[str, Any]:
        """The subset that identifies the *book*, with nothing about this fetch in it.

        This is what goes into a committed passage file, and the omission of
        ``fetched_at`` is the point. A committed artifact that embeds when it happened to
        be built cannot be byte-compared against a rebuild, so "regenerate the corpus and
        confirm nothing changed" — the cheapest possible check that the extractor still
        does what it did — stops working. When the fetch happened is real information, but
        it belongs to the run log (``results/provenance/``), not to the description of a
        1925 novel.
        """
        return {
            "ebook_id": self.ebook_id,
            "url": self.url,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "release_date": self.release_date,
            "original_publication": self.original_publication,
            "rights_statement": self.rights_statement,
        }


def book_url(ebook_id: int) -> str:
    """Canonical plain-text UTF-8 URL for a Project Gutenberg ebook id."""
    return f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


def fetch_book(
    spec: SourceSpec,
    cache_dir: str | Path,
    *,
    session: requests.Session | None = None,
    timeout: float = 60.0,
    allow_network: bool = True,
) -> tuple[str, SourceBook]:
    """Return ``(normalised_full_text, provenance)`` for one source book.

    The raw bytes are cached under ``cache_dir`` and reused on later runs, so a rebuild of
    the corpus is offline and free. The cache is keyed by ebook id only; integrity comes
    from the SHA-256 check below, not from the filename.

    Args:
        spec: The source to fetch.
        cache_dir: Where raw book files live (gitignored; they are large and derivable).
        session: Reuse an existing :class:`requests.Session` across several fetches.
        timeout: Per-request timeout in seconds.
        allow_network: When ``False``, a cache miss is an error instead of a download.
            Used by ``--offline`` so a rebuild can be proven not to depend on the network.

    Raises:
        CorpusError: The download failed, the cache is missing under ``allow_network=False``,
            or the file's SHA-256 does not match ``spec.expected_sha256``.
    """
    cache_path = Path(cache_dir) / f"pg{spec.ebook_id}.txt"
    url = book_url(spec.ebook_id)

    if cache_path.is_file():
        raw = cache_path.read_bytes()
        fetched_at = _mtime_iso(cache_path)
    elif not allow_network:
        raise CorpusError(
            f"{spec.key}: no cached copy at {cache_path} and network access is disabled. "
            f"Run without --offline once to populate the cache."
        )
    else:
        raw = _download(url, session=session, timeout=timeout)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
        fetched_at = utc_now()

    digest = sha256_bytes(raw)
    if spec.expected_sha256 is not None and digest != spec.expected_sha256:
        raise CorpusError(
            f"{spec.key}: SHA-256 mismatch for {url}\n"
            f"  expected {spec.expected_sha256}\n"
            f"  got      {digest}\n"
            f"Project Gutenberg has re-issued this file. Do NOT just update the hash: the "
            f"anchors may now point at different text, so re-verify each passage of this "
            f"source before pinning the new digest. The cached copy is at {cache_path}."
        )

    text = normalise_newlines(raw.decode("utf-8", errors="strict"))
    header = _parse_header(text)
    book = SourceBook(
        ebook_id=spec.ebook_id,
        url=url,
        sha256=digest,
        byte_length=len(raw),
        fetched_at=fetched_at,
        title=header.get("title"),
        author=header.get("author"),
        language=header.get("language"),
        release_date=header.get("release date"),
        original_publication=header.get("original publication"),
        rights_statement=header.get("_rights"),
    )
    return text, book


def _download(url: str, *, session: requests.Session | None, timeout: float) -> bytes:
    client = session or requests.Session()
    try:
        response = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CorpusError(f"failed to fetch {url}: {exc}") from exc
    if not response.content:
        raise CorpusError(f"fetched {url} but the body was empty")
    time.sleep(FETCH_DELAY_SECONDS)
    return response.content


def _mtime_iso(path: Path) -> str:
    from datetime import datetime

    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return stamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------------------
# Boilerplate
# --------------------------------------------------------------------------------------

# Both spellings occur: files re-generated in recent years say "THE", older ones "THIS".
# The title is echoed in the marker, hence the `.*`.
_START_RE = re.compile(
    r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_END_RE = re.compile(
    r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_FIELD_RE = re.compile(
    r"^(Title|Author|Language|Release date|Original publication|Credits):[ \t]*(.*)$",
    re.MULTILINE | re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Return the book body, with Project Gutenberg's header and footer removed.

    Raises:
        CorpusError: A marker is missing or the two are in the wrong order. This is fatal
            rather than a fallback to "use the whole file", because the header contains
            the words "Project Gutenberg" a few dozen times and the footer contains the
            entire trademark licence — text which would otherwise land in a corpus of
            literary prose and be measured as if Dickens wrote it.
    """
    start = _START_RE.search(text)
    if start is None:
        raise CorpusError(
            "no Project Gutenberg START marker found; this file is not in the expected "
            "plain-text format"
        )
    end = _END_RE.search(text, start.end())
    if end is None:
        raise CorpusError("Project Gutenberg START marker found but no END marker follows it")
    return text[start.end() : end.start()]


def _parse_header(text: str) -> dict[str, str]:
    """Pull Title/Author/Language/Release date and the rights sentence from the header."""
    start = _START_RE.search(text)
    header = text[: start.start()] if start else text[:4000]
    fields: dict[str, str] = {}
    for match in _FIELD_RE.finditer(header):
        fields[match.group(1).lower()] = match.group(2).strip()

    # The rights sentence is the first paragraph after the title line. Recorded verbatim
    # rather than paraphrased: NOTICE cites it, and paraphrasing a licence is a bad habit.
    for block_start, block_end in paragraph_spans(header):
        block = " ".join(header[block_start:block_end].split())
        if "no cost" in block.lower() or "restrictions" in block.lower():
            fields["_rights"] = block
            break
    return fields


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------


def extract_passage(
    body: str,
    spec: PassageSpec,
    extraction: ExtractionSpec,
) -> tuple[str, tuple[int, int]]:
    """Cut one passage out of a book body.

    The passage starts at the beginning of the paragraph containing ``spec.anchor`` and
    runs to a sentence boundary at or after ``spec.target_words``, then optionally to the
    end of that paragraph (see :attr:`ExtractionSpec.complete_paragraph_slack_frac`).

    Args:
        body: Book text with boilerplate already stripped and newlines normalised.
        spec: The passage to cut.
        extraction: Corpus-wide bounds.

    Returns:
        ``(text, (start, end))`` where ``body[start:end] == text`` exactly.

    Raises:
        CorpusError: The anchor is absent or ambiguous, or the resulting passage falls
            outside the configured word range.
    """
    occurrences = _find_all(body, spec.anchor)
    if not occurrences:
        raise CorpusError(
            f"passage {spec.id}: anchor not found in the book body: {spec.anchor!r}. "
            f"Check for a typographic apostrophe or an em dash — the anchor is matched "
            f"literally against the source, which is not typographically normalised."
        )
    if len(occurrences) > 1:
        raise CorpusError(
            f"passage {spec.id}: anchor occurs {len(occurrences)} times "
            f"(at {occurrences[:5]}...); it must be unique or the cut is not reproducible. "
            f"Lengthen it: {spec.anchor!r}"
        )
    anchor_at = occurrences[0]

    para_spans = paragraph_spans(body)
    start = _containing_span(para_spans, anchor_at)
    if start is None:
        raise CorpusError(f"passage {spec.id}: anchor is not inside any paragraph")

    # Walk sentences from the start of the anchor's paragraph.
    sent_spans = [s for s in sentence_spans(body) if s[0] >= start]
    if not sent_spans:
        raise CorpusError(f"passage {spec.id}: no sentences follow the anchor")

    total = 0
    end = start
    for sent_start, sent_end in sent_spans:
        total += word_count(body[sent_start:sent_end])
        end = sent_end
        if total >= spec.target_words:
            break
    else:
        raise CorpusError(
            f"passage {spec.id}: the book ends after {total} words, short of the "
            f"target of {spec.target_words}; move the anchor earlier"
        )

    end = _complete_paragraph(body, para_spans, end, total, extraction)

    text = body[start:end]
    words_taken = word_count(text)
    if not extraction.min_words <= words_taken <= extraction.max_words:
        raise CorpusError(
            f"passage {spec.id}: extracted {words_taken} words, outside the configured "
            f"range {extraction.min_words}–{extraction.max_words}. A single very long "
            f"sentence or paragraph overshot the target of {spec.target_words}; adjust "
            f"the target or pick a different anchor."
        )
    return text, (start, end)


def _complete_paragraph(
    body: str,
    para_spans: list[tuple[int, int]],
    end: int,
    total: int,
    extraction: ExtractionSpec,
) -> int:
    """Extend ``end`` to the end of its paragraph when the extra words are affordable."""
    para = _containing_span_pair(para_spans, max(end - 1, 0))
    if para is None or end >= para[1]:
        return end
    extra = word_count(body[end : para[1]])
    budget = int(total * extraction.complete_paragraph_slack_frac)
    if extra <= budget and total + extra <= extraction.max_words:
        return para[1]
    return end


def _find_all(haystack: str, needle: str) -> list[int]:
    if not needle:
        raise CorpusError("anchor must not be empty")
    out: list[int] = []
    at = haystack.find(needle)
    while at != -1:
        out.append(at)
        at = haystack.find(needle, at + 1)
    return out


def _containing_span(spans: list[tuple[int, int]], index: int) -> int | None:
    pair = _containing_span_pair(spans, index)
    return pair[0] if pair else None


def _containing_span_pair(spans: list[tuple[int, int]], index: int) -> tuple[int, int] | None:
    for span in spans:
        if span[0] <= index < span[1]:
            return span
    return None


def passage_record(
    spec: PassageSpec,
    source: SourceSpec,
    book: SourceBook,
    text: str,
    span: tuple[int, int],
) -> dict[str, Any]:
    """Build the committed JSON record for one passage.

    ``text_sha256`` identifies the round-0 text of every experiment that uses this
    passage, so a metric row can be tied to the exact string it was computed from without
    embedding the string again.
    """
    return {
        "passage_id": spec.id,
        "stratum": spec.stratum,
        "author_id": source.author_id,
        "author_display": source.author_display,
        "fame": source.fame,
        "title": book.title or source.title_hint,
        "note": spec.note,
        "text": text,
        "text_sha256": sha256_text(text),
        "word_count": word_count(text),
        "char_span": list(span),
        "anchor": spec.anchor,
        "extractor_version": EXTRACTOR_VERSION,
        "source": book.identity(),
    }
