"""Append-only JSONL artifacts: the only thing a run is allowed to leave behind.

plan.md §9 makes this Phase 0's acceptance criterion — "runner produces resumable JSONL
provenance per round; metrics reproducible from artifacts alone" — and §11 fixes the
shape: one line per trial, carrying enough to recompute every metric without re-running
a model.

Three properties are load-bearing, and each exists to prevent a specific silent failure:

1. **Crash-safe.** Every row is flushed and ``fsync``-ed before :meth:`JsonlWriter.write`
   returns. A revision loop is a long sequence of slow, expensive model calls; losing the
   buffered tail of a 10-round run to a crash means re-running it, and re-running it with
   a different model state means the rounds are no longer comparable.

2. **Truncation is detected, not skipped.** A process killed mid-write leaves a partial
   final line. :func:`read_jsonl` tolerates exactly that — an unterminated, unparseable
   *last* line — and refuses anything else. The tempting alternative, ``try: parse;
   except: continue``, turns a corrupted file into a smaller valid-looking dataset, which
   is the failure mode this whole repo is organised against.

3. **Resume is by key, not by count.** :func:`resume_index` reads back the identifying
   fields of completed rows so a re-run skips work it has already done. Counting rows
   instead would silently mis-resume the moment any trial is filtered, reordered or
   retried.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

__all__ = [
    "JsonlError",
    "JsonlWriter",
    "read_jsonl",
    "resume_index",
    "write_json",
]


class JsonlError(RuntimeError):
    """A JSONL artifact is malformed, or a row lacks the fields it is required to have."""


class JsonlWriter:
    """Append rows to a JSONL file, durably.

    Use as a context manager::

        with JsonlWriter(path, provenance=prov) as out:
            out.write({"passage_id": "woolf-01", "round": 0, ...})

    Args:
        path: Destination file. Parent directories are created. Opened in append mode,
            so re-running into an existing file continues it rather than truncating it —
            resume depends on that, and truncating an artifact that cost GPU-hours to
            produce is not a recoverable mistake.
        provenance: Optional mapping stamped onto every row under the keys given by
            ``stamp_fields``. Denormalising these onto each row keeps a file
            self-describing even after rows are concatenated across runs.
        stamp_fields: Which provenance keys to copy onto rows.
        fsync: Force the OS to commit each row to disk. Defaults to ``True``. Tests that
            write thousands of throwaway rows may set it ``False``; a real run must not.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        provenance: Mapping[str, Any] | None = None,
        stamp_fields: Sequence[str] = ("run_id", "config_hash"),
        fsync: bool = True,
    ) -> None:
        self.path = Path(path)
        self._fsync = fsync
        self._stamp: dict[str, Any] = {}
        if provenance is not None:
            missing = [f for f in stamp_fields if f not in provenance]
            if missing:
                raise JsonlError(
                    f"provenance is missing stamp field(s): {', '.join(missing)}. Either "
                    f"supply them or narrow stamp_fields; a row stamped with a partial "
                    f"identity cannot be traced back to its run."
                )
            self._stamp = {f: provenance[f] for f in stamp_fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self._rows_written = 0

    @property
    def rows_written(self) -> int:
        """Rows written by *this* writer instance (not rows already in the file)."""
        return self._rows_written

    def write(self, row: Mapping[str, Any]) -> None:
        """Serialise and durably append one row.

        Raises:
            JsonlError: The row is not JSON-serialisable, or one of its own keys collides
                with a provenance stamp field and disagrees with it. A row that carries
                ``run_id`` from somewhere else is either a copy-paste error or a genuine
                cross-run merge; either way, silently overwriting one with the other
                produces an artifact whose provenance is wrong.
        """
        merged = dict(row)
        for key, value in self._stamp.items():
            existing = merged.get(key, value)
            if existing != value:
                raise JsonlError(
                    f"row sets {key!r}={existing!r} but this writer stamps {value!r}; "
                    f"refusing to overwrite provenance"
                )
            merged[key] = value

        try:
            # `ensure_ascii=False` keeps prose readable in the artifact. These rows hold
            # em dashes, curly quotes and the occasional diacritic; escaping every one of
            # them to \uXXXX produces a file no human will ever spot-check, and
            # spot-checking the text a metric was computed from is the main defence
            # against a plausible wrong number. The file is opened UTF-8, so this is safe.
            line = json.dumps(merged, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise JsonlError(
                f"row is not JSON-serialisable ({exc}); convert numpy scalars with "
                f"float()/int() and paths with str() before writing"
            ) from exc

        self._fh.write(line + "\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())
        self._rows_written += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_jsonl(path: str | Path, *, allow_truncated_tail: bool = True) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file.

    Args:
        path: The file to read. A missing file yields nothing — an unstarted run has no
            rows, which is different from an error.
        allow_truncated_tail: Tolerate an unterminated, unparseable final line, which is
            what a process killed mid-write leaves. Set ``False`` when validating an
            artifact that is supposed to be complete (e.g. before publishing results).

    Raises:
        JsonlError: Any line other than a tolerated final fragment fails to parse, or a
            line parses to something that is not a JSON object. The line number is
            included: these files run to thousands of lines and "invalid JSON" without a
            position is not actionable.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return

    raw = file_path.read_bytes()
    if not raw:
        return
    text = raw.decode("utf-8")
    # A final "\n" produces a trailing empty element; anything else in that slot is an
    # unterminated last line, i.e. the crash signature described in the module docstring.
    parts = text.split("\n")
    unterminated = parts[-1] != ""
    lines = parts[:-1] if not unterminated else parts

    for index, line in enumerate(lines, start=1):
        is_last = index == len(lines)
        if not line.strip():
            # Blank lines are noise from manual editing, not data. Skip them rather than
            # failing; they cannot hide a row.
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if is_last and unterminated and allow_truncated_tail:
                # The one tolerated case: a row that was being written when the process
                # died. Everything before it is intact, which is the point of fsync.
                return
            raise JsonlError(
                f"{file_path}:{index}: invalid JSON ({exc.msg} at column {exc.colno}). "
                f"This is not the truncated-tail case, so the file is corrupt rather "
                f"than merely interrupted; do not delete it, and do not resume onto it."
            ) from exc
        if not isinstance(value, dict):
            raise JsonlError(
                f"{file_path}:{index}: expected a JSON object, got {type(value).__name__}"
            )
        yield value


def resume_index(
    path: str | Path,
    key_fields: Sequence[str],
) -> set[tuple[Any, ...]]:
    """Return the set of identifying keys already present in a JSONL artifact.

    A runner builds its full plan, then skips any unit whose key is in this set. That
    makes a resumed run idempotent with respect to work already done, without assuming
    the plan is enumerated in the same order twice.

    Args:
        path: The artifact to read.
        key_fields: The fields that jointly identify a unit of work — for a revision loop
            that is something like ``("passage_id", "arm", "round", "reviser")``.

    Raises:
        ValueError: ``key_fields`` is empty. An empty key makes every row identical and
            would resume a run past everything.
        JsonlError: A row is missing one of ``key_fields``. A row that cannot be
            identified cannot be skipped *or* redone safely, so this is fatal rather
            than ignorable.
    """
    if not key_fields:
        raise ValueError(
            "resume_index needs at least one key field; an empty key matches every row "
            "and would skip the entire run"
        )
    seen: set[tuple[Any, ...]] = set()
    for row_number, row in enumerate(read_jsonl(path), start=1):
        missing = [f for f in key_fields if f not in row]
        if missing:
            raise JsonlError(
                f"{Path(path)}: row {row_number} is missing key field(s) "
                f"{', '.join(missing)}, so it cannot be matched against the run plan. "
                f"Either the artifact was written by a different schema or key_fields "
                f"is wrong."
            )
        seen.add(tuple(row[f] for f in key_fields))
    return seen


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """Write one JSON document, atomically, creating parent directories.

    Atomic because these are the small descriptive files — corpus passages, run headers,
    metric summaries — that other steps read. A half-written one read by the next step is
    a confusing failure a long way from its cause. Write to a sibling temp file, then
    ``os.replace``, which is atomic on both Windows and POSIX when both paths are on the
    same filesystem.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(target)
