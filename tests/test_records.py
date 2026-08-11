"""JSONL artifacts: durability, truncation handling, and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from revisionbench.records import JsonlError, JsonlWriter, read_jsonl, resume_index, write_json


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"passage_id": "woolf-01", "round": k, "text": "em — dash"} for k in range(3)]
    with JsonlWriter(path, fsync=False) as out:
        for row in rows:
            out.write(row)
        assert out.rows_written == 3
    assert list(read_jsonl(path)) == rows


def test_append_does_not_truncate(tmp_path: Path) -> None:
    """Re-running into an existing artifact must continue it; resume depends on that."""
    path = tmp_path / "rows.jsonl"
    with JsonlWriter(path, fsync=False) as out:
        out.write({"k": 1})
    with JsonlWriter(path, fsync=False) as out:
        out.write({"k": 2})
    assert [r["k"] for r in read_jsonl(path)] == [1, 2]


def test_non_ascii_is_not_escaped(tmp_path: Path) -> None:
    """Escaping em dashes to \\uXXXX makes an artifact no human will spot-check."""
    path = tmp_path / "rows.jsonl"
    with JsonlWriter(path, fsync=False) as out:
        out.write({"text": "a — b “c” it’s"})
    assert "—" in path.read_text(encoding="utf-8")


class TestTruncation:
    def test_truncated_tail_is_tolerated(self, tmp_path: Path) -> None:
        """The crash signature: a row that was mid-write when the process died."""
        path = tmp_path / "rows.jsonl"
        with JsonlWriter(path, fsync=False) as out:
            out.write({"k": 1})
            out.write({"k": 2})
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"k": 3, "text": "half a ro')
        assert [r["k"] for r in read_jsonl(path)] == [1, 2]

    def test_truncated_tail_can_be_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        with JsonlWriter(path, fsync=False) as out:
            out.write({"k": 1})
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"k": 2, "tex')
        with pytest.raises(JsonlError):
            list(read_jsonl(path, allow_truncated_tail=False))

    def test_corrupt_middle_line_is_fatal(self, tmp_path: Path) -> None:
        """The failure this design exists to prevent: a corrupt file read as a smaller one."""
        path = tmp_path / "rows.jsonl"
        path.write_text('{"k": 1}\nNOT JSON\n{"k": 3}\n', encoding="utf-8")
        with pytest.raises(JsonlError, match=r":2:"):
            list(read_jsonl(path))

    def test_non_object_line_is_fatal(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text('{"k": 1}\n[1, 2, 3]\n', encoding="utf-8")
        with pytest.raises(JsonlError, match="expected a JSON object"):
            list(read_jsonl(path))

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text('{"k": 1}\n\n{"k": 2}\n', encoding="utf-8")
        assert [r["k"] for r in read_jsonl(path)] == [1, 2]


def test_missing_file_yields_nothing(tmp_path: Path) -> None:
    """An unstarted run has no rows, which is not an error."""
    assert list(read_jsonl(tmp_path / "absent.jsonl")) == []


class TestProvenanceStamp:
    def test_stamp_is_applied(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        prov = {"run_id": "r1", "config_hash": "abc", "extra": "ignored"}
        with JsonlWriter(path, provenance=prov, fsync=False) as out:
            out.write({"k": 1})
        row = next(iter(read_jsonl(path)))
        assert row["run_id"] == "r1" and row["config_hash"] == "abc"
        assert "extra" not in row

    def test_missing_stamp_field_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(JsonlError, match="missing stamp field"):
            JsonlWriter(tmp_path / "rows.jsonl", provenance={"run_id": "r1"}, fsync=False)

    def test_conflicting_row_value_is_rejected(self, tmp_path: Path) -> None:
        """Silently overwriting one run's id with another's yields lying provenance."""
        prov = {"run_id": "r1", "config_hash": "abc"}
        with (
            JsonlWriter(tmp_path / "rows.jsonl", provenance=prov, fsync=False) as out,
            pytest.raises(JsonlError, match="refusing to overwrite provenance"),
        ):
            out.write({"run_id": "other", "k": 1})

    def test_matching_row_value_is_allowed(self, tmp_path: Path) -> None:
        prov = {"run_id": "r1", "config_hash": "abc"}
        with JsonlWriter(tmp_path / "rows.jsonl", provenance=prov, fsync=False) as out:
            out.write({"run_id": "r1", "k": 1})


def test_unserialisable_row_is_rejected(tmp_path: Path) -> None:
    with (
        JsonlWriter(tmp_path / "rows.jsonl", fsync=False) as out,
        pytest.raises(JsonlError, match="not JSON-serialisable"),
    ):
        out.write({"path": Path("x")})


class TestResumeIndex:
    def test_keys_are_read_back(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        with JsonlWriter(path, fsync=False) as out:
            out.write({"passage_id": "w-01", "arm": "A0", "round": 0})
            out.write({"passage_id": "w-01", "arm": "A0", "round": 1})
        keys = resume_index(path, ("passage_id", "arm", "round"))
        assert keys == {("w-01", "A0", 0), ("w-01", "A0", 1)}

    def test_empty_key_fields_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one key field"):
            resume_index(tmp_path / "rows.jsonl", ())

    def test_row_missing_key_field_is_fatal(self, tmp_path: Path) -> None:
        """A row that cannot be identified can be neither skipped nor safely redone."""
        path = tmp_path / "rows.jsonl"
        with JsonlWriter(path, fsync=False) as out:
            out.write({"passage_id": "w-01"})
        with pytest.raises(JsonlError, match="missing key field"):
            resume_index(path, ("passage_id", "round"))


class TestWriteJson:
    def test_roundtrip_and_no_temp_left(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "doc.json"
        write_json(target, {"b": 1, "a": [1, 2]})
        assert json.loads(target.read_text(encoding="utf-8")) == {"b": 1, "a": [1, 2]}
        assert list(tmp_path.rglob("*.tmp")) == []

    def test_overwrite_is_atomic_in_effect(self, tmp_path: Path) -> None:
        target = tmp_path / "doc.json"
        write_json(target, {"v": 1})
        write_json(target, {"v": 2})
        assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}
