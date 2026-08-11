"""Config loading, strict key checking, and hash stability."""

from __future__ import annotations

from pathlib import Path

import pytest

from revisionbench.config import (
    ConfigError,
    canonical_json,
    config_hash,
    load_config,
    merge,
    require_keys,
)


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class TestLoad:
    def test_plain_load(self, tmp_path: Path) -> None:
        cfg = write(tmp_path / "a.yaml", "version: 1\nname: x\n")
        assert load_config(cfg) == {"version": 1, "name": "x"}

    def test_empty_file_is_an_empty_mapping(self, tmp_path: Path) -> None:
        assert load_config(write(tmp_path / "a.yaml", "")) == {}

    def test_extends_merges_nested(self, tmp_path: Path) -> None:
        write(tmp_path / "base.yaml", "a: 1\nblock:\n  x: 1\n  y: 2\n")
        child = write(tmp_path / "child.yaml", "extends: base.yaml\nblock:\n  y: 99\n")
        assert load_config(child) == {"a": 1, "block": {"x": 1, "y": 99}}

    def test_cycle_is_detected(self, tmp_path: Path) -> None:
        write(tmp_path / "a.yaml", "extends: b.yaml\n")
        write(tmp_path / "b.yaml", "extends: a.yaml\n")
        with pytest.raises(ConfigError, match="circular"):
            load_config(tmp_path / "a.yaml")

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_non_mapping_top_level(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_config(write(tmp_path / "a.yaml", "- 1\n- 2\n"))


def test_merge_replaces_lists_wholesale() -> None:
    """A half-overridden list is never what anyone means."""
    assert merge({"xs": [1, 2, 3]}, {"xs": [9]}) == {"xs": [9]}


class TestRequireKeys:
    def test_accepts_exact_keys(self) -> None:
        require_keys({"a": 1, "b": 2}, required=("a",), optional=("b",), where="block")

    def test_missing_required_named(self) -> None:
        with pytest.raises(ConfigError, match="missing required key"):
            require_keys({"b": 2}, required=("a",), optional=("b",), where="block")

    def test_unknown_key_named(self) -> None:
        """The important direction: a typo silently falls back to a default otherwise."""
        with pytest.raises(ConfigError, match=r"unknown key\(s\): targt_words"):
            require_keys({"targt_words": 900}, optional=("target_words",), where="passages[0]")

    def test_error_lists_the_allowed_set(self) -> None:
        with pytest.raises(ConfigError, match="Allowed keys are: a, b"):
            require_keys({"c": 1}, required=("a",), optional=("b",), where="block")


class TestHash:
    def test_key_order_does_not_change_the_hash(self) -> None:
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_value_change_changes_the_hash(self) -> None:
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_list_order_does_change_the_hash(self) -> None:
        """Passage order is meaningful; two orderings are two different configs."""
        assert config_hash({"xs": [1, 2]}) != config_hash({"xs": [2, 1]})

    def test_length_is_respected(self) -> None:
        assert len(config_hash({"a": 1}, length=64)) == 64
        assert len(config_hash({"a": 1})) == 16

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match=r"length must be in 1\.\.64"):
            config_hash({"a": 1}, length=0)

    def test_canonical_json_is_compact_and_sorted(self) -> None:
        assert (
            canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == '{"a":[1,{"c":3,"d":2}],"b":1}'
        )

    def test_non_ascii_survives_canonicalisation(self) -> None:
        assert canonical_json({"a": "Iruña"}) == '{"a":"Iruña"}'
