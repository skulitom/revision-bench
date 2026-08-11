"""Run provenance stamps."""

from __future__ import annotations

import re
from pathlib import Path

from revisionbench.provenance import (
    RunProvenance,
    env_flag,
    git_describe,
    package_versions,
    sha256_bytes,
    sha256_file,
    sha256_text,
    utc_now,
)


class TestHashes:
    def test_text_and_bytes_agree(self) -> None:
        assert sha256_text("hello") == sha256_bytes(b"hello")

    def test_file_hash_matches_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "f.bin"
        payload = b"x" * (1 << 21)  # larger than the streaming chunk
        path.write_bytes(payload)
        assert sha256_file(path) == sha256_bytes(payload)

    def test_non_ascii_is_hashed_as_utf8(self) -> None:
        assert sha256_text("Iruña") == sha256_bytes("Iruña".encode())

    def test_newline_convention_changes_the_hash(self) -> None:
        """Documented behaviour: callers must normalise before hashing."""
        assert sha256_text("a\r\nb") != sha256_text("a\nb")


class TestUtcNow:
    def test_shape(self) -> None:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", utc_now())


class TestGitDescribe:
    def test_keys_are_always_present(self) -> None:
        described = git_describe()
        assert set(described) == {"sha", "branch", "dirty"}

    def test_non_repo_returns_unknowns_rather_than_raising(self, tmp_path: Path) -> None:
        assert git_describe(tmp_path) == {"sha": None, "branch": None, "dirty": None}


class TestPackageVersions:
    def test_missing_package_is_recorded_as_absent(self) -> None:
        versions = package_versions(("numpy", "definitely-not-installed-xyz"))
        assert versions["numpy"] is not None
        assert versions["definitely-not-installed-xyz"] is None


class TestRunProvenance:
    def test_as_dict_and_summary(self) -> None:
        prov = RunProvenance(run_id="r1", started_at=utc_now(), config_hash="abc123")
        data = prov.as_dict()
        assert data["run_id"] == "r1"
        assert set(data) >= {"git", "packages", "python", "platform", "hostname", "artifacts"}
        assert "run=r1" in prov.summary()
        assert "cfg=abc123" in prov.summary()

    def test_with_artifacts_is_additive_and_does_not_mutate(self) -> None:
        prov = RunProvenance(run_id="r1", started_at=utc_now(), config_hash="abc")
        extended = prov.with_artifacts(reviser="gemma2:2b").with_artifacts(digest="deadbeef")
        assert extended.artifacts == {"reviser": "gemma2:2b", "digest": "deadbeef"}
        assert prov.artifacts == {}


class TestEnvFlag:
    def test_truthy_spellings(self, monkeypatch) -> None:
        for raw in ("1", "true", "TRUE", " yes ", "on"):
            monkeypatch.setenv("RB_TEST_FLAG", raw)
            assert env_flag("RB_TEST_FLAG") is True

    def test_falsy_and_default(self, monkeypatch) -> None:
        monkeypatch.setenv("RB_TEST_FLAG", "0")
        assert env_flag("RB_TEST_FLAG") is False
        monkeypatch.delenv("RB_TEST_FLAG")
        assert env_flag("RB_TEST_FLAG", default=True) is True
