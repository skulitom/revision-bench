"""Every artifact this repo produces must be traceable to the code, configs and models
that produced it (plan.md §11 — "every trial one JSONL line ... model+version, prompt
hash, seed").

The rule of thumb: if a number could change because of it, it belongs in the stamp.
Nothing here needs a GPU, a network connection or an API key, so it is cheap to call and
safe to unit-test.

Adapted from MirrorBench's ``mirrorbench/provenance.py`` (plan.md §10). The torch/CUDA
block is gone — revision-bench's metrics half is pure Python — and is replaced by
:func:`ollama_environment`, because the thing that can silently move a number *here* is
which local model weights answered, not which device they ran on.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

__all__ = [
    "RunProvenance",
    "env_flag",
    "git_describe",
    "package_versions",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "utc_now",
]

_HASH_CHUNK = 1 << 20  # 1 MiB

#: Packages whose version can move a number. Missing ones are recorded as absent rather
#: than skipped, because "numpy was not installed" is itself a fact.
_TRACKED_PACKAGES = ("numpy", "pyyaml", "requests")


def sha256_file(path: str | Path) -> str:
    """Full SHA-256 hex digest of a file's contents, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Full SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of ``text`` encoded as UTF-8.

    Used for prompt hashes and for the identity of a passage revision. Note that this
    hashes the string *as given*: callers that care about newline conventions must
    normalise before calling, because ``"a\\r\\nb"`` and ``"a\\nb"`` are different texts
    to this function and identical texts to a reader. :mod:`revisionbench.corpus`
    normalises on the way in for exactly that reason.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with a ``Z`` suffix, to whole seconds."""
    from datetime import datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_describe(repo: str | Path | None = None) -> dict[str, Any]:
    """Return ``{sha, branch, dirty}`` for the repo, or ``None`` values if unavailable.

    A dirty tree is recorded rather than rejected — blocking a quick exploratory run on
    an uncommitted edit is the kind of friction that gets provenance switched off
    entirely. The flag is what matters: a dirty run is not a reproducible run, and
    aggregation can filter on it.
    """
    root = Path(repo) if repo is not None else Path(__file__).resolve().parent.parent
    out: dict[str, Any] = {"sha": None, "branch": None, "dirty": None}

    def _git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    if sha is None:
        # No git, not a repo, or no commits yet. All three mean "unknown".
        return out
    out["sha"] = sha
    out["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    out["dirty"] = bool(status) if status is not None else None
    return out


def package_versions(names: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each tracked package, ``None`` where not installed."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _python_version() -> str:
    return sys.version.split()[0]


# Bound here rather than referenced inline below: the dataclass has a field named
# `platform`, which shadows the module inside the class body.
_platform_string = platform.platform
_hostname = platform.node


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """The provenance block stamped onto every run (and denormalised onto rows).

    ``run_id`` and ``started_at`` are supplied by the caller rather than generated here,
    so a resumed run re-stamps itself with its original identity instead of silently
    forking into a new one.
    """

    run_id: str
    started_at: str
    config_hash: str
    git: dict[str, Any] = field(default_factory=git_describe)
    packages: dict[str, str | None] = field(default_factory=package_versions)
    python: str = field(default_factory=_python_version)
    platform: str = field(default_factory=_platform_string)
    hostname: str = field(default_factory=_hostname)
    #: Model and data identities, filled in by whoever loads them: reviser tag *and*
    #: digest, judge panel members, corpus config hash, slop lexicon version.
    artifacts: dict[str, Any] = field(default_factory=dict)

    def with_artifacts(self, **kwargs: Any) -> RunProvenance:
        """Return a copy with additional artifact identities recorded."""
        from dataclasses import replace

        return replace(self, artifacts={**self.artifacts, **kwargs})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """One-line human-readable form for logs and run headers."""
        sha = (self.git.get("sha") or "nogit")[:8]
        dirty = "+dirty" if self.git.get("dirty") else ""
        return f"run={self.run_id} git={sha}{dirty} cfg={self.config_hash}"


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable using the usual truthy spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
