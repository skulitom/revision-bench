"""YAML configs are the single source of truth (plan.md §11 — full provenance).

Two jobs live here: turning a config file into a plain resolved dict, and hashing that
resolved dict so every result row traces back to the exact inputs that produced it.

The hash is computed over the *resolved* structure rather than the file bytes, so
reformatting a config or reordering keys does not invalidate previous runs, while any
change to an actual value does.

Adapted from MirrorBench's ``mirrorbench/config.py`` (same author, Apache-2.0; reuse is
sanctioned by plan.md §10), with one addition: :func:`require_keys`. MirrorBench shipped
a defect where a script read flat top-level keys while its config nested them, so the run
silently used hardcoded defaults *and stamped the config's hash* — a result file that
claims provenance it does not have. Rejecting unknown keys by name is the cheap guard.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "EXTENDS_KEY",
    "ConfigError",
    "canonical_json",
    "config_hash",
    "load_config",
    "merge",
    "require_keys",
]

#: Key used by a config to inherit from another config, resolved relative to the
#: inheriting file. Chains are allowed; cycles are an error.
EXTENDS_KEY = "extends"


class ConfigError(RuntimeError):
    """A config file is missing, malformed, inherits in a cycle, or has unknown keys."""


def load_config(path: str | Path, *, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving ``extends:`` chains into one dict.

    Values in the inheriting file win over the ones it extends; nested mappings merge
    key-by-key rather than being replaced wholesale, so a config can override a single
    field of a nested block.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, resolved))
        raise ConfigError(f"circular extends chain: {chain}")
    if not resolved.is_file():
        raise ConfigError(f"config not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config must be a mapping at the top level: {resolved}")

    parent_ref = raw.pop(EXTENDS_KEY, None)
    if parent_ref is None:
        return raw

    if not isinstance(parent_ref, str):
        raise ConfigError(f"{EXTENDS_KEY!r} must be a string path in {resolved}")
    parent_path = (resolved.parent / parent_ref).resolve()
    parent = load_config(parent_path, _seen=(*_seen, resolved))
    return merge(parent, raw)


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Mappings merge; every other type (including lists) is replaced outright. Lists are
    replaced deliberately: a half-overridden passage list is never what anyone means.
    """
    out = dict(base)
    for key, value in override.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = merge(current, value)
        else:
            out[key] = value
    return out


def require_keys(
    section: Mapping[str, Any],
    *,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
    where: str,
) -> None:
    """Validate that ``section`` has exactly the keys it is allowed to have.

    Both directions are checked, and the unknown-key direction is the important one. A
    missing required key usually raises somewhere downstream; a *misspelled* key does
    not — the code falls back to its default, the run completes, and the result row is
    stamped with a config hash that implies the misspelled value was honoured. That is
    provenance which actively lies, so it is rejected here by name.

    Args:
        section: The mapping to check (one config block, not the whole file).
        required: Keys that must be present.
        optional: Keys that may be present.
        where: Human-readable location, used verbatim in the error (e.g. ``"passages[3]"``).

    Raises:
        ConfigError: A required key is absent, or a key outside ``required | optional``
            is present. The message names every offending key and lists what was allowed,
            because "unexpected key" without the allowed set is a guessing game.
    """
    required_set = set(required)
    allowed = required_set | set(optional)
    present = set(section)

    missing = sorted(required_set - present)
    unknown = sorted(present - allowed)
    problems = []
    if missing:
        problems.append(f"missing required key(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown key(s): {', '.join(unknown)}")
    if not problems:
        return
    raise ConfigError(
        f"{where}: {'; '.join(problems)}. Allowed keys are: "
        f"{', '.join(sorted(allowed)) or '(none)'}"
    )


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to JSON with sorted keys and no incidental whitespace.

    This is the exact byte sequence :func:`config_hash` digests, exposed separately so a
    test can assert on it and a debugging session can eyeball it.
    """
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(obj: Any, *, length: int = 16) -> str:
    """SHA-256 of the canonical form of ``obj``, truncated to ``length`` hex chars.

    Truncation keeps result rows readable; 16 hex chars (64 bits) is far beyond collision
    range for the number of configs this project will ever have. Pass ``length=64`` for
    the full digest.
    """
    if not 1 <= length <= 64:
        raise ValueError(f"length must be in 1..64, got {length}")
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:length]


def _normalise(obj: Any) -> Any:
    """Coerce YAML-loaded values into JSON-serialisable ones, deterministically."""
    if isinstance(obj, Mapping):
        return {str(k): _normalise(v) for k, v in obj.items()}
    if isinstance(obj, (str, bytes)):
        return obj.decode("utf-8") if isinstance(obj, bytes) else obj
    if isinstance(obj, Sequence):
        return [_normalise(v) for v in obj]
    if isinstance(obj, (bool, int, float)) or obj is None:
        return obj
    # Dates, paths, and anything else YAML hands back: stringify rather than crash, so an
    # exotic value still contributes to the hash.
    return str(obj)
