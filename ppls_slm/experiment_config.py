"""Centralised experiment configuration loader.

Goal
----
Avoid scattered hyperparameters and conflicting defaults across scripts.
All experiment entry points should read from a single config JSON.

This module provides:
- `load_config(path)`
- `get_experiment_cfg(cfg, name)`
- `require_keys(d, keys, ctx)`

The repository uses `config.json` in the repo root by default, but entry points
should require a `--config` argument to avoid implicit defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config root must be an object/dict, got {type(cfg).__name__}")
    return cfg


def _get_nested(cfg: Mapping[str, Any], keys: Iterable[str], *, ctx: str) -> Any:
    cur: Any = cfg
    path = []
    for k in keys:
        path.append(k)
        if not isinstance(cur, Mapping) or k not in cur:
            raise ConfigError(f"Missing config key: {'.'.join(path)} (context: {ctx})")
        cur = cur[k]
    return cur


def get_experiment_cfg(cfg: Mapping[str, Any], name: str) -> Dict[str, Any]:
    exp = _get_nested(cfg, ["experiments", name], ctx=f"experiments.{name}")
    if not isinstance(exp, dict):
        raise ConfigError(f"experiments.{name} must be an object/dict")
    return dict(exp)


def require_keys(d: Mapping[str, Any], keys: Iterable[str], *, ctx: str) -> None:
    missing = [k for k in keys if k not in d or d.get(k) is None]
    if missing:
        raise ConfigError(f"Missing required keys for {ctx}: {missing}")


def coerce_int(d: MutableMapping[str, Any], key: str, *, ctx: str) -> int:
    if key not in d:
        raise ConfigError(f"Missing key {key} for {ctx}")
    try:
        d[key] = int(d[key])
    except Exception as e:
        raise ConfigError(f"Invalid int for {ctx}.{key}: {d.get(key)!r}") from e
    return int(d[key])


def coerce_float(d: MutableMapping[str, Any], key: str, *, ctx: str) -> float:
    if key not in d:
        raise ConfigError(f"Missing key {key} for {ctx}")
    try:
        d[key] = float(d[key])
    except Exception as e:
        raise ConfigError(f"Invalid float for {ctx}.{key}: {d.get(key)!r}") from e
    return float(d[key])


def coerce_bool(d: MutableMapping[str, Any], key: str, *, ctx: str) -> bool:
    if key not in d:
        raise ConfigError(f"Missing key {key} for {ctx}")
    d[key] = bool(d[key])
    return bool(d[key])


def deep_merge(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge two mapping objects.

    - `override` wins on conflicts.
    - Nested dicts are merged recursively.

    This is intentionally small and dependency-free (no yaml/omegaconf/pydantic).
    """

    merged: Dict[str, Any] = dict(defaults)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(merged.get(k), Mapping):
            merged[k] = deep_merge(merged[k], v)  # type: ignore[arg-type]
        else:
            merged[k] = v
    return merged


def load_config_with_defaults(path: str | Path, *, defaults: Mapping[str, Any]) -> Dict[str, Any]:
    """Load config JSON then apply a deep-merge of `defaults`.

    This keeps entry points tolerant to missing keys while still allowing a single source
    of truth when `config.json` is fully specified.
    """

    cfg = load_config(path)
    return deep_merge(defaults, cfg)

