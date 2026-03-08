from __future__ import annotations

from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def repo_root() -> Path:
    """Return repository root as an absolute `Path`.

    Assumes this file lives at `ppls_slm/utils/paths.py`.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(base: Path, path: PathLike) -> Path:
    """Resolve `path` against `base` if it is relative."""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (including parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
