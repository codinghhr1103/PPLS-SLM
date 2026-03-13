from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object/dict: {p}")
    return data


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)
        f.write("\n")
