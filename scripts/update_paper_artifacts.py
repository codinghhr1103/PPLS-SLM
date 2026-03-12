"""Sync experiment outputs into `paper/artifacts/`.

Usage (repo root):
  python scripts/update_paper_artifacts.py

This is a tiny wrapper around:
  python scripts/sync_artifacts.py

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.check_call([sys.executable, "-u", "scripts/sync_artifacts.py"], cwd=str(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
