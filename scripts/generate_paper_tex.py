"""Generate LaTeX table snippets + metric macros under `paper/generated/`.

Usage (repo root):
  python scripts/generate_paper_tex.py

This is a tiny wrapper around:
  python scripts/generate_paper_tables.py

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.check_call([sys.executable, "-u", "scripts/generate_paper_tables.py"], cwd=str(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
