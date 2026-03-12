"""Compile the LaTeX paper with pdflatex (no latexmk).

Usage (repo root):
  python scripts/compile_paper_pdflatex.py

"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile paper/main.tex via pdflatex")
    p.add_argument("--runs", type=int, default=2, help="Number of pdflatex passes (default: 2)")
    p.add_argument("--main", type=str, default="main.tex", help="Main tex filename (default: main.tex)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paper_dir = repo_root / "paper"
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper dir not found: {paper_dir}")

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        str(args.main),
    ]

    for i in range(int(args.runs)):
        print(f"[pdflatex] pass {i+1}/{int(args.runs)}")
        subprocess.check_call(cmd, cwd=str(paper_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
