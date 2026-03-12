"""(Deprecated) Previously: one-click run for CITE-seq + paper update.

We now keep the workflow as explicit steps (more controllable, easier to debug):

  1) Run Phase 1:
     python scripts/run_citeseq_phase1.py

  2) Sync artifacts:
     python scripts/update_paper_artifacts.py

  3) Generate LaTeX snippets/macros:
     python scripts/generate_paper_tex.py

  4) Compile (optional):
     python scripts/compile_paper_pdflatex.py

This file is kept only for backward compatibility; it does not run anything by default.
"""


from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 80)
    print("RUN  :", " ".join(cmd))
    print("CWD  :", str(cwd))
    print("=" * 80)
    subprocess.check_call(cmd, cwd=str(cwd))


def _pdflatex_compile(paper_dir: Path, main_tex: str = "main.tex", runs: int = 2) -> None:
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        main_tex,
    ]
    for i in range(int(runs)):
        print(f"\n[pdflatex] pass {i+1}/{runs}")
        _run(cmd, cwd=paper_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CITE-seq Phase 1 and update paper (tables + prose + PDF)")
    p.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")
    p.add_argument("--skip-paper", action="store_true", help="Skip pdflatex compilation")
    p.add_argument("--pdflatex-runs", type=int, default=2, help="Number of pdflatex passes (default: 2)")
    return p.parse_args()


def main() -> int:
    # This script is deprecated and intentionally does nothing.
    # Use the explicit step scripts in `scripts/` instead.
    print(__doc__)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
