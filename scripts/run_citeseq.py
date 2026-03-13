"""Run CITE-seq prediction, calibration, and loading export.


Usage (repo root):
  python scripts/run_citeseq.py

This is intentionally a single-purpose wrapper that delegates to:
  python -m ppls_slm.apps.citeseq_prediction --config config.json

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CITE-seq prediction benchmark (prediction + calibration + loadings)")
    p.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cmd = [sys.executable, "-u", "-m", "ppls_slm.apps.citeseq_prediction", "--config", str(config_path)]
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
