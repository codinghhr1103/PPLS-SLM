"""Convenience wrapper for PPCA verification. Equivalent to: python -m ppls_slm.apps.ppca_verification --config config.json"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PPCA noise variance estimator")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-u", "-m", "ppls_slm.apps.ppca_verification", "--config", args.config]
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
