"""PPCA sanity check for the spectral noise estimator.

Goal:
  Verify that ppls_slm.ppls_model.estimate_noise_variance(X, r)
  matches the classical PPCA noise MLE of Tipping & Bishop (1999):

    sigma^2 = (1/(p-r)) * sum_{i=1}^{p-r} lambda_i(S)

where lambda_i are the smallest eigenvalues of the centered sample covariance.

We choose p=20, r=3, N=500 so that p/N=0.04 < 0.1 and the code's optional
Marchenko--Pastur correction does NOT trigger; the two estimators should match
numerically to machine precision.

Outputs (under output_dir):
  - per_trial.csv
  - summary.csv
  - summary.json

Usage:
  python scripts/run_ppca_verification.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from ppls_slm.ppls_model import estimate_noise_variance



def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _random_orthonormal(p: int, r: int, rng: np.random.RandomState) -> np.ndarray:
    a = rng.randn(p, r)
    q, rr = np.linalg.qr(a)
    s = np.sign(np.diag(rr))
    s[s == 0] = 1.0
    return q[:, :r] * s[:r]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PPCA noise variance estimator")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _read_json((repo_root / args.config).resolve())

    exp_cfg = cfg.get("experiments", {}).get("ppca_verification", None)
    if not isinstance(exp_cfg, dict):
        raise ValueError("Missing config.experiments.ppca_verification")

    out_dir = repo_root / str(exp_cfg.get("output_dir", "output/ppca_verification"))
    out_dir.mkdir(parents=True, exist_ok=True)

    p = int(exp_cfg.get("p", 20))
    r = int(exp_cfg.get("r", 3))
    n_samples = int(exp_cfg.get("n_samples", 500))
    n_trials = int(exp_cfg.get("n_trials", 20))
    seed = int(exp_cfg.get("seed", 42))
    sigma_e2_true = float(exp_cfg.get("sigma_e2", 0.1))

    theta_t2 = np.array([1.0, 0.65, 0.3], dtype=float)
    if theta_t2.shape != (r,):
        raise ValueError("theta_t2 must have length r")

    rows = []

    for trial in range(n_trials):
        trial_seed = seed + trial
        rng = np.random.RandomState(trial_seed)

        W = _random_orthonormal(p, r, rng)
        T = rng.randn(n_samples, r) @ np.diag(np.sqrt(theta_t2))
        E = np.sqrt(sigma_e2_true) * rng.randn(n_samples, p)
        X = T @ W.T + E

        sigma_e2_spectral = float(estimate_noise_variance(X, r))

        Xc = X - X.mean(axis=0, keepdims=True)
        S = (Xc.T @ Xc) / float(n_samples)
        evals = np.sort(np.linalg.eigvalsh(S))
        sigma_e2_tb = float(np.mean(evals[: p - r]))

        rows.append(
            {
                "trial": trial,
                "seed": trial_seed,
                "sigma_e2_true": sigma_e2_true,
                "sigma_e2_spectral": sigma_e2_spectral,
                "sigma_e2_tb": sigma_e2_tb,
                "abs_diff_spectral_tb": abs(sigma_e2_spectral - sigma_e2_tb),
                "abs_err_spectral": abs(sigma_e2_spectral - sigma_e2_true),
                "abs_err_tb": abs(sigma_e2_tb - sigma_e2_true),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_trial.csv", index=False)

    summary = {
        "p": p,
        "r": r,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "seed": seed,
        "sigma_e2_true": sigma_e2_true,
        "sigma_e2_spectral_mean": float(df["sigma_e2_spectral"].mean()),
        "sigma_e2_spectral_std": float(df["sigma_e2_spectral"].std(ddof=0)),
        "sigma_e2_tb_mean": float(df["sigma_e2_tb"].mean()),
        "sigma_e2_tb_std": float(df["sigma_e2_tb"].std(ddof=0)),
        "abs_diff_mean": float(df["abs_diff_spectral_tb"].mean()),
        "abs_diff_max": float(df["abs_diff_spectral_tb"].max()),
        "abs_err_mean": float(df["abs_err_spectral"].mean()),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    _write_json(out_dir / "summary.json", summary)

    print(f"[OK] Wrote PPCA verification results into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
