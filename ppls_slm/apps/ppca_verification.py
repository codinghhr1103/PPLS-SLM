"""PPCA sanity check for the spectral noise estimator."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from ppls_slm.experiment_config import load_config
from ppls_slm.ppls_model import estimate_noise_variance
from ppls_slm.utils.json_io import save_json
from ppls_slm.utils.paths import repo_root, resolve_path


def _random_orthonormal(p: int, r: int, rng: np.random.RandomState) -> np.ndarray:
    a = rng.randn(p, r)
    q, rr = np.linalg.qr(a)
    s = np.sign(np.diag(rr))
    s[s == 0] = 1.0
    return q[:, :r] * s[:r]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PPCA noise variance estimator")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")
    args = parser.parse_args(argv)

    root = repo_root()
    cfg = load_config(resolve_path(root, args.config))

    exp_cfg = cfg.get("experiments", {}).get("ppca_verification", None)
    if not isinstance(exp_cfg, dict):
        raise ValueError("Missing config.experiments.ppca_verification")

    out_dir = resolve_path(root, str(exp_cfg.get("output_dir", "output/ppca_verification")))
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

    rows: list[dict[str, Any]] = []

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
    save_json(out_dir / "summary.json", summary)

    print(f"[OK] Wrote PPCA verification results into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
