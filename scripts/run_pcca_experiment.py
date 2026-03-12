"""Run the PCCA simulation requested for Table 1.

Setting:
  p=q=20, r=3, N=500, M=20 trials
  PCCA specialization: B = I_r, sigma_h^2 = 0

We generate synthetic data using RandomOrthogonalDataGenerator with b_values = ones(r)
(which makes B=I_r) and sigma_h2=0.

We run:
  - BCD-SLM in PCCA mode (model='pcca')
  - EM baseline (unconstrained; estimates full PPLS parameters)

Outputs (under output_dir):
  - mse_table.json: mean/std table strings (x1e2) for MSE_W, MSE_C, MSE_Sigma_t
  - per_trial_mse.csv: per-trial MSE records
  - experiment_summary.json: metadata + raw aggregates

Usage:
  python scripts/run_pcca_experiment.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from ppls_slm.algorithms import EMAlgorithm, InitialPointGenerator
from ppls_slm.bcd_slm import BCDScalarLikelihoodMethod
from ppls_slm.data_generator import RandomOrthogonalDataGenerator
from ppls_slm.experiment import PerformanceMetrics



def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _table_cell_x1e2(mean: float, std: float) -> str:
    return f"{100.0 * float(mean):.2f}±{100.0 * float(std):.2f}"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PCCA simulation (BCD-SLM vs EM)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON (default: config.json)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _read_json((repo_root / args.config).resolve())

    exp_cfg = cfg.get("experiments", {}).get("pcca_simulation", None)
    if not isinstance(exp_cfg, dict):
        raise ValueError("Missing config.experiments.pcca_simulation")

    out_dir = repo_root / str(exp_cfg.get("output_dir", "output/pcca_simulation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    p = int(exp_cfg.get("p", 20))
    q = int(exp_cfg.get("q", 20))
    r = int(exp_cfg.get("r", 3))
    n_samples = int(exp_cfg.get("n_samples", 500))
    n_trials = int(exp_cfg.get("n_trials", 20))
    seed = int(exp_cfg.get("seed", 42))

    sigma_e2 = float(exp_cfg.get("sigma_e2", 0.1))
    sigma_f2 = float(exp_cfg.get("sigma_f2", 0.1))
    sigma_h2 = float(exp_cfg.get("sigma_h2", 0.0))

    b_values = np.asarray(exp_cfg.get("b_values", [1.0] * r), dtype=float)
    if b_values.shape != (r,):
        raise ValueError(f"b_values must have length r={r}, got shape={b_values.shape}")

    n_starts = int(exp_cfg.get("n_starts", cfg.get("algorithms", {}).get("common", {}).get("n_starts", 8)))

    # --- Ground truth (fixed across trials)
    gen = RandomOrthogonalDataGenerator(p=p, q=q, r=r, n_samples=n_samples, random_seed=seed, output_dir=str(out_dir))
    true_params = gen.generate_true_parameters(
        sigma_e2=sigma_e2,
        sigma_f2=sigma_f2,
        sigma_h2=sigma_h2,
        b_values=b_values,
    )

    # --- Starting points (fixed across trials for fairness)
    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    # --- Algorithms
    bcd_cfg = cfg.get("algorithms", {}).get("bcd_slm", {})
    if not isinstance(bcd_cfg, dict):
        bcd_cfg = {}

    bcd = BCDScalarLikelihoodMethod(
        p=p,
        q=q,
        r=r,
        model="pcca",
        max_outer_iter=int(bcd_cfg.get("max_outer_iter", 200)),
        n_cg_steps_W=int(bcd_cfg.get("n_cg_steps_W", 5)),
        n_cg_steps_C=int(bcd_cfg.get("n_cg_steps_C", 5)),
        tolerance=float(bcd_cfg.get("tolerance", 1e-4)),
        use_noise_preestimation=bool(bcd_cfg.get("use_noise_preestimation", True)),
    )

    em_cfg = cfg.get("algorithms", {}).get("em", {})
    if not isinstance(em_cfg, dict):
        em_cfg = {}

    em = EMAlgorithm(
        p=p,
        q=q,
        r=r,
        max_iter=int(em_cfg.get("max_iter", 2000)),
        tolerance=float(em_cfg.get("tolerance", 0.005)),
    )

    metrics_calc = PerformanceMetrics(true_params)

    rows = []

    for trial in range(n_trials):
        trial_seed = seed + trial
        X, Y = gen.generate_samples(true_params, seed=trial_seed)

        t0 = time.time()
        bcd_res = bcd.fit(X, Y, starting_points)
        bcd_time = time.time() - t0

        t1 = time.time()
        em_res = em.fit(X, Y, starting_points)
        em_time = time.time() - t1

        bcd_mse = metrics_calc.compute_mse(bcd_res)
        em_mse = metrics_calc.compute_mse(em_res)

        # Only keep the metrics requested for the PCCA panel.
        for method, mse, runtime in (
            ("BCD-SLM", bcd_mse, bcd_time),
            ("EM", em_mse, em_time),
        ):
            rows.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "method": method,
                    "mse_W": float(mse.get("mse_W", np.nan)),
                    "mse_C": float(mse.get("mse_C", np.nan)),
                    "mse_Sigma_t": float(mse.get("mse_Sigma_t", np.nan)),
                    "runtime_sec": float(runtime),
                }
            )

        print(
            f"trial {trial+1:02d}/{n_trials} | "
            f"BCD-SLM: {bcd_time:.2f}s | EM: {em_time:.2f}s"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_trial_mse.csv", index=False)

    def agg(method: str) -> Dict[str, Dict[str, float]]:
        sub = df[df["method"] == method]
        out: Dict[str, Dict[str, float]] = {}
        for key in ("mse_W", "mse_C", "mse_Sigma_t"):
            vals = sub[key].astype(float).to_numpy()
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return out

    bcd_agg = agg("BCD-SLM")
    em_agg = agg("EM")

    mse_table = {
        "meta": {
            "p": p,
            "q": q,
            "r": r,
            "n_samples": n_samples,
            "n_trials": n_trials,
            "seed": seed,
            "sigma_e2": sigma_e2,
            "sigma_f2": sigma_f2,
            "sigma_h2": sigma_h2,
            "b_values": b_values.tolist(),
            "model": "pcca",
        },
        "bcd_slm": {
            "W": {**bcd_agg["mse_W"], "table_str_x1e2": _table_cell_x1e2(bcd_agg["mse_W"]["mean"], bcd_agg["mse_W"]["std"])},
            "C": {**bcd_agg["mse_C"], "table_str_x1e2": _table_cell_x1e2(bcd_agg["mse_C"]["mean"], bcd_agg["mse_C"]["std"])},
            "Sigma_t": {**bcd_agg["mse_Sigma_t"], "table_str_x1e2": _table_cell_x1e2(bcd_agg["mse_Sigma_t"]["mean"], bcd_agg["mse_Sigma_t"]["std"])},
        },
        "em": {
            "W": {**em_agg["mse_W"], "table_str_x1e2": _table_cell_x1e2(em_agg["mse_W"]["mean"], em_agg["mse_W"]["std"])},
            "C": {**em_agg["mse_C"], "table_str_x1e2": _table_cell_x1e2(em_agg["mse_C"]["mean"], em_agg["mse_C"]["std"])},
            "Sigma_t": {**em_agg["mse_Sigma_t"], "table_str_x1e2": _table_cell_x1e2(em_agg["mse_Sigma_t"]["mean"], em_agg["mse_Sigma_t"]["std"])},
        },
    }

    _write_json(out_dir / "mse_table.json", mse_table)

    exp_summary = {
        "experiment": "pcca_simulation",
        "output_dir": str(out_dir),
        "meta": mse_table["meta"],
        "runtime": {
            "bcd_slm_mean_sec": float(df[df["method"] == "BCD-SLM"]["runtime_sec"].mean()),
            "em_mean_sec": float(df[df["method"] == "EM"]["runtime_sec"].mean()),
        },
        "mse": {
            "bcd_slm": bcd_agg,
            "em": em_agg,
        },
    }
    _write_json(out_dir / "experiment_summary.json", exp_summary)

    print(f"\n[OK] Wrote PCCA results into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
