"""Calibration of PPLS-SLM predictive credible intervals on BRCA TCGA data.

This script uses the same 5-fold split as `ppls_slm.apps.brca_prediction` and
computes empirical coverage of element-wise credible intervals for
alpha in {0.05,0.10,0.15,0.20,0.25}.

By default, it reads the best r (CV-MSE optimal) for PPLS-SLM from:
  results_prediction_brca/brca_prediction_summary.csv
and then runs calibration at that r.

Outputs
-------
- brca_calibration_table.csv: rows=alpha, columns=[Expected, Fold1..Fold5, Mean]
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ppls_slm.apps.data_utils import (
    load_brca_combined_raw,
    standardize_train_test,
    top_variance_indices,
    unstandardize_cov,
    unstandardize_y,
)


from ppls_slm.algorithms import InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction import (
    compute_credible_intervals,
    empirical_coverage,
    predict_conditional_covariance,
    predict_conditional_mean,
    select_shrinkage_alpha_cv,
    slm_method_name,
)











def _fit_slm(

    X_train_s,
    Y_train_s,
    *,
    r: int,
    n_starts: int,
    seed: int,
    max_iter: int,
    optimizer: str,
    use_noise_preestimation: bool,
    gtol: float,
    xtol: float,
    barrier_tol: float,
    initial_constr_penalty: float,
    constraint_slack: float,
    verbose: bool,
    progress_every: int,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    slm = ScalarLikelihoodMethod(
        p=p,
        q=q,
        r=r,
        optimizer=str(optimizer),
        max_iter=int(max_iter),
        use_noise_preestimation=bool(use_noise_preestimation),
        gtol=float(gtol),
        xtol=float(xtol),
        barrier_tol=float(barrier_tol),
        initial_constr_penalty=float(initial_constr_penalty),
        constraint_slack=float(constraint_slack),
        verbose=bool(verbose),
        progress_every=int(progress_every),
    )
    res = slm.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
    }




def _best_r_from_summary(path: str) -> int:
    df = pd.read_csv(path)
    methods = df["method"].astype(str)
    sub = df[methods.str.startswith("PPLS-SLM", na=False)]
    if sub.empty:
        raise ValueError(f"PPLS-SLM* row not found in: {path}")
    r = sub.iloc[0]["r"]
    return int(r)



def run_calibration(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r: int,
    n_folds: int,
    seed: int,
    n_starts: int,
    max_iter: int,
    alphas: List[float],
    x_top_k: int | None = None,
    y_top_k: int | None = None,
    slm_optimizer: str = "trust-constr",
    slm_use_noise_preestimation: bool = True,
    slm_gtol: float = 1e-3,
    slm_xtol: float = 1e-3,
    slm_barrier_tol: float = 1e-3,
    slm_initial_constr_penalty: float = 1.0,
    slm_constraint_slack: float = 1e-2,
    slm_verbose: bool = False,
    slm_progress_every: int = 1,
    slm_adaptive_shrinkage: bool = False,
    slm_shrinkage_alpha_grid: List[float] | None = None,
    slm_adaptive_shrinkage_folds: int = 5,
) -> pd.DataFrame:


    """Compute empirical coverage for element-wise credible intervals.

    Efficiency note: fitting the PPLS-SLM model does not depend on alpha, so we fit once per fold
    and then evaluate all alpha values on the same predictive distribution.
    """
    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, int(n_folds))

    slm_method = slm_method_name(slm_optimizer=slm_optimizer, adaptive=slm_adaptive_shrinkage)

    # Fit once per fold and cache (Y_test, y_pred, Cov) on original Y scale.

    fold_cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []

    for fold_idx, test_idx in enumerate(folds):
        print(f"[Calibration] fitting fold {fold_idx + 1}/{n_folds} (r={r}, starts={n_starts}, max_iter={max_iter})...", flush=True)
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])
        X_train0, Y_train0 = X[train_idx], Y[train_idx]
        X_test0, Y_test0 = X[test_idx], Y[test_idx]

        x_idx = top_variance_indices(X_train0, x_top_k)
        y_idx = top_variance_indices(Y_train0, y_top_k)


        X_train = X_train0 if x_idx is None else X_train0[:, x_idx]
        X_test = X_test0 if x_idx is None else X_test0[:, x_idx]
        Y_train = Y_train0 if y_idx is None else Y_train0[:, y_idx]
        Y_test = Y_test0 if y_idx is None else Y_test0[:, y_idx]

        X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = standardize_train_test(X_train, Y_train, X_test, Y_test)


        params = _fit_slm(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            max_iter=max_iter,
            optimizer=slm_optimizer,
            use_noise_preestimation=slm_use_noise_preestimation,
            gtol=slm_gtol,
            xtol=slm_xtol,
            barrier_tol=slm_barrier_tol,
            initial_constr_penalty=slm_initial_constr_penalty,
            constraint_slack=slm_constraint_slack,
            verbose=bool(slm_verbose),
            progress_every=int(slm_progress_every),
        )




        shrinkage_alpha_slm = 1.0
        if bool(slm_adaptive_shrinkage):
            grid = slm_shrinkage_alpha_grid or [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
            shrinkage_alpha_slm, _cv = select_shrinkage_alpha_cv(
                X_train_s,
                Y_train_s,
                params=params,
                alpha_grid=grid,
                n_folds=int(slm_adaptive_shrinkage_folds),
                seed=int(seed + fold_idx),
            )

        y_pred_s = predict_conditional_mean(X_test_s, params, shrinkage_alpha=shrinkage_alpha_slm)
        Cov_s = predict_conditional_covariance(params, shrinkage_alpha=shrinkage_alpha_slm)

        y_pred = unstandardize_y(y_pred_s, sy)
        Cov = unstandardize_cov(Cov_s, sy)


        fold_cache.append((Y_test, y_pred, Cov, int(X_train.shape[1]), int(Y_train.shape[1]), float(shrinkage_alpha_slm)))




    shrinkage_alphas = [float(x[5]) for x in fold_cache]
    shrinkage_alpha_mean = float(np.mean(shrinkage_alphas)) if shrinkage_alphas else float("nan")
    shrinkage_alpha_std = float(np.std(shrinkage_alphas, ddof=1)) if len(shrinkage_alphas) > 1 else 0.0

    # Table with rows alpha, cols expected + fold1..foldK + mean
    rows = []

    for a in alphas:
        row = {
            "method": slm_method,
            "Alpha": float(a),
            "Expected Coverage": f"{100.0 * (1.0 - float(a)):.2f}%",
            "shrinkage_alpha_mean": shrinkage_alpha_mean,
            "shrinkage_alpha_std": shrinkage_alpha_std,
            "n_folds": int(n_folds),
            "p": int(fold_cache[0][3]) if fold_cache else None,
            "q": int(fold_cache[0][4]) if fold_cache else None,
            "x_top_k": x_top_k,
            "y_top_k": y_top_k,
        }


        covs = []
        for fold_idx, (Y_test, y_pred, Cov, _p, _q, _shrink_a) in enumerate(fold_cache):

            lower, upper = compute_credible_intervals(y_pred, Cov, alpha=float(a))
            cov_pct = 100.0 * empirical_coverage(Y_test, lower, upper)
            covs.append(float(cov_pct))
            row[f"Fold {fold_idx + 1}"] = f"{cov_pct:.2f}%"

        row["Mean"] = f"{np.mean(covs):.2f}%"
        rows.append(row)

    return pd.DataFrame(rows)



def parse_args():
    p = argparse.ArgumentParser(description="BRCA calibration (PPLS-SLM)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON (single source of truth)")
    return p.parse_args()


def main():
    args = parse_args()

    from ppls_slm.experiment_config import (
        coerce_float,
        coerce_int,
        get_experiment_cfg,
        load_config,
        require_keys,
    )

    cfg = load_config(args.config)
    calib_cfg = get_experiment_cfg(cfg, "calibration_brca")

    require_keys(
        calib_cfg,
        [
            "thread_limit",
            "brca_data",
            "output_dir",
            "prediction_summary",
            "seed",
            "n_folds",
            "n_starts",
            "max_iter",
        ],
        ctx="experiments.calibration_brca",
    )


    for k in ("thread_limit", "seed", "n_folds", "n_starts", "max_iter"):
        coerce_int(calib_cfg, k, ctx="experiments.calibration_brca")

    output_dir = str(calib_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Runtime thread limiting
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(calib_cfg["thread_limit"]))
    except Exception:
        pass

    # r=null means "read best r from summary"
    r_val = calib_cfg.get("r")
    if r_val is None:
        r = _best_r_from_summary(str(calib_cfg["prediction_summary"]))
    else:
        r = int(r_val)

    X, Y = load_brca_combined_raw(str(calib_cfg["brca_data"]))
    print(f"Loaded BRCA: X={X.shape}, Y={Y.shape}")
    print(f"Using r={r} for calibration")

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Feature screening defaults (auto-enable for BRCA if not specified).
    x_top_k_raw = calib_cfg.get("x_top_k", None)
    x_top_k = None if x_top_k_raw in (None, "none", "None", "null") else int(x_top_k_raw)
    y_top_k_raw = calib_cfg.get("y_top_k", None)
    y_top_k = None if y_top_k_raw in (None, "none", "None", "null") else int(y_top_k_raw)

    if x_top_k is None:
        x_top_k = min(60, int(X.shape[1]))
    if y_top_k is None:
        y_top_k = min(60, int(Y.shape[1]))

    # SLM knobs (defaults are looser for BRCA runtime).
    slm_optimizer = str(calib_cfg.get("slm_optimizer", "trust-constr"))
    slm_use_noise_preestimation = bool(calib_cfg.get("slm_use_noise_preestimation", True))
    slm_gtol = float(calib_cfg.get("slm_gtol", 0.05))
    slm_xtol = float(calib_cfg.get("slm_xtol", 0.05))
    slm_barrier_tol = float(calib_cfg.get("slm_barrier_tol", 0.05))
    slm_initial_constr_penalty = float(calib_cfg.get("slm_initial_constr_penalty", 1.0))
    slm_constraint_slack = float(calib_cfg.get("slm_constraint_slack", 0.01))

    slm_verbose = bool(calib_cfg.get("slm_verbose", False))
    slm_progress_every = int(calib_cfg.get("slm_progress_every", 5))

    print(
        "Config (calibration_brca): "
        f"n_folds={int(calib_cfg['n_folds'])}, n_starts={int(calib_cfg['n_starts'])}, max_iter={int(calib_cfg['max_iter'])}, "
        f"x_top_k={x_top_k}, y_top_k={y_top_k}, "
        f"slm_optimizer={slm_optimizer}, slm_gtol={slm_gtol}, slm_xtol={slm_xtol}, slm_barrier_tol={slm_barrier_tol}, slm_constraint_slack={slm_constraint_slack}, "
        f"slm_verbose={slm_verbose}, slm_progress_every={slm_progress_every}",
        flush=True,
    )


    table = run_calibration(
        X,
        Y,
        r=r,
        n_folds=int(calib_cfg["n_folds"]),
        seed=int(calib_cfg["seed"]),
        n_starts=int(calib_cfg["n_starts"]),
        max_iter=int(calib_cfg["max_iter"]),
        alphas=alphas,
        x_top_k=x_top_k,
        y_top_k=y_top_k,
        slm_optimizer=slm_optimizer,
        slm_use_noise_preestimation=slm_use_noise_preestimation,
        slm_gtol=slm_gtol,
        slm_xtol=slm_xtol,
        slm_barrier_tol=slm_barrier_tol,
        slm_initial_constr_penalty=slm_initial_constr_penalty,
        slm_constraint_slack=slm_constraint_slack,
        slm_verbose=slm_verbose,
        slm_progress_every=slm_progress_every,
        slm_adaptive_shrinkage=bool(calib_cfg.get("slm_adaptive_shrinkage", False)),
        slm_shrinkage_alpha_grid=calib_cfg.get("slm_shrinkage_alpha_grid", None),
        slm_adaptive_shrinkage_folds=int(calib_cfg.get("slm_adaptive_shrinkage_folds", 5)),
    )





    out_csv = os.path.join(output_dir, "brca_calibration_table.csv")
    table.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

