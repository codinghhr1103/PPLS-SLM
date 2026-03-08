"""Real-data prediction benchmark on TCGA-BRCA (Section 8.3 extension).

This script evaluates predictive accuracy on the BRCA TCGA paired multi-omics dataset:
- X: gene expression (N=705, p=604)
- Y: protein expression (N=705, q=223)

Protocol
--------
- 5-fold CV with a fixed random seed
- Per-fold standardisation (fit on train, apply to test)
- Methods:
    * PPLS-SLM (multi-start, spectral noise pre-estimation)
    * PPLS-EM
    * PLS regression (PLSR)
    * Ridge regression (RidgeCV)
- Latent dimension grid: r in {3,5,8,10} for PPLS-SLM/EM/PLSR (Ridge uses '-')

Outputs
-------
Writes CSVs under --output_dir (default: results_prediction_brca):
- brca_prediction_per_fold.csv   : fold-level metrics for all methods and r values
- brca_prediction_by_r.csv       : aggregated mean/std for each (method,r)
- brca_prediction_summary.csv    : best-r summary per method (r* minimising CV-MSE)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ppls_slm.apps.data_utils import (
    load_brca_combined_raw,
    select_feature_indices,
    standardize_train_test,
    unstandardize_cov,
    unstandardize_y,
)



from ppls_slm.algorithms import EMAlgorithm, InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction_baselines import compute_regression_metrics, run_plsr_prediction, run_ridge_prediction
from ppls_slm.apps.prediction import (
    _data_driven_theta0,
    compute_credible_intervals,
    empirical_coverage,
    fit_latent_recalibration_head,
    predict_conditional_covariance,
    predict_conditional_mean,
    predict_recalibrated_mean,
    select_shrinkage_alpha_cv,
    select_shrinkage_alpha_nested_cv,
    slm_method_name,
)














def _fit_ppls_slm(

    X_train_s,
    Y_train_s,
    *,
    r: int,
    n_starts: int,
    seed: int,
    max_iter: int,
    optimizer: str = "trust-constr",
    use_noise_preestimation: bool = True,
    gtol: float = 1e-3,
    xtol: float = 1e-3,
    barrier_tol: float = 1e-3,
    initial_constr_penalty: float = 1.0,
    constraint_slack: float = 1e-2,
    verbose: bool = False,
    progress_every: int = 1,
    early_stop_patience: Optional[int] = None,
    early_stop_rel_improvement: Optional[float] = None,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()
    if starting_points:
        starting_points[0] = _data_driven_theta0(X_train_s, Y_train_s, r=r)

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
        early_stop_patience=early_stop_patience,
        early_stop_rel_improvement=early_stop_rel_improvement,
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




def _fit_ppls_em(X_train_s, Y_train_s, *, r: int, n_starts: int, seed: int, max_iter: int, tol: float) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()
    if starting_points:
        starting_points[0] = _data_driven_theta0(X_train_s, Y_train_s, r=r)

    em = EMAlgorithm(p=p, q=q, r=r, max_iter=int(max_iter), tolerance=float(tol))

    res = em.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
    }


def _predict_ppls(X_test_s, params: Dict, *, shrinkage_alpha: float = 1.0):
    y_pred_s = predict_conditional_mean(X_test_s, params, shrinkage_alpha=float(shrinkage_alpha))
    Cov_s = predict_conditional_covariance(params, shrinkage_alpha=float(shrinkage_alpha))
    return y_pred_s, Cov_s



def run_brca_prediction(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r_grid: List[int],
    n_folds: int,
    seed: int,
    slm_n_starts: int,
    slm_max_iter: int,
    em_n_starts: int,
    em_max_iter: int,
    em_tol: float,
    ridge_cv: Optional[int] = None,
    x_top_k: Optional[int] = None,
    y_top_k: Optional[int] = None,
    feature_screening_method: str = "variance",
    feature_screening_mix: float = 0.5,
    slm_optimizer: str = "trust-constr",
    slm_use_noise_preestimation: bool = True,
    slm_gtol: float = 1e-3,
    slm_xtol: float = 1e-3,
    slm_barrier_tol: float = 1e-3,
    slm_initial_constr_penalty: float = 1.0,
    slm_constraint_slack: float = 1e-2,
    slm_verbose: bool = False,
    slm_progress_every: int = 1,
    slm_early_stop_patience: Optional[int] = None,
    slm_early_stop_rel_improvement: Optional[float] = None,
    slm_adaptive_shrinkage: bool = False,
    slm_adaptive_shrinkage_mode: str = "nested_refit",
    slm_adaptive_shrinkage_inner_n_starts: Optional[int] = None,
    slm_adaptive_shrinkage_inner_max_iter: Optional[int] = None,
    slm_adaptive_shrinkage_verbose: bool = False,
    slm_shrinkage_alpha_grid: Optional[List[float]] = None,
    slm_adaptive_shrinkage_folds: int = 5,
    slm_latent_recalibration: bool = False,
    slm_latent_recalibration_cv: Optional[int] = None,
    slm_latent_recalibration_alphas: Optional[List[float]] = None,
    slm_latent_recalibration_include_x: bool = False,
) -> pd.DataFrame:






    import time

    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, int(n_folds))

    slm_method = slm_method_name(slm_optimizer=slm_optimizer, adaptive=slm_adaptive_shrinkage)
    latent_head_alphas = slm_latent_recalibration_alphas or [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    rows: List[Dict] = []



    # Precompute fold splits and (optional) feature indices once per fold,
    # then reuse for all r values (important for speed).
    fold_data: list[dict] = []
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        X_train0, Y_train0 = X[train_idx], Y[train_idx]
        X_test0, Y_test0 = X[test_idx], Y[test_idx]

        x_idx, y_idx = select_feature_indices(
            X_train0,
            Y_train0,
            x_top_k=x_top_k,
            y_top_k=y_top_k,
            method=feature_screening_method,
            hybrid_mix=feature_screening_mix,
        )

        X_train = X_train0 if x_idx is None else X_train0[:, x_idx]
        X_test = X_test0 if x_idx is None else X_test0[:, x_idx]
        Y_train = Y_train0 if y_idx is None else Y_train0[:, y_idx]
        Y_test = Y_test0 if y_idx is None else Y_test0[:, y_idx]


        # Standardize once per fold for PPLS-based methods.
        X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = standardize_train_test(X_train, Y_train, X_test, Y_test)


        fold_data.append(
            {
                "fold_idx": fold_idx,
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "X_train_s": X_train_s,
                "Y_train_s": Y_train_s,
                "X_test_s": X_test_s,
                "sy": sy,
            }
        )

    # Ridge baseline (does not depend on r)
    # NOTE: For BRCA, doing an inner K-fold CV *per output dimension* is extremely expensive.
    # Default is generalized CV (ridge_cv=None).
    for fd in fold_data:
        fold_idx = int(fd["fold_idx"])
        t0 = time.perf_counter()
        print(f"[Ridge] fold {fold_idx + 1}/{n_folds} (RidgeCV cv={ridge_cv})...", flush=True)

        ridge = run_ridge_prediction(fd["X_train"], fd["Y_train"], fd["X_test"], fd["Y_test"], cv=ridge_cv)
        m = ridge["metrics"]

        rows.append(
            {
                "method": "Ridge",
                "r": "-",
                "fold": fold_idx + 1,
                "mse": m.mse,
                "mae": m.mae,
                "r2": m.r2_mean,
                "n_folds": int(n_folds),
                "p": int(fd["X_train"].shape[1]),
                "q": int(fd["Y_train"].shape[1]),
                "x_top_k": x_top_k,
                "y_top_k": y_top_k,
            }
        )

        print(f"[Ridge] fold {fold_idx + 1}/{n_folds} done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Methods that depend on r
    for r in r_grid:
        for fd in fold_data:
            fold_idx = int(fd["fold_idx"])

            # --- PPLS-SLM (optionally: Manifold + adaptive shrinkage) ---
            t_slm0 = time.perf_counter()
            print(
                f"[{slm_method}] r={r} fold {fold_idx + 1}/{n_folds} (starts={slm_n_starts}, max_iter={slm_max_iter})...",
                flush=True,
            )
            slm_params = _fit_ppls_slm(
                fd["X_train_s"],
                fd["Y_train_s"],
                r=r,
                n_starts=slm_n_starts,
                seed=seed + fold_idx,
                max_iter=slm_max_iter,
                optimizer=slm_optimizer,
                use_noise_preestimation=slm_use_noise_preestimation,
                gtol=slm_gtol,
                xtol=slm_xtol,
                barrier_tol=slm_barrier_tol,
                initial_constr_penalty=slm_initial_constr_penalty,
                constraint_slack=slm_constraint_slack,
                verbose=bool(slm_verbose),
                progress_every=int(slm_progress_every),
                early_stop_patience=slm_early_stop_patience,
                early_stop_rel_improvement=slm_early_stop_rel_improvement,
            )

            shrinkage_alpha_slm = 1.0
            if bool(slm_adaptive_shrinkage):
                grid = slm_shrinkage_alpha_grid or [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
                inner_n_starts = int(slm_adaptive_shrinkage_inner_n_starts or max(1, min(2, int(slm_n_starts))))
                inner_max_iter = int(slm_adaptive_shrinkage_inner_max_iter or max(40, min(int(slm_max_iter), 80)))
                mode = str(slm_adaptive_shrinkage_mode).strip().lower()
                if mode in ("nested_refit", "nested", "refit"):
                    print(
                        f"[{slm_method}] r={r} fold {fold_idx + 1}/{n_folds} adaptive-shrinkage nested CV (inner_folds={int(slm_adaptive_shrinkage_folds)}, inner_starts={inner_n_starts}, inner_max_iter={inner_max_iter})...",
                        flush=True,
                    )
                    shrinkage_alpha_slm, _cv, _cov_scale = select_shrinkage_alpha_nested_cv(
                        fd["X_train_s"],
                        fd["Y_train_s"],
                        fit_model_fn=lambda X_in, Y_in, inner_seed: _fit_ppls_slm(
                            X_in,
                            Y_in,
                            r=r,
                            n_starts=inner_n_starts,
                            seed=int(inner_seed),
                            max_iter=inner_max_iter,
                            optimizer=slm_optimizer,
                            use_noise_preestimation=slm_use_noise_preestimation,
                            gtol=slm_gtol,
                            xtol=slm_xtol,
                            barrier_tol=slm_barrier_tol,
                            initial_constr_penalty=slm_initial_constr_penalty,
                            constraint_slack=slm_constraint_slack,
                            verbose=bool(slm_adaptive_shrinkage_verbose),
                            progress_every=int(slm_progress_every),
                            early_stop_patience=slm_early_stop_patience,
                            early_stop_rel_improvement=slm_early_stop_rel_improvement,
                        ),
                        alpha_grid=grid,
                        n_folds=int(slm_adaptive_shrinkage_folds),
                        seed=int(seed + fold_idx),
                        verbose=bool(slm_adaptive_shrinkage_verbose),
                        progress_label=f"[{slm_method}] r={r} fold {fold_idx + 1}/{n_folds}",
                        validation_predictor_fn=(
                            (lambda X_tr_in, Y_tr_in, X_va_in, params_in, a: predict_recalibrated_mean(
                                X_va_in,
                                params_in,
                                fit_latent_recalibration_head(
                                    X_tr_in,
                                    Y_tr_in,
                                    params_in,
                                    shrinkage_alpha=float(a),
                                    ridge_alphas=latent_head_alphas,
                                    cv=slm_latent_recalibration_cv,
                                    include_original_x=bool(slm_latent_recalibration_include_x),
                                ),

                                shrinkage_alpha=float(a),
                            ))
                            if bool(slm_latent_recalibration)
                            else None
                        ),
                    )

                else:
                    shrinkage_alpha_slm, _cv = select_shrinkage_alpha_cv(
                        fd["X_train_s"],
                        fd["Y_train_s"],
                        params=slm_params,
                        alpha_grid=grid,
                        n_folds=int(slm_adaptive_shrinkage_folds),
                        seed=int(seed + fold_idx),
                    )

            if bool(slm_latent_recalibration):
                slm_head = fit_latent_recalibration_head(
                    fd["X_train_s"],
                    fd["Y_train_s"],
                    slm_params,
                    shrinkage_alpha=shrinkage_alpha_slm,
                    ridge_alphas=latent_head_alphas,
                    cv=slm_latent_recalibration_cv,
                    include_original_x=bool(slm_latent_recalibration_include_x),
                )

                y_pred_s = predict_recalibrated_mean(
                    fd["X_test_s"],
                    slm_params,
                    slm_head,
                    shrinkage_alpha=shrinkage_alpha_slm,
                )
            else:
                y_pred_s, _Cov_s = _predict_ppls(fd["X_test_s"], slm_params, shrinkage_alpha=shrinkage_alpha_slm)

            y_pred = unstandardize_y(y_pred_s, fd["sy"])
            m = compute_regression_metrics(fd["Y_test"], y_pred)


            rows.append(
                {
                    "method": slm_method,
                    "r": int(r),
                    "fold": fold_idx + 1,
                    "mse": m.mse,
                    "mae": m.mae,
                    "r2": m.r2_mean,
                    "shrinkage_alpha": float(shrinkage_alpha_slm),
                    "n_folds": int(n_folds),
                    "p": int(fd["X_train"].shape[1]),
                    "q": int(fd["Y_train"].shape[1]),
                    "x_top_k": x_top_k,
                    "y_top_k": y_top_k,
                }
            )
            a_msg = f", shrink_alpha={float(shrinkage_alpha_slm):.3g}" if bool(slm_adaptive_shrinkage) else ""
            print(
                f"[{slm_method}] r={r} fold {fold_idx + 1}/{n_folds} done in {time.perf_counter() - t_slm0:.1f}s{a_msg}",
                flush=True,
            )


            # --- PPLS-EM ---
            t_em0 = time.perf_counter()
            print(f"[PPLS-EM] r={r} fold {fold_idx + 1}/{n_folds} (starts={em_n_starts}, max_iter={em_max_iter})...", flush=True)
            em_params = _fit_ppls_em(fd["X_train_s"], fd["Y_train_s"], r=r, n_starts=em_n_starts, seed=seed + fold_idx, max_iter=em_max_iter, tol=em_tol)
            y_pred_s, _Cov_s = _predict_ppls(fd["X_test_s"], em_params)
            y_pred = unstandardize_y(y_pred_s, fd["sy"])
            m = compute_regression_metrics(fd["Y_test"], y_pred)

            rows.append(
                {
                    "method": "PPLS-EM",
                    "r": int(r),
                    "fold": fold_idx + 1,
                    "mse": m.mse,
                    "mae": m.mae,
                    "r2": m.r2_mean,
                    "n_folds": int(n_folds),
                    "p": int(fd["X_train"].shape[1]),
                    "q": int(fd["Y_train"].shape[1]),
                    "x_top_k": x_top_k,
                    "y_top_k": y_top_k,
                }
            )
            print(f"[PPLS-EM] r={r} fold {fold_idx + 1}/{n_folds} done in {time.perf_counter() - t_em0:.1f}s", flush=True)

            # --- PLSR ---
            t_pls0 = time.perf_counter()
            print(f"[PLSR] r={r} fold {fold_idx + 1}/{n_folds}...", flush=True)
            plsr = run_plsr_prediction(fd["X_train"], fd["Y_train"], fd["X_test"], fd["Y_test"], n_components=r)
            m = plsr["metrics"]
            rows.append(
                {
                    "method": "PLSR",
                    "r": int(r),
                    "fold": fold_idx + 1,
                    "mse": m.mse,
                    "mae": m.mae,
                    "r2": m.r2_mean,
                    "n_folds": int(n_folds),
                    "p": int(fd["X_train"].shape[1]),
                    "q": int(fd["Y_train"].shape[1]),
                    "x_top_k": x_top_k,
                    "y_top_k": y_top_k,
                }
            )
            print(f"[PLSR] r={r} fold {fold_idx + 1}/{n_folds} done in {time.perf_counter() - t_pls0:.1f}s", flush=True)

    return pd.DataFrame(rows)




def _aggregate_by_r(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, r), sub in df.groupby(["method", "r"], sort=False):
        out.append(
            {
                "method": method,
                "r": r,
                "mse_mean": float(sub["mse"].mean()),
                "mse_std": float(sub["mse"].std(ddof=1)),
                "mae_mean": float(sub["mae"].mean()),
                "mae_std": float(sub["mae"].std(ddof=1)),
                "r2_mean": float(sub["r2"].mean()),
                "r2_std": float(sub["r2"].std(ddof=1)),
            }
        )
    return pd.DataFrame(out)


def _select_best_r(df_by_r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df_by_r.groupby("method", sort=False):
        if method == "Ridge":
            best = sub.iloc[0]
        else:
            # r stored as int for these methods
            sub2 = sub.copy()
            sub2["r_int"] = sub2["r"].astype(int)
            best = sub2.sort_values(["mse_mean", "r_int"], ascending=[True, True]).iloc[0]
        rows.append(best.drop(labels=[c for c in ("r_int",) if c in best.index]))
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="BRCA prediction benchmark (5-fold CV)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON (single source of truth)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a very small/fast configuration (for debugging the pipeline)",
    )
    return p.parse_args()



def main():
    args = parse_args()

    from ppls_slm.experiment_config import (
        coerce_bool,
        coerce_float,
        coerce_int,
        get_experiment_cfg,
        load_config,
        require_keys,
    )

    cfg = load_config(args.config)
    brca_cfg = get_experiment_cfg(cfg, "prediction_brca")

    require_keys(
        brca_cfg,
        [
            "thread_limit",
            "brca_data",
            "output_dir",
            "seed",
            "n_folds",
            "r_grid",
            "n_starts",
            "max_iter",
            "em_tol",
        ],
        ctx="experiments.prediction_brca",
    )

    for k in ("thread_limit", "seed", "n_folds", "n_starts", "max_iter"):
        coerce_int(brca_cfg, k, ctx="experiments.prediction_brca")
    coerce_float(brca_cfg, "em_tol", ctx="experiments.prediction_brca")

    output_dir = str(brca_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Runtime thread limiting (helps avoid BLAS thread deadlocks / oversubscription on Windows).
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(brca_cfg["thread_limit"]))
    except Exception:
        pass

    r_grid = [int(x.strip()) for x in str(brca_cfg["r_grid"]).split(",") if x.strip()]

    X, Y = load_brca_combined_raw(str(brca_cfg["brca_data"]))
    print(f"Loaded BRCA: X={X.shape}, Y={Y.shape}")

    ridge_cv_raw = brca_cfg.get("ridge_cv", None)
    ridge_cv = None if ridge_cv_raw in (None, "none", "None", "null") else int(ridge_cv_raw)

    # SLM knobs (defaults here are *deliberately* looser for BRCA to keep runtime manageable)
    slm_verbose = bool(brca_cfg.get("slm_verbose", False))
    slm_progress_every = int(brca_cfg.get("slm_progress_every", 5))
    slm_early_stop_patience = brca_cfg.get("slm_early_stop_patience", 2)
    slm_early_stop_rel_improvement = brca_cfg.get("slm_early_stop_rel_improvement", 0.001)

    slm_optimizer = str(brca_cfg.get("slm_optimizer", "trust-constr"))
    slm_use_noise_preestimation = bool(brca_cfg.get("slm_use_noise_preestimation", True))
    slm_gtol = float(brca_cfg.get("slm_gtol", 0.05))
    slm_xtol = float(brca_cfg.get("slm_xtol", 0.05))
    slm_barrier_tol = float(brca_cfg.get("slm_barrier_tol", 0.05))
    slm_initial_constr_penalty = float(brca_cfg.get("slm_initial_constr_penalty", 1.0))
    slm_constraint_slack = float(brca_cfg.get("slm_constraint_slack", 0.01))

    # Feature screening (no leakage: indices computed on training folds only).
    # If not provided, we auto-enable a moderate top-variance screen for BRCA.
    x_top_k_raw = brca_cfg.get("x_top_k", None)
    x_top_k = None if x_top_k_raw in (None, "none", "None", "null") else int(x_top_k_raw)
    y_top_k_raw = brca_cfg.get("y_top_k", None)
    y_top_k = None if y_top_k_raw in (None, "none", "None", "null") else int(y_top_k_raw)

    if x_top_k is None:
        x_top_k = min(60, int(X.shape[1]))
    if y_top_k is None:
        y_top_k = min(60, int(Y.shape[1]))

    feature_screening_method = str(brca_cfg.get("feature_screening", "hybrid"))
    feature_screening_mix = float(brca_cfg.get("feature_screening_mix", 0.7))

    if args.smoke:

        # Minimal settings that should finish quickly and validate the end-to-end pipeline.
        r_grid = r_grid[:1]
        brca_cfg["n_folds"] = min(int(brca_cfg["n_folds"]), 2)
        brca_cfg["n_starts"] = min(int(brca_cfg["n_starts"]), 2)
        brca_cfg["max_iter"] = min(int(brca_cfg["max_iter"]), 50)
        x_top_k = 30
        y_top_k = 30
        slm_verbose = True
        slm_progress_every = 1
        print("[SMOKE] overriding config for a fast debug run", flush=True)

    slm_n_starts = int(brca_cfg.get("slm_n_starts", brca_cfg["n_starts"]))
    em_n_starts = int(brca_cfg.get("em_n_starts", brca_cfg["n_starts"]))
    slm_max_iter = int(brca_cfg.get("slm_max_iter", brca_cfg["max_iter"]))
    em_max_iter = int(brca_cfg.get("em_max_iter", brca_cfg["max_iter"]))
    slm_adaptive_shrinkage_mode = str(brca_cfg.get("slm_adaptive_shrinkage_mode", "nested_refit"))
    slm_adaptive_shrinkage_inner_n_starts = int(brca_cfg.get("slm_adaptive_shrinkage_inner_n_starts", max(1, min(2, slm_n_starts))))
    slm_adaptive_shrinkage_inner_max_iter = int(brca_cfg.get("slm_adaptive_shrinkage_inner_max_iter", max(40, min(slm_max_iter, 80))))
    slm_adaptive_shrinkage_verbose = bool(brca_cfg.get("slm_adaptive_shrinkage_verbose", False))
    slm_latent_recalibration = bool(brca_cfg.get("slm_latent_recalibration", False))
    slm_latent_recalibration_cv_raw = brca_cfg.get("slm_latent_recalibration_cv", None)
    slm_latent_recalibration_cv = None if slm_latent_recalibration_cv_raw in (None, "none", "None", "null") else int(slm_latent_recalibration_cv_raw)
    slm_latent_recalibration_alphas = brca_cfg.get("slm_latent_recalibration_alphas", None)
    slm_latent_recalibration_include_x = bool(brca_cfg.get("slm_latent_recalibration_include_x", False))

    print(


        "Config (prediction_brca): "
        f"n_folds={int(brca_cfg['n_folds'])}, r_grid={r_grid}, slm_n_starts={slm_n_starts}, em_n_starts={em_n_starts}, slm_max_iter={slm_max_iter}, em_max_iter={em_max_iter}, "
        f"ridge_cv={ridge_cv}, x_top_k={x_top_k}, y_top_k={y_top_k}, feature_screening={feature_screening_method}, feature_screening_mix={feature_screening_mix}, "
        f"slm_optimizer={slm_optimizer}, slm_gtol={slm_gtol}, slm_xtol={slm_xtol}, slm_barrier_tol={slm_barrier_tol}, slm_constraint_slack={slm_constraint_slack}, "
        f"slm_verbose={slm_verbose}, slm_progress_every={slm_progress_every}, "
        f"slm_early_stop_patience={slm_early_stop_patience}, slm_early_stop_rel_improvement={slm_early_stop_rel_improvement}, "
        f"adaptive_mode={slm_adaptive_shrinkage_mode}, adaptive_inner_starts={slm_adaptive_shrinkage_inner_n_starts}, adaptive_inner_max_iter={slm_adaptive_shrinkage_inner_max_iter}, "
        f"latent_recalibration={slm_latent_recalibration}, latent_recalibration_cv={slm_latent_recalibration_cv}, latent_recalibration_include_x={slm_latent_recalibration_include_x}",
        flush=True,
    )





    df = run_brca_prediction(
        X,
        Y,
        r_grid=r_grid,
        n_folds=int(brca_cfg["n_folds"]),
        seed=int(brca_cfg["seed"]),
        slm_n_starts=slm_n_starts,
        slm_max_iter=slm_max_iter,
        em_n_starts=em_n_starts,
        em_max_iter=em_max_iter,
        em_tol=float(brca_cfg["em_tol"]),

        ridge_cv=ridge_cv,
        x_top_k=x_top_k,
        y_top_k=y_top_k,
        feature_screening_method=feature_screening_method,
        feature_screening_mix=feature_screening_mix,
        slm_optimizer=slm_optimizer,
        slm_use_noise_preestimation=slm_use_noise_preestimation,
        slm_gtol=slm_gtol,
        slm_xtol=slm_xtol,
        slm_barrier_tol=slm_barrier_tol,
        slm_initial_constr_penalty=slm_initial_constr_penalty,
        slm_constraint_slack=slm_constraint_slack,
        slm_verbose=slm_verbose,
        slm_progress_every=slm_progress_every,
        slm_early_stop_patience=slm_early_stop_patience,
        slm_early_stop_rel_improvement=slm_early_stop_rel_improvement,
        slm_adaptive_shrinkage=bool(brca_cfg.get("slm_adaptive_shrinkage", False)),
        slm_adaptive_shrinkage_mode=slm_adaptive_shrinkage_mode,
        slm_adaptive_shrinkage_inner_n_starts=slm_adaptive_shrinkage_inner_n_starts,
        slm_adaptive_shrinkage_inner_max_iter=slm_adaptive_shrinkage_inner_max_iter,
        slm_adaptive_shrinkage_verbose=slm_adaptive_shrinkage_verbose,
        slm_shrinkage_alpha_grid=brca_cfg.get("slm_shrinkage_alpha_grid", None),
        slm_adaptive_shrinkage_folds=int(brca_cfg.get("slm_adaptive_shrinkage_folds", 5)),
        slm_latent_recalibration=slm_latent_recalibration,
        slm_latent_recalibration_cv=slm_latent_recalibration_cv,
        slm_latent_recalibration_alphas=slm_latent_recalibration_alphas,
        slm_latent_recalibration_include_x=slm_latent_recalibration_include_x,
    )









    df.to_csv(os.path.join(output_dir, "brca_prediction_per_fold.csv"), index=False)

    df_by_r = _aggregate_by_r(df)
    df_by_r.to_csv(os.path.join(output_dir, "brca_prediction_by_r.csv"), index=False)

    df_best = _select_best_r(df_by_r)
    df_best.to_csv(os.path.join(output_dir, "brca_prediction_summary.csv"), index=False)

    # Diagnostic: selected adaptive shrinkage alphas (when enabled)
    if "shrinkage_alpha" in df.columns:
        sub = df[df["method"].astype(str).str.startswith("PPLS-SLM", na=False)]
        if len(sub):
            alpha_diag = (
                sub[["fold", "r", "method", "shrinkage_alpha"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["method", "r", "fold"])
            )
            alpha_diag.to_csv(os.path.join(output_dir, "brca_selected_shrinkage_alpha.csv"), index=False)

    print("\nSaved:")

    print(f"  {output_dir}/brca_prediction_per_fold.csv")
    print(f"  {output_dir}/brca_prediction_by_r.csv")
    print(f"  {output_dir}/brca_prediction_summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

