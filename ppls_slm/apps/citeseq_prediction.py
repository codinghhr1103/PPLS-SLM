"""CITE-seq protein imputation on PBMC (Hao et al., 2021).

This app mirrors `ppls_slm.apps.brca_prediction` but targets large-scale paired
single-cell RNA/ADT data.

Outputs (under experiments.citeseq_prediction.output_dir)
--------------------------------------------------------
- citeseq_prediction_per_fold.csv
- citeseq_prediction_by_r.csv
- citeseq_prediction_summary.csv
- citeseq_calibration_per_fold.csv
- citeseq_calibration_summary.csv
- citeseq_selected_shrinkage_alpha.csv (diagnostic)
- citeseq_loadings_top.csv (Phase 2; best-effort)


"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


import numpy as np
import pandas as pd

from ppls_slm.bcd_slm import BCDScalarLikelihoodMethod

from ppls_slm.apps.data_utils import load_citeseq_data, standardize_train_test, unstandardize_cov, unstandardize_y
from ppls_slm.apps.prediction_baselines import compute_regression_metrics, run_plsr_prediction, run_ridge_prediction
from ppls_slm.apps.prediction_common import (
    aggregate_prediction_by_r,
    build_cv_folds,
    build_starting_points,
    fit_ppls_em,
    fit_ppls_slm,
    select_best_r,
)
from ppls_slm.apps.prediction import (

    _data_driven_theta0,
    compute_credible_intervals,
    empirical_coverage,
    estimate_predictive_covariance_scale_cv,
    fit_latent_recalibration_head,
    predict_conditional_covariance,
    predict_conditional_mean,
    predict_recalibrated_mean,
    select_shrinkage_alpha_cv,
    select_shrinkage_alpha_nested_cv,
    slm_method_name,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Timeout helper (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────


class TimeoutError(RuntimeError):
    pass


@dataclass
class _Heartbeat:
    """Background heartbeat printer for long in-process stages."""

    label: str
    heartbeat_sec: float = 60.0

    def __post_init__(self):
        self.heartbeat_sec = max(1.0, float(self.heartbeat_sec))
        self._stop = threading.Event()
        self._t0 = None
        self._thr = None

    def __enter__(self):
        self._t0 = time.time()

        def _loop():
            # First sleep, then print, so short stages don't spam.
            while not self._stop.wait(self.heartbeat_sec):
                elapsed = time.time() - float(self._t0)
                print(f"  heartbeat: {self.label}; elapsed {elapsed/60:.1f} min", flush=True)

        self._thr = threading.Thread(target=_loop, name="heartbeat", daemon=True)
        self._thr.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        try:
            if self._thr is not None:
                self._thr.join(timeout=1.0)
        except Exception:
            pass
        return False


def _timeout_worker(q_out, f, a, k):
    """Top-level worker for Windows spawn compatibility."""

    try:
        out = f(*a, **k)
        q_out.put(("ok", out))
    except Exception as e:
        q_out.put(("err", repr(e)))


def _run_in_subprocess_with_timeout(

    func,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    timeout_sec: float,
    heartbeat_sec: float = 60.0,
    label: str = "job",
):
    """Run `func(*args, **kwargs)` in a fresh process with a wall-clock timeout.

    NOTE: On Windows (spawn start method), the process target must be picklable.
    We therefore use a top-level helper (`_timeout_worker`) rather than a nested function.

    We also print a periodic heartbeat while waiting, so long runs don't look "stuck".
    """

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue(maxsize=1)

    p = ctx.Process(target=_timeout_worker, args=(q, func, args, kwargs))
    p.daemon = True
    p.start()

    t0 = time.time()
    last_beat = t0
    hb = max(1.0, float(heartbeat_sec))
    timeout = float(timeout_sec)

    while True:
        if not p.is_alive():
            break
        elapsed = time.time() - t0
        remaining = timeout - elapsed
        if remaining <= 0:
            break
        p.join(timeout=min(hb, remaining))
        now = time.time()
        if p.is_alive() and (now - last_beat) >= hb:
            print(f"  heartbeat: {label} still running; elapsed {elapsed/60:.1f} min", flush=True)
            last_beat = now

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        raise TimeoutError(f"Timed out after {timeout_sec} sec")

    if q.empty():
        raise RuntimeError("Subprocess finished but returned no result")

    status, payload = q.get()
    if status == "ok":
        return payload
    raise RuntimeError(f"Subprocess error: {payload}")




# ─────────────────────────────────────────────────────────────────────────────
#  Fit wrappers (match dict keys across solvers)
# ─────────────────────────────────────────────────────────────────────────────





def _fit_ppls_bcd(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    max_outer_iter: int = 200,
    n_cg_steps_W: int = 5,
    n_cg_steps_C: int = 5,
    tolerance: float = 1e-4,
    use_noise_preestimation: bool = True,
    early_stop_patience: Optional[int] = None,
    early_stop_rel_improvement: Optional[float] = None,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    starting_points = build_starting_points(
        X_train_s,
        Y_train_s,
        r=r,
        n_starts=n_starts,
        seed=seed,
        use_data_driven_init=True,
        data_driven_init_fn=_data_driven_theta0,
    )


    bcd = BCDScalarLikelihoodMethod(
        p=p,
        q=q,
        r=r,
        model="ppls",
        max_outer_iter=int(max_outer_iter),
        n_cg_steps_W=int(n_cg_steps_W),
        n_cg_steps_C=int(n_cg_steps_C),
        tolerance=float(tolerance),
        use_noise_preestimation=bool(use_noise_preestimation),
        early_stop_patience=early_stop_patience,
        early_stop_rel_improvement=early_stop_rel_improvement,
    )

    res = bcd.fit(X_train_s, Y_train_s, starting_points)

    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
    }








def run_citeseq_prediction(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r_grid: Sequence[int],
    n_folds: int = 5,
    seed: int = 42,
    slm_n_starts: int = 4,
    slm_max_iter: int = 200,
    slm_progress_every: int = 20,
    slm_verbose: bool = False,
    heartbeat_sec: float = 60.0,
    em_n_starts: int = 2,
    em_max_iter: int = 200,
    em_tol: float = 1e-4,
    slm_optimizer: str = "manifold",
    slm_use_noise_preestimation: bool = True,
    slm_gtol: float = 1e-2,
    slm_xtol: float = 1e-2,
    slm_barrier_tol: float = 1e-2,
    slm_constraint_slack: float = 5e-3,
    slm_early_stop_patience: Optional[int] = None,
    slm_early_stop_rel_improvement: Optional[float] = None,
    slm_adaptive_shrinkage: bool = True,
    slm_adaptive_shrinkage_mode: str = "nested_refit",
    slm_adaptive_shrinkage_inner_n_starts: Optional[int] = None,
    slm_adaptive_shrinkage_inner_max_iter: Optional[int] = None,
    slm_adaptive_shrinkage_verbose: bool = False,
    slm_shrinkage_alpha_grid: Optional[Sequence[float]] = None,
    slm_adaptive_shrinkage_folds: int = 5,
    slm_latent_recalibration: bool = True,
    slm_latent_recalibration_cv: Optional[int] = 5,
    slm_latent_recalibration_alphas: Optional[Sequence[float]] = None,
    slm_latent_recalibration_include_x: bool = True,
    alphas_for_coverage: Sequence[float] = (0.05, 0.10, 0.20),
    em_timeout_sec: float = 2 * 3600,
) -> Tuple[pd.DataFrame, pd.DataFrame]:


    """K-fold CV prediction + calibration on CITE-seq data.

    Returns
    -------
    df_pred: per-fold prediction metrics (mse/mae/r2)
    df_cov : per-fold coverage metrics for probabilistic methods
    """

    N = int(X.shape[0])
    folds = build_cv_folds(n_samples=N, n_folds=int(n_folds), seed=int(seed))


    slm_method = slm_method_name(slm_optimizer=slm_optimizer, adaptive=slm_adaptive_shrinkage)
    latent_head_alphas = list(slm_latent_recalibration_alphas or [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0, 100.0])

    # Prepare fold data (no feature screening here; data loader already applies HVGs)
    fold_data: List[Dict[str, Any]] = []
    for fold_idx in range(int(n_folds)):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(int(n_folds)) if j != fold_idx])

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_test = X[test_idx]
        Y_test = Y[test_idx]

        X_train_s, Y_train_s, X_test_s, Y_test_s, sx, sy = standardize_train_test(X_train, Y_train, X_test, Y_test)
        fold_data.append(
            {
                "fold_idx": int(fold_idx),
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "X_train_s": X_train_s,
                "Y_train_s": Y_train_s,
                "X_test_s": X_test_s,
                "Y_test_s": Y_test_s,
                "sx": sx,
                "sy": sy,
            }
        )

    pred_rows: List[Dict[str, Any]] = []
    cov_rows: List[Dict[str, Any]] = []

    # Ridge baseline (no r)
    for fd in fold_data:
        fold_idx = int(fd["fold_idx"])
        t0 = time.perf_counter()
        ridge = run_ridge_prediction(fd["X_train"], fd["Y_train"], fd["X_test"], fd["Y_test"], cv=None)
        m = ridge["metrics"]
        pred_rows.append(
            {
                "method": "Ridge",
                "r": "-",
                "fold": fold_idx + 1,
                "mse": m.mse,
                "mae": m.mae,
                "r2": m.r2_mean,
                "time_sec": float(time.perf_counter() - t0),
                "N": int(fd["X_train"].shape[0] + fd["X_test"].shape[0]),
                "p": int(fd["X_train"].shape[1]),
                "q": int(fd["Y_train"].shape[1]),
            }
        )

    # Methods that depend on r
    for r in [int(x) for x in r_grid]:
        for fd in fold_data:
            fold_idx = int(fd["fold_idx"])

            # --- SLM (with augmentations) ---
            t0 = time.perf_counter()
            print(f"[{slm_method}] r={r} fold {fold_idx + 1}/{n_folds}...", flush=True)

            # SLM fit can be long; print a periodic heartbeat even when solver logs are off.
            with _Heartbeat(
                label=f"{slm_method} r={r} fold {fold_idx + 1}/{n_folds} (SLM fit)",
                heartbeat_sec=float(heartbeat_sec),
            ):
                slm_params = fit_ppls_slm(
                    fd["X_train_s"],
                    fd["Y_train_s"],
                    r=int(r),
                    n_starts=int(slm_n_starts),
                    seed=int(seed + fold_idx),
                    max_iter=int(slm_max_iter),
                    optimizer=str(slm_optimizer),
                    use_noise_preestimation=bool(slm_use_noise_preestimation),
                    gtol=float(slm_gtol),
                    xtol=float(slm_xtol),
                    barrier_tol=float(slm_barrier_tol),
                    initial_constr_penalty=1.0,
                    constraint_slack=float(slm_constraint_slack),
                    verbose=bool(slm_verbose),
                    progress_every=int(max(1, slm_progress_every)),
                    early_stop_patience=slm_early_stop_patience,
                    early_stop_rel_improvement=slm_early_stop_rel_improvement,
                    data_driven_init_fn=_data_driven_theta0,
                )



            shrinkage_alpha = 1.0
            cov_scale = 1.0


            if bool(slm_adaptive_shrinkage):
                grid = list(slm_shrinkage_alpha_grid or [0.3, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0])
                inner_n_starts = int(slm_adaptive_shrinkage_inner_n_starts or max(1, min(2, int(slm_n_starts))))
                inner_max_iter = int(slm_adaptive_shrinkage_inner_max_iter or max(40, min(int(slm_max_iter), 80)))
                mode = str(slm_adaptive_shrinkage_mode).strip().lower()

                if mode in ("nested_refit", "nested", "refit"):
                    with _Heartbeat(
                        label=f"{slm_method} r={r} fold {fold_idx + 1}/{n_folds} (adaptive shrinkage CV)",
                        heartbeat_sec=float(heartbeat_sec),
                    ):
                        shrinkage_alpha, _cv, cov_scale = select_shrinkage_alpha_nested_cv(
                            fd["X_train_s"],
                            fd["Y_train_s"],
                            fit_model_fn=lambda X_in, Y_in, inner_seed: fit_ppls_slm(
                                X_in,
                                Y_in,
                                r=int(r),
                                n_starts=int(inner_n_starts),
                                seed=int(inner_seed),
                                max_iter=int(inner_max_iter),
                                optimizer=str(slm_optimizer),
                                use_noise_preestimation=bool(slm_use_noise_preestimation),
                                gtol=float(slm_gtol),
                                xtol=float(slm_xtol),
                                barrier_tol=float(slm_barrier_tol),
                                initial_constr_penalty=1.0,
                                constraint_slack=float(slm_constraint_slack),
                                verbose=bool(slm_adaptive_shrinkage_verbose),
                                progress_every=999999,
                                early_stop_patience=slm_early_stop_patience,
                                early_stop_rel_improvement=slm_early_stop_rel_improvement,
                                data_driven_init_fn=_data_driven_theta0,
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
                    shrinkage_alpha, _cv = select_shrinkage_alpha_cv(
                        fd["X_train_s"],
                        fd["Y_train_s"],
                        params=slm_params,
                        alpha_grid=grid,
                        n_folds=int(slm_adaptive_shrinkage_folds),
                        seed=int(seed + fold_idx),
                    )
                    with _Heartbeat(
                        label=f"{slm_method} r={r} fold {fold_idx + 1}/{n_folds} (covariance scale CV)",
                        heartbeat_sec=float(heartbeat_sec),
                    ):
                        cov_scale = estimate_predictive_covariance_scale_cv(
                            fd["X_train_s"],
                            fd["Y_train_s"],
                            params=slm_params,
                            shrinkage_alpha=float(shrinkage_alpha),
                            n_folds=int(slm_adaptive_shrinkage_folds),
                            seed=int(seed + fold_idx),
                        )
            else:
                with _Heartbeat(
                    label=f"{slm_method} r={r} fold {fold_idx + 1}/{n_folds} (covariance scale CV)",
                    heartbeat_sec=float(heartbeat_sec),
                ):
                    cov_scale = estimate_predictive_covariance_scale_cv(
                        fd["X_train_s"],
                        fd["Y_train_s"],
                        params=slm_params,
                        shrinkage_alpha=float(shrinkage_alpha),
                        n_folds=max(2, int(slm_adaptive_shrinkage_folds)),
                        seed=int(seed + fold_idx),
                    )


            # Optional latent recalibration head for mean prediction
            if bool(slm_latent_recalibration):
                with _Heartbeat(
                    label=f"{slm_method} r={r} fold {fold_idx + 1}/{n_folds} (latent head fit)",
                    heartbeat_sec=float(heartbeat_sec),
                ):
                    slm_head = fit_latent_recalibration_head(
                        fd["X_train_s"],
                        fd["Y_train_s"],
                        slm_params,
                        shrinkage_alpha=float(shrinkage_alpha),
                        ridge_alphas=latent_head_alphas,
                        cv=slm_latent_recalibration_cv,
                        include_original_x=bool(slm_latent_recalibration_include_x),
                    )

                y_pred_s = predict_recalibrated_mean(
                    fd["X_test_s"],
                    slm_params,
                    slm_head,
                    shrinkage_alpha=float(shrinkage_alpha),
                )
            else:
                y_pred_s = predict_conditional_mean(fd["X_test_s"], slm_params, shrinkage_alpha=float(shrinkage_alpha))

            Cov_s = predict_conditional_covariance(slm_params, shrinkage_alpha=float(shrinkage_alpha))
            Cov_s = float(cov_scale) * Cov_s

            y_pred = unstandardize_y(y_pred_s, fd["sy"])
            Cov = unstandardize_cov(Cov_s, fd["sy"])

            m = compute_regression_metrics(fd["Y_test"], y_pred)

            pred_rows.append(
                {
                    "method": slm_method,
                    "r": int(r),
                    "fold": fold_idx + 1,
                    "mse": m.mse,
                    "mae": m.mae,
                    "r2": m.r2_mean,
                    "time_sec": float(time.perf_counter() - t0),
                    "shrinkage_alpha": float(shrinkage_alpha),
                    "cov_scale": float(cov_scale),
                    "N": int(fd["X_train"].shape[0] + fd["X_test"].shape[0]),
                    "p": int(fd["X_train"].shape[1]),
                    "q": int(fd["Y_train"].shape[1]),
                }
            )

            for a in alphas_for_coverage:
                lo, hi = compute_credible_intervals(y_pred, Cov, alpha=float(a))
                cov = empirical_coverage(fd["Y_test"], lo, hi)
                cov_rows.append(
                    {
                        "method": slm_method,
                        "r": int(r),
                        "fold": fold_idx + 1,
                        "alpha": float(a),
                        "coverage": float(cov),
                        "shrinkage_alpha": float(shrinkage_alpha),
                        "cov_scale": float(cov_scale),
                    }
                )

            # --- EM (guarded; optional) ---
            # Skip EM cleanly when em_n_starts<=0 or timeout<=0 to avoid Windows spawn issues.
            if int(em_n_starts) > 0 and float(em_timeout_sec) > 0:
                t1 = time.perf_counter()
                em_status = "ok"
                em_params: Optional[Dict] = None
                try:
                    em_params = _run_in_subprocess_with_timeout(
                        fit_ppls_em,
                        (fd["X_train_s"], fd["Y_train_s"]),
                        dict(
                            r=int(r),
                            n_starts=int(em_n_starts),
                            seed=int(seed + fold_idx),
                            max_iter=int(em_max_iter),
                            tol=float(em_tol),
                            data_driven_init_fn=_data_driven_theta0,
                        ),

                        timeout_sec=float(em_timeout_sec),
                        heartbeat_sec=float(heartbeat_sec),
                        label=f"EM prediction r={r} fold {fold_idx + 1}/{n_folds}",
                    )

                except TimeoutError:
                    em_status = "DNF"
                except Exception:
                    em_status = "error"

                if em_params is not None:
                    y_em_s = predict_conditional_mean(fd["X_test_s"], em_params, shrinkage_alpha=1.0)
                    Cov_em_s = predict_conditional_covariance(em_params, shrinkage_alpha=1.0)
                    cov_scale_em = estimate_predictive_covariance_scale_cv(
                        fd["X_train_s"],
                        fd["Y_train_s"],
                        params=em_params,
                        shrinkage_alpha=1.0,
                        n_folds=max(2, int(slm_adaptive_shrinkage_folds)),
                        seed=int(seed + fold_idx),
                    )
                    Cov_em_s = float(cov_scale_em) * Cov_em_s

                    y_em = unstandardize_y(y_em_s, fd["sy"])
                    Cov_em = unstandardize_cov(Cov_em_s, fd["sy"])

                    m_em = compute_regression_metrics(fd["Y_test"], y_em)
                    pred_rows.append(
                        {
                            "method": "PPLS-EM",
                            "r": int(r),
                            "fold": fold_idx + 1,
                            "mse": m_em.mse,
                            "mae": m_em.mae,
                            "r2": m_em.r2_mean,
                            "time_sec": float(time.perf_counter() - t1),
                            "em_status": em_status,
                            "N": int(fd["X_train"].shape[0] + fd["X_test"].shape[0]),
                            "p": int(fd["X_train"].shape[1]),
                            "q": int(fd["Y_train"].shape[1]),
                        }
                    )

                    for a in alphas_for_coverage:
                        lo, hi = compute_credible_intervals(y_em, Cov_em, alpha=float(a))
                        cov = empirical_coverage(fd["Y_test"], lo, hi)
                        cov_rows.append(
                            {
                                "method": "PPLS-EM",
                                "r": int(r),
                                "fold": fold_idx + 1,
                                "alpha": float(a),
                                "coverage": float(cov),
                                "em_status": em_status,
                            }
                        )

                else:
                    pred_rows.append(
                        {
                            "method": "PPLS-EM",
                            "r": int(r),
                            "fold": fold_idx + 1,
                            "mse": np.nan,
                            "mae": np.nan,
                            "r2": np.nan,
                            "time_sec": float(time.perf_counter() - t1),
                            "em_status": em_status,
                            "N": int(fd["X_train"].shape[0] + fd["X_test"].shape[0]),
                            "p": int(fd["X_train"].shape[1]),
                            "q": int(fd["Y_train"].shape[1]),
                        }
                    )


            # --- PLSR baseline ---
            t2 = time.perf_counter()
            plsr = run_plsr_prediction(fd["X_train"], fd["Y_train"], fd["X_test"], fd["Y_test"], n_components=int(r))
            m_pls = plsr["metrics"]
            pred_rows.append(
                {
                    "method": "PLSR",
                    "r": int(r),
                    "fold": fold_idx + 1,
                    "mse": m_pls.mse,
                    "mae": m_pls.mae,
                    "r2": m_pls.r2_mean,
                    "time_sec": float(time.perf_counter() - t2),
                    "N": int(fd["X_train"].shape[0] + fd["X_test"].shape[0]),
                    "p": int(fd["X_train"].shape[1]),
                    "q": int(fd["Y_train"].shape[1]),
                }
            )

    return pd.DataFrame(pred_rows), pd.DataFrame(cov_rows)





def _coverage_summary(df_cov: pd.DataFrame) -> pd.DataFrame:
    if df_cov.empty:
        return df_cov
    out = []
    for (method, r, alpha), sub in df_cov.groupby(["method", "r", "alpha"], sort=False):
        out.append(
            {
                "method": method,
                "r": r,
                "alpha": float(alpha),
                "coverage_mean": float(sub["coverage"].mean()),
                "coverage_std": float(sub["coverage"].std(ddof=0)),
            }
        )
    return pd.DataFrame(out)


def _format_seconds(x: float) -> str:
    if not np.isfinite(x):
        return "DNF"
    x = float(x)
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{x/60:.1f}m"
    return f"{x/3600:.1f}h"


def _export_top_loadings(params: Dict, gene_names: Sequence[str], protein_names: Sequence[str], *, out_csv: str, top_genes: int = 10, top_proteins: int = 5, n_components: int = 5) -> None:
    W = np.asarray(params["W"], dtype=float)
    C = np.asarray(params["C"], dtype=float)

    k = int(min(int(n_components), W.shape[1], C.shape[1]))

    rows: List[Dict[str, Any]] = []
    for i in range(k):
        w = W[:, i]
        c = C[:, i]

        gi = np.argsort(-np.abs(w), kind="mergesort")[: int(top_genes)]
        pi = np.argsort(-np.abs(c), kind="mergesort")[: int(top_proteins)]

        for rank, j in enumerate(gi, start=1):
            rows.append({"component": i + 1, "view": "RNA", "rank": rank, "feature": str(gene_names[j]), "loading": float(w[j])})
        for rank, j in enumerate(pi, start=1):
            rows.append({"component": i + 1, "view": "ADT", "rank": rank, "feature": str(protein_names[j]), "loading": float(c[j])})

    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="CITE-seq PBMC protein prediction (CV prediction + calibration)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON (single source of truth)")
    p.add_argument("--smoke", action="store_true", help="Fast debug run (small N, fewer folds/iters)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from ppls_slm.experiment_config import coerce_float, coerce_int, get_experiment_cfg, load_config, require_keys

    cfg = load_config(args.config)
    c_cfg = get_experiment_cfg(cfg, "citeseq_prediction")

    require_keys(
        c_cfg,
        [
            "thread_limit",
            "citeseq_data",
            "output_dir",
            "seed",
            "n_folds",
            "r_grid",
            "n_top_genes",
            "subsample_n_prediction",
            "slm_n_starts",
            "slm_max_iter",
            "em_n_starts",
            "em_max_iter",
            "em_tol",
        ],
        ctx="experiments.citeseq_prediction",
    )

    for k in ("thread_limit", "seed", "n_folds", "n_top_genes", "subsample_n_prediction", "slm_n_starts", "slm_max_iter", "em_n_starts", "em_max_iter"):
        coerce_int(c_cfg, k, ctx="experiments.citeseq_prediction")
    coerce_float(c_cfg, "em_tol", ctx="experiments.citeseq_prediction")

    if args.smoke:
        # Minimal settings that should finish quickly and validate the end-to-end pipeline.
        print("[SMOKE] overriding config for a fast debug run", flush=True)

        # Guardrail: smoke runs should not overwrite the full paper outputs by accident.
        if str(c_cfg.get("output_dir", "")) == "results_citeseq":
            c_cfg["output_dir"] = "results_citeseq_smoke"

        c_cfg["n_folds"] = min(int(c_cfg.get("n_folds", 5)), 2)
        c_cfg["subsample_n_prediction"] = min(int(c_cfg.get("subsample_n_prediction", 30000)), 400)


        # Make smoke runs fast: fewer restarts/iters and skip expensive nested CV components.
        c_cfg["slm_n_starts"] = 1
        c_cfg["slm_max_iter"] = min(int(c_cfg.get("slm_max_iter", 200)), 25)
        c_cfg["slm_early_stop_patience"] = min(int(c_cfg.get("slm_early_stop_patience", 2)), 1)
        c_cfg["slm_verbose"] = True
        c_cfg["slm_progress_every"] = 1

        # Optional components that can dominate runtime.
        c_cfg["slm_adaptive_shrinkage"] = False
        c_cfg["slm_latent_recalibration"] = False

        # Ensure EM is skipped in smoke (it runs in subprocesses and can add overhead).
        c_cfg["em_n_starts"] = 0
        c_cfg["em_timeout_sec_prediction"] = 0


    output_dir = str(c_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Runtime thread limiting (helps avoid BLAS thread deadlocks / oversubscription on Windows).
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(c_cfg["thread_limit"]))
    except Exception:
        pass

    r_grid = [int(x.strip()) for x in str(c_cfg["r_grid"]).split(",") if x.strip()]
    if args.smoke:
        r_grid = r_grid[:1] or [10]

    # Load data (bounded to the prediction subset we need)

    max_need = int(c_cfg.get("subsample_n_prediction", 30000))


    data_path = str(c_cfg["citeseq_data"])
    if not os.path.exists(data_path):
        # Convenience fallback: if user exported CSVs into `application/`, allow pointing to a non-existent
        # placeholder file (e.g., application/pbmc_citeseq.h5ad) while still loading from the directory.
        parent = os.path.dirname(data_path) or "."
        rna_csv = os.path.join(parent, "citeseq_rna.csv")
        adt_csv = os.path.join(parent, "citeseq_adt.csv")
        if os.path.exists(rna_csv) and os.path.exists(adt_csv):
            data_path = parent
        else:
            raise FileNotFoundError(
                f"CITE-seq input not found: {data_path}. "
                f"Provide a .h5ad/.h5mu at that path, or export CSVs to {parent}/citeseq_rna.csv and {parent}/citeseq_adt.csv."
            )

    X, Y, gene_names, protein_names = load_citeseq_data(
        data_path,
        n_top_genes=int(c_cfg.get("n_top_genes", 2000)),
        subsample_n=max_need,
        seed=int(c_cfg.get("seed", 42)),
        return_names=True,
    )


    print(f"Loaded CITE-seq: X={X.shape}, Y={Y.shape} (subsample_n={max_need})")






    # 1) Prediction + calibration (CV) on a fixed subset



    N_pred = int(c_cfg.get("subsample_n_prediction", min(30000, X.shape[0])))
    if N_pred < X.shape[0]:
        rng = np.random.RandomState(int(c_cfg.get("seed", 42)))
        idx = rng.choice(X.shape[0], size=N_pred, replace=False)
        idx = np.sort(idx)
        Xp = X[idx]
        Yp = Y[idx]
    else:
        Xp, Yp = X, Y

    df_pred, df_cov = run_citeseq_prediction(
        Xp,
        Yp,
        r_grid=r_grid,
        n_folds=int(c_cfg.get("n_folds", 5)),
        seed=int(c_cfg.get("seed", 42)),
        slm_n_starts=int(c_cfg.get("slm_n_starts", 4)),
        slm_max_iter=int(c_cfg.get("slm_max_iter", 200)),
        slm_progress_every=int(c_cfg.get("slm_progress_every", 20)),
        slm_verbose=bool(c_cfg.get("slm_verbose", False)),
        heartbeat_sec=float(c_cfg.get("heartbeat_sec", 60)),
        em_n_starts=int(c_cfg.get("em_n_starts", 2)),
        em_max_iter=int(c_cfg.get("em_max_iter", 200)),
        em_tol=float(c_cfg.get("em_tol", 1e-4)),

        slm_optimizer=str(c_cfg.get("slm_optimizer", "manifold")),
        slm_use_noise_preestimation=bool(c_cfg.get("slm_use_noise_preestimation", True)),
        slm_gtol=float(c_cfg.get("slm_gtol", 0.01)),
        slm_xtol=float(c_cfg.get("slm_xtol", 0.01)),
        slm_barrier_tol=float(c_cfg.get("slm_barrier_tol", 0.01)),
        slm_constraint_slack=float(c_cfg.get("slm_constraint_slack", 0.005)),
        slm_early_stop_patience=c_cfg.get("slm_early_stop_patience", None),
        slm_early_stop_rel_improvement=c_cfg.get("slm_early_stop_rel_improvement", None),
        slm_adaptive_shrinkage=bool(c_cfg.get("slm_adaptive_shrinkage", True)),
        slm_adaptive_shrinkage_mode=str(c_cfg.get("slm_adaptive_shrinkage_mode", "nested_refit")),
        slm_adaptive_shrinkage_inner_n_starts=int(c_cfg.get("slm_adaptive_shrinkage_inner_n_starts", 2)),
        slm_adaptive_shrinkage_inner_max_iter=int(c_cfg.get("slm_adaptive_shrinkage_inner_max_iter", 80)),
        slm_adaptive_shrinkage_verbose=bool(c_cfg.get("slm_adaptive_shrinkage_verbose", False)),
        slm_shrinkage_alpha_grid=c_cfg.get("slm_shrinkage_alpha_grid", None),
        slm_adaptive_shrinkage_folds=int(c_cfg.get("slm_adaptive_shrinkage_folds", 5)),
        slm_latent_recalibration=bool(c_cfg.get("slm_latent_recalibration", True)),
        slm_latent_recalibration_cv=int(c_cfg.get("slm_latent_recalibration_cv", 5)) if c_cfg.get("slm_latent_recalibration_cv", None) is not None else None,
        slm_latent_recalibration_include_x=bool(c_cfg.get("slm_latent_recalibration_include_x", True)),
        slm_latent_recalibration_alphas=c_cfg.get("slm_latent_recalibration_alphas", None),
        em_timeout_sec=float(c_cfg.get("em_timeout_sec_prediction", 2 * 3600)),
    )


    df_pred.to_csv(os.path.join(output_dir, "citeseq_prediction_per_fold.csv"), index=False)
    df_by_r = aggregate_prediction_by_r(df_pred, ddof=0)

    df_by_r.to_csv(os.path.join(output_dir, "citeseq_prediction_by_r.csv"), index=False)
    df_best = select_best_r(df_by_r, ignore_dash_r=True)

    df_best.to_csv(os.path.join(output_dir, "citeseq_prediction_summary.csv"), index=False)

    df_cov.to_csv(os.path.join(output_dir, "citeseq_calibration_per_fold.csv"), index=False)
    df_cov_sum = _coverage_summary(df_cov)
    df_cov_sum.to_csv(os.path.join(output_dir, "citeseq_calibration_summary.csv"), index=False)

    # Diagnostic: selected adaptive shrinkage alphas
    if "shrinkage_alpha" in df_pred.columns:
        sub = df_pred[df_pred["method"].astype(str).str.startswith("PPLS-SLM", na=False)]
        if len(sub):
            alpha_diag = (
                sub[["fold", "r", "method", "shrinkage_alpha", "cov_scale"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["method", "r", "fold"])
            )
            alpha_diag.to_csv(os.path.join(output_dir, "citeseq_selected_shrinkage_alpha.csv"), index=False)

    # Phase 2 (best-effort): loading interpretability on one fitted model
    try:
        r_best = None
        try:
            r_best = int(df_best.loc[df_best["method"].astype(str).str.startswith("PPLS-SLM"), "r"].iloc[0])
        except Exception:
            r_best = int(r_grid[0])

        from sklearn.preprocessing import StandardScaler

        sx = StandardScaler().fit(Xp)
        sy = StandardScaler().fit(Yp)
        Xs = sx.transform(Xp)
        Ys = sy.transform(Yp)
        params_best = fit_ppls_slm(
            Xs,
            Ys,
            r=int(r_best),
            n_starts=int(c_cfg.get("slm_n_starts", 4)),
            seed=int(c_cfg.get("seed", 42)),
            max_iter=int(c_cfg.get("slm_max_iter", 200)),
            optimizer=str(c_cfg.get("slm_optimizer", "manifold")),
            use_noise_preestimation=bool(c_cfg.get("slm_use_noise_preestimation", True)),
            gtol=float(c_cfg.get("slm_gtol", 0.01)),
            xtol=float(c_cfg.get("slm_xtol", 0.01)),
            barrier_tol=float(c_cfg.get("slm_barrier_tol", 0.01)),
            initial_constr_penalty=1.0,
            constraint_slack=float(c_cfg.get("slm_constraint_slack", 0.005)),
            verbose=False,
            progress_every=10,
            data_driven_init_fn=_data_driven_theta0,
        )


        _export_top_loadings(
            params_best,
            gene_names,
            protein_names,
            out_csv=os.path.join(output_dir, "citeseq_loadings_top.csv"),
            top_genes=10,
            top_proteins=5,
            n_components=5,
        )
    except Exception as e:
        print(f"[WARN] loading export failed: {e}", flush=True)

    print("\nSaved (CITE-seq):")
    for fn in (
        "citeseq_prediction_per_fold.csv",

        "citeseq_prediction_by_r.csv",
        "citeseq_prediction_summary.csv",
        "citeseq_calibration_per_fold.csv",
        "citeseq_calibration_summary.csv",
        "citeseq_selected_shrinkage_alpha.csv",
        "citeseq_loadings_top.csv",
    ):
        p = os.path.join(output_dir, fn)
        if os.path.exists(p):
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
