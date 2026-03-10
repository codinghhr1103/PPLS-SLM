"""Application 2: Prediction with Uncertainty Quantification (Section 8.3)

This module reproduces and extends the prediction experiments from the paper
"Scalar Likelihood Method for Probabilistic Partial Least Squares Model with Rank n Update".

Two experiment tracks are supported:

A) Synthetic PPLS data (paper's main prediction sandbox)
   - p=q=100, r=20, N=120 (defaults; reduced for faster reproducible runs)
   - 5-fold CV
   - Compare predictive accuracy across four methods:

        * PPLS-SLM (fitted per fold)
        * PPLS-EM  (fitted per fold)
        * Classical PLS regression (PLSR)
        * Ridge regression (RidgeCV)
   - For PPLS-SLM and PPLS-EM, additionally evaluate calibration of credible intervals.

B) (Implemented in separate scripts) Real BRCA TCGA data prediction and calibration.

Evaluation protocol
-------------------
- All methods use the same CV folds (fixed RNG seed) and the same standardisation flow:
  per fold, fit a z-score transform on the training split and apply it to the test split.
- Accuracy metrics are evaluated on the original Y scale:
    MSE, MAE, and mean R2 across output dimensions.
- PPLS credible intervals are constructed element-wise using the predictive covariance.

Usage
-----
    python -m ppls_slm.apps.prediction --config config.json

Configuration
-------------
All hyperparameters and output paths are read from a single JSON config file.
See `config.json` under `experiments.prediction`.

"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Tuple



# NOTE (Windows stability): some BLAS/LAPACK builds may hang or become extremely slow
# on QR/SVD due to oversubscription or thread deadlocks. Defaulting to 1 thread makes
# the synthetic-data generation and optimisation steps deterministic and robust.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats


from ppls_slm.algorithms import EMAlgorithm, InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction_baselines import compute_regression_metrics, run_plsr_prediction, run_ridge_prediction
from ppls_slm.ppls_model import PPLSModel


# ─────────────────────────────────────────────────────────────────────────────
#  Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_ppls_data(
    p: int = 200,
    q: int = 200,
    r: int = 50,
    n_samples: int = 100,
    sigma_e2: float = 0.1,
    sigma_f2: float = 0.1,
    sigma_h2: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate synthetic PPLS data for the prediction experiment."""
    rng = np.random.RandomState(seed)

    # Random orthonormal loading matrices
    W, _ = np.linalg.qr(rng.randn(p, r))
    C, _ = np.linalg.qr(rng.randn(q, r))

    # Decreasing signals so identifiability holds: theta_t2[i] * b[i] decreasing
    theta_t2 = np.linspace(1.5, 0.3, r)
    b = np.linspace(2.0, 0.5, r)
    assert np.all(np.diff(theta_t2 * b) < 0), "Identifiability violated."

    B = np.diag(b)
    Sigma_t = np.diag(theta_t2)

    model = PPLSModel(p, q, r)
    np.random.seed(seed + 1)  # reproducible sampling
    X, Y = model.sample(n_samples, W, C, B, Sigma_t, sigma_e2, sigma_f2, sigma_h2)

    true_params = {
        "W": W,
        "C": C,
        "B": B,
        "Sigma_t": Sigma_t,
        "sigma_e2": sigma_e2,
        "sigma_f2": sigma_f2,
        "sigma_h2": sigma_h2,
    }
    return X, Y, true_params


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction helpers (Property 1 / Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────

def predict_conditional_mean(
    x_new: np.ndarray,
    params: Dict,
    *,
    shrinkage_alpha: float = 1.0,
) -> np.ndarray:
    """Compute E[y_new | x_new, params] via the conditional Gaussian formula.

    The adaptive-shrinkage parameter alpha enters through

        (W Sigma_t W^T + alpha * sigma_e^2 I)^{-1}.
    """
    W, C, B = params["W"], params["C"], params["B"]
    Sigma_t = params["Sigma_t"]
    sigma_e2 = float(params["sigma_e2"])

    a = float(shrinkage_alpha)
    if not (a > 0):
        raise ValueError(f"shrinkage_alpha must be > 0, got {a}")

    Sigma_xx = W @ Sigma_t @ W.T + (a * sigma_e2) * np.eye(W.shape[0])

    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    tmp = np.linalg.solve(L, W)
    tmp = np.linalg.solve(L.T, tmp)  # Sigma_xx^{-1} W

    A = C @ B @ Sigma_t @ tmp.T  # (q, p)

    if x_new.ndim == 1:
        return A @ x_new
    return x_new @ A.T


def posterior_latent_mean(
    x_new: np.ndarray,
    params: Dict,
    *,
    shrinkage_alpha: float = 1.0,
) -> np.ndarray:
    r"""Compute the posterior mean \\(\mathbb{E}[t_{new}\mid x_{new}]\\)."""

    W = params["W"]
    Sigma_t = params["Sigma_t"]
    sigma_e2 = float(params["sigma_e2"])

    a = float(shrinkage_alpha)
    if not (a > 0):
        raise ValueError(f"shrinkage_alpha must be > 0, got {a}")

    Sigma_xx = W @ Sigma_t @ W.T + (a * sigma_e2) * np.eye(W.shape[0])
    WS = W @ Sigma_t

    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    tmp = np.linalg.solve(L, WS)
    tmp = np.linalg.solve(L.T, tmp)  # Sigma_xx^{-1} W Sigma_t

    if x_new.ndim == 1:
        return x_new @ tmp
    return x_new @ tmp


def fit_latent_recalibration_head(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    params: Dict,
    *,
    shrinkage_alpha: float = 1.0,
    ridge_alphas: Optional[Sequence[float]] = None,
    cv: Optional[int] = None,
    include_original_x: bool = False,
):

    """Fit a low-dimensional ridge head on posterior latent means.

    This keeps the PPLS latent representation but relaxes the strict diagonal
    latent-to-response map during prediction, which is especially helpful in
    high-dimensional real-data settings.
    """
    from sklearn.linear_model import RidgeCV

    if ridge_alphas is None:
        ridge_alphas = np.logspace(-4, 4, 17)

    Z_train = posterior_latent_mean(X_train_s, params, shrinkage_alpha=float(shrinkage_alpha))
    X_head = np.hstack([Z_train, X_train_s]) if bool(include_original_x) else Z_train
    head = RidgeCV(alphas=np.asarray(list(ridge_alphas), dtype=float), cv=cv, fit_intercept=True)
    head.fit(X_head, Y_train_s)
    return {
        "model": head,
        "include_original_x": bool(include_original_x),
    }



def predict_recalibrated_mean(
    X_new_s: np.ndarray,
    params: Dict,
    head,
    *,
    shrinkage_alpha: float = 1.0,
) -> np.ndarray:
    """Predict Y from posterior latent means using a fitted low-dimensional head."""
    Z_new = posterior_latent_mean(X_new_s, params, shrinkage_alpha=float(shrinkage_alpha))
    model = head["model"] if isinstance(head, dict) else head
    include_original_x = bool(head.get("include_original_x", False)) if isinstance(head, dict) else False
    X_head = np.hstack([Z_new, X_new_s]) if include_original_x else Z_new
    return np.asarray(model.predict(X_head), dtype=float)



def predict_conditional_covariance(
    params: Dict,
    *,
    shrinkage_alpha: float = 1.0,
) -> np.ndarray:

    r"""Compute \(\mathrm{Cov}[y_{new}\mid x_{new}]\) (x-independent).

    The adaptive-shrinkage parameter \(\alpha\) enters through

        \((W\Sigma_tW^\top + \alpha\,\sigma_e^2 I)^{-1}\).
    """
    W, C, B = params["W"], params["C"], params["B"]
    Sigma_t = params["Sigma_t"]
    sigma_e2 = float(params["sigma_e2"])
    sigma_f2 = float(params["sigma_f2"])
    sigma_h2 = float(params["sigma_h2"])

    a = float(shrinkage_alpha)
    if not (a > 0):
        raise ValueError(f"shrinkage_alpha must be > 0, got {a}")

    r = W.shape[1]
    q = C.shape[0]

    b = np.diag(B)
    theta_t2 = np.diag(Sigma_t)

    sigma_eff = a * sigma_e2
    Sigma_xx = W @ Sigma_t @ W.T + sigma_eff * np.eye(W.shape[0])
    L = np.linalg.cholesky(Sigma_xx + 1e-9 * np.eye(Sigma_xx.shape[0]))
    WtSig_inv = np.linalg.solve(L.T, np.linalg.solve(L, W))  # Sigma_xx^{-1} W (p,r)

    B2Sigma_t = np.diag(b**2 * theta_t2)
    term1 = C @ (B2Sigma_t + sigma_h2 * np.eye(r)) @ C.T + sigma_f2 * np.eye(q)

    K = W.T @ WtSig_inv
    K = (K + K.T) / 2
    M = C @ B @ Sigma_t @ K @ Sigma_t @ B @ C.T

    Cov = term1 - M
    Cov = (Cov + Cov.T) / 2 + 1e-9 * np.eye(q)
    return Cov



def compute_credible_intervals(
    y_pred: np.ndarray,
    Cov_yx: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Element-wise symmetric credible intervals: mean ± z * sqrt(diag(Cov))."""
    z = stats.norm.ppf(1 - alpha / 2)
    std = np.sqrt(np.diag(Cov_yx))
    lower = y_pred - z * std[np.newaxis, :]
    upper = y_pred + z * std[np.newaxis, :]
    return lower, upper


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of (test-sample, feature) entries within [lower, upper]."""
    within = (y_true >= lower) & (y_true <= upper)
    return float(within.mean())


def _slm_method_name_from_cfg(slm_cfg: Dict) -> str:
    opt = str(slm_cfg.get("optimizer", "")).lower()
    adaptive = bool(slm_cfg.get("adaptive_shrinkage", False))

    if opt in ("manifold", "pymanopt", "riemannian", "stiefel"):
        return "PPLS-SLM-Manifold-Adaptive" if adaptive else "PPLS-SLM-Manifold"

    return "PPLS-SLM-Adaptive" if adaptive else "PPLS-SLM"


def slm_method_name(*, slm_optimizer: str, adaptive: bool) -> str:
    """Human-readable method name used in result tables/plots."""

    opt = str(slm_optimizer).lower()
    if opt in ("manifold", "pymanopt", "riemannian", "stiefel"):
        return "PPLS-SLM-Manifold-Adaptive" if bool(adaptive) else "PPLS-SLM-Manifold"
    return "PPLS-SLM-Adaptive" if bool(adaptive) else "PPLS-SLM"


def select_shrinkage_alpha_cv(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    params: Dict,
    alpha_grid: Sequence[float],
    n_folds: int = 5,
    seed: int = 0,
) -> Tuple[float, pd.DataFrame]:
    r"""Select alpha by inner CV using fixed PPLS parameters.

    We do NOT refit PPLS per inner fold; we only vary \(\alpha\) in the prediction
    rule, as described in the paper revision.

    Returns `(alpha_star, cv_table)` where `cv_table` contains per-alpha mean MSE.
    """
    from sklearn.model_selection import KFold

    alpha_grid = [float(a) for a in alpha_grid]
    if not alpha_grid:
        raise ValueError("alpha_grid must be non-empty")

    n = int(X_train_s.shape[0])
    k = int(min(max(2, int(n_folds)), n))

    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    rows: List[Dict] = []
    for a in alpha_grid:
        if not (a > 0):
            continue
        mses = []
        for tr, va in kf.split(X_train_s):
            X_va = X_train_s[va]
            Y_va = Y_train_s[va]
            Y_hat = predict_conditional_mean(X_va, params, shrinkage_alpha=float(a))
            mses.append(float(np.mean((Y_va - Y_hat) ** 2)))
        rows.append({"shrinkage_alpha": float(a), "mse_mean": float(np.mean(mses))})

    cv_table = pd.DataFrame(rows).sort_values("shrinkage_alpha").reset_index(drop=True)
    if cv_table.empty:
        raise ValueError("alpha_grid produced no valid candidates (all <= 0?)")

    best_row = cv_table.loc[int(cv_table["mse_mean"].idxmin())]
    return float(best_row["shrinkage_alpha"]), cv_table


def select_shrinkage_alpha_nested_cv(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    fit_model_fn: Callable[[np.ndarray, np.ndarray, int], Dict],
    alpha_grid: Sequence[float],
    n_folds: int = 5,
    seed: int = 0,
    verbose: bool = False,
    progress_label: Optional[str] = None,
    validation_predictor_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, Dict, float], np.ndarray]] = None,
) -> Tuple[float, pd.DataFrame, float]:

    r"""Select alpha via nested inner CV with refits on inner-training splits.

    Unlike :func:`select_shrinkage_alpha_cv`, this routine refits the PPLS model on
    each inner-training split and then evaluates all alpha candidates on the held-out
    validation block. This avoids the optimistic bias of recycling outer-fold
    parameters for inner validation, and in practice yields more informative
    fold-to-fold alpha variation on both synthetic and BRCA prediction tasks.
    """
    from sklearn.model_selection import KFold

    alpha_grid = sorted({float(a) for a in alpha_grid if float(a) > 0.0})
    if not alpha_grid:
        raise ValueError("alpha_grid must contain at least one positive value")

    n = int(X_train_s.shape[0])
    k = int(min(max(2, int(n_folds)), n))
    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    rows: List[Dict] = []
    scale_rows: List[Dict] = []

    for inner_fold, (tr, va) in enumerate(kf.split(X_train_s), start=1):
        if verbose and progress_label:
            print(f"{progress_label} inner-fit {inner_fold}/{k}...", flush=True)

        params_inner = fit_model_fn(X_train_s[tr], Y_train_s[tr], int(seed + inner_fold - 1))
        X_va = X_train_s[va]
        Y_va = Y_train_s[va]

        for a in alpha_grid:
            if validation_predictor_fn is None:
                Y_hat = predict_conditional_mean(X_va, params_inner, shrinkage_alpha=float(a))
            else:
                Y_hat = validation_predictor_fn(X_train_s[tr], Y_train_s[tr], X_va, params_inner, float(a))
            mse = float(np.mean((Y_va - Y_hat) ** 2))
            rows.append({"inner_fold": inner_fold, "shrinkage_alpha": float(a), "mse": mse})

            Cov_s = predict_conditional_covariance(params_inner, shrinkage_alpha=float(a))

            # Paper calibration scale: kappa = (1/(n_val q)) sum r_i^T V^{-1} r_i.
            resid = Y_va - Y_hat
            kappa = _kappa_from_residuals(resid, Cov_s)
            if np.isfinite(kappa) and kappa > 0.0:
                scale_rows.append(
                    {
                        "inner_fold": inner_fold,
                        "shrinkage_alpha": float(a),
                        "covariance_scale": float(kappa),
                    }
                )


    cv_long = pd.DataFrame(rows)
    if cv_long.empty:
        raise ValueError("nested alpha CV produced no validation rows")

    cv_table = (
        cv_long.groupby("shrinkage_alpha", sort=True)["mse"]
        .agg([("mse_mean", "mean"), ("mse_std", "std")])
        .reset_index()
        .sort_values("shrinkage_alpha")
        .reset_index(drop=True)
    )
    cv_table["mse_std"] = cv_table["mse_std"].fillna(0.0)
    cv_table["distance_to_one"] = np.abs(np.log(cv_table["shrinkage_alpha"].astype(float)))

    best_row = cv_table.sort_values(["mse_mean", "mse_std", "distance_to_one", "shrinkage_alpha"]).iloc[0]
    alpha_star = float(best_row["shrinkage_alpha"])

    covariance_scale = 1.0
    if scale_rows:
        scale_df = pd.DataFrame(scale_rows)
        best_scales = scale_df.loc[
            np.isclose(scale_df["shrinkage_alpha"].to_numpy(dtype=float), alpha_star),
            "covariance_scale",
        ]
        if len(best_scales):
            covariance_scale = float(np.mean(best_scales.to_numpy(dtype=float)))


    cv_table = cv_table.drop(columns=["distance_to_one"])
    return alpha_star, cv_table, covariance_scale


def _kappa_from_residuals(residuals: np.ndarray, Cov: np.ndarray) -> float:
    r"""Compute the paper's calibration scale \(\hat\kappa\).


    \[\hat\kappa = \frac{1}{n_{val} q}\sum_{i=1}^{n_{val}} r_i^\top V^{-1} r_i\]

    Here `residuals` is (n_val, q) and `Cov` is (q, q).
    """
    R = np.asarray(residuals, dtype=float)
    V = np.asarray(Cov, dtype=float)
    if R.ndim != 2 or V.ndim != 2 or V.shape[0] != V.shape[1] or R.shape[1] != V.shape[0]:
        raise ValueError(f"shape mismatch: residuals={R.shape}, Cov={V.shape}")

    n_val, q = R.shape
    if n_val <= 0 or q <= 0:
        return 1.0

    V = (V + V.T) / 2.0

    # Compute sum_i r_i^T V^{-1} r_i via Cholesky solves.
    try:
        L = np.linalg.cholesky(V + 1e-12 * np.eye(q))
        Z = np.linalg.solve(L, R.T)
        W = np.linalg.solve(L.T, Z)  # V^{-1} R^T
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(V) @ R.T

    quad = np.sum(R.T * W, axis=0)  # length n_val
    kappa = float(np.mean(quad) / float(q))
    if not np.isfinite(kappa) or kappa <= 0.0:
        return 1.0
    return kappa


def estimate_predictive_covariance_scale_cv(

    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    params: Dict,
    shrinkage_alpha: float,
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    r"""Estimate the paper's covariance scale \(\hat\kappa\) via inner CV.


    We compute \(\hat\kappa\) on each validation block and return the mean across folds.
    """
    from sklearn.model_selection import KFold

    n = int(X_train_s.shape[0])
    k = int(min(max(2, int(n_folds)), n))
    kf = KFold(n_splits=k, shuffle=True, random_state=int(seed))

    Cov_s = predict_conditional_covariance(params, shrinkage_alpha=float(shrinkage_alpha))

    kappas: List[float] = []
    for _tr, va in kf.split(X_train_s):
        X_va = X_train_s[va]
        Y_va = Y_train_s[va]
        Y_hat = predict_conditional_mean(X_va, params, shrinkage_alpha=float(shrinkage_alpha))
        resid = Y_va - Y_hat
        kappas.append(_kappa_from_residuals(resid, Cov_s))

    return float(np.mean(kappas)) if kappas else 1.0




# ─────────────────────────────────────────────────────────────────────────────
#  Fold-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _standardize_train_test(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
):
    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler()
    sy = StandardScaler()

    X_train_s = sx.fit_transform(X_train)
    Y_train_s = sy.fit_transform(Y_train)
    X_test_s = sx.transform(X_test)
    Y_test_s = sy.transform(Y_test)

    return X_train_s, Y_train_s, X_test_s, Y_test_s, sx, sy


def _unstandardize_y_pred(y_pred_s: np.ndarray, sy) -> np.ndarray:
    return sy.inverse_transform(y_pred_s)


def _unstandardize_cov_y(Cov_s: np.ndarray, sy) -> np.ndarray:
    # If y_s = (y - mean)/scale, then Cov(y) = D * Cov(y_s) * D, D = diag(scale)
    scale = getattr(sy, "scale_", None)
    if scale is None:
        return Cov_s
    D = np.diag(np.asarray(scale, dtype=float))
    return D @ Cov_s @ D


def _data_driven_theta0(X_train_s: np.ndarray, Y_train_s: np.ndarray, *, r: int) -> np.ndarray:
    """Deterministic PLS-style starting point based on cross-covariance.

    Compared with separate PCA starts on X and Y, the leading singular vectors of
    X^T Y align the initial W/C pair with directions that are directly predictive
    across the two views. This tends to improve fit quality for the prediction
    benchmarks while remaining essentially free computationally.
    """
    eps = 1e-3
    n = max(1, int(X_train_s.shape[0]))

    try:
        cross = (X_train_s.T @ Y_train_s) / float(n)
        Uxy, _sxy, Vtxy = np.linalg.svd(cross, full_matrices=False)
        W0 = Uxy[:, :r]
        C0 = Vtxy.T[:, :r]
    except np.linalg.LinAlgError:
        _Ux, _sx, Vt_x = np.linalg.svd(X_train_s, full_matrices=False)
        _Uy, _sy, Vt_y = np.linalg.svd(Y_train_s, full_matrices=False)
        W0 = Vt_x.T[:, :r]
        C0 = Vt_y.T[:, :r]

    T0 = X_train_s @ W0
    U0 = Y_train_s @ C0

    cross_diag = np.mean(T0 * U0, axis=0)
    signs = np.where(cross_diag < 0.0, -1.0, 1.0)
    C0 = C0 * signs[np.newaxis, :]
    U0 = U0 * signs[np.newaxis, :]

    theta0 = np.clip(np.var(T0, axis=0, ddof=0), eps, 5.0)
    cross_diag = np.abs(np.mean(T0 * U0, axis=0))
    b0 = np.clip(cross_diag / (theta0 + 1e-8), eps, 5.0)

    signal = theta0 * b0
    order = np.argsort(-signal)
    W0 = W0[:, order]
    C0 = C0[:, order]
    T0 = T0[:, order]
    U0 = U0[:, order]
    theta0 = theta0[order]
    b0 = b0[order]

    sigma_h2_0 = float(np.clip(np.mean((U0 - T0 * b0[np.newaxis, :]) ** 2), eps, 2.0))

    return np.concatenate([W0.flatten(), C0.flatten(), theta0, b0, [sigma_h2_0]])






def _fit_ppls_params_slm(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    slm_max_iter: int,
    slm_optimizer: str,
    slm_gtol: float,
    slm_xtol: float,
    slm_barrier_tol: float,
    slm_constraint_slack: float,
    slm_progress_every: int,
    slm_early_stop_patience: Optional[int],
    slm_early_stop_rel_improvement: Optional[float],
    slm_data_start: bool,
    slm_verbose: bool,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    if bool(slm_data_start) and starting_points:
        starting_points[0] = _data_driven_theta0(X_train_s, Y_train_s, r=r)

    slm = ScalarLikelihoodMethod(
        p=p,
        q=q,
        r=r,
        optimizer=str(slm_optimizer),
        max_iter=int(slm_max_iter),
        use_noise_preestimation=True,
        gtol=float(slm_gtol),
        xtol=float(slm_xtol),
        barrier_tol=float(slm_barrier_tol),
        constraint_slack=float(slm_constraint_slack),
        verbose=bool(slm_verbose),
        progress_every=int(slm_progress_every),
        early_stop_patience=slm_early_stop_patience,
        early_stop_rel_improvement=slm_early_stop_rel_improvement,
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
        "_meta": {"n_iterations": res.get("n_iterations"), "success": res.get("success")},
    }





def _fit_ppls_params_em(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    em_max_iter: int,
    em_tol: float,
    em_data_start: bool,
) -> Dict:
    p, q = X_train_s.shape[1], Y_train_s.shape[1]

    init_gen = InitialPointGenerator(p=p, q=q, r=r, n_starts=n_starts, random_seed=seed)
    starting_points = init_gen.generate_starting_points()

    if bool(em_data_start) and starting_points:
        starting_points[0] = _data_driven_theta0(X_train_s, Y_train_s, r=r)

    em = EMAlgorithm(p=p, q=q, r=r, max_iter=int(em_max_iter), tolerance=float(em_tol))
    res = em.fit(X_train_s, Y_train_s, starting_points)


    return {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
        "_meta": {"n_iterations": res.get("n_iterations"), "log_likelihood": res.get("log_likelihood")},
    }


def _predict_ppls(
    X_test_s: np.ndarray,
    *,
    params: Dict,
    shrinkage_alpha: float = 1.0,
):
    y_pred_s = predict_conditional_mean(X_test_s, params, shrinkage_alpha=float(shrinkage_alpha))
    Cov_s = predict_conditional_covariance(params, shrinkage_alpha=float(shrinkage_alpha))
    return y_pred_s, Cov_s


def _select_slm_shrinkage_and_scale(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    params_outer: Dict,
    fold_seed: int,
    slm_cfg: Dict,
    fit_inner_model_fn: Callable[[np.ndarray, np.ndarray, int], Dict],
    progress_label: Optional[str] = None,
) -> Tuple[float, float, Optional[pd.DataFrame]]:
    """Tune adaptive shrinkage and covariance scaling for SLM prediction."""

    if not bool(slm_cfg.get("adaptive_shrinkage", False)):
        return 1.0, 1.0, None

    alpha_grid = slm_cfg.get(
        "shrinkage_alpha_grid",
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
    )
    n_inner = int(slm_cfg.get("adaptive_shrinkage_folds", 5))
    mode = str(slm_cfg.get("adaptive_shrinkage_mode", "nested_refit")).strip().lower()

    if mode in ("nested_refit", "nested", "refit"):
        shrinkage_alpha, cv_table, covariance_scale = select_shrinkage_alpha_nested_cv(
            X_train_s,
            Y_train_s,
            fit_model_fn=fit_inner_model_fn,
            alpha_grid=alpha_grid,
            n_folds=n_inner,
            seed=int(fold_seed),
            verbose=bool(slm_cfg.get("adaptive_shrinkage_verbose", False)),
            progress_label=progress_label,
        )
        return float(shrinkage_alpha), float(covariance_scale), cv_table

    shrinkage_alpha, cv_table = select_shrinkage_alpha_cv(
        X_train_s,
        Y_train_s,
        params=params_outer,
        alpha_grid=alpha_grid,
        n_folds=n_inner,
        seed=int(fold_seed),
    )
    covariance_scale = estimate_predictive_covariance_scale_cv(
        X_train_s,
        Y_train_s,
        params=params_outer,
        shrinkage_alpha=shrinkage_alpha,
        n_folds=n_inner,
        seed=int(fold_seed),
    )
    return float(shrinkage_alpha), float(covariance_scale), cv_table


# ─────────────────────────────────────────────────────────────────────────────

#  Single-fold worker (for parallel CV)
# ─────────────────────────────────────────────────────────────────────────────


def _run_single_fold_prediction(
    *,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    r: int,
    n_starts: int,
    seed: int,
    slm_max_iter: int,
    em_max_iter: int,
    em_tol: float,
    include_baselines: bool,
    alphas: List[float],
    thread_limit: int,
    slm_cfg: Dict,
):
    """Run one fold without printing (spawn-safe)."""
    # Apply thread limiting again inside the worker process.
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(thread_limit))
    except Exception:
        pass

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(
        X_train, Y_train, X_test, Y_test
    )

    metrics_rows: List[Dict] = []
    calib_rows: List[Dict] = []

    # --- PPLS-SLM ---
    if bool(slm_cfg.get("verbose", False)):
        print(f"    [fold {fold_idx + 1}] fitting PPLS-SLM/EM...", flush=True)

    slm_params = _fit_ppls_params_slm(
        X_train_s,
        Y_train_s,
        r=r,
        n_starts=n_starts,
        seed=seed + fold_idx,
        slm_max_iter=slm_max_iter,
        slm_optimizer=str(slm_cfg["optimizer"]),
        slm_gtol=float(slm_cfg["gtol"]),
        slm_xtol=float(slm_cfg["xtol"]),
        slm_barrier_tol=float(slm_cfg["barrier_tol"]),
        slm_constraint_slack=float(slm_cfg["constraint_slack"]),
        slm_progress_every=int(slm_cfg["progress_every"]),
        slm_early_stop_patience=slm_cfg.get("early_stop_patience"),
        slm_early_stop_rel_improvement=slm_cfg.get("early_stop_rel_improvement"),
        slm_data_start=bool(slm_cfg["data_start"]),
        slm_verbose=bool(slm_cfg.get("verbose", False)),
    )




    slm_method = _slm_method_name_from_cfg(slm_cfg)
    inner_n_starts = int(slm_cfg.get("adaptive_shrinkage_inner_n_starts", max(1, min(3, int(n_starts)))))
    inner_max_iter = int(slm_cfg.get("adaptive_shrinkage_inner_max_iter", max(40, min(int(slm_max_iter), 120))))
    shrinkage_alpha_slm, covariance_scale_slm, _cv_table = _select_slm_shrinkage_and_scale(
        X_train_s,
        Y_train_s,
        params_outer=slm_params,
        fold_seed=int(seed + fold_idx),
        slm_cfg=slm_cfg,
        fit_inner_model_fn=lambda X_in, Y_in, inner_seed: _fit_ppls_params_slm(
            X_in,
            Y_in,
            r=r,
            n_starts=inner_n_starts,
            seed=int(inner_seed),
            slm_max_iter=inner_max_iter,
            slm_optimizer=str(slm_cfg["optimizer"]),
            slm_gtol=float(slm_cfg["gtol"]),
            slm_xtol=float(slm_cfg["xtol"]),
            slm_barrier_tol=float(slm_cfg["barrier_tol"]),
            slm_constraint_slack=float(slm_cfg["constraint_slack"]),
            slm_progress_every=int(slm_cfg["progress_every"]),
            slm_early_stop_patience=slm_cfg.get("early_stop_patience"),
            slm_early_stop_rel_improvement=slm_cfg.get("early_stop_rel_improvement"),
            slm_data_start=bool(slm_cfg["data_start"]),
            slm_verbose=bool(slm_cfg.get("adaptive_shrinkage_verbose", False)),
        ),
        progress_label=f"    [fold {fold_idx + 1}] adaptive-shrinkage",
    )

    y_pred_slm_s, Cov_slm_s = _predict_ppls(X_test_s, params=slm_params, shrinkage_alpha=shrinkage_alpha_slm)

    Cov_slm_s = float(covariance_scale_slm) * Cov_slm_s
    y_pred_slm = _unstandardize_y_pred(y_pred_slm_s, sy)


    m_slm = compute_regression_metrics(Y_test, y_pred_slm)
    metrics_rows.append(
        {
            "fold": fold_idx + 1,
            "method": slm_method,
            "mse": m_slm.mse,
            "mae": m_slm.mae,
            "r2": m_slm.r2_mean,
            "shrinkage_alpha": float(shrinkage_alpha_slm),
        }
    )

    Cov_slm = _unstandardize_cov_y(Cov_slm_s, sy)
    for a in alphas:
        lower, upper = compute_credible_intervals(y_pred_slm, Cov_slm, alpha=float(a))
        cov = 100.0 * empirical_coverage(Y_test, lower, upper)
        calib_rows.append(
            {
                "fold": fold_idx + 1,
                "method": slm_method,
                "alpha": float(a),
                "coverage": cov,
                "shrinkage_alpha": float(shrinkage_alpha_slm),
                "covariance_scale": float(covariance_scale_slm),
            }

        )


    # --- PPLS-EM ---
    em_params = _fit_ppls_params_em(
        X_train_s,
        Y_train_s,
        r=r,
        n_starts=n_starts,
        seed=seed + fold_idx,
        em_max_iter=em_max_iter,
        em_tol=em_tol,
        em_data_start=bool(slm_cfg["data_start"]),
    )

    y_pred_em_s, Cov_em_s = _predict_ppls(X_test_s, params=em_params)
    y_pred_em = _unstandardize_y_pred(y_pred_em_s, sy)

    m_em = compute_regression_metrics(Y_test, y_pred_em)
    metrics_rows.append(
        {
            "fold": fold_idx + 1,
            "method": "PPLS-EM",
            "mse": m_em.mse,
            "mae": m_em.mae,
            "r2": m_em.r2_mean,
        }
    )

    Cov_em = _unstandardize_cov_y(Cov_em_s, sy)
    for a in alphas:
        lower, upper = compute_credible_intervals(y_pred_em, Cov_em, alpha=float(a))
        cov = 100.0 * empirical_coverage(Y_test, lower, upper)
        calib_rows.append({"fold": fold_idx + 1, "method": "PPLS-EM", "alpha": float(a), "coverage": cov})

    # --- Baselines ---
    if include_baselines:
        plsr = run_plsr_prediction(X_train, Y_train, X_test, Y_test, n_components=r)
        m_plsr = plsr["metrics"]
        metrics_rows.append(
            {
                "fold": fold_idx + 1,
                "method": "PLSR",
                "mse": m_plsr.mse,
                "mae": m_plsr.mae,
                "r2": m_plsr.r2_mean,
            }
        )

        ridge = run_ridge_prediction(X_train, Y_train, X_test, Y_test)
        m_ridge = ridge["metrics"]
        metrics_rows.append(
            {
                "fold": fold_idx + 1,
                "method": "Ridge",
                "mse": m_ridge.mse,
                "mae": m_ridge.mae,
                "r2": m_ridge.r2_mean,
            }
        )

    return metrics_rows, calib_rows


# ─────────────────────────────────────────────────────────────────────────────
#  k-fold benchmark (Synthetic)
# ─────────────────────────────────────────────────────────────────────────────



def kfold_prediction_benchmark(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r: int,
    n_folds: int,
    n_starts: int,
    seed: int,
    slm_max_iter: int,
    em_max_iter: int,
    em_tol: float,
    include_baselines: bool,
    thread_limit: int,
    parallel_folds: bool,
    fold_workers: int,
    cv_heartbeat_sec: int,
    slm_cfg: Dict,
    alphas: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    import time

    """Run 5-fold CV and return:

    - metrics_per_fold: long table (fold, method, MSE/MAE/R2)
    - metrics_summary : per method mean±std
    - calib_summary   : per alpha expected + (SLM/EM) mean±std (percent)
    """
    if alphas is None:
        alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    N = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    folds = np.array_split(indices, int(n_folds))

    metrics_rows: List[Dict] = []
    calib_rows: List[Dict] = []

    folds_iter = list(enumerate(folds))

    if bool(parallel_folds) and int(fold_workers) > 1 and int(n_folds) > 1:
        import multiprocessing
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        max_workers = max(1, min(int(fold_workers), int(n_folds)))
        heartbeat = max(1, int(cv_heartbeat_sec))

        if verbose:
            print(f"  Running folds in parallel (workers={max_workers})...", flush=True)
            print(f"  (heartbeat every {heartbeat}s; enable per-start logs via slm_verbose=true)", flush=True)

        mp_ctx = multiprocessing.get_context("spawn")
        futures = {}
        t_parallel = time.time()

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            for fold_idx, test_idx in folds_iter:
                if n_folds <= 1:
                    train_idx = indices
                else:
                    train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

                fut = executor.submit(
                    _run_single_fold_prediction,
                    fold_idx=fold_idx,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    X=X,
                    Y=Y,
                    r=r,
                    n_starts=n_starts,
                    seed=seed,
                    slm_max_iter=slm_max_iter,
                    em_max_iter=em_max_iter,
                    em_tol=em_tol,
                    include_baselines=include_baselines,
                    alphas=alphas,
                    thread_limit=thread_limit,
                    slm_cfg=slm_cfg,
                )
                futures[fut] = fold_idx

            if verbose:
                print(f"  Submitted {len(futures)}/{n_folds} fold jobs.", flush=True)

            completed = 0
            pending = set(futures.keys())
            last_beat = time.time()

            while pending:
                done, pending = wait(pending, timeout=heartbeat, return_when=FIRST_COMPLETED)

                for fut in done:
                    fold_done = int(futures[fut])
                    rows_m, rows_c = fut.result()
                    metrics_rows.extend(rows_m)
                    calib_rows.extend(rows_c)
                    completed += 1
                    if verbose:
                        print(f"  Fold {fold_done + 1}/{n_folds} done ({completed}/{n_folds}).", flush=True)

                now = time.time()
                if verbose and pending and (now - last_beat) >= heartbeat:
                    elapsed = now - t_parallel
                    print(
                        f"  Still running... {completed}/{n_folds} folds done (elapsed {elapsed:.0f}s)",
                        flush=True,
                    )
                    last_beat = now

        if verbose:
            print(f"  Parallel CV done in {time.time() - t_parallel:.1f}s.", flush=True)

        # Skip the sequential path below.
        folds_iter = []


    for fold_idx, test_idx in folds_iter:

        if n_folds <= 1:
            train_idx = indices
        else:
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        if verbose:
            print(
                f"  Fold {fold_idx+1}/{n_folds}  (train={len(train_idx)}, test={len(test_idx)})",
                flush=True,
            )

        # Shared fold standardisation for PPLS models
        X_train_s, Y_train_s, X_test_s, _Y_test_s, _sx, sy = _standardize_train_test(
            X_train, Y_train, X_test, Y_test
        )

        # --- PPLS-SLM ---
        if verbose:
            print(f"    Fitting PPLS-SLM (starts={n_starts}, max_iter={slm_max_iter})...", flush=True)
        _t_fit = time.time()
        slm_params = _fit_ppls_params_slm(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            slm_max_iter=slm_max_iter,
            slm_optimizer=str(slm_cfg["optimizer"]),
            slm_gtol=float(slm_cfg["gtol"]),
            slm_xtol=float(slm_cfg["xtol"]),
            slm_barrier_tol=float(slm_cfg["barrier_tol"]),
            slm_constraint_slack=float(slm_cfg["constraint_slack"]),
            slm_progress_every=int(slm_cfg["progress_every"]),
            slm_early_stop_patience=slm_cfg.get("early_stop_patience"),
            slm_early_stop_rel_improvement=slm_cfg.get("early_stop_rel_improvement"),
            slm_data_start=bool(slm_cfg["data_start"]),
            slm_verbose=bool(slm_cfg.get("verbose", True)),
        )


        if verbose:
            meta = slm_params.get("_meta", {}) if isinstance(slm_params, dict) else {}
            iters = meta.get("n_iterations")
            iters_s = f", iters={iters}" if iters is not None else ""
            print(f"    Done PPLS-SLM in {time.time()-_t_fit:.1f}s{iters_s}.", flush=True)

        slm_method = _slm_method_name_from_cfg(slm_cfg)
        inner_n_starts = int(slm_cfg.get("adaptive_shrinkage_inner_n_starts", max(1, min(3, int(n_starts)))))
        inner_max_iter = int(slm_cfg.get("adaptive_shrinkage_inner_max_iter", max(40, min(int(slm_max_iter), 120))))
        shrinkage_alpha_slm, covariance_scale_slm, _cv_table = _select_slm_shrinkage_and_scale(
            X_train_s,
            Y_train_s,
            params_outer=slm_params,
            fold_seed=int(seed + fold_idx),
            slm_cfg=slm_cfg,
            fit_inner_model_fn=lambda X_in, Y_in, inner_seed: _fit_ppls_params_slm(
                X_in,
                Y_in,
                r=r,
                n_starts=inner_n_starts,
                seed=int(inner_seed),
                slm_max_iter=inner_max_iter,
                slm_optimizer=str(slm_cfg["optimizer"]),
                slm_gtol=float(slm_cfg["gtol"]),
                slm_xtol=float(slm_cfg["xtol"]),
                slm_barrier_tol=float(slm_cfg["barrier_tol"]),
                slm_constraint_slack=float(slm_cfg["constraint_slack"]),
                slm_progress_every=int(slm_cfg["progress_every"]),
                slm_early_stop_patience=slm_cfg.get("early_stop_patience"),
                slm_early_stop_rel_improvement=slm_cfg.get("early_stop_rel_improvement"),
                slm_data_start=bool(slm_cfg["data_start"]),
                slm_verbose=bool(slm_cfg.get("adaptive_shrinkage_verbose", False)),
            ),
            progress_label=f"    [fold {fold_idx + 1}] adaptive-shrinkage",
        )

        y_pred_slm_s, Cov_slm_s = _predict_ppls(X_test_s, params=slm_params, shrinkage_alpha=shrinkage_alpha_slm)
        Cov_slm_s = float(covariance_scale_slm) * Cov_slm_s



        y_pred_slm = _unstandardize_y_pred(y_pred_slm_s, sy)

        m_slm = compute_regression_metrics(Y_test, y_pred_slm)
        metrics_rows.append(
            {
                "fold": fold_idx + 1,
                "method": slm_method,
                "mse": m_slm.mse,
                "mae": m_slm.mae,
                "r2": m_slm.r2_mean,
                "shrinkage_alpha": float(shrinkage_alpha_slm),
            }
        )

        # Calibration (percent)
        Cov_slm = _unstandardize_cov_y(Cov_slm_s, sy)
        for a in alphas:
            lower, upper = compute_credible_intervals(y_pred_slm, Cov_slm, alpha=float(a))
            cov = 100.0 * empirical_coverage(Y_test, lower, upper)
            calib_rows.append(
                {
                    "fold": fold_idx + 1,
                    "method": slm_method,
                    "alpha": float(a),
                    "coverage": cov,
                    "shrinkage_alpha": float(shrinkage_alpha_slm),
                    "covariance_scale": float(covariance_scale_slm),
                }

            )


        # --- PPLS-EM ---
        if verbose:
            print(f"    Fitting PPLS-EM  (starts={n_starts}, max_iter={em_max_iter}, tol={em_tol})...", flush=True)
        _t_fit = time.time()
        em_params = _fit_ppls_params_em(
            X_train_s,
            Y_train_s,
            r=r,
            n_starts=n_starts,
            seed=seed + fold_idx,
            em_max_iter=em_max_iter,
            em_tol=em_tol,
            em_data_start=bool(slm_cfg["data_start"]),
        )

        if verbose:
            meta = em_params.get("_meta", {}) if isinstance(em_params, dict) else {}
            iters = meta.get("n_iterations")
            iters_s = f", iters={iters}" if iters is not None else ""
            print(f"    Done PPLS-EM in {time.time()-_t_fit:.1f}s{iters_s}.", flush=True)

        y_pred_em_s, Cov_em_s = _predict_ppls(X_test_s, params=em_params)

        y_pred_em = _unstandardize_y_pred(y_pred_em_s, sy)

        m_em = compute_regression_metrics(Y_test, y_pred_em)
        metrics_rows.append(
            {
                "fold": fold_idx + 1,
                "method": "PPLS-EM",
                "mse": m_em.mse,
                "mae": m_em.mae,
                "r2": m_em.r2_mean,
            }
        )

        Cov_em = _unstandardize_cov_y(Cov_em_s, sy)
        for a in alphas:
            lower, upper = compute_credible_intervals(y_pred_em, Cov_em, alpha=float(a))
            cov = 100.0 * empirical_coverage(Y_test, lower, upper)
            calib_rows.append({"fold": fold_idx + 1, "method": "PPLS-EM", "alpha": float(a), "coverage": cov})

        # --- Baselines ---
        if include_baselines:
            if verbose:
                print(f"    Fitting PLSR (n_components={r})...", flush=True)
            _t_fit = time.time()
            plsr = run_plsr_prediction(X_train, Y_train, X_test, Y_test, n_components=r)
            if verbose:
                print(f"    Done PLSR in {time.time()-_t_fit:.1f}s.", flush=True)

            m_plsr = plsr["metrics"]
            metrics_rows.append(
                {
                    "fold": fold_idx + 1,
                    "method": "PLSR",
                    "mse": m_plsr.mse,
                    "mae": m_plsr.mae,
                    "r2": m_plsr.r2_mean,
                }
            )

            if verbose:
                print("    Fitting RidgeCV...", flush=True)
            _t_fit = time.time()
            ridge = run_ridge_prediction(X_train, Y_train, X_test, Y_test)
            if verbose:
                print(f"    Done RidgeCV in {time.time()-_t_fit:.1f}s.", flush=True)

            m_ridge = ridge["metrics"]
            metrics_rows.append(
                {
                    "fold": fold_idx + 1,
                    "method": "Ridge",
                    "mse": m_ridge.mse,
                    "mae": m_ridge.mae,
                    "r2": m_ridge.r2_mean,
                }
            )


    metrics_per_fold = pd.DataFrame(metrics_rows)

    def _summarise(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for method, sub in df.groupby("method", sort=False):
            out.append(
                {
                    "method": method,
                    "mse_mean": float(sub["mse"].mean()),
                    "mse_std": float(sub["mse"].std(ddof=1)),
                    "mae_mean": float(sub["mae"].mean()),
                    "mae_std": float(sub["mae"].std(ddof=1)),
                    "r2_mean": float(sub["r2"].mean()),
                    "r2_std": float(sub["r2"].std(ddof=1)),
                }
            )
        return pd.DataFrame(out)

    metrics_summary = _summarise(metrics_per_fold)

    calib_long = pd.DataFrame(calib_rows)

    # Summary by alpha for SLM/EM (method name depends on slm_cfg)
    slm_method = _slm_method_name_from_cfg(slm_cfg)

    calib_summary_rows = []
    for a, suba in calib_long.groupby("alpha", sort=True):
        row = {"alpha": float(a), "expected_coverage": 100.0 * (1.0 - float(a))}
        for method in (slm_method, "PPLS-EM"):
            subm = suba[suba["method"] == method]
            row[f"{method}_mean"] = float(subm["coverage"].mean()) if len(subm) else float("nan")
            row[f"{method}_std"] = float(subm["coverage"].std(ddof=1)) if len(subm) > 1 else float("nan")
        calib_summary_rows.append(row)

    calib_summary = pd.DataFrame(calib_summary_rows)


    return metrics_per_fold, metrics_summary, calib_summary


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisation (optional)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(calib_summary: pd.DataFrame, output_dir: str, *, slm_method: str = "PPLS-SLM"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not found – skipping calibration plot.")
        return

    x = calib_summary["expected_coverage"].to_numpy()

    slm_col = f"{slm_method}_mean"
    if slm_col not in calib_summary.columns:
        warnings.warn(f"Missing calibration column: {slm_col} – skipping plot.")
        return

    y_slm = calib_summary[slm_col].to_numpy()
    y_em = calib_summary["PPLS-EM_mean"].to_numpy() if "PPLS-EM_mean" in calib_summary.columns else None

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x, y_slm, "o-", label=slm_method)
    if y_em is not None:
        ax.plot(x, y_em, "s-", label="PPLS-EM")

    ax.plot([x.min(), x.max()], [x.min(), x.max()], "k--", linewidth=1.2, label="Perfect")

    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Calibration of predictive credible intervals")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = os.path.join(output_dir, "calibration_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PPLS prediction experiment (synthetic)")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON (single source of truth)")
    return p.parse_args()


def main():
    import time

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
    pred_cfg = get_experiment_cfg(cfg, "prediction")

    require_keys(
        pred_cfg,
        [
            "thread_limit",
            "output_dir",
            "p",
            "q",
            "r",
            "n_samples",
            "n_folds",
            "n_starts",
            "seed",
            "sigma_e2",
            "sigma_f2",
            "sigma_h2",
            "max_iter",
            "em_tol",
            "plot",
            "no_baselines",
            # speed & stability knobs
            "parallel_folds",
            "fold_workers",
            "cv_heartbeat_sec",
            "slm_optimizer",
            "slm_gtol",
            "slm_xtol",
            "slm_barrier_tol",
            "slm_constraint_slack",
            "slm_progress_every",
            "slm_early_stop_patience",
            "slm_early_stop_rel_improvement",
            "slm_data_start",
            "slm_verbose",
        ],

        ctx="experiments.prediction",
    )


    # Coerce basic types
    for k in (
        "thread_limit",
        "p",
        "q",
        "r",
        "n_samples",
        "n_folds",
        "n_starts",
        "seed",
        "max_iter",
        "fold_workers",
        "cv_heartbeat_sec",
        "slm_progress_every",
        "slm_early_stop_patience",
    ):
        coerce_int(pred_cfg, k, ctx="experiments.prediction")

    for k in (
        "sigma_e2",
        "sigma_f2",
        "sigma_h2",
        "em_tol",
        "slm_gtol",
        "slm_xtol",
        "slm_barrier_tol",
        "slm_constraint_slack",
        "slm_early_stop_rel_improvement",
    ):
        coerce_float(pred_cfg, k, ctx="experiments.prediction")
    for k in ("plot", "no_baselines", "parallel_folds", "slm_data_start", "slm_verbose"):
        coerce_bool(pred_cfg, k, ctx="experiments.prediction")



    output_dir = str(pred_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    # Runtime thread limiting (more reliable than env vars if NumPy was imported early).
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(pred_cfg["thread_limit"]))
    except Exception:
        pass

    print(f"[prediction] using: {__file__}", flush=True)
    print(f"[prediction] config: {args.config}", flush=True)

    print("=" * 60, flush=True)
    print("Prediction experiment (synthetic)", flush=True)
    print("=" * 60, flush=True)

    p_dim = int(pred_cfg["p"])
    q_dim = int(pred_cfg["q"])
    r = int(pred_cfg["r"])
    n_samples = int(pred_cfg["n_samples"])
    n_folds = int(pred_cfg["n_folds"])
    n_starts = int(pred_cfg["n_starts"])
    seed = int(pred_cfg["seed"])

    sigma_e2 = float(pred_cfg["sigma_e2"])
    sigma_f2 = float(pred_cfg["sigma_f2"])
    sigma_h2 = float(pred_cfg["sigma_h2"])

    max_iter = int(pred_cfg["max_iter"])
    em_tol = float(pred_cfg["em_tol"])

    parallel_folds = bool(pred_cfg["parallel_folds"])
    fold_workers = int(pred_cfg["fold_workers"])
    cv_heartbeat_sec = int(pred_cfg["cv_heartbeat_sec"])


    slm_early_stop_patience = int(pred_cfg["slm_early_stop_patience"])
    if slm_early_stop_patience <= 0:
        slm_early_stop_patience = None

    slm_early_stop_rel_improvement = float(pred_cfg["slm_early_stop_rel_improvement"])
    if slm_early_stop_rel_improvement <= 0:
        slm_early_stop_rel_improvement = None

    slm_cfg = {
        "optimizer": str(pred_cfg["slm_optimizer"]),
        "gtol": float(pred_cfg["slm_gtol"]),
        "xtol": float(pred_cfg["slm_xtol"]),
        "barrier_tol": float(pred_cfg["slm_barrier_tol"]),
        "constraint_slack": float(pred_cfg["slm_constraint_slack"]),
        "progress_every": int(pred_cfg["slm_progress_every"]),
        "early_stop_patience": slm_early_stop_patience,
        "early_stop_rel_improvement": slm_early_stop_rel_improvement,
        "data_start": bool(pred_cfg["slm_data_start"]),
        "verbose": bool(pred_cfg["slm_verbose"]),
        "n_starts": int(n_starts),
        "max_iter": int(max_iter),
        # Adaptive shrinkage at prediction time (does not change estimation).
        "adaptive_shrinkage": bool(pred_cfg.get("slm_adaptive_shrinkage", False)),
        "adaptive_shrinkage_mode": str(pred_cfg.get("slm_adaptive_shrinkage_mode", "nested_refit")),
        "adaptive_shrinkage_verbose": bool(pred_cfg.get("slm_adaptive_shrinkage_verbose", False)),
        "adaptive_shrinkage_inner_n_starts": int(pred_cfg.get("slm_adaptive_shrinkage_inner_n_starts", max(1, min(3, int(n_starts))))),
        "adaptive_shrinkage_inner_max_iter": int(pred_cfg.get("slm_adaptive_shrinkage_inner_max_iter", max(40, min(int(max_iter), 120)))),
        "shrinkage_alpha_grid": list(
            pred_cfg.get(
                "slm_shrinkage_alpha_grid",
                [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
            )
        ),
        "adaptive_shrinkage_folds": int(pred_cfg.get("slm_adaptive_shrinkage_folds", 5)),
    }





    print(f"  p={p_dim}, q={q_dim}, r={r}, N={n_samples}")
    print(f"  folds={n_folds}, starts={n_starts}")
    print(f"  threads: thread_limit={int(pred_cfg['thread_limit'])}")
    print(f"  CV: parallel_folds={parallel_folds}, fold_workers={fold_workers}, heartbeat={cv_heartbeat_sec}s")

    print(f"  SLM/EM: max_iter={max_iter} (spectral noise pre-estimation), tol={em_tol}")
    print(
        "  SLM: "
        f"optimizer={slm_cfg['optimizer']}, "
        f"gtol={slm_cfg['gtol']}, xtol={slm_cfg['xtol']}, barrier_tol={slm_cfg['barrier_tol']}, "
        f"slack={slm_cfg['constraint_slack']}, data_start={slm_cfg['data_start']}, "
        f"early_stop_patience={slm_cfg['early_stop_patience']}, early_stop_rel_improvement={slm_cfg['early_stop_rel_improvement']}, "
        f"adaptive_mode={slm_cfg['adaptive_shrinkage_mode']}, adaptive_inner_folds={slm_cfg['adaptive_shrinkage_folds']}, "
        f"adaptive_inner_starts={slm_cfg['adaptive_shrinkage_inner_n_starts']}, adaptive_inner_max_iter={slm_cfg['adaptive_shrinkage_inner_max_iter']}"
    )


    print("=" * 60)

    print("\nGenerating synthetic PPLS data...", flush=True)
    print("  - generating orthonormal loadings + sampling latent variables", flush=True)
    t0 = time.time()
    X, Y, _true_params = generate_ppls_data(
        p=p_dim,
        q=q_dim,
        r=r,
        n_samples=n_samples,
        sigma_e2=sigma_e2,
        sigma_f2=sigma_f2,
        sigma_h2=sigma_h2,
        seed=seed,
    )
    print(f"  X: {X.shape}, Y: {Y.shape}", flush=True)
    print(f"  data generation took {time.time()-t0:.3f}s", flush=True)

    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\nRunning {n_folds}-fold CV benchmark...", flush=True)
    metrics_per_fold, metrics_summary, calib_summary = kfold_prediction_benchmark(
        X,
        Y,
        r=r,
        n_folds=n_folds,
        n_starts=n_starts,
        seed=seed,
        slm_max_iter=max_iter,
        em_max_iter=max_iter,
        em_tol=em_tol,
        include_baselines=(not bool(pred_cfg["no_baselines"])),
        thread_limit=int(pred_cfg["thread_limit"]),
        parallel_folds=parallel_folds,
        fold_workers=fold_workers,
        cv_heartbeat_sec=cv_heartbeat_sec,
        slm_cfg=slm_cfg,
        alphas=alphas,
        verbose=True,
    )



    # Save
    metrics_per_fold.to_csv(os.path.join(output_dir, "prediction_metrics_per_fold.csv"), index=False)
    metrics_summary.to_csv(os.path.join(output_dir, "prediction_metrics_summary.csv"), index=False)
    calib_summary.to_csv(os.path.join(output_dir, "calibration_comparison.csv"), index=False)

    # Diagnostic: selected adaptive shrinkage alphas (if enabled)
    if "shrinkage_alpha" in metrics_per_fold.columns:
        alpha_diag = (
            metrics_per_fold[["fold", "method", "shrinkage_alpha"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["method", "fold"])
        )
        if len(alpha_diag):
            alpha_diag.to_csv(os.path.join(output_dir, "selected_shrinkage_alpha.csv"), index=False)


    print("\n── Prediction metrics (mean ± std across folds) ──")
    disp = metrics_summary.copy()
    for k in ("mse", "mae", "r2"):
        disp[f"{k}"] = disp[f"{k}_mean"].map(lambda x: f"{x:.4g}") + " ± " + disp[f"{k}_std"].map(lambda x: f"{x:.4g}")
    print(disp[["method", "mse", "mae", "r2"]].to_string(index=False))

    print("\n── Calibration summary (mean ± std, %) ──")
    slm_method = _slm_method_name_from_cfg(slm_cfg)
    disp_c = calib_summary.copy()

    slm_mean_col = f"{slm_method}_mean"
    slm_std_col = f"{slm_method}_std"
    if (slm_mean_col in disp_c.columns) and (slm_std_col in disp_c.columns):
        disp_c[slm_method] = disp_c[slm_mean_col].map(lambda x: f"{x:.2f}") + " ± " + disp_c[slm_std_col].map(lambda x: f"{x:.2f}")

    if ("PPLS-EM_mean" in disp_c.columns) and ("PPLS-EM_std" in disp_c.columns):
        disp_c["PPLS-EM"] = disp_c["PPLS-EM_mean"].map(lambda x: f"{x:.2f}") + " ± " + disp_c["PPLS-EM_std"].map(lambda x: f"{x:.2f}")

    cols = ["alpha", "expected_coverage"]
    if slm_method in disp_c.columns:
        cols.append(slm_method)
    if "PPLS-EM" in disp_c.columns:
        cols.append("PPLS-EM")
    print(disp_c[cols].to_string(index=False))


    if bool(pred_cfg["plot"]):
        plot_calibration(calib_summary, output_dir, slm_method=_slm_method_name_from_cfg(slm_cfg))

        print(f"\nSaved plot: {output_dir}/calibration_plot.png")

    print(f"\nResults saved to: {output_dir}/")
    print("Prediction experiment complete.")

    return metrics_summary


if __name__ == "__main__":
    main()

