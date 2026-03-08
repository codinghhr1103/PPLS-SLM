"""Shared data utilities for the `ppls_slm.apps` scripts.

These helpers are intentionally lightweight and keep behaviour stable across apps:
- fold-wise standardisation (fit on train, apply to test)
- inverse-transform helpers for predictions/covariances
- BRCA combined dataset loader (bundled CSV(.zip) format)
- deterministic feature screening on the training split only
"""


from __future__ import annotations

import io
import zipfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_brca_combined_raw(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the bundled BRCA combined dataset without global standardisation.

    The expected schema matches the repository's bundled artifact:
    - gene expression columns prefix: `rs_`
    - protein expression columns prefix: `pp_`

    Supports `.csv` and `.zip` containing a single `.csv`.
    """

    lower = str(path).lower()

    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if not names:
                raise ValueError(f"Empty zip file: {path}")
            data = z.read(names[0])
            df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_csv(path)

    rs_cols = [c for c in df.columns if str(c).startswith("rs_")]
    pp_cols = [c for c in df.columns if str(c).startswith("pp_")]
    if not rs_cols or not pp_cols:
        raise ValueError(
            "BRCA combined dataset must contain `rs_` (genes) and `pp_` (proteins) columns. "
            f"Found rs_={len(rs_cols)}, pp_={len(pp_cols)}"
        )

    X = df[rs_cols].to_numpy(dtype=float)
    Y = df[pp_cols].to_numpy(dtype=float)

    # Defensive cleaning: drop rows/cols with NaN/inf.
    good_rows = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[good_rows]
    Y = Y[good_rows]

    good_x_cols = np.isfinite(X).all(axis=0)
    good_y_cols = np.isfinite(Y).all(axis=0)
    X = X[:, good_x_cols]
    Y = Y[:, good_y_cols]

    return X, Y


def standardize_train_test(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
):
    """Standardise X and Y using training-set statistics only.

    Returns
    -------
    X_train_s, Y_train_s, X_test_s, Y_test_s, scaler_x, scaler_y
    """

    from sklearn.preprocessing import StandardScaler

    sx = StandardScaler()
    sy = StandardScaler()

    X_train_s = sx.fit_transform(X_train)
    Y_train_s = sy.fit_transform(Y_train)
    X_test_s = sx.transform(X_test)
    Y_test_s = sy.transform(Y_test)

    return X_train_s, Y_train_s, X_test_s, Y_test_s, sx, sy


def unstandardize_y(y_s: np.ndarray, scaler_y) -> np.ndarray:
    """Inverse-transform standardized Y back to the original scale."""

    return scaler_y.inverse_transform(y_s)


def unstandardize_cov(Cov_s: np.ndarray, scaler_y) -> np.ndarray:
    """Inverse-transform a covariance matrix from standardized-Y space.

    If Y_s = (Y - mu) / scale element-wise, then Cov(Y) = D Cov(Y_s) D with D=diag(scale).
    """

    scale = getattr(scaler_y, "scale_", None)
    if scale is None:
        return Cov_s
    D = np.diag(np.asarray(scale, dtype=float))
    return D @ Cov_s @ D


def _top_k_indices_from_scores(scores: np.ndarray, k: Optional[int]) -> Optional[np.ndarray]:
    if k is None:
        return None
    k = int(k)
    if k <= 0 or k >= int(scores.shape[0]):
        return None

    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    idx_desc = np.argsort(-scores, kind="mergesort")
    return np.sort(idx_desc[:k])


def _cross_view_screen_scores(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return cross-view relevance scores for X and Y features.

    The score is the RMS empirical cross-correlation with the opposite view,
    computed on the training split only after z-scoring each feature.
    """

    Xc = np.asarray(X, dtype=float)
    Yc = np.asarray(Y, dtype=float)
    if Xc.ndim != 2 or Yc.ndim != 2:
        raise ValueError("X and Y must be 2D arrays")
    if Xc.shape[0] != Yc.shape[0]:
        raise ValueError("X and Y must have the same number of rows")

    Xc = Xc - np.mean(Xc, axis=0, keepdims=True)
    Yc = Yc - np.mean(Yc, axis=0, keepdims=True)

    sx = np.std(Xc, axis=0, ddof=0)
    sy = np.std(Yc, axis=0, ddof=0)
    sx_safe = np.where(sx > 1e-12, sx, 1.0)
    sy_safe = np.where(sy > 1e-12, sy, 1.0)

    Xs = Xc / sx_safe[np.newaxis, :]
    Ys = Yc / sy_safe[np.newaxis, :]

    denom = float(max(1, Xs.shape[0] - 1))
    cross_corr = (Xs.T @ Ys) / denom

    x_scores = np.sqrt(np.mean(cross_corr**2, axis=1))
    y_scores = np.sqrt(np.mean(cross_corr**2, axis=0))
    return x_scores, y_scores


def _standardize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    mu = float(np.mean(scores))
    sd = float(np.std(scores, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(scores, dtype=float)
    return (scores - mu) / sd


def top_variance_indices(M: np.ndarray, k: Optional[int]) -> Optional[np.ndarray]:
    """Select top-k columns by variance.

    IMPORTANT: This must be computed on the training split only.

    Determinism: uses stable sorting and returns indices in ascending order.
    """

    v = np.var(np.asarray(M, dtype=float), axis=0)
    return _top_k_indices_from_scores(v, k)


def top_cross_covariance_indices(M_self: np.ndarray, M_other: np.ndarray, k: Optional[int], *, target: str) -> Optional[np.ndarray]:
    """Select top-k features by cross-view relevance.

    Parameters
    ----------
    target:
        Either ``"x"`` or ``"y"`` to indicate which view should be ranked.
    """

    x_scores, y_scores = _cross_view_screen_scores(M_self, M_other)
    tgt = str(target).strip().lower()
    if tgt == "x":
        return _top_k_indices_from_scores(x_scores, k)
    if tgt == "y":
        return _top_k_indices_from_scores(y_scores, k)
    raise ValueError(f"target must be 'x' or 'y', got {target!r}")


def select_feature_indices(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    x_top_k: Optional[int],
    y_top_k: Optional[int],
    method: str = "variance",
    hybrid_mix: float = 0.5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Select BRCA features using a train-fold-only screening rule.

    Supported methods
    -----------------
    ``variance``:
        Keep the original unsupervised top-variance screen.
    ``cross_covariance`` / ``supervised``:
        Rank each feature by its RMS empirical cross-correlation with the opposite view.
    ``hybrid``:
        Weighted combination of standardized variance and cross-view scores.
    """

    method_key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    if method_key in ("variance", "var"):
        return top_variance_indices(X_train, x_top_k), top_variance_indices(Y_train, y_top_k)

    x_cross, y_cross = _cross_view_screen_scores(X_train, Y_train)
    if method_key in ("cross_covariance", "crosscovariance", "crosscov", "supervised"):
        return _top_k_indices_from_scores(x_cross, x_top_k), _top_k_indices_from_scores(y_cross, y_top_k)

    if method_key == "hybrid":
        mix = float(np.clip(hybrid_mix, 0.0, 1.0))
        x_var = np.var(np.asarray(X_train, dtype=float), axis=0)
        y_var = np.var(np.asarray(Y_train, dtype=float), axis=0)
        x_score = (1.0 - mix) * _standardize_scores(x_var) + mix * _standardize_scores(x_cross)
        y_score = (1.0 - mix) * _standardize_scores(y_var) + mix * _standardize_scores(y_cross)
        return _top_k_indices_from_scores(x_score, x_top_k), _top_k_indices_from_scores(y_score, y_top_k)

    raise ValueError(f"Unsupported feature screening method: {method}")

