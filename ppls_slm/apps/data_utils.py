"""Shared data utilities for the `ppls_slm.apps` scripts.

These helpers are intentionally lightweight and keep behaviour stable across apps:
- fold-wise standardisation (fit on train, apply to test)
- inverse-transform helpers for predictions/covariances
- BRCA combined dataset loader (bundled CSV(.zip) format)
- deterministic top-variance feature selection
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


def top_variance_indices(M: np.ndarray, k: Optional[int]) -> Optional[np.ndarray]:
    """Select top-k columns by variance.

    IMPORTANT: This must be computed on the training split only.

    Determinism: uses stable sorting and returns indices in ascending order.
    """

    if k is None:
        return None
    k = int(k)
    if k <= 0 or k >= M.shape[1]:
        return None

    v = np.var(np.asarray(M, dtype=float), axis=0)
    idx_desc = np.argsort(-v, kind="mergesort")
    idx = np.sort(idx_desc[:k])
    return idx
