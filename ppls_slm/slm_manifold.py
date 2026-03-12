"""SLM-Manifold: exact-feasible optimization on the PPLS product manifold.

This module provides a drop-in optimizer for the Scalar Likelihood Method (SLM)
that maintains exact orthonormality constraints

    W^T W = I_r,   C^T C = I_r

by optimizing on the product manifold

    St(p, r) × St(q, r) × R^r × R^r × R

using a smooth positive reparameterization for diagonal/scalar parameters.

We use a numerically stable softplus transform to avoid overflow in high-dimensional
or difficult runs:

    theta_t2 = softplus(theta_t2_tilde) + eps   (latent variances)
    b        = softplus(b_tilde)        + eps   (regression coefficients)
    sigma_h2 = softplus(sigma_h2_tilde) + eps   (latent noise)

This keeps strict positivity while preventing `exp` overflow. (sigma_e2, sigma_f2)
remain fixed at spectral pre-estimates.

Implementation notes
--------------------
- Uses Pymanopt if available.
- Uses sign-consistent thin-QR retraction (diag(R) > 0) on Stiefel factors.
- Supplies analytic Euclidean gradients and applies the log-coordinate chain rule.
- Includes a post-processing step to enforce identifiability ordering
  (theta_t2[i] * b[i]) decreasing by permuting latent components.

If Pymanopt is not installed, importing/using this module will raise an
ImportError with instructions.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .ppls_model import PPLSObjective


_POS_EPS = 1e-8


def _softplus(x: np.ndarray) -> np.ndarray:
    """Stable softplus: log(1 + exp(x)) without overflow."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid for chain rule with softplus."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    """Approximate inverse of softplus for y>0 (stable across regimes)."""
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 1e-12)
    # For large y, softplus(x) ~ x.
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


def _qr_with_positive_diagonal(A: np.ndarray) -> np.ndarray:
    """QR-based retraction with consistent sign (positive diagonal in R)."""
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    s = np.sign(d)
    s[s == 0] = 1.0
    return Q * s


def _enforce_identifiability_order(
    W: np.ndarray,
    C: np.ndarray,
    theta_t2: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reorder components so (theta_t2[i] * b[i]) is decreasing."""
    score = np.asarray(theta_t2, dtype=float) * np.asarray(b, dtype=float)
    order = np.argsort(score)[::-1]
    return W[:, order], C[:, order], theta_t2[order], b[order]


def _objective_from_parts(
    obj: PPLSObjective,
    W: np.ndarray,
    C: np.ndarray,
    theta_t2: np.ndarray,
    b: np.ndarray,
    sigma_h2: float,
) -> float:
    se2, sf2 = float(obj.sigma_e2), float(obj.sigma_f2)
    ln_det = obj._compute_ln_det(W, C, b, theta_t2, se2, sf2, float(sigma_h2))
    trace = obj._compute_trace(W, C, b, theta_t2, se2, sf2, float(sigma_h2))
    out = float(ln_det + trace)
    return out if np.isfinite(out) else float(1e10)


def _euclidean_gradient_from_parts(
    obj: PPLSObjective,
    W: np.ndarray,
    C: np.ndarray,
    theta_t2: np.ndarray,
    b: np.ndarray,
    sigma_h2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytic Euclidean gradients for (W, C, theta_t2, b, sigma_h2)."""

    S_xx = obj.S_xx
    S_xy = obj.S_xy
    S_yy = obj.S_yy

    se2, sf2 = float(obj.sigma_e2), float(obj.sigma_f2)
    sh2 = float(sigma_h2)

    theta_t2 = np.asarray(theta_t2, dtype=float)
    b = np.asarray(b, dtype=float)

    # Guard against invalid values (optimizer should stay in the nonnegative orthant).
    # Note: for the PCCA specialization we allow sigma_h2 = 0.
    if np.any(theta_t2 <= 0) or np.any(b <= 0) or (sh2 < 0):
        p, r = W.shape
        q = C.shape[0]
        return (
            np.zeros((p, r)),
            np.zeros((q, r)),
            np.zeros_like(theta_t2),
            np.zeros_like(b),
            np.zeros((1,)),
        )


    # Per-component denominators and precision weights.
    D = (sf2 + sh2) * (theta_t2 + se2) + (b**2) * theta_t2 * se2

    Phi_x = (sf2 + sh2) * theta_t2 / D
    Phi_y = (sh2 * (theta_t2 + se2) + (b**2) * theta_t2 * se2) / D
    Phi_xy = (b * theta_t2) / D

    hat_Phi_x = Phi_x / se2
    hat_Phi_y = Phi_y / sf2

    # --- W, C gradients (paper Eqs. (20)-(21))
    SxxW = S_xx @ W
    SxyC = S_xy @ C
    grad_W = -2.0 * (SxxW * hat_Phi_x[np.newaxis, :] + SxyC * Phi_xy[np.newaxis, :])

    SyyC = S_yy @ C
    SyxW = S_xy.T @ W
    grad_C = -2.0 * (SyyC * hat_Phi_y[np.newaxis, :] + SyxW * Phi_xy[np.newaxis, :])

    # --- Scalar gradients
    # A_i, B_i, C_i as described in the trace expansion.
    A = np.sum(W * SxxW, axis=0) / se2
    Bq = np.sum(C * SyyC, axis=0) / sf2
    Cc = 2.0 * np.sum(W * SxyC, axis=0)

    # Useful numerators.
    Nx = (sf2 + sh2) * theta_t2
    Ny = sh2 * (theta_t2 + se2) + (b**2) * theta_t2 * se2
    Nxy = b * theta_t2

    # D derivatives.
    dD_db = 2.0 * b * theta_t2 * se2
    dD_dtheta = (sf2 + sh2) + (b**2) * se2
    dD_dsh2 = theta_t2 + se2

    # Numerator derivatives.
    dNx_db = np.zeros_like(b)
    dNx_dtheta = (sf2 + sh2) * np.ones_like(theta_t2)
    dNx_dsh2 = theta_t2

    dNy_db = 2.0 * b * theta_t2 * se2
    dNy_dtheta = sh2 + (b**2) * se2
    dNy_dsh2 = theta_t2 + se2

    dNxy_db = theta_t2
    dNxy_dtheta = b
    dNxy_dsh2 = np.zeros_like(theta_t2)

    invD2 = 1.0 / (D**2)

    # dPhi = (dN * D - N * dD) / D^2
    dPhi_x_db = (dNx_db * D - Nx * dD_db) * invD2
    dPhi_x_dtheta = (dNx_dtheta * D - Nx * dD_dtheta) * invD2
    dPhi_x_dsh2 = (dNx_dsh2 * D - Nx * dD_dsh2) * invD2

    dPhi_y_db = (dNy_db * D - Ny * dD_db) * invD2
    dPhi_y_dtheta = (dNy_dtheta * D - Ny * dD_dtheta) * invD2
    dPhi_y_dsh2 = (dNy_dsh2 * D - Ny * dD_dsh2) * invD2

    dPhi_xy_db = (dNxy_db * D - Nxy * dD_db) * invD2
    dPhi_xy_dtheta = (dNxy_dtheta * D - Nxy * dD_dtheta) * invD2
    dPhi_xy_dsh2 = (dNxy_dsh2 * D - Nxy * dD_dsh2) * invD2

    dlogD_db = dD_db / D
    dlogD_dtheta = dD_dtheta / D
    dlogD_dsh2 = dD_dsh2 / D

    grad_b = dlogD_db - (A * dPhi_x_db + Bq * dPhi_y_db + Cc * dPhi_xy_db)
    grad_theta = dlogD_dtheta - (A * dPhi_x_dtheta + Bq * dPhi_y_dtheta + Cc * dPhi_xy_dtheta)

    grad_sigma_h2 = float(np.sum(dlogD_dsh2 - (A * dPhi_x_dsh2 + Bq * dPhi_y_dsh2 + Cc * dPhi_xy_dsh2)))

    return grad_W, grad_C, grad_theta, grad_b, np.array([grad_sigma_h2], dtype=float)


@dataclass
class ManifoldSolveResult:
    x: np.ndarray
    fun: float
    success: bool
    nit: int
    message: str = ""
    raw: Optional[Any] = None


def solve_slm_manifold_single_start(
    objective: PPLSObjective,
    theta0: np.ndarray,
    *,
    max_iter: int = 100,
    verbosity: int = 0,
) -> ManifoldSolveResult:

    """Run one manifold optimization starting from an SLM-style theta vector."""

    try:
        from pymanopt import Problem, function
        from pymanopt.manifolds import Euclidean, Product, Stiefel
        from pymanopt.optimizers import ConjugateGradient
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "SLM-Manifold requires 'pymanopt'. Install via: pip install pymanopt"
        ) from e

    class _StiefelQF(Stiefel):
        """Stiefel manifold with sign-consistent thin-QR retraction (diag(R) > 0)."""

        def retraction(self, point, tangent_vector):  # type: ignore[override]
            return _qr_with_positive_diagonal(point + tangent_vector)


    p, q, r = int(objective.p), int(objective.q), int(objective.r)

    W0, C0, B0, Sigma_t0, sigma_h2_0 = objective._theta_to_params(np.asarray(theta0, dtype=float))
    theta_t2_0 = np.maximum(np.diag(Sigma_t0).astype(float), 1e-8)
    b0 = np.maximum(np.diag(B0).astype(float), 1e-8)
    sigma_h2_0 = float(max(float(sigma_h2_0), 1e-8))

    W0 = _qr_with_positive_diagonal(W0)[:, :r]
    C0 = _qr_with_positive_diagonal(C0)[:, :r]

    W0, C0, theta_t2_0, b0 = _enforce_identifiability_order(W0, C0, theta_t2_0, b0)

    # We optimize positive parameters in log-coordinates on Euclidean factors.
    manifold = Product(
        [
            _StiefelQF(p, r),
            _StiefelQF(q, r),
            Euclidean(r, 1),  # theta_t2_tilde
            Euclidean(r, 1),  # b_tilde
            Euclidean(1, 1),  # sigma_h2_tilde
        ]
    )


    @function.numpy(manifold)
    def cost(W, C, theta_t2_tilde, b_tilde, sigma_h2_tilde):
        theta_t2_v = _softplus(np.asarray(theta_t2_tilde, dtype=float)).reshape(-1) + _POS_EPS
        b_v = _softplus(np.asarray(b_tilde, dtype=float)).reshape(-1) + _POS_EPS
        sh2 = float(_softplus(np.asarray(sigma_h2_tilde, dtype=float)).reshape(-1)[0] + _POS_EPS)
        return _objective_from_parts(objective, W, C, theta_t2_v, b_v, sh2)

    @function.numpy(manifold)
    def egrad(W, C, theta_t2_tilde, b_tilde, sigma_h2_tilde):
        theta_t2_tilde_v = np.asarray(theta_t2_tilde, dtype=float).reshape(-1)
        b_tilde_v = np.asarray(b_tilde, dtype=float).reshape(-1)
        sh2_tilde_s = float(np.asarray(sigma_h2_tilde, dtype=float).reshape(-1)[0])

        theta_t2_v = _softplus(theta_t2_tilde_v) + _POS_EPS
        b_v = _softplus(b_tilde_v) + _POS_EPS
        sh2 = float(_softplus(np.asarray([sh2_tilde_s]))[0] + _POS_EPS)

        gW, gC, gtheta, gb, gsh2 = _euclidean_gradient_from_parts(objective, W, C, theta_t2_v, b_v, sh2)

        # Chain rule for softplus: d softplus(x) / dx = sigmoid(x).
        gtheta_tilde = np.asarray(gtheta, dtype=float).reshape(-1) * _sigmoid(theta_t2_tilde_v)
        gb_tilde = np.asarray(gb, dtype=float).reshape(-1) * _sigmoid(b_tilde_v)
        gsh2_tilde = float(np.asarray(gsh2, dtype=float).reshape(-1)[0]) * float(_sigmoid(np.asarray([sh2_tilde_s]))[0])

        return (
            gW,
            gC,
            np.asarray(gtheta_tilde, dtype=float).reshape(r, 1),
            np.asarray(gb_tilde, dtype=float).reshape(r, 1),
            np.asarray([[gsh2_tilde]], dtype=float),
        )


    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad)

    # NOTE: Pymanopt has had minor API changes across versions; keep construction and
    # invocation conservative for compatibility.
    solver = ConjugateGradient(max_iterations=int(max_iter), verbosity=int(verbosity))

    initial_point = (
        W0,
        C0,
        np.log(np.asarray(theta_t2_0, dtype=float).reshape(r, 1)),
        np.log(np.asarray(b0, dtype=float).reshape(r, 1)),
        np.log(np.asarray([[sigma_h2_0]], dtype=float)),
    )

    try:
        result = solver.run(problem, initial_point=initial_point)
    except TypeError:
        # Older versions: positional initial point.
        result = solver.run(problem, initial_point)


    try:
        point = result.point
        fun = float(result.cost)
        nit = int(getattr(result, "iterations", getattr(result, "iteration", 0)))
    except Exception:
        # Fallback for version differences.
        point = getattr(result, "point", None)
        fun = float(getattr(result, "cost", np.nan))
        nit = int(getattr(result, "iterations", 0) or 0)

    if point is None:
        raise RuntimeError("Pymanopt solver returned no point")

    W_hat, C_hat, theta_t2_tilde_hat, b_tilde_hat, sh2_tilde_arr = point

    theta_t2_hat_v = _softplus(np.asarray(theta_t2_tilde_hat, dtype=float)).reshape(-1) + _POS_EPS
    b_hat_v = _softplus(np.asarray(b_tilde_hat, dtype=float)).reshape(-1) + _POS_EPS
    sigma_h2_hat = float(_softplus(np.asarray(sh2_tilde_arr, dtype=float)).reshape(-1)[0] + _POS_EPS)


    W_hat, C_hat, theta_t2_hat_v, b_hat_v = _enforce_identifiability_order(
        np.asarray(W_hat, dtype=float),
        np.asarray(C_hat, dtype=float),
        theta_t2_hat_v,
        b_hat_v,
    )

    theta_hat = objective._params_to_theta(
        W_hat,
        C_hat,
        np.diag(b_hat_v),
        np.diag(theta_t2_hat_v),
        float(sigma_h2_hat),
    )

    # Recompute objective value for safety.
    fun = float(objective.scalar_log_likelihood(theta_hat))

    return ManifoldSolveResult(
        x=theta_hat,
        fun=fun,
        success=bool(np.isfinite(fun)),
        nit=nit,
        message="pymanopt ConjugateGradient",
        raw=result,
    )
