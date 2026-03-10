"""BCD-SLM: Block Coordinate Descent for the Scalar Likelihood Method.

This module implements the NeurIPS 2025 paper's Algorithm 1 (Section 4.4):
- alternate W-step and C-step on Stiefel manifolds (few Riemannian CG steps)
- update diagonal parameters (theta_t^2, b) via closed-form conditional updates
- update sigma_h^2 via 1D bounded search

Design goals
------------
- Match the objective used by SLM-Fixed: the same scalar negative log-likelihood
  implemented in `ppls_slm.ppls_model.PPLSObjective`.
- Keep (sigma_e^2, sigma_f^2) fixed (spectral pre-estimates by default).
- Provide a drop-in algorithm class with the same `fit()` return dict keys as
  `ScalarLikelihoodMethod.fit()`.

Notes
-----
- This implementation intentionally reuses internal helpers from `slm_manifold.py`
  (objective evaluation and Euclidean gradients), since they already match the
  paper's scalar expansion and gradients.
- The W/C subproblems are solved approximately using pymanopt's
  `ConjugateGradient` for a small number of iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.optimize import minimize_scalar

from .algorithms import PPLSAlgorithm
from .ppls_model import NoiseEstimator, PPLSObjective
from .slm_manifold import (
    _enforce_identifiability_order as _enforce_identifiability_order_vec,
    _euclidean_gradient_from_parts,
    _objective_from_parts,
    _qr_with_positive_diagonal,
)


_EPS_POS = 1e-8


def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _riemannian_grad_stiefel(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project Euclidean gradient G to the Stiefel tangent space at X."""
    return G - X @ _sym(X.T @ G)


def compute_projected_quadratics(
    objective: PPLSObjective,
    W: np.ndarray,
    C: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute paper Eqs. (13)-(15): Qx, Qy, Qxy as length-r vectors."""

    S_xx = objective.S_xx
    S_yy = objective.S_yy
    S_xy = objective.S_xy
    se2 = float(objective.sigma_e2)
    sf2 = float(objective.sigma_f2)

    # diag(W^T Sxx W)
    SxxW = S_xx @ W
    Qx = np.sum(W * SxxW, axis=0) / max(se2, _EPS_POS)

    # diag(C^T Syy C)
    SyyC = S_yy @ C
    Qy = np.sum(C * SyyC, axis=0) / max(sf2, _EPS_POS)

    # 2 * diag(W^T Sxy C)
    SxyC = S_xy @ C
    Qxy = 2.0 * np.sum(W * SxyC, axis=0)

    return Qx.astype(float), Qy.astype(float), Qxy.astype(float)


def update_theta_t2_prop4(
    *,
    Qx: float,
    Qy: float,
    Qxy: float,
    b: float,
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
) -> float:
    """Proposition 4: closed-form update for theta_t^2 (scalar)."""
    se2 = float(sigma_e2)
    sf2 = float(sigma_f2)
    sh2 = float(sigma_h2)
    b = float(b)

    di = (sf2 + sh2) + (b**2) * se2
    ni = (sf2 + sh2) * float(Qx) + (sh2 + (b**2) * se2) * float(Qy) + b * float(Qxy)

    # theta = se2 * [ (ni - di)*(sf2+sh2) - di*sh2*Qy ] / di^2
    num = (ni - di) * (sf2 + sh2) - di * sh2 * float(Qy)
    out = se2 * num / (di**2)

    if not np.isfinite(out):
        return _EPS_POS
    return float(max(out, _EPS_POS))


def _pick_positive_real_root(roots: np.ndarray, *, imag_tol: float = 1e-10) -> Optional[float]:
    candidates: List[float] = []
    for z in np.asarray(roots):
        if abs(z.imag) <= imag_tol and z.real > 0:
            candidates.append(float(z.real))
    if not candidates:
        return None
    # There should be a unique positive real root under Qxy > 0.
    # Still, be conservative: pick the smallest positive root (stable if numerical noise creates near-duplicates).
    return float(min(candidates))


def update_b_prop5(
    *,
    Qx: float,
    Qy: float,
    Qxy: float,
    theta_t2: float,
    b_prev: float,
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
    fallback: str = "keep",
) -> float:
    """Proposition 5: update b via the unique positive real root of a cubic.

    When Qxy <= 0, the uniqueness guarantee no longer holds.
    We default to a conservative fallback (keep previous b).
    """

    Qxy = float(Qxy)
    if Qxy <= 0:
        if fallback == "keep":
            return float(max(float(b_prev), _EPS_POS))
        # Could add other fallbacks (e.g., bounded line search), but keep minimal.
        return float(max(float(b_prev), _EPS_POS))

    a = float(sigma_e2)
    beta = float(sigma_f2)
    gamma = float(sigma_h2)
    s = float(theta_t2)

    Qx = float(Qx)
    Qy = float(Qy)

    R = (beta + gamma) * ((s + a) * (1.0 - Qy) + s * Qx) + gamma * (s + a) * Qy

    c3 = 2.0 * (a**2) * s
    c2 = a * s * Qxy
    c1 = 2.0 * a * R
    c0 = -Qxy * (beta + gamma) * (s + a)

    coeffs = np.array([c3, c2, c1, c0], dtype=float)

    try:
        roots = np.roots(coeffs)
    except Exception:
        return float(max(float(b_prev), _EPS_POS))

    root = _pick_positive_real_root(roots)
    if root is None or (not np.isfinite(root)):
        return float(max(float(b_prev), _EPS_POS))

    return float(max(root, _EPS_POS))


def reduced_objective_from_Q(
    *,
    Qx: np.ndarray,
    Qy: np.ndarray,
    Qxy: np.ndarray,
    theta_t2: np.ndarray,
    b: np.ndarray,
    sigma_e2: float,
    sigma_f2: float,
    sigma_h2: float,
) -> float:
    r"""Compute sum_i \ell_i(theta_t2_i, b_i, sigma_h2) using the paper's scalar form."""


    se2 = float(sigma_e2)
    sf2 = float(sigma_f2)
    sh2 = float(sigma_h2)

    theta_t2 = np.asarray(theta_t2, dtype=float)
    b = np.asarray(b, dtype=float)
    Qx = np.asarray(Qx, dtype=float)
    Qy = np.asarray(Qy, dtype=float)
    Qxy = np.asarray(Qxy, dtype=float)

    if np.any(theta_t2 <= 0) or np.any(b <= 0) or not (se2 > 0) or not (sf2 > 0) or not (sh2 > 0):
        return float(1e10)

    D = (sf2 + sh2) * (theta_t2 + se2) + (b**2) * theta_t2 * se2
    N = (sf2 + sh2) * theta_t2 * Qx + sh2 * (theta_t2 + se2) * Qy + (b**2) * theta_t2 * se2 * Qy + b * theta_t2 * Qxy

    if np.any(D <= 0) or np.any(~np.isfinite(D)) or np.any(~np.isfinite(N)):
        return float(1e10)

    val = np.sum(np.log(D) - N / D)
    out = float(val)
    return out if np.isfinite(out) else float(1e10)


def update_sigma_h2_bounded(
    *,
    Qx: np.ndarray,
    Qy: np.ndarray,
    Qxy: np.ndarray,
    theta_t2: np.ndarray,
    b: np.ndarray,
    sigma_e2: float,
    sigma_f2: float,
    bounds: Tuple[float, float] = (1e-8, 100.0),
) -> float:
    """Update sigma_h^2 by minimizing the reduced objective over R_{++}."""

    lo, hi = float(bounds[0]), float(bounds[1])
    lo = max(lo, _EPS_POS)
    hi = max(hi, lo * 10.0)

    def f(sh2: float) -> float:
        return reduced_objective_from_Q(
            Qx=Qx,
            Qy=Qy,
            Qxy=Qxy,
            theta_t2=theta_t2,
            b=b,
            sigma_e2=sigma_e2,
            sigma_f2=sigma_f2,
            sigma_h2=float(sh2),
        )

    res = minimize_scalar(f, method="bounded", bounds=(lo, hi), options={"xatol": 1e-6})

    if getattr(res, "success", False) and np.isfinite(getattr(res, "x", np.nan)):
        return float(max(float(res.x), _EPS_POS))

    # Fallback: keep within bounds.
    x0 = float(np.clip((lo + hi) * 0.5, lo, hi))
    return float(max(x0, _EPS_POS))


@dataclass
class _SubproblemResult:
    X: np.ndarray
    success: bool
    nit: int
    fun: float


def _solve_stiefel_subproblem(
    *,
    objective: PPLSObjective,
    var0: np.ndarray,
    other: np.ndarray,
    theta_t2: np.ndarray,
    b: np.ndarray,
    sigma_h2: float,
    which: str,
    n_cg_steps: int,
) -> _SubproblemResult:
    """Solve W-step or C-step on a single Stiefel manifold using a few CG iterations."""

    try:
        from pymanopt import Problem, function
        from pymanopt.manifolds import Stiefel
        from pymanopt.optimizers import ConjugateGradient
    except Exception as e:  # pragma: no cover
        raise ImportError("BCD-SLM requires 'pymanopt'. Install via: pip install pymanopt") from e

    which_l = str(which).lower().strip()
    if which_l not in {"w", "c"}:
        raise ValueError("which must be 'w' or 'c'")

    p, r = var0.shape

    class _StiefelQF(Stiefel):
        def retraction(self, point, tangent_vector):  # type: ignore[override]
            return _qr_with_positive_diagonal(point + tangent_vector)

    manifold = _StiefelQF(p, r)

    @function.numpy(manifold)
    def cost(X):
        if which_l == "w":
            return _objective_from_parts(objective, X, other, theta_t2, b, float(sigma_h2))
        return _objective_from_parts(objective, other, X, theta_t2, b, float(sigma_h2))

    @function.numpy(manifold)
    def egrad(X):
        if which_l == "w":
            gW, _, _, _, _ = _euclidean_gradient_from_parts(objective, X, other, theta_t2, b, float(sigma_h2))
            return gW
        _, gC, _, _, _ = _euclidean_gradient_from_parts(objective, other, X, theta_t2, b, float(sigma_h2))
        return gC

    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=egrad)
    solver = ConjugateGradient(max_iterations=int(n_cg_steps), verbosity=0)

    try:
        result = solver.run(problem, initial_point=np.asarray(var0, dtype=float))
    except TypeError:
        result = solver.run(problem, np.asarray(var0, dtype=float))

    point = getattr(result, "point", None)
    cost_val = float(getattr(result, "cost", np.nan))
    nit = int(getattr(result, "iterations", getattr(result, "iteration", 0)) or 0)

    if point is None:
        return _SubproblemResult(X=np.asarray(var0, dtype=float), success=False, nit=nit, fun=float("nan"))

    X_hat = np.asarray(point, dtype=float)
    X_hat = _qr_with_positive_diagonal(X_hat)[:, :r]

    return _SubproblemResult(X=X_hat, success=bool(np.isfinite(cost_val)), nit=nit, fun=cost_val)


class BCDScalarLikelihoodMethod(PPLSAlgorithm):
    """BCD-SLM algorithm (paper Algorithm 1) with fixed (sigma_e^2, sigma_f^2)."""

    def __init__(
        self,
        p: int,
        q: int,
        r: int,
        *,
        max_outer_iter: int = 200,
        n_cg_steps_W: int = 5,
        n_cg_steps_C: int = 5,
        tolerance: float = 1e-4,
        use_noise_preestimation: bool = True,
        fixed_sigma_e2: Optional[float] = None,
        fixed_sigma_f2: Optional[float] = None,
        sigma_h2_bounds: Tuple[float, float] = (1e-8, 100.0),
    ):
        super().__init__(p, q, r)
        self.max_outer_iter = int(max_outer_iter)
        self.n_cg_steps_W = int(n_cg_steps_W)
        self.n_cg_steps_C = int(n_cg_steps_C)
        self.tolerance = float(tolerance)
        self.use_noise_preestimation = bool(use_noise_preestimation)
        self.fixed_sigma_e2 = fixed_sigma_e2
        self.fixed_sigma_f2 = fixed_sigma_f2
        self.sigma_h2_bounds = (float(sigma_h2_bounds[0]), float(sigma_h2_bounds[1]))

    def fit(self, X, Y, starting_points, experiment_dir=None, trial_id=None) -> Dict:
        N = X.shape[0]
        XY = np.hstack([X, Y])
        XY_c = XY - XY.mean(axis=0)
        S = (XY_c.T @ XY_c) / float(N)

        # Noise variances (fixed during optimization)
        if (self.fixed_sigma_e2 is not None) and (self.fixed_sigma_f2 is not None):
            sigma_e2 = float(self.fixed_sigma_e2)
            sigma_f2 = float(self.fixed_sigma_f2)
        elif self.use_noise_preestimation:
            sigma_e2, sigma_f2 = NoiseEstimator.estimate_noise_variances(X, Y, r=self.r)
        else:
            sigma_e2, sigma_f2 = 0.01, 0.01

        if self.fixed_sigma_e2 is not None:
            sigma_e2 = float(self.fixed_sigma_e2)
        if self.fixed_sigma_f2 is not None:
            sigma_f2 = float(self.fixed_sigma_f2)

        objective = PPLSObjective(self.p, self.q, self.r, S)
        objective.sigma_e2 = float(sigma_e2)
        objective.sigma_f2 = float(sigma_f2)

        best: Optional[Dict] = None

        for theta0 in starting_points:
            try:
                res = self._run_single_start(theta0, objective)
            except Exception:
                continue

            if best is None or float(res.get("objective_value", np.inf)) < float(best.get("objective_value", np.inf)):
                best = res

        if best is None:
            # Mirror other algorithms: return a failure dict with required keys.
            W = np.zeros((self.p, self.r))
            C = np.zeros((self.q, self.r))
            B = np.diag(np.ones(self.r))
            Sigma_t = np.diag(np.ones(self.r))
            return {
                "W": W,
                "C": C,
                "B": B,
                "Sigma_t": Sigma_t,
                "sigma_e2": float(sigma_e2),
                "sigma_f2": float(sigma_f2),
                "sigma_h2": float(0.01),
                "objective_value": float("inf"),
                "n_iterations": 0,
                "success": False,
            }

        return best

    def _run_single_start(self, theta0: np.ndarray, objective: PPLSObjective) -> Dict:
        W0, C0, B0, Sigma_t0, sigma_h2_0 = objective._theta_to_params(np.asarray(theta0, dtype=float))

        W = _qr_with_positive_diagonal(np.asarray(W0, dtype=float))[:, : self.r]
        C = _qr_with_positive_diagonal(np.asarray(C0, dtype=float))[:, : self.r]

        theta_t2 = np.maximum(np.diag(Sigma_t0).astype(float), _EPS_POS)
        b = np.maximum(np.diag(B0).astype(float), _EPS_POS)
        sigma_h2 = float(max(float(sigma_h2_0), _EPS_POS))

        # Enforce identifiability order at start.
        W, C, theta_t2, b = _enforce_identifiability_order_vec(W, C, theta_t2, b)

        prev_obj = float(_objective_from_parts(objective, W, C, theta_t2, b, sigma_h2))

        converged = False
        outer_it_used = 0

        for k in range(int(self.max_outer_iter)):
            outer_it_used = k + 1

            # ---- W-step
            wres = _solve_stiefel_subproblem(
                objective=objective,
                var0=W,
                other=C,
                theta_t2=theta_t2,
                b=b,
                sigma_h2=sigma_h2,
                which="w",
                n_cg_steps=self.n_cg_steps_W,
            )
            W = wres.X

            # ---- C-step
            cres = _solve_stiefel_subproblem(
                objective=objective,
                var0=C,
                other=W,
                theta_t2=theta_t2,
                b=b,
                sigma_h2=sigma_h2,
                which="c",
                n_cg_steps=self.n_cg_steps_C,
            )
            C = cres.X

            # ---- Diagonal updates based on projected quadratics
            Qx, Qy, Qxy = compute_projected_quadratics(objective, W, C)

            # theta_t2 update (Prop 4)
            theta_new = np.empty_like(theta_t2)
            for i in range(self.r):
                theta_new[i] = update_theta_t2_prop4(
                    Qx=float(Qx[i]),
                    Qy=float(Qy[i]),
                    Qxy=float(Qxy[i]),
                    b=float(b[i]),
                    sigma_e2=float(objective.sigma_e2),
                    sigma_f2=float(objective.sigma_f2),
                    sigma_h2=float(sigma_h2),
                )
            theta_t2 = np.maximum(theta_new, _EPS_POS)

            # b update (Prop 5)
            b_new = np.empty_like(b)
            for i in range(self.r):
                b_new[i] = update_b_prop5(
                    Qx=float(Qx[i]),
                    Qy=float(Qy[i]),
                    Qxy=float(Qxy[i]),
                    theta_t2=float(theta_t2[i]),
                    b_prev=float(b[i]),
                    sigma_e2=float(objective.sigma_e2),
                    sigma_f2=float(objective.sigma_f2),
                    sigma_h2=float(sigma_h2),
                    fallback="keep",
                )
            b = np.maximum(b_new, _EPS_POS)

            # ---- sigma_h2 update (1D bounded)
            sigma_h2 = update_sigma_h2_bounded(
                Qx=Qx,
                Qy=Qy,
                Qxy=Qxy,
                theta_t2=theta_t2,
                b=b,
                sigma_e2=float(objective.sigma_e2),
                sigma_f2=float(objective.sigma_f2),
                bounds=self.sigma_h2_bounds,
            )

            # ---- Reorder components for identifiability
            W, C, theta_t2, b = _enforce_identifiability_order_vec(W, C, theta_t2, b)

            # ---- Convergence check
            obj_val = float(_objective_from_parts(objective, W, C, theta_t2, b, sigma_h2))
            if abs(obj_val - prev_obj) <= float(self.tolerance):
                converged = True
                prev_obj = obj_val
                break

            prev_obj = obj_val

        # Gradient norm (diagnostic + optional convergence notion)
        gW, gC, gtheta, gb, gsh2 = _euclidean_gradient_from_parts(objective, W, C, theta_t2, b, sigma_h2)
        rgW = _riemannian_grad_stiefel(W, gW)
        rgC = _riemannian_grad_stiefel(C, gC)
        grad_norm = float(
            np.sqrt(
                np.linalg.norm(rgW, ord="fro") ** 2
                + np.linalg.norm(rgC, ord="fro") ** 2
                + np.linalg.norm(np.asarray(gtheta, dtype=float).reshape(-1)) ** 2
                + np.linalg.norm(np.asarray(gb, dtype=float).reshape(-1)) ** 2
                + float(np.asarray(gsh2, dtype=float).reshape(-1)[0]) ** 2
            )
        )

        if (not converged) and grad_norm <= float(self.tolerance):
            converged = True

        results = {
            "W": W,
            "C": C,
            "B": np.diag(np.asarray(b, dtype=float)),
            "Sigma_t": np.diag(np.asarray(theta_t2, dtype=float)),
            "sigma_e2": float(objective.sigma_e2),
            "sigma_f2": float(objective.sigma_f2),
            "sigma_h2": float(sigma_h2),
            "objective_value": float(prev_obj),
            "n_iterations": int(outer_it_used),
            "success": bool(converged and np.isfinite(prev_obj)),
            # Extra diagnostics (kept optional / non-breaking)
            "grad_norm": grad_norm,
        }
        return results
