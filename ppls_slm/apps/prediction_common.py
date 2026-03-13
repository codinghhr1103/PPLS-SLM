from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence


import numpy as np
import pandas as pd

from ppls_slm.algorithms import EMAlgorithm, InitialPointGenerator, ScalarLikelihoodMethod


def build_cv_folds(n_samples: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(int(n_samples))
    return [np.asarray(f, dtype=int) for f in np.array_split(indices, int(n_folds))]


def build_starting_points(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    use_data_driven_init: bool = True,
    data_driven_init_fn: Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = None,
) -> list[np.ndarray]:
    p, q = int(X_train_s.shape[1]), int(Y_train_s.shape[1])
    init_gen = InitialPointGenerator(p=p, q=q, r=int(r), n_starts=int(n_starts), random_seed=int(seed))
    starting_points = init_gen.generate_starting_points()
    if bool(use_data_driven_init) and starting_points and data_driven_init_fn is not None:
        starting_points[0] = data_driven_init_fn(X_train_s, Y_train_s, int(r))
    return starting_points


def _extract_params(res: Mapping[str, Any], *, include_meta: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "W": res["W"],
        "C": res["C"],
        "B": res["B"],
        "Sigma_t": res["Sigma_t"],
        "sigma_e2": res["sigma_e2"],
        "sigma_f2": res["sigma_f2"],
        "sigma_h2": res["sigma_h2"],
    }
    if include_meta:
        out["_meta"] = {k: res.get(v) for k, v in include_meta.items()}
    return out


def fit_ppls_slm(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
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
    early_stop_patience: Optional[int] = None,
    early_stop_rel_improvement: Optional[float] = None,
    use_data_driven_init: bool = True,
    data_driven_init_fn: Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = None,
    include_meta: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    p, q = int(X_train_s.shape[1]), int(Y_train_s.shape[1])
    starting_points = build_starting_points(
        X_train_s,
        Y_train_s,
        r=int(r),
        n_starts=int(n_starts),
        seed=int(seed),
        use_data_driven_init=bool(use_data_driven_init),
        data_driven_init_fn=data_driven_init_fn,
    )

    slm = ScalarLikelihoodMethod(
        p=p,
        q=q,
        r=int(r),
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
    return _extract_params(res, include_meta=include_meta)


def fit_ppls_em(
    X_train_s: np.ndarray,
    Y_train_s: np.ndarray,
    *,
    r: int,
    n_starts: int,
    seed: int,
    max_iter: int,
    tol: float,
    use_data_driven_init: bool = True,
    data_driven_init_fn: Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = None,
    include_meta: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    p, q = int(X_train_s.shape[1]), int(Y_train_s.shape[1])
    starting_points = build_starting_points(
        X_train_s,
        Y_train_s,
        r=int(r),
        n_starts=int(n_starts),
        seed=int(seed),
        use_data_driven_init=bool(use_data_driven_init),
        data_driven_init_fn=data_driven_init_fn,
    )

    em = EMAlgorithm(p=p, q=q, r=int(r), max_iter=int(max_iter), tolerance=float(tol))
    res = em.fit(X_train_s, Y_train_s, starting_points)
    return _extract_params(res, include_meta=include_meta)


def aggregate_prediction_by_r(df: pd.DataFrame, *, ddof: int) -> pd.DataFrame:
    out = []
    for (method, r), sub in df.groupby(["method", "r"], sort=False):
        out.append(
            {
                "method": method,
                "r": r,
                "mse_mean": float(sub["mse"].mean()),
                "mse_std": float(sub["mse"].std(ddof=int(ddof))),
                "mae_mean": float(sub["mae"].mean()),
                "mae_std": float(sub["mae"].std(ddof=int(ddof))),
                "r2_mean": float(sub["r2"].mean()),
                "r2_std": float(sub["r2"].std(ddof=int(ddof))),
            }
        )
    return pd.DataFrame(out)


def select_best_r(
    df_by_r: pd.DataFrame,
    *,
    ridge_method_names: Sequence[str] = ("Ridge",),
    ignore_dash_r: bool = True,
) -> pd.DataFrame:
    rows = []
    ridge_set = set(str(x) for x in ridge_method_names)
    for method, sub in df_by_r.groupby("method", sort=False):
        if str(method) in ridge_set:
            best = sub.iloc[0]
            rows.append(best)
            continue

        sub2 = sub.copy()
        if bool(ignore_dash_r):
            sub2 = sub2[sub2["r"].astype(str) != "-"]

        if sub2.empty:
            best = sub.sort_values(["mse_mean"], ascending=[True]).iloc[0]
            rows.append(best)
            continue

        try:
            sub2["r_int"] = sub2["r"].astype(int)
            best = sub2.sort_values(["mse_mean", "r_int"], ascending=[True, True]).iloc[0]
        except Exception:
            best = sub2.sort_values(["mse_mean"], ascending=[True]).iloc[0]

        rows.append(best.drop(labels=[c for c in ("r_int",) if c in best.index]))
    return pd.DataFrame(rows)
