"""BRCA TCGA model selection experiment for latent dimension r (BIC + CV).

This script uses the same BRCA paired multi-omics dataset as the paper's association
analysis section (gene expression X and protein expression Y). We report:
- BIC(r) and log-likelihood on the full standardized dataset
- 5-fold CV prediction MSE(r)

Outputs are written under an output directory (default: results_model_selection/brca),
so they can be synced into `paper/artifacts/` via `scripts/sync_artifacts.py`.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ppls_slm.algorithms import InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.model_selection import compute_bic, select_r_by_cv_prediction_mse
from ppls_slm.apps.data_utils import load_brca_combined_raw



def _zscore_global(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _fit_slm_fixed_noise(
    X: np.ndarray,
    Y: np.ndarray,
    r: int,
    *,
    n_starts: int,
    max_iter: int,
    seed: int,
    verbose: bool,
    slm_progress_every: int,
) -> Dict:
    p, q = X.shape[1], Y.shape[1]
    gen = InitialPointGenerator(p, q, int(r), n_starts=int(n_starts), random_seed=int(seed))
    starts = gen.generate_starting_points()

    slm = ScalarLikelihoodMethod(
        p,
        q,
        int(r),
        max_iter=int(max_iter),
        use_noise_preestimation=True,
        optimize_noise_variances=False,
        verbose=bool(verbose),
        progress_every=int(slm_progress_every),
    )
    return slm.fit(X, Y, starts)



def run_brca_model_selection(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    r_candidates: List[int],
    slm_n_starts: int,
    slm_max_iter: int,
    seed: int,
    n_folds: int,
    verbose: bool,
    slm_progress_every: int,
) -> Tuple[pd.DataFrame, int, int]:
    # Global standardization for the full-data BIC computation.
    Xs = _zscore_global(X)
    Ys = _zscore_global(Y)

    if verbose:
        print(f"[ModelSelection/BRCA] running CV ({int(n_folds)}-fold) over r={r_candidates} ...", flush=True)

    # CV prediction MSE (fold-wise standardization inside helper)
    cv_res = select_r_by_cv_prediction_mse(
        X,
        Y,
        r_candidates=[int(r) for r in r_candidates],
        fit_fn=lambda Xt, Yt, rr: _fit_slm_fixed_noise(
            Xt,
            Yt,
            rr,
            n_starts=int(slm_n_starts),
            max_iter=int(slm_max_iter),
            seed=int(seed) + 10_000 + int(rr),
            verbose=verbose,
            slm_progress_every=slm_progress_every,
        ),
        n_folds=int(n_folds),
        random_state=int(seed),
    )

    rows = []

    for r in r_candidates:
        # BIC / log-likelihood on full data
        if verbose:
            print(f"  [BIC] fit r={int(r)} (starts={int(slm_n_starts)}, max_iter={int(slm_max_iter)})", flush=True)

        try:
            params_hat = _fit_slm_fixed_noise(
                Xs,
                Ys,
                int(r),
                n_starts=int(slm_n_starts),
                max_iter=int(slm_max_iter),
                seed=int(seed) + int(r),
                verbose=verbose,
                slm_progress_every=slm_progress_every,
            )
            bic, ll = compute_bic(Xs, Ys, params_hat, r=int(r), assume_centered=True)
        except Exception:
            bic, ll = float("inf"), float("-inf")

        rows.append(
            {
                "r": int(r),
                "bic": float(bic),
                "cv_mse_mean": float(cv_res.cv_mse[int(r)]),
                "cv_mse_std": float(cv_res.cv_mse_std[int(r)]),
                "log_likelihood": float(ll),
                "n_folds": int(n_folds),
            }
        )


    df = pd.DataFrame(rows).sort_values("r")


    best_r_bic = int(df.sort_values(["bic", "r"], ascending=[True, True]).iloc[0]["r"])
    best_r_cv = int(df.sort_values(["cv_mse_mean", "r"], ascending=[True, True]).iloc[0]["r"])

    df["is_best_bic"] = df["r"].astype(int) == int(best_r_bic)
    df["is_best_cv"] = df["r"].astype(int) == int(best_r_cv)

    return df, best_r_bic, best_r_cv


def _plot_brca(df: pd.DataFrame, *, out_path: str):
    import matplotlib.pyplot as plt

    rs = df["r"].astype(int).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # BIC
    ax = axes[0]
    ax.plot(rs, df["bic"].to_numpy(), color="#6A1B9A", linewidth=2.2, marker="o")
    rb = int(df[df["is_best_bic"]].iloc[0]["r"])
    ax.axvline(rb, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("BRCA: BIC(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("BIC")
    ax.grid(True, alpha=0.25)

    # CV-MSE
    ax = axes[1]
    y = df["cv_mse_mean"].to_numpy()
    yerr = df["cv_mse_std"].to_numpy()
    ax.errorbar(rs, y, yerr=yerr, color="#1565C0", linewidth=2.0, marker="o", capsize=4)
    rc = int(df[df["is_best_cv"]].iloc[0]["r"])
    ax.axvline(rc, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("BRCA: 5-fold CV-MSE(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("CV-MSE")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="BRCA model selection (BIC + 5-fold CV)")
    p.add_argument("--brca_data", type=str, default="application/brca_data_w_subtypes.csv.zip")
    p.add_argument("--output_dir", type=str, default="results_model_selection/brca")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--r_candidates", type=str, default="2,3,5,8,10,15,20")
    p.add_argument("--slm_n_starts", type=int, default=8)
    p.add_argument("--slm_max_iter", type=int, default=50)
    p.add_argument("--slm_progress_every", type=int, default=1)
    p.add_argument("--verbose", action="store_true", help="Print progress (recommended)")
    return p.parse_args()



def main():
    args = parse_args()

    out_dir = str(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")

    r_candidates = [int(x.strip()) for x in str(args.r_candidates).split(",") if x.strip()]

    X, Y = load_brca_combined_raw(str(args.brca_data))
    print(f"Loaded BRCA: X={X.shape}, Y={Y.shape}")

    df, best_bic, best_cv = run_brca_model_selection(
        X,
        Y,
        r_candidates=r_candidates,
        slm_n_starts=int(args.slm_n_starts),
        slm_max_iter=int(args.slm_max_iter),
        seed=int(args.seed),
        n_folds=int(args.n_folds),
        verbose=bool(args.verbose),
        slm_progress_every=int(args.slm_progress_every),
    )


    df.to_csv(os.path.join(out_dir, "brca_r_selection.csv"), index=False)

    _plot_brca(df, out_path=os.path.join(fig_dir, "figure_brca_bic_cv.png"))

    print("Saved:")
    print(f"  {out_dir}/brca_r_selection.csv")
    print(f"  {fig_dir}/figure_brca_bic_cv.png")
    print(f"Best r (BIC) = {best_bic}, Best r (CV) = {best_cv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
