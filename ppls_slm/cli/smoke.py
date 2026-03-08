"""Minimal smoke test for the PPLS-SLM codebase.

Goal
----
Provide a fast, dependency-light sanity check that the core pipeline still runs:
- generate a tiny synthetic PPLS dataset
- fit one SLM model with a single start and very small iteration budget
- run the conditional-mean prediction helper

This is NOT a unit test and does not assert numerical quality.

Run:
  python -m ppls_slm.cli.smoke
"""

from __future__ import annotations

import argparse

import numpy as np

from ppls_slm.algorithms import InitialPointGenerator, ScalarLikelihoodMethod
from ppls_slm.apps.prediction import predict_conditional_mean
from ppls_slm.ppls_model import PPLSModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPLS-SLM smoke test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p", type=int, default=8)
    p.add_argument("--q", type=int, default=6)
    p.add_argument("--r", type=int, default=2)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--max_iter", type=int, default=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    rng = np.random.RandomState(int(args.seed))

    p, q, r, n = int(args.p), int(args.q), int(args.r), int(args.n)

    # Tiny, well-posed synthetic model.
    W, _ = np.linalg.qr(rng.randn(p, r))
    C, _ = np.linalg.qr(rng.randn(q, r))
    B = np.diag(np.linspace(1.2, 0.8, r))
    Sigma_t = np.diag(np.linspace(1.0, 0.4, r))

    model = PPLSModel(p, q, r)
    np.random.seed(int(args.seed) + 1)
    X, Y = model.sample(
        n_samples=n,
        W=W,
        C=C,
        B=B,
        Sigma_t=Sigma_t,
        sigma_e2=0.1,
        sigma_f2=0.1,
        sigma_h2=0.05,
    )

    init = InitialPointGenerator(p=p, q=q, r=r, n_starts=1, random_seed=int(args.seed))
    starts = init.generate_starting_points()

    slm = ScalarLikelihoodMethod(
        p=p,
        q=q,
        r=r,
        optimizer="trust-constr",
        max_iter=int(args.max_iter),
        use_noise_preestimation=True,
        verbose=False,
        progress_every=999999,
    )

    res = slm.fit(X, Y, starts)

    required = ("W", "C", "B", "Sigma_t", "sigma_e2", "sigma_f2", "sigma_h2")
    missing = [k for k in required if k not in res]
    if missing:
        raise RuntimeError(f"SLM fit result missing keys: {missing}")

    # Quick prediction sanity check.
    y_hat = predict_conditional_mean(X[:5], res)
    assert y_hat.shape == (5, q), f"Unexpected prediction shape: {y_hat.shape}"

    print("[OK] smoke passed")
    print(f"X={X.shape}, Y={Y.shape}, y_hat={y_hat.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
