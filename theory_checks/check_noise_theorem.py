import numpy as np

from common import check, random_stiefel
from ppls_slm.ppls_model import NoiseEstimator


def run(trials: int = 300, seed: int = 4, delta: float = 0.05) -> dict:
    rng = np.random.RandomState(seed)

    p = 40
    r = 5
    n = 1200
    sigma_e2_true = 0.4

    W = random_stiefel(p, r, rng)
    theta = np.linspace(2.0, 0.8, r)

    estimates = []
    bound_hits = 0
    for _ in range(trials):
        T = rng.randn(n, r) @ np.diag(np.sqrt(theta))
        E = np.sqrt(sigma_e2_true) * rng.randn(n, p)
        X = T @ W.T + E
        est = NoiseEstimator.estimate_noise_variances(X, X, r=r)[0]
        estimates.append(est)

        err = abs(est - sigma_e2_true)
        lead_bound = sigma_e2_true * np.sqrt(2.0 * np.log(4.0 / delta) / (n * (p - r)))
        if err <= lead_bound:
            bound_hits += 1

    estimates = np.asarray(estimates)
    mae = float(np.mean(np.abs(estimates - sigma_e2_true)))
    rhs_expect = float(2.0 * sigma_e2_true / np.sqrt(n * (p - r)))
    hit_rate = float(bound_hits / trials)

    check(mae <= rhs_expect * 1.1, f"Expectation bound not met: mae={mae:.4e}, rhs={rhs_expect:.4e}")
    check(hit_rate >= 0.90, f"High-probability leading-rate bound hit-rate too low: {hit_rate:.3f}")

    return {
        "name": "Theorem 4 spectral noise estimator empirical verification",
        "trials": trials,
        "mae": mae,
        "expectation_bound_rhs": rhs_expect,
        "bound_hit_rate": hit_rate,
        "status": "PASS",
    }


if __name__ == "__main__":
    result = run()
    print(result)
