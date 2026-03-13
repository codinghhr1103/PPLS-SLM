import numpy as np

from common import check, random_stiefel
from ppls_slm.ppls_model import PPLSModel, PPLSObjective


def run(trials: int = 120, seed: int = 2) -> dict:
    rng = np.random.RandomState(seed)
    max_abs = 0.0

    for _ in range(trials):
        r = int(rng.randint(1, 4))
        p = int(rng.randint(r + 1, r + 10))
        q = int(rng.randint(r + 1, r + 10))
        n = int(rng.randint(120, 260))

        W = random_stiefel(p, r, rng)
        C = random_stiefel(q, r, rng)
        b = rng.uniform(0.2, 2.0, size=r)
        theta_t2 = rng.uniform(0.1, 1.8, size=r)
        sh2 = float(rng.uniform(0.01, 1.0))
        se2 = float(rng.uniform(0.02, 0.8))
        sf2 = float(rng.uniform(0.02, 0.8))

        B = np.diag(b)
        Sigma_t = np.diag(theta_t2)

        model = PPLSModel(p, q, r)
        X, Y = model.sample(
            n_samples=n,
            W=W,
            C=C,
            B=B,
            Sigma_t=Sigma_t,
            sigma_e2=se2,
            sigma_f2=sf2,
            sigma_h2=sh2,
        )
        XY = np.hstack([X, Y])
        XY = XY - XY.mean(axis=0, keepdims=True)
        S = (XY.T @ XY) / float(n)

        Sigma = model.compute_covariance_matrix(W, C, B, Sigma_t, se2, sf2, sh2)
        matrix_obj = model.log_likelihood_matrix(S, Sigma)

        objective = PPLSObjective(p, q, r, S)
        objective.sigma_e2 = se2
        objective.sigma_f2 = sf2
        theta = objective._params_to_theta(W, C, B, Sigma_t, sh2)
        scalar_obj = objective.scalar_log_likelihood(theta)

        diff = abs(matrix_obj - scalar_obj)
        max_abs = max(max_abs, diff)
        check(diff < 1e-8, f"Scalar expansion mismatch: abs={diff:.3e}")

    return {
        "name": "Theorem A scalar expansion vs matrix-form objective",
        "trials": trials,
        "max_abs_diff": max_abs,
        "status": "PASS",
    }


if __name__ == "__main__":
    result = run()
    print(result)
