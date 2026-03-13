import numpy as np

from common import check, random_stiefel, rel_error


def run(trials: int = 200, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    max_det_rel = 0.0
    max_inv_rel = 0.0

    for _ in range(trials):
        n = int(rng.randint(1, 6))
        m = int(rng.randint(n + 1, n + 10))
        A = random_stiefel(m, n, rng)
        sigma = rng.uniform(0.05, 3.0, size=n)
        D = np.diag(sigma)
        k = float(rng.uniform(0.05, 2.5))

        M = A @ D @ A.T + k * np.eye(m)

        det_lhs = np.linalg.det(M)
        det_rhs = (k ** (m - n)) * np.prod(k + sigma)
        det_rel = abs(det_lhs - det_rhs) / max(1.0, abs(det_lhs), abs(det_rhs))

        inv_lhs = np.linalg.inv(M)
        inv_rhs = (1.0 / k) * (np.eye(m) - A @ np.linalg.inv(np.eye(n) + k * np.linalg.inv(D)) @ A.T)
        inv_rel = rel_error(inv_lhs, inv_rhs)

        max_det_rel = max(max_det_rel, det_rel)
        max_inv_rel = max(max_inv_rel, inv_rel)

        check(det_rel < 1e-8, f"Determinant identity failed: rel={det_rel:.3e}")
        check(inv_rel < 1e-10, f"Inverse identity failed: rel={inv_rel:.3e}")

    return {
        "name": "Lemma rank-n determinant/inverse identity",
        "trials": trials,
        "max_det_rel": max_det_rel,
        "max_inv_rel": max_inv_rel,
        "status": "PASS",
    }


if __name__ == "__main__":
    result = run()
    print(result)
