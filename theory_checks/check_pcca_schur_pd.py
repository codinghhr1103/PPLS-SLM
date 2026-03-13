import numpy as np

from common import check, random_stiefel, rel_error


def run(trials: int = 200, seed: int = 1) -> dict:
    rng = np.random.RandomState(seed)
    max_rel = 0.0
    min_eig_schur = np.inf
    min_eig_joint = np.inf

    for _ in range(trials):
        r = int(rng.randint(1, 5))
        p = int(rng.randint(r + 1, r + 8))
        q = int(rng.randint(r + 1, r + 8))

        W = random_stiefel(p, r, rng)
        C = random_stiefel(q, r, rng)
        theta = rng.uniform(0.05, 2.0, size=r)
        Sigma_t = np.diag(theta)
        se2 = float(rng.uniform(0.01, 1.5))
        sf2 = float(rng.uniform(0.01, 1.5))

        Sigma_xx = W @ Sigma_t @ W.T + se2 * np.eye(p)
        Sigma_xy = W @ Sigma_t @ C.T  # PCCA: B=I
        Sigma_yy = C @ Sigma_t @ C.T + sf2 * np.eye(q)

        schur = Sigma_yy - Sigma_xy.T @ np.linalg.inv(Sigma_xx) @ Sigma_xy
        closed = C @ np.diag(theta * se2 / (theta + se2)) @ C.T + sf2 * np.eye(q)

        rel = rel_error(schur, closed)
        max_rel = max(max_rel, rel)
        min_eig_schur = min(min_eig_schur, float(np.min(np.linalg.eigvalsh(schur))))

        joint = np.block([[Sigma_xx, Sigma_xy], [Sigma_xy.T, Sigma_yy]])
        min_eig_joint = min(min_eig_joint, float(np.min(np.linalg.eigvalsh(joint))))

        check(rel < 1e-10, f"Schur closed form mismatch: rel={rel:.3e}")
        check(np.all(np.linalg.eigvalsh(schur) > 1e-10), "Schur complement is not SPD")
        check(np.all(np.linalg.eigvalsh(joint) > 1e-10), "Joint covariance is not SPD")

    return {
        "name": "PCCA Schur complement and positive definiteness",
        "trials": trials,
        "max_rel": max_rel,
        "min_eig_schur": min_eig_schur,
        "min_eig_joint": min_eig_joint,
        "status": "PASS",
    }


if __name__ == "__main__":
    result = run()
    print(result)
