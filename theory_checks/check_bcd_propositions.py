import numpy as np

from common import check
from ppls_slm.bcd_slm import reduced_objective_from_Q, update_b_prop5, update_theta_t2_prop4


def ell_i(Qx: float, Qy: float, Qxy: float, s: float, b: float, se2: float, sf2: float, sh2: float) -> float:
    return float(
        reduced_objective_from_Q(
            Qx=np.array([Qx]),
            Qy=np.array([Qy]),
            Qxy=np.array([Qxy]),
            theta_t2=np.array([s]),
            b=np.array([b]),
            sigma_e2=se2,
            sigma_f2=sf2,
            sigma_h2=sh2,
        )
    )


def run(trials: int = 300, seed: int = 3) -> dict:
    rng = np.random.RandomState(seed)
    max_theta_grad = 0.0
    max_cubic_res = 0.0
    positive_root_uniqueness_ok = 0

    for _ in range(trials):
        Qx = float(rng.uniform(0.1, 2.0))
        Qy = float(rng.uniform(0.1, 2.0))
        Qxy = float(rng.uniform(0.05, 2.0))

        se2 = float(rng.uniform(0.05, 0.8))
        sf2 = float(rng.uniform(0.05, 0.8))
        sh2 = float(rng.uniform(0.01, 0.5))
        b = float(rng.uniform(0.2, 2.0))

        s_star = update_theta_t2_prop4(Qx=Qx, Qy=Qy, Qxy=Qxy, b=b, sigma_e2=se2, sigma_f2=sf2, sigma_h2=sh2)
        if s_star > 1e-4:
            eps = 1e-6
            grad_fd = (ell_i(Qx, Qy, Qxy, s_star + eps, b, se2, sf2, sh2) - ell_i(Qx, Qy, Qxy, s_star - eps, b, se2, sf2, sh2)) / (2 * eps)
            max_theta_grad = max(max_theta_grad, abs(grad_fd))
            check(abs(grad_fd) < 2e-3, f"Prop 4 finite-difference stationarity failed: {grad_fd:.3e}")


        s_for_b = float(rng.uniform(0.1, 2.0))
        Qxy_pos = float(rng.uniform(0.1, 2.0))
        b_star = update_b_prop5(
            Qx=Qx,
            Qy=Qy,
            Qxy=Qxy_pos,
            theta_t2=s_for_b,
            b_prev=b,
            sigma_e2=se2,
            sigma_f2=sf2,
            sigma_h2=sh2,
            fallback="keep",
        )

        a = se2
        beta = sf2
        gamma = sh2
        R = (beta + gamma) * ((s_for_b + a) * (1.0 - Qy) + s_for_b * Qx) + gamma * (s_for_b + a) * Qy
        c3 = 2.0 * (a**2) * s_for_b
        c2 = a * s_for_b * Qxy_pos
        c1 = 2.0 * a * R
        c0 = -Qxy_pos * (beta + gamma) * (s_for_b + a)

        cubic_res = ((c3 * b_star + c2) * b_star + c1) * b_star + c0
        max_cubic_res = max(max_cubic_res, abs(cubic_res))
        check(b_star > 0, "Prop 5 root is not positive")
        check(abs(cubic_res) < 1e-6, f"Prop 5 cubic residual too large: {cubic_res:.3e}")

        roots = np.roots(np.array([c3, c2, c1, c0], dtype=float))
        n_pos_real = sum((abs(z.imag) < 1e-8) and (z.real > 0) for z in roots)
        check(n_pos_real == 1, f"Descartes uniqueness violated, positive real roots={n_pos_real}")
        positive_root_uniqueness_ok += 1

    return {
        "name": "Propositions 4-5 (closed-form theta update and cubic b update)",
        "trials": trials,
        "max_theta_grad_fd": max_theta_grad,
        "max_cubic_residual": max_cubic_res,
        "positive_root_uniqueness_cases": positive_root_uniqueness_ok,
        "status": "PASS",
    }


if __name__ == "__main__":
    result = run()
    print(result)
