import numpy as np

from ppls_slm.bcd_slm import (
    reduced_objective_from_Q,
    update_b_prop5,
    update_sigma_h2_bounded,
    update_theta_t2_prop4,
)


def _ell_i_from_Q(*, Qx: float, Qy: float, Qxy: float, s: float, b: float, se2: float, sf2: float, sh2: float) -> float:
    Qxv = np.array([float(Qx)])
    Qyv = np.array([float(Qy)])
    Qxyv = np.array([float(Qxy)])
    sv = np.array([float(s)])
    bv = np.array([float(b)])
    return float(
        reduced_objective_from_Q(
            Qx=Qxv,
            Qy=Qyv,
            Qxy=Qxyv,
            theta_t2=sv,
            b=bv,
            sigma_e2=float(se2),
            sigma_f2=float(sf2),
            sigma_h2=float(sh2),
        )
    )


def test_prop4_theta_update_stationary_by_finite_difference():
    rng = np.random.RandomState(0)

    # Use a typical regime (positive Qxy).
    Qx = float(rng.uniform(0.2, 2.0))
    Qy = float(rng.uniform(0.2, 2.0))
    Qxy = float(rng.uniform(0.1, 2.0))

    se2, sf2, sh2 = 0.1, 0.1, 0.05
    b = 1.3

    s_star = update_theta_t2_prop4(
        Qx=Qx,
        Qy=Qy,
        Qxy=Qxy,
        b=b,
        sigma_e2=se2,
        sigma_f2=sf2,
        sigma_h2=sh2,
    )

    eps = 1e-6

    def f(s: float) -> float:
        return _ell_i_from_Q(Qx=Qx, Qy=Qy, Qxy=Qxy, s=s, b=b, se2=se2, sf2=sf2, sh2=sh2)

    # Central difference derivative near optimum should be ~0.
    d = (f(s_star + eps) - f(s_star - eps)) / (2 * eps)
    assert abs(d) < 1e-3


def test_prop5_b_update_solves_cubic_when_Qxy_positive():
    rng = np.random.RandomState(1)

    Qx = float(rng.uniform(0.1, 2.0))
    Qy = float(rng.uniform(0.1, 2.0))
    Qxy = float(rng.uniform(0.2, 2.0))  # enforce > 0

    se2, sf2, sh2 = 0.1, 0.1, 0.05
    s = 0.8

    b_prev = 1.0
    b_star = update_b_prop5(
        Qx=Qx,
        Qy=Qy,
        Qxy=Qxy,
        theta_t2=s,
        b_prev=b_prev,
        sigma_e2=se2,
        sigma_f2=sf2,
        sigma_h2=sh2,
        fallback="keep",
    )

    # Verify root satisfies the cubic coefficients (Prop 5)
    a = se2
    beta = sf2
    gamma = sh2
    R = (beta + gamma) * ((s + a) * (1.0 - Qy) + s * Qx) + gamma * (s + a) * Qy

    c3 = 2.0 * (a**2) * s
    c2 = a * s * Qxy
    c1 = 2.0 * a * R
    c0 = -Qxy * (beta + gamma) * (s + a)

    poly = ((c3 * b_star + c2) * b_star + c1) * b_star + c0
    assert b_star > 0
    assert abs(poly) < 1e-6


def test_prop5_fallback_when_Qxy_nonpositive_keeps_previous():
    b_prev = 1.23
    b_star = update_b_prop5(
        Qx=1.0,
        Qy=1.0,
        Qxy=0.0,
        theta_t2=1.0,
        b_prev=b_prev,
        sigma_e2=0.1,
        sigma_f2=0.1,
        sigma_h2=0.05,
        fallback="keep",
    )
    assert abs(b_star - b_prev) < 1e-12


def test_sigma_h2_bounded_search_improves_objective():
    rng = np.random.RandomState(2)

    r = 3
    Qx = rng.uniform(0.2, 2.0, size=r)
    Qy = rng.uniform(0.2, 2.0, size=r)
    Qxy = rng.uniform(0.1, 2.0, size=r)

    theta = rng.uniform(0.2, 1.5, size=r)
    b = rng.uniform(0.5, 2.0, size=r)

    se2, sf2 = 0.1, 0.1
    lo, hi = 1e-8, 10.0

    sh2_star = update_sigma_h2_bounded(
        Qx=Qx,
        Qy=Qy,
        Qxy=Qxy,
        theta_t2=theta,
        b=b,
        sigma_e2=se2,
        sigma_f2=sf2,
        bounds=(lo, hi),
    )

    f_lo = reduced_objective_from_Q(
        Qx=Qx, Qy=Qy, Qxy=Qxy, theta_t2=theta, b=b, sigma_e2=se2, sigma_f2=sf2, sigma_h2=lo
    )
    f_hi = reduced_objective_from_Q(
        Qx=Qx, Qy=Qy, Qxy=Qxy, theta_t2=theta, b=b, sigma_e2=se2, sigma_f2=sf2, sigma_h2=hi
    )
    f_star = reduced_objective_from_Q(
        Qx=Qx, Qy=Qy, Qxy=Qxy, theta_t2=theta, b=b, sigma_e2=se2, sigma_f2=sf2, sigma_h2=sh2_star
    )

    assert lo <= sh2_star <= hi
    assert f_star <= min(f_lo, f_hi) + 1e-9
