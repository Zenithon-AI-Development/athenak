"""
Exact 1-D Euler (ideal-gas) Riemann solver -- Layer-1 analytic oracle.

Standard Toro (2009, ch. 4) algorithm: a Newton iteration for the star-region pressure,
then self-similar sampling of the rarefaction / contact / shock wave structure.  Shared by
the Sod shock-tube verification tests -- planar ``test_verify_sod_cpu.py`` and cylindrical
``test_verify_cyl_sod_cpu.py`` -- as the independent ground truth: the exact solution of
the Riemann initial-value problem, derived independently of AthenaK's own output.

Reference: Toro, E.F., "Riemann Solvers and Numerical Methods for Fluid Dynamics",
3rd ed., Springer (2009), ch. 4.  Classic Sod problem: Sod, G.A., JCP 27, pp. 1-31 (1978).
"""

import numpy as np


def exact_riemann(xi, gamma, wl, wr):
    """Exact solution of the 1-D Euler (ideal-gas) Riemann problem.

    Parameters
    ----------
    xi : array_like
        Self-similar sampling speeds ``(x - x_interface) / t``.
    gamma : float
        Ideal-gas adiabatic index.
    wl, wr : tuple(float, float, float)
        Left / right primitive states ``(rho, u, p)``.

    Returns
    -------
    rho, u, pr : numpy.ndarray
        Density, velocity and pressure sampled at each ``xi`` (same shape as ``xi``).
    """
    rho_l, u_l, p_l = wl
    rho_r, u_r, p_r = wr
    g = gamma
    gm1, gp1 = g - 1.0, g + 1.0
    a_l = np.sqrt(g * p_l / rho_l)
    a_r = np.sqrt(g * p_r / rho_r)
    g1 = gm1 / (2.0 * g)
    g2 = gp1 / (2.0 * g)

    def f_k(p, rho_k, p_k, a_k):
        if p > p_k:  # shock branch
            big_a = 2.0 / (gp1 * rho_k)
            big_b = gm1 / gp1 * p_k
            return (p - p_k) * np.sqrt(big_a / (p + big_b))
        return (2.0 * a_k / gm1) * ((p / p_k) ** g1 - 1.0)  # rarefaction branch

    def fp_k(p, rho_k, p_k, a_k):
        if p > p_k:
            big_a = 2.0 / (gp1 * rho_k)
            big_b = gm1 / gp1 * p_k
            return np.sqrt(big_a / (p + big_b)) * (1.0 - 0.5 * (p - p_k) / (p + big_b))
        return (1.0 / (rho_k * a_k)) * (p / p_k) ** (-g2)

    # Newton iteration for the star-region pressure.
    p = max(1.0e-8, 0.5 * (p_l + p_r))
    for _ in range(100):
        f = f_k(p, rho_l, p_l, a_l) + f_k(p, rho_r, p_r, a_r) + (u_r - u_l)
        fp = fp_k(p, rho_l, p_l, a_l) + fp_k(p, rho_r, p_r, a_r)
        p_new = p - f / fp
        if p_new <= 0.0:
            p_new = 0.5 * p
        if abs(p_new - p) < 1.0e-13 * (p_new + p):
            p = p_new
            break
        p = p_new
    p_star = p
    u_star = 0.5 * (u_l + u_r) + 0.5 * (
        f_k(p_star, rho_r, p_r, a_r) - f_k(p_star, rho_l, p_l, a_l)
    )

    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    pr = np.empty_like(xi)
    for i, s in enumerate(xi):
        if s <= u_star:  # left of the contact
            if p_star > p_l:  # left shock
                s_l = u_l - a_l * np.sqrt(g2 * (p_star / p_l) + g1)
                if s <= s_l:
                    rho[i], u[i], pr[i] = rho_l, u_l, p_l
                else:
                    rho[i] = rho_l * ((p_star / p_l) + gm1 / gp1) / (
                        gm1 / gp1 * (p_star / p_l) + 1.0
                    )
                    u[i], pr[i] = u_star, p_star
            else:  # left rarefaction
                a_sl = a_l * (p_star / p_l) ** g1
                s_h, s_t = u_l - a_l, u_star - a_sl
                if s <= s_h:
                    rho[i], u[i], pr[i] = rho_l, u_l, p_l
                elif s >= s_t:
                    rho[i] = rho_l * (p_star / p_l) ** (1.0 / g)
                    u[i], pr[i] = u_star, p_star
                else:
                    u[i] = 2.0 / gp1 * (a_l + 0.5 * gm1 * u_l + s)
                    a = 2.0 / gp1 * (a_l + 0.5 * gm1 * (u_l - s))
                    rho[i] = rho_l * (a / a_l) ** (2.0 / gm1)
                    pr[i] = p_l * (a / a_l) ** (2.0 * g / gm1)
        else:  # right of the contact
            if p_star > p_r:  # right shock
                s_r = u_r + a_r * np.sqrt(g2 * (p_star / p_r) + g1)
                if s >= s_r:
                    rho[i], u[i], pr[i] = rho_r, u_r, p_r
                else:
                    rho[i] = rho_r * ((p_star / p_r) + gm1 / gp1) / (
                        gm1 / gp1 * (p_star / p_r) + 1.0
                    )
                    u[i], pr[i] = u_star, p_star
            else:  # right rarefaction
                a_sr = a_r * (p_star / p_r) ** g1
                s_h, s_t = u_r + a_r, u_star + a_sr
                if s >= s_h:
                    rho[i], u[i], pr[i] = rho_r, u_r, p_r
                elif s <= s_t:
                    rho[i] = rho_r * (p_star / p_r) ** (1.0 / g)
                    u[i], pr[i] = u_star, p_star
                else:
                    u[i] = 2.0 / gp1 * (-a_r + 0.5 * gm1 * u_r + s)
                    a = 2.0 / gp1 * (a_r - 0.5 * gm1 * (u_r - s))
                    rho[i] = rho_r * (a / a_r) ** (2.0 / gm1)
                    pr[i] = p_r * (a / a_r) ** (2.0 * g / gm1)
    return rho, u, pr
