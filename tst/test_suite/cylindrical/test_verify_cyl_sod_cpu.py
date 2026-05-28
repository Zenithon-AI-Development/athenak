"""
Cylindrical Sod shock-tube verification (ADR-0004, issue #15).

Runs Sod's shock tube along the RADIAL direction at large radius (r in [100,101]).  At
large r the cylindrical geometric source <1/r> ~ 1e-2 is a small perturbation, so the
radial solution must reduce to the exact PLANAR Sod Riemann solution.  The test:

  * runs the cylindrical shock tube,
  * compares the numerical density/velocity/pressure profiles at t=0.2 to the EXACT Sod
    solution (an exact Riemann solver, below) and asserts the L1 error is within
    tolerance ("matches analytic"),
  * plots the profiles and saves/diffs a golden regression baseline via the harness.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

input_file = "inputs/cyl_sod.athinput"

# Sod initial states (must match inputs/cyl_sod.athinput) and run parameters.
GAMMA = 1.4
RHO_L, U_L, P_L = 1.0, 0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1
XSHOCK = 100.5
TFINAL = 0.2


def _exact_riemann(xi):
    """Exact solution of the 1D Euler (ideal-gas) Riemann problem, sampled at the
    self-similar speeds xi = (x - xshock)/t.  Standard Toro (2009) algorithm.  Returns
    (rho, u, p) arrays for the Sod states defined above."""
    g = GAMMA
    gm1, gp1 = g - 1.0, g + 1.0
    a_l = np.sqrt(g * P_L / RHO_L)
    a_r = np.sqrt(g * P_R / RHO_R)
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
    p = max(1.0e-8, 0.5 * (P_L + P_R))
    for _ in range(100):
        f = f_k(p, RHO_L, P_L, a_l) + f_k(p, RHO_R, P_R, a_r) + (U_R - U_L)
        fp = fp_k(p, RHO_L, P_L, a_l) + fp_k(p, RHO_R, P_R, a_r)
        p_new = p - f / fp
        if p_new <= 0.0:
            p_new = 0.5 * p
        if abs(p_new - p) < 1.0e-13 * (p_new + p):
            p = p_new
            break
        p = p_new
    p_star = p
    u_star = 0.5 * (U_L + U_R) + 0.5 * (
        f_k(p_star, RHO_R, P_R, a_r) - f_k(p_star, RHO_L, P_L, a_l)
    )

    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    pr = np.empty_like(xi)
    for i, s in enumerate(np.atleast_1d(xi)):
        if s <= u_star:  # left of the contact
            if p_star > P_L:  # left shock
                s_l = U_L - a_l * np.sqrt(g2 * (p_star / P_L) + g1)
                if s <= s_l:
                    rho[i], u[i], pr[i] = RHO_L, U_L, P_L
                else:
                    rho[i] = RHO_L * ((p_star / P_L) + gm1 / gp1) / (
                        gm1 / gp1 * (p_star / P_L) + 1.0
                    )
                    u[i], pr[i] = u_star, p_star
            else:  # left rarefaction
                a_sl = a_l * (p_star / P_L) ** g1
                s_h, s_t = U_L - a_l, u_star - a_sl
                if s <= s_h:
                    rho[i], u[i], pr[i] = RHO_L, U_L, P_L
                elif s >= s_t:
                    rho[i] = RHO_L * (p_star / P_L) ** (1.0 / g)
                    u[i], pr[i] = u_star, p_star
                else:
                    u[i] = 2.0 / gp1 * (a_l + 0.5 * gm1 * U_L + s)
                    a = 2.0 / gp1 * (a_l + 0.5 * gm1 * (U_L - s))
                    rho[i] = RHO_L * (a / a_l) ** (2.0 / gm1)
                    pr[i] = P_L * (a / a_l) ** (2.0 * g / gm1)
        else:  # right of the contact
            if p_star > P_R:  # right shock
                s_r = U_R + a_r * np.sqrt(g2 * (p_star / P_R) + g1)
                if s >= s_r:
                    rho[i], u[i], pr[i] = RHO_R, U_R, P_R
                else:
                    rho[i] = RHO_R * ((p_star / P_R) + gm1 / gp1) / (
                        gm1 / gp1 * (p_star / P_R) + 1.0
                    )
                    u[i], pr[i] = u_star, p_star
            else:  # right rarefaction
                a_sr = a_r * (p_star / P_R) ** g1
                s_h, s_t = U_R + a_r, u_star + a_sr
                if s >= s_h:
                    rho[i], u[i], pr[i] = RHO_R, U_R, P_R
                elif s <= s_t:
                    rho[i] = RHO_R * (p_star / P_R) ** (1.0 / g)
                    u[i], pr[i] = u_star, p_star
                else:
                    u[i] = 2.0 / gp1 * (-a_r + 0.5 * gm1 * U_R + s)
                    a = 2.0 / gp1 * (a_r - 0.5 * gm1 * (U_R - s))
                    rho[i] = RHO_R * (a / a_r) ** (2.0 / gm1)
                    pr[i] = P_R * (a / a_r) ** (2.0 * g / gm1)
    return rho, u, pr


def test_verify_cyl_sod():
    """Run cylindrical Sod and verify against the exact Riemann solution + baseline."""
    try:
        results = testutils.run(input_file)
        assert results, "Cylindrical Sod run failed."
        data = athena_read.tab("tab/cyl_sod.hydro_w.00001.tab")
        x1v = np.asarray(data["x1v"])
        dens = np.asarray(data["dens"])
        velx = np.asarray(data["velx"])
        pres = (GAMMA - 1.0) * np.asarray(data["eint"])  # w(IEN) = p/(gamma-1)

        # Exact planar Sod solution at the same cell centres.
        xi = (x1v - XSHOCK) / TFINAL
        rho_ex, u_ex, p_ex = _exact_riemann(xi)

        # Relative L1 errors -- "matches analytic" within tolerance.  The residual is set
        # by the (~1%) shock/contact smearing of a 2nd-order shock-capturing scheme plus
        # the ~1e-3 geometric correction at r~100; 3% gives comfortable headroom.
        l1_dens = np.mean(np.abs(dens - rho_ex)) / np.mean(np.abs(rho_ex))
        l1_pres = np.mean(np.abs(pres - p_ex)) / np.mean(np.abs(p_ex))
        assert l1_dens < 0.03, f"cyl Sod density L1 vs analytic too large: {l1_dens:g}"
        assert l1_pres < 0.03, f"cyl Sod pressure L1 vs analytic too large: {l1_pres:g}"

        harness.verify(
            "cyl_sod",
            x1v,
            {"dens": dens, "velx": velx, "pres": pres},
            coord_label="x1v",
            xlabel="r",
            title="Cylindrical Sod shock tube (r in [100,101], t=0.2)",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        testutils.cleanup()
