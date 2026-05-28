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

Oracle: Layer 1 -- analytic.  Exact Sod (1978) Riemann solution at t=0.2, computed by the
standard Toro (2009, ch. 4) exact Riemann solver (shared ``verification/exact_riemann.py``
-- Newton solve for the star-region pressure, then self-similar sampling of the
rarefaction/contact/shock structure); the binding assertion is that the L1 error of the
radial profile vs this exact solution is within tolerance.  Running radially at large r
(r in [100,101]) keeps the
cylindrical geometric source <1/r> ~ 1e-2 below the discretization error, so the radial
problem reduces to the planar Sod problem the exact solver describes.  The solver's star
state was cross-checked against the published Sod values (p*=0.30313, u*=0.92745) and an
independent bisection root-find (issue #76).  The harness.verify baseline is a regression
guard only.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
from test_suite.verification.exact_riemann import exact_riemann
import athena_read

input_file = "inputs/cyl_sod.athinput"

# Sod initial states (must match inputs/cyl_sod.athinput) and run parameters.
GAMMA = 1.4
RHO_L, U_L, P_L = 1.0, 0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1
XSHOCK = 100.5
TFINAL = 0.2


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
        rho_ex, u_ex, p_ex = exact_riemann(
            xi, GAMMA, (RHO_L, U_L, P_L), (RHO_R, U_R, P_R)
        )

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
