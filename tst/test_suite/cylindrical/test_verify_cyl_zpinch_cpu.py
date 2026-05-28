"""
Static cylindrical Z-pinch magnetohydrostatic equilibrium verification (ADR-0004, #22).

This is the "magnoh / Z-pinch equilibrium" verification slice.  A purely azimuthal field
B_phi(r) = b0 (r/a) exp(-(r/a)^2/2) is confined by a gas-pressure gradient in radial force
balance (a Gaussian-current Z-pinch); see inputs/cylindrical/cyl_zpinch.athinput.  This is
the discriminating test of the B_phi^2/r magnetic HOOP STRESS source term (issue #16): a
correct cylindrical MHD update holds the column STATIC -- only a small, bounded,
truncation-level radial drift develops -- whereas a wrong-sign or missing hoop stress
makes the pinch implode/explode at O(1).  The test:

  * runs the 1-D radial Z-pinch (cyl_zpinch is a user pgen -> run_pgen),
  * asserts the developed radial/azimuthal velocity stays small (equilibrium maintained),
  * asserts the density / B_phi / pressure profiles barely move from their initial state,
  * plots B_phi, pressure, density vs r and saves/diffs a golden regression baseline.

Oracle: Layer 1 -- analytic.  The Gaussian-current Z-pinch field
B_phi(r) = b0 (r/a) exp(-(r/a)^2/2) has the closed-form magnetohydrostatic pressure
p(r) = p0 - B_phi^2/2 + (b0^2/2)(e^{-(r/a)^2} - e^{-(rmax/a)^2}), which satisfies the
radial force balance dp/dr = -(B_phi/r) d(r B_phi)/dr exactly (mu0=1; round-off residual),
with peak B_phi = b0 exp(-1/2) = 0.6065 at r = a.  Reference: standard screw-/Z-pinch
magnetohydrostatic equilibrium (Freidberg, Ideal Magnetohydrodynamics 2014).  The
harness.verify baseline is a regression guard only.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os
import shutil
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

GAMMA = 1.666666666666667

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "cyl_zpinch.athinput"
)
tab_dir = os.path.join(testutils.pgen_run_dir("cyl_zpinch"), "tab")


def test_verify_cyl_zpinch():
    """Run the cylindrical Z-pinch equilibrium and verify it stays static + baseline."""
    try:
        assert testutils.run_pgen("cyl_zpinch", input_file), "cyl_zpinch run failed."
        d0 = athena_read.tab(os.path.join(tab_dir, "cyl_zpinch.mhd_w_bcc.00000.tab"))
        d1 = athena_read.tab(os.path.join(tab_dir, "cyl_zpinch.mhd_w_bcc.00001.tab"))

        x1v = np.asarray(d1["x1v"])
        bphi = np.asarray(d1["bcc2"])                       # azimuthal field B_phi
        pres = (GAMMA - 1.0) * np.asarray(d1["eint"])       # w(IEN) = p/(gamma-1)
        dens = np.asarray(d1["dens"])

        # (1) No appreciable flow develops: the hoop stress balances the pressure
        # gradient, so the column stays static to ~truncation level.
        max_vr = np.abs(np.asarray(d1["velx"])).max()
        max_vphi = np.abs(np.asarray(d1["vely"])).max()
        assert max_vr < 5.0e-3, f"Z-pinch not static: spurious max|v_r| = {max_vr:g}"
        assert max_vphi < 5.0e-3, f"Z-pinch spun up: spurious max|v_phi| = {max_vphi:g}"

        # (2) The equilibrium profiles barely move from their initial state.
        d_bphi = np.abs(bphi - np.asarray(d0["bcc2"])).max()
        d_dens = np.abs(dens - np.asarray(d0["dens"])).max()
        d_pres = np.abs(pres - (GAMMA - 1.0) * np.asarray(d0["eint"])).max()
        assert d_bphi < 1.0e-3, f"B_phi drifted from equilibrium: max|d|={d_bphi:g}"
        assert d_dens < 1.0e-3, f"density drifted from equilibrium: max|d|={d_dens:g}"
        assert d_pres < 5.0e-3, f"pressure drifted from equilibrium: max|d|={d_pres:g}"

        # (3) The peak field matches the analytic profile b0*(r/a)*exp(-(r/a)^2/2):
        # max at r=a with b0=a=1 -> exp(-1/2) = 0.6065.
        assert abs(bphi.max() - np.exp(-0.5)) < 0.02, (
            f"peak B_phi {bphi.max():g} != analytic {np.exp(-0.5):g}"
        )

        harness.verify(
            "cyl_zpinch",
            x1v,
            {"B_phi": bphi, "pressure": pres, "density": dens},
            coord_label="x1v",
            xlabel="r",
            title="Cylindrical Z-pinch magnetohydrostatic equilibrium (t=0.5)",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(tab_dir, ignore_errors=True)
