"""
MHD cylindrical linear-wave convergence verification (ADR-0004, issue #22).

A fast magnetosonic wave propagating RADIALLY at large radius (r in [1e4, 1e4+1]) in
cylindrical coordinates, with a perpendicular axial background field B_z (an exact
cylindrical equilibrium, divergence-free for any B_z(r)).  See
inputs/cylindrical/cyl_lwave_mhd.athinput.  At large r the geometric source acts on the
1e-6 wave only at ~1e-10, far below the discretization error, so the radial solution
reduces to a planar linear wave and the L1 error must converge at the scheme's 2nd-order
rate.  The test:

  * runs the wave at two resolutions (64, 128) using the built-in linear_wave pgen,
  * reads the auto-emitted L1 error norms and asserts the convergence ratio matches the
    2nd-order reference (halving dx -> error x ~1/4),
  * plots the L1 error vs resolution and saves/diffs a golden regression baseline.

Oracle: Layer 1 -- analytic.  A linear MHD eigenmode is dispersionless and exact at
infinitesimal amplitude, so the L1 error of a finite-volume solver is pure truncation
error ~ C dx^2; halving dx must drop the L1 norm to ~1/4 (the scheme's 2nd-order
convergence rate).  Reference: standard finite-volume convergence analysis (LeVeque 2002;
Stone et al. 2008, ApJS 178, 137 -- the Athena linear-wave convergence test).  The
harness.verify baseline is a regression guard only.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

input_file = "inputs/cyl_lwave_mhd.athinput"
errs_file = "cyl_lwave_mhd-errs.dat"
RESOLUTIONS = [64, 128]
L1_RMS_INDEX = 4  # column index of the RMS-L1 error in the -errs.dat file


def test_verify_cyl_lwave():
    """Run the cylindrical MHD fast wave at two resolutions; verify 2nd-order rate."""
    try:
        if os.path.exists(errs_file):
            os.remove(errs_file)
        for n in RESOLUTIONS:
            assert testutils.run(
                input_file, [f"mesh/nx1={n}", f"meshblock/nx1={n}"]
            ), f"cylindrical MHD lwave run failed at nx1={n}"

        data = athena_read.error_dat(errs_file)
        l1 = np.array([data[i][L1_RMS_INDEX] for i in range(len(RESOLUTIONS))])

        # 2nd-order accuracy: error ~ dx^2, so doubling the resolution drops the L1 error
        # ~4x (ratio ~0.25).  0.35 gives headroom (cf Cartesian nr lwave plm threshold).
        ratio = float(l1[1] / l1[0])
        assert ratio < 0.35, (
            f"cylindrical MHD lwave not converging at 2nd order: "
            f"L1(64)={l1[0]:g} L1(128)={l1[1]:g} ratio={ratio:g}"
        )
        # the field stays div-free (B_r=B_phi=0): the B1/B2 error columns must be zero.
        assert data[0][8] == 0.0 and data[0][9] == 0.0, (
            "cyl MHD lwave generated spurious B_r/B_phi (B1_L1/B2_L1 != 0)"
        )

        harness.verify(
            "cyl_lwave_mhd",
            np.array(RESOLUTIONS, dtype=float),
            {"L1_RMS": l1},
            coord_label="nx1",
            xlabel="radial resolution nx1",
            title=f"MHD cylindrical fast-wave convergence (ratio={ratio:.3f})",
            rtol=2.0e-2,
            atol=1.0e-12,
        )
    finally:
        testutils.cleanup()
