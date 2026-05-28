"""
Static cylindrical equilibrium test (ADR-0004, issue #14): a uniform medium (rho, p
constant, v = 0) on a 1D radial grid r in [0,1] is an exact static equilibrium in
cylindrical coordinates.  The radial flux divergence carries a spurious outward
"geometric pressure" (the r-face areas grow with r); the radial geometric source term
S_1 = <1/r>(rho v_phi^2 + p) cancels it exactly to round-off.  A correct implementation
therefore shows NO spurious radial drift after many cycles.

This run also exercises the `axis` (r=0 axisymmetric) boundary condition at inner-x1.

Auto-collected by run_test_suite.py (no runner edits) because the module name contains
``_cpu``.  See inputs/cylindrical/cyl_equilibrium.athinput.
"""

# Modules
import numpy as np
import test_suite.testutils as testutils
import athena_read

input_file = "inputs/cyl_equilibrium.athinput"


def test_cylindrical_equilibrium():
    """Run the uniform cylindrical state and assert no spurious radial drift."""
    try:
        results = testutils.run(input_file)
        assert results, "Cylindrical equilibrium run failed."
        d0 = athena_read.tab("tab/cyl_equilibrium.hydro_w.00000.tab")
        d1 = athena_read.tab("tab/cyl_equilibrium.hydro_w.00001.tab")

        # No spurious radial (v_r) or azimuthal (v_phi) velocity should develop.
        max_vr = np.abs(d1["velx"]).max()
        max_vphi = np.abs(d1["vely"]).max()
        assert max_vr < 1.0e-10, f"spurious radial drift: max|v_r| = {max_vr:g}"
        assert max_vphi < 1.0e-10, f"spurious azimuthal drift: max|v_phi| = {max_vphi:g}"

        # Density and internal energy must be held to round-off (the medium is static).
        max_ddens = np.abs(d1["dens"] - d0["dens"]).max()
        max_deint = np.abs(d1["eint"] - d0["eint"]).max()
        assert max_ddens < 1.0e-12, f"density drifted: max|d(dens)| = {max_ddens:g}"
        assert max_deint < 1.0e-12, f"energy drifted: max|d(eint)| = {max_deint:g}"
    finally:
        testutils.cleanup()
