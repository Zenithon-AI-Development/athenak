"""
Auto-collected wrapper for the <mhd> mrad_opacity_local wiring pgen-based unit test
(#204).

Builds src/pgen/unit_tests/mrad_opacity_local_wiring_test.cpp
(-D PROBLEM=unit_tests/mrad_opacity_local_wiring_test) into the isolated tst/build_unit
directory, runs it against the real committed aluminum IONMIX table, and asserts it
exited 0.  The test verifies that with `<mhd> mrad_opacity_local=true` the grey
matter-radiation coupling evaluates its Planck absorption coefficient PER CELL as
chi_a(rho,Te) = kappa_P(rho,Te)*rho_cgs*L (grey Planck mean of the multigroup table,
same Te closure as ConsToPrim2T) instead of the frozen solid-liner constant: one
production MatterRadCouplingHalf step lands exactly on the PointImplicitGreyCoupling
oracle evaluated with the local chain, and the tenuous vacuum cell's energy exchange is
suppressed >=100x relative to the liner cell (no more spurious equilibrium-locking of
near-massless gap material).

Oracle: Layer 1 -- the closed-form backward-Euler coupling kernel evaluated with the
independently-recomputed local chi_a chain, plus the liner/vacuum exchange ordering.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the mrad_opacity_local wiring unit test; pass iff it exits 0."""
    table = os.path.join(
        testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
    )
    passed = testutils.run_unit_test(
        "mrad_opacity_local_wiring_test",
        args=[f"mhd/eos_table={table}"],
    )
    assert passed, (
        "mrad_opacity_local_wiring_test reported a failing check (nonzero exit)"
    )
