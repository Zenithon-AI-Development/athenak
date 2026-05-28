"""
Auto-collected wrapper for the point-implicit grey matter-radiation coupling unit test.

Builds src/pgen/unit_tests/matter_radiation_coupling_test.cpp
(-D PROBLEM=unit_tests/matter_radiation_coupling_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed). The test exercises the
per-cell point-implicit (backward-Euler) grey matter-radiation emission/absorption
coupling (radiationfld::PointImplicitGreyCoupling, issue [8b]/#23, ADR-0001/0002) on real
device kernels. It verifies: single-cell relaxation to radiative equilibrium
(a T^4 = E_r); exact conservation of E_r + e_mat; no spatial dependence (identical cells);
correct exchange direction (hot matter cools / radiation heats and vice versa); and
unconditional stability for stiff absorption (c chi_a dt >> 1 reaches equilibrium in one
step without overshoot).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the matter-radiation coupling unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("matter_radiation_coupling_test")
    assert passed, "matter_radiation_coupling_test reported a failing check (nonzero)"
