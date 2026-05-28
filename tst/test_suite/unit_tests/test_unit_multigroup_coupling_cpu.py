"""
Auto-collected wrapper for the point-implicit multigroup matter-radiation coupling test.

Builds src/pgen/unit_tests/multigroup_coupling_test.cpp
(-D PROBLEM=unit_tests/multigroup_coupling_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed). The test exercises the
per-cell backward-Euler MULTIGROUP matter-radiation coupling solved as an arrowhead system
(O(N_group) Schur elimination over the photon-energy groups + a safeguarded Newton in the
electron temperature) plus the electron-ion exchange (radiationfld::
MultigroupArrowheadCoupling, issue [17b]/#35, ADR-0001/0002) on real device kernels.

It verifies: the fractional Planck function F(x) (endpoints, monotonicity, series
continuity, and agreement with an independent host quadrature oracle); relaxation of the
radiation groups to the Planck spectrum E_g = a T_e^4 [F(x_{g+1}) - F(x_g)]; exact
conservation of sum_g E_g + e_ele + e_ion every step; reduction to the grey coupling
(#23) for a single group spanning the spectrum; electron-ion exchange relaxing T_e and T_i
to the common temperature; and unconditional stability for stiff absorption.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the multigroup coupling unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("multigroup_coupling_test")
    assert passed, "multigroup_coupling_test reported a failing check (nonzero)"
