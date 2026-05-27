"""
Auto-collected wrapper for the (converted) Gauss-Legendre spherical-harmonic unit test.

Builds src/pgen/unit_tests/gauss_legendre_test.cpp into the isolated tst/build_unit
directory, runs it, and asserts it exited 0. The pgen checks that cross integrals of
spin-weighted spherical harmonics on a GaussLegendreGrid are delta functions, reporting
pass/fail via the shared unit_test.hpp helpers.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the gauss_legendre unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("gauss_legendre_test")
    assert passed, "gauss_legendre_test reported a failing check (nonzero exit)"
