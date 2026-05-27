"""
Auto-collected wrapper for the RKL2 super-time-stepping integrator pgen-based unit test.

Builds src/pgen/unit_tests/rkl2_sts_test.cpp
(-D PROBLEM=unit_tests/rkl2_sts_test) into the isolated tst/build_unit directory, runs it,
and asserts it exited 0 (all checks passed). The test verifies that the RKL2 integrator in
driver/parabolic_integrator.hpp (issue [4a]/#9, ADR-0001): reproduces the 2nd-order
amplification factor R_s(z) = 1 + z + z^2/2 + O(z^3) of the super-time-stepping recursion;
advances a 1D scalar diffusion eigenmode with measured 2nd-order temporal convergence vs
the analytic decay exp(lambda t); derives the minimal adaptive RKL2 stage count from the
superstep/explicit dt ratio; and remains stable (L2 non-increasing, mean conserved, field
decaying) at dt = 256 dt_exp, far beyond the explicit parabolic limit where forward Euler
would blow up. It also checks the <time> parabolic_integrator selector parses to the sts
backend.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the RKL2 super-time-stepping unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("rkl2_sts_test")
    assert passed, "rkl2_sts_test reported a failing check (nonzero exit)"
