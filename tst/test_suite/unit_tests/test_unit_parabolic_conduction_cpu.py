"""
Auto-collected wrapper for the operator-split conduction pgen-based unit test.

Builds src/pgen/unit_tests/parabolic_conduction_test.cpp
(-D PROBLEM=unit_tests/parabolic_conduction_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed). The test re-expresses
AthenaK's existing isotropic thermal conduction as a parabolic::ParabolicOperator
(ConductionOperator) and advances it operator-split by the RKL2 super-time-stepping
integrator (issue [4b]/#13, ADR-0001). It verifies, on a 1D static uniform-density medium:
the operator action M(u) equals lambda*(E-E0) for the analytic cosine diffusion eigenmode
(pinning both the conduction flux divergence and the coefficient D = kappa(gamma-1)); the
STS-advanced result matches the analytic exp(lambda t) decay AND the explicit
forward-Euler sub-cycled conduction result (the acceptance comparison); 2nd-order temporal
convergence
under superstep refinement; exact total-energy conservation by the insulated operator; and
stability/decay at dt = 256 dt_exp where forward Euler would diverge. It drives the same
parabolic::OperatorSplitStep tasklist-task body the production Hydro task uses.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the operator-split conduction unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("parabolic_conduction_test")
    assert passed, "parabolic_conduction_test reported a failing check (nonzero exit)"
