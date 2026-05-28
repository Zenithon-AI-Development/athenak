"""
Auto-collected wrapper for the grey FLD operator pgen-based unit test.

Builds src/pgen/unit_tests/fld_grey_operator_test.cpp
(-D PROBLEM=unit_tests/fld_grey_operator_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test exercises the grey
flux-limited radiation diffusion operator (FLDGreyOperator) with the Larsen flux limiter,
re-expressed as a parabolic::ParabolicOperator (issue [8a]/#17, ADR-0001), on real device
kernels. It verifies: the Larsen limiter limits (lambda(0)=1/3 diffusion limit, the
free-streaming flux cap lambda*R<=1 with lambda*R->1 as R->inf, monotonicity); the
operator
action M(E) reproduces the discrete diffusion eigenvalue lambda_eig*(E-E0) with
D = c/(3 chi) on an optically-thick cosine eigenmode; conservation by the insulated
operator; and a finite, positive explicit stable dt.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the grey FLD operator unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("fld_grey_operator_test")
    assert passed, "fld_grey_operator_test reported a failing check (nonzero exit)"
