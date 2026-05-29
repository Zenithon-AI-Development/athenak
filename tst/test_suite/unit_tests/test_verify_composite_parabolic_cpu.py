"""
Composite parabolic operator verification ([B1]/#114, ADR-0009).

Builds src/pgen/unit_tests/composite_parabolic_test.cpp
(-D PROBLEM=unit_tests/composite_parabolic_test) into the isolated tst/build_unit
directory, runs it on FOUR MeshBlocks (one rank), and asserts it exited 0 (all checks
passed). The pgen wraps TWO isotropic ConductionOperators with different conductivities
over the SAME conserved energy field in a parabolic::CompositeParabolicOperator and
checks, on a 1D static uniform-density medium carrying the global discrete cosine
diffusion eigenmode:

  - the composite action M(u) equals the SUM of the two analytic eigenvalues
    (lambda1+lambda2)*(E-E0) -- accumulate, not overwrite (operators touching the same
    conserved component sum); the result is materially DISTINCT from a single operator's
    action, and 0 is written into the non-energy components (frozen background);
  - ExplicitStableDt() returns the MINIMUM across sub-operators (the stiffer, larger-kappa
    operator sets the dt), so the RKL2 stage count comes from the single stiffest term;
  - a single RKL2 super-step over the composite reproduces the analytic
    exp((lambda1+lambda2) t) decay across block boundaries and conserves total energy;
  - the empty composite is inert (M=0, no parabolic dt limit);
  - the global min-dt MPI all-reduce (exercised cross-rank by the _mpicpu companion)
    returns the same global minimum on this rank.

Oracle: Layer 1 -- analytic.  The discrete cosine mode is an exact eigenvector of
isotropic conduction; the SUM of two such operators has eigenvalue lambda1+lambda2 and
semi-discrete solution E_i(t)=E0+A cos(..) exp((l1+l2)t).  References: closed form
(diffusion eigenmode); Meyer, Balsara & Aslam 2014 (RKL2); ADR-0009 (one super-step over
the summed operator, dt = min over operators).

Auto-collected by run_test_suite.py under --cpu (module name contains ``_cpu``).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the composite parabolic operator unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("composite_parabolic_test")
    assert passed, "composite_parabolic_test reported a failing check (nonzero exit)"
