"""
Multi-block operator-split conduction verification ([A1]/#108, ADR-0001/ADR-0009).

Builds src/pgen/unit_tests/conduction_multiblock_test.cpp
(-D PROBLEM=unit_tests/conduction_multiblock_test) into the isolated tst/build_unit
directory, runs it on FOUR MeshBlocks (mesh nx1=64, meshblock nx1=16) on a single rank,
and asserts it exited 0 (all checks passed).

This is the multi-block half of the framework-canary verification for the shared
per-substage ghost-exchange helper MeshBoundaryValuesCC::SyncParabolicGhosts. The pgen
seeds the live hydro conserved field with the GLOBAL discrete cosine diffusion eigenmode,
advances it operator-split by the RKL2 STS integrator through the exact production task
body (parabolic::OperatorSplitStep on hydro u0 -- whose ApplyBoundary delegates the
cross-block neighbor refresh to SyncParabolicGhosts), and checks the global mode
reproduces the analytic exp(lambda t) decay ACROSS the three internal block boundaries,
conserves total energy, and actually diffuses. The single-block analytic correctness of
the same operator is pinned separately by test_unit_parabolic_conduction. With the
insulated-only (pre-#108) ghost fill each block would decay as an independent sub-box and
the decay-match check fails -- so this test is red without the wiring and green with it.

Oracle: Layer 1 -- analytic. On the global insulated box the discrete cosine mode
E_i = E0 + A cos(theta (ig+1/2)), theta = m pi / N, is an exact eigenvector of the
discrete conduction operator with eigenvalue lambda = -(2 D/dx^2)(1 - cos theta),
D = kappa(g-1)/rho, so E_i(t) = E0 + A cos(...) exp(lambda t). The run must reproduce this
decay. Reference: closed form (diffusion eigenmode); Meyer, Balsara & Aslam 2014 (RKL2).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the multi-block operator-split conduction test; pass iff it exits 0."""
    passed = testutils.run_unit_test("conduction_multiblock_test")
    assert passed, (
        "conduction_multiblock_test reported a failing check (nonzero exit): the "
        "operator-split conduction did not reproduce the analytic decay across MeshBlock "
        "boundaries (SyncParabolicGhosts wiring)"
    )
