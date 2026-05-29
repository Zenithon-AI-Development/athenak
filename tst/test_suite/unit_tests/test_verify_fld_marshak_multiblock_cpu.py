"""
Multi-block operator-split grey-FLD Marshak verification ([A3]/#110, ADR-0001/ADR-0009).

Builds src/pgen/unit_tests/fld_marshak_multiblock_test.cpp
(-D PROBLEM=unit_tests/fld_marshak_multiblock_test) into the isolated tst/build_unit
directory, runs it on FOUR MeshBlocks (mesh nx1=128, meshblock nx1=32) on a single rank,
and asserts it exited 0 (all checks passed).

This is the multi-block verification of the grey-FLD wiring into the MHD timestep. The
pgen seeds the live MHD radiation-energy field (mhd/fld_operator_split) with a cold slab,
heats it from the inner-x1 Dirichlet source, advances it operator-split by the RKL2 STS
integrator through the exact production task body (parabolic::OperatorSplitStep on the
live MHD erad -- whose ApplyBoundary delegates the cross-block neighbor refresh to
MeshBoundaryValuesCC::SyncParabolicGhosts), and checks the wave reproduces the analytic
erfc Marshak profile ACROSS the three internal block boundaries, crosses the first block
boundary, and (a separate insulated field) conserves total radiation energy. With the
insulated-only ghost fill (no exchange) the wave cannot leave the source block and the
erfc-match / cross-block checks fail -- so this test is red without the wiring and green
with it. The single-block analytic correctness of the same operator is pinned separately
by verification/test_verify_marshak_fld.

Oracle: Layer 1 -- analytic. In the optically-thick (Larsen limiter -> 1/3) equilibrium-
diffusion limit grey FLD reduces to linear diffusion with D = c/(3 chi), whose cold-slab
half-space solution is E_r(x,t)=e_floor+(e_source-e_floor)*erfc((x-x1min)/(2*sqrt(Dt))).
Reference: closed form (erfc Marshak wave); Meyer, Balsara & Aslam 2014 (RKL2).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the multi-block grey-FLD Marshak test; pass iff it exits 0."""
    passed = testutils.run_unit_test("fld_marshak_multiblock_test")
    assert passed, (
        "fld_marshak_multiblock_test reported a failing check (nonzero exit): the "
        "operator-split grey FLD did not reproduce the analytic Marshak wave across "
        "MeshBlock boundaries (SyncParabolicGhosts wiring)"
    )
