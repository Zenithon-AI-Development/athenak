"""
Multi-block operator-split MULTIGROUP-FLD Marshak verification ([A4]/#111).

Builds src/pgen/unit_tests/mgfld_marshak_multiblock_test.cpp
(-D PROBLEM=unit_tests/mgfld_marshak_multiblock_test) into the isolated tst/build_unit
directory, runs it on FOUR MeshBlocks (mesh nx1=128, meshblock nx1=32) on a single rank,
and asserts it exited 0 (all checks passed).

This is the multigroup analogue of test_verify_fld_marshak_multiblock (#110/[A3], grey):
it verifies the wiring of the N-group FLDMultigroupOperator into the MHD timestep
(mhd/mgfld_operator_split). The pgen seeds the live MHD per-group radiation field with a
cold slab, heats it from the inner-x1 Dirichlet source (all groups), advances it
operator-split by RKL2 STS through the production task body (parabolic::OperatorSplitStep
on the live MHD erad_mg -- whose ApplyBoundary delegates the cross-block neighbor refresh
to MeshBoundaryValuesCC::SyncParabolicGhosts), and checks that EVERY group reproduces its
own analytic erfc Marshak profile (each with D_g = c/(3 chi_g)) across the three internal
block boundaries, that each group crossed the first block boundary, and that an insulated
field conserves total radiation energy. With the insulated-only ghost fill (no exchange) a
wave cannot leave the source block, so this test is red without the wiring, green with it.

Oracle: Layer 1 -- analytic. In the optically-thick (Larsen limiter -> 1/3) equilibrium-
diffusion limit each group reduces to linear diffusion with D_g = c/(3 chi_g), whose
cold-slab half-space solution is E_g(x,t) = e_floor + (e_source - e_floor) *
erfc((x - x1min)/(2 sqrt(D_g t))). Reference: closed form (erfc Marshak wave); Meyer,
Balsara & Aslam 2014.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "mgfld_marshak_multiblock_test"


def _fixture_arg():
    fixture = os.path.join(
        testutils._repo_root(), "inputs", "radiation_fld", "multigroup_opacity.cn4"
    )
    return [f"mhd/mgfld_opacity_file={fixture}"]


def test_run():
    """Build + run the multi-block multigroup-FLD Marshak test; pass iff it exits 0."""
    passed = testutils.run_unit_test(NAME, args=_fixture_arg())
    assert passed, (
        "mgfld_marshak_multiblock_test reported a failing check (nonzero exit): the "
        "operator-split multigroup FLD did not reproduce the per-group analytic Marshak "
        "waves across MeshBlock boundaries (SyncParabolicGhosts wiring)"
    )
