"""
Static-SMR (AMR coarse/fine) operator-split anisotropic (Braginskii) conduction
verification ([A5]/#112, ADR-0006/ADR-0001).

Runs the SAME pgen (src/pgen/unit_tests/aniso_conduction_multiblock_test.cpp) as the
uniform multi-block test, but with the static-SMR athinput
(inputs/unit_tests/aniso_conduction_amr_test.athinput): a 2D [-1,1]^2 mesh of 4x4 root
MeshBlocks with the central [-0.5,0.5]^2 region refined to level 1, so a refined patch is
surrounded by coarse blocks. This exercises the coarse/fine path of the operator's
per-substage MeshBoundaryValuesCC::SyncParabolicGhosts (RestrictCC / FillCoarse /
ProlongateCC) and the conservative pbval_flux_ flux correction (#33) that carries
field-aligned heat across the refinement boundaries.

The pgen detects the multilevel mesh and (a) loosens the field-aligned analytic-decay
tolerance (the c/f prolongation is only O(dx^2)-consistent), (b) skips the no-exchange RED
discriminator (ill-defined under static refinement), and (c) checks conservation on the
VOLUME-WEIGHTED energy (fine cells are smaller). refinement = static, so the mesh never
regrids -- the operator diffuses the live conserved field, which is already
AMR-regrid-registered as the MHD field, so dynamic regrid needs no new code; the per-
substage c/f exchange is what carries the operator across the refinement boundary.

Oracle: Layer 1 -- analytic (see test_verify_aniso_conduction_multiblock_cpu).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the static-SMR anisotropic-conduction test; pass iff it exits 0."""
    repo_root = testutils._repo_root()
    input_file = os.path.join(
        repo_root, "inputs", "unit_tests", "aniso_conduction_amr_test.athinput"
    )
    passed = testutils.run_unit_test(
        "aniso_conduction_multiblock_test", input_file=input_file
    )
    assert passed, (
        "aniso_conduction_multiblock_test (static-SMR) reported a failing check (nonzero "
        "exit): the operator-split anisotropic conduction did not keep heat aligned "
        "and conservative across the AMR coarse/fine boundaries (SyncParabolicGhosts c/f "
        "path + pbval_flux_ correction)"
    )
