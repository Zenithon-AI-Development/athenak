"""
Static-SMR operator-split CYLINDRICAL resistive B_phi diffusion verification
([A6]/#113, ADR-0004/ADR-0001).

Builds the SAME pgen as test_verify_resb_bphi_multiblock_cpu
(src/pgen/unit_tests/resb_bphi_multiblock_test.cpp) but runs it on a static-SMR mesh
(input file resb_bphi_amr_test.athinput): a 1-D radial cylinder r in [0,1] with 8 root
MeshBlocks (mesh nx1=128, meshblock nx1=16), the mid-radius band r in [0.25,0.625] refined
to level 1, so a refined radial patch is surrounded by coarse blocks. This exercises the
coarse/fine path of the operator's per-substage MeshBoundaryValuesCC::SyncParabolicGhosts
(RestrictCC / FillCoarse / ProlongateCC) and the conservative pbval_flux_ flux correction
(#33) that carries the diffused B_phi (the area-weighted poloidal flux) across the
refinement boundaries. The c/f faces sit in smooth regions of J_1 (away from the axis
singularity).

The pgen detects the multilevel mesh and loosens the analytic-decay / shape / rate
tolerances (the c/f prolongation is only O(dx^2)-consistent) and skips the no-exchange RED
discriminator (ill-defined under static refinement). refinement = static, so the mesh
never regrids; the per-substage c/f exchange is what carries the operator across the
refinement boundary. Single rank, so the per-block explicit dt is reduced globally on the
rank -> one RKL2 stage count.

Oracle: Layer 1 -- analytic (see test_verify_resb_bphi_multiblock_cpu).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the static-SMR resistive-B_phi test; pass iff it exits 0."""
    repo_root = testutils._repo_root()
    input_file = os.path.join(
        repo_root, "inputs", "unit_tests", "resb_bphi_amr_test.athinput"
    )
    passed = testutils.run_unit_test(
        "resb_bphi_multiblock_test", input_file=input_file
    )
    assert passed, (
        "resb_bphi_multiblock_test (static-SMR) reported a failing check (nonzero exit): "
        "the operator-split cylindrical resistive B_phi diffusion did not reproduce the "
        "analytic J_1 eigenmode decay across the AMR coarse/fine boundaries "
        "(SyncParabolicGhosts c/f path + pbval_flux_ correction)"
    )
