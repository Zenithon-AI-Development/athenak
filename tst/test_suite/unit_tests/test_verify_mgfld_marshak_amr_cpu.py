"""
AMR (static-SMR) operator-split MULTIGROUP-FLD Marshak verification ([A4]/#111).

The block-AMR companion to test_verify_mgfld_marshak_multiblock_cpu: builds the same pgen
(src/pgen/unit_tests/mgfld_marshak_multiblock_test.cpp) and runs it on a 2D STATICALLY
REFINED mesh (inputs/unit_tests/mgfld_marshak_amr_test.athinput) where the interior region
x1 in [0.1,0.3] is refined to level 1. Each group's Marshak front (uniform in x2) crosses
the fine/coarse boundaries at x1=0.1 and x1=0.3, so this exercises BOTH the restriction/
prolongation path of MeshBoundaryValuesCC::SyncParabolicGhosts AND the conservative
refinement-boundary flux correction (CorrectFlux) for the multigroup operator under
block-AMR -- the single-level _cpu test only exercises the same-level neighbor copy. It
proves the multigroup FLD wiring (#111/[A4]) reproduces the per-group erfc Marshak
profiles across a coarse/fine boundary AND conserves energy across it (battery 4).

Static refinement => the mesh never regrids (dynamic-regrid array registration is pinned
separately by test_verify_mgfld_regrid, #111/[A4]).

Oracle: Layer 1 -- analytic (see test_verify_mgfld_marshak_multiblock_cpu for the waves).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "mgfld_marshak_multiblock_test"


def test_run():
    """Build + run the multigroup-FLD Marshak test on a static-SMR mesh (exit 0)."""
    repo_root = testutils._repo_root()
    input_file = os.path.join(
        repo_root, "inputs", "unit_tests", "mgfld_marshak_amr_test.athinput"
    )
    fixture = os.path.join(
        repo_root, "inputs", "radiation_fld", "multigroup_opacity.cn4"
    )
    passed = testutils.run_unit_test(
        NAME, input_file=input_file, args=[f"mhd/mgfld_opacity_file={fixture}"]
    )
    assert passed, (
        "mgfld_marshak_multiblock_test (static-SMR mesh) reported a failing check "
        "(nonzero exit): the operator-split multigroup FLD did not reproduce the "
        "per-group Marshak waves across a coarse/fine boundary, or conserve energy "
        "across it (SyncParabolicGhosts restriction/prolongation + CorrectFlux)"
    )
