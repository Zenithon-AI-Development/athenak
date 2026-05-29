"""
AMR (static-SMR) operator-split grey-FLD Marshak verification ([A3]/#110, ADR-0001/0009).

The block-AMR companion to test_verify_fld_marshak_multiblock_cpu: builds the same pgen
(src/pgen/unit_tests/fld_marshak_multiblock_test.cpp) and runs it on a 2D STATICALLY
REFINED mesh (inputs/unit_tests/fld_marshak_amr_test.athinput) where the interior region
x1 in [0.1,0.3] is refined to level 1. The Marshak front (uniform in x2) crosses the
fine/coarse boundaries at x1=0.1 and x1=0.3, so this exercises the restriction/prolong
path of MeshBoundaryValuesCC::SyncParabolicGhosts under block-AMR -- the single-level _cpu
test only exercises the same-level direct neighbor copy. It proves the grey FLD wiring
(#110/[A3]) reproduces the analytic erfc Marshak profile across a coarse/fine boundary.

Static refinement => the mesh never regrids, so the standalone radiation array is never
re-mapped (dynamic-regrid registration for radiation group arrays is #111/[A4]).

Oracle: Layer 1 -- analytic (see test_verify_fld_marshak_multiblock_cpu for the wave).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the grey-FLD Marshak test on a static-SMR mesh; pass iff it exits 0."""
    repo_root = testutils._repo_root()
    input_file = os.path.join(
        repo_root, "inputs", "unit_tests", "fld_marshak_amr_test.athinput"
    )
    passed = testutils.run_unit_test(
        "fld_marshak_multiblock_test", input_file=input_file
    )
    assert passed, (
        "fld_marshak_multiblock_test (static-SMR mesh) reported a failing check (nonzero "
        "exit): the operator-split grey FLD did not reproduce the analytic Marshak wave "
        "across a coarse/fine boundary (SyncParabolicGhosts restriction/prolongation)"
    )
