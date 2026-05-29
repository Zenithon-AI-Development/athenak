"""
Dynamic-AMR MULTIGROUP-FLD regrid (group-array restrict/prolong) verification ([A4]/#111).

Builds src/pgen/unit_tests/mgfld_regrid_test.cpp (-D PROBLEM=unit_tests/mgfld_regrid_test)
and RUNS THE DRIVER (nlim>0) on a dynamic-AMR mesh
(inputs/unit_tests/mgfld_regrid_test.athinput) where a user refinement criterion refines
the central x1 blocks for the first half of the run and de-refines them for the second
half. The standalone per-group radiation-energy array MHD::erad_mg is diffused every step
by an INSULATED operator-split multigroup FLD operator (so the conserved quantity is the
volume-weighted total radiation energy), and is registered with the AMR regrid machinery
(mesh_refinement.cpp DerefineCCSameRank / CopyForRefinementCC / RefineCC) so it is
conservatively restricted/prolonged whenever the mesh refines or de-refines.

This is THE dynamic-regrid array-registration gate that the static-SMR Marshak test cannot
reach (a static mesh never regrids; #110/[A3] used static SMR for this reason). The
finalize hook asserts (a) the mesh actually refined at least once and (b) the volume-
weighted total radiation energy is conserved across the regrids. Without erad_mg
in the regrid path, the field on refined / de-refined blocks is never re-mapped (stale or
uninitialized) and the total is not conserved -- so this test is red without the
registration and green with it.

Oracle: Layer 1 -- conservation. A conservative restriction/prolongation preserves the
volume-weighted integral exactly (to round-off), and the insulated diffusion preserves it
too, so the total radiation energy is invariant across a regrid.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "mgfld_regrid_test"


def test_run():
    """Build + run the dynamic-AMR multigroup-FLD regrid test; pass iff it exits 0."""
    fixture = os.path.join(
        testutils._repo_root(), "inputs", "radiation_fld", "multigroup_opacity.cn4"
    )
    passed = testutils.run_unit_test(NAME, args=[f"mhd/mgfld_opacity_file={fixture}"])
    assert passed, (
        "mgfld_regrid_test reported a failing check (nonzero exit): the multigroup "
        "group-energy arrays were not conserved across a dynamic AMR regrid "
        "(erad_mg not correctly restricted/prolonged in the regrid path)"
    )
