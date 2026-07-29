"""
Dynamic-AMR GREY-FLD + resb-B_phi regrid (array restrict/prolong) verification (#248).

Builds src/pgen/unit_tests/grey_resb_regrid_test.cpp
(-D PROBLEM=unit_tests/grey_resb_regrid_test) and RUNS THE DRIVER (nlim>0) on a
dynamic-AMR mesh (inputs/unit_tests/grey_resb_regrid_test.athinput) where a user
refinement criterion refines the central x1 blocks for the first half of the run and
de-refines them for the second half. The standalone GREY radiation-energy array MHD::erad
is diffused every step by an INSULATED operator-split grey FLD operator (so the conserved
quantity is the volume-weighted total radiation energy), and the standalone resb B_phi
array MHD::bphi is held frozen (resb_eta = 0; the resb x1 boundary is Dirichlet-0 by
construction, so only a frozen field furnishes a conservation oracle). Both arrays must
be registered with the AMR regrid machinery (mesh_refinement.cpp DerefineCCSameRank /
CopyCC / CopyForRefinementCC / RefineCC) so they are conservatively restricted/prolonged
whenever the mesh refines or de-refines -- the grey/resb port of the #111/[A4] erad_mg
registration.

This is THE dynamic-regrid array-registration gate that the static-SMR multiblock tests
cannot reach (a static mesh never regrids). The finalize hook asserts (a) the mesh
actually refined at least once and (b) the volume-weighted totals of erad AND bphi are
conserved across the regrids -- both at the END of the run and at EVERY AMR check
mid-run (the refinement hook tracks the running max deviation; the mid-run tracking is
what catches a dropped FROZEN field, since a symmetric refine-then-derefine cycle
returns the never-re-mapped stale array slots to their original geometry by the end).
Without erad/bphi in the regrid path, the fields on refined / de-refined blocks are
never re-mapped (stale) and the totals deviate -- so this test is red without the
registration and green with it.

Oracle: Layer 1 -- conservation. A conservative restriction/prolongation preserves the
volume-weighted integral exactly (to round-off); the insulated grey diffusion (and the
frozen bphi) preserve it too, so both totals are invariant at every AMR check.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

import test_suite.testutils as testutils

NAME = "grey_resb_regrid_test"


def test_run():
    """Build + run the dynamic-AMR grey-FLD + resb regrid test; pass iff it exits 0."""
    passed = testutils.run_unit_test(NAME)
    assert passed, (
        "grey_resb_regrid_test reported a failing check (nonzero exit): the grey "
        "radiation-energy array (erad) and/or the resb B_phi array (bphi) were not "
        "conserved across a dynamic AMR regrid (not restricted/prolonged in the "
        "regrid path)"
    )
