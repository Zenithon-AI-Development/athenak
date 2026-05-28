"""
Auto-collected wrapper for the diffusive-operator AMR flux-correction unit test.

Builds src/pgen/unit_tests/fld_amr_flux_correction_test.cpp
(-D PROBLEM=unit_tests/fld_amr_flux_correction_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed).

The test verifies that the operator-split diffusive operators conserve the diffused
quantity across a STATIC refinement (SMR) boundary once their face flux is registered for
AthenaK's conservative fine->coarse flux correction (issue [19a]/#33). It builds a 2D
periodic mesh with a single interior root block refined to level 1, fills the grey
radiation-energy field (and its cross-block ghosts) from a periodic analytic profile,
evaluates the FLDGreyOperator action M(E) = div(D grad E), and checks the discrete
conservation residual SUM(dV*M) is at round-off relative to its scale SUM|dV*M| (it is
O(1) without the flux correction -> the red->green discriminator). The same registration
is also wired into the anisotropic conduction (#18) and resistive B_phi (#27) operators.

Oracle: Layer 1 -- analytic.  With the operator's face flux registered for AthenaK's
conservative fine->coarse flux correction, the diffusion operator action M(E) = div(D grad
E) conserves the diffused quantity across a static refinement boundary: the discrete
conservation residual SUM(dV*M) sits at round-off vs SUM|dV*M| (it is O(1) without
the correction -- the red->green discriminator).  References: closed form (conservation
identity).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the AMR flux-correction unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("fld_amr_flux_correction_test")
    assert passed, "fld_amr_flux_correction_test reported a failing check (nonzero exit)"
