"""
Auto-collected wrapper for the per-cell grey chi(rho,Te) refresh pgen-based unit test
(#204).

Builds src/pgen/unit_tests/fld_local_opacity_refresh_test.cpp
(-D PROBLEM=unit_tests/fld_local_opacity_refresh_test) into the isolated tst/build_unit
directory, runs it against the REAL committed aluminum IONMIX table
(inputs/ionmix/al-imx-004.cn4, both the EOS closure and the multigroup opacity), and
asserts it exited 0.  The test verifies radiationfld::RefreshGreyChiField
(radiation_fld/local_grey_opacity.hpp): per cell, Te is inverted through the same
tabulated_3t closure ConsToPrim2T uses, the grey Rosseland mean of the multigroup
opacity is evaluated at the LOCAL (rho,Te), and the code-unit extinction
chi = kappa_R*rho_cgs*L lands in the field the grey FLD operator diffuses with -- so the
solid liner is opaque (chi within the physical band of the old frozen constant) while
the tenuous vacuum gap is >= 1e3 x more transparent (#204), chi tracks the live
temperature, and a positive floor guards the operator's 1/chi.

Oracle: Layer 1 -- structural/physical on the real table: public-lookup-chain
equivalence, the liner/gap transparency ordering, temperature liveness, and the floor.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the local-opacity refresh unit test; pass iff it exits 0."""
    table = os.path.join(
        testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
    )
    passed = testutils.run_unit_test(
        "fld_local_opacity_refresh_test",
        args=[f"problem/eos_table={table}"],
    )
    assert passed, (
        "fld_local_opacity_refresh_test reported a failing check (nonzero exit)"
    )
