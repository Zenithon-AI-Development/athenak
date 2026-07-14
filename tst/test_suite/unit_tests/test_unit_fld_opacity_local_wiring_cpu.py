"""
Auto-collected wrapper for the <mhd> fld_opacity_local wiring pgen-based unit test
(#204).

Builds src/pgen/unit_tests/fld_opacity_local_wiring_test.cpp
(-D PROBLEM=unit_tests/fld_opacity_local_wiring_test) into the isolated tst/build_unit
directory, runs it against the real committed aluminum IONMIX table, and asserts it
exited 0.  The test verifies the production wiring of the per-cell grey opacity (#204):
`<mhd> fld_opacity_local=true` on a tabulated_3t MHD package reads the multigroup
opacity from the SAME cn4 file as the EOS (density axis rescaled to g/cc via
eos_mass_per_ion), allocates the per-cell chi field, registers it with the grey FLD
operator (EnableLocalChi), and MHD::RefreshFldLocalChi() fills it from the live u0 --
a two-zone liner|vacuum state refreshes to an opaque liner and a >= 1e3 x more
transparent gap, ghosts included.

Oracle: Layer 1 -- structural/physical through the production member path.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the fld_opacity_local wiring unit test; pass iff it exits 0."""
    table = os.path.join(
        testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
    )
    passed = testutils.run_unit_test(
        "fld_opacity_local_wiring_test",
        args=[f"mhd/eos_table={table}"],
    )
    assert passed, (
        "fld_opacity_local_wiring_test reported a failing check (nonzero exit)"
    )
