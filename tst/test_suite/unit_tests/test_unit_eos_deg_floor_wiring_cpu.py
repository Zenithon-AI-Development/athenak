"""
Auto-collected wrapper for the <mhd> eos_deg_zstar wiring pgen-based unit test (#209).

Builds src/pgen/unit_tests/eos_deg_floor_wiring_test.cpp
(-D PROBLEM=unit_tests/eos_deg_floor_wiring_test) into the isolated tst/build_unit
directory, runs it against the real committed aluminum IONMIX table, and asserts it
exited 0.  The test verifies the production wiring of the Fermi degeneracy-pressure
floor (#209): `<mhd> eos_deg_zstar=3` on a tabulated_3t MHD package enables the floor
on BOTH table copies (the package cons->prim closure and the Riemann/newdt
eos_data.eos_tbl), builds K from eos_mass_per_ion + <units> (checked against the
independent CGS oracle), floors TabulatedGasPressure at the #209 cold-dense collapse
state, and stays exactly zero at the solid reference.
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
        "eos_deg_floor_wiring_test",
        args=[f"mhd/eos_table={table}"],
    )
    assert passed, (
        "eos_deg_floor_wiring_test reported a failing check (nonzero exit)"
    )
