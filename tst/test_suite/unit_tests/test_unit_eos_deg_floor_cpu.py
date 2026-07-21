"""
Auto-collected wrapper for the degeneracy-pressure floor pgen-based unit test (#209).

Builds src/pgen/unit_tests/eos_deg_floor_test.cpp
(-D PROBLEM=unit_tests/eos_deg_floor_test) into the isolated tst/build_unit directory,
runs it against the REAL committed aluminum IONMIX table (inputs/ionmix/al-imx-004.cn4)
at the B1 deck's <units> and ion mass, and asserts it exited 0.

The test verifies the zero-temperature Fermi floor on the tabulated 3T electron
pressure (eos_table_3t::EosTable3T::SetDegeneracyFloor / DegeneracyPressureFloor and
its application inside ConsToPrim2T): opt-in (fresh table byte-identical), an
independent CGS oracle at the OBSERVED #209 collapse densities (23/46/164 code =
62/124/443 g/cc), exact zero at and below the solid reference, restored rho^(5/3)
stiffness beyond the 44.8 g/cc table edge (where edge-clamping made dP/drho = 0),
floor-dominated closure pressure at the collapse state, bare-table hot states, and an
untouched Te inversion.

Oracle: Layer 2 -- hardcoded first-principles values (CODATA hbar/m_e, Z*=3, Al ion
mass, B1 units) computed outside the implementation.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the degeneracy-floor unit test; pass iff it exits 0."""
    table = os.path.join(
        testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4")
    passed = testutils.run_unit_test(
        "eos_deg_floor_test",
        args=[f"problem/eos_table={table}"],
    )
    assert passed, (
        "eos_deg_floor_test reported a failing check (nonzero exit)"
    )
