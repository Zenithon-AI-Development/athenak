"""
Tabulated 3T (IONMIX) EOS live MHD energy-update test (issue [P2]/#162).

Builds src/pgen/unit_tests/tabulated_mhd_energy_test.cpp (-D
PROBLEM=unit_tests/tabulated_mhd_energy_test) into the isolated tst/build_unit directory,
runs it against the real-material aluminum IONMIX table (inputs/ionmix/al-imx-004.cn4),
and asserts it exited 0 (all checks passed).

Where the #159 cons->prim unit test runs no integration, this test RUNS THE DRIVER
(nlim>0): a 1D periodic, field-free, at-rest aluminum plasma with a central electron+ion
temperature bump is integrated with rsolver=llf, which routes the hyperbolic gas pressure
through the tabulated inversion (#162, p_gas = p_ele(T_e) + p_ion(T_i) from the table).
The pgen's finalize hook then asserts:
  - the volume-integrated conserved total energy and the passive electron-energy scalar
    are each conserved to round-off across the live update (periodic + conservative LLF);
  - on the EVOLVED state the cached derived temperatures match the table forward relation
    rho*EnergyEle(rho,T_e) == e_ele (and the ion analogue) -- the live step used the
    table's T(rho,e), not an ideal-gamma value;
  - the tabulated pressure actually drove flow (max|v_x| grew to an O(1) fraction of the
    initial sound speed) -- the RED->GREEN discriminator, since without the [P2] routing
    the tabulated EOS hits the isothermal LLF branch (iso_cs=0) and stays at rest.

The .cn4 path is passed as an absolute-path command-line override because the unit-test
binary runs from tst/build_unit/<name>/src.

Oracle: Layer 1 -- method-independent analytic anchors (exact conservation of two
conserved integrals + the table's own forward/inverse round-trip on the final state).
References: tabulated EOS inversion; FLASH IONMIX aluminum table (al-imx-004.cn4).
"""

# Modules
import os

import test_suite.testutils as testutils


def _table_path(fname):
    return os.path.join(testutils._repo_root(), "inputs", "ionmix", fname)


def test_run():
    """Build + run the tabulated-3T live energy-update test; pass iff it exits 0."""
    table = _table_path("al-imx-004.cn4")
    passed = testutils.run_unit_test(
        "tabulated_mhd_energy_test", args=[f"mhd/eos_table={table}"]
    )
    assert passed, "tabulated_mhd_energy_test reported a failing check (nonzero exit)"
