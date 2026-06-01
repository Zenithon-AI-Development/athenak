"""
Tabulated 3T (IONMIX) EOS live-MHD cons->prim wiring unit test (issue [P1]/#159).

Builds src/pgen/unit_tests/tabulated_mhd_c2p_test.cpp (-D
PROBLEM=unit_tests/tabulated_mhd_c2p_test) into the isolated tst/build_unit directory,
runs it against the real-material aluminum IONMIX table (inputs/ionmix/al-imx-004.cn4),
and asserts it exited 0 (all checks passed).

The test exercises the new end-to-end wiring of the tabulated 3T EOS into the live MHD
solver (ADR-0002/0007): the mhd.cpp EOS selector accepting `eos=tabulated_3t` and reading
the .cn4 table (via ionmix_eos_reader) into the EosTable3T held on the MHD package (AC#1);
and the package's own peos->ConsToPrim routing through eos_table_3t::ConsToPrim2T on a
uniform state to produce table-consistent derived temperatures T_e/T_i, with a
cons->prim->cons round-trip that conserves total energy + the electron-energy scalar
(AC#2).  The .cn4 path is passed as an absolute-path command-line override because the
unit-test binary runs from tst/build_unit/<name>/src.

Oracle: Layer 1 -- analytic round-trip.  Seeding the conserved electron/ion internal
energies from known (rho, T_e, T_i) via the table's forward lookups and inverting them
back through ConsToPrim2T must recover the same temperatures (the same monotone e->T
round-trip the cons_to_prim_2t unit test verifies, now with the real Al table); the
cons->prim->cons assembly is EOS-independent so it conserves energy exactly.
References: tabulated EOS inversion; FLASH IONMIX aluminum table (al-imx-004.cn4).
"""

# Modules
import os

import test_suite.testutils as testutils


def _table_path(fname):
    return os.path.join(testutils._repo_root(), "inputs", "ionmix", fname)


def test_run():
    """Build + run the tabulated-3T live-MHD cons->prim unit test; pass iff it exits 0."""
    table = _table_path("al-imx-004.cn4")
    passed = testutils.run_unit_test(
        "tabulated_mhd_c2p_test", args=[f"mhd/eos_table={table}"]
    )
    assert passed, "tabulated_mhd_c2p_test reported a failing check (nonzero exit)"
