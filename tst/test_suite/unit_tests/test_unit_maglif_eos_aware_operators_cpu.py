"""
EOS-aware operator closures unit test (issue [P7c]/#183, ADR-0012 gap c).

Builds src/pgen/unit_tests/maglif_eos_aware_operators_test.cpp (-D
PROBLEM=unit_tests/maglif_eos_aware_operators_test) into the isolated tst/build_unit
directory and runs it against the real aluminum IONMIX table, asserting it exited 0 (all
checks passed).

Two reference-model-set consistency facts, both checked against the SAME tabulated EOS
closure ``ConsToPrim2T`` uses, that the faithful B1 run needs (ADR-0012 gap c):

  (AC#1, conduction) The anisotropic-conduction temperature recovery
  ``anisocond::EosAwareTemp::Temp`` in EOS-aware mode equals the tabulated_3t electron
  temperature ``T_e = table.Te(rho, e_ele/rho)`` (the derived/cached field of the 3T
  formulation), and recovers the node temperature each test cell was filled at -- NOT the
  ideal-gamma bookkeeping value ``T = (gamma-1) eint/rho``.  The test ALSO asserts (RED
  first) that the ideal-gamma recovery, the pre-#183 conduction closure, is grossly
  inconsistent with the eV closure, so the EOS-aware fix is genuinely necessary.

  (AC#2, matter-radiation) The EOS-aware grey-coupling heat capacity equals the tabulated
  closure ``c_v = rho*(c_v,e(rho,T_e) + c_v,i(rho,T_i))`` (volumetric), is positive, and
  is NOT the constant ``mrad_cv`` placeholder the coupling used before #183.

The table path is passed as an absolute-path command-line override because the unit-test
binary runs from tst/build_unit/<name>/src.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the EOS-aware operator closures unit test; pass iff it exits 0."""
    repo = testutils._repo_root()
    table = os.path.join(repo, "inputs", "ionmix", "al-imx-004.cn4")
    passed = testutils.run_unit_test(
        "maglif_eos_aware_operators_test",
        args=[f"mhd/eos_table={table}"],
    )
    assert passed, (
        "maglif_eos_aware_operators_test reported a failing check (nonzero exit)"
    )
