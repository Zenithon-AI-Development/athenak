"""
SI operator-coefficient calibration anchor unit test (issue [P7d]/#184, ADR-0014).

Builds src/pgen/unit_tests/operator_si_calibration_test.cpp (-D
PROBLEM=unit_tests/operator_si_calibration_test) into the isolated tst/build_unit dir and
runs it against the real aluminum IONMIX table, asserting it exited 0 (all checks passed).

Closes the ADR-0012 coefficient-calibration gap (gap d): the reference-model-set operator
coefficients are SI-calibrated from first-principles transport / opacity and converted
into the code-unit system fixed by the <units> block (the operator analogue of the
ADR-0010 drive calibration), instead of the order-unity B1-deck placeholders.

Anchors verified (ADR-0008 Layer-1 independent oracle):

  (AC#1, resistivity) The Spitzer-Braginskii parallel resistivity equals the NRL practical
  value eta_|| = 5.2e-5 Z lnL T_e[eV]^(-3/2) Ohm m at a known classical state.

  (AC#1, conduction) The Braginskii parallel electron thermal conductivity equals the
  NRL/Spitzer-Harm value kappa_||e = 3.16 n_e k_B^2 T_e tau_e / m_e.

  (AC#1, opacity) The FLD/mrad opacity coefficient is the real aluminum IONMIX cn4
  Planck-absorption / Rosseland opacity at a known (group, T, rho) node, converted to the
  code absorption coefficient kappa[cm^2/g] * rho[g/cc] * length_cgs.

  (RED meta-checks) Each calibrated code-unit coefficient is grossly inconsistent with the
  order-unity placeholder the deck carried before #184 (resb_eta=1e-3,
  acond_kappa_conv=0.05, mrad_chi_a=0.04), proving the calibration is genuinely necessary.

The opacity table path is passed as an absolute-path command-line override because the
unit-test binary runs from tst/build_unit/<name>/src (a relative path would not resolve).
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the SI operator-calibration anchor unit test; pass iff it exits 0."""
    repo = testutils._repo_root()
    table = os.path.join(repo, "inputs", "ionmix", "al-imx-004.cn4")
    passed = testutils.run_unit_test(
        "operator_si_calibration_test",
        args=[f"problem/opacity_file={table}"],
    )
    assert passed, (
        "operator_si_calibration_test reported a failing check (nonzero exit)"
    )
