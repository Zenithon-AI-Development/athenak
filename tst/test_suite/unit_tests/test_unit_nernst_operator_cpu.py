"""
RED-first wrapper for the Nernst-operator pgen-based unit test (#238, ADR-0017).

Will build src/pgen/unit_tests/nernst_operator_test.cpp
(-D PROBLEM=unit_tests/nernst_operator_test) into the isolated tst/build_unit directory,
run it, and assert it exits 0 (all checks passed).  Per ADR-0017 (Proposed -- human
decision pending) the C++ test must verify, against closed forms:

  1. the magnetization-reduced Nernst coefficient beta_wedge(x)/x in
     diffusion/braginskii_transport.hpp reproduces the Braginskii Z=1 limits
     (-> 3.053/3.7703 ~= 0.810 unmagnetized, -> 1.5/x^2 strongly magnetized) and rolls
     off monotonically in between;
  2. the Nernst velocity points down the electron-temperature gradient with magnitude
     (beta_wedge/x)*(tau_e/m_e)*|grad(k_B T_e)|;
  3. the EMF assembly is UPWINDED: the advected B is taken from the donor cell selected
     by the sign of v_N (the Ellison section-2.1.2 cure for on-axis negative-B_z spots);
  4. gate off (<mhd> nernst = false, the default) => the operator contributes exactly
     zero (byte-identical acceptance criterion);
  5. div(B) of the CT-updated face field stays at round-off after a Nernst step.

RED-FIRST (#238): neither the operator nor the unit-test pgen exists yet, so the cmake
configure fails ("Cannot find source file ... nernst_operator_test.cpp") and
testutils.run_unit_test raises RuntimeError.  Marked xfail(strict=True) so CI stays
green now and FAILS on an unexpected pass, forcing removal of the marker when the
operator and its pgen land (only after the ADR-0017 human decision gate accepts).

Oracle: Layer 1 -- literature.  Braginskii 1965 (Reviews of Plasma Physics 1, 205)
Z=1 thermoelectric fit; Haines 1986 (Plasma Phys. Control. Fusion 28, 1705).
"""

# Modules
import pytest
import test_suite.testutils as testutils


@pytest.mark.xfail(
    strict=True,
    reason="#238 red-first: the Nernst operator and its unit-test pgen "
           "(src/pgen/unit_tests/nernst_operator_test.cpp) do not exist "
           "(ADR-0017 Proposed, human decision pending)",
)
def test_run():
    """Build + run the Nernst-operator unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("nernst_operator_test")
    assert passed, "nernst_operator_test reported a failing check (nonzero exit)"
