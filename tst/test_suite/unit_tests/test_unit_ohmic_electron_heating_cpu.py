"""
Auto-collected wrapper for the Ohmic->electron heating + 2T radiation-coupling unit test.

Builds src/pgen/unit_tests/ohmic_electron_heating_test.cpp
(-D PROBLEM=unit_tests/ohmic_electron_heating_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed). The test exercises, on
real device kernels (issue [14c]/#30, ADR-0002/0003):

  * three_temp::OhmicHeatingRate / ApplyOhmicElectronHeating -- the Ohmic (Joule) heating
    source deposited on the electron internal energy e_ele, consistently with the
    constrained-transport EMF (eta|J|^2 == E_res . J), so that depositing the local
    magnetic-energy decrement on e_ele and recovering e_ion by subtraction leaves the ions
    untouched and conserves total energy; and
  * radiationfld::PointImplicitElectronRadiationCoupling -- the 2T re-targeting of the
    point-implicit matter-radiation coupling to the electron T_e (relaxation to
    a T_e^4 = E_r, conservation of E_r + e_ele), including the full Ohmic -> electron ->
    radiation equilibration chain (ions untouched; all Ohmic input accounted for).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the Ohmic->electron heating unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("ohmic_electron_heating_test")
    assert passed, "ohmic_electron_heating_test reported a failing check (nonzero)"
