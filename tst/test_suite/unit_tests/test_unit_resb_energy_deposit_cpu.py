"""
resb B_phi write-back energy-budget unit test (issue #193).

Builds src/pgen/unit_tests/resb_energy_deposit_test.cpp (-D
PROBLEM=unit_tests/resb_energy_deposit_test) into the isolated tst/build_unit directory
and runs it TWICE with different <mhd>/resb_deposit_heat overrides, asserting each leg
exited 0 (all checks passed).

BACKGROUND.  The #192 fix made the resb bphi->b0.x2f representation swap energy-consistent
(it bills 0.5*(b_new^2-b_old^2) to u0(IEN) so e_int is invariant under the swap) but DROPS
the resistively-dissipated field energy (the Ohmic eta*J^2 heat) -- a one-signed,
stabilising omission, so on the #192 baseline TOTAL energy decreases by exactly the
dropped field energy.  #193 adds a gated deposit (<mhd> resb_deposit_heat, default false
=> #192 byte-identical) that adds it back, to the ELECTRONS (scalar 0 = u0(nmhd)), closing
the budget: field energy lost == electron heat gained, E_tot invariant, ions unchanged.

The pgen seeds a div-free axisymmetric b0, a known E_tot and a known e_ele, populates
`bphi` with the phi-mean seed shrunk by a known factor (a clean one-step resb stand-in
that removes a known POSITIVE field energy), calls the real CoupleResbBphiToB0(), and --
branching on the active resb_deposit_heat gate -- asserts:
  deposit OFF (#192 baseline): TOTAL energy drops by exactly the field-energy decrease,
                               the electrons did NOT gain the heat, ions unchanged;
  deposit ON  (#193):          TOTAL energy conserved to round-off, electrons gained
                               exactly the dissipated field energy, ions unchanged.
"""

# Modules
import test_suite.testutils as testutils


def test_run_deposit_off():
    """#192 baseline leg: drop the heat => E_tot drops by the field decrease."""
    passed = testutils.run_unit_test(
        "resb_energy_deposit_test",
        args=["mhd/resb_deposit_heat=false"],
    )
    assert passed, (
        "resb_energy_deposit_test (deposit OFF) reported a failing check (nonzero exit): "
        "the #192 baseline must drop the dissipated field energy (E_tot drops by exactly "
        "the field-energy decrease; electrons unheated)."
    )


def test_run_deposit_on():
    """#193 leg: deposit the dissipated heat on the electrons => E_tot conserved."""
    passed = testutils.run_unit_test(
        "resb_energy_deposit_test",
        args=["mhd/resb_deposit_heat=true"],
    )
    assert passed, (
        "resb_energy_deposit_test (deposit ON) reported a failing check (nonzero exit): "
        "the #193 deposit must conserve TOTAL energy (field lost == electron heat "
        "gained) and leave the ion internal energy unchanged."
    )
