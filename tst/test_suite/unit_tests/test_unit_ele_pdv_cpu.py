"""
Gated electron PdV (compression-heating) driver-source unit test (issue #231).

Builds src/pgen/unit_tests/ele_pdv_test.cpp (-D PROBLEM=unit_tests/ele_pdv_test) into an
isolated tst/build_unit directory and runs it with different <mhd>/ele_pdv overrides,
asserting each leg exited 0 (all checks passed).

BACKGROUND.  three_temp::ElectronPdVRate (ADR-0002) existed but was never wired into the
MHD driver: the evolved electron internal-energy scalar e_ele (passive scalar 0 =
u0(nmhd)) changed by ADVECTION ONLY, so electrons received no compression heating and
T_e stayed cold through the MagLIF implosion while every T_e consumer (opacity, EOS-aware
conduction/coupling, the e_ion = E - e_ele subtraction) read a wrong temperature.  #231
wires d e_ele/dt = -p_ele div(v) as a gated per-stage driver source (<mhd> ele_pdv,
default false => byte-identical).

The pgen RUNS THE DRIVER on a 1D periodic ideal-gas box with a sinusoidal compressive
velocity and a uniform-fraction electron-energy scalar, then checks in a finalize hook:
  gate ON  (#231):     every cell tracks the analytic electron adiabat
                       e_ele/rho = s0*(rho/rho0)^(gamma-1); total energy conserved to
                       round-off (the source repartitions, never touches u0(IEN)).
                       RED before the wiring: e_ele/rho stays the seeded constant, so
                       the adiabat check fails by the full compression signal (~13%).
  gate OFF (baseline): the specific e_ele stays exactly the seeded constant despite the
                       compression (advection-only, today's behaviour); total energy
                       conserved.

Two further DIRECT-CALL stencil legs (problem/check overrides, no time integration) pin
the source's geometry and EOS branches:
  cyl_stencil: in cylindrical coordinates div(v) must carry the radial metric term
               (1/r) d(r v_r)/dr (for v_r=-a*r the increment is exactly 2*a*p_ele*dt; a
               cartesian-naive stencil lands at half);
  tab_stencil: with eos=tabulated_3t p_ele must close from the table at the cached
               derived T_e (PressureEle(rho, T_e)), not the ideal-gamma fallback.
"""

# Modules
import os

import test_suite.testutils as testutils


def _table_path(name):
    """Absolute path to a real IONMIX table (the binary runs from tst/build_unit)."""
    return os.path.join(testutils._repo_root(), "inputs", "ionmix", name)


def test_tab_stencil():
    """Tabulated leg: p_ele comes from the table at the cached derived T_e."""
    passed = testutils.run_unit_test(
        "ele_pdv_test",
        args=[
            "problem/check=tab_stencil",
            "mhd/eos=tabulated_3t",
            f"mhd/eos_table={_table_path('al-imx-004.cn4')}",
            "mhd/eos_mass_per_ion=4.4804e-23",  # Al: number-density axis -> g/cc
            "problem/dt=0.5",  # direct call (no CFL): keep the increment well above
            # the round-off floor of the CGS-magnitude e_ele it lands on
            "time/nlim=0",
        ],
    )
    assert passed, (
        "ele_pdv_test (tab_stencil) reported a failing check (nonzero exit): with "
        "eos=tabulated_3t the PdV source must close p_ele from the table at the cached "
        "derived T_e (PressureEle(rho, T_e)), not the ideal-gamma fallback "
        "(gamma-1)*e_ele."
    )


def test_cyl_stencil():
    """Cylindrical leg: div(v) carries the radial metric term (1/r) d(r v_r)/dr."""
    passed = testutils.run_unit_test(
        "ele_pdv_test",
        args=[
            "problem/check=cyl_stencil",
            "coord/system=cylindrical",
            "mesh/x1min=1.0",
            "mesh/x1max=2.0",
            "time/nlim=0",
        ],
    )
    assert passed, (
        "ele_pdv_test (cyl_stencil) reported a failing check (nonzero exit): in "
        "cylindrical coordinates the PdV source's div(v) must include the radial metric "
        "term, div(v) = (1/r) d(r v_r)/dr (for v_r=-a*r the increment is exactly "
        "2*a*p_ele*dt; a cartesian-naive stencil lands at half)."
    )


def test_gate_off_baseline():
    """Default-off leg: e_ele stays advection-only (byte-identical baseline)."""
    passed = testutils.run_unit_test(
        "ele_pdv_test",
        args=["mhd/ele_pdv=false"],
    )
    assert passed, (
        "ele_pdv_test (gate OFF) reported a failing check (nonzero exit): with the gate "
        "off (the default) the specific e_ele must stay exactly the seeded constant "
        "under compression (advection-only) and total energy must be conserved."
    )


def test_adiabat_gate_on():
    """#231 leg: gate on => e_ele tracks the analytic compression adiabat T_e(rho)."""
    passed = testutils.run_unit_test(
        "ele_pdv_test",
        args=["mhd/ele_pdv=true"],
    )
    assert passed, (
        "ele_pdv_test (gate ON) reported a failing check (nonzero exit): with the #231 "
        "electron PdV source wired, e_ele/rho must track the analytic adiabat "
        "s0*(rho/rho0)^(gamma-1) in every cell and total energy must be conserved to "
        "round-off."
    )
