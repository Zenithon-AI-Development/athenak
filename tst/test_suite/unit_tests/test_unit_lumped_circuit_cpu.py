"""
Auto-collected wrapper for the lumped-element circuit ODE pgen-based unit test.

Builds src/pgen/unit_tests/lumped_circuit_test.cpp
(-D PROBLEM=unit_tests/lumped_circuit_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test verifies the coupled
lumped-element circuit ODE (src/circuit/lumped_circuit.hpp) (issue [18b]/#34, ADR-0005
modes B/C): the once-per-step RK4 advance of the series RLC circuit driven by an
open-circuit voltage source reproduces the analytic series-RLC step response (underdamped
and overdamped current, capacitor voltage, and the RL no-capacitor limit); mode C consumes
a Faraday load voltage (a constant V_load lowers the effective source to V0-V_load) while
mode B (fixed RLC) ignores it; and the integrated current drives the B_phi boundary via
BoundaryBphi == mu0*I/(2*pi*r).

Oracle: Layer 1 -- analytic.  The once-per-step RK4 advance of the series-RLC circuit
reproduces the analytic step response (underdamped/overdamped current, capacitor voltage,
the RL no-capacitor limit); mode C consumes a Faraday V_load (lowers the effective source
to V0-V_load) while mode B ignores it; B_phi = mu0*I/(2*pi*r).  References:
closed form (series-RLC step response).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the lumped-circuit unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("lumped_circuit_test")
    assert passed, "lumped_circuit_test reported a failing check (nonzero exit)"
