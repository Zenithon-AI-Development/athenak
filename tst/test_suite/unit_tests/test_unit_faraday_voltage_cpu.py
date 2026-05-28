"""
Auto-collected wrapper for the Faraday load-voltage reduction pgen-based unit test.

Builds src/pgen/unit_tests/faraday_voltage_test.cpp
(-D PROBLEM=unit_tests/faraday_voltage_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test exercises
circuit/faraday_voltage.hpp on real device kernels in a cylindrical (ADR-0004) MHD block
(issue [18a]/#31, ADR-0005 mode-C feedback term).

It verifies: the global Kokkos reduction of the poloidal magnetic flux
Phi = int int B_phi dr dz matches the exact discrete poloidal-plane sum (including the
divide-by-(global nphi) per-plane average, with nx2=2 phi-planes); the Faraday load
voltage V_load = d(Phi)/dt from the FaradayVoltage monitor equals the analytic rate (and
the first sample, with no prior, returns 0); and FillFaradayHistory exposes the reduced
flux and voltage as the two history-output columns "Bphi_flux" and "V_load".

Oracle: Layer 1 -- analytic.  The Kokkos-reduced poloidal flux Phi = int int B_phi dr dz
matches the exact discrete poloidal-plane sum, and the Faraday load voltage
V_load = d(Phi)/dt matches the analytic rate (first sample returns 0), exposed as the
history columns Bphi_flux and V_load.  References: closed form (Faraday's law; discrete
flux quadrature).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the Faraday load-voltage unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("faraday_voltage_test")
    assert passed, "faraday_voltage_test reported a failing check (nonzero exit)"
