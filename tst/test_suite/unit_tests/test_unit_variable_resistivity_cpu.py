"""
Auto-collected wrapper for the variable Spitzer/Braginskii resistivity pgen-based unit
test.

Builds src/pgen/unit_tests/variable_resistivity_test.cpp
(-D PROBLEM=unit_tests/variable_resistivity_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed). The test verifies that
the variable magnetic-diffusivity helper in diffusion/variable_resistivity.hpp composes
the SIM-61 transport chain correctly (rho,T_e -> eta_par/eta_perp [Ohm m] -> eta/mu0 ->
code units), applies the linear code<->SI conversions, reproduces the density-independence
and T^-1.5 scaling of Spitzer resistivity, distinguishes eta_perp >= eta_par under
magnetization, and activates the vacuum-resistivity floor below a density threshold
(issue [7a], ADR-0003).

Oracle: Layer 1 -- analytic.  The variable magnetic-diffusivity helper composes the
Spitzer/Braginskii chain: density-independent T^-1.5 scaling, eta_perp >= eta_par
under magnetization, the code<->SI conversions, and the vacuum-resistivity floor below a
density threshold.  References: Spitzer 1962; Braginskii 1965 (Reviews of Plasma Physics
1, 205).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the variable-resistivity unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("variable_resistivity_test")
    assert passed, "variable_resistivity_test reported a failing check (nonzero exit)"
