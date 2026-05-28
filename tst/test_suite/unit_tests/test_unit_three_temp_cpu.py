"""
Auto-collected wrapper for the conservative 3T energy-reconciliation pgen-based unit test.

Builds src/pgen/unit_tests/three_temp_test.cpp
(-D PROBLEM=unit_tests/three_temp_test) into the isolated tst/build_unit directory, runs
it, and asserts it exited 0 (all checks passed). The test exercises the 3T energy
functions in eos/three_temperature.hpp (issue [14a]/#19, ADR-0002) on real device kernels.

It verifies: electron internal energy `e_ele` plus ion-by-subtraction
(e_ion = E_tot - KE [- 1/2 B^2] - e_ele) conserves total energy by construction (hydro and
MHD); the PdV compression work splits between species by pressure fraction
(-p_ele*div(v)*dt to the electrons, the complement to the ions); irreversible shock
dissipation lands entirely on the ions; the 1T limit (T_e = T_i) recovers the
single-temperature internal energy / pressure and preserves species fractions under PdV;
and the pressure fraction fed from the tabulated 3T EOS (e -> T -> p, #6) matches it.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the 3T energy-reconciliation unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("three_temp_test")
    assert passed, "three_temp_test reported a failing check (nonzero exit)"
