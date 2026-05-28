"""
Auto-collected wrapper for the tabulated 2T EOS conserved->primitive closure unit test.

Builds src/pgen/unit_tests/cons_to_prim_2t_test.cpp
(-D PROBLEM=unit_tests/cons_to_prim_2t_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test exercises the per-cell
closure eos_table_3t::ConsToPrim2T in eos/cons_to_prim_2t.hpp (issue [14b]/#26, ADR-0002)
on real device kernels.

It verifies the tabulated 2T EOS ConsToPrim path: given (rho, e_ele, e_ion) -- density and
the electron/ion internal energy densities (the conserved inputs of the 3T formulation,
#19) -- the closure inverts each species energy to its temperature via the tabulated EOS
(#6), caching T_e/T_i, and returns p_gas = p_ele(rho,T_e) + p_ion(rho,T_i). Batteries:
two-temperature recovery (T_e != T_i) matching the table closed forms; species
independence (distinct gammas, p_ele uses T_e and p_ion uses T_i); out-of-range edge
clamping; and the shared-gamma ideal-gas limit p_gas = (gamma-1)(e_ele + e_ion).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the tabulated 2T EOS C2P closure unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("cons_to_prim_2t_test")
    assert passed, "cons_to_prim_2t_test reported a failing check (nonzero exit)"
