"""
Auto-collected wrapper for the tabulated 3T EOS representation pgen-based unit test.

Builds src/pgen/unit_tests/eos_table_3t_test.cpp
(-D PROBLEM=unit_tests/eos_table_3t_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test verifies that the common
tabulated 3T EOS representation in eos/eos_table_3t.hpp (issue [13a]/#6, ADR-0002/0007):
reproduces a synthetic ideal-gas table's analytic relations (forward energies, pressures,
mean ionization, heat capacities) under log-log bilinear interpolation; round-trips
T->e->T; clamps out-of-range (rho, T, e) queries to the table edges (no out-of-bounds
lookups); and inverts e->T monotonically and convergently over a representative (rho,e)
sweep for both electrons and ions.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the tabulated 3T EOS unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("eos_table_3t_test")
    assert passed, "eos_table_3t_test reported a failing check (nonzero exit)"
