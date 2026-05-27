"""
Auto-collected wrapper for the Braginskii transport-coefficient pgen-based unit test.

Builds src/pgen/unit_tests/braginskii_transport_test.cpp
(-D PROBLEM=unit_tests/braginskii_transport_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed). The test verifies that
the magnetized Braginskii transport coefficients in diffusion/braginskii_transport.hpp
reproduce the Braginskii reference values (collision times, parallel kappa/eta
coefficients, Spitzer parallel resistivity), the unmagnetized (x->0) and
strongly-magnetized limits, and that the Bohm-like anomalous term and the
vacuum-resistivity floor activate as specified (ADR-0003/0006, issue [5]).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the Braginskii-transport unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("braginskii_transport_test")
    assert passed, "braginskii_transport_test reported a failing check (nonzero exit)"
