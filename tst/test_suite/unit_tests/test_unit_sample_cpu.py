"""
Auto-collected wrapper for the sample pgen-based unit test.

Builds src/pgen/unit_tests/sample_unit_test.cpp (-D PROBLEM=unit_tests/sample_unit_test)
into the isolated tst/build_unit directory, runs it, and asserts it exited 0 (all checks
passed). The unit-test binary itself reports per-check failures and exits nonzero on any
failure -- see src/pgen/unit_tests/unit_test.hpp.

This file demonstrates the auto-collection contract: dropping a
test_unit_<name>_<device>.py here is enough; no edit to run_test_suite.py is needed.

Oracle: n/a (not a physics test).  Scaffold for the auto-collection contract;
ApproxEqual self-checks vs hardcoded values, with no independent oracle -- it verifies
the harness wiring, not a physical result.  Excluded from the Layer-1/Layer-2 provenance
classification in docs/verification.md.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the sample unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("sample_unit_test")
    assert passed, "sample_unit_test reported a failing check (nonzero exit)"
