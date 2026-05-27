"""
Auto-collected wrapper for the SESAME 3T EOS reader pgen-based unit test.

Builds src/pgen/unit_tests/sesame_eos_test.cpp (-D PROBLEM=unit_tests/sesame_eos_test)
into the isolated tst/build_unit directory and exercises the SESAME EOS reader
(src/eos/sesame_eos_reader.hpp) + the common 3T EOS representation
(src/eos/eos_table_3t.hpp) (issue [13c]/#11, ADR-0007):

  test_run               -- against the 3T-native fixture (electron table 304 + ion
                            table 305 + ionization table 601), asserts the binary exits 0
                            (all checks pass): the reader parses the grid, populates the
                            representation's per-species electron/ion specific energies /
                            pressures / mean ionization, derives the heat capacities
                            (c_v = de/dT), the (rho,T) interpolation reproduces the
                            synthetic laws, the e->T inversion the representation feeds
                            round-trips, and out-of-range queries clamp to the edges.
  test_rejects_1t        -- against a 1T-only material (only a total table 301), asserts
                            the binary exits NONZERO: a non-3T material must be rejected
                            with a clear error (ADR-0007, no 1T->2T split modelling).

The fixture paths (repo-relative) are resolved to absolute paths here and passed to the
binary via the `problem/eos_file=<abspath>` command-line override, because the test runs
from tst/build*/src where a repo-relative path would not resolve.
"""

# Modules
import os

import test_suite.testutils as testutils


def _fixture(name):
    return os.path.join(testutils._repo_root(), "inputs", "unit_tests", name)


def test_run():
    """Build + run the SESAME EOS reader unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test(
        "sesame_eos_test",
        args=[f"problem/eos_file={_fixture('sesame_eos_test.ses')}"],
    )
    assert passed, "sesame_eos_test reported a failing check (nonzero exit)"


def test_rejects_1t():
    """A 1T-only (total-table-only) material must be rejected: the reader FATALs."""
    passed = testutils.run_unit_test(
        "sesame_eos_test",
        args=[f"problem/eos_file={_fixture('sesame_eos_test_1t.ses')}"],
    )
    assert not passed, "1T-only SESAME material was NOT rejected (binary exited 0)"
