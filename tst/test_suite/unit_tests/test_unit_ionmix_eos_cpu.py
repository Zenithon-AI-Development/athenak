"""
Auto-collected wrapper for the IONMIX 3T EOS reader pgen-based unit test.

Builds src/pgen/unit_tests/ionmix_eos_test.cpp (-D PROBLEM=unit_tests/ionmix_eos_test)
into the isolated tst/build_unit directory and exercises the IONMIX EOS reader
(src/eos/ionmix_eos_reader.hpp) + the common 3T EOS representation
(src/eos/eos_table_3t.hpp) (issue [13b]/#10, ADR-0007):

  test_run               -- against the 3T-native fixture, asserts the binary exits 0
                            (all checks pass): the reader parses the grid, populates the
                            representation's per-species electron/ion specific energies /
                            pressures / heat capacities / mean ionization, the (rho,T)
                            interpolation reproduces the synthetic laws, the e->T
                            inversion
                            the representation feeds round-trips, and out-of-range queries
                            clamp to the table edges.
  test_rejects_non3t     -- against a 1T-only (nspec=1) fixture, asserts the binary exits
                            NONZERO: a non-3T table must be rejected with a clear error
                            (ADR-0007, no 1T->2T split modelling).

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
    """Build + run the IONMIX EOS reader unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test(
        "ionmix_eos_test",
        args=[f"problem/eos_file={_fixture('ionmix_eos_test.cn4')}"],
    )
    assert passed, "ionmix_eos_test reported a failing check (nonzero exit)"


def test_rejects_non3t():
    """A non-3T (1T-only) table must be rejected: the reader FATALs (nonzero exit)."""
    passed = testutils.run_unit_test(
        "ionmix_eos_test",
        args=[f"problem/eos_file={_fixture('ionmix_eos_test_non3t.cn4')}"],
    )
    assert not passed, "non-3T EOS table was NOT rejected (binary exited 0)"
