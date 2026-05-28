"""
Auto-collected wrapper for the IONMIX multigroup-opacity reader pgen-based unit test.

Builds src/pgen/unit_tests/ionmix_opacity_test.cpp
(-D PROBLEM=unit_tests/ionmix_opacity_test) into the isolated tst/build_unit directory,
runs it against the committed IONMIX fixture table, and asserts it exited 0 (all checks
passed). The test verifies that the IONMIX opacity reader
(src/opacity/ionmix_opacity_reader.hpp) + the common multigroup opacity representation
(src/opacity/multigroup_opacity.hpp) (issue [16a]/#7, ADR-0007): parse the group
structure (count + photon-energy boundaries) from the table; reproduce the known per-group
Planck-absorption / Planck-emission / Rosseland mass-opacity entries; interpolate in
(rho, T_e) against reference off-grid points; and clamp out-of-range (rho, T_e) queries to
the table edges (no out-of-bounds lookups), for the synthetic power-law fixture.

The fixture path (repo-relative) is resolved to an absolute path here and passed to the
binary via the `problem/opacity_file=<abspath>` command-line override, because the test
runs from tst/build*/src where a repo-relative path would not resolve.

Oracle: Layer 1 -- analytic.  Against a synthetic power-law IONMIX fixture, the reader
reproduces the group structure and the known per-group Planck-absorption / Planck-emission
/ Rosseland mass opacities, interpolates (rho, T_e) at off-grid reference points, and
clamps out-of-range queries to the table edges.  References: closed form (log-log bilinear
on the synthetic table).
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the IONMIX opacity reader unit test; pass iff it exits 0."""
    fixture = os.path.join(
        testutils._repo_root(), "inputs", "unit_tests", "ionmix_opacity_test.cn4"
    )
    passed = testutils.run_unit_test(
        "ionmix_opacity_test",
        args=[f"problem/opacity_file={fixture}"],
    )
    assert passed, "ionmix_opacity_test reported a failing check (nonzero exit)"
