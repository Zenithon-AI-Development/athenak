"""
Auto-collected wrapper for the circuit drive-source pgen-based unit test.

Builds src/pgen/unit_tests/drive_source_test.cpp
(-D PROBLEM=unit_tests/drive_source_test) into the isolated tst/build_unit directory, runs
it against the committed tabulated-waveform fixture, and asserts it exited 0 (all checks
passed). The test verifies the pluggable circuit drive source + driven B_phi boundary
formulas (src/circuit/drive_source.hpp) (issue [9a]/#21, ADR-0005): the prescribed-I(t)
waveforms (constant / linear_ramp / sin_squared) at known sample points; a tabulated
(t, I) waveform read from file (nodes, piecewise-linear interpolation, end clamping); and
the boundary formulas the user-BC writes -- driven B_phi == mu0*I/(2*pi*r) (so r*B_phi
is the constant enclosed current) and the nocurrent vacuum extrapolation d_r(r*B_phi)=0
(B_phi ~ 1/r decay), shown consistent with the driven profile.

The fixture path (repo-relative) is resolved to an absolute path here and passed to the
binary via the `problem/current_file=<abspath>` command-line override, because the test
runs from tst/build*/src where a repo-relative path would not resolve.

Oracle: Layer 1 -- analytic.  The prescribed-I(t) waveforms (constant / linear_ramp /
sin_squared) and a file-tabulated (t, I) waveform (piecewise-linear interpolation, end
clamping) are checked at known sample points, and the driven boundary obeys Ampere's law
B_phi = mu0*I/(2*pi*r) (r*B_phi = enclosed current) with the nocurrent d_r(r*B_phi)=0
vacuum extrapolation.  References: closed form (Ampere's law; prescribed waveforms).
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the drive-source unit test; pass iff it exits 0."""
    fixture = os.path.join(
        testutils._repo_root(), "inputs", "unit_tests", "drive_source_test_iwave.dat"
    )
    passed = testutils.run_unit_test(
        "drive_source_test",
        args=[f"problem/current_file={fixture}"],
    )
    assert passed, "drive_source_test reported a failing check (nonzero exit)"
