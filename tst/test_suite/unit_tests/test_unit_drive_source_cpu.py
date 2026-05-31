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

It also reads the committed faithful z2173 drive trace (tst/inputs/z2173_current.dat,
issue #160/[P3]) through the same ReadCurrentWaveform path the maglif pgen uses and
asserts the published peak current (~20 MA) and rise time (~100 ns).

Oracle: Layer 1 -- analytic + published-literature.  The prescribed-I(t) waveforms
(constant / linear_ramp / sin_squared) and a file-tabulated (t, I) waveform
(piecewise-linear interpolation, end clamping) are checked at known sample points, and
the driven boundary obeys Ampere's law B_phi = mu0*I/(2*pi*r) (r*B_phi = enclosed current)
with the nocurrent d_r(r*B_phi)=0 vacuum extrapolation (closed form).  The committed z2173
trace is anchored to the published load-current characteristics of Z shot z2173
(M. R. Gomez / R. D. McBride et al., Phys. Rev. Lett. 109, 135004, 2012, Fig. 1(d)):
peak ~20 MA, rise ~100 ns.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the drive-source unit test; pass iff it exits 0."""
    repo_root = testutils._repo_root()
    fixture = os.path.join(
        repo_root, "inputs", "unit_tests", "drive_source_test_iwave.dat"
    )
    # Committed faithful z2173 Be-liner drive trace (issue #160/[P3]): the unit test reads
    # it through the same circuit::ReadCurrentWaveform path the maglif pgen uses and
    # asserts the published peak current (~20 MA) and rise time (~100 ns) (McBride 2012,
    # PRL 109, 135004, Fig. 1(d)).  Both fixtures are passed as absolute-path command-line
    # overrides because the binary runs from tst/build_unit/<name>/src where a
    # repo-relative path would not resolve.
    z2173 = os.path.join(repo_root, "tst", "inputs", "z2173_current.dat")
    passed = testutils.run_unit_test(
        "drive_source_test",
        args=[
            f"problem/current_file={fixture}",
            f"problem/z2173_file={z2173}",
        ],
    )
    assert passed, "drive_source_test reported a failing check (nonzero exit)"
