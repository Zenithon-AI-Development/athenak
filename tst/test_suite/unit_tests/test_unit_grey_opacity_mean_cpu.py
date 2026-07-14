"""
Auto-collected wrapper for the grey opacity-mean pgen-based unit test (#204).

Builds src/pgen/unit_tests/grey_opacity_mean_test.cpp
(-D PROBLEM=unit_tests/grey_opacity_mean_test) into the isolated tst/build_unit
directory, runs it against the committed IONMIX power-law fixture table, and asserts it
exited 0 (all checks passed).  The test verifies the grey (frequency-integrated)
reduction of the tabulated multigroup opacity (radiation_fld/grey_opacity_mean.hpp): the
dB/dT-weighted harmonic Rosseland mean over the photon-energy groups -- the per-cell
chi(rho,Te) lookup that lets the grey FLD operator treat the dense liner as opaque and
the tenuous vacuum gap as transparent (#204), replacing the frozen density-independent
constant.

Oracle: Layer 1 -- analytic / independent quadrature.  Against the synthetic power-law
IONMIX fixture the grey mean matches an independent host Simpson quadrature of the
weighted harmonic mean, is bounded by the per-group extremes, collapses to the
lowest/highest group's kappa for cold/hot spectra, inherits the fixture's exact rho^0.6
scaling, and its group-weighting factor rises monotonically with Te.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the grey opacity-mean unit test; pass iff it exits 0."""
    fixture = os.path.join(
        testutils._repo_root(), "inputs", "unit_tests", "ionmix_opacity_test.cn4"
    )
    passed = testutils.run_unit_test(
        "grey_opacity_mean_test",
        args=[f"problem/opacity_file={fixture}"],
    )
    assert passed, "grey_opacity_mean_test reported a failing check (nonzero exit)"
