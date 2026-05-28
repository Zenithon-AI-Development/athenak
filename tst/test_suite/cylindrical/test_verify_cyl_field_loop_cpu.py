"""
Advected cylindrical (r,phi) field-loop div(B) verification (ADR-0004, issue #22).

A weak magnetic field loop -- a circle in PHYSICAL (x,y) space centred off the axis -- is
laid down from a z-directed vector potential (using the *cylindrical* discrete curl, so
the initial field is divergence-free under the finite-volume div(B) operator) and advected
by a rigid azimuthal rotation on a full-annulus 2D (r,phi) polar grid.  See
inputs/cylindrical/cyl_field_loop.athinput.  The test verifies that the cylindrical
constrained-transport / r-weighted EMF (issue #16) keeps div(B) at ROUND-OFF as the loop
is advected -- including across internal MeshBlock boundaries of the 2x2 decomposition.

The div(B) it reads is the cylindrical finite-volume divergence
(1/V)[A_{i+1/2}B_{i+1/2} - ...] from the now coordinate-aware ``mhd_divb`` derived
variable (this file's companion change in src/outputs/derived_variables.cpp); the legacy
Cartesian (B_{i+1}-B_i)/dx form would report a spurious O(B/r) "divergence" on this
genuinely div-free field.

The test:
  * runs the advected loop (cyl_field_loop is a user pgen -> run_pgen),
  * asserts max|div(B)| stays at round-off both initially and after advection,
  * plots the azimuthally-averaged field magnitude vs r and saves/diffs a golden baseline.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os
import sys
import shutil
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "cyl_field_loop.athinput"
)
bin_dir = os.path.join(testutils.pgen_run_dir("cyl_field_loop"), "bin")

# div(B) at round-off: the field amplitude is ~1e-3 and dx ~ 1e-2, so the scale of an O(1)
# divergence error would be ~0.1; round-off div(B) is ~1e-16.  1e-10 is a generous
# "round-off" gate that still fails loudly on a real CT divergence bug.
DIVB_TOL = 1.0e-10


def _max_abs_divb(tag):
    d = bin_convert.read_binary_as_athdf(
        os.path.join(bin_dir, f"cyl_field_loop.divb.{tag}.bin")
    )
    return float(np.max(np.abs(np.asarray(d["divb"], dtype=float))))


def test_verify_cyl_field_loop():
    """Run the advected cylindrical field loop and verify div(B) round-off + baseline."""
    try:
        assert testutils.run_pgen("cyl_field_loop", input_file), (
            "cyl_field_loop run failed."
        )

        # (1) div(B) is at round-off initially (cylindrical discrete curl is div-free) ...
        divb0 = _max_abs_divb("00000")
        assert divb0 < DIVB_TOL, f"initial max|div(B)| not at round-off: {divb0:g}"
        # ... and stays at round-off after advection (CT is divergence-preserving).
        divb1 = _max_abs_divb("00001")
        assert divb1 < DIVB_TOL, f"advected max|div(B)| not at round-off: {divb1:g}"

        # (2) field structure: azimuthally-averaged |B|(r) at t=tlim.  Rigid rotation
        # preserves the radial distribution, so this is a smooth, reproducible curve
        # capturing the loop's radial extent (a regression baseline for the evolution).
        b = bin_convert.read_binary_as_athdf(
            os.path.join(bin_dir, "cyl_field_loop.bcc.00001.bin")
        )
        x1v = np.asarray(b["x1v"], dtype=float)               # (nx1,)
        bcc1 = np.asarray(b["bcc1"], dtype=float)[0]          # (nx2, nx1)
        bcc2 = np.asarray(b["bcc2"], dtype=float)[0]
        bmag_rms = np.sqrt(np.mean(bcc1 ** 2 + bcc2 ** 2, axis=0))  # rms over phi

        harness.verify(
            "cyl_field_loop",
            x1v,
            {"Bmag_rms": bmag_rms},
            coord_label="x1v",
            xlabel="r",
            title=(
                f"Cylindrical field-loop |B|_rms(r), t=0.5 "
                f"(max|divB|={divb1:.1e})"
            ),
            rtol=1.0e-4,
            atol=1.0e-10,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
