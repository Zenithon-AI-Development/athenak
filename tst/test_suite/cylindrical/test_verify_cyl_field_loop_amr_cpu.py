"""
Cylindrical (r,phi) field-loop div(B) across an AMR refinement boundary (issue #38).

A divergence-free magnetic field loop (a circle in PHYSICAL (x,y) laid down from a
z-directed vector potential via the *cylindrical* discrete curl, so div(B)=0 under the
finite-volume operator) is placed on an ADAPTIVE 2D (r,phi) polar grid.  A `location`
refinement criterion refines the single root MeshBlock straddling the loop, so a fine
block is created from its coarse parent DURING the run by RefineFC's face-B prolongation.
See inputs/cyl_field_loop_amr.athinput.

This is the acceptance test for the cylindrical Balsara (divergence-preserving)
prolongation: with the legacy Cartesian internal-face prolongation the prolonged fine
field is NOT div-free under the cylindrical div(B) operator (an O(B*dr/r) residual here),
so max|div(B)| jumps FAR above round-off when the fine block appears (the red state); the
cylindrical flux-form prolongation (Phi = Area*B is metric-agnostic for the Toth & Roe
construction) makes every fine cell div-free to round-off (the green state).  Constrained
transport then preserves that div(B) on the active faces, so a post-refinement history row
faithfully probes the prolongation.

div(B) is read from a user history output (max|div(B)| over all active cells, with the
MeshBlock count) rather than a gridded dump, because history is divergence/AMR-safe.  The
test:
  * runs the adaptive loop (cyl_field_loop is a user pgen -> run_pgen),
  * ASSERTS a refinement actually happened (block count rises above the 4 root blocks) --
    otherwise prolongation was never exercised and round-off div(B) would be a false pass,
  * asserts max|div(B)| stays at round-off on every history row, incl. after refinement.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).  Serial (_cpu): the
rank reduction is the identity, so "max-divb" is a true global maximum.

Oracle: Layer 1 -- analytic.  The binding correctness assertion is the conservation
identity div(B)=0: a finite-volume div-free coarse field prolonged by the cylindrical
divergence-preserving (Balsara) operator yields fine cells whose cylindrical finite-volume
div(B) is zero to round-off (max|div(B)| < 1e-10), measured by the same cyl-aware
finite-volume divergence as the mhd_divb derived variable.  References: Balsara, JCP 174,
614 (2001) (divergence-free AMR prolongation); Toth & Roe, JCP 180, 736 (2002); ADR-0004.
"""

# Modules
import os
import glob
import shutil
import numpy as np
import test_suite.testutils as testutils
import athena_read

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "cyl_field_loop_amr.athinput"
)
run_dir = testutils.pgen_run_dir("cyl_field_loop")

# div(B) at round-off: the field amplitude is ~1e-3, so a real prolongation divergence bug
# shows up at ~1e-4; round-off div(B) is ~1e-14.  1e-10 cleanly separates the two.
DIVB_TOL = 1.0e-10
NROOT_MB = 4  # 2x2 root MeshBlocks; refinement pushes the count above this


def _hst_path():
    files = glob.glob(os.path.join(run_dir, "*.user.hst"))
    assert files, f"no *.user.hst history output found in {run_dir}"
    return files[0]


def test_verify_cyl_field_loop_amr():
    """Run the adaptive cylindrical field loop; verify div(B) round-off post-refine."""
    try:
        assert testutils.run_pgen("cyl_field_loop", input_file), (
            "cyl_field_loop (AMR) run failed."
        )

        hst = athena_read.hst(_hst_path())
        max_divb = np.asarray(hst["max-divb"], dtype=float)
        nmb = np.asarray(hst["nmb"], dtype=float)

        # (1) refinement actually happened: the MeshBlock count rose above the root grid.
        # Guards against a false pass where no fine block was ever created (so RefineFC's
        # prolongation was never exercised).
        assert nmb.max() > NROOT_MB, (
            f"no refinement occurred -- AMR prolongation was not exercised "
            f"(max MeshBlock count {nmb.max():g} <= {NROOT_MB} root blocks)"
        )

        # (2) div(B) stays at round-off on every row, including the post-refinement rows
        # (CT preserves whatever div(B) the prolongation produced).
        worst = float(max_divb.max())
        assert worst < DIVB_TOL, (
            f"max|div(B)| not at round-off after cylindrical AMR refinement: "
            f"{worst:g} (>= {DIVB_TOL:g}) -- prolongation is not divergence-preserving"
        )
    finally:
        for f in glob.glob(os.path.join(run_dir, "*.user.hst")):
            try:
                os.remove(f)
            except OSError:
                pass
        shutil.rmtree(os.path.join(run_dir, "bin"), ignore_errors=True)
