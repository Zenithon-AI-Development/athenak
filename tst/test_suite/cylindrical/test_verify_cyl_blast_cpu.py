"""
Cylindrical blast-wave sphericity verification (ADR-0004, issue #15).

A blast wave is initialized as a circular over-pressured region in PHYSICAL space,
centered OFF the axis at (r,phi)=(1.5,0), and evolved on a 2D (r,phi) polar grid.  On a
correct curvilinear scheme the shock front must stay CIRCULAR in physical space: its
radius (measured from the blast center) is the same in all directions.  The test:

  * runs the 2D cylindrical blast,
  * reads the full 2D field (binary output -> bin_convert), maps each cell to physical
    (x,y), and measures the shock-front radius vs angle around the blast center,
  * asserts the azimuthal distortion (rmax-rmin)/rave is within tolerance,
  * plots shock-radius vs angle and saves/diffs a golden regression baseline.

Oracle: Layer 1 -- analytic.  The 2-D (cylindrical) Sedov-Taylor self-similar blast is
circularly symmetric -- its solution depends only on distance from the blast center, so
the exact shock front is a single-radius circle R_s(t) at every azimuth (Sedov 1959;
Taylor 1950).  Initialized OFF the axis on an (r,phi) polar grid deliberately MISALIGNED
with that physical circular symmetry, the measured azimuthal distortion
(r_max-r_min)/r_ave of the front is therefore a direct, snapshot-independent measure of
the curvilinear hydro scheme's geometric error -- exactly the correctness criterion of
the Athena/Athena++ blast-wave benchmark (Londrillo & Del Zanna 2000; Stone et al. 2008).
The binding assertion is this symmetry bound; the harness.verify baseline is a regression
guard only.

A quantitative absolute-radius Sedov-Taylor check is intentionally NOT used: this
finite-size, finite-strength blast lies outside the strong-shock self-similar regime
(post-shock density jump ~2.4 << the strong-shock limit (gamma+1)/(gamma-1)=6; ambient
pressure not negligible), so the strong-shock Sedov radius overpredicts the measured
front by ~2x and is not a valid oracle here.  Front-POSITION / radial-profile
correctness of the cylindrical scheme is established independently by the Layer-1
cyl_sod (exact Riemann) and cyl_mignone (geometric dilution) tests; this test isolates
the 2-D angular isotropy they cannot see.  The 0.12 distortion tolerance is justified by
the measured ~2-3% distortion -- a ~4-5x margin for cross-platform FP plus the polar
grid's coarser azimuthal resolution and PLM directional truncation.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os
import sys
import shutil
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# "blast" is a user pgen (built with -D PROBLEM=blast), so it needs its own binary built
# into a per-problem dir; outputs land in testutils.pgen_run_dir("blast").
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "cyl_blast.athinput"
)
binary_file = os.path.join(
    testutils.pgen_run_dir("blast"), "bin", "cyl_blast.blast.00001.bin"
)

# Blast center in code coords (must match inputs/cyl_blast.athinput).
RC, PHIC = 1.5, 0.0
RHO_AMB = 1.0           # ambient density (dn_amb in the athinput)
N_SECTORS = 12          # azimuthal sectors around the blast center
D_LO, D_HI = 0.08, 0.45  # radial band (from centre) in which to locate the shock front


def test_verify_cyl_blast():
    """Run the 2D cylindrical blast and verify shock-front sphericity + baseline."""
    try:
        assert testutils.run_pgen("blast", input_file), "Cylindrical blast run failed."
        data = bin_convert.read_binary_as_athdf(binary_file)

        r = np.asarray(data["x1v"], dtype=float)            # (nx1,)
        phi = np.asarray(data["x2v"], dtype=float)          # (nx2,)
        dens = np.asarray(data["dens"], dtype=float)[0]     # (nx2, nx1)

        # Map every cell to physical (x,y) and to distance/angle from the blast center.
        rr, pp = np.meshgrid(r, phi)                        # (nx2, nx1)
        x = rr * np.cos(pp)
        y = rr * np.sin(pp)
        xc, yc = RC * np.cos(PHIC), RC * np.sin(PHIC)
        dist = np.sqrt((x - xc) ** 2 + (y - yc) ** 2).ravel()
        ang = np.arctan2(y - yc, x - xc).ravel()
        rho = dens.ravel()

        # Within the shock band the blast's dense shell carries the excess density
        # (rho - rho_amb).  The shell radius in each angular sector is the
        # excess-density-weighted mean distance from the center -- a continuous estimator
        # (robust to grid/round-off, unlike a discrete argmax) of where the shock front
        # sits.  A circular shock gives the same shell radius at every angle.
        band = (dist >= D_LO) & (dist <= D_HI)
        excess = np.clip(rho - RHO_AMB, 0.0, None)
        edges = np.linspace(-np.pi, np.pi, N_SECTORS + 1)
        centers, radii = [], []
        for s in range(N_SECTORS):
            sel = band & (ang >= edges[s]) & (ang < edges[s + 1])
            w = excess[sel]
            if np.count_nonzero(sel) < 5 or w.sum() <= 0.0:
                continue
            centers.append(0.5 * (edges[s] + edges[s + 1]))
            radii.append(float(np.sum(w * dist[sel]) / np.sum(w)))
        radii = np.asarray(radii)
        centers = np.asarray(centers)

        assert radii.size >= N_SECTORS - 1, (
            f"too few populated sectors ({radii.size}); blast may have left the domain"
        )
        rave = float(np.mean(radii))
        distortion = float((radii.max() - radii.min()) / rave)
        # BINDING ORACLE (see module docstring): the self-similar blast front is exactly
        # circular, so its azimuthal distortion measures the curvilinear scheme's
        # geometric error.  A 2nd-order scheme on a polar grid keeps an off-axis blast
        # circular to a few % (measured ~2-3%); 0.12 leaves a cross-platform-FP margin.
        assert distortion < 0.12, (
            f"cyl blast shock-front distortion too large: {distortion:g} "
            f"(rave={rave:g}, rmin={radii.min():g}, rmax={radii.max():g})"
        )

        harness.verify(
            "cyl_blast",
            np.degrees(centers),
            {"shock_radius": radii},
            coord_label="angle_deg",
            xlabel="angle around blast center [deg]",
            title=f"Cylindrical blast sphericity (distortion={distortion:.3f})",
            rtol=1.0e-4,
            atol=1.0e-6,
        )
    finally:
        shutil.rmtree(
            os.path.join(testutils.pgen_run_dir("blast"), "bin"), ignore_errors=True
        )
