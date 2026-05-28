"""
Converging single-mode Richtmyer-Meshkov verification (FLASH MagLIF benchmark 3, #36).

Runs the integrated MagLIF/Z-pinch problem generator (issue [10a]/#32) in a converging
configuration -- a dense cylindrical liner driven radially inward by a prescribed constant
load current, with a single axial mode seeded on the liner interface.  As the magnetic
piston converges the shell, the seeded perturbation grows (the shock-driven
Richtmyer-Meshkov feed plus Bell-Plesset convergence amplification); the radiography
observable is the growth of that single-mode amplitude.  This is benchmark 3 of the FLASH
MagLIF validation ladder (Ellison et al. 2025, arXiv:2504.10760).

Physics scope (honest, per the #32 MagLIF-pgen scope note + the PRD Phase-1 sequencing):
the grey-FLD radiation, anisotropic-conduction and variable-resistivity operators are
operator-split modules that are NOT yet wired into the production MHD driver, so this
benchmark exercises the WIRED stack -- cylindrical ideal-MHD (ADR-0004) + the prescribed-
I(t) circuit drive (ADR-0005 mode A).  The radiation coupling is verified separately
(#28 coupled grey rad-hydro, #41 multigroup).  The converging-RM growth measured here is
the adiabatic-MHD baseline; radiation is layered in once the driver wiring lands.

Discriminating quantities (see inputs/maglif_rm.athinput):
  * R_liner(t)  -- mass-weighted mean liner radius: the bulk convergence trajectory.
  * R_front(t)  -- inner (fuel-side) half-max interface: the leading imploding front.
  * a_outer(t)  -- single-mode amplitude on the driven (liner/vacuum) interface: this is
                   the magnetically-driven, MRT/RM-unstable surface -- it grows.
  * a_inner(t)  -- single-mode amplitude on the fuel-side interface: RT-stabilised during
                   the run-in (it does NOT run away), the expected MagLIF asymmetry.

The test runs the implosion, checks that (1) the liner converges, (2) the leading front
moves inward, (3) the seeded single mode grows on the driven surface and stays dominant
(no spurious mode cascade), then plots the four trajectories and saves/diffs a golden
regression baseline.  Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import glob
import os
import shutil
import sys

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_rm.athinput.
PERT_MODE = 2          # seeded axial mode number (wavelength = Lz/pert_mode)
PERT_AMP = 0.05        # seeded radial interface amplitude
D_LINER = 1.0          # liner (dense shell) density
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

# Verification thresholds.
CONV_FRAC = 0.75       # require >= 25% bulk convergence (R_liner shrinks)
GROWTH_MIN = 1.5       # require >= 1.5x single-mode growth on the driven surface
DOM_FRAC = 0.6         # seeded mode must hold >= 60% of the interface z-variance

input_file = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif_rm.athinput")
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")


def _cross(r, p, i, j):
    """Sub-cell radius where the density profile p crosses HALF between cells i and j."""
    f = (HALF - p[i]) / (p[j] - p[i])
    return r[i] + f * (r[j] - r[i])


def _interfaces(data):
    """Per-z liner interfaces (phi-averaged; the axisymmetric IC makes phi a replica).

    For each axial height z, reduce to the radial density profile and return the inner
    and outer half-max crossings of the dense-liner slab (continuous, FP-robust) plus the
    mass-weighted mean liner radius (rho*r dr weighting).  Each is an array of length nz.
    """
    r = np.asarray(data["x1v"], dtype=float)            # (nr,)
    dens = np.asarray(data["dens"], dtype=float)        # (nz, nphi, nr)
    nz = dens.shape[0]
    r_in = np.full(nz, np.nan)
    r_out = np.full(nz, np.nan)
    r_mw = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)                       # (nr,) phi-averaged
        dense = p >= HALF
        idx = np.where(dense)[0]
        if len(idx) == 0:
            continue
        w = np.where(dense, p, 0.0)
        r_mw[kz] = np.sum(w * r * r) / np.sum(w * r)
        lo, hi = idx[0], idx[-1]
        r_in[kz] = r[lo] if lo == 0 else _cross(r, p, lo - 1, lo)
        r_out[kz] = r[hi] if hi == len(r) - 1 else _cross(r, p, hi, hi + 1)
    return r_in, r_out, r_mw


def _mode_amp(rz, z, k):
    """Magnitude of the discrete cos/sin projection of r(z) onto axial mode k.

    Magnitude (not signed cosine) so the measure is robust to any phase drift of the
    growing mode.  z is the axial coordinate on [0, Lz=1].
    """
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    zz = z[good]
    a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz))
    b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz))
    return float(np.hypot(a, b))


def test_verify_maglif_rm():
    """Run the converging RM implosion and verify convergence + single-mode growth."""
    try:
        assert testutils.run_pgen("maglif", input_file), "MagLIF RM run failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_rm.prim.*.bin")))
        assert len(files) > 8, f"too few RM snapshots: {len(files)}"

        z = None
        times, r_liner, r_front, a_outer, a_inner = [], [], [], [], []
        for fp in files:
            d = bin_convert.read_binary_as_athdf(fp)
            if z is None:
                z = np.asarray(d["x3v"], dtype=float)
            r_in, r_out, r_mw = _interfaces(d)
            times.append(float(d["Time"]))
            r_liner.append(float(np.nanmean(r_mw)))
            r_front.append(float(np.nanmean(r_in)))
            a_outer.append(_mode_amp(r_out, z, PERT_MODE))
            a_inner.append(_mode_amp(r_in, z, PERT_MODE))
        times = np.array(times)
        r_liner = np.array(r_liner)
        r_front = np.array(r_front)
        a_outer = np.array(a_outer)
        a_inner = np.array(a_inner)

        # Final snapshot is finite and stable (the implosion ran in cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # (1) The liner actually converges (bulk implosion, not a static/expanding shell).
        assert r_liner[-1] < CONV_FRAC * r_liner[0], (
            f"liner did not converge: R0={r_liner[0]:.4f} -> R={r_liner[-1]:.4f}"
        )

        # (2) The leading (fuel-side) front moves inward as the shock/liner converge.
        assert r_front[-1] < r_front[0], (
            f"leading front did not move inward: {r_front[0]:.4f} -> {r_front[-1]:.4f}"
        )

        # (3) The seeded single mode grows on the magnetically-driven (outer) interface --
        # the radiography growth signature -- starting from ~the seeded amplitude.
        assert abs(a_outer[0] - PERT_AMP) < 0.25 * PERT_AMP, (
            f"initial driven-surface amplitude {a_outer[0]:.4f} != seed {PERT_AMP}"
        )
        assert a_outer[-1] > GROWTH_MIN * a_outer[0], (
            f"single-mode did not grow: a_out {a_outer[0]:.4f} -> {a_outer[-1]:.4f} "
            f"(need > {GROWTH_MIN}x)"
        )

        # (4) The seeded mode stays dominant on the driven interface (no spurious
        # cascade): its modal power is the majority of the z-variance at stagnation.
        r_out_fin = _interfaces(fin)[1]
        ro = r_out_fin[np.isfinite(r_out_fin)]
        var_tot = float(np.mean((ro - ro.mean()) ** 2))
        seed_pow = 0.5 * a_outer[-1] ** 2
        dom = seed_pow / var_tot
        assert dom > DOM_FRAC, (
            f"seeded mode not dominant at stagnation: {dom:.2f} < {DOM_FRAC}"
        )

        harness.verify(
            "maglif_rm",
            times,
            {
                "R_liner": r_liner,
                "R_front": r_front,
                "a_outer": a_outer,
                "a_inner": a_inner,
            },
            coord_label="t",
            xlabel="t",
            title="Converging single-mode RM (MagLIF benchmark 3): convergence + growth",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
