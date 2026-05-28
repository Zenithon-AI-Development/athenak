"""
MagLIF / Z-pinch problem-generator smoke + IC test (ADR-0004/0005, issue [10a]/#32).

Exercises the three acceptance criteria of the integrated MagLIF setup pgen:

  1. liner/fuel/vacuum initial conditions on a cylindrical (r,phi,z) grid with a uniform
     axial B_z premagnetization field;
  2. single-mode AND multi-mode (surface-roughness) axial perturbation seeding of the
     liner interface;
  3. the prescribed-I(t) circuit drive is wired in -- a basic driven run launches and is
     stable (runs to completion with everything finite, the driven B_phi appears in the
     vacuum, and the magnetic piston starts the implosion).

The pgen is a USER pgen (built with -D PROBLEM=maglif), so it gets its own per-problem
binary; outputs land in testutils.pgen_run_dir("maglif").  Auto-collected by
run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import glob
import os
import sys
import shutil
import numpy as np
import test_suite.testutils as testutils

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

input_file = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif.athinput")

# Values that must match inputs/maglif.athinput.
MU0, I0, BZ0 = 1.0, 0.5, 0.1
D_FUEL, D_LINER, D_VAC = 0.05, 1.0, 0.01
PERT_AMP = 0.05


def _liner_radius_vs_z(data):
    """Dense-liner mass-weighted mean radius as a function of z (averaged over phi).

    A continuous, FP-robust estimator of where the imploding shell sits at each axial
    height -- it tracks the z-dependent interface displacement dr(z) the perturbation
    seeds.  Returns an array of length nz.
    """
    r = np.asarray(data["x1v"], dtype=float)            # (nr,)
    dens = np.asarray(data["dens"], dtype=float)        # (nz, nphi, nr)
    nz = dens.shape[0]
    rc = np.empty(nz)
    for kz in range(nz):
        sl = dens[kz]                                   # (nphi, nr)
        w = np.where(sl >= 0.5 * D_LINER, sl, 0.0)      # dense-liner cells only
        rr = np.broadcast_to(r, sl.shape)
        rc[kz] = np.sum(w * rr) / np.sum(w)
    return rc


def test_maglif_single_mode_and_drive():
    """Single-mode 3D driven run: verify the ICs, B_z premag, drive, and stability."""
    try:
        assert testutils.run_pgen("maglif", input_file), "MagLIF driven run failed."

        run_dir = testutils.pgen_run_dir("maglif")
        snaps = sorted(glob.glob(os.path.join(run_dir, "bin", "maglif.prim.*.bin")))
        assert len(snaps) >= 2, f"expected >=2 output snapshots, found {len(snaps)}"

        # ---- (1)+(2) initial conditions (t=0 snapshot) ----
        ic = bin_convert.read_binary_as_athdf(snaps[0])
        r = np.asarray(ic["x1v"], dtype=float)
        dens = np.asarray(ic["dens"], dtype=float)
        bcc1 = np.asarray(ic["bcc1"], dtype=float)      # B_r
        bcc2 = np.asarray(ic["bcc2"], dtype=float)      # B_phi
        bcc3 = np.asarray(ic["bcc3"], dtype=float)      # B_z

        # three distinct density regions (fuel | liner | vacuum) are present.
        for label, val in (("fuel", D_FUEL), ("liner", D_LINER), ("vacuum", D_VAC)):
            assert np.any(np.isclose(dens, val, rtol=1e-6)), f"{label} density missing"

        # axial B_z premagnetization is uniform and equal to bz0; B_r is identically zero.
        assert np.allclose(bcc3, BZ0, atol=1e-12), "B_z premag not uniform == bz0"
        assert np.allclose(bcc1, 0.0, atol=1e-12), "B_r should be zero initially"

        # the driven B_phi = mu0*I0/(2*pi*r) is present in the vacuum (drive wired in).
        ridx = int(np.argmin(np.abs(r - 0.9)))          # a radius well inside the vacuum
        bphi_expected = MU0 * I0 / (2.0 * np.pi * r[ridx])
        bphi_meas = float(np.mean(bcc2[:, :, ridx]))
        assert np.isclose(bphi_meas, bphi_expected, rtol=0.02), (
            f"driven B_phi(r=0.9)={bphi_meas:g} != {bphi_expected:g}"
        )

        # single-mode (n=1) axial perturbation: the liner radius varies with z by ~2*amp
        # and the profile is symmetric about the axial midplane (a clean cosine).
        rc = _liner_radius_vs_z(ic)
        assert np.ptp(rc) > 1.2 * PERT_AMP, (
            f"single-mode interface barely varies with z (ptp={np.ptp(rc):g})"
        )
        mirror_resid = np.max(np.abs(rc - rc[::-1]))
        assert mirror_resid < 0.2 * np.ptp(rc), (
            f"single-mode profile not z-mirror-symmetric (resid={mirror_resid:g})"
        )

        # ---- (3) launches and is stable (final snapshot) ----
        fin = bin_convert.read_binary_as_athdf(snaps[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"
        # the magnetic piston has started the implosion -> inward radial velocity.
        assert float(np.asarray(fin["velx"]).min()) < 0.0, "no inward implosion velocity"
    finally:
        shutil.rmtree(
            os.path.join(testutils.pgen_run_dir("maglif"), "bin"), ignore_errors=True
        )


def test_maglif_roughness_seeding():
    """Roughness (multi-mode) seeding: a band of axial modes breaks the single-mode
    symmetry, confirming the surface-roughness option lays down a distinct profile."""
    try:
        args = [
            "time/nlim=0",                   # IC only -- just check the seeded surface
            "problem/perturbation=roughness",
            "problem/pert_amp=0.06",
            "problem/pert_nmin=2",
            "problem/pert_nmax=6",
            "problem/pert_seed=7",
            "job/basename=maglif_rough",
        ]
        ok = testutils.run_pgen("maglif", input_file, args=args)
        assert ok, "roughness run failed."

        snap = os.path.join(
            testutils.pgen_run_dir("maglif"), "bin", "maglif_rough.prim.00000.bin"
        )
        ic = bin_convert.read_binary_as_athdf(snap)
        rc = _liner_radius_vs_z(ic)

        # the rough surface displaces the interface with z ...
        assert np.ptp(rc) > 0.02, f"roughness interface too flat (ptp={np.ptp(rc):g})"
        # ... and is NOT the symmetric single-mode cosine (multi-mode, random phases).
        mirror_resid = np.max(np.abs(rc - rc[::-1]))
        assert mirror_resid > 0.2 * np.ptp(rc), (
            f"roughness profile looks single-mode-symmetric (resid={mirror_resid:g})"
        )
    finally:
        shutil.rmtree(
            os.path.join(testutils.pgen_run_dir("maglif"), "bin"), ignore_errors=True
        )
