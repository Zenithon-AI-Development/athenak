"""
Faithful tabulated-3T B1 MagLIF initial-conditions + drive-calibration CPU verification
(issue [P5]/#174, ADR-0010).

Runs the faithful single-mode MRT setup (inputs/maglif_b1_sinars.athinput) -- now
``eos = tabulated_3t`` with the real aluminum IONMIX table, the SI-calibrated z2173 drive,
and the ``dt_max`` cold-start cap -- at REDUCED resolution on CPU.  This is the cheap CPU
gate that exercises the exact new #174 code paths (the load-time table scaling, the
tabulated IC ``else`` branch that fills the conserved energy + the ``e_ele`` passive
scalar, and the ns/MA->code drive conversion); the paper-resolution quantitative GPU
replication is test_verify_maglif_b1_sinars_gpu.py.

Binding checks (the issue's acceptance criteria, all CPU-cheap):
  * TABULATED IC FILLED (AC#1 red->green): the t=0 ``e_ele/rho`` passive scalar (s_00) is
    strictly positive in the liner.  Before the #174 ``else`` branch the tabulated energy
    + e_ele scalar were left uninitialised (energy was filled only ``if (is_ideal)``), so
    this is exactly the "uninitialised tabulated energy" failure the branch fixes.
  * IMPLOSION RUNS IN (not a degenerate single step): the dense liner's mass-weighted
    radius shrinks over the run -- proving the SI-calibrated magnetic drive does real work
    and the ``dt_max`` cap let the integrator step through the field-free cold start.
  * 1-D STAYS 1-D (AC#4): the same drive+geometry with ``pert_amp=0`` develops no axial
    structure (the outer interface r(z) stays flat), proving any axial growth in the
    seeded run comes from the seed, not the numerics.
  * COARSE TWO-GRID CONVERGENCE (AC#4): the 1-D control's bulk convergence ratio agrees
    between two resolutions within a loose band (a sanity check, not a convergence-order
    measurement).

CPU-only (``_cpu`` suffix); excluded from the GPU job.  The maglif build dir is shared
with the _gpu tests, so a CUDA-configured dir is cleared to rebuild as a Serial binary.
"""

# Modules
import glob
import os
import shutil

import numpy as np
import test_suite.testutils as testutils

import sys

sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

D_LINER = 1.0          # liner density in code units (2.7 g/cc / density_cgs 2.7)
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_b1_sinars.athinput"
)
table_file = os.path.join(
    testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
)
trace_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "z2173_current.dat"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")
build_dir = os.path.join(testutils._repo_root(), "tst", "build_pgen", "maglif")


def _build_dir_is_cuda():
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, "r") as f:
        return "Kokkos_ENABLE_CUDA:BOOL=ON" in f.read().upper()


def _run(basename, nx1, nx3, pert_amp=None, tlim=90.0):
    """Run the faithful tabulated B1 at reduced resolution; return snapshot paths."""
    args = [
        f"mhd/eos_table={table_file}",
        f"problem/current_file={trace_file}",
        f"job/basename={basename}",
        f"mesh/nx1={nx1}", "mesh/nx2=4", f"mesh/nx3={nx3}",
        f"meshblock/nx1={nx1}", "meshblock/nx2=4", f"meshblock/nx3={nx3}",
        f"time/tlim={tlim}", "output1/dt=15.0", "output2/dt=1000",
    ]
    if pert_amp is not None:
        args.append(f"problem/pert_amp={pert_amp}")
    ok = testutils.run_pgen("maglif", input_file, args=args)
    assert ok, f"tabulated B1 reduced CPU run failed (basename={basename})"
    files = sorted(glob.glob(os.path.join(bin_dir, f"{basename}.prim.*.bin")))
    assert len(files) >= 3, f"too few snapshots for {basename}: {len(files)}"
    return files


def _liner_radius(d):
    """Mass-weighted mean radius of the dense liner (phi+z averaged half-max material)."""
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)          # (nz, nphi, nr)
    p = dens.mean(axis=(0, 1))                          # (nr,) phi+z averaged profile
    w = np.where(p >= HALF, p, 0.0)
    return float(np.sum(w * r) / np.sum(w)) if w.sum() > 0 else np.nan


def _outer_interface_z(d):
    """Per-z outer half-max interface r_out(z) (len nz); std measures axial waviness."""
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)          # (nz, nphi, nr)
    nz = dens.shape[0]
    r_out = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)                      # (nr,)
        idx = np.where(p >= HALF)[0]
        if len(idx):
            r_out[kz] = r[idx[-1]]
    return r_out


def test_verify_maglif_b1_tabulated_cpu():
    """Reduced-resolution CPU gate for the #174 tabulated-IC + SI-drive wiring."""
    if _build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        # ---- (1) seeded faithful run: IC fill + implosion runs in + axial structure ----
        sf = _run("b1_tab", 72, 16)
        d0 = bin_convert.read_binary_as_athdf(sf[0])
        dN = bin_convert.read_binary_as_athdf(sf[-1])

        # tabulated IC filled the e_ele/rho scalar (s_00) in the liner (#174 else-branch)
        dens0 = np.asarray(d0["dens"], dtype=float)
        s00 = np.asarray(d0["s_00"], dtype=float)
        liner = dens0 > HALF
        assert liner.any(), "no dense liner cells at t=0"
        assert s00[liner].min() > 0.0, (
            f"tabulated IC did not fill the e_ele scalar (min s_00 in liner "
            f"{s00[liner].min():.3e} <= 0): the energy/e_ele fill is uninitialised"
        )
        # finite, stable end state
        for v in ("dens", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3", "s_00"):
            assert np.all(np.isfinite(np.asarray(dN[v]))), f"{v} non-finite (unstable)"

        # the SI-calibrated drive imploded the liner (mass-weighted radius shrinks)
        r0 = _liner_radius(d0)
        rN = _liner_radius(dN)
        assert rN < r0, f"liner did not converge: R0={r0:.4f} -> R={rN:.4f}"

        # ---- (2) 1-D control: same drive+geometry, no seed -> stays axisymmetric ----
        # (the seeded-mode axial GROWTH curve is the paper-resolution GPU benchmark's
        # quantitative observable; the 20 um seed is sub-cell on this coarse CPU grid, so
        # only the 1-D-stays-1-D direction of AC#4 is asserted here.)
        cf = _run("b1_tab_ctl", 72, 16, pert_amp=0.0)
        dc0 = bin_convert.read_binary_as_athdf(cf[0])
        dcN = bin_convert.read_binary_as_athdf(cf[-1])
        ro_ctl = _outer_interface_z(dcN)
        std_ctl = float(np.nanstd(ro_ctl))
        assert std_ctl < 1.0e-6, (
            f"pert_amp=0 control developed axial structure (std r_out(z)={std_ctl:.3e}); "
            f"not 1-D without the seed"
        )
        # the control still implodes (the drive is seed-independent)
        rc0 = _liner_radius(dc0)
        rcN = _liner_radius(dcN)
        assert rcN < rc0, f"control liner did not converge: {rc0:.4f} -> {rcN:.4f}"

        # ---- (3) coarse two-grid convergence sanity: coarser control grid agrees ----
        cf2 = _run("b1_tab_ctl2", 48, 12, pert_amp=0.0)
        dc20 = bin_convert.read_binary_as_athdf(cf2[0])
        dc2N = bin_convert.read_binary_as_athdf(cf2[-1])
        conv_fine = rcN / rc0
        conv_coarse = _liner_radius(dc2N) / _liner_radius(dc20)
        assert abs(conv_fine - conv_coarse) < 0.15, (
            f"coarse two-grid convergence ratio disagrees: fine {conv_fine:.3f} vs "
            f"coarse {conv_coarse:.3f} (|diff| >= 0.15)"
        )
    finally:
        for pat in ("b1_tab.*", "b1_tab_ctl.*", "b1_tab_ctl2.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        shutil.rmtree(bin_dir, ignore_errors=True)
