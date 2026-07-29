"""
Faithful single-mode magneto-Rayleigh-Taylor QUANTITATIVE GPU replication (Ellison
benchmark 1 / [C3], issue #120).  Paper-resolution GPU run of the FAITHFUL single-mode
MRT setup -- a dimensional aluminum liner driven by the measured z2173 Z-machine load
current with a single seeded axial sinusoid -- reduced to the seeded-mode amplitude-growth
history and compared against the digitized Sinars-2011 experimental growth curve.

This is the faithful replication the 2026-05-31 reframe of #120 asked for: it replaces the
idealized code-unit surrogate (test_verify_maglif_mrt_cpu.py, #142) with the #160/[P3]
dimensional SI geometry + the real z2173 tabulated drive, seeded single-mode, on the
ideal-MHD core, at the paper resolution (12.5 um finest) on the GPU.  The committed
faithful artifact is inputs/maglif_b1_sinars.athinput.

VALIDATION TIER -- the seeded-mode amplitude-growth curve is the QUANTITATIVE observable,
compared point-by-point against the committed Sinars-2011 curve (now digitized via #154)
through the curve oracle (ADR-0008).  As of #174/[P5] this is now a FAITHFUL DIMENSIONAL
comparison: the IONMIX tabulated 3T EOS is wired into BOTH the LLF solver (#162) and the
multi-material maglif IC (the tabulated IC branch fills the conserved energy + the e_ele
scalar; below-table-floor densities are edge-clamped), and the z2173 drive is consumed in
SI units (ns->code time, MA->code current; ADR-0010), so the amplitude(t) curve is
compared on ABSOLUTE TIME and ABSOLUTE AMPLITUDE [mm] -- the time axis gates implosion
timing (drive+EOS calibration), the magnitude gates growth.

That comparison is REPORTED in the scorecard (binding=False), NOT asserted.  Per ADR-0011
(attribution policy) any remaining growth shortfall is NOT attributable to material
strength -- the Ellison/FLASH reference (arXiv:2504.10760) matched B1 with a STRENGTHLESS
hydrodynamic liner.  Once the upstream anchors are cleared (the closed-form
magnetic-pressure anchor for calibration -- test_unit_maglif_si_calibration_cpu; implosion
timing for bulk dynamics; the 1D-stays-1D + coarse two-grid convergence for numerics --
test_verify_maglif_b1_tabulated_cpu), a surviving residual is bounded by the reference's
model set (conduction / resistivity / gray radiation diffusion) and filed as a follow-up
with that evidence; it does not gate this run.  #175/[P6] then enabled that model set one
operator at a time (test_verify_maglif_b1_operators_gpu) and found NONE yet altered the
tabulated B1 implosion faithfully (ADR-0012), bounding the residual by three model-set
wiring/consistency gaps (still NOT strength), closed one at a time: resb is now COUPLED to
the live driven b0.x2f (#181/[P7a], resb_couple_b0) and ACTIVE; fld/fld+mrad are now
ACTIVE too (erad SOURCED from the gas, #182/[P7b], fld_source_erad_from_gas) though
uncalibrated; acond remains EOS-inconsistent (ideal-gamma T, #183).  The
BINDING gate here is therefore the
qualitative single-mode MRT signature -- the liner converges, the SEEDED mode grows and
stays dominant, the synthetic radiograph limb modulates, and a pert_amp=0 control stays
exactly 1-D -- with the quantitative curve verdict reported alongside.  Flipping the
report to a binding assert closes #120 AC#2 if the run lands in-band.

PHYSICS SCOPE: the single-mode MRT feed is a magneto-HYDRODYNAMIC instability set by the
inward acceleration (magnetic pressure) and the Atwood number across the liner/vacuum
interface -- carried by the cylindrical MHD core (ADR-0004) with the tabulated 3T Al EOS
(#162/#174) + the SI-calibrated prescribed-I(t) circuit drive (ADR-0005 mode A, ADR-0010).
The coupled radiation-conduction stack is OFF here (see inputs/maglif_b1_sinars.athinput).

GPU-only (``_gpu`` suffix): built+run with CUDA; auto-collected by ``run_test_suite.py
--gpu`` (excluded from CPU CI).  Heavy GPU CI harness is #123/[C6].
"""

# Modules
import glob
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import test_suite.testutils as testutils  # noqa: E402
import test_suite.verification.harness as harness  # noqa: E402
import test_suite.verification.ground_truth_oracle as gto  # noqa: E402
import test_suite.verification.scorecard as scorecard  # noqa: E402
import test_suite.verification.experiment_overlay as overlay  # noqa: E402

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_b1_sinars.athinput.
PERT_MODE = 4          # seeded axial mode number (wavelength = Lz/4 = 400 um)
PERT_AMP = 0.02        # seeded radial amplitude in CODE units (0.002 cm / length_cgs 0.1)
LZ = 1.6               # axial domain extent [mm] (x3max - x3min); used to normalise z
D_LINER = 1.0          # liner density in code units (2.7 g/cc / density_cgs 2.7)
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

# Synthetic-radiograph instrument response (#212).  A real radiograph cannot resolve
# structure finer than its own spatial resolution, so the synthetic limb we compare
# against the Sinars curve must be band-limited the same way -- otherwise the
# (max - min) excursion also counts grid-seeded MRT, which grows like sqrt(k) and so
# strengthens with every refinement (a_sb 0.0162 / 0.0403 / 0.0798 mm at 60 ns for
# 288 / 432 / 864, diverging while the hydro state itself converged to 4.4%).
RADIOGRAPH_RES_UM = 15.0                                  # quoted Sinars radiograph res
K_MAX_BAND = int(LZ * 1.0e3 / (2.0 * RADIOGRAPH_RES_UM))  # cycles across LZ (= 53)

# Paper density-contour observable (#222).  Ellison SS3.1 (Fig. 3) measures the B1
# spike-bubble reduction as the (max - min) radius of the ABSOLUTE rho = 0.1 g/cc
# density contour at the vacuum/conductor interface -- stated to be "not very
# sensitive to the exact choice of density threshold".  This is independent of the
# radiograph limb (a_sb): the limb's half-max is a RELATIVE criterion that moves as
# the shell decompresses and can switch features when a brighter structure appears
# in the projection (the suspected 51->54 ns 2.1x jump on the 432 leg).
DENSITY_CGS = 2.7                  # g/cc per code density; must match athinput units
RHO_CONTOUR_GCC = 0.1              # the paper's contour threshold [g/cc]
CONTOUR_THRESHOLDS_GCC = (0.05, 0.1, 0.2)   # the paper's insensitivity band [g/cc]

# Qualitative pre-check thresholds (fast sign/shape gates; NOT experiment-anchored bars --
# the seeded growth is anchored against the Sinars curve via the oracle, reported below).
# NOTE (#174): the "cal ~" values were calibrated on the pre-#174 ideal-MHD raw-current
# surrogate; the run is now tabulated-3T + SI-calibrated drive (gentler, longer run-in),
# so these bars may need re-tuning when the #123 GPU harness next runs this benchmark.
# They stay conservative sign/shape gates (converges, seed grows, control stays 1-D), not
# magnitude matches (the magnitude is the reported oracle verdict).
CONV_FRAC = 0.85       # require >= 15% bulk convergence (R_liner shrinks)
DOM_FRAC = 0.2         # seeded mode holds >= 20% of interface z-variance at its peak
GROWTH_MIN = 1.5       # seeded-mode amplitude grows >= 1.5x at peak (MRT feed)
FLAT_TOL = 1.0e-6      # pert_amp=0 control seeded-mode amplitude stays below

# GPU build flags (athenakdev / A100); overridable for other runners (#123 generalizes).
GPU_ARCH = os.environ.get("ATHENAK_GPU_ARCH", "Kokkos_ARCH_AMPERE80")
NVCC_WRAPPER = os.environ.get(
    "ATHENAK_NVCC_WRAPPER",
    os.path.join(testutils._repo_root(), "kokkos", "bin", "nvcc_wrapper"),
)
GPU_FLAGS = [
    "-D", "Kokkos_ENABLE_CUDA=On",
    "-D", f"{GPU_ARCH}=On",
    "-D", f"CMAKE_CXX_COMPILER={NVCC_WRAPPER}",
]

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_b1_sinars.athinput"
)
trace_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "z2173_current.dat"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")
# The maglif _cpu tests build into the SAME build_pgen/maglif dir as a Serial-Kokkos (CPU)
# binary; a parent-level reconfigure does NOT switch Kokkos's backend, so force a CLEAN
# build dir so the CUDA backend is configured fresh.
build_dir = os.path.join(testutils._repo_root(), "tst", "build_pgen", "maglif")


def _build_dir_is_cuda():
    """True iff build_pgen/maglif is already configured with the Kokkos CUDA backend."""
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, "r") as f:
        return "KOKKOS_ENABLE_CUDA:BOOL=ON" in f.read().upper()


def _run(basename, pert_amp=None):
    """Run the faithful paper-resolution setup on GPU; return sorted snapshot paths.

    The faithful run (pert_amp=None) uses the committed athinput seed amplitude unchanged;
    only the absolute current_file path and basename are set.  The 1-D control passes
    pert_amp=0.0 (an athinput-units CGS value) to suppress the seed; the geometry, drive,
    resolution and EOS stay the committed faithful values either way.
    """
    args = [
        f"problem/current_file={trace_file}",
        f"job/basename={basename}",
    ]
    if pert_amp is not None:
        args.append(f"problem/pert_amp={pert_amp}")
    ok = testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS, args=args)
    assert ok, f"faithful Sinars B1 GPU run failed (basename={basename})"
    files = sorted(glob.glob(os.path.join(bin_dir, f"{basename}.prim.*.bin")))
    assert len(files) > 8, f"too few snapshots for {basename}: {len(files)}"
    return files


def _cross(r, p, i, j, level=HALF):
    """Sub-cell radius where the density profile p crosses level between cells i and j."""
    return r[i] + (level - p[i]) / (p[j] - p[i]) * (r[j] - r[i])


def _interfaces(d):
    """Per-z inner/outer half-max interfaces + mass-weighted mean radius (length nz)."""
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)            # (nz, nphi, nr)
    nz = dens.shape[0]
    r_in = np.full(nz, np.nan)
    r_out = np.full(nz, np.nan)
    r_mw = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)                        # (nr,) phi-averaged
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


def _mode_amp(rz, zn, k):
    """Magnitude of the cos/sin projection of r(z) onto axial mode k.

    zn is the axial coordinate NORMALISED to [0, 1) (z / Lz) so mode k is k full
    wavelengths across the domain; magnitude (not signed) is robust to phase drift.
    """
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    zz = zn[good]
    a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz))
    b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz))
    return float(np.hypot(a, b))


def _synthetic_radiograph(d):
    """Synthetic side-on radiograph = forward Abel projection of the axisymmetric density.

    Per-annulus chord length L(x,r) = 2[sqrt(r_out^2 - x^2) - sqrt(r_in^2 - x^2)]
    (singularity-free), so Sigma(x,z) = sum_r rho(r,z) L(x,r); the dense liner is a bright
    limb whose axial position is modulated by the seeded MRT mode.
    """
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)            # (nz, nphi, nr)
    rho_rz = dens.mean(axis=1)                            # (nz, nr) phi-averaged
    dr = r[1] - r[0]
    r_lo = np.clip(r - 0.5 * dr, 0.0, None)
    r_hi = r + 0.5 * dr
    x = r.copy()
    x2 = x[:, None] ** 2
    a = np.sqrt(np.clip(r_hi[None, :] ** 2 - x2, 0.0, None))
    b = np.sqrt(np.clip(r_lo[None, :] ** 2 - x2, 0.0, None))
    weight = 2.0 * (a - b)
    sigma = rho_rz @ weight.T                            # (nz, nx)
    return x, sigma


def _radiograph_mod(x, sigma, zn, k):
    """Single-mode axial contrast of the synthetic radiograph at the bright limb."""
    s_mean = sigma.mean(axis=0)
    ilimb = int(np.argmax(s_mean))
    col = sigma[:, ilimb]
    base = float(np.mean(col))
    cc = col - base
    a = 2.0 * np.mean(cc * np.cos(2.0 * np.pi * k * zn))
    b = 2.0 * np.mean(cc * np.sin(2.0 * np.pi * k * zn))
    return float(np.hypot(a, b)) / (abs(base) + 1.0e-30)


def _band_limit(rz, k_max):
    """Truncate an axial trace to k <= k_max cycles across the domain (#212).

    NaN rows (failed interface finds) are filled by periodic interpolation first so
    the transform stays well-posed; the axial direction is periodic by construction
    (the deck seeds PERT_MODE whole wavelengths across LZ).
    """
    r = np.asarray(rz, dtype=float)
    good = np.isfinite(r)
    if not np.any(good):
        return r
    if not np.all(good):
        idx = np.arange(r.size, dtype=float)
        r = np.interp(idx, idx[good], r[good], period=float(r.size))
    f = np.fft.rfft(r)
    f[int(k_max) + 1:] = 0.0
    return np.fft.irfft(f, n=r.size)


def _spike_bubble_amp(rz, k_max=None):
    """Spike-bubble amplitude (max - min)/2 of an interface/limb trace.

    The Sinars 2011 observable: radiograph spike-to-bubble excursion (B2 convention,
    spike = max radius on the run-in).  Unlike the mode-k Fourier projection this does
    NOT saturate near lambda/2pi or phase-cancel once the mode goes nonlinear -- in
    b1_window120_204 the fundamental reported 1.3x while this grew 4.5x by 84 ns.

    With k_max set (#212) the trace is first band-limited to the radiograph's own
    resolution, so the excursion counts the seeded mode and its resolvable harmonics
    but not grid-seeded structure the instrument could never have seen.  Without it
    the metric is resolution-divergent and cannot be compared to the measurement.
    """
    if k_max is not None:
        rz = _band_limit(rz, k_max)
    good = rz[np.isfinite(rz)]
    return float(0.5 * (np.max(good) - np.min(good)))


def _limb_edge(x, sigma):
    """Per-z outermost half-max crossing of the synthetic radiograph Sigma(z, x).

    The bright-limb edge position -- the contour a radiograph measurement tracks.
    """
    nz = sigma.shape[0]
    out = np.full(nz, np.nan)
    for kz in range(nz):
        s = sigma[kz]
        pk = s.max()
        if pk <= 0.0:
            continue
        half = 0.5 * pk
        idx = np.where(s >= half)[0]
        hi = idx[-1]
        if hi == len(x) - 1:
            out[kz] = x[hi]
        else:
            out[kz] = x[hi] + (half - s[hi]) / (s[hi + 1] - s[hi]) * (x[hi + 1] - x[hi])
    return out


def _contour_edge(d, rho_gcc=RHO_CONTOUR_GCC):
    """Per-z outermost radial crossing of the ABSOLUTE density level rho_gcc [g/cc].

    The paper's contour criterion (#222, Ellison SS3.1/Fig. 3): unlike the limb's
    relative half-max, the level is fixed in physical units, so it keeps tracking the
    same material edge as the shell decompresses and cannot be re-keyed by a brighter
    interior feature.  Slices with no material at/above the level give NaN.
    """
    level = float(rho_gcc) / DENSITY_CGS
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)            # (nz, nphi, nr)
    nz = dens.shape[0]
    out = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)                        # (nr,) phi-averaged
        idx = np.where(p >= level)[0]
        if len(idx) == 0:
            continue
        hi = idx[-1]
        if hi == len(r) - 1:
            out[kz] = r[hi]
        else:
            out[kz] = _cross(r, p, hi, hi + 1, level)
    return out


def _contour_amp(rz):
    """a_contour: FULL max - min excursion [mm] of a contour trace over z.

    Paper convention (#222): the B1 Fig.-3 quantity is the max-minus-min radius of
    the rho = 0.1 g/cc contour -- NOT halved (a_sb is the HALF excursion).  NaN rows
    (slices below the threshold) are skipped; all-NaN gives NaN.
    """
    good = rz[np.isfinite(rz)]
    if good.size == 0:
        return float("nan")
    return float(np.max(good) - np.min(good))


def _contour_threshold_spread(d, thresholds_gcc=CONTOUR_THRESHOLDS_GCC):
    """a_contour at each threshold + max relative deviation from the paper's 0.1.

    Mirrors the paper's insensitivity claim (#222 AC): a_contour should vary by
    <10% for thresholds 0.05-0.2 g/cc.  Returns ({rho_gcc: amp}, spread); spread is
    NaN when the reference amplitude is not finite/positive.
    """
    amps = {thr: _contour_amp(_contour_edge(d, thr)) for thr in thresholds_gcc}
    ref = _contour_amp(_contour_edge(d, RHO_CONTOUR_GCC))
    if not np.isfinite(ref) or ref <= 0.0:
        return amps, float("nan")
    spread = max(abs(a - ref) for a in amps.values()) / ref
    return amps, float(spread)


def _trajectory(files):
    """Reduce a run's snapshots to (z, zn, times, R_liner, R_front, a_outer, rad_mod,
    a_sb, a_sb_raw, a_contour).

    a_sb is the radiograph limb-edge spike-bubble amplitude [mm] band-limited to the
    instrument resolution (#212) -- the quantity compared against Sinars.  a_sb_raw is
    the unfiltered excursion, kept as a diagnostic: the gap between the two IS the
    grid-seeded content, so a widening gap under refinement is the tell.  a_contour
    (#222) is the paper's independent second observable: the FULL max - min radius of
    the absolute rho = 0.1 g/cc density contour (Ellison SS3.1/Fig. 3).
    """
    z = zn = None
    times, r_liner, r_front, a_outer, rad_mod = [], [], [], [], []
    a_sb, a_sb_raw, a_contour = [], [], []
    for fp in files:
        d = bin_convert.read_binary_as_athdf(fp)
        if z is None:
            z = np.asarray(d["x3v"], dtype=float)
            zn = z / LZ
        r_in, r_out, r_mw = _interfaces(d)
        xrad, sigma = _synthetic_radiograph(d)
        times.append(float(d["Time"]))
        r_liner.append(float(np.nanmean(r_mw)))
        r_front.append(float(np.nanmean(r_in)))
        a_outer.append(_mode_amp(r_out, zn, PERT_MODE))
        rad_mod.append(_radiograph_mod(xrad, sigma, zn, PERT_MODE))
        edge = _limb_edge(xrad, sigma)
        a_sb.append(_spike_bubble_amp(edge, k_max=K_MAX_BAND))
        a_sb_raw.append(_spike_bubble_amp(edge))
        a_contour.append(_contour_amp(_contour_edge(d)))
    return (z, zn, np.array(times), np.array(r_liner), np.array(r_front),
            np.array(a_outer), np.array(rad_mod), np.array(a_sb),
            np.array(a_sb_raw), np.array(a_contour))


def _plot_snapshot(files, z):
    """Save the initial vs final phi-averaged density r-z snapshot as a PNG."""
    harness._ensure_dirs()
    d0 = bin_convert.read_binary_as_athdf(files[0])
    dN = bin_convert.read_binary_as_athdf(files[-1])
    r = np.asarray(d0["x1v"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    ext = [r[0], r[-1], z[0], z[-1]]
    for ax, d, ttl in ((axes[0], d0, "initial"), (axes[1], dN, "final")):
        rho = np.asarray(d["dens"], dtype=float).mean(axis=1)   # (nz, nr)
        im = ax.imshow(rho, origin="lower", aspect="auto", extent=ext, cmap="inferno")
        ax.set_title(f"{ttl}  (t={float(d['Time']):.2f} ns)")
        ax.set_xlabel("r [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$")
    axes[0].set_ylabel("axial z [mm]")
    fig.suptitle("Faithful single-mode MRT (MagLIF benchmark 1): density r-z")
    fig.tight_layout()
    out = os.path.join(harness.PLOT_DIR, "maglif_b1_sinars_snapshot.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def test_verify_maglif_b1_sinars_gpu():
    """Paper-resolution GPU single-mode MRT; qualitative gate + reported Sinars curve."""
    if not _build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        files = _run("maglif_b1_sinars")
        (z, zn, times, r_liner, r_front, a_outer, rad_mod, a_sb,
         a_sb_raw, a_contour) = _trajectory(files)

        # Final snapshot is finite and stable (the implosion ran in cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # (1) The liner actually converges (bulk implosion, not a static/expanding shell).
        assert r_liner[-1] < CONV_FRAC * r_liner[0], (
            f"liner did not converge: R0={r_liner[0]:.4f} -> R={r_liner[-1]:.4f}"
        )

        # (2) The leading (fuel-side) front moves inward as the liner accelerates inward.
        assert r_front[-1] < r_front[0], (
            f"leading front did not move inward: {r_front[0]:.4f} -> {r_front[-1]:.4f}"
        )

        # (3) Seed-amplitude sanity: the driven-surface single-mode amplitude starts at
        # ~the seeded value (the reduction picked up the right mode at the seed level).
        assert abs(a_outer[0] - PERT_AMP) < 0.35 * PERT_AMP, (
            f"initial driven-surface amplitude {a_outer[0]:.4f} != seed {PERT_AMP}"
        )

        # (4) The seeded mode GROWS under the sustained acceleration (the MRT feed).  The
        # measure is the PEAK amplitude over the run-in: in this idealized ideal-MHD
        # implosion the single mode e-folds order-unity times during the magnetic run-in,
        # then deep convergence (compression-dominated, no material strength) crushes it
        # back -- the documented ceiling -- so the peak, not the final, is the growth
        # signal (the quantitative shortfall vs Sinars is the reported oracle verdict).
        i_peak = int(np.argmax(a_outer))
        growth = a_outer[i_peak] / a_outer[0]
        assert growth > GROWTH_MIN, (
            f"seeded mode did not grow: peak a_outer {a_outer[i_peak]:.4f} vs seed "
            f"{a_outer[0]:.4f} (= {growth:.2f}x < {GROWTH_MIN}x)"
        )
        # The seeded mode carries a substantial share of the driven-interface axial
        # structure at its growth peak.  By the peak the single mode has gone NONLINEAR
        # (mushroom head + harmonics spread power off the fundamental), so it no longer
        # holds a majority of the variance, but its fundamental power is still a sizeable
        # fraction (calibrated ~0.27) -- versus the pert_amp=0 control's exactly-zero
        # structure (below), this is the seed-driven single-mode signature.
        peak_file = files[i_peak]
        r_out_pk = _interfaces(bin_convert.read_binary_as_athdf(peak_file))[1]
        ro = r_out_pk[np.isfinite(r_out_pk)]
        var_tot = float(np.mean((ro - ro.mean()) ** 2))
        dom = (0.5 * a_outer[i_peak] ** 2) / var_tot if var_tot > 0.0 else 0.0
        assert dom > DOM_FRAC, (
            f"seeded mode share at growth peak too small: {dom:.2f} < {DOM_FRAC}"
        )

        # (5) The synthetic-radiograph limb modulation (the observable MRT signal) grows
        # above its seeded level at some point during the run-in.
        assert rad_mod.max() > rad_mod[0], (
            f"radiograph modulation did not grow: {rad_mod[0]:.4e} -> peak "
            f"{rad_mod.max():.4e}"
        )

        # (6) DISCRIMINATOR: the SAME drive+geometry+convergence with pert_amp=0 stays
        # exactly 1-D (no axial structure) -- proving the growth comes from the seed, not
        # the numerics.  This is the binding control.
        cfiles = _run("maglif_b1_sinars_ctl", 0.0)
        _, _, ctimes, cr_liner, _, ca_outer, _, _, _, _ = _trajectory(cfiles)
        assert ca_outer.max() < FLAT_TOL, (
            f"pert_amp=0 control developed axial structure (max a_outer="
            f"{ca_outer.max():.3e} >= {FLAT_TOL}); not 1-D without the seed"
        )
        assert cr_liner[-1] < CONV_FRAC * cr_liner[0], (
            f"control liner did not converge: {cr_liner[0]:.4f} -> {cr_liner[-1]:.4f}"
        )

        _plot_snapshot(files, z)

        # --- QUANTITATIVE ANCHOR (REPORTED): the seeded-mode amplitude-growth history
        # (times, a_outer) is the growth curve this benchmark reproduces, compared against
        # the committed Sinars-2011 curve via the oracle.  As of #174 this is a FAITHFUL
        # DIMENSIONAL comparison -- the SI-calibrated drive makes `times` absolute ns, and
        # the code length unit is mm (length_cgs 0.1), so `a_outer` is absolute mm: the
        # curve is compared point-for-point on the experiment's own (ns, mm) axes, NO
        # window/growth-factor remap.  binding=False per ADR-0011 (docstring): a residual
        # is bounded by the reference model set (conduction/resistivity/radiation, NOT
        # strength) and filed as a follow-up; it does not gate this run.
        oracle = gto.GroundTruthOracle.from_committed()
        datum = oracle.get("B1", "single_mode_amplitude_growth")
        if datum.is_pending:                       # forward-safe (curve is committed now)
            scorecard.record_pending(
                "B1", "single_mode_amplitude_growth", issue=120,
                note="Sinars 2011 amplitude-growth curve pending digitization",
            )
            exp_points = None
        else:
            # The committed curve is a radiograph SPIKE-BUBBLE measurement; compare it
            # against the same observable (a_sb), not the mode-4 Fourier projection,
            # which saturates near lambda/2pi and phase-rotates once the MRT goes
            # nonlinear (b1_window120_204: fundamental 1.3x vs spike-bubble 4.5x).
            # a_sb carries the instrument band limit (#212), so this compares two
            # traces of the same spatial resolution.
            res = oracle.compare(
                "B1", "single_mode_amplitude_growth", (times, a_sb)
            )
            y_seed = float(min(datum.points, key=lambda p: float(p["x"]))["y"])
            exp_span = max(p["y"] for p in datum.points) / y_seed
            scorecard.record_result(
                res.worst_point, binding=False,
                note="faithful SI tabulated-3T run, absolute time + amp [mm]; "
                     "residual bounded by reference model set (conduction/resistivity/"
                     "radiation, NOT strength; ADR-0011), filed as follow-up",
            )
            print(f"[B1] {res}  (reported, not asserted)")
            print(f"[B1] faithful-SI spike-bubble peak amplitude {a_sb.max():.4f} mm "
                  f"(mode-4 fundamental peak {a_outer.max():.4f} mm; Sinars curve spans "
                  f"~{exp_span:.0f}x its seed)")
            # Band-limited vs raw: the gap IS the sub-instrument (grid-seeded) content.
            print(f"[B1] instrument band k<={K_MAX_BAND} ({RADIOGRAPH_RES_UM:.0f} um): "
                  f"peak a_sb {a_sb.max():.4f} mm vs unfiltered {a_sb_raw.max():.4f} mm "
                  f"({a_sb_raw.max() / max(a_sb.max(), 1.0e-30):.2f}x)")
            # #222: the paper's independent contour observable, reported next to a_sb,
            # with the threshold-insensitivity check (paper claims <10% over
            # 0.05-0.2 g/cc) evaluated at the a_contour peak snapshot.
            i_ct = int(np.nanargmax(a_contour))
            amps, spread = _contour_threshold_spread(
                bin_convert.read_binary_as_athdf(files[i_ct])
            )
            amp_str = ", ".join(
                f"{thr:g}: {amps[thr]:.4f}" for thr in CONTOUR_THRESHOLDS_GCC
            )
            print(f"[B1] rho={RHO_CONTOUR_GCC:g} g/cc contour (Ellison Fig. 3): peak "
                  f"a_contour {a_contour[i_ct]:.4f} mm at t={times[i_ct]:.1f} ns; "
                  f"threshold sweep [g/cc: mm] {{{amp_str}}} -> spread {spread:.1%} "
                  f"(paper claims <10%)")
            exp_points = [
                {"x": p.x, "y": p.exp_value, "band": p.band}
                for p in res.point_results
            ]

        overlay.overlay_curve(
            "maglif_b1_sinars", times, a_sb, exp_points=exp_points,
            xlabel="t", ylabel="spike-bubble amplitude [mm]",
            x_unit="ns", unit="mm", pending_issue=120,
            title="Faithful single-mode MRT growth vs Sinars 2011 (MagLIF benchmark 1)",
        )

        harness.verify(
            "maglif_b1_sinars",
            times,
            {
                "R_liner": r_liner,
                "R_front": r_front,
                "a_outer": a_outer,
                "rad_mod": rad_mod,
                "a_sb": a_sb,
                "a_sb_raw": a_sb_raw,
                "a_contour": a_contour,
            },
            coord_label="t",
            xlabel="t [ns]",
            title="Faithful single-mode MRT (MagLIF benchmark 1): convergence + growth",
            rtol=1.0e-4,
            atol=1.0e-8,
        )
    finally:
        for pat in ("maglif_b1_sinars.*", "maglif_b1_sinars_ctl.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        shutil.rmtree(bin_dir, ignore_errors=True)
