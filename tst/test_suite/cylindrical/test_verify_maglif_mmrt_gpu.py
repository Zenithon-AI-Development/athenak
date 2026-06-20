"""
Multi-mode magneto-Rayleigh-Taylor QUALITATIVE GPU replication (Ellison benchmark 2 /
[C5], issue #122).  Paper-resolution GPU run of the multi-mode MRT surface-roughness
implosion, validated QUALITATIVELY against the experimental radiographs (limb-modulation
behaviour) without overclaiming on inherently stochastic data.

Runs the integrated MagLIF/Z-pinch problem generator (issue [10a]/#32) at paper resolution
on the GPU: a dense cylindrical (Be-like) liner is sustainedly accelerated radially inward
by a prescribed constant load current, the magnetic pressure of the enclosed B_phi field
(the light fluid) driving the dense liner (the heavy fluid) -- the magneto-Rayleigh-Taylor
(MRT) unstable configuration.  The liner carries a BAND of axial modes with deterministic
random phases (a manufactured surface-roughness spectrum, the experimental Be-liner
machining/roughness seed); under the continuous inward acceleration the broadband
roughness grows nonlinearly (the multi-mode MRT feed).  This is FLASH MagLIF benchmark 2
(multi-mode MRT from surface roughness; Ellison et al. 2025, arXiv:2504.10760).

VALIDATION TIER -- QUALITATIVE (per the tiered validation bar, ADR-0008).  Multi-mode MRT
from surface roughness is INHERENTLY STOCHASTIC: the seeded phases are not the
experiment's, and the nonlinear cascade is chaotic, so a quantitative curve match would
OVERCLAIM.  This benchmark therefore confirms the QUALITATIVE limb-modulation behaviour --
the synthetic side-on radiograph develops a broadband, multi-mode, driven-surface-dominant
modulation on the bright limb that GROWS under the sustained acceleration -- and documents
the verdict as qualitative in the suite scorecard (binding=False).  The quantitative
McBride-2012 growth CURVE remains the reduced-_cpu reported anchor
(test_verify_maglif_mmrt_cpu.py, #143); it is NOT asserted here.

PHYSICS SCOPE: the multi-mode MRT feed is a magneto-HYDRODYNAMIC instability set by the
inward acceleration and the Atwood number across the liner/vacuum interface -- carried by
the cylindrical ideal-MHD core (ADR-0004) + the prescribed-I(t) circuit drive (ADR-0005
mode A).  The coupled radiation-conduction stack (ADR-0009) is OFF here (see
inputs/maglif_mmrt_si.athinput) for a PHYSICS reason: its toy coefficients would damp the
MRT growth.  (The operator-split parabolic GPU runtime segfault was fixed in #153/#139;
the coupled GPU path now runs, guarded by test_verify_maglif_smoke_gpu.py.)
The COUPLED multi-mode run is the _cpu benchmark (#116/[B3], #143).

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
import test_suite.verification.scorecard as scorecard  # noqa: E402
import test_suite.verification.experiment_overlay as overlay  # noqa: E402

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_mmrt_si.athinput.
KMIN = 2               # lowest seeded axial mode number
KMAX = 6               # highest seeded axial mode number
PERT_AMP = 0.08        # aggregate radial roughness amplitude (split per mode)
NMODES = KMAX - KMIN + 1
# RMS of a sum of NMODES cosines of equal amplitude (pert_amp/NMODES) with random phases.
SEED_RMS = (PERT_AMP / NMODES) * np.sqrt(NMODES / 2.0)
D_LINER = 1.0          # liner (dense shell) density
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

# Qualitative pre-check thresholds (fast sign/shape gates; NOT experiment-anchored bars).
CONV_FRAC = 0.75       # require >= 25% bulk convergence (R_liner shrinks)
PART_MIN = 2.0         # outer spectrum must keep >= 2 effective modes (multi-mode)
DOM_MAX = 0.6          # no single seeded mode may hold > 60% of band power (no collapse)

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
    testutils._repo_root(), "tst", "inputs", "maglif_mmrt_si.athinput"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")
# The maglif _cpu tests build into the SAME build_pgen/maglif dir as a Serial-Kokkos (CPU)
# binary; a parent-level reconfigure does NOT switch Kokkos's backend from Serial to Cuda
# (the kokkos sub-config is sticky), so a GPU run in a CPU-contaminated dir would silently
# run on the host.  Force a CLEAN build dir so the CUDA backend is configured fresh.
build_dir = os.path.join(testutils._repo_root(), "tst", "build_pgen", "maglif")


def _build_dir_is_cuda():
    """True iff build_pgen/maglif is already configured with the Kokkos CUDA backend."""
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, "r") as f:
        return "KOKKOS_ENABLE_CUDA:BOOL=ON" in f.read().upper()


def _cross(r, p, i, j):
    """Sub-cell radius where the density profile p crosses HALF between cells i and j."""
    f = (HALF - p[i]) / (p[j] - p[i])
    return r[i] + f * (r[j] - r[i])


def _interfaces(data):
    """Per-z liner interfaces (phi-averaged) + mass-weighted mean liner radius.

    For each axial height z, reduce to the radial density profile and return the inner and
    outer half-max crossings of the dense-liner slab (continuous, FP-robust) plus the
    mass-weighted mean liner radius.  Each is an array of length nz.
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


def _rms(rz):
    """RMS amplitude of the mean-subtracted interface profile r(z) -- total roughness."""
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    return float(np.sqrt(np.mean(rr * rr)))


def _band_spectrum(rz, z):
    """Per-mode amplitude (cos/sin projection magnitude) over the seeded band [KMIN,KMAX].

    Magnitude (not signed cosine) so each measure is robust to phase drift of the growing
    mode.  z is the axial coordinate on [0, Lz=1].
    """
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    zz = z[good]
    amps = []
    for k in range(KMIN, KMAX + 1):
        a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz))
        b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz))
        amps.append(np.hypot(a, b))
    return np.array(amps)


def _participation(amps):
    """Spectral participation number (sum p)^2 / sum p^2, p = mode power amps^2.

    Equals N for N equal-amplitude modes, 1 for a single dominant mode -> the effective
    number of modes carrying the roughness (the broadband-vs-single-mode discriminator).
    """
    p = amps ** 2
    s = float(p.sum())
    return float(s * s / np.sum(p * p)) if s > 0.0 else 0.0


def _dominant_frac(amps):
    """Fraction of the band power held by the single largest seeded mode."""
    p = amps ** 2
    s = float(p.sum())
    return float(p.max() / s) if s > 0.0 else 0.0


def _synthetic_radiograph(data):
    """Synthetic side-on radiograph = Abel forward projection of the axisymmetric density.

    A parallel X-ray beam crosses the cylinder perpendicular to its axis (z).  For an
    axisymmetric density rho(r,z) the line-of-sight areal density at transverse impact
    parameter x is the forward Abel transform, evaluated by the exact per-annulus chord
    length L(x,r) = 2[sqrt(r_out^2 - x^2) - sqrt(r_in^2 - x^2)] (singularity-free), so
    Sigma(x,z) = sum_r rho(r,z) L(x,r).  Returns (x, Sigma) with Sigma shaped (nz, nx);
    the dense liner is a bright limb whose axial position is modulated by MRT roughness.
    """
    r = np.asarray(data["x1v"], dtype=float)            # (nr,) uniform cell centres
    dens = np.asarray(data["dens"], dtype=float)        # (nz, nphi, nr)
    rho_rz = dens.mean(axis=1)                          # (nz, nr) phi-averaged
    dr = r[1] - r[0]
    r_in = np.clip(r - 0.5 * dr, 0.0, None)             # annulus inner faces (axis -> 0)
    r_out = r + 0.5 * dr                                # annulus outer faces
    x = r.copy()                                        # impact parameters = cell radii
    x2 = x[:, None] ** 2                                # (nx, 1)
    a = np.sqrt(np.clip(r_out[None, :] ** 2 - x2, 0.0, None))   # (nx, nr)
    b = np.sqrt(np.clip(r_in[None, :] ** 2 - x2, 0.0, None))
    weight = 2.0 * (a - b)                              # chord length L(x_i, r_j)
    sigma = rho_rz @ weight.T                           # (nz, nx)
    return x, sigma


def _radiograph_mod(x, sigma):
    """Broadband axial contrast of the synthetic radiograph at the bright limb.

    The limb impact parameter is the x maximising the z-averaged areal density (the
    tangent ray grazing the dense shell).  At that fixed x, the areal density modulates as
    the roughened liner moves through the sight line; return the RMS modulation normalised
    by the mean limb brightness (dimensionless multi-mode radiograph contrast).
    """
    s_mean = sigma.mean(axis=0)                          # (nx,)
    ilimb = int(np.argmax(s_mean))
    col = sigma[:, ilimb]                                # Sigma(x_limb, z)
    base = float(np.mean(col))
    return float(np.std(col)) / (abs(base) + 1.0e-30)


def _plot_radiograph(x, z, sig0, sigN, t0, tN):
    """Save the initial vs final synthetic radiograph (Abel projection) as a PNG."""
    harness._ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    ext = [x[0], x[-1], z[0], z[-1]]
    for ax, sig, t, ttl in (
        (axes[0], sig0, t0, "initial"),
        (axes[1], sigN, tN, "final"),
    ):
        im = ax.imshow(sig, origin="lower", aspect="auto", extent=ext, cmap="inferno")
        ax.set_title(f"{ttl}  (t={t:.3f})")
        ax.set_xlabel("impact parameter x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\Sigma$")
    axes[0].set_ylabel("axial z")
    fig.suptitle("Synthetic side-on radiograph (MagLIF benchmark 2, multi-mode MRT, GPU)")
    fig.tight_layout()
    out = os.path.join(harness.PLOT_DIR, "maglif_mmrt_gpu_radiograph.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def test_verify_maglif_mmrt_gpu():
    """Paper-resolution GPU multi-mode MRT; QUALITATIVE limb-modulation replication."""
    # Clean build dir so Kokkos is (re)configured with the CUDA backend, not a stale
    # Serial backend left by a prior maglif _cpu build sharing build_pgen/maglif.
    if not _build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        assert testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS), \
            "MagLIF multi-mode MRT GPU run failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_mmrt_si.prim.*.bin")))
        assert len(files) > 8, f"too few multi-mode MRT snapshots: {len(files)}"

        z = None
        times, r_liner, r_front, rms_outer, rms_inner, npart, rad_mod = (
            [], [], [], [], [], [], [])
        rad0 = radN = None
        for fp in files:
            d = bin_convert.read_binary_as_athdf(fp)
            if z is None:
                z = np.asarray(d["x3v"], dtype=float)
            r_in, r_out, r_mw = _interfaces(d)
            xrad, sigma = _synthetic_radiograph(d)
            times.append(float(d["Time"]))
            r_liner.append(float(np.nanmean(r_mw)))
            r_front.append(float(np.nanmean(r_in)))
            rms_outer.append(_rms(r_out))
            rms_inner.append(_rms(r_in))
            npart.append(_participation(_band_spectrum(r_out, z)))
            rad_mod.append(_radiograph_mod(xrad, sigma))
            if rad0 is None:
                rad0 = sigma
            radN = sigma
        times = np.array(times)
        r_liner = np.array(r_liner)
        r_front = np.array(r_front)
        rms_outer = np.array(rms_outer)
        rms_inner = np.array(rms_inner)
        npart = np.array(npart)
        rad_mod = np.array(rad_mod)

        # Final snapshot is finite and stable (the implosion ran in cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # The seeded roughness amplitude is laid down as designed (RMS ~ analytic seed).
        assert abs(rms_outer[0] - SEED_RMS) < 0.35 * SEED_RMS, (
            f"initial driven-surface roughness {rms_outer[0]:.4f} != seed {SEED_RMS:.4f}"
        )

        # (1) The liner actually converges (bulk implosion, not a static/expanding shell).
        assert r_liner[-1] < CONV_FRAC * r_liner[0], (
            f"liner did not converge: R0={r_liner[0]:.4f} -> R={r_liner[-1]:.4f}"
        )

        # (2) The leading (fuel-side) front moves inward as the liner accelerates inward.
        assert r_front[-1] < r_front[0], (
            f"leading front did not move inward: {r_front[0]:.4f} -> {r_front[-1]:.4f}"
        )

        # (3) The broadband roughness GROWS on the magnetically-driven (outer) interface
        # -- the multi-mode MRT feed (qualitative sign check; no quantitative bar).
        assert rms_outer[-1] > rms_outer[0], (
            f"driven-surface roughness did not grow: {rms_outer[0]:.4f} -> "
            f"{rms_outer[-1]:.4f}"
        )

        # (4) MagLIF asymmetry: the driven (outer) interface roughens MORE than the fuel-
        # side (inner) interface, which is RT-stabilised during run-in (stays bounded).
        g_out = rms_outer[-1] / rms_outer[0]
        g_in = rms_inner[-1] / rms_inner[0]
        assert g_out > g_in, (
            f"driven surface did not roughen more than the fuel side: "
            f"g_out={g_out:.2f} <= g_in={g_in:.2f}"
        )
        assert rms_inner[-1] < 1.5 * rms_inner[0], (
            f"fuel-side roughness ran away (not RT-stabilised): "
            f"{rms_inner[0]:.4f} -> {rms_inner[-1]:.4f}"
        )

        # (5) The seeded spectrum stays MULTI-MODE -- it does not collapse to one mode
        # (the inverse of benchmark 1's single-mode dominance): outer-interface power is
        # spread over >= 2 effective modes and no single mode holds the majority.
        amps_fin = _band_spectrum(_interfaces(fin)[1], z)
        assert npart[-1] > PART_MIN, (
            f"outer spectrum collapsed to ~1 mode: participation {npart[-1]:.2f} "
            f"< {PART_MIN}"
        )
        dom = _dominant_frac(amps_fin)
        assert dom < DOM_MAX, (
            f"a single seeded mode dominates (not multi-mode): {dom:.2f} >= {DOM_MAX}"
        )

        # (6) The synthetic-radiograph LIMB MODULATION (the experimentally observable
        # multi-mode MRT signal) GROWS -- the qualitative replication target of this
        # benchmark (sign check; stochastic, so not quantitatively asserted).
        rad_growth = rad_mod[-1] / rad_mod[0] if rad_mod[0] > 0 else float("nan")
        assert rad_mod[-1] > rad_mod[0], (
            f"radiograph limb modulation did not grow: {rad_mod[0]:.4e} -> "
            f"{rad_mod[-1]:.4e}"
        )

        # --- DOCUMENTED AS QUALITATIVE (#122 [C5], tiered bar): record the qualitative
        # verdict in the suite scorecard (binding=False, reported not asserted).  Multi-
        # mode MRT from surface roughness is inherently stochastic, so the synthetic
        # radiograph reproducing the limb-modulation BEHAVIOUR (broadband, multi-mode,
        # driven-surface-dominant, growing) is the validation -- NOT a curve match (the
        # McBride-2012 curve stays the reduced-_cpu reported anchor, #143).
        scorecard.record(
            "B2", "multimode_mrt_limb_modulation",
            sim=rad_growth, experiment=None, band=None, unit=None,
            verdict="QUALITATIVE-MATCH", binding=False,
            note=(
                f"GPU paper-res ideal-MHD multi-mode MRT (#122): synthetic-radiograph "
                f"limb modulation present and grows (x{rad_growth:.2f}), broadband "
                f"(participation {npart[-1]:.1f} modes), driven-surface-dominant "
                f"(g_out={g_out:.2f} > g_in={g_in:.2f}); qualitative tier, not a "
                f"quantitative anchor"
            ),
        )
        print(
            f"[B2] QUALITATIVE multi-mode MRT (GPU): radiograph limb modulation x"
            f"{rad_growth:.2f}, participation {npart[-1]:.2f} modes, asymmetry "
            f"g_out/g_in={g_out / g_in:.2f} (reported, not asserted)"
        )

        # Experiment-overlay diagnostic: the sim growth curve (broadband roughness vs t)
        # with a pending-digitization annotation (the McBride curve is paywalled, #143);
        # the synthetic radiograph (initial vs final) is the qualitative limb-modulation
        # visual.
        overlay.overlay_curve(
            "maglif_mmrt_gpu", times, rms_outer, exp_points=None,
            xlabel="t", ylabel="multi-mode roughness rms_outer",
            x_unit="code units", unit="code units", pending_issue=122,
            title="Multi-mode MRT growth (GPU paper-res, benchmark 2, qualitative)",
        )
        _plot_radiograph(xrad, z, rad0, radN, times[0], times[-1])

        harness.verify(
            "maglif_mmrt_gpu",
            times,
            {
                "R_liner": r_liner,
                "R_front": r_front,
                "rms_outer": rms_outer,
                "rms_inner": rms_inner,
                "npart": npart,
                "rad_mod": rad_mod,
            },
            coord_label="t",
            xlabel="t",
            title="Multi-mode MRT (benchmark 2, GPU): convergence + broadband growth",
            rtol=1.0e-4,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
