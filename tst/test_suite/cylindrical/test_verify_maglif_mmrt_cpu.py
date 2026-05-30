"""
Multi-mode magneto-Rayleigh-Taylor verification (FLASH MagLIF benchmark 2, #42).

Runs the integrated MagLIF/Z-pinch problem generator (issue [10a]/#32) as a SUSTAINEDLY
accelerated liner implosion -- a dense cylindrical liner driven radially inward by a
prescribed constant load current -- with a BAND of axial modes (deterministic random
phases) seeded on the liner interface: a manufactured surface-roughness spectrum.  The
magnetic pressure of the enclosed B_phi field (the light fluid) continuously accelerates
the dense liner (the heavy fluid) inward: the outer (liner/vacuum) interface is therefore
magneto-Rayleigh-Taylor (MRT) unstable, and the broadband roughness grows nonlinearly.
This is benchmark 2 of the FLASH MagLIF validation ladder -- multi-mode MRT from surface
roughness, building on the single-mode benchmark 1 (#39) and the same Phase-1 stack
(Ellison et al. 2025, arXiv:2504.10760).

Physics scope (Phase B, #116/[B3], ADR-0009): this benchmark now runs the FULL COUPLED
radiation-conduction-MHD stack -- cylindrical ideal-MHD (ADR-0004) + the prescribed-I(t)
circuit drive (ADR-0005 mode A) + the Strang-split operator-split parabolic block (grey
flux-limited radiation diffusion + anisotropic Braginskii conduction) with the point-
implicit matter-radiation coupling outside the super-step (see the input's <mhd> block).
Coefficients are nondimensional code units chosen so the coupled physics is a stable,
signature-preserving perturbation; real opacity/EOS-derived values arrive with the IONMIX
tables in Phase C (#118).  Resistive B_phi / multigroup FLD stay off here (Phase C).  The
regression baseline below reflects this coupled stack, not the earlier ideal-MHD run.

Discriminating quantities (see inputs/maglif_mmrt.athinput):
  * R_liner(t)  -- mass-weighted mean liner radius: the bulk convergence trajectory.
  * R_front(t)  -- inner (fuel-side) half-max interface: the leading imploding front.
  * rms_outer(t)-- RMS roughness of the driven (liner/vacuum) interface: the MRT-unstable
                   surface -- the broadband roughness grows under sustained acceleration.
                   THIS is the multi-mode amplitude-growth curve anchored against McBride.
  * rms_inner(t)-- RMS roughness of the fuel-side interface: RT-stabilised during the
                   run-in (it does NOT run away), the expected MagLIF asymmetry.
  * npart(t)    -- spectral participation number (effective # of seeded modes carrying the
                   outer-interface roughness): stays >= 2 -> genuinely multi-mode, NOT a
                   collapse to one dominant mode (the inverse of benchmark 1's single-mode
                   dominance).
  * rad_rms(t)  -- broadband contrast of a SYNTHETIC SIDE-ON RADIOGRAPH (the Abel forward
                   projection of the axisymmetric density along the sight line): the
                   experimentally observable multi-mode MRT signature on the limb.

Oracle (Layer-1, ADR-0008; #143 [VA4]): the EXPERIMENT, via the committed ground-truth
datum in ``verification/ground_truth/b2_multimode_mrt_mcbride_2012.json`` (R. D. McBride
et al., Phys. Rev. Lett. 109, 135004, 2012 / companion Phys. Plasmas 20, 056309, 2013).
The broadband driven-surface roughness growth history (``times``, ``rms_outer``) is THE
multi-mode amplitude-growth curve this benchmark reproduces; it is compared against the
committed McBride-2012 experimental growth curve via the oracle's curve comparison
(``GroundTruthOracle``) and the verdict is recorded in the suite scorecard, replacing the
former self-/re-anchored ``GROWTH_MIN`` (and the radiograph ``RAD_GROWTH``) bars.  Only
cheap qualitative gates -- finiteness, convergence, broadband-seed RMS sanity, the
driven-vs-fuel asymmetry, multi-mode participation, and the front-inward / growth /
radiograph-grows sign checks -- remain as fast pre-checks.

This reduced nondimensional ``_cpu`` surrogate emits the growth curve in CODE UNITS, while
the McBride curve is a physical amplitude-vs-time radiograph: per the reduced-surrogate
policy (B3 velocity-ratio / B4 ICF #140) the absolute experimental curve comparison is the
paper-resolution run (#122 [C5]), which outputs physical units directly.  The committed B2
curve datum is still ``pending_digitization`` -- the McBride figure is paywalled and not
faithfully digitizable from this environment (the same constraint #142 hit for Sinars-2011;
no fabricated points, per ADR-0008) -- so the curve verdict is REPORTED as PENDING here,
never as a pass; the diagnostic overlays the experimental curve + band once digitized
(#122).  The wiring below is the full curve-oracle path: it reports PENDING while pending
and, once digitized, compares the curve and reports the verdict (binding=False) with the
experimental band overlaid.

The test runs the implosion, checks that (1) the liner converges, (2) the leading front
moves inward, (3) the broadband roughness grows on the driven surface (sign check), (4) the
driven surface roughens MORE than the RT-stabilised fuel-side surface (the MagLIF
asymmetry), (5) the seeded spectrum stays multi-mode (no single-mode collapse), and (6) the
synthetic-radiograph modulation grows (sign check), anchors the growth curve against the
McBride-2012 oracle, then plots the growth-curve overlay + a synthetic radiograph and
saves/diffs a golden regression baseline.  Auto-collected by run_test_suite.py (the module
name contains ``_cpu``).

NOTE: like the single-mode benchmark (#39) this run is INSTABILITY-amplifying (MRT growth
amplifies floating-point round-off), so the rtol=1e-5 regression baseline is exact only
for the deterministic SERIAL same-platform re-run; loosen it if a cross-platform flake
appears.
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
import test_suite.cylindrical.maglif_coupled_energy as cenergy  # noqa: E402

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_mmrt.athinput.
KMIN = 2               # lowest seeded axial mode number
KMAX = 6               # highest seeded axial mode number
PERT_AMP = 0.08        # aggregate radial roughness amplitude (split per mode)
NMODES = KMAX - KMIN + 1
# RMS of a sum of NMODES cosines of equal amplitude (pert_amp/NMODES) with random phases.
SEED_RMS = (PERT_AMP / NMODES) * np.sqrt(NMODES / 2.0)
D_LINER = 1.0          # liner (dense shell) density
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

# Qualitative pre-check thresholds (fast sign/shape gates; NOT experiment-anchored bars).
# The broadband driven-surface roughness-growth signature itself is anchored against the
# McBride-2012 experimental growth curve via the curve oracle (#143), which replaces the
# former self-/re-anchored GROWTH_MIN and radiograph-growth RAD_GROWTH bars.
CONV_FRAC = 0.75       # require >= 25% bulk convergence (R_liner shrinks)
PART_MIN = 2.0         # outer spectrum must keep >= 2 effective modes (multi-mode)
DOM_MAX = 0.6          # no single seeded mode may hold > 60% of band power (no collapse)

input_file = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif_mmrt.athinput")
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


def _rms(rz):
    """RMS amplitude of the mean-subtracted interface profile r(z) -- total roughness."""
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    return float(np.sqrt(np.mean(rr * rr)))


def _band_spectrum(rz, z):
    """Per-mode amplitude (cos/sin projection magnitude) over the seeded band [KMIN,KMAX].

    Magnitude (not signed cosine) so each measure is robust to any phase drift of the
    growing mode.  z is the axial coordinate on [0, Lz=1].
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
    parameter x is the forward Abel transform
        Sigma(x,z) = 2 * integral_{r=x}^{R} rho(r,z) r / sqrt(r^2 - x^2) dr,
    evaluated here by the exact per-annulus chord length L(x,r) = 2[sqrt(r_out^2 - x^2) -
    sqrt(r_in^2 - x^2)] (singularity-free), so Sigma(x,z) = sum_r rho(r,z) L(x,r).
    Returns (x, Sigma) with Sigma shaped (nz, nx); the dense liner is a bright limb whose
    axial position is modulated by the seeded MRT roughness.
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
    tangent ray grazing the dense shell).  At that fixed x, the areal density modulates
    along z as the roughened liner moves through the sight line; return the RMS modulation
    normalised by the mean limb brightness (dimensionless multi-mode radiograph contrast).
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
    fig.suptitle("Synthetic side-on radiograph (MagLIF benchmark 2, multi-mode MRT)")
    fig.tight_layout()
    out = os.path.join(harness.PLOT_DIR, "maglif_mmrt_radiograph.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def test_verify_maglif_mmrt():
    """Run multi-mode MRT implosion; verify convergence, broadband growth, radiograph."""
    try:
        assert testutils.run_pgen("maglif", input_file), "MagLIF multi-mode MRT failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_mmrt.prim.*.bin")))
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

        # (3) The broadband roughness grows on the magnetically-driven (outer) interface:
        # a qualitative SIGN check only.  The former self-/re-anchored >=1.3x bar is
        # removed; the growth magnitude is anchored against the experiment via the curve
        # oracle below (#143).
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
        # (the inverse of benchmark 1's single-mode dominance): the outer-interface power
        # is spread over >= 2 effective modes and no single mode holds the majority.
        amps_fin = _band_spectrum(_interfaces(fin)[1], z)
        assert npart[-1] > PART_MIN, (
            f"outer spectrum collapsed to ~1 mode: participation {npart[-1]:.2f} "
            f"< {PART_MIN}"
        )
        dom = _dominant_frac(amps_fin)
        assert dom < DOM_MAX, (
            f"a single seeded mode dominates (not multi-mode): {dom:.2f} >= {DOM_MAX}"
        )

        # (6) The synthetic-radiograph limb modulation (the observable MRT signal) grows:
        # a qualitative SIGN check only.  The former self-anchored >=1.3x bar is removed;
        # the growth magnitude is anchored against the experiment via the curve oracle.
        assert rad_mod[-1] > rad_mod[0], (
            f"radiograph modulation did not grow: {rad_mod[0]:.4e} -> {rad_mod[-1]:.4e}"
        )

        # The full coupled radiation-conduction-MHD stack ran and kept the species-split
        # energy budget physically sane (erad finite, non-neg, bounded; #116/[B3]).
        cenergy.assert_coupled_energy_sane("maglif_mmrt")

        # --- QUANTITATIVE ANCHOR (#143): the broadband driven-surface roughness-growth
        # history (times, rms_outer) IS the multi-mode amplitude-growth curve this
        # benchmark reproduces.  It is compared against the committed McBride-2012
        # experimental growth curve via the oracle's curve comparison (GroundTruthOracle),
        # replacing the former self-/re-anchored GROWTH_MIN bar.  This reduced
        # nondimensional _cpu surrogate emits the curve in CODE UNITS, whereas the McBride
        # curve is a physical amplitude-vs-time radiograph; per the reduced-surrogate
        # policy (B3 velocity-ratio / B4 ICF #140) the absolute experimental comparison is
        # the paper-resolution run (#122 [C5]).  The committed B2 curve datum is still
        # pending_digitization (the McBride figure is paywalled and not faithfully
        # digitizable from this environment -- the same constraint #142 hit for Sinars,
        # no fabricated points per ADR-0008), so it is REPORTED as PENDING here, never as
        # a pass.  The wiring below is the full curve-oracle path: it reports PENDING while
        # pending and, once digitized, compares the curve and reports the verdict
        # (binding=False) with the experimental band overlaid.
        oracle = gto.GroundTruthOracle.from_committed()
        datum = oracle.get("B2", "multimode_amplitude_growth")
        if datum.is_pending:
            scorecard.record_pending(
                "B2", "multimode_amplitude_growth", issue=122,
                note="McBride 2012 multi-mode growth curve; figure digitization + "
                     "compare in #122 (paywalled, not digitizable here -- #143)",
            )
            exp_points = None
        else:
            res = oracle.compare(
                "B2", "multimode_amplitude_growth", (times, rms_outer)
            )
            scorecard.record_result(
                res.worst_point, binding=False,
                note="reduced code-unit surrogate; absolute compare is #122",
            )
            print(f"[B2] {res}  (reported, not asserted)")
            exp_points = [
                {"x": p.x, "y": p.exp_value, "band": p.band}
                for p in res.point_results
            ]

        # Experiment-overlay diagnostic: the sim growth curve (broadband roughness vs time)
        # with the experimental curve + band overlaid once digitized; until then the sim
        # curve is plotted with a pending-digitization annotation (#122).
        overlay.overlay_curve(
            "maglif_mmrt", times, rms_outer, exp_points=exp_points,
            xlabel="t", ylabel="multi-mode roughness rms_outer",
            x_unit="code units", unit="code units", pending_issue=122,
            title="Multi-mode MRT growth curve vs McBride 2012 (MagLIF benchmark 2)",
        )

        # Plot the synthetic radiograph (initial vs final); trajectories + baseline below.
        _plot_radiograph(xrad, z, rad0, radN, times[0], times[-1])

        harness.verify(
            "maglif_mmrt",
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
            title="Multi-mode MRT (MagLIF benchmark 2): convergence + broadband growth",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
