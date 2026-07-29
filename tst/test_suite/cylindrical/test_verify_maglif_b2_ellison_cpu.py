"""
Faithful Ellison benchmark 2 Stage-1 qualitative MRT replication (HARD gate, #163).

REPLACES the multi-mode MRT *surrogate* (test_verify_maglif_mmrt_cpu.py / #122), which
ran a code-unit toy (constant ``i0=3.0`` drive, a manufactured geometric surface-roughness
seed, code-unit geometry, radiation ON) -- it differed from the published Ellison
benchmark 2 on every knob, so it validated against the code's own output, not the
This test runs the *faithful* Stage-1 setup (``tst/inputs/maglif_b2_ellison.athinput``):
the AR=6 beryllium liner (#160 geometry), the measured z2173 Z-machine drive (#160), the
Ellison Eq.16 RANDOM-TEMPERATURE multi-mode seed (delta_T = 100 K, #161 --
``perturbation=temperature``, NOT a roughness spectrum), radiation OFF (ideal-MHD).

Ellison et al. 2025 (arXiv:2504.10760, sec 3.2, validated vs McBride 2012/2013) state the
B2 success criterion QUALITATIVELY: "the prominent characteristics of the MRT features are
in good agreement... the amplitude of the largest mode, the approximate wavelength of the
largest mode, and general qualitative agreement in the macroscopic multimode behavior."
Stage 1 (this test) is that qualitative morphology gate; the soft *quantitative* anchor to
the McBride fit laws is Stage 2 (#155 [B2-S2]).

The temperature seed lays down a per-zone random ELECTRON+ION temperature (a pressure
factor at FIXED density, so it is mass-conserving and the density interface starts EXACTLY
1-D); the noisy pressure then drives differential motion that the inward magnetic
acceleration amplifies into multi-mode MRT structure on the driven interface.
With delta_T = 0 the factor is identically 1 -> the run stays axisymmetric (1-D) -> NO
axial structure: that ``delta_T=0`` control run is the binding discriminator below.

This is the reduced-resolution CPU GATE for the faithful setup: it runs the committed
faithful athinput but OVERRIDES ONLY the mesh resolution and integration length
(``nx*``/``tlim``) to keep the CPU path fast -- the drive, geometry, EOS and seed are the
faithful committed values.  The full 12.5 um ("paper-resolution") uniform run is the
committed athinput itself (the GPU target, as for the other benchmarks).  Per Ellison the
comparison is qualitative, so the gate asserts the MORPHOLOGY SIGNATURE -- structure
develops (delta_T>0) vs stays 1-D (delta_T=0), the dominant mode has a finite amplitude at
an approximate (sub-mm, well-resolved) wavelength, the spectrum stays multi-mode, and the
structure grows as the liner converges toward stagnation -- rather than a digitized-curve
match (that is Stage 2).

Discriminating quantities (interfaces are phi-averaged; the axisymmetric IC makes phi a
replica, and the temperature seed breaks the symmetry along z):
  * R_liner(t)  -- mass-weighted mean liner radius (bulk convergence trajectory).
  * rms_outer(t)-- RMS axial roughness of the driven (liner/vacuum) interface: THE
                   multi-mode MRT amplitude.  Grows for delta_T>0; flat for delta_T=0.
  * rms_inner(t)-- RMS roughness of the fuel-side interface (RT-stabilised in run-in).
  * kdom(t)     -- dominant axial mode number of the driven surface (wavelength Lz/kdom);
                   the largest-mode wavelength Ellison compares.
  * npart(t)    -- spectral participation number (effective # modes): >= 2 == multimode.

NOTE: like the surrogate, this run is INSTABILITY-amplifying (MRT amplifies floating-point
round-off), so the harness regression baseline is exact only for the deterministic SERIAL
same-platform re-run; loosen/refresh it if a cross-platform flake appears.  Auto-collected
by run_test_suite.py (the module name contains ``_cpu``).
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
import test_suite.verification.experiment_overlay as overlay  # noqa: E402
import test_suite.verification.ground_truth_oracle as gto  # noqa: E402
import test_suite.verification.harness as harness  # noqa: E402
import test_suite.verification.mcbride_b2_anchor as mb2  # noqa: E402
import test_suite.verification.scorecard as scorecard  # noqa: E402

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the working directory (the binary runs from build_pgen/maglif/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Faithful committed input (12.5 um artifact); the gate overrides only nx*/tlim below.
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_b2_ellison.athinput"
)
trace_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "z2173_current.dat"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")

# Reduced CPU-gate mesh + integration (faithful drive/geometry/seed unchanged): calibrated
# so the liner converges measurably and the seed develops clear multi-mode MRT structure
# while keeping the CPU run fast; the 12.5 um uniform run is the committed athinput.
NX1 = 64             # radial zones over [0, 3.6] mm (dr ~ 56 um; ~10 cells / liner)
NX3 = 48              # axial zones over [0, 1.6] mm (dz ~ 33 um; Nyquist mode 24)
TLIM = 20.0          # code time (ns): clear growth + ~15% convergence, pre-stagnation
LZ = 1.6              # axial domain length [mm] (matches the athinput x3max)
HALF = 0.5           # half-max density level isolating the dense Be liner

# Qualitative morphology thresholds (sign/shape gates; McBride quant anchor is #155).
CONV_FRAC = 0.95     # require >= 5% bulk convergence (R_liner shrinks; calibrated ~15%)

# Synthetic-radiograph instrument response for the McBride spike/bubble extrema (#229,
# mirroring the #212 B1 fix): McBride 2012's monochromatic radiography has 15 um spatial
# resolution, so the synthetic extrema compared against its fit laws are band-limited to
# the instrument's own band -- otherwise (max - min) also counts grid-scale structure the
# radiograph could never have seen (resolution-divergent, like B1's a_sb before #212).
# At the reduced CPU gate (nx3 48, Nyquist mode 24 < 53) this is a no-op by construction;
# it bites at the committed paper-resolution deck (nx3 128, Nyquist 64 > 53).
RADIOGRAPH_RES_UM = 15.0                                  # McBride 2012 quoted resolution
K_MAX_BAND = int(LZ * 1.0e3 / (2.0 * RADIOGRAPH_RES_UM))  # cycles across LZ (= 53)
STRUCT_MIN = 1.0e-2  # delta_T>0 driven roughness must reach this (calibrated ~5.8e-2)
FLAT_TOL = 1.0e-8    # delta_T=0 control roughness stays below this (calibrated ~4e-16)
GROW_FACTOR = 5.0    # late roughness exceeds early roughness by this (calibrated ~20x)
PART_MIN = 2.0       # driven spectrum stays multi-mode (effective modes; calibrated 8-11)
LAM_MIN = 0.05       # largest-mode wavelength lower bound [mm] (resolved, not grid noise)


def _run(basename, dT):
    """Run the faithful setup at reduced resolution; return sorted snapshot paths."""
    ok = testutils.run_pgen(
        "maglif", input_file,
        args=[
            f"problem/current_file={trace_file}",
            f"problem/pert_dT={dT}",
            f"mesh/nx1={NX1}", f"mesh/nx3={NX3}",
            f"meshblock/nx1={NX1}", f"meshblock/nx3={NX3}",
            f"time/tlim={TLIM}", "output1/dt=5.0",
            f"job/basename={basename}",
        ],
    )
    assert ok, f"faithful Ellison B2 run failed (basename={basename}, dT={dT})"
    files = sorted(glob.glob(os.path.join(bin_dir, f"{basename}.prim.*.bin")))
    assert len(files) >= 4, f"too few snapshots for {basename}: {len(files)}"
    return files


def _cross(r, p, i, j):
    """Sub-cell radius where the density profile p crosses HALF between cells i and j."""
    return r[i] + (HALF - p[i]) / (p[j] - p[i]) * (r[j] - r[i])


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
        idx = np.where(p >= HALF)[0]
        if len(idx) == 0:
            continue
        w = np.where(p >= HALF, p, 0.0)
        r_mw[kz] = np.sum(w * r * r) / np.sum(w * r)
        lo, hi = idx[0], idx[-1]
        r_in[kz] = r[lo] if lo == 0 else _cross(r, p, lo - 1, lo)
        r_out[kz] = r[hi] if hi == len(r) - 1 else _cross(r, p, hi, hi + 1)
    return r_in, r_out, r_mw


def _rms(rz):
    """RMS amplitude of the mean-subtracted interface profile r(z) -- axial roughness."""
    good = np.isfinite(rz)
    if not good.any():
        return 0.0
    rr = rz[good] - np.mean(rz[good])
    return float(np.sqrt(np.mean(rr * rr)))


def _spectrum(rz, z):
    """Per-mode magnitude over axial modes k = 1..nz//2; returns (ks, amps)."""
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    zz = z[good]
    ks = np.arange(1, len(z) // 2)
    amps = np.array([
        np.hypot(2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz / LZ)),
                 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz / LZ)))
        for k in ks
    ])
    return ks, amps


def _kdom(rz, z):
    """Dominant axial mode number of the interface r(z) (0 if no finite structure)."""
    ks, amps = _spectrum(rz, z)
    return int(ks[np.argmax(amps)]) if amps.size and amps.max() > 0.0 else 0


def _participation(rz, z):
    """Participation number (sum p)^2 / sum p^2, p = amps^2: effective # modes."""
    _, amps = _spectrum(rz, z)
    p = amps ** 2
    s = float(p.sum())
    return float(s * s / np.sum(p * p)) if s > 0.0 else 0.0


def _trajectory(files):
    """Reduce a run's snapshots to the discriminating time series.

    Also returns the driven-surface spike/bubble extrema per snapshot (R_spike=max,
    R_bubble=min of the finite outer interface, band-limited to the radiograph's own
    resolution, #229) -- the inputs to the McBride soft quantitative anchor
    (#155 [B2-S2]); they do NOT enter the regression baseline.
    """
    z = None
    times, r_liner, rms_outer, rms_inner, kdom, npart = [], [], [], [], [], []
    r_spike, r_bubble = [], []
    for fp in files:
        d = bin_convert.read_binary_as_athdf(fp)
        if z is None:
            z = np.asarray(d["x3v"], dtype=float)
        r_in, r_out, r_mw = _interfaces(d)
        times.append(float(d["Time"]))
        r_liner.append(float(np.nanmean(r_mw)))
        rms_outer.append(_rms(r_out))
        rms_inner.append(_rms(r_in))
        kdom.append(_kdom(r_out, z))
        npart.append(_participation(r_out, z))
        sp, bu = mb2.outer_surface_extrema(r_out, k_max=K_MAX_BAND)
        r_spike.append(sp)
        r_bubble.append(bu)
    return (z, np.array(times), np.array(r_liner), np.array(rms_outer),
            np.array(rms_inner), np.array(kdom, dtype=float), np.array(npart),
            np.array(r_spike), np.array(r_bubble))


def _report_mcbride_anchor(times, r_spike, r_bubble, kdom):
    """SOFT (report-only) quantitative anchor to the McBride 2012 fit laws (#155 [B2-S2]).

    Ellison B2 is a QUALITATIVE benchmark (the asserts above are the binding gate); this
    block REPORTS -- never asserts -- the dimensionless growth fraction against McBride's
    stated [0.05, 0.15] band (scorecard + overlay), and reports the amplitude(um) /
    wavelength(um) fit laws.  A persistent miss is escalated as a "needs investigation"
    note (#155 Stage 3), not a hard fail (FLASH matched B2, so a miss is a diagnostic
    signal about our setup).  Returns ``(growth_fraction_final, x_final)`` for the log.
    """
    oracle = gto.GroundTruthOracle.from_committed()
    laws = mb2.fit_laws(oracle.meta("B2"))

    # Initial outer radius R0 = R_outer(0): the interface starts EXACTLY 1-D (the seed
    # perturbs pressure at fixed density), so spike == bubble at t=0.
    r0 = 0.5 * (float(r_spike[0]) + float(r_bubble[0]))
    gf = np.array([mb2.growth_fraction(s, b, r0)
                   for s, b in zip(r_spike, r_bubble)])
    x_conv = np.array([mb2.convergence_x(b, r0) for b in r_bubble])
    gf_final = float(gf[-1])
    x_final = float(x_conv[-1])

    # --- AC#1: growth fraction vs the McBride [0.05, 0.15] band (reported only) ---
    # The band is a published claim ONLY over McBride's measured abscissa domain
    # x in [0.4, 0.95] (the nonlinear regime near stagnation; #229): McBride's own
    # laws imply gf ~ (450x-90)/(R0*x) ~ 0.008 at the gate's x~0.214, so an
    # out-of-window comparison cannot be met even by the reference data.  Below the
    # window the measured value is recorded PENDING (deeper-convergence run needed),
    # not FAIL; inside it the band comparison stands (still report-only, #155).
    datum = oracle.get("B2", "growth_fraction")
    if mb2.in_validity_window(x_final, laws):
        res = oracle.compare("B2", "growth_fraction", gf_final)
        note = f"reduced CPU gate, deepest convergence x~{x_final:.3f}"
        if not res.passed:
            # Stage-3 escalation: a persistent in-window miss is a diagnostic note.
            note = (f"needs investigation (#155 Stage 3): {note}; FLASH matched B2, "
                    f"so a persistent miss flags our setup, not a regression")
        scorecard.record_result(res, binding=False, note=note)
    else:
        lo, hi = laws["x_valid_range"]
        scorecard.record(
            "B2", datum.observable, gf_final, datum.value, datum.tolerance_band(),
            datum.unit, scorecard.PENDING, binding=False,
            note=(f"out of the McBride validity window: deepest convergence "
                  f"x~{x_final:.3f} not in [{lo}, {hi}] (band stated for the "
                  f"nonlinear regime near stagnation); comparison deferred to a "
                  f"deeper-convergence run (#229)"))

    # gf(t) overlay against the stated band (NaN early points -- before any convergence,
    # R_bubble == R0 -- are dropped so the band line is the focus).
    finite = np.isfinite(gf)
    overlay.overlay_scalars(
        "maglif_b2_growth_fraction",
        np.asarray(times)[finite],
        {"growth_fraction": gf[finite]},
        {"growth_fraction": {"exp_value": 0.10, "band": 0.05, "unit": "",
                             "sim_value": gf_final}},
        xlabel="t [ns]",
        title="Faithful Ellison B2: MRT growth fraction vs McBride [0.05, 0.15]",
    )

    # --- AC#2: amplitude(um) / wavelength(um) vs the McBride fit laws ---
    # The geometry is dimensional (1 code length = 1 mm), so the reduced run yields
    # amplitude/wavelength in um.  BUT amplitude_um = 450*x - 90 is only physical for
    # x > 0.2 and the coarse CPU gate under-resolves um-scale amplitudes, so the
    # *quantitative* law match needs the paper-resolution dimensional SI run -> reported
    # PENDING that GPU artifact, with a diagnostic overlay of the reduced-run values.
    amp = laws["amplitude_um"]
    wav = laws["wavelength_um"]
    amp_meas = (np.asarray(r_spike) - np.asarray(r_bubble)) * 1.0e3      # mm -> um
    kf = np.asarray(kdom, dtype=float)
    lam_meas = np.where(kf > 0, LZ / np.where(kf > 0, kf, 1.0) * 1.0e3, np.nan)
    amp_law = np.array([mb2.linear_law(x, amp["slope"], amp["intercept"])
                        for x in x_conv])
    lam_law = np.array([mb2.linear_law(x, wav["slope"], wav["intercept"])
                        for x in x_conv])
    scorecard.record_pending(
        "B2", "MRT amplitude um vs McBride 450x-90", issue=123,
        note=(f"reduced CPU gate at x~{x_final:.3f} under-resolves the um-scale "
              f"amplitude; quantitative law match needs the paper-res SI run"))
    scorecard.record_pending(
        "B2", "MRT wavelength um vs McBride 750x", issue=123,
        note="needs the paper-resolution dimensional SI run")
    _plot_fit_laws(x_conv, amp_meas, amp_law, lam_meas, lam_law)
    return gf_final, x_final


def _plot_fit_laws(x_conv, amp_meas, amp_law, lam_meas, lam_law):
    """Diagnostic overlay: reduced-run amplitude/wavelength (um) vs McBride fit laws."""
    harness._ensure_dirs()
    good = np.isfinite(x_conv)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(x_conv[good], amp_meas[good], "-o", ms=3, color="C0",
                 label="reduced CPU gate")
    axes[0].plot(x_conv[good], amp_law[good], "--", color="C3",
                 label="McBride 450x-90")
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_ylabel("amplitude [um]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    fin = good & np.isfinite(lam_meas)
    axes[1].plot(x_conv[fin], lam_meas[fin], "-o", ms=3, color="C0",
                 label="reduced CPU gate")
    axes[1].plot(x_conv[good], lam_law[good], "--", color="C3",
                 label="McBride 750x")
    axes[1].set_ylabel("wavelength [um]")
    axes[1].set_xlabel("convergence x = 1 - R_bubble/R_outer(0)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Faithful Ellison B2: amplitude/wavelength vs McBride fit laws "
                 "(report-only; SI run needed for x>0.2)")
    fig.tight_layout()
    fig.savefig(os.path.join(harness.PLOT_DIR, "maglif_b2_fit_laws.png"), dpi=110)
    plt.close(fig)


def _plot_snapshot(files, z):
    """Density r-z snapshot (initial vs final) showing the developed MRT morphology."""
    harness._ensure_dirs()
    d0 = bin_convert.read_binary_as_athdf(files[0])
    dN = bin_convert.read_binary_as_athdf(files[-1])
    r = np.asarray(d0["x1v"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, d, ttl in ((axes[0], d0, "initial"), (axes[1], dN, "final")):
        rho = np.asarray(d["dens"], dtype=float).mean(axis=1)   # (nz, nr) phi-averaged
        im = ax.pcolormesh(r, z, rho, shading="auto", cmap="inferno")
        ax.set_title(f"{ttl}  (t={float(d['Time']):.1f})")
        ax.set_xlabel("radius r [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$")
    axes[0].set_ylabel("axial z [mm]")
    fig.suptitle("Faithful Ellison B2 (multi-mode MRT, delta_T=100 K): density r-z")
    fig.tight_layout()
    fig.savefig(os.path.join(harness.PLOT_DIR, "maglif_b2_ellison_density.png"), dpi=110)
    plt.close(fig)


def test_verify_maglif_b2_ellison():
    """Faithful Ellison B2: structure develops (delta_T>0) vs stays 1-D (delta_T=0)."""
    try:
        # --- faithful run (delta_T = 100 K, the Ellison B2 seed) ---
        files = _run("maglif_b2_ellison", 100.0)
        (z, times, r_liner, rms_outer, rms_inner,
         kdom, npart, r_spike, r_bubble) = _trajectory(files)

        # Final snapshot is finite (the implosion ran in cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # The seed perturbs PRESSURE at fixed density, so the interface starts 1-D.
        assert rms_outer[0] < FLAT_TOL, (
            f"driven interface not initially 1-D: rms_outer[0]={rms_outer[0]:.3e}"
        )

        # (1) The liner actually converges (bulk implosion under the z2173 drive).
        assert r_liner[-1] < CONV_FRAC * r_liner[0], (
            f"liner did not converge: R0={r_liner[0]:.4f} -> R={r_liner[-1]:.4f}"
        )

        # (2) Multi-mode structure DEVELOPS on the driven interface from the temperature
        # seed (the faithful Ellison signature): the driven-surface roughness reaches a
        # finite, macroscopic amplitude.
        assert rms_outer[-1] > STRUCT_MIN, (
            f"no multi-mode structure developed: rms_outer[-1]={rms_outer[-1]:.3e} "
            f"<= {STRUCT_MIN}"
        )

        # (3) The structure GROWS as the liner converges toward stagnation:
        # the late roughness far exceeds the early (post-seed) roughness, and grows
        # monotonically over the final stretch.
        early = rms_outer[1]   # first post-IC snapshot (seed has imprinted)
        assert rms_outer[-1] > GROW_FACTOR * early, (
            f"structure did not grow toward stagnation: {early:.3e} -> "
            f"{rms_outer[-1]:.3e}"
        )
        assert rms_outer[-1] > rms_outer[-2] > rms_outer[-3], (
            f"roughness not increasing toward stagnation: ...{rms_outer[-3:]} "
        )
        # (rms_inner is recorded as a diagnostic but NOT asserted against rms_outer: the
        # driven surface leads during run-in, but as the liner nears stagnation the fuel-
        # side interface also roughens -- feedthrough + small-radius amplification -- so
        # the outer>inner "MagLIF asymmetry" is a run-in signature, not a stagnation one,
        # and is not one of the binding Ellison B2 Stage-1 criteria.)

        # (4) The largest mode sits at an APPROXIMATE wavelength in the right qualitative
        # regime -- a real, well-resolved sub-mm axial mode (not grid noise, not a single
        # whole-box mode) -- and the spectrum stays MULTI-MODE (Ellison's "macroscopic
        # multimode behavior", the inverse of single-mode dominance).
        kdf = int(kdom[-1])
        lam = LZ / kdf if kdf > 0 else 0.0
        assert 1 <= kdf <= NX3 // 4, (
            f"largest mode not in the resolved regime: kdom={kdf} (need 1..{NX3 // 4})"
        )
        assert LAM_MIN < lam <= LZ, (
            f"largest-mode wavelength {lam:.3f} mm outside qualitative regime "
            f"({LAM_MIN}..{LZ}]"
        )
        assert npart[-1] >= PART_MIN, (
            f"driven spectrum collapsed to ~1 mode: participation {npart[-1]:.2f} "
            f"< {PART_MIN}"
        )

        # (5) DISCRIMINATOR: the SAME drive+geometry+convergence with delta_T=0 stays 1-D
        # (axisymmetric, NO axial structure) -- proving the morphology comes from
        # the Ellison temperature seed, not the numerics.  This is the binding control.
        cfiles = _run("maglif_b2_ellison_ctl", 0.0)
        _, ctimes, cr_liner, crms_outer, _, _, _, _, _ = _trajectory(cfiles)
        assert crms_outer.max() < FLAT_TOL, (
            f"delta_T=0 control developed axial structure (max rms_outer="
            f"{crms_outer.max():.3e} >= {FLAT_TOL}); not 1-D without the seed"
        )
        # The control converges like the seeded run (same drive) -- it is a true control.
        assert cr_liner[-1] < CONV_FRAC * cr_liner[0], (
            f"control liner did not converge: {cr_liner[0]:.4f} -> {cr_liner[-1]:.4f}"
        )

        # Density r-z snapshot (initial vs developed multi-mode morphology).
        _plot_snapshot(files, z)

        # Stage-2 SOFT quantitative anchor (#155 [B2-S2]): report the growth fraction vs
        # the McBride [0.05, 0.15] band + the amplitude/wavelength fit laws on this same
        # faithful Stage-1 run.  REPORT-ONLY -- never asserts (Ellison B2 is qualitative;
        # the binding gate is the morphology signature above), so it cannot fail the test.
        gf_final, x_final = _report_mcbride_anchor(times, r_spike, r_bubble, kdom)
        print(f"[B2-S2] McBride soft anchor: growth_fraction={gf_final:.4f} "
              f"at convergence x={x_final:.4f} (band [0.05, 0.15], report-only)")

        # Regression baseline: the faithful-run trajectory (captured on first run).
        harness.verify(
            "maglif_b2_ellison",
            times,
            {
                "R_liner": r_liner,
                "rms_outer": rms_outer,
                "rms_inner": rms_inner,
                "npart": npart,
            },
            # kdom is omitted from the regression baseline: it is a DISCRETE spectral
            # argmax that can flip discontinuously under round-off; it is gated
            # qualitatively (the resolved-regime range check above) instead.
            coord_label="t",
            xlabel="t [ns]",
            title="Faithful Ellison B2 MRT: convergence + seed-driven growth",
            rtol=1.0e-4,
            atol=1.0e-8,
        )
    finally:
        for pat in ("maglif_b2_ellison.*", "maglif_b2_ellison_ctl.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        shutil.rmtree(bin_dir, ignore_errors=True)
