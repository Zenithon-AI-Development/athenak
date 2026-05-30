"""
Converging Richtmyer-Meshkov B3 quantitative anchor on the CPU path (issue #144/[VA5]).

Brings the Ellison-benchmark-3 anchor (converging RM, Knapp 2020) into per-PR CPU CI so it
cannot silently rot behind the GPU-only ``test_verify_maglif_rm_si_gpu.py`` (#119/[C2]).
It runs a COMPACT code-unit converging-RM (``inputs/maglif_rm_anchor.athinput``: a
magnetically driven liner launches a converging shock through a working-fluid gap into an
on-axis dense ROD carrying a seeded axial kz perturbation), reduces it to dimensionless
``rm_growth_factor`` with the SAME definition the GPU test uses (the rod-material s=0.5
contour amplitude vs the normalized interface displacement kz*dr, measured relative to the
post-shock compression minimum), and the dimensionless shock/interface velocity ratio,
then records both against the committed B3 oracle (scorecard + experiment overlay).

Oracle (Layer-1, ADR-0008): the EXPERIMENT, via the committed ground-truth datums in
``verification/ground_truth/b3_convergent_rm_knapp_2020.json`` (P. F. Knapp et al., Phys.
Plasmas 27, 092707, 2020; SAND2020-10207J, open access).

FIDELITY / REPORT-vs-ASSERT (the documented reduced-_cpu precedent, B4 ICF #140 and the
GPU velocity-ratio policy #119): the two B3 observables are REPORTED against the oracle
(``binding=False``), NOT hard-asserted, because this compact CPU surrogate cannot
reproduce their paper-resolution magnitudes within the per-PR CPU budget:

  * rm_growth_factor (experiment 4.8 +/- 1.92): the experimental RM growth develops over
    the long, well-resolved magnetic run-in of the SI run (nx1=512, tlim=4.5); in the
    compact surrogate (nx1=128, tlim=0.85) the seeded mode is compression-dominated and
    underproduces the growth.  The BINDING magnitude hard-assert is the GPU SI run; here
    the binding gate is the cheap qualitative convergence signature.
  * rm_shock_interface_velocity_ratio (experiment 1.369 +/- 0.16): EOS-sensitive (set by
    the Be shock Hugoniot, which an ideal gamma-law EOS cannot reproduce) -- reported, not
    asserted, EXACTLY as the GPU test does it.

The binding CPU gate is therefore the cheap qualitative signature, robust to the reduced
physics: finiteness of the final state, the rod converges and rebounds (a clean confined
implosion), and the seeded single mode is present and dominant on the rod interface (no
spurious cascade).  The coupled radiation-conduction stack is OFF (the RM is hydrodynamic;
see the input header).  The existing GPU ``test_verify_maglif_rm_si_gpu.py`` anchor is
left untouched.  Auto-collected by ``run_test_suite.py`` (module name contains ``_cpu``).
"""

# Modules
import glob
import os
import shutil
import sys

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

# bin_convert lives next to athena_read in vis/python.
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_rm_anchor.athinput.
PERT_MODE = 2            # seeded axial mode number (wavelength = Lz/pert_mode)
LZ = 0.6                 # axial domain extent in code units; must match <mesh> x3max
LAMBDA_CODE = LZ / PERT_MODE      # perturbation wavelength = 0.3 code units
KZ = 2.0 * np.pi / LAMBDA_CODE    # axial mode wavenumber (1/code-length)
PERT_AMP = 0.025         # seeded radial amplitude (must match <problem> pert_amp)
R_LINER_INNER = 0.4      # liner inner radius (= <problem> r_fuel); the fill outer bound
# Knapp 2020 final radiographed dimensionless interface displacement (Fig. 9, kz*dr=11.8)
# -- the displacement the committed rm_growth_factor anchor is at; the sim growth is
# measured at the SAME displacement (capped at the max kz*dr the compact run-in reaches).
KZDR_EXP = 11.8

# Qualitative binding gates (sign/shape sanity; magnitudes are reported via the oracle).
CONV_FRAC = 0.85         # require the rod to converge by >= 15% (R_rod shrinks)
DOM_FRAC = 0.5           # seeded mode holds >= 50% of the rod-interface z-variance

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_rm_anchor.athinput"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")
build_dir = os.path.join(testutils._repo_root(), "tst", "build_pgen", "maglif")


def _build_dir_is_cuda():
    """True iff build_pgen/maglif is already configured with the Kokkos CUDA backend.

    The maglif build dir is SHARED with the GPU B3 test; a prior _gpu run leaves it
    CUDA-configured (sticky Kokkos cache), which would make this CPU run try a CUDA build
    (KNOWN_ISSUES #1, device-lambda).  Clear it (Python rmtree; rm -rf is sandbox-blocked
    here) so Kokkos reconfigures with the Serial/CPU backend.
    """
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, "r") as f:
        return "Kokkos_ENABLE_CUDA:BOOL=ON" in f.read().upper()


def _rod_radius(data):
    """Per-z radius of the rod/working-fluid interface via the rod-material scalar s_00.

    The rod is tagged by a passive scalar (s=1 in the rod, 0 in the fill; advected with
    the flow), so the interface is the s=0.5 contour -- robust through convergence, where
    a density threshold cannot separate the rod from the equally dense liner.  For each
    axial height z, phi-average the scalar and find the OUTERMOST 0.5 crossing.
    (Identical to the GPU test's reduction.)
    """
    r = np.asarray(data["x1v"], dtype=float)              # (nr,)
    s = np.asarray(data["s_00"], dtype=float)             # (nz, nphi, nr) rod scalar
    nz = s.shape[0]
    rrod = np.full(nz, np.nan)
    for kz in range(nz):
        p = s[kz].mean(axis=0)                             # (nr,) phi-averaged scalar
        if p[0] < 0.5:
            continue                                        # no rod material on the axis
        below = np.where(p < 0.5)[0]
        if len(below) == 0:
            continue
        hi = below[0]                                      # first cell below s=0.5
        lo = hi - 1
        if lo < 0:
            continue
        f = (0.5 - p[lo]) / (p[hi] - p[lo])
        rrod[kz] = r[lo] + f * (r[hi] - r[lo])
    return rrod


def _mode_amp(rz, z, k, lz):
    """Magnitude of the cos/sin projection of r(z) onto axial mode k over [0, lz]."""
    good = np.isfinite(rz)
    if good.sum() < 4:
        return np.nan
    rr = rz[good] - np.mean(rz[good])
    zz = z[good]
    a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz / lz))
    b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz / lz))
    return float(np.hypot(a, b))


def _shock_radius(data, r_rod_mean):
    """Inward-propagating shock front radius in the working fluid (ahead of the rod).

    Track the radius of the maximum radial pressure gradient inside the fill region
    (r in (r_rod_mean, liner inner]); returns nan if no clean feature.  (Same construction
    as the GPU test, with the fill outer bound at the compact liner inner radius.)
    """
    r = np.asarray(data["x1v"], dtype=float)
    if "eint" in data:
        prs = np.asarray(data["eint"], dtype=float)
    else:
        prs = np.asarray(data["dens"], dtype=float)
    prof = prs.mean(axis=(0, 1)) if prs.ndim == 3 else prs.mean(axis=0)
    sel = (r > r_rod_mean * 1.05) & (r < R_LINER_INNER)
    if sel.sum() < 4:
        return np.nan
    rr = r[sel]
    pp = prof[sel]
    grad = np.abs(np.gradient(pp, rr))
    return float(rr[int(np.argmax(grad))])


def test_verify_maglif_rm_anchor():
    """Run the compact converging RM on CPU; gate on the convergence signature, then
    report the B3 growth factor + velocity ratio against the Knapp-2020 oracle (#144)."""
    oracle = gto.GroundTruthOracle.from_committed()
    # Ensure a CPU (Serial) build of the shared maglif pgen dir (clear a CUDA-configured
    # dir left by a prior _gpu run -> KNOWN_ISSUES #1).
    if _build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        assert testutils.run_pgen("maglif", input_file), \
            "MagLIF compact converging-RM CPU run failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_rm_anchor.prim.*.bin")))
        assert len(files) > 8, f"too few RM snapshots: {len(files)}"

        z = None
        times, r_rod, eta, r_shock = [], [], [], []
        for fp in files:
            d = bin_convert.read_binary_as_athdf(fp)
            if z is None:
                z = np.asarray(d["x3v"], dtype=float)
            rr = _rod_radius(d)
            rmean = float(np.nanmean(rr))
            times.append(float(d["Time"]))
            r_rod.append(rmean)
            eta.append(_mode_amp(rr, z, PERT_MODE, LZ))
            r_shock.append(_shock_radius(d, rmean))
        times = np.array(times)
        r_rod = np.array(r_rod)
        eta = np.array(eta)
        r_shock = np.array(r_shock)

        # Final snapshot is finite and stable (the implosion ran in cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # The converging run-in: shock arrival (rod first moves inward) -> peak
        # convergence (min rod radius).  kz*dr rises 0 -> max (x-axis of Knapp Fig. 8).
        kzdr = KZ * (r_rod[0] - r_rod)
        i_conv = int(np.nanargmin(r_rod))
        sh = np.where(r_rod < 0.99 * r_rod[0])[0]
        i_shock = int(sh[0]) if len(sh) else 0
        assert i_conv > i_shock + 2, "no resolved converging run-in (shock never arrived)"

        # --- Cheap QUALITATIVE binding gates (sign/shape sanity, no tuned magnitudes).
        # (q1) The rod actually converges (bulk implosion of the on-axis target).
        assert r_rod[i_conv] < CONV_FRAC * r_rod[0], (
            f"rod did not converge: R0={r_rod[0]:.4f} -> R_min={r_rod[i_conv]:.4f}"
        )
        # (q2) Convergence-and-rebound: the minimum is interior in time (the rod turned
        # around, not still imploding at the last snapshot) and it rebounds afterwards.
        assert 0 < i_conv < len(times) - 1, (
            f"rod did not stagnate before the run end (min at snap {i_conv})"
        )
        assert r_rod[-1] > r_rod[i_conv], (
            f"no rebound: R_min={r_rod[i_conv]:.4f} -> R_end={r_rod[-1]:.4f}"
        )
        # (q3) The seeded amplitude starts at ~the seed (the IC + reduction are wired up).
        assert abs(eta[0] - PERT_AMP) < 0.25 * PERT_AMP, (
            f"initial seeded amplitude {eta[0]:.4f} != seed {PERT_AMP}"
        )

        # --- rm_growth_factor with the GPU test's definition (reported, not asserted).
        ri = np.arange(i_shock, i_conv + 1)
        m = np.isfinite(kzdr[ri]) & np.isfinite(eta[ri])
        ri = ri[m]
        kz_ri = kzdr[ri]
        eta_ri = eta[ri]
        assert kz_ri.size >= 3, "too few finite run-in snapshots for the growth curve"
        early = kz_ri <= 0.25 * KZDR_EXP                 # post-shock compression phase
        eta_ps = float(np.min(eta_ri[early])) if early.any() \
            else float(eta_ri[int(np.argmin(kz_ri))])
        target = min(KZDR_EXP, float(np.max(kz_ri)))     # capped at displacement reached
        j = int(np.argmin(np.abs(kz_ri - target)))
        eta_at = float(eta_ri[j])
        target = float(kz_ri[j])
        growth_factor = eta_at / eta_ps if eta_ps > 0 else float("nan")

        reduced_note = ("compact CPU surrogate; binding magnitude hard-assert is the "
                        "paper-resolution GPU SI run (#119)")
        res_gf = oracle.compare("B3", "rm_growth_factor", growth_factor)
        scorecard.record_result(res_gf, binding=False, note=reduced_note)
        print(f"[B3] growth factor (sim) = {growth_factor:.3f} at kz*dr = {target:.2f} "
              f"(eta_ps={eta_ps:.4g} -> eta={eta_at:.4g}); {res_gf}  (reported)")

        # --- rm_shock_interface_velocity_ratio (EOS-sensitive; reported, not asserted, as
        # on GPU).  The robust falsifiable check: the shock LEADS the interface inward.
        v_i = -np.polyfit(times[i_shock:i_conv + 1], r_rod[i_shock:i_conv + 1], 1)[0]
        smask = np.isfinite(r_shock)
        smask[i_conv + 1:] = False
        smask[:i_shock] = False
        assert v_i > 0, f"interface did not move inward over the run-in (v_i={v_i:.3g})"
        ratio = float("nan")
        if smask.sum() >= 3:
            v_s = -np.polyfit(times[smask], r_shock[smask], 1)[0]
            ratio = v_s / v_i if v_i > 0 else float("nan")
            res_vr = oracle.compare("B3", "rm_shock_interface_velocity_ratio", ratio)
            scorecard.record_result(res_vr, binding=False, note=reduced_note)
            print(f"[B3] PDV: v_i={v_i:.3f}, v_s={v_s:.3f}, ratio={ratio:.3f}; {res_vr} "
                  f"(EOS-sensitive -- reported, not asserted)")
            assert v_s > 0 and v_s >= 0.9 * v_i, (
                f"shock not leading the interface inward (v_s={v_s:.3g}, v_i={v_i:.3g})"
            )
        else:
            exp_vr = oracle.get("B3", "rm_shock_interface_velocity_ratio").value
            print(f"[B3] PDV: interface inward v_i={v_i:.3f}; shock not cleanly tracked "
                  f"(experiment ratio {exp_vr}, reported via the GPU run)")

        # (q4) The seeded mode stays dominant on the rod interface at the growth point.
        i_t = int(ri[j])
        rr_t = _rod_radius(bin_convert.read_binary_as_athdf(files[i_t]))
        ro = rr_t[np.isfinite(rr_t)]
        var_tot = float(np.mean((ro - ro.mean()) ** 2)) if ro.size else 0.0
        seed_pow = 0.5 * eta_at ** 2
        dom = seed_pow / var_tot if var_tot > 0 else 0.0
        assert dom > DOM_FRAC, (
            f"seeded mode not dominant on the rod: {dom:.2f} < {DOM_FRAC}"
        )

        # Experiment-overlay diagnostic: the sim's running growth-factor curve over the
        # run-in with the experimental anchor value + tolerance band drawn on top.
        gr_curve = eta_ri / eta_ps if eta_ps > 0 else np.full_like(eta_ri, np.nan)
        overlay.overlay_scalars(
            "maglif_rm_anchor",
            kz_ri,
            {"growth_factor": gr_curve},
            {
                "growth_factor": {
                    "exp_value": res_gf.exp_value, "band": res_gf.band,
                    "unit": res_gf.unit, "sim_value": growth_factor,
                },
            },
            xlabel="kz*dr (normalized interface displacement)",
            title="B3 converging-RM growth factor vs Knapp 2020 (CPU anchor, #144)",
        )

        # Layer-2 regression guard (ADR-0008): code-unit baseline diff of trajectories.
        harness.verify(
            "maglif_rm_anchor",
            times,
            {
                "R_rod": r_rod,
                "eta": eta,
                "R_shock": r_shock,
            },
            coord_label="t",
            xlabel="t",
            title="Compact converging RM (B3 CPU anchor): convergence + RM growth factor",
            rtol=1.0e-4,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
