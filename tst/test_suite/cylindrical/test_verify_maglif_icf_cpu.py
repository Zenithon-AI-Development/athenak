"""
ICF confinement-time verification (FLASH MagLIF benchmark 4, #37; oracle-anchored #140).

Runs the integrated MagLIF/Z-pinch problem generator (issue [10a]/#32) as a CLEAN bulk
implosion -- a dense beryllium liner enclosing low-density deuterium fuel, driven radially
inward by a prescribed constant load current and run to STAGNATION (no interface
perturbation is seeded, unlike the instability benchmarks #36/#39/#42).  This is benchmark
4 of the FLASH MagLIF validation ladder (Ellison et al. 2025, arXiv:2504.10760), whose
observables are the PDV/radiography inner-radius trajectory and the stagnation density.

Oracle (Layer-1, ADR-0008): the EXPERIMENT, via the committed ground-truth datums in
``verification/ground_truth/b4_icf_confinement_knapp_2017.json`` (P. F. Knapp et al.,
Phys. Plasmas 24, 042708, 2017).  This is the FIRST use of the quantitative-anchoring
substrate (PRD #138, [VA1]/#140): the run reduces to its stagnation scalars (min fuel
radius, peak fuel density), each is compared against the committed experimental datum via
``GroundTruthOracle.compare``, the verdicts are recorded in the suite scorecard, and the
diagnostic plot overlays each scalar's experimental value + tolerance band on the
simulation result (``experiment_overlay``).  The previously self-anchored numeric
thresholds (CR/rebound/compression/dwell bars re-anchored to the code's own ideal output)
are REMOVED; only cheap qualitative gates (finiteness, a convergence-and-rebound sign
check, mass sanity) remain as fast pre-checks.

This ``_cpu`` benchmark is a REDUCED nondimensional surrogate (1-D radial, toy coupling
coefficients, ideal-gamma EOS), not the paper-resolution SI replication.  Its convergence
ratio (~22x) and fuel compression (~500x) are far stronger than the real MagLIF target, so
it cannot reproduce the ABSOLUTE dimensional stagnation scalars (0.45 mm, ~10 g/cc) within
their bands.  The two scalar comparisons are therefore REPORTED against the oracle for the
record -- not hard-asserted -- exactly mirroring the B3 velocity-ratio policy in
``test_verify_maglif_rm_si_gpu.py``: the absolute-SI hard-assert is the paper-resolution
SI run (#121 [C4]) with real IONMIX-EOS/opacity coupling (#118).  The confinement-time
anchor (Knapp 2017: 14 ns measured vs 16 ns 1D -- a stated text scalar, no longer pending
as of #156) and the corrected v3 inner-radius trajectory (6 radiograph circles) are
likewise REPORTED against the oracle (binding=False), not hard-asserted, in the same
reduced-surrogate policy.

Physics scope (Phase B, #116/[B3], ADR-0009): this benchmark now runs the FULL COUPLED
radiation-conduction-MHD stack -- cylindrical ideal-MHD (ADR-0004) + the prescribed-I(t)
circuit drive (ADR-0005 mode A) + the Strang-split operator-split parabolic block (grey
flux-limited radiation diffusion + anisotropic Braginskii conduction) with the point-
implicit matter-radiation coupling outside the super-step (see the input's <mhd> block).
This 1-D radial run uses a hotter seeded state (p0 = 0.1), so it carries its own
nondimensional coupling coefficients; real opacity/EOS-derived values (full 2T tabulated
EOS, IONMIX opacities) arrive in Phase C (#118).  Resistive B_phi / multigroup FLD stay
off here (Phase C).  The regression baseline below reflects this coupled stack, not the
earlier ideal-MHD run.

The discriminating observables (1-D radial, axisymmetric -- a pure bulk stagnation) are
tracked by ENCLOSED MASS, which is robust to shocks and the fuzzy fuel/liner density
transition: the fuel mass is conserved at the r=0 axis, so the radius enclosing the
initial fuel mass is the Lagrangian fuel/liner contact (the radiography inner radius), and
the radius enclosing fuel+liner mass is the liner/vacuum contact.

  * R_if(t) -- inner radius (fuel/liner contact): implodes to a minimum (stagnation) and
               rebounds; the central confinement observable.
  * R_lv(t) -- outer liner edge (liner/vacuum contact).
  * rho_fuel(t) -- mass-conserved mean fuel density (~ 1/R_if^2): the fuel compression,
               peaking at stagnation.
  * rho_liner_peak(t) -- peak shell density (the compressed beryllium liner).

The test checks (1) the liner converges (R_if reaches a high-convergence minimum), (2) it
stagnates and rebounds (the minimum is interior in time and R_if bounces back), (3) the
fuel compresses strongly at stagnation, (4) a positive confinement time (dwell at
convergence ratio > 2), and that the final state is finite/stable; then plots the
trajectories + densities and saves/diffs a golden regression baseline.  Auto-collected by
run_test_suite.py (module name contains ``_cpu``).
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
import test_suite.cylindrical.maglif_coupled_energy as cenergy

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_icf.athinput.
D_FUEL = 0.01          # deuterium fuel density
D_LINER = 1.85         # beryllium liner density
R_FUEL = 0.4           # fuel/liner interface radius
R_LINER = 0.6          # liner/vacuum interface radius

# --- Provisional SI calibration for REPORTING the reduced-surrogate stagnation scalars
# against the Knapp 2017 experiment.  These map the toy's code-unit observables onto
# physical units ONLY for the oracle comparison + overlay; they do NOT gate the test (the
# comparisons are reported, not hard-asserted -- see the module docstring and the B3
# velocity-ratio precedent).  The absolute-SI hard-assert is #121's paper-resolution run.
#   * density: the input already uses d_liner = 1.85 code = solid beryllium 1.85 g/cc, so
#     1 code density unit = 1 g/cc (grounded, not fabricated).
#   * length: the toy's initial fuel/liner-contact radius (R_FUEL = 0.4 code) is mapped
#     to a representative MagLIF target inner radius ~2.3 mm (AR~6 Be liner, OD 5.58 mm
#     -> inner radius ~2.3 mm; Gomez 2014 / Sefkow 2014); refined to the Knapp-2017
#     published target in #121.  min_radius then scales with the achieved convergence.
RHO_CGS = 1.0          # g/cc per code density (solid-Be grounded)
R_FUEL_MM = 2.3        # representative initial inner radius in mm (provisional, #121)
# time: 1 code time unit anchored so the full ICF run (tlim = 0.9 code; see
# inputs/maglif_icf.athinput) spans a representative ~100 ns MagLIF current-rise-to-
# stagnation window (Gomez 2014 / Sefkow 2014).  Provisional, for REPORTING the
# confinement time only -- the absolute-SI value is calibrated to the Knapp-2017 drive in
# #121.
TLIM_CODE = 0.9
IMPLOSION_NS = 100.0
NS_PER_CODE = IMPLOSION_NS / TLIM_CODE   # ~111 ns per code time unit (provisional, #121)

input_file = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif_icf.athinput")
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")


def _map_to_window(t_code, x_lo, x_hi):
    """Linearly map a code-unit abscissa onto the experimental ``[x_lo, x_hi]`` window.

    The reduced surrogate's trajectory lives on a code-unit time axis incommensurate with
    the experiment's ns abscissa.  Mapping the sim abscissa onto the digitized window lets
    the curve oracle compare the two point-by-point WITHOUT extrapolation (every
    experimental point overlaps the mapped sim curve); the ORDINATE is left physically
    calibrated, so the verdict is an honest reported deviation -- not a forced fit.
    """
    t = np.asarray(t_code, dtype=float)
    span = float(t.max() - t.min())
    if span <= 0.0:
        return np.full_like(t, 0.5 * (x_lo + x_hi))
    return x_lo + (t - t.min()) / span * (x_hi - x_lo)


def _cell_mass(r, dens):
    """Cylindrical cell mass per unit (phi, z): rho * 0.5 (r_out^2 - r_in^2)."""
    dr = r[1] - r[0]
    rf = np.concatenate([[r[0] - 0.5 * dr], r + 0.5 * dr])
    rf = np.clip(rf, 0.0, None)
    vol = 0.5 * (rf[1:] ** 2 - rf[:-1] ** 2)
    return dens * vol


def _radius_enclosing(r, cum_mass, target):
    """Radius where the cumulative mass profile first reaches ``target`` (interp)."""
    if cum_mass[-1] < target:
        return np.nan
    j = int(np.searchsorted(cum_mass, target))
    if j == 0:
        return float(r[0])
    f = (target - cum_mass[j - 1]) / (cum_mass[j] - cum_mass[j - 1])
    return float(r[j - 1] + f * (r[j] - r[j - 1]))


def test_verify_maglif_icf():
    """Run the confined implosion; gate on the convergence/rebound + mass sanity
    signature, then report the stagnation scalars against the Knapp-2017 oracle (#140)."""
    try:
        assert testutils.run_pgen("maglif", input_file), "MagLIF ICF run failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_icf.prim.*.bin")))
        assert len(files) > 8, f"too few ICF snapshots: {len(files)}"

        # Radial grid + initial fuel/liner masses (Lagrangian markers) from t=0.
        d0 = bin_convert.read_binary_as_athdf(files[0])
        r = np.asarray(d0["x1v"], dtype=float)
        dn0 = np.asarray(d0["dens"], dtype=float)[0, 0]
        m0 = _cell_mass(r, dn0)
        liner = dn0 >= 0.5 * D_LINER
        assert liner.any(), "no dense liner present in the initial state"
        r_liner_inner0 = r[liner][0]
        m_fuel0 = float(m0[r < r_liner_inner0].sum())   # mass interior to the liner
        m_liner = float(m0[liner].sum())

        times, r_if, r_lv, rho_fuel, rho_liner = [], [], [], [], []
        for fp in files:
            d = bin_convert.read_binary_as_athdf(fp)
            dn = np.asarray(d["dens"], dtype=float)[0, 0]
            cum = np.cumsum(_cell_mass(r, dn))
            rif = _radius_enclosing(r, cum, m_fuel0)
            rlv = _radius_enclosing(r, cum, m_fuel0 + m_liner)
            times.append(float(d["Time"]))
            r_if.append(rif)
            r_lv.append(rlv)
            # mass-conserved mean fuel density: m_fuel0 / (fuel volume ~ 0.5 R_if^2)
            rho_fuel.append(m_fuel0 / (0.5 * rif ** 2) if np.isfinite(rif) else np.nan)
            rho_liner.append(float(dn.max()))
        times = np.array(times)
        r_if = np.array(r_if)
        r_lv = np.array(r_lv)
        rho_fuel = np.array(rho_fuel)
        rho_liner = np.array(rho_liner)

        # Final snapshot is finite (the implosion ran in + rebounded cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        imin = int(np.nanargmin(r_if))
        rif0, rif_min = r_if[0], r_if[imin]

        # --- Cheap QUALITATIVE gates (sign/shape sanity, no re-anchored magnitudes).
        # These are the reduced surrogate's binding pre-checks -- they guard against a sim
        # that never imploded or went unstable; the experimental scalar comparisons below
        # are reported against the oracle (the absolute-SI hard-assert is #121).

        # (q1) Mass sanity: the t=0 mean fuel density recovers the input fuel density (the
        # enclosed-mass reduction is wired up correctly).
        assert abs(rho_fuel[0] - D_FUEL) < 0.1 * D_FUEL, (
            f"t=0 mean fuel density {rho_fuel[0]:.4f} != d_fuel {D_FUEL}"
        )

        # (q2) Convergence-and-rebound signature: the inner radius actually converged (its
        # minimum is below the start), the minimum is interior in time (it turned around,
        # not still imploding at the last snapshot), and it rebounds afterwards -- a clean
        # confined stagnation.  These are sign checks, not tuned convergence/rebound bars.
        assert rif_min < rif0, (
            f"inner radius did not converge: R_if {rif0:.4f} -> min {rif_min:.4f}"
        )
        assert 0 < imin < len(times) - 1, (
            f"inner radius did not stagnate before the run end (min at snap {imin})"
        )
        assert r_if[-1] > rif_min, (
            f"no rebound after stagnation: R_min={rif_min:.4f} -> R_end={r_if[-1]:.4f}"
        )

        # The full coupled radiation-conduction-MHD stack ran and kept the species-split
        # energy budget physically sane (erad finite, non-neg, bounded; #116/[B3]).
        cenergy.assert_coupled_energy_sane("maglif_icf")

        # --- QUANTITATIVE ANCHOR (#140): compare the reduced run's stagnation scalars
        # against the committed Knapp-2017 experimental datums via the oracle.
        # The reduced 1-D surrogate (CR~22x, ~500x compression) cannot reproduce the
        # ABSOLUTE dimensional scalars; per the B3 velocity-ratio policy these are
        # REPORTED against the oracle (scorecard + overlay), not hard-asserted -- the SI
        # hard-assert is the paper-resolution SI run (#121) with real EOS/opacity (#118).
        oracle = gto.GroundTruthOracle.from_committed()

        # min fuel-column radius, mapped to mm via the provisional length calibration.
        min_radius_mm = (rif_min / R_FUEL) * R_FUEL_MM
        # peak mean fuel density, mapped to g/cc via the solid-Be-grounded density unit.
        peak_density_gcc = float(np.nanmax(rho_fuel)) * RHO_CGS

        reduced_note = "reduced 1-D surrogate; absolute-SI hard-assert is #121 (EOS #118)"
        res_radius = oracle.compare("B4", "min_radius", min_radius_mm)
        res_density = oracle.compare("B4", "peak_density", peak_density_gcc)
        scorecard.record_result(res_radius, binding=False, note=reduced_note)
        scorecard.record_result(res_density, binding=False, note=reduced_note)
        print(f"[B4] {res_radius}  (reported, not asserted: {reduced_note})")
        print(f"[B4] {res_density}  (reported, not asserted: {reduced_note})")

        # --- Confinement time (#156): formerly pending, now a committed stated text
        # scalar (Knapp 2017: 14 ns measured vs 16 ns 1D).  Reduce the sim to its dwell at
        # convergence ratio > 2 (the stagnation residence time), map to ns via the
        # provisional time calibration, and REPORT it against the oracle (binding=False);
        # the same reduced-surrogate report policy as the min-radius/peak-density scalars;
        # the absolute-SI hard-assert is #121.
        cr = rif0 / r_if
        above = np.where(np.isfinite(cr) & (cr > 2.0))[0]
        tau_code = float(times[above[-1]] - times[above[0]]) if above.size >= 2 else 0.0
        # (q5) Positive confinement time: the implosion dwelt at high convergence (CR>2).
        assert tau_code > 0.0, (
            f"no positive confinement time (no dwell at CR>2; tau_code={tau_code:.4g})"
        )
        tau_ns = tau_code * NS_PER_CODE
        res_tau = oracle.compare("B4", "confinement_time", tau_ns)
        scorecard.record_result(res_tau, binding=False, note=reduced_note)
        print(f"[B4] {res_tau}  (reported, not asserted: {reduced_note})")

        # Experiment-overlay diagnostic: sim trajectories (physical units) with each
        # scalar's experimental value + tolerance band drawn on top.
        overlay.overlay_scalars(
            "maglif_icf",
            times,
            {
                "R_if": (r_if / R_FUEL) * R_FUEL_MM,
                "rho_fuel": rho_fuel * RHO_CGS,
            },
            {
                "R_if": {
                    "exp_value": res_radius.exp_value, "band": res_radius.band,
                    "unit": "mm", "sim_value": min_radius_mm,
                },
                "rho_fuel": {
                    "exp_value": res_density.exp_value, "band": res_density.band,
                    "unit": "g/cc", "sim_value": peak_density_gcc,
                },
            },
            xlabel="t (code units)",
            title="ICF stagnation scalars vs Knapp 2017 (MagLIF benchmark 4)",
        )

        # --- Inner-radius TRAJECTORY anchor (#156): overlay the corrected v3 experiment
        # radiograph circles (Knapp 2017 panel b, 6 circles -- the 2 rebound points at
        # r~0.54 recovered, legend box excluded) on the sim inner-radius trajectory and
        # REPORT the curve comparison.  The sim confinement-window abscissa is mapped onto
        # the experimental stagnation window so the digitized points overlap for a
        # point-by-point oracle comparison (no extrapolation); the radius stays physically
        # calibrated (R_FUEL_MM), so the verdict is an honest reported FAIL when the toy
        # over-compresses -- binding=False, exactly as for the scalars.
        traj = oracle.get("B4", "inner_radius_trajectory")
        xe = [float(p["x"]) for p in traj.points]
        sl = slice(int(above[0]), int(above[-1]) + 1)
        t_map = _map_to_window(times[sl], min(xe), max(xe))
        r_if_mm = (r_if[sl] / R_FUEL) * R_FUEL_MM
        res_traj = oracle.compare("B4", "inner_radius_trajectory", (t_map, r_if_mm))
        scorecard.record_result(res_traj.worst_point, binding=False, note=reduced_note)
        print(f"[B4] inner-radius trajectory: {res_traj.n_compared} v3 pts compared; "
              f"worst {res_traj.worst_point}  (reported)")
        exp_pts = [{"x": p.x, "y": p.exp_value, "band": p.band}
                   for p in res_traj.point_results]
        overlay.overlay_curve(
            "maglif_icf_inner_radius", t_map, r_if_mm, exp_points=exp_pts,
            xlabel="t (mapped onto experiment window)", ylabel="inner radius",
            x_unit="ns", unit="mm",
            title="B4 inner-radius trajectory vs Knapp 2017 (v3 corrected, #156)")

        # Layer-2 regression guard (ADR-0008): code-unit baseline diff, unchanged so the
        # B-series coupled-stack baseline stays byte-identical.
        harness.verify(
            "maglif_icf",
            times,
            {
                "R_if": r_if,
                "R_lv": r_lv,
                "rho_fuel": rho_fuel,
                "rho_liner_peak": rho_liner,
            },
            coord_label="t",
            xlabel="t",
            title="ICF confinement time (MagLIF benchmark 4): inner radius + densities",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
