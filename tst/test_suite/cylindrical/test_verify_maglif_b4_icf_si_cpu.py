"""
Faithful Ellison benchmark 4 ICF-confinement replication (#121 [C4]).

REPLACES the idealized code-unit ICF *surrogate* (test_verify_maglif_icf_cpu.py / #140),
which ran a code-unit toy (a tuned constant drive, code-unit geometry, provisional
length/time calibrations) and so could only REPORT the stagnation scalars against fudge
factors.  This test runs the *faithful*, DIMENSIONAL setup
(``tst/inputs/maglif_b4_icf_si.athinput``), built with the same faithful conventions as
the B1/B2 replications (maglif_b1_sinars.athinput / #120, maglif_b2_ellison.athinput /
#163): the AR=6 BERYLLIUM liner enclosing deuterium fuel (#160 geometry, solid-Be density
unit), the measured z2173 Z-machine drive (#160), radiation OFF (ideal-MHD core).  Because
the run is dimensional, the observables emerge directly in mm / g-cc / ns -- the
provisional
``R_FUEL_MM`` / ``NS_PER_CODE`` calibrations the surrogate flagged "refined in #121" are
gone.

Oracle (Layer-1, ADR-0008): the EXPERIMENT, via the committed Knapp-2017 datums in
``verification/ground_truth/b4_icf_confinement_knapp_2017.json`` (P. F. Knapp et al.,
Phys. Plasmas 24, 042708, 2017): the stagnation scalars (min inner radius 0.45 mm, peak
fuel density ~10 g/cc, confinement time 14 ns) and the digitized inner-radius trajectory
(6 radiograph circles).  The run reduces to its inner-radius (fuel/liner contact)
trajectory + fuel/liner densities, compares the stagnation scalars + trajectory against
the oracle, records the verdicts in the suite scorecard, and overlays the experimental
value/band on the diagnostic plots.

HONEST SCOPE (the engineering residual to a BINDING quantitative MATCH -- the three
residuals documented in maglif_b4_icf_si.athinput, shared with the faithful B1/B2 runs):
(a) the IONMIX tabulated 3T EOS/opacity (#118/#162) is not yet wired into the
multi-material maglif IC, so this faithful run uses the ideal-MHD core (radiation OFF) and
the confining rebound is the ADIABATIC fuel cushion; (b) the drive is consumed in code
units (no SI current/time calibration), so the trajectory is compared as a dimensionless
implosion mapped onto the experimental stagnation window; (c) the ideal-gamma EOS (no
material strength, no degenerate-DD pressure) over-compresses the fuel column.  So the
absolute stagnation scalars and the trajectory are REPORTED against the oracle
(binding=False), exactly as the surrogate (#140) and the faithful B1 run (#120); the
absolute-SI hard-assert is the paper-resolution GPU run on the tabulated-EOS coupled stack
(residual a + the GPU coupled segfault #139).  The density-vs-radius PROFILE (acceptance
criterion 3) has no committed experimental profile datum yet (only the radius-vs-time
trajectory is digitized), so it is recorded PENDING (needs a digitized Abel-inverted
profile), per ADR-0008's no-fabrication rule.

This is the reduced-resolution CPU GATE for the faithful setup (the #163 faithful-vs-gate
pattern): it runs the committed faithful athinput but OVERRIDES ONLY the mesh resolution
and integration length (``nx1`` / ``tlim``) to keep the CPU path fast -- the drive,
geometry and EOS are the faithful committed values.  The full nx1=512 ("paper-resolution")
run is the committed athinput itself (the GPU target, as for the other benchmarks).  The
BINDING gate is the qualitative confinement signature (the dimensional scalars are
reported, not asserted):

  * the faithful z2173 drive IMPLODES the liner -- the inner radius converges to a deep,
    interior-in-time minimum (stagnation) and REBOUNDS (a confined bulk implosion);
  * mass is conserved (the t=0 enclosed-mass reduction recovers the input fuel density);
  * the final state is finite/stable;
  * DISCRIMINATOR: a no-drive control (current_waveform=constant, i0=0 -> zero load
    current) does NOT converge (the inner radius stays flat) -- so the convergence is
    drive-caused, not a setup artifact.  That control is the binding red->green
    discriminator (feeding its flat trajectory into the convergence assert FAILS).

Discriminating quantities (1-D radial, axisymmetric -- a pure bulk stagnation; tracked by
ENCLOSED MASS, robust to shocks and the fuzzy fuel/liner density transition):
  * R_if(t)  -- inner radius (fuel/liner contact): implodes to a minimum and rebounds.
  * R_lv(t)  -- outer liner edge (liner/vacuum contact).
  * rho_fuel(t) -- mass-conserved mean fuel density (~ 1/R_if^2): peaks at stagnation.
  * rho_liner_peak(t) -- peak shell density (the compressed beryllium liner).

NOTE: this is a deterministic bulk implosion (no instability seed), so the harness
regression baseline is reproducible for the serial same-platform re-run.  Auto-collected
by run_test_suite.py (the module name contains ``_cpu``).
"""

# Modules
import glob
import os
import sys

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the working directory (the binary runs from build_pgen/maglif/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Faithful committed input (dimensional artifact); the gate overrides only nx1/tlim below.
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_b4_icf_si.athinput"
)
trace_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "z2173_current.dat"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")

# Values that must match inputs/maglif_b4_icf_si.athinput (PHYSICAL CGS / units).
RHO_CGS = 1.85        # g/cc per code density (density_cgs = solid beryllium 1.85)
D_FUEL_GCC = 0.0185   # deuterium fuel density [g/cc] (d_fuel = 0.0185 -> 0.01 code)
D_LINER_CODE = 1.0    # liner density in code units (d_liner 1.85 g/cc / RHO_CGS)

# Reduced CPU-gate mesh + integration (faithful drive/geometry/EOS unchanged): calibrated
# so the liner implodes to a deep interior minimum and rebounds within a fast CPU run; the
# nx1=512 paper-resolution run is the committed athinput (the GPU target).
NX1 = 64             # radial zones over [0, 3.2] mm (dr 50 um; resolves the column)
TLIM = 34.0          # code time (ns): run-in + stagnation + clear rebound (CR ~ 11)
OUT_DT = 1.0         # snapshot cadence [ns]

# Confinement-signature thresholds (sign/shape gates; scalars are reported, not gated).
CONV_MIN = 3.0       # faithful run must reach >= 3x convergence ratio (calibrated ~11x)
CTRL_MAX = 1.3       # no-drive control stays below this conv. ratio (calibrated 1.0)
CR_DWELL = 2.0       # convergence-ratio level defining the confinement dwell window


def _run(basename, drive_args):
    """Run the faithful setup at reduced resolution; return sorted snapshot paths."""
    ok = testutils.run_pgen(
        "maglif", input_file,
        args=[
            f"problem/current_file={trace_file}",
            f"mesh/nx1={NX1}", f"meshblock/nx1={NX1}",
            f"time/tlim={TLIM}", f"output1/dt={OUT_DT}",
            f"job/basename={basename}",
        ] + drive_args,
    )
    assert ok, f"faithful B4 ICF run failed (basename={basename})"
    files = sorted(glob.glob(os.path.join(bin_dir, f"{basename}.prim.*.bin")))
    assert len(files) > 8, f"too few snapshots for {basename}: {len(files)}"
    return files


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


def _map_to_window(t_code, x_lo, x_hi):
    """Linearly map a code-time abscissa onto the experimental ``[x_lo, x_hi]`` window.

    The faithful run's stagnation occurs at code-time tens of ns; the Knapp radiograph
    circles sit at absolute experiment time ~3120 ns (near peak current).  Mapping the sim
    stagnation window onto the digitized window lets the curve oracle compare the two
    point-by-point WITHOUT extrapolation; the ORDINATE (radius, mm) is left physically
    calibrated, so the verdict is an honest reported deviation -- not a forced fit.
    """
    t = np.asarray(t_code, dtype=float)
    span = float(t.max() - t.min())
    if span <= 0.0:
        return np.full_like(t, 0.5 * (x_lo + x_hi))
    return x_lo + (t - t.min()) / span * (x_hi - x_lo)


def _trajectory(files):
    """Reduce snapshots to the inner-radius + density time series (dimensional)."""
    d0 = bin_convert.read_binary_as_athdf(files[0])
    r = np.asarray(d0["x1v"], dtype=float)
    dn0 = np.asarray(d0["dens"], dtype=float)[0, 0]
    m0 = _cell_mass(r, dn0)
    liner = dn0 >= 0.5 * D_LINER_CODE
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
        rho_fuel.append(m_fuel0 / (0.5 * rif ** 2) if np.isfinite(rif) else np.nan)
        rho_liner.append(float(dn.max()))
    return (np.array(times), np.array(r_if), np.array(r_lv),
            np.array(rho_fuel), np.array(rho_liner), m_fuel0)


def test_verify_maglif_b4_icf_si():
    """Run the faithful confined implosion; gate on the convergence-rebound + no-drive
    control signature, then REPORT the dimensional stagnation scalars + inner-radius
    trajectory against the Knapp-2017 oracle (#121 [C4])."""
    try:
        # --- Faithful run (measured z2173 tabulated drive) + no-drive control.
        files = _run("maglif_b4_icf_si", [])
        ctl_files = _run(
            "maglif_b4_icf_si_ctl",
            ["problem/current_waveform=constant", "problem/i0=0.0"],
        )

        times, r_if, r_lv, rho_fuel, rho_liner, m_fuel0 = _trajectory(files)
        ct, c_rif = _trajectory(ctl_files)[:2]

        # Final snapshot is finite (the implosion ran in + rebounded, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        imin = int(np.nanargmin(r_if))
        rif0, rif_min = r_if[0], r_if[imin]
        cr_main = rif0 / rif_min
        cr_ctl = c_rif[0] / np.nanmin(c_rif)

        # --- BINDING qualitative gates (the dimensional scalars are reported below).

        # (q1) Mass sanity: the t=0 mean fuel density recovers the input fuel density (the
        # enclosed-mass reduction is wired up correctly), in physical g/cc.
        assert abs(rho_fuel[0] * RHO_CGS - D_FUEL_GCC) < 0.1 * D_FUEL_GCC, (
            f"t=0 mean fuel density {rho_fuel[0] * RHO_CGS:.4f} g/cc != "
            f"d_fuel {D_FUEL_GCC}"
        )

        # (q2) Convergence: the faithful drive imploded the liner to a deep minimum.
        assert cr_main >= CONV_MIN, (
            f"liner under-converged: CR {cr_main:.2f} < {CONV_MIN} "
            f"(R_if {rif0:.4f} -> min {rif_min:.4f} mm)"
        )

        # (q3) Stagnation: the minimum is interior in time (it turned around, not still
        # imploding at the last snapshot) -- a confined stagnation, not a runaway crush.
        assert 0 < imin < len(times) - 1, (
            f"inner radius did not stagnate before the run end (min at snap {imin})"
        )

        # (q4) Rebound: the inner radius bounces back after the minimum (the fuel cushion
        # halted and reversed the liner) -- the confinement signature.
        assert r_if[-1] > rif_min, (
            f"no rebound after stagnation: R_min {rif_min:.4f} -> R_end "
            f"{r_if[-1]:.4f} mm"
        )

        # DISCRIMINATOR (binding red->green): the no-drive control does NOT converge --
        # convergence above is caused by the faithful z2173 drive, not the IC/geometry.
        # Feeding this flat control into (q2) FAILS (cr_ctl ~ 1.0 < CONV_MIN).
        assert cr_ctl <= CTRL_MAX, (
            f"no-drive control converged unexpectedly: CR {cr_ctl:.3f} > {CTRL_MAX} "
            f"(drive should be off)"
        )

        # --- QUANTITATIVE ANCHOR (#121): compare the faithful DIMENSIONAL stagnation
        # scalars against the committed Knapp-2017 datums via the oracle.  Per the
        # reduced-surrogate / faithful-B1 policy (residuals a-c in the input header) these
        # are REPORTED (binding=False), not hard-asserted; the SI hard-assert is the
        # paper-resolution GPU run on the tabulated-EOS coupled stack.
        oracle = gto.GroundTruthOracle.from_committed()
        reduced_note = (
            "faithful reduced-res CPU gate; SI hard-assert is the paper-res GPU run"
        )

        min_radius_mm = float(rif_min)              # dimensional, no fudge factor
        peak_density_gcc = float(np.nanmax(rho_fuel)) * RHO_CGS

        res_radius = oracle.compare("B4", "min_radius", min_radius_mm)
        res_density = oracle.compare("B4", "peak_density", peak_density_gcc)
        scorecard.record_result(res_radius, binding=False, note=reduced_note)
        scorecard.record_result(res_density, binding=False, note=reduced_note)
        print(f"[B4] {res_radius}  (reported: {reduced_note})")
        print(f"[B4] {res_density}  (reported: {reduced_note})")

        # Confinement time: the dwell at convergence ratio > CR_DWELL (the stagnation
        # residence time).  Code time = ns here (the z2173 trace + tlim are in ns), so the
        # dwell is reported directly in ns -- no provisional NS_PER_CODE calibration.
        cr = rif0 / r_if
        above = np.where(np.isfinite(cr) & (cr > CR_DWELL))[0]
        tau_ns = float(times[above[-1]] - times[above[0]]) if above.size >= 2 else 0.0
        assert tau_ns > 0.0, (
            f"no positive confinement time (no dwell at CR>{CR_DWELL}; tau={tau_ns:.4g})"
        )
        res_tau = oracle.compare("B4", "confinement_time", tau_ns)
        scorecard.record_result(res_tau, binding=False, note=reduced_note)
        print(f"[B4] {res_tau}  (reported: {reduced_note})")

        # Experiment-overlay diagnostic: sim trajectories (physical units) with each
        # scalar's experimental value + tolerance band drawn on top.
        overlay.overlay_scalars(
            "maglif_b4_icf_si",
            times,
            {"R_if": r_if, "rho_fuel": rho_fuel * RHO_CGS},
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
            xlabel="t (ns, code units)",
            title="B4 ICF stagnation scalars vs Knapp 2017 (faithful gate, #121)",
        )

        # --- Inner-radius TRAJECTORY anchor: overlay the digitized Knapp v3 radiograph
        # circles on the sim inner-radius trajectory and REPORT the curve comparison.  The
        # sim confinement-window abscissa maps onto the experimental stagnation window
        # so the digitized points overlap for a point-by-point oracle comparison (no
        # extrapolation); the radius stays physically calibrated (mm), so the verdict is
        # honest reported deviation -- binding=False, as for the scalars.
        traj = oracle.get("B4", "inner_radius_trajectory")
        xe = [float(p["x"]) for p in traj.points]
        sl = slice(int(above[0]), int(above[-1]) + 1)
        t_map = _map_to_window(times[sl], min(xe), max(xe))
        res_traj = oracle.compare("B4", "inner_radius_trajectory", (t_map, r_if[sl]))
        scorecard.record_result(res_traj.worst_point, binding=False, note=reduced_note)
        print(f"[B4] inner-radius trajectory: {res_traj.n_compared} Knapp pts; "
              f"worst {res_traj.worst_point}  (reported)")
        exp_pts = [{"x": p.x, "y": p.exp_value, "band": p.band}
                   for p in res_traj.point_results]
        overlay.overlay_curve(
            "maglif_b4_icf_si_inner_radius", t_map, r_if[sl], exp_points=exp_pts,
            xlabel="t (mapped onto experiment window)", ylabel="inner radius",
            x_unit="ns", unit="mm",
            title="B4 inner-radius trajectory vs Knapp 2017 (v3 corrected, #121)")

        # --- Density-vs-radius PROFILE (acceptance criterion 3): no committed exp.
        # profile datum exists yet (only the radius-vs-time trajectory is digitized).
        # Record it PENDING -- a digitized Abel-inverted profile is the data-supply step
        # that flips this to a comparison (ADR-0008 forbids fabricating one).
        scorecard.record_pending("B4", "density_radius_profile", issue=121)
        print("[B4] density-vs-radius profile: PENDING (no profile datum; #121)")

        # Layer-2 regression guard (ADR-0008): code-unit baseline diff.
        harness.verify(
            "maglif_b4_icf_si",
            times,
            {
                "R_if": r_if,
                "R_lv": r_lv,
                "rho_fuel": rho_fuel,
                "rho_liner_peak": rho_liner,
            },
            coord_label="t",
            xlabel="t",
            title="B4 faithful ICF confinement (benchmark 4): inner radius + densities",
            rtol=1.0e-4,
            atol=1.0e-8,
        )
    finally:
        for pat in ("maglif_b4_icf_si.prim.*.bin", "maglif_b4_icf_si_ctl.prim.*.bin"):
            for fp in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(fp)
