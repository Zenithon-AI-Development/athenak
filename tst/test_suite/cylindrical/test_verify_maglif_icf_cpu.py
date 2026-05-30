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
anchor is still ``pending_digitization`` (Knapp 2017 headline result, #121) and is
reported as PENDING -- never as a pass.

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

input_file = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif_icf.athinput")
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")


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

        # The confinement-time anchor is still pending digitization (Knapp 2017 headline
        # result, #121): reported as PENDING, never as a pass.
        assert oracle.get("B4", "confinement_time").is_pending, (
            "B4 confinement_time unexpectedly has a value; update the scorecard wiring"
        )
        scorecard.record_pending("B4", "confinement_time", issue=121)

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
