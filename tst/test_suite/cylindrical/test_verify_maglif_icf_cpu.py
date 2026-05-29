"""
ICF confinement-time verification (FLASH MagLIF benchmark 4, #37).

Runs the integrated MagLIF/Z-pinch problem generator (issue [10a]/#32) as a CLEAN bulk
implosion -- a dense beryllium liner enclosing low-density deuterium fuel, driven radially
inward by a prescribed constant load current and run to STAGNATION (no interface
perturbation is seeded, unlike the instability benchmarks #36/#39/#42).  This is benchmark
4 of the FLASH MagLIF validation ladder (Ellison et al. 2025, arXiv:2504.10760), whose
observables are the PDV/radiography inner-radius trajectory and the stagnation density.

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

# Verification thresholds (comfortable margins; the qualitative convergence/stagnation/
# rebound signature is preserved under the coupled stack -- observed adiabatic-MHD run:
# R_if ~ 0.40 -> 0.045 (convergence ratio ~9) -> rebound to ~0.23; fuel compresses ~75x).
CR_MIN = 2.0           # require >= 2x convergence of the inner radius (clean implosion)
REBOUND_MIN = 0.04     # require the inner radius to bounce back >= this after stagnation
COMPRESS_MIN = 5.0     # require the fuel to compress >= 5x in mean density
TAU_MIN = 0.05         # require a positive confinement dwell time (CR > 2 interval)

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


def _interval_below(t, y, thr):
    """Time span over which y(t) is below thr, with linear edge interpolation."""
    below = y < thr
    if not below.any():
        return 0.0
    idx = np.where(below)[0]
    lo, hi = idx[0], idx[-1]
    # entry: interpolate between lo-1 and lo (if there is a crossing)
    if lo == 0:
        t_in = t[0]
    else:
        f = (thr - y[lo - 1]) / (y[lo] - y[lo - 1])
        t_in = t[lo - 1] + f * (t[lo] - t[lo - 1])
    # exit: interpolate between hi and hi+1 (if there is a crossing)
    if hi == len(y) - 1:
        t_out = t[-1]
    else:
        f = (thr - y[hi]) / (y[hi + 1] - y[hi])
        t_out = t[hi] + f * (t[hi + 1] - t[hi])
    return float(t_out - t_in)


def test_verify_maglif_icf():
    """Run the confined implosion and verify stagnation + rebound + compression."""
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

        # sanity: the t=0 mean fuel density recovers the input fuel density.
        assert abs(rho_fuel[0] - D_FUEL) < 0.1 * D_FUEL, (
            f"t=0 mean fuel density {rho_fuel[0]:.4f} != d_fuel {D_FUEL}"
        )

        # Final snapshot is finite (the implosion ran in + rebounded cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        imin = int(np.nanargmin(r_if))
        rif0, rif_min = r_if[0], r_if[imin]

        # (1) The liner converges: the inner radius reaches a high-convergence minimum.
        assert rif_min < rif0 / CR_MIN, (
            f"liner did not converge: R_if {rif0:.4f} -> min {rif_min:.4f} "
            f"(need < {rif0 / CR_MIN:.4f}, i.e. CR > {CR_MIN})"
        )

        # (2) It stagnates and rebounds: the minimum is interior in time (not at the last
        # snapshot -- it actually turns around) and the inner radius bounces back.
        assert 0 < imin < len(times) - 1, (
            f"inner radius did not stagnate before the run end (min at snap {imin})"
        )
        assert r_if[-1] > rif_min + REBOUND_MIN, (
            f"no rebound after stagnation: R_min={rif_min:.4f} -> R_end={r_if[-1]:.4f}"
        )

        # (3) The fuel compresses strongly at stagnation.
        compress = float(np.nanmax(rho_fuel) / D_FUEL)
        assert compress > COMPRESS_MIN, (
            f"fuel compression {compress:.1f}x < required {COMPRESS_MIN}x"
        )

        # (4) A positive confinement time: the dwell during which the inner radius is
        # below half its initial value (CR > 2) -- the stagnation burn-width proxy.
        tau_conf = _interval_below(times, r_if, 0.5 * rif0)
        assert tau_conf > TAU_MIN, (
            f"confinement time {tau_conf:.4f} <= {TAU_MIN} (no sustained stagnation)"
        )

        # The full coupled radiation-conduction-MHD stack ran and kept the species-split
        # energy budget physically sane (erad finite, non-neg, bounded; #116/[B3]).
        cenergy.assert_coupled_energy_sane("maglif_icf")

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
