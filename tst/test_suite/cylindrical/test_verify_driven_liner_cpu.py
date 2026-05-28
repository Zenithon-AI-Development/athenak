"""
Current-driven cylindrical liner implosion verification (ADR-0005, #29).

This is the "verify driven pinch" slice: a prescribed-I(t) drive compresses a dense
cylindrical liner shell radially inward (the magnetic piston).  See
inputs/cylindrical/driven_liner.athinput -- a 1-D radial cylindrical MHD run reusing the
driven-pinch drive source + circuit-driven B_phi outer boundary (issue #21) with the
``profile=liner`` IC: fuel | liner | vacuum regions, the driven 1/r azimuthal field laid
down only in the vacuum gap OUTSIDE the liner, and a constant load current I(t).  The
magnetic pressure just outside the liner drives the shell inward; the bulk implosion is
purely 1-D radial so no Rayleigh-Taylor modes grow ("bulk implosion only").

The implosion trajectory R(t) is the discriminating quantity: a thin-shell (slug) model
balances the liner line-mass inertia against the J x B magnetic drive,

    m_l d^2R/dt^2 = -2*pi*R * (B_phi(R)^2 / 2)
                  = -mu0^2 I(t)^2 / (4*pi*R) ,   B_phi(R) = mu0 I / (2*pi*R) ,

with the line mass m_l and initial mass-averaged radius R0 measured from the t=0 state.
A correct cylindrical MHD update (hoop stress + flux divergence, #16) reproduces this
trajectory to a few percent during the run-in; a wrong magnetic force would implode at a
visibly different rate (or not at all).  The test:

  * runs the 1-D radial liner implosion (driven_pinch is a user pgen -> run_pgen),
  * extracts the mass-averaged liner radius R_sim(t) from each snapshot,
  * integrates the thin-shell ODE with the same I(t), m_l, R0 to get R_thinshell(t),
  * asserts the liner actually compresses and the trajectory matches within tolerance,
  * plots R_sim / R_thinshell / v_sim vs t and saves/diffs a golden regression baseline.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import glob
import os
import re
import shutil
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# Drive/IC parameters -- must match inputs/cylindrical/driven_liner.athinput.
I0 = 1.0           # constant prescribed load current [code units]
MU0 = 1.0          # code-unit permeability (Heaviside-Lorentz)
DTHRESH = 0.3      # density threshold isolating the dense liner material
TRAJ_RTOL = 0.10   # max |R_sim - R_thinshell| / R0 allowed during the implosion

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "driven_liner.athinput"
)
tab_dir = os.path.join(testutils.pgen_run_dir("driven_pinch"), "tab")


def _read_time(path):
    """Parse the simulation time from a tab snapshot's header line."""
    with open(path) as f:
        first = f.readline()
    return float(re.search(r"time=([0-9eE+.\-]+)", first).group(1))


def _load(path):
    """Return (x1v, dens, velx) by column position (gid i x1v dens velx ...)."""
    data = np.loadtxt(path, comments="#")
    return data[:, 2], data[:, 3], data[:, 4]


def _mass_avg_radius(x1v, dens, dr):
    """Mass-weighted mean radius of the liner (cylindrical mass element rho*2*pi*r*dr)."""
    mask = dens > DTHRESH
    w = dens[mask] * x1v[mask]            # 2*pi*dr cancels in the ratio
    return np.sum(w * x1v[mask]) / np.sum(w), mask


def _line_mass(x1v, dens, dr):
    """Liner line mass (per unit length): integral rho * 2*pi*r dr over liner cells."""
    mask = dens > DTHRESH
    return np.sum(dens[mask] * 2.0 * np.pi * x1v[mask] * dr)


def _thin_shell_trajectory(times, r0, m_l):
    """Integrate m_l R'' = -mu0^2 I0^2/(4*pi*R) (constant I) sampled at ``times``."""
    a = MU0 ** 2 * I0 ** 2 / (4.0 * np.pi * m_l)
    r, v, tcur = r0, 0.0, 0.0
    dt = 1.0e-4
    out = []
    for tsample in times:
        while tcur < tsample - 1.0e-12:
            h = min(dt, tsample - tcur)
            k1x, k1v = v, -a / r
            k2x, k2v = v + 0.5 * h * k1v, -a / (r + 0.5 * h * k1x)
            k3x, k3v = v + 0.5 * h * k2v, -a / (r + 0.5 * h * k2x)
            k4x, k4v = v + h * k3v, -a / (r + h * k3x)
            r += (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v += (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            tcur += h
        out.append(r)
    return np.array(out)


def test_verify_driven_liner():
    """Run the liner implosion and verify R(t) vs the thin-shell model + baseline."""
    try:
        assert testutils.run_pgen("driven_pinch", input_file), "driven_liner run failed."

        files = sorted(glob.glob(os.path.join(tab_dir, "driven_liner.prim.*.tab")))
        assert len(files) > 5, f"too few liner snapshots: {len(files)}"

        x1v0, dens0, _ = _load(files[0])
        dr = x1v0[1] - x1v0[0]
        m_l = _line_mass(x1v0, dens0, dr)
        r0, _ = _mass_avg_radius(x1v0, dens0, dr)

        times, r_sim, v_sim = [], [], []
        for fp in files:
            x1v, dens, velx = _load(fp)
            r, mask = _mass_avg_radius(x1v, dens, dr)
            wsum = np.sum(dens[mask] * x1v[mask])
            vr = np.sum(dens[mask] * velx[mask] * x1v[mask]) / wsum
            times.append(_read_time(fp))
            r_sim.append(r)
            v_sim.append(vr)
        times = np.array(times)
        r_sim = np.array(r_sim)
        v_sim = np.array(v_sim)

        r_thin = _thin_shell_trajectory(times, r0, m_l)

        # (1) The liner actually compresses: a clear net inward implosion, no spurious
        # expansion (a missing/wrong magnetic drive would leave R ~ R0 or grow it).
        assert r_sim[-1] < 0.85 * r0, (
            f"liner did not compress: R0={r0:.4f} -> R={r_sim[-1]:.4f}"
        )
        assert np.all(np.diff(r_sim) < 1.0e-6), "liner radius not monotonically imploding"

        # (2) The shell moves inward (mass-averaged radial velocity negative + growing).
        assert v_sim[-1] < -0.1, f"implosion velocity too small: v={v_sim[-1]:.4f}"

        # (3) The trajectory tracks the analytic thin-shell model within tolerance.
        rel = np.abs(r_sim - r_thin) / r0
        assert rel.max() < TRAJ_RTOL, (
            f"R(t) deviates from thin-shell model: max rel err {rel.max():.4f} "
            f">= {TRAJ_RTOL} (R0={r0:.4f}, m_l={m_l:.4f})"
        )

        harness.verify(
            "driven_liner",
            times,
            {"R_sim": r_sim, "R_thinshell": r_thin, "v_sim": v_sim},
            coord_label="t",
            xlabel="t",
            title="Current-driven cylindrical liner implosion: R(t) vs thin-shell model",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(tab_dir, ignore_errors=True)
