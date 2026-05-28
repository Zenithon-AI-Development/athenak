"""
Coupled-circuit cylindrical liner implosion verification (ADR-0005 mode C, #40).

This is the Tier-2 verification of the *coupled* circuit-driven boundary: instead of a
prescribed current (the #29 driven-pinch slice), the load current is produced by a
voltage-driven lumped-element circuit ODE integrated alongside the MHD, with the Faraday
load voltage V_load = d(Phi)/dt fed back into the circuit (current loss + the stagnation
voltage spike).  See inputs/cylindrical/coupled_liner.athinput -- a 1-D radial cylindrical
MHD liner implosion (coupled_liner is a USER pgen -> run_pgen) wiring the three ADR-0005
pieces: the LumpedCircuit ODE (#34), the Faraday poloidal-flux reduction (#31), and the
circuit-driven B_phi outer boundary (#21).

The circuit is a series loop with an open-circuit voltage source V_oc, inductance L,
(here) zero series resistance Z0 and capacitance C, and the coupled load.  With Z0=C=0 the
loop reduces to a pure inductor in series with the imploding load, L dI/dt = V_oc -
V_load, so flux is conserved in the (series L + load) loop:

    L * I(t) + Phi(t) = V_oc * t        (constant V_oc, started from rest),

where Phi is the poloidal magnetic flux linking the circuit (the load inductance times I).
This exact relation -- the coupled-circuit *reference* (the verification anchor's circuit,
arXiv:2504.10760) reduced to its analytic core -- is the discriminating check: it holds
iff the circuit ODE, the Faraday feedback V_load=d(Phi)/dt, and the B_phi boundary are
wired together correctly.  The test:

  * runs the 1-D radial coupled-circuit liner implosion,
  * reads the dense circuit current/voltage waveforms (I, V_load, Phi) from the user
    history output and the liner trajectory R(t) from the tab snapshots,
  * checks flux conservation L*I + Phi = V_oc*t (the coupled-circuit reference),
  * checks the Faraday law V_load = d(Phi)/dt (the mode-C feedback term),
  * checks the implosion trajectory R(t) against the analytic thin-shell model driven by
    the *measured* (circuit-determined) current I(t) -- the hydro response (cf. #29),
  * checks the stagnation voltage spike (V_load peaks at peak compression, far above the
    run-in value),
  * plots I / V_load / Phi / R(t) and saves/diffs a golden regression baseline.

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

# Circuit / IC parameters -- must match inputs/cylindrical/coupled_liner.athinput.
V_OC = 2.0          # open-circuit (constant) voltage source [code units]
L_CIRCUIT = 1.0     # series inductance
MU0 = 1.0           # code-unit permeability (Heaviside-Lorentz)
TLIM = 1.25         # run end time
DTHRESH = 0.3       # density threshold isolating the dense liner material
TRAJ_RTOL = 0.15    # max |R_sim - R_thinshell| / R0 allowed during the implosion

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "coupled_liner.athinput"
)
run_dir = testutils.pgen_run_dir("coupled_liner")
tab_dir = os.path.join(run_dir, "tab")
hst_file = os.path.join(run_dir, "coupled_liner.user.hst")


def _read_time(path):
    """Parse the simulation time from a tab snapshot's header line."""
    with open(path) as f:
        first = f.readline()
    return float(re.search(r"time=([0-9eE+.\-]+)", first).group(1))


def _load(path):
    """Return (x1v, dens, velx) by column position (gid i x1v dens velx ...)."""
    data = np.loadtxt(path, comments="#")
    return data[:, 2], data[:, 3], data[:, 4]


def _mass_avg_radius(x1v, dens):
    """Mass-weighted mean radius of the liner (cylindrical mass element rho*2*pi*r*dr)."""
    mask = dens > DTHRESH
    w = dens[mask] * x1v[mask]            # 2*pi*dr cancels in the ratio
    return np.sum(w * x1v[mask]) / np.sum(w), mask


def _line_mass(x1v, dens, dr):
    """Liner line mass (per unit length): integral rho * 2*pi*r dr over liner cells."""
    mask = dens > DTHRESH
    return np.sum(dens[mask] * 2.0 * np.pi * x1v[mask] * dr)


def _thin_shell_trajectory(times, r0, m_l, i_of_t):
    """Integrate m_l R'' = -mu0^2 I^2/(4 pi R), I=i_of_t, sampled at the given times."""
    r, v, tcur = r0, 0.0, 0.0
    dt = 2.0e-4
    out = []
    for tsample in times:
        while tcur < tsample - 1.0e-12:
            h = min(dt, tsample - tcur)

            def acc(rr, tt):
                return -MU0 ** 2 * i_of_t(tt) ** 2 / (4.0 * np.pi * m_l * rr)

            k1x, k1v = v, acc(r, tcur)
            k2x, k2v = v + 0.5 * h * k1v, acc(r + 0.5 * h * k1x, tcur + 0.5 * h)
            k3x, k3v = v + 0.5 * h * k2v, acc(r + 0.5 * h * k2x, tcur + 0.5 * h)
            k4x, k4v = v + h * k3v, acc(r + h * k3x, tcur + h)
            r += (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            v += (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
            tcur += h
        out.append(r)
    return np.array(out)


def test_verify_coupled_liner():
    """Run the coupled-circuit liner implosion and verify the circuit + trajectory."""
    try:
        import athena_read

        ok = testutils.run_pgen("coupled_liner", input_file)
        assert ok, "coupled_liner run failed."

        # --- circuit current/voltage waveforms from the user history output ---
        hst = athena_read.hst(hst_file)
        th = hst["time"]
        cur = hst["I_circuit"]
        v_load = hst["V_load"]
        flux = hst["Bphi_flux"]
        assert len(th) > 50, f"too few history samples: {len(th)}"

        # (A) Flux conservation -- the coupled-circuit reference L*I + Phi = V_oc*t.
        # The explicit one-step feedback lag makes the residual ~ V_oc*dt (largest at the
        # first step); it is tight (<1%) once past the start-up transient.
        resid = np.abs(L_CIRCUIT * cur + flux - V_OC * th)
        assert resid.max() < 0.02, (
            f"flux conservation L*I+Phi=V_oc*t violated: max abs residual "
            f"{resid.max():.4f} (>= 0.02)"
        )
        late = th > 0.3
        rel_resid = (resid[late] / (V_OC * th[late])).max()
        assert rel_resid < 0.02, (
            f"flux conservation relative residual too large past startup: {rel_resid:.4f}"
        )

        # (B) Faraday law -- the mode-C feedback term V_load = d(Phi)/dt.  Compare the
        # circuit's load voltage to a central-difference of the reduced flux (skip the
        # first/last single-step samples, which are noisy finite differences).
        dphidt = np.gradient(flux, th)
        win = (th > 0.1) & (th < 0.98 * TLIM)
        rel_far = np.abs(v_load[win] - dphidt[win]) / (np.abs(dphidt[win]) + 1.0e-3)
        assert np.median(rel_far) < 0.03, (
            f"Faraday V_load != d(Phi)/dt: median rel error {np.median(rel_far):.4f}"
        )

        # Current rises from rest to a clear load current.
        assert cur[0] == 0.0 and cur[-1] > 1.0, (
            f"current did not ramp from rest: I0={cur[0]:.3f} I_end={cur[-1]:.3f}"
        )

        # (C) Stagnation voltage spike: the load voltage peaks at peak compression (late),
        # far above the run-in value (the inductance change as the liner stagnates).
        v_smooth = dphidt
        t_peak = th[np.argmax(v_smooth)]
        v_early = np.interp(0.2, th, v_smooth)
        assert t_peak > 0.7 * TLIM, f"load-voltage peak not at stagnation: t={t_peak:.3f}"
        assert v_smooth.max() > 10.0 * abs(v_early), (
            f"no stagnation spike: peak {v_smooth.max():.3f} vs early {v_early:.4f}"
        )

        # --- liner trajectory R(t) from the tab snapshots ---
        files = sorted(glob.glob(os.path.join(tab_dir, "coupled_liner.prim.*.tab")))
        assert len(files) > 10, f"too few liner snapshots: {len(files)}"

        x1v0, dens0, _ = _load(files[0])
        dr = x1v0[1] - x1v0[0]
        m_l = _line_mass(x1v0, dens0, dr)
        r0, _ = _mass_avg_radius(x1v0, dens0)

        times, r_sim, vr_sim = [], [], []
        for fp in files:
            x1v, dens, velx = _load(fp)
            r, mask = _mass_avg_radius(x1v, dens)
            wsum = np.sum(dens[mask] * x1v[mask])
            vr = np.sum(dens[mask] * velx[mask] * x1v[mask]) / wsum
            times.append(_read_time(fp))
            r_sim.append(r)
            vr_sim.append(vr)
        times = np.array(times)
        r_sim = np.array(r_sim)
        vr_sim = np.array(vr_sim)

        # (D) The liner implodes (clear net inward compression + inward velocity).
        assert r_sim[-1] < 0.8 * r0, f"no implosion: R0={r0:.4f} -> R={r_sim[-1]:.4f}"
        assert vr_sim[-1] < -0.1, f"implosion velocity too small: v={vr_sim[-1]:.4f}"

        # (E) The trajectory tracks the thin-shell model driven by the measured current
        # I(t) (the hydro response to the circuit-determined drive; cf. the #29 anchor).
        def i_of_t(t):
            return np.interp(t, th, cur)

        r_thin = _thin_shell_trajectory(times, r0, m_l, i_of_t)
        rel_traj = np.abs(r_sim - r_thin) / r0
        assert rel_traj.max() < TRAJ_RTOL, (
            f"R(t) deviates from thin-shell model: max rel err {rel_traj.max():.4f} "
            f">= {TRAJ_RTOL} (R0={r0:.4f}, m_l={m_l:.4f})"
        )

        # --- regression baseline (everything on the snapshot time grid) ---
        harness.verify(
            "coupled_liner",
            times,
            {
                "R_sim": r_sim,
                "R_thinshell": r_thin,
                "I_circuit": np.interp(times, th, cur),
                "V_load": np.interp(times, th, v_smooth),
                "Bphi_flux": np.interp(times, th, flux),
            },
            coord_label="t",
            xlabel="t",
            title="Coupled-circuit liner implosion: I/V/Phi/R(t) vs reference",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(tab_dir, ignore_errors=True)
        if os.path.exists(hst_file):
            os.remove(hst_file)
