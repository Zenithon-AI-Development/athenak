"""
Circuit drive modes B (voltage+RLC) and C (coupled feedback) in the MAGLIF path (#236).

The lumped-element circuit ODE (src/circuit/lumped_circuit.hpp, #34) and the Faraday
load-voltage reduction (src/circuit/faraday_voltage.hpp, #31) were wired end-to-end only
in the `coupled_liner` pgen (#40); the `maglif` pgen -- the reusable suite path every
MagLIF benchmark runs through -- hard-coded mode A (prescribed I(t)).  Issue #236
promotes ADR-0005 drive modes B and C into `maglif` behind `<problem> current_mode`:

  * ``voltage_rlc``     (B) -- open-circuit voltage source + fixed series RLC, no load
                               feedback: the circuit ODE is INDEPENDENT of the hydro;
  * ``coupled_circuit`` (C) -- identical, plus the Faraday load voltage
                               V_load = d(Phi)/dt fed back as a series voltage drop.

Two verifications, both against inputs/maglif_circuit.athinput (1-D radial cylindrical
MagLIF target; the mode-B test overrides the circuit block on the command line):

(1) Mode B vs a HIGH-ACCURACY ODE reference (the paper's own circuit-coupling check,
    Ellison et al. 2025 arXiv:2504.10760 SS2.2.1/SS3.5, with frozen/prescribed hydro):
    because mode B has no load feedback, the in-code once-per-step RK4 circuit must
    reproduce a fine-step reference integration of the SAME series-RLC ODE
        L dI/dt + (Z0 + R_loss) I + V_C = V_oc(t),   dV_C/dt = I / C,
    driven by the SAME deck-tabulated V_oc(t) waveform (exercising the deck-driven
    tabulated voltage source + R/L/C + loss-resistor hooks).  I(t) and V_C(t) from the
    user history must track the reference within a few percent of peak (the history
    samples the circuit one MHD cycle behind the output clock, so the band absorbs that
    sub-dt registration offset; any WIRING error -- wrong element, missing capacitor,
    un-consumed waveform, feedback applied in mode B -- is an order-of-magnitude miss).

(2) Mode C flux conservation + the Faraday tally: with Z0 = C = R_loss = 0 the series
    loop reduces to the pure inductor + coupled load, L dI/dt = V_oc - V_load, whose
    exact integral is flux conservation in the (series L + load) loop,
        L*I(t) + Phi(t) = V_oc * t        (constant V_oc, started from rest).
    This holds iff the circuit ODE, the Faraday feedback V_load = d(Phi)/dt (a global
    reduction), and the B_phi boundary are wired together correctly in the maglif path.
    The V_load history tally is validated against d(Phi)/dt differenced from the
    co-reported flux trace (Faraday's law on the run's own implosion), and the
    stagnation voltage spike -- the physical signature mode C exists to capture -- must
    appear late in the implosion.

The MPI companion (test_verify_maglif_circuit_mpicpu.py) reruns (2) on 2 ranks to pin
the reduction's MPI-safety.  Mode-A byte-identity is pinned by the existing maglif
suite (golden baselines + smoke test), which this change must not perturb.

Oracle: Layer 1 -- analytic / high-accuracy ODE reference (ADR-0008).
Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import numpy as np
import test_suite.testutils as testutils

# Values that must match inputs/maglif_circuit.athinput (the mode-C base deck).
V_OC = 2.0          # open-circuit (constant) voltage source [code units]
L_CIRCUIT = 1.0     # series inductance (mode C: pure L + coupled load)
TLIM = 1.25         # mode-C run end time
D_LINER = 1.0       # liner (dense shell) density

# Mode-B (voltage_rlc) circuit parameters -- passed as command-line overrides.
B_TLIM = 1.0        # mode-B run end time
B_L = 1.0           # series inductance
B_C = 0.5           # series capacitance
B_Z0 = 0.3          # fixed source/line resistance
B_RLOSS = 0.2       # loss resistor (constant; the z3018 calibration hook)
B_VPEAK = 2.0       # peak of the tabulated V_oc(t) = B_VPEAK * sin^2(pi t) pulse
ODE_TOL = 0.03      # max |sim - reference| / peak(reference) for I and V_C

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_circuit.athinput"
)
run_dir = testutils.pgen_run_dir("maglif")


def _hst_path(basename):
    return os.path.join(run_dir, f"{basename}.user.hst")


def _clear_stale(basename):
    """AthenaK APPENDS to an existing .user.hst; clear ours so the parse is this run."""
    if os.path.exists(_hst_path(basename)):
        os.remove(_hst_path(basename))


def _read_hst(basename):
    import athena_read

    return athena_read.hst(_hst_path(basename))


def _reference_rlc(t_end, ell, cap, r_tot, v_of_t, dt=1.0e-4):
    """High-accuracy (fine-step RK4) reference integration of the series-RLC ODE.

    State (I, V_C) from rest:  dI/dt = (V_oc(t) - r_tot*I - V_C)/L,  dV_C/dt = I/C
    (capacitor dropped for cap <= 0).  This is the frozen-hydro reference the paper's
    circuit-coupling verification integrates the code circuit against.
    """
    nstep = int(round(t_end / dt))
    times = np.empty(nstep + 1)
    cur = np.empty(nstep + 1)
    vcap = np.empty(nstep + 1)
    ii, vc, t = 0.0, 0.0, 0.0
    times[0], cur[0], vcap[0] = t, ii, vc

    def deriv(tt, i_, vc_):
        di = (v_of_t(tt) - r_tot * i_ - vc_) / ell
        dv = (i_ / cap) if cap > 0.0 else 0.0
        return di, dv

    for n in range(nstep):
        k1i, k1v = deriv(t, ii, vc)
        k2i, k2v = deriv(t + 0.5 * dt, ii + 0.5 * dt * k1i, vc + 0.5 * dt * k1v)
        k3i, k3v = deriv(t + 0.5 * dt, ii + 0.5 * dt * k2i, vc + 0.5 * dt * k2v)
        k4i, k4v = deriv(t + dt, ii + dt * k3i, vc + dt * k3v)
        ii += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
        vc += (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t += dt
        times[n + 1], cur[n + 1], vcap[n + 1] = t, ii, vc
    return times, cur, vcap


def test_maglif_voltage_rlc_matches_reference_ode():
    """Mode B (voltage + fixed RLC) in the maglif path vs a high-accuracy ODE reference.

    The paper's own circuit-coupling verification with frozen hydro: mode B has no load
    feedback, so I(t)/V_C(t) from the run must reproduce a fine-step integration of the
    same deck-tabulated series-RLC circuit.
    """
    basename = "maglif_rlc"
    _clear_stale(basename)
    os.makedirs(run_dir, exist_ok=True)

    # Deck-driven TABULATED open-circuit voltage: V_oc(t) = B_VPEAK * sin^2(pi t),
    # densely sampled so both the code and the reference consume the same
    # piecewise-linear waveform (the drive_source file reader's clamped interpolation).
    t_tab = np.arange(0.0, 1.3 + 1.0e-12, 0.005)
    v_tab = B_VPEAK * np.sin(np.pi * t_tab) ** 2
    vfile = os.path.join(run_dir, "maglif_rlc_voltage.dat")
    with open(vfile, "w") as f:
        f.write("# tabulated open-circuit voltage V_oc(t) (mode-B verification, #236)\n")
        f.write(f"{len(t_tab)}\n")
        for tt, vv in zip(t_tab, v_tab):
            f.write(f"{tt:.10e} {vv:.10e}\n")

    try:
        args = [
            f"job/basename={basename}",
            f"time/tlim={B_TLIM}",
            "problem/current_mode=voltage_rlc",
            "problem/voltage_waveform=tabulated",
            f"problem/voltage_file={vfile}",
            f"problem/circuit_L={B_L}",
            f"problem/circuit_C={B_C}",
            f"problem/circuit_Z0={B_Z0}",
            f"problem/r_loss={B_RLOSS}",
        ]
        ok = testutils.run_pgen("maglif", input_file, args=args)
        assert ok, "maglif voltage_rlc (mode B) run failed."

        hst = _read_hst(basename)
        assert "I_circuit" in hst and "V_cap" in hst, (
            "maglif history lacks the circuit columns (I_circuit/V_cap): drive modes "
            "B/C are not wired into the maglif path (#236)"
        )
        th = hst["time"]
        assert len(th) > 50, f"too few history samples: {len(th)}"

        # High-accuracy reference of the SAME piecewise-linear tabulated waveform.
        def v_of_t(tt):
            return np.interp(tt, t_tab, v_tab)

        tref, i_ref, vc_ref = _reference_rlc(
            B_TLIM, B_L, B_C, B_Z0 + B_RLOSS, v_of_t
        )
        i_ref_h = np.interp(th, tref, i_ref)
        vc_ref_h = np.interp(th, tref, vc_ref)

        i_peak = np.abs(i_ref).max()
        vc_peak = np.abs(vc_ref).max()
        assert i_peak > 0.1, f"reference current is degenerate (peak {i_peak:.3e})"

        err_i = np.abs(hst["I_circuit"] - i_ref_h).max() / i_peak
        assert err_i < ODE_TOL, (
            f"mode-B circuit current deviates from the high-accuracy ODE reference: "
            f"max |dI|/peak = {err_i:.4f} >= {ODE_TOL}"
        )
        err_v = np.abs(hst["V_cap"] - vc_ref_h).max() / vc_peak
        assert err_v < ODE_TOL, (
            f"mode-B capacitor voltage deviates from the ODE reference: "
            f"max |dV_C|/peak = {err_v:.4f} >= {ODE_TOL}"
        )

        # The integrated current actually drives the B_phi boundary: poloidal flux
        # entered the domain (a wiring check on ApplyDriveBphiBC in circuit mode).
        assert np.abs(hst["Bphi_flux"]).max() > 0.01, (
            f"no poloidal flux entered the domain (max |Phi| = "
            f"{np.abs(hst['Bphi_flux']).max():.3e}): circuit current not driving the "
            "B_phi boundary"
        )
    finally:
        _clear_stale(basename)
        if os.path.exists(vfile):
            os.remove(vfile)


def test_maglif_coupled_circuit_flux_conservation():
    """Mode C (coupled feedback) in the maglif path: exact flux conservation + Faraday.

    With Z0 = C = R_loss = 0 the coupled system obeys L*I + Phi = V_oc*t exactly -- the
    coupled-circuit reference (high-accuracy ODE with the run's own prescribed load) --
    and the V_load tally must equal d(Phi)/dt of the co-reported flux (Faraday's law).
    """
    basename = "maglif_circuit"
    _clear_stale(basename)
    try:
        ok = testutils.run_pgen("maglif", input_file)
        assert ok, "maglif coupled_circuit (mode C) run failed."

        hst = _read_hst(basename)
        assert "I_circuit" in hst and "V_load" in hst and "Bphi_flux" in hst, (
            "maglif history lacks the circuit columns: drive mode C is not wired into "
            "the maglif path (#236)"
        )
        th = hst["time"]
        cur = hst["I_circuit"]
        v_load = hst["V_load"]
        flux = hst["Bphi_flux"]
        assert len(th) > 50, f"too few history samples: {len(th)}"

        # (A) Flux conservation -- the coupled-circuit reference L*I + Phi = V_oc*t.
        # The explicit one-step feedback lag makes the residual ~ V_oc*dt; it is tight
        # (<1%) once past the start-up transient.
        resid = np.abs(L_CIRCUIT * cur + flux - V_OC * th)
        assert resid.max() < 0.02, (
            f"flux conservation L*I+Phi=V_oc*t violated in the maglif path: max abs "
            f"residual {resid.max():.4f} (>= 0.02)"
        )
        late = th > 0.3
        rel_resid = (resid[late] / (V_OC * th[late])).max()
        assert rel_resid < 0.02, (
            f"flux conservation relative residual too large past startup: "
            f"{rel_resid:.4f}"
        )

        # (B) Faraday tally -- V_load = d(Phi)/dt on the run's own implosion (skip the
        # noisy single-step end samples).
        dphidt = np.gradient(flux, th)
        win = (th > 0.1) & (th < 0.98 * TLIM)
        rel_far = np.abs(v_load[win] - dphidt[win]) / (np.abs(dphidt[win]) + 1.0e-3)
        assert np.median(rel_far) < 0.03, (
            f"Faraday V_load != d(Phi)/dt: median rel error {np.median(rel_far):.4f}"
        )

        # Current ramps from rest to a clear load current.
        assert cur[0] == 0.0 and cur[-1] > 1.0, (
            f"current did not ramp from rest: I0={cur[0]:.3f} I_end={cur[-1]:.3f}"
        )

        # (C) Stagnation voltage spike: the load voltage peaks at peak compression
        # (late), far above the run-in value -- the mode-C physics signature.
        t_peak = th[np.argmax(dphidt)]
        v_early = np.interp(0.2, th, dphidt)
        assert t_peak > 0.7 * TLIM, (
            f"load-voltage peak not at stagnation: t={t_peak:.3f}"
        )
        assert dphidt.max() > 10.0 * abs(v_early), (
            f"no stagnation spike: peak {dphidt.max():.3f} vs early {v_early:.4f}"
        )

        # (D) The magnetic piston actually imploded the liner (peak compression from
        # the cons history columns the deck also enrolls).
        assert "maxdens" in hst, "cons history columns missing alongside circuit ones"
        assert hst["maxdens"].max() > 1.3 * D_LINER, (
            f"no liner compression: max density {hst['maxdens'].max():.3f} "
            f"(IC liner {D_LINER})"
        )
    finally:
        _clear_stale(basename)
