"""
Resistive magnetic-field-decay verification (issue [7a], ADR-0003).

Drives full MHD with the variable Spitzer/Braginskii resistivity (<mhd> resistivity =
spitzer) on a uniform, static 1-D periodic box seeded with a small single-mode transverse
field B_y(x) = by0 sin(k x), via the ``resistive_decay`` user pgen (testutils.run_pgen).
With rho and p uniform and the perturbation linear, the magnetic diffusivity eta(rho,T_e)
is uniform, so B_y obeys the induction-equation diffusion

    B_y(x,t) = by0 sin(k x) exp(-eta k^2 t),

i.e. the antinode amplitude decays at the analytic rate eta k^2.  The test:

  1. measures the decay rate from the simulated amplitude (projection onto sin(k x) at t=0
     and t=tlim) and checks it against the analytic Spitzer eta(rho,T_e) computed
     independently from the SIM-61 transport chain -- "magnetic-field decay with
     T-dependent eta matches the analytic rate";
  2. confirms the decayed profile is still a clean single sine mode (no spurious
     harmonics);
  3. baselines the t=tlim B_y profile through the shared verification harness.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# Column indices in the non-sliced 1-D tab dump (gid, i, x1v, dens, velx, vely, velz,
# eint, bcc1, bcc2, bcc3); athena_read.tab assumes a sliced layout, so read by position.
COL_X1V = 2
COL_BCC2 = 9

PROBLEM = "resistive_decay"
input_file = os.path.join(testutils._repo_root(), "tst", "inputs",
                          "resistive_decay.athinput")

# Run parameters -- MUST match tst/inputs/resistive_decay.athinput.
LX = 1.0
NWAVE = 1.0
K = 2.0 * math.pi * NWAVE / LX
TLIM = 1.0
BY0 = 1.0e-2
# Resistivity model inputs (code state + linear code<->SI conversions).
RHO0, P0 = 1.0, 1.0
TEMP_CODE = P0 / RHO0          # ideal-gas code temperature T = p/rho
DENS_SI = 0.167
TEMP_SI = 1.160451e6
ETA_SI = 0.1
ZBAR = 1.0
LNLAMBDA = 5.0

# SI physical constants (match diffusion/braginskii_transport.hpp).
KB = 1.380649e-23
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
ALPHA0 = 0.5129               # Braginskii Z=1 parallel resistivity coefficient


def _spitzer_eta_code():
    """Analytic Spitzer parallel magnetic diffusivity (code units), replaying the exact
    SIM-61 chain that diffusion/variable_resistivity.hpp uses."""
    n_i = RHO0 * DENS_SI / M_P
    n_e = ZBAR * n_i
    t_e = TEMP_CODE * TEMP_SI
    kt = KB * t_e
    fourpieps0_sq = (4.0 * math.pi * EPS0) ** 2
    tau_e = (3.0 * math.sqrt(M_E) * kt ** 1.5 * fourpieps0_sq /
             (4.0 * math.sqrt(2.0 * math.pi) * n_e * ZBAR * QE ** 4 * LNLAMBDA))
    eta0 = M_E / (n_e * QE ** 2 * tau_e)
    eta_par_si = ALPHA0 * eta0          # Ohm m
    eta_mag_si = eta_par_si / MU0       # m^2/s
    return eta_mag_si * ETA_SI          # code units


def _sine_amplitude(x, by):
    """Projection of by(x) onto sin(k x): for by = A sin(k x) sampled over a full period
    this returns A."""
    return 2.0 * float(np.mean(by * np.sin(K * x)))


def test_verify_resistive_decay():
    """Run resistive decay; verify the decay rate matches the analytic Spitzer eta."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    f0 = os.path.join(run_dir, "tab", "resistive_decay.bcc.00000.tab")
    f1 = os.path.join(run_dir, "tab", "resistive_decay.bcc.00001.tab")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "resistive_decay run failed"

        arr0 = np.loadtxt(f0, comments="#")
        arr1 = np.loadtxt(f1, comments="#")
        x = arr0[:, COL_X1V]
        by_init = arr0[:, COL_BCC2]
        by_final = arr1[:, COL_BCC2]

        # (1) decay rate from the sine-mode amplitude at t=0 and t=tlim.
        a0 = _sine_amplitude(x, by_init)
        a1 = _sine_amplitude(x, by_final)
        assert a0 > 0.0 and a1 > 0.0, f"unexpected amplitudes a0={a0:g} a1={a1:g}"
        # t=0 amplitude must recover the seeded by0.
        assert abs(a0 - BY0) < 1.0e-4 * BY0, f"initial amplitude {a0:g} != by0 {BY0:g}"
        eta_meas = -math.log(a1 / a0) / (K * K * TLIM)
        eta_analytic = _spitzer_eta_code()
        rel = abs(eta_meas / eta_analytic - 1.0)
        # 2nd-order CT diffusion of mode k has discrete eigenvalue eta*k^2*(1-(k dx)^2/12)
        # ~ 0.08% low here; 2% tolerance is tight enough to catch a wiring/factor bug.
        assert rel < 0.02, (
            f"measured decay rate eta={eta_meas:.6e} does not match analytic Spitzer "
            f"eta={eta_analytic:.6e} (rel err {rel:.3e})"
        )

        # (2) the decayed field is still a clean single sine mode.
        resid = by_final - a1 * np.sin(K * x)
        assert np.max(np.abs(resid)) < 1.0e-2 * abs(a1), (
            "decayed B_y is not a clean single sine mode"
        )

        # (3) baseline the t=tlim profile (regression guard).
        harness.verify(
            "resistive_decay",
            x.tolist(),
            {"By_final": by_final.tolist()},
            coord_label="x1v",
            xlabel="x",
            title="Resistive decay of B_y(x) with Spitzer eta(rho,T_e)",
            rtol=1.0e-5,
            atol=1.0e-10,
        )
    finally:
        testutils.cleanup()
