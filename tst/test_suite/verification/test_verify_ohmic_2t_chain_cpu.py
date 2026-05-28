"""
2T equilibration-chain verification (ADR-0002/0003, issue [14c]/#30).

Drives the Ohmic->electron heating source (three_temp::ApplyOhmicElectronHeating) and the
2T point-implicit radiation coupling targeting T_e
(radiationfld::PointImplicitElectronRadiationCoupling) via the ``ohmic_2t_chain`` pgen
(built/run with testutils.run_pgen), then checks the coupled chain
Ohmic -> electron -> radiation/ion and stores a golden baseline + plot through the shared
verification harness.

A spatially-uniform cell carries four energy reservoirs (magnetic, electron, ion,
radiation).  Each step (1) moves the Ohmic magnetic-energy decrement Q = eta|J|^2 dt onto
the electrons (E_tot held fixed -> ions untouched) and (2) exchanges energy between the
electrons (T_e) and the radiation via the point-implicit backward-Euler coupling (mirrored
in E_tot -> ions untouched).  The test asserts:

  * the (E_r, e_ele) trajectory matches a high-accuracy RK4 integration of the coupled ODE
    ``de_ele/dt = Q - c chi_a (a T_e^4 - E_r)``, ``dE_r/dt = c chi_a (a T_e^4 - E_r)``
    (within the backward-Euler / operator-split truncation -- a routing bug diverges);
  * both the electrons and the radiation are heated by the chain;
  * the IONS are UNTOUCHED: e_ion (recovered by subtraction) stays constant to machine
    precision (the whole point of routing Ohmic + radiation to T_e, ADR-0003); and
  * the closed system me + e_ele + e_ion + E_r is conserved to machine precision.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# Problem (constants must match tst/inputs/ohmic_2t_chain.athinput).
PROBLEM = "ohmic_2t_chain"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "ohmic_2t_chain.athinput"
)

C_LIGHT, A_RAD, CV_ELE, CHI_A = 1.0, 1.0, 1.0, 1.0
ETA, JX, JY, JZ = 0.1, 1.0, 0.0, 0.0
Q_RATE = ETA * (JX * JX + JY * JY + JZ * JZ)   # Ohmic dissipation rate eta|J|^2
EION0 = 1.0


def _rk4_chain(er0, ee0, tgrid, nsub=80):
    """High-accuracy reference: RK4 of the driven coupled ODE
    de_ele/dt = Q - c chi_a (a (e_ele/cv)^4 - E_r),
    dE_r/dt   =     c chi_a (a (e_ele/cv)^4 - E_r), sampled on tgrid."""
    er, ee = er0, ee0
    out_er, out_ee = [er], [ee]

    def f(e_r, e_e):
        rate = C_LIGHT * CHI_A * (A_RAD * (e_e / CV_ELE) ** 4 - e_r)
        return rate, Q_RATE - rate     # dE_r/dt, de_ele/dt

    for k in range(1, len(tgrid)):
        h = (tgrid[k] - tgrid[k - 1]) / nsub
        for _ in range(nsub):
            k1 = f(er, ee)
            k2 = f(er + 0.5 * h * k1[0], ee + 0.5 * h * k1[1])
            k3 = f(er + 0.5 * h * k2[0], ee + 0.5 * h * k2[1])
            k4 = f(er + h * k3[0], ee + h * k3[1])
            er += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            ee += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        out_er.append(er)
        out_ee.append(ee)
    return np.array(out_er), np.array(out_ee)


def test_verify_ohmic_2t_chain():
    """Run the 2T equilibration-chain driver; verify trajectory, ion isolation, energy."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    hist_path = os.path.join(run_dir, "ohmic_2t_chain.dat")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "2T chain run failed"

        hist = np.loadtxt(hist_path)
        t, e_r, e_ele, e_ion, e_sys = hist.T

        # (i) trajectory matches the RK4 reference (within backward-Euler truncation).
        ref_er, ref_ee = _rk4_chain(e_r[0], e_ele[0], t)
        assert np.allclose(e_r, ref_er, rtol=0.08, atol=0.04), (
            f"E_r trajectory off RK4 reference (max|d|={np.max(np.abs(e_r-ref_er)):.3e})"
        )
        assert np.allclose(e_ele, ref_ee, rtol=0.08, atol=0.04), (
            f"e_ele trajectory off RK4 ref (max|d|={np.max(np.abs(e_ele-ref_ee)):.3e})"
        )

        # (ii) the chain heats both the electrons and the radiation.
        assert e_ele[-1] > e_ele[0] + 1.0e-3, "electrons not heated by the chain"
        assert e_r[-1] > e_r[0] + 1.0e-3, "radiation not heated by the electron coupling"

        # (iii) IONS UNTOUCHED: e_ion stays constant to machine precision (ADR-0003).
        ion_drift = float(np.max(np.abs(e_ion - EION0)))
        assert ion_drift < 1.0e-9, f"ions NOT untouched: e_ion drift {ion_drift:.3e}"

        # (iv) closed system me + e_ele + e_ion + E_r conserved to machine precision.
        sys_drift = float(np.max(np.abs(e_sys - e_sys[0])))
        assert sys_drift < 1.0e-9, f"total energy not conserved (drift {sys_drift:.3e})"

        # subsample for a clean baseline + plot.
        s = slice(None, None, 10)
        harness.verify(
            "ohmic_2t_chain",
            t[s].tolist(),
            {
                "E_r": e_r[s].tolist(),
                "e_ele": e_ele[s].tolist(),
                "e_ion": e_ion[s].tolist(),
            },
            coord_label="t",
            xlabel="time",
            title="2T chain: Ohmic->electron->radiation (ions untouched)",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        try:
            os.remove(hist_path)
        except OSError:
            pass
        testutils.cleanup()
