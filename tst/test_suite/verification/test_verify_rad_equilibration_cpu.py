"""
Coupled grey radiation-hydro verification (ADR-0001, issue #28).

Drives the grey FLD spatial diffusion operator (FLDGreyOperator, #17) together with the
per-cell point-implicit matter-radiation coupling (PointImplicitGreyCoupling, #23) via the
``radiation_fld_equilibration`` user pgen (built/run with testutils.run_pgen), then checks
two coupled radiation-hydro problems and stores golden baselines + plots through the
shared verification harness:

  1. Matter-radiation EQUILIBRATION.  A spatially-uniform box relaxes from thermal
     non-equilibrium to radiative equilibrium ``a T^4 = E_r`` by the point-implicit
     coupling.  Two cases (hot radiation preheating cold matter, and hot matter cooling
     into cold radiation) are compared against a high-accuracy RK4 integration of the
     coupled ODE (the reference solution); the final state must match the analytic
     equilibrium (``a T_eq^4 + c_v T_eq = E_r^0 + e_mat^0``); and total energy
     ``E_r + e_mat`` is conserved to machine precision throughout.

  2. Coupled FLD diffusion + matter equilibration (radiating absorption front).  A cold
     optically-thick slab is heated from an inner-x1 Dirichlet radiation source; the
     radiation diffuses (FLD) while the strong local coupling locks the matter to the
     local radiation field.  The test asserts local radiative equilibrium ``a T^4 = E_r``
     behind the front, that the radiation profile tracks the equilibrium-diffusion erfc
     Marshak wave ``E_r(x) = e_floor + (e_source-e_floor) erfc(x/(2 sqrt(D t)))`` with
     ``D = c/(3 chi)``, and that the matter is heated to local equilibrium.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

_erfc = np.vectorize(math.erfc)

# Problem (constants must match tst/inputs/rad_equilibration.athinput).
PROBLEM = "radiation_fld_equilibration"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "rad_equilibration.athinput"
)

C_LIGHT, A_RAD, GAMMA = 1.0, 1.0, 5.0 / 3.0

# -- PART 1 (equilibration) parameters --
EQ_RHO, EQ_CHI_A = 1.0, 1.0
CV_EQ = EQ_RHO / (GAMMA - 1.0)               # matter volumetric heat capacity
TMAT_A, TRAD_A = 0.5, 1.5                     # case A: cold matter, hot radiation
TMAT_B, TRAD_B = 1.5, 0.5                     # case B: hot matter, cold radiation

# -- PART 2 (absorption front) parameters --
CHI = 1.0e3
E_SOURCE, E_FLOOR, T_FLOOR = 1.0, 1.0e-3, 0.1
TLIM_FLD, X1MIN = 50.0, 0.0
D_DIFF = C_LIGHT / (3.0 * CHI)                # equilibrium-diffusion coefficient
WIN_LO, WIN_HI = 0.01, 0.15                   # well-resolved interior of the front


def _rk4_relax(er0, em0, tgrid, nsub=80):
    """High-accuracy reference: RK4 integration of the coupled relaxation ODE
    dE_r/dt = c chi_a (a (e_mat/cv)^4 - E_r), de_mat/dt = -dE_r/dt, sampled on tgrid."""
    er, em = er0, em0
    out_er, out_em = [er], [em]

    def f(e_r, e_m):
        rate = C_LIGHT * EQ_CHI_A * (A_RAD * (e_m / CV_EQ) ** 4 - e_r)
        return rate, -rate

    for k in range(1, len(tgrid)):
        h = (tgrid[k] - tgrid[k - 1]) / nsub
        for _ in range(nsub):
            k1 = f(er, em)
            k2 = f(er + 0.5 * h * k1[0], em + 0.5 * h * k1[1])
            k3 = f(er + 0.5 * h * k2[0], em + 0.5 * h * k2[1])
            k4 = f(er + h * k3[0], em + h * k3[1])
            er += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            em += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        out_er.append(er)
        out_em.append(em)
    return np.array(out_er), np.array(out_em)


def _equilibrium_temp(total_energy):
    """Analytic radiative-equilibrium temperature: root of a T^4 + cv T = S (Newton)."""
    T = (total_energy / A_RAD) ** 0.25
    for _ in range(100):
        g = A_RAD * T ** 4 + CV_EQ * T - total_energy
        gp = 4.0 * A_RAD * T ** 3 + CV_EQ
        T -= g / gp
    return T


def _marshak_exact(x):
    """Equilibrium-diffusion erfc Marshak wave (radiation-dominated coupling limit)."""
    z = (x - X1MIN) / (2.0 * np.sqrt(D_DIFF * TLIM_FLD))
    return E_FLOOR + (E_SOURCE - E_FLOOR) * _erfc(z)


def test_verify_rad_equilibration():
    """Run the coupled radiation-hydro driver; verify equilibration + absorption front."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    hist_path = os.path.join(run_dir, "rad_equil_history.dat")
    front_path = os.path.join(run_dir, "rad_equil_front.dat")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "coupled rad-hydro run failed"

        # ===== Part 1: matter-radiation equilibration =====
        hist = np.loadtxt(hist_path)
        t, er_a, em_a, er_b, em_b = hist.T

        for tag, er, em in (("A", er_a, em_a), ("B", er_b, em_b)):
            # (i) trajectory matches the RK4 reference solution (within backward-Euler
            #     truncation -- a coupling bug diverges from the physical relaxation).
            ref_er, ref_em = _rk4_relax(er[0], em[0], t)
            assert np.allclose(er, ref_er, rtol=0.08, atol=0.04), (
                f"case {tag}: E_r trajectory off RK4 reference "
                f"(max|d|={np.max(np.abs(er - ref_er)):.3e})"
            )
            assert np.allclose(em, ref_em, rtol=0.08, atol=0.04), (
                f"case {tag}: e_mat trajectory off RK4 reference "
                f"(max|d|={np.max(np.abs(em - ref_em)):.3e})"
            )
            # (ii) total energy E_r + e_mat conserved to machine precision every step.
            drift = np.max(np.abs((er + em) - (er[0] + em[0])))
            assert drift < 1.0e-9, f"case {tag}: energy not conserved (drift={drift:.3e})"
            # (iii) final state matches the analytic radiative equilibrium.
            t_eq = _equilibrium_temp(er[0] + em[0])
            assert abs(er[-1] - A_RAD * t_eq ** 4) < 1.0e-4 * (er[0] + em[0]), (
                f"case {tag}: final E_r off analytic equilibrium a T_eq^4"
            )
            assert abs(em[-1] - CV_EQ * t_eq) < 1.0e-4 * (er[0] + em[0]), (
                f"case {tag}: final e_mat off analytic equilibrium c_v T_eq"
            )
            # (iv) monotone (non-overshooting) approach of e_mat toward equilibrium.
            dist = np.abs(em - em[-1])
            assert np.all(np.diff(dist) <= 1.0e-9), (
                f"case {tag}: matter energy not monotone toward equilibrium"
            )

        # subsample the trajectory for a clean baseline + plot.
        s = slice(None, None, 10)
        harness.verify(
            "rad_equilibration",
            t[s].tolist(),
            {
                "E_r_A": er_a[s].tolist(),
                "e_mat_A": em_a[s].tolist(),
                "E_r_B": er_b[s].tolist(),
                "e_mat_B": em_b[s].tolist(),
            },
            coord_label="t",
            xlabel="time",
            title="Matter-radiation equilibration (A: rad heats matter; B: matter cools)",
            rtol=1.0e-5,
            atol=1.0e-8,
        )

        # ===== Part 2: coupled FLD diffusion + matter equilibration (front) =====
        front = np.loadtxt(front_path)
        x, e_r, e_mat, temp, at4 = front.T

        # (i) local radiative equilibrium a T^4 = E_r enforced by the strong coupling.
        loc_eq = np.max(np.abs(at4 - e_r) / np.maximum(e_r, 1.0e-12))
        assert loc_eq < 1.0e-3, f"local equilibrium a T^4 == E_r broken: rel {loc_eq:.3e}"

        # (ii) radiation profile tracks the equilibrium-diffusion erfc Marshak wave.
        e_an = _marshak_exact(x)
        mask = (x > WIN_LO) & (x < WIN_HI)
        contrast = E_SOURCE - E_FLOOR
        l1 = float(np.mean(np.abs(e_r[mask] - e_an[mask])))
        linf = float(np.max(np.abs(e_r[mask] - e_an[mask])))
        assert l1 < 1.0e-2 * contrast, f"front L1 off erfc Marshak: {l1:.3e}"
        assert linf < 2.0e-2 * contrast, f"front Linf off erfc Marshak: {linf:.3e}"

        # (iii) matter is heated to local equilibrium deep behind the front.
        assert temp[0] > 0.5, f"matter not heated behind the front: T={temp[0]:.3e}"
        assert temp[0] > 4.0 * T_FLOOR, "matter barely above its cold floor"

        harness.verify(
            "rad_equilibration_front",
            x[mask].tolist(),
            {
                "E_r": e_r[mask].tolist(),
                "E_r_erfc": e_an[mask].tolist(),
                "T_matter": temp[mask].tolist(),
            },
            coord_label="x1v",
            xlabel="x",
            title="Coupled FLD absorption front: E_r vs erfc Marshak + locked matter T",
            rtol=1.0e-5,
            atol=1.0e-7,
        )
    finally:
        for p in (hist_path, front_path):
            try:
                os.remove(p)
            except OSError:
                pass
        testutils.cleanup()
