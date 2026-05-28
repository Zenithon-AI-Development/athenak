"""
Coupled MULTIGROUP radiation-hydro verification (ADR-0001, issue [17c]/#41).

Drives the N-group FLD spatial diffusion operator (FLDMultigroupOperator, #24) together
with the per-cell point-implicit MULTIGROUP matter-radiation coupling
(MultigroupArrowheadCoupling, #35) and the ideal-gas matter heat capacity via the
``radiation_fld_multigroup_equilibration`` user pgen (built/run with testutils.run_pgen),
then checks two coupled multigroup radiation-hydro problems and stores golden baselines +
plots through the shared verification harness.  It is the multigroup analogue of the grey
``test_verify_rad_equilibration`` (#28).

  1. Multigroup NON-EQUILIBRIUM spectral relaxation.  A spatially-uniform box relaxes from
     a NON-Planck radiation spectrum (all energy in the lowest group) + thermal
     non-equilibrium (electrons, ions and radiation at different temperatures) to the
     PLANCK SPECTRUM at a common temperature, by the multigroup point-implicit coupling
     alone.  Two cases (hot radiation preheating cold electrons, and hot electrons cooling
     into cold radiation, each exchanging with the ions) are compared against a
     high-accuracy RK4 integration of the coupled multigroup ODE (the reference solution);
     the final per-group state must match the analytic Planck-spectrum equilibrium
     ``E_g = a T_eq^4 [F(x_{g+1}) - F(x_g)]`` (the SPECTRAL check); and total energy
     ``sum_g E_g + e_ele + e_ion`` is conserved to machine precision.

  2. Coupled multigroup FLD diffusion + matter coupling (multigroup Marshak front).  A
     cold optically-thick slab is heated from an inner-x1 Dirichlet radiation source; each
     group diffuses (FLD, own D_g) while the strong local coupling thermalises the
     spectrum to the local Planck spectrum at the matter temperature.  The test asserts
     the LOCAL PLANCK-SPECTRUM LOCK ``E_g(x) = a T_e(x)^4 [F(x_{g+1}) - F(x_g)]`` in the
     propagating front (the spectral reference), a monotone front and matter heating.

The Planck fraction ``F(x) = 15/pi^4 int_0^x t^3/(e^t-1) dt`` is built here by an
INDEPENDENT numerical quadrature (not the code's own series), so the spectral check is a
genuine cross-check.  Per-group extinctions and group boundaries are read back from the
pgen's groups file (so the oracle uses the actual table lookups).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# Problem (constants must match tst/inputs/multigroup_rad_equilibration.athinput).
PROBLEM = "radiation_fld_multigroup_equilibration"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "multigroup_rad_equilibration.athinput"
)
fixture = os.path.join(
    testutils._repo_root(), "inputs", "radiation_fld", "multigroup_rad_hydro_opacity.cn4"
)

C_LIGHT, A_RAD, KBOLTZ = 1.0, 1.0, 30.0
GAMMA = 2.0
CV_ELE, CV_ION = 1.0 / (GAMMA - 1.0), 1.0 / (GAMMA - 1.0)   # rho_ele/ion = 1.0
C_EI = 2.0

# PART 1 initial conditions (case A: cold e + hot ion + radiation in lowest group;
# case B: hot e + cold ion + cold radiation).
TE_A, TI_A, ERAD_A = 0.3, 0.8, 1.0
TE_B, TI_B = 1.5, 0.5
ERAD_FLOOR = 1.0e-3

# PART 2 interior window: clear of the near-wall boundary layer (the white Dirichlet
# source is spectrally inconsistent with the thermalised interior there) and in the front.
WIN_LO, WIN_HI = 0.13, 0.45


def _planck_fraction(x):
    """F(x) = (15/pi^4) int_0^x t^3/(e^t-1) dt, by independent numerical quadrature."""
    x = np.asarray(x, dtype=float)
    flat = np.ravel(x)
    out = np.zeros_like(flat)
    for i, xv in enumerate(flat):
        if xv <= 0.0:
            continue
        n = max(400, int(xv * 400))
        t = np.linspace(0.0, xv, n + 1)
        safe = np.where(t > 1.0e-8, t, 1.0)
        integ = np.where(t > 1.0e-8, t ** 3 / np.expm1(safe), t ** 2)
        out[i] = 15.0 / np.pi ** 4 * np.trapezoid(integ, t)
    return out.reshape(x.shape)


def _planck_spectrum(temp, bounds):
    """Per-group Planck energy slice a T^4 [F(eps_hi/(kB T)) - F(eps_lo/(kB T))]."""
    temp = np.asarray(temp, dtype=float)
    frac = (_planck_fraction(bounds[1:][None, :] / (KBOLTZ * temp[:, None]))
            - _planck_fraction(bounds[:-1][None, :] / (KBOLTZ * temp[:, None])))
    return A_RAD * (temp[:, None] ** 4) * frac


def _equilibrium_temp(total_energy, bounds):
    """Analytic common equilibrium temperature: root of
    a T^4 sum_g[F(x_hi)-F(x_lo)] + (cv_ele+cv_ion) T = total_energy (Newton)."""
    cv = CV_ELE + CV_ION

    def resid(temp):
        spec = _planck_spectrum(np.array([temp]), bounds).sum()
        return spec + cv * temp - total_energy

    temp = (total_energy / A_RAD) ** 0.25
    for _ in range(80):
        f = resid(temp)
        h = 1.0e-6 * temp
        fp = (resid(temp + h) - f) / h
        temp -= f / fp
    return temp


def _rk4_relax(eg0, ee0, ei0, tgrid, chi, bounds, nsub=40):
    """High-accuracy reference: RK4 integration of the coupled multigroup ODE
    dE_g/dt = c chi_g (B_g(Te) - E_g); de_ele/dt = -sum dE_g/dt + c_ei (Ti - Te);
    de_ion/dt = -c_ei (Ti - Te), sampled on tgrid."""
    def deriv(eg, ee, ei):
        te, ti = ee / CV_ELE, ei / CV_ION
        bg = _planck_spectrum(np.array([te]), bounds)[0]
        deg = C_LIGHT * chi * (bg - eg)
        dee = -deg.sum() + C_EI * (ti - te)
        dei = -C_EI * (ti - te)
        return deg, dee, dei

    eg, ee, ei = eg0.copy(), ee0, ei0
    out_eg, out_ee, out_ei = [eg.copy()], [ee], [ei]
    for k in range(1, len(tgrid)):
        h = (tgrid[k] - tgrid[k - 1]) / nsub
        for _ in range(nsub):
            k1 = deriv(eg, ee, ei)
            k2 = deriv(eg + 0.5 * h * k1[0], ee + 0.5 * h * k1[1], ei + 0.5 * h * k1[2])
            k3 = deriv(eg + 0.5 * h * k2[0], ee + 0.5 * h * k2[1], ei + 0.5 * h * k2[2])
            k4 = deriv(eg + h * k3[0], ee + h * k3[1], ei + h * k3[2])
            eg = eg + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            ee = ee + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            ei = ei + h / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        out_eg.append(eg.copy())
        out_ee.append(ee)
        out_ei.append(ei)
    return np.array(out_eg), np.array(out_ee), np.array(out_ei)


def test_verify_multigroup_rad_hydro():
    """Run the coupled multigroup rad-hydro driver; verify relaxation + Marshak front."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    hist_path = os.path.join(run_dir, "multigroup_rad_equil.dat")
    grp_path = os.path.join(run_dir, "multigroup_rad_groups.dat")
    front_path = os.path.join(run_dir, "multigroup_rad_front.dat")
    try:
        assert testutils.run_pgen(
            PROBLEM, input_file, args=[f"problem/opacity_file={fixture}"]
        ), "coupled multigroup rad-hydro run failed"

        # per-group coefficients: ig gb_lo gb_hi chi_P1 chi_P2 chi_R2 D2
        grp = np.atleast_2d(np.loadtxt(grp_path))
        gb_lo, gb_hi = grp[:, 1], grp[:, 2]
        chi_p1, chi_r2, d2 = grp[:, 3], grp[:, 5], grp[:, 6]
        ngroups = grp.shape[0]
        bounds = np.concatenate([[gb_lo[0]], gb_hi])
        assert ngroups >= 2, f"expected multiple groups, got {ngroups}"

        # ===== Part 1: multigroup non-equilibrium spectral relaxation =====
        hist = np.loadtxt(hist_path)
        t = hist[:, 0]
        eg_a = hist[:, 1:1 + ngroups]
        ee_a, ei_a = hist[:, 1 + ngroups], hist[:, 2 + ngroups]
        off = 3 + ngroups
        eg_b = hist[:, off:off + ngroups]
        ee_b, ei_b = hist[:, off + ngroups], hist[:, off + ngroups + 1]

        for tag, eg, ee, ei in (("A", eg_a, ee_a, ei_a), ("B", eg_b, ee_b, ei_b)):
            etot0 = eg[0].sum() + ee[0] + ei[0]

            # (i) trajectory matches the RK4 reference (within backward-Euler truncation).
            ref_eg, ref_ee, ref_ei = _rk4_relax(eg[0], ee[0], ei[0], t, chi_p1, bounds)
            for g in range(ngroups):
                assert np.allclose(eg[:, g], ref_eg[:, g], rtol=0.08, atol=0.05), (
                    f"case {tag}: group {g} E trajectory off RK4 reference "
                    f"(max|d|={np.max(np.abs(eg[:, g] - ref_eg[:, g])):.3e})"
                )
            assert np.allclose(ee, ref_ee, rtol=0.08, atol=0.05), (
                f"case {tag}: e_ele trajectory off RK4 reference"
            )
            assert np.allclose(ei, ref_ei, rtol=0.08, atol=0.05), (
                f"case {tag}: e_ion trajectory off RK4 reference"
            )

            # (ii) total energy sum_g E_g + e_ele + e_ion conserved to roundoff each step.
            etot = eg.sum(1) + ee + ei
            drift = np.max(np.abs(etot - etot0))
            assert drift < 1.0e-9, f"case {tag}: energy not conserved (drift={drift:.3e})"

            # (iii) final state matches the analytic Planck-spectrum equilibrium.
            t_eq = _equilibrium_temp(etot0, bounds)
            eg_eq = _planck_spectrum(np.array([t_eq]), bounds)[0]
            for g in range(ngroups):
                assert abs(eg[-1, g] - eg_eq[g]) < 1.0e-3 * etot0, (
                    f"case {tag}: final group {g} off analytic Planck spectrum "
                    f"(E={eg[-1, g]:.5e} vs {eg_eq[g]:.5e})"
                )
            assert abs(ee[-1] - CV_ELE * t_eq) < 1.0e-3 * etot0, (
                f"case {tag}: final e_ele off equilibrium cv_ele T_eq"
            )
            assert abs(ei[-1] - CV_ION * t_eq) < 1.0e-3 * etot0, (
                f"case {tag}: final e_ion off equilibrium cv_ion T_eq"
            )

        # (iv) the spectrum genuinely thermalised (not a no-op): case A starts with all
        #      radiation in the lowest group, ending with it spread to higher groups.
        assert eg_a[0, 0] > 0.5, "case A should start with radiation in the lowest group"
        assert eg_a[-1, 0] < 0.1 * eg_a[0, 0], "case A lowest group should drain"
        assert eg_a[-1, 1] > 10.0 * eg_a[0, 1], "case A higher groups should fill"

        # subsample the trajectories for a clean baseline + plot.
        s = slice(None, None, 10)
        relax_fields = {}
        for g in range(ngroups):
            relax_fields[f"EgA_{g}"] = eg_a[s, g].tolist()
            relax_fields[f"EgB_{g}"] = eg_b[s, g].tolist()
        relax_fields["eele_A"] = ee_a[s].tolist()
        relax_fields["eele_B"] = ee_b[s].tolist()
        harness.verify(
            "multigroup_rad_equil",
            t[s].tolist(),
            relax_fields,
            coord_label="t",
            xlabel="time",
            title="Multigroup non-equilibrium spectral relaxation to the Planck spectrum",
            rtol=1.0e-5,
            atol=1.0e-8,
        )

        # ===== Part 2: coupled multigroup FLD Marshak front =====
        front = np.loadtxt(front_path)
        x = front[:, 0]
        eg_x = front[:, 1:1 + ngroups]
        te_x = front[:, 1 + ngroups + 1]
        etot_x = front[:, 1 + ngroups + 2]
        mask = (x > WIN_LO) & (x < WIN_HI)
        assert mask.sum() > 10, "PART 2 verification window too small"

        # (i) per-group D distinct/ordered (group-resolved diffusion: thicker chi -> less
        #     D -> shallower penetration), read from the Rosseland extinctions.
        assert all(
            d2[i + 1] > d2[i] * 1.2 for i in range(ngroups - 1)
        ), f"per-group D not distinct/ordered: D={d2}, chi_R={chi_r2}"

        # (ii) LOCAL PLANCK-SPECTRUM LOCK: each group equals its Planck slice at the local
        #      matter temperature (the spectral reference for the coupled front).
        lock = _planck_spectrum(te_x, bounds)
        for g in range(ngroups):
            rel = np.abs(eg_x[mask, g] - lock[mask, g]) / np.maximum(
                A_RAD * te_x[mask] ** 4, 1.0e-30
            )
            assert np.max(rel) < 5.0e-3, (
                f"PART 2 group {g} off local Planck spectrum: rel {np.max(rel):.3e}"
            )

        # (iii) monotone (non-increasing) radiation front in the interior window.
        assert np.all(np.diff(etot_x[mask]) <= 1.0e-9), (
            "PART 2 total radiation not monotone in the front window"
        )

        # (iv) matter is heated behind the front (radiation-hydro coupling at work).
        assert te_x[0] > 0.8, f"matter not heated behind the front: Te={te_x[0]:.3e}"
        assert te_x[0] > 4.0 * te_x[-1], "no clear front (wall not much hotter than tail)"

        front_fields = {"Te": te_x[mask].tolist(), "E_tot": etot_x[mask].tolist()}
        for g in range(ngroups):
            front_fields[f"E_g{g}"] = eg_x[mask, g].tolist()
            front_fields[f"E_g{g}_planck"] = lock[mask, g].tolist()
        harness.verify(
            "multigroup_rad_front",
            x[mask].tolist(),
            front_fields,
            coord_label="x1v",
            xlabel="x",
            title="Coupled multigroup FLD Marshak front: E_g vs local Planck spectrum",
            rtol=1.0e-5,
            atol=1.0e-7,
        )
    finally:
        for p in (hist_path, grp_path, front_path):
            try:
                os.remove(p)
            except OSError:
                pass
        testutils.cleanup()
