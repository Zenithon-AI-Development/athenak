"""
Cylindrical field-aligned anisotropic-conduction ring-test verification
(ADR-0006 / ADR-0004, issue [6b]/#25).

Verifies the anisotropic (magnetized) Braginskii conduction operator
(AnisotropicConductionOperator) UNDER THE CYLINDRICAL (r,phi) METRIC -- the cylindrical
analog of the Parrish-Stone ring test.  A 2D r-phi annulus carries a purely azimuthal
frozen field B = b0 e_phi (field lines = concentric circles of constant radius), with a
smooth Gaussian temperature blob on the mid-annulus ring at (r0, phi=0).  With the
strongly magnetized Braginskii coefficients (kappa_par >> kappa_perp) heat must conduct
AZIMUTHALLY around the ring (along the field, at constant r) while the cross-field RADIAL
leakage stays a tiny fraction -- heat follows field lines under the cylindrical metric.

The operator is advanced operator-split by RKL2 super-time-stepping entirely inside the
``cyl_aniso_ring`` user pgen (it is not yet wired into the driver tasklist), which writes
two ASCII profiles read here:

  * cyl_aniso_ring_phi.dat -- T vs phi along the ring radius r=r0 (azimuthal spread),
  * cyl_aniso_ring_r.dat   -- T vs r at phi=0 (radial confinement / cross-field leakage).

The test asserts the temperature broadens strongly AZIMUTHALLY but barely RADIALLY (field
alignment), the azimuthal far-field is heated while cross-field radial leakage stays a
tiny fraction (leakage within tolerance), no new extrema form (Sharma-Hammett
monotonicity), and baselines both profiles through the shared verification harness.

Oracle: Layer 2 -- self-snapshot.  This cylindrical-metric ring profile has no
closed-form solution, so the harness.verify baselines (the azimuthal-spread and
radial-confinement profiles) are a regression guard only, over an
anisotropic-conduction method verified independently elsewhere.  The
method-correctness anchors are test_unit_aniso_conduction (analytic field-aligned
heat-flux projection: a mode along B reduces to kappa_par diffusion, across B to
kappa_perp, to machine precision) and test_unit_parabolic_conduction (analytic cosine
diffusion eigenmode, exp(lambda t) decay).  The qualitative asserts reproduce the
published ring-test behavior -- heat conducts along the field lines while cross-field
leakage stays a tiny fraction (Parrish & Stone 2005, the canonical
anisotropic-conduction ring test), with no new extrema created on the grid (Sharma &
Hammett 2007, the monotonised anisotropic-diffusion slope limiter).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# Run parameters -- MUST match tst/inputs/cyl_aniso_ring.athinput.
PROBLEM = "cyl_aniso_ring"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "cyl_aniso_ring.athinput"
)
T_COLD = 1.0
DT_PEAK = 0.1
R0 = 1.0
SIGMA_R = 0.05
SIGMA_PHI = 0.19634954        # pi/16


def _weighted_sigma(coord, temp):
    """Intensity-weighted standard deviation of the temperature excess T - T_cold; a
    coordinate-agnostic measure of how far the heat has spread along an axis."""
    w = np.clip(temp - T_COLD, 0.0, None)
    s = w.sum()
    mu = (coord * w).sum() / s
    var = ((coord - mu) ** 2 * w).sum() / s
    return float(np.sqrt(var))


def test_verify_cyl_aniso_ring():
    """Run the cylindrical ring test; verify field-aligned conduction + baseline."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    phi_path = os.path.join(run_dir, "cyl_aniso_ring_phi.dat")
    r_path = os.path.join(run_dir, "cyl_aniso_ring_r.dat")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "cyl_aniso_ring run failed"

        prof_phi = np.loadtxt(phi_path)
        prof_r = np.loadtxt(r_path)
        phi, t_phi_final, t_phi_init = prof_phi[:, 0], prof_phi[:, 1], prof_phi[:, 2]
        r, t_r_final, t_r_init = prof_r[:, 0], prof_r[:, 1], prof_r[:, 2]

        # (1) Field alignment: the blob broadens strongly ALONG the field (azimuthal) but
        # is barely broadened ACROSS it (radial) -- heat stays on the field-line ring.
        sig_phi0 = _weighted_sigma(phi, t_phi_init)
        sig_phi1 = _weighted_sigma(phi, t_phi_final)
        sig_r0 = _weighted_sigma(r, t_r_init)
        sig_r1 = _weighted_sigma(r, t_r_final)
        azim_broadening = sig_phi1 / sig_phi0
        radial_broadening = sig_r1 / sig_r0
        assert azim_broadening > 2.0, (
            f"heat did not spread azimuthally along the field: "
            f"sigma_phi {sig_phi0:.4f} -> {sig_phi1:.4f} ({azim_broadening:.2f}x)"
        )
        assert radial_broadening < 1.1, (
            f"too much cross-field radial spreading under the cylindrical metric: "
            f"sigma_r {sig_r0:.4f} -> {sig_r1:.4f} ({radial_broadening:.3f}x)"
        )
        # field-line confinement: the azimuthal spreading dwarfs the radial spreading
        # (an isotropic operator would broaden comparably in r and phi -> this fails).
        assert (azim_broadening - 1.0) > 10.0 * (radial_broadening - 1.0), (
            "azimuthal spread does not dominate radial spread (field alignment weak): "
            f"azim {azim_broadening:.2f}x vs radial {radial_broadening:.3f}x"
        )

        # (2) Heat reached the azimuthal far field along the field lines, while the
        # cross-field radial far field stays a tiny fraction (leakage within tolerance).
        dT_azim_far = float(
            (t_phi_final[np.abs(phi) >= 3.0 * SIGMA_PHI] - T_COLD).max()
        )
        dT_radial_far = float(
            (t_r_final[np.abs(r - R0) >= 3.0 * SIGMA_R] - T_COLD).max()
        )
        assert dT_azim_far > 0.02 * DT_PEAK, (
            f"no heat reached the azimuthal far side: dT_azim_far={dT_azim_far:.3e}"
        )
        assert dT_radial_far < 0.05 * dT_azim_far, (
            f"cross-field radial leakage too large: dT_radial_far={dT_radial_far:.3e} "
            f"vs dT_azim_far={dT_azim_far:.3e}"
        )

        # (3) Sharma-Hammett monotonicity: no new extrema (no cross-field over/undershoot)
        # are created on the grid.
        tmin = float(min(t_phi_final.min(), t_r_final.min()))
        tmax = float(max(t_phi_final.max(), t_r_final.max()))
        assert tmin > T_COLD - 0.02 * DT_PEAK, f"undershoot below T_cold: tmin={tmin:.6f}"
        assert tmax < T_COLD + 1.02 * DT_PEAK, f"overshoot: tmax={tmax:.6f}"

        # (4) Regression baselines (azimuthal spread + radial confinement profiles).
        harness.verify(
            "cyl_aniso_ring_phi",
            phi.tolist(),
            {"T_final": t_phi_final.tolist(), "T_init": t_phi_init.tolist()},
            coord_label="phi",
            xlabel="phi (along field lines, ring radius r=r0)",
            title="Cylindrical aniso conduction: azimuthal (field-aligned) heat spread",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
        harness.verify(
            "cyl_aniso_ring_r",
            r.tolist(),
            {"T_final": t_r_final.tolist(), "T_init": t_r_init.tolist()},
            coord_label="x1v",
            xlabel="r (across field lines, phi=0)",
            title="Cylindrical aniso conduction: radial (cross-field) confinement",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        for p in (phi_path, r_path):
            try:
                os.remove(p)
            except OSError:
                pass
        testutils.cleanup()
