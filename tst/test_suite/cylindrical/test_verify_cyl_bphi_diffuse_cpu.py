"""
Cylindrical resistive B_phi diffusion verification (ADR-0004, issue [7b]/#27).

Verifies the special CYLINDRICAL resistive diffusion of B_phi (ResistiveBphiOperator) --
"the one true cylindrical exception" of ADR-0004.  Unlike the cylindrical scalar
Laplacian, the phi-component of the resistive curl-curl operator carries a genuine
-eta*B_phi/r^2 term:

    dB_phi/dt = eta [ d^2B_phi/dr^2 + (1/r) dB_phi/dr - B_phi/r^2 ] ,

implemented in the conservative (1/r)d_r(r*flux) finite-volume form with an antisymmetric
axis ghost B_phi(i-1) = -B_phi(i).  The exact decaying solution is the Bessel J_1
eigenmode

    B_phi(r,t) = A J_1(k r) exp(-eta k^2 t)   (J_1'' + (1/r)J_1' + (1-1/r^2)J_1 = 0),

so on r in [0, R] with k = j_{1,1}/R the mode is zero on the axis (J_1(0)=0) AND at the
outer radius (J_1(j_{1,1})=0).  The -B_phi/r^2 curl-curl term is ESSENTIAL: naive scalar
diffusion (dropping it) makes J_1 no longer an eigenmode, distorting the near-axis shape
and changing the decay rate -- so this test discriminates the correct cylindrical operator
NEAR and AWAY from the axis.

The operator is advanced operator-split by RKL2 super-time-stepping entirely inside the
``cyl_bphi_diffuse`` user pgen (not yet wired into the driver tasklist), which writes one
ASCII profile read here:

  * cyl_bphi_diffuse.dat -- (r, B_phi_numerical, B_phi_initial, B_phi_analytic).

The test asserts the numerical profile matches the analytic J_1-decay profile near and
away from the axis, the eigenmode shape is preserved (single-scalar decay), the measured
decay matches the analytic rate, and baselines the profile through the shared
verification harness.

Oracle: Layer 1 -- analytic.  B_phi(r,t) = A J_1(k r) exp(-eta k^2 t) is the exact
decaying eigenmode of the cylindrical resistive curl-curl operator: J_1 solves Bessel's
equation of order 1, so [d_rr + (1/r) d_r - 1/r^2] J_1(k r) = -k^2 J_1(k r), giving the
decay rate eta k^2.  With k = j_{1,1}/R (j_{1,1} = 3.83171, the first zero of J_1) the
mode vanishes on the axis (J_1(0)=0) and at the outer radius (J_1(j_{1,1})=0).  Reference:
Bessel-function theory (Abramowitz & Stegun 1972, ch. 9); resistive MHD diffusion.  The
harness.verify baseline is a regression guard only.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

# Run parameters -- MUST match tst/inputs/cyl_bphi_diffuse.athinput.
PROBLEM = "cyl_bphi_diffuse"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "cyl_bphi_diffuse.athinput"
)


def _decay_amplitude(b_target, b_ref, weight):
    """Least-squares scalar A minimizing ||b_target - A b_ref|| under the cylindrical
    inner-product weight r (Bessel J_1 modes are r-orthogonal); = the decay factor if the
    eigenmode shape is preserved."""
    return float((b_target * b_ref * weight).sum() / (b_ref * b_ref * weight).sum())


def test_verify_cyl_bphi_diffuse():
    """Run the cylindrical B_phi-diffusion test; verify J_1 eigenmode decay + baseline."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    dat_path = os.path.join(run_dir, "cyl_bphi_diffuse.dat")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "cyl_bphi_diffuse run failed"

        prof = np.loadtxt(dat_path)
        r = prof[:, 0]
        b_num = prof[:, 1]
        b_init = prof[:, 2]
        b_ana = prof[:, 3]
        w = r  # cylindrical weight r*dr (dr constant -> drop it from the ratios)
        bscale = float(np.abs(b_init).max())

        # (1) Eigenmode shape preservation: the numerical field is a single scalar
        # multiple of the initial J_1 mode (no shape distortion -> resistive operator
        # reproduces it).  Naive scalar diffusion (no -eta B/r^2) distorts the shape here.
        a_num = _decay_amplitude(b_num, b_init, w)
        resid = float(
            np.sqrt(
                ((b_num - a_num * b_init) ** 2 * w).sum()
                / ((a_num * b_init) ** 2 * w).sum()
            )
        )
        assert resid < 1.0e-2, (
            f"J_1 eigenmode shape not preserved (shape residual {resid:.3e}); "
            f"the cylindrical -eta B_phi/r^2 term is likely wrong/missing"
        )

        # (2) Decay rate: the measured amplitude matches the analytic exp(-eta k^2 t)
        # decay (recovered from the analytic profile the pgen wrote).
        a_ana = _decay_amplitude(b_ana, b_init, w)
        assert a_ana < 0.999, f"analytic decay not applied (a_ana={a_ana:.4f})"
        rate_err = abs(a_num - a_ana) / a_ana
        assert rate_err < 2.0e-2, (
            f"decay rate mismatch: measured A={a_num:.5f} vs analytic A={a_ana:.5f} "
            f"(rel err {rate_err:.3e})"
        )

        # (3) The numerical profile matches the analytic J_1-decay profile NEAR the axis
        # (r < 0.2 R, where the -eta B_phi/r^2 term dominates) AND AWAY (r > 0.5 R).
        err = np.abs(b_num - b_ana)
        rmax = r.max()
        near = r < 0.2 * rmax
        far = r > 0.5 * rmax
        near_err = float(err[near].max()) / bscale
        far_err = float(err[far].max()) / bscale
        assert near_err < 2.0e-2, (
            f"near-axis profile does not match the analytic J_1 decay: "
            f"max|dB|/Bscale={near_err:.3e} for r<{0.2 * rmax:.3f}"
        )
        assert far_err < 2.0e-2, (
            f"far-field profile does not match the analytic J_1 decay: "
            f"max|dB|/Bscale={far_err:.3e} for r>{0.5 * rmax:.3f}"
        )

        # (4) Regression baseline (numerical + analytic radial profiles).
        harness.verify(
            "cyl_bphi_diffuse",
            r.tolist(),
            {"B_phi": b_num.tolist(), "B_phi_analytic": b_ana.tolist()},
            coord_label="x1v",
            xlabel="r",
            title="Cylindrical resistive B_phi diffusion: J_1 eigenmode decay",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        try:
            os.remove(dat_path)
        except OSError:
            pass
        testutils.cleanup()
