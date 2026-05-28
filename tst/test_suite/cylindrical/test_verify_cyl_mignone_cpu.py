"""
Mignone cylindrical radial-advection convergence verification (ADR-0004, issue #15).

A smooth density profile is advected radially at constant velocity (evolution=kinematic),
for which the cylindrical continuity equation has the exact solution

    rho(r,t) = (r0/r) * rho_init(r0),   r0 = r - v_r * t        (geometric dilution).

The test runs several resolutions, measures the L1 error against this analytic solution
over the well-resolved interior, and checks that the convergence rate matches the
reference for the 2nd-order PLM reconstruction (error ~ N^-2, i.e. it falls by ~4x per
doubling).  Plot + golden baseline of the error-vs-resolution curve via the harness.

Oracle: Layer 1 -- analytic.  Closed-form cylindrical radial-advection (geometric
dilution) solution rho(r,t) = (r0/r) rho_init(r0), r0 = r - v_r t.  It follows from the
cylindrical continuity equation d_t rho + (1/r) d_r(r rho v_r) = 0 with constant v_r:
along the characteristics dr/dt = v_r the quantity r*rho is conserved (mass in an
annulus), giving the (r0/r) dilution factor.  The binding assertion is that the L1 error
vs this exact solution exhibits 2nd-order (PLM) convergence, error ~ N^-2.  The solution
was cross-checked by confirming it satisfies the continuity PDE to finite-difference
truncation and the exact r*rho characteristic invariant (issue #76).  Cf. Mignone (2014)
/ PLUTO cylindrical-advection test problem.  The harness.verify baseline is a regression
guard only.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

input_file = "inputs/cyl_mignone.athinput"

# Run parameters (must match inputs/cyl_mignone.athinput).
X1MIN, X1MAX = 1.0, 2.0
LBOX = X1MAX - X1MIN
VEL = 1.0
AMP = 0.2
TFINAL = 0.2
RESOLUTIONS = [64, 128, 256]
# Compare only where the solution descends from interior initial data (r0 >= x1min) and
# is clear of the outer boundary -- isolates the scheme's interior accuracy.
WIN_LO, WIN_HI = 1.3, 1.9


def _rho_init(r):
    """Initial smooth profile laid down by the advection pgen (iproblem=1, sine)."""
    return 1.0 + AMP * np.sin(2.0 * np.pi * (r - X1MIN) / LBOX)


def _rho_exact(r, t):
    """Exact cylindrical radial-advection solution rho(r,t) = (r0/r) rho_init(r0)."""
    r0 = r - VEL * t
    return (r0 / r) * _rho_init(r0)


def _run_one(res):
    """Run one resolution; return interior L1 error of density vs the exact solution."""
    args = [
        f"mesh/nx1={res}",
        f"meshblock/nx1={res}",
    ]
    assert testutils.run(input_file, args), f"cyl Mignone run failed at N={res}"
    data = athena_read.tab("tab/cyl_mignone.hydro_w.00001.tab")
    r = np.asarray(data["x1v"])
    dens = np.asarray(data["dens"])
    mask = (r >= WIN_LO) & (r <= WIN_HI)
    err = np.abs(dens[mask] - _rho_exact(r[mask], TFINAL))
    return float(np.mean(err))


def test_verify_cyl_mignone():
    """Run several resolutions and verify 2nd-order convergence + save a baseline."""
    try:
        errors = [_run_one(res) for res in RESOLUTIONS]

        # Convergence order between successive (doubled) resolutions.
        orders = [
            np.log(errors[i] / errors[i + 1]) / np.log(2.0)
            for i in range(len(errors) - 1)
        ]
        # PLM is 2nd order on a smooth profile; require the finest pair to be clearly
        # 2nd-order and every pair to be at least solidly above 1st order.
        assert orders[-1] > 1.7, f"cyl Mignone not 2nd-order: orders={orders}"
        assert min(orders) > 1.5, f"cyl Mignone convergence too slow: orders={orders}"
        # Errors must actually decrease with resolution.
        assert all(
            errors[i + 1] < errors[i] for i in range(len(errors) - 1)
        ), f"cyl Mignone error not monotonically decreasing: {errors}"

        harness.verify(
            "cyl_mignone",
            [float(r) for r in RESOLUTIONS],
            {"L1_error": errors},
            coord_label="resolution",
            xlabel="radial resolution N",
            title="Mignone cylindrical radial advection: L1 error vs resolution",
            rtol=1.0e-4,
            atol=1.0e-12,
        )
    finally:
        testutils.cleanup()
