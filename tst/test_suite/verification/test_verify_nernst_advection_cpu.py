"""
Nernst B_z-advection verification -- RED-first for issue #238 (ADR-0017, Proposed).

Drives full MHD on a static, uniform-pressure 1-D box carrying a linear electron-
temperature ramp T(x) = (p0/d0)*(1 - eps*(x - xt0)) (imposed through rho = p0/T so the
v=0 state is an exact hydro equilibrium) and a passive Gaussian axial-field bump
B_z(x) = bz0 exp(-((x-xc)/w)^2), via the ``nernst_advection`` user pgen
(testutils.run_pgen).  Under the upwinded Nernst operator (#238) the bump advects down
the temperature gradient at the Braginskii Nernst velocity

    v_N = -(beta_wedge(x_e)/x_e) * (tau_e/m_e) * grad(k_B T_e),

with the Z=1 wedge-thermoelectric fit beta_wedge(x) = x(1.5 x^2 + 3.053)/Delta(x),
Delta(x) = x^4 + 14.79 x^2 + 3.7703 (magnetization-reduced: beta_wedge/x -> 0.810
unmagnetized, -> 1.5/x^2 strongly magnetized).  With grad(T_e) uniform the bump
translates rigidly, so its centroid displacement over tlim must equal v_N*tlim.  The
test measures that displacement from the t=0 and t=tlim tab dumps and checks it against
the analytic v_N computed independently here from the same SI transport chain the code's
diffusion/braginskii_transport.hpp implements (conversions from the deck's nernst_*_si
parameters, the resist_*_si pattern).

RED-FIRST (#238): the Nernst operator does not exist in AthenaK (ADR-0003/0006 scoped it
out; ADR-0017 -- Proposed, human decision pending -- reverses that for B5/B6).  Nothing
advects B_z, the measured displacement is ~0, and the final assertion fails for exactly
the right reason.  Marked xfail(strict=True) so CI stays green now and FAILS on an
unexpected pass, forcing removal of the marker when the operator lands.  Green must also
add the harness.verify baseline of the advected profile (not committed red, to avoid
baselining the wrong -- unadvected -- state).

Oracle: Layer 1 -- analytic.  Rigid translation B_z(x,t) = B_z(x - v_N t, 0) of a
passive field bump under a uniform Nernst velocity is the exact solution of
dB/dt = curl(v_N x B) for uniform v_N; the velocity is the Braginskii Z=1
wedge-thermoelectric form above.  References: Braginskii 1965 (Reviews of Plasma Physics
1, 205); Haines 1986 (Plasma Phys. Control. Fusion 28, 1705); Nishiguchi et al. 1984
(PRL 53, 262); Ellison et al. 2025 (arXiv:2504.10760, section 2.1.2).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math
import os

import numpy as np
import pytest
import test_suite.testutils as testutils

# Column indices in the non-sliced 1-D tab dump (gid, i, x1v, dens, velx, vely, velz,
# eint, bcc1, bcc2, bcc3); athena_read.tab assumes a sliced layout, so read by position.
COL_X1V = 2
COL_BCC3 = 10

PROBLEM = "nernst_advection"
input_file = os.path.join(testutils._repo_root(), "tst", "inputs",
                          "nernst_advection.athinput")

# Run parameters -- MUST match tst/inputs/nernst_advection.athinput.
D0, P0 = 1.0, 1.0
EPS = 0.2                      # fractional T drop per unit x
XT0 = 0.5                      # ramp anchor: T(XT0) = P0/D0
BZ0 = 1.0e-3
XC = 0.4                       # initial bump center
W = 0.05
TLIM = 1.0
NX1 = 256
# Nernst model inputs (code state + linear code<->SI conversions, resist_*_si pattern).
DENS_SI = 0.167                # code density -> kg m^-3
TEMP_SI = 1.160451e6           # code temperature -> K (T=1 <-> 100 eV)
BMAG_SI = 1.0                  # code |B| -> Tesla
LEN_SI = 1.0e-3                # code length -> m
VEL_SI = 1.0e4                 # code velocity -> m/s
ZBAR = 1.0
LNLAMBDA = 5.0

# SI physical constants (match diffusion/braginskii_transport.hpp).
KB = 1.380649e-23
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
# Braginskii Z=1 wedge-thermoelectric fit constants (beta_wedge = x(B1 x^2 + B0)/Delta,
# Delta = x^4 + D1 x^2 + D0; the same electron Delta as braginskii_transport.hpp).
BETA1 = 1.5
BETA0 = 3.053
DELTA1 = 14.79
DELTA0 = 3.7703


def _nernst_velocity_code():
    """Analytic Braginskii Nernst velocity (code units) at the initial bump center,
    replaying the exact SI transport chain of diffusion/braginskii_transport.hpp."""
    t_code = (P0 / D0) * (1.0 - EPS * (XC - XT0))
    rho_code = P0 / t_code
    n_e = ZBAR * rho_code * DENS_SI / M_P
    t_e = t_code * TEMP_SI
    kt = KB * t_e
    fourpieps0_sq = (4.0 * math.pi * EPS0) ** 2
    tau_e = (3.0 * math.sqrt(M_E) * kt ** 1.5 * fourpieps0_sq /
             (4.0 * math.sqrt(2.0 * math.pi) * n_e * ZBAR * QE ** 4 * LNLAMBDA))
    # magnetization at the bump peak (deeply unmagnetized here, coeff -> BETA0/DELTA0)
    x_e = (QE * BZ0 * BMAG_SI / M_E) * tau_e
    x2 = x_e * x_e
    coeff = (BETA1 * x2 + BETA0) / (x2 * x2 + DELTA1 * x2 + DELTA0)
    # uniform grad(T_e): dT_code/dx_code = -EPS*(P0/D0) -> SI via temp/len conversions
    dtdx_si = -EPS * (P0 / D0) * TEMP_SI / LEN_SI
    v_n_si = -coeff * (tau_e / M_E) * KB * dtdx_si   # down-gradient: +x here
    return v_n_si / VEL_SI


def _centroid(x, bz):
    """B_z-weighted centroid of the (non-negative) bump."""
    tot = float(np.sum(bz))
    assert tot > 0.0, f"unexpected non-positive total B_z flux {tot:g}"
    return float(np.sum(x * bz) / tot)


@pytest.mark.xfail(
    strict=True,
    reason="#238 red-first: Nernst advection of B by the electron heat flux is not "
           "implemented (ADR-0017 Proposed, human decision pending); B_z does not move",
)
def test_verify_nernst_advection():
    """Run the T_e-ramp/B_z-bump deck; verify the bump advects at the Nernst velocity."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    f0 = os.path.join(run_dir, "tab", "nernst_advection.bcc.00000.tab")
    f1 = os.path.join(run_dir, "tab", "nernst_advection.bcc.00001.tab")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "nernst_advection run failed"

        arr0 = np.loadtxt(f0, comments="#")
        arr1 = np.loadtxt(f1, comments="#")
        x = arr0[:, COL_X1V]
        bz_init = arr0[:, COL_BCC3]
        bz_final = arr1[:, COL_BCC3]

        # (1) sanity: the seeded bump is where the deck put it.
        dx = 1.0 / NX1
        c0 = _centroid(x, bz_init)
        assert abs(c0 - XC) < 2.0 * dx, f"initial bump centroid {c0:g} != xc {XC:g}"

        # (2) sanity: conservative advection preserves the total B_z flux (the bump
        # stays interior over tlim; upwind diffusion moves flux, never creates it).
        flux0 = float(np.sum(bz_init)) * dx
        flux1 = float(np.sum(bz_final)) * dx
        assert abs(flux1 / flux0 - 1.0) < 0.10, (
            f"total B_z flux not conserved: {flux0:g} -> {flux1:g}"
        )

        # (3) the oracle: centroid displacement = v_N * tlim.  Donor-cell upwinding
        # diffuses the bump but transports its centroid at the advection speed; the 25%
        # band covers the O(eps) variation of v_N along the path.
        disp_meas = _centroid(x, bz_final) - c0
        disp_oracle = _nernst_velocity_code() * TLIM
        rel = abs(disp_meas - disp_oracle) / abs(disp_oracle)
        assert rel < 0.25, (
            f"B_z bump centroid displacement {disp_meas:.4e} does not match the "
            f"analytic Nernst-advection displacement v_N*tlim = {disp_oracle:.4e} "
            f"(rel err {rel:.2f}): Nernst advection of B by the electron heat flux "
            f"is absent (#238, ADR-0017)"
        )
    finally:
        testutils.cleanup()
