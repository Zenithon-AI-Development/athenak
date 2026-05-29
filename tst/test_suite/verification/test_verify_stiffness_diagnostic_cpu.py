"""
Per-operator stiffness diagnostic (issue [A2]/#109, ADR-0001 routing addendum).

This is a *component/logic* test, not a physics-sim test.  It exercises the stiffness
diagnostic that reports each parabolic operator's forward-Euler stable timestep
``dt_exp`` relative to the hyperbolic timestep ``dt_hyp`` and turns that ratio into a
per-operator routing decision: STS (operator-split RKL2 super-time-stepping, which pays
~sqrt(ratio) substages and one ghost exchange per substage) versus the existing
flux-fused diffusion path (the diffusive flux rides the hyperbolic update, global
dt = min(dt_hyp, dt_exp), and no per-substage exchange is added).

The diagnostic mirrors the C++ ``ExplicitStableDt`` closed forms of every shipped
operator (``src/diffusion/conduction_operator.cpp``, ``aniso_conduction_operator.cpp``,
``resistivity.cpp``, ``resistive_bphi_operator.cpp``, ``src/radiation_fld/
fld_grey_operator.cpp`` / ``fld_multigroup_operator.cpp``) and the hyperbolic limit of
``src/hydro/hydro_newdt.cpp`` plus the driver's CFL scaling, and the adaptive RKL2 stage
count of ``parabolic::RKL2NumStages`` (``src/driver/parabolic_integrator.hpp``).

Two layers of assertion:
  * a CONTROLLED, dimensionless config whose dt_exp / dt_hyp / ratio / stage-count are
    hand-computable in closed form -- this is the "known ratios" gate (red-first);
  * a representative-DIFFICULT MagLIF stagnation config built from standard plasma
    closed forms (NRL Plasma Formulary; Spitzer 1962 / Braginskii 1965) with the assumed
    state documented -- the routing decision (STS for conduction, resistive B, and
    optically-thick radiation) is robust to order-of-magnitude uncertainty in the
    transport constants, which is the property the test pins.

Oracle: Layer 1 -- analytic.  The ground truth is the closed-form forward-Euler
diffusion stability limit dt = 1/(2 D sum_i 1/dx_i^2) (and its conduction/resistive
specializations), the hyperbolic CFL limit dt = C dx/(|v|+c_s), and the shifted-Chebyshev
RKL2 stage count s = ceil((sqrt(9+16 r)-1)/2) (Meyer, Balsara & Aslam 2014); the
diagnostic reproduces those numbers exactly on a controlled config.  Per ADR-0008 this
component test verifies the diagnostic's arithmetic and routing logic, not a simulation.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math

import pytest

import test_suite.verification.stiffness_diagnostic as sd

GAMMA = 5.0 / 3.0


# ----------------------------------------------------------------------------------------
# Controlled config: every number below is hand-computable in closed form.
# 1-D, dx = 0.1, forward-Euler prefactor fac = 1/2 (1-D).
# ----------------------------------------------------------------------------------------
DX = 0.1
CFL = 0.4
CS = 1.0


def test_conduction_stable_dt_matches_closed_form():
    """dt = fac dx^2 rho/(kappa (gamma-1)); fac=1/2 in 1-D (conduction_operator.cpp)."""
    # 0.5 * 0.1^2 * 2.0 / (0.5 * 2/3) = 0.5*0.01*2.0/(1/3) = 0.03
    dt = sd.conduction_stable_dt(DX, rho=2.0, kappa=0.5, gamma=GAMMA, ndim=1)
    assert dt == pytest.approx(0.03, rel=1e-12)


def test_conduction_fac_drops_with_dimension():
    """fac = 1/2, 1/4, 1/6 in 1/2/3-D -- the C++ ExplicitStableDt prefactor."""
    base = sd.conduction_stable_dt(DX, rho=1.0, kappa=1.0, gamma=GAMMA, ndim=1)
    two = sd.conduction_stable_dt(DX, rho=1.0, kappa=1.0, gamma=GAMMA, ndim=2)
    three = sd.conduction_stable_dt(DX, rho=1.0, kappa=1.0, gamma=GAMMA, ndim=3)
    assert two == pytest.approx(base / 2.0, rel=1e-12)   # 1/4 vs 1/2
    assert three == pytest.approx(base / 3.0, rel=1e-12)  # 1/6 vs 1/2


def test_resistivity_stable_dt_matches_closed_form():
    """dt = fac dx^2/eta (Resistivity::NewTimeStep)."""
    # 0.5 * 0.01 / 0.5 = 0.01
    dt = sd.resistivity_stable_dt(DX, eta=0.5, ndim=1)
    assert dt == pytest.approx(0.01, rel=1e-12)


def test_resistive_bphi_near_axis_term_tightens_dt():
    """The cylindrical 1/r^2 curl-curl term shrinks dt near the axis (bphi operator)."""
    far = sd.resistive_bphi_stable_dt(DX, eta=0.5, ndim=1, radius=10.0)
    near = sd.resistive_bphi_stable_dt(DX, eta=0.5, ndim=1, radius=0.05)
    no_r = sd.resistive_bphi_stable_dt(DX, eta=0.5, ndim=1, radius=None)
    # rate = 2 eta/dx^2 (+ eta/r^2); dt = 1/rate.  Near-axis adds the largest 1/r^2.
    assert near < far < no_r
    # closed form with no curvature term: rate = 2*0.5/0.01 = 100 -> dt = 0.01
    assert no_r == pytest.approx(0.01, rel=1e-12)


def test_larsen_limiter_thick_and_thin_limits():
    """lambda(R) = (3^n + R^n)^(-1/n): -> 1/3 (thick, R->0); -> 1/R (thin, R>>1)."""
    assert sd.larsen_limiter(0.0, n=2.0) == pytest.approx(1.0 / 3.0, rel=1e-12)
    big = sd.larsen_limiter(1.0e6, n=2.0)
    assert big == pytest.approx(1.0e-6, rel=1e-3)  # ~1/R free-streaming


def test_fld_stable_dt_matches_closed_form():
    """dt = 1/(2 D sum_i 1/dx_i^2); thick D = c lambda(0)/chi = c/(3 chi)."""
    dt = sd.fld_stable_dt(DX, D=0.05, ndim=1)
    # 1/(2*0.05*(1/0.01)) = 1/(0.1*100) = 0.1
    assert dt == pytest.approx(0.1, rel=1e-12)
    # diffusivity helper: thick limit
    d_thick = sd.fld_diffusivity(c=3.0, chi=1.0, R=0.0)
    assert d_thick == pytest.approx(1.0, rel=1e-12)  # 3 * (1/3) / 1


def test_hyperbolic_dt_matches_closed_form():
    """dt_hyp = cfl * min_i dx_i/(|v_i| + c_s) (hydro_newdt + driver CFL)."""
    dt = sd.hyperbolic_dt(DX, vel=0.0, cs=CS, cfl=CFL, ndim=1)
    assert dt == pytest.approx(0.04, rel=1e-12)  # 0.4 * 0.1 / 1.0


def test_rkl2_num_stages_matches_chebyshev_recursion():
    """s = max(2, ceil((sqrt(9+16 r)-1)/2)) -- the integrator's RKL2NumStages."""
    assert sd.rkl2_num_stages(dt_super=1.0, dt_exp=1.0) == 2     # r=1 -> floor 2
    assert sd.rkl2_num_stages(dt_super=4.0, dt_exp=1.0) == 4     # r=4 -> 4 (sqrt73)
    assert sd.rkl2_num_stages(dt_super=0.0, dt_exp=1.0) == 2     # r=0 -> floor 2
    # general: a 100x stiffness ratio collapses to ~10 substages (sqrt scaling)
    s = sd.rkl2_num_stages(dt_super=100.0, dt_exp=1.0)
    assert s == math.ceil(0.5 * (math.sqrt(9.0 + 1600.0) - 1.0))
    assert 10 <= s <= 21  # sqrt(100)..sqrt(ratio)+ -- far below 100 explicit substeps


# ----------------------------------------------------------------------------------------
# Routing classifier on the controlled config.
# ----------------------------------------------------------------------------------------
def test_route_flux_fused_when_not_stiff():
    """ratio = dt_hyp/dt_exp <= FLUX_FUSE_RATIO -> ride the flux-fused path (no STS)."""
    dt_exp = sd.conduction_stable_dt(DX, rho=2.0, kappa=0.5, gamma=GAMMA, ndim=1)  # 0.03
    dt_hyp = sd.hyperbolic_dt(DX, vel=0.0, cs=CS, cfl=CFL, ndim=1)                 # 0.04
    assert sd.stiffness_ratio(dt_hyp, dt_exp) == pytest.approx(4.0 / 3.0, rel=1e-12)
    assert sd.route(dt_hyp, dt_exp) == "flux_fused"


def test_route_sts_when_stiff():
    """ratio > FLUX_FUSE_RATIO -> super-time-step (amortize stiffness)."""
    dt_exp = sd.resistivity_stable_dt(DX, eta=0.5, ndim=1)         # 0.01
    dt_hyp = sd.hyperbolic_dt(DX, vel=0.0, cs=CS, cfl=CFL, ndim=1)  # 0.04
    assert sd.stiffness_ratio(dt_hyp, dt_exp) == pytest.approx(4.0, rel=1e-12)
    assert sd.route(dt_hyp, dt_exp) == "sts"


def test_route_threshold_is_configurable():
    """The pragmatic flux-fuse cutoff is a documented, tunable parameter."""
    # ratio = 4: flux-fuse only if we raise the cutoff above 4.
    dt_exp, dt_hyp = 0.01, 0.04
    assert sd.route(dt_hyp, dt_exp, flux_fuse_ratio=2.0) == "sts"
    assert sd.route(dt_hyp, dt_exp, flux_fuse_ratio=5.0) == "flux_fused"


# ----------------------------------------------------------------------------------------
# Representative-difficult MagLIF stagnation config -> the documented routing decision.
# ----------------------------------------------------------------------------------------
def test_maglif_config_is_documented_and_difficult():
    """The representative config carries its assumed plasma state + provenance basis."""
    cfg = sd.maglif_representative_config()
    for key in ("dx_cm", "cfl", "basis"):
        assert key in cfg, f"MagLIF config missing '{key}'"
    assert cfg["dx_cm"] == pytest.approx(5.0e-4, rel=1e-9)  # 5 micron paper resolution
    assert isinstance(cfg["basis"], str) and len(cfg["basis"]) > 0


def test_maglif_operators_all_route_to_sts():
    """On the difficult config, conduction, resistive B, and thick radiation are stiff by
    orders of magnitude -> all route to STS (the ADR-0001 decision)."""
    report = sd.maglif_stiffness_report()
    for op in ("conduction", "resistive_b", "radiation_thick"):
        assert op in report, f"missing operator '{op}' in stiffness report"
        assert report[op]["ratio"] > 50.0, (
            f"{op}: expected strongly-stiff ratio, got {report[op]['ratio']}")
        assert report[op]["route"] == "sts"
        # STS pays far fewer substages than the explicit subcycle count (= ratio).
        assert report[op]["rkl2_stages"] < report[op]["ratio"]


def test_maglif_thin_radiation_hits_c_cfl_floor():
    """In optically-thin cells the Larsen limiter free-streams at c: dt -> c-CFL floor
    (dt ~ dx/c), which STS cannot accelerate (ADR-0001 known consequence)."""
    report = sd.maglif_stiffness_report()
    assert "radiation_thin" in report
    thin = report["radiation_thin"]
    # free-streaming dt_exp is set by the speed of light, not the diffusive dx^2/D limit
    assert thin["dt_exp_s"] == pytest.approx(thin["c_cfl_floor_s"], rel=0.5)


def test_spitzer_magnetic_diffusivity_scales_as_T_to_minus_three_halves():
    """Spitzer resistivity eta ~ T^{-3/2} -> magnetic diffusivity colder = stiffer."""
    d_hot = sd.spitzer_magnetic_diffusivity(t_ev=1000.0, z=1.0, ln_lambda=7.0)
    d_cold = sd.spitzer_magnetic_diffusivity(t_ev=10.0, z=1.0, ln_lambda=7.0)
    # (1000/10)^{3/2} = 1000x colder-is-more-resistive
    assert d_cold / d_hot == pytest.approx(1000.0, rel=1e-6)
    assert d_cold > d_hot
