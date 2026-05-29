"""
Per-operator stiffness diagnostic + STS-vs-flux-fuse routing (issue [A2]/#109).

The Phase-Integration epic (#106) wires several stiff parabolic HED operators into the
MHD timestep.  Each must be advanced one of two ways (ADR-0001):

  * **STS** -- operator-split RKL2 super-time-stepping.  Covers the full hyperbolic step
    ``dt_hyp`` in one super-step of ``s ~ ceil((sqrt(9+16 r)-1)/2)`` substages
    (``r = dt_hyp/dt_exp``), each needing one per-substage multi-block ghost exchange
    (the shared ``SyncParabolicGhosts`` helper, #108).  Cost grows like ``sqrt(r)``.
  * **flux-fused** -- the diffusive flux rides the existing hyperbolic flux update; the
    global timestep simply becomes ``min(dt_hyp, dt_exp)`` (see ``Hydro::NewTimeStep`` in
    ``src/hydro/hydro_newdt.cpp``, the ``cond_operator_split == false`` branch).  No
    per-substage exchange is added, but if ``dt_exp < dt_hyp`` the *whole* simulation
    slows by the factor ``r``.

This module is the diagnostic that decides, per operator, which path to take, by
reporting each operator's forward-Euler stable timestep ``dt_exp`` relative to the
hyperbolic ``dt_hyp`` on a representative-difficult MagLIF stagnation configuration.

Every ``*_stable_dt`` below mirrors the corresponding C++ ``ExplicitStableDt`` closed
form so the Python diagnostic and the production kernels agree by construction:

  * ``conduction_stable_dt``        <- ``src/diffusion/conduction_operator.cpp`` (and
                                       ``aniso_conduction_operator.cpp``, with kappa_par)
  * ``resistivity_stable_dt``       <- ``src/diffusion/resistivity.cpp``
  * ``resistive_bphi_stable_dt``    <- ``src/diffusion/resistive_bphi_operator.cpp``
  * ``larsen_limiter`` / ``fld_*``  <- ``src/radiation_fld/fld_grey_operator.{hpp,cpp}``
                                       and ``fld_multigroup_operator.cpp``
  * ``hyperbolic_dt``               <- ``src/hydro/hydro_newdt.cpp`` + driver CFL scaling
  * ``rkl2_num_stages``             <- ``parabolic::RKL2NumStages``
                                       (``src/driver/parabolic_integrator.hpp``)

The routing decision this diagnostic produces is documented in the ADR-0001 routing
addendum and registered in ``docs/verification.md``.
"""

import math

# Default ideal-gas adiabatic index used by the conduction stability limit.
GAMMA_DEFAULT = 5.0 / 3.0

# Pragmatic routing cutoff (ADR-0001 routing addendum, issue #109): an operator whose
# explicit stable dt is within FLUX_FUSE_RATIO of the hyperbolic dt is not stiff enough to
# justify a per-substage ghost exchange -- it rides the flux-fused path for free.  Above
# it, STS amortizes the stiffness with ~sqrt(ratio) substages instead of ratio-x more
# hydro steps.  Tunable; the default is deliberately conservative (only genuinely-mild
# operators flux-fuse).
FLUX_FUSE_RATIO = 2.0

# Physical constants (CGS practical units; NRL Plasma Formulary conventions).
C_LIGHT_CGS = 2.99792458e10   # speed of light [cm/s]
MU0_SI = 4.0e-7 * math.pi     # vacuum permeability [H/m]


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------
def _as_dims(value, ndim):
    """Broadcast a scalar to ``ndim`` per-direction values, or pass a sequence through."""
    if isinstance(value, (list, tuple)):
        if len(value) != ndim:
            raise ValueError(f"expected {ndim} per-direction values, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * ndim


def _fe_prefactor(ndim):
    """Forward-Euler diffusion prefactor fac = 1/2, 1/4, 1/6 in 1/2/3-D -- the factor the
    C++ ConductionOperator/Resistivity ExplicitStableDt loops multiply dx^2/D by."""
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim must be 1, 2, or 3 (got {ndim})")
    return {1: 0.5, 2: 0.25, 3: 1.0 / 6.0}[ndim]


# ----------------------------------------------------------------------------------------
# Forward-Euler stable timesteps -- one per shipped ParabolicOperator
# ----------------------------------------------------------------------------------------
def conduction_stable_dt(dx, rho, kappa, gamma=GAMMA_DEFAULT, ndim=1):
    """dt = fac * min_i(dx_i^2) * rho/(kappa (gamma-1)).

    Mirrors ConductionOperator::ExplicitStableDt (and AnisotropicConductionOperator with
    kappa = kappa_par, the most restrictive coefficient).  ``dx`` may be a scalar or a
    per-direction sequence.
    """
    dxs = _as_dims(dx, ndim)
    gm1 = gamma - 1.0
    return _fe_prefactor(ndim) * min(d * d for d in dxs) * rho / (kappa * gm1)


def resistivity_stable_dt(dx, eta, ndim=1):
    """dt = fac * min_i(dx_i^2)/eta.  Mirrors Resistivity::NewTimeStep (eta = magnetic
    diffusivity)."""
    dxs = _as_dims(dx, ndim)
    return _fe_prefactor(ndim) * min(d * d for d in dxs) / eta


def resistive_bphi_stable_dt(dx, eta, ndim=1, radius=None):
    """dt = 1/rate, rate = sum_i 2 eta/w_i^2 (+ eta/r^2 near the axis).

    Mirrors ResistiveBphiOperator::ExplicitStableDt: the cylindrical curl-curl 1/r^2 term
    tightens dt near the axis.  Pass ``radius`` (cell-centre r) to include it.
    """
    dxs = _as_dims(dx, ndim)
    rate = sum(2.0 * eta / (d * d) for d in dxs)
    if radius is not None:
        rate += eta / (radius * radius)
    return 1.0 / rate


def larsen_limiter(R, n=2.0):
    """Larsen flux limiter lambda(R) = (3^n + R^n)^(-1/n).

    Mirrors radiationfld::LarsenLimiter.  lambda -> 1/3 in the thick (R->0) limit and
    -> 1/R (free-streaming) in the thin (R>>1) limit.
    """
    rc = R if R > 0.0 else 0.0
    return (3.0 ** n + rc ** n) ** (-1.0 / n)


def fld_diffusivity(c, chi, R, nlarsen=2.0):
    """Flux-limited radiation diffusivity D = c lambda(R)/chi (Larsen limiter)."""
    return c * larsen_limiter(R, nlarsen) / chi


def fld_stable_dt(dx, D, ndim=1):
    """dt = 1/(2 D sum_i 1/dx_i^2).

    Mirrors FLD{Grey,Multigroup}Operator::ExplicitStableDt (D = cell-centred flux-limited
    diffusivity; the multigroup form minimises this over groups).
    """
    dxs = _as_dims(dx, ndim)
    inv = sum(1.0 / (d * d) for d in dxs)
    return 1.0 / (2.0 * D * inv)


# ----------------------------------------------------------------------------------------
# Hyperbolic timestep and the RKL2 stage count
# ----------------------------------------------------------------------------------------
def hyperbolic_dt(dx, vel, cs, cfl=0.3, ndim=1):
    """dt_hyp = cfl * min_i dx_i/(|v_i| + c_s).

    Mirrors the non-relativistic branch of Hydro::NewTimeStep (src/hydro/hydro_newdt.cpp)
    folded with the driver's CFL scaling.
    """
    dxs = _as_dims(dx, ndim)
    vels = _as_dims(vel if vel is not None else 0.0, ndim)
    return cfl * min(d / (abs(v) + cs) for d, v in zip(dxs, vels))


def rkl2_num_stages(dt_super, dt_exp):
    """s = max(2, ceil((sqrt(9 + 16 r) - 1)/2)),  r = dt_super/dt_exp.

    Mirrors parabolic::RKL2NumStages: the smallest RKL2 stage count whose stability
    interval (s^2+s-2)/4 covers the requested super-step/explicit ratio (Meyer, Balsara &
    Aslam 2014), floored at 2.
    """
    ratio = (dt_super / dt_exp) if dt_exp > 0.0 else 0.0
    if ratio < 0.0:
        ratio = 0.0
    s = math.ceil(0.5 * (math.sqrt(9.0 + 16.0 * ratio) - 1.0))
    return max(2, int(s))


# ----------------------------------------------------------------------------------------
# Stiffness ratio + routing classifier
# ----------------------------------------------------------------------------------------
def stiffness_ratio(dt_hyp, dt_exp):
    """The stiffness ratio r = dt_hyp/dt_exp: how many forward-Euler substeps the
    operator would need per hyperbolic step (= the flux-fused slowdown factor)."""
    return dt_hyp / dt_exp


def route(dt_hyp, dt_exp, flux_fuse_ratio=FLUX_FUSE_RATIO):
    """Return ``"flux_fused"`` if the operator is mild (r <= flux_fuse_ratio) or ``"sts"``
    if it is stiff.  See the ADR-0001 routing addendum for the rationale."""
    return "flux_fused" if stiffness_ratio(dt_hyp, dt_exp) <= flux_fuse_ratio else "sts"


# ----------------------------------------------------------------------------------------
# Standard plasma closed forms (NRL Plasma Formulary; Spitzer 1962 / Braginskii 1965)
# ----------------------------------------------------------------------------------------
def spitzer_thermal_diffusivity(t_ev, n_e_cm3, z=1.0, ln_lambda=7.0):
    """Electron (parallel) Spitzer-Haerm thermal diffusivity chi_e [cm^2/s].

    chi_e ~ 3.2 v_te^2 tau_e, with v_te^2 = 1.7556e15 T_e[eV] (cm^2/s^2) and the NRL
    electron collision time tau_e = 3.44e5 T_e[eV]^{3/2}/(n_e lnLambda Z) s.  Used only to
    set a representative-difficult conduction stiffness; the routing decision is robust to
    O(1) uncertainty in the prefactor.
    """
    v_te2 = 1.7556e15 * t_ev
    tau_e = 3.44e5 * t_ev ** 1.5 / (n_e_cm3 * ln_lambda * z)
    return 3.2 * v_te2 * tau_e


def spitzer_magnetic_diffusivity(t_ev, z=1.0, ln_lambda=7.0):
    """Magnetic diffusivity D_eta [cm^2/s] from Spitzer perpendicular resistivity.

    eta_perp[Ohm m] = 1.03e-4 Z lnLambda T_e[eV]^{-3/2} (Spitzer 1962); the magnetic
    diffusivity D_eta = eta/mu0 [m^2/s], converted to cm^2/s.  The T^{-3/2} scaling makes
    cold material (the liner edge / vacuum floor) the resistively-stiffest region.
    """
    eta_perp_si = 1.03e-4 * z * ln_lambda * t_ev ** (-1.5)
    d_eta_si = eta_perp_si / MU0_SI           # m^2/s
    return d_eta_si * 1.0e4                    # cm^2/s


def ion_sound_speed(t_ev, mu=2.0, z=1.0, gamma=GAMMA_DEFAULT):
    """Ion-acoustic / sound speed c_s [cm/s] ~ 9.79e5 sqrt(gamma Z T_e[eV]/mu) (NRL),
    mu = ion mass in proton units (D fuel mu=2)."""
    return 9.79e5 * math.sqrt(gamma * z * t_ev / mu)


# ----------------------------------------------------------------------------------------
# Representative-difficult MagLIF stagnation configuration + per-operator report
# ----------------------------------------------------------------------------------------
def maglif_representative_config():
    """The representative-DIFFICULT MagLIF stagnation state used to route the operators.

    Numbers are order-of-magnitude engineering inputs (CGS), each grounded in the MagLIF
    stagnation literature (Slutz & Vesey 2012; Gomez et al. 2014; Ellison et al. 2025,
    arXiv:2504.10760) and standard plasma transport closed forms -- they are NOT validated
    experimental data (those live in ground_truth/, per ADR-0008).  The routing decision
    is insensitive to O(1) errors in these inputs because the operators are stiff by many
    orders of magnitude at 5 micron resolution.
    """
    return {
        "dx_cm": 5.0e-4,           # 5 micron finest-AMR-level (paper) resolution
        "ndim": 2,                 # r-z MagLIF benchmark geometry
        "cfl": 0.3,
        # hot DD fuel at stagnation (sets the thermal-conduction stiffness)
        "fuel_te_ev": 2.0e3,       # ~2 keV (Gomez 2014 measured ~3 keV)
        "fuel_ne_cm3": 1.0e21,     # compressed, preheated DD
        "fuel_mu": 2.0, "fuel_z": 1.0,
        # cold liner edge / vacuum floor (sets the resistive stiffness; eta ~ T^-3/2)
        "liner_te_ev": 1.0e1,      # ~10 eV cold material
        "liner_mu": 27.0, "liner_z": 13.0,  # aluminium liner
        # optically-thick liner Rosseland opacity (sets thick-radiation diffusivity)
        "rosseland_chi_cm": 1.0e3,
        # optically-thin opacity (free-streaming radiation -> c-CFL floor)
        "thin_chi_cm": 1.0,
        "ln_lambda": 7.0,
        "basis": ("CGS engineering estimates from MagLIF stagnation literature "
                  "(Slutz & Vesey 2012; Gomez 2014; Ellison 2025) + Spitzer/Braginskii "
                  "transport closed forms (NRL Plasma Formulary). Routing-decision "
                  "inputs only; not experimental ground truth."),
    }


def _classified(dt_hyp, dt_exp):
    """Bundle the ratio / RKL2 stage count / routing for one operator."""
    r = stiffness_ratio(dt_hyp, dt_exp)
    return {
        "dt_exp_s": dt_exp,
        "dt_hyp_s": dt_hyp,
        "ratio": r,
        "rkl2_stages": rkl2_num_stages(dt_hyp, dt_exp),
        "route": route(dt_hyp, dt_exp),
    }


def maglif_stiffness_report():
    """Report ``{dt_exp, dt_hyp, ratio, rkl2_stages, route}`` per operator on the
    representative-difficult MagLIF config.  This is the diagnostic the routing decision
    (ADR-0001 addendum) is read off."""
    cfg = maglif_representative_config()
    dx = cfg["dx_cm"]
    ndim = cfg["ndim"]
    cfl = cfg["cfl"]
    lnl = cfg["ln_lambda"]

    # Electron thermal conduction (hot fuel): chi_e via Spitzer-Haerm.
    chi_e = spitzer_thermal_diffusivity(cfg["fuel_te_ev"], cfg["fuel_ne_cm3"],
                                        z=cfg["fuel_z"], ln_lambda=lnl)
    cs_fuel = ion_sound_speed(cfg["fuel_te_ev"], mu=cfg["fuel_mu"], z=cfg["fuel_z"])
    # conduction dt_exp via the diffusivity form (kappa*(gamma-1)/rho = chi_e):
    dt_cond = conduction_stable_dt(dx, rho=1.0, kappa=chi_e, gamma=2.0, ndim=ndim)
    report = {"conduction": _classified(hyperbolic_dt(dx, 0.0, cs_fuel, cfl, ndim),
                                        dt_cond)}

    # Resistive B (cold liner edge): D_eta via Spitzer perpendicular resistivity.
    d_eta = spitzer_magnetic_diffusivity(cfg["liner_te_ev"], z=cfg["liner_z"],
                                         ln_lambda=lnl)
    cs_liner = ion_sound_speed(cfg["liner_te_ev"], mu=cfg["liner_mu"], z=cfg["liner_z"])
    dt_res = resistive_bphi_stable_dt(dx, d_eta, ndim=ndim)
    dt_hyp_liner = hyperbolic_dt(dx, 0.0, cs_liner, cfl, ndim)
    report["resistive_b"] = _classified(dt_hyp_liner, dt_res)

    # Multigroup FLD radiation -- optically thick (diffusive) limit, R->0.
    d_rad = fld_diffusivity(C_LIGHT_CGS, cfg["rosseland_chi_cm"], R=0.0)
    dt_rad = fld_stable_dt(dx, d_rad, ndim=ndim)
    report["radiation_thick"] = _classified(dt_hyp_liner, dt_rad)

    # Multigroup FLD radiation -- optically thin (free-streaming) limit.  A one-cell-scale
    # gradient gives R = 1/(chi dx) >> 1, so lambda -> 1/R and D -> c dx: the diffusive
    # dt_exp collapses to the c-CFL floor dt ~ dx/(2 ndim c), which STS cannot accelerate
    # (ADR-0001 known consequence -- mitigated by local reduced-c only if it sets dt).
    r_thin = 1.0 / (cfg["thin_chi_cm"] * dx)
    d_thin = fld_diffusivity(C_LIGHT_CGS, cfg["thin_chi_cm"], R=r_thin)
    dt_thin = fld_stable_dt(dx, d_thin, ndim=ndim)
    thin = _classified(dt_hyp_liner, dt_thin)
    thin["c_cfl_floor_s"] = dx / (2.0 * ndim * C_LIGHT_CGS)
    report["radiation_thin"] = thin

    return report
