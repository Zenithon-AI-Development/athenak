"""
Multigroup LF streaming-dissipation convergence study (issue #221).

#215 ported the grey #194 gated Lax-Friedrichs (LF) streaming dissipation to
``FLDMultigroupOperator``.  It is required for stability, but it is FIRST-ORDER
numerical diffusion: at the committed ``multigroup_rad_front`` resolution (256 cells) it
shifted the observable by up to 12.6 % (E_g2, the most transparent group, where the
``(1 - 3 lambda)_+`` gate opens widest).  The correct claim for such a term is that its
error CONVERGES AWAY as ``dx -> 0``; this test demonstrates it rather than assumes it.

What the offline study behind this test (256..4096 cells, LF on/off, n_super up to 320)
measured, and what is asserted here per the #221 acceptance criteria:

  1. The committed deck is structurally PRE-ASYMPTOTIC: chi_R,g = {300, 200, 150} on a
     unit domain gives only 0.9-1.7 CELLS per photon mean free path at 256 cells.  The
     LF-attributable deviation (LF-on vs LF-off on the same grid, window L1 on E_g2) is
     flat ~0.11-0.12 over 256-512-1024 and DECREASES once the mfp is resolved:
     1.14e-1 (1024) -> 9.2e-2 (2048).  The study triple is therefore (1024, 2048,
     4096), not the deck's 1x/2x/4x.
  2. The PRODUCTION scheme (LF on) self-converges: Richardson decrements on E_g2 are
     8.8e-2 / 9.0e-2 / 7.9e-2 / 4.7e-2 for successive doublings 256->4096 -- flat
     through the pre-asymptotic window, then halving (observed order 0.74, climbing
     toward 1) once the front structure is resolved.
  3. The BARE CENTERED scheme (``mgfld_upwind = 0``) has NO stable refinement limit on
     this deck: it runs away (E_g2 -> 1e58..NaN) at 4096 cells with n_super = 320, and
     already at 2048 cells at the committed n_super = 40 -- the #194 imaginary-mode
     runaway reappearing as refinement steepens the front foot.  The AC-3 separation
     leg therefore ends by demonstrating the LF term is what gives the multigroup path
     a continuum limit at all; the gate does NOT need re-deriving -- the plateau the
     issue feared is the pre-asymptotic window, not the gate.
  4. Every healthy leg stays inside the committed analytic Planck-lock oracle
     ``E_g(x) = a T_e(x)^4 [F(x_{g+1}) - F(x_g)]`` (independent quadrature, tolerance
     5e-3; measured <= 9.5e-5 on the study legs) -- the analytic reference of AC 1.

The study legs run with ``n_super = 320`` (dt_super = 0.0625): at the committed
``n_super = 40`` the RKL2 stage count at fine dx grows so large that BOTH gates
degrade at 4096 (even LF-on overshoots to E_g2 ~ 84) -- a super-step-size limit of
RKL2 at extreme stage counts, reported on #221 as follow-up, distinct from the LF
question studied here.  The temporal error of n_super = 320 is <= 3e-3 on E_g2
(measured by n_super doubling), subdominant to every spatial signal asserted.  The
knob-liveness pair runs at the committed deck (256 cells, n_super = 40) where the
12.6 % shift of #221 is reproduced exactly.  Fixed dt_super means that within every
LF-on/LF-off pair the ONLY difference is the LF term.  Total runtime ~4 minutes,
single core, dominated by the 4096-cell legs.

Shares the pgen build (and run directory) with ``test_verify_multigroup_rad_hydro_cpu``;
like the other multigroup verifications it must not run concurrently with them (the
.dat outputs share filenames).  Auto-collected by run_test_suite.py (name has ``_cpu``).
"""

# Modules
import os

import numpy as np
import test_suite.testutils as testutils

# numpy < 2.0 has no np.trapezoid (the renamed np.trapz); numpy >= 2.0 has no np.trapz.
# The fallback must be LAZY: an eager getattr default (np.trapz) touches the removed
# attribute at import time on numpy 2.x and aborts the whole suite's collection.
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# Problem (constants must match tst/inputs/multigroup_rad_equilibration.athinput).
PROBLEM = "radiation_fld_multigroup_equilibration"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "multigroup_rad_equilibration.athinput"
)
fixture = os.path.join(
    testutils._repo_root(), "inputs", "radiation_fld", "multigroup_rad_hydro_opacity.cn4"
)

A_RAD, KBOLTZ = 1.0, 30.0

# Front window: identical to test_verify_multigroup_rad_hydro_cpu (clear of the
# spectrally-inconsistent near-wall boundary layer, containing the group fronts).
WIN_LO, WIN_HI = 0.13, 0.45

# The three-resolution study triple (AC 1), chosen INSIDE the asymptotic regime: the
# most transparent group has a photon mfp of 1/150 of the domain, i.e. 6.8 / 13.7 /
# 27.3 cells per mfp here -- the committed 256-cell deck (1.7 cells/mfp) cannot resolve
# the front foot the LF term acts on (see module docstring).
RESOLUTIONS = (1024, 2048, 4096)
N_SUPER_STUDY = 320    # dt_super = 0.0625: keeps RKL2 stage counts sane at 4096 cells

# The committed deck (resolution + n_super): the knob-liveness pair runs here, where
# the LF term moves E_g2 by the 12.6 % documented in #221.
NX_COMMITTED = 256
N_SUPER_COMMITTED = 40

# Baseline-diff-style relative-difference floor (harness atol for this slice).
FIELD_ATOL = 1.0e-7

# The committed analytic-oracle tolerance on the Planck-spectrum lock (the per-group
# spectral check of test_verify_multigroup_rad_hydro_cpu); every healthy leg holds it.
LOCK_TOL = 5.0e-3

# A leg is HEALTHY when finite and of physical magnitude: E is bounded by a few times
# the Dirichlet source e_source = 1 (the centered scheme's runaway reaches 1e58+ while
# still formally finite, so finiteness alone is not health).
E_PHYSICAL_MAX = 10.0


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
        out[i] = 15.0 / np.pi ** 4 * _trapz(integ, t)
    return out.reshape(x.shape)


def _planck_spectrum(temp, bounds):
    """Per-group Planck energy slice a T^4 [F(eps_hi/(kB T)) - F(eps_lo/(kB T))]."""
    temp = np.asarray(temp, dtype=float)
    frac = (_planck_fraction(bounds[1:][None, :] / (KBOLTZ * temp[:, None]))
            - _planck_fraction(bounds[:-1][None, :] / (KBOLTZ * temp[:, None])))
    return A_RAD * (temp[:, None] ** 4) * frac


def _run_front_leg(nx1, upwind, n_super):
    """Run the PART-2 multigroup Marshak front at nx1 cells with the LF gate set to
    ``upwind`` (problem/mgfld_upwind) over n_super fixed super-steps, returning
    (x, E_g(x,g), Te(x), E_tot(x), bounds) on the active cells, with the .dat outputs
    consumed (removed)."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    front_path = os.path.join(run_dir, "multigroup_rad_front.dat")
    grp_path = os.path.join(run_dir, "multigroup_rad_groups.dat")
    hist_path = os.path.join(run_dir, "multigroup_rad_equil.dat")
    args = [
        f"problem/opacity_file={fixture}",
        f"mesh/nx1={nx1}",
        f"meshblock/nx1={nx1}",
        f"problem/mgfld_upwind={upwind}",
        f"problem/n_super={n_super}",
        # PART 1 is resolution-independent (0-D relaxation); one step keeps the legs
        # cheap without touching the PART-2 front this study measures.
        "problem/eq_nstep=1",
    ]
    assert testutils.run_pgen(PROBLEM, input_file, args=args), (
        f"multigroup front leg failed: nx1={nx1} mgfld_upwind={upwind}"
    )
    try:
        front = np.loadtxt(front_path)
        grp = np.atleast_2d(np.loadtxt(grp_path))
    finally:
        for p in (front_path, grp_path, hist_path):
            try:
                os.remove(p)
            except OSError:
                pass
    ngroups = grp.shape[0]
    assert ngroups >= 2, f"expected multiple groups, got {ngroups}"
    bounds = np.concatenate([[grp[0, 1]], grp[:, 2]])
    x = front[:, 0]
    assert x.size == nx1, f"expected {nx1} cells, got {x.size}"
    eg_x = front[:, 1:1 + ngroups]
    te_x = front[:, 1 + ngroups + 1]
    etot_x = front[:, 1 + ngroups + 2]
    return x, eg_x, te_x, etot_x, bounds


def _healthy(leg):
    """Finite AND physically bounded (the centered runaway can be finite yet 1e58)."""
    x, eg_x = leg[0], leg[1]
    mask = (x > WIN_LO) & (x < WIN_HI)
    return bool(np.all(np.isfinite(eg_x))
                and np.max(np.abs(eg_x[mask])) < E_PHYSICAL_MAX)


def _lock_deviation(leg):
    """Max relative deviation from the analytic Planck-spectrum lock per group over the
    front window, normalised by the local a Te^4 (as the committed spectral oracle)."""
    x, eg_x, te_x, _, bounds = leg
    mask = (x > WIN_LO) & (x < WIN_HI)
    assert mask.sum() > 10, "front verification window too small"
    lock = _planck_spectrum(te_x[mask], bounds)
    dev = []
    for g in range(eg_x.shape[1]):
        rel = np.abs(eg_x[mask, g] - lock[:, g]) / np.maximum(
            A_RAD * te_x[mask] ** 4, 1.0e-30
        )
        dev.append(float(np.max(rel)))
    return dev


def _window_l1(x, ua, ub):
    """Relative L1 difference over the front window (a front-position-shift norm,
    robust where pointwise max-rel at an ever-steepening front conflates sharpness
    with error)."""
    mask = (x > WIN_LO) & (x < WIN_HI)
    return float(np.sum(np.abs(ua[mask] - ub[mask])) / np.sum(np.abs(ub[mask])))


def _window_maxrel(x, ua, ub):
    """Max relative pointwise difference over the front window (baseline-diff norm)."""
    mask = (x > WIN_LO) & (x < WIN_HI)
    return float(np.max(
        np.abs(ua[mask] - ub[mask]) / np.maximum(np.abs(ub[mask]), FIELD_ATOL)
    ))


def _richardson(leg_coarse, leg_fine):
    """Coarse-interpolated-to-fine window relative L1 on E_g2 (self-convergence)."""
    xc, egc = leg_coarse[0], leg_coarse[1]
    xf, egf = leg_fine[0], leg_fine[1]
    g = egc.shape[1] - 1
    mask = (xf > WIN_LO) & (xf < WIN_HI)
    uci = np.interp(xf[mask], xc, egc[:, g])
    return float(np.sum(np.abs(uci - egf[mask, g])) / np.sum(np.abs(egf[mask, g])))


def test_verify_multigroup_lf_convergence():
    """Three-resolution LF-on/LF-off study of the multigroup Marshak front (#221)."""
    legs = {}
    try:
        # knob-liveness pair at the committed deck.
        for upwind in (1, 0):
            legs[(NX_COMMITTED, upwind)] = _run_front_leg(
                NX_COMMITTED, upwind, N_SUPER_COMMITTED
            )
        # the convergence study triple.
        for nx1 in RESOLUTIONS:
            for upwind in (1, 0):
                legs[(nx1, upwind)] = _run_front_leg(nx1, upwind, N_SUPER_STUDY)
    finally:
        testutils.cleanup()

    print("\n[#221] multigroup LF streaming-dissipation convergence study")
    print(f"  front window: {WIN_LO} < x < {WIN_HI}; committed deck {NX_COMMITTED} "
          f"cells (n_super={N_SUPER_COMMITTED}); study triple {RESOLUTIONS} "
          f"(n_super={N_SUPER_STUDY})")

    # ---- the production scheme (LF on) must be healthy everywhere: stability is the
    #      very thing the LF term exists to provide (#215/#194).
    for nx1 in (NX_COMMITTED,) + RESOLUTIONS:
        assert _healthy(legs[(nx1, 1)]), (
            f"LF-on leg unphysical/not finite at nx1={nx1}: the production "
            f"multigroup scheme lost the stability the LF term provides (#215/#194)"
        )

    # ---- (AC 1) deviation from the analytic Planck-lock reference per resolution;
    #      asserted on the production legs, reported for the healthy centered legs.
    for nx1 in (NX_COMMITTED,) + RESOLUTIONS:
        for upwind in (1, 0):
            leg = legs[(nx1, upwind)]
            if not _healthy(leg):
                print(f"  nx1={nx1:5d} mgfld_upwind={upwind}: RUNAWAY (centered "
                      f"scheme, see AC-3 below)")
                continue
            dev = _lock_deviation(leg)
            rep = ", ".join(f"g{g}={d:.3e}" for g, d in enumerate(dev))
            print(f"  nx1={nx1:5d} mgfld_upwind={upwind}: Planck-lock dev [{rep}]")
            if upwind == 1:
                assert max(dev) < LOCK_TOL, (
                    f"nx1={nx1} mgfld_upwind=1: Planck-lock deviation {max(dev):.3e} "
                    f"outside the committed oracle tolerance {LOCK_TOL:g}"
                )

    # ---- knob liveness: at the committed deck the LF term moves E_g2 by the 12.6 %
    #      documented in #221, so the pair must differ well clear of noise.  RED while
    #      the pgen does not plumb problem/mgfld_upwind through to
    #      FLDMultigroupOperator (both legs then silently run LF-on and the difference
    #      is identically zero).
    on_c, off_c = legs[(NX_COMMITTED, 1)], legs[(NX_COMMITTED, 0)]
    assert _healthy(off_c), f"LF-off leg unphysical at nx1={NX_COMMITTED}"
    g2 = on_c[1].shape[1] - 1
    knob = _window_maxrel(on_c[0], on_c[1][:, g2], off_c[1][:, g2])
    print(f"  knob check nx1={NX_COMMITTED}: LF-on vs LF-off E_g2 max rel "
          f"{knob:.3e} (#221 measured 1.26e-1)")
    assert knob > 1.0e-2, (
        f"LF-on vs LF-off differ by only {knob:.3e} at nx1={NX_COMMITTED}: the "
        f"mgfld_upwind gate is not reaching the operator"
    )

    # ---- (AC 2) the LF-attributable deviation decreases under refinement in the
    #      resolved regime.  Measured (E_g2, window L1): 1.10e-1 (256), 1.23e-1 (512),
    #      1.14e-1 (1024), 9.2e-2 (2048) -- flat while the mfp is unresolved, then
    #      falling; asserted on the healthy pairs of the study triple.
    lf_delta = {}
    for nx1 in RESOLUTIONS:
        leg_on, leg_off = legs[(nx1, 1)], legs[(nx1, 0)]
        if not _healthy(leg_off):
            continue
        lf_delta[nx1] = _window_l1(leg_on[0], leg_on[1][:, g2], leg_off[1][:, g2])
        print(f"  nx1={nx1:5d} LF-attributable E_g2 window-L1 delta: "
              f"{lf_delta[nx1]:.3e}")
    comparable = [r for r in RESOLUTIONS if r in lf_delta]
    assert len(comparable) >= 2, (
        "fewer than two healthy LF-off legs in the study triple: cannot measure the "
        "LF-attributable deviation trend at all"
    )
    for coarse, fine in zip(comparable[:-1], comparable[1:]):
        assert lf_delta[fine] < lf_delta[coarse], (
            f"LF-attributable deviation does not decrease in the resolved regime: "
            f"delta({coarse})={lf_delta[coarse]:.3e} -> "
            f"delta({fine})={lf_delta[fine]:.3e}; the (1-3*lambda)_+ gate is too "
            f"aggressive after all and needs re-deriving (#221)"
        )

    # ---- (AC 2) the production scheme self-converges at approaching first order:
    #      Richardson decrements on E_g2 must SHRINK across the study triple
    #      (measured 7.9e-2 -> 4.7e-2, observed order 0.74 and climbing; a flat or
    #      growing decrement is the signature of the pre-asymptotic plateau, or of a
    #      gate injecting non-vanishing dissipation).
    d1 = _richardson(legs[(RESOLUTIONS[0], 1)], legs[(RESOLUTIONS[1], 1)])
    d2 = _richardson(legs[(RESOLUTIONS[1], 1)], legs[(RESOLUTIONS[2], 1)])
    order = float(np.log2(d1 / d2)) if d2 > 0.0 else float("inf")
    print(f"  LF-on Richardson decrements (E_g2 L1): "
          f"{RESOLUTIONS[0]}->{RESOLUTIONS[1]} {d1:.3e}, "
          f"{RESOLUTIONS[1]}->{RESOLUTIONS[2]} {d2:.3e}, observed order {order:.2f}")
    assert d2 < d1, (
        f"the production (LF-on) scheme is not converging under refinement: "
        f"Richardson decrement grew {d1:.3e} -> {d2:.3e} (#221)"
    )
    assert order > 0.4, (
        f"the production (LF-on) scheme converges too slowly (observed order "
        f"{order:.2f} < 0.4, expected to approach 1): the LF gate injects "
        f"non-vanishing dissipation and needs re-deriving (#221)"
    )

    # ---- (AC 3) LF-off separation leg at the finest resolution.  Two admissible
    #      outcomes, both reported: (a) the centered leg is healthy -> its analytic-
    #      reference deviation bounds the underlying discretization error and the
    #      LF-on leg must sit within a factor 2 of it; (b) the centered leg runs away
    #      (the measured outcome: E_g2 -> 1e58..NaN at 4096) -- the #194
    #      imaginary-mode runaway on this very deck: the underlying discretization has
    #      NO stable refinement limit and the LF term is what provides one.  Either
    #      way the LF error is separated from the underlying discretization error.
    fin = RESOLUTIONS[-1]
    lock_on = max(_lock_deviation(legs[(fin, 1)]))
    leg_off_fin = legs[(fin, 0)]
    if _healthy(leg_off_fin):
        lock_off = max(_lock_deviation(leg_off_fin))
        print(f"  AC-3 separation at nx1={fin}: Planck-lock dev LF-on {lock_on:.3e} "
              f"vs LF-off {lock_off:.3e}")
        assert lock_on < 2.0 * max(lock_off, 1.0e-4), (
            f"at nx1={fin} the LF term still dominates the analytic-reference "
            f"deviation: LF-on {lock_on:.3e} vs LF-off {lock_off:.3e}"
        )
    else:
        print(f"  AC-3 separation at nx1={fin}: the mgfld_upwind=0 (bare centered) "
              f"leg ran away -- the #194/#215 streaming instability on this deck; "
              f"the underlying discretization has no stable refinement limit, the "
              f"LF term provides it (LF-on Planck-lock dev {lock_on:.3e})")
