"""
RADIATION-ON BINDING grid convergence for the faithful B1 coupled benchmark (#199).

The radiation-ON analog of the #195 radiation-OFF gate (test_verify_maglif_b1_coupled).
#194 made the full-physics stack (resb + acond + grey FLD + mrad) run NaN-clean to 70ns,
adding the Lax-Friedrichs streaming dissipation + face-consistent FLD super-step dt; "NaN-
clean at two grids" is STABILITY, not CONVERGENCE.  This gate supplies the missing BINDING
grid-convergence verdict for the full-physics stack, the prerequisite that lets the B1
quantitative anchor (#120) be trusted.

Runs the radiation-ON stack (mhd/fld_operator_split=true + mhd/mrad_coupling=true,
fld_efloor=1e-10, fld_upwind=1 the #194 LF gate) on paper (288x4x128) / refined
(432x4x192) / 2x (576x4x256), all to 70 ns.  Each leg must stay bounded + finite +
div(B)-clean (the #192/#194 invariants) and grow the seeded mode.  Grid convergence is
BINDING on the refined<->2x pair, judged against RE-DERIVED radiation-ON bands
(RADON_CONV_RHO_RTOL / RADON_CONV_AMP_RTOL / RADON_CONV_TPEAK_TOL in
maglif_grid_convergence) -- NOT the radiation-OFF #195 bands.  Paper-288 is the
UNDER-RESOLVED outlier (kept as a stability anchor, its under-resolution REPORTED,
excluded from the binding pair) -- the #195 finding re-confirmed radiation-ON.

The pure band logic + the recorded radiation-ON bracket are pinned offline (no GPU) in
test_unit_maglif_grid_convergence_radon_cpu, so gate and unit test share one definition of
"converged" and the bands run in CPU CI.  Bands re-derived per ADR-0015 addendum 3.

Radiation is nearly inert at the current constant FLD opacity (#204 will make it
kappa(rho,T) from IONMIX), so the radiation-ON bracket tracks the radiation-OFF #195
bracket closely; the bands are nonetheless DERIVED from the radiation-ON bracket, not
reused.  No regression to the radiation-OFF #192/#195 gate (a separate file, untouched).

GPU-only (_gpu suffix): built + run with CUDA on athenakdev; auto-collected by
run_test_suite.py --gpu (excluded from CPU CI).

(Adapted from test_verify_maglif_b1_coupled_gpu / issues #195 / #192.)
"""

# Modules
import glob
import os
import shutil

import numpy as np
import test_suite.testutils as testutils

# Reuse the faithful-B1 reduction pipeline (interfaces -> seeded-mode amplitude) and the
# committed deck paths from the operators-OFF paper-resolution benchmark so the two tests
# can never drift apart on what "the faithful B1 observable" means.
import test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu as b1
from test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu import (  # noqa: F401
    GPU_FLAGS, bin_convert, bin_dir, build_dir, input_file, trace_file,
)
# Shared grid-convergence verdict + bands (pure; CPU-tested in
# test_unit_maglif_grid_convergence_cpu) so gate and unit test agree on "converged".
from test_suite.cylindrical.maglif_grid_convergence import (
    RADON_CONV_AMP_RTOL, RADON_CONV_RHO_RTOL, RADON_CONV_TPEAK_TOL,
    grid_convergence_verdict,
)

# Absolute table path: the per-problem build dir runs with cwd build_pgen/maglif/src, so
# the deck's repo-relative eos_table must be overridden (same pattern as the operators
# attribution test).
table_file = os.path.join(
    testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
)

# The radiation-ON #194 full-physics stack: the Strang-wrapped parabolic block
# (ADR-0009) of the #195 radiation-OFF gate -- resb (driven b0.x2f) + acond (EOS-aware)
# + the ox1 inflow guard -- with grey FLD + mrad now FLIPPED ON.  #194 (the LF streaming
# dissipation + face-consistent FLD super-step dt) makes this stack NaN-clean to 70 ns;
# this gate adds the BINDING grid-convergence verdict the radon stability gate does not.
COUPLED_ARGS = [
    "mhd/strang_split=true",
    "mhd/resb_operator_split=true",
    "mhd/acond_operator_split=true",
    "mhd/fld_operator_split=true",    # radiation ON (#194 fixed the streaming runaway)
    "mhd/mrad_coupling=true",         # radiation ON
    "mhd/fld_efloor=1.0e-10",         # erad positivity floor (#194 green); fld_upwind=1
                                      # (the #194 LF streaming gate) on via deck default
    # #192: pin the open outer boundary against ghost-fed inflow.  The zero-gradient
    # ghost copy is an unbounded mass reservoir once the resb-diffused drive field
    # raises an inward wind in the vacuum gap (mass x77,000 by 70 ns at paper res);
    # the guard swaps inflowing columns to a static vacuum ghost state (ADR-0015).
    "problem/ox1_inflow_guard=true",
]

TLIM = 70.0          # ns; the Sinars experiment window (last committed datum 69.9 ns)
SNAP_MIN = 20        # >= 20 of the ~24 expected 3-ns snapshots (run reached tlim)
RHO_MAX_CODE = 1.0e3  # bounded-density gate, code units (solid Al = 1.0); see docstring

# Honest floor on the radiation-ON seeded-mode growth (peak a_outer / seed): only
# guards against a "fix" that over-damps and kills the MRT feed -- NOT an experiment-
# magnitude growth certification (that is #120's quantitative anchor).  Radiation is
# nearly inert at the constant opacity, so this tracks the radiation-OFF floor.
GROWTH_MIN_RADON = 1.20

# Grid-convergence legs in the meshed (r, z) plane, same physics, committed 144x4x64
# MeshBlock tiling.  The BINDING convergence pair is refined <-> 2x (the #195 converged
# pair); paper-288 (the deck default) is the UNDER-RESOLVED outlier, excluded from it.
REF_MESH = [   # 1.5x refinement: 432x4x192 (3x1x3 = 9 blocks)
    "mesh/nx1=432", "mesh/nx3=192",
    "meshblock/nx1=144", "meshblock/nx3=64",
]
TWOX_MESH = [  # 2x refinement: 576x4x256 (4x1x4 = 16 blocks)
    "mesh/nx1=576", "mesh/nx3=256",
    "meshblock/nx1=144", "meshblock/nx3=64",
]
# The grid-convergence bands (CONV_RHO_RTOL / CONV_AMP_RTOL / CONV_TPEAK_TOL) live in
# maglif_grid_convergence (imported above) and are exercised on CPU by
# test_unit_maglif_grid_convergence_cpu.  #195 bracket: with the div(B) runaway fixed the
# refined (432) and 2x (576) legs converge (rho_max 16.106 vs 16.089, 0.11%), so the gate
# now BINDS convergence on that pair; the paper (288) under-resolves (rho 2.89) and is
# reported, not in the binding pair.
# max|div B| (cyl FV) gate; clean ~1e-13, the #192 runaway hit 0.16 -> NaN
DIVB_MAX = 1.0e-6


def _run_coupled(basename, extra_args=None):
    """Run the faithful B1 deck with the radiation-ON #194 stack; return snapshot paths.

    Removes any stale ``{basename}.user.hst`` first: AthenaK history output
    APPENDS across runs, so a leftover file from a previous (possibly runaway)
    run would poison the div(B) gate that
    reads it (``_max_divb``).  A fresh run then writes a clean, single-run history file.
    """
    hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{basename}.user.hst")
    if os.path.exists(hst):
        os.remove(hst)
    args = [
        f"mhd/eos_table={table_file}",
        f"problem/current_file={trace_file}",
        f"job/basename={basename}",
        f"time/tlim={TLIM}",
    ] + COUPLED_ARGS
    if extra_args:
        args += extra_args
    ok = testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS, args=args)
    assert ok, f"radiation-ON #199 B1 GPU run failed (basename={basename})"
    files = sorted(glob.glob(os.path.join(bin_dir, f"{basename}.prim.*.bin")))
    assert len(files) >= SNAP_MIN, (
        f"too few snapshots for {basename}: {len(files)} < {SNAP_MIN} "
        f"(run did not reach tlim={TLIM} ns)"
    )
    return files


def _density_and_growth(files):
    """Per-snapshot (times, rho_max, a_outer) reduction of a coupled run."""
    times, rho_max, a_outer = [], [], []
    zn = None
    for fp in files:
        d = bin_convert.read_binary_as_athdf(fp)
        if zn is None:
            zn = np.asarray(d["x3v"], dtype=float) / b1.LZ
        dens = np.asarray(d["dens"], dtype=float)
        r_out = b1._interfaces(d)[1]
        times.append(float(d["Time"]))
        rho_max.append(float(np.max(dens)))
        a_outer.append(b1._mode_amp(r_out, zn, b1.PERT_MODE))
    return np.array(times), np.array(rho_max), np.array(a_outer)


def _print_table(tag, times, rho_max, a_outer):
    """Print the reduction so the run's evidence survives in the pytest log."""
    print(f"[{tag}] t [ns]   max rho [code]   a_outer [mm]")
    for t, rh, a in zip(times, rho_max, a_outer):
        print(f"[{tag}] {t:7.2f}   {rh:14.6e}   {a:12.6e}")


def _assert_bounded(tag, times, rho_max):
    """The #192 gate: peak density stays finite AND physically bounded to tlim."""
    assert np.all(np.isfinite(rho_max)), f"[{tag}] non-finite density"
    i_bad = int(np.argmax(rho_max))
    assert rho_max[i_bad] < RHO_MAX_CODE, (
        f"[{tag}] RUNAWAY COMPRESSION: max rho {rho_max[i_bad]:.3e} code units "
        f"(= {2.7 * rho_max[i_bad]:.3e} g/cc) at t={times[i_bad]:.1f} ns exceeds the "
        f"physical bound {RHO_MAX_CODE:.1e} (solid Al = 1.0; issue #192)"
    )


def _max_divb(basename):
    """Max |div(B)| (cylindrical FV) over the run, from the maglif
    {basename}.user.hst (col 6).

    The #192 resb div(B) write-back runaway surfaced here first: max|div B|
    climbed 1e-11 -> 0.16 on the refined grid and LED the density into NaN.  With
    the phi-uniform-delta write-back the solenoidal constraint stays at machine
    precision (~1e-13), so this is the direct integration-level gate on the fix
    (the unit test resb_divb_couple_test pins the per-cell invariant).
    """
    hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{basename}.user.hst")
    vals = []
    with open(hst) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) >= 6:
                vals.append(abs(float(cols[5])))  # col 6 (1-indexed) = maxdivb
    assert vals, f"no div(B) history rows in {hst}"
    return max(vals)


def test_verify_maglif_b1_coupled_radon_conv_gpu():
    """Radiation-off #192 stack: bounded + finite + div(B)-clean through 70 ns on all
    three grids; seeded mode grows; grid convergence BINDING on refined<->2x (#195)."""
    if not b1._build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        # --- (1) STABILITY at paper resolution (288x4x128): the #192 COMPRESSION-runaway
        # red->green gate (rho ran to ~1e9 here before the fix).  Paper-288 UNDER-RESOLVES
        # the implosion (rho ~2.89 vs converged ~16; #195), so it is NOT in the binding
        # convergence pair -- kept as the compression-runaway anchor; reported below.
        files = _run_coupled("b1rconv")
        times, rho_max, a_outer = _density_and_growth(files)
        _print_table("b1rconv 288x4x128", times, rho_max, a_outer)
        _assert_bounded("b1rconv 288x4x128", times, rho_max)
        divb = _max_divb("b1rconv")
        print(f"[b1rconv 288x4x128] max|div B| = {divb:.3e} (gate {DIVB_MAX:.0e}; #192)")
        assert divb < DIVB_MAX, (
            f"[b1rconv 288x4x128] div(B) CONSTRAINT VIOLATED: "
            f"max|div B| {divb:.3e} > {DIVB_MAX:.1e} "
            f"(the resb write-back must keep div(B) at machine precision; issue #192)"
        )

        # The stack must still GROW the seeded mode -- a "fix" that stabilizes by killing
        # the
        # MRT feed is not a fix.  Radiation-off this is a WEAK floor (see
        # GROWTH_MIN_RADON), not
        # an experiment-magnitude growth check.
        assert np.all(np.isfinite(a_outer)), \
            "[b1rconv 288x4x128] non-finite seeded-mode amplitude"
        growth = float(a_outer.max() / a_outer[0])
        assert growth > GROWTH_MIN_RADON, (
            f"radiation-off stack no longer grows the seeded mode: "
            f"peak/seed {growth:.2f}x "
            f"< {GROWTH_MIN_RADON}x (over-damped by the stabilization?)"
        )

        # --- (2) The refined leg (432x4x192): the #192 DIV(B)-runaway red->green
        # gate -- this is where div B ran 1e-11 -> 0.16 -> NaN before the fix.  Bounded +
        # div(B)-clean.  COARSE leg of the binding refined<->2x convergence pair (#195).
        rfiles = _run_coupled("b1rconv_ref", extra_args=REF_MESH)
        rtimes, rrho_max, ra_outer = _density_and_growth(rfiles)
        _print_table("b1rconv_ref 432x4x192", rtimes, rrho_max, ra_outer)
        _assert_bounded("b1rconv_ref 432x4x192", rtimes, rrho_max)
        rdivb = _max_divb("b1rconv_ref")
        print(f"[b1rconv_ref 432x4x192] max|div B| = {rdivb:.3e} "
              f"(gate {DIVB_MAX:.0e}; #192)")
        assert rdivb < DIVB_MAX, (
            f"[b1rconv_ref 432x4x192] div(B) CONSTRAINT VIOLATED: "
            f"max|div B| {rdivb:.3e} > {DIVB_MAX:.1e} "
            f"-- the refined-grid resb div(B) runaway (#192) has returned"
        )

        # --- (3) The 2x leg (576x4x256, 16 blocks): bounded + div(B)-clean.  FINE
        # (best-estimate) leg of the binding refined<->2x convergence pair (#195).
        xfiles = _run_coupled("b1rconv_2x", extra_args=TWOX_MESH)
        xtimes, xrho_max, xa_outer = _density_and_growth(xfiles)
        _print_table("b1rconv_2x 576x4x256", xtimes, xrho_max, xa_outer)
        _assert_bounded("b1rconv_2x 576x4x256", xtimes, xrho_max)
        xdivb = _max_divb("b1rconv_2x")
        print(f"[b1rconv_2x 576x4x256] max|div B| = {xdivb:.3e} "
              f"(gate {DIVB_MAX:.0e}; #192)")
        assert xdivb < DIVB_MAX, (
            f"[b1rconv_2x 576x4x256] div(B) CONSTRAINT VIOLATED: "
            f"max|div B| {xdivb:.3e} > {DIVB_MAX:.1e} "
            f"-- the resb div(B) runaway (#192) has returned at 2x resolution"
        )

        # --- (4) GRID CONVERGENCE: now BINDING on refined<->2x (#195 bracket).  With the
        # runaway fixed the implosion converges between 432 and 576 (rho_max within
        # CONV_RHO_RTOL) -- this closes #192's last open AC.  The shared verdict logic is
        # unit-tested on CPU (test_unit_maglif_grid_convergence_cpu).
        def _peak(t_, rho_, amp_):
            i = int(np.argmax(amp_))
            return {"rho_max": float(np.max(rho_)),
                    "amp_peak": float(amp_[i]), "t_peak_ns": float(t_[i])}

        ref_leg = _peak(rtimes, rrho_max, ra_outer)
        twox_leg = _peak(xtimes, xrho_max, xa_outer)
        verdict = grid_convergence_verdict(coarse=ref_leg, fine=twox_leg,
                                       rho_rtol=RADON_CONV_RHO_RTOL,
                                       amp_rtol=RADON_CONV_AMP_RTOL,
                                       tpk_tol=RADON_CONV_TPEAK_TOL)
        print(
            f"[CONVERGENCE #199 BINDING] refined<->2x: rho_max "
            f"{ref_leg['rho_max']:.3f} vs {twox_leg['rho_max']:.3f} "
            f"(dev {verdict['rho_dev']:.4f}/{RADON_CONV_RHO_RTOL}); seeded-mode peak "
            f"{ref_leg['amp_peak']:.4f} mm @ t={ref_leg['t_peak_ns']:.1f} ns vs "
            f"{twox_leg['amp_peak']:.4f} mm @ t={twox_leg['t_peak_ns']:.1f} ns "
            f"(amp dev {verdict['amp_dev']:.3f}/{RADON_CONV_AMP_RTOL}; "
            f"timing dev {verdict['tpk_dev']:.1f}/{RADON_CONV_TPEAK_TOL} ns)"
        )
        assert verdict["converged"], (
            "[CONVERGENCE #199] refined(432)<->2x(576) NOT grid-converged: "
            + "; ".join(verdict["failures"])
            + " -- #192's grid-convergence AC has regressed"
        )

        # REPORTED: paper-288 UNDER-RESOLVES the implosion (the #195 finding that excludes
        # it from the binding pair; kept above as the compression-runaway anchor).  NOT a
        # numerical defect -- div(B) clean and bounded on all three legs (gated above).
        paper_leg = _peak(times, rho_max, a_outer)
        paper_v = grid_convergence_verdict(coarse=paper_leg, fine=ref_leg,
                                       rho_rtol=RADON_CONV_RHO_RTOL,
                                       amp_rtol=RADON_CONV_AMP_RTOL,
                                       tpk_tol=RADON_CONV_TPEAK_TOL)
        print(
            f"[CONVERGENCE #199 REPORTED] paper-288 UNDER-RESOLVED vs refined: "
            f"rho_max {paper_leg['rho_max']:.2f} vs {ref_leg['rho_max']:.2f} "
            f"(rho dev {paper_v['rho_dev']:.2f}, amp dev {paper_v['amp_dev']:.2f}, "
            f"timing dev {paper_v['tpk_dev']:.1f} ns); excluded from the binding pair"
        )
    finally:
        for pat in ("b1rconv.*", "b1rconv_ref.*", "b1rconv_2x.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        for bn in ("b1rconv", "b1rconv_ref", "b1rconv_2x"):
            hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{bn}.user.hst")
            if os.path.exists(hst):
                os.remove(hst)
