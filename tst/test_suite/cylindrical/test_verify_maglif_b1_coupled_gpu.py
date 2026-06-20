"""
Coupled-stack DENSITY-STABILITY + div(B)-PRESERVATION gate for the faithful B1 run (issue
#192 [P8]).
(Grid convergence is now BINDING on the refined<->2x pair (#195 bracket): with the
runaway fixed the refined 432x4x192 and 2x 576x4x256 legs converge to 0.11% on peak
density (16.106 vs 16.089), so the gate asserts that pair; the paper 288x4x128 leg
UNDER-RESOLVES the implosion (rho~2.89), reported alongside but not in the binding pair.)

#192 was a paper-resolution (288x4x128) RUNAWAY COMPRESSION in the operator-split
#parabolic
block: rho ran to ~1e9 code units (orders above solid Al) before stagnation, all values
finite,
so no finiteness gate tripped.  THREE stacked defects drove it, all now fixed on this
branch:
  * resb energy-pump write-back (bphi->b0), commit 53fd9f11;
  * ox1 ghost-fed mass reservoir, gated behind ``problem/ox1_inflow_guard`` (commit
  01abd6cf),
    recorded in ADR-0015;
  * resb div(B) write-back: CoupleResbBphiToB0 overwrote b0.x2f per phi-slice, bypassing
  constrained
    transport and seeding a div(B) constraint violation that compounded to a NaN runaway
    on the
    1.5x-refined grid (the LEADING indicator there: div B 1e-11 -> 0.16 -> NaN, not a
    symptom of
    compression).  Fixed by applying a phi-UNIFORM B_phi increment so the write cannot
    change the
    discrete (1/r) dB_phi/dphi (ADR-0015; unit test resb_divb_couple_test).

This test is the TDD red->green anchor for #192: it asserts what the runaway violated -- a
PHYSICALLY BOUNDED, FINITE peak density AND a clean solenoidal constraint (max|div B| at
machine
precision) through the 70 ns experiment window, on ALL THREE grids (paper 288, refined
432, 2x 576).  The seeded mode must still grow (not be over-damped).  Grid convergence of
the implosion is now BINDING on the refined<->2x pair (#195 bracket) -- see the
convergence block below.

RADIATION IS OFF in this gate (``mhd/fld_operator_split=false``,
``mhd/mrad_coupling=false``), and
deliberately so.  The two #192 defects live in the operator-split parabolic block (resb +
acond);
the bisect isolated the runaway to resb ALONE (-> ~1e9), so the radiation-OFF stack
(resb + acond + guard) faithfully reproduces the runaway regime while staying NaN-clean.
The FULL
coupled stack (with grey FLD + mrad) cannot yet run NaN-clean: the grey FLD
spatial-diffusion
super-step overshoots ``erad`` negative in low-opacity cells and goes NaN at t~33 ns
(paper) /
t~18 ns (refined) -- a SEPARATE, pre-existing defect tracked as #194.  An earlier coupled
"green"
was poisoned by that NaN.  Only the COUPLED figures "75x / 1.58 mm / ~14x growth" were
read after NaN
onset and are INVALID (see #194).  rho=2.89 is NOT one of them: it is the VALID
radiation-OFF
paper-resolution density (resbguard/reduced.json; r_liner 3.146 matches ADR-0015
verbatim), reproduced
by leg-1 of this test (rho_max 2.892 at 70 ns).  NOTE rho=2.89 is the PAPER-RES value and
is UNDER-RESOLVED: the refined (432) and 2x (576) legs converge to rho~16.1 (a sharper,
better-resolved stagnation; #195 bracket), so the gate binds convergence on that pair.
Re-enabling FLD+mrad in this gate is gated on #194.

Measured radiation-OFF behaviour (this fix, full 70 ns window): paper res is WEAK and
monotone --
rho_max 1.0 -> 2.89, seeded mode grows 1.92x (a_outer 0.0208 -> 0.0400 mm, still rising at
t=70), div(B)
~4e-16.  The refined (432) and 2x (576) legs are div(B)-clean (~2e-13 / ~4e-13) and
bounded too, but implode HARDER (rho -> 16.106 / 16.089; sharp stagnation onset ~t=57/54)
with seeded-mode peaks 0.0275 mm @ t=57 / 0.0255 mm @ t=54.  Paper-288 UNDER-RESOLVES the
implosion (rho 2.89), but refined and 2x AGREE to 0.11% on peak density -- the #195
bracket pair, now the gate's binding pair.  NONE are the contaminated coupled ~14x.
GROWTH_MIN_NORAD
is an honest floor on the paper-res signal (see below), not borrowed from the stronger
operators-OFF
baseline; its job is only to catch a "fix" that stabilizes by killing the MRT feed, not a
growth rate.

The density bound is deliberately GENEROUS: RHO_MAX_CODE = 1e3 code units = 2.7 kg/cc,
two-plus
orders above any credible Z-liner stagnation compression of solid Al in this window and
many orders
below the observed runaway -- it can only discriminate "physical implosion" from
"numerical
collapse", never tune physics.

The operators-OFF faithful baseline (test_verify_maglif_b1_sinars_gpu) is untouched by
this gate and
must stay byte-identical under any #192 fix (the fix is gated to the operator-split path).
The
quantitative Sinars-curve verdict stays REPORTED there (binding=False) -- flipping it to a
binding
assert is #120's call, gated on the full coupled stack (i.e. on #194).

GPU-only (``_gpu`` suffix): built+run with CUDA; auto-collected by ``run_test_suite.py
--gpu``
(excluded from CPU CI).  Heavy GPU CI harness is #123/[C6].
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
    CONV_AMP_RTOL, CONV_RHO_RTOL, CONV_TPEAK_TOL, grid_convergence_verdict,
)

# Absolute table path: the per-problem build dir runs with cwd build_pgen/maglif/src, so
# the deck's repo-relative eos_table must be overridden (same pattern as the operators
# attribution test).
table_file = os.path.join(
    testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
)

# The radiation-OFF #192 stability stack: Strang-wrapped parabolic block (ADR-0009)
# carrying
# the two operators that hosted #192's defects -- resb (driven b0.x2f) + acond (EOS-aware)
# --
# with the deck's committed coefficients (resb_couple_b0, operators_si_calibrate) left
# untouched.
# Grey FLD + mrad are OFF: they cannot run NaN-clean yet (#194 -- erad overshoots negative
# in
# low-opacity cells), and an earlier coupled "green" was contaminated by that NaN.  resb
# ALONE
# reproduces the runaway (bisect), so this stack is the faithful, NaN-clean gate on the
# fix;
# FLD+mrad re-enable is gated on #194.
COUPLED_ARGS = [
    "mhd/strang_split=true",
    "mhd/resb_operator_split=true",
    "mhd/acond_operator_split=true",
    "mhd/fld_operator_split=false",   # radiation OFF -- gated on #194 (FLD erad -> NaN)
    "mhd/mrad_coupling=false",        # radiation OFF -- gated on #194
    # #192: pin the open outer boundary against ghost-fed inflow.  The zero-gradient
    # ghost copy is an unbounded mass reservoir once the resb-diffused drive field
    # raises an inward wind in the vacuum gap (mass x77,000 by 70 ns at paper res);
    # the guard swaps inflowing columns to a static vacuum ghost state (ADR-0015).
    "problem/ox1_inflow_guard=true",
]

TLIM = 70.0          # ns; the Sinars experiment window (last committed datum 69.9 ns)
SNAP_MIN = 20        # >= 20 of the ~24 expected 3-ns snapshots (run reached tlim)
RHO_MAX_CODE = 1.0e3  # bounded-density gate, code units (solid Al = 1.0); see docstring

# Honest floor on the radiation-OFF seeded-mode growth (peak a_outer / seed).
# Radiation-off the
# implosion is weak and monotone: paper-res measured ~1.4x and still rising at 57 ns
# (probe
# athenakdev:~/p192/runs/radcheck), far below the contaminated coupled run's ~14x.  This
# threshold
# only guards against a "fix" that over-damps and kills the MRT feed -- it is NOT borrowed
# from the
# stronger operators-OFF b1 baseline (b1.GROWTH_MIN) and does not certify
# experiment-magnitude
# growth (that needs the full coupled stack, gated on #194).
GROWTH_MIN_NORAD = 1.20

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
    """Run the faithful B1 deck with the radiation-OFF #192 stack; return snapshot paths.

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
    assert ok, f"radiation-off #192 B1 GPU run failed (basename={basename})"
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


def test_verify_maglif_b1_coupled_gpu():
    """Radiation-off #192 stack: bounded + finite + div(B)-clean through 70 ns on all
    three grids; seeded mode grows; grid convergence BINDING on refined<->2x (#195)."""
    if not b1._build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        # --- (1) STABILITY at paper resolution (288x4x128): the #192 COMPRESSION-runaway
        # red->green gate (rho ran to ~1e9 here before the fix).  Paper-288 UNDER-RESOLVES
        # the implosion (rho ~2.89 vs converged ~16; #195), so it is NOT in the binding
        # convergence pair -- kept as the compression-runaway anchor; reported below.
        files = _run_coupled("b1cpl")
        times, rho_max, a_outer = _density_and_growth(files)
        _print_table("b1cpl 288x4x128", times, rho_max, a_outer)
        _assert_bounded("b1cpl 288x4x128", times, rho_max)
        divb = _max_divb("b1cpl")
        print(f"[b1cpl 288x4x128] max|div B| = {divb:.3e} (gate {DIVB_MAX:.0e}; #192)")
        assert divb < DIVB_MAX, (
            f"[b1cpl 288x4x128] div(B) CONSTRAINT VIOLATED: "
            f"max|div B| {divb:.3e} > {DIVB_MAX:.1e} "
            f"(the resb write-back must keep div(B) at machine precision; issue #192)"
        )

        # The stack must still GROW the seeded mode -- a "fix" that stabilizes by killing
        # the
        # MRT feed is not a fix.  Radiation-off this is a WEAK floor (see
        # GROWTH_MIN_NORAD), not
        # an experiment-magnitude growth check.
        assert np.all(np.isfinite(a_outer)), \
            "[b1cpl 288x4x128] non-finite seeded-mode amplitude"
        growth = float(a_outer.max() / a_outer[0])
        assert growth > GROWTH_MIN_NORAD, (
            f"radiation-off stack no longer grows the seeded mode: "
            f"peak/seed {growth:.2f}x "
            f"< {GROWTH_MIN_NORAD}x (over-damped by the stabilization?)"
        )

        # --- (2) The refined leg (432x4x192): the #192 DIV(B)-runaway red->green
        # gate -- this is where div B ran 1e-11 -> 0.16 -> NaN before the fix.  Bounded +
        # div(B)-clean.  COARSE leg of the binding refined<->2x convergence pair (#195).
        rfiles = _run_coupled("b1cpl_ref", extra_args=REF_MESH)
        rtimes, rrho_max, ra_outer = _density_and_growth(rfiles)
        _print_table("b1cpl_ref 432x4x192", rtimes, rrho_max, ra_outer)
        _assert_bounded("b1cpl_ref 432x4x192", rtimes, rrho_max)
        rdivb = _max_divb("b1cpl_ref")
        print(f"[b1cpl_ref 432x4x192] max|div B| = {rdivb:.3e} "
              f"(gate {DIVB_MAX:.0e}; #192)")
        assert rdivb < DIVB_MAX, (
            f"[b1cpl_ref 432x4x192] div(B) CONSTRAINT VIOLATED: "
            f"max|div B| {rdivb:.3e} > {DIVB_MAX:.1e} "
            f"-- the refined-grid resb div(B) runaway (#192) has returned"
        )

        # --- (3) The 2x leg (576x4x256, 16 blocks): bounded + div(B)-clean.  FINE
        # (best-estimate) leg of the binding refined<->2x convergence pair (#195).
        xfiles = _run_coupled("b1cpl_2x", extra_args=TWOX_MESH)
        xtimes, xrho_max, xa_outer = _density_and_growth(xfiles)
        _print_table("b1cpl_2x 576x4x256", xtimes, xrho_max, xa_outer)
        _assert_bounded("b1cpl_2x 576x4x256", xtimes, xrho_max)
        xdivb = _max_divb("b1cpl_2x")
        print(f"[b1cpl_2x 576x4x256] max|div B| = {xdivb:.3e} "
              f"(gate {DIVB_MAX:.0e}; #192)")
        assert xdivb < DIVB_MAX, (
            f"[b1cpl_2x 576x4x256] div(B) CONSTRAINT VIOLATED: "
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
        verdict = grid_convergence_verdict(coarse=ref_leg, fine=twox_leg)
        print(
            f"[CONVERGENCE #195 BINDING] refined<->2x: rho_max "
            f"{ref_leg['rho_max']:.3f} vs {twox_leg['rho_max']:.3f} "
            f"(dev {verdict['rho_dev']:.4f}/{CONV_RHO_RTOL}); seeded-mode peak "
            f"{ref_leg['amp_peak']:.4f} mm @ t={ref_leg['t_peak_ns']:.1f} ns vs "
            f"{twox_leg['amp_peak']:.4f} mm @ t={twox_leg['t_peak_ns']:.1f} ns "
            f"(amp dev {verdict['amp_dev']:.3f}/{CONV_AMP_RTOL}; "
            f"timing dev {verdict['tpk_dev']:.1f}/{CONV_TPEAK_TOL} ns)"
        )
        assert verdict["converged"], (
            "[CONVERGENCE #195] refined(432)<->2x(576) NOT grid-converged: "
            + "; ".join(verdict["failures"])
            + " -- #192's grid-convergence AC has regressed"
        )

        # REPORTED: paper-288 UNDER-RESOLVES the implosion (the #195 finding that excludes
        # it from the binding pair; kept above as the compression-runaway anchor).  NOT a
        # numerical defect -- div(B) clean and bounded on all three legs (gated above).
        paper_leg = _peak(times, rho_max, a_outer)
        paper_v = grid_convergence_verdict(coarse=paper_leg, fine=ref_leg)
        print(
            f"[CONVERGENCE #195 REPORTED] paper-288 UNDER-RESOLVED vs refined: "
            f"rho_max {paper_leg['rho_max']:.2f} vs {ref_leg['rho_max']:.2f} "
            f"(rho dev {paper_v['rho_dev']:.2f}, amp dev {paper_v['amp_dev']:.2f}, "
            f"timing dev {paper_v['tpk_dev']:.1f} ns); excluded from the binding pair"
        )
    finally:
        for pat in ("b1cpl.*", "b1cpl_ref.*", "b1cpl_2x.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        for bn in ("b1cpl", "b1cpl_ref", "b1cpl_2x"):
            hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{bn}.user.hst")
            if os.path.exists(hst):
                os.remove(hst)
