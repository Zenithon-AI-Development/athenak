"""
Coupled-stack STABILITY + GRID-CONVERGENCE gate for the faithful B1 run (issue #192 [P8]).

The P7 chain (#181-#184) wired and SI-calibrated the reference-model-set operators
(resb coupled to the live driven b0.x2f, gray-FLD erad sourced from the gas, EOS-aware
acond/mrad, ADR-0014 coefficients).  Enabling the FULL coupled stack on the faithful B1
deck (inputs/maglif_b1_sinars.athinput) drives experiment-magnitude MRT growth -- but the
2026-06-03 A100 evidence (issue #192) is that the run is NOT numerically converged:

  * reduced grid (144x4x64):  bounded growth, seeded-mode peak 1.58 mm @ 84 ns (75x seed);
  * paper grid  (288x4x128):  RUNAWAY compression -- rho -> 3.6e14 code units (14 orders
    above solid Al) by t ~ 60 ns, all values finite (no NaN), so no finiteness gate trips.

This test is the TDD red->green anchor for #192: it asserts what the runaway violates --
a PHYSICALLY BOUNDED peak density on the coupled paper-resolution run through the
experiment window -- and then, once stable, that the seeded-mode amplitude(t) no longer
flips between bounded growth and runaway under grid refinement (a 1.5x-refined companion
run must stay bounded and reproduce the paper-res growth curve within tolerance).

The density bound is deliberately GENEROUS: RHO_MAX_CODE = 1e3 code units = 2.7 kg/cc,
two-plus orders above any credible Z-liner shock/stagnation compression of solid Al in
this pre-stagnation window, and eleven orders below the observed runaway -- it can only
discriminate "physical implosion" from "numerical collapse", never tune physics.

The operators-OFF faithful baseline (test_verify_maglif_b1_sinars_gpu) is untouched by
this gate and must stay byte-identical under any #192 fix (the fix is gated to the
coupled path).  The quantitative Sinars-curve verdict stays REPORTED here (binding=False)
-- flipping it to a binding assert is #120's call, gated on this test going green.

GPU-only (``_gpu`` suffix): built+run with CUDA; auto-collected by ``run_test_suite.py
--gpu`` (excluded from CPU CI).  Heavy GPU CI harness is #123/[C6].
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

# Absolute table path: the per-problem build dir runs with cwd build_pgen/maglif/src, so
# the deck's repo-relative eos_table must be overridden (same pattern as the operators
# attribution test).
table_file = os.path.join(
    testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
)

# The full P7-calibrated coupled stack, exactly the #192 configuration: Strang-wrapped
# parabolic block (ADR-0009) + all four reference-model-set operators, with the deck's
# committed defaults (resb_couple_b0, fld_source_erad_from_gas, acond/mrad EOS-aware,
# operators_si_calibrate) left untouched.
COUPLED_ARGS = [
    "mhd/strang_split=true",
    "mhd/resb_operator_split=true",
    "mhd/acond_operator_split=true",
    "mhd/fld_operator_split=true",
    "mhd/mrad_coupling=true",
]

TLIM = 70.0          # ns; the Sinars experiment window (last committed datum 69.9 ns)
SNAP_MIN = 20        # >= 20 of the ~24 expected 3-ns snapshots (run reached tlim)
RHO_MAX_CODE = 1.0e3  # bounded-density gate, code units (solid Al = 1.0); see docstring

# Grid-convergence companion: 1.5x refinement in the meshed (r, z) plane, same physics.
# 432x4x192 keeps the committed 144x4x64 MeshBlock tiling (3x1x3 = 9 blocks).
REF_MESH = [
    "mesh/nx1=432", "mesh/nx3=192",
    "meshblock/nx1=144", "meshblock/nx3=64",
]
# Convergence tolerance on the seeded-mode growth curve between the two refinements:
# the runaway-vs-bounded flip this gate exists to kill is ORDERS of magnitude, so a
# 30% band on the peak amplitude plus a 2-output-frame (6 ns) band on its timing is an
# honest "same physical answer" criterion for a 1.5x PLM refinement near nonlinear
# saturation without tuning to either run.
CONV_AMP_RTOL = 0.30
CONV_TPEAK_TOL = 6.0  # ns


def _run_coupled(basename, extra_args=None):
    """Run the faithful B1 deck with the full coupled stack ON; return snapshot paths."""
    args = [
        f"mhd/eos_table={table_file}",
        f"problem/current_file={trace_file}",
        f"job/basename={basename}",
        f"time/tlim={TLIM}",
    ] + COUPLED_ARGS
    if extra_args:
        args += extra_args
    ok = testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS, args=args)
    assert ok, f"coupled-stack B1 GPU run failed (basename={basename})"
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


def test_verify_maglif_b1_coupled_gpu():
    """Coupled stack at paper resolution: bounded through 70 ns + grid-converged."""
    if not b1._build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        # --- (1) STABILITY at paper resolution (288x4x128): the #192 red->green gate.
        files = _run_coupled("b1cpl")
        times, rho_max, a_outer = _density_and_growth(files)
        _print_table("b1cpl 288x4x128", times, rho_max, a_outer)
        _assert_bounded("b1cpl 288x4x128", times, rho_max)

        # The coupled physics must still GROW the seeded mode (the whole point of P7:
        # the calibrated stack broke the O(1)-efolding ceiling) -- a "fix" that
        # stabilizes by killing the MRT feed is not a fix.
        growth = float(a_outer.max() / a_outer[0])
        assert growth > b1.GROWTH_MIN, (
            f"coupled stack no longer grows the seeded mode: peak/seed {growth:.2f}x "
            f"< {b1.GROWTH_MIN}x (over-damped by the stabilization?)"
        )

        # --- (2) GRID CONVERGENCE: a 1.5x refinement must give the same physical
        # answer -- bounded, and the same growth curve within tolerance (no
        # bounded-vs-runaway flip with resolution; issue #192 AC).
        rfiles = _run_coupled("b1cpl_ref", extra_args=REF_MESH)
        rtimes, rrho_max, ra_outer = _density_and_growth(rfiles)
        _print_table("b1cpl_ref 432x4x192", rtimes, rrho_max, ra_outer)
        _assert_bounded("b1cpl_ref 432x4x192", rtimes, rrho_max)

        i_pk, ir_pk = int(np.argmax(a_outer)), int(np.argmax(ra_outer))
        amp_dev = abs(ra_outer[ir_pk] - a_outer[i_pk]) / a_outer[i_pk]
        assert amp_dev <= CONV_AMP_RTOL, (
            f"seeded-mode peak amplitude not grid-converged: paper "
            f"{a_outer[i_pk]:.4f} mm vs refined {ra_outer[ir_pk]:.4f} mm "
            f"(rel dev {amp_dev:.2f} > {CONV_AMP_RTOL})"
        )
        assert abs(rtimes[ir_pk] - times[i_pk]) <= CONV_TPEAK_TOL, (
            f"seeded-mode peak TIMING not grid-converged: paper t={times[i_pk]:.1f} ns "
            f"vs refined t={rtimes[ir_pk]:.1f} ns (> {CONV_TPEAK_TOL} ns apart)"
        )
    finally:
        for pat in ("b1cpl.*", "b1cpl_ref.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
