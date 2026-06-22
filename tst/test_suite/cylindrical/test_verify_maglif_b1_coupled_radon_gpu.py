"""
RADIATION-ON coupled-stack STABILITY gate for the faithful B1 run (issue #194).

#194 is the grey-FLD positivity defect: with radiation ON (``mhd/fld_operator_split``
+ ``mhd/mrad_coupling``) the grey-FLD spatial-diffusion super-step overshoots ``erad``
NEGATIVE in low-opacity cells, then the radiation energy collapses
(``erad`` -> -6.1e178 -> NaN) at t~33 ns (paper 288x4x128) / t~18 ns (refined 432x4x192)
when the read floor is the OLD 1e-30 guard.  An earlier coupled "green" was poisoned by
exactly this NaN, so the radiation-ON stack had never run clean on GPU.

The fix (floor-the-read + PostSuperstepProject, commit 5b4c457e, already on main) clamps
the ``erad`` read into the FLD diffusion solve to ``mhd/fld_efloor`` (read-floor also caps
the implied flux ``|F| <= c*efloor``) and re-projects ``erad`` back onto the floor AFTER
the super-step, so a negative overshoot can no longer seed the runaway.  This test is the
TDD red->green anchor for that fix EXERCISED RADIATION-ON on GPU for the first time: it
flips FLD + matter-radiation coupling ON and asserts the radiation energy stays
NaN-clean and non-negative through the full 70 ns Sinars window.

The ``mhd/fld_efloor`` knob is the red/green selector:
  * RED   -- the DEFAULT 1e-30 == the old guard reproduces the NaN
             (``erad`` -> -6.1e178 -> NaN; onset t~33 ns paper / t~18 ns refined).
  * GREEN -- 1e-10 (~ a_rad*(0.1)^4, ~9 orders below physical ``erad`` ~O(0.1);
             read-floor caps ``|F| <= c*efloor = 1e-9``) holds positivity and the
             stack runs clean to tlim.

This gate is STABILITY ONLY: NaN-clean + ``erad`` >= 0 (the #194 core) + density bounded
+ ``div(B)`` clean through 70 ns, on BOTH the paper and refined legs.  It is NOT a
binding convergence verdict -- grid convergence of the radiation-ON stack is a separate
issue (#199).  The radiation-OFF #192/#195 BINDING convergence gate
(``test_verify_maglif_b1_coupled_gpu.py``) is untouched by this file and stays
byte-identical; this is a deliberately SEPARATE gate.

GPU-only (``_gpu`` suffix): built+run with CUDA; auto-collected by ``run_test_suite.py
--gpu`` (excluded from CPU CI).  Heavy GPU CI harness is #123/[C6].
"""

# Modules
import glob
import os
import shutil

import numpy as np
import test_suite.testutils as testutils

# Reuse the faithful-B1 committed deck paths + GPU build/run plumbing from the
# operators-OFF paper-resolution benchmark so the radiation-ON gate can never drift apart
# from the radiation-OFF gate on what "the faithful B1 deck" means.
import test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu as b1
from test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu import (  # noqa: F401
    GPU_FLAGS, bin_convert, bin_dir, build_dir, input_file, trace_file,
)

# Absolute table path: the per-problem build dir runs with cwd build_pgen/maglif/src, so
# the deck's repo-relative eos_table must be overridden (same pattern as the
# radiation-OFF gate).
table_file = os.path.join(
    testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
)

# The RADIATION-ON #194 stability stack: the same Strang-wrapped parabolic block as the
# radiation-OFF #192/#195 gate (resb + acond + ox1 guard) but with grey FLD + mrad
# FLIPPED ON, and the erad positivity floor set to the GREEN value.  fld_efloor at the
# DEFAULT 1e-30 reproduces the #194 NaN (the red); 1e-10 holds positivity (the green).
COUPLED_ARGS_RADON = [
    "mhd/strang_split=true",
    "mhd/resb_operator_split=true",
    "mhd/acond_operator_split=true",
    "mhd/fld_operator_split=true",     # radiation ON (#194)
    "mhd/mrad_coupling=true",          # radiation ON (#194)
    # erad positivity floor (#194).  DEFAULT 1e-30 == old guard reproduces the NaN
    # (the red).  1e-10 ~ a_rad*(0.1)^4, ~9 orders below physical erad ~O(0.1);
    # read-floor caps |F| <= c*efloor = 1e-9 (the green).
    "mhd/fld_efloor=1.0e-10",
    # #192: pin the open outer boundary against ghost-fed inflow (ADR-0015); carried
    # over from the radiation-OFF stack so the regime is identical apart from radiation.
    "problem/ox1_inflow_guard=true",
]

TLIM = 70.0           # ns; the Sinars experiment window (last committed datum 69.9 ns)
RHO_MAX_CODE = 1.0e3  # bounded-density gate, code units (solid Al = 1.0)
# max|div B| (cyl FV) gate; clean ~1e-13.  Same value as the radiation-OFF gate.
DIVB_MAX = 1.0e-6

# Refined leg: 1.5x refinement 432x4x192 (3x1x3 = 9 blocks), committed 144x4x64 tiling.
# Same mesh as the radiation-OFF gate's refined leg.  At the OLD 1e-30 guard the #194 NaN
# onset is EARLIER here (t~18 ns) than at paper res (t~33 ns).
REF_MESH = [
    "mesh/nx1=432", "mesh/nx3=192",
    "meshblock/nx1=144", "meshblock/nx3=64",
]


def _read_history(basename):
    """Parse the maglif {basename}.user.hst (radiation-ON: 10 whitespace columns).

    Columns (0-indexed): [0]=time [1]=dt [2]=mass [3]=etot [4]=massr
    [5]=maxdivb [6]=maxdens [7]=nmb [8]=ncells [9]=erad.  Everything the #194
    stability gate needs lives in the history, so no snapshots are read.  Skips
    blank lines and '#' comments; requires >= 10 columns per data row.  Returns
    numpy arrays (times, etot, maxdivb, maxdens, erad).
    """
    hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{basename}.user.hst")
    times, etot, maxdivb, maxdens, erad = [], [], [], [], []
    with open(hst) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 10:
                continue
            times.append(float(cols[0]))
            etot.append(float(cols[3]))
            maxdivb.append(abs(float(cols[5])))
            maxdens.append(float(cols[6]))
            erad.append(float(cols[9]))
    assert times, f"no radiation-ON history rows in {hst}"
    return (np.array(times), np.array(etot), np.array(maxdivb),
            np.array(maxdens), np.array(erad))


def _run_radon(basename, extra_args=None):
    """Run the faithful B1 deck with the RADIATION-ON #194 stack.

    Removes any stale ``{basename}.user.hst`` first: AthenaK history output
    APPENDS across runs, so a leftover file from a previous (possibly NaN) run
    would poison the parse.  A fresh run then writes a clean, single-run history.
    """
    hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{basename}.user.hst")
    if os.path.exists(hst):
        os.remove(hst)
    args = [
        f"mhd/eos_table={table_file}",
        f"problem/current_file={trace_file}",
        f"job/basename={basename}",
        f"time/tlim={TLIM}",
    ] + COUPLED_ARGS_RADON
    if extra_args:
        args += extra_args
    ok = testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS, args=args)
    assert ok, f"radiation-ON #194 B1 GPU run failed (basename={basename})"


def _print_table(tag, times, etot, erad, maxdens, maxdivb):
    """Print the history reduction so the run's evidence survives the pytest log."""
    print(f"[{tag}] t [ns]   etot [code]   erad [code]   maxdens [code]   maxdivb")
    for t, e, er, rh, db in zip(times, etot, erad, maxdens, maxdivb):
        print(f"[{tag}] {t:7.2f}   {e:12.5e}   {er:12.5e}   "
              f"{rh:13.6e}   {db:.3e}")


def _assert_stable(tag, times, etot, maxdivb, maxdens, erad):
    """The #194 STABILITY gate, applied to one leg: NaN-clean + erad >= 0 +
    bounded + div(B)-clean through tlim."""
    # reached tlim (one 3-ns snapshot of slack)
    assert times.size and float(times.max()) >= TLIM - 3.0, (
        f"[{tag}] run did not reach tlim: t_max "
        f"{float(times.max()) if times.size else float('nan'):.1f} < "
        f"{TLIM - 3.0:.1f} ns"
    )

    # total energy finite
    assert np.all(np.isfinite(etot)), f"[{tag}] non-finite total energy"

    # erad POSITIVITY -- the #194 core: NaN-clean AND non-negative.
    assert np.all(np.isfinite(erad)), (
        f"[{tag}] NON-FINITE radiation energy: erad went NaN -- the #194 "
        f"grey-FLD overshoot (erad -> -6.1e178 -> NaN) has returned; the "
        f"floor-the-read + PostSuperstepProject positivity floor must hold"
    )
    i_neg = int(np.argmin(erad))
    assert np.all(erad >= 0.0), (
        f"[{tag}] NEGATIVE radiation energy: erad {erad[i_neg]:.3e} at "
        f"t={times[i_neg]:.1f} ns -- the #194 grey-FLD super-step overshot "
        f"erad negative (precursor to the -6.1e178 -> NaN runaway); the "
        f"positivity floor (fld_efloor) must keep erad >= 0"
    )

    # density bounded
    assert np.all(np.isfinite(maxdens)), f"[{tag}] non-finite density"
    i_rho = int(np.argmax(maxdens))
    assert maxdens[i_rho] < RHO_MAX_CODE, (
        f"[{tag}] RUNAWAY COMPRESSION: max rho {maxdens[i_rho]:.3e} code "
        f"units at t={times[i_rho]:.1f} ns exceeds the physical bound "
        f"{RHO_MAX_CODE:.1e} (solid Al = 1.0)"
    )

    # div(B) clean
    divb = float(np.max(maxdivb))
    assert divb < DIVB_MAX, (
        f"[{tag}] div(B) CONSTRAINT VIOLATED: max|div B| {divb:.3e} > "
        f"{DIVB_MAX:.1e} (the resb write-back must keep div(B) at machine "
        f"precision through the radiation-ON run; #192/#194)"
    )


def test_verify_maglif_b1_coupled_radon_gpu():
    """Radiation-ON #194 stack: NaN-clean + erad >= 0 + bounded + div(B)-clean
    through 70 ns on both the paper and refined legs (STABILITY, not convergence
    -- convergence is #199)."""
    if not b1._build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        # --- leg 1: paper resolution (288x4x128).  At the OLD 1e-30 guard the
        # #194 NaN onset is t~33 ns here; with the positivity floor it runs clean.
        _run_radon("b1cpl_radon")
        times, etot, maxdivb, maxdens, erad = _read_history("b1cpl_radon")
        _print_table("b1cpl_radon 288x4x128", times, etot, erad, maxdens, maxdivb)
        _assert_stable("b1cpl_radon 288x4x128", times, etot, maxdivb,
                       maxdens, erad)

        # --- leg 2: refined (432x4x192).  At the OLD 1e-30 guard the #194 NaN
        # onset is EARLIER here (t~18 ns); with the positivity floor it runs clean.
        _run_radon("b1cpl_radon_ref", extra_args=REF_MESH)
        rtimes, retot, rmaxdivb, rmaxdens, rerad = _read_history("b1cpl_radon_ref")
        _print_table("b1cpl_radon_ref 432x4x192", rtimes, retot, rerad,
                     rmaxdens, rmaxdivb)
        _assert_stable("b1cpl_radon_ref 432x4x192", rtimes, retot, rmaxdivb,
                       rmaxdens, rerad)
    finally:
        for pat in ("b1cpl_radon.*", "b1cpl_radon_ref.*"):
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        for bn in ("b1cpl_radon", "b1cpl_radon_ref"):
            hst = os.path.join(testutils.pgen_run_dir("maglif"), f"{bn}.user.hst")
            if os.path.exists(hst):
                os.remove(hst)
