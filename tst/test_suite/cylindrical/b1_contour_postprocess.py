"""CPU post-processing: the paper's rho = 0.1 g/cc contour observable (#222).

Reduces SAVED faithful-B1 snapshots (``{basename}.prim.*.bin``) to the Ellison
SS3.1/Fig.-3 contour spike-bubble amplitude ``a_contour(t)`` -- per-z outermost
radial crossing of the absolute rho = 0.1 g/cc density contour, FULL max - min
over z -- next to the existing radiograph observables ``a_sb(t)``/``a_sb_raw(t)``,
plus the paper's threshold-insensitivity sweep (0.05 / 0.1 / 0.2 g/cc).  No GPU,
no build, no re-run: point it at a directory holding the saved 432/864-leg
snapshots (#120) on any CPU box.

Usage (from ``tst/``, on the machine holding the snapshots):

    python3 -m test_suite.cylindrical.b1_contour_postprocess \\
        /path/to/snapshots --basename maglif_b1_sinars [--json out.json]

PURE core: ``reduce_snapshot_dicts`` takes already-read snapshot dicts, so the
reduction is unit-testable offline (test_unit_b1_contour_observable_cpu) and
shares its definitions with the GPU harness (test_verify_maglif_b1_sinars_gpu).
"""

import argparse
import glob
import json
import os

import numpy as np

import test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu as b1

THRESHOLDS_GCC = b1.CONTOUR_THRESHOLDS_GCC


def reduce_snapshot_dicts(dicts, thresholds_gcc=THRESHOLDS_GCC):
    """Reduce snapshot dicts to the #222 observables, sorted by snapshot time.

    Returns ``{"times", "a_contour" ({rho_gcc: [mm]}), "a_sb", "a_sb_raw"}``:
    ``a_contour`` at each threshold (the paper's value is at
    ``b1.RHO_CONTOUR_GCC``), ``a_sb``/``a_sb_raw`` recomputed with the committed
    reductions so the two observables come from the very same snapshots.
    """
    dicts = sorted(dicts, key=lambda d: float(d["Time"]))
    times, a_sb, a_sb_raw = [], [], []
    a_contour = {thr: [] for thr in thresholds_gcc}
    for d in dicts:
        times.append(float(d["Time"]))
        for thr in thresholds_gcc:
            a_contour[thr].append(b1._contour_amp(b1._contour_edge(d, thr)))
        x, sigma = b1._synthetic_radiograph(d)
        edge = b1._limb_edge(x, sigma)
        a_sb.append(b1._spike_bubble_amp(edge, k_max=b1.K_MAX_BAND))
        a_sb_raw.append(b1._spike_bubble_amp(edge))
    return {
        "times": times,
        "a_contour": a_contour,
        "a_sb": a_sb,
        "a_sb_raw": a_sb_raw,
    }


def reduce_files(files, thresholds_gcc=THRESHOLDS_GCC):
    """Read saved ``.prim.*.bin`` snapshots and reduce them (CPU only)."""
    dicts = [b1.bin_convert.read_binary_as_athdf(fp) for fp in files]
    return reduce_snapshot_dicts(dicts, thresholds_gcc)


def _format_table(out, thresholds_gcc=THRESHOLDS_GCC):
    """Human-readable table of the reduced observables (one row per snapshot)."""
    thr_heads = "  ".join(f"a_ct@{thr:g}".rjust(10) for thr in thresholds_gcc)
    lines = [f"{'t [ns]':>8}  {thr_heads}  {'a_sb':>10}  {'a_sb_raw':>10}"]
    for i, t in enumerate(out["times"]):
        thr_vals = "  ".join(
            f"{out['a_contour'][thr][i]:10.4f}" for thr in thresholds_gcc
        )
        lines.append(
            f"{t:8.1f}  {thr_vals}  {out['a_sb'][i]:10.4f}  "
            f"{out['a_sb_raw'][i]:10.4f}"
        )
    return "\n".join(lines)


def main(argv=None):
    """CLI: reduce a directory of saved snapshots and print/emit the observables."""
    ap = argparse.ArgumentParser(
        description="a_contour(t) (#222) from saved faithful-B1 snapshots"
    )
    ap.add_argument("snapshot_dir", help="directory holding {basename}.prim.*.bin")
    ap.add_argument("--basename", default="maglif_b1_sinars",
                    help="snapshot basename (default: maglif_b1_sinars)")
    ap.add_argument("--json", default=None,
                    help="also write the reduced observables to this JSON path")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(
        os.path.join(args.snapshot_dir, f"{args.basename}.prim.*.bin")
    ))
    if not files:
        ap.error(f"no {args.basename}.prim.*.bin under {args.snapshot_dir}")
    out = reduce_files(files)

    print(f"[#222] {len(files)} snapshots ({args.basename}), amplitudes in mm; "
          f"a_ct = FULL max-min of the rho-contour, a_sb = HALF limb excursion")
    print(_format_table(out))
    ac = np.asarray(out["a_contour"][b1.RHO_CONTOUR_GCC], dtype=float)
    if np.any(np.isfinite(ac)):
        i_pk = int(np.nanargmax(ac))
        ref = ac[i_pk]
        spread = max(
            abs(out["a_contour"][thr][i_pk] - ref) for thr in THRESHOLDS_GCC
        ) / ref if ref > 0.0 else float("nan")
        print(f"[#222] peak a_contour {ref:.4f} mm at t={out['times'][i_pk]:.1f} ns; "
              f"threshold spread there {spread:.1%} (paper claims <10%)")
    if args.json:
        payload = {
            "basename": args.basename,
            "times_ns": out["times"],
            "a_contour_mm": {str(k): v for k, v in out["a_contour"].items()},
            "a_sb_mm": out["a_sb"],
            "a_sb_raw_mm": out["a_sb_raw"],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[#222] wrote {args.json}")


if __name__ == "__main__":
    main()
