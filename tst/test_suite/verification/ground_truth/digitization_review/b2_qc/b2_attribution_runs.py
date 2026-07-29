"""
Discriminating-run driver for the #229 B2 Stage-3 growth-fraction attribution.

The scorecard recorded ``growth_fraction = 0.345`` against McBride's [0.05, 0.15]
band from the reduced CPU gate (64x4x48, tlim 20 -> deepest convergence x ~ 0.214).
Issue #229 requires the 3.5x miss decomposed among {seed amplitude dT, reduction
mismatch, timing convention, resolution, stale pre-#211 data} with at least one
discriminating run or measurement each.  This driver produces the run side:

  repro_dT100 : the exact gate (dT=100, 64/48, tlim 20)      -> staleness check
                (post-#211 code either reproduces ~0.345 or the number was stale)
  dT10/dT1000 : the Ellison delta-T scan around the nominal   -> seed amplitude
  res2x       : 128/96 at the same tlim                       -> resolution
                (compare gf at MATCHED convergence x, not at matched time)
  deep        : the gate resolution run extended in time      -> timing convention
                (measure gf INSIDE McBride's validity window x in [0.4, 0.95])

Reduction mismatch is a measurement, not a run: every case records the growth
fraction from raw extrema, from instrument-band-limited extrema (k <= 53 across
the 1.6 mm axial extent = the 15 um McBride radiograph resolution, #212-mirror),
and from a percentile (p95/p5) variant that bounds extreme-value inflation of
max/min versus McBride's dominant-bubble + spike-center-of-mass definitions.

Runs on athenakdev (CPU, OpenMP); results land in one JSON per case.  Committed
under the digitization-review tree as the provenance of the #229 attribution.

Usage (from a directory containing the built maglif binary):
  python3 b2_attribution_runs.py --binary .../build/src/athena \
      --repo .../src --workdir .../runs [--cases repro_dT100,dT10,...]
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HALF = 0.5      # half-max density level isolating the dense Be liner
LZ = 1.6        # axial extent [mm]
K_MAX_BAND = int(LZ * 1.0e3 / (2.0 * 15.0))   # McBride 15 um radiograph -> k <= 53

CASES = {
    # name: (dT, nx1, nx3, tlim, out_dt)
    "repro_dT100": (100.0, 64, 48, 20.0, 5.0),
    "dT10":        (10.0, 64, 48, 20.0, 5.0),
    "dT1000":      (1000.0, 64, 48, 20.0, 5.0),
    "res2x":       (100.0, 128, 96, 20.0, 5.0),
    "deep":        (100.0, 64, 48, 60.0, 2.5),
}


def _interfaces(d):
    """Per-z inner/outer half-max interfaces (mirror of the B2 CPU gate reduction)."""
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)            # (nz, nphi, nr)
    nz = dens.shape[0]
    r_out = np.full(nz, np.nan)
    r_mw = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)
        idx = np.where(p >= HALF)[0]
        if len(idx) == 0:
            continue
        w = np.where(p >= HALF, p, 0.0)
        r_mw[kz] = np.sum(w * r * r) / np.sum(w * r)
        hi = idx[-1]
        if hi == len(r) - 1:
            r_out[kz] = r[hi]
        else:
            r_out[kz] = r[hi] + (HALF - p[hi]) / (p[hi + 1] - p[hi]) * (r[hi + 1] - r[hi])
    return r_out, r_mw


def _band_limit(rz, k_max):
    r = np.asarray(rz, dtype=float)
    good = np.isfinite(r)
    if not np.any(good):
        return r
    if not np.all(good):
        idx = np.arange(r.size, dtype=float)
        r = np.interp(idx, idx[good], r[good], period=float(r.size))
    f = np.fft.rfft(r)
    f[int(k_max) + 1:] = 0.0
    return np.fft.irfft(f, n=r.size)


def _gf(spike, bubble, r0):
    den = r0 - bubble
    return float("nan") if den <= 0.0 else (spike - bubble) / den


def _reduce(files):
    """Snapshot series -> {t, x, gf_raw, gf_band, gf_pct, r_bubble, ...} lists."""
    out = {k: [] for k in ("t", "x", "gf_raw", "gf_band", "gf_pct",
                           "r_spike", "r_bubble", "r_liner")}
    r0 = None
    for fp in files:
        d = bin_convert.read_binary_as_athdf(fp)
        r_out, r_mw = _interfaces(d)
        good = np.isfinite(r_out)
        if not good.any():
            continue
        sp_raw, bu_raw = float(np.max(r_out[good])), float(np.min(r_out[good]))
        rb = _band_limit(r_out, K_MAX_BAND)
        sp_b, bu_b = float(np.max(rb)), float(np.min(rb))
        sp_p, bu_p = float(np.percentile(rb, 95)), float(np.percentile(rb, 5))
        if r0 is None:
            r0 = 0.5 * (sp_raw + bu_raw)   # interface starts exactly 1-D
        out["t"].append(float(d["Time"]))
        out["x"].append(1.0 - bu_b / r0)
        out["gf_raw"].append(_gf(sp_raw, bu_raw, r0))
        out["gf_band"].append(_gf(sp_b, bu_b, r0))
        out["gf_pct"].append(_gf(sp_p, bu_p, r0))
        out["r_spike"].append(sp_b)
        out["r_bubble"].append(bu_b)
        out["r_liner"].append(float(np.nanmean(r_mw)))
    out["r0"] = r0
    return out


def _run_case(name, spec, binary, repo, workdir):
    dT, nx1, nx3, tlim, out_dt = spec
    case_dir = os.path.join(workdir, name)
    os.makedirs(case_dir, exist_ok=True)
    inp = os.path.join(repo, "tst", "inputs", "maglif_b2_ellison.athinput")
    trace = os.path.join(repo, "tst", "inputs", "z2173_current.dat")
    cmd = [
        binary, "-i", inp,
        f"problem/current_file={trace}",
        f"problem/pert_dT={dT}",
        f"mesh/nx1={nx1}", f"mesh/nx3={nx3}",
        f"meshblock/nx1={nx1}", f"meshblock/nx3={nx3}",
        f"time/tlim={tlim}", f"output1/dt={out_dt}",
        f"job/basename={name}",
    ]
    log = os.path.join(case_dir, "run.log")
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, cwd=case_dir, stdout=fh, stderr=subprocess.STDOUT)
    files = sorted(glob.glob(os.path.join(case_dir, "bin", f"{name}.prim.*.bin")))
    result = {"case": name, "dT": dT, "nx1": nx1, "nx3": nx3, "tlim": tlim,
              "exit_code": rc, "n_snapshots": len(files)}
    if files:
        result.update(_reduce(files))
    with open(os.path.join(workdir, f"{name}.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"[{name}] rc={rc} snaps={len(files)} "
          f"gf_final={result.get('gf_raw', [float('nan')])[-1]:.4f} "
          f"x_final={result.get('x', [float('nan')])[-1]:.4f}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(args.repo, "vis", "python"))
    global bin_convert
    import bin_convert  # noqa: F401
    os.makedirs(args.workdir, exist_ok=True)
    for name in args.cases.split(","):
        _run_case(name, CASES[name], os.path.abspath(args.binary),
                  os.path.abspath(args.repo), os.path.abspath(args.workdir))


if __name__ == "__main__":
    main()
