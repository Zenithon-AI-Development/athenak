"""
Summarize the #229 discriminating runs into the growth-fraction attribution table.

Reads the per-case JSONs produced by ``b2_attribution_runs.py`` and emits
``attribution_summary.json`` -- the evidence behind the #229 written attribution
(one discriminating run or measurement per candidate cause).  See that file and
issue #229 for the narrative.

Usage:  python3 b2_attribution_summary.py --runs <dir-with-case-jsons>
"""

import argparse
import json
import os

import numpy as np

R0_MM = 3.4688          # initial outer-liner radius [mm]
BAND = (0.05, 0.15)     # McBride stated growth-fraction band
WINDOW = (0.4, 0.95)    # McBride Fig. 7a measured abscissa domain (fit_laws)


def law_gf(x):
    """McBride's own fit-law-implied growth fraction (450x - 90)/(R0_um * x)."""
    return (450.0 * x - 90.0) / (R0_MM * 1.0e3 * x)


def gf_at_x(case, x_target, key="gf_band"):
    """Interpolate a case's gf(x) series at a matched convergence x_target."""
    x = np.asarray(case["x"], dtype=float)
    g = np.asarray(case[key], dtype=float)
    good = np.isfinite(x) & np.isfinite(g) & (x > 0.02)   # drop the pre-imprint spike
    x, g = x[good], g[good]
    if x.size < 2 or x_target > x.max() or x_target < x.min():
        return float("nan")
    return float(np.interp(x_target, x, g))


def window_series(case):
    """The (x, gf_band, gf_pct, t) samples inside McBride's validity window."""
    x = np.asarray(case["x"], dtype=float)
    inw = (x >= WINDOW[0]) & (x <= WINDOW[1])
    return [
        {"t": case["t"][i], "x": round(float(x[i]), 4),
         "gf_band": round(case["gf_band"][i], 4),
         "gf_pct": round(case["gf_pct"][i], 4),
         "law_implied": round(law_gf(float(x[i])), 4)}
        for i in np.where(inw)[0]
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    cases = {}
    for name in ("repro_dT100", "dT10", "dT1000", "res2x", "deep"):
        path = os.path.join(args.runs, f"{name}.json")
        if os.path.isfile(path):
            with open(path) as fh:
                cases[name] = json.load(fh)

    base = cases["repro_dT100"]
    x_gate = float(base["x"][-1])
    summary = {
        "recorded_miss": {"growth_fraction": 0.345496, "x": 0.214,
                          "band": list(BAND), "source": "plots/scorecard.txt"},
        "timing_convention": {
            "law_implied_gf_at_gate_x": round(law_gf(x_gate), 5),
            "law_implied_gf_in_window": {str(x): round(law_gf(x), 4)
                                         for x in (0.4, 0.6, 0.8, 0.95)},
            "deep_run_window_series": window_series(cases["deep"]) if "deep" in cases
            else "deep run unavailable",
        },
        "stale_pre211_data": {
            "gate_rerun_gf": round(base["gf_band"][-1], 5),
            "gate_rerun_x": round(x_gate, 5),
            "deck_eos": "ideal (cn4/#210 path never exercised)",
        },
        "seed_amplitude_dT": {
            name: {"gf_final": round(cases[name]["gf_band"][-1], 4),
                   "x_final": round(cases[name]["x"][-1], 4),
                   "gf_at_x0.14": round(gf_at_x(cases[name], 0.14), 4)}
            for name in ("dT10", "repro_dT100", "dT1000") if name in cases
        },
        "resolution": {
            "gate_gf_at_matched_x": round(gf_at_x(base, min(x_gate, 0.19)), 4),
            "res2x_gf_at_matched_x": round(gf_at_x(cases["res2x"],
                                                   min(x_gate, 0.19)), 4)
            if "res2x" in cases else "res2x unavailable",
            "matched_x": round(min(x_gate, 0.19), 4),
        },
        "reduction_mismatch": {
            "gate_gf_raw_vs_band": [round(base["gf_raw"][-1], 4),
                                    round(base["gf_band"][-1], 4)],
            "note": "identical at the gate by construction (Nyquist 24 < k_band 53)",
            "gate_gf_pct_final": round(base["gf_pct"][-1], 4),
        },
    }
    out = os.path.join(args.runs, "attribution_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
