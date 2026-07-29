#!/usr/bin/env python3
"""Digitize the Ruiz 2023 II Fig. 2 open-circuit voltage trace (issue #241).

Reads figures/ruiz_fig2_voltage_trace.png (300-dpi crop of arXiv:2209.14911v2
p. 2, Fig. 2), auto-calibrates axes from the plot frame + y ticks, extracts the
blue phi_oc(t) curve per column, resamples to a 5-ns grid, and writes
extracted/b6_voltage_trace.json plus an overlay plot for human QC (ADR-0008).

Calibration: left/right spines = -100 / 200 ns; bottom spine (0-tick) = 0 MV;
topmost y tick = 8 MV (y axis is linear with ticks 0,2,4,6,8).
"""
import json
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.environ.get("FIGDIR", "figures")
OUT = os.environ.get("OUT", "extracted")
os.makedirs(OUT, exist_ok=True)

img = np.asarray(
    Image.open(os.path.join(FIGDIR, "ruiz_fig2_voltage_trace.png")).convert("RGB")
).astype(int)
H, W, _ = img.shape
dark = img.sum(axis=2) < 3 * 110

# --- frame detection: long dark lines -----------------------------------------
col_frac = dark.mean(axis=0)          # fraction of dark pixels per column
row_frac = dark.mean(axis=1)
vlines = np.where(col_frac > 0.5)[0]  # spines span most of the height
hlines = np.where(row_frac > 0.5)[0]
x_left, x_right = vlines.min(), vlines.max()
y_top, y_bottom = hlines.min(), hlines.max()
print(f"frame: x [{x_left},{x_right}]  y [{y_top},{y_bottom}]")

# --- y ticks: short dark segments just left of the left spine ------------------
tick_band = dark[:, max(0, x_left - 12):x_left - 2]
tick_rows = np.where(tick_band.mean(axis=1) > 0.5)[0]
tick_rows = tick_rows[tick_rows <= y_bottom + 3]   # x-label glyphs sit below the frame
# cluster consecutive rows into tick centres
ticks = []
if len(tick_rows):
    start = prev = tick_rows[0]
    for r in tick_rows[1:]:
        if r > prev + 2:
            ticks.append(0.5 * (start + prev))
            start = r
        prev = r
    ticks.append(0.5 * (start + prev))
print("y tick rows:", [round(t, 1) for t in ticks])
py_v8 = min(ticks)          # topmost tick = 8 MV
py_v0 = max(ticks)          # bottom tick = 0 MV (coincides with bottom spine)


def x_ns(px):
    return -100.0 + (px - x_left) * 300.0 / (x_right - x_left)


def y_mv(py):
    return 8.0 * (py_v0 - py) / (py_v0 - py_v8)


# --- blue curve extraction (tab:blue ~ (31,119,180)) ---------------------------
d_blue = np.abs(img - np.array([31, 119, 180])).sum(axis=2)
blue = d_blue < 120
blue[:y_top + 2, :] = False
blue[y_bottom - 1:, :] = False
blue[:, :x_left + 2] = False
blue[:, x_right - 1:] = False

cols, ts, vs = [], [], []
for px in range(x_left + 3, x_right - 2):
    ys = np.where(blue[:, px])[0]
    if len(ys) == 0:
        continue
    cols.append(px)
    ts.append(x_ns(px))
    vs.append(y_mv(ys.mean()))
ts, vs = np.array(ts), np.array(vs)
print(f"extracted {len(ts)} columns, t [{ts.min():.1f},{ts.max():.1f}] ns, "
      f"phi peak {vs.max():.2f} MV @ {ts[np.argmax(vs)]:.0f} ns")

# FWHM sanity
half = 0.5 * vs.max()
above = np.where(vs >= half)[0]
print(f"FWHM {ts[above[-1]] - ts[above[0]]:.1f} ns "
      f"[{ts[above[0]]:.1f}, {ts[above[-1]]:.1f}]")

# --- resample to a 5-ns grid ---------------------------------------------------
t_grid = np.arange(-100.0, 200.0 + 1e-9, 5.0)
v_grid = np.interp(t_grid, ts, vs)

with open(os.path.join(OUT, "b6_voltage_trace.json"), "w") as f:
    json.dump({
        "figure": "ruiz_fig2_voltage_trace.png",
        "x_axis": "time (ns)", "y_axis": "phi_oc (MV)",
        "series": "open-circuit voltage source (blue curve)",
        "qc": "PENDING",
        "calibration": {
            "x_left_px": int(x_left), "x_right_px": int(x_right),
            "x_left_ns": -100.0, "x_right_ns": 200.0,
            "y0_px": float(py_v0), "y8_px": float(py_v8),
        },
        "points": [
            {"t_ns": round(float(t), 1), "phi_oc_MV": round(float(v), 3)}
            for t, v in zip(t_grid, v_grid)
        ],
    }, f, indent=1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.imshow(img.astype(np.uint8),
          extent=[x_ns(0), x_ns(W), y_mv(H), y_mv(0)], aspect="auto")
ax.plot(ts, vs, "r-", lw=0.8, alpha=0.8, label="per-column extraction")
ax.plot(t_grid, v_grid, "g.", ms=3, label="committed 5-ns points")
ax.set_xlim(x_ns(0), x_ns(W))
ax.set_ylim(y_mv(H), y_mv(0))
ax.legend(loc="upper right")
fig.savefig(os.path.join(OUT, "B6_voltage_overlay.png"), dpi=140)
print("wrote", os.path.join(OUT, "b6_voltage_trace.json"),
      "and B6_voltage_overlay.png")
