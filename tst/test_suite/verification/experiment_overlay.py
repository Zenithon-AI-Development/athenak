"""
Experiment-overlay plotting (issue [VA1]/#140).

Part of the quantitative-anchoring substrate (PRD #138, ADR-0008). Where
``harness.verify`` overlays a run against its own golden baseline (self-regression), this
module overlays a run against the **experiment**: each scalar observable's stated
experimental value is drawn as a horizontal line with its tolerance band shaded across the
simulation time series, the simulation's reduced value is marked, and -- where stored -- a
secondary-reference code value (FLASH/LASNEX/HYDRA) is drawn as a clearly-labelled,
non-binding dashed line. The plot makes sim-vs-experiment visible at a glance, with units
and the abscissa stated.

The MagLIF curve benchmarks (#142/#143) overlay digitized *curves* the same way, via
``overlay_curve`` (issue [VA3]/#142); ``overlay_scalars`` is the scalar overlay used by
the B4 ICF substrate.
"""

import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Default output location: the shared verification plot dir (anchored to this module so it
# lands in-repo regardless of the test's working directory).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLOT_DIR = os.path.join(_THIS_DIR, "plots")


def overlay_scalars(name, x, series, overlays, xlabel="t", title=None, outdir=None):
    """Plot simulation time series with each scalar's experimental value + band overlaid.

    Parameters
    ----------
    name : str
        Slug naming the output PNG (``<name>_overlay.png``).
    x : array_like
        Shared abscissa (e.g. time) for the simulation series.
    series : dict[str, array_like]
        Named simulation time series; one subplot per entry.
    overlays : dict[str, dict]
        Maps a ``series`` label to its experimental overlay spec:
        ``{"exp_value", "band", "unit", "sim_value" (optional),
        "secondary" (optional {"code", "value"})}``. The experimental value is drawn as a
        solid horizontal line, the band as a shaded region ``[exp-band, exp+band]``, the
        simulation's reduced scalar (if given) as a marked horizontal line, and the
        secondary code value (if given) as a labelled dashed line.
    xlabel : str
        Label for the shared x-axis.
    title : str, optional
        Figure title (defaults to ``name``).
    outdir : str, optional
        Output directory (defaults to the shared verification plot dir).

    Returns
    -------
    str
        Absolute path to the written PNG.
    """
    outdir = outdir or DEFAULT_PLOT_DIR
    os.makedirs(outdir, exist_ok=True)
    x = np.asarray(x, dtype=float)
    labels = list(series.keys())
    nfields = len(labels)
    fig, axes = plt.subplots(
        nfields, 1, figsize=(7, 2.8 * nfields), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for ax, label in zip(axes, labels):
        ax.plot(x, np.asarray(series[label], dtype=float), "-", lw=1.4,
                color="C0", label="simulation")
        ov = overlays.get(label)
        if ov is not None:
            unit = ov.get("unit", "")
            exp = float(ov["exp_value"])
            band = float(ov["band"])
            ax.axhline(exp, color="C3", lw=1.6,
                       label=f"experiment {exp:g} {unit}".strip())
            ax.axhspan(exp - band, exp + band, color="C3", alpha=0.15,
                       label=f"exp band +/-{band:g}")
            sim_val = ov.get("sim_value")
            if sim_val is not None:
                ax.axhline(float(sim_val), color="C0", lw=1.0, ls=":",
                           label=f"sim reduced {float(sim_val):g}")
            sec = ov.get("secondary")
            if sec is not None:
                code = sec.get("code", "secondary")
                ax.axhline(float(sec["value"]), color="0.4", lw=1.0, ls="--",
                           label=f"{code} {float(sec['value']):g} (ref, non-binding)")
            ylabel = f"{label} [{unit}]" if unit else label
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title if title else name)
    fig.tight_layout()
    out = os.path.join(outdir, f"{name}_overlay.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    logging.info(f"[overlay] wrote experiment-overlay plot -> {out}")
    return out


def overlay_curve(name, x_sim, y_sim, exp_points=None, xlabel="t", ylabel=None,
                  x_unit=None, unit=None, title=None, pending_issue=None, outdir=None):
    """Overlay a digitized experimental *curve* (+ per-point band) on a simulation curve.

    The curve analogue of :func:`overlay_scalars`, used by the growth-curve benchmarks
    (B1 #142 / B2 #143 / their SI replications #120/#122). The simulation growth curve is
    drawn as a marked line; each experimental point is drawn with an error bar of its own
    tolerance band and the band envelope is shaded, so sim-vs-experiment is visible at a
    glance. When the experimental curve is not yet digitized the simulation curve is still
    plotted, annotated as ``pending digitization (#NNN)`` -- never a fabricated band.

    Parameters
    ----------
    name : str
        Slug naming the output PNG (``<name>_growth_overlay.png``).
    x_sim, y_sim : array_like
        The simulation growth curve (e.g. time and seeded-mode amplitude).
    exp_points : list[dict] or None
        Digitized experimental points, each ``{"x", "y", "band"}`` with ``band`` the
        absolute per-point tolerance (``max(exp_err, dig_err)``, resolved by the caller
        via the oracle). ``None``/empty means the curve is still pending digitization: sim
        curve is plotted with a pending annotation and no experimental band is drawn.
    xlabel, ylabel : str
        Axis labels (units are appended from ``x_unit``/``unit`` when given).
    x_unit, unit : str, optional
        Abscissa / ordinate units, appended to the axis labels.
    title : str, optional
        Figure title (defaults to ``name``).
    pending_issue : int or str, optional
        Issue that will digitize the curve; woven into the pending annotation.
    outdir : str, optional
        Output directory (defaults to the shared verification plot dir).

    Returns
    -------
    str
        Absolute path to the written PNG.
    """
    outdir = outdir or DEFAULT_PLOT_DIR
    os.makedirs(outdir, exist_ok=True)
    x_sim = np.asarray(x_sim, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(x_sim, y_sim, "-o", lw=1.4, ms=3, color="C0", label="simulation")

    if exp_points:
        xe = np.array([float(p["x"]) for p in exp_points])
        ye = np.array([float(p["y"]) for p in exp_points])
        be = np.array([float(p["band"]) for p in exp_points])
        order = np.argsort(xe)
        xe, ye, be = xe[order], ye[order], be[order]
        ax.errorbar(xe, ye, yerr=be, fmt="s", ms=4, color="C3", ecolor="C3",
                    elinewidth=1.2, capsize=3, label="experiment")
        ax.fill_between(xe, ye - be, ye + be, color="C3", alpha=0.15,
                        label="exp tolerance band")
    else:
        tag = ""
        if pending_issue is not None:
            pi = str(pending_issue)
            it = pi if pi.startswith("#") else f"#{pi}"
            tag = f" ({it})"
        ax.text(0.5, 0.95, f"experiment: pending digitization{tag}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="C3", bbox=dict(boxstyle="round", fc="white", ec="C3", alpha=0.8))

    ax.set_xlabel(f"{xlabel} [{x_unit}]" if x_unit else xlabel)
    yl = ylabel if ylabel else name
    ax.set_ylabel(f"{yl} [{unit}]" if unit else yl)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.suptitle(title if title else name)
    fig.tight_layout()
    out = os.path.join(outdir, f"{name}_growth_overlay.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    logging.info(f"[overlay] wrote experiment-curve overlay plot -> {out}")
    return out
