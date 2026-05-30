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

The MagLIF curve benchmarks (#142/#143) overlay digitized *curves* the same way; this
slice delivers the scalar overlay used by the B4 ICF substrate.
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
