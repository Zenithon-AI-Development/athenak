"""
McBride 2012 B2 multi-mode MRT *soft quantitative* anchor reductions (issue [B2-S2]/#155).

Part of the quantitative-anchoring substrate (PRD #138, ADR-0008).  Ellison benchmark 2 is
a QUALITATIVE morphology benchmark (the binding gate is the Stage-1 signature test,
``cylindrical/test_verify_maglif_b2_ellison_cpu.py`` / #163): there is no published
amplitude-vs-time curve to digitize, and absolute-time comparison is explicitly disclaimed
by the authors (shot-to-shot drive-timing scatter).  The correct *quantitative* anchor is
McBride's own fit laws vs convergence, stated in the paper text / Fig. 7 printed fits:

  * growth fraction  ``(R_spike - R_bubble) / (R0 - R_bubble)``  in **[0.05, 0.15]**
    (McBride 2012 text), with ``R0 = R_outer(0)`` the initial outer-liner radius,
    ``R_spike = max`` and ``R_bubble = min`` of the driven (liner/vacuum) interface;
  * amplitude(um) ``= 450*x - 90`` and wavelength(um) ``= 750*x`` (Fig. 7a/b fits),
    with the dimensionless convergence ``x = 1 - R_bubble/R_outer(0)``.

These are *stated published laws*, not figure-digitized points (ADR-0008: never fabricate
points/tolerances).  This module holds the pure reductions so they are unit-testable
without running a simulation; the law COEFFICIENTS live in the committed ground-truth
datum file (``ground_truth/b2_multimode_mrt_mcbride_2012.json`` -> ``fit_laws`` block,
reachable via ``GroundTruthOracle.meta("B2")``) so there is one provenance-tagged source
of truth.  Per the issue these anchors are SOFT (report-only): a persistent miss is
reported (scorecard + overlay) and escalated as a "needs investigation" note -- not a fail
(FLASH matched B2, so a miss is a diagnostic signal about our setup, not a regression).
"""

import numpy as np


def outer_surface_extrema(r_out):
    """Return ``(R_spike, R_bubble)`` = (max, min) of the finite outer-interface radii.

    ``r_out`` is the per-axial-zone driven (liner/vacuum) interface radius r(z) (the same
    half-max crossing the Stage-1 test reduces); NaNs (zones with no resolved interface)
    are ignored.  Returns ``(nan, nan)`` if nothing is finite.
    """
    r = np.asarray(r_out, dtype=float)
    good = np.isfinite(r)
    if not good.any():
        return float("nan"), float("nan")
    return float(np.max(r[good])), float(np.min(r[good]))


def growth_fraction(r_spike, r_bubble, r0):
    """McBride dimensionless growth fraction ``(R_spike - R_bubble)/(R0 - R_bubble)``.

    ``R0`` is the initial outer-liner radius.  The denominator is the bubble's convergence
    distance; before any convergence (``R_bubble == R0``) it is 0 and the fraction is
    undefined -> returns ``nan`` (the caller reports it as not-yet-meaningful rather than
    dividing by zero).
    """
    denom = float(r0) - float(r_bubble)
    if denom <= 0.0:
        return float("nan")
    return (float(r_spike) - float(r_bubble)) / denom


def convergence_x(r_bubble, r_outer0):
    """Dimensionless convergence ``x = 1 - R_bubble/R_outer(0)`` (McBride abscissa)."""
    if float(r_outer0) <= 0.0:
        return float("nan")
    return 1.0 - float(r_bubble) / float(r_outer0)


def linear_law(x, slope, intercept):
    """Evaluate a McBride printed fit law ``slope*x + intercept`` at convergence ``x``."""
    return float(slope) * float(x) + float(intercept)


def fit_laws(meta):
    """Pull the committed B2 ``fit_laws`` block from a ``GroundTruthOracle.meta("B2")``.

    Returns the raw dict (``{"amplitude_um": {slope, intercept}, "wavelength_um": {...},
    "x_definition", "growth_fraction_band", "source"}``) so the benchmark and the
    component test read the SAME committed coefficients.  Raises ``KeyError`` if absent.
    """
    return meta["fit_laws"]
