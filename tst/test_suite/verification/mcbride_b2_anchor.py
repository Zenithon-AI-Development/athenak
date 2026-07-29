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


def band_limit(rz, k_max):
    """Truncate a periodic axial trace to modes ``k <= k_max`` cycles across the domain.

    Mirror of the #212 B1 fix: a radiograph cannot resolve structure finer than its own
    spatial resolution, so a synthetic spike/bubble excursion compared against it must be
    band-limited the same way -- otherwise (max - min) also counts grid-seeded modes,
    which grow like sqrt(k) and strengthen with every refinement.  NaN zones (failed
    interface finds) are filled by periodic interpolation first so the transform stays
    well-posed (the axial domain is periodic by construction).
    """
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


def outer_surface_extrema(r_out, k_max=None):
    """Return ``(R_spike, R_bubble)`` = (max, min) of the finite outer-interface radii.

    ``r_out`` is the per-axial-zone driven (liner/vacuum) interface radius r(z) (the same
    half-max crossing the Stage-1 test reduces); NaNs (zones with no resolved interface)
    are ignored.  Returns ``(nan, nan)`` if nothing is finite.

    With ``k_max`` set (#229, mirroring #212) the trace is first band-limited to
    ``k <= k_max`` cycles across the periodic axial domain, so the extrema count the
    seeded modes and their resolvable harmonics but not grid-scale structure the
    radiograph (McBride 2012: 15 um spatial resolution) could never have seen.
    """
    r = np.asarray(r_out, dtype=float)
    if not np.isfinite(r).any():
        return float("nan"), float("nan")
    if k_max is not None:
        r = band_limit(r, k_max)
    good = np.isfinite(r)
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


def in_validity_window(x, laws):
    """Whether convergence ``x`` lies in McBride's measured abscissa domain (#229).

    The [0.05, 0.15] growth-fraction band is stated for the NONLINEAR regime near
    stagnation -- Fig. 7a's data span x in [0.4, 0.95] (the committed
    ``fit_laws["x_valid_range"]``).  Outside that domain the band is not a published
    claim at all (the amplitude law 450x - 90 is even negative below x = 0.2), so a
    comparison there is a category error, not a physics verdict.  NaN -> False.
    """
    lo, hi = laws["x_valid_range"]
    x = float(x)
    if not np.isfinite(x):
        return False
    return lo <= x <= hi


def law_implied_growth_fraction(x, laws, r0_um):
    """The growth fraction McBride's OWN fit laws imply at convergence ``x`` (#229).

    growth fraction = amplitude / distance moved = (450x - 90) / (r0_um * x), with
    ``r0_um`` the initial outer-liner radius in um.  Inside the validity window this
    lands in the published [0.05, 0.15] band (self-consistency); at shallow convergence
    it falls far below the band (and is negative below x = 0.2), which is WHY an
    out-of-window band comparison cannot be met even by the reference data themselves.
    Returns ``nan`` for ``x <= 0`` (no convergence -> fraction undefined).
    """
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        return float("nan")
    amp = laws["amplitude_um"]
    return linear_law(x, amp["slope"], amp["intercept"]) / (float(r0_um) * x)


def fit_laws(meta):
    """Pull the committed B2 ``fit_laws`` block from a ``GroundTruthOracle.meta("B2")``.

    Returns the raw dict (``{"amplitude_um": {slope, intercept}, "wavelength_um": {...},
    "x_definition", "growth_fraction_band", "source"}``) so the benchmark and the
    component test read the SAME committed coefficients.  Raises ``KeyError`` if absent.
    """
    return meta["fit_laws"]
