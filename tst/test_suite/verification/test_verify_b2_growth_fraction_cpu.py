"""
Red-first component test for the B2 McBride soft quantitative anchor (issue [B2-S2]/#155,
PRD #138, ADR-0008).

A *component* test (not a physics-sim test): it exercises the exact reductions + oracle
dispatch the faithful B2 Stage-1 benchmark
(``cylindrical/test_verify_maglif_b2_ellison_cpu.py``) wires into for its SOFT
quantitative anchor, without running a simulation, so it stays in the fast CPU component
band.  It pins:

* the committed McBride ``growth_fraction`` scalar datum (band [0.05, 0.15]) and that the
  oracle PASSES an in-band growth fraction and FAILS an out-of-band one (the anchor is
  real, not vacuous);
* the pure reductions in ``mcbride_b2_anchor`` (spike/bubble extrema, growth fraction,
  convergence x, the McBride printed fit laws read from the committed ``fit_laws`` block);
* that a reported (non-binding) row + the overlay render without hard-failing -- the
  report-only contract (#155: never hard-fail; escalate a persistent miss as a note).

Oracle: Layer 1 -- literature (experiment), via ``GroundTruthOracle``.  Auto-collected by
run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math
import os

import numpy as np
import pytest

import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.mcbride_b2_anchor as mb2
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

B2_BENCH = "B2"
GF_DATUM = "growth_fraction"
R0_UM = 3468.8  # committed initial outer-liner radius r_o 3.4688 mm in um


@pytest.fixture(autouse=True)
def _clean_scorecard():
    """Each test runs against an empty scorecard, then any suite rows are restored.

    The scorecard is a process-wide singleton shared with the benchmark tests; snapshot
    and restore so this component test never clobbers the real suite scorecard, whatever
    the collection order.
    """
    saved = scorecard.rows()
    scorecard.reset()
    yield
    scorecard.set_rows(saved)


def test_b2_growth_fraction_datum_band_cpu():
    """The committed McBride growth-fraction datum is non-pending with the [0.05,0.15]
    band (midpoint 0.10 +/- 0.05), carrying full provenance."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B2_BENCH, GF_DATUM)
    assert datum.kind == "scalar"
    assert not datum.is_pending, "growth-fraction datum must be a stated value"
    assert datum.value == pytest.approx(0.10)
    assert datum.tolerance_band() == pytest.approx(0.05)
    # Provenance is enforced on load (ADR-0008): the datum carries its citation.
    assert datum.source.get("doi")


def test_b2_growth_fraction_oracle_discriminates_cpu():
    """The oracle PASSES an in-band growth fraction and FAILS an out-of-band one -- the
    soft anchor genuinely discriminates (red->green proof for the datum)."""
    oracle = gto.GroundTruthOracle.from_committed()
    # In band: 0.10 (centre) and the band edges 0.05 / 0.15 all pass.
    for gf in (0.05, 0.10, 0.15):
        assert oracle.compare(B2_BENCH, GF_DATUM, gf).passed, f"{gf} should be in band"
    # Out of band on either side: a flat (no-growth) surface and an over-grown one fail.
    for gf in (0.0, 0.30):
        assert not oracle.compare(B2_BENCH, GF_DATUM, gf).passed, f"{gf} should fail"


def test_b2_growth_fraction_reduction_cpu():
    """``growth_fraction`` / ``convergence_x`` / extrema reduce a known surface right."""
    # Surface r(z) with a spike at 2.0, a bubble at 1.0, initial outer radius 3.0.
    r_out = [2.0, 1.0, 1.5, float("nan"), 1.8]
    r_spike, r_bubble = mb2.outer_surface_extrema(r_out)
    assert r_spike == pytest.approx(2.0)
    assert r_bubble == pytest.approx(1.0)
    # (R_spike - R_bubble)/(R0 - R_bubble) = (2-1)/(3-1) = 0.5.
    assert mb2.growth_fraction(r_spike, r_bubble, 3.0) == pytest.approx(0.5)
    # x = 1 - R_bubble/R_outer(0) = 1 - 1/4 = 0.75.
    assert mb2.convergence_x(1.0, 4.0) == pytest.approx(0.75)
    # No convergence yet (R_bubble == R0) -> growth fraction undefined (nan, not div0).
    assert math.isnan(mb2.growth_fraction(3.0, 3.0, 3.0))
    # All-NaN surface -> (nan, nan).
    sp, bu = mb2.outer_surface_extrema([float("nan"), float("nan")])
    assert math.isnan(sp) and math.isnan(bu)


def test_b2_fit_laws_committed_cpu():
    """The McBride printed fit laws are committed in the B2 ``fit_laws`` block and
    evaluate as 450*x - 90 (amplitude, um) and 750*x (wavelength, um)."""
    oracle = gto.GroundTruthOracle.from_committed()
    laws = mb2.fit_laws(oracle.meta(B2_BENCH))
    assert laws["x_definition"] == "1 - R_bubble/R_outer(0)"
    assert laws["growth_fraction_band"] == [0.05, 0.15]
    amp = laws["amplitude_um"]
    wav = laws["wavelength_um"]
    # amplitude_um(0.4) = 450*0.4 - 90 = 90 um; it is negative below x = 0.2 (90/450).
    assert mb2.linear_law(0.4, amp["slope"], amp["intercept"]) == pytest.approx(90.0)
    assert mb2.linear_law(0.1, amp["slope"], amp["intercept"]) < 0.0
    # wavelength_um(0.4) = 750*0.4 = 300 um.
    assert mb2.linear_law(0.4, wav["slope"], wav["intercept"]) == pytest.approx(300.0)
    assert laws["source"].get("doi")


def test_b2_growth_fraction_reported_not_binding_cpu():
    """A growth-fraction verdict is recorded as a NON-binding (report-only) scorecard row,
    even when it fails -- the #155 report-only contract (never hard-fail)."""
    oracle = gto.GroundTruthOracle.from_committed()
    # An out-of-band sim value: recorded FAIL but binding=False (reported, not asserted).
    res = oracle.compare(B2_BENCH, GF_DATUM, 0.40)
    assert not res.passed
    row = scorecard.record_result(res, binding=False, note="needs investigation (#155)")
    assert row.binding is False
    assert row.verdict == scorecard.FAIL
    assert "155" in str(row)
    assert len(scorecard.rows()) == 1


def test_b2_gf_validity_window_committed_cpu():
    """The committed ``fit_laws`` block carries McBride's measured abscissa domain
    x in [0.4, 0.95] (Fig. 7a; the nonlinear regime the 0.05-0.15 fraction is stated
    for), and ``in_validity_window`` discriminates inside from outside (#229)."""
    oracle = gto.GroundTruthOracle.from_committed()
    laws = mb2.fit_laws(oracle.meta(B2_BENCH))
    assert laws["x_valid_range"] == [0.4, 0.95]
    # The reduced CPU gate's deepest convergence (x~0.214) is OUTSIDE the window.
    assert not mb2.in_validity_window(0.214, laws)
    # The window itself (closed interval) is inside.
    for x in (0.4, 0.6, 0.95):
        assert mb2.in_validity_window(x, laws), f"x={x} should be in window"
    # Beyond the measured domain / undefined convergence are outside.
    assert not mb2.in_validity_window(0.96, laws)
    assert not mb2.in_validity_window(float("nan"), laws)


def test_b2_law_implied_growth_fraction_cpu():
    """McBride's OWN committed fit laws imply a growth fraction (450x - 90)/(R0*x)
    that sits inside the [0.05, 0.15] band across the validity window but FAR BELOW
    it at the reduced gate's x~0.214 -- the band is simply not attainable there, so
    an out-of-window comparison is a category error, not a physics miss (#229)."""
    oracle = gto.GroundTruthOracle.from_committed()
    laws = mb2.fit_laws(oracle.meta(B2_BENCH))
    lo, hi = laws["growth_fraction_band"]
    # Across the measured window the law-implied fraction is in McBride's own band.
    for x in (0.4, 0.6, 0.8, 0.95):
        gf = mb2.law_implied_growth_fraction(x, laws, R0_UM)
        assert lo <= gf <= hi, f"law-implied gf({x})={gf} outside [{lo},{hi}]"
    # At the gate's deepest convergence the law-implied fraction is ~0.008 << 0.05.
    gf_gate = mb2.law_implied_growth_fraction(0.214, laws, R0_UM)
    assert gf_gate == pytest.approx(0.00849, abs=5.0e-4)
    assert gf_gate < lo
    # Below x = 0.2 the printed amplitude law is negative (unphysical domain).
    assert mb2.law_implied_growth_fraction(0.15, laws, R0_UM) < 0.0
    # x <= 0 (no convergence): undefined, nan.
    assert math.isnan(mb2.law_implied_growth_fraction(0.0, laws, R0_UM))


def test_b2_band_limited_extrema_cpu():
    """``outer_surface_extrema`` with ``k_max`` band-limits the interface trace to the
    radiograph's own resolution before taking extrema (mirror of the #212 B1 fix):
    grid-scale (Nyquist) noise the instrument could never see must not inflate the
    spike-bubble excursion, while the resolved mode survives untouched (#229)."""
    n = 128  # paper-resolution axial zone count (Nyquist mode 64 > instrument k 53)
    i = np.arange(n)
    mode = 0.05 * np.cos(2.0 * np.pi * 3.0 * i / n)     # resolved k=3, amp 0.05
    nyq = 0.2 * np.where(i % 2 == 0, 1.0, -1.0)         # pure-Nyquist grid noise
    trace = 3.0 + mode + nyq
    # Unfiltered extrema count the grid noise: excursion ~ 2*(0.05 + 0.2) = 0.5
    # (up to the discrete sampling of the k=3 cosine extrema).
    sp_raw, bu_raw = mb2.outer_surface_extrema(trace)
    assert sp_raw - bu_raw == pytest.approx(0.5, abs=2.0e-3)
    # Band-limited to k <= 53 the Nyquist mode vanishes exactly; k=3 survives.
    sp, bu = mb2.outer_surface_extrema(trace, k_max=53)
    assert sp - bu == pytest.approx(0.1, abs=2.0e-3)
    assert sp == pytest.approx(3.05, rel=1.0e-6)
    # NaN zones (failed interface finds) are interpolated, not fatal.
    trace_nan = 3.0 + mode
    trace_nan[10] = float("nan")
    sp2, bu2 = mb2.outer_surface_extrema(trace_nan, k_max=53)
    assert sp2 - bu2 == pytest.approx(0.1, rel=1.0e-2)
    # All-NaN stays (nan, nan) with k_max too.
    sp3, bu3 = mb2.outer_surface_extrema([float("nan")] * 4, k_max=53)
    assert math.isnan(sp3) and math.isnan(bu3)


def _synthetic_trajectory(x_final, gf_final, r0=3.4688, nsnap=5):
    """Times + spike/bubble arrays converging to (x_final, gf_final) at the end."""
    times = np.linspace(0.0, 20.0, nsnap)
    xs = np.linspace(0.0, x_final, nsnap)
    r_bubble = r0 * (1.0 - xs)
    r_spike = r_bubble + gf_final * (r0 - r_bubble)
    kdom = np.array([0.0] + [3.0] * (nsnap - 1))
    return times, r_spike, r_bubble, kdom


def test_b2_gf_report_out_of_window_records_pending_cpu():
    """The Stage-2 report at a convergence OUTSIDE McBride's validity window records
    the growth fraction as PENDING (out-of-regime; deeper run needed), NOT as a FAIL:
    comparing x~0.214 against a band stated for x in [0.4, 0.95] near stagnation is
    the #229 timing-convention category error."""
    import test_suite.cylindrical.test_verify_maglif_b2_ellison_cpu as b2

    # Replicate the recorded miss: gf = 0.3455 at x = 0.214 (scorecard.txt, #229).
    times, r_spike, r_bubble, kdom = _synthetic_trajectory(0.214, 0.3455)
    gf_final, x_final = b2._report_mcbride_anchor(times, r_spike, r_bubble, kdom)
    assert gf_final == pytest.approx(0.3455, rel=1.0e-6)
    assert x_final == pytest.approx(0.214, rel=1.0e-6)
    gf_rows = [r for r in scorecard.rows() if "growth fraction" in r.observable]
    assert len(gf_rows) == 1
    row = gf_rows[0]
    assert row.verdict == scorecard.PENDING, (
        "an out-of-window comparison must be reported PENDING, not "
        f"{row.verdict}: the McBride band is only stated for x in [0.4, 0.95]"
    )
    assert row.binding is False
    assert "0.4" in str(row.note)          # the note names the validity window
    assert row.sim == pytest.approx(0.3455, rel=1.0e-6)  # measured value still shown


def test_b2_gf_report_in_window_compares_cpu():
    """The Stage-2 report at a convergence INSIDE the validity window compares against
    the [0.05, 0.15] band as before (report-only): in-band -> PASS row."""
    import test_suite.cylindrical.test_verify_maglif_b2_ellison_cpu as b2

    times, r_spike, r_bubble, kdom = _synthetic_trajectory(0.6, 0.09)
    gf_final, x_final = b2._report_mcbride_anchor(times, r_spike, r_bubble, kdom)
    assert gf_final == pytest.approx(0.09, rel=1.0e-6)
    assert x_final == pytest.approx(0.6, rel=1.0e-6)
    gf_rows = [r for r in scorecard.rows() if "growth fraction" in r.observable]
    assert len(gf_rows) == 1
    assert gf_rows[0].verdict == scorecard.PASS
    assert gf_rows[0].binding is False     # Stage 2 stays report-only (soft anchor)


def test_b2_qc_provenance_committed_cpu():
    """The Stage-2 digitization-QC evidence (fit-law transcription checks) is COMMITTED
    under the digitization-review tree with a provenance README (#229 AC#2) -- the
    provenance trail for the 450x-90 / 750x laws is reproducible from the repo."""
    here = os.path.dirname(os.path.abspath(__file__))
    qc_dir = os.path.join(here, "ground_truth", "digitization_review", "b2_qc")
    expected = [
        "README.md",
        "fig7a_fitlaw.png",
        "fig7b_fitlaw.png",
        "mcbride2012_fig6_7_column.png",
        "mcbride2012_fig6_radii_vs_time.png",
        "mcbride2012_fig7a_amplitude.png",
    ]
    for name in expected:
        path = os.path.join(qc_dir, name)
        assert os.path.isfile(path), f"missing committed B2 QC evidence: {path}"
        assert os.path.getsize(path) > 0
    with open(os.path.join(qc_dir, "README.md")) as fh:
        readme = fh.read()
    assert "10.1103/PhysRevLett.109.135004" in readme   # primary-source DOI
    assert "450" in readme and "750" in readme          # the QC'd fit laws


def test_b2_growth_fraction_overlay_writes_png_cpu(tmp_path):
    """The growth-fraction overlay (gf(t) time series + the McBride [0.05,0.15] band)
    renders to a PNG without error."""
    t = [0.0, 5.0, 10.0, 15.0, 20.0]
    gf = [0.0, 0.02, 0.06, 0.10, 0.13]
    out = overlay.overlay_scalars(
        "b2_growth_fraction_test", t, {"growth_fraction": gf},
        {"growth_fraction": {"exp_value": 0.10, "band": 0.05, "unit": "",
                             "sim_value": gf[-1]}},
        xlabel="t [ns]", title="B2 growth fraction (test)", outdir=str(tmp_path))
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0
