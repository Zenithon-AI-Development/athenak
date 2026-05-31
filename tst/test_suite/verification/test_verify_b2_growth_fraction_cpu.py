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

import pytest

import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.mcbride_b2_anchor as mb2
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

B2_BENCH = "B2"
GF_DATUM = "growth_fraction"


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
