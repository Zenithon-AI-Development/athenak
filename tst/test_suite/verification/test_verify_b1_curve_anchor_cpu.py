"""
Red-first component test for the B1 single-mode MRT curve anchor (issue [VA6]/#154,
follow-up to [VA3]/#142, PRD #138, ADR-0008).

This is a *component* test, not a physics-sim test: it exercises the exact curve-anchor
dispatch the B1 ``_cpu`` benchmark (``cylindrical/test_verify_maglif_mrt_cpu.py``) wires
into -- without running a simulation, so it stays in the fast CPU component band.

Unlike the B2 anchor (``test_verify_b2_curve_anchor_cpu.py``, still pending the paywalled
McBride figure), the Sinars-2011 single-mode amplitude-growth curve **has been digitized
and human-QC'd GOOD** (11 points; ``digitization_review/extracted/
extracted_points_v3.json`` key ``B1_single_mode_amplitude``) and promoted into the
committed datum here.  This test
is therefore the inverse of the B2 one: it proves the committed B1 datum is **no longer
pending** and that the oracle compares the sim growth curve against it without raising
``PendingDatumError`` (PRD #138 story 8 -- "the bar is the experiment").

It proves:

* the committed B1 datum is a populated ``kind: curve`` with full ADR-0008 provenance
  (y/x units, per-point experimental + digitization error) and is no longer pending;
* the **populated** branch -- the oracle compares the sim observable against the digitized
  Sinars points and the verdict's ``worst_point`` feeds ``scorecard.record_result`` as a
  non-binding row (the reduced ``_cpu`` surrogate reports rather than asserts absolute
  magnitudes; the binding absolute-SI assert is #120, per the reduced-surrogate precedent
  B3 #144 / B4 #140);
* the overlay renders the digitized experimental curve + its tolerance band.

Oracle: Layer 1 -- literature (experiment), via ``GroundTruthOracle``. Auto-collected by
run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import pytest

import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

B1_BENCH = "B1"
B1_DATUM = "single_mode_amplitude_growth"


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


def test_b1_committed_datum_is_populated_cpu():
    """The committed Sinars-2011 B1 curve datum carries the digitized growth curve and is
    no longer pending, so the benchmark takes the populated (oracle-compare) branch and
    ``PendingDatumError`` is never raised for B1 (#154, PRD #138 story 8)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B1_BENCH, B1_DATUM)
    assert datum.kind == "curve"
    assert not datum.is_pending, (
        "B1 curve must be digitized + committed (no longer pending_digitization)"
    )
    assert datum.points and len(datum.points) >= 8, "B1 curve must carry its points"
    # A populated curve compares without raising (the inverse of the pending B2 case).
    res = oracle.compare(B1_BENCH, B1_DATUM, ([0.0, 100.0], [0.0, 1.0]))
    assert isinstance(res, gto.CurveComparisonResult)
    assert res.n_compared >= 1


def test_b1_points_carry_full_provenance_cpu():
    """Every digitized point carries y/x units + per-point experimental & digitization
    error (ADR-0008); the datum validated on load, so just confirm the schema is real."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B1_BENCH, B1_DATUM)
    assert datum.unit and datum.x_unit, "curve needs y + x units"
    for pt in datum.points:
        for field in ("x", "y", "experimental_error", "digitization_error"):
            assert pt.get(field) is not None, f"point missing {field}"
        # Per-point band is positive and finite (real tolerance, not zero/NaN).
        band = datum._point_band(pt)
        assert band > 0.0


def test_b1_populated_branch_compares_and_records_non_binding_cpu():
    """Populated datum -> oracle compares the (growth-normalized) sim curve and the
    worst point is recorded as a non-binding row (the reduced-surrogate report path)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B1_BENCH, B1_DATUM)
    xe = [float(p["x"]) for p in datum.points]
    ye = [float(p["y"]) for p in datum.points]
    # A sim curve that exactly traces the digitized experimental points is well in band.
    res = oracle.compare(B1_BENCH, B1_DATUM, (xe, ye))
    assert res.n_compared == len(datum.points)
    assert res.passed
    row = scorecard.record_result(res.worst_point, binding=False)
    assert row.binding is False
    assert row.verdict == scorecard.PASS
    assert len(scorecard.rows()) == 1


def test_b1_curve_overlay_writes_png_cpu(tmp_path):
    """The B1 overlay draws the digitized experimental curve + its tolerance band on the
    sim growth curve (PRD #138 story 14 -- sim-vs-experiment at a glance)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B1_BENCH, B1_DATUM)
    exp_points = [
        {"x": float(p["x"]), "y": float(p["y"]), "band": datum._point_band(p)}
        for p in datum.points
    ]
    x = [p["x"] for p in exp_points]
    y = [p["y"] for p in exp_points]
    out = overlay.overlay_curve(
        "b1_mrt_overlay_test", x, y, exp_points=exp_points,
        xlabel="t", ylabel="seeded-mode amplitude", x_unit=datum.x_unit,
        unit=datum.unit, outdir=str(tmp_path))
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0
