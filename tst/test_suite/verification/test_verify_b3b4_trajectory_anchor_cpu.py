"""
Red-first component test for the B3/B4 corrected-v3 trajectory anchors and the B4
``confinement_time`` scalar (issue [VA8]/#156, PRD #138, ADR-0008).

This is a *component* test, not a physics-sim test: it exercises the exact oracle/overlay
dispatch the B3 converging-RM anchor (``cylindrical/test_verify_maglif_rm_anchor_cpu.py``)
and the B4 ICF benchmark (``cylindrical/test_verify_maglif_icf_cpu.py``) wire into --
without running a simulation, so it stays in the fast CPU component band.

Context: the figure-traced **trajectory** point clouds shipped under the now-closed B3
(#119/#144) and B4 (#140) work were found wrong on human visual QC (2026-05-30) and were
corrected to **v3** (B3 shock: spurious upper strand culled -> 30 monotonic pts; B4
inner-radius: the 2 rebound points at r~0.54 recovered and the legend box excluded -> 6
circles).  #156 ingests those corrected v3 point clouds into the committed B3/B4 datums as
``kind: curve`` anchors, and promotes the B4 ``confinement_time`` (a stated text scalar in
Knapp 2017, 14 ns vs 16 ns 1D) out of ``pending_digitization`` into the scalar oracle.

It proves:

* the committed B3 ``rm_liner_trajectory`` / ``rm_shock_trajectory`` and the B4
  ``inner_radius_trajectory`` are populated ``kind: curve`` datums with full ADR-0008
  provenance (y/x units, per-point experimental + digitization error), no longer pending,
  and carry the v3 corrected point counts (30 / 30 / 6) -- no phantom/dropped points;
* the populated curve branch -- the oracle compares a sim trajectory against the digitized
  points and the verdict's ``worst_point`` feeds ``scorecard.record_result`` as a
  non-binding row (the reduced ``_cpu`` surrogate reports rather than asserts absolute
  trajectories; the binding absolute-SI assert is the GPU/SI run, per the reduced
  surrogate precedent B3 #144 / B4 #140 / B1 #154);
* the B4 ``confinement_time`` is **no longer pending** and the scalar oracle compares it
  (14 ns +/- band) without raising ``PendingDatumError``.

Oracle: Layer 1 -- literature (experiment), via ``GroundTruthOracle``. Auto-collected by
run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import pytest

import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

# (benchmark, datum_id, expected v3 point count) for the corrected trajectory curves.
TRAJECTORIES = [
    ("B3", "rm_liner_trajectory", 30),
    ("B3", "rm_shock_trajectory", 30),
    ("B4", "inner_radius_trajectory", 6),
]


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


@pytest.mark.parametrize("bench,datum_id,n_expect", TRAJECTORIES)
def test_v3_trajectory_datum_is_populated_cpu(bench, datum_id, n_expect):
    """Each corrected v3 trajectory is a populated ``kind: curve`` with the right point
    count (no phantom/dropped points) and is no longer pending (#156, AC2)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(bench, datum_id)
    assert datum.kind == "curve"
    assert not datum.is_pending, (
        f"{bench}/{datum_id} must be digitized + committed (not pending_digitization)"
    )
    assert len(datum.points) == n_expect, (
        f"{bench}/{datum_id} must carry exactly the {n_expect} v3 corrected points "
        f"(got {len(datum.points)})"
    )
    # A populated curve compares without raising (the oracle is wired for the sim curve).
    xe = [float(p["x"]) for p in datum.points]
    ye = [float(p["y"]) for p in datum.points]
    res = oracle.compare(bench, datum_id, (xe, ye))
    assert isinstance(res, gto.CurveComparisonResult)
    assert res.n_compared == n_expect


@pytest.mark.parametrize("bench,datum_id,n_expect", TRAJECTORIES)
def test_v3_trajectory_points_carry_full_provenance_cpu(bench, datum_id, n_expect):
    """Every v3 point carries y/x units + per-point experimental & digitization error
    (ADR-0008); the per-point band is a positive, finite real tolerance."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(bench, datum_id)
    assert datum.unit == "mm" and datum.x_unit == "ns", "trajectory needs mm vs ns units"
    for pt in datum.points:
        for field in ("x", "y", "experimental_error", "digitization_error"):
            assert pt.get(field) is not None, f"point missing {field}"
        band = datum._point_band(pt)
        assert band > 0.0


def test_v3_trajectory_compares_and_records_non_binding_cpu():
    """Populated datum -> oracle compares a sim trajectory that traces the digitized
    points and the worst point is recorded as a non-binding row (the report path)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get("B4", "inner_radius_trajectory")
    xe = [float(p["x"]) for p in datum.points]
    ye = [float(p["y"]) for p in datum.points]
    res = oracle.compare("B4", "inner_radius_trajectory", (xe, ye))
    assert res.n_compared == len(datum.points)
    assert res.passed
    row = scorecard.record_result(res.worst_point, binding=False)
    assert row.binding is False
    assert row.verdict == scorecard.PASS
    assert len(scorecard.rows()) == 1


def test_b4_confinement_time_is_no_longer_pending_cpu():
    """The committed B4 ``confinement_time`` is a stated text scalar (Knapp 2017: 14 ns vs
    16 ns 1D), no longer ``pending_digitization``; the scalar oracle compares it without
    raising ``PendingDatumError`` (#156, AC3)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get("B4", "confinement_time")
    assert datum.kind == "scalar"
    assert not datum.is_pending, "B4 confinement_time must no longer be pending (#156)"
    assert datum.value == pytest.approx(14.0)
    assert datum.unit == "ns"
    res = oracle.compare("B4", "confinement_time", 14.0)
    assert res.passed and res.band > 0.0
    row = scorecard.record_result(res, binding=False)
    assert row.verdict == scorecard.PASS


def test_v3_trajectory_overlay_writes_png_cpu(tmp_path):
    """The trajectory overlay draws the digitized v3 experimental points + per-point band
    on the sim trajectory (AC4 -- overlays reproduce the v3 figures)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get("B3", "rm_shock_trajectory")
    exp_points = [
        {"x": float(p["x"]), "y": float(p["y"]), "band": datum._point_band(p)}
        for p in datum.points
    ]
    x = [p["x"] for p in exp_points]
    y = [p["y"] for p in exp_points]
    out = overlay.overlay_curve(
        "b3_shock_trajectory_overlay_test", x, y, exp_points=exp_points,
        xlabel="t", ylabel="shock radius", x_unit=datum.x_unit,
        unit=datum.unit, outdir=str(tmp_path))
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0
