"""
Red-first component test for the B2 multi-mode MRT curve anchor (issue [VA4]/#143,
PRD #138, ADR-0008).

This is a *component* test, not a physics-sim test: it exercises the exact curve-anchor
dispatch the B2 ``_cpu`` benchmark (``cylindrical/test_verify_maglif_mmrt_cpu.py``) wires
into -- without running a simulation, so it stays in the fast CPU component band. It
proves
both dispatch branches the benchmark uses:

* the **pending** branch -- the committed McBride-2012 datum has no digitized points yet
  (the figure is paywalled; #143), so it must report ``is_pending`` and drive
  ``scorecard.record_pending`` (never a silent pass, per ADR-0008);
* the **populated** branch -- once a curve is digitized into the same schema, the oracle
  compares the sim observable against it and the verdict's ``worst_point`` feeds
  ``scorecard.record_result`` as a non-binding row (the live path once #122 digitizes the
  figure).

Oracle: Layer 1 -- literature (experiment), via ``GroundTruthOracle``. Auto-collected by
run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import pytest

import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay

B2_BENCH = "B2"
B2_DATUM = "multimode_amplitude_growth"


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


def _pt(x, y, exp=0.2, dig=0.1, kind="absolute"):
    """One experimental curve point with its per-point error bars."""
    return {
        "x": x, "y": y,
        "experimental_error": {"value": exp, "kind": kind, "basis": "test"},
        "digitization_error": {"value": dig, "kind": kind, "basis": "test"},
    }


def _populated_b2_curve(points):
    """Build a populated B2-schema ``kind: curve`` datum (synthetic digitization)."""
    return gto.Datum.from_dict(B2_BENCH, {
        "id": B2_DATUM,
        "observable": "multi-mode MRT amplitude growth (dominant-mode envelope) vs time",
        "kind": "curve",
        "status": "digitized",
        "unit": "dimensionless",
        "x_unit": "ns",
        "confidence": "medium",
        "extraction_method": "digitized (WebPlotDigitizer) -- synthetic test fixture",
        "oracle_kind": "experiment",
        "source": {"paper": "McBride 2012 (synthetic test)", "figure": "Fig. 3",
                   "doi": "10.1103/PhysRevLett.109.135004"},
        "points": points,
    })


def test_b2_committed_datum_is_pending_cpu():
    """The committed McBride-2012 B2 curve datum is wired with full provenance but still
    awaiting digitization, so the benchmark must take the pending branch (no fabricated
    points, per ADR-0008)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B2_BENCH, B2_DATUM)
    assert datum.kind == "curve"
    assert datum.is_pending, "B2 curve must stay pending until the figure is digitized"
    # Comparing a pending curve raises rather than fabricating a verdict.
    with pytest.raises(gto.PendingDatumError):
        oracle.compare(B2_BENCH, B2_DATUM, ([0.0, 1.0], [0.0, 1.0]))


def test_b2_pending_branch_records_pending_cpu():
    """Pending datum -> recorded as an explicit PENDING scorecard row (never a pass)."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get(B2_BENCH, B2_DATUM)
    assert datum.is_pending
    row = scorecard.record_pending(B2_BENCH, B2_DATUM, issue=122)
    assert row.verdict == scorecard.PENDING
    assert row.verdict != scorecard.PASS
    assert "122" in str(row)
    assert row.sim is None and row.experiment is None
    assert len(scorecard.rows()) == 1


def test_b2_populated_branch_compares_and_records_non_binding_cpu():
    """Populated datum -> oracle compares the sim curve and the worst point is recorded as
    a non-binding scorecard row (the live path once #122 digitizes the McBride figure)."""
    datum = _populated_b2_curve([_pt(0.0, 1.0), _pt(1.0, 2.0), _pt(2.0, 3.0)])
    assert not datum.is_pending
    # Sim curve A(t) = 1 + t matches the digitized points exactly (well inside band).
    x_sim = [0.0, 0.5, 1.0, 1.5, 2.0]
    y_sim = [1.0 + x for x in x_sim]
    res = datum.compare((x_sim, y_sim))
    assert res.n_compared == 3
    assert res.passed
    row = scorecard.record_result(res.worst_point, binding=False)
    assert row.binding is False
    assert row.verdict == scorecard.PASS
    assert len(scorecard.rows()) == 1


def test_b2_curve_overlay_pending_writes_png_cpu(tmp_path):
    """The pending curve overlay (sim growth curve + #122 annotation) renders to a PNG."""
    x = [0.0, 0.2, 0.4, 0.6]
    y = [0.02, 0.05, 0.12, 0.30]
    out = overlay.overlay_curve(
        "b2_mmrt_overlay_test", x, y, exp_points=None, pending_issue=122,
        xlabel="t", ylabel="multi-mode roughness", outdir=str(tmp_path))
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0
