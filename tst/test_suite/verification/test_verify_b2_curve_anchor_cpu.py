"""Red-first component tests for the B2 multi-mode MRT curve anchor (issue #143
[VA4], PRD #138, ADR-0008).

These exercise the exact curve-anchor dispatch the B2 `_cpu` benchmark
(`cylindrical/test_verify_maglif_mmrt_cpu.py`) wires into — without running a
simulation, so they stay in the fast CPU component band:

* the **pending** branch: the committed McBride-2012 datum has no digitized
  points yet (paywalled figure; see #143), so it must report `is_pending` and
  drive `scorecard.record_pending`;
* the **populated** branch: once a curve is digitized into the same schema, the
  oracle compares the sim observable against it and the verdict's `worst_point`
  feeds `scorecard.record_result` as a non-binding row.

Run via `run_test_suite.py --cpu` (collected by the `_cpu` suffix).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verification import ground_truth_oracle as gto  # noqa: E402
from verification import scorecard  # noqa: E402
from verification import experiment_overlay as eo  # noqa: E402

B2_BENCH = "b2_multimode_mrt"
B2_DATUM = "mmrt_growth_curve"


def _reset():
    scorecard.set_rows([])


def _write_datum(tmp_path, payload):
    p = tmp_path / "b2_gt.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_b2_committed_datum_is_pending_cpu():
    """The committed McBride-2012 B2 curve datum is wired and provenance-rich
    but still awaiting digitization, so the benchmark must take the pending
    branch (no fabricated points, per ADR-0008)."""
    oracle = gto.GroundTruthOracle()
    datum = oracle.get_datum(B2_BENCH, B2_DATUM)
    assert datum.kind == "curve"
    assert datum.is_pending, "B2 curve must stay pending until digitized"
    assert datum.x_unit == "ns"
    assert datum.unit == "dimensionless"


def test_b2_pending_branch_records_pending_cpu():
    """Pending datum -> recorded as a PENDING scorecard row tagged with #143."""
    _reset()
    saved = scorecard.get_rows()
    try:
        oracle = gto.GroundTruthOracle()
        datum = oracle.get_datum(B2_BENCH, B2_DATUM)
        assert datum.is_pending
        scorecard.record_pending(B2_BENCH, B2_DATUM, issue=143)
        rows = scorecard.get_rows()
        assert len(rows) == 1
        assert rows[0].pending
        assert rows[0].benchmark == B2_BENCH
        assert rows[0].datum_id == B2_DATUM
        assert rows[0].issue == 143
    finally:
        scorecard.set_rows(saved)


def test_b2_populated_branch_compares_cpu(tmp_path):
    """Populated datum -> oracle compares the sim curve and the worst point is
    recorded as a non-binding scorecard row (the live path once #143 digitizes
    the McBride-2012 figure)."""
    _reset()
    saved = scorecard.get_rows()
    try:
        payload = {
            "benchmark": B2_BENCH,
            "source": {"citation": "McBride 2012 (synthetic test)",
                       "doi": "x", "figure": "Fig. 3"},
            "datums": {
                B2_DATUM: {
                    "kind": "curve",
                    "status": "digitized",
                    "x_unit": "ns",
                    "unit": "dimensionless",
                    "points": [
                        {"x": 0.0, "y": 1.0, "experimental_error": 0.2,
                         "digitization_error": 0.0},
                        {"x": 1.0, "y": 2.0, "experimental_error": 0.2,
                         "digitization_error": 0.0},
                        {"x": 2.0, "y": 3.0, "experimental_error": 0.2,
                         "digitization_error": 0.0},
                    ],
                }
            },
        }
        path = _write_datum(tmp_path, payload)
        oracle = gto.GroundTruthOracle(path=path)
        datum = oracle.get_datum(B2_BENCH, B2_DATUM)
        assert not datum.is_pending
        # sim curve A(t) = 1 + t exactly matches the digitized points
        x_sim = np.linspace(0.0, 2.0, 21)
        y_sim = 1.0 + x_sim
        res = oracle.compare(B2_BENCH, B2_DATUM, (x_sim, y_sim))
        assert res.n_compared == 3
        assert res.passed
        scorecard.record_result(res.worst_point, binding=False)
        rows = scorecard.get_rows()
        assert len(rows) == 1
        assert rows[0].binding is False
        assert rows[0].passed
    finally:
        scorecard.set_rows(saved)


def test_b2_curve_overlay_pending_writes_png_cpu(tmp_path):
    """The pending curve overlay (sim growth curve + #143 annotation) renders."""
    _reset()
    x = np.linspace(0.0, 0.6, 8)
    y = 0.02 * np.exp(3.0 * x)
    out = eo.overlay_curve(
        "b2_mmrt_overlay_test", x, y, exp_points=None, pending_issue=143,
        outdir=str(tmp_path))
    assert os.path.exists(out)


# (no module-level execution; pytest collects the test_* functions above)
