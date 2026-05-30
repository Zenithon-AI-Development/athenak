"""
Experiment-anchoring substrate component test (issue [VA1]/#140, ADR-0008).

This is a *component* test, not a physics-sim test: it exercises the reusable plumbing the
MagLIF benchmark replications use to turn a self-regression harness into an experiment-
validation harness -- the **suite scorecard** (``verification/scorecard.py``) that records
each observable's simulation-vs-experiment verdict, and the **experiment-overlay** plotter
(``verification/experiment_overlay.py``) that draws each scalar's experimental value and
tolerance band on top of the simulation result. Both consume the ``GroundTruthOracle``
comparison results so the later curve benchmarks (#142/#143) plug straight in.

Oracle: Layer 1 -- literature (experiment), via ``GroundTruthOracle``. This module has no
simulation; it proves the substrate's gating logic (a within-band comparison records a
PASS row, an out-of-band one records a FAIL row, a not-yet-digitized datum records a
explicit PENDING row -- never a silent pass) and that the overlay plotter emits a figure.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import pytest

import test_suite.verification.ground_truth_oracle as gto
import test_suite.verification.scorecard as scorecard
import test_suite.verification.experiment_overlay as overlay


# A small in-band / out-of-band datum pair built directly (no committed file needed).
def _datum(value, exp_err, dig_err=0.0):
    return gto.Datum.from_dict("Bx", {
        "id": "obs", "observable": "test observable", "kind": "scalar",
        "value": value, "unit": "u", "confidence": "high",
        "extraction_method": "stated scalar",
        "experimental_error": {"value": exp_err, "kind": "absolute", "basis": "test"},
        "digitization_error": {"value": dig_err, "kind": "absolute", "basis": "test"},
        "source": {"paper": "p", "figure": "f", "doi": "10.0/x"},
    })


@pytest.fixture(autouse=True)
def _clean_scorecard():
    """Each test runs against an empty scorecard, then any suite rows are restored.

    The scorecard is a process-wide singleton shared with the benchmark tests; snapshot
    and restore so these component tests never clobber the real suite scorecard, whatever
    the collection order.
    """
    saved = scorecard.rows()
    scorecard.reset()
    yield
    scorecard.set_rows(saved)


def test_scorecard_starts_empty():
    assert scorecard.rows() == []


def test_in_band_comparison_records_a_pass_row():
    res = _datum(10.0, 1.0).compare(10.5)            # |Δ|=0.5 <= band 1.0
    row = scorecard.record_result(res)
    assert res.passed
    assert row.verdict == scorecard.PASS
    assert row.benchmark == "Bx" and row.observable == "test observable"
    assert row.sim == pytest.approx(10.5)
    assert row.experiment == pytest.approx(10.0)
    assert row.band == pytest.approx(1.0)
    assert len(scorecard.rows()) == 1


def test_out_of_band_comparison_records_a_fail_row():
    res = _datum(10.0, 1.0).compare(12.0)            # |Δ|=2.0 > band 1.0
    row = scorecard.record_result(res)
    assert not res.passed
    assert row.verdict == scorecard.FAIL


def test_pending_datum_records_an_explicit_pending_row_never_a_pass():
    """A not-yet-digitized anchor is reported as pending (#N), never as a pass."""
    row = scorecard.record_pending("B4", "confinement_time", issue=121)
    assert row.verdict == scorecard.PENDING
    assert row.verdict != scorecard.PASS
    assert "121" in str(row)
    assert row.sim is None and row.experiment is None


def test_reported_not_asserted_row_is_marked_non_binding():
    """A scalar the reduced run only *reports* (B3 velocity-ratio policy) is flagged."""
    res = _datum(10.0, 1.0).compare(99.0)
    binding = scorecard.record_result(res)
    reported = scorecard.record_result(res, binding=False, note="reduced surrogate")
    assert binding.binding is True
    assert reported.binding is False
    assert "report" in str(reported).lower()


def test_render_is_a_table_with_all_verdicts_and_pending_marker():
    scorecard.record_result(_datum(10.0, 1.0).compare(10.2))      # PASS
    scorecard.record_result(_datum(10.0, 1.0).compare(20.0))      # FAIL
    scorecard.record_pending("B4", "confinement_time", issue=121)  # PENDING
    table = scorecard.render()
    for token in ("Benchmark", "Observable", "Sim", "Experiment", "Band", "Verdict"):
        assert token in table
    assert scorecard.PASS in table
    assert scorecard.FAIL in table
    assert scorecard.PENDING in table
    assert "#121" in table


def test_oracle_pending_b4_confinement_time_round_trips_to_pending_row():
    """The committed B4 confinement_time is still pending -> compare raises, not pass."""
    oracle = gto.GroundTruthOracle.from_committed()
    datum = oracle.get("B4", "confinement_time")
    assert datum.is_pending
    with pytest.raises(gto.PendingDatumError):
        oracle.compare("B4", "confinement_time", 1.0)
    row = scorecard.record_pending("B4", "confinement_time", issue=121)
    assert row.verdict == scorecard.PENDING


def test_overlay_emits_a_figure_with_experiment_band_and_secondary(tmp_path):
    """The overlay plotter draws sim series + each scalar's exp value/band (+ FLASH)."""
    x = [0.0, 0.1, 0.2, 0.3, 0.4]
    series = {"R_if": [0.4, 0.3, 0.2, 0.15, 0.18],
              "rho_fuel": [0.01, 0.5, 2.0, 5.0, 3.0]}
    overlays = {
        "R_if": {"exp_value": 0.16, "band": 0.02, "unit": "mm", "sim_value": 0.15},
        "rho_fuel": {"exp_value": 4.5, "band": 0.9, "unit": "g/cc", "sim_value": 5.0,
                     "secondary": {"code": "FLASH", "value": 4.0}},
    }
    out = overlay.overlay_scalars(
        "scorecard_overlay_selftest", x, series, overlays,
        xlabel="t (code units)", title="overlay self-test",
        outdir=str(tmp_path),
    )
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0


def test_overlay_handles_a_scalar_with_no_secondary_reference(tmp_path):
    out = overlay.overlay_scalars(
        "scorecard_overlay_nosecondary", [0.0, 1.0], {"q": [1.0, 2.0]},
        {"q": {"exp_value": 1.5, "band": 0.3, "unit": "x", "sim_value": 2.0}},
        outdir=str(tmp_path),
    )
    assert os.path.isfile(out)
