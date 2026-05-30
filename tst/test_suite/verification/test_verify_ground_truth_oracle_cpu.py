"""
GroundTruthOracle verification component (issue [A0]/#107, ADR-0008).

This is a *component* test, not a physics-sim test: it exercises the Layer-1 ground-truth
oracle that the Ellison 1-4 MagLIF benchmark replications (#119-#122, Phase C) compare
their simulation observables against. The oracle loads the committed, provenance-tagged
experimental data directory (tst/test_suite/verification/ground_truth/), validates that
every datum carries its citation (source paper + figure + DOI + extraction method), and
compares a simulation observable against an experimental datum within a tolerance band =
max(experimental error, digitization error). The experiment is the oracle; FLASH/LASNEX/
HYDRA values, where present, are stored only as secondary reference and never as the
binding ground truth.

Oracle: Layer 1 -- literature (experiment).  The ground truth is the digitized/stated
experimental data from the primary Sandia papers (Sinars PoP 18,056301 (2011);
McBride PRL 109,135004 (2012) / PoP 20,056309 (2013); Knapp PoP 24,042708 (2017);
Knapp PoP 27,092707 (2020)), per ADR-0008.  This module verifies the oracle's load /
provenance / tolerance-band / compare logic against that committed data; it has no
simulation and no harness.verify baseline.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import copy

import pytest

import test_suite.verification.ground_truth_oracle as gto

# The four Ellison benchmarks the oracle must cover (ids per PRD #106 / arXiv:2504.10760).
ELLISON_BENCHMARKS = ("B1", "B2", "B3", "B4")


def _oracle():
    """Load the committed ground-truth directory (also asserts it validates on load)."""
    return gto.GroundTruthOracle.from_committed()


def test_loads_committed_data_for_all_four_benchmarks():
    """The committed directory loads and covers Ellison benchmarks 1-4."""
    oracle = _oracle()
    for b in ELLISON_BENCHMARKS:
        assert oracle.has_benchmark(b), f"missing committed data for benchmark {b}"
    # Every benchmark carries at least one datum.
    for b in ELLISON_BENCHMARKS:
        assert len(oracle.data(b)) >= 1, f"benchmark {b} has no datums"


def test_every_committed_datum_carries_full_provenance():
    """ADR-0008 Layer-1: every datum names paper + figure + DOI + extraction method."""
    oracle = _oracle()
    for b in ELLISON_BENCHMARKS:
        for datum in oracle.data(b):
            src = datum.source
            assert src.get("paper"), f"{b}/{datum.id}: missing source paper"
            assert src.get("figure"), f"{b}/{datum.id}: missing source figure"
            assert src.get("doi"), f"{b}/{datum.id}: missing source DOI"
            assert datum.extraction_method, f"{b}/{datum.id}: missing extraction method"
            # The oracle is the experiment, never another code.
            assert datum.oracle_kind == "experiment", (
                f"{b}/{datum.id}: oracle must be the experiment, got {datum.oracle_kind}"
            )


def test_doi_strings_are_well_formed():
    """DOIs follow the 10.NNNN/... prefix (sources were verified during extraction)."""
    oracle = _oracle()
    for b in ELLISON_BENCHMARKS:
        for datum in oracle.data(b):
            doi = datum.source["doi"]
            assert doi.startswith("10."), f"{b}/{datum.id}: malformed DOI {doi!r}"
            assert "/" in doi, f"{b}/{datum.id}: malformed DOI {doi!r}"


def test_b4_known_scalars_extracted_faithfully():
    """B4 (Knapp 2017) carries the stated scalars given in the issue, faithfully."""
    oracle = _oracle()
    min_radius = oracle.get("B4", "min_radius")
    assert min_radius.unit == "mm"
    assert min_radius.value == pytest.approx(0.45)
    peak_density = oracle.get("B4", "peak_density")
    assert peak_density.unit == "g/cc"
    assert peak_density.value == pytest.approx(10.0, rel=0.05)


def test_digitized_points_flagged_with_confidence():
    """Any committed datum carries a confidence flag; curve points say how digitized."""
    oracle = _oracle()
    for b in ELLISON_BENCHMARKS:
        for datum in oracle.data(b):
            assert datum.confidence in ("high", "medium", "low"), (
                f"{b}/{datum.id}: bad confidence {datum.confidence!r}"
            )


def test_tolerance_band_is_max_of_experimental_and_digitization_error():
    """Tolerance band = max(experimental error, digitization error), in absolute units."""
    # exp error dominates (abs 0.5 > abs 0.2)
    d1 = gto.Datum.from_dict("Bx", {
        "id": "t1", "observable": "o", "kind": "scalar", "value": 10.0, "unit": "u",
        "confidence": "high", "extraction_method": "stated scalar",
        "experimental_error": {"value": 0.5, "kind": "absolute", "basis": "test"},
        "digitization_error": {"value": 0.2, "kind": "absolute", "basis": "test"},
        "source": {"paper": "p", "figure": "f", "doi": "10.0/x"},
    })
    assert d1.tolerance_band() == pytest.approx(0.5)

    # digitization error dominates, and relative errors convert against |value|
    d2 = gto.Datum.from_dict("Bx", {
        "id": "t2", "observable": "o", "kind": "curve_point", "value": 10.0, "unit": "u",
        "confidence": "medium", "extraction_method": "digitized (WebPlotDigitizer)",
        "experimental_error": {"value": 0.01, "kind": "relative", "basis": "test"},
        "digitization_error": {"value": 0.10, "kind": "relative", "basis": "test"},
        "source": {"paper": "p", "figure": "f", "doi": "10.0/x"},
    })
    # max(0.01*10, 0.10*10) = 1.0
    assert d2.tolerance_band() == pytest.approx(1.0)


def test_compare_passes_inside_band_and_fails_outside():
    """compare() passes iff |sim - exp| <= tolerance band."""
    d = gto.Datum.from_dict("Bx", {
        "id": "t", "observable": "o", "kind": "scalar", "value": 10.0, "unit": "u",
        "confidence": "high", "extraction_method": "stated scalar",
        "experimental_error": {"value": 1.0, "kind": "absolute", "basis": "test"},
        "digitization_error": {"value": 0.0, "kind": "absolute", "basis": "test"},
        "source": {"paper": "p", "figure": "f", "doi": "10.0/x"},
    })
    inside = d.compare(10.7)
    assert inside.passed and inside.deviation == pytest.approx(0.7)
    edge = d.compare(11.0)
    assert edge.passed, "value exactly on the band edge must pass"
    outside = d.compare(11.5)
    assert not outside.passed
    assert "outside" in str(outside).lower() or "fail" in str(outside).lower()


def test_oracle_compare_routes_through_committed_datum():
    """oracle.compare(benchmark, id, value) delegates to the named committed datum."""
    oracle = _oracle()
    res = oracle.compare("B4", "min_radius", 0.45)
    assert res.passed
    assert res.benchmark == "B4" and res.datum_id == "min_radius"


def test_pending_digitization_datum_cannot_be_compared():
    """A not-yet-digitized datum has no value; comparing it is an error, not a pass."""
    pend = gto.Datum.from_dict("Bx", {
        "id": "p", "observable": "growth curve", "kind": "curve",
        "status": "pending_digitization", "confidence": "low",
        "extraction_method": "pending_digitization (deferred to #119)",
        "source": {"paper": "p", "figure": "Fig. 4", "doi": "10.0/x"},
    })
    assert pend.is_pending
    with pytest.raises(gto.PendingDatumError):
        pend.compare(1.0)


def test_missing_provenance_field_raises_on_load():
    """A datum missing its DOI (or figure/method) is a provenance error, not silent."""
    good = {
        "id": "g", "observable": "o", "kind": "scalar", "value": 1.0, "unit": "u",
        "confidence": "high", "extraction_method": "stated scalar",
        "experimental_error": {"value": 0.1, "kind": "absolute", "basis": "test"},
        "digitization_error": {"value": 0.0, "kind": "absolute", "basis": "test"},
        "source": {"paper": "p", "figure": "f", "doi": "10.0/x"},
    }
    # Sanity: the good datum loads.
    gto.Datum.from_dict("Bx", good)

    for missing in ("doi", "figure"):
        bad = copy.deepcopy(good)
        del bad["source"][missing]
        with pytest.raises(gto.ProvenanceError):
            gto.Datum.from_dict("Bx", bad)

    bad_method = copy.deepcopy(good)
    del bad_method["extraction_method"]
    with pytest.raises(gto.ProvenanceError):
        gto.Datum.from_dict("Bx", bad_method)


def test_secondary_code_reference_is_not_the_oracle():
    """FLASH/HYDRA/LASNEX values may be stored, but never as the binding oracle."""
    d = gto.Datum.from_dict("Bx", {
        "id": "t", "observable": "o", "kind": "scalar", "value": 10.0, "unit": "u",
        "confidence": "high", "extraction_method": "stated scalar",
        "experimental_error": {"value": 1.0, "kind": "absolute", "basis": "test"},
        "digitization_error": {"value": 0.0, "kind": "absolute", "basis": "test"},
        "source": {"paper": "p", "figure": "f", "doi": "10.0/x"},
        "secondary_reference": {"code": "FLASH", "value": 9.7},
    })
    assert d.oracle_kind == "experiment"
    assert d.secondary_reference["code"] == "FLASH"
    # The binding value is still the experiment.
    assert d.value == pytest.approx(10.0)


# --------------------------------------------------------------------------------------
# Curve-comparison engine ([VA2]/#141): the substrate the growth-curve benchmarks
# (B1 single-mode MRT #120, B2 multi-mode MRT #122) compare their amplitude-vs-time
# curves against. The oracle interpolates the simulation curve onto the experiment's
# abscissa over the overlapping support, tests each point within its own band, and
# returns per-point results + an aggregate pass/fail. No benchmark is wired here; these
# tests prove the comparison logic in isolation against synthetic curves.
# --------------------------------------------------------------------------------------

def _curve_datum(points, *, status=None, drop_source_field=None, x_unit="ns",
                 unit="um", confidence="medium"):
    """Build a ``kind: curve`` datum dict (optionally pending / provenance-broken)."""
    raw = {
        "id": "growth_curve",
        "observable": "seeded MRT amplitude vs time",
        "kind": "curve",
        "unit": unit,
        "x_unit": x_unit,
        "confidence": confidence,
        "extraction_method": "digitized (WebPlotDigitizer)",
        "oracle_kind": "experiment",
        "source": {"paper": "p", "figure": "Fig. 4", "doi": "10.0/x"},
        "points": points,
    }
    if status is not None:
        raw["status"] = status
    if drop_source_field is not None:
        del raw["source"][drop_source_field]
    return gto.Datum.from_dict("Bcurve", raw)


def _pt(x, y, exp=0.2, dig=0.1, kind="absolute"):
    """One experimental curve point with its per-point error bars."""
    return {
        "x": x, "y": y,
        "experimental_error": {"value": exp, "kind": kind, "basis": "test"},
        "digitization_error": {"value": dig, "kind": kind, "basis": "test"},
    }


def test_curve_datum_with_points_is_not_pending():
    """A curve datum carrying digitized points is comparable (value lives in points)."""
    d = _curve_datum([_pt(1.0, 2.0), _pt(2.0, 4.0)])
    assert not d.is_pending
    assert d.kind == "curve"


def test_curve_in_band_at_every_point_passes():
    """Sim curve within each point's band over the whole support -> aggregate PASS."""
    exp_pts = [_pt(1.0, 1.0), _pt(2.0, 2.0), _pt(3.0, 3.0)]  # band = max(0.2,0.1)=0.2
    d = _curve_datum(exp_pts)
    # Sim hugs the experiment (offset 0.1 < 0.2 band) on a denser grid.
    x_sim = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    y_sim = [x + 0.1 for x in x_sim]
    res = d.compare((x_sim, y_sim))
    assert res.passed
    assert res.n_compared == 3
    assert res.n_failed == 0
    assert "pass" in str(res).lower()


def test_curve_out_of_band_on_any_point_fails():
    """A single point outside its band fails the whole curve verdict."""
    exp_pts = [_pt(1.0, 1.0), _pt(2.0, 2.0), _pt(3.0, 3.0)]  # band 0.2 each
    d = _curve_datum(exp_pts)
    # Within band everywhere except a 1.0 spike at x=2.
    x_sim = [1.0, 2.0, 3.0]
    y_sim = [1.0, 3.0, 3.0]
    res = d.compare((x_sim, y_sim))
    assert not res.passed
    assert res.n_failed == 1
    assert res.n_compared == 3
    assert "fail" in str(res).lower() or "outside" in str(res).lower()


def test_curve_interpolates_sim_onto_experimental_abscissa():
    """Sim is sampled (linearly interpolated) at the experiment's x, not its own grid."""
    exp_pts = [_pt(1.0, 1.0), _pt(2.0, 2.0)]
    d = _curve_datum(exp_pts)
    # Sim is the line y = x sampled only at the endpoints; interpolation must recover
    # y(1.5) etc. Here we read the interpolated sim value back off the point results.
    x_sim = [0.0, 4.0]
    y_sim = [0.0, 4.0]
    res = d.compare((x_sim, y_sim))
    by_x = {p.x: p.sim_value for p in res.point_results}
    assert by_x[1.0] == pytest.approx(1.0)
    assert by_x[2.0] == pytest.approx(2.0)
    assert res.passed


def test_curve_compares_only_overlapping_support():
    """Experimental points outside the sim x-range are not compared (no extrapolation)."""
    exp_pts = [_pt(1.0, 1.0), _pt(2.0, 2.0), _pt(3.0, 3.0), _pt(4.0, 4.0)]
    d = _curve_datum(exp_pts)
    # Sim only covers x in [1.5, 3.2]: only the x=2 and x=3 points overlap.
    x_sim = [1.5, 2.0, 2.5, 3.0, 3.2]
    y_sim = [x for x in x_sim]
    res = d.compare((x_sim, y_sim))
    assert res.n_compared == 2
    assert {p.x for p in res.point_results} == {2.0, 3.0}
    assert res.passed


def test_curve_per_point_band_is_max_exp_dig_relative():
    """Per-point band = max(experimental, digitization) error, relative scales by |y|."""
    # exp 1% of 100 = 1.0; dig 5% of 100 = 5.0 -> band 5.0 at the y=100 point.
    d = _curve_datum([_pt(1.0, 100.0, exp=0.01, dig=0.05, kind="relative")])
    res = d.compare(([0.0, 2.0], [100.0, 100.0]))  # sim = 100 at x=1 after interp
    p = res.point_results[0]
    assert p.band == pytest.approx(5.0)
    # 4.5 deviation is inside band 5.0; 5.5 is outside.
    inside = d.compare(([0.0, 2.0], [104.5, 104.5]))
    assert inside.passed
    outside = d.compare(([0.0, 2.0], [105.5, 105.5]))
    assert not outside.passed


def test_pending_curve_datum_cannot_be_compared():
    """A curve flagged pending_digitization raises rather than fabricating a verdict."""
    d = _curve_datum([], status="pending_digitization")
    assert d.is_pending
    with pytest.raises(gto.PendingDatumError):
        d.compare(([0.0, 1.0], [0.0, 1.0]))


def test_curve_missing_provenance_raises_on_load():
    """A curve datum missing its DOI is a provenance error, like a scalar datum."""
    with pytest.raises(gto.ProvenanceError):
        _curve_datum([_pt(1.0, 1.0)], drop_source_field="doi")


def test_curve_result_stringifies_to_a_legible_verdict():
    """The aggregate result reads cleanly in an assertion message / the scorecard."""
    d = _curve_datum([_pt(1.0, 1.0), _pt(2.0, 2.0)])
    res = d.compare(([1.0, 2.0], [1.0, 5.0]))
    s = str(res)
    assert "Bcurve" in s and "growth_curve" in s
    assert "2" in s  # mentions how many points were compared
    assert "FAIL" in s
