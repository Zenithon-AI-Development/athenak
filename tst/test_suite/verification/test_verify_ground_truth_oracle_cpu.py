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
