"""
GroundTruthOracle -- Layer-1 experimental ground truth for the Ellison MagLIF benchmarks.

This is the verification component the Phase-C replications (#119-#122) compare their
simulation observables against. Per ADR-0008, a verification test is "grounded" only when
it diffs against a ground truth derived *independently of AthenaK's own output*. For the
four Ellison benchmarks (PRD #106 / arXiv:2504.10760) that independent oracle is the
**experiment** -- digitized/stated data from the primary Sandia papers -- never another
code: FLASH/LASNEX/HYDRA values, where stored, are secondary reference only.

The oracle loads a committed, provenance-tagged data directory
(``tst/test_suite/verification/ground_truth/``: one JSON file per benchmark), validates
that every datum carries its citation (source paper + figure + DOI + extraction method),
and compares a simulation observable against an experimental datum within a tolerance
band::

    tolerance_band = max(experimental_error, digitization_error)   # in absolute units

Design notes
------------
* The data directory is anchored to *this file* (not the working directory) so it is found
  regardless of where the test runs from (the suite runs from ``tst/build*/src``).
* Provenance is validated eagerly on load: a datum missing any mandated citation field
  is a ``ProvenanceError``, not a silent pass -- the whole point of ADR-0008.
* A not-yet-digitized observable is committed as a ``pending_digitization`` datum carrying
  full provenance (which paper/figure it will come from) but no value; comparing it raises
  ``PendingDatumError`` rather than fabricating a pass. The Phase-C benchmark issues fill
  these in as they digitize the curves.
* Error bars carry a ``basis`` string stating where each bar came from (published value,
  inferred from significant figures, or a placeholder for a downstream issue), so no
  tolerance is fabricated without saying so.
"""

import json
import os

# Anchor the committed data directory to this module so it is found from any CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(_THIS_DIR, "ground_truth")

# Citation fields ADR-0008 mandates on every datum's ``source`` block.
REQUIRED_SOURCE_FIELDS = ("paper", "figure", "doi")
# Fields a *comparable* (non-pending) datum needs beyond provenance.
REQUIRED_VALUE_FIELDS = ("value", "unit", "experimental_error", "digitization_error")

VALID_CONFIDENCE = ("high", "medium", "low")


class ProvenanceError(ValueError):
    """A committed datum is missing a mandated citation/provenance field (ADR-0008)."""


class PendingDatumError(RuntimeError):
    """Attempted to compare against a datum whose value is not yet digitized."""


def _error_to_absolute(spec, value):
    """Convert an {value, kind} error spec to an absolute magnitude against ``value``."""
    mag = float(spec["value"])
    kind = spec.get("kind", "absolute")
    if kind == "relative":
        return abs(mag) * abs(float(value))
    if kind == "absolute":
        return abs(mag)
    raise ProvenanceError(f"unknown error kind {kind!r} (expected absolute|relative)")


class ComparisonResult:
    """Outcome of comparing a simulation observable against an experimental datum."""

    def __init__(self, benchmark, datum_id, observable, sim_value, exp_value,
                 deviation, band, unit):
        self.benchmark = benchmark
        self.datum_id = datum_id
        self.observable = observable
        self.sim_value = sim_value
        self.exp_value = exp_value
        self.deviation = deviation
        self.band = band
        self.unit = unit
        self.passed = deviation <= band

    def __str__(self):
        verdict = "PASS (within band)" if self.passed else "FAIL (outside band)"
        return (
            f"[{self.benchmark}/{self.datum_id}] {self.observable}: "
            f"sim={self.sim_value:g} vs exp={self.exp_value:g} {self.unit} "
            f"|Δ|={self.deviation:.3g} band=±{self.band:.3g} -> {verdict}"
        )


class Datum:
    """One provenance-tagged experimental observable the simulation must reproduce."""

    def __init__(self, benchmark, raw):
        self.benchmark = benchmark
        self.raw = raw
        self.id = raw.get("id")
        self.observable = raw.get("observable")
        self.kind = raw.get("kind")
        self.confidence = raw.get("confidence")
        self.extraction_method = raw.get("extraction_method")
        self.source = raw.get("source", {})
        self.status = raw.get("status", "extracted")
        # The oracle is always the experiment; another code may be a secondary reference.
        self.oracle_kind = raw.get("oracle_kind", "experiment")
        self.secondary_reference = raw.get("secondary_reference")
        self.value = raw.get("value")
        self.unit = raw.get("unit")
        self._validate()

    @property
    def is_pending(self):
        """True iff the value is not yet digitized (no binding comparison possible)."""
        return self.status == "pending_digitization" or self.value is None

    def _validate(self):
        where = f"{self.benchmark}/{self.id or '<no-id>'}"
        if not self.id:
            raise ProvenanceError(f"{self.benchmark}: a datum is missing its id")
        if self.confidence not in VALID_CONFIDENCE:
            raise ProvenanceError(
                f"{where}: confidence must be one of {VALID_CONFIDENCE}, "
                f"got {self.confidence!r}"
            )
        if not self.extraction_method:
            raise ProvenanceError(f"{where}: missing extraction_method")
        if self.oracle_kind != "experiment":
            raise ProvenanceError(
                f"{where}: oracle must be the experiment (got {self.oracle_kind!r}); "
                f"another code belongs in 'secondary_reference', not as the oracle"
            )
        for field in REQUIRED_SOURCE_FIELDS:
            if not self.source.get(field):
                raise ProvenanceError(f"{where}: source is missing '{field}'")
        # Comparable (extracted) datums must also carry value + units + error bars.
        if not self.is_pending:
            for field in REQUIRED_VALUE_FIELDS:
                if self.raw.get(field) is None:
                    raise ProvenanceError(
                        f"{where}: extracted datum is missing '{field}'"
                    )

    def tolerance_band(self):
        """Absolute tolerance band = max(experimental error, digitization error)."""
        if self.is_pending:
            raise PendingDatumError(
                f"{self.benchmark}/{self.id}: value pending digitization"
            )
        exp = _error_to_absolute(self.raw["experimental_error"], self.value)
        dig = _error_to_absolute(self.raw["digitization_error"], self.value)
        return max(exp, dig)

    def compare(self, sim_value):
        """Compare a simulation observable against this datum within its tolerance."""
        if self.is_pending:
            raise PendingDatumError(
                f"{self.benchmark}/{self.id}: cannot compare -- value pending "
                f"digitization ({self.extraction_method})"
            )
        sim_value = float(sim_value)
        deviation = abs(sim_value - float(self.value))
        return ComparisonResult(
            self.benchmark, self.id, self.observable, sim_value, float(self.value),
            deviation, self.tolerance_band(), self.unit,
        )

    @classmethod
    def from_dict(cls, benchmark, raw):
        """Build (and validate) a single datum from its raw dict."""
        return cls(benchmark, raw)


class GroundTruthOracle:
    """Loads and serves the committed Layer-1 experimental ground truth for B1-B4."""

    def __init__(self, benchmarks):
        # benchmarks: {benchmark_id: {"meta": {...}, "data": [Datum, ...]}}
        self._benchmarks = benchmarks

    # -- loading ----------------------------------------------------------------
    @classmethod
    def from_committed(cls):
        """Load the committed ground-truth directory anchored next to this module."""
        return cls.from_directory(GROUND_TRUTH_DIR)

    @classmethod
    def from_directory(cls, directory):
        """Load every ``*.json`` benchmark file in ``directory`` (provenance checked)."""
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"ground-truth directory not found: {directory}")
        benchmarks = {}
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".json") or fname.startswith("_"):
                continue
            with open(os.path.join(directory, fname), "r") as f:
                doc = json.load(f)
            bid = doc["benchmark"]
            data = [Datum(bid, raw) for raw in doc.get("data", [])]
            ids = [d.id for d in data]
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                raise ProvenanceError(f"{bid}: duplicate datum id(s) {sorted(dupes)}")
            benchmarks[bid] = {
                "meta": {k: v for k, v in doc.items() if k != "data"},
                "data": data,
            }
        return cls(benchmarks)

    # -- access -----------------------------------------------------------------
    def has_benchmark(self, benchmark):
        return benchmark in self._benchmarks

    def benchmarks(self):
        return tuple(sorted(self._benchmarks))

    def meta(self, benchmark):
        return self._benchmarks[benchmark]["meta"]

    def data(self, benchmark):
        """All datums for ``benchmark``."""
        return list(self._benchmarks[benchmark]["data"])

    def get(self, benchmark, datum_id):
        """Fetch a single datum by benchmark + id."""
        for d in self._benchmarks[benchmark]["data"]:
            if d.id == datum_id:
                return d
        raise KeyError(f"no datum {datum_id!r} in benchmark {benchmark!r}")

    # -- comparison -------------------------------------------------------------
    def compare(self, benchmark, datum_id, sim_value):
        """Compare a simulation observable against the named committed datum."""
        return self.get(benchmark, datum_id).compare(sim_value)
