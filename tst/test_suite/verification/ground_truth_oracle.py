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

import numpy as np

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
                 deviation, band, unit, x=None, x_unit=None):
        self.benchmark = benchmark
        self.datum_id = datum_id
        self.observable = observable
        self.sim_value = sim_value
        self.exp_value = exp_value
        self.deviation = deviation
        self.band = band
        self.unit = unit
        # For a curve point: its abscissa (and abscissa unit); None for a scalar.
        self.x = x
        self.x_unit = x_unit
        self.passed = deviation <= band

    def __str__(self):
        verdict = "PASS (within band)" if self.passed else "FAIL (outside band)"
        at = "" if self.x is None else f"@x={self.x:g}{self.x_unit or ''} "
        return (
            f"[{self.benchmark}/{self.datum_id}] {self.observable}: {at}"
            f"sim={self.sim_value:g} vs exp={self.exp_value:g} {self.unit} "
            f"|Δ|={self.deviation:.3g} band=±{self.band:.3g} -> {verdict}"
        )


class CurveComparisonResult:
    """Aggregate outcome of comparing a simulation curve against an experimental curve.

    The oracle interpolates the simulation curve onto the experiment's abscissa over the
    overlapping support and tests each compared point within its own per-point band. The
    aggregate verdict passes iff every compared point is within its band (and at least one
    point was compared -- a curve with no overlapping support is *not* a vacuous pass).
    """

    def __init__(self, benchmark, datum_id, observable, unit, x_unit, point_results,
                 support):
        self.benchmark = benchmark
        self.datum_id = datum_id
        self.observable = observable
        self.unit = unit
        self.x_unit = x_unit
        self.point_results = list(point_results)
        # (x_lo, x_hi) overlapping support actually compared.
        self.support = support
        self.n_compared = len(self.point_results)
        self.n_failed = sum(1 for p in self.point_results if not p.passed)
        self.passed = self.n_compared > 0 and self.n_failed == 0

    @property
    def worst_point(self):
        """The compared point with the largest band-relative deviation (None if empty).

        A single ``ComparisonResult`` suitable for ``scorecard.record_result``.
        """
        if not self.point_results:
            return None

        def severity(p):
            return p.deviation / p.band if p.band > 0 else float("inf")

        return max(self.point_results, key=severity)

    def __str__(self):
        verdict = "PASS (within band)" if self.passed else "FAIL (outside band)"
        if self.n_compared == 0:
            verdict = "FAIL (no overlapping support)"
        lo, hi = self.support if self.support else (float("nan"), float("nan"))
        xu = self.x_unit or ""
        return (
            f"[{self.benchmark}/{self.datum_id}] {self.observable}: "
            f"{self.n_compared} pts compared over x=[{lo:g},{hi:g}]{xu}, "
            f"{self.n_failed} outside band -> {verdict}"
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
        # Curve datums carry their value as a list of (x, y, error-bars) points instead
        # of a single scalar ``value``; the abscissa has its own unit.
        self.points = raw.get("points")
        self.x_unit = raw.get("x_unit")
        self._validate()

    @property
    def is_curve(self):
        """True iff this datum is a digitized curve (compared point-by-point)."""
        return self.kind == "curve"

    @property
    def is_pending(self):
        """True iff the value is not yet digitized (no binding comparison possible)."""
        if self.status == "pending_digitization":
            return True
        if self.is_curve:
            # A curve's "value" lives in its points list.
            return not self.points
        return self.value is None

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
            if self.is_curve:
                self._validate_curve(where)
            else:
                for field in REQUIRED_VALUE_FIELDS:
                    if self.raw.get(field) is None:
                        raise ProvenanceError(
                            f"{where}: extracted datum is missing '{field}'"
                        )

    def _validate_curve(self, where):
        """A digitized (non-pending) curve carries y/x units and complete points."""
        if not self.unit:
            raise ProvenanceError(f"{where}: curve datum is missing 'unit' (y axis)")
        if not self.x_unit:
            raise ProvenanceError(f"{where}: curve datum is missing 'x_unit' (abscissa)")
        if not self.points:
            raise ProvenanceError(f"{where}: extracted curve datum has no 'points'")
        for i, pt in enumerate(self.points):
            for field in ("x", "y", "experimental_error", "digitization_error"):
                if pt.get(field) is None:
                    raise ProvenanceError(
                        f"{where}: curve point {i} is missing '{field}'"
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

    def compare(self, sim_observable):
        """Compare a simulation observable against this datum within its tolerance.

        For a scalar/curve_point datum, ``sim_observable`` is a single value. For a
        ``kind: curve`` datum it is the simulation curve ``(x_sim, y_sim)`` (two equal-
        length sequences), and the return is a :class:`CurveComparisonResult`.
        """
        if self.is_pending:
            raise PendingDatumError(
                f"{self.benchmark}/{self.id}: cannot compare -- value pending "
                f"digitization ({self.extraction_method})"
            )
        if self.is_curve:
            return self._compare_curve(sim_observable)
        sim_value = float(sim_observable)
        deviation = abs(sim_value - float(self.value))
        return ComparisonResult(
            self.benchmark, self.id, self.observable, sim_value, float(self.value),
            deviation, self.tolerance_band(), self.unit,
        )

    def _point_band(self, pt):
        """Per-point tolerance band = max(experimental, digitization) error, absolute."""
        exp = _error_to_absolute(pt["experimental_error"], pt["y"])
        dig = _error_to_absolute(pt["digitization_error"], pt["y"])
        return max(exp, dig)

    def _compare_curve(self, sim_observable):
        """Interpolate the sim curve onto the experiment's abscissa and test each point.

        Only experimental points inside the overlapping support
        ``[max(min x), min(max x)]`` are compared -- the simulation is never
        extrapolated past its own domain. Returns a :class:`CurveComparisonResult` whose
        aggregate passes iff every compared point lies within its own band.
        """
        x_sim, y_sim = sim_observable
        x_sim = np.asarray(x_sim, dtype=float)
        y_sim = np.asarray(y_sim, dtype=float)
        if x_sim.ndim != 1 or x_sim.shape != y_sim.shape or x_sim.size < 1:
            raise ValueError(
                f"{self.benchmark}/{self.id}: sim curve must be two equal-length 1-D "
                f"sequences (x_sim, y_sim)"
            )
        # np.interp requires a monotone-increasing abscissa.
        order = np.argsort(x_sim)
        x_sim, y_sim = x_sim[order], y_sim[order]
        x_lo = max(float(x_sim[0]), min(float(p["x"]) for p in self.points))
        x_hi = min(float(x_sim[-1]), max(float(p["x"]) for p in self.points))

        results = []
        for pt in sorted(self.points, key=lambda p: float(p["x"])):
            xe = float(pt["x"])
            if xe < x_lo or xe > x_hi:
                continue  # outside overlapping support -- not compared (no extrapolation)
            ye = float(pt["y"])
            ys = float(np.interp(xe, x_sim, y_sim))
            band = self._point_band(pt)
            results.append(ComparisonResult(
                self.benchmark, self.id, self.observable, ys, ye,
                abs(ys - ye), band, self.unit, x=xe, x_unit=self.x_unit,
            ))
        return CurveComparisonResult(
            self.benchmark, self.id, self.observable, self.unit, self.x_unit,
            results, (x_lo, x_hi),
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
