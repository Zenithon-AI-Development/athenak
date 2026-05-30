"""
Suite scorecard -- the end-of-run experiment-vs-simulation verdict table ([VA1]/#140).

Part of the quantitative-anchoring substrate (PRD #138, ADR-0008). As each MagLIF
benchmark reduces its run to an observable and compares it against the committed
experimental ground truth (``GroundTruthOracle``), it records the verdict here; at the end
of the test session a single table is emitted (printed and written to ``plots/scorecard``)
so a reader can see, at a glance, which observables reproduce the experiment, which do
not, and which anchors are still ``pending digitization`` (reported as pending, **never**
as a pass).

The registry is a process-wide singleton: pytest runs the whole device session in one
process, so every ``record_*`` call accumulates into the same table. ``conftest.py``
renders it from ``pytest_terminal_summary``. Each row carries:

  * ``benchmark`` / ``observable`` -- which datum,
  * ``sim`` / ``experiment`` / ``band`` -- the compared values and tolerance (``None``
    when the anchor is pending),
  * ``verdict`` -- ``PASS`` / ``FAIL`` / ``PENDING``,
  * ``binding`` -- whether this comparison is the test's *binding* assertion (``True``) or
    only *reported* for the record (``False``; e.g. a dimensionless/EOS- or
    scale-sensitive quantity a reduced ideal run cannot reproduce in absolute terms -- the
    B3 velocity-ratio policy), and an optional ``note``.
"""

PASS = "PASS"
FAIL = "FAIL"
PENDING = "PENDING"

# Process-wide registry of scorecard rows.
_ROWS = []


class ScorecardRow:
    """One observable's simulation-vs-experiment line in the suite scorecard."""

    def __init__(self, benchmark, observable, sim, experiment, band, unit, verdict,
                 binding=True, note=None):
        self.benchmark = benchmark
        self.observable = observable
        self.sim = sim
        self.experiment = experiment
        self.band = band
        self.unit = unit
        self.verdict = verdict
        self.binding = binding
        self.note = note

    def _fmt(self, v):
        if v is None:
            return "--"
        return f"{v:g}"

    def __str__(self):
        sim = self._fmt(self.sim)
        exp = self._fmt(self.experiment)
        band = "--" if self.band is None else f"+/-{self.band:g}"
        unit = f" {self.unit}" if self.unit and self.experiment is not None else ""
        mode = "" if self.binding else " (reported, not asserted)"
        verdict = self.verdict + mode
        note = f"  [{self.note}]" if self.note else ""
        return (
            f"[{self.benchmark}] {self.observable}: sim={sim}{unit} "
            f"exp={exp}{unit} band={band} -> {verdict}{note}"
        )


def reset():
    """Clear the registry (called per pytest session, and by component tests)."""
    _ROWS.clear()


def rows():
    """A copy of the recorded rows, in insertion order."""
    return list(_ROWS)


def set_rows(rowlist):
    """Replace the registry contents (used by component tests to save/restore state)."""
    _ROWS.clear()
    _ROWS.extend(rowlist)


def record_result(result, binding=True, note=None):
    """Record a ``GroundTruthOracle`` ComparisonResult as a PASS/FAIL row."""
    verdict = PASS if result.passed else FAIL
    row = ScorecardRow(
        result.benchmark, result.observable, result.sim_value, result.exp_value,
        result.band, result.unit, verdict, binding=binding, note=note,
    )
    _ROWS.append(row)
    return row


def record_pending(benchmark, observable, issue, note=None):
    """Record a not-yet-digitized anchor as an explicit PENDING row (never a pass).

    ``issue`` is the GitHub issue that will digitize it (int or ``"#NNN"`` string); it is
    woven into the note as ``pending digitization (#NNN)`` so the table points at it.
    """
    issue_tag = issue if str(issue).startswith("#") else f"#{issue}"
    text = f"pending digitization ({issue_tag})"
    if note:
        text = f"{text}; {note}"
    row = ScorecardRow(
        benchmark, observable, None, None, None, None, PENDING,
        binding=False, note=text,
    )
    _ROWS.append(row)
    return row


def record(benchmark, observable, sim, experiment, band, unit, verdict,
           binding=True, note=None):
    """Record a fully-specified row directly (escape hatch for non-oracle observables)."""
    row = ScorecardRow(benchmark, observable, sim, experiment, band, unit, verdict,
                       binding=binding, note=note)
    _ROWS.append(row)
    return row


def render(the_rows=None):
    """Render the scorecard as a fixed-width text table (returns the string)."""
    the_rows = _ROWS if the_rows is None else the_rows
    headers = ["Benchmark", "Observable", "Sim", "Experiment", "Band", "Verdict"]

    def cells(r):
        sim = r._fmt(r.sim)
        exp = r._fmt(r.experiment)
        band = "--" if r.band is None else f"+/-{r.band:g}"
        unit = f" {r.unit}" if r.unit else ""
        sim = sim if r.sim is None else f"{sim}{unit}"
        exp = exp if r.experiment is None else f"{exp}{unit}"
        verdict = r.verdict if r.binding else f"{r.verdict} (reported)"
        if r.verdict == PENDING and r.note:
            verdict = f"{r.verdict} ({r.note.split('(')[-1].rstrip(')')})"
        return [r.benchmark, r.observable, sim, exp, band, verdict]

    table = [headers] + [cells(r) for r in the_rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    lines = []
    sep = "  "

    def fmt_row(row):
        return sep.join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip()

    lines.append(fmt_row(headers))
    lines.append(sep.join("-" * widths[i] for i in range(len(headers))))
    for r, src in zip(table[1:], the_rows):
        lines.append(fmt_row(r))
        if src.note and src.verdict != PENDING:
            lines.append(f"    note: {src.note}")
    return "\n".join(lines)


def write(path):
    """Write the rendered scorecard to ``path`` (returns the rendered string)."""
    text = render()
    with open(path, "w") as f:
        f.write(text + "\n")
    return text
