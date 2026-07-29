"""
Red-first component tests for the scorecard surfacing + ratchet (issue #228, PRD #138).

The scorecard is the only artifact carrying the experiment-vs-simulation verdicts, yet it
was printed by ``conftest`` into the gitignored ``plots/`` directory and asserted by
nothing -- an all-FAIL experimental tier was a green session. This module pins the fix:

* the scorecard is written to a **committed deterministic path**
  (``verification/scorecard.txt``, tracked in git -- never the gitignored ``plots/``);
* a **ratchet**: a committed expected-status file
  (``verification/scorecard_ratchet.json``); any row regressing (PASS->FAIL, a binding
  row disappearing, a binding assertion softened) fails the session, and improvements
  must update the ratchet file in the same PR (enforced as a mismatch with an explicit
  update instruction; ``ATHENAK_UPDATE_BASELINES=1`` regenerates the file);
* the live B2 Stage-3 growth-fraction miss (0.345 vs the McBride band [0.05, 0.15]) is a
  tracked FAIL entry in the committed ratchet, referencing the investigation issue #229
  -- never a silently green session.

Enforcement is opt-in via ``ATHENAK_SCORECARD_RATCHET=1`` (set by the CI full-CPU-tier
job): only a full suite run records the complete row set, so partial local runs stay
inert. These are component tests (no simulation); auto-collected by run_test_suite.py
(module name contains ``_cpu``).
"""

# Modules
import json
import os
import types

import pytest

import test_suite.conftest as conftest
import test_suite.verification.scorecard as scorecard
import test_suite.verification.scorecard_ratchet as ratchet


@pytest.fixture(autouse=True)
def _clean_scorecard():
    """Each test runs against an empty scorecard, then any suite rows are restored.

    The scorecard is a process-wide singleton shared with the benchmark tests; snapshot
    and restore so these component tests never clobber the real suite scorecard,
    whatever the collection order.
    """
    saved = scorecard.rows()
    scorecard.reset()
    yield
    scorecard.set_rows(saved)


@pytest.fixture(autouse=True)
def _no_ratchet_env(monkeypatch):
    """Enforcement env vars never leak into (or out of) these tests."""
    monkeypatch.delenv("ATHENAK_SCORECARD_RATCHET", raising=False)
    monkeypatch.delenv("ATHENAK_UPDATE_BASELINES", raising=False)


def _row(benchmark="B2", observable="component observable", verdict=scorecard.FAIL,
         binding=False):
    return scorecard.ScorecardRow(
        benchmark, observable, 1.0, 2.0, 0.5, "u", verdict, binding=binding)


def _entry(benchmark="B2", observable="component observable", verdict=scorecard.FAIL,
           binding=False, count=1, **extra):
    e = {"benchmark": benchmark, "observable": observable, "verdict": verdict,
         "binding": binding, "count": count}
    e.update(extra)
    return e


class _FakeReporter:
    def __init__(self):
        self.lines = []

    def write_sep(self, sep, title):
        self.lines.append(title)

    def write_line(self, line):
        self.lines.append(line)


# --------------------------------------------------------------------------------------
# AC1: the scorecard is written to a committed deterministic path on every full run.
# --------------------------------------------------------------------------------------
def test_scorecard_path_is_committed_and_not_gitignored_plots_cpu():
    """The scorecard lands at verification/scorecard.txt (tracked), not under the
    gitignored plots/ directory, and the committed snapshot exists there."""
    path = conftest._SCORECARD_PATH
    assert path.endswith(os.path.join("verification", "scorecard.txt"))
    assert (os.sep + "plots" + os.sep) not in path
    assert os.path.isfile(path), "committed scorecard snapshot missing"


def test_terminal_summary_writes_the_scorecard_to_the_deterministic_path_cpu(
        tmp_path, monkeypatch):
    """The end-of-session hook renders the recorded rows to _SCORECARD_PATH."""
    target = str(tmp_path / "scorecard.txt")
    monkeypatch.setattr(conftest, "_SCORECARD_PATH", target)
    scorecard.record("B9", "hook-write observable", 1.0, 2.0, 0.5, "u",
                     scorecard.FAIL, binding=False)
    conftest.pytest_terminal_summary(_FakeReporter())
    assert os.path.isfile(target)
    with open(target) as f:
        text = f.read()
    assert "hook-write observable" in text


# --------------------------------------------------------------------------------------
# AC2: a committed expected-status file; regressions fail the suite; improvements must
# update the ratchet file in the same PR. Demonstrated red-first.
# --------------------------------------------------------------------------------------
def test_ratchet_file_is_committed_and_covers_the_experimental_tier_cpu():
    """The expected-status file is committed and covers all four MagLIF benchmarks."""
    assert os.path.isfile(ratchet.RATCHET_PATH)
    entries = ratchet.load()
    assert entries, "committed ratchet must not be empty"
    for e in entries:
        assert e["verdict"] in (scorecard.PASS, scorecard.FAIL, scorecard.PENDING)
    benchmarks = {e["benchmark"] for e in entries}
    assert {"B1", "B2", "B3", "B4"} <= benchmarks
    total = sum(int(e.get("count", 1)) for e in entries)
    assert total >= 10, "ratchet must cover the whole recorded experimental tier"


def test_matching_session_has_no_violations_cpu():
    expected = [_entry(verdict=scorecard.FAIL)]
    rows = [_row(verdict=scorecard.FAIL)]
    assert ratchet.check(rows, expected) == []


def test_pass_to_fail_is_a_regression_cpu():
    """A row that was PASS in the committed ratchet coming back FAIL fails the check."""
    expected = [_entry(verdict=scorecard.PASS)]
    rows = [_row(verdict=scorecard.FAIL)]
    violations = ratchet.check(rows, expected)
    assert violations
    assert all(v.kind == ratchet.REGRESSION for v in violations)
    assert any("PASS" in v.message and "FAIL" in v.message for v in violations)


def test_fail_to_pending_is_a_regression_cpu():
    """Losing an active comparison (FAIL -> PENDING) is a regression in anchoring."""
    expected = [_entry(verdict=scorecard.FAIL)]
    rows = [_row(verdict=scorecard.PENDING)]
    violations = ratchet.check(rows, expected)
    assert violations and all(v.kind == ratchet.REGRESSION for v in violations)


def test_binding_row_disappearing_is_a_regression_cpu():
    expected = [_entry(verdict=scorecard.PASS, binding=True)]
    violations = ratchet.check([], expected)
    assert violations
    assert all(v.kind == ratchet.REGRESSION for v in violations)
    assert any("disappear" in v.message for v in violations)


def test_reported_row_disappearing_is_also_a_regression_cpu():
    """Even a non-binding (reported) row vanishing is a loss of visibility -> red."""
    expected = [_entry(verdict=scorecard.FAIL, binding=False)]
    violations = ratchet.check([], expected)
    assert violations and all(v.kind == ratchet.REGRESSION for v in violations)


def test_binding_softened_to_reported_is_a_regression_cpu():
    """A binding assertion demoted to reported-only regresses the ratchet."""
    expected = [_entry(verdict=scorecard.PASS, binding=True)]
    rows = [_row(verdict=scorecard.PASS, binding=False)]
    violations = ratchet.check(rows, expected)
    assert violations and all(v.kind == ratchet.REGRESSION for v in violations)


def test_improvement_requires_updating_the_ratchet_in_the_same_pr_cpu():
    """FAIL -> PASS is an improvement: the check flags it so the same PR updates the
    committed ratchet file (it must never silently drift)."""
    expected = [_entry(verdict=scorecard.FAIL)]
    rows = [_row(verdict=scorecard.PASS)]
    violations = ratchet.check(rows, expected)
    assert violations
    assert all(v.kind == ratchet.IMPROVEMENT for v in violations)
    assert any("ratchet" in v.message for v in violations)


def test_pending_to_fail_is_an_improvement_cpu():
    """A newly-digitized anchor that now compares (and misses) is still an improvement
    over PENDING -- more anchoring, not less."""
    expected = [_entry(verdict=scorecard.PENDING)]
    rows = [_row(verdict=scorecard.FAIL)]
    violations = ratchet.check(rows, expected)
    assert violations and all(v.kind == ratchet.IMPROVEMENT for v in violations)


def test_new_row_not_in_the_ratchet_must_be_added_cpu():
    violations = ratchet.check([_row()], [])
    assert violations and all(v.kind == ratchet.IMPROVEMENT for v in violations)
    assert any("ratchet" in v.message for v in violations)


def test_duplicate_rows_are_ratcheted_by_count_cpu():
    """Two identical committed rows (count=2, the B4 pattern) demand two session rows;
    one disappearing is a regression."""
    expected = [_entry(verdict=scorecard.FAIL, count=2)]
    ok = ratchet.check([_row(), _row()], expected)
    assert ok == []
    violations = ratchet.check([_row()], expected)
    assert violations and all(v.kind == ratchet.REGRESSION for v in violations)


# --------------------------------------------------------------------------------------
# AC2 (wiring): the end-of-session hook fails the suite on a ratchet violation.
# --------------------------------------------------------------------------------------
def _write_ratchet(tmp_path, entries):
    path = str(tmp_path / "scorecard_ratchet.json")
    with open(path, "w") as f:
        json.dump({"rows": entries}, f)
    return path


def test_sessionfinish_fails_the_session_on_a_regression_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ratchet, "RATCHET_PATH",
        _write_ratchet(tmp_path, [_entry(verdict=scorecard.PASS)]))
    monkeypatch.setenv("ATHENAK_SCORECARD_RATCHET", "1")
    scorecard.record("B2", "component observable", 1.0, 2.0, 0.5, "u",
                     scorecard.FAIL, binding=False)
    session = types.SimpleNamespace(exitstatus=0)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1


def test_sessionfinish_is_inert_without_the_optin_env_cpu(tmp_path, monkeypatch):
    """Partial local runs (no env var) never trip the ratchet."""
    monkeypatch.setattr(
        ratchet, "RATCHET_PATH",
        _write_ratchet(tmp_path, [_entry(verdict=scorecard.PASS)]))
    scorecard.record("B2", "component observable", 1.0, 2.0, 0.5, "u",
                     scorecard.FAIL, binding=False)
    session = types.SimpleNamespace(exitstatus=0)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_sessionfinish_green_when_the_session_matches_the_ratchet_cpu(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        ratchet, "RATCHET_PATH",
        _write_ratchet(tmp_path, [_entry(verdict=scorecard.FAIL)]))
    monkeypatch.setenv("ATHENAK_SCORECARD_RATCHET", "1")
    scorecard.record("B2", "component observable", 1.0, 2.0, 0.5, "u",
                     scorecard.FAIL, binding=False)
    session = types.SimpleNamespace(exitstatus=0)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0


def test_update_mode_rewrites_the_ratchet_instead_of_failing_cpu(
        tmp_path, monkeypatch):
    """ATHENAK_UPDATE_BASELINES=1 (the harness convention) regenerates the ratchet from
    the session rows, preserving hand-curated issue annotations for surviving keys."""
    path = _write_ratchet(
        tmp_path, [_entry(verdict=scorecard.FAIL, issue="#229")])
    monkeypatch.setattr(ratchet, "RATCHET_PATH", path)
    monkeypatch.setenv("ATHENAK_SCORECARD_RATCHET", "1")
    monkeypatch.setenv("ATHENAK_UPDATE_BASELINES", "1")
    scorecard.record("B2", "component observable", 1.0, 2.0, 0.5, "u",
                     scorecard.PASS, binding=False)
    session = types.SimpleNamespace(exitstatus=0)
    conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0
    entries = ratchet.load(path)
    assert len(entries) == 1
    assert entries[0]["verdict"] == scorecard.PASS
    assert entries[0]["issue"] == "#229"


# --------------------------------------------------------------------------------------
# AC3: the live B2 Stage-3 miss is a tracked ratchet entry referencing the
# investigation issue (#229) -- not a silently green session.
# --------------------------------------------------------------------------------------
def test_b2_stage3_growth_fraction_miss_is_a_tracked_fail_entry_cpu():
    entries = ratchet.load()
    gf = [e for e in entries
          if e["benchmark"] == "B2" and "growth fraction" in e["observable"]]
    assert len(gf) == 1, "exactly one committed B2 growth-fraction ratchet entry"
    entry = gf[0]
    assert entry["verdict"] == scorecard.FAIL
    assert entry["binding"] is False
    reference = str(entry.get("issue", "")) + " " + str(entry.get("note", ""))
    assert "#229" in reference, "must reference the B2 investigation issue #229"
