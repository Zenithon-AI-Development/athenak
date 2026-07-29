"""
Pytest session hooks for the AthenaK test suite.

Emits the experiment-vs-simulation **scorecard** (issue [VA1]/#140, PRD #138) at the end
of a test session: as the MagLIF benchmarks compare their reduced observables against the
committed experimental ground truth, they record each verdict in
``test_suite.verification.scorecard``; here we render the accumulated table to the
terminal and write it to the committed deterministic path
``test_suite/verification/scorecard.txt`` (issue #228 -- never the gitignored ``plots/``
directory) so a reader sees, in one place, which observables reproduce the experiment,
which do not, and which anchors are still pending digitization. The scorecard is empty
(and nothing is emitted) for sessions that run no anchored benchmark -- so this hook is
inert for the upstream tiers.

The scorecard is also **ratcheted** (issue #228): with ``ATHENAK_SCORECARD_RATCHET=1``
(set by the CI full-CPU-tier job -- only a full run records the complete row set),
``pytest_sessionfinish`` compares the session rows against the committed expected-status
file ``verification/scorecard_ratchet.json``. Any regression (PASS->FAIL, a row
disappearing, a binding assertion softened) fails the session; improvements must update
the ratchet file in the same PR (``ATHENAK_UPDATE_BASELINES=1`` regenerates it, mirroring
the harness baseline convention). An all-FAIL experimental tier is thus visible and
pinned -- it can no longer regress silently inside a green session.
"""

import os

import test_suite.verification.scorecard as scorecard
import test_suite.verification.scorecard_ratchet as ratchet

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCORECARD_PATH = os.path.join(_THIS_DIR, "verification", "scorecard.txt")

# Ratchet violations found by pytest_sessionfinish, re-rendered in the terminal summary.
_RATCHET_VIOLATIONS = []


def pytest_sessionfinish(session, exitstatus=0):
    """Enforce the committed scorecard ratchet at end of session (issue #228)."""
    rows = scorecard.rows()
    if not rows or os.environ.get("ATHENAK_SCORECARD_RATCHET") != "1":
        return
    if os.environ.get("ATHENAK_UPDATE_BASELINES") == "1":
        entries = ratchet.update_file(rows)
        print(f"[scorecard-ratchet] regenerated {ratchet.RATCHET_PATH} "
              f"({len(entries)} entries) -- commit it with the PR")
        return
    violations = ratchet.check(rows, ratchet.load())
    del _RATCHET_VIOLATIONS[:]
    _RATCHET_VIOLATIONS.extend(violations)
    if violations:
        for v in violations:
            print(f"[scorecard-ratchet] {v.kind.upper()}: {v.message}")
        print("[scorecard-ratchet] session scorecard does not match the committed "
              "scorecard_ratchet.json: fix regressions; ratchet improvements in the "
              "same PR (ATHENAK_UPDATE_BASELINES=1 regenerates the file).")
        if getattr(session, "exitstatus", 0) == 0:
            session.exitstatus = 1


def pytest_terminal_summary(terminalreporter, exitstatus=None, config=None):
    """Render the experiment-anchoring scorecard once, at end of session."""
    rows = scorecard.rows()
    if not rows:
        return
    table = scorecard.render()
    terminalreporter.write_sep("=", "experiment-anchoring scorecard (PRD #138)")
    terminalreporter.write_line(table)
    try:
        os.makedirs(os.path.dirname(_SCORECARD_PATH), exist_ok=True)
        scorecard.write(_SCORECARD_PATH)
        terminalreporter.write_line(f"(written to {_SCORECARD_PATH})")
    except OSError:
        pass
    if _RATCHET_VIOLATIONS:
        terminalreporter.write_sep("=", "scorecard ratchet violations (issue #228)")
        for v in _RATCHET_VIOLATIONS:
            terminalreporter.write_line(f"{v.kind.upper()}: {v.message}")
