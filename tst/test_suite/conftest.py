"""
Pytest session hooks for the AthenaK test suite.

Emits the experiment-vs-simulation **scorecard** (issue [VA1]/#140, PRD #138) at the end
of a test session: as the MagLIF benchmarks compare their reduced observables against the
committed experimental ground truth, they record each verdict in
``test_suite.verification.scorecard``; here we render the accumulated table to the
terminal and write it to ``test_suite/verification/plots/scorecard.txt`` so a reader sees,
in one place, which observables reproduce the experiment, which do not, and which anchors
are still pending digitization. The scorecard is empty (and nothing is emitted) for
sessions that run no anchored benchmark -- so this hook is inert for the upstream tiers.
"""

import os

import test_suite.verification.scorecard as scorecard

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCORECARD_PATH = os.path.join(_THIS_DIR, "verification", "plots", "scorecard.txt")


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
