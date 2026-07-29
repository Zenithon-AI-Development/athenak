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

import pytest

import test_suite.verification.scorecard as scorecard

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCORECARD_PATH = os.path.join(_THIS_DIR, "verification", "plots", "scorecard.txt")

# ---------------------------------------------------------------------------------------
# CI tiering (#250): modules listed here are marked `heavy` at collection time and are
# EXCLUDED from the fast per-PR CI tier (`run_test_suite.py --fast` adds -m "not heavy");
# the weekly heavy-ci workflow runs the full suite (no filter), so they still run.
# Durations are from CI run 30278391756 (per-line log timestamps) -- keep them current
# when adding entries, and add any new test expected to exceed ~5 minutes of *simulation*
# time (build overhead does not count: that is fixed separately by the per-source
# PROBLEM defines, also #250). Guarded by test_suite/style/test_ci_tiers.py.
# ---------------------------------------------------------------------------------------
HEAVY_TEST_MODULES = {
    "test_verify_maglif_b1_tabulated_cpu.py": "~80 min: full faithful B1 implosion",
    "test_verify_maglif_b2_ellison_cpu.py": "~5 min: B2 multimode MRT implosion",
    "test_verify_multigroup_rad_hydro_cpu.py": "~4 min: multigroup rad-hydro battery",
}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: long-running simulation test; excluded from the fast per-PR CI tier, "
        "run by the weekly heavy-ci workflow (#250)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-apply the `heavy` marker to the modules listed in HEAVY_TEST_MODULES."""
    for item in items:
        if os.path.basename(str(item.fspath)) in HEAVY_TEST_MODULES:
            item.add_marker(pytest.mark.heavy)


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
