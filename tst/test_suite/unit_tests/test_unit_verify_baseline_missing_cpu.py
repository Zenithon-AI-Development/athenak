"""Layer-2 harness must fail fast on a missing golden baseline (#226).

The 2026-07-29 adversarial audit found a structural loophole under every snapshot
test in the suite: ``harness.verify`` silently CAPTURED and PASSED when the golden
baseline JSON was absent.  Combined with the git-ignored plots directory and
``ATHENAK_UPDATE_BASELINES``, a deleted or never-committed baseline converted a
red into a green with no signal.

Contract pinned here:

  * comparing against a missing baseline is a FAILURE, and the error names both
    the expected baseline path and the explicit capture procedure
    (``ATHENAK_UPDATE_BASELINES=1``);
  * no baseline file is written as a side effect of that failure;
  * capture happens ONLY when explicitly requested via ``ATHENAK_UPDATE_BASELINES=1``;
  * the ordinary compare path (baseline present) is unchanged: identical data
    passes, out-of-tolerance data raises.

Pure-python harness tests: no build, no run.  Oracle: n/a (test-infrastructure
contract, not a physical result); excluded from the Layer-1/Layer-2 provenance
classification in docs/verification.md.
"""

import os

import numpy as np
import pytest

import test_suite.verification.harness as harness


@pytest.fixture()
def isolated_harness(tmp_path, monkeypatch):
    """Redirect baseline/plot output to a tmp dir; clear the capture env var."""
    baseline_dir = tmp_path / "baselines"
    plot_dir = tmp_path / "plots"
    monkeypatch.setattr(harness, "BASELINE_DIR", str(baseline_dir))
    monkeypatch.setattr(harness, "PLOT_DIR", str(plot_dir))
    monkeypatch.delenv(harness.UPDATE_ENV, raising=False)
    return harness


def _run(name):
    """Call verify() on a tiny synthetic profile under slice ``name``."""
    coord = np.linspace(0.0, 1.0, 8)
    fields = {"dens": np.ones(8), "velx": np.zeros(8)}
    harness.verify(name, coord, fields, coord_label="x1v", xlabel="x")
    return coord, fields


def test_missing_baseline_fails_by_default(isolated_harness):
    """No baseline + no env var => AssertionError naming path and capture procedure."""
    name = "no_such_slice"
    with pytest.raises(AssertionError) as excinfo:
        _run(name)
    message = str(excinfo.value)
    expected_path = harness.baseline_path(name)
    assert expected_path in message, (
        f"failure message must name the expected baseline path {expected_path}; "
        f"got: {message}"
    )
    assert harness.UPDATE_ENV in message, (
        f"failure message must name the capture procedure ({harness.UPDATE_ENV}=1); "
        f"got: {message}"
    )


def test_missing_baseline_failure_does_not_capture(isolated_harness):
    """The missing-baseline failure must not write a baseline as a side effect."""
    name = "no_such_slice"
    with pytest.raises(AssertionError):
        _run(name)
    assert not os.path.exists(harness.baseline_path(name)), (
        "verify() wrote a baseline while failing on a missing one -- capture must "
        f"only happen under {harness.UPDATE_ENV}=1"
    )


def test_capture_only_under_explicit_env(isolated_harness, monkeypatch):
    """ATHENAK_UPDATE_BASELINES=1 => capture the run as golden and pass."""
    name = "fresh_slice"
    monkeypatch.setenv(harness.UPDATE_ENV, "1")
    _run(name)  # must not raise
    path = harness.baseline_path(name)
    assert os.path.exists(path), f"explicit capture did not write {path}"
    captured = harness.load_baseline(name)
    assert captured["name"] == name
    assert set(captured["fields"]) == {"dens", "velx"}


def test_existing_baseline_compare_path_unchanged(isolated_harness, monkeypatch):
    """Baseline present: identical run passes; perturbed run raises with tolerances."""
    name = "guarded_slice"
    monkeypatch.setenv(harness.UPDATE_ENV, "1")
    coord, fields = _run(name)  # capture golden
    monkeypatch.delenv(harness.UPDATE_ENV)

    # Identical rerun passes.
    harness.verify(name, coord, fields, coord_label="x1v", xlabel="x")

    # Out-of-tolerance rerun raises and names the offending field.
    bad = dict(fields, dens=np.asarray(fields["dens"]) + 0.5)
    with pytest.raises(AssertionError, match="dens"):
        harness.verify(name, coord, bad, coord_label="x1v", xlabel="x")
