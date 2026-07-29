"""
Reusable Tier-2 verification harness for AthenaK.

This is the shared substrate every verification slice plugs into. A single call to
``verify(...)`` does three things for a problem that has already been run:

  1. emits a standard diagnostic plot (the key profiles/curves) to a known location,
  2. diffs the current run against a committed "golden" baseline of those quantities
     within a tolerance and reports pass/fail (raising ``AssertionError`` on a
     regression), and
  3. fails fast if the baseline is missing: capture happens ONLY when explicitly
     requested via ``ATHENAK_UPDATE_BASELINES=1`` (#226) — a deleted or
     never-committed baseline must never convert a red into a green.

Design notes
------------
* Baselines and plots are anchored to *this file* (not the working directory), so they
  always land inside the repo tree regardless of where the test binary runs from
  (the suite runs tests from ``tst/build/src``). Baselines are committed; plots are
  build artifacts (git-ignored).
* Baselines are stored as human-readable JSON so they diff cleanly in code review.
* To capture a new baseline, or to re-capture one deliberately, set the environment
  variable ``ATHENAK_UPDATE_BASELINES=1`` before running the suite and commit the
  resulting JSON. This is the ONLY path that writes a baseline; a compare against
  a missing one raises.
* Adding a new verification slice requires no edit to ``run_test_suite.py``: drop a
  ``test_verify_<name>_<device>.py`` file in a ``test_suite`` subdirectory and call
  ``verify(...)``. pytest auto-collects it.
"""

import json
import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Anchor output locations to this module so they live in-repo, not under the build dir.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(_THIS_DIR, "baselines")
PLOT_DIR = os.path.join(_THIS_DIR, "plots")

# Set this env var (to any non-empty value) to (re)capture baselines on this run.
UPDATE_ENV = "ATHENAK_UPDATE_BASELINES"


def _ensure_dirs():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def baseline_path(name):
    """Path to the golden baseline JSON for verification slice ``name``."""
    return os.path.join(BASELINE_DIR, f"{name}.json")


def plot_path(name):
    """Path to the diagnostic plot PNG for verification slice ``name``."""
    return os.path.join(PLOT_DIR, f"{name}.png")


def _as_float_lists(coord, fields):
    """Normalise inputs to plain Python floats/lists for JSON serialisation."""
    coord_list = np.asarray(coord, dtype=float).tolist()
    field_lists = {k: np.asarray(v, dtype=float).tolist() for k, v in fields.items()}
    return coord_list, field_lists


def capture_baseline(name, coord, fields, coord_label, rtol, atol):
    """Write the current run as the golden baseline for ``name`` and return it."""
    _ensure_dirs()
    coord_list, field_lists = _as_float_lists(coord, fields)
    baseline = {
        "name": name,
        "coord_label": coord_label,
        "coord": coord_list,
        "fields": field_lists,
        "rtol": rtol,
        "atol": atol,
    }
    with open(baseline_path(name), "w") as f:
        json.dump(baseline, f, indent=2)
    logging.info(f"[verify] captured golden baseline -> {baseline_path(name)}")
    return baseline


def load_baseline(name):
    """Load the golden baseline for ``name``, or ``None`` if none exists yet."""
    path = baseline_path(name)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def compare_to_baseline(baseline, coord, fields, rtol, atol):
    """
    Diff the current run against ``baseline`` within tolerance.

    Returns a list of human-readable failure strings (empty list == pass).
    """
    failures = []

    def _check(label, current, golden):
        cur = np.asarray(current, dtype=float)
        gold = np.asarray(golden, dtype=float)
        if cur.shape != gold.shape:
            failures.append(
                f"{label}: shape changed {gold.shape} -> {cur.shape}"
            )
            return
        if not np.allclose(cur, gold, rtol=rtol, atol=atol):
            max_abs = float(np.max(np.abs(cur - gold)))
            denom = np.maximum(np.abs(gold), atol)
            max_rel = float(np.max(np.abs(cur - gold) / denom))
            failures.append(
                f"{label}: max|Δ|={max_abs:.3e} max rel={max_rel:.3e} "
                f"(rtol={rtol:g} atol={atol:g})"
            )

    _check(baseline["coord_label"], coord, baseline["coord"])
    golden_fields = baseline["fields"]
    for label, current in fields.items():
        if label not in golden_fields:
            failures.append(f"{label}: not present in baseline")
            continue
        _check(label, current, golden_fields[label])
    for label in golden_fields:
        if label not in fields:
            failures.append(f"{label}: present in baseline but missing from run")
    return failures


def make_plot(name, coord, fields, baseline, xlabel, title, passed):
    """Emit a standard diagnostic plot (current run, golden overlay if present)."""
    _ensure_dirs()
    coord = np.asarray(coord, dtype=float)
    nfields = len(fields)
    fig, axes = plt.subplots(
        nfields, 1, figsize=(7, 2.6 * nfields), sharex=True, squeeze=False
    )
    axes = axes[:, 0]
    for ax, (label, current) in zip(axes, fields.items()):
        ax.plot(coord, np.asarray(current, dtype=float), "-", lw=1.4, label="run")
        if baseline is not None and label in baseline["fields"]:
            ax.plot(
                baseline["coord"], baseline["fields"][label],
                "--", lw=1.0, color="k", alpha=0.6, label="golden",
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel(xlabel)
    status = "PASS" if passed else (
        "MISSING BASELINE" if baseline is None else "FAIL"
    )
    suptitle = title if title else name
    fig.suptitle(f"{suptitle}  [{status}]")
    fig.tight_layout()
    fig.savefig(plot_path(name), dpi=110)
    plt.close(fig)
    logging.info(f"[verify] wrote diagnostic plot -> {plot_path(name)}")


def verify(name, coord, fields, coord_label="x", rtol=1.0e-5, atol=1.0e-8,
           xlabel="x", title=None):
    """
    One-call verification step for a problem that has already been run.

    Parameters
    ----------
    name : str
        Slug identifying this slice; names the baseline and plot files.
    coord : array_like
        1-D independent coordinate (e.g. cell-centre ``x1v``).
    fields : dict[str, array_like]
        Named diagnostic profiles/curves to regression-test and plot.
    coord_label : str
        Label stored for the coordinate (e.g. ``"x1v"``).
    rtol, atol : float
        Tolerances for the baseline diff (``numpy.allclose`` semantics).
    xlabel : str
        Axis label for the plot's shared x-axis.
    title : str, optional
        Plot title (defaults to ``name``).

    Behaviour
    ---------
    Always writes a diagnostic PNG. If ``ATHENAK_UPDATE_BASELINES`` is set, captures
    the current run as the golden baseline and passes — the ONLY path that writes a
    baseline. If no baseline exists and capture was not requested, raises
    ``AssertionError`` naming the expected path and the capture procedure (#226): a
    missing baseline must fail loudly, never silently self-capture and pass.
    Otherwise diffs every field (and the coordinate) against the baseline, raising
    ``AssertionError`` listing any quantities that exceed tolerance.
    """
    baseline = load_baseline(name)
    updating = bool(os.environ.get(UPDATE_ENV))

    if updating:
        # Plot against the previous baseline (if any) before overwriting it.
        make_plot(name, coord, fields, baseline, xlabel, title, passed=True)
        capture_baseline(name, coord, fields, coord_label, rtol, atol)
        return

    if baseline is None:
        # Fail fast (#226): a deleted or never-committed baseline must not convert
        # a red into a green. Still emit the diagnostic plot of the current run so
        # the failure can be inspected.
        make_plot(name, coord, fields, None, xlabel, title, passed=False)
        raise AssertionError(
            f"verification '{name}' has no golden baseline at "
            f"{baseline_path(name)} — comparing against a missing baseline is a "
            f"failure by default (#226). If this is a new slice (or a deliberate "
            f"re-baseline), capture it explicitly by re-running with "
            f"{UPDATE_ENV}=1 and committing the resulting JSON."
        )

    # Use the tolerances stored with the baseline if present (keeps captures and
    # diffs consistent); fall back to the call-site values otherwise.
    b_rtol = baseline.get("rtol", rtol)
    b_atol = baseline.get("atol", atol)
    failures = compare_to_baseline(baseline, coord, fields, b_rtol, b_atol)
    make_plot(name, coord, fields, baseline, xlabel, title, passed=not failures)
    if failures:
        raise AssertionError(
            f"verification '{name}' regressed against baseline:\n  "
            + "\n  ".join(failures)
        )
    logging.info(f"[verify] '{name}' matches baseline within tolerance")
