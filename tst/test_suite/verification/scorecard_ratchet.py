"""
Scorecard ratchet -- the committed expected-status file for the experimental tier
(issue #228, PRD #138, ADR-0008).

The scorecard (``scorecard.py``) carries the suite's experiment-vs-simulation verdicts,
but by itself it is *reported*, not *asserted*: an all-FAIL experimental tier was still a
green session. The ratchet closes that hole. ``scorecard_ratchet.json`` (next to this
module) commits the expected status of every recorded row -- (benchmark, observable) ->
verdict/binding, with an occurrence ``count`` for observables recorded by more than one
test (the B4 pattern) and optional ``issue``/``note`` annotations pointing at the
tracking issue for a known miss (e.g. the B2 Stage-3 growth-fraction miss, #229).

``check(rows, expected)`` compares a session's recorded rows against the committed
expectations and returns ``Violation``s:

* ``REGRESSION`` -- a row got *worse* (verdict rank dropped ``PASS > FAIL > PENDING``, a
  binding assertion was softened to reported-only, or an expected row disappeared).
* ``IMPROVEMENT`` -- a row got *better* (or a new row appeared). Improvements are real
  progress, but the committed file must ratchet up with them **in the same PR**, so they
  are still violations until ``scorecard_ratchet.json`` is updated (rerun with
  ``ATHENAK_UPDATE_BASELINES=1`` to regenerate it, mirroring the harness baseline
  convention).

Enforcement is wired in ``conftest.pytest_sessionfinish`` behind
``ATHENAK_SCORECARD_RATCHET=1`` (set by the CI full-CPU-tier job): only a full suite run
records the complete row set, so partial local runs stay inert.
"""

# Modules
import collections
import json
import os

import test_suite.verification.scorecard as scorecard

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# The committed expected-status file (the ratchet).
RATCHET_PATH = os.path.join(_THIS_DIR, "scorecard_ratchet.json")

# Violation kinds.
REGRESSION = "regression"
IMPROVEMENT = "improvement"

Violation = collections.namedtuple("Violation", ["kind", "message"])

# Verdict "goodness" ranking: a session verdict ranking BELOW the committed expectation
# is a regression. PENDING (no comparison at all) ranks below FAIL (a real comparison
# that misses) which ranks below PASS.
_RANK = {scorecard.PENDING: 0, scorecard.FAIL: 1, scorecard.PASS: 2}


def load(path=None):
    """Load and validate the committed expected-status entries (a list of dicts)."""
    if path is None:
        path = RATCHET_PATH
    with open(path) as f:
        payload = json.load(f)
    entries = payload["rows"] if isinstance(payload, dict) else payload
    for e in entries:
        for field in ("benchmark", "observable", "verdict"):
            if field not in e:
                raise ValueError(f"ratchet entry missing '{field}': {e}")
        if e["verdict"] not in _RANK:
            raise ValueError(f"ratchet entry has unknown verdict: {e}")
    return entries


def _expected_per_key(entries):
    """Expand entries (with their counts) into {(benchmark, observable): [(verdict,
    binding), ...]}."""
    per_key = {}
    for e in entries:
        key = (e["benchmark"], e["observable"])
        vb = (e["verdict"], bool(e.get("binding", False)))
        per_key.setdefault(key, []).extend([vb] * int(e.get("count", 1)))
    return per_key


def _actual_per_key(rows):
    per_key = {}
    for r in rows:
        per_key.setdefault((r.benchmark, r.observable), []).append(
            (r.verdict, bool(r.binding)))
    return per_key


def _canon(vb_list):
    """Canonical order: best verdict first, binding before reported."""
    return sorted(vb_list, key=lambda vb: (-_RANK[vb[0]], not vb[1]))


def _fmt(vb):
    verdict, binding = vb
    return f"{verdict}{' (binding)' if binding else ''}"


def check(rows, expected):
    """Compare session scorecard rows against the committed expectations.

    Returns a list of ``Violation``s (empty when the session matches the ratchet
    exactly). ANY violation must fail the suite: regressions are fixed, improvements
    update ``scorecard_ratchet.json`` in the same PR.
    """
    exp = _expected_per_key(expected)
    act = _actual_per_key(rows)
    violations = []
    for key in sorted(set(exp) | set(act)):
        e_list = _canon(exp.get(key, []))
        a_list = _canon(act.get(key, []))
        if e_list == a_list:
            continue
        bench, obs = key
        where = f"[{bench}] {obs}"
        for i in range(max(len(e_list), len(a_list))):
            e = e_list[i] if i < len(e_list) else None
            a = a_list[i] if i < len(a_list) else None
            if e == a:
                continue
            if a is None:
                violations.append(Violation(
                    REGRESSION,
                    f"{where}: expected {_fmt(e)} row disappeared from the session "
                    f"scorecard"))
            elif e is None:
                violations.append(Violation(
                    IMPROVEMENT,
                    f"{where}: new {_fmt(a)} row not in the ratchet -- add it to "
                    f"scorecard_ratchet.json in the same PR"))
            else:
                worse = (_RANK[a[0]] < _RANK[e[0]]) or (e[1] and not a[1])
                if worse:
                    violations.append(Violation(
                        REGRESSION, f"{where}: {_fmt(e)} regressed to {_fmt(a)}"))
                else:
                    violations.append(Violation(
                        IMPROVEMENT,
                        f"{where}: {_fmt(e)} improved to {_fmt(a)} -- ratchet it by "
                        f"updating scorecard_ratchet.json in the same PR"))
    return violations


def update_file(rows, path=None):
    """Regenerate the ratchet file from the session rows (ATHENAK_UPDATE_BASELINES=1).

    Duplicate (benchmark, observable, verdict, binding) rows aggregate into a single
    entry with a ``count``; hand-curated ``issue``/``note`` annotations survive for keys
    that persist. Returns the written entries.
    """
    if path is None:
        path = RATCHET_PATH
    old = {}
    if os.path.isfile(path):
        for e in load(path):
            old.setdefault((e["benchmark"], e["observable"]), e)
    entries = []
    index = {}
    for r in rows:
        key = (r.benchmark, r.observable, r.verdict, bool(r.binding))
        if key in index:
            index[key]["count"] += 1
            continue
        entry = {"benchmark": r.benchmark, "observable": r.observable,
                 "verdict": r.verdict, "binding": bool(r.binding), "count": 1}
        prev = old.get((r.benchmark, r.observable))
        if prev:
            for field in ("issue", "note"):
                if field in prev:
                    entry[field] = prev[field]
        index[key] = entry
        entries.append(entry)
    payload = {
        "_doc": ("Committed scorecard ratchet (issue #228): the expected status of "
                 "every experiment-vs-simulation row the full CPU suite records. Any "
                 "regression against this file fails the suite; improvements must "
                 "update it in the same PR (ATHENAK_UPDATE_BASELINES=1 regenerates "
                 "it). See scorecard_ratchet.py."),
        "rows": entries,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return entries
