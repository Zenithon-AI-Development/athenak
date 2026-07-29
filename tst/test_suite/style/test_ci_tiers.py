"""
CI-speed invariants (#250).

Oracle: Layer 1 -- structural invariants of the build system and test-suite tiering,
checked directly against the repository files (no simulation, no baseline).

Two regressions this file guards against:

1. ``config.hpp.in`` must stay pgen-independent. When ``PROBLEM_GENERATOR`` /
   ``USER_PROBLEM_ENABLED`` lived in the global config header, every per-test pgen
   build (``-D PROBLEM=...``) invalidated every translation unit, so ~70 CI tests each
   paid a ~2.2-minute whole-tree recompile (~2.6 h per run, measured on CI run
   30278391756). Those macros are injected as per-source compile definitions on the
   only two consumers (``pgen/pgen.cpp``, ``utils/show_config.cpp``) instead.

2. The ``heavy`` tier must actually exclude the long-running simulation tests from the
   fast (per-PR) tier: ``-m "not heavy"`` deselects them, ``-m heavy`` selects them.
"""

import os
import re
import sys
from subprocess import Popen, PIPE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
_TST_DIR = os.path.join(_REPO_ROOT, "tst")

# The whale: 80 minutes on CI (run 30278391756). If the tiering ever silently stops
# excluding it, the PR gate regresses to multi-hour runs.
_HEAVY_EXAMPLE = "test_suite/cylindrical/test_verify_maglif_b1_tabulated_cpu.py"


def test_config_template_is_pgen_independent():
    """The global config header template must not depend on the PROBLEM cmake var."""
    template = os.path.join(_REPO_ROOT, "config.hpp.in")
    with open(template) as f:
        text = f.read()
    offenders = re.findall(
        r"^\s*#\s*define\s+(?:PROBLEM_GENERATOR|USER_PROBLEM_ENABLED)\b.*|@PROBLEM@",
        text,
        flags=re.MULTILINE,
    )
    assert not offenders, (
        "config.hpp.in references the problem generator; this makes every "
        "-D PROBLEM=... build recompile the whole tree (see #250). Inject these as "
        "per-source compile definitions in src/CMakeLists.txt instead: "
        f"{offenders}"
    )


def _collect(marker_expr):
    """Return pytest's collect-only output for the whale under a marker filter."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-m",
        marker_expr,
        _HEAVY_EXAMPLE,
    ]
    process = Popen(command, stdout=PIPE, stderr=PIPE, cwd=_TST_DIR, text=True)
    output, errors = process.communicate()
    return output + errors


def test_heavy_tier_excludes_the_whale_from_fast():
    """-m "not heavy" must deselect the 80-minute B1 CPU implosion test."""
    out = _collect("not heavy")
    assert "deselected" in out and "1 test collected" not in out.replace(
        "tests", "test"
    ), (
        f"{_HEAVY_EXAMPLE} is not marked heavy, so the fast (per-PR) CI tier would "
        f"run it (~80 min). Collection output:\n{out}"
    )


def test_heavy_tier_still_selects_the_whale():
    """-m heavy must select it, so the weekly full run still covers it."""
    out = _collect("heavy")
    assert re.search(r"\b1 test collected\b|\b1/1 tests collected\b", out.replace(
        "tests collected", "test collected"
    )), (
        f"{_HEAVY_EXAMPLE} is not selectable via -m heavy; the weekly tier would "
        f"silently drop it. Collection output:\n{out}"
    )
