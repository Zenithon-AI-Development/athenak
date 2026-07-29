"""
CI-speed invariants (#250).

Oracle: Layer 1 -- structural invariants of the build system and test-suite tiering,
checked directly against the repository files (no simulation, no baseline).

Regressions this file guards against:

1. ``config.hpp.in`` must stay pgen-independent. When ``PROBLEM_GENERATOR`` /
   ``USER_PROBLEM_ENABLED`` lived in the global config header, every per-test pgen
   build (``-D PROBLEM=...``) produced a different ``config.hpp``, so the shared ccache
   missed on every translation unit of every per-test build and each one recompiled
   the whole tree (~2.2 min apiece, ~70 times per CI run, measured on run
   30278391756). Those macros are injected as per-source compile definitions on the
   only two consumers (``pgen/pgen.cpp``, ``utils/show_config.cpp``) instead.

2. The ``heavy``-marking hook in ``test_suite/conftest.py`` must actually mark the
   listed modules (else the fast per-PR tier would run the ~80-minute B1 implosion),
   and the listed modules must exist (else the weekly tier would silently cover
   nothing). The hook is exercised directly with stand-in items rather than via a
   pytest subprocess, so this check does not depend on the heavy modules' own imports
   (they need h5py etc., which the style CI job does not install).
"""

import importlib.util
import os
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SUITE_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_SUITE_DIR, "..", ".."))

# The whale: ~80 minutes on CI (run 30278391756). If it ever leaves the heavy list,
# the PR gate regresses to multi-hour runs.
_WHALE = "test_verify_maglif_b1_tabulated_cpu.py"


def test_config_template_is_pgen_independent():
    """The global config header template must not define the pgen-identity macros."""
    template = os.path.join(_REPO_ROOT, "config.hpp.in")
    with open(template) as f:
        text = f.read()
    offenders = re.findall(
        r"^\s*#\s*define\s+(?:PROBLEM_GENERATOR|USER_PROBLEM_ENABLED)\b.*|@PROBLEM@",
        text,
        flags=re.MULTILINE,
    )
    assert not offenders, (
        "config.hpp.in defines the problem-generator macros; this makes every "
        "-D PROBLEM=... build recompile the whole tree (see #250). Inject these as "
        "per-source compile definitions in src/CMakeLists.txt instead: "
        f"{offenders}"
    )


def _load_suite_conftest():
    """Import test_suite/conftest.py by path (it is not an importable module name)."""
    spec = importlib.util.spec_from_file_location(
        "suite_conftest_under_test", os.path.join(_SUITE_DIR, "conftest.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeItem:
    """Minimal stand-in for a pytest item: the hook reads fspath, calls add_marker."""

    def __init__(self, path):
        self.fspath = path
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


def _marker_names(item):
    return [getattr(m, "name", getattr(m, "markname", None)) for m in item.markers]


def test_heavy_hook_marks_the_whale():
    """The conftest hook must mark every listed module heavy, and only those."""
    conftest = _load_suite_conftest()
    whale = _FakeItem(os.path.join(_SUITE_DIR, "cylindrical", _WHALE))
    bystander = _FakeItem(
        os.path.join(_SUITE_DIR, "unit_tests", "test_unit_sample_cpu.py")
    )
    conftest.pytest_collection_modifyitems(None, [whale, bystander])
    assert "heavy" in _marker_names(whale), (
        f"{_WHALE} is not marked heavy by the conftest hook, so the fast (per-PR) "
        "CI tier (--fast / -m 'not heavy') would run it (~80 min)."
    )
    assert "heavy" not in _marker_names(bystander), (
        "the heavy hook marked a module that is not in HEAVY_TEST_MODULES -- the "
        "fast tier would silently lose coverage."
    )


def test_heavy_list_matches_files_on_disk():
    """Every HEAVY_TEST_MODULES entry must name a real test file (list-rot guard)."""
    conftest = _load_suite_conftest()
    assert _WHALE in conftest.HEAVY_TEST_MODULES, (
        f"{_WHALE} (~80 min on CI) must stay in HEAVY_TEST_MODULES."
    )
    on_disk = set()
    for dirpath, _dirnames, filenames in os.walk(_SUITE_DIR):
        on_disk.update(filenames)
    missing = sorted(set(conftest.HEAVY_TEST_MODULES) - on_disk)
    assert not missing, (
        "HEAVY_TEST_MODULES lists modules that do not exist under test_suite/ -- "
        f"stale entries mark nothing: {missing}"
    )
