"""
Auto-collected wrapper for the coordinate-geometry pgen-based unit test.

Builds src/pgen/unit_tests/coord_geometry_test.cpp
(-D PROBLEM=unit_tests/coord_geometry_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test verifies that the Cartesian
accessors in coordinates/coord_geometry.hpp return the analytic uniform-grid constants and
that the flux-divergence helpers reduce to the legacy (F_{i+1}-F_i)/dx form -- the
foundation that lets the hydro/MHD update be routed through the accessors without changing
Cartesian behavior (ADR-0004, issue [1a]).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the coord-geometry unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("coord_geometry_test")
    assert passed, "coord_geometry_test reported a failing check (nonzero exit)"
