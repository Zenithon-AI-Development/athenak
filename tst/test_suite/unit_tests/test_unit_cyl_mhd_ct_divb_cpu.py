"""
Auto-collected wrapper for the cylindrical CT divergence-preservation unit test.

Builds src/pgen/unit_tests/cyl_mhd_ct_divb_test.cpp
(-D PROBLEM=unit_tests/cyl_mhd_ct_divb_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test builds the cylindrical
constrained-transport curl B = curl(E) on a small cylindrical grid using the same
r-weighted edge lengths / face areas as mhd_ct.cpp's cylindrical branch, then verifies
the cylindrical finite-volume divergence of B vanishes to machine precision -- i.e.
constrained transport preserves div(B)=0 in curvilinear geometry (ADR-0004, issue [3a]).

Oracle: Layer 1 -- analytic.  The cylindrical constrained-transport curl B = curl(E),
built with the same r-weighted edge lengths / face areas as mhd_ct.cpp, obeys the
finite-volume identity div(curl E) = 0 exactly; the test pins the cylindrical div(B) to
machine precision (closed-form CT identity).  This unit test is the method-correctness
anchor for the Layer-2 test_verify_cyl_field_loop (advected div(B) snapshot; TODO #83).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the cylindrical CT div(B)=0 unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("cyl_mhd_ct_divb_test")
    assert passed, "cyl_mhd_ct_divb_test reported a failing check (nonzero exit)"
