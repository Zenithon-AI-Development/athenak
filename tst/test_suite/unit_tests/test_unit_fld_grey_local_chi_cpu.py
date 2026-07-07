"""
Auto-collected wrapper for the grey-FLD per-cell (local) chi pgen-based unit test (#204).

Builds src/pgen/unit_tests/fld_grey_local_chi_test.cpp
(-D PROBLEM=unit_tests/fld_grey_local_chi_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed).  The test verifies
FLDGreyOperator::EnableLocalChi (radiation_fld/fld_grey_operator.hpp): a per-cell
extinction field chi(m,k,j,i) replaces the frozen scalar constant, so a thick|thin
two-zone chi transports radiation across a transparent gap the all-thick constant
operator blocks (#204: opaque liner, transparent gap), while a uniformly-filled chi
field reproduces the constant-chi operator exactly and the explicit stability dt reads
the local chi.

Oracle: Layer 1 -- analytic/structural.  Equivalence of the uniform local field with the
scalar-constant operator; sign + >=1e3x magnitude ordering of the interface-cell energy
drain between the transparent-gap and all-thick operators; >=1e3x dt tightening from the
transparent (huge-D) half.
"""

import test_suite.testutils as testutils


def test_run():
    """Build + run the grey-FLD local-chi unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("fld_grey_local_chi_test")
    assert passed, "fld_grey_local_chi_test reported a failing check (nonzero exit)"
