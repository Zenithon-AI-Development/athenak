"""
Auto-collected wrapper for the multigroup FLD operator pgen-based unit test.

Builds src/pgen/unit_tests/fld_multigroup_operator_test.cpp
(-D PROBLEM=unit_tests/fld_multigroup_operator_test) into the isolated tst/build_unit
directory, runs it against the committed optically-thick multigroup-opacity fixture, and
asserts it exited 0 (all checks passed). The test exercises the N-group flux-limited
radiation diffusion operator (FLDMultigroupOperator) with a per-group Larsen flux limiter
and a per-group Rosseland diffusion coefficient, re-expressed as a
parabolic::ParabolicOperator (issue [17a]/#24, ADR-0001/0007), on real device kernels. It
verifies: the group structure (count) is read from the opacity table; the per-group
extinction chi_g = rho*kappa_R,g comes from the tabulated Rosseland transport opacity; the
per-group operator action reproduces the discrete diffusion eigenvalue lambda_g*(E_g-E0)
with D_g = c/(3 chi_g) on an optically-thick cosine eigenmode (distinct rate per group);
per-group conservation by the insulated operator; and a finite, positive explicit dt.

The fixture path (repo-relative) is resolved to an absolute path here and passed to the
binary via the `problem/opacity_file=<abspath>` command-line override, because the test
runs from tst/build*/src where a repo-relative path would not resolve.

Oracle: Layer 1 -- analytic.  Per-group analog of the grey operator test: each group's
action reproduces the discrete diffusion eigenvalue lambda_g*(E_g-E0) with
D_g = c/(3 chi_g), chi_g = rho*kappa_R,g from the tabulated Rosseland transport opacity,
on an optically-thick cosine eigenmode (distinct rate per group), plus per-group
conservation.  References: closed form (per-group eigenmode); Larsen flux limiter.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the multigroup FLD operator unit test; pass iff it exits 0."""
    fixture = os.path.join(
        testutils._repo_root(), "inputs", "radiation_fld", "multigroup_opacity.cn4"
    )
    passed = testutils.run_unit_test(
        "fld_multigroup_operator_test",
        args=[f"problem/opacity_file={fixture}"],
    )
    assert passed, "fld_multigroup_operator_test reported a failing check (nonzero exit)"
