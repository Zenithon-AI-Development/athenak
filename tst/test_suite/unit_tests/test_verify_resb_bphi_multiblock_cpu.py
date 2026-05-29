"""
Multi-block operator-split CYLINDRICAL resistive B_phi diffusion verification
([A6]/#113, ADR-0004/ADR-0001).

Builds src/pgen/unit_tests/resb_bphi_multiblock_test.cpp
(-D PROBLEM=unit_tests/resb_bphi_multiblock_test) into the isolated tst/build_unit
directory, runs it on FOUR radial MeshBlocks (mesh nx1=128, meshblock nx1=32) on a single
rank, and asserts it exited 0 (all checks passed).

This is the multi-block half of the verification that the ResistiveBphiOperator (the
special cylindrical -eta B_phi/r^2 curl-curl operator) reproduces the analytic Bessel-J_1
eigenmode decay while running across MeshBlock boundaries via the shared per-substage
ghost-exchange helper MeshBoundaryValuesCC::SyncParabolicGhosts (#108/[A1]). The
single-block cylindrical physics (the -eta B_phi/r^2 term, the conservative
(1/r)d_r(r*flux) radial form, the antisymmetric axis ghost) is pinned separately by
test_verify_cyl_bphi_diffuse; here we add ONLY what #113 builds: the cross-block exchange
that keeps the antisymmetric axis ghost only at the true r=0 face and lets the operator
span >1 MeshBlock.

RED -> GREEN discriminator (embedded in the pgen, mirroring the #108 canary): the same
J_1(kr*r) eigenmode is advanced once WITH the exchange (operator built with ``pin``) and
once WITHOUT it (built with ``pin == nullptr``, skipping SyncParabolicGhosts). The
internal radial block faces sit at r=0.25,0.5,0.75 where B_phi is large, so the
no-exchange run sign-flips B_phi there and the global Linf decay error is ~28% of the
amplitude (RED),
while the exchange run reproduces the analytic A J_1(kr) exp(-eta kr^2 t) decay to ~1e-5
(GREEN). The pgen also checks the J_1 eigenmode shape is preserved and the J_1-projected
decay factor matches analytic, near the axis (the 1/r^2 region) and away from it alike.

Oracle: Layer 1 -- analytic. The axisymmetric resistive B_phi equation has the exact
decaying eigenmode B_phi(r,t) = A J_1(kr) exp(-eta kr^2 t) (J_1 solves Bessel's equation
of order 1, so the -B_phi/r^2 curl-curl term is essential); kr = j_{1,1}/R puts J_1 zeros
on the axis and the outer radius. References: closed form (Bessel-J_1 resistive
eigenmode); Abramowitz & Stegun 1972; Meyer, Balsara & Aslam 2014 (RKL2 STS).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the multi-block resistive-B_phi test; pass iff it exits 0."""
    passed = testutils.run_unit_test("resb_bphi_multiblock_test")
    assert passed, (
        "resb_bphi_multiblock_test reported a failing check (nonzero exit): the operator-"
        "split cylindrical resistive B_phi diffusion did not reproduce the analytic J_1 "
        "eigenmode decay across MeshBlock boundaries (SyncParabolicGhosts wiring / "
        "antisymmetric axis ghost)"
    )
