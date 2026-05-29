"""
Multi-block operator-split anisotropic (Braginskii) conduction verification
([A5]/#112, ADR-0006/ADR-0001).

Builds src/pgen/unit_tests/aniso_conduction_multiblock_test.cpp
(-D PROBLEM=unit_tests/aniso_conduction_multiblock_test) into the isolated
tst/build_unit directory, runs it on FOUR MeshBlocks (mesh 64x64, meshblock 32x32 => 2x2)
on a single rank, and asserts it exited 0 (all checks passed).

This is the multi-block half of the verification that the AnisotropicConductionOperator
keeps heat ON the field lines while running across MeshBlock boundaries via the shared
per-substage ghost-exchange helper MeshBoundaryValuesCC::SyncParabolicGhosts (#108/[A1])
and the new internal-face-only insulating BC (#112). The single-block physics (SIM-61
Braginskii coefficients, field-aligned projection, the Parrish-Stone ring, the Larsen cap)
is pinned separately by test_unit_aniso_conduction; here we add ONLY what #112 builds: the
cross-block exchange that lets the operator span >1 MeshBlock without changing the
field-aligned physics.

RED -> GREEN discriminator (embedded in the pgen, mirroring the #108 canary): the same
field-aligned cosine mode is advanced once WITH the exchange (the operator built with
``pin``) and once WITHOUT it (built with ``pin == nullptr``, skipping
SyncParabolicGhosts). The single internal x1 block face sits on the mode's steep
zero-crossing (m_mode=1 over the symmetric domain), so the no-exchange run self-insulates
each block and the global Linf decay error is ~60% of the amplitude (RED), while the
exchange run reproduces the analytic exp(lambda_par t) parallel decay to <2% (GREEN). The
pgen also checks the perpendicular mode barely decays (heat stays on field lines), the
ring stays on its circular field lines across blocks, the cross-field leakage stays
bounded + converged across an RKL2 substage sweep, and the box conserves the
volume-weighted energy.

Oracle: Layer 1 -- analytic. On the global insulated box a tiny cosine mode along the
uniform field bhat=xhat is an exact eigenvector of the discrete PARALLEL conduction
operator with eigenvalue lambda_par = -(2 D_par/dx^2)(1 - cos theta), theta = m pi / N,
D_par = kpar(gamma-1)/rho, so it decays as exp(lambda_par t); the cross-field
perpendicular mode (D_perp << D_par) is frozen on that timescale. References: closed form
(anisotropic
diffusion eigenmode); Parrish & Stone 2005 (field-aligned ring test); Meyer, Balsara &
Aslam 2014 (RKL2 STS).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the multi-block anisotropic-conduction test; pass iff it exits 0."""
    passed = testutils.run_unit_test("aniso_conduction_multiblock_test")
    assert passed, (
        "aniso_conduction_multiblock_test reported a failing check (nonzero exit): the "
        "operator-split anisotropic conduction did not keep heat field-aligned across "
        "MeshBlock boundaries (SyncParabolicGhosts wiring / internal-face insulating BC)"
    )
