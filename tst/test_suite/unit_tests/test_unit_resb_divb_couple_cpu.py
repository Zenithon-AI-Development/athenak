"""
resb B_phi write-back div(B)-preservation unit test (issue #192).

Builds src/pgen/unit_tests/resb_divb_couple_test.cpp (-D
PROBLEM=unit_tests/resb_divb_couple_test) into the isolated tst/build_unit directory and
runs it, asserting it exited 0 (all checks passed).

The operator-split cylindrical resistive-B_phi step (#113) writes its diffused `bphi`
back onto the staggered face field b0.x2f via MHD::CoupleResbBphiToB0(), OUTSIDE the
constrained-transport curl that keeps div(B)=0.  The cylindrical FV divergence changes
under that write by exactly the phi term A2/V * d(dB_phi)/dphi, so the write is
divergence-free iff the APPLIED B_phi change is phi-uniform.  At the 1.5x-refined grid a
sub-machine phi asymmetry crept into `bphi`, the per-phi-slice write injected div(B), and
constrained transport grew the seeded violation into a density runaway -> NaN (#192).

This is the fast CPU unit-level red->green anchor for that fix: it seeds a div-free
axisymmetric b0, populates `bphi` with an axisymmetric increment PLUS a zero-mean phi
perturbation, calls the real CoupleResbBphiToB0(), and asserts div(B) is still ~0 (RED
with the per-slice write b2f(j)=bphi(j); GREEN with the phi-uniform-delta write), that the
write applied the axisymmetric increment, and that every face collapsed to the intended
phi-uniform target (the physical phi-mean diffusion is kept, only the phi noise dropped).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the resb div(B)-preservation unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("resb_divb_couple_test")
    assert passed, (
        "resb_divb_couple_test reported a failing check (nonzero exit): "
        "CoupleResbBphiToB0 injected div(B) under a phi-structured bphi (#192)"
    )
