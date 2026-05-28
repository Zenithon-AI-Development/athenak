"""
Auto-collected wrapper for the anisotropic Braginskii conduction pgen-based unit test.

Builds src/pgen/unit_tests/aniso_conduction_test.cpp
(-D PROBLEM=unit_tests/aniso_conduction_test) into the isolated tst/build_unit directory,
runs it, and asserts it exited 0 (all checks passed). The test exercises the anisotropic
(magnetized) Braginskii electron+ion thermal-conduction operator
(AnisotropicConductionOperator), re-expressed as a parabolic::ParabolicOperator and
advanced operator-split by RKL2 STS (issue [6a]/#18, ADR-0006), on real device kernels.

It verifies: the SIM-61 Braginskii coefficients (kappa_par >> kappa_perp when magnetized,
the (omega_c tau)^2 anisotropy law, electron+ion summation); the field-aligned heat-flux
projection (a temperature mode along B reduces to kappa_par diffusion, across B to
kappa_perp, to machine precision); a Parrish-Stone ring test where heat conducts along the
azimuthal field lines while cross-field radial leakage stays a tiny fraction; the
Sharma-Hammett monotonic limiter (no cross-field over/undershoot); and the Larsen
field-aligned flux cap (saturation at the free-streaming limit).

Oracle: Layer 1 -- literature.  Field-aligned heat-flux projection (kappa_par >>
kappa_perp; mode along B -> kappa_par diffusion, across B -> kappa_perp, to machine
precision), the Parrish-Stone ring test (heat follows the azimuthal field; cross-field
leakage tiny), the Sharma-Hammett monotonic limiter, and the Larsen free-streaming cap.
References: Parrish & Stone 2005; Sharma & Hammett 2007; Braginskii 1965.  This unit test
(with test_unit_parabolic_conduction) is a method-correctness anchor for the Layer-2
test_verify_cyl_aniso_ring (grounded in #81).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the anisotropic conduction unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("aniso_conduction_test")
    assert passed, "aniso_conduction_test reported a failing check (nonzero exit)"
