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
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the anisotropic conduction unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("aniso_conduction_test")
    assert passed, "aniso_conduction_test reported a failing check (nonzero exit)"
