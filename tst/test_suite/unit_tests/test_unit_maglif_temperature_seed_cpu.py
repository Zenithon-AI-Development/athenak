"""
MagLIF Ellison Eq.16 temperature-seed unit test (issue [P4]/#161).

Builds src/pgen/unit_tests/maglif_temperature_seed_test.cpp (-D
PROBLEM=unit_tests/maglif_temperature_seed_test) into the isolated tst/build_unit
directory, runs it, and asserts it exited 0 (all checks passed).

The test exercises the SHARED device-inline helper src/pgen/maglif_temperature_seed.hpp
that the maglif pgen's `perturbation=temperature` mode uses, so it pins the exact code
that runs in the multi-mode MRT benchmark.  Eq.16 (Ellison et al. 2025, arXiv:2504.10760):

    T(r,z) = max( T_min, T_0 + exp((r - r_o)/lambda) * N_dT(i,j) ),  N_dT ~ Normal(0, dT)

with T_0 = 293 K, T_min = 273 K, r_o the outer liner radius, lambda the radial decay
length, and dT the swept amplitude.  The checks verify the perturbation statistics
required by AC#3: per-zone mean ~ 0 and std ~ dT at full envelope; the std decaying to
~ dT/e one decay length inside r_o; the T_min floor respected AND actually engaging at
large dT; the field reproducible by seed (identical for the same seed, different for a
different seed); and dT = 0 reducing to the unperturbed state (T == T_0 everywhere).

Oracle: Layer 1 -- analytic.  The Gaussian draw is a deterministic, counter-based
standard-normal (SplitMix64 hashes + Box-Muller), so its sample statistics converge to
the closed-form Normal(0, dT) moments and the envelope/floor/reproducibility properties
are exact by construction.
References: Ellison et al. 2025 (arXiv:2504.10760) Eq.16.
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the maglif temperature-seed unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("maglif_temperature_seed_test")
    assert passed, (
        "maglif_temperature_seed_test reported a failing check (nonzero exit)"
    )
