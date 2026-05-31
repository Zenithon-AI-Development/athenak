//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file maglif_temperature_seed_test.cpp
//! \brief Unit test for the Ellison Eq.16 temperature seed (issue [P4]/#161).
//!
//! Exercises the SHARED device-inline helper pgen/maglif_temperature_seed.hpp that the
//! maglif pgen's `perturbation=temperature` mode uses, so this test pins the exact code
//! that runs in the benchmark.  Eq.16 (Ellison et al. 2025, arXiv:2504.10760):
//!   T(r,z) = max( T_min, T_0 + exp((r-r_o)/lambda) * N_dT ),  N_dT ~ Normal(0, dT).
//! The checks verify the perturbation statistics (AC#3):
//!   (1) the per-zone draw has mean ~ 0 and standard deviation ~ dT at full envelope;
//!   (2) the radial decay: std falls to ~ dT/e one decay length lambda inside r_o;
//!   (3) the T_min floor is respected (no zone below T_min) AND actually engages;
//!   (4) the field is reproducible by seed (same seed -> identical; differs by seed);
//!   (5) dT = 0 reduces to the unperturbed state (T == T_0 everywhere).
//! Reductions run on DevExeSpace (SERIAL on CPU), so the tested path is the device one.
//! Auto-run by tst/test_suite/unit_tests/test_unit_maglif_temperature_seed_cpu.py.

#include <cstdint>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "pgen/maglif_temperature_seed.hpp"
#include "pgen/unit_tests/unit_test.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("maglif_temperature_seed_test");

  // Eq.16 parameters (overridable from <problem>); defaults match the maglif pgen.
  const std::uint64_t n = static_cast<std::uint64_t>(
      pin->GetOrAddInteger("problem", "n", 500000));   // number of zones sampled
  const Real T0     = pin->GetOrAddReal("problem", "pert_T0", 293.0);
  const Real Tmin   = pin->GetOrAddReal("problem", "pert_Tmin", 273.0);
  const Real dT     = pin->GetOrAddReal("problem", "pert_dT", 100.0);
  const Real lambda = pin->GetOrAddReal("problem", "pert_decay", 0.05);
  const Real r_o    = pin->GetOrAddReal("problem", "pert_decay_r0", 1.0);
  const std::uint64_t seed = static_cast<std::uint64_t>(
      static_cast<unsigned>(pin->GetOrAddInteger("problem", "pert_seed", 12345)));

  // A floor far below any draw, used for the unbiased mean/std checks so the clip does
  // not skew them (the floor itself is checked separately below with the real T_min).
  const Real kNoFloor = -1.0e30;

  // (1) Mean ~ 0 and std ~ dT of the perturbation (T - T0) at the outer surface r = r_o
  //     (envelope exp(0) = 1), with the floor disabled so the raw Gaussian stats show.
  {
    Real sum = 0.0, sumsq = 0.0;
    Kokkos::parallel_reduce("tseed_stats_full",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
      KOKKOS_LAMBDA(const std::int64_t s, Real &lsum, Real &lsq) {
        Real g = maglif_tseed::CellGaussian(static_cast<std::uint64_t>(s), seed);
        Real T = maglif_tseed::EllisonTemperature(r_o, r_o, lambda, T0, kNoFloor, dT, g);
        Real p = T - T0;
        lsum += p;
        lsq  += p*p;
      }, Kokkos::Sum<Real>(sum), Kokkos::Sum<Real>(sumsq));
    Real mean = sum/static_cast<Real>(n);
    Real var  = sumsq/static_cast<Real>(n) - mean*mean;
    Real std  = Kokkos::sqrt(var);
    test.CheckNear(mean, 0.0, 0.0, 2.0e-2*dT, "perturbation mean ~ 0 at full envelope");
    test.CheckNear(std, dT, 3.0e-2, 0.0, "perturbation std ~ dT at full envelope");
  }

  // (2) Radial decay: one decay length inside the outer surface (r = r_o - lambda) the
  //     envelope is exp(-1), so the std of the perturbation is ~ dT/e.
  {
    Real sumsq = 0.0;
    const Real rin = r_o - lambda;
    Kokkos::parallel_reduce("tseed_stats_decay",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
      KOKKOS_LAMBDA(const std::int64_t s, Real &lsq) {
        Real g = maglif_tseed::CellGaussian(static_cast<std::uint64_t>(s), seed);
        Real T = maglif_tseed::EllisonTemperature(rin, r_o, lambda, T0, kNoFloor, dT, g);
        Real p = T - T0;
        lsq += p*p;
      }, Kokkos::Sum<Real>(sumsq));
    Real std_decay = Kokkos::sqrt(sumsq/static_cast<Real>(n));
    Real expected  = dT*Kokkos::exp(static_cast<Real>(-1.0));
    test.CheckNear(std_decay, expected, 4.0e-2, 0.0,
                   "perturbation std ~ dT/e one decay length inside r_o");
  }

  // (3) Floor respected AND engaged.  With the real T_min and a LARGE dT, the unfloored
  //     draws dip below T_min (proving the floor is necessary), while the floored field
  //     never does.
  {
    const Real big_dT = 1000.0;
    Real floored_min = 1.0e30, unfloored_min = 1.0e30;
    Kokkos::parallel_reduce("tseed_floor",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
      KOKKOS_LAMBDA(const std::int64_t s, Real &lfmin, Real &lumin) {
        Real g = maglif_tseed::CellGaussian(static_cast<std::uint64_t>(s), seed);
        Real Tf = maglif_tseed::EllisonTemperature(r_o, r_o, lambda, T0, Tmin, big_dT, g);
        Real Tu = T0 + big_dT*g;   // same draw, no floor
        lfmin = Kokkos::fmin(lfmin, Tf);
        lumin = Kokkos::fmin(lumin, Tu);
      }, Kokkos::Min<Real>(floored_min), Kokkos::Min<Real>(unfloored_min));
    test.CheckTrue(floored_min >= Tmin - 1.0e-9, "no zone falls below the T_min floor");
    test.CheckTrue(unfloored_min < Tmin, "floor actually engages (unfloored dips below)");
  }

  // (4) Reproducible by seed: the same seed gives a bitwise-identical field; a different
  //     seed gives a different field (max |diff| > 0).
  {
    const std::uint64_t seed2 = seed + 777ULL;
    Real same_maxdiff = 0.0, diff_maxdiff = 0.0;
    Kokkos::parallel_reduce("tseed_repro",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
      KOKKOS_LAMBDA(const std::int64_t s, Real &lsame, Real &ldiff) {
        std::uint64_t gid = static_cast<std::uint64_t>(s);
        Real a  = maglif_tseed::CellGaussian(gid, seed);
        Real a2 = maglif_tseed::CellGaussian(gid, seed);
        Real b  = maglif_tseed::CellGaussian(gid, seed2);
        lsame = Kokkos::fmax(lsame, Kokkos::fabs(a - a2));
        ldiff = Kokkos::fmax(ldiff, Kokkos::fabs(a - b));
      }, Kokkos::Max<Real>(same_maxdiff), Kokkos::Max<Real>(diff_maxdiff));
    test.CheckTrue(same_maxdiff == 0.0, "same seed -> identical field (reproducible)");
    test.CheckTrue(diff_maxdiff > 0.0, "different seed -> different field");
  }

  // (5) dT = 0 reduces to the unperturbed state: T == T0 for every zone.
  {
    Real maxdev = 0.0;
    Kokkos::parallel_reduce("tseed_zero_dT",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
      KOKKOS_LAMBDA(const std::int64_t s, Real &ldev) {
        Real g = maglif_tseed::CellGaussian(static_cast<std::uint64_t>(s), seed);
        Real T = maglif_tseed::EllisonTemperature(r_o, r_o, lambda, T0, Tmin, 0.0, g);
        ldev = Kokkos::fmax(ldev, Kokkos::fabs(T - T0));
      }, Kokkos::Max<Real>(maxdev));
    test.CheckNear(maxdev, 0.0, 0.0, 1.0e-12, "dT=0 reduces to unperturbed (T == T0)");
  }

  test.Finish();
  return;
}
