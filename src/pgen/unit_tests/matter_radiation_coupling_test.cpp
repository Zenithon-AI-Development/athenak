//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file matter_radiation_coupling_test.cpp
//  \brief Unit test: per-cell point-implicit (backward-Euler) grey matter-radiation
//  emission/absorption coupling (radiationfld::PointImplicitGreyCoupling), exercised on
//  real device kernels (issue [8b]/#23, ADR-0001/0002).
//
//  The coupling exchanges energy between the grey radiation energy density E_r and a
//  single matter internal energy density e_mat (T = e_mat/c_v):
//     dE_r/dt = c chi_a (a T^4 - E_r),   de_mat/dt = -dE_r/dt,
//  conserving E_r + e_mat and relaxing to radiative equilibrium a T^4 = E_r.  It is
//  purely local (no spatial dependence) and unconditionally stable (backward Euler), so a
//  stiff cell (c chi_a dt >> 1) reaches equilibrium in one step.
//
//  Batteries:
//   (A) Single-cell RELAXATION to radiative equilibrium: many moderate steps drive
//       (E_r, e_mat) to a T_eq^4 = E_r,eq with a T_eq^4 + c_v T_eq = E_r^0 + e_mat^0
//       (host Newton oracle); energy conserved to machine precision throughout.
//   (B) NO SPATIAL DEPENDENCE: identically-initialised cells stay bit-for-bit identical.
//   (C) Exchange DIRECTION: hot matter (a T^4 > E_r) cools while radiation heats, and
//       vice versa; each single step conserves E_r + e_mat.
//   (D) STIFF STABILITY: with c chi_a dt = 1e8 a single step lands at equilibrium from
//       either side, finite and without overshoot.
//   (E) chi_a = 0 is an exact no-op.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/matter_radiation_coupling_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_matter_radiation_coupling_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::fabs, std::pow, std::isfinite
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
//! \brief Host oracle: radiative-equilibrium temperature solving a T^4 + cv T = S, by
//! safeguarded Newton (g monotone increasing, convex for T>0 -> Newton from the upper
//! bracket converges).
double EquilibriumTemp(double a_rad, double cv, double S) {
  double T = std::pow(S/a_rad, 0.25);   // a T^4 = S => upper bound on the root
  for (int it = 0; it < 200; ++it) {
    double g = a_rad*T*T*T*T + cv*T - S;
    double gp = 4.0*a_rad*T*T*T + cv;
    double dT = g/gp;
    T -= dT;
    if (std::fabs(dT) <= 1.0e-15*T) { break; }
  }
  return T;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Point-implicit grey matter-radiation coupling unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("matter_radiation_coupling_test");

  // code-unit physical constants (c = a = cv = 1 keeps the equilibrium algebra clean)
  const Real cv = 1.0, c_light = 1.0, a_rad = 1.0;

  // ===== (A)+(B) relaxation to equilibrium on a set of identical cells =====
  const int ncell = 8;
  const Real e_mat0 = 1.0;       // T0 = 1
  const Real e_rad0 = 0.0;       // cold radiation
  const Real S = e_rad0 + e_mat0;
  const Real chi_a = 2.0, dt = 0.5;   // beta = c chi_a dt = 1.0 per step
  const int nstep = 200;
  const double T_eq = EquilibriumTemp(a_rad, cv, S);
  const Real em_eq = cv*T_eq;
  const Real er_eq = a_rad*T_eq*T_eq*T_eq*T_eq;

  DvceArray1D<Real> d_er("er", ncell), d_em("em", ncell), d_cons("cons", ncell);
  par_for("mrc_relax", DevExeSpace(), 0, ncell-1, KOKKOS_LAMBDA(const int n) {
    Real er = e_rad0, em = e_mat0;
    Real maxdrift = 0.0;
    for (int s = 0; s < nstep; ++s) {
      Real er_new, em_new;
      radiationfld::PointImplicitGreyCoupling(er, em, cv, chi_a, c_light, a_rad, dt,
                                              er_new, em_new);
      er = er_new; em = em_new;
      Real drift = Kokkos::fabs(er + em - S);
      maxdrift = Kokkos::fmax(maxdrift, drift);
    }
    d_er(n) = er; d_em(n) = em; d_cons(n) = maxdrift;
  });
  auto h_er = Kokkos::create_mirror_view(d_er);
  auto h_em = Kokkos::create_mirror_view(d_em);
  auto h_cons = Kokkos::create_mirror_view(d_cons);
  Kokkos::deep_copy(h_er, d_er);
  Kokkos::deep_copy(h_em, d_em);
  Kokkos::deep_copy(h_cons, d_cons);

  test.CheckNear(h_em(0), em_eq, 1.0e-8, 0.0,
                 "(A) matter relaxes to equilibrium internal energy c_v T_eq");
  test.CheckNear(h_er(0), er_eq, 1.0e-8, 0.0,
                 "(A) radiation relaxes to equilibrium energy a T_eq^4");
  test.CheckNear(a_rad*Kokkos::pow(h_em(0)/cv, 4.0), h_er(0), 1.0e-7, 0.0,
                 "(A) radiative equilibrium reached: a T^4 == E_r");

  Real max_drift = 0.0;
  for (int n = 0; n < ncell; ++n) { max_drift = std::fmax(max_drift, h_cons(n)); }
  test.CheckNear(max_drift, 0.0, 0.0, 1.0e-12*S,
                 "(A) energy E_r + e_mat conserved to machine precision every step");

  // (B) no spatial dependence: identical inputs -> identical outputs, bit-for-bit.
  Real er_lo = h_er(0), er_hi = h_er(0), em_lo = h_em(0), em_hi = h_em(0);
  for (int n = 1; n < ncell; ++n) {
    er_lo = std::fmin(er_lo, h_er(n)); er_hi = std::fmax(er_hi, h_er(n));
    em_lo = std::fmin(em_lo, h_em(n)); em_hi = std::fmax(em_hi, h_em(n));
  }
  test.CheckNear(er_hi - er_lo, 0.0, 0.0, 0.0,
                 "(B) coupling is purely local: identical cells stay identical (E_r)");
  test.CheckNear(em_hi - em_lo, 0.0, 0.0, 0.0,
                 "(B) coupling is purely local: identical cells stay identical (e_mat)");

  // ===== (C)+(D)+(E) single-step scenarios =====
  // 0: hot matter (a T^4 > E_r) -> matter cools, radiation heats
  // 1: cold matter (a T^4 < E_r) -> matter heats, radiation cools
  // 2: stiff, hot matter -> equilibrium in one step (S = 1)
  // 3: stiff, cold matter (e_mat=0) -> equilibrium in one step (S = 1)
  // 4: chi_a = 0 -> exact no-op
  const int nsc = 5;
  DvceArray1D<Real> d_eri("eri", nsc), d_emi("emi", nsc);
  DvceArray1D<Real> d_chia("chia", nsc), d_dts("dts", nsc);
  DvceArray1D<Real> d_ero("ero", nsc), d_emo("emo", nsc);
  auto h_eri = Kokkos::create_mirror_view(d_eri);
  auto h_emi = Kokkos::create_mirror_view(d_emi);
  auto h_chia = Kokkos::create_mirror_view(d_chia);
  auto h_dts = Kokkos::create_mirror_view(d_dts);
  h_eri(0) = 1.0; h_emi(0) = 2.0; h_chia(0) = 2.0;  h_dts(0) = 0.5;  // beta = 1
  h_eri(1) = 5.0; h_emi(1) = 0.1; h_chia(1) = 2.0;  h_dts(1) = 0.5;
  h_eri(2) = 0.0; h_emi(2) = 1.0; h_chia(2) = 1.0e8; h_dts(2) = 1.0;  // beta = 1e8
  h_eri(3) = 1.0; h_emi(3) = 0.0; h_chia(3) = 1.0e8; h_dts(3) = 1.0;
  h_eri(4) = 0.0; h_emi(4) = 1.0; h_chia(4) = 0.0;  h_dts(4) = 0.5;  // no coupling
  Kokkos::deep_copy(d_eri, h_eri);
  Kokkos::deep_copy(d_emi, h_emi);
  Kokkos::deep_copy(d_chia, h_chia);
  Kokkos::deep_copy(d_dts, h_dts);

  par_for("mrc_step", DevExeSpace(), 0, nsc-1, KOKKOS_LAMBDA(const int n) {
    Real er, em;
    radiationfld::PointImplicitGreyCoupling(d_eri(n), d_emi(n), cv, d_chia(n), c_light,
                                            a_rad, d_dts(n), er, em);
    d_ero(n) = er; d_emo(n) = em;
  });
  auto h_ero = Kokkos::create_mirror_view(d_ero);
  auto h_emo = Kokkos::create_mirror_view(d_emo);
  Kokkos::deep_copy(h_ero, d_ero);
  Kokkos::deep_copy(h_emo, d_emo);

  // (C) direction of energy exchange + single-step conservation
  test.CheckTrue(h_emo(0) < h_emi(0) && h_ero(0) > h_eri(0),
                 "(C) hot matter (a T^4 > E_r): matter cools, radiation heats");
  test.CheckTrue(h_emo(1) > h_emi(1) && h_ero(1) < h_eri(1),
                 "(C) cold matter (a T^4 < E_r): matter heats, radiation cools");
  test.CheckNear(h_ero(0) + h_emo(0), h_eri(0) + h_emi(0), 0.0, 1.0e-12,
                 "(C) single moderate step conserves E_r + e_mat (hot)");
  test.CheckNear(h_ero(1) + h_emo(1), h_eri(1) + h_emi(1), 0.0, 1.0e-12,
                 "(C) single moderate step conserves E_r + e_mat (cold)");

  // (D) stiff stability: one step reaches equilibrium from either side, finite, bounded.
  // scenarios 2 and 3 share S = 1 (same T_eq, em_eq, er_eq as battery A).
  test.CheckTrue(std::isfinite(h_emo(2)) && std::isfinite(h_ero(2)) &&
                 std::isfinite(h_emo(3)) && std::isfinite(h_ero(3)),
                 "(D) stiff coupling stays finite (no blow-up for rate >> 1/dt)");
  test.CheckNear(h_emo(2), em_eq, 1.0e-6, 0.0,
                 "(D) stiff step from hot matter lands at equilibrium e_mat");
  test.CheckNear(h_ero(2), er_eq, 1.0e-6, 0.0,
                 "(D) stiff step from hot matter lands at equilibrium E_r");
  test.CheckNear(h_emo(3), em_eq, 1.0e-6, 0.0,
                 "(D) stiff step from cold matter lands at equilibrium e_mat");
  test.CheckNear(h_ero(3), er_eq, 1.0e-6, 0.0,
                 "(D) stiff step from cold matter lands at equilibrium E_r");
  // no overshoot past equilibrium (monotone approach from each side)
  test.CheckTrue(h_emo(2) <= h_emi(2) + 1.0e-14 && h_emo(2) >= em_eq - 1.0e-9,
                 "(D) stiff hot-matter step does not overshoot equilibrium");
  test.CheckTrue(h_emo(3) >= h_emi(3) - 1.0e-14 && h_emo(3) <= em_eq + 1.0e-9,
                 "(D) stiff cold-matter step does not overshoot equilibrium");
  test.CheckNear(h_ero(2) + h_emo(2), S, 0.0, 1.0e-12,
                 "(D) stiff step conserves E_r + e_mat (hot)");
  test.CheckNear(h_ero(3) + h_emo(3), S, 0.0, 1.0e-12,
                 "(D) stiff step conserves E_r + e_mat (cold)");

  // (E) chi_a = 0 -> exact no-op (state unchanged bit-for-bit)
  test.CheckNear(h_emo(4), h_emi(4), 0.0, 0.0, "(E) chi_a = 0 leaves e_mat unchanged");
  test.CheckNear(h_ero(4), h_eri(4), 0.0, 0.0, "(E) chi_a = 0 leaves E_r unchanged");

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
