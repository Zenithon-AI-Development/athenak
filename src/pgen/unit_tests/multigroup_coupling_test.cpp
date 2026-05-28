//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file multigroup_coupling_test.cpp
//  \brief Unit test: per-cell point-implicit (backward-Euler) MULTIGROUP matter-radiation
//  coupling solved as an arrowhead system (O(N_group) Schur + Newton over T_e) plus the
//  electron-ion exchange (radiationfld::MultigroupArrowheadCoupling), exercised on real
//  device kernels (issue [17b]/#35, ADR-0001/0002).
//
//  The coupling exchanges energy between N photon-energy radiation groups E_g, the
//  electrons (T_e) and the ions (T_i), with the groups coupling only through T_e:
//     dE_g/dt   = c chi_g (B_g(T_e) - E_g),   B_g = a T_e^4 [F(x_{g+1}) - F(x_g)],
//     de_ele/dt = -sum_g dE_g/dt + c_ei (T_i - T_e),    de_ion/dt = -c_ei (T_i - T_e),
//  conserving sum_g E_g + e_ele + e_ion and relaxing the radiation to the Planck spectrum
//  E_g = B_g(T_e) at the common equilibrium temperature.
//
//  Batteries:
//   (P) PLANCK FRACTION F(x): endpoints F(0)=0, F(inf)->1, monotone, series continuity at
//       x=1, partition sum -> 1, and -- the key discriminator -- agreement with an
//       INDEPENDENT host quadrature of (15/pi^4) int_0^x t^3/(e^t-1) dt (catches a wrong
//       normalisation/coefficient that a self-consistent solver check would miss).
//   (A) Multigroup EQUILIBRATION to the Planck spectrum: cold groups + hot electrons
//       relax so every group lands on E_g = B_g(T_e_final); energy conserved; stationary.
//   (B) ENERGY conservation to machine precision every step (max drift).
//   (C) REDUCES TO GREY: N=1 group spanning the spectrum, c_ei=0 -> matches the grey
//       PointImplicitGreyCoupling (#23) bit-closely.
//   (D) ELECTRON-ION exchange: chi=0, c_ei>0, T_e != T_i -> relax to common temperature
//       (cv_e T_e + cv_i T_i)/(cv_e+cv_i); radiation untouched; energy conserved.
//   (E) STIFF stability: c chi_g dt = 1e8 -> a single step lands each group on its Planck
//       slice, finite and bounded.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/multigroup_coupling_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_multigroup_coupling_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::fabs, std::pow, std::isfinite, std::expm1
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "radiation_fld/multigroup_coupling.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
//! \brief INDEPENDENT host oracle for the fractional Planck function
//!   F(x) = (15/pi^4) int_0^x t^3/(e^t-1) dt
//! by composite Simpson quadrature (fine grid) -- deliberately not the series in
//! multigroup_coupling.hpp, so a wrong coefficient/normalisation there is caught.
double PlanckFractionQuad(double x) {
  if (x <= 0.0) { return 0.0; }
  const int n = 200000;                 // even number of intervals
  const double h = x/static_cast<double>(n);
  auto integrand = [](double t) -> double {
    if (t <= 0.0) { return 0.0; }       // t^3/(e^t-1) -> t^2 -> 0
    return t*t*t/std::expm1(t);
  };
  double s = integrand(0.0) + integrand(x);
  for (int i = 1; i < n; ++i) {
    double t = i*h;
    s += (i & 1) ? 4.0*integrand(t) : 2.0*integrand(t);
  }
  double integral = s*h/3.0;
  const double c15pi4 = 15.0/(M_PI*M_PI*M_PI*M_PI);
  return c15pi4*integral;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Point-implicit multigroup arrowhead matter-radiation coupling unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("multigroup_coupling_test");

  // code-unit physical constants (c = a = kb = 1 keeps the equilibrium algebra clean).
  const Real c_light = 1.0, a_rad = 1.0, kboltz = 1.0;

  // ============================ (P) PLANCK FRACTION F(x) ============================
  // Host-callable (KOKKOS_INLINE_FUNCTION); the device path is exercised by (A)-(E).
  test.CheckNear(radiationfld::PlanckFraction(0.0), 0.0, 0.0, 0.0,
                 "(P) F(0) = 0");
  test.CheckNear(radiationfld::PlanckFraction(200.0), 1.0, 0.0, 1.0e-10,
                 "(P) F(x) -> 1 as x -> infinity");
  // monotone increasing
  bool mono = true;
  Real fprev = -1.0;
  for (Real x = 0.0; x <= 30.0; x += 0.25) {
    Real f = radiationfld::PlanckFraction(x);
    if (f < fprev - 1.0e-14) { mono = false; }
    fprev = f;
  }
  test.CheckTrue(mono, "(P) F(x) is monotone increasing");
  // series continuity across the x = 1 switch (small-x vs large-x series)
  test.CheckNear(radiationfld::PlanckFraction(0.9999999),
                 radiationfld::PlanckFraction(1.0000001), 1.0e-6, 1.0e-9,
                 "(P) F continuous across the x=1 series switch");
  // agreement with the independent host quadrature oracle at several x
  const Real xs[5] = {0.5, 1.0, 2.0, 4.0, 8.0};
  for (int q = 0; q < 5; ++q) {
    Real f = radiationfld::PlanckFraction(xs[q]);
    Real fq = static_cast<Real>(PlanckFractionQuad(static_cast<double>(xs[q])));
    test.CheckNear(f, fq, 1.0e-8, 1.0e-10,
                   "(P) F(x) matches independent quadrature oracle");
  }
  // a spanning partition's group fractions sum to 1 (sum_g B_g = a T^4)
  {
    const Real bnds[6] = {1.0e-3, 0.3, 1.0, 3.0, 10.0, 1.0e3};
    Real sumfrac = 0.0;
    for (int g = 0; g < 5; ++g) {
      sumfrac += radiationfld::PlanckFraction(bnds[g+1])
                 - radiationfld::PlanckFraction(bnds[g]);
    }
    test.CheckNear(sumfrac, 1.0, 0.0, 1.0e-9,
                   "(P) spanning group fractions sum to 1");
  }
  // x F'(x) consistency with a central difference of F
  {
    Real x = 2.5, hh = 1.0e-5;
    Real num = x*(radiationfld::PlanckFraction(x+hh)
                  - radiationfld::PlanckFraction(x-hh))/(2.0*hh);
    test.CheckNear(radiationfld::PlanckFractionXDeriv(x), num, 1.0e-6, 1.0e-9,
                   "(P) x F'(x) = PlanckFractionXDeriv matches d/dx of F");
  }

  // =============== (A)+(B)+(E) device equilibration / conservation =================
  // 4 groups spanning the spectrum; cold radiation + hot electrons, no ions (c_ei=0).
  const int NGA = 4;
  DvceArray1D<Real> d_egA("egA", NGA);
  DvceArray1D<Real> d_diagA("diagA", 6);  // Te, maxdrift, planck_resid, stat, S, e_ele
  const Real dtA = 0.5;
  const int nstepA = 400;
  par_for("mg_equil", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real gb[NGA+1] = {0.01, 0.5, 2.0, 8.0, 100.0};
    Real chi[NGA] = {1.0, 2.0, 4.0, 8.0};
    Real eold[NGA], enew[NGA];
    for (int g = 0; g < NGA; ++g) { eold[g] = 0.0; }   // cold radiation, all groups
    Real cv_ele = 1.0, cv_ion = 0.0;
    Real e_ele = 3.0, e_ion = 0.0;                     // hot electrons (T_e0 = 3)
    Real S = e_ele;
    for (int g = 0; g < NGA; ++g) { S += eold[g]; }
    Real maxdrift = 0.0;
    for (int s = 0; s < nstepA; ++s) {
      Real ele_new, ion_new;
      radiationfld::MultigroupArrowheadCoupling(NGA, eold, gb, chi, e_ele, e_ion, cv_ele,
          cv_ion, c_light, a_rad, kboltz, 0.0, dtA, enew, ele_new, ion_new);
      Real tot = ele_new + ion_new;
      for (int g = 0; g < NGA; ++g) { tot += enew[g]; }
      maxdrift = Kokkos::fmax(maxdrift, Kokkos::fabs(tot - S));
      for (int g = 0; g < NGA; ++g) { eold[g] = enew[g]; }
      e_ele = ele_new; e_ion = ion_new;
    }
    // final electron temperature + Planck-spectrum residual E_g - B_g(T_e)
    Real te = e_ele/cv_ele;
    Real te4 = te*te*te*te;
    Real inv_kt = 1.0/(kboltz*te);
    Real planck_resid = 0.0;
    for (int g = 0; g < NGA; ++g) {
      Real bg = a_rad*te4*(radiationfld::PlanckFraction(gb[g+1]*inv_kt)
                           - radiationfld::PlanckFraction(gb[g]*inv_kt));
      planck_resid = Kokkos::fmax(planck_resid, Kokkos::fabs(eold[g] - bg));
    }
    // stationarity: one more step changes nothing (fixed point)
    Real enew2[NGA], ele2, ion2;
    radiationfld::MultigroupArrowheadCoupling(NGA, eold, gb, chi, e_ele, e_ion, cv_ele,
        cv_ion, c_light, a_rad, kboltz, 0.0, dtA, enew2, ele2, ion2);
    Real stat = Kokkos::fabs(ele2 - e_ele);
    for (int g = 0; g < NGA; ++g) {
      stat = Kokkos::fmax(stat, Kokkos::fabs(enew2[g] - eold[g]));
    }
    d_diagA(0) = te; d_diagA(1) = maxdrift; d_diagA(2) = planck_resid;
    d_diagA(3) = stat; d_diagA(4) = S; d_diagA(5) = e_ele;
    for (int g = 0; g < NGA; ++g) { d_egA(g) = eold[g]; }
  });
  auto h_egA = Kokkos::create_mirror_view(d_egA);
  auto h_diagA = Kokkos::create_mirror_view(d_diagA);
  Kokkos::deep_copy(h_egA, d_egA);
  Kokkos::deep_copy(h_diagA, d_diagA);

  test.CheckNear(h_diagA(2), 0.0, 0.0, 1.0e-8,
                 "(A) radiation relaxes to the Planck spectrum E_g = B_g(T_e)");
  test.CheckNear(h_diagA(3), 0.0, 0.0, 1.0e-9,
                 "(A) equilibrium state is stationary (fixed point of the step)");
  test.CheckNear(h_diagA(1), 0.0, 0.0, 1.0e-12*h_diagA(4),
                 "(B) total energy sum_g E_g + e_ele conserved every step");
  // explicit host re-check that the final spectrum is Planck at the final T_e
  {
    const Real gb[NGA+1] = {0.01, 0.5, 2.0, 8.0, 100.0};
    Real te = h_diagA(0);
    Real te4 = te*te*te*te;
    Real inv_kt = 1.0/(kboltz*te);
    Real maxrel = 0.0;
    for (int g = 0; g < NGA; ++g) {
      Real bg = a_rad*te4*(radiationfld::PlanckFraction(gb[g+1]*inv_kt)
                           - radiationfld::PlanckFraction(gb[g]*inv_kt));
      maxrel = std::fmax(maxrel, std::fabs(h_egA(g) - bg));
    }
    test.CheckNear(maxrel, 0.0, 0.0, 1.0e-8,
                   "(A) host re-check: every group equals its Planck slice");
  }

  // =================== (C) REDUCES TO THE GREY CASE (single group) ==================
  // N=1 group spanning the spectrum (F -> 1), c_ei = 0, beta = c chi dt = 1.
  const Real chiC = 2.0, dtC = 0.5, er0C = 0.0, ee0C = 1.0;
  DvceArray1D<Real> d_C("Cout", 4);   // e_rad_new, e_ele_new, grey_er, grey_em
  par_for("mg_grey", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real gb[2] = {1.0e-3, 1.0e5};   // x in [1e-3, 1e5]/T -> F(hi)-F(lo) ~ 1
    Real chi[1] = {chiC};
    Real eold[1] = {er0C}, enew[1];
    Real ele_new, ion_new;
    radiationfld::MultigroupArrowheadCoupling(1, eold, gb, chi, ee0C, 0.0, 1.0, 0.0,
        c_light, a_rad, kboltz, 0.0, dtC, enew, ele_new, ion_new);
    // grey reference (#23): same numbers, single grey energy + matter reservoir
    Real grey_er, grey_em;
    radiationfld::PointImplicitGreyCoupling(er0C, ee0C, 1.0, chiC, c_light, a_rad, dtC,
                                            grey_er, grey_em);
    d_C(0) = enew[0]; d_C(1) = ele_new; d_C(2) = grey_er; d_C(3) = grey_em;
  });
  auto h_C = Kokkos::create_mirror_view(d_C);
  Kokkos::deep_copy(h_C, d_C);
  test.CheckNear(h_C(0), h_C(2), 1.0e-7, 1.0e-9,
                 "(C) single-group radiation energy matches the grey solver");
  test.CheckNear(h_C(1), h_C(3), 1.0e-7, 1.0e-9,
                 "(C) single-group electron energy matches the grey matter energy");

  // ======================= (D) ELECTRON-ION EXCHANGE (chi=0) ========================
  // No radiation coupling (chi=0 -> groups frozen); electrons & ions relax to common T.
  const int NGD = 2;
  DvceArray1D<Real> d_egD("egD", NGD);
  DvceArray1D<Real> d_diagD("diagD", 6);  // Te, Ti, maxdrift, therm_drift, S, Tcommon
  const Real dtD = 0.5;
  const int nstepD = 200;
  par_for("mg_ei", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real gb[NGD+1] = {0.1, 1.0, 10.0};
    Real chi[NGD] = {0.0, 0.0};                  // radiation decoupled
    Real eold[NGD] = {1.5, 0.7}, enew[NGD];
    Real cv_ele = 1.0, cv_ion = 2.0, c_ei = 1.0;
    Real e_ele = 3.0, e_ion = 2.0;               // T_e0 = 3, T_i0 = 1
    Real Tcommon = (cv_ele*(e_ele/cv_ele) + cv_ion*(e_ion/cv_ion))/(cv_ele + cv_ion);
    Real therm0 = cv_ele*(e_ele/cv_ele) + cv_ion*(e_ion/cv_ion);
    Real S = e_ele + e_ion;
    for (int g = 0; g < NGD; ++g) { S += eold[g]; }
    Real maxdrift = 0.0, therm_drift = 0.0;
    for (int s = 0; s < nstepD; ++s) {
      Real ele_new, ion_new;
      radiationfld::MultigroupArrowheadCoupling(NGD, eold, gb, chi, e_ele, e_ion, cv_ele,
          cv_ion, c_light, a_rad, kboltz, c_ei, dtD, enew, ele_new, ion_new);
      Real tot = ele_new + ion_new;
      for (int g = 0; g < NGD; ++g) { tot += enew[g]; }
      maxdrift = Kokkos::fmax(maxdrift, Kokkos::fabs(tot - S));
      Real therm = ele_new + ion_new;   // cv_e T_e + cv_i T_i = e_ele + e_ion
      therm_drift = Kokkos::fmax(therm_drift, Kokkos::fabs(therm - therm0));
      for (int g = 0; g < NGD; ++g) { eold[g] = enew[g]; }
      e_ele = ele_new; e_ion = ion_new;
    }
    d_diagD(0) = e_ele/cv_ele; d_diagD(1) = e_ion/cv_ion; d_diagD(2) = maxdrift;
    d_diagD(3) = therm_drift; d_diagD(4) = S; d_diagD(5) = Tcommon;
    for (int g = 0; g < NGD; ++g) { d_egD(g) = eold[g]; }
  });
  auto h_egD = Kokkos::create_mirror_view(d_egD);
  auto h_diagD = Kokkos::create_mirror_view(d_diagD);
  Kokkos::deep_copy(h_egD, d_egD);
  Kokkos::deep_copy(h_diagD, d_diagD);
  test.CheckNear(h_diagD(0), h_diagD(5), 1.0e-8, 1.0e-10,
                 "(D) electron temperature relaxes to the common temperature");
  test.CheckNear(h_diagD(1), h_diagD(5), 1.0e-8, 1.0e-10,
                 "(D) ion temperature relaxes to the common temperature");
  test.CheckNear(h_diagD(3), 0.0, 0.0, 1.0e-10,
                 "(D) electron-ion exchange conserves cv_e T_e + cv_i T_i");
  test.CheckNear(h_egD(0), 1.5, 0.0, 0.0, "(D) chi=0 leaves group 0 untouched");
  test.CheckNear(h_egD(1), 0.7, 0.0, 0.0, "(D) chi=0 leaves group 1 untouched");
  test.CheckNear(h_diagD(2), 0.0, 0.0, 1.0e-12*h_diagD(4),
                 "(D) total energy conserved with electron-ion exchange");

  // ============================ (E) STIFF single step ==============================
  const int NGE = 3;
  DvceArray1D<Real> d_diagE("diagE", 3);   // planck_resid, finite(1/0), Te
  par_for("mg_stiff", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real gb[NGE+1] = {0.05, 1.0, 5.0, 50.0};
    Real chi[NGE] = {1.0e8, 1.0e8, 1.0e8};   // c chi dt = 1e8
    Real eold[NGE] = {0.0, 0.0, 0.0}, enew[NGE];
    Real ele_new, ion_new;
    radiationfld::MultigroupArrowheadCoupling(NGE, eold, gb, chi, 4.0, 0.0, 1.0, 0.0,
        c_light, a_rad, kboltz, 0.0, 1.0, enew, ele_new, ion_new);
    Real te = ele_new;
    Real te4 = te*te*te*te;
    Real inv_kt = 1.0/(kboltz*te);
    Real resid = 0.0;
    bool fin = Kokkos::isfinite(ele_new);
    for (int g = 0; g < NGE; ++g) {
      Real bg = a_rad*te4*(radiationfld::PlanckFraction(gb[g+1]*inv_kt)
                           - radiationfld::PlanckFraction(gb[g]*inv_kt));
      resid = Kokkos::fmax(resid, Kokkos::fabs(enew[g] - bg));
      fin = fin && Kokkos::isfinite(enew[g]);
    }
    d_diagE(0) = resid; d_diagE(1) = fin ? 1.0 : 0.0; d_diagE(2) = te;
  });
  auto h_diagE = Kokkos::create_mirror_view(d_diagE);
  Kokkos::deep_copy(h_diagE, d_diagE);
  test.CheckTrue(h_diagE(1) > 0.5,
                 "(E) stiff coupling stays finite (no blow-up for rate >> 1/dt)");
  test.CheckNear(h_diagE(0), 0.0, 0.0, 1.0e-6,
                 "(E) stiff single step lands each group on its Planck slice");

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
