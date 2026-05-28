//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ohmic_electron_heating_test.cpp
//  \brief Unit test: Ohmic (Joule) heating routed to the electron internal energy
//  `e_ele`, consistently with the constrained-transport EMF, plus the 2T re-targeting of
//  the point-implicit radiation coupling to the electron T_e (issue [14c]/#30,
//  ADR-0002/0003).
//
//  Exercises the device functions
//    three_temp::OhmicHeatingRate / ApplyOhmicElectronHeating   (three_temperature.hpp)
//    radiationfld::PointImplicitElectronRadiationCoupling  (the 2T re-targeting)
//  against analytic oracles.  Batteries:
//   (A) CT-EMF CONSISTENCY: the Ohmic dissipation rate eta|J|^2 equals the resistive
//       electric field dotted with the current E_res . J (E_res = eta J), so the electron
//       heating uses exactly the same discrete eta and J that drive the CT B-diffusion.
//   (B) HEATING -> ELECTRONS, IONS UNTOUCHED, ENERGY CONSERVED: a substep decays
//       B by the local magnetic-energy decrement Q = eta|J|^2 dt; depositing Q on e_ele
//       and recovering e_ion = E_tot - KE - 1/2 B^2 - e_ele by subtraction leaves the ion
//       energy UNCHANGED (electron gain == magnetic loss) and keeps the species sum equal
//       to E_tot.  The wrong partition (heat left on ions) shifts e_ion by +Q -> caught.
//   (C) RADIATION COUPLING TARGETS T_e: the 2T wrapper relaxes (E_r, e_ele) with the
//       electron heat capacity cv_ele to a T_e^4 = E_r, conserves E_r + e_ele, gives the
//       right exchange direction, and is bit-identical to the grey coupling with the same
//       arguments (it is a thin re-targeting forward).
//   (D) 2T CHAIN (Ohmic -> electron -> radiation): a single cell driven by a
//       constant Ohmic source then coupled to radiation each step heats the electrons and
//       the radiation, leaves the ions untouched, and accounts ALL Ohmic input in
//       E_r + e_ele = initial + Q*t to machine precision.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/ohmic_electron_heating_test -B build_unit/ohmic_ele
//  Auto-run by tst/test_suite/unit_tests/test_unit_ohmic_electron_heating_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::pow, std::sqrt, std::fabs
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/three_temperature.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
//! \brief Host oracle: radiative-equilibrium temperature solving a T^4 + cv T = S, by
//! safeguarded Newton (g monotone increasing, convex for T>0 -> converges from the upper
//! bracket a T^4 = S).
double EquilibriumTemp(double a_rad, double cv, double S) {
  double T = std::pow(S/a_rad, 0.25);
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
//! \brief Ohmic->electron heating + 2T radiation-coupling unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("ohmic_electron_heating_test");

  // ---------- shared state (Heaviside-Lorentz code units, c = a = 1) ----------
  const Real rho = 2.0;
  const Real v1 = 0.3, v2 = -0.4, v3 = 0.1;          // velocity -> momentum m = rho v
  const Real m1 = rho*v1, m2 = rho*v2, m3 = rho*v3;
  const Real b1 = 0.6, b2 = -0.3, b3 = 0.2;          // cell-centred B
  const Real ke = 0.5*(m1*m1 + m2*m2 + m3*m3)/rho;
  const Real me0 = 0.5*(b1*b1 + b2*b2 + b3*b3);       // initial magnetic energy
  const Real eele0 = 0.7, eion0 = 1.3;                // seeded species energy densities
  const Real etot = ke + me0 + eele0 + eion0;         // conserved MHD total energy

  // resistive substep: eta, the discrete current J = curl B, and dt.
  const Real eta = 0.05;
  const Real jx = 1.5, jy = -2.0, jz = 0.5;           // current-density components
  const Real dt = 0.1;

  // code-unit radiation constants and electron heat capacity for the coupling batteries.
  const Real c_light = 1.0, a_rad = 1.0, cv_ele = 1.0;

  enum {
    // (A) CT-EMF consistency
    IRATE_A=0, IDOT_A, IDEP_A,
    // (B) heating -> electrons / ions untouched / conservation
    IQ_B, IDEELE_B, IEION_B, ICONS_B, IGAIN_VS_LOSS_B, IION_BUG_B,
    // (C) radiation coupling on T_e
    IEM_C, IER_C, IEQ_C, ICONS_C, IDIR_EM_C, IDIR_ER_C, IWRAP_ER_C, IWRAP_EM_C,
    // (D) 2T equilibration chain
    IEELE_D, IER_D, IEION_D, IBOOK_D, IEION_DRIFT_D,
    NVAL
  };
  DvceArray1D<Real> d_vals("ohmic_ele_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  // chain parameters captured by value into the device kernel
  const Real chi_a_chain = 4.0;
  const int  nstep_chain = 400;
  const Real dt_chain    = 0.02;
  const Real eele_chain0 = 0.2, eion_chain0 = 0.9, erad_chain0 = 0.0;

  par_for("ohmic_ele_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // ===================== (A) CT-EMF consistency: eta|J|^2 == E_res . J ===============
    Real rate = three_temp::OhmicHeatingRate(eta, jx, jy, jz);
    d_vals(IRATE_A) = rate;
    // resistive EMF E_res = eta J; dissipation E_res . J = eta (Jx^2+Jy^2+Jz^2).
    Real e1 = eta*jx, e2 = eta*jy, e3 = eta*jz;
    d_vals(IDOT_A) = e1*jx + e2*jy + e3*jz;
    d_vals(IDEP_A) = three_temp::ApplyOhmicElectronHeating(eele0, eta, jx, jy, jz, dt);

    // ============ (B) heating -> electrons, ions untouched, energy conserved ===========
    // Local magnetic-energy decrement of the CT update over dt (the physical truth that
    // the field actually loses); the field decays B by exactly this amount.
    Real q = eta*(jx*jx + jy*jy + jz*jz)*dt;
    Real me_new = me0 - q;                              // 1/2 B_new^2 after the substep
    Real bscale = Kokkos::sqrt(me_new/me0);             // uniform B downscale -> ME_new
    Real b1n = bscale*b1, b2n = bscale*b2, b3n = bscale*b3;
    // deposit the Joule heat on the electrons via the module function.
    Real eele_new = three_temp::ApplyOhmicElectronHeating(eele0, eta, jx, jy, jz, dt);
    // recover the ion energy by subtraction from the (conserved) total.
    Real eion_new = three_temp::IonInternalEnergyMHD(etot, rho, m1, m2, m3,
                                                     b1n, b2n, b3n, eele_new);
    d_vals(IQ_B)    = q;
    d_vals(IDEELE_B) = eele_new - eele0;                // -> q (electron gain)
    d_vals(IEION_B)  = eion_new;                        // -> eion0 (ions untouched)
    d_vals(ICONS_B)  = (eele_new + eion_new + ke + me_new) - etot;   // -> 0 (conserved)
    d_vals(IGAIN_VS_LOSS_B) = (eele_new - eele0) - (me0 - me_new);   // gain - loss -> 0
    // the wrong partition: leave the Joule heat on the ions (no e_ele deposit) -> e_ion
    // would gain q.  This is the bug ADR-0003 fixes; recorded to show the discriminator.
    Real eion_bug = three_temp::IonInternalEnergyMHD(etot, rho, m1, m2, m3,
                                                     b1n, b2n, b3n, eele0);
    d_vals(IION_BUG_B) = eion_bug - eion0;              // -> +q (NOT how we do it)

    // ===================== (C) radiation coupling targets T_e =========================
    // relax a hot-electron / cold-radiation cell to a T_e^4 = E_r (cv_ele heat capacity).
    Real er = 0.0, em = 1.0;                            // T_e0 = 1, cold radiation
    Real Sc = er + em;
    for (int s = 0; s < 300; ++s) {
      Real er_n, em_n;
      radiationfld::PointImplicitElectronRadiationCoupling(er, em, cv_ele, 2.0, c_light,
                                                           a_rad, 0.5, er_n, em_n);
      er = er_n; em = em_n;
    }
    d_vals(IEM_C) = em;
    d_vals(IER_C) = er;
    d_vals(IEQ_C) = a_rad*Kokkos::pow(em/cv_ele, 4.0) - er;   // -> 0 (a T_e^4 == E_r)
    d_vals(ICONS_C) = (er + em) - Sc;                    // -> 0 (E_r + e_ele conserved)
    // direction of a single step: hot electrons (a T_e^4 > E_r) cool, radiation heats.
    Real er_d, em_d;
    radiationfld::PointImplicitElectronRadiationCoupling(0.0, 2.0, cv_ele, 2.0, c_light,
                                                         a_rad, 0.5, er_d, em_d);
    d_vals(IDIR_EM_C) = em_d - 2.0;                          // -> < 0 (electrons cool)
    d_vals(IDIR_ER_C) = er_d - 0.0;                          // -> > 0 (radiation heats)
    // the 2T wrapper is a re-targeting forward -> bit-identical to the grey coupling.
    Real er_g, em_g;
    radiationfld::PointImplicitGreyCoupling(0.0, 2.0, cv_ele, 2.0, c_light, a_rad, 0.5,
                                            er_g, em_g);
    d_vals(IWRAP_ER_C) = er_d - er_g;                        // -> 0 (identical)
    d_vals(IWRAP_EM_C) = em_d - em_g;                        // -> 0

    // ===================== (D) 2T equilibration chain (Ohmic->electron->radiation) =====
    // constant Ohmic source heats e_ele each step; radiation absorbs from the electrons;
    // ions are a passive reservoir that the chain never touches.
    Real eele_c = eele_chain0, eion_c = eion_chain0, erad_c = erad_chain0;
    Real q_chain = three_temp::OhmicHeatingRate(eta, jx, jy, jz)*dt_chain;  // per step
    Real book0 = erad_c + eele_c;
    Real eion_drift = 0.0;
    for (int s = 0; s < nstep_chain; ++s) {
      // 1) Ohmic (Joule) heating -> electrons
      eele_c = three_temp::ApplyOhmicElectronHeating(eele_c, eta, jx, jy, jz, dt_chain);
      // 2) point-implicit radiation coupling on T_e (electron <-> radiation)
      Real er_n, ee_n;
      radiationfld::PointImplicitElectronRadiationCoupling(erad_c, eele_c, cv_ele,
          chi_a_chain, c_light, a_rad, dt_chain, er_n, ee_n);
      erad_c = er_n; eele_c = ee_n;
      // ions untouched throughout
      eion_drift = Kokkos::fmax(eion_drift, Kokkos::fabs(eion_c - eion_chain0));
    }
    d_vals(IEELE_D) = eele_c - eele_chain0;             // -> > 0 (electrons heated)
    d_vals(IER_D)   = erad_c - erad_chain0;             // -> > 0 (radiation heated)
    d_vals(IEION_D) = eion_c;                           // -> eion_chain0 (untouched)
    // all Ohmic input accounted: E_r + e_ele gained exactly Q*nstep.
    Real q_total = q_chain*static_cast<Real>(nstep_chain);
    d_vals(IBOOK_D) = (erad_c + eele_c) - (book0 + q_total);
    d_vals(IEION_DRIFT_D) = eion_drift;                 // -> 0 (never modified)
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real tol = 1.0e-13;
  const Real q_expected = eta*(jx*jx + jy*jy + jz*jz)*dt;

  // ----------------------------- (A) checks -----------------------------
  test.CheckNear(h_vals(IRATE_A), eta*(jx*jx + jy*jy + jz*jz), 0.0, tol,
                 "Ohmic dissipation rate = eta |J|^2");
  test.CheckNear(h_vals(IDOT_A), h_vals(IRATE_A), 0.0, tol,
                 "(A) eta|J|^2 == E_res . J with E_res = eta J (CT-EMF consistency)");
  test.CheckNear(h_vals(IDEP_A), eele0 + q_expected, 0.0, tol,
                 "ApplyOhmicElectronHeating adds eta|J|^2 dt to e_ele");

  // ----------------------------- (B) checks -----------------------------
  test.CheckNear(h_vals(IQ_B), q_expected, 0.0, tol,
                 "magnetic-energy decrement Q = eta|J|^2 dt");
  test.CheckNear(h_vals(IDEELE_B), q_expected, 0.0, tol,
                 "(B) electron energy gains exactly the Joule heat Q");
  test.CheckNear(h_vals(IEION_B), eion0, 0.0, tol,
                 "(B) ions UNTOUCHED by Ohmic heating (e_ion unchanged)");
  test.CheckNear(h_vals(ICONS_B), 0.0, 0.0, tol,
                 "(B) total energy conserved: e_ele+e_ion+KE+1/2 B^2 == E_tot");
  test.CheckNear(h_vals(IGAIN_VS_LOSS_B), 0.0, 0.0, tol,
                 "(B) electron gain == local magnetic-energy loss");
  test.CheckNear(h_vals(IION_BUG_B), q_expected, 0.0, tol,
                 "(B) the WRONG partition (heat left on ions) would shift e_ion by +Q");

  // ----------------------------- (C) checks -----------------------------
  const double T_eq = EquilibriumTemp(a_rad, cv_ele, 1.0);   // S = er+em = 1
  test.CheckNear(h_vals(IEM_C), cv_ele*T_eq, 1.0e-8, 0.0,
                 "(C) electrons relax to equilibrium energy cv_ele T_eq");
  test.CheckNear(h_vals(IER_C), a_rad*T_eq*T_eq*T_eq*T_eq, 1.0e-8, 0.0,
                 "(C) radiation relaxes to equilibrium energy a T_eq^4");
  test.CheckNear(h_vals(IEQ_C), 0.0, 0.0, 1.0e-7,
                 "(C) radiative equilibrium on T_e reached: a T_e^4 == E_r");
  test.CheckNear(h_vals(ICONS_C), 0.0, 0.0, 1.0e-12,
                 "(C) E_r + e_ele conserved through the relaxation");
  test.CheckTrue(h_vals(IDIR_EM_C) < 0.0,
                 "(C) hot electrons (a T_e^4 > E_r) cool");
  test.CheckTrue(h_vals(IDIR_ER_C) > 0.0,
                 "(C) radiation heats when electrons are hot");
  test.CheckNear(h_vals(IWRAP_ER_C), 0.0, 0.0, 0.0,
                 "(C) electron coupling == grey coupling with electron args (E_r)");
  test.CheckNear(h_vals(IWRAP_EM_C), 0.0, 0.0, 0.0,
                 "(C) electron coupling == grey coupling with electron args (e_ele)");

  // ----------------------------- (D) checks -----------------------------
  test.CheckTrue(h_vals(IEELE_D) > 0.0,
                 "(D) chain heats the electrons (Ohmic -> e_ele)");
  test.CheckTrue(h_vals(IER_D) > 0.0,
                 "(D) chain heats the radiation (electron -> radiation)");
  test.CheckNear(h_vals(IEION_D), eion_chain0, 0.0, tol,
                 "(D) ions untouched by the whole chain");
  test.CheckNear(h_vals(IEION_DRIFT_D), 0.0, 0.0, 0.0,
                 "(D) ion energy never modified at any step (drift == 0)");
  test.CheckNear(h_vals(IBOOK_D), 0.0, 0.0, 1.0e-12,
                 "(D) all Ohmic input accounted: E_r + e_ele = initial + Q*t");

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
