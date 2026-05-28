//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file three_temp_test.cpp
//  \brief Unit test: conservative 3T energy reconciliation -- electron internal energy
//  `e_ele` (advected as the passive scalar e_ele/rho) + PdV pressure-fraction split +
//  ion-by-subtraction (issue [14a]/#19, ADR-0002).
//
//  Exercises the device functions in eos/three_temperature.hpp against analytic oracles.
//  Batteries:
//   (A) e_ion by subtraction + total-energy conservation (hydro and MHD): recovering
//       e_ion = E_tot - KE [- 1/2 B^2] - e_ele reproduces the seeded ion energy, and the
//       species sum equals E_tot exactly (conservation by construction).
//   (B) PdV split by pressure fraction: the electron PdV increment -p_ele*div(v)*dt
//       equals f_ele = p_ele/p_gas times the total reversible work -p_gas*div(v)*dt; the
//       ion gets the complementary (1-f_ele) share; compression heats, expansion cools.
//   (C) Shock dissipation lands on ions: when the gas internal energy rises by more than
//       the reversible electron PdV (irreversible shock heat in E_tot), recovering e_ion
//       by subtraction puts ALL of the excess on the ions and none on the electrons.
//   (D) 1T limit: with T_e = T_i (ideal gas, shared gamma) the 3T partition sums to the
//       single-temperature internal energy / pressure, and the species pressure fractions
//       are preserved under PdV compression (the gas stays single-temperature).
//   (E) EOS-fed pressures (ties #6, previews #26): invert a specific electron/ion energy
//       to T_e/T_i via eos_table_3t::EosTable3T, look up p_ele/p_ion, and verify the
//       resulting pressure fraction matches the synthetic ideal-gas closed form.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/three_temp_test -B build_unit/three_temp
//    (cd build_unit/.../src && make) && ./athena -i <three_temp_test.athinput>
//  Auto-run by tst/test_suite/unit_tests/test_unit_three_temp_cpu.py.

#include <cmath>     // std::pow
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/three_temperature.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// --- Battery E synthetic ideal-gas EOS table (per-species power laws; p=(g-1) rho e) ---
constexpr Real CVE = 1.5, AE =  0.25, GE = 5.0/3.0;   // electron Cv, rho-exponent, gamma
constexpr Real CVI = 0.9, AI = -0.10, GI = 5.0/3.0;   // ion
constexpr int  NT = 11, NR = 9;                       // non-square (catches axis swap)
constexpr Real TMIN = 1.0,    TMAX = 1.0e4;
constexpr Real RMIN = 1.0e-3, RMAX = 1.0e1;
Real EAnalytic(int s, Real rho, Real temp) {          // host fill helper
  Real cv = (s == eos_table_3t::kElectron) ? CVE : CVI;
  Real a  = (s == eos_table_3t::kElectron) ? AE  : AI;
  return cv*std::pow(rho, a)*temp;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Conservative 3T energy reconciliation unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("three_temp_test");

  // ---------- build the synthetic ideal-gas 3T EOS table for Battery E ----------
  eos_table_3t::EosTable3T table;
  table.Allocate(NT, NR, TMIN, TMAX, RMIN, RMAX);
  auto h_e_ele  = Kokkos::create_mirror_view(table.e_ele);
  auto h_e_ion  = Kokkos::create_mirror_view(table.e_ion);
  auto h_p_ele  = Kokkos::create_mirror_view(table.p_ele);
  auto h_p_ion  = Kokkos::create_mirror_view(table.p_ion);
  auto h_zbar   = Kokkos::create_mirror_view(table.zbar);
  auto h_cv_ele = Kokkos::create_mirror_view(table.cv_ele);
  auto h_cv_ion = Kokkos::create_mirror_view(table.cv_ion);
  for (int it = 0; it < NT; ++it) {
    Real temp = table.TempAt(it);
    for (int ir = 0; ir < NR; ++ir) {
      Real rho = table.RhoAt(ir);
      h_e_ele(it, ir)  = EAnalytic(eos_table_3t::kElectron, rho, temp);
      h_e_ion(it, ir)  = EAnalytic(eos_table_3t::kIon, rho, temp);
      h_cv_ele(it, ir) = CVE*std::pow(rho, AE);
      h_cv_ion(it, ir) = CVI*std::pow(rho, AI);
      h_p_ele(it, ir)  = (GE - 1.0)*rho*h_e_ele(it, ir);
      h_p_ion(it, ir)  = (GI - 1.0)*rho*h_e_ion(it, ir);
      h_zbar(it, ir)   = 1.0;
    }
  }
  Kokkos::deep_copy(table.e_ele, h_e_ele);
  Kokkos::deep_copy(table.e_ion, h_e_ion);
  Kokkos::deep_copy(table.p_ele, h_p_ele);
  Kokkos::deep_copy(table.p_ion, h_p_ion);
  Kokkos::deep_copy(table.zbar, h_zbar);
  Kokkos::deep_copy(table.cv_ele, h_cv_ele);
  Kokkos::deep_copy(table.cv_ion, h_cv_ion);

  // ---------- shared test state ----------
  const Real rho = 2.0;
  const Real v1 = 0.3, v2 = -0.4, v3 = 0.1;      // velocity -> momentum density m = rho v
  const Real m1 = rho*v1, m2 = rho*v2, m3 = rho*v3;
  const Real b1 = 0.5, b2 = -0.2, b3 = 0.3;      // cell-centred B (Heaviside-Lorentz)
  const Real ke = 0.5*(m1*m1 + m2*m2 + m3*m3)/rho;
  const Real me = 0.5*(b1*b1 + b2*b2 + b3*b3);
  const Real eele0 = 0.7, eion0 = 1.3;           // seeded species energy densities
  const Real eint0 = eele0 + eion0;
  const Real etot_hyd = ke + eint0;              // hydro total energy density
  const Real etot_mhd = ke + me + eint0;         // MHD total energy density

  // (B) PdV inputs
  const Real pele = 0.8, pion = 1.2;             // f_ele = 0.4
  const Real divv_c = -0.5;                      // compression
  const Real divv_e =  0.5;                      // expansion
  const Real dt = 0.1;

  // (C) shock heat added to the gas internal energy (lands in E_tot via the conservative
  //     update); the reversible electron PdV is the only electron source.
  const Real ushock = 0.5;

  // (D) 1T-limit (T_e = T_i) ideal-gas pressures, shared gamma -> p propto number density
  const Real gamma = GE;                         // shared species gamma
  const Real pele_1t = 1.0, pion_1t = 0.5;       // f_1t = 2/3
  const Real divv_1t = -0.3;

  // (E) EOS-fed query point (off-grid)
  const Real rho_e = 0.5, te_true = 300.0, ti_true = 300.0;

  enum {
    // (A)
    IEION_HYD=0, ICONS_HYD, IEION_MHD, ICONS_MHD,
    // (B)
    IFRAC_B, IDEELE_B, ITOTPDV_B, IDEION_B, IDEELE_EXPAND,
    // (C)
    IDEELE_C, IDEION_C, ISHOCK_ELE, ISHOCK_ION, ICONS_C,
    // (D)
    IEINT_1T, IPREC_1T, IFRAC_1T, IFRACELE_PDV, IFRACION_PDV,
    // (E)
    IPELE_E, IPION_E, IFRAC_E, IDEELE_E,
    NVAL
  };
  DvceArray1D<Real> d_vals("three_temp_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("three_temp_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // ===================== (A) e_ion by subtraction + conservation ====================
    Real eion_hyd = three_temp::IonInternalEnergyHyd(etot_hyd, rho, m1, m2, m3, eele0);
    d_vals(IEION_HYD) = eion_hyd;
    d_vals(ICONS_HYD) = (eele0 + eion_hyd + ke) - etot_hyd;       // -> 0
    Real eion_mhd = three_temp::IonInternalEnergyMHD(etot_mhd, rho, m1, m2, m3,
                                                     b1, b2, b3, eele0);
    d_vals(IEION_MHD) = eion_mhd;
    d_vals(ICONS_MHD) = (eele0 + eion_mhd + ke + me) - etot_mhd;  // -> 0

    // ========================= (B) PdV split by pressure fraction =====================
    Real f = three_temp::ElectronPressureFraction(pele, pion);
    d_vals(IFRAC_B) = f;
    Real eele1 = three_temp::ApplyElectronPdV(eele0, pele, divv_c, dt);
    Real deele = eele1 - eele0;                            // electron PdV increment
    Real totpdv = -(pele + pion)*divv_c*dt;                       // total reversible work
    d_vals(IDEELE_B)  = deele;
    d_vals(ITOTPDV_B) = totpdv;
    d_vals(IDEION_B)  = totpdv - deele;                           // ion PdV share
    Real eele_exp = three_temp::ApplyElectronPdV(eele0, pele, divv_e, dt);
    d_vals(IDEELE_EXPAND) = eele_exp - eele0;                     // -> negative (cooling)

    // ===================== (C) shock dissipation lands on ions ========================
    // After a step: gas internal energy rose by reversible PdV + irreversible shock heat.
    Real eint1   = eint0 + totpdv + ushock;
    Real etot1   = ke + me + eint1;
    Real eele1_c = three_temp::ApplyElectronPdV(eele0, pele, divv_c, dt);  // PdV only
    Real eion1_c = three_temp::IonInternalEnergyMHD(etot1, rho, m1, m2, m3,
                                                    b1, b2, b3, eele1_c);
    Real deele_c = eele1_c - eele0;
    Real deion_c = eion1_c - eion0;
    d_vals(IDEELE_C) = deele_c;
    d_vals(IDEION_C) = deion_c;
    // reversible-only shares
    Real ele_rev = -pele*divv_c*dt;
    Real ion_rev = -pion*divv_c*dt;
    d_vals(ISHOCK_ELE) = deele_c - ele_rev;                       // -> 0 (no shock on e)
    d_vals(ISHOCK_ION) = deion_c - ion_rev;                // -> ushock (all on ions)
    d_vals(ICONS_C)    = (eele1_c + eion1_c + ke + me) - etot1;   // -> 0 (conserved)

    // ============================ (D) 1T limit (T_e = T_i) ============================
    Real eele_1t = pele_1t/(gamma - 1.0);
    Real eion_1t = pion_1t/(gamma - 1.0);
    Real eint_1t_3t = eele_1t + eion_1t;                          // 3T partition sum
    d_vals(IEINT_1T) = eint_1t_3t;                                // == (pele+pion)/(g-1)
    d_vals(IPREC_1T) = (gamma - 1.0)*eint_1t_3t;           // recovered total pressure
    d_vals(IFRAC_1T) = three_temp::ElectronPressureFraction(pele_1t, pion_1t);
    // PdV at equal temperature: species energy fractions are preserved (stays 1T).
    Real eele_1t_new = three_temp::ApplyElectronPdV(eele_1t, pele_1t, divv_1t, dt);
    Real etot_1t_new = eint_1t_3t + (-(pele_1t + pion_1t)*divv_1t*dt);  // v=0 -> KE=0
    Real eion_1t_new = three_temp::IonInternalEnergyHyd(etot_1t_new, rho, 0.0, 0.0, 0.0,
                                                        eele_1t_new);
    Real eint_1t_new = eele_1t_new + eion_1t_new;
    d_vals(IFRACELE_PDV) = eele_1t_new/eint_1t_new;               // -> f_1t (preserved)
    d_vals(IFRACION_PDV) = eion_1t_new/eint_1t_new;               // -> 1 - f_1t

    // ===================== (E) EOS-fed pressures (e -> T -> p -> f) ====================
    Real eele_spec = table.EnergyEle(rho_e, te_true);      // specific electron energy
    Real eion_spec = table.EnergyIon(rho_e, ti_true);
    Real te = table.Te(rho_e, eele_spec);                         // invert back
    Real ti = table.Ti(rho_e, eion_spec);
    Real pele_eos = table.PressureEle(rho_e, te);
    Real pion_eos = table.PressureIon(rho_e, ti);
    d_vals(IPELE_E) = pele_eos;
    d_vals(IPION_E) = pion_eos;
    d_vals(IFRAC_E) = three_temp::ElectronPressureFraction(pele_eos, pion_eos);
    Real eele_e1 = three_temp::ApplyElectronPdV(eele0, pele_eos, divv_c, dt);
    d_vals(IDEELE_E) = eele_e1 - eele0;                           // == -pele_eos*divv*dt
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real tol = 1.0e-13;

  // ----------------------------- (A) checks -----------------------------
  test.CheckNear(h_vals(IEION_HYD), eion0, 0.0, tol,
                 "hydro e_ion = E_tot - KE - e_ele recovers seeded e_ion");
  test.CheckNear(h_vals(ICONS_HYD), 0.0, 0.0, tol,
                 "hydro total energy conserved: e_ele + e_ion + KE == E_tot");
  test.CheckNear(h_vals(IEION_MHD), eion0, 0.0, tol,
                 "MHD e_ion = E_tot - KE - 1/2 B^2 - e_ele recovers seeded e_ion");
  test.CheckNear(h_vals(ICONS_MHD), 0.0, 0.0, tol,
                 "MHD total energy conserved: e_ele + e_ion + KE + 1/2 B^2 == E_tot");

  // ----------------------------- (B) checks -----------------------------
  test.CheckNear(h_vals(IFRAC_B), pele/(pele + pion), 0.0, tol,
                 "electron pressure fraction f_ele = p_ele/(p_ele+p_ion)");
  test.CheckNear(h_vals(IDEELE_B), (pele/(pele + pion))*h_vals(ITOTPDV_B), 0.0, tol,
                 "PdV split: electron increment = f_ele * total reversible work");
  test.CheckNear(h_vals(IDEELE_B), -pele*divv_c*dt, 0.0, tol,
                 "electron PdV increment = -p_ele*div(v)*dt");
  test.CheckNear(h_vals(IDEION_B), -pion*divv_c*dt, 0.0, tol,
                 "ion PdV share = -p_ion*div(v)*dt (complementary 1-f_ele)");
  test.CheckTrue(h_vals(IDEELE_B) > 0.0,
                 "compression (div v < 0) heats electrons");
  test.CheckTrue(h_vals(IDEELE_EXPAND) < 0.0,
                 "expansion (div v > 0) cools electrons");

  // ----------------------------- (C) checks -----------------------------
  test.CheckNear(h_vals(ISHOCK_ELE), 0.0, 0.0, tol,
                 "shock dissipation puts NO irreversible heat on electrons");
  test.CheckNear(h_vals(ISHOCK_ION), ushock, 0.0, tol,
                 "shock dissipation lands entirely on ions");
  test.CheckNear(h_vals(IDEELE_C), -pele*divv_c*dt, 0.0, tol,
                 "electron energy gains only the reversible PdV (no shock heat)");
  test.CheckNear(h_vals(IDEION_C), -pion*divv_c*dt + ushock, 0.0, tol,
                 "ion energy gains reversible PdV + all shock dissipation");
  test.CheckNear(h_vals(ICONS_C), 0.0, 0.0, tol,
                 "total energy still conserved after shock + PdV step");

  // ----------------------------- (D) checks -----------------------------
  test.CheckNear(h_vals(IEINT_1T), (pele_1t + pion_1t)/(gamma - 1.0), 0.0, tol,
                 "1T limit: e_ele + e_ion == single-temperature internal energy");
  test.CheckNear(h_vals(IPREC_1T), pele_1t + pion_1t, 0.0, tol,
                 "1T limit: recovered p_gas == single-temperature ideal-gas pressure");
  test.CheckNear(h_vals(IFRAC_1T), pele_1t/(pele_1t + pion_1t), 0.0, tol,
                 "1T limit: pressure fraction = number fraction (T_e = T_i)");
  test.CheckNear(h_vals(IFRACELE_PDV), pele_1t/(pele_1t + pion_1t), 0.0, tol,
                 "1T limit: electron energy fraction preserved under PdV (stays 1T)");
  test.CheckNear(h_vals(IFRACION_PDV), pion_1t/(pele_1t + pion_1t), 0.0, tol,
                 "1T limit: ion energy fraction preserved under PdV (stays 1T)");

  // ----------------------------- (E) checks -----------------------------
  // round-trip e->T->p reproduces the synthetic ideal-gas closed form p = (g-1) rho e.
  const Real rt = 1.0e-9;  // log10/pow round-off-limited tolerance
  Real pele_cf = (GE - 1.0)*rho_e*EAnalytic(eos_table_3t::kElectron, rho_e, te_true);
  Real pion_cf = (GI - 1.0)*rho_e*EAnalytic(eos_table_3t::kIon, rho_e, ti_true);
  test.CheckNear(h_vals(IPELE_E), pele_cf, rt, 0.0,
                 "EOS-fed p_ele (e_ele -> T_e -> p_ele) matches ideal-gas closed form");
  test.CheckNear(h_vals(IPION_E), pion_cf, rt, 0.0,
                 "EOS-fed p_ion (e_ion -> T_i -> p_ion) matches ideal-gas closed form");
  test.CheckNear(h_vals(IFRAC_E), pele_cf/(pele_cf + pion_cf), rt, 0.0,
                 "EOS-fed electron pressure fraction matches closed form (ties #6/#26)");
  test.CheckNear(h_vals(IDEELE_E), -h_vals(IPELE_E)*divv_c*dt, 0.0, tol,
                 "EOS-fed PdV increment = -p_ele(EOS)*div(v)*dt");

  test.Finish();
  return;
}
