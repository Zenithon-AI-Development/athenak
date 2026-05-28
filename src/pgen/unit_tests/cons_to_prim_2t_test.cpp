//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cons_to_prim_2t_test.cpp
//  \brief Unit test: tabulated 2T EOS conserved->primitive closure (issue [14b]/#26,
//  ADR-0002).
//
//  Exercises the per-cell closure eos_table_3t::ConsToPrim2T (eos/cons_to_prim_2t.hpp)
//  against analytic oracles on real device kernels.  Given (rho, e_ele, e_ion) -- the
//  density and the electron/ion internal energy *densities* (the conserved inputs of the
//  3T formulation, #19) -- it inverts each species energy to its temperature via the
//  tabulated EOS (#6) and returns the derived/cached temperatures T_e, T_i plus
//  p_gas = p_ele(rho,T_e) + p_ion(rho,T_i).
//
//  Batteries:
//   (A) Two-temperature recovery (T_e != T_i): seed species energies from KNOWN
//       (rho, T_e, T_i) via the table's forward lookups, run ConsToPrim2T, and verify the
//       recovered temperatures and species/total pressures match the table (the synthetic
//       per-species ideal-gas closed form p = (gamma-1) rho e).  This is "pressures /
//       temperatures match the table for known (rho, e_ele, e_ion)".
//   (B) Species independence (discriminator): with DIFFERENT species gammas, p_ele is set
//       by T_e (electron table) and p_ion by T_i (ion table); a closure that mixed up the
//       species or used a single temperature would miss the distinct closed forms.  Also
//       checks the structural identity p_gas == p_ele + p_ion.
//   (C) Out-of-range edge clamping: an electron energy below e(Tmin) clamps T_e to the
//       table-minimum temperature (table-edge clamp, #6) -- robust, no OOB read.
//   (D) Ideal-gas limit (shared gamma): with a single-gamma table and T_e = T_i the
//       closure recovers the exact single-temperature ideal-gas pressure
//       p_gas = (gamma-1) (e_ele + e_ion), and the two derived temperatures coincide.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/cons_to_prim_2t_test -B build_unit/cons_to_prim_2t
//    (cd build_unit/.../src && make) && ./athena -i <cons_to_prim_2t_test.athinput>
//  Auto-run by tst/test_suite/unit_tests/test_unit_cons_to_prim_2t_cpu.py.

#include <cmath>     // std::pow
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/cons_to_prim_2t.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// --- synthetic per-species ideal-gas EOS tables (power laws; p = (gamma-1) rho e) ---
// Battery A/B/C table: DIFFERENT species gammas so the closure is forced to use T_e for
// p_ele and T_i for p_ion (a single-T mix-up would fail the closed-form checks).
constexpr Real CVE = 1.5, AE =  0.25, GE = 5.0/3.0;   // electron Cv, rho-exponent, gamma
constexpr Real CVI = 0.9, AI = -0.10, GI = 7.0/5.0;   // ion (distinct gamma)
// Battery D table: a SINGLE shared gamma (true 1T ideal gas).
constexpr Real CVE1 = 1.2, AE1 =  0.10, G1 = 5.0/3.0; // electron, shared gamma
constexpr Real CVI1 = 0.7, AI1 = -0.20;               // ion (same gamma G1)
constexpr int  NT = 11, NR = 9;                       // non-square (catches axis swap)
constexpr Real TMIN = 1.0,    TMAX = 1.0e4;
constexpr Real RMIN = 1.0e-3, RMAX = 1.0e1;

// specific internal energy e(T,rho) = Cv * rho^a * T (ideal gas; exact under log-log)
Real ESpec(Real cv, Real a, Real rho, Real temp) {
  return cv*std::pow(rho, a)*temp;
}

// Fill a synthetic ideal-gas EosTable3T from per-species (Cv, a, gamma) power laws.
void FillIdealTable(eos_table_3t::EosTable3T *table,
                    Real cve, Real ae, Real ge, Real cvi, Real ai, Real gi) {
  table->Allocate(NT, NR, TMIN, TMAX, RMIN, RMAX);
  auto h_e_ele  = Kokkos::create_mirror_view(table->e_ele);
  auto h_e_ion  = Kokkos::create_mirror_view(table->e_ion);
  auto h_p_ele  = Kokkos::create_mirror_view(table->p_ele);
  auto h_p_ion  = Kokkos::create_mirror_view(table->p_ion);
  auto h_zbar   = Kokkos::create_mirror_view(table->zbar);
  auto h_cv_ele = Kokkos::create_mirror_view(table->cv_ele);
  auto h_cv_ion = Kokkos::create_mirror_view(table->cv_ion);
  for (int it = 0; it < NT; ++it) {
    Real temp = table->TempAt(it);
    for (int ir = 0; ir < NR; ++ir) {
      Real rho = table->RhoAt(ir);
      h_e_ele(it, ir)  = ESpec(cve, ae, rho, temp);
      h_e_ion(it, ir)  = ESpec(cvi, ai, rho, temp);
      h_cv_ele(it, ir) = cve*std::pow(rho, ae);
      h_cv_ion(it, ir) = cvi*std::pow(rho, ai);
      h_p_ele(it, ir)  = (ge - 1.0)*rho*h_e_ele(it, ir);
      h_p_ion(it, ir)  = (gi - 1.0)*rho*h_e_ion(it, ir);
      h_zbar(it, ir)   = 1.0;
    }
  }
  Kokkos::deep_copy(table->e_ele, h_e_ele);
  Kokkos::deep_copy(table->e_ion, h_e_ion);
  Kokkos::deep_copy(table->p_ele, h_p_ele);
  Kokkos::deep_copy(table->p_ion, h_p_ion);
  Kokkos::deep_copy(table->zbar, h_zbar);
  Kokkos::deep_copy(table->cv_ele, h_cv_ele);
  Kokkos::deep_copy(table->cv_ion, h_cv_ion);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Tabulated 2T EOS conserved->primitive closure unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("cons_to_prim_2t_test");

  // ---------- build the two synthetic ideal-gas EOS tables ----------
  eos_table_3t::EosTable3T table2t;   // distinct species gammas (Batteries A/B/C)
  FillIdealTable(&table2t, CVE, AE, GE, CVI, AI, GI);
  eos_table_3t::EosTable3T table1g;   // single shared gamma (Battery D)
  FillIdealTable(&table1g, CVE1, AE1, G1, CVI1, AI1, G1);

  // ---------- query points (off-grid) ----------
  const Real rho_a   = 0.5;
  const Real te_true = 300.0, ti_true = 120.0;   // distinct temperatures (genuine 2T)
  const Real rho_d   = 2.0;
  const Real t_eq    = 250.0;                     // shared temperature for the 1T limit

  enum {
    // (A)/(B)
    ITE, ITI, IPELE, IPION, IPGAS, IPSUM,
    // (C)
    ITE_LO, IPELE_LO,
    // (D)
    ITE_D, ITI_D, IPGAS_D, IPGAS_IDEAL, ITDIFF_D,
    NVAL
  };
  DvceArray1D<Real> d_vals("c2p2t_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("c2p2t_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // ===================== (A)/(B) two-temperature recovery ===========================
    // Seed conserved species energy DENSITIES from known (rho, T_e, T_i): the table holds
    // SPECIFIC energy, so the density input is rho * e_specific.
    Real e_ele_dens = rho_a*table2t.EnergyEle(rho_a, te_true);
    Real e_ion_dens = rho_a*table2t.EnergyIon(rho_a, ti_true);
    eos_table_3t::GasState2T s = eos_table_3t::ConsToPrim2T(table2t, rho_a,
                                                            e_ele_dens, e_ion_dens);
    d_vals(ITE)   = s.te;
    d_vals(ITI)   = s.ti;
    d_vals(IPELE) = s.p_ele;
    d_vals(IPION) = s.p_ion;
    d_vals(IPGAS) = s.p_gas;
    d_vals(IPSUM) = s.p_gas - (s.p_ele + s.p_ion);   // -> 0 (structural identity)

    // ===================== (C) out-of-range edge clamping =============================
    // An electron energy below e(Tmin) clamps T_e to the table-minimum temperature.
    Real e_ele_lo = 0.5*rho_a*table2t.EnergyEle(rho_a, table2t.TempMin());
    eos_table_3t::GasState2T s_lo = eos_table_3t::ConsToPrim2T(table2t, rho_a,
                                                               e_ele_lo, e_ion_dens);
    d_vals(ITE_LO)   = s_lo.te;       // -> TempMin
    d_vals(IPELE_LO) = s_lo.p_ele;    // -> (GE-1) rho e(Tmin)

    // ===================== (D) ideal-gas limit (shared gamma) =========================
    Real ee_d = rho_d*table1g.EnergyEle(rho_d, t_eq);
    Real ei_d = rho_d*table1g.EnergyIon(rho_d, t_eq);
    eos_table_3t::GasState2T s_d = eos_table_3t::ConsToPrim2T(table1g, rho_d, ee_d, ei_d);
    d_vals(ITE_D)       = s_d.te;
    d_vals(ITI_D)       = s_d.ti;
    d_vals(IPGAS_D)     = s_d.p_gas;
    d_vals(IPGAS_IDEAL) = (G1 - 1.0)*(ee_d + ei_d);  // single-T ideal-gas pressure
    d_vals(ITDIFF_D)    = s_d.te - s_d.ti;           // -> 0 (T_e == T_i)
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real rt = 1.0e-9;   // log10/pow round-off-limited tol for the EOS round-trip
  const Real tol = 1.0e-13; // structural-identity tolerance

  // closed-form oracles (host): p = (gamma-1) rho e_specific
  const Real eele_spec_a = ESpec(CVE, AE, rho_a, te_true);
  const Real eion_spec_a = ESpec(CVI, AI, rho_a, ti_true);
  const Real pele_cf = (GE - 1.0)*rho_a*eele_spec_a;
  const Real pion_cf = (GI - 1.0)*rho_a*eion_spec_a;

  // ----------------------------- (A) checks -----------------------------
  test.CheckNear(h_vals(ITE), te_true, rt, 0.0,
                 "T_e recovered from e_ele/rho matches the table temperature");
  test.CheckNear(h_vals(ITI), ti_true, rt, 0.0,
                 "T_i recovered from e_ion/rho matches the table temperature");
  test.CheckNear(h_vals(IPELE), pele_cf, rt, 0.0,
                 "p_ele(rho,T_e) matches the table ideal-gas closed form");
  test.CheckNear(h_vals(IPION), pion_cf, rt, 0.0,
                 "p_ion(rho,T_i) matches the table ideal-gas closed form");
  test.CheckNear(h_vals(IPGAS), pele_cf + pion_cf, rt, 0.0,
                 "p_gas = p_ele(rho,T_e) + p_ion(rho,T_i)");

  // ----------------------------- (B) checks -----------------------------
  test.CheckNear(h_vals(IPSUM), 0.0, 0.0, tol,
                 "structural identity p_gas == p_ele + p_ion (exact)");
  test.CheckTrue(h_vals(ITE) > h_vals(ITI),
                 "genuine 2T state: recovered T_e != T_i (T_e > T_i here)");
  // species independence: p_ele uses the ELECTRON table at T_e, not the ion energy/gamma.
  test.CheckTrue(!unit_test::ApproxEqual(h_vals(IPELE), (GI - 1.0)*rho_a*eion_spec_a,
                                         rt, 0.0),
                 "p_ele uses the electron species (not the ion energy) -- 2T, not mixed");

  // ----------------------------- (C) checks -----------------------------
  test.CheckNear(h_vals(ITE_LO), TMIN, rt, 0.0,
                 "T_e clamps to the table-minimum temperature for e_ele below e(Tmin)");
  test.CheckNear(h_vals(IPELE_LO), (GE - 1.0)*rho_a*ESpec(CVE, AE, rho_a, TMIN), rt, 0.0,
                 "p_ele at the clamped T_e equals the closed form at Tmin");

  // ----------------------------- (D) checks -----------------------------
  test.CheckNear(h_vals(ITE_D), t_eq, rt, 0.0,
                 "ideal-gas limit: T_e recovered at the shared temperature");
  test.CheckNear(h_vals(ITI_D), t_eq, rt, 0.0,
                 "ideal-gas limit: T_i recovered at the shared temperature");
  test.CheckNear(h_vals(ITDIFF_D), 0.0, 0.0, rt*t_eq,
                 "ideal-gas limit: T_e == T_i for a single-temperature state");
  test.CheckNear(h_vals(IPGAS_D), h_vals(IPGAS_IDEAL), rt, 0.0,
                 "ideal-gas limit: p_gas == (gamma-1)(e_ele + e_ion) recovered");

  test.Finish();
  return;
}
