//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_table_3t_test.cpp
//  \brief Unit test: the common tabulated 3T EOS representation (eos/eos_table_3t.hpp)
//  evaluated on the device reproduces a synthetic ideal-gas table's analytic relations,
//  round-trips T->e->T, clamps out-of-range queries to the table edges, and inverts
//  e->T monotonically and convergently over a representative (rho,e) sweep (issue
//  [13a]/#6, ADR-0002/0007).
//
//  The synthetic table is an ideal gas per species:
//      e_s(rho,T)  = Cv_s * (rho/rho_ref)^a_s * T     (specific internal energy)
//      cv_s(rho,T) = Cv_s * (rho/rho_ref)^a_s         (= de_s/dT)
//      p_s(rho,T)  = (gamma_s-1) * rho * e_s
//      Zbar        = Z0                               (constant)
//  In log10(value) vs (log10 rho, log10 T) space every field above is affine, so the
//  log-log bilinear interpolation reproduces it *exactly* (on and off grid) -- which is
//  the ideal-gas-limit agreement we assert, and makes the e->T inversion recover the
//  closed-form temperature to the Newton tolerance.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/eos_table_3t_test -B build_unit
//    (cd build_unit/src && make) && ./build_unit/src/athena \
//        -i inputs/unit_tests/eos_table_3t_test.athinput
//  Auto-run by tst/test_suite/unit_tests/test_unit_eos_table_3t_cpu.py.

#include <cmath>     // std::pow
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// Synthetic ideal-gas table parameters (unit-agnostic synthetic values).
constexpr Real CVE = 1.5,  AE = 0.25, GE = 5.0/3.0;   // electron Cv, rho-exponent, gamma
constexpr Real CVI = 0.9,  AI = -0.1, GI = 5.0/3.0;   // ion
constexpr Real Z0  = 3.5;                             // constant mean ionization
constexpr Real RHO_REF = 1.0;                         // reference density
constexpr int  NT = 41, NR = 31;                      // non-square (catches axis swap)
constexpr Real TMIN = 1.0,    TMAX = 1.0e4;           // temperature range
constexpr Real RMIN = 1.0e-3, RMAX = 1.0e1;           // density range

// Closed-form ideal-gas specific energy / heat capacity per species (host helper).
Real EAnalytic(int s, Real rho, Real temp) {
  Real cv = (s == eos_table_3t::kElectron) ? CVE : CVI;
  Real a  = (s == eos_table_3t::kElectron) ? AE  : AI;
  return cv*std::pow(rho/RHO_REF, a)*temp;
}
Real CvAnalytic(int s, Real rho) {
  Real cv = (s == eos_table_3t::kElectron) ? CVE : CVI;
  Real a  = (s == eos_table_3t::kElectron) ? AE  : AI;
  return cv*std::pow(rho/RHO_REF, a);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Tabulated 3T EOS representation + e->T inversion unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("eos_table_3t_test");

  // --- build the synthetic ideal-gas table on the host, copy to the device ---
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
      h_cv_ele(it, ir) = CvAnalytic(eos_table_3t::kElectron, rho);
      h_cv_ion(it, ir) = CvAnalytic(eos_table_3t::kIon, rho);
      h_p_ele(it, ir)  = (GE - 1.0)*rho*h_e_ele(it, ir);
      h_p_ion(it, ir)  = (GI - 1.0)*rho*h_e_ion(it, ir);
      h_zbar(it, ir)   = Z0;
    }
  }
  Kokkos::deep_copy(table.e_ele, h_e_ele);
  Kokkos::deep_copy(table.e_ion, h_e_ion);
  Kokkos::deep_copy(table.p_ele, h_p_ele);
  Kokkos::deep_copy(table.p_ion, h_p_ion);
  Kokkos::deep_copy(table.zbar, h_zbar);
  Kokkos::deep_copy(table.cv_ele, h_cv_ele);
  Kokkos::deep_copy(table.cv_ion, h_cv_ion);

  // Off-grid query point (neither rho nor T lands on a node), and one on-grid node.
  const Real rho_q = 0.3, T_q = 250.0;
  const Real T_node = table.TempAt(20);   // = 100  (exact node)
  const Real rho_node = table.RhoAt(15);  // = 0.1  (exact node)
  const int NSWEEP = 7;                   // per-axis points in the convergence sweep

  enum {
    IE_E=0, IE_I, IP_E, IP_I, IZ, ICV_E, ICV_I,
    IROUND_TE, IROUND_TI,
    ICLAMP_ELO, ICLAMP_EHI, ICLAMP_RHO_LO, ICLAMP_T_LO, ICLAMP_T_HI,
    IONGRID_E,
    ISWEEP_OK, ISWEEP_MAXERR,
    ITMIN, ITMAX, IRHOMIN,
    NVAL
  };
  DvceArray1D<Real> d_vals("eos3t_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("eos3t_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // Forward lookups at the off-grid point.
    d_vals(IE_E)  = table.EnergyEle(rho_q, T_q);
    d_vals(IE_I)  = table.EnergyIon(rho_q, T_q);
    d_vals(IP_E)  = table.PressureEle(rho_q, T_q);
    d_vals(IP_I)  = table.PressureIon(rho_q, T_q);
    d_vals(IZ)    = table.Zbar(rho_q, T_q);
    d_vals(ICV_E) = table.CvEle(rho_q, T_q);
    d_vals(ICV_I) = table.CvIon(rho_q, T_q);

    // Round-trip T -> e -> T for both species.
    d_vals(IROUND_TE) = table.Te(rho_q, table.EnergyEle(rho_q, T_q));
    d_vals(IROUND_TI) = table.Ti(rho_q, table.EnergyIon(rho_q, T_q));

    // Table-edge clamping: out-of-range energies map to the edge temperatures; out-of-
    // range axis queries return the edge-node (in-bounds) value (no OOB read).
    d_vals(ICLAMP_ELO)    = table.Te(rho_q, 1.0e-30);
    d_vals(ICLAMP_EHI)    = table.Te(rho_q, 1.0e+30);
    d_vals(ICLAMP_RHO_LO) = table.EnergyEle(1.0e-8, T_q);          // rho clamps to RhoMin
    d_vals(ICLAMP_T_LO)   = table.EnergyEle(rho_q, 1.0e-6);        // T clamps to TempMin
    d_vals(ICLAMP_T_HI)   = table.EnergyEle(rho_q, 1.0e+9);        // T clamps to TempMax

    // On-grid node value (exact up to log10/pow round-off).
    d_vals(IONGRID_E) = table.EnergyEle(rho_node, T_node);

    // Edge temperatures/density the clamps should resolve to.
    d_vals(ITMIN)   = table.TempMin();
    d_vals(ITMAX)   = table.TempMax();
    d_vals(IRHOMIN) = table.RhoMin();

    // Convergence sweep: invert e->T over a representative interior (rho,T) grid for
    // both species; all must converge to the generating temperature.
    int n_ok = 0;
    Real max_err = 0.0;
    for (int a = 0; a < NSWEEP; ++a) {
      Real lT = table.log_temp_min + ((a + 0.5)/NSWEEP)*(NT - 1)*table.dlog_temp;
      Real t_true = Kokkos::pow(10.0, lT);
      for (int b = 0; b < NSWEEP; ++b) {
        Real lR = table.log_rho_min + ((b + 0.5)/NSWEEP)*(NR - 1)*table.dlog_rho;
        Real rho = Kokkos::pow(10.0, lR);
        Real t_rec_e = table.Te(rho, table.EnergyEle(rho, t_true));
        Real err_e = Kokkos::fabs(t_rec_e - t_true)/t_true;
        if (err_e < 1.0e-9) { ++n_ok; }
        max_err = Kokkos::fmax(max_err, err_e);
        Real t_rec_i = table.Ti(rho, table.EnergyIon(rho, t_true));
        Real err_i = Kokkos::fabs(t_rec_i - t_true)/t_true;
        if (err_i < 1.0e-9) { ++n_ok; }
        max_err = Kokkos::fmax(max_err, err_i);
      }
    }
    d_vals(ISWEEP_OK)     = static_cast<Real>(n_ok);
    d_vals(ISWEEP_MAXERR) = max_err;
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real rt = 1.0e-9;  // log10/pow round-off-limited relative tolerance

  // (1) Ideal-gas-limit agreement: forward lookups equal the closed-form relations,
  //     interpolated off-grid.
  test.CheckNear(h_vals(IE_E), EAnalytic(eos_table_3t::kElectron, rho_q, T_q), rt, 0.0,
                 "EnergyEle == Cv_e (rho/rho_ref)^a_e T");
  test.CheckNear(h_vals(IE_I), EAnalytic(eos_table_3t::kIon, rho_q, T_q), rt, 0.0,
                 "EnergyIon == Cv_i (rho/rho_ref)^a_i T");
  test.CheckNear(h_vals(IP_E), (GE - 1.0)*rho_q*EAnalytic(eos_table_3t::kElectron,
                 rho_q, T_q), rt, 0.0, "PressureEle == (gamma_e-1) rho e_e");
  test.CheckNear(h_vals(IP_I), (GI - 1.0)*rho_q*EAnalytic(eos_table_3t::kIon,
                 rho_q, T_q), rt, 0.0, "PressureIon == (gamma_i-1) rho e_i");
  test.CheckNear(h_vals(IZ), Z0, rt, 0.0, "Zbar == Z0 (constant, raw bilinear)");
  test.CheckNear(h_vals(ICV_E), CvAnalytic(eos_table_3t::kElectron, rho_q), rt, 0.0,
                 "CvEle == Cv_e (rho/rho_ref)^a_e");
  test.CheckNear(h_vals(ICV_I), CvAnalytic(eos_table_3t::kIon, rho_q), rt, 0.0,
                 "CvIon == Cv_i (rho/rho_ref)^a_i");
  // Ideal-gas identity e = c_v T.
  test.CheckNear(h_vals(IE_E)/T_q, h_vals(ICV_E), rt, 0.0, "e_e/T == c_v,e (ideal gas)");

  // (2) Round-trip T -> e -> T recovers the original temperature.
  test.CheckNear(h_vals(IROUND_TE), T_q, rt, 0.0, "round-trip T_e -> e_e -> T_e");
  test.CheckNear(h_vals(IROUND_TI), T_q, rt, 0.0, "round-trip T_i -> e_i -> T_i");

  // (3) Table-edge clamping behaves (no out-of-range lookups).
  test.CheckNear(h_vals(ICLAMP_ELO), h_vals(ITMIN), rt, 0.0,
                 "e below table -> T clamps to TempMin");
  test.CheckNear(h_vals(ICLAMP_EHI), h_vals(ITMAX), rt, 0.0,
                 "e above table -> T clamps to TempMax");
  test.CheckNear(h_vals(ICLAMP_RHO_LO),
                 EAnalytic(eos_table_3t::kElectron, h_vals(IRHOMIN), T_q), rt, 0.0,
                 "rho below table -> lookup uses RhoMin edge");
  test.CheckNear(h_vals(ICLAMP_T_LO),
                 EAnalytic(eos_table_3t::kElectron, rho_q, h_vals(ITMIN)), rt, 0.0,
                 "T below table -> lookup uses TempMin edge");
  test.CheckNear(h_vals(ICLAMP_T_HI),
                 EAnalytic(eos_table_3t::kElectron, rho_q, h_vals(ITMAX)), rt, 0.0,
                 "T above table -> lookup uses TempMax edge");

  // (4) On-grid node value is exact, and the inversion converges over the whole sweep.
  test.CheckNear(h_vals(IONGRID_E),
                 EAnalytic(eos_table_3t::kElectron, rho_node, T_node), rt, 0.0,
                 "on-grid node EnergyEle is exact");
  test.CheckNear(h_vals(ISWEEP_OK), static_cast<Real>(2*NSWEEP*NSWEEP), 0.0, 0.0,
                 "every (rho,e) in the sweep inverts to the generating T");
  test.CheckTrue(h_vals(ISWEEP_MAXERR) < 1.0e-9,
                 "max relative inversion error over the sweep < 1e-9");
  // Sanity: edge temperatures equal the configured table bounds.
  test.CheckNear(h_vals(ITMIN), TMIN, rt, 0.0, "TempMin == configured TMIN");
  test.CheckNear(h_vals(ITMAX), TMAX, rt, 0.0, "TempMax == configured TMAX");

  test.Finish();
  return;
}
