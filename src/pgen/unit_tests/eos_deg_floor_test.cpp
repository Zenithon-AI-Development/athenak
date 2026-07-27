//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_deg_floor_test.cpp
//  \brief Unit test: the zero-temperature Fermi degeneracy-pressure floor on the
//  tabulated 3T electron pressure (issue #209).  The IONMIX table is ideal-ion nkT with
//  a nearly-neutral Zbar below ~1 eV AND its density axis ends at 44.8 g/cc (edge
//  clamping => dP/drho = 0 beyond) -- so a magnetically driven cold shell is pressureless
//  dust and its peak density diverges with resolution (4/23/164 code at dr
//  12.5/8.3/4.2 um).  The floor
//      P_ele >= K * max(0, rho^(5/3) - rho0^(5/3)),
//      K = (2/5)(hbar^2/2 m_e)(3 pi^2)^(2/3) (Z* rho_u/m_ion)^(5/3) / (rho_u v_u^2)
//  restores a physical stiffness that keeps rising through and beyond the table edge
//  (Z* = cold/valence ionization; the table's own Zbar ~ 0.004 at 1 eV would neuter it),
//  zeroed at the solid reference rho0 so the quiescent pre-drive liner feels nothing.
//
//  Uses the REAL committed aluminum table (inputs/ionmix/al-imx-004.cn4) at the B1
//  deck's <units> and ion mass, at the ACTUAL observed collapse states of issue #209.
//
//  Batteries:
//   (A) OFF BY DEFAULT: a freshly loaded table has a zero floor everywhere and the
//       ConsToPrim2T closure reproduces the bare table pressures (byte-identical path).
//   (B) ORACLE: after SetDegeneracyFloor with the B1 parameters (Z*=3, Al ion mass,
//       B1 <units>, rho0=1 code = solid), the floor matches an independently computed
//       CGS oracle at the observed collapse densities 23/46/164 code, and is EXACTLY
//       zero at and below solid density.
//   (C) CLOSURE FLOORED: the cold-dense closure pressure is floor-dominated (>= 50x the
//       bare-table value at rho=23 code, 1 eV) -- the state that collapses in #209.
//   (D) OFF-TABLE STIFFNESS: dP/drho > 0 beyond the 44.8 g/cc table edge -- the
//       floor(46)/floor(23) ratio is the rho^(5/3) value, NOT the edge-clamped 1.
//   (E) HOT STATE UNCHANGED: where the table pressure exceeds the floor (5.4 g/cc at
//       1 keV) the closure returns the bare-table pressure exactly.
//   (F) INVERSION UNTOUCHED: Te(rho, e_ele) is identical with and without the floor
//       (the floor touches pressure only, never the energy tables).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/eos_deg_floor_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_eos_deg_floor_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "eos/cons_to_prim_2t.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Degeneracy-pressure floor unit test (#209).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("eos_deg_floor_test");

  // --- the real aluminum IONMIX table at the B1 deck's units/ion mass ---
  const std::string tbl_file = pin->GetString("problem", "eos_table");
  const Real mion  = pin->GetOrAddReal("problem", "eos_mass_per_ion", 4.4804e-23);
  const Real u_rho = pin->GetOrAddReal("units", "density_cgs", 2.7);
  const Real u_vel = pin->GetOrAddReal("units", "velocity_cgs", 1.0e8);

  eos_table_3t::EosTable3T eos;
  eos_table_3t::ReadIonmixCn4Eos(tbl_file, eos, mion);
  eos.ScaleToCodeUnits(u_rho, u_vel);          // exactly what mhd.cpp does at load

  // B1 floor parameters + the #209 observed collapse states (code units).
  const Real zstar = 3.0;                       // Al valence (cold) ionization
  const Real rho0  = 1.0;                       // solid Al reference (floor zero here)
  const Real rho_states[3] = {23.0, 46.0, 164.0};
  // Independent CGS oracle: K and P_floor = K*(rho^(5/3)-1) precomputed outside the
  // implementation (python, CODATA hbar/m_e), B1 units (rho_u=2.7, v_u=1e8), Z*=3,
  // m_ion=4.4804e-23 g.
  const Real k_oracle = 5.002924413768e-05;
  const Real p_oracle[3] = {9.256126893381e-03, 2.949517484082e-02, 2.457774086791e-01};

  // Device evaluation buffer: floors + closure pressures before/after enabling.
  enum {
    IFLR_OFF = 0,                    // floor(23) on the fresh table (must be 0)
    IPG_OFF,                         // cold-dense p_gas, fresh table
    ITE_OFF,                         // Te(23, e_cold), fresh table
    IFLR_HALF, IFLR_SOLID,           // floor(0.5), floor(1.0) after enabling (== 0)
    IFLR_23, IFLR_46, IFLR_164,      // floor at the collapse states
    IPG_ON,                          // cold-dense p_gas, floored closure
    IPI_23,                          // bare-table ion pressure at (23, ~1 eV)
    IPG_HOT_ON,                      // hot p_gas with floor enabled
    IPG_HOT_TBL,                     // hot p_gas assembled from bare-table lookups
    ITE_ON,                          // Te(23, e_cold) with floor enabled
    NVAL
  };
  DvceArray1D<Real> d_vals("degflr_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  // Cold-dense electron/ion energies: the table's own energy at ~its 1 eV floor, so the
  // closure inverts back to the clamped table minimum (the #209 collapse state).
  const Real rho_cold = rho_states[0];
  const Real rho_hot = 2.0;          // 5.4 g/cc: ON-table, hot
  const Real te_hot = 1000.0;        // 1 keV: table pressure >> floor there

  // (A) fresh table: floor identically zero, closure is the bare table.
  par_for("degflr_off", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real e_ele = rho_cold*eos.EnergyEle(rho_cold, 1.0);
    Real e_ion = rho_cold*eos.EnergyIon(rho_cold, 1.0);
    d_vals(IFLR_OFF) = eos.DegeneracyPressureFloor(rho_cold);
    eos_table_3t::GasState2T s = eos_table_3t::ConsToPrim2T(eos, rho_cold, e_ele, e_ion);
    d_vals(IPG_OFF) = s.p_gas;
    d_vals(ITE_OFF) = s.te;
  });

  // Enable the floor with the B1 parameters (host-side setup, like table loading).
  eos.SetDegeneracyFloor(zstar, mion, u_rho, u_vel, rho0);

  par_for("degflr_on", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    d_vals(IFLR_HALF)  = eos.DegeneracyPressureFloor(0.5);
    d_vals(IFLR_SOLID) = eos.DegeneracyPressureFloor(1.0);
    d_vals(IFLR_23)  = eos.DegeneracyPressureFloor(rho_states[0]);
    d_vals(IFLR_46)  = eos.DegeneracyPressureFloor(rho_states[1]);
    d_vals(IFLR_164) = eos.DegeneracyPressureFloor(rho_states[2]);

    Real e_ele = rho_cold*eos.EnergyEle(rho_cold, 1.0);
    Real e_ion = rho_cold*eos.EnergyIon(rho_cold, 1.0);
    eos_table_3t::GasState2T s = eos_table_3t::ConsToPrim2T(eos, rho_cold, e_ele, e_ion);
    d_vals(IPG_ON) = s.p_gas;
    d_vals(ITE_ON) = s.te;
    d_vals(IPI_23) = eos.PressureIon(rho_cold, s.ti);

    Real eh_ele = rho_hot*eos.EnergyEle(rho_hot, te_hot);
    Real eh_ion = rho_hot*eos.EnergyIon(rho_hot, te_hot);
    eos_table_3t::GasState2T sh = eos_table_3t::ConsToPrim2T(eos, rho_hot, eh_ele,
                                                             eh_ion);
    d_vals(IPG_HOT_ON) = sh.p_gas;
    d_vals(IPG_HOT_TBL) = eos.PressureEle(rho_hot, sh.te)
                        + eos.PressureIon(rho_hot, sh.ti);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // (A) off by default
  test.CheckTrue(h_vals(IFLR_OFF) == 0.0,
                 "fresh table: DegeneracyPressureFloor == 0 (floor is opt-in)");
  test.CheckTrue(h_vals(IPG_OFF) < 1.0e-3,
                 "fresh table: cold-dense closure stays at the tiny bare-table pressure");

  // (B) oracle
  test.CheckTrue(h_vals(IFLR_HALF) == 0.0, "floor(rho < rho0) == 0 exactly");
  test.CheckTrue(h_vals(IFLR_SOLID) == 0.0, "floor(rho0) == 0 exactly (quiescent solid)");
  test.CheckNear(h_vals(IFLR_23), p_oracle[0], 1.0e-6, 0.0,
                 "floor(23 code = 62 g/cc) == independent CGS oracle (~250 Mbar)");
  test.CheckNear(h_vals(IFLR_46), p_oracle[1], 1.0e-6, 0.0,
                 "floor(46 code) == independent CGS oracle (~796 Mbar)");
  test.CheckNear(h_vals(IFLR_164), p_oracle[2], 1.0e-6, 0.0,
                 "floor(164 code = 443 g/cc) == independent CGS oracle (~6.6 Gbar)");

  // (C) the closure is floor-dominated at the collapse state
  test.CheckTrue(h_vals(IPG_ON) >= h_vals(IFLR_23),
                 "cold-dense closure p_gas >= the electron floor");
  test.CheckNear(h_vals(IPG_ON), h_vals(IFLR_23) + h_vals(IPI_23), 1.0e-12, 0.0,
                 "cold-dense p_gas == floored p_ele + bare-table p_ion");
  test.CheckTrue(h_vals(IPG_ON) > 50.0*h_vals(IPG_OFF),
                 "the floor dominates the bare-table pressure at the #209 state (>50x)");

  // (D) off-table stiffness restored: rho^(5/3), not edge-clamped flat
  test.CheckNear(h_vals(IFLR_46)/h_vals(IFLR_23), 3.186557, 1.0e-4, 0.0,
                 "floor(46)/floor(23) == (46^(5/3)-1)/(23^(5/3)-1): stiff off-table");

  // (E) hot state unchanged
  test.CheckNear(h_vals(IPG_HOT_ON), h_vals(IPG_HOT_TBL), 1.0e-14, 0.0,
                 "hot on-table state: closure == bare table (floor inactive there)");

  // (F) the inversion is untouched
  test.CheckNear(h_vals(ITE_ON), h_vals(ITE_OFF), 1.0e-14, 0.0,
                 "Te inversion identical with/without the floor (pressure-only floor)");

  test.Finish();
  std::exit(0);
}
