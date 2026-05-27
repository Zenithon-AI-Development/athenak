//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_eos_test.cpp
//  \brief Unit test: the IONMIX 3T EOS reader + the common 3T EOS representation
//  (eos/ionmix_eos_reader.hpp + eos/eos_table_3t.hpp).  Reads a committed IONMIX-style
//  fixture table, then on the device checks that the parsed grid, the per-species
//  electron/ion specific energies / pressures / heat capacities / mean ionization, the
//  (rho,T) interpolation, the e->T inversion the representation feeds, and the table-edge
//  clamping all reproduce the fixture's known synthetic laws (issue [13b]/#10, ADR-0007).
//
//  The fixture (inputs/unit_tests/ionmix_eos_test.cn4) stores a synthetic 3T-native EOS
//  on a log-spaced (T, rho) grid (T [eV], rho [g/cc]; specific energies/heat capacities):
//      e_ele (rho,T) = 1.5*rho^0.10*T     cv_ele(rho,T) = 1.5*rho^0.10   (= de_e/dT)
//      e_ion (rho,T) = 0.9*rho^0.05*T     cv_ion(rho,T) = 0.9*rho^0.05   (= de_i/dT)
//      p_ele (rho,T) = 0.8*rho^1.0 *T     p_ion (rho,T) = 0.5*rho^1.0 *T
//      zbar  (rho,T) = 3.0 + 0.5*log10(rho) + 0.25*log10(T)
//  The e/p/cv laws are power laws (affine in log10) so the log-log bilinear interpolation
//  reproduces them exactly on AND off grid; zbar is affine in (log10 rho, log10 T) so the
//  raw bilinear interpolation reproduces it exactly too -- so we can assert the lookups
//  against the closed-form laws to round-off, and (since e is linear in T with slope cv)
//  the e->T inversion recovers the generating temperature to the Newton tolerance.
//
//  The fixture path is passed on the command line as `problem/eos_file=<abspath>` by the
//  Python wrapper (tst/test_suite/unit_tests/test_unit_ionmix_eos_cpu.py).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/ionmix_eos_test -B build_unit
//    (cd build_unit/src && make) && ./build_unit/src/athena \
//        -i inputs/unit_tests/ionmix_eos_test.athinput \
//        problem/eos_file=$PWD/inputs/unit_tests/ionmix_eos_test.cn4

#include <cmath>     // std::pow, std::log10
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// Synthetic 3T-native EOS laws used to generate the fixture (host helpers; T in eV,
// rho in g/cc; specific energies/heat capacities).
Real Eele(Real rho, Real T) { return 1.5*std::pow(rho, 0.10)*T; }
Real Eion(Real rho, Real T) { return 0.9*std::pow(rho, 0.05)*T; }
Real Cvele(Real rho)        { return 1.5*std::pow(rho, 0.10); }
Real Cvion(Real rho)        { return 0.9*std::pow(rho, 0.05); }
Real Pele(Real rho, Real T) { return 0.8*rho*T; }
Real Pion(Real rho, Real T) { return 0.5*rho*T; }
Real Zbar(Real rho, Real T) { return 3.0 + 0.5*std::log10(rho) + 0.25*std::log10(T); }
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief IONMIX 3T EOS reader + common-representation unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("ionmix_eos_test");

  // --- read the IONMIX fixture table from disk into the common representation ---
  std::string fname = pin->GetString("problem", "eos_file");
  eos_table_3t::EosTable3T table;
  eos_table_3t::ReadIonmixEos(fname, table);

  // (1) Grid parsed correctly (PODs are host-readable).
  test.CheckTrue(table.ntemp == 5, "parsed ntemp == 5");
  test.CheckTrue(table.nrho == 4, "parsed nrho == 4");

  // Off-grid query point (neither rho nor T lands on a node) and an on-grid node.
  const Real rho_og = 3.0e-2, T_og = 300.0;
  const Real T_node = 100.0, rho_node = 1.0e-2;  // exact nodes

  enum {
    IE_E=0, IE_I, IP_E, IP_I, IZ, ICV_E, ICV_I,    // off-grid forward lookups
    IE_E_NODE, IZ_NODE,                            // on-grid node values
    IROUND_TE, IROUND_TI,                          // e->T inversion round-trip
    IPGAS,                                         // p_gas = p_ele + p_ion (the #26 use)
    ICLAMP_RHO_LO, ICLAMP_T_HI, ICLAMP_ELO, ICLAMP_EHI,  // edge clamping
    ITMIN, ITMAX, IRHOMIN, IRHOMAX,                // resolved table edges
    NVAL
  };
  DvceArray1D<Real> d_vals("eos_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("eos_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // Off-grid forward lookups across all fields.
    d_vals(IE_E)  = table.EnergyEle(rho_og, T_og);
    d_vals(IE_I)  = table.EnergyIon(rho_og, T_og);
    d_vals(IP_E)  = table.PressureEle(rho_og, T_og);
    d_vals(IP_I)  = table.PressureIon(rho_og, T_og);
    d_vals(IZ)    = table.Zbar(rho_og, T_og);
    d_vals(ICV_E) = table.CvEle(rho_og, T_og);
    d_vals(ICV_I) = table.CvIon(rho_og, T_og);

    // On-grid node values (exact up to log10/pow round-off).
    d_vals(IE_E_NODE) = table.EnergyEle(rho_node, T_node);
    d_vals(IZ_NODE)   = table.Zbar(rho_node, T_node);

    // e->T inversion the representation feeds: T -> e -> T must recover the temperature.
    d_vals(IROUND_TE) = table.Te(rho_og, table.EnergyEle(rho_og, T_og));
    d_vals(IROUND_TI) = table.Ti(rho_og, table.EnergyIon(rho_og, T_og));

    // Total gas pressure split between species (the ConsToPrim consumer, #26).
    d_vals(IPGAS) = table.PressureEle(rho_og, T_og) + table.PressureIon(rho_og, T_og);

    // Table-edge clamping (no out-of-range memory access).
    d_vals(ICLAMP_RHO_LO) = table.EnergyEle(1.0e-8, T_og);    // rho -> RhoMin edge
    d_vals(ICLAMP_T_HI)   = table.EnergyEle(rho_og, 1.0e9);   // T   -> TempMax edge
    d_vals(ICLAMP_ELO)    = table.Te(rho_og, 1.0e-30);        // e below -> TempMin
    d_vals(ICLAMP_EHI)    = table.Te(rho_og, 1.0e+30);        // e above -> TempMax

    // Resolved table edges (used to express the clamp expectations).
    d_vals(ITMIN)   = table.TempMin();
    d_vals(ITMAX)   = table.TempMax();
    d_vals(IRHOMIN) = table.RhoMin();
    d_vals(IRHOMAX) = table.RhoMax();
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real rt = 1.0e-9;  // log10/pow round-off-limited relative tolerance

  // (2) Per-species values match the known fixture laws, interpolated off-grid.
  test.CheckNear(h_vals(IE_E), Eele(rho_og, T_og), rt, 0.0,
                 "EnergyEle == 1.5 rho^0.10 T (off-grid)");
  test.CheckNear(h_vals(IE_I), Eion(rho_og, T_og), rt, 0.0,
                 "EnergyIon == 0.9 rho^0.05 T (off-grid)");
  test.CheckNear(h_vals(IP_E), Pele(rho_og, T_og), rt, 0.0,
                 "PressureEle == 0.8 rho T (off-grid)");
  test.CheckNear(h_vals(IP_I), Pion(rho_og, T_og), rt, 0.0,
                 "PressureIon == 0.5 rho T (off-grid)");
  test.CheckNear(h_vals(ICV_E), Cvele(rho_og), rt, 0.0,
                 "CvEle == 1.5 rho^0.10 (off-grid)");
  test.CheckNear(h_vals(ICV_I), Cvion(rho_og), rt, 0.0,
                 "CvIon == 0.9 rho^0.05 (off-grid)");
  test.CheckNear(h_vals(IZ), Zbar(rho_og, T_og), rt, 0.0,
                 "Zbar == 3 + 0.5 log10(rho) + 0.25 log10(T) (off-grid, raw bilinear)");
  // Specific-energy/heat-capacity identity for the linear-in-T fixture: e_e = cv_e T.
  test.CheckNear(h_vals(IE_E)/T_og, h_vals(ICV_E), rt, 0.0, "e_e/T == cv_e");

  // (3) On-grid nodes reproduce the fixture entries exactly.
  test.CheckNear(h_vals(IE_E_NODE), Eele(rho_node, T_node), rt, 0.0,
                 "EnergyEle at node (rho=1e-2,T=100) is exact");
  test.CheckNear(h_vals(IZ_NODE), Zbar(rho_node, T_node), rt, 0.0,
                 "Zbar at node (rho=1e-2,T=100) is exact");

  // (4) The representation's e->T inversion round-trips on the parsed table.
  test.CheckNear(h_vals(IROUND_TE), T_og, rt, 0.0, "round-trip T_e -> e_e -> T_e");
  test.CheckNear(h_vals(IROUND_TI), T_og, rt, 0.0, "round-trip T_i -> e_i -> T_i");

  // (5) Total gas pressure is the species sum (p_gas = p_ele + p_ion).
  test.CheckNear(h_vals(IPGAS), Pele(rho_og, T_og) + Pion(rho_og, T_og), rt, 0.0,
                 "p_gas == p_ele + p_ion");

  // (6) Table-edge clamping behaves (no out-of-range lookups).
  test.CheckNear(h_vals(ICLAMP_RHO_LO), Eele(h_vals(IRHOMIN), T_og), rt, 0.0,
                 "rho below table -> lookup uses RhoMin edge");
  test.CheckNear(h_vals(ICLAMP_T_HI), Eele(rho_og, h_vals(ITMAX)), rt, 0.0,
                 "T above table -> lookup uses TempMax edge");
  test.CheckNear(h_vals(ICLAMP_ELO), h_vals(ITMIN), rt, 0.0,
                 "e below table -> T clamps to TempMin");
  test.CheckNear(h_vals(ICLAMP_EHI), h_vals(ITMAX), rt, 0.0,
                 "e above table -> T clamps to TempMax");

  // Sanity: resolved edges equal the configured fixture bounds.
  test.CheckNear(h_vals(ITMIN), 1.0e0, rt, 0.0, "TempMin == 1 eV");
  test.CheckNear(h_vals(ITMAX), 1.0e4, rt, 0.0, "TempMax == 1e4 eV");
  test.CheckNear(h_vals(IRHOMIN), 1.0e-3, rt, 0.0, "RhoMin == 1e-3 g/cc");
  test.CheckNear(h_vals(IRHOMAX), 1.0e0, rt, 0.0, "RhoMax == 1 g/cc");

  test.Finish();
  return;
}
