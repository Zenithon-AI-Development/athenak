//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_cn4_eos_test.cpp
//  \brief Unit test: ingest a real FLASH IONMIX `.cn4` EOS table (aluminum / beryllium /
//  deuterium) through `eos/ionmix_eos_reader.hpp`'s `ReadIonmixCn4Eos` into the common 3T
//  EOS representation (eos/eos_table_3t.hpp), and verify the parsed grid, the per-species
//  electron/ion energies / pressures / ionization reproduce the table's known values,
//  the e->T inversion round-trips, and out-of-range queries clamp to the table edges
//  (issue [C1]/#118, ADR-0007/0008).
//
//  The "known table values" are supplied by the Python wrapper
//  (tst/test_suite/unit_tests/test_unit_ionmix_cn4_eos_cpu.py), which *independently*
//  decodes the same packed cn4 file (the opacplot2 reference format) and passes the
//  reference dimensions, the material charge state Z, and the node values at two interior
//  probe nodes as `problem/...` command-line overrides.  This pgen reads the table itself
//  and asserts the reader reproduces them, plus the format-agnostic physical invariants
//  (Zbar in [0,Z], e monotone in T, round-trip, edge clamping).
//
//  Built/run by the wrapper:
//    cmake -D PROBLEM=unit_tests/ionmix_cn4_eos_test -B build_unit/... && make
//    ./athena -i inputs/unit_tests/ionmix_cn4_eos_test.athinput \
//        problem/eos_file=<abspath to a real .cn4> problem/exp_*=<oracle values>

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Real-material IONMIX cn4 EOS ingestion + lookup-verification unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("ionmix_cn4_eos_test");

  // --- read the real IONMIX cn4 EOS table into the common representation ---
  std::string fname = pin->GetString("problem", "eos_file");
  eos_table_3t::EosTable3T table;
  eos_table_3t::ReadIonmixCn4Eos(fname, table);  // density axis = number density

  // Reference values from the independent Python decode (the #118 oracle).
  const int exp_ntemp = pin->GetInteger("problem", "exp_ntemp");
  const int exp_ndens = pin->GetInteger("problem", "exp_ndens");
  const Real exp_zsum = pin->GetReal("problem", "exp_zsum");  // sum z*frac = Z_material

  // (1) Grid parsed correctly.
  test.CheckTrue(table.ntemp == exp_ntemp, "parsed ntemp matches the cn4 header");
  test.CheckTrue(table.nrho == exp_ndens, "parsed ndens matches the cn4 header");

  // Two interior probe nodes (indices + expected per-species values) from the oracle.
  const int pit0 = pin->GetInteger("problem", "pit0");
  const int pid0 = pin->GetInteger("problem", "pid0");
  const int pit1 = pin->GetInteger("problem", "pit1");
  const int pid1 = pin->GetInteger("problem", "pid1");

  enum {
    E0_EELE=0, E0_EION, E0_PELE, E0_PION, E0_ZBAR, E0_CVELE, E0_CVION,
    E1_EELE, E1_EION, E1_PELE, E1_PION, E1_ZBAR, E1_CVELE, E1_CVION,
    ROUND_TE, ROUND_TI,                 // e->T round-trip at probe node 0
    ZMIN, ZMAX,                         // global Zbar bounds (over all nodes)
    EMONO,                              // 1 if e_ele strictly increasing in T everywhere
    CLAMP_TLO, CLAMP_THI, CLAMP_ELO, CLAMP_EHI,
    TMIN, TMAX, RHOMIN, RHOMAX,
    INTERP_MID, NODE_LO, NODE_HI,       // off-node bracket sanity (probe-node column)
    NVAL
  };
  DvceArray1D<Real> d_vals("cn4_eos_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);
  const int ntemp = table.ntemp, nrho = table.nrho;

  par_for("cn4_eos_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // Probe nodes: read the RAW stored array entries (a faithful-load check, free of the
    // ~1e-5 error the table's ideal log-uniform-axis reconstruction would add to an
    // at-node lookup on a 6-significant-figure grid).  EosTable3T stores (it, id).
    d_vals(E0_EELE)  = table.e_ele(pit0, pid0);
    d_vals(E0_EION)  = table.e_ion(pit0, pid0);
    d_vals(E0_PELE)  = table.p_ele(pit0, pid0);
    d_vals(E0_PION)  = table.p_ion(pit0, pid0);
    d_vals(E0_ZBAR)  = table.zbar(pit0, pid0);
    d_vals(E0_CVELE) = table.cv_ele(pit0, pid0);
    d_vals(E0_CVION) = table.cv_ion(pit0, pid0);
    d_vals(E1_EELE)  = table.e_ele(pit1, pid1);
    d_vals(E1_EION)  = table.e_ion(pit1, pid1);
    d_vals(E1_PELE)  = table.p_ele(pit1, pid1);
    d_vals(E1_PION)  = table.p_ion(pit1, pid1);
    d_vals(E1_ZBAR)  = table.zbar(pit1, pid1);
    d_vals(E1_CVELE) = table.cv_ele(pit1, pid1);
    d_vals(E1_CVION) = table.cv_ion(pit1, pid1);

    // e->T inversion round-trip at node 0 (consistent interpolant -> recovers T).
    Real rho0 = table.RhoAt(pid0), T0 = table.TempAt(pit0);
    d_vals(ROUND_TE) = table.Te(rho0, table.EnergyEle(rho0, T0));
    d_vals(ROUND_TI) = table.Ti(rho0, table.EnergyIon(rho0, T0));

    // Global Zbar bounds and electron-energy monotonicity in T over EVERY node.
    Real zmin = table.zbar(0, 0), zmax = table.zbar(0, 0);
    Real mono = 1.0;
    for (int ir = 0; ir < nrho; ++ir) {
      for (int it = 0; it < ntemp; ++it) {
        Real z = table.zbar(it, ir);
        zmin = Kokkos::fmin(zmin, z);
        zmax = Kokkos::fmax(zmax, z);
        if (it > 0 && !(table.e_ele(it, ir) > table.e_ele(it-1, ir))) { mono = 0.0; }
      }
    }
    d_vals(ZMIN) = zmin;
    d_vals(ZMAX) = zmax;
    d_vals(EMONO) = mono;

    // Table-edge clamping (no out-of-range memory access).
    d_vals(TMIN)   = table.TempMin();
    d_vals(TMAX)   = table.TempMax();
    d_vals(RHOMIN) = table.RhoMin();
    d_vals(RHOMAX) = table.RhoMax();
    d_vals(CLAMP_TLO) = table.EnergyEle(rho0, 1.0e-30*table.TempMin());  // T below
    d_vals(CLAMP_THI) = table.EnergyEle(rho0, 1.0e+30*table.TempMax());  // T above
    d_vals(CLAMP_ELO) = table.Te(rho0, 1.0e-300);   // e far below -> TempMin
    d_vals(CLAMP_EHI) = table.Te(rho0, 1.0e+300);   // e far above -> TempMax

    // Off-node interpolation is bracketed by its two bounding nodes (monotone in T).
    Real Tmid = table.TempAt(pit0) *
                Kokkos::sqrt(table.TempAt(pit0+1)/table.TempAt(pit0));  // log-midpoint
    d_vals(INTERP_MID) = table.EnergyEle(rho0, Tmid);
    d_vals(NODE_LO) = table.e_ele(pit0,   pid0);
    d_vals(NODE_HI) = table.e_ele(pit0+1, pid0);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real rt = 1.0e-6;        // self-consistent (round-trip/clamp) round-off tolerance
  const Real rt_load = 1.0e-9;   // raw-load tolerance (stored entry vs oracle decode)

  // (2) The raw per-species node entries reproduce the known table values (the oracle's
  //     independent decode of the same cn4 bytes) -- a faithful-load check.
  test.CheckNear(h_vals(E0_EELE), pin->GetReal("problem", "exp_eele0"), rt_load, 0.0,
                 "e_ele at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E0_EION), pin->GetReal("problem", "exp_eion0"), rt_load, 0.0,
                 "e_ion at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E0_PELE), pin->GetReal("problem", "exp_pele0"), rt_load, 0.0,
                 "p_ele at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E0_PION), pin->GetReal("problem", "exp_pion0"), rt_load, 0.0,
                 "p_ion at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E0_ZBAR), pin->GetReal("problem", "exp_zbar0"), rt_load, 1.0e-12,
                 "Zbar at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E0_CVELE), pin->GetReal("problem", "exp_cvele0"), rt_load, 0.0,
                 "c_v,e at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E0_CVION), pin->GetReal("problem", "exp_cvion0"), rt_load, 0.0,
                 "c_v,i at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(E1_EELE), pin->GetReal("problem", "exp_eele1"), rt_load, 0.0,
                 "e_ele at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(E1_EION), pin->GetReal("problem", "exp_eion1"), rt_load, 0.0,
                 "e_ion at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(E1_PELE), pin->GetReal("problem", "exp_pele1"), rt_load, 0.0,
                 "p_ele at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(E1_PION), pin->GetReal("problem", "exp_pion1"), rt_load, 0.0,
                 "p_ion at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(E1_ZBAR), pin->GetReal("problem", "exp_zbar1"), rt_load, 1.0e-12,
                 "Zbar at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(E1_CVELE), pin->GetReal("problem", "exp_cvele1"), rt_load, 0.0,
                 "c_v,e at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(E1_CVION), pin->GetReal("problem", "exp_cvion1"), rt_load, 0.0,
                 "c_v,i at probe node 1 reproduces the table value");

  // (3) Mean ionization stays physical: 0 <= Zbar <= Z_material across the whole table.
  test.CheckTrue(h_vals(ZMIN) >= -1.0e-12, "Zbar >= 0 over the whole table");
  test.CheckTrue(h_vals(ZMAX) <= exp_zsum*(1.0 + 1.0e-6) + 1.0e-9,
                 "Zbar <= Z_material over the whole table");

  // (4) Electron specific energy is strictly increasing in T at every density (so the
  //     e->T inversion is well-posed), and the inversion round-trips at a node.
  test.CheckTrue(h_vals(EMONO) == 1.0, "e_ele strictly increasing in T at every density");
  test.CheckNear(h_vals(ROUND_TE), table.TempAt(pit0), rt, 0.0,
                 "round-trip T_e -> e_e -> T_e on the real table");
  test.CheckNear(h_vals(ROUND_TI), table.TempAt(pit0), rt, 0.0,
                 "round-trip T_i -> e_i -> T_i on the real table");

  // (5) Table-edge clamping: out-of-range (T, e) queries return the edge result.
  test.CheckNear(h_vals(CLAMP_TLO), table.EnergyEle(table.RhoAt(pid0), h_vals(TMIN)),
                 rt, 0.0, "T below table -> EnergyEle uses TempMin edge");
  test.CheckNear(h_vals(CLAMP_THI), table.EnergyEle(table.RhoAt(pid0), h_vals(TMAX)),
                 rt, 0.0, "T above table -> EnergyEle uses TempMax edge");
  test.CheckNear(h_vals(CLAMP_ELO), h_vals(TMIN), rt, 0.0,
                 "e below table -> T clamps to TempMin");
  test.CheckNear(h_vals(CLAMP_EHI), h_vals(TMAX), rt, 0.0,
                 "e above table -> T clamps to TempMax");

  // (6) Off-node interpolation is bracketed by its bounding node values (monotone).
  test.CheckTrue(h_vals(INTERP_MID) >= h_vals(NODE_LO)*(1.0 - 1.0e-9)
                 && h_vals(INTERP_MID) <= h_vals(NODE_HI)*(1.0 + 1.0e-9),
                 "off-node EnergyEle lies between its bracketing node values");

  test.Finish();
  return;
}
