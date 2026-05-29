//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_cn4_opacity_test.cpp
//  \brief Unit test: ingest a *real* FLASH IONMIX `.cn4` multigroup-opacity table
//  (aluminum / beryllium / deuterium) through `opacity/ionmix_opacity_reader.hpp`'s
//  `ReadIonmixCn4Opacity` into the common opacity representation
//  (opacity/multigroup_opacity.hpp), and verify the parsed grid + group structure and the
//  per-group Planck-absorption / Planck-emission / Rosseland opacities reproduce the
//  table's known values, with table-edge clamping (issue [C1]/#118, ADR-0007/0008).
//
//  The "known table values" are supplied by the Python wrapper
//  (tst/test_suite/unit_tests/test_unit_ionmix_cn4_opacity_cpu.py), which independently
//  decodes the same packed cn4 file and passes the dimensions, group boundaries,
//  and node opacities at two interior probe nodes as `problem/...` overrides.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Real-material IONMIX cn4 multigroup-opacity ingestion + lookup-verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("ionmix_cn4_opacity_test");

  std::string fname = pin->GetString("problem", "opacity_file");
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixCn4Opacity(fname, table);  // density axis = number density

  const int exp_ntemp = pin->GetInteger("problem", "exp_ntemp");
  const int exp_ndens = pin->GetInteger("problem", "exp_ndens");
  const int exp_ngroups = pin->GetInteger("problem", "exp_ngroups");

  // (1) Grid + group structure parsed correctly.
  test.CheckTrue(table.ntemp == exp_ntemp, "parsed ntemp matches the cn4 header");
  test.CheckTrue(table.ndens == exp_ndens, "parsed ndens matches the cn4 header");
  test.CheckTrue(table.ngroups == exp_ngroups, "parsed ngroups matches the cn4 header");

  // Two interior probe nodes (group + T + rho indices) from the oracle.
  const int pg0 = pin->GetInteger("problem", "pg0");
  const int pit0 = pin->GetInteger("problem", "pit0");
  const int pid0 = pin->GetInteger("problem", "pid0");
  const int pg1 = pin->GetInteger("problem", "pg1");
  const int pit1 = pin->GetInteger("problem", "pit1");
  const int pid1 = pin->GetInteger("problem", "pid1");

  enum {
    O0_PA=0, O0_PE, O0_RO,
    O1_PA, O1_PE, O1_RO,
    GB_LO, GB_HI,                       // first/last group boundary
    GB_ASC,                             // 1 if group bounds strictly ascending
    O_POS,                              // 1 if every opacity entry is positive
    CLAMP_RO_TLO, CLAMP_RO_THI,         // edge clamping (Rosseland, group pg0)
    TMIN, TMAX,
    INTERP_MID, NODE_LO, NODE_HI,       // off-node bracket sanity (Rosseland)
    NVAL
  };
  DvceArray1D<Real> d_vals("cn4_opac_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);
  const int ntemp = table.ntemp, ndens = table.ndens, ngroups = table.ngroups;

  par_for("cn4_opac_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // Raw stored opacity entries (faithful load; MultigroupOpacity stores (ig,it,id)).
    d_vals(O0_PA) = table.planck_absorb(pg0, pit0, pid0);
    d_vals(O0_PE) = table.planck_emiss(pg0, pit0, pid0);
    d_vals(O0_RO) = table.rosseland(pg0, pit0, pid0);
    d_vals(O1_PA) = table.planck_absorb(pg1, pit1, pid1);
    d_vals(O1_PE) = table.planck_emiss(pg1, pit1, pid1);
    d_vals(O1_RO) = table.rosseland(pg1, pit1, pid1);

    d_vals(GB_LO) = table.GroupBound(0);
    d_vals(GB_HI) = table.GroupBound(ngroups);
    Real asc = 1.0;
    for (int ig = 0; ig < ngroups; ++ig) {
      if (!(table.GroupBound(ig+1) > table.GroupBound(ig))) { asc = 0.0; }
    }
    d_vals(GB_ASC) = asc;

    // Every stored opacity is strictly positive (log-interpolated).
    Real pos = 1.0;
    for (int ig = 0; ig < ngroups; ++ig) {
      for (int it = 0; it < ntemp; ++it) {
        for (int id = 0; id < ndens; ++id) {
          if (!(table.planck_absorb(ig, it, id) > 0.0)
              || !(table.planck_emiss(ig, it, id) > 0.0)
              || !(table.rosseland(ig, it, id) > 0.0)) { pos = 0.0; }
        }
      }
    }
    d_vals(O_POS) = pos;

    Real rho0 = table.DensAt(pid0);
    d_vals(TMIN) = table.TempMin();
    d_vals(TMAX) = table.TempMax();
    d_vals(CLAMP_RO_TLO) = table.RosselandTransport(pg0, rho0, 1.0e-30*table.TempMin());
    d_vals(CLAMP_RO_THI) = table.RosselandTransport(pg0, rho0, 1.0e+30*table.TempMax());

    // Off-node interpolation bracketed by bounding node values (group pg0, density node).
    Real Tmid = table.TempAt(pit0)*Kokkos::sqrt(table.TempAt(pit0+1)/table.TempAt(pit0));
    d_vals(INTERP_MID) = table.RosselandTransport(pg0, rho0, Tmid);
    Real lo = table.rosseland(pg0, pit0,   pid0);
    Real hi = table.rosseland(pg0, pit0+1, pid0);
    d_vals(NODE_LO) = Kokkos::fmin(lo, hi);
    d_vals(NODE_HI) = Kokkos::fmax(lo, hi);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real rt = 1.0e-6;        // self-consistent (clamp/bracket) round-off tolerance
  const Real rt_load = 1.0e-9;   // raw-load tolerance (stored entry vs oracle decode)

  // (2) The raw per-group node opacities reproduce the known table values (the oracle's
  //     independent decode of the same cn4 bytes) -- a faithful-load check.
  test.CheckNear(h_vals(O0_PA), pin->GetReal("problem", "exp_pa0"), rt_load, 0.0,
                 "planck_absorb at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(O0_PE), pin->GetReal("problem", "exp_pe0"), rt_load, 0.0,
                 "planck_emiss at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(O0_RO), pin->GetReal("problem", "exp_ro0"), rt_load, 0.0,
                 "rosseland at probe node 0 reproduces the table value");
  test.CheckNear(h_vals(O1_PA), pin->GetReal("problem", "exp_pa1"), rt_load, 0.0,
                 "planck_absorb at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(O1_PE), pin->GetReal("problem", "exp_pe1"), rt_load, 0.0,
                 "planck_emiss at probe node 1 reproduces the table value");
  test.CheckNear(h_vals(O1_RO), pin->GetReal("problem", "exp_ro1"), rt_load, 0.0,
                 "rosseland at probe node 1 reproduces the table value");

  // (3) Group structure: ascending boundaries reproduced from the table.
  test.CheckTrue(h_vals(GB_ASC) == 1.0, "group-energy boundaries strictly ascending");
  test.CheckNear(h_vals(GB_LO), pin->GetReal("problem", "exp_gb_lo"), rt_load, 0.0,
                 "first group boundary reproduces the table value");
  test.CheckNear(h_vals(GB_HI), pin->GetReal("problem", "exp_gb_hi"), rt_load, 0.0,
                 "last group boundary reproduces the table value");

  // (4) Every opacity entry is positive (well-posed log-log interpolation).
  test.CheckTrue(h_vals(O_POS) == 1.0, "all stored opacities are strictly positive");

  // (5) Table-edge clamping: out-of-range T queries return the edge result.
  test.CheckNear(h_vals(CLAMP_RO_TLO),
                 table.RosselandTransport(pg0, table.DensAt(pid0), h_vals(TMIN)), rt, 0.0,
                 "T below table -> Rosseland uses TempMin edge");
  test.CheckNear(h_vals(CLAMP_RO_THI),
                 table.RosselandTransport(pg0, table.DensAt(pid0), h_vals(TMAX)), rt, 0.0,
                 "T above table -> Rosseland uses TempMax edge");

  // (6) Off-node interpolation lies between its bracketing node values.
  test.CheckTrue(h_vals(INTERP_MID) >= h_vals(NODE_LO)*(1.0 - 1.0e-9)
                 && h_vals(INTERP_MID) <= h_vals(NODE_HI)*(1.0 + 1.0e-9),
                 "off-node Rosseland lies between its bracketing node values");

  test.Finish();
  return;
}
