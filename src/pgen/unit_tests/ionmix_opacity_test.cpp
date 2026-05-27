//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_opacity_test.cpp
//  \brief Unit test: the IONMIX multigroup-opacity reader + the common opacity
//  representation (opacity/ionmix_opacity_reader.hpp + opacity/multigroup_opacity.hpp).
//  Reads a committed IONMIX-style fixture table, then on the device checks that the
//  parsed group structure, the per-group Planck-absorption / Planck-emission / Rosseland
//  values, the (rho,T_e) interpolation, and the table-edge clamping all reproduce the
//  fixture's known synthetic power-law opacities (issue [16a]/#7, ADR-0007).
//
//  The fixture (inputs/unit_tests/ionmix_opacity_test.cn4) stores positive power-law mass
//  opacities on a log-spaced (T_e, rho) grid:
//      planck_absorb(ig,rho,T) = 2.0*(ig+1)*rho^0.5*T^-1.5
//      planck_emiss (ig,rho,T) = 3.0*(ig+1)*rho^0.4*T^-1.0
//      rosseland    (ig,rho,T) = 1.5*(ig+1)*rho^0.6*T^-2.0
//  In log10(opacity) vs (log10 rho, log10 T) space each per-group field is affine, so the
//  log-log bilinear interpolation reproduces it *exactly* on AND off grid -- so we can
//  assert the interpolated lookups against the closed-form law to round-off.
//
//  The fixture path is passed on the command line as `problem/opacity_file=<abspath>` by
//  the Python wrapper (tst/test_suite/unit_tests/test_unit_ionmix_opacity_cpu.py).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/ionmix_opacity_test -B build_unit
//    (cd build_unit/src && make) && ./build_unit/src/athena \
//        -i inputs/unit_tests/ionmix_opacity_test.athinput \
//        problem/opacity_file=$PWD/inputs/unit_tests/ionmix_opacity_test.cn4

#include <cmath>     // std::pow
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// Synthetic power-law mass-opacity laws used to generate the fixture (host helpers).
Real Kpa(int ig, Real rho, Real T) {
  return 2.0*(ig + 1)*std::pow(rho, 0.5)*std::pow(T, -1.5);
}
Real Kpe(int ig, Real rho, Real T) {
  return 3.0*(ig + 1)*std::pow(rho, 0.4)*std::pow(T, -1.0);
}
Real Kro(int ig, Real rho, Real T) {
  return 1.5*(ig + 1)*std::pow(rho, 0.6)*std::pow(T, -2.0);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief IONMIX multigroup-opacity reader + lookup-module unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("ionmix_opacity_test");

  // --- read the IONMIX fixture table from disk into the common representation ---
  std::string fname = pin->GetString("problem", "opacity_file");
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixOpacity(fname, table);

  // (1) Group structure (count + boundaries) parsed correctly.  Dimensions are PODs
  //     (host-readable); boundaries live on the device, so read them in the kernel below.
  test.CheckTrue(table.ngroups == 4, "parsed ngroups == 4");
  test.CheckTrue(table.ntemp == 5, "parsed ntemp == 5");
  test.CheckTrue(table.ndens == 4, "parsed ndens == 4");

  // Expected fixture group-energy boundaries [eV].
  const Real bounds_expect[5] = {1.0, 10.0, 100.0, 1000.0, 10000.0};

  // Off-grid query point (neither rho nor T_e lands on a node).
  const Real rho_og = 3.0e-2, T_og = 300.0;

  enum {
    IB0=0, IB1, IB2, IB3, IB4,         // group boundaries
    IPA_A, IPA_B, IRO_C, IPE_D,        // on-grid node values
    IPA_OG, IPE_OG, IRO_OG,            // off-grid interpolated values
    ICLAMP_RHO_LO, ICLAMP_T_HI, ICLAMP_BOTH,  // edge clamping
    IDENSMIN, IDENSMAX, ITMIN, ITMAX,  // resolved table edges
    IGEN_PA,                           // generic Opacity() dispatch consistency
    NVAL
  };
  DvceArray1D<Real> d_vals("opac_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("opac_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    // Group boundaries (device array).
    d_vals(IB0) = table.GroupBound(0);
    d_vals(IB1) = table.GroupBound(1);
    d_vals(IB2) = table.GroupBound(2);
    d_vals(IB3) = table.GroupBound(3);
    d_vals(IB4) = table.GroupBound(4);

    // On-grid node values (exact up to log10/pow round-off): pick spread-out (ig,rho,T).
    d_vals(IPA_A) = table.PlanckAbsorption(0, 1.0e-3, 1.0);     // corner (id0,it0)
    d_vals(IPA_B) = table.PlanckAbsorption(3, 1.0,    1.0e4);   // corner (id3,it4)
    d_vals(IRO_C) = table.RosselandTransport(2, 1.0e-2, 100.0); // interior node
    d_vals(IPE_D) = table.PlanckEmission(1, 1.0e-1, 1.0e3);     // interior node

    // Off-grid interpolation at (rho_og, T_og) across kinds & groups.
    d_vals(IPA_OG) = table.PlanckAbsorption(2, rho_og, T_og);
    d_vals(IPE_OG) = table.PlanckEmission(0, rho_og, T_og);
    d_vals(IRO_OG) = table.RosselandTransport(3, rho_og, T_og);

    // Table-edge clamping (no out-of-range memory access): out-of-range axis queries
    // resolve to the nearest edge node.
    d_vals(ICLAMP_RHO_LO) = table.PlanckAbsorption(1, 1.0e-8, T_og);   // rho -> DensMin
    d_vals(ICLAMP_T_HI)   = table.PlanckAbsorption(1, 1.0e-2, 1.0e9);  // T   -> TempMax
    d_vals(ICLAMP_BOTH)   = table.PlanckAbsorption(0, 1.0e-8, 1.0e-6); // both edges

    // Resolved table edges (used to express the clamp expectations).
    d_vals(IDENSMIN) = table.DensMin();
    d_vals(IDENSMAX) = table.DensMax();
    d_vals(ITMIN)    = table.TempMin();
    d_vals(ITMAX)    = table.TempMax();

    // Generic kind-dispatched lookup must equal the named accessor.
    d_vals(IGEN_PA) = table.Opacity(opacity::kPlanckAbsorption, 2, rho_og, T_og);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  const Real rt = 1.0e-9;  // log10/pow round-off-limited relative tolerance

  // (1, cont.) Group boundaries match the fixture.
  test.CheckNear(h_vals(IB0), bounds_expect[0], rt, 0.0, "group bound 0 == 1 eV");
  test.CheckNear(h_vals(IB1), bounds_expect[1], rt, 0.0, "group bound 1 == 10 eV");
  test.CheckNear(h_vals(IB2), bounds_expect[2], rt, 0.0, "group bound 2 == 100 eV");
  test.CheckNear(h_vals(IB3), bounds_expect[3], rt, 0.0, "group bound 3 == 1000 eV");
  test.CheckNear(h_vals(IB4), bounds_expect[4], rt, 0.0, "group bound 4 == 10000 eV");

  // (2) Per-group Planck/Rosseland values match known fixture entries (on-grid nodes).
  test.CheckNear(h_vals(IPA_A), Kpa(0, 1.0e-3, 1.0), rt, 0.0,
                 "PlanckAbsorption(ig0) at (rho_min,T_min) node");
  test.CheckNear(h_vals(IPA_B), Kpa(3, 1.0, 1.0e4), rt, 0.0,
                 "PlanckAbsorption(ig3) at (rho_max,T_max) node");
  test.CheckNear(h_vals(IRO_C), Kro(2, 1.0e-2, 100.0), rt, 0.0,
                 "RosselandTransport(ig2) at interior node");
  test.CheckNear(h_vals(IPE_D), Kpe(1, 1.0e-1, 1.0e3), rt, 0.0,
                 "PlanckEmission(ig1) at interior node");

  // (3) Interpolation in (rho,T_e) verified against reference points (off-grid, exact for
  //     the power-law fixture under log-log interpolation).
  test.CheckNear(h_vals(IPA_OG), Kpa(2, rho_og, T_og), rt, 0.0,
                 "PlanckAbsorption(ig2) interpolated off-grid");
  test.CheckNear(h_vals(IPE_OG), Kpe(0, rho_og, T_og), rt, 0.0,
                 "PlanckEmission(ig0) interpolated off-grid");
  test.CheckNear(h_vals(IRO_OG), Kro(3, rho_og, T_og), rt, 0.0,
                 "RosselandTransport(ig3) interpolated off-grid");

  // (3, cont.) Edges are clamped: an out-of-range axis query uses the edge node.
  test.CheckNear(h_vals(ICLAMP_RHO_LO), Kpa(1, h_vals(IDENSMIN), T_og), rt, 0.0,
                 "rho below table -> lookup uses DensMin edge");
  test.CheckNear(h_vals(ICLAMP_T_HI), Kpa(1, 1.0e-2, h_vals(ITMAX)), rt, 0.0,
                 "T above table -> lookup uses TempMax edge");
  test.CheckNear(h_vals(ICLAMP_BOTH), Kpa(0, h_vals(IDENSMIN), h_vals(ITMIN)), rt, 0.0,
                 "rho & T below table -> lookup uses (DensMin,TempMin) corner");

  // Sanity: resolved edges equal the configured fixture bounds.
  test.CheckNear(h_vals(IDENSMIN), 1.0e-3, rt, 0.0, "DensMin == 1e-3 g/cc");
  test.CheckNear(h_vals(IDENSMAX), 1.0e0, rt, 0.0, "DensMax == 1 g/cc");
  test.CheckNear(h_vals(ITMIN), 1.0e0, rt, 0.0, "TempMin == 1 eV");
  test.CheckNear(h_vals(ITMAX), 1.0e4, rt, 0.0, "TempMax == 1e4 eV");

  // (4) Generic kind-dispatched lookup matches the named accessor.
  test.CheckNear(h_vals(IGEN_PA), h_vals(IPA_OG), 0.0, 0.0,
                 "Opacity(kPlanckAbsorption,...) == PlanckAbsorption(...)");

  test.Finish();
  return;
}
