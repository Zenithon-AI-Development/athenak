//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_deg_floor_wiring_test.cpp
//  \brief Unit test: the <mhd> eos_deg_zstar wiring (issue #209).  Selecting
//  eos=tabulated_3t + eos_deg_zstar > 0 on a fully-initialized MHD package must
//  (a) parse the knob and enable the Fermi degeneracy-pressure floor on BOTH table
//  copies -- the package's own pmhd->eos_tbl (the cons->prim closure) AND the
//  peos->eos_data.eos_tbl handed to the Riemann/newdt path (TabulatedGasPressure) --
//  with the K coefficient built from the deck's eos_mass_per_ion and <units>; and
//  (b) actually floor the interface pressure: TabulatedGasPressure at the #209
//  cold-dense collapse state (rho = 23 code) returns the floor-dominated pressure,
//  while the default rho0 = 1 solid reference stays quiescent (zero floor).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/eos_deg_floor_wiring_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_eos_deg_floor_wiring_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief <mhd> eos_deg_zstar wiring unit test (#209).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("eos_deg_floor_wiring_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  mhd::MHD *pmhd = pmbp->pmhd;

  test.CheckTrue(pmhd != nullptr, "MHD package constructed");
  // (a) both table copies carry the enabled floor with the same coefficient.
  test.CheckTrue(pmhd->eos_tbl.deg_k > 0.0,
                 "package eos_tbl floor enabled (deg_k > 0)");
  test.CheckTrue(pmhd->peos->eos_data.eos_tbl.deg_k == pmhd->eos_tbl.deg_k,
                 "Riemann-path eos_data.eos_tbl carries the SAME floor coefficient");
  // B1 deck parameters (Z*=3, Al mion, rho_u=2.7, v_u=1e8, rho0=1): independent oracle.
  test.CheckNear(pmhd->eos_tbl.deg_k, 5.002924413768e-05, 1.0e-6, 0.0,
                 "deg_k == independent CGS oracle for Z*=3 Al at the B1 <units>");

  // (b) the interface-pressure path floors the #209 collapse state on the device.
  auto eosd = pmhd->peos->eos_data;
  auto tbl = pmhd->eos_tbl;
  DvceArray1D<Real> d_vals("degwire_vals", 3);
  auto h_vals = Kokkos::create_mirror_view(d_vals);
  par_for("degwire", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    const Real rho_c = 23.0;
    Real e_ele = rho_c*tbl.EnergyEle(rho_c, 1.0);
    Real e_ion = rho_c*tbl.EnergyIon(rho_c, 1.0);
    d_vals(0) = eosd.TabulatedGasPressure(rho_c, e_ele + e_ion, e_ele);
    d_vals(1) = tbl.DegeneracyPressureFloor(rho_c);
    d_vals(2) = tbl.DegeneracyPressureFloor(1.0);   // solid reference: quiescent
  });
  Kokkos::deep_copy(h_vals, d_vals);
  test.CheckTrue(h_vals(0) >= h_vals(1) && h_vals(1) > 0.0,
                 "TabulatedGasPressure(23 code, cold) is floor-dominated");
  test.CheckNear(h_vals(1), 9.256126893381e-03, 1.0e-6, 0.0,
                 "floor(23 code) through the wired package == CGS oracle (~250 Mbar)");
  test.CheckTrue(h_vals(2) == 0.0, "floor(rho0 = 1 solid) == 0 (quiescent liner)");

  test.Finish();
}
