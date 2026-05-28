//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file faraday_voltage_test.cpp
//  \brief Unit test: Faraday load-voltage global reduction (issue [18a]/#31, ADR-0005).
//
//  Exercises circuit/faraday_voltage.hpp on real device kernels in a cylindrical
//  (ADR-0004) MHD block.  The diagnostic computes the poloidal magnetic flux
//      Phi = \int\int B_phi dr dz   (averaged over phi-planes)
//  via a Kokkos parallel_reduce over the domain, and the Faraday load voltage
//      V_load = d(Phi)/dt
//  as a backward time difference of successive flux samples.
//
//  Batteries:
//   (A) Reduction matches the analytic flux: with B_phi(r,z) set to a known affine field
//       on the x2-faces, PoloidalFluxGlobal equals the exact discrete poloidal-plane sum
//       Sum_{i,k} B_phi(r_i,z_k) * dr*dz.  Because the mesh uses nx2=2 phi-planes with an
//       axisymmetric field, this also pins the divide-by-(global nphi) per-plane average.
//   (B) Faraday voltage matches d(Phi)/dt: after a second field at a later time, the
//       FaradayVoltage monitor returns (Phi1-Phi0)/(t1-t0), equal to the analytic rate;
//       the first sample (no previous) returns 0.
//   (C) History exposure: FillFaradayHistory writes the two diagnostic columns
//       ("Bphi_flux", "V_load") into a HistoryData with the reduced values.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/faraday_voltage_test -B build_unit/faraday_voltage
//    (cd build_unit/.../src && make) && ./athena -i inputs/unit_tests/<this>.athinput
//  Auto-run by tst/test_suite/unit_tests/test_unit_faraday_voltage_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "outputs/outputs.hpp"
#include "circuit/faraday_voltage.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// Affine B_phi(r,z) = c0 + cr*r + cz*z used to set the x2-face field.
struct BphiField {
  Real c0, cr, cz;
  KOKKOS_INLINE_FUNCTION
  Real operator()(Real r, Real z) const { return c0 + cr*r + cz*z; }
};
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Faraday load-voltage reduction unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  unit_test::UnitTest test("faraday_voltage_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  test.CheckTrue(pmbp->pmhd != nullptr, "mhd block constructed (B_phi flux source)");
  if (pmbp->pmhd == nullptr) { test.Finish(); return; }
  test.CheckTrue(pmbp->pcoord->coord_system == CoordSystem::cylindrical,
                 "<coord> system = cylindrical (r,phi,z)");

  // mesh geometry (single MeshBlock spans the whole domain in this test)
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is; int ie = indcs.ie; int nx1 = indcs.nx1;
  int js = indcs.js; int je = indcs.je;
  int ks = indcs.ks; int ke = indcs.ke; int nx3 = indcs.nx3;
  Real x1min = pmy_mesh_->mesh_size.x1min, x1max = pmy_mesh_->mesh_size.x1max;
  Real x3min = pmy_mesh_->mesh_size.x3min, x3max = pmy_mesh_->mesh_size.x3max;
  Real dx1 = (x1max - x1min)/static_cast<Real>(nx1);
  Real dx3 = (x3max - x3min)/static_cast<Real>(nx3);

  // host helper: exact discrete poloidal-plane flux Sum_{i,k} B_phi(r_i,z_k)*dr*dz.
  auto analytic_flux = [&](const BphiField &f) -> Real {
    Real phi = 0.0;
    for (int i = 0; i < nx1; ++i) {
      Real r = CellCenterX(i, nx1, x1min, x1max);
      for (int k = 0; k < nx3; ++k) {
        Real z = CellCenterX(k, nx3, x3min, x3max);
        phi += f(r, z)*dx1*dx3;
      }
    }
    return phi;
  };

  // device helper: write B_phi(r,z) onto the active x2-faces (b0.x2f is cell-centred in
  // r,z; axisymmetric so every phi-plane is identical) and return the reduced flux.
  auto &size = pmbp->pmb->mb_size;
  auto &bx2f = pmbp->pmhd->b0.x2f;
  auto set_field_and_reduce = [&](const BphiField &f) -> Real {
    par_for("set_bphi", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &xmin = size.d_view(m).x1min;
      Real &xmax = size.d_view(m).x1max;
      Real &zmin = size.d_view(m).x3min;
      Real &zmax = size.d_view(m).x3max;
      Real r = CellCenterX(i-is, nx1, xmin, xmax);
      Real z = CellCenterX(k-ks, nx3, zmin, zmax);
      bx2f(m,k,j,i) = f(r, z);
    });
    Kokkos::fence();
    return circuit::PoloidalFluxGlobal(pmy_mesh_);
  };

  // Two affine B_phi(r,z) fields sampled at t0 < t1 (overridable from <problem>).  The
  // defaults are arbitrary nonzero coefficients chosen so the flux and its rate are both
  // nonzero (a discriminating test).
  BphiField f0{pin->GetOrAddReal("problem", "b0_const", 0.6),
               pin->GetOrAddReal("problem", "b0_slope_r", 0.25),
               pin->GetOrAddReal("problem", "b0_slope_z", -0.15)};
  BphiField f1{pin->GetOrAddReal("problem", "b1_const", 1.1),
               pin->GetOrAddReal("problem", "b1_slope_r", -0.10),
               pin->GetOrAddReal("problem", "b1_slope_z", 0.30)};
  Real t0 = pin->GetOrAddReal("problem", "t0", 0.0);
  Real t1 = pin->GetOrAddReal("problem", "t1", 0.4);

  // (A) Reduction matches the analytic flux for the first (t0) field.
  Real flux0 = set_field_and_reduce(f0);
  Real flux0_exact = analytic_flux(f0);
  test.CheckNear(flux0, flux0_exact, 1.0e-12, 1.0e-14,
                 "PoloidalFluxGlobal == analytic poloidal flux (t0)");
  test.CheckTrue(flux0_exact != 0.0, "analytic flux is nonzero (test is discriminating)");

  // (B) Faraday voltage matches d(Phi)/dt across two samples.
  Real flux1 = set_field_and_reduce(f1);
  Real flux1_exact = analytic_flux(f1);
  test.CheckNear(flux1, flux1_exact, 1.0e-12, 1.0e-14,
                 "PoloidalFluxGlobal == analytic poloidal flux (t1)");

  circuit::FaradayVoltage mon;
  Real v_first = mon.Update(flux0, t0);          // no previous sample -> 0
  test.CheckTrue(mon.initialized(), "monitor initialized after first sample");
  test.CheckNear(v_first, 0.0, 0.0, 1.0e-14, "first Faraday sample returns V=0");

  Real v_load = mon.Update(flux1, t1);           // V = (Phi1 - Phi0)/(t1 - t0)
  Real v_exact = (flux1_exact - flux0_exact)/(t1 - t0);
  test.CheckNear(v_load, v_exact, 1.0e-12, 1.0e-14,
                 "Faraday V_load == analytic d(flux)/dt");
  test.CheckNear(mon.voltage(), v_load, 0.0, 1.0e-14, "monitor caches the last voltage");
  test.CheckNear(mon.flux(), flux1, 0.0, 1.0e-14, "monitor caches the last flux");

  // (C) History exposure: FillFaradayHistory writes the two diagnostic columns.  Re-seed
  // the field to f1 (the reduction reads the live b0.x2f) and use a fresh monitor whose
  // prior sample is flux0 at t0, so the history voltage equals the analytic rate.
  set_field_and_reduce(f1);
  circuit::FaradayVoltage hmon;
  hmon.Update(flux0, t0);
  HistoryData hd(PhysicsModule::UserDefined);
  // give the Mesh the t1 time used for the history difference.
  pmy_mesh_->time = t1;
  circuit::FillFaradayHistory(&hd, pmy_mesh_, hmon);
  test.CheckTrue(hd.nhist == 2, "FillFaradayHistory sets nhist = 2");
  test.CheckTrue(hd.label[0] == "Bphi_flux", "history column 0 labelled Bphi_flux");
  test.CheckTrue(hd.label[1] == "V_load", "history column 1 labelled V_load");
  test.CheckNear(hd.hdata[0], flux1_exact, 1.0e-12, 1.0e-14,
                 "history flux column == analytic flux");
  test.CheckNear(hd.hdata[1], v_exact, 1.0e-12, 1.0e-14,
                 "history voltage column == analytic d(flux)/dt");

  test.Finish();
  // All checks live in UserProblem on the live MHD field + local objects; exit cleanly on
  // success rather than proceed into the (pointless) trivial driver run (which would run
  // ConsToPrim on the unset state).
  std::exit(EXIT_SUCCESS);
  return;
}
