//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coord_geometry_test.cpp
//  \brief Unit test: the Cartesian coordinate-geometry accessors (coord_geometry.hpp)
//  return the analytic uniform-grid constants, and the flux-divergence helpers reduce to
//  the legacy (F_{i+1}-F_i)/dx form, when evaluated on the device.
//
// This is the [1a] geometry-accessor foundation (ADR-0004): it locks in that the
// Cartesian specialization is exactly the previous arithmetic, so routing the hydro/MHD
// update through the accessors leaves Cartesian behavior unchanged.  Built/run by
//   cmake -D PROBLEM=unit_tests/coord_geometry_test -B build_unit
//   (cd build_unit/src && make) && ./build_unit/src/athena \
//       -i inputs/unit_tests/coord_geometry_test.athinput
// Auto-run by tst/test_suite/unit_tests/test_unit_coord_geometry_cpu.py.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Cartesian geometry-accessor unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("coord_geometry_test");

  // (0) The default `<coord> system` parses to CoordSystem::cartesian.
  CoordSystem csys = pmy_mesh_->pmb_pack->pcoord->coord_system;
  test.CheckTrue(csys == CoordSystem::cartesian,
                 "default <coord> system parses to cartesian");

  // Distinct, non-trivial cell spacings so each accessor's product is unambiguous.
  const Real dx1 = 0.5, dx2 = 0.25, dx3 = 2.0;
  // Left/right x1-face fluxes of a linear flux with slope `slope`: F(x)=F0+slope*x,
  // so (Fr-Fl)/dx1 must recover `slope` exactly (the discrete divergence of a linear
  // flux is its slope).  Reuse the same slope for the x2/x3 helpers.
  const Real slope = 3.0;
  const Real f1l = 7.0,            f1r = f1l + slope*dx1;
  const Real f2l = 7.0,            f2r = f2l + slope*dx2;
  const Real f3l = 7.0,            f3r = f3l + slope*dx3;

  // (1) Evaluate every accessor inside a real device kernel and copy results to host.
  enum {
    IVOL=0, IA1, IA2, IA3, IL1, IL2, IL3, ICW1, ICW2, ICW3,
    IS1, IS2, IS3, ID1, ID2, ID3, NVAL
  };
  DvceArray1D<Real> d_vals("coord_geom_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("coord_geom_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    d_vals(IVOL) = CellVolume(csys, dx1, dx2, dx3);
    d_vals(IA1)  = Face1Area(csys, dx2, dx3);
    d_vals(IA2)  = Face2Area(csys, dx1, dx3);
    d_vals(IA3)  = Face3Area(csys, dx1, dx2);
    d_vals(IL1)  = Edge1Length(csys, dx1);
    d_vals(IL2)  = Edge2Length(csys, dx2);
    d_vals(IL3)  = Edge3Length(csys, dx3);
    d_vals(ICW1) = CenterWidth1(csys, dx1);
    d_vals(ICW2) = CenterWidth2(csys, dx2);
    d_vals(ICW3) = CenterWidth3(csys, dx3);
    d_vals(IS1)  = CoordSrc1Coeff(csys);
    d_vals(IS2)  = CoordSrc2Coeff(csys);
    d_vals(IS3)  = CoordSrc3Coeff(csys);
    d_vals(ID1)  = FluxDivX1(csys, f1l, f1r, dx1);
    d_vals(ID2)  = FluxDivX2(csys, f2l, f2r, dx2);
    d_vals(ID3)  = FluxDivX3(csys, f3l, f3r, dx3);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // (2) Cartesian cell volume and face areas are the uniform-grid products (exact).
  test.CheckNear(h_vals(IVOL), dx1*dx2*dx3, 0.0, 0.0, "CellVolume == dx1*dx2*dx3");
  test.CheckNear(h_vals(IA1),  dx2*dx3,     0.0, 0.0, "Face1Area == dx2*dx3");
  test.CheckNear(h_vals(IA2),  dx1*dx3,     0.0, 0.0, "Face2Area == dx1*dx3");
  test.CheckNear(h_vals(IA3),  dx1*dx2,     0.0, 0.0, "Face3Area == dx1*dx2");

  // (3) Cartesian edge lengths are the cell widths (exact); they are the CT denominators
  // (mhd_ct divides EMF differences by Edge?Length in the differentiation direction).
  test.CheckNear(h_vals(IL1), dx1, 0.0, 0.0, "Edge1Length == dx1");
  test.CheckNear(h_vals(IL2), dx2, 0.0, 0.0, "Edge2Length == dx2");
  test.CheckNear(h_vals(IL3), dx3, 0.0, 0.0, "Edge3Length == dx3");

  // (3b) Cartesian cell-center widths are the cell widths (exact).  These are the newdt
  // CFL widths (hydro/mhd_newdt.cpp divide CenterWidth? by the signal speed).
  test.CheckNear(h_vals(ICW1), dx1, 0.0, 0.0, "CenterWidth1 == dx1");
  test.CheckNear(h_vals(ICW2), dx2, 0.0, 0.0, "CenterWidth2 == dx2");
  test.CheckNear(h_vals(ICW3), dx3, 0.0, 0.0, "CenterWidth3 == dx3");

  // (4) Cartesian space is flat: the coordinate-source coefficients vanish.
  test.CheckNear(h_vals(IS1), 0.0, 0.0, 0.0, "CoordSrc1Coeff == 0");
  test.CheckNear(h_vals(IS2), 0.0, 0.0, 0.0, "CoordSrc2Coeff == 0");
  test.CheckNear(h_vals(IS3), 0.0, 0.0, 0.0, "CoordSrc3Coeff == 0");

  // (5) The flux-divergence helpers reduce to (Fr-Fl)/dx, i.e. the slope of a linear
  // flux.  Checked both against the analytic slope and bit-for-bit against the legacy
  // closed form, since this is the equality that keeps the Cartesian update unchanged.
  test.CheckNear(h_vals(ID1), slope, 1.0e-14, 0.0, "FluxDivX1 recovers linear slope");
  test.CheckNear(h_vals(ID2), slope, 1.0e-14, 0.0, "FluxDivX2 recovers linear slope");
  test.CheckNear(h_vals(ID3), slope, 1.0e-14, 0.0, "FluxDivX3 recovers linear slope");
  test.CheckNear(h_vals(ID1), (f1r-f1l)/dx1, 0.0, 0.0, "FluxDivX1 == legacy (Fr-Fl)/dx1");
  test.CheckNear(h_vals(ID2), (f2r-f2l)/dx2, 0.0, 0.0, "FluxDivX2 == legacy (Fr-Fl)/dx2");
  test.CheckNear(h_vals(ID3), (f3r-f3l)/dx3, 0.0, 0.0, "FluxDivX3 == legacy (Fr-Fl)/dx3");

  // ======================================================================================
  // CYLINDRICAL accessors (issue #14): coordinates (x1,x2,x3)=(r,phi,z).  Evaluated with
  // explicit face radii rm,rp and cell-center radius x1v; the analytic factors checked
  // directly (the literal CoordSystem::cylindrical is passed -- no cylindrical mesh).
  const CoordSystem cyl = CoordSystem::cylindrical;
  const Real rm = 2.0, rp = 2.5;            // left/right radial face radii (dr=dx1=0.5)
  const Real x1v = 0.5*(rm + rp);           // arithmetic-mean cell-center radius = 2.25
  const Real half_dr2 = 0.5*(rp*rp - rm*rm);  // 1/2(rp^2-rm^2), the radial volume factor

  enum {CVOL=0, CA1, CA2, CA3, CL2, CCW2, CS1, CS2, CD1, CD2, CDBAL, NCVAL};
  DvceArray1D<Real> d_cvals("coord_geom_cyl_vals", NCVAL);
  auto h_cvals = Kokkos::create_mirror_view(d_cvals);

  // A uniform radial flux F=Cuni on both faces: its conservative divergence must equal
  // Cuni*<1/r> (= Cuni*CoordSrc1Coeff).  This is the static-equilibrium balance property
  // that keeps a uniform-pressure cylindrical state from drifting (issue #14 criterion).
  const Real Cuni = 5.0;

  par_for("coord_geom_cyl_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    d_cvals(CVOL) = CellVolume(cyl, dx1, dx2, dx3, rm, rp);
    d_cvals(CA1)  = Face1Area(cyl, dx2, dx3, rp);
    d_cvals(CA2)  = Face2Area(cyl, dx1, dx3);
    d_cvals(CA3)  = Face3Area(cyl, dx1, dx2, rm, rp);
    d_cvals(CL2)  = Edge2Length(cyl, dx2, rp);
    d_cvals(CCW2) = CenterWidth2(cyl, dx2, x1v);
    d_cvals(CS1)  = CoordSrc1Coeff(cyl, rm, rp);
    d_cvals(CS2)  = CoordSrc2Coeff(cyl, rm, rp);
    d_cvals(CD1)  = FluxDivX1(cyl, f1l, f1r, dx1, rm, rp);
    d_cvals(CD2)  = FluxDivX2(cyl, f2l, f2r, dx2, x1v);
    // uniform-flux radial divergence == Cuni*<1/r>
    d_cvals(CDBAL) = FluxDivX1(cyl, Cuni, Cuni, dx1, rm, rp)
                     - Cuni*CoordSrc1Coeff(cyl, rm, rp);
  });
  Kokkos::deep_copy(h_cvals, d_cvals);

  // (6) Cylindrical cell volume / face areas are the annular metric factors.
  test.CheckNear(h_cvals(CVOL), half_dr2*dx2*dx3, 1.0e-14, 0.0,
                 "cyl CellVolume == 1/2(rp^2-rm^2)*dx2*dx3");
  test.CheckNear(h_cvals(CA1), rp*dx2*dx3, 1.0e-14, 0.0, "cyl Face1Area == rp*dx2*dx3");
  test.CheckNear(h_cvals(CA2), dx1*dx3, 0.0, 0.0, "cyl Face2Area == dx1*dx3");
  test.CheckNear(h_cvals(CA3), half_dr2*dx2, 1.0e-14, 0.0,
                 "cyl Face3Area == 1/2(rp^2-rm^2)*dx2");

  // (7) Azimuthal edge length r*dphi and the near-axis CFL width x1v*dphi.
  test.CheckNear(h_cvals(CL2), rp*dx2, 1.0e-14, 0.0, "cyl Edge2Length == rp*dx2");
  test.CheckNear(h_cvals(CCW2), x1v*dx2, 1.0e-14, 0.0, "cyl CenterWidth2 == x1v*dx2");

  // (8) Geometric source coefficients: volume-averaged <1/r> and the angular-momentum
  // re-symmetrization factor 2/(rp+rm)^2.
  test.CheckNear(h_cvals(CS1), (rp-rm)/half_dr2, 1.0e-14, 0.0,
                 "cyl CoordSrc1Coeff == (rp-rm)/[1/2(rp^2-rm^2)]");
  test.CheckNear(h_cvals(CS2), 2.0/((rp+rm)*(rp+rm)), 1.0e-14, 0.0,
                 "cyl CoordSrc2Coeff == 2/(rp+rm)^2");

  // (9) Cylindrical flux divergences match the analytic curvilinear forms.
  test.CheckNear(h_cvals(CD1), (rp*f1r - rm*f1l)/half_dr2, 1.0e-13, 0.0,
                 "cyl FluxDivX1 == (rp*Fr-rm*Fl)/[1/2(rp^2-rm^2)]");
  test.CheckNear(h_cvals(CD2), (f2r-f2l)/(x1v*dx2), 1.0e-13, 0.0,
                 "cyl FluxDivX2 == (Fr-Fl)/(x1v*dx2)");

  // (10) Static-equilibrium balance: a uniform radial flux gives divergence = flux*<1/r>,
  // so the pressure flux divergence is cancelled by the radial geometric source.
  test.CheckNear(h_cvals(CDBAL), 0.0, 1.0e-13, 0.0,
                 "cyl uniform-flux FluxDivX1 balances Cuni*CoordSrc1Coeff");

  // Print the summary and exit nonzero if any check failed.
  test.Finish();
  return;
}
