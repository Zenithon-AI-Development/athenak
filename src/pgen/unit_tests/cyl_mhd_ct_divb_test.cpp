//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyl_mhd_ct_divb_test.cpp
//  \brief Unit test: the cylindrical constrained-transport curl preserves div(B)=0.
//
// Cylindrical MHD (ADR-0004, issue #16) updates the staggered magnetic field with the
// r-weighted finite-volume curl of the edge EMFs.  For the scheme to be
// divergence-preserving the curl must be the discrete companion of the finite-volume
// divergence: div(curl E) = 0 for ANY edge field E (to round-off).  This test builds the
// face-centered field B = curl(E) on a small cylindrical grid using EXACTLY the same
// arithmetic as mhd_ct.cpp's cylindrical branch (the r-weighted azimuthal edge lengths
// r*dphi, the conservative (1/r) d(r E_phi)/dr via FluxDivX1, and the cell-center arc
// width x1v*dphi via CenterWidth2), then verifies the cylindrical finite-volume
// divergence
//   div(B) = (1/V)[A1(i+1)B1(i+1) - A1(i)B1(i) + A2(B2(j+1)-B2(j)) + A3(B3(k+1)-B3(k))]
// vanishes to machine precision in every interior cell.  A non-r-weighted CT curl (e.g.
// dividing the radial E_phi derivative by dr instead of the r-weighted edge) leaves an
// O(1) divergence, so this is a strict red->green check of the r-weighting.
//
// Built/run by
//   cmake -D PROBLEM=unit_tests/cyl_mhd_ct_divb_test -B build_unit/cyl_mhd_ct_divb_test
//   (cd build_unit/.../src && make) && ./athena -i inputs/unit_tests/<this>.athinput
// Auto-run by tst/test_suite/unit_tests/test_unit_cyl_mhd_ct_divb_cpu.py.

#include <algorithm>  // std::max
#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// A smooth, O(1) analytic "edge EMF" (vector potential / electric field).  Polynomial in
// (r,phi,z) so the test needs no transcendental device functions; the divergence-of-curl
// identity holds for ANY field, so the precise form is irrelevant -- only that the three
// components vary in all directions and are O(1) over the grid.
KOKKOS_INLINE_FUNCTION
Real EmfR(Real r, Real ph, Real z) {     // r-component (on r-aligned edges)
  return r*ph + 0.5*z*z + 0.3*r*z;
}
KOKKOS_INLINE_FUNCTION
Real EmfPhi(Real r, Real ph, Real z) {    // phi-component (on phi-aligned edges)
  return r*r*ph - 0.3*z + 0.6*ph*z;
}
KOKKOS_INLINE_FUNCTION
Real EmfZ(Real r, Real ph, Real z) {      // z-component (on z-aligned edges)
  return r*ph*z + 0.7*r*r - 0.4*ph*ph;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Cylindrical constrained-transport divergence-preservation unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("cyl_mhd_ct_divb_test");

  // Small, deliberately non-cubic cylindrical grid with r_min > 0 (interior of a pinch).
  const CoordSystem cyl = CoordSystem::cylindrical;
  const int nr = 5, nph = 4, nz = 3;
  const Real r0 = 0.7, dr = 0.3;     // r in [0.7, 2.2]
  const Real ph0 = 0.1, dph = 0.2;   // phi in [0.1, 0.9]
  const Real z0 = 0.2, dz = 0.5;     // z in [0.2, 1.7]

  // Edge-centered EMFs and face-centered fields, indexed [k][j][i].  Allocated one larger
  // than the cell count in every direction so all faces/edges needed by the curl and the
  // divergence are in range; unused over-allocated entries are simply never read.
  DvceArray3D<Real> e1("e1", nz+1, nph+1, nr+1);  // r-aligned edge EMF
  DvceArray3D<Real> e2("e2", nz+1, nph+1, nr+1);  // phi-aligned edge EMF
  DvceArray3D<Real> e3("e3", nz+1, nph+1, nr+1);  // z-aligned edge EMF
  DvceArray3D<Real> b1("b1", nz+1, nph+1, nr+1);  // B_r   on r-faces
  DvceArray3D<Real> b2("b2", nz+1, nph+1, nr+1);  // B_phi on phi-faces
  DvceArray3D<Real> b3("b3", nz+1, nph+1, nr+1);  // B_z   on z-faces
  DvceArray3D<Real> dvb("dvb", nz, nph, nr);      // div(B) per cell

  // (1) Fill the edge EMFs at the physical location of each staggered edge.  An r-aligned
  // edge (e1) spans cell i radially (center radius x1v) at the (phi,z)-face corner; a
  // phi-aligned edge (e2) sits at the r-face radius and z-face, centered in phi; a
  // z-aligned edge (e3) sits at the r-face radius and phi-face, centered in z.
  par_for("fill_edges", DevExeSpace(), 0, nz, 0, nph, 0, nr,
  KOKKOS_LAMBDA(int k, int j, int i) {
    Real rf  = r0 + i*dr;          // r at the i-th radial face
    Real rc  = r0 + (i+0.5)*dr;    // r at the i-th cell center
    Real phf = ph0 + j*dph;        // phi at the j-th face
    Real phc = ph0 + (j+0.5)*dph;  // phi at the j-th cell center
    Real zf  = z0 + k*dz;          // z at the k-th face
    Real zc  = z0 + (k+0.5)*dz;    // z at the k-th cell center
    e1(k,j,i) = EmfR(rc, phf, zf);
    e2(k,j,i) = EmfPhi(rf, phc, zf);
    e3(k,j,i) = EmfZ(rf, phf, zc);
  });

  // (2) Build B = curl(E) on the faces using EXACTLY the cylindrical CT arithmetic of
  // mhd_ct.cpp.  B_r on the r-face at radius rf: the azimuthal (E_z) derivative divides
  // by the arc length rf*dphi (= Edge2Length at rf), the axial (E_phi) derivative by dz.
  par_for("build_b1", DevExeSpace(), 0, nz-1, 0, nph-1, 0, nr,
  KOKKOS_LAMBDA(int k, int j, int i) {
    Real rf = r0 + i*dr;
    Real l2 = Edge2Length(cyl, dph, rf);  // rf*dphi
    Real l3 = Edge3Length(cyl, dz);       // dz
    b1(k,j,i) = -((e3(k,j+1,i) - e3(k,j,i))/l2 - (e2(k+1,j,i) - e2(k,j,i))/l3);
  });

  // B_phi on the phi-face: identical to Cartesian (no r-weighting), E_z derivative over
  // dr and E_r derivative over dz.
  par_for("build_b2", DevExeSpace(), 0, nz-1, 0, nph, 0, nr-1,
  KOKKOS_LAMBDA(int k, int j, int i) {
    Real l1 = Edge1Length(cyl, dr);  // dr
    Real l3 = Edge3Length(cyl, dz);  // dz
    b2(k,j,i) = (e3(k,j,i+1) - e3(k,j,i))/l1 - (e1(k+1,j,i) - e1(k,j,i))/l3;
  });

  // B_z on the z-face spanning [rm,rp] (center radius x1v): the radial (E_phi) derivative
  // is the conservative (1/r) d(r E_phi)/dr = FluxDivX1 (r-weighted edges rm,rp), and the
  // azimuthal (E_r) derivative divides by the center arc width x1v*dphi (CenterWidth2).
  par_for("build_b3", DevExeSpace(), 0, nz, 0, nph-1, 0, nr-1,
  KOKKOS_LAMBDA(int k, int j, int i) {
    Real rm  = r0 + i*dr;
    Real rp  = r0 + (i+1)*dr;
    Real x1v = r0 + (i+0.5)*dr;
    Real cw2 = CenterWidth2(cyl, dph, x1v);  // x1v*dphi
    b3(k,j,i) = -FluxDivX1(cyl, e2(k,j,i), e2(k,j,i+1), dr, rm, rp)
                + (e1(k,j+1,i) - e1(k,j,i))/cw2;
  });

  // (3) Cylindrical finite-volume divergence of B in every cell.  The radial faces carry
  // the radius-dependent area A1 = r*dphi*dz; the axial faces the annular-sector area
  // A3 = 1/2(rp^2-rm^2)*dphi; V the cell volume.  div(curl E) must vanish to round-off.
  par_for("divb", DevExeSpace(), 0, nz-1, 0, nph-1, 0, nr-1,
  KOKKOS_LAMBDA(int k, int j, int i) {
    Real rm = r0 + i*dr;
    Real rp = r0 + (i+1)*dr;
    Real a1m = Face1Area(cyl, dph, dz, rm);     // area of inner r-face
    Real a1p = Face1Area(cyl, dph, dz, rp);     // area of outer r-face
    Real a2  = Face2Area(cyl, dr, dz);          // phi-face area
    Real a3  = Face3Area(cyl, dr, dph, rm, rp); // z-face area
    Real vol = CellVolume(cyl, dr, dph, dz, rm, rp);
    dvb(k,j,i) = (a1p*b1(k,j,i+1) - a1m*b1(k,j,i)
                  + a2*(b2(k,j+1,i) - b2(k,j,i))
                  + a3*(b3(k+1,j,i) - b3(k,j,i)))/vol;
  });

  // (4) Reduce the maximum |div(B)| and the maximum |B| (the natural scale) to the host.
  auto h_dvb = Kokkos::create_mirror_view(dvb);
  Kokkos::deep_copy(h_dvb, dvb);
  auto h_b1 = Kokkos::create_mirror_view(b1);
  auto h_b2 = Kokkos::create_mirror_view(b2);
  auto h_b3 = Kokkos::create_mirror_view(b3);
  Kokkos::deep_copy(h_b1, b1);
  Kokkos::deep_copy(h_b2, b2);
  Kokkos::deep_copy(h_b3, b3);

  Real max_divb = 0.0, max_b = 0.0;
  for (int k=0; k<nz; ++k) {
    for (int j=0; j<nph; ++j) {
      for (int i=0; i<nr; ++i) {
        max_divb = std::max(max_divb, std::fabs(h_dvb(k,j,i)));
      }
    }
  }
  for (int k=0; k<nz; ++k) {
    for (int j=0; j<=nph; ++j) {
      for (int i=0; i<=nr; ++i) {
        max_b = std::max(max_b, std::fabs(h_b1(k,j,i)));
        max_b = std::max(max_b, std::fabs(h_b2(k,j,i)));
        max_b = std::max(max_b, std::fabs(h_b3(k,j,i)));
      }
    }
  }

  // The field components are O(1) (so a broken r-weighting would give O(1) divergence).
  test.CheckTrue(max_b > 0.1, "constructed |B| is O(1) (curl is non-trivial)");

  // The discrete cylindrical curl is divergence-free to machine precision: exactly the
  // property that makes constrained transport preserve div(B)=0 under evolution.
  test.CheckNear(max_divb, 0.0, 0.0, 1.0e-12,
                 "cylindrical CT curl preserves div(B)=0 to machine precision");

  test.Finish();
  return;
}
