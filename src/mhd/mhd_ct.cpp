//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_ct.cpp
//  \brief

// Athena++ headers
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "srcterms/srcterms.hpp"
#include "driver/driver.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn  void MHD::CT
//  \brief Constrained Transport implementation of dB/dt = -Curl(E), where E=-(v X B)
//  To be clear, the edge-centered variable 'efld' stores E = -(v X B).
//  Temporal update uses multi-step SSP integrators, e.g. RK2, RK3

TaskStatus MHD::CT(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // capture class variables for the kernels
  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;

  // The constrained-transport curl dB_n/dt = -(curl E)_n is the conservative
  // (1/A_face) d(L_edge*E)/dx.  For B1/B2 in Cartesian -- and B2 in cylindrical -- the
  // bounding edge length is constant along the differentiation direction, so each term
  // reduces to the EMF difference over the cell-edge length in THAT direction
  // (A_face/L_perp = Edge_dx).  Dividing by Edge?Length(csys, dx) -- which returns dx on
  // the Cartesian path -- keeps the legacy (beta_dt*dE)/dx arithmetic byte-identical.
  //
  // CYLINDRICAL (ADR-0004, issue #16): coordinates (x1,x2,x3)=(r,phi,z).  The r-weighting
  // enters two terms, so those take an explicit csys branch (the Cartesian path is left
  // textually unchanged -> byte-identical):
  //   B1 (B_r, r-face at radius rf): the azimuthal E_z derivative divides by the arc
  //       length rf*dphi (= Edge2Length at rf); the axial E_phi derivative by dz.
  //   B3 (B_z, z-face spanning [rm,rp]): the radial E_phi derivative is the conservative
  //       (1/r) d(r E_phi)/dr = FluxDivX1 (r-weighted edges rm,rp); the azimuthal E_r
  //       derivative divides by the cell-center arc width x1v*dphi (= CenterWidth2).
  // B2 (B_phi, phi-face) carries no r-weighting (edges dr, dz; area dr*dz) -> unchanged.

  //---- update B1 (only for 2D/3D problems)
  if (multi_d) {
    auto bx1f = b0.x1f;
    auto bx1f_old = b1.x1f;
    par_for("CT-b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
      if (csys == CoordSystem::cartesian) {
        bx1f(m,k,j,i) -= beta_dt*(e3(m,k,j+1,i) - e3(m,k,j,i))
                         /Edge2Length(csys, mbsize.d_view(m).dx2);
        if (three_d) {
          bx1f(m,k,j,i) += beta_dt*(e2(m,k+1,j,i) - e2(m,k,j,i))
                           /Edge3Length(csys, mbsize.d_view(m).dx3);
        }
      } else {
        // cylindrical: B_r on the x1-face at radius rf; the azimuthal E_z difference is
        // weighted by the arc length rf*dphi (Edge2Length at rf).  The axis face rf=0 has
        // zero arc length and carries B_r=0 (set by the axis BC), so skip its E_z term.
        Real rf = LeftEdgeX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        Real l2 = Edge2Length(csys, mbsize.d_view(m).dx2, rf);
        if (l2 != 0.0) {
          bx1f(m,k,j,i) -= beta_dt*(e3(m,k,j+1,i) - e3(m,k,j,i))/l2;
        }
        if (three_d) {
          bx1f(m,k,j,i) += beta_dt*(e2(m,k+1,j,i) - e2(m,k,j,i))
                           /Edge3Length(csys, mbsize.d_view(m).dx3);
        }
      }
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  auto bx2f = b0.x2f;
  auto bx2f_old = b1.x2f;
  par_for("CT-b2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bx2f(m,k,j,i) = gam0*bx2f(m,k,j,i) + gam1*bx2f_old(m,k,j,i);
    bx2f(m,k,j,i) += beta_dt*(e3(m,k,j,i+1) - e3(m,k,j,i))
                     /Edge1Length(csys, mbsize.d_view(m).dx1);
    if (three_d) {
      bx2f(m,k,j,i) -= beta_dt*(e1(m,k+1,j,i) - e1(m,k,j,i))
                       /Edge3Length(csys, mbsize.d_view(m).dx3);
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  auto bx3f = b0.x3f;
  auto bx3f_old = b1.x3f;
  par_for("CT-b3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    bx3f(m,k,j,i) = gam0*bx3f(m,k,j,i) + gam1*bx3f_old(m,k,j,i);
    if (csys == CoordSystem::cartesian) {
      bx3f(m,k,j,i) -= beta_dt*(e2(m,k,j,i+1) - e2(m,k,j,i))
                       /Edge1Length(csys, mbsize.d_view(m).dx1);
      if (multi_d) {
        bx3f(m,k,j,i) += beta_dt*(e1(m,k,j+1,i) - e1(m,k,j,i))
                         /Edge2Length(csys, mbsize.d_view(m).dx2);
      }
    } else {
      // cylindrical: B_z on the x3-face spanning [rm,rp], center radius x1v.  The radial
      // E_phi derivative is the conservative (1/r) d(r E_phi)/dr -- exactly FluxDivX1,
      // the r-weighted azimuthal edges rm,rp; the azimuthal E_r derivative divides by the
      // cell-center arc width x1v*dphi (= CenterWidth2).
      Real x1min = mbsize.d_view(m).x1min, x1max = mbsize.d_view(m).x1max;
      Real rm = LeftEdgeX(i-is,   nx1, x1min, x1max);
      Real rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      bx3f(m,k,j,i) -= beta_dt*FluxDivX1(csys, e2(m,k,j,i), e2(m,k,j,i+1),
                                         mbsize.d_view(m).dx1, rm, rp);
      if (multi_d) {
        Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
        bx3f(m,k,j,i) += beta_dt*(e1(m,k,j+1,i) - e1(m,k,j,i))
                         /CenterWidth2(csys, mbsize.d_view(m).dx2, x1v);
      }
    }
  });

  return TaskStatus::complete;
}
} // namespace mhd
