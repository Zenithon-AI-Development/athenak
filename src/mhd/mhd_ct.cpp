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
  // (1/A_face) d(L_edge*E)/dx.  In every implemented coordinate system the bounding edge
  // length is constant along the differentiation direction, so each term reduces to the
  // EMF difference over the cell-edge length in THAT direction (A_face/L_perp = Edge_dx).
  // Dividing by Edge?Length(csys, dx) -- which returns dx on the Cartesian path -- keeps
  // the legacy (beta_dt*dE)/dx arithmetic byte-identical while letting the curvilinear
  // specializations (#16) supply r-weighted edges.

  //---- update B1 (only for 2D/3D problems)
  if (multi_d) {
    auto bx1f = b0.x1f;
    auto bx1f_old = b1.x1f;
    par_for("CT-b1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      bx1f(m,k,j,i) = gam0*bx1f(m,k,j,i) + gam1*bx1f_old(m,k,j,i);
      bx1f(m,k,j,i) -= beta_dt*(e3(m,k,j+1,i) - e3(m,k,j,i))
                       /Edge2Length(csys, mbsize.d_view(m).dx2);
      if (three_d) {
        bx1f(m,k,j,i) += beta_dt*(e2(m,k+1,j,i) - e2(m,k,j,i))
                         /Edge3Length(csys, mbsize.d_view(m).dx3);
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
    bx3f(m,k,j,i) -= beta_dt*(e2(m,k,j,i+1) - e2(m,k,j,i))
                     /Edge1Length(csys, mbsize.d_view(m).dx1);
    if (multi_d) {
      bx3f(m,k,j,i) += beta_dt*(e1(m,k,j+1,i) - e1(m,k,j,i))
                       /Edge2Length(csys, mbsize.d_view(m).dx2);
    }
  });

  return TaskStatus::complete;
}
} // namespace mhd
