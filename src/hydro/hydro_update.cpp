//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_update.cpp
//! \brief Performs explicit update of Hydro conserved variables (u0) for each stage of
//! the SSP RK integrators (e.g. RK1, RK2, RK3) implemented in AthenaK, using weighted
//! average and partial time step update of flux divergence. Source terms are added in
//! the HydroSrcTerms() function.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
#include "hydro.hpp"

namespace hydro {
//----------------------------------------------------------------------------------------
//! \fn  void Hydro::Update
//  \brief Explicit RK update including flux divergence terms

TaskStatus Hydro::RKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  Real &gam0 = pdriver->gam0[stage-1];
  Real &gam1 = pdriver->gam1[stage-1];
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = nhydro + nscalars;
  auto u0_ = u0;
  auto u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  int nx1 = indcs.nx1;

  // hierarchical parallel loop that updates conserved variables to intermediate step
  // using weights and fractional time step appropriate to stages of time-integrator.
  // Vector inner loop used for good performance on cpus
  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1);

  par_for_outer("h_update",DevExeSpace(),scr_size,scr_level,0,nmb1,0,nvar-1,ks,ke,js,je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int j) {
    ScrArray1D<Real> divf(member.team_scratch(scr_level), ncells1);

    // compute dF1/dx1 (geometry-agnostic; Cartesian reduces to (F_{i+1}-F_i)/dx1).  In
    // cylindrical the radial face radii rm,rp weight the flux divergence; they are only
    // computed off the Cartesian path so the Cartesian update stays byte-identical.
    par_for_inner(member, is, ie, [&](const int i) {
      Real rm = 0.0, rp = 0.0;
      if (csys != CoordSystem::cartesian) {
        Real x1min = mbsize.d_view(m).x1min, x1max = mbsize.d_view(m).x1max;
        rm = LeftEdgeX(i-is,   nx1, x1min, x1max);
        rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      }
      divf(i) = FluxDivX1(csys, flx1(m,n,k,j,i), flx1(m,n,k,j,i+1),
                          mbsize.d_view(m).dx1, rm, rp);
    });
    member.team_barrier();

    // Add dF2/dx2
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (multi_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        Real x1v = 0.0;
        if (csys != CoordSystem::cartesian) {
          x1v = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
        }
        divf(i) += FluxDivX2(csys, flx2(m,n,k,j,i), flx2(m,n,k,j+1,i),
                             mbsize.d_view(m).dx2, x1v);
      });
      member.team_barrier();
    }

    // Add dF3/dx3
    // Fluxes must be summed in pairs to symmetrize round-off error in each dir
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        divf(i) += FluxDivX3(csys, flx3(m,n,k,j,i), flx3(m,n,k+1,j,i),
                             mbsize.d_view(m).dx3);
      });
      member.team_barrier();
    }

    par_for_inner(member, is, ie, [&](const int i) {
      u0_(m,n,k,j,i) = gam0*u0_(m,n,k,j,i) + gam1*u1_(m,n,k,j,i) - beta_dt*divf(i);
    });
  });
  return TaskStatus::complete;
}
} // namespace hydro
