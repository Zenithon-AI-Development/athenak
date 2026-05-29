//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_operator.cpp
//! \brief Implements ConductionOperator: isotropic thermal conduction as a
//! parabolic::ParabolicOperator advanced operator-split by RKL2 STS (issue [4b]/#13).

#include <float.h>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "bvals/bvals.hpp"
#include "diffusion/conduction_operator.hpp"

//----------------------------------------------------------------------------------------
//! \brief ConductionOperator constructor.  Allocates the operator's own cell-centered
//! boundary-values object (its per-substage multi-block/MPI ghost exchange) sized to the
//! diffused field's variable width, plus a coarse-mesh scratch array for the
//! restriction/prolongation done at fine/coarse boundaries under SMR/AMR (#108/[A1]).

ConductionOperator::ConductionOperator(MeshBlockPack *pp, ParameterInput *pin,
  const DvceArray5D<Real> &cons, Real kappa, Real gamma) :
  pmy_pack(pp),
  cons_(cons),
  kappa_(kappa),
  gamma_(gamma),
  hflx_("cond_op_hflx", cons.extent_int(0), cons.extent_int(1),
        cons.extent_int(2), cons.extent_int(3), cons.extent_int(4)),
  coarse_("cond_op_coarse", 1, 1, 1, 1, 1) {
  const int nvar = cons.extent_int(1);
  // own boundary-values object: a unique MPI communicator for this operator's exchange,
  // buffers sized to the diffused field's variable width (configurable per operator).
  pbval_ = new MeshBoundaryValuesCC(pp, pin, false);
  pbval_->InitializeBuffers(nvar);
  // coarse-mesh scratch (only meaningful with SMR/AMR; left 1^5 on a uniform grid, where
  // SyncParabolicGhosts never touches it)
  if (pp->pmesh->multilevel) {
    auto &indcs = pp->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_, cons.extent_int(0), nvar, n_ccells3, n_ccells2, n_ccells1);
  }
}

//----------------------------------------------------------------------------------------
//! \brief ConductionOperator destructor: free the owned boundary-values object.

ConductionOperator::~ConductionOperator() {
  delete pbval_;
}

//----------------------------------------------------------------------------------------
//! \fn Real TempFromCons()
//! \brief Gas temperature T = (gamma-1) eint/rho from a conserved cell (frozen
//! background: kinetic energy subtracted, magnetic energy absent in the hydro proof).
KOKKOS_INLINE_FUNCTION
Real TempFromCons(const DvceArray5D<Real> &u, Real gm1,
                  int m, int k, int j, int i) {
  Real rho = u(m,IDN,k,j,i);
  Real ke = 0.5*(SQR(u(m,IM1,k,j,i)) + SQR(u(m,IM2,k,j,i)) + SQR(u(m,IM3,k,j,i)))/rho;
  Real eint = u(m,IEN,k,j,i) - ke;
  return gm1*eint/rho;
}

//----------------------------------------------------------------------------------------
//! \fn void ConductionOperator::OperatorAction()
//! \brief M(u): write the isotropic heat-flux divergence into rhs_out(IEN), 0 elsewhere.

void ConductionOperator::OperatorAction(const DvceArray5D<Real> &u_in,
  DvceArray5D<Real> &rhs_out) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  Real gm1 = gamma_ - 1.0;
  Real kappa = kappa_;
  auto &flx1 = hflx_.x1f;
  auto &flx2 = hflx_.x2f;
  auto &flx3 = hflx_.x3f;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;

  // rhs_out is fully overwritten: zero every conserved component first, so the components
  // this operator does not evolve (everything but IEN) carry M = 0 and the RKL2 recursion
  // leaves them frozen.
  Kokkos::deep_copy(rhs_out, 0.0);

  // x1 heat flux F1 = -kappa dT/dx1 (matches Conduction::IsotropicHeatFlux).
  par_for("cond_op_flx1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real dtempdx = (TempFromCons(u_in, gm1, m, k, j, i)
                  - TempFromCons(u_in, gm1, m, k, j, i-1)) / size.d_view(m).dx1;
    flx1(m,IEN,k,j,i) = -kappa*dtempdx;
  });

  if (!one_d) {
    par_for("cond_op_flx2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtempdx = (TempFromCons(u_in, gm1, m, k, j, i)
                    - TempFromCons(u_in, gm1, m, k, j-1, i)) / size.d_view(m).dx2;
      flx2(m,IEN,k,j,i) = -kappa*dtempdx;
    });
  }
  if (!one_d && !two_d) {
    par_for("cond_op_flx3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real dtempdx = (TempFromCons(u_in, gm1, m, k, j, i)
                    - TempFromCons(u_in, gm1, m, k-1, j, i)) / size.d_view(m).dx3;
      flx3(m,IEN,k,j,i) = -kappa*dtempdx;
    });
  }

  // dE/dt = -div(F): difference the face heat fluxes through the geometry accessors,
  // exactly as the hydro/MHD RKUpdate differences the conserved fluxes.
  par_for("cond_op_div", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real divf = FluxDivX1(csys, flx1(m,IEN,k,j,i), flx1(m,IEN,k,j,i+1),
                          size.d_view(m).dx1);
    if (!one_d) {
      divf += FluxDivX2(csys, flx2(m,IEN,k,j,i), flx2(m,IEN,k,j+1,i),
                        size.d_view(m).dx2);
    }
    if (!one_d && !two_d) {
      divf += FluxDivX3(csys, flx3(m,IEN,k,j,i), flx3(m,IEN,k+1,j,i),
                        size.d_view(m).dx3);
    }
    rhs_out(m,IEN,k,j,i) = -divf;
  });
}

//----------------------------------------------------------------------------------------
//! \fn Real ConductionOperator::ExplicitStableDt()
//! \brief Forward-Euler conduction stability dt = fac * min(dx^2 rho/(kappa (gamma-1))),
//! the same limit Conduction::NewTimeStep imposes (fac = 1/2, 1/4, 1/6 in 1/2/3-D).

Real ConductionOperator::ExplicitStableDt() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;
  auto size = pmy_pack->pmb->mb_size;
  auto &u = cons_;
  Real gm1 = gamma_ - 1.0;
  Real kappa = kappa_;
  Real fac = (three_d) ? (1.0/6.0) : ((multi_d) ? 0.25 : 0.5);

  Real dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Kokkos::parallel_reduce("cond_op_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real rho = u(m,IDN,k,j,i);
    min_dt = fmin(min_dt, SQR(size.d_view(m).dx1)*rho/(kappa*gm1));
    if (multi_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx2)*rho/(kappa*gm1));
    }
    if (three_d) {
      min_dt = fmin(min_dt, SQR(size.d_view(m).dx3)*rho/(kappa*gm1));
    }
  }, Kokkos::Min<Real>(dtnew));

  return fac*dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void ConductionOperator::ApplyBoundary()
//! \brief Refresh the trial state's ghost zones before every M(.) evaluation so the RKL2
//! substeps see a boundary-consistent stencil.  Two steps, in order:
//!  (1) Insulated (zero-gradient) fill of ALL ghost zones: ghost cells copy the nearest
//!      active cell, so dT/dx = 0 across that face => zero heat flux.  This is the
//!      correct PHYSICAL boundary condition (insulated box).
//!  (2) The shared multi-block/MPI/AMR neighbor exchange (SyncParabolicGhosts) then
//!      OVERWRITES the ghosts on every INTERNAL block face (where a neighbor exists) with
//!      the neighbor's real data, leaving only the true domain-boundary ghosts insulated.
//! Doing the insulated fill first and the exchange second is what makes the single global
//! box (split across blocks/ranks) insulated only at its outer edges -- so the discrete
//! cosine eigenmode decays as exp(lambda t) across block boundaries.  On a single block
//! with no neighbors step (2) is a no-op and the behaviour is the original fill.

void ConductionOperator::ApplyBoundary(DvceArray5D<Real> &u) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ng = indcs.ng;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nvar = u.extent_int(1);
  int n3 = u.extent_int(2);
  int n2 = u.extent_int(3);
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;

  // x1 ghosts (always)
  par_for("cond_op_bcx1", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int g) {
    u(m,n,k,j,is-1-g) = u(m,n,k,j,is);
    u(m,n,k,j,ie+1+g) = u(m,n,k,j,ie);
  });
  if (!one_d) {
    int n1 = u.extent_int(4);
    par_for("cond_op_bcx2", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, ng-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int g, const int i) {
      u(m,n,k,js-1-g,i) = u(m,n,k,js,i);
      u(m,n,k,je+1+g,i) = u(m,n,k,je,i);
    });
  }
  if (!one_d && !two_d) {
    int n1 = u.extent_int(4);
    par_for("cond_op_bcx3", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, ng-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int g, const int j, const int i) {
      u(m,n,ks-1-g,j,i) = u(m,n,ks,j,i);
      u(m,n,ke+1+g,j,i) = u(m,n,ke,j,i);
    });
  }

  // (2) overwrite internal block-face ghosts (and coarse/fine boundary ghosts under
  // SMR/AMR) with neighbor data via the shared synchronous exchange.  Untouched at true
  // physical boundaries (no neighbor), so those keep the insulated fill from step (1).
  pbval_->SyncParabolicGhosts(u, coarse_);
}
