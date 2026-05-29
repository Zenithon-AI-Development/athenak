//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_bphi_operator.cpp
//! \brief Implements ResistiveBphiOperator: the special cylindrical resistive B_phi
//! diffusion (the -eta B_phi/r^2 curl-curl operator, conservative (1/r)d_r(r*flux) radial
//! form, antisymmetric axis ghost) as a parabolic::ParabolicOperator advanced operator-
//! split by RKL2 super-time-stepping (issue [7b]/#27, ADR-0004/ADR-0001).  See
//! resistive_bphi_operator.hpp for the model and the axis treatment.

#include <float.h>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "diffusion/resistive_bphi_operator.hpp"

//----------------------------------------------------------------------------------------
//! \brief ResistiveBphiOperator constructor.

ResistiveBphiOperator::ResistiveBphiOperator(MeshBlockPack *pp, ParameterInput *pin,
  const DvceArray5D<Real> &bphi, const DvceArray4D<Real> &eta) :
  pmy_pack(pp),
  bphi_(bphi),
  eta_(eta),
  bflx_("resb_bflx", bphi.extent_int(0), bphi.extent_int(1),
        bphi.extent_int(2), bphi.extent_int(3), bphi.extent_int(4)),
  pbval_(nullptr),
  coarse_("resb_coarse", 1, 1, 1, 1, 1),
  pbval_flux_(nullptr) {
  const int nvar = bphi.extent_int(1);   // single B_phi component
  // own boundary-values object for the per-substage neighbour ghost exchange: unique MPI
  // communicator for this operator's exchange (avoids tag collision with the MHD field's
  // pbval), buffers sized to the single diffused B_phi variable (#108/[A1], #113/[A6]).
  // Only built when `pin` is supplied (the production wiring); legacy single-block-only
  // callers pass nullptr to skip it (then ApplyBoundary applies only the physical fill).
  if (pin != nullptr) {
    pbval_ = new MeshBoundaryValuesCC(pp, pin, false);
    pbval_->InitializeBuffers(nvar);
  }
  // coarse-mesh scratch (only meaningful with SMR/AMR; left 1^5 on a uniform grid, where
  // SyncParabolicGhosts never touches it)
  if (pp->pmesh->multilevel) {
    auto &indcs = pp->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_, bphi.extent_int(0), nvar, n_ccells3, n_ccells2, n_ccells1);
    // On a multilevel (SMR/AMR) mesh, register the resistive flux for the conservative
    // fine->coarse flux correction (#33), so the diffused B_phi (and the area-weighted
    // poloidal flux it carries) is conserved across refinement boundaries.  (pin is
    // unused by the boundary-values constructor.)
    pbval_flux_ = new MeshBoundaryValuesCC(pp, nullptr, false);
    pbval_flux_->InitializeBuffers(bflx_.x1f.extent_int(1));
  }
}

//----------------------------------------------------------------------------------------
//! \brief ResistiveBphiOperator destructor.

ResistiveBphiOperator::~ResistiveBphiOperator() {
  if (pbval_ != nullptr) { delete pbval_; }
  if (pbval_flux_ != nullptr) { delete pbval_flux_; }
}

//----------------------------------------------------------------------------------------
//! \fn void ResistiveBphiOperator::OperatorAction()
//! \brief M(u): write the cylindrical resistive B_phi diffusion into rhs_out(IBPHI), 0
//! elsewhere.  Radial part = conservative (1/r)d_r(r*flux) (FluxDivX1) MINUS the
//! curl-curl -eta B_phi/r^2 term at cell centres; axial (z) part = plain diffusion; the
//! azimuthal (phi) part = cylindrical scalar phi-Laplacian via the r-weighted accessors.

void ResistiveBphiOperator::OperatorAction(const DvceArray5D<Real> &u_in,
  DvceArray5D<Real> &rhs_out) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = indcs.nx1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  auto eta = eta_;
  const int ib = IBPHI;
  auto &flx1 = bflx_.x1f;
  auto &flx2 = bflx_.x2f;
  auto &flx3 = bflx_.x3f;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;

  // rhs_out is fully overwritten: zero every component first, so any component this
  // operator does not evolve carries M = 0 and the RKL2 recursion leaves it frozen.
  Kokkos::deep_copy(rhs_out, 0.0);

  // x1 resistive flux F1 = -eta dB_phi/dr on the i-face (radial; the (1/r)d_r(r*flux)
  // conservative form is applied by FluxDivX1 in the divergence below).
  par_for("resb_flx1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real eta_f = 0.5*(eta(m,k,j,i-1) + eta(m,k,j,i));
    Real dbdx1 = (u_in(m,ib,k,j,i) - u_in(m,ib,k,j,i-1))/size.d_view(m).dx1;
    flx1(m,ib,k,j,i) = -eta_f*dbdx1;
  });

  // x2 resistive flux F2 = -eta (1/r) dB_phi/dphi on the j-face (the physical azimuthal
  // gradient uses the arc length CenterWidth2 = x1v*dx2 at the cell-centre radius).
  if (!one_d) {
    par_for("resb_flx2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x1v = 0.0;
      if (csys != CoordSystem::cartesian) {
        x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real eta_f = 0.5*(eta(m,k,j-1,i) + eta(m,k,j,i));
      Real dbdx2 = (u_in(m,ib,k,j,i) - u_in(m,ib,k,j-1,i))
                 /CenterWidth2(csys, size.d_view(m).dx2, x1v);
      flx2(m,ib,k,j,i) = -eta_f*dbdx2;
    });
  }

  // x3 resistive flux F3 = -eta dB_phi/dz on the k-face (axial; z is flat -> plain).
  if (!one_d && !two_d) {
    par_for("resb_flx3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real eta_f = 0.5*(eta(m,k-1,j,i) + eta(m,k,j,i));
      Real dbdx3 = (u_in(m,ib,k,j,i) - u_in(m,ib,k-1,j,i))/size.d_view(m).dx3;
      flx3(m,ib,k,j,i) = -eta_f*dbdx3;
    });
  }

  // conservative fine->coarse flux correction at SMR/AMR level boundaries (#33): replace
  // the coarse-side boundary face flux with the restricted (area-averaged) fine flux.
  // No-op on a uniform mesh.  A radial (x1) refinement face sits at a fixed radius, so
  // the coarse and fine faces share rface and the bare-flux average matches the r-weight
  // area ratio FluxDivX1 applies below; full curvilinear-face refinement (Balsara
  // prolongation) is ADR-0004's deferred #38.
  if (pbval_flux_ != nullptr) { pbval_flux_->CorrectFlux(bflx_); }

  // dB_phi/dt = -div(F) + the curl-curl -eta B_phi/r^2 term.  The radial divergence is
  // the conservative (1/r)d_r(r F) form (FluxDivX1 with the face radii rm/rp); the rm=0
  // axis face carries zero radial flux automatically.  The -eta B_phi/r^2 term (the
  // genuine cylindrical exception, ADR-0004) is added at the cell centre (r = x1v); it is
  // absent in flat Cartesian space.
  par_for("resb_div", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rm = 0.0, rp = 0.0, x1v = 0.0;
    if (csys != CoordSystem::cartesian) {
      Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      rm  = LeftEdgeX(i-is,   nx1, x1min, x1max);
      rp  = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      x1v = CellCenterX(i-is,  nx1, x1min, x1max);
    }
    Real divf = FluxDivX1(csys, flx1(m,ib,k,j,i), flx1(m,ib,k,j,i+1),
                          size.d_view(m).dx1, rm, rp);
    if (!one_d) {
      divf += FluxDivX2(csys, flx2(m,ib,k,j,i), flx2(m,ib,k,j+1,i),
                        size.d_view(m).dx2, x1v);
    }
    if (!one_d && !two_d) {
      divf += FluxDivX3(csys, flx3(m,ib,k,j,i), flx3(m,ib,k+1,j,i),
                        size.d_view(m).dx3);
    }
    Real rhs = -divf;
    if (csys == CoordSystem::cylindrical) {
      rhs -= eta(m,k,j,i)*u_in(m,ib,k,j,i)/(x1v*x1v);   // -eta B_phi/r^2 (curl-curl)
    }
    rhs_out(m,ib,k,j,i) = rhs;
  });
}

//----------------------------------------------------------------------------------------
//! \fn Real ResistiveBphiOperator::ExplicitStableDt()
//! \brief Forward-Euler resistive stability dt.  The decay rate at a cell is bounded by
//! eta*(2/w1^2 [+2/w2^2] [+2/w3^2]) from the diffusion stencil PLUS eta/r^2 from the
//! curl-curl term (which tightens dt near the axis); dt_exp = min_cell 1/rate (a factor
//! ~2 below the FE limit 2/rate, so the RKL2 stage count is conservative -> stable).

Real ResistiveBphiOperator::ExplicitStableDt() {
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
  auto csys = pmy_pack->pcoord->coord_system;
  auto eta = eta_;

  Real dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Kokkos::parallel_reduce("resb_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real eta_c = eta(m,k,j,i);
    Real x1v = 0.0;
    if (csys != CoordSystem::cartesian) {
      x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    }
    Real w1 = CenterWidth1(csys, size.d_view(m).dx1);
    Real rate = 2.0*eta_c/(w1*w1);
    if (multi_d) {
      Real w2 = CenterWidth2(csys, size.d_view(m).dx2, x1v);
      rate += 2.0*eta_c/(w2*w2);
    }
    if (three_d) {
      Real w3 = CenterWidth3(csys, size.d_view(m).dx3);
      rate += 2.0*eta_c/(w3*w3);
    }
    if (csys == CoordSystem::cylindrical) {
      rate += eta_c/(x1v*x1v);
    }
    if (rate > 0.0) {
      min_dt = fmin(min_dt, 1.0/rate);
    }
  }, Kokkos::Min<Real>(dtnew));

  return dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void ResistiveBphiOperator::ApplyBoundary()
//! \brief Antisymmetric ghost B_phi(i-1) = -B_phi(i) at BOTH radial ends: at inner_x1
//! this is the r=0 axisymmetric axis ghost (B_phi is odd across the axis); at outer_x1 it
//! realizes a Dirichlet B_phi=0 at the outer face (the node of the J_1 verification mode,
//! and a sensible "no field at the outer radius" default until the circuit-driven B_phi
//! BC (#21/#29) replaces it in a wired run).  x2/x3 use a zero-gradient (copy nearest).

void ResistiveBphiOperator::ApplyBoundary(DvceArray5D<Real> &u) {
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

  // x1 ghosts: antisymmetric at both ends (axis at inner; Dirichlet-0 node at outer).
  par_for("resb_bcx1", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int g) {
    u(m,n,k,j,is-1-g) = -u(m,n,k,j,is+g);
    u(m,n,k,j,ie+1+g) = -u(m,n,k,j,ie-g);
  });
  if (!one_d) {
    int n1 = u.extent_int(4);
    par_for("resb_bcx2", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, ng-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int g, const int i) {
      u(m,n,k,js-1-g,i) = u(m,n,k,js,i);
      u(m,n,k,je+1+g,i) = u(m,n,k,je,i);
    });
  }
  if (!one_d && !two_d) {
    int n1 = u.extent_int(4);
    par_for("resb_bcx3", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, ng-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int g, const int j, const int i) {
      u(m,n,ks-1-g,j,i) = u(m,n,ks,j,i);
      u(m,n,ke+1+g,j,i) = u(m,n,ke,j,i);
    });
  }

  // After the PHYSICAL fill above, OVERWRITE the ghosts on every INTERNAL block face (and
  // the coarse/fine boundary ghosts under SMR/AMR) with the neighbour's real B_phi via
  // the shared synchronous exchange (#108/[A1], wired by #113/[A6]).  Untouched at true
  // radial domain faces (no neighbour), so the axis keeps its antisymmetric ghost and the
  // outer radius its Dirichlet-0 ghost -- a radially decomposed column behaves as one
  // global cylinder.  Skipped (no-op) with pin==nullptr (legacy single-block run).
  if (pbval_ != nullptr) { pbval_->SyncParabolicGhosts(u, coarse_); }
}
