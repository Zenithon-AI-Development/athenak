//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_grey_operator.cpp
//! \brief Implements FLDGreyOperator: grey flux-limited radiation diffusion (Larsen
//! limiter) as a parabolic::ParabolicOperator advanced operator-split by RKL2 STS
//! (issue [8a]/#17, ADR-0001).  See fld_grey_operator.hpp for the model and contract.

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
#include "radiation_fld/fld_grey_operator.hpp"

using radiationfld::LarsenLimiter;
using radiationfld::FaceDeff;

//----------------------------------------------------------------------------------------
//! \brief FLDGreyOperator constructor.  Allocates the operator's own cell-centered
//! boundary-values object (its per-substage multi-block/MPI ghost exchange) sized to the
//! diffused radiation field's variable width, plus a coarse-mesh scratch array for the
//! restriction/prolongation done at fine/coarse boundaries under SMR/AMR (#108/[A1],
//! #110/[A3]) -- mirrors ConductionOperator.

FLDGreyOperator::FLDGreyOperator(MeshBlockPack *pp, ParameterInput *pin,
  const DvceArray5D<Real> &erad,
  Real c_light, Real chi, Real n_larsen, Real e_source, Real e_floor,
  Real upwind) :
  pmy_pack(pp),
  erad_(erad),
  c_(c_light),
  chi_(chi),
  nlarsen_(n_larsen),
  esrc_(e_source),
  efloor_((e_floor > 0.0) ? e_floor : static_cast<Real>(1.0e-30)),
  upwind_(upwind),
  injected_energy_(0.0),
  irad_(0),
  chi_c_("fld_op_chic", 1, 1, 1, 1, 1),
  rflx_("fld_op_rflx", erad.extent_int(0), erad.extent_int(1),
        erad.extent_int(2), erad.extent_int(3), erad.extent_int(4)),
  pbval_flux_(nullptr),
  pbval_(nullptr),
  coarse_("fld_op_coarse", 1, 1, 1, 1, 1) {
  const int nvar = erad.extent_int(1);
  // own boundary-values object for the per-substage neighbor ghost exchange: a unique MPI
  // communicator for this operator's exchange, buffers sized to the diffused field's
  // variable width (the multigroup operator widens this to batch all groups).
  pbval_ = new MeshBoundaryValuesCC(pp, pin, false);
  pbval_->InitializeBuffers(nvar);
  // coarse-mesh scratch (only meaningful with SMR/AMR; left 1^5 on a uniform grid, where
  // SyncParabolicGhosts never touches it)
  if (pp->pmesh->multilevel) {
    auto &indcs = pp->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_, erad.extent_int(0), nvar, n_ccells3, n_ccells2, n_ccells1);
    // also register the radiative flux for the conservative fine->coarse flux correction
    // (#33), so the diffused radiation energy is conserved across refinement boundaries.
    pbval_flux_ = new MeshBoundaryValuesCC(pp, nullptr, false);
    pbval_flux_->InitializeBuffers(rflx_.x1f.extent_int(1));
  }
}

//----------------------------------------------------------------------------------------
//! \brief FLDGreyOperator destructor: free the owned boundary-values objects.

FLDGreyOperator::~FLDGreyOperator() {
  if (pbval_flux_ != nullptr) { delete pbval_flux_; }
  if (pbval_ != nullptr) { delete pbval_; }
}

//----------------------------------------------------------------------------------------
//! \fn void FLDGreyOperator::OperatorAction()
//! \brief M(u): write the grey FLD flux divergence div(D grad E_r) into rhs_out(irad),
//! 0 elsewhere.  The face diffusivity D = c lambda(R)/chi uses the Larsen limiter of the
//! local face gradient R = |grad E_r|/(chi E_face), so the diffusive flux saturates at
//! the free-streaming limit |F| -> c E in optically-thin cells.

void FLDGreyOperator::OperatorAction(const DvceArray5D<Real> &u_in,
  DvceArray5D<Real> &rhs_out) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = indcs.nx1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  Real cc = c_, chi = chi_, nl = nlarsen_, efloor = efloor_, upwind = upwind_;
  auto chic = chi_c_;
  const bool lchi = local_chi_;
  int irad = irad_;
  auto &flx1 = rflx_.x1f;
  auto &flx2 = rflx_.x2f;
  auto &flx3 = rflx_.x3f;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;

  // rhs_out is fully overwritten: zero every component first, so the components this
  // operator does not evolve (everything but irad) carry M = 0 and the RKL2 recursion
  // leaves them frozen (operator-split frozen background).
  Kokkos::deep_copy(rhs_out, 0.0);

  // x1 radiative flux F1 = -D dE/dx1, D = c lambda(R)/chi at the i-face.
  par_for("fld_op_flx1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real el = u_in(m,irad,k,j,i-1);
    Real er = u_in(m,irad,k,j,i);
    // per-cell chi (#204): face value = HARMONIC mean of the two cells' extinction, so
    // a thick|thin (liner|vacuum-gap) interface face is dominated by the TRANSPARENT
    // side -- the limiter then sees a large R and the face vents at the streaming-
    // limited surface flux ~c E (the physical surface luminosity), instead of being
    // throttled to the opaque side's diffusive trickle as an arithmetic mean would.
    // For chi_L ~ chi_R (smooth interior) harmonic == arithmetic to O(contrast^2).
    Real chif = lchi ? 2.0*chic(m,0,k,j,i-1)*chic(m,0,k,j,i)
                       /(chic(m,0,k,j,i-1) + chic(m,0,k,j,i)) : chi;
    Real dedx = (er - el)/size.d_view(m).dx1;
    Real eface = 0.5*(el + er);
    Real R = Kokkos::fabs(dedx)/(chif*Kokkos::fmax(eface, efloor));
    Real lam = LarsenLimiter(R, nl);
    Real D = cc*lam/chif;
    // LF streaming dissipation (#194): signal speed cc*lam*R (-> c streaming) GATED by
    // the advective fraction (1-3 lam)_+ (0 thick where lam=1/3, -> 1 streaming) so it
    // vanishes as ~R^2 where the scheme is already a stable diffusion (battery B)
    // and damps the centered-advection (imaginary) mode RKL2 cannot at a sharp front.
    Real alf = cc*lam*R*Kokkos::fmax(0.0, 1.0 - 3.0*lam);
    flx1(m,irad,k,j,i) = -D*dedx - 0.5*upwind*alf*(er - el);
  });

  if (!one_d) {
    par_for("fld_op_flx2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real el = u_in(m,irad,k,j-1,i);
      Real er = u_in(m,irad,k,j,i);
      // azimuthal gradient uses the physical arc length (cyl CenterWidth2 = x1v*dx2,
      // Cartesian = dx2 -> bit-identical), so the cross-axis phi diffusion is geometric.
      Real x1v2 = 0.0;
      if (csys == CoordSystem::cylindrical) {
        x1v2 = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real chif = lchi ? 2.0*chic(m,0,k,j-1,i)*chic(m,0,k,j,i)
                         /(chic(m,0,k,j-1,i) + chic(m,0,k,j,i)) : chi;  // #204 harmonic
      Real dedx = (er - el)/CenterWidth2(csys, size.d_view(m).dx2, x1v2);
      Real eface = 0.5*(el + er);
      Real R = Kokkos::fabs(dedx)/(chif*Kokkos::fmax(eface, efloor));
      Real lam = LarsenLimiter(R, nl);
      Real D = cc*lam/chif;
      Real alf = cc*lam*R*Kokkos::fmax(0.0, 1.0 - 3.0*lam);  // gated LF (#194)
      flx2(m,irad,k,j,i) = -D*dedx - 0.5*upwind*alf*(er - el);
    });
  }
  if (!one_d && !two_d) {
    par_for("fld_op_flx3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real el = u_in(m,irad,k-1,j,i);
      Real er = u_in(m,irad,k,j,i);
      Real chif = lchi ? 2.0*chic(m,0,k-1,j,i)*chic(m,0,k,j,i)
                         /(chic(m,0,k-1,j,i) + chic(m,0,k,j,i)) : chi;  // #204 harmonic
      Real dedx = (er - el)/size.d_view(m).dx3;
      Real eface = 0.5*(el + er);
      Real R = Kokkos::fabs(dedx)/(chif*Kokkos::fmax(eface, efloor));
      Real lam = LarsenLimiter(R, nl);
      Real D = cc*lam/chif;
      Real alf = cc*lam*R*Kokkos::fmax(0.0, 1.0 - 3.0*lam);  // gated LF (#194)
      flx3(m,irad,k,j,i) = -D*dedx - 0.5*upwind*alf*(er - el);
    });
  }

  // conservative fine->coarse flux correction at SMR/AMR level boundaries (#33): replace
  // the coarse-side boundary face flux with the restricted (area-averaged) fine flux so
  // the coarse divergence matches the sum of the fine faces.  No-op on a uniform mesh.
  if (pbval_flux_ != nullptr) { pbval_flux_->CorrectFlux(rflx_); }

  // dE/dt = -div(F): difference the face fluxes through the geometry accessors, exactly
  // as the hydro/MHD RKUpdate and ConductionOperator do.
  par_for("fld_op_div", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // cylindrical face radii (rm,rp) / cell radius (x1v) feed the curvilinear flux
    // divergence (1/r) d(r F)/dr and (1/r) dF/dphi; default 0 -> Cartesian (fr-fl)/dx,
    // byte-identical.  Required for the operator to run on the cylindrical MagLIF mesh
    // (#116/[B3]) -- without them FluxDivX1 evaluates 0/0 at every cell (NaN).
    Real rm = 0.0, rp = 0.0, x1v = 0.0;
    if (csys == CoordSystem::cylindrical) {
      Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      rm  = LeftEdgeX(i-is,    nx1, x1min, x1max);
      rp  = LeftEdgeX(i+1-is,  nx1, x1min, x1max);
      x1v = CellCenterX(i-is,  nx1, x1min, x1max);
    }
    Real divf = FluxDivX1(csys, flx1(m,irad,k,j,i), flx1(m,irad,k,j,i+1),
                          size.d_view(m).dx1, rm, rp);
    if (!one_d) {
      divf += FluxDivX2(csys, flx2(m,irad,k,j,i), flx2(m,irad,k,j+1,i),
                        size.d_view(m).dx2, x1v);
    }
    if (!one_d && !two_d) {
      divf += FluxDivX3(csys, flx3(m,irad,k,j,i), flx3(m,irad,k+1,j,i),
                        size.d_view(m).dx3);
    }
    rhs_out(m,irad,k,j,i) = -divf;
  });
}

//----------------------------------------------------------------------------------------
//! \fn Real FLDGreyOperator::ExplicitStableDt()
//! \brief Forward-Euler stability dt for the flux-limited radiation diffusion of the
//! current field: dt = 1/(2 D_max (1/dx1^2 + 1/dx2^2 + 1/dx3^2)), with the cell-centred
//! diffusivity D = c lambda(R)/chi from the Larsen limiter.  Because lambda <= 1/3 the
//! diffusive D is bounded by c/(3 chi) in flat (thick) cells and saturates to the
//! free-streaming D = c E/|grad E| ~ c dx in steep (thin) cells -- so dt naturally
//! recovers the c-CFL limit dt ~ dx/c the limiter imposes there (ADR-0001).

Real FLDGreyOperator::ExplicitStableDt() {
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
  auto &u = erad_;
  Real cc = c_, chi = chi_, nl = nlarsen_, efloor = efloor_;
  Real upwind = upwind_;
  auto chic = chi_c_;
  const bool lchi = local_chi_;
  auto csys = pmy_pack->pcoord->coord_system;
  int irad = irad_;

  Real dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Kokkos::parallel_reduce("fld_op_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    // face-consistent forward-Euler bound (#194): the effective diffusivity D_eff
    // (flux-limited D + the LF streaming viscosity) is built per face from the SAME
    // el/er/eface/R the flux kernel uses, summed over the two faces per active dim:
    //   dt <= 1 / sum_dim (D_eff_minus + D_eff_plus)/w^2.
    // In a flat thick cell D_eff_minus=D_eff_plus=D and this reduces to 1/(2 D inv) (the
    // old estimate); at a streaming front it recovers ~w/(2c), which the old cell-centred
    // central-difference estimate over-counted (a floored cell beside a bright neighbour
    // reported a near-zero D -> too-large dt -> RKL2 under-staged -> runaway).
    Real e0 = u(m,irad,k,j,i);
    Real chi0 = lchi ? chic(m,0,k,j,i) : chi;
    Real w1 = size.d_view(m).dx1;
    // per-face HARMONIC chi (#204), same as the flux kernels (face-consistent dt).
    Real chm = lchi ? 2.0*chic(m,0,k,j,i-1)*chi0/(chic(m,0,k,j,i-1) + chi0) : chi;
    Real chp = lchi ? 2.0*chi0*chic(m,0,k,j,i+1)/(chi0 + chic(m,0,k,j,i+1)) : chi;
    Real Dm = FaceDeff(u(m,irad,k,j,i-1), e0, w1, cc, chm, nl, efloor, upwind);
    Real Dp = FaceDeff(e0, u(m,irad,k,j,i+1), w1, cc, chp, nl, efloor, upwind);
    Real invsum = (Dm + Dp)/SQR(w1);
    if (multi_d) {
      Real x1v = 0.0;
      if (csys == CoordSystem::cylindrical) {
        x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real w2 = CenterWidth2(csys, size.d_view(m).dx2, x1v);
      Real chm2 = lchi ? 2.0*chic(m,0,k,j-1,i)*chi0/(chic(m,0,k,j-1,i) + chi0) : chi;
      Real chp2 = lchi ? 2.0*chi0*chic(m,0,k,j+1,i)/(chi0 + chic(m,0,k,j+1,i)) : chi;
      Real Dm2 = FaceDeff(u(m,irad,k,j-1,i), e0, w2, cc, chm2, nl, efloor, upwind);
      Real Dp2 = FaceDeff(e0, u(m,irad,k,j+1,i), w2, cc, chp2, nl, efloor, upwind);
      invsum += (Dm2 + Dp2)/SQR(w2);
    }
    if (three_d) {
      Real w3 = size.d_view(m).dx3;
      Real chm3 = lchi ? 2.0*chic(m,0,k-1,j,i)*chi0/(chic(m,0,k-1,j,i) + chi0) : chi;
      Real chp3 = lchi ? 2.0*chi0*chic(m,0,k+1,j,i)/(chi0 + chic(m,0,k+1,j,i)) : chi;
      Real Dm3 = FaceDeff(u(m,irad,k-1,j,i), e0, w3, cc, chm3, nl, efloor, upwind);
      Real Dp3 = FaceDeff(e0, u(m,irad,k+1,j,i), w3, cc, chp3, nl, efloor, upwind);
      invsum += (Dm3 + Dp3)/SQR(w3);
    }
    if (invsum > 0.0) { min_dt = fmin(min_dt, 1.0/invsum); }
  }, Kokkos::Min<Real>(dtnew));

  return dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void FLDGreyOperator::ApplyBoundary()
//! \brief Refresh the trial state's ghost zones before every M(.) evaluation.  Two steps,
//! in order (mirrors ConductionOperator, #108/[A1]):
//!  (1) PHYSICAL-BC fill of ALL ghost zones: DIRICHLET radiation source at the inner-x1
//!      face (E_r(x1min) = esrc_ via ghost = 2 esrc - E[is], so the FACE value is esrc),
//!      and ZERO-GRADIENT (insulated/outflow) everywhere else.  With esrc_ < 0 the
//!      inner-x1 face is zero-gradient too (an insulated box).
//!  (2) The shared multi-block/MPI/AMR neighbor exchange (SyncParabolicGhosts) then
//!      OVERWRITES the ghosts on every INTERNAL block (and coarse/fine) face with the
//!      neighbor's real data, leaving only the true domain-boundary ghosts at their
//!      physical value.  This is what makes the inner-x1 Dirichlet source apply only at
//!      the GLOBAL x1min face -- not at every block's left edge -- so a block-decomposed
//!      run reproduces the single global Marshak wave.  On a single block with no
//!      neighbors step (2) is a no-op and the behaviour is the original fill.

void FLDGreyOperator::ApplyBoundary(DvceArray5D<Real> &u) {
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
  Real esrc = esrc_;
  int irad = irad_;

  // x1 ghosts (always): zero-gradient at outer-x1; inner-x1 Dirichlet source if esrc>=0.
  par_for("fld_op_bcx1", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int g) {
    if (n == irad && esrc >= 0.0) {
      // Dirichlet: ghost reflects the active cell about the source value so the inner-x1
      // FACE value is esrc (linear extrapolation).
      u(m,n,k,j,is-1-g) = 2.0*esrc - u(m,n,k,j,is);
    } else {
      u(m,n,k,j,is-1-g) = u(m,n,k,j,is);
    }
    u(m,n,k,j,ie+1+g) = u(m,n,k,j,ie);
  });
  if (!one_d) {
    int n1 = u.extent_int(4);
    par_for("fld_op_bcx2", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, ng-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int g, const int i) {
      u(m,n,k,js-1-g,i) = u(m,n,k,js,i);
      u(m,n,k,je+1+g,i) = u(m,n,k,je,i);
    });
  }
  if (!one_d && !two_d) {
    int n1 = u.extent_int(4);
    par_for("fld_op_bcx3", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, ng-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int g, const int j, const int i) {
      u(m,n,ks-1-g,j,i) = u(m,n,ks,j,i);
      u(m,n,ke+1+g,j,i) = u(m,n,ke,j,i);
    });
  }

  // (2) overwrite internal block-face ghosts (and coarse/fine boundary ghosts under
  // SMR/AMR) with neighbor data via the shared synchronous exchange.  Untouched at true
  // physical boundaries (no neighbor), so those keep the physical fill from step (1).
  pbval_->SyncParabolicGhosts(u, coarse_);
}

//----------------------------------------------------------------------------------------
//! \fn void FLDGreyOperator::PostSuperstepProject()
//! \brief Positivity projection run once after each RKL2 super-step (#194): floor the
//! committed radiation energy to efloor_ over active cells and accumulate the energy
//! added by the floor (Sum of max(efloor-erad,0)*cell_volume) so the injection stays
//! bounded and accounted.  Without it the free-streaming / transition-regime FLD operator
//! (advective under RKL2) overshoots erad negative and the divide-guard cascades to NaN
//! in the coupled run.
void FLDGreyOperator::PostSuperstepProject(DvceArray5D<Real> &u) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;
  Real efloor = efloor_;
  int irad = irad_;

  Real injected = 0.0;
  Kokkos::parallel_reduce("fld_op_project",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &linj) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real e = u(m,irad,k,j,i);
    Real deficit = efloor - e;
    if (deficit > 0.0) {
      Real dx2 = one_d ? 1.0 : size.d_view(m).dx2;
      Real dx3 = (one_d || two_d) ? 1.0 : size.d_view(m).dx3;
      Real vol;
      if (csys == CoordSystem::cylindrical) {
        Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
        Real rm = LeftEdgeX(i-is,   nx1, x1min, x1max);
        Real rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
        vol = 0.5*(rp*rp - rm*rm)*dx2*dx3;
      } else {
        vol = size.d_view(m).dx1*dx2*dx3;
      }
      linj += deficit*vol;
      u(m,irad,k,j,i) = efloor;
    }
  }, Kokkos::Sum<Real>(injected));
  injected_energy_ += injected;
}
