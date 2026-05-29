//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file aniso_conduction_operator.cpp
//! \brief Implements AnisotropicConductionOperator: field-aligned Braginskii electron+ion
//! thermal conduction as a parabolic::ParabolicOperator advanced operator-split by RKL2
//! (issue [6a]/#18, ADR-0006/ADR-0001).  See aniso_conduction_operator.hpp for the model,
//! limiters (Sharma-Hammett + Larsen) and unit convention.

#include <float.h>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "diffusion/aniso_conduction_operator.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real TempMHD()
//! \brief Gas temperature T = (gamma-1) eint/rho from an MHD conserved cell, subtracting
//! BOTH kinetic and (cell-centred) magnetic energy from the total energy.
KOKKOS_INLINE_FUNCTION
Real TempMHD(const DvceArray5D<Real> &u, const DvceArray5D<Real> &bcc, Real gm1,
             int m, int k, int j, int i) {
  Real rho = u(m,IDN,k,j,i);
  Real ke = 0.5*(SQR(u(m,IM1,k,j,i)) + SQR(u(m,IM2,k,j,i)) + SQR(u(m,IM3,k,j,i)))/rho;
  Real me = 0.5*(SQR(bcc(m,IBX,k,j,i)) + SQR(bcc(m,IBY,k,j,i)) + SQR(bcc(m,IBZ,k,j,i)));
  Real eint = u(m,IEN,k,j,i) - ke - me;
  return gm1*eint/rho;
}

//----------------------------------------------------------------------------------------
//! \fn bool InsulatingBdry()
//! \brief True iff a MeshBlock face is an INSULATING domain boundary -- i.e. it has no
//! neighbour (not a `block` internal face) and is not connected to one across the domain
//! (not `periodic`/`shear_periodic`).  At such faces the anisotropic boundary flux is
//! zeroed (insulated box); at internal/periodic faces it is left intact so heat conducts
//! across the block boundary after the SyncParabolicGhosts exchange (#112/[A5]).
KOKKOS_INLINE_FUNCTION
bool InsulatingBdry(BoundaryFlag flag) {
  return (flag != BoundaryFlag::block && flag != BoundaryFlag::periodic
          && flag != BoundaryFlag::shear_periodic && flag != BoundaryFlag::undef);
}

//----------------------------------------------------------------------------------------
//! \brief AnisotropicConductionOperator constructor.

AnisotropicConductionOperator::AnisotropicConductionOperator(MeshBlockPack *pp,
  ParameterInput *pin, const DvceArray5D<Real> &cons, const DvceArray5D<Real> &bcc,
  Real gamma, const anisocond::AnisoCondParams &par) :
  pmy_pack(pp),
  cons_(cons),
  bcc_(bcc),
  gamma_(gamma),
  par_(par),
  hflx_("aniso_cond_hflx", cons.extent_int(0), cons.extent_int(1),
        cons.extent_int(2), cons.extent_int(3), cons.extent_int(4)),
  pbval_(nullptr),
  coarse_("aniso_cond_coarse", 1, 1, 1, 1, 1),
  pbval_flux_(nullptr) {
  const int nvar = cons.extent_int(1);
  // own boundary-values object for the per-substage neighbour ghost exchange: unique MPI
  // communicator for this operator's exchange, buffers sized to the diffused field's
  // variable width (#112/[A5]).  Only built when `pin` is supplied (the production
  // wiring); legacy single-block-only callers may pass nullptr to skip it.
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
    Kokkos::realloc(coarse_, cons.extent_int(0), nvar, n_ccells3, n_ccells2, n_ccells1);
    // On a multilevel (SMR/AMR) mesh, register the heat flux for the conservative
    // fine->coarse flux correction (#33), so conducted energy conserves across refinement
    // boundaries.  (pin is unused by the boundary-values constructor.)
    pbval_flux_ = new MeshBoundaryValuesCC(pp, nullptr, false);
    pbval_flux_->InitializeBuffers(hflx_.x1f.extent_int(1));
  }
}

//----------------------------------------------------------------------------------------
//! \brief AnisotropicConductionOperator destructor.

AnisotropicConductionOperator::~AnisotropicConductionOperator() {
  if (pbval_ != nullptr) { delete pbval_; }
  if (pbval_flux_ != nullptr) { delete pbval_flux_; }
}

//----------------------------------------------------------------------------------------
//! \fn void AnisotropicConductionOperator::OperatorAction()
//! \brief M(u): write anisotropic heat-flux divergence into rhs_out(IEN), 0 elsewhere.
//! Per face: the face-normal dT is the direct two-point difference; the transverse dT are
//! Sharma-Hammett van-Leer-limited (VL4) so heat cannot leak across field lines; the
//! field-aligned flux is capped at the free-streaming limit by the Larsen limiter.

void AnisotropicConductionOperator::OperatorAction(const DvceArray5D<Real> &u_in,
  DvceArray5D<Real> &rhs_out) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nx1 = indcs.nx1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  Real gm1 = gamma_ - 1.0;
  auto par = par_;
  auto bcc = bcc_;
  auto &flx1 = hflx_.x1f;
  auto &flx2 = hflx_.x2f;
  auto &flx3 = hflx_.x3f;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;
  const Real tiny = 1.0e-30;

  // rhs_out is fully overwritten: zero every conserved component first, so the components
  // this operator does not evolve (everything but IEN) carry M = 0 and the RKL2 recursion
  // leaves them frozen (operator-split frozen background).
  Kokkos::deep_copy(rhs_out, 0.0);

  // device closure: field-aligned flux component along direction `dir` (1,2,3) at a face,
  // given the face-normal derivative `dndir` and the two transverse derivatives, the
  // face-averaged state (rho_f,T_f) and the face-averaged B components.
  // x1 heat flux on the i-face.
  par_for("aniso_flx1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // physical phi-gradient at the i-face uses the arc length at the FACE radius rface
    // (Edge2Length = rface*dx2 in cylindrical; = dx2 in Cartesian -> byte-identical).
    Real rface = 0.0;
    if (csys != CoordSystem::cartesian) {
      rface = LeftEdgeX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    }
    Real tl = TempMHD(u_in, bcc, gm1, m, k, j, i-1);
    Real tr = TempMHD(u_in, bcc, gm1, m, k, j, i);
    Real dtdx1 = (tr - tl)/size.d_view(m).dx1;
    Real dtdx2 = 0.0, dtdx3 = 0.0;
    if (!one_d) {
      Real fr = TempMHD(u_in,bcc,gm1,m,k,j+1,i)   - tr;
      Real br = tr - TempMHD(u_in,bcc,gm1,m,k,j-1,i);
      Real fl = TempMHD(u_in,bcc,gm1,m,k,j+1,i-1) - tl;
      Real bl = tl - TempMHD(u_in,bcc,gm1,m,k,j-1,i-1);
      dtdx2 = anisocond::VL4(fr, br, fl, bl)/Edge2Length(csys, size.d_view(m).dx2, rface);
    }
    if (!one_d && !two_d) {
      Real fr = TempMHD(u_in,bcc,gm1,m,k+1,j,i)   - tr;
      Real br = tr - TempMHD(u_in,bcc,gm1,m,k-1,j,i);
      Real fl = TempMHD(u_in,bcc,gm1,m,k+1,j,i-1) - tl;
      Real bl = tl - TempMHD(u_in,bcc,gm1,m,k-1,j,i-1);
      dtdx3 = anisocond::VL4(fr, br, fl, bl)/size.d_view(m).dx3;
    }
    Real bx = 0.5*(bcc(m,IBX,k,j,i-1) + bcc(m,IBX,k,j,i));
    Real by = 0.5*(bcc(m,IBY,k,j,i-1) + bcc(m,IBY,k,j,i));
    Real bz = 0.5*(bcc(m,IBZ,k,j,i-1) + bcc(m,IBZ,k,j,i));
    Real bmag = Kokkos::sqrt(bx*bx + by*by + bz*bz);
    Real rho_f = 0.5*(u_in(m,IDN,k,j,i-1) + u_in(m,IDN,k,j,i));
    Real t_f = 0.5*(tl + tr);
    Real kpar, kperp;
    anisocond::Conductivities(par, rho_f, t_f, bmag, kpar, kperp);
    Real ib = (bmag > tiny) ? 1.0/bmag : 0.0;
    Real hx = bx*ib, hy = by*ib, hz = bz*ib;
    Real bdg = hx*dtdx1 + hy*dtdx2 + hz*dtdx3;          // bhat . grad T
    Real kpar_eff = kpar;
    if (par.vfs > 0.0) {
      Real R = 3.0*kpar*Kokkos::fabs(bdg)/(par.vfs*Kokkos::fmax(t_f, tiny));
      kpar_eff = 3.0*kpar*anisocond::LarsenLimiter(R, par.nlarsen);
    }
    flx1(m,IEN,k,j,i) = -(kperp*dtdx1 + (kpar_eff - kperp)*hx*bdg);
  });

  if (!one_d) {
    par_for("aniso_flx2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // physical phi-gradient on the j-face uses the arc length at the cell-center radius
      // x1v (CenterWidth2 = x1v*dx2 in cylindrical; = dx2 in Cartesian -> bit-identical).
      Real x1v = 0.0;
      if (csys != CoordSystem::cartesian) {
        x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real tl = TempMHD(u_in, bcc, gm1, m, k, j-1, i);
      Real tr = TempMHD(u_in, bcc, gm1, m, k, j, i);
      Real dtdx2 = (tr - tl)/CenterWidth2(csys, size.d_view(m).dx2, x1v);
      Real fr = TempMHD(u_in,bcc,gm1,m,k,j,i+1)   - tr;
      Real br = tr - TempMHD(u_in,bcc,gm1,m,k,j,i-1);
      Real fl = TempMHD(u_in,bcc,gm1,m,k,j-1,i+1) - tl;
      Real bl = tl - TempMHD(u_in,bcc,gm1,m,k,j-1,i-1);
      Real dtdx1 = anisocond::VL4(fr, br, fl, bl)/size.d_view(m).dx1;
      Real dtdx3 = 0.0;
      if (!two_d) {
        Real fr3 = TempMHD(u_in,bcc,gm1,m,k+1,j,i)   - tr;
        Real br3 = tr - TempMHD(u_in,bcc,gm1,m,k-1,j,i);
        Real fl3 = TempMHD(u_in,bcc,gm1,m,k+1,j-1,i) - tl;
        Real bl3 = tl - TempMHD(u_in,bcc,gm1,m,k-1,j-1,i);
        dtdx3 = anisocond::VL4(fr3, br3, fl3, bl3)/size.d_view(m).dx3;
      }
      Real bx = 0.5*(bcc(m,IBX,k,j-1,i) + bcc(m,IBX,k,j,i));
      Real by = 0.5*(bcc(m,IBY,k,j-1,i) + bcc(m,IBY,k,j,i));
      Real bz = 0.5*(bcc(m,IBZ,k,j-1,i) + bcc(m,IBZ,k,j,i));
      Real bmag = Kokkos::sqrt(bx*bx + by*by + bz*bz);
      Real rho_f = 0.5*(u_in(m,IDN,k,j-1,i) + u_in(m,IDN,k,j,i));
      Real t_f = 0.5*(tl + tr);
      Real kpar, kperp;
      anisocond::Conductivities(par, rho_f, t_f, bmag, kpar, kperp);
      Real ib = (bmag > tiny) ? 1.0/bmag : 0.0;
      Real hx = bx*ib, hy = by*ib, hz = bz*ib;
      Real bdg = hx*dtdx1 + hy*dtdx2 + hz*dtdx3;
      Real kpar_eff = kpar;
      if (par.vfs > 0.0) {
        Real R = 3.0*kpar*Kokkos::fabs(bdg)/(par.vfs*Kokkos::fmax(t_f, tiny));
        kpar_eff = 3.0*kpar*anisocond::LarsenLimiter(R, par.nlarsen);
      }
      flx2(m,IEN,k,j,i) = -(kperp*dtdx2 + (kpar_eff - kperp)*hy*bdg);
    });
  }
  if (!one_d && !two_d) {
    par_for("aniso_flx3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      // physical phi-gradient on the k-face uses the arc length at the cell-center radius
      // x1v (CenterWidth2 = x1v*dx2 in cylindrical; = dx2 in Cartesian -> bit-identical).
      Real x1v = 0.0;
      if (csys != CoordSystem::cartesian) {
        x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real tl = TempMHD(u_in, bcc, gm1, m, k-1, j, i);
      Real tr = TempMHD(u_in, bcc, gm1, m, k, j, i);
      Real dtdx3 = (tr - tl)/size.d_view(m).dx3;
      Real fr1 = TempMHD(u_in,bcc,gm1,m,k,j,i+1)   - tr;
      Real br1 = tr - TempMHD(u_in,bcc,gm1,m,k,j,i-1);
      Real fl1 = TempMHD(u_in,bcc,gm1,m,k-1,j,i+1) - tl;
      Real bl1 = tl - TempMHD(u_in,bcc,gm1,m,k-1,j,i-1);
      Real dtdx1 = anisocond::VL4(fr1, br1, fl1, bl1)/size.d_view(m).dx1;
      Real fr2 = TempMHD(u_in,bcc,gm1,m,k,j+1,i)   - tr;
      Real br2 = tr - TempMHD(u_in,bcc,gm1,m,k,j-1,i);
      Real fl2 = TempMHD(u_in,bcc,gm1,m,k-1,j+1,i) - tl;
      Real bl2 = tl - TempMHD(u_in,bcc,gm1,m,k-1,j-1,i);
      Real dtdx2 = anisocond::VL4(fr2, br2, fl2, bl2)
                 /CenterWidth2(csys, size.d_view(m).dx2, x1v);
      Real bx = 0.5*(bcc(m,IBX,k-1,j,i) + bcc(m,IBX,k,j,i));
      Real by = 0.5*(bcc(m,IBY,k-1,j,i) + bcc(m,IBY,k,j,i));
      Real bz = 0.5*(bcc(m,IBZ,k-1,j,i) + bcc(m,IBZ,k,j,i));
      Real bmag = Kokkos::sqrt(bx*bx + by*by + bz*bz);
      Real rho_f = 0.5*(u_in(m,IDN,k-1,j,i) + u_in(m,IDN,k,j,i));
      Real t_f = 0.5*(tl + tr);
      Real kpar, kperp;
      anisocond::Conductivities(par, rho_f, t_f, bmag, kpar, kperp);
      Real ib = (bmag > tiny) ? 1.0/bmag : 0.0;
      Real hx = bx*ib, hy = by*ib, hz = bz*ib;
      Real bdg = hx*dtdx1 + hy*dtdx2 + hz*dtdx3;
      Real kpar_eff = kpar;
      if (par.vfs > 0.0) {
        Real R = 3.0*kpar*Kokkos::fabs(bdg)/(par.vfs*Kokkos::fmax(t_f, tiny));
        kpar_eff = 3.0*kpar*anisocond::LarsenLimiter(R, par.nlarsen);
      }
      flx3(m,IEN,k,j,i) = -(kperp*dtdx3 + (kpar_eff - kperp)*hz*bdg);
    });
  }

  // INSULATED boundary: zero out the heat flux on the true domain-boundary faces ONLY.
  // For anisotropic conduction the parallel flux has a normal component (b_normal != 0)
  // even when the temperature gradient normal to the boundary vanishes (zero-gradient
  // ghost fill), so the ConductionOperator-style ghost-only BC does NOT close the box.
  // Setting the boundary face flux to 0 makes the operator energy-conserving at an
  // insulated outer edge (the "insulated box" the ring run + cosine-eigenmode oracle
  // assume).  On a multi-block / AMR / MPI run only TRUE domain faces are zeroed
  // (InsulatingBdry); internal block faces (`block`) and periodic faces keep the computed
  // flux, so heat conducts across them after the SyncParabolicGhosts neighbour exchange
  // refreshes their ghosts (#112/[A5]).  On a single block every face is a domain
  // boundary, so the behaviour is byte-identical to the previous unconditional zeroing.
  auto mbbcs = pmy_pack->pmb->mb_bcs;
  par_for("aniso_zerobflx1", DevExeSpace(), 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(const int m, const int k, const int j) {
    if (InsulatingBdry(mbbcs.d_view(m,BoundaryFace::inner_x1))) {
      flx1(m,IEN,k,j,is)   = 0.0;
    }
    if (InsulatingBdry(mbbcs.d_view(m,BoundaryFace::outer_x1))) {
      flx1(m,IEN,k,j,ie+1) = 0.0;
    }
  });
  if (!one_d) {
    par_for("aniso_zerobflx2", DevExeSpace(), 0, nmb1, ks, ke, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int i) {
      if (InsulatingBdry(mbbcs.d_view(m,BoundaryFace::inner_x2))) {
        flx2(m,IEN,k,js,i)   = 0.0;
      }
      if (InsulatingBdry(mbbcs.d_view(m,BoundaryFace::outer_x2))) {
        flx2(m,IEN,k,je+1,i) = 0.0;
      }
    });
  }
  if (!one_d && !two_d) {
    par_for("aniso_zerobflx3", DevExeSpace(), 0, nmb1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int j, const int i) {
      if (InsulatingBdry(mbbcs.d_view(m,BoundaryFace::inner_x3))) {
        flx3(m,IEN,ks,j,i)   = 0.0;
      }
      if (InsulatingBdry(mbbcs.d_view(m,BoundaryFace::outer_x3))) {
        flx3(m,IEN,ke+1,j,i) = 0.0;
      }
    });
  }

  // conservative fine->coarse flux correction at SMR/AMR level boundaries (#33): replace
  // the coarse-side boundary face flux with the restricted (area-averaged) fine flux.
  // No-op on a uniform mesh.
  if (pbval_flux_ != nullptr) { pbval_flux_->CorrectFlux(hflx_); }

  // dE/dt = -div(F): difference the face heat fluxes through the geometry accessors,
  // exactly as the hydro/MHD RKUpdate and ConductionOperator do.
  par_for("aniso_div", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // cylindrical radial face radii / cell-center radius weight the conservative flux
    // divergence; computed off the Cartesian path so Cartesian stays byte-identical.
    Real rm = 0.0, rp = 0.0, x1v = 0.0;
    if (csys != CoordSystem::cartesian) {
      Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      rm  = LeftEdgeX(i-is,   nx1, x1min, x1max);
      rp  = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      x1v = CellCenterX(i-is,  nx1, x1min, x1max);
    }
    Real divf = FluxDivX1(csys, flx1(m,IEN,k,j,i), flx1(m,IEN,k,j,i+1),
                          size.d_view(m).dx1, rm, rp);
    if (!one_d) {
      divf += FluxDivX2(csys, flx2(m,IEN,k,j,i), flx2(m,IEN,k,j+1,i),
                        size.d_view(m).dx2, x1v);
    }
    if (!one_d && !two_d) {
      divf += FluxDivX3(csys, flx3(m,IEN,k,j,i), flx3(m,IEN,k+1,j,i),
                        size.d_view(m).dx3);
    }
    rhs_out(m,IEN,k,j,i) = -divf;
  });
}

//----------------------------------------------------------------------------------------
//! \fn Real AnisotropicConductionOperator::ExplicitStableDt()
//! \brief Forward-Euler stability dt = fac * min(dx^2 rho/(kappa_par (gamma-1))); the
//! parallel conductivity is the most restrictive coefficient for anisotropic diffusion.

Real AnisotropicConductionOperator::ExplicitStableDt() {
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
  auto &u = cons_;
  auto bcc = bcc_;
  auto par = par_;
  Real gm1 = gamma_ - 1.0;
  Real fac = (three_d) ? (1.0/6.0) : ((multi_d) ? 0.25 : 0.5);
  const Real tiny = 1.0e-30;

  Real dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Kokkos::parallel_reduce("aniso_newdt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real rho = u(m,IDN,k,j,i);
    Real tcode = TempMHD(u, bcc, gm1, m, k, j, i);
    Real bmag = Kokkos::sqrt(SQR(bcc(m,IBX,k,j,i)) + SQR(bcc(m,IBY,k,j,i))
                           + SQR(bcc(m,IBZ,k,j,i)));
    Real kpar, kperp;
    anisocond::Conductivities(par, rho, tcode, bmag, kpar, kperp);
    Real kuse = Kokkos::fmax(kpar, tiny);
    // physical cell-center widths (cylindrical x2 width = x1v*dx2); reduce to dx in
    // Cartesian -> byte-identical there.
    Real x1v = 0.0;
    if (csys != CoordSystem::cartesian) {
      x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    }
    Real w1 = CenterWidth1(csys, size.d_view(m).dx1);
    Real w2 = CenterWidth2(csys, size.d_view(m).dx2, x1v);
    Real w3 = CenterWidth3(csys, size.d_view(m).dx3);
    min_dt = fmin(min_dt, SQR(w1)*rho/(kuse*gm1));
    if (multi_d) {
      min_dt = fmin(min_dt, SQR(w2)*rho/(kuse*gm1));
    }
    if (three_d) {
      min_dt = fmin(min_dt, SQR(w3)*rho/(kuse*gm1));
    }
  }, Kokkos::Min<Real>(dtnew));

  return fac*dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void AnisotropicConductionOperator::ApplyBoundary()
//! \brief Refresh the trial state's ghost zones before every M(.) evaluation.  Two steps,
//! in order (mirrors ConductionOperator #108/[A1] and FLDGreyOperator #110/[A3]):
//!  (1) Zero-gradient (insulated) PHYSICAL fill of all ghost zones: ghost cells copy the
//!      nearest active cell.  Combined with OperatorAction's flux zeroing, this is
//!      the insulated-box BC at a true domain edge (zero heat flux out of the box).
//!  (2) The shared multi-block/MPI/AMR neighbour exchange (SyncParabolicGhosts) then
//!      OVERWRITES the ghosts on every INTERNAL block (and coarse/fine) face with the
//!      neighbour's real data, leaving only the true domain ghosts insulated -- so
//!      a block-decomposed run is a single insulated global box and heat conducts across
//!      the internal block boundaries.  On a lone block with no neighbours (or when built
//!      with pin==nullptr) step (2) is a no-op and the behaviour is the original fill.

void AnisotropicConductionOperator::ApplyBoundary(DvceArray5D<Real> &u) {
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

  par_for("aniso_bcx1", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int g) {
    u(m,n,k,j,is-1-g) = u(m,n,k,j,is);
    u(m,n,k,j,ie+1+g) = u(m,n,k,j,ie);
  });
  if (!one_d) {
    int n1 = u.extent_int(4);
    par_for("aniso_bcx2", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, ng-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int g, const int i) {
      u(m,n,k,js-1-g,i) = u(m,n,k,js,i);
      u(m,n,k,je+1+g,i) = u(m,n,k,je,i);
    });
  }
  if (!one_d && !two_d) {
    int n1 = u.extent_int(4);
    par_for("aniso_bcx3", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, ng-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int g, const int j, const int i) {
      u(m,n,ks-1-g,j,i) = u(m,n,ks,j,i);
      u(m,n,ke+1+g,j,i) = u(m,n,ke,j,i);
    });
  }

  // (2) overwrite internal block-face ghosts (and coarse/fine boundary ghosts under
  // SMR/AMR) with neighbour data via the shared synchronous exchange.  Untouched at true
  // physical boundaries (no neighbour), so those keep the insulated fill from step (1).
  // Skipped only when built with pin==nullptr (legacy single-block-only construction).
  if (pbval_ != nullptr) { pbval_->SyncParabolicGhosts(u, coarse_); }
}
