//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_multigroup_operator.cpp
//! \brief Implements FLDMultigroupOperator: N-group flux-limited radiation diffusion
//! (per-group Larsen limiter, per-group Rosseland D) as a parabolic::ParabolicOperator
//! advanced operator-split by RKL2 STS (issue [17a]/#24, ADR-0001/0007).  See
//! fld_multigroup_operator.hpp for the model and contract.

#include <float.h>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"

using radiationfld::LarsenLimiter;

//----------------------------------------------------------------------------------------
//! \brief FLDMultigroupOperator constructor.  Precomputes the per-group extinction
//! chi_g = rho_bg * kappa_R,g(rho_bg, te_bg) from the tabulated Rosseland transport
//! opacity (the group structure and per-group D both come from the opacity table).

FLDMultigroupOperator::FLDMultigroupOperator(MeshBlockPack *pp,
  const DvceArray5D<Real> &erad, const opacity::MultigroupOpacity &table, Real c_light,
  Real rho_bg, Real te_bg, Real n_larsen, Real e_source) :
  pmy_pack(pp),
  erad_(erad),
  c_(c_light),
  nlarsen_(n_larsen),
  esrc_(e_source),
  ngroups_(table.ngroups),
  chi_("fld_mg_chi", table.ngroups),
  rflx_("fld_mg_rflx", erad.extent_int(0), erad.extent_int(1),
        erad.extent_int(2), erad.extent_int(3), erad.extent_int(4)) {
  // Precompute the per-group extinction chi_g from the tabulated Rosseland transport
  // (mass) opacity at the background: chi_g = rho * kappa_R,g(rho, te) [1/length].
  opacity::MultigroupOpacity tab = table;   // shallow View copy -> device-valid in kernel
  auto chi = chi_;
  const Real rho = rho_bg, te = te_bg;
  const int ng = ngroups_;
  par_for("fld_mg_chi_init", DevExeSpace(), 0, ng-1, KOKKOS_LAMBDA(const int ig) {
    chi(ig) = rho*tab.RosselandTransport(ig, rho, te);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void FLDMultigroupOperator::OperatorAction()
//! \brief M(u): write the per-group FLD flux divergence div(D_g grad E_g) into
//! rhs_out(ig) for every group.  Each group's face diffusivity D_g = c lambda(R_g)/chi_g
//! uses the Larsen limiter of the group's local face gradient R_g = |grad E_g|/(chi_g
//! E_face), so each group's diffusive flux saturates at its own free-streaming limit.

void FLDMultigroupOperator::OperatorAction(const DvceArray5D<Real> &u_in,
  DvceArray5D<Real> &rhs_out) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int ng1 = ngroups_ - 1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  auto chi = chi_;
  Real cc = c_, nl = nlarsen_;
  auto &flx1 = rflx_.x1f;
  auto &flx2 = rflx_.x2f;
  auto &flx3 = rflx_.x3f;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;
  const Real tiny = 1.0e-30;

  // rhs_out is fully overwritten: zero first (every group is evolved below, but this
  // keeps the contract -- any unused component carries M = 0 -> frozen background).
  Kokkos::deep_copy(rhs_out, 0.0);

  // x1 per-group radiative flux F1_g = -D_g dE_g/dx1, D_g = c lambda(R_g)/chi_g.
  par_for("fld_mg_flx1", DevExeSpace(), 0, nmb1, 0, ng1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
    Real cg = chi(g);
    Real el = u_in(m,g,k,j,i-1);
    Real er = u_in(m,g,k,j,i);
    Real dedx = (er - el)/size.d_view(m).dx1;
    Real eface = 0.5*(el + er);
    Real R = Kokkos::fabs(dedx)/(cg*Kokkos::fmax(eface, tiny));
    Real D = cc*LarsenLimiter(R, nl)/cg;
    flx1(m,g,k,j,i) = -D*dedx;
  });

  if (!one_d) {
    par_for("fld_mg_flx2", DevExeSpace(), 0, nmb1, 0, ng1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
      Real cg = chi(g);
      Real el = u_in(m,g,k,j-1,i);
      Real er = u_in(m,g,k,j,i);
      Real dedx = (er - el)/size.d_view(m).dx2;
      Real eface = 0.5*(el + er);
      Real R = Kokkos::fabs(dedx)/(cg*Kokkos::fmax(eface, tiny));
      Real D = cc*LarsenLimiter(R, nl)/cg;
      flx2(m,g,k,j,i) = -D*dedx;
    });
  }
  if (!one_d && !two_d) {
    par_for("fld_mg_flx3", DevExeSpace(), 0, nmb1, 0, ng1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
      Real cg = chi(g);
      Real el = u_in(m,g,k-1,j,i);
      Real er = u_in(m,g,k,j,i);
      Real dedx = (er - el)/size.d_view(m).dx3;
      Real eface = 0.5*(el + er);
      Real R = Kokkos::fabs(dedx)/(cg*Kokkos::fmax(eface, tiny));
      Real D = cc*LarsenLimiter(R, nl)/cg;
      flx3(m,g,k,j,i) = -D*dedx;
    });
  }

  // dE_g/dt = -div(F_g): difference the per-group face fluxes through the geometry
  // accessors, exactly as the hydro/MHD RKUpdate and FLDGreyOperator do.
  par_for("fld_mg_div", DevExeSpace(), 0, nmb1, 0, ng1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
    Real divf = FluxDivX1(csys, flx1(m,g,k,j,i), flx1(m,g,k,j,i+1),
                          size.d_view(m).dx1);
    if (!one_d) {
      divf += FluxDivX2(csys, flx2(m,g,k,j,i), flx2(m,g,k,j+1,i),
                        size.d_view(m).dx2);
    }
    if (!one_d && !two_d) {
      divf += FluxDivX3(csys, flx3(m,g,k,j,i), flx3(m,g,k+1,j,i),
                        size.d_view(m).dx3);
    }
    rhs_out(m,g,k,j,i) = -divf;
  });
}

//----------------------------------------------------------------------------------------
//! \fn Real FLDMultigroupOperator::ExplicitStableDt()
//! \brief Forward-Euler stability dt for the flux-limited diffusion of the current field,
//! minimised over all groups: dt = min_g 1/(2 D_g (1/dx1^2 + 1/dx2^2 + 1/dx3^2)), with
//! the cell-centred per-group diffusivity D_g = c lambda(R_g)/chi_g.

Real FLDMultigroupOperator::ExplicitStableDt() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int ng = ngroups_;
  const int ngmkji = (pmy_pack->nmb_thispack)*ng*nx3*nx2*nx1;
  const int ngkji = ng*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto &multi_d = pmy_pack->pmesh->multi_d;
  auto &three_d = pmy_pack->pmesh->three_d;
  auto size = pmy_pack->pmb->mb_size;
  auto &u = erad_;
  auto chi = chi_;
  Real cc = c_, nl = nlarsen_;
  const Real tiny = 1.0e-30;

  Real dtnew = static_cast<Real>(std::numeric_limits<float>::max());
  Kokkos::parallel_reduce("fld_mg_newdt",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ngmkji),
  KOKKOS_LAMBDA(const int &idx, Real &min_dt) {
    int m = (idx)/ngkji;
    int g = (idx - m*ngkji)/nkji;
    int k = (idx - m*ngkji - g*nkji)/nji;
    int j = (idx - m*ngkji - g*nkji - k*nji)/nx1;
    int i = (idx - m*ngkji - g*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real cg = chi(g);
    // cell-centred gradient magnitude (central differences over active dims)
    Real e0 = u(m,g,k,j,i);
    Real gx1 = (u(m,g,k,j,i+1) - u(m,g,k,j,i-1))/(2.0*size.d_view(m).dx1);
    Real g2 = gx1*gx1;
    Real inv = 1.0/SQR(size.d_view(m).dx1);
    if (multi_d) {
      Real gx2 = (u(m,g,k,j+1,i) - u(m,g,k,j-1,i))/(2.0*size.d_view(m).dx2);
      g2 += gx2*gx2;
      inv += 1.0/SQR(size.d_view(m).dx2);
    }
    if (three_d) {
      Real gx3 = (u(m,g,k+1,j,i) - u(m,g,k-1,j,i))/(2.0*size.d_view(m).dx3);
      g2 += gx3*gx3;
      inv += 1.0/SQR(size.d_view(m).dx3);
    }
    Real R = Kokkos::sqrt(g2)/(cg*Kokkos::fmax(e0, tiny));
    Real D = cc*LarsenLimiter(R, nl)/cg;
    min_dt = fmin(min_dt, 1.0/(2.0*D*inv));
  }, Kokkos::Min<Real>(dtnew));

  return dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn void FLDMultigroupOperator::ApplyBoundary()
//! \brief Refresh ghost zones for every group: DIRICHLET radiation source at the inner-x1
//! face (E_g(x1min) = esrc_ via linear extrapolation ghost = 2 esrc - E[is]) when
//! esrc_ >= 0, and ZERO-GRADIENT (insulated/outflow) everywhere else.  With esrc_ < 0 the
//! inner-x1 face is zero-gradient too (an insulated box).

void FLDMultigroupOperator::ApplyBoundary(DvceArray5D<Real> &u) {
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

  // x1 ghosts (always): zero-gradient at outer-x1; inner-x1 Dirichlet source (every
  // group) if esrc>=0, else zero-gradient.
  par_for("fld_mg_bcx1", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, ng-1,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int g) {
    if (esrc >= 0.0) {
      // Dirichlet: ghost reflects the active cell about the source so the inner-x1 FACE
      // value is esrc (linear extrapolation).
      u(m,n,k,j,is-1-g) = 2.0*esrc - u(m,n,k,j,is);
    } else {
      u(m,n,k,j,is-1-g) = u(m,n,k,j,is);
    }
    u(m,n,k,j,ie+1+g) = u(m,n,k,j,ie);
  });
  if (!one_d) {
    int n1 = u.extent_int(4);
    par_for("fld_mg_bcx2", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, n3-1, 0, ng-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int g, const int i) {
      u(m,n,k,js-1-g,i) = u(m,n,k,js,i);
      u(m,n,k,je+1+g,i) = u(m,n,k,je,i);
    });
  }
  if (!one_d && !two_d) {
    int n1 = u.extent_int(4);
    par_for("fld_mg_bcx3", DevExeSpace(), 0, nmb1, 0, nvar-1, 0, ng-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int g, const int j, const int i) {
      u(m,n,ks-1-g,j,i) = u(m,n,ks,j,i);
      u(m,n,ke+1+g,j,i) = u(m,n,ke,j,i);
    });
  }
}
