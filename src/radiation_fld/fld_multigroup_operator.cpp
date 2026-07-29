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
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "radiation_fld/fld_grey_operator.hpp"    // radiationfld::FaceDeff (#215)
#include "radiation_fld/fld_multigroup_operator.hpp"

using radiationfld::FaceDeff;
using radiationfld::LarsenLimiter;

//----------------------------------------------------------------------------------------
//! \brief FLDMultigroupOperator constructor.  Precomputes the per-group extinction
//! chi_g = rho_bg * kappa_R,g(rho_bg, te_bg) from the tabulated Rosseland transport
//! opacity (the group structure and per-group D both come from the opacity table), and --
//! exactly like FLDGreyOperator (#110/[A3]) -- allocates the operator's own cell-centered
//! boundary-values object (its per-substage multi-block/MPI ghost exchange, sized to ALL
//! groups so the batched field exchanges in one call), plus a coarse-mesh scratch array
//! used both by SyncParabolicGhosts and by the AMR regrid restrict/prolong (#111/[A4]).

FLDMultigroupOperator::FLDMultigroupOperator(MeshBlockPack *pp, ParameterInput *pin,
  const DvceArray5D<Real> &erad, const opacity::MultigroupOpacity &table, Real c_light,
  Real rho_bg, Real te_bg, Real n_larsen, Real e_source, Real e_floor, Real upwind) :
  pmy_pack(pp),
  erad_(erad),
  c_(c_light),
  nlarsen_(n_larsen),
  esrc_(e_source),
  // clamp efloor_ > 0 to avoid a 0/0 NaN from a misconfigured deck (mirror grey #194).
  efloor_((e_floor > 0.0) ? e_floor : static_cast<Real>(1.0e-30)),
  upwind_(upwind),
  injected_energy_(0.0),
  ngroups_(table.ngroups),
  chi_("fld_mg_chi", table.ngroups),
  rflx_("fld_mg_rflx", erad.extent_int(0), erad.extent_int(1),
        erad.extent_int(2), erad.extent_int(3), erad.extent_int(4)),
  pbval_flux_(nullptr),
  pbval_(nullptr),
  coarse_("fld_mg_coarse", 1, 1, 1, 1, 1) {
  // Precompute the per-group extinction chi_g (in a member function, not here: nvcc
  // forbids the extended-lambda par_for inside a constructor).
  InitChi(table, rho_bg, te_bg);

  // own boundary-values object for the per-substage neighbor ghost exchange: a unique MPI
  // communicator for this operator's exchange, buffers sized to ALL groups (the
  // multigroup operator batches every group into one exchange).
  const int nvar = erad.extent_int(1);   // == ngroups
  pbval_ = new MeshBoundaryValuesCC(pp, pin, false);
  pbval_->InitializeBuffers(nvar);
  // coarse-mesh scratch (only meaningful with SMR/AMR; left 1^5 on a uniform grid, where
  // SyncParabolicGhosts never touches it).  This SAME array is reused by the regrid
  // restrict/prolong of erad_mg (exposed via coarse()).
  if (pp->pmesh->multilevel) {
    auto &indcs = pp->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1) ? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1) ? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_, erad.extent_int(0), nvar, n_ccells3, n_ccells2, n_ccells1);
    // register the per-group radiative flux for the conservative fine->coarse flux
    // correction (#33/#111), so the diffused radiation energy is conserved across
    // refinement boundaries for the multigroup operator (the grey operator already did).
    pbval_flux_ = new MeshBoundaryValuesCC(pp, nullptr, false);
    pbval_flux_->InitializeBuffers(rflx_.x1f.extent_int(1));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FLDMultigroupOperator::InitChi()
//! \brief Precompute the per-group extinction chi_g = rho * kappa_R,g(rho, te) [1/length]
//! from the tabulated Rosseland transport (mass) opacity at the uniform background.
//! Separated from the constructor because nvcc rejects an extended __device__ lambda
//! whose enclosing parent function is a constructor (cannot take its address).

void FLDMultigroupOperator::InitChi(const opacity::MultigroupOpacity &table,
                                    Real rho_bg, Real te_bg) {
  opacity::MultigroupOpacity tab = table;   // shallow View copy -> device-valid in kernel
  auto chi = chi_;
  const Real rho = rho_bg, te = te_bg;
  const int ng = ngroups_;
  par_for("fld_mg_chi_init", DevExeSpace(), 0, ng-1, KOKKOS_LAMBDA(const int ig) {
    chi(ig) = rho*tab.RosselandTransport(ig, rho, te);
  });
}

//----------------------------------------------------------------------------------------
//! \brief FLDMultigroupOperator destructor: free the owned boundary-values objects.

FLDMultigroupOperator::~FLDMultigroupOperator() {
  if (pbval_flux_ != nullptr) { delete pbval_flux_; }
  if (pbval_ != nullptr) { delete pbval_; }
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
  Real cc = c_, nl = nlarsen_, efloor = efloor_, upwind = upwind_;
  int nx1 = indcs.nx1;
  auto &flx1 = rflx_.x1f;
  auto &flx2 = rflx_.x2f;
  auto &flx3 = rflx_.x3f;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;

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
    Real R = Kokkos::fabs(dedx)/(cg*Kokkos::fmax(eface, efloor));
    Real lam = LarsenLimiter(R, nl);
    Real D = cc*lam/cg;
    // LF streaming dissipation (#215, per-group port of grey #194): signal speed
    // cc*lam*R (-> c streaming) GATED by the advective fraction (1-3 lam)_+ (0 where
    // lam=1/3 thick, -> 1 streaming), so it vanishes as ~R^2 where the group is already
    // a stable diffusion and damps the centered-advection (imaginary) mode RKL2 cannot
    // at a sharp front.  Without it a sustained steep front runs away to NaN per group.
    Real alf = cc*lam*R*Kokkos::fmax(0.0, 1.0 - 3.0*lam);
    flx1(m,g,k,j,i) = -D*dedx - 0.5*upwind*alf*(er - el);
  });

  if (!one_d) {
    par_for("fld_mg_flx2", DevExeSpace(), 0, nmb1, 0, ng1, ks, ke, js, je+1, is, ie,
    KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
      Real cg = chi(g);
      Real el = u_in(m,g,k,j-1,i);
      Real er = u_in(m,g,k,j,i);
      // azimuthal gradient uses the physical ARC LENGTH (cyl CenterWidth2 = x1v*dx2,
      // Cartesian = dx2 -> bit-identical), so the cross-axis phi diffusion is geometric.
      // Dividing by the raw dx2 (an ANGLE on a cylindrical mesh) is wrong by r (#215).
      Real x1v2 = 0.0;
      if (csys == CoordSystem::cylindrical) {
        x1v2 = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real dedx = (er - el)/CenterWidth2(csys, size.d_view(m).dx2, x1v2);
      Real eface = 0.5*(el + er);
      Real R = Kokkos::fabs(dedx)/(cg*Kokkos::fmax(eface, efloor));
      Real lam = LarsenLimiter(R, nl);
      Real D = cc*lam/cg;
      Real alf = cc*lam*R*Kokkos::fmax(0.0, 1.0 - 3.0*lam);  // gated LF (#215)
      flx2(m,g,k,j,i) = -D*dedx - 0.5*upwind*alf*(er - el);
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
      Real R = Kokkos::fabs(dedx)/(cg*Kokkos::fmax(eface, efloor));
      Real lam = LarsenLimiter(R, nl);
      Real D = cc*lam/cg;
      Real alf = cc*lam*R*Kokkos::fmax(0.0, 1.0 - 3.0*lam);  // gated LF (#215)
      flx3(m,g,k,j,i) = -D*dedx - 0.5*upwind*alf*(er - el);
    });
  }

  // conservative fine->coarse flux correction at SMR/AMR level boundaries (#33/#111):
  // replace each coarse-side boundary face flux with the restricted (area-averaged) fine
  // flux so the coarse divergence matches the sum of the fine faces -- for EVERY group
  // (pbval_flux_ buffers are sized to all groups).  No-op on a uniform mesh.
  if (pbval_flux_ != nullptr) { pbval_flux_->CorrectFlux(rflx_); }

  // dE_g/dt = -div(F_g): difference the per-group face fluxes through the geometry
  // accessors, exactly as the hydro/MHD RKUpdate and FLDGreyOperator do.
  par_for("fld_mg_div", DevExeSpace(), 0, nmb1, 0, ng1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
    // cylindrical face radii (rm,rp) / cell radius (x1v) feed the curvilinear flux
    // divergence (1/r) d(r F)/dr and (1/r) dF/dphi; default 0 -> Cartesian (fr-fl)/dx,
    // byte-identical.  Required for the operator to run on the cylindrical MagLIF mesh
    // at all (#215) -- without them FluxDivX1 evaluates 0/0 at EVERY cell and every group
    // goes NaN, exactly as the grey operator did before #116/[B3].  Benchmarks 5-6 run on
    // that mesh, so this is a hard blocker for them, not an accuracy detail.
    Real rm = 0.0, rp = 0.0, x1v = 0.0;
    if (csys == CoordSystem::cylindrical) {
      Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      rm  = LeftEdgeX(i-is,    nx1, x1min, x1max);
      rp  = LeftEdgeX(i+1-is,  nx1, x1min, x1max);
      x1v = CellCenterX(i-is,  nx1, x1min, x1max);
    }
    Real divf = FluxDivX1(csys, flx1(m,g,k,j,i), flx1(m,g,k,j,i+1),
                          size.d_view(m).dx1, rm, rp);
    if (!one_d) {
      divf += FluxDivX2(csys, flx2(m,g,k,j,i), flx2(m,g,k,j+1,i),
                        size.d_view(m).dx2, x1v);
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
//! minimised over all groups.  Face-consistent (#215, per-group port of grey #194): the
//! effective per-face diffusivity D_eff (flux-limited D_g + the Lax-Friedrichs streaming
//! viscosity) is built from the SAME el/er/eface the flux kernel uses, so the reported
//! bound is the one the scheme it advances actually needs.

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
  auto csys = pmy_pack->pcoord->coord_system;
  auto &u = erad_;
  auto chi = chi_;
  Real cc = c_, nl = nlarsen_, efloor = efloor_, upwind = upwind_;

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
    // face-consistent forward-Euler bound (#215, per-group port of grey #194): the
    // effective diffusivity D_eff (flux-limited D_g + the LF streaming viscosity) is
    // built
    // per face from the SAME el/er/eface/R_g the flux kernel uses, summed over the two
    // faces per active dim:  dt <= 1 / sum_dim (D_eff_minus + D_eff_plus)/w^2.
    // In a flat thick cell D_eff_minus = D_eff_plus = D and this reduces to 1/(2 D inv)
    // (the old cell-centred estimate).  At a front the old estimate over-counted: a DARK
    // cell beside a BRIGHT neighbour has R_g = |grad|/(chi_g e0) -> huge from the central
    // difference, so lambda ~ 1/R, D ~ 0, and it reported an effectively infinite dt.
    Real e0 = u(m,g,k,j,i);
    Real w1 = size.d_view(m).dx1;
    Real Dm = FaceDeff(u(m,g,k,j,i-1), e0, w1, cc, cg, nl, efloor, upwind);
    Real Dp = FaceDeff(e0, u(m,g,k,j,i+1), w1, cc, cg, nl, efloor, upwind);
    Real invsum = (Dm + Dp)/SQR(w1);
    if (multi_d) {
      Real x1v = 0.0;
      if (csys == CoordSystem::cylindrical) {
        x1v = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      }
      Real w2 = CenterWidth2(csys, size.d_view(m).dx2, x1v);
      Real Dm2 = FaceDeff(u(m,g,k,j-1,i), e0, w2, cc, cg, nl, efloor, upwind);
      Real Dp2 = FaceDeff(e0, u(m,g,k,j+1,i), w2, cc, cg, nl, efloor, upwind);
      invsum += (Dm2 + Dp2)/SQR(w2);
    }
    if (three_d) {
      Real w3 = size.d_view(m).dx3;
      Real Dm3 = FaceDeff(u(m,g,k-1,j,i), e0, w3, cc, cg, nl, efloor, upwind);
      Real Dp3 = FaceDeff(e0, u(m,g,k+1,j,i), w3, cc, cg, nl, efloor, upwind);
      invsum += (Dm3 + Dp3)/SQR(w3);
    }
    if (invsum > 0.0) { min_dt = fmin(min_dt, 1.0/invsum); }
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

  // overwrite internal block-face ghosts (and coarse/fine boundary ghosts under SMR/AMR)
  // with neighbor data via the shared synchronous exchange -- ALL groups in one call
  // (#108/[A1], #111/[A4]).  Untouched at true physical boundaries (no neighbor), so
  // those keep the physical fill above; a no-op on a single block.
  pbval_->SyncParabolicGhosts(u, coarse_);
}

//----------------------------------------------------------------------------------------
//! \fn void FLDMultigroupOperator::PostSuperstepProject()
//! \brief Positivity projection run once after each RKL2 super-step (#197, per-group port
//! of grey #194): floor the committed per-group radiation energy E_g to efloor_ over all
//! active cells of every group and accumulate the energy added by the floor (Sum over ALL
//! groups of max(efloor-E_g,0)*cell_volume) so the injection stays bounded and accounted.
//! Without it the free-streaming / transition-regime FLD operator (advective under RKL2)
//! overshoots a thin group's E_g negative and the divide-guard cascades to NaN in the
//! coupled run.
void FLDMultigroupOperator::PostSuperstepProject(DvceArray5D<Real> &u) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  const int ng = ngroups_;
  const int ngmkji = (pmy_pack->nmb_thispack)*ng*nx3*nx2*nx1;
  const int ngkji = ng*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  auto size = pmy_pack->pmb->mb_size;
  auto csys = pmy_pack->pcoord->coord_system;
  bool one_d = pmy_pack->pmesh->one_d;
  bool two_d = pmy_pack->pmesh->two_d;
  Real efloor = efloor_;

  Real injected = 0.0;
  Kokkos::parallel_reduce("fld_mg_project",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ngmkji),
  KOKKOS_LAMBDA(const int &idx, Real &linj) {
    int m = (idx)/ngkji;
    int g = (idx - m*ngkji)/nkji;
    int k = (idx - m*ngkji - g*nkji)/nji;
    int j = (idx - m*ngkji - g*nkji - k*nji)/nx1;
    int i = (idx - m*ngkji - g*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real e = u(m,g,k,j,i);
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
      u(m,g,k,j,i) = efloor;
    }
  }, Kokkos::Sum<Real>(injected));
  injected_energy_ += injected;
}
