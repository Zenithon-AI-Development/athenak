#ifndef RADIATION_FLD_FLD_MULTIGROUP_OPERATOR_HPP_
#define RADIATION_FLD_FLD_MULTIGROUP_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_multigroup_operator.hpp
//! \brief N-group flux-limited radiation diffusion (FLD) as a
//! parabolic::ParabolicOperator, advanced operator-split by the RKL2 super-time-stepping
//! integrator (issue [17a]/#24, ADR-0001/0007).
//
// The multigroup extension of the grey FLDGreyOperator (#17, ADR-0001).  The radiation
// energy is resolved into the N photon-energy GROUPS of a tabulated multigroup opacity
// (CONTEXT.md "Group"); each group is an independent diffusing energy field
//
//     dE_g/dt = div( D_g grad E_g ),     D_g = c * lambda(R_g) / chi_g,
//
// with a PER-GROUP extinction coefficient chi_g [1/length] and a PER-GROUP Larsen flux
// limiter lambda(R_g) of the group's own dimensionless gradient R_g = |grad E_g|/(chi_g
// E_g).  The group structure (count N) and the per-group Rosseland TRANSPORT opacity come
// from the shared opacity::MultigroupOpacity table (#7, ADR-0007): chi_g = rho *
// kappa_R,g(rho, T_e) (the IONMIX mass opacity multiplied by the mass density), so in the
// optically-thick limit D_g -> c/(3 chi_g) = c/(3 rho kappa_R,g) -- the ADR-0007 consumer
// of the Rosseland transport opacity.
//
// SHAPE.  An exact multigroup clone of FLDGreyOperator.  The field is a single conserved
// array E_g(m, ig, k, j, i) carrying one energy density per group (ig in [0,N)); the
// operator evolves EVERY group component (the grey operator evolved only component 0), so
// the RKL2 5D recursion advances all N groups together over one superstep.  Each
// face-centred diffusive flux F_g = -D_g grad E_g is differenced into the evolved field
// through the shared geometry accessors (coord_geometry.hpp FluxDivX*), so the operator
// is coordinate-agnostic.  D_g depends on the LOCAL group gradient through the limiter,
// so it is recomputed per face every M(.) evaluation and the explicit stability dt is the
// min over all groups of the flux-limited diffusion limit.
//
// BACKGROUND (rho, T_e).  This slice keeps the operator STANDALONE (like the grey #17): a
// uniform background (rho, T_e) sets the per-group extinction chi_g, precomputed once
// from the table in the ctor.  The production wiring that couples chi_g to the live
// per-cell hydro state (and the point-implicit emission/absorption coupling) is the
// consuming issues (#35 multigroup point-implicit coupling, #41 radiation-hydro verify).
//
// BOUNDARY.  Mirrors FLDGreyOperator: a DIRICHLET radiation source at the inner-x1 face
// applied to EVERY group (E_g(x1min) = e_source) and ZERO-GRADIENT elsewhere; e_source<0
// makes inner-x1 zero-gradient too (an insulated box, used by the unit test's conserving
// cosine-eigenmode oracle).  Multi-block / MPI ghost exchange between substeps is left to
// the consuming issues, as in FLDGreyOperator.

#include "athena.hpp"
#include "driver/parabolic_operator.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "radiation_fld/fld_grey_operator.hpp"  // radiationfld::LarsenLimiter

// forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class FLDMultigroupOperator
//! \brief N-group flux-limited radiation diffusion (per-group Larsen limiter, per-group
//! Rosseland D) as a ParabolicOperator.

class FLDMultigroupOperator : public parabolic::ParabolicOperator {
 public:
  //! \brief Build the multigroup FLD operator over the per-group radiation-energy field
  //! `erad` (shape (nmb, ngroups, n3, n2, n1); its var extent MUST equal table.ngroups),
  //! with code-unit light speed `c_light`, the tabulated multigroup opacity `table`
  //! (supplies the group count and per-group Rosseland transport opacity), the uniform
  //! background mass density `rho_bg` and electron temperature `te_bg` that set the
  //! per-group extinction chi_g = rho_bg * kappa_R,g(rho_bg, te_bg), the Larsen exponent
  //! `n_larsen`, and the inner-x1 Dirichlet source `e_source` (e_source < 0 => inner-x1
  //! zero-gradient).
  FLDMultigroupOperator(MeshBlockPack *pp, ParameterInput *pin,
                        const DvceArray5D<Real> &erad,
                        const opacity::MultigroupOpacity &table, Real c_light,
                        Real rho_bg, Real te_bg, Real n_larsen, Real e_source);
  ~FLDMultigroupOperator();

  //! \brief M(u): per-group FLD flux divergence div(D_g grad E_g) into rhs_out(ig) for
  //! every group ig; reads ghost zones of u_in.  (All components are evolved.)
  void OperatorAction(const DvceArray5D<Real> &u_in,
                      DvceArray5D<Real> &rhs_out) override;

  //! \brief Forward-Euler stability dt for the flux-limited diffusion of the current
  //! field, minimised over all groups (the most restrictive per-group D_g).
  Real ExplicitStableDt() override;

  //! \brief Dirichlet inner-x1 (radiation source) + zero-gradient ghost-zone refresh,
  //! applied to every group.
  void ApplyBoundary(DvceArray5D<Real> &u) override;

  // accessors (used by the verification driver / unit test)
  int ngroups() const { return ngroups_; }
  Real c_light() const { return c_; }
  DvceArray1D<Real> chi() const { return chi_; }  // per-group extinction chi_g
  // coarse-mesh scratch (nmb, ngroups, cn3, cn2, cn1); the SAME array the operator uses
  // per-substage (SyncParabolicGhosts) is reused by the AMR regrid restrict/prolong
  // (#111/[A4]), exposed as a non-const lvalue reference for mesh_refinement.cpp.
  DvceArray5D<Real> &coarse() { return coarse_; }

 private:
  MeshBlockPack *pmy_pack;
  DvceArray5D<Real> erad_;    // the live per-group radiation-energy field (explicit dt)
  Real c_;                    // code-unit speed of light
  Real nlarsen_;              // Larsen flux-limiter exponent n
  Real esrc_;                 // inner-x1 Dirichlet source value (<0 => zero-gradient)
  int ngroups_;               // number of photon-energy groups (= table.ngroups)
  DvceArray1D<Real> chi_;     // per-group extinction chi_g = rho*kappa_R,g [1/length]
  DvceFaceFld5D<Real> rflx_;  // scratch face-centred per-group radiative flux
  // conservative fine->coarse flux correction at AMR/SMR level boundaries (#33/#111);
  // built only on a multilevel mesh, nullptr otherwise -> no-op on a uniform grid.
  MeshBoundaryValuesCC *pbval_flux_;
  // per-substage multi-block/MPI/AMR neighbor ghost exchange for the batched per-group
  // field (#108/[A1] SyncParabolicGhosts, all groups in one exchange), with a coarse-mesh
  // scratch for SMR/AMR restriction/prolongation at fine/coarse boundaries (#111/[A4]).
  MeshBoundaryValuesCC *pbval_;
  DvceArray5D<Real> coarse_;
};

#endif  // RADIATION_FLD_FLD_MULTIGROUP_OPERATOR_HPP_
