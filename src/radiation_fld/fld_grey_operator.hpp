#ifndef RADIATION_FLD_FLD_GREY_OPERATOR_HPP_
#define RADIATION_FLD_FLD_GREY_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_grey_operator.hpp
//! \brief Grey (single-group) flux-limited radiation diffusion (FLD) re-expressed as a
//! parabolic::ParabolicOperator, so it is advanced operator-split by the RKL2
//! super-time-stepping integrator (issue [8a]/#17, ADR-0001).
//
// This is the FIRST radiation slice of the HED/MagLIF stack and the ADR-0001
// "measure-first" go/no-go experiment for the explicit STS approach: it advances the grey
// radiation energy density E_r by the flux-limited diffusion equation
//
//     dE_r/dt = div( D grad E_r ),     D = c * lambda(R) / chi,
//
// where chi [1/length] is the (Rosseland) extinction coefficient (= kappa_R * rho; here
// a fixed/analytic constant -- the tabulated multigroup opacity of #7/#24 is a later
// slice), c is the code-unit speed of light, and lambda(R) is the LARSEN flux limiter of
// the dimensionless gradient R = |grad E_r| / (chi E_r):
//
//     lambda(R) = ( 3^n + R^n )^(-1/n)   ->  1/3  (R->0, optically-thick diffusion limit)
//                                        ->  1/R  (R->inf, optically-thin free-stream).
//
// In the thick limit D -> c/(3 chi) (the equilibrium-diffusion coefficient); in the thin
// limit the flux saturates at |F| = D|grad E| -> c E_r (free streaming), the c-CFL term
// STS cannot accelerate (ADR-0001's known consequence -- which the substage-count sweep
// in the Marshak verification reports across optical depths).
//
// SHAPE.  Mirrors the existing ConductionOperator (#13) exactly: a face-centred diffusive
// flux F = -D grad E_r differenced into the evolved field through the shared geometry
// accessors (coord_geometry.hpp FluxDivX*), so the operator is coordinate-agnostic.  The
// FLD field is a single-variable conserved array E_r(m, irad, k, j, i); the operator
// evolves component `irad` and writes 0 into every other component, so the RKL2 recursion
// leaves them frozen (the operator-split "frozen background").  Unlike isotropic
// conduction, D depends on the LOCAL gradient through the limiter, so it is recomputed
// per face every M(.) evaluation and the explicit stability dt is taken from the
// flux-limited D of the current state.
//
// BOUNDARY.  The RKL2 substeps run inside the integrator, so ghosts are refreshed via the
// ParabolicOperator::ApplyBoundary hook.  The grey operator supports the boundary
// conditions the analytic Marshak wave needs: a DIRICHLET radiation source at the
// inner-x1 face (the hot wall E_r(x1min) = e_source) and ZERO-GRADIENT (insulated /
// outflow) elsewhere.  With e_source < 0 the inner-x1 face is also zero-gradient (an
// insulated box, used by the operator unit test's conserving cosine-eigenmode oracle).
// The PHYSICAL-BC fill above is followed by the shared multi-block/MPI/AMR neighbor ghost
// exchange MeshBoundaryValuesCC::SyncParabolicGhosts (#108/[A1]), which OVERWRITES the
// ghosts on every internal block (and coarse/fine) face with the neighbor's real data,
// leaving only true domain-boundary ghosts at their physical value -- exactly as
// ConductionOperator does.  So the operator runs operator-split across >1 MeshBlock, >1
// MPI rank, and under block-AMR (the radiation FLD wiring of #110/[A3]).  On a single
// block with no neighbors the exchange is a no-op and the behaviour is the original fill.

#include "athena.hpp"
#include "driver/parabolic_operator.hpp"

// forward declarations
class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace radiationfld {

//----------------------------------------------------------------------------------------
//! \fn Real LarsenLimiter
//! \brief Larsen flux limiter lambda(R) = (3^n + R^n)^(-1/n) for the dimensionless
//! radiation-energy gradient R = |grad E_r|/(chi E_r) and exponent n (default 2).
//! Smooth across the thick/thin transition: lambda(0)=1/3 (diffusion), lambda(R->inf)=1/R
//! (free-streaming).  KOKKOS_INLINE_FUNCTION so it is host- and device-callable and
//! shared by the operator and its unit test.
KOKKOS_INLINE_FUNCTION
Real LarsenLimiter(Real R, Real n) {
  // R is |grad E|/(chi E) >= 0 by construction; clamp defensively.
  Real Rc = (R > 0.0) ? R : 0.0;
  Real three_n = Kokkos::pow(static_cast<Real>(3.0), n);
  Real Rn = Kokkos::pow(Rc, n);
  return Kokkos::pow(three_n + Rn, -1.0/n);
}

}  // namespace radiationfld

//----------------------------------------------------------------------------------------
//! \class FLDGreyOperator
//! \brief Grey flux-limited radiation diffusion (Larsen limiter) as a ParabolicOperator.

class FLDGreyOperator : public parabolic::ParabolicOperator {
 public:
  //! \brief Build the grey FLD operator over the radiation-energy field `erad` it
  //! diffuses (read for the flux-limited explicit-dt), with code-unit light speed
  //! `c_light`, constant extinction coefficient `chi` [1/length], Larsen exponent
  //! `n_larsen`, and inner-x1 Dirichlet source value `e_source` (e_source < 0 =>
  //! inner-x1 is zero-gradient too).  `pin` is needed to construct the operator's own
  //! cell-centered boundary-values object for the per-substage multi-block/MPI/AMR ghost
  //! exchange (#108/[A1], #110/[A3]).
  FLDGreyOperator(MeshBlockPack *pp, ParameterInput *pin,
                  const DvceArray5D<Real> &erad,
                  Real c_light, Real chi, Real n_larsen, Real e_source,
                  Real e_floor = 1.0e-30);
  ~FLDGreyOperator();

  //! \brief M(u): grey FLD flux divergence div(D grad E_r) into rhs_out(irad); 0 in all
  //! other components (frozen background).  Reads ghost zones of u_in.
  void OperatorAction(const DvceArray5D<Real> &u_in,
                      DvceArray5D<Real> &rhs_out) override;

  //! \brief Forward-Euler stability dt for the flux-limited radiation diffusion of the
  //! current field (D from the Larsen limiter, so it recovers the c-CFL cap when thin).
  Real ExplicitStableDt() override;

  //! \brief Dirichlet inner-x1 (radiation source) + zero-gradient ghost-zone refresh.
  void ApplyBoundary(DvceArray5D<Real> &u) override;

  //! \brief Post-superstep positivity projection (#194): floor the committed radiation
  //! energy to efloor_ over active cells and accumulate the injected (floored-in) energy.
  void PostSuperstepProject(DvceArray5D<Real> &u) override;

  // accessors (used by the verification driver / unit test)
  Real chi() const { return chi_; }
  Real c_light() const { return c_; }
  Real efloor() const { return efloor_; }
  //! \brief Total energy added by the positivity floor across all PostSuperstepProject
  //! calls (Sum of max(efloor-erad,0)*cell_volume): the #194 "accounted injection".
  //! Rank-local (no MPI reduce); a serial diagnostic / per-rank budget for now.
  Real injected_energy() const { return injected_energy_; }

 private:
  MeshBlockPack *pmy_pack;
  DvceArray5D<Real> erad_;      // the live radiation-energy field (for the explicit dt)
  Real c_;                      // code-unit speed of light
  Real chi_;                    // constant extinction coefficient kappa_R*rho [1/length]
  Real nlarsen_;                // Larsen flux-limiter exponent n
  Real esrc_;                   // inner-x1 Dirichlet source value (<0 => zero-gradient)
  Real efloor_;                 // positivity floor for erad (#194): read-floor+projection
  Real injected_energy_;        // energy added by the positivity floor (#194)
  int irad_;                    // evolved radiation-energy component (default 0)
  DvceFaceFld5D<Real> rflx_;    // scratch face-centred radiative flux
  // conservative fine->coarse flux correction at AMR/SMR level boundaries (#33); built
  // only on a multilevel mesh, nullptr otherwise -> no-op on a uniform grid.
  MeshBoundaryValuesCC *pbval_flux_;
  // per-substage multi-block/MPI/AMR neighbor ghost exchange for the diffused field
  // (#108/[A1] SyncParabolicGhosts), with a coarse-mesh scratch for the SMR/AMR
  // restriction/prolongation at fine/coarse boundaries (#110/[A3]).
  MeshBoundaryValuesCC *pbval_;
  DvceArray5D<Real> coarse_;
};

#endif  // RADIATION_FLD_FLD_GREY_OPERATOR_HPP_
