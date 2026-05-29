#ifndef DIFFUSION_CONDUCTION_OPERATOR_HPP_
#define DIFFUSION_CONDUCTION_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_operator.hpp
//! \brief AthenaK's existing isotropic thermal conduction re-expressed as a
//! parabolic::ParabolicOperator, so it can be advanced operator-split by the RKL2
//! super-time-stepping integrator (issue [4b]/#13, ADR-0001).
//
// The fused conduction path (Conduction::AddHeatFlux, added each RK stage to the
// conserved-variable fluxes and differenced by the hyperbolic RKUpdate) collapses the
// timestep to the diffusive limit dt ~ dx^2/D.  This operator instead exposes the SAME
// isotropic heat-flux divergence as a standalone operator action M(u), which the
// ParabolicIntegrator advances over a superstep with an adaptive RKL2 stage count -- so
// the hyperbolic timestep is no longer parabolic-limited.
//
// The interior flux formula is byte-identical to Conduction::IsotropicHeatFlux:
//   F_face = -kappa * dT/dx,   T = (gamma-1) * eint / rho,
// differenced through the shared geometry accessors (coord_geometry.hpp FluxDivX*), as
// the hydro/MHD RKUpdate does, so the operator is coordinate-agnostic.  Splitting
// freezes the background (density and momentum), so the operator evolves ONLY the total
// energy IEN and writes 0 into every other conserved component (the RKL2 recursion then
// leaves them unchanged).
//
// BOUNDARY: the RKL2 substeps run inside the integrator, so ghost zones are refreshed via
// the ParabolicOperator::ApplyBoundary hook.  On the PHYSICAL domain boundary this is a
// zero-gradient / insulated fill (zero heat flux through the boundary, so total energy is
// conserved and the box has cosine diffusion eigenmodes -- the analytic oracle).  Across
// MeshBlock and MPI-rank boundaries (and coarse/fine faces under SMR/AMR) it delegates to
// the shared MeshBoundaryValuesCC::SyncParabolicGhosts helper (#108/[A1]), which bundles
// AthenaK's existing CC exchange + restriction/prolongation synchronously.  The operator
// therefore owns its own boundary-values object (so its per-substage exchange uses a
// dedicated MPI communicator, never colliding with the hydro/MHD tasklist exchange).

#include "athena.hpp"
#include "driver/parabolic_operator.hpp"

// forward declarations
class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

//----------------------------------------------------------------------------------------
//! \class ConductionOperator
//! \brief Isotropic constant-conductivity thermal conduction as a ParabolicOperator.

class ConductionOperator : public parabolic::ParabolicOperator {
 public:
  //! \brief Build the operator over the conserved field `cons` it diffuses (used for the
  //! explicit-dt density), with constant conductivity `kappa` and adiabatic index gamma.
  //! `pin` is needed to construct the operator's own cell-centered boundary-values object
  //! (the per-substage multi-block/MPI ghost exchange, #108).
  ConductionOperator(MeshBlockPack *pp, ParameterInput *pin,
                     const DvceArray5D<Real> &cons, Real kappa, Real gamma);
  ~ConductionOperator();

  //! \brief M(u): isotropic heat-flux divergence into rhs_out(IEN); 0 in all other
  //! conserved components (frozen background).  Reads ghost zones of u_in.
  void OperatorAction(const DvceArray5D<Real> &u_in,
                      DvceArray5D<Real> &rhs_out) override;

  //! \brief Forward-Euler conduction stability dt (mirrors Conduction::NewTimeStep).
  Real ExplicitStableDt() override;

  //! \brief Ghost-zone refresh for the operator-split substeps: insulated (zero-gradient)
  //! fill at PHYSICAL boundaries, then the shared multi-block/MPI/AMR neighbor exchange.
  void ApplyBoundary(DvceArray5D<Real> &u) override;

 private:
  MeshBlockPack *pmy_pack;
  DvceArray5D<Real> cons_;      // the live conserved field (for the explicit-dt density)
  Real kappa_;                  // (constant, isotropic) thermal conductivity
  Real gamma_;                  // adiabatic index, for T = (gamma-1) eint/rho
  DvceFaceFld5D<Real> hflx_;    // scratch face-centred heat flux
  MeshBoundaryValuesCC *pbval_; // owned bvals object for the per-substage ghost exchange
  DvceArray5D<Real> coarse_;    // coarse-mesh scratch for restriction/prolongation (AMR)
};

#endif  // DIFFUSION_CONDUCTION_OPERATOR_HPP_
