#ifndef DRIVER_PARABOLIC_OPERATOR_HPP_
#define DRIVER_PARABOLIC_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parabolic_operator.hpp
//  \brief The ParabolicOperator interface: the operator-action contract shared by the
//  operator-split ParabolicIntegrator (RKL2 super-time-stepping, #9) and any future
//  matrix-free implicit backend (issue [4b]/#13, ADR-0001).
//
// ADR-0001 advances all stiff spatial diffusion (multigroup FLD radiation, electron/ion
// conduction, resistive B) by OPERATOR SPLITTING in the driver's once-per-step
// `before/after_timeintegrator` tasklist slot.  The integrator owns only the stepping
// recursion; the PHYSICS is supplied through this interface as the operator action
//
//     M(u) = du/dt|_parabolic = (flux divergence of the diffusive flux),
//
// shaped exactly like the existing `Conduction::AddHeatFlux` (a face-centred diffusive
// flux differenced into a conserved-variable source).  Concrete operators (#13 isotropic
// conduction; #18 Braginskii conduction; #17 multigroup FLD; resistive B) each implement
// `OperatorAction` and `ExplicitStableDt`; the integrator then advances them over a
// superstep without knowing which physics it is integrating.
//
// CONTRACT (the two methods):
//   OperatorAction(u_in, rhs_out): given a (trial) conserved state `u_in`, OVERWRITE
//     `rhs_out` with M(u_in).  The integrator evaluates this at intermediate RKL2 stages,
//     so `u_in` is a trial state, NOT necessarily the live `u0`.  Operators that evolve
//     only a subset of the conserved variables (conduction touches only energy) MUST
//     write 0 into the components they do not evolve, so the RKL2 recursion leaves those
//     components unchanged (the frozen-background operator splitting).  The stencil reads
//     ghost zones of `u_in`; it is the CALLER's responsibility to present a
//     boundary-consistent `u_in` (refresh ghost zones) before each evaluation -- see
//     `ApplyBoundary` below.  Geometry enters ONLY through the shared metric accessors
//     (coord_geometry.hpp), so the operator stays coordinate-agnostic (the FLASH lesson:
//     the r rides in via face area / cell volume).
//   ExplicitStableDt(): the forward-Euler stability limit dt_exp for this operator's
//     current state.  The integrator derives its adaptive RKL2 stage count from the ratio
//     dt_super/dt_exp (RKL2NumStages): the bridge between physics and integrator.
//
// BOUNDARY refresh (ApplyBoundary): the RKL2 substeps happen INSIDE the integrator (the
// driver does not re-enter the boundary tasklist between substages), so the operator
// exposes a hook to refresh the ghost zones of a trial state.  The operator-split task
// wraps `OperatorAction` in a closure that calls `ApplyBoundary` first, so every M(.)
// evaluation sees boundary-consistent ghosts.  The default is a no-op (interior-only or
// periodic-by-array operators need nothing); single-block physical BCs override it (#13);
// the multi-block MPI ghost exchange is the consuming physics issues' responsibility.

#include "athena.hpp"   // Real, DvceArray5D

namespace parabolic {

//----------------------------------------------------------------------------------------
//! \class ParabolicOperator
//! \brief Abstract operator-action interface consumed by the ParabolicIntegrator (#9) and
//! any future implicit backend.  See the file comment for the full contract.

class ParabolicOperator {
 public:
  virtual ~ParabolicOperator() = default;

  //! \brief Operator action M(u): OVERWRITE `rhs_out` with the parabolic time derivative
  //! du/dt of the (trial) conserved state `u_in`.  Components not evolved are set to 0.
  virtual void OperatorAction(const DvceArray5D<Real> &u_in,
                              DvceArray5D<Real> &rhs_out) = 0;

  //! \brief Forward-Euler stability timestep dt_exp for this operator's current state.
  //! The integrator picks its RKL2 stage count from dt_super/dt_exp.
  virtual Real ExplicitStableDt() = 0;

  //! \brief Refresh the ghost zones of a trial state so the next OperatorAction sees a
  //! boundary-consistent stencil.  Default: no-op (override for single-block BCs).
  virtual void ApplyBoundary(DvceArray5D<Real> &u) { (void)u; }
};

}  // namespace parabolic

#endif  // DRIVER_PARABOLIC_OPERATOR_HPP_
