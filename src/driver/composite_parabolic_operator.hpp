#ifndef DRIVER_COMPOSITE_PARABOLIC_OPERATOR_HPP_
#define DRIVER_COMPOSITE_PARABOLIC_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file composite_parabolic_operator.hpp
//! \brief A ParabolicOperator that runs SEVERAL stiff parabolic operators as a single
//! RKL2 super-step over their sum (issue [B1]/#114, ADR-0009).
//
// ADR-0009 settles the open question from ADR-0001: when several stiff spatial operators
// are active (anisotropic Braginskii e/i conduction, resistive B_phi diffusion,
// multigroup FLD radiation), AthenaK runs ONE super-step over their SUM rather than one
// sweep per operator.  This is the proven design in every production STS FV code
// (Athena++, AthenaPK, PLUTO, MHDSTS): the RKL2 stability polynomial only has to bound
// largest eigenvalue of the summed operator, which `dt_diff = min_i dt_exp_i` guarantees,
// so summing carries no stability penalty and dissolves any Lie/Strang ordering AMONG the
// parabolic terms (they are added, never sequenced).
//
// This CompositeParabolicOperator IS itself a parabolic::ParabolicOperator, so the
// existing integrator (parabolic::OperatorSplitStep / ParabolicIntegrator::Superstep)
// advances it unchanged.  It holds a list of (non-owning) sub-operator pointers and:
//   OperatorAction(u_in, rhs_out): M(u) = sum_i M_i(u).  Each sub-operator OVERWRITES its
//     output (the ParabolicOperator contract), so the composite ACCUMULATES: it writes
//     first sub-operator's action into rhs_out, then for each remaining sub-operator
//     evaluates M_i into a scratch register and ADDS it.  Operators that touch the SAME
//     conserved component (e.g. isotropic + anisotropic conduction -> energy) therefore
//     sum; operators that touch disjoint components combine without interference because
//     every operator already writes 0 into the components it does not evolve (the
//     frozen-background rule).
//   ExplicitStableDt(): min_i dt_exp_i (RKL2 stage count is set by the single stiffest
//     term) -- AND the global minimum-dt MPI all-reduce lives HERE, behind the stable-dt
//     query.  Each sub-operator's ExplicitStableDt is a per-RANK Kokkos reduction with no
//     MPI; the composite reduces across operators on the rank and then MPI_Allreduce-MINs
//     across ranks, so every rank derives the SAME RKL2 stage count.  This is also what
//     removes the per-operator MPI deadlock risk (a spatially-varying dt_exp giving ranks
//     different substage counts -> the synchronous SyncParabolicGhosts busy-wait hangs):
//     with one global dt_exp all ranks agree on the stage count (see #113 learnings).
//   ApplyBoundary(u): refresh ghost zones for every sub-operator's stencil (each sub-op's
//     physical-BC fill + its own SyncParabolicGhosts exchange), so each M_i(.) evaluation
//     inside the RKL2 recursion sees a boundary-consistent trial state.  (Fusing the
//     per-substage halo into one shared exchange is the ADR-0009 performance follow-on,
//     this correctness-first wiring.)

#include <vector>

#include "athena.hpp"   // Real, DvceArray5D
#include "driver/parabolic_operator.hpp"

namespace parabolic {

//----------------------------------------------------------------------------------------
//! \class CompositeParabolicOperator
//! \brief Summed-action / min-dt parabolic operator over a set of sub-operators (#114).

class CompositeParabolicOperator : public ParabolicOperator {
 public:
  CompositeParabolicOperator() = default;
  ~CompositeParabolicOperator() override = default;

  //! \brief Register a sub-operator (NON-owning; the caller keeps ownership/lifetime).
  void AddOperator(ParabolicOperator *op) {
    if (op != nullptr) { ops_.push_back(op); }
  }

  //! \brief Number of registered sub-operators.
  int num_operators() const { return static_cast<int>(ops_.size()); }

  //! \brief M(u) = sum_i M_i(u): the first sub-operator overwrites rhs_out, each later
  //! one is evaluated into a scratch register and ADDED (accumulating where two operators
  //! write the same conserved component).  With no sub-operators, rhs_out is zeroed.
  void OperatorAction(const DvceArray5D<Real> &u_in,
                      DvceArray5D<Real> &rhs_out) override;

  //! \brief min_i dt_exp_i, then the global minimum-dt MPI all-reduce across all ranks.
  Real ExplicitStableDt() override;

  //! \brief Refresh ghost zones for every sub-operator's stencil (in registration order).
  void ApplyBoundary(DvceArray5D<Real> &u) override;

 private:
  std::vector<ParabolicOperator *> ops_;  //!< non-owning sub-operators
  DvceArray5D<Real> scratch_;             //!< accumulation scratch for the summed action
};

}  // namespace parabolic

#endif  // DRIVER_COMPOSITE_PARABOLIC_OPERATOR_HPP_
