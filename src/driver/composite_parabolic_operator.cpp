//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file composite_parabolic_operator.cpp
//! \brief Implementation of CompositeParabolicOperator (summed action, min-dt, global
//! min-dt MPI all-reduce) -- issue [B1]/#114, ADR-0009.  See the header for the contract.

#include <limits>

#include "athena.hpp"
#include "driver/composite_parabolic_operator.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace parabolic {

//----------------------------------------------------------------------------------------
//! \fn void CompositeParabolicOperator::OperatorAction()
//! \brief M(u) = sum_i M_i(u).  The first sub-operator OVERWRITES rhs_out (the
//! ParabolicOperator contract); each remaining sub-operator is evaluated into a scratch
//! register and ADDED.  Operators touching the same conserved component therefore sum;
//! the frozen-background rule (each writes 0 into components it does not evolve) makes
//! disjoint operators combine without interference.

void CompositeParabolicOperator::OperatorAction(const DvceArray5D<Real> &u_in,
                                                DvceArray5D<Real> &rhs_out) {
  const int nmb  = rhs_out.extent_int(0);
  const int nvar = rhs_out.extent_int(1);
  const int n3   = rhs_out.extent_int(2);
  const int n2   = rhs_out.extent_int(3);
  const int n1   = rhs_out.extent_int(4);

  // No sub-operators: M(u) = 0 (a no-op composite leaves the field frozen).
  if (ops_.empty()) {
    par_for("composite_zero", DevExeSpace(),
    0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      rhs_out(m, v, k, j, i) = 0.0;
    });
    return;
  }

  // First sub-operator overwrites rhs_out with M_0(u).
  ops_[0]->OperatorAction(u_in, rhs_out);
  if (ops_.size() == 1) { return; }

  // Lazily (re)allocate the accumulation scratch to the field shape.
  if (scratch_.extent_int(0) != nmb || scratch_.extent_int(1) != nvar ||
      scratch_.extent_int(2) != n3  || scratch_.extent_int(3) != n2 ||
      scratch_.extent_int(4) != n1) {
    Kokkos::realloc(scratch_, nmb, nvar, n3, n2, n1);
  }
  auto scr = scratch_;

  // Remaining sub-operators: evaluate M_i into scratch and ADD into rhs_out.
  for (std::size_t op = 1; op < ops_.size(); ++op) {
    ops_[op]->OperatorAction(u_in, scratch_);
    par_for("composite_accum", DevExeSpace(),
    0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      rhs_out(m, v, k, j, i) += scr(m, v, k, j, i);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CompositeParabolicOperator::ExplicitStableDt()
//! \brief min_i dt_exp_i across the sub-operators, then the global minimum-dt MPI
//! all-reduce across all ranks.  The RKL2 stage count is set by the single stiffest term
//! (the smallest dt_exp), and a SINGLE global reduction lives here so every rank derives
//! the SAME stage count (correctness AND the per-operator MPI-deadlock fix, see #113).

Real CompositeParabolicOperator::ExplicitStableDt() {
  Real dt = static_cast<Real>(std::numeric_limits<float>::max());
  for (auto *op : ops_) {
    dt = (dt < op->ExplicitStableDt()) ? dt : op->ExplicitStableDt();
  }
#if MPI_PARALLEL_ENABLED
  // global minimum dt over all MPI ranks (the same pattern as Mesh::NewTimeStep).
  MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
#endif
  return dt;
}

//----------------------------------------------------------------------------------------
//! \fn void CompositeParabolicOperator::ApplyBoundary()
//! \brief Refresh ghost zones for every sub-operator's stencil (in registration order),
//! so each M_i(.) evaluation inside the RKL2 recursion sees a boundary-consistent state.

void CompositeParabolicOperator::ApplyBoundary(DvceArray5D<Real> &u) {
  for (auto *op : ops_) {
    op->ApplyBoundary(u);
  }
}

}  // namespace parabolic
