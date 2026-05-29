//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file composite_parabolic_test.cpp
//  \brief Unit test: parabolic::CompositeParabolicOperator runs SEVERAL stiff parabolic
//  operators as a single RKL2 super-step over their SUM (issue [B1]/#114, ADR-0009).
//
//  This verifies the three things #114 builds, on a 1D static uniform-density medium
//  carrying the global discrete cosine diffusion eigenmode (decomposed into MeshBlocks,
//  under MPI across ranks):
//   - SUMMED ACTION with a known combined solution.  Two analytic isotropic conduction
//     operators with DIFFERENT conductivities kappa1, kappa2 share the SAME conserved
//     energy field.  Each is a parabolic eigenoperator of the cosine mode with eigenvalue
//     lambda_a = -(2 D_a/dx^2)(1 - cos theta), D_a = kappa_a (gamma-1)/rho.  Because both
//     touch the SAME component (energy), the composite must ACCUMULATE: M(u)=M1+M2 acts
//     as the single eigenvalue lambda1 + lambda2, NOT lambda2 (the overwrite bug).  The
//     summed STS evolution then matches exp((lambda1+lambda2) t) -- the known combined
//     solution.
//   - MIN-DT SELECTION.  CompositeParabolicOperator::ExplicitStableDt() returns the min
//     across sub-operators (the RKL2 stage count is set by the single stiffest term); the
//     stiffer (larger-kappa) operator has the smaller dt_exp.
//   - GLOBAL MIN-DT MPI ALL-REDUCE behind the stable-dt query.  A separate sub-operator
//     pair built over a PER-RANK-VARYING density makes each rank's local dt_exp differ;
//     composite's ExplicitStableDt() must return the SAME global minimum on every rank
//     (the all-reduce), <= every rank's local min, and strictly below the cross-rank
//     maximum when nranks>1 (the reduction genuinely combines ranks).
//
//  ORACLE (Layer 1, analytic).  The discrete cosine mode E_i=E0+A cos(k(x-x1min)) is an
//  exact eigenvector of the isotropic conduction operator; the SUM of two such operators
//  has eigenvalue l1+l2 and semi-discrete solution E_i(t)=E0+A cos(..) exp((l1+l2)t).
//  References: closed form (diffusion eigenmode); Meyer, Balsara & Aslam 2014 (RKL2);
//  ADR-0009 (one super-step over the summed operator, dt = min over operators).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_composite_parabolic_cpu.py (4
//  MeshBlocks on one rank) and ..._mpicpu.py (4 MeshBlocks across MPI ranks).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::exp, std::fabs, std::acos
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction_operator.hpp"
#include "driver/parabolic_integrator.hpp"
#include "driver/composite_parabolic_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
//! \enum MPI_Op_kind -- which reduction (avoids leaking the MPI_Op type when MPI is off).
enum class MPI_Op_kind { kMin, kMax, kSum };

//! \brief MPI_Allreduce a scalar across all ranks (identity on a single rank), so every
//! rank records the same global value and Finish() exits consistently everywhere.
Real GlobalAllreduce(Real local, MPI_Op_kind kind) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Op op = (kind == MPI_Op_kind::kMax) ? MPI_MAX
            : (kind == MPI_Op_kind::kSum) ? MPI_SUM : MPI_MIN;
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, op, MPI_COMM_WORLD);
#else
  (void)kind;
#endif
  return global;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief CompositeParabolicOperator (summed action, min-dt, global reduction) test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("composite_parabolic_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // requires hydro (for the conserved-field sizing + ideal-gas gamma)
  bool have_hydro = (pmbp->phydro != nullptr);
  test.CheckTrue(have_hydro, "hydro block constructed (conserved-field sizing)");
  if (!have_hydro) { test.Finish(); return; }
  auto *phydro = pmbp->phydro;

  // selector: <time> parabolic_integrator parses to the RKL2 STS backend
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- problem constants (GLOBAL mesh) ----
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const int Nglob = pin->GetInteger("mesh", "nx1");
  const Real L = x1max - x1min;
  const Real dx = L/static_cast<Real>(Nglob);

  const Real rho0 = 1.0;
  const Real E0 = 1.0;                        // background energy (T0 = gm1 E0 > 0)
  const Real amp = 1.0e-2;                    // eigenmode amplitude (small => positive E)
  // TWO conductivities -> two operators with DIFFERENT eigenvalues and DIFFERENT dt_exp,
  // so the summed action (l1+l2) is distinct from either alone and the min-dt is the
  // stiffer (kappa2) operator.  kappa2 > kappa1.
  const Real kappa1 = 0.05;
  const Real kappa2 = 0.15;
  // m=6 over 4 blocks of 16 cells puts the three internal block faces on the mode's steep
  // zero-crossings (not extrema), so a broken cross-block exchange would be visible (the
  // #108 discriminator); here the exchange is correct so the analytic match is tight.
  const int m_mode = 6;
  const Real PI = std::acos(-1.0);
  const Real kwave = m_mode*PI/L;                          // continuous wavenumber
  const Real theta = m_mode*PI/static_cast<Real>(Nglob);   // discrete phase per cell
  const Real D1 = kappa1*gm1/rho0, D2 = kappa2*gm1/rho0;
  const Real lambda1 = -(2.0*D1/(dx*dx))*(1.0 - std::cos(theta));
  const Real lambda2 = -(2.0*D2/(dx*dx))*(1.0 - std::cos(theta));
  const Real lambda_sum = lambda1 + lambda2;               // combined eigenvalue

  // ---- seed the live hydro conserved field u0 with the global cosine eigenmode ----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto u0 = phydro->u0;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n3 = u0.extent_int(2);
  const int n2 = u0.extent_int(3);
  const int n1 = u0.extent_int(4);
  par_for("comp_ic", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = E0 + amp*Kokkos::cos(kwave*(x1 - x1min));
  });

  // the two sub-operators (different kappa) over the SAME live field, and the composite
  ConductionOperator op1(pmbp, pin, u0, kappa1, gamma);
  ConductionOperator op2(pmbp, pin, u0, kappa2, gamma);
  parabolic::CompositeParabolicOperator comp;
  comp.AddOperator(&op1);
  comp.AddOperator(&op2);
  test.CheckTrue(comp.num_operators() == 2, "composite holds the two registered ops");

  // conserved-field work arrays
  DvceArray5D<Real> u_work("u_work", nmb1+1, u0.extent_int(1), n3, n2, n1);
  DvceArray5D<Real> rhs("rhs", nmb1+1, u0.extent_int(1), n3, n2, n1);

  // ===== (A) summed action == (l1+l2)*(E-E0) on the eigenmode (machine precision) =====
  Kokkos::deep_copy(u_work, u0);
  comp.ApplyBoundary(u_work);
  comp.OperatorAction(u_work, rhs);
  Real local_errA = 0.0, local_excess = 0.0, local_rho_rhs = 0.0;
  {
    auto h_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs);
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real x1 = size.h_view(m).x1min
                      + (static_cast<Real>(i - is) + 0.5)*size.h_view(m).dx1;
            Real pert = amp*std::cos(kwave*(x1 - x1min));   // E_i - E0
            Real expected = lambda_sum*pert;                // SUM of the two eigenvalues
            local_errA = std::fmax(local_errA, std::fabs(h_rhs(m,IEN,k,j,i) - expected));
            // distinctness: how far the sum is from a single (kappa2) operator's action
            local_excess = std::fmax(local_excess,
                                     std::fabs(h_rhs(m,IEN,k,j,i) - lambda2*pert));
            local_rho_rhs = std::fmax(local_rho_rhs, std::fabs(h_rhs(m,IDN,k,j,i)));
            local_rho_rhs = std::fmax(local_rho_rhs, std::fabs(h_rhs(m,IM1,k,j,i)));
          }
        }
      }
    }
  }
  const Real max_errA = GlobalAllreduce(local_errA, MPI_Op_kind::kMax);
  const Real max_excess = GlobalAllreduce(local_excess, MPI_Op_kind::kMax);
  const Real max_rho_rhs = GlobalAllreduce(local_rho_rhs, MPI_Op_kind::kMax);
  test.CheckTrue(max_errA < 1.0e-9*std::fabs(lambda_sum)*amp + 1.0e-13,
                 "composite action M(u) == (l1+l2)*(E-E0) (summed, not overwrite)");
  test.CheckTrue(max_excess > 0.5*std::fabs(lambda1)*amp,
                 "summed action is DISTINCT from a single operator (accumulate)");
  test.CheckTrue(max_rho_rhs < 1.0e-30,
                 "composite writes 0 into non-energy components (frozen background)");

  // ===== (B) min-dt selection: composite dt_exp == min over sub-operators ==============
  const Real e1 = op1.ExplicitStableDt();
  const Real e2 = op2.ExplicitStableDt();
  const Real comp_dt = comp.ExplicitStableDt();      // min over ops, then global reduce
  const Real expected_min = GlobalAllreduce(std::fmin(e1, e2), MPI_Op_kind::kMin);
  test.CheckTrue(e2 < e1,
                 "stiffer (larger-kappa) operator has the smaller explicit dt");
  test.CheckNear(comp_dt, expected_min, 1.0e-12, 1.0e-30,
                 "composite ExplicitStableDt == min over sub-operators");

  // ===== (C) empty composite: M(u)=0, num_operators=0, dt_exp = +inf-like ==============
  parabolic::CompositeParabolicOperator empty;
  test.CheckTrue(empty.num_operators() == 0, "default composite has no sub-operators");
  empty.OperatorAction(u_work, rhs);
  Real local_zero = 0.0;
  {
    auto h_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs);
    for (int m = 0; m <= nmb1; ++m) {
      for (int i = is; i <= ie; ++i) {
        local_zero = std::fmax(local_zero, std::fabs(h_rhs(m,IEN,ks,js,i)));
      }
    }
  }
  test.CheckTrue(GlobalAllreduce(local_zero, MPI_Op_kind::kMax) < 1.0e-30,
                 "empty composite action is identically zero");
  test.CheckTrue(empty.ExplicitStableDt() > 1.0e30,
                 "empty composite imposes no parabolic dt limit (dt_exp ~ +inf)");

  // ===== (D) combined STS evolution matches exp((lambda1+lambda2) t) ===================
  // advance the live field by the production task body over the composite (one super-step
  // per call covers BOTH operators; the stage count comes from the stiffer dt_exp).
  test.CheckTrue(integ.NumStages(0.7/std::fabs(lambda_sum)/8.0, comp_dt) > 2,
                 "STS superstep over the composite uses >2 RKL2 substages");
  // interior reductions (energy sum, peak excess) on this rank's blocks
  auto interior_sum_energy = [&]() -> Real {
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u0);
    Real s = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      for (int i = is; i <= ie; ++i) { s += h_u(m,IEN,ks,js,i); }
    }
    return s;
  };
  const Real e_init = GlobalAllreduce(interior_sum_energy(), MPI_Op_kind::kSum);
  const Real Tfinal = 0.7/std::fabs(lambda_sum);   // mode decays to ~exp(-0.7) ~ 0.5
  const int nsuper = 8;
  const Real dt_super = Tfinal/static_cast<Real>(nsuper);
  for (int n = 0; n < nsuper; ++n) {
    parabolic::OperatorSplitStep(integ, comp, u0, dt_super);
  }
  const Real decay = std::exp(lambda_sum*Tfinal);
  Real local_errD = 0.0, local_peak = 0.0;
  {
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u0);
    for (int m = 0; m <= nmb1; ++m) {
      for (int i = is; i <= ie; ++i) {
        Real x1 = size.h_view(m).x1min
                  + (static_cast<Real>(i - is) + 0.5)*size.h_view(m).dx1;
        Real analytic = E0 + amp*std::cos(kwave*(x1 - x1min))*decay;
        local_errD = std::fmax(local_errD, std::fabs(h_u(m,IEN,ks,js,i) - analytic));
        local_peak = std::fmax(local_peak, std::fabs(h_u(m,IEN,ks,js,i) - E0));
      }
    }
  }
  const Real max_errD = GlobalAllreduce(local_errD, MPI_Op_kind::kMax);
  const Real peak_final = GlobalAllreduce(local_peak, MPI_Op_kind::kMax);
  test.CheckTrue(max_errD < 2.0e-2*amp,
                 "summed STS reproduces analytic exp((lambda1+lambda2) t) decay");
  test.CheckTrue(peak_final < 0.9*amp,
                 "summed STS diffuses the mode (perturbation decays)");
  const Real e_final = GlobalAllreduce(interior_sum_energy(), MPI_Op_kind::kSum);
  test.CheckNear(e_final, e_init, 0.0, 1.0e-10*std::fabs(e_init),
                 "summed STS conserves total energy (both operators insulated)");

  // ===== (E) GLOBAL min-dt reduction across ranks ======================================
  // A separate operator pair over a PER-RANK-VARYING density makes each rank's local
  // dt_exp differ; the composite's ExplicitStableDt returns the SAME global min on all.
  const Real rho_rank = rho0*(1.0 + static_cast<Real>(global_variable::my_rank));
  DvceArray5D<Real> u_rank("u_rank", nmb1+1, u0.extent_int(1), n3, n2, n1);
  par_for("comp_rank_ic", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u_rank(m,IDN,k,j,i) = rho_rank;
    u_rank(m,IM1,k,j,i) = 0.0;
    u_rank(m,IM2,k,j,i) = 0.0;
    u_rank(m,IM3,k,j,i) = 0.0;
    u_rank(m,IEN,k,j,i) = E0;
  });
  ConductionOperator opR1(pmbp, pin, u_rank, kappa1, gamma);
  ConductionOperator opR2(pmbp, pin, u_rank, kappa2, gamma);
  parabolic::CompositeParabolicOperator compR;
  compR.AddOperator(&opR1);
  compR.AddOperator(&opR2);
  const Real local_min = std::fmin(opR1.ExplicitStableDt(), opR2.ExplicitStableDt());
  const Real compR_dt = compR.ExplicitStableDt();        // global all-reduce lives here
  const Real global_min = GlobalAllreduce(local_min, MPI_Op_kind::kMin);
  const Real global_max = GlobalAllreduce(local_min, MPI_Op_kind::kMax);
  test.CheckNear(compR_dt, global_min, 1.0e-12, 1.0e-30,
                 "composite ExplicitStableDt == global min over ranks (MPI all-reduce)");
  test.CheckTrue(compR_dt <= local_min*(1.0 + 1.0e-12),
                 "global min-dt never exceeds this rank's local min");
  test.CheckTrue(global_variable::nranks == 1 || (global_min < global_max),
                 "per-rank dt_exp genuinely varies across ranks (non-trivial reduce)");

  if (global_variable::my_rank == 0) {
    std::cout << "### composite_parabolic_test: nmb_total=" << pmy_mesh_->nmb_total
              << " nranks=" << global_variable::nranks
              << " l1=" << lambda1 << " l2=" << lambda2
              << " comp_dt=" << comp_dt << " global_min=" << global_min
              << " global_max=" << global_max
              << " decayLinf/amp=" << (max_errD/amp) << std::endl;
  }

  test.Finish();
  // All checks have run on the live/local state; the athinput sets nlim=tlim=0, so
  // returning here shuts down cleanly through main (MPI_Finalize); no time integration.
  return;
}
