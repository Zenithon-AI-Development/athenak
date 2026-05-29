//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file conduction_multiblock_test.cpp
//  \brief Unit test: the operator-split isotropic conduction operator (ConductionOperator
//  + RKL2 STS) reproduces the analytic cosine diffusion decay while running across
//  MeshBlock and MPI-rank boundaries, via the shared per-substage ghost-exchange helper
//  MeshBoundaryValuesCC::SyncParabolicGhosts (issue [A1]/#108, ADR-0001/ADR-0009).
//
//  This is the multi-block / multi-rank verification of the framework canary.  The
//  single-block analytic correctness of the same operator (action = lambda(E-E0),
//  exp(lambda t) decay, 2nd-order convergence, conservation, stability) is pinned by
//  unit_tests/parabolic_conduction_test; here we add ONLY the thing #108 builds: the
//  inter-substage neighbor ghost exchange that lets the operator span >1 MeshBlock and >1
//  MPI rank without changing the physics.
//
//  ORACLE (Layer 1, analytic).  A static (v=0), uniform-density (rho=1) gas on the GLOBAL
//  domain [x1min,x1max] of N cells carries the discrete cosine eigenmode
//      E_i = E0 + A cos(theta (ig + 1/2)),   theta = m pi / N,   ig = global cell index,
//  which (because cos(theta(ig+1/2)) = cos(k (x - x1min)), k = m pi / L) is identical to
//  the continuous cosine sampled at cell centers, so it can be seeded per block from each
//  cell's physical x1.  On the insulated global box it is an EXACT eigenvector of the
//  discrete conduction operator with eigenvalue lambda = -(2D/dx^2)(1 - cos theta),
//  D = kappa(gamma-1)/rho, so the semi-discrete solution is E_i(t) = E0 + A cos(...)
//  exp(lambda t).  The mode is GLOBAL: only the correct cross-block ghost exchange lets a
//  block-decomposed run reproduce it.  Run insulated-only (no exchange) and each block
//  decays as its own insulated sub-box -- a kinked profile -- so the decay-match check
//  below is RED without SyncParabolicGhosts and GREEN with it.
//
//  Batteries (all reductions are MPI_Allreduce'd so every rank checks the SAME globals):
//   (1) The operator-split operator is built by hydro (conduction_operator_split) and the
//       run really is multi-block (nmb_total > 1).
//   (2) After advancing a fixed time by RKL2 STS through the production task body
//       (parabolic::OperatorSplitStep on the live hydro u0), the global L-inf error vs
//       the analytic exp(lambda t) decay is small -- the decay is reproduced ACROSS block
//       (and, under MPI, rank) boundaries.
//   (3) The insulated global box conserves total energy across the decomposition.
//   (4) Diffusion actually happened (the perturbation decayed).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_conduction_multiblock_cpu.py (4
//  MeshBlocks on one rank) and ..._mpicpu.py (4 MeshBlocks across MPI ranks).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::exp, std::fabs, std::acos, std::ceil
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/conduction_operator.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
//! \brief MPI_Allreduce a scalar across all ranks (identity on a single rank), so every
//! rank records the same global value and Finish() exits consistently everywhere.
Real GlobalReduce(Real local, bool want_max) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Op op = want_max ? MPI_MAX : MPI_SUM;
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, op, MPI_COMM_WORLD);
#else
  (void)want_max;
#endif
  return global;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Multi-block / multi-rank operator-split conduction verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("conduction_multiblock_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // (1) requires conduction-enabled, operator-split hydro running on >1 MeshBlock
  bool have_hydro = (pmbp->phydro != nullptr);
  test.CheckTrue(have_hydro, "hydro block constructed");
  bool have_op = have_hydro && (pmbp->phydro->pcond_op != nullptr);
  test.CheckTrue(have_op,
                 "operator-split conduction operator built (conduction_operator_split)");
  test.CheckTrue(pmy_mesh_->nmb_total > 1,
                 "more than one MeshBlock (cross-block ghost exchange is exercised)");
  if (!have_op) { test.Finish(); return; }
  auto *phydro = pmbp->phydro;

  // selector: <time> parabolic_integrator parses to the RKL2 STS backend
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- problem constants (GLOBAL mesh) ----
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real kappa = phydro->pcond->kappa;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const int Nglob = pin->GetInteger("mesh", "nx1");
  const Real L = x1max - x1min;
  const Real dx = L/static_cast<Real>(Nglob);

  const Real rho0 = 1.0;
  const Real E0 = 1.0;                       // background energy (T0 = gm1 E0 > 0)
  const Real amp = 1.0e-2;                   // eigenmode amplitude (small => positive E)
  // Cosine mode chosen so the four block boundaries (global cells 16/32/48) fall on the
  // mode's steep ZERO-CROSSINGS, not its extrema: m=6 => face phase m*pi/4 = 1.5pi.  A
  // mode that is a multiple of 4 would place block faces exactly on cosine extrema
  // (dT/dx=0), where the insulated-only ghost fill is accidentally correct and the test
  // would NOT discriminate the exchange.  m=6 makes the wrong ghost maximally visible.
  const int m_mode = 6;
  const Real PI = std::acos(-1.0);
  const Real kwave = m_mode*PI/L;            // continuous wavenumber
  const Real theta = m_mode*PI/static_cast<Real>(Nglob);   // discrete phase per cell
  const Real D = kappa*gm1/rho0;             // energy diffusivity
  const Real lambda = -(2.0*D/(dx*dx))*(1.0 - std::cos(theta));   // discrete eigenvalue

  // ---- seed the live hydro conserved field u0 with the global cosine eigenmode ----
  // Each cell (incl. ghosts) is seeded from its own physical x1, so the IC is the global
  // mode regardless of how the domain is split into blocks/ranks.
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
  par_for("cmb_ic", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = E0 + amp*Kokkos::cos(kwave*(x1 - x1min));
  });

  // ---- shared host reductions over the INTERIOR cells of this rank's blocks ----
  auto interior_sum_energy = [&]() -> Real {
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u0);
    Real s = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) { s += h_u(m,IEN,k,j,i); }
        }
      }
    }
    return s;
  };
  auto interior_peak_excess = [&]() -> Real {
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u0);
    Real p = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            p = std::fmax(p, std::fabs(h_u(m,IEN,k,j,i) - E0));
          }
        }
      }
    }
    return p;
  };

  const Real e_init = GlobalReduce(interior_sum_energy(), false);
  const Real peak_init = GlobalReduce(interior_peak_excess(), true);

  // ---- advance a fixed time by RKL2 STS through the production task body ----
  // parabolic::OperatorSplitStep is EXACTLY what Hydro::OperatorSplitConduction runs; its
  // per-substage ApplyBoundary delegates to SyncParabolicGhosts for the cross-block/rank
  // exchange.  Use the operator hydro itself constructed (pcond_op).
  ConductionOperator &op = *(phydro->pcond_op);
  const Real dt_exp = op.ExplicitStableDt();
  test.CheckTrue(dt_exp > 0.0, "explicit conduction dt is positive");
  const Real Tfinal = 0.7/std::fabs(lambda);   // mode decays to ~exp(-0.7) ~ 0.5
  const int nsuper = 8;
  const Real dt_super = Tfinal/static_cast<Real>(nsuper);
  test.CheckTrue(integ.NumStages(dt_super, dt_exp) > 2,
                 "STS superstep uses >2 RKL2 substages (dt_super >> dt_exp)");
  for (int n = 0; n < nsuper; ++n) {
    parabolic::OperatorSplitStep(integ, op, u0, dt_super);
  }

  // ---- (2) global L-inf error vs analytic decay; across block/rank boundaries ----
  const Real decay = std::exp(lambda*Tfinal);
  Real local_err = 0.0;
  {
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u0);
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real x1 = size.h_view(m).x1min
                      + (static_cast<Real>(i - is) + 0.5)*size.h_view(m).dx1;
            Real analytic = E0 + amp*std::cos(kwave*(x1 - x1min))*decay;
            local_err = std::fmax(local_err, std::fabs(h_u(m,IEN,k,j,i) - analytic));
          }
        }
      }
    }
  }
  const Real max_err = GlobalReduce(local_err, true);
  test.CheckTrue(max_err < 2.0e-2*amp,
                 "operator-split conduction reproduces analytic decay across blocks");

  // ---- (3) total energy conserved across the decomposition (insulated global box) ----
  const Real e_final = GlobalReduce(interior_sum_energy(), false);
  test.CheckNear(e_final, e_init, 0.0, 1.0e-10*std::fabs(e_init),
                 "operator-split conduction conserves total energy (insulated box)");

  // ---- (4) diffusion actually happened (the perturbation decayed) ----
  const Real peak_final = GlobalReduce(interior_peak_excess(), true);
  test.CheckTrue(peak_final < 0.9*peak_init,
                 "operator-split conduction diffuses (global perturbation decays)");

  if (global_variable::my_rank == 0) {
    std::cout << "### conduction_multiblock_test: nmb_total=" << pmy_mesh_->nmb_total
              << " nranks=" << global_variable::nranks
              << " max_err/amp=" << (max_err/amp)
              << " peak " << peak_init << " -> " << peak_final << std::endl;
  }

  test.Finish();
  // All checks have run on the live state; the athinput sets nlim=tlim=0, so returning
  // here lets the program shut down cleanly through main (MPI_Finalize) with no further
  // time integration.  (Finish() already std::exit'd nonzero on any failed check.)
  return;
}
