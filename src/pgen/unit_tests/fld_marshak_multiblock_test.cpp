//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_marshak_multiblock_test.cpp
//  \brief Unit test: the grey FLD radiation operator (FLDGreyOperator + RKL2 STS), wired
//  operator-split into the MHD timestep (#110/[A3]), reproduces the analytic Marshak wave
//  while running across MeshBlock, MPI-rank, and AMR coarse/fine boundaries via the
//  per-substage ghost-exchange helper MeshBoundaryValuesCC::SyncParabolicGhosts
//  (#108/[A1], ADR-0001/ADR-0009).
//
//  This is the multi-block / AMR / multi-rank verification of the grey-FLD wiring.  The
//  single-block analytic correctness of the same operator (erfc Marshak profile,
//  substage-count sweep) is pinned by verification/test_verify_marshak_fld; here we add
//  ONLY the thing #110 wires in: the inter-substage neighbor ghost exchange that lets the
//  operator span >1 MeshBlock, >1 MPI rank, and a static-refinement (SMR/AMR) mesh
//  changing the physics.
//
//  ORACLE (Layer 1, analytic).  An optically-thick (chi large => Larsen limiter in its
//  diffusion limit lambda=1/3) cold slab E_r = e_floor on the GLOBAL domain
//  [x1min,x1max] is heated from the inner-x1 face held at the Dirichlet source
//  E_r(x1min) = e_source.  In the equilibrium-diffusion limit grey FLD reduces to linear
//  diffusion with D = c/(3 chi), whose half-space solution into a cold medium is the
//  complementary-error-function profile
//      E_r(x,t) = e_floor + (e_source - e_floor) erfc((x - x1min)/(2 sqrt(D t))).
//  The wave must propagate from the inner-x1 source THROUGH the internal block (and AMR
//  coarse/fine) boundaries to reproduce this profile.  Run insulated-only (no exchange)
//  and the wave cannot leave the source block -- the downstream blocks stay at the cold
//  floor -- so the erfc-match check below is RED without SyncParabolicGhosts and GREEN
//  with it.
//
//  Batteries (all reductions are MPI_Allreduce'd so every rank checks the SAME globals):
//   (1) the grey-FLD operator-split slot is wired (pmhd->pfld_op built, erad allocated)
//       and the run really is multi-block (nmb_total > 1).
//   (2) after advancing a fixed time by RKL2 STS through the production task body
//       (parabolic::OperatorSplitStep on the live MHD erad), the global L-inf/L1 error vs
//       the analytic erfc Marshak wave is small -- the wave is reproduced ACROSS the
//       (and, under MPI/AMR, rank / coarse-fine) boundaries.
//   (3) the wave actually CROSSED the first internal boundary (the downstream block
//       heated above the cold floor) -- the discriminating cross-block transport check.
//   (4) an INSULATED grey-FLD diffusion (separate field, e_source < 0) conserves total
//       radiation energy across the decomposition.
//
//  Built/run by tst/test_suite/unit_tests/test_verify_fld_marshak_multiblock_cpu.py
//  (4 MeshBlocks, one rank), ..._amr_cpu.py (static-SMR multilevel mesh), and
//  ..._mpicpu.py (4 MeshBlocks across MPI ranks).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::erfc, std::sqrt, std::fabs
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
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
//! \brief Multi-block / AMR / multi-rank operator-split grey-FLD Marshak verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_marshak_multiblock_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // (1) requires the grey-FLD operator-split slot wired into MHD, running on >1 MeshBlock
  bool have_mhd = (pmbp->pmhd != nullptr);
  test.CheckTrue(have_mhd, "MHD block constructed");
  bool have_op = have_mhd && (pmbp->pmhd->pfld_op != nullptr);
  test.CheckTrue(have_op,
                 "operator-split grey FLD operator built (mhd/fld_operator_split)");
  test.CheckTrue(pmy_mesh_->nmb_total > 1,
                 "more than one MeshBlock (cross-block ghost exchange is exercised)");
  if (!have_op) { test.Finish(); return; }
  auto *pmhd = pmbp->pmhd;

  // <time> parabolic_integrator selector parses to the RKL2 STS backend
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- problem constants (GLOBAL mesh); FLD params come from <mhd> (single source of
  //      truth -- the same values MHD built pfld_op from), IC/time from <problem) ----
  const Real c_light = pin->GetReal("mhd", "fld_c_light");
  const Real chi     = pin->GetReal("mhd", "fld_chi");
  const Real e_src   = pin->GetReal("mhd", "fld_e_source");
  const Real e_floor = pin->GetOrAddReal("problem", "e_floor", 0.1);
  const Real tlim    = pin->GetOrAddReal("problem", "tlim_fld", 50.0);
  const int  nsuper  = pin->GetOrAddInteger("problem", "n_super", 20);
  const Real x1min   = pin->GetReal("mesh", "x1min");
  const Real x1max   = pin->GetReal("mesh", "x1max");
  const Real L       = x1max - x1min;
  const Real contrast = e_src - e_floor;
  // optically-thick equilibrium-diffusion coefficient and erfc half-width.
  const Real D = c_light/(3.0*chi);
  const Real half_width = 2.0*std::sqrt(D*tlim);
  test.CheckTrue(e_src >= 0.0 && contrast > 0.0,
                 "Dirichlet source hotter than the cold floor (Marshak setup)");

  // ---- seed the live MHD radiation field erad = e_floor (cold slab), incl. ghosts ----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto erad = pmhd->erad;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n3 = erad.extent_int(2);
  const int n2 = erad.extent_int(3);
  const int n1 = erad.extent_int(4);
  Real efl = e_floor;
  par_for("fmb_ic", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    erad(m,0,k,j,i) = efl;
  });

  // also seed the MHD conserved state with a trivial uniform gas so the driver's initial
  // ConToPrim / NewTimeStep see a valid state (no actual integration: nlim = tlim = 0).
  auto u0 = pmhd->u0;
  auto b0 = pmhd->b0;
  const int nvaru = u0.extent_int(1);
  par_for("fmb_u0", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int n=0; n<nvaru; ++n) { u0(m,n,k,j,i) = 0.0; }
    u0(m,IDN,k,j,i) = 1.0;
    u0(m,IEN,k,j,i) = 1.0;
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = 0.0;
    if (i==n1-1) { b0.x1f(m,k,j,i+1) = 0.0; }
    if (j==n2-1) { b0.x2f(m,k,j+1,i) = 0.0; }
    if (k==n3-1) { b0.x3f(m,k+1,j,i) = 0.0; }
  });

  // ---- advance a fixed time by RKL2 STS through the production task body ----
  // parabolic::OperatorSplitStep is EXACTLY what MHD::OperatorSplitFLD runs; its
  // per-substage ApplyBoundary delegates to SyncParabolicGhosts for the cross-block /
  // rank / coarse-fine exchange.  Use the operator MHD itself constructed (pfld_op).
  FLDGreyOperator &op = *(pmhd->pfld_op);
  const Real dt_exp = op.ExplicitStableDt();
  test.CheckTrue(dt_exp > 0.0, "explicit FLD diffusion dt is positive");
  const Real dt_super = tlim/static_cast<Real>(nsuper);
  test.CheckTrue(integ.NumStages(dt_super, dt_exp) > 2,
                 "STS superstep uses >2 RKL2 substages (dt_super >> dt_exp)");
  for (int n = 0; n < nsuper; ++n) {
    parabolic::OperatorSplitStep(integ, op, erad, dt_super);
  }

  // ---- (2) global L-inf/L1 error vs the analytic erfc Marshak wave, in an interior
  //      window that avoids the immediate source cell and the far cold tail ----
  const Real win_lo = x1min + 0.04*L;
  const Real win_hi = x1min + 0.85*half_width;   // out to ~erfc(0.85/2)~0.55 contrast
  Real local_linf = 0.0, local_l1 = 0.0;
  int  local_cnt = 0;
  Real local_down_peak = 0.0;
  {
    auto h_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), erad);
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real x1 = size.h_view(m).x1min
                      + (static_cast<Real>(i - is) + 0.5)*size.h_view(m).dx1;
            if (x1 > win_lo && x1 < win_hi) {
              Real z = (x1 - x1min)/half_width;
              Real analytic = e_floor + contrast*std::erfc(z);
              Real err = std::fabs(h_e(m,0,k,j,i) - analytic);
              local_linf = std::fmax(local_linf, err);
              local_l1 += err;
              local_cnt += 1;
            }
            // downstream-of-the-first-internal-boundary heating (cross-block transport):
            // sample a band a third of the way into the domain, past block 0.
            if (x1 > x1min + 0.30*L && x1 < x1min + 0.45*L) {
              local_down_peak = std::fmax(local_down_peak,
                                          h_e(m,0,k,j,i) - e_floor);
            }
          }
        }
      }
    }
  }
  const Real linf = GlobalReduce(local_linf, true);
  const Real l1sum = GlobalReduce(local_l1, false);
  const Real cnt = GlobalReduce(static_cast<Real>(local_cnt), false);
  const Real l1 = (cnt > 0.0) ? l1sum/cnt : 0.0;
  test.CheckTrue(linf < 5.0e-2*contrast,
                 "grey FLD reproduces the analytic Marshak Linf across blocks/AMR/ranks");
  test.CheckTrue(l1 < 1.0e-2*contrast,
                 "grey FLD reproduces the analytic Marshak L1 across blocks/AMR/ranks");

  // ---- (3) the wave crossed the first internal block boundary (RED w/o exchange) ----
  const Real down_peak = GlobalReduce(local_down_peak, true);
  test.CheckTrue(down_peak > 0.10*contrast,
                 "Marshak wave crossed the internal block boundary (cross-block)");

  // ---- (4) insulated grey-FLD diffusion conserves total radiation energy across the
  //      decomposition (a separate field + insulated operator, e_source < 0) ----
  DvceArray5D<Real> econs("fmb_econs", erad.extent_int(0), 1, n3, n2, n1);
  const Real PI = std::acos(-1.0);
  const Real kcons = 2.0*PI/L;   // one smooth period across the global domain
  par_for("fmb_cons_ic", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    econs(m,0,k,j,i) = 1.0 + 0.1*Kokkos::cos(kcons*(x1 - x1min));
  });
  // e_source < 0 => insulated box (zero-gradient inner-x1 too)
  FLDGreyOperator op_ins(pmbp, pin, econs, c_light, chi, 2.0, -1.0);
  // total radiation ENERGY = sum(E_i * cell_volume): the conserved quantity.  Volume
  // weighting matters on a multilevel mesh, where fine cells are smaller than coarse ones
  // (an unweighted cell sum is not conserved when diffusion moves energy across a c/f
  // boundary, even though the volume-weighted total is).
  auto interior_sum = [&]() -> Real {
    auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), econs);
    Real s = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      Real vol = size.h_view(m).dx1 * size.h_view(m).dx2 * size.h_view(m).dx3;
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) { s += h_c(m,0,k,j,i)*vol; }
        }
      }
    }
    return s;
  };
  const Real cons_init = GlobalReduce(interior_sum(), false);
  const Real dt_cons = op_ins.ExplicitStableDt();
  for (int n = 0; n < 4; ++n) {
    parabolic::OperatorSplitStep(integ, op_ins, econs, 5.0*dt_cons);
  }
  const Real cons_final = GlobalReduce(interior_sum(), false);
  test.CheckNear(cons_final, cons_init, 1.0e-10, 1.0e-12,
                 "insulated grey FLD conserves total radiation energy across blocks");

  if (global_variable::my_rank == 0) {
    std::cout << "### fld_marshak_multiblock_test: nmb_total=" << pmy_mesh_->nmb_total
              << " multilevel=" << pmy_mesh_->multilevel
              << " nranks=" << global_variable::nranks
              << " linf/contrast=" << (linf/contrast)
              << " l1/contrast=" << (l1/contrast)
              << " down_peak/contrast=" << (down_peak/contrast) << std::endl;
  }

  test.Finish();
  // All checks have run on the live state; the athinput sets nlim = tlim = 0, so we
  // return here to shut down cleanly through main (MPI_Finalize) with no integration.
  return;
}
