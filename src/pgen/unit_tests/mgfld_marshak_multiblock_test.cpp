//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mgfld_marshak_multiblock_test.cpp
//  \brief Unit test: the MULTIGROUP FLD radiation operator (FLDMultigroupOperator + RKL2
//  STS), wired operator-split into the MHD timestep (#111/[A4]), reproduces the analytic
//  per-group Marshak wave while running across MeshBlock, MPI-rank, and AMR coarse/fine
//  boundaries via the per-substage ghost-exchange helper
//  MeshBoundaryValuesCC::SyncParabolicGhosts (#108/[A1], ADR-0001/ADR-0007/ADR-0009).
//
//  This is the multigroup analogue of fld_marshak_multiblock_test (#110/[A3], grey).  The
//  single-block analytic correctness of the same operator (per-group erfc profiles with
//  distinct D_g) is pinned by verification/test_verify_multigroup_fld; we add ONLY the
//  thing #111 wires in: the inter-substage neighbor ghost exchange + conservative
//  refinement-boundary flux correction that let the operator span >1 MeshBlock, >1 MPI
//  rank, and a (static-SMR) coarse/fine boundary without changing the physics.
//
//  ORACLE (Layer 1, analytic).  An optically-thick cold slab E_g = e_floor (ALL groups)
//  on the GLOBAL domain [x1min,x1max] is heated from the inner-x1 face held at the
//  Dirichlet source E_g(x1min) = e_source (all groups).  In the equilibrium-diffusion
//  limit each group g reduces to linear diffusion with its OWN coefficient D_g = c/(3
//  chi_g) (chi_g from the tabulated Rosseland transport opacity), whose half-space
//  solution into a cold medium is the complementary-error-function profile
//      E_g(x,t) = e_floor + (e_source - e_floor) erfc((x - x1min)/(2 sqrt(D_g t))).
//  The groups penetrate to DIFFERENT depths, and each wave must propagate from the
//  inner-x1 source THROUGH the internal block (and AMR coarse/fine) boundaries to
//  reproduce its own profile.  Run insulated-only (no exchange) and a wave cannot leave
//  the source block, so the per-group erfc-match checks below are RED without
//  SyncParabolicGhosts and GREEN with it.
//
//  Batteries (all reductions are MPI_Allreduce'd so every rank checks the SAME globals):
//   (1) the multigroup-FLD operator-split slot is wired (pmhd->pmg_op built, erad_mg
//       allocated with ngroups>1 var width) and the run is multi-block (nmb_total>1).
//   (2) after advancing a fixed time by RKL2 STS through the production task body
//       (parabolic::OperatorSplitStep on the live MHD erad_mg), EACH group's global
//       L-inf/L1 error vs its analytic erfc Marshak wave is small -- every group is
//       reproduced ACROSS the (and, under MPI/AMR, rank / coarse-fine) boundaries.
//   (3) each group's wave actually CROSSED the first internal boundary -- the
//       discriminating cross-block transport check, with the groups ordered by
//       penetration depth (largest D_g penetrates deepest).
//   (4) an INSULATED multigroup-FLD diffusion (separate field, e_source < 0) conserves
//       total per-group radiation energy across the decomposition (volume-weighted, so it
//       also pins the conservative flux correction across a coarse/fine boundary).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_mgfld_marshak_multiblock_cpu.py
//  (4 MeshBlocks, one rank), ..._amr_cpu.py (static-SMR multilevel mesh), and
//  ..._mpicpu.py (4 MeshBlocks across MPI ranks).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cstdio>    // std::snprintf
#include <cmath>     // std::erfc, std::sqrt, std::fabs
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
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
//! \brief Multi-block/AMR/multi-rank operator-split multigroup-FLD Marshak verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("mgfld_marshak_multiblock_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // (1) requires the multigroup-FLD operator-split slot wired into MHD, on >1 MeshBlock
  bool have_mhd = (pmbp->pmhd != nullptr);
  test.CheckTrue(have_mhd, "MHD block constructed");
  bool have_op = have_mhd && (pmbp->pmhd->pmg_op != nullptr);
  test.CheckTrue(have_op,
                 "operator-split multigroup FLD operator built (mgfld_operator_split)");
  test.CheckTrue(pmy_mesh_->nmb_total > 1,
                 "more than one MeshBlock (cross-block ghost exchange is exercised)");
  if (!have_op) { test.Finish(); return; }
  auto *pmhd = pmbp->pmhd;
  FLDMultigroupOperator &op = *(pmhd->pmg_op);
  const int NG = op.ngroups();
  test.CheckTrue(NG > 1, "multiple photon-energy groups (multigroup, not grey)");
  test.CheckTrue(pmhd->erad_mg.extent_int(1) == NG,
                 "erad_mg variable width equals the group count");

  // <time> parabolic_integrator selector parses to the RKL2 STS backend
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- problem constants (GLOBAL mesh); FLD params come from <mhd> (single source of
  //      truth -- the same values MHD built pmg_op from), IC/time from <problem> ----
  const Real c_light = pin->GetReal("mhd", "mgfld_c_light");
  const Real e_src   = pin->GetReal("mhd", "mgfld_e_source");
  const Real e_floor = pin->GetOrAddReal("problem", "e_floor", 0.1);
  const Real tlim    = pin->GetOrAddReal("problem", "tlim_fld", 50.0);
  const int  nsuper  = pin->GetOrAddInteger("problem", "n_super", 20);
  const Real x1min   = pin->GetReal("mesh", "x1min");
  const Real x1max   = pin->GetReal("mesh", "x1max");
  const Real L       = x1max - x1min;
  const Real contrast = e_src - e_floor;
  test.CheckTrue(e_src >= 0.0 && contrast > 0.0,
                 "Dirichlet source hotter than the cold floor (Marshak setup)");

  // per-group extinction chi_g (precomputed in the operator from the Rosseland opacity);
  // optically-thick equilibrium-diffusion coefficient D_g = c/(3 chi_g) per group.
  auto d_chi = op.chi();
  auto h_chi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_chi);
  std::vector<Real> Dg(NG), half_width(NG);
  for (int g = 0; g < NG; ++g) {
    Dg[g] = c_light/(3.0*h_chi(g));
    half_width[g] = 2.0*std::sqrt(Dg[g]*tlim);
  }

  // ---- seed the live MHD per-group radiation field erad_mg = e_floor (cold slab) ----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto erad = pmhd->erad_mg;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n3 = erad.extent_int(2);
  const int n2 = erad.extent_int(3);
  const int n1 = erad.extent_int(4);
  Real efl = e_floor;
  par_for("mgfmb_ic", DevExeSpace(), 0, nmb1, 0, NG-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
    erad(m,g,k,j,i) = efl;
  });

  // also seed the MHD conserved state with a trivial uniform gas so the driver's initial
  // ConToPrim / NewTimeStep see a valid state (no actual integration: nlim = tlim = 0).
  auto u0 = pmhd->u0;
  auto b0 = pmhd->b0;
  const int nvaru = u0.extent_int(1);
  par_for("mgfmb_u0", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
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
  // parabolic::OperatorSplitStep is EXACTLY what MHD::OperatorSplitMultigroupFLD runs;
  // its per-substage ApplyBoundary delegates to SyncParabolicGhosts for the cross-block /
  // rank / coarse-fine exchange.  Use the operator MHD itself constructed (pmg_op).
  const Real dt_exp = op.ExplicitStableDt();
  test.CheckTrue(dt_exp > 0.0, "explicit multigroup FLD diffusion dt is positive");
  const Real dt_super = tlim/static_cast<Real>(nsuper);
  test.CheckTrue(integ.NumStages(dt_super, dt_exp) > 2,
                 "STS superstep uses >2 RKL2 substages (dt_super >> dt_exp)");
  for (int n = 0; n < nsuper; ++n) {
    parabolic::OperatorSplitStep(integ, op, erad, dt_super);
  }

  // ---- (2)+(3) per-group L-inf/L1 error vs the analytic erfc Marshak wave + cross-block
  //      transport, in an interior window avoiding the source cell and the cold tail ----
  std::vector<Real> local_linf(NG, 0.0), local_l1(NG, 0.0);
  std::vector<Real> local_cnt(NG, 0.0), local_down(NG, 0.0);
  {
    auto h_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), erad);
    for (int g = 0; g < NG; ++g) {
      const Real win_lo = x1min + 0.04*L;
      const Real win_hi = x1min + 0.85*half_width[g];   // ~erfc(0.85/2)~0.55 contrast
      for (int m = 0; m <= nmb1; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
              Real x1 = size.h_view(m).x1min
                        + (static_cast<Real>(i - is) + 0.5)*size.h_view(m).dx1;
              if (x1 > win_lo && x1 < win_hi) {
                Real z = (x1 - x1min)/half_width[g];
                Real analytic = e_floor + contrast*std::erfc(z);
                Real err = std::fabs(h_e(m,g,k,j,i) - analytic);
                local_linf[g] = std::fmax(local_linf[g], err);
                local_l1[g] += err;
                local_cnt[g] += 1.0;
              }
              // downstream-of-the-first-internal-boundary heating (cross-block transport)
              if (x1 > x1min + 0.30*L && x1 < x1min + 0.45*L) {
                local_down[g] = std::fmax(local_down[g], h_e(m,g,k,j,i) - e_floor);
              }
            }
          }
        }
      }
    }
  }
  for (int g = 0; g < NG; ++g) {
    const Real linf = GlobalReduce(local_linf[g], true);
    const Real l1sum = GlobalReduce(local_l1[g], false);
    const Real cnt = GlobalReduce(local_cnt[g], false);
    const Real l1 = (cnt > 0.0) ? l1sum/cnt : 0.0;
    const Real down = GlobalReduce(local_down[g], true);
    char msg[160];
    std::snprintf(msg, sizeof(msg),
        "group %d reproduces analytic Marshak Linf across blocks/AMR/ranks", g);
    test.CheckTrue(linf < 6.0e-2*contrast, msg);
    std::snprintf(msg, sizeof(msg),
        "group %d reproduces analytic Marshak L1 across blocks/AMR/ranks", g);
    test.CheckTrue(l1 < 1.5e-2*contrast, msg);
    std::snprintf(msg, sizeof(msg),
        "group %d Marshak wave crossed the internal block boundary (cross-block)", g);
    test.CheckTrue(down > 0.05*contrast, msg);
    if (global_variable::my_rank == 0) {
      std::cout << "### mgfld group " << g << ": D_g=" << Dg[g]
                << " linf/contrast=" << (linf/contrast)
                << " l1/contrast=" << (l1/contrast)
                << " down/contrast=" << (down/contrast) << std::endl;
    }
  }

  // ---- (4) insulated multigroup-FLD diffusion conserves total per-group radiation
  //      energy across the decomposition (volume-weighted -> also pins the conservative
  //      flux correction across a coarse/fine boundary).  Separate field + insulated
  //      operator (e_source < 0); reuse the SAME opacity table MHD built op from. ----
  DvceArray5D<Real> econs("mgfmb_econs", erad.extent_int(0), NG, n3, n2, n1);
  const Real PI = std::acos(-1.0);
  const Real kcons = 2.0*PI/L;   // one smooth period across the global domain
  par_for("mgfmb_cons_ic", DevExeSpace(), 0, nmb1, 0, NG-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    econs(m,g,k,j,i) = 1.0 + 0.1*Kokkos::cos(kcons*(x1 - x1min));
  });
  const std::string fname = pin->GetString("mhd", "mgfld_opacity_file");
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixOpacity(fname, table);
  const Real rho_bg = pin->GetReal("mhd", "mgfld_rho_bg");
  const Real te_bg  = pin->GetReal("mhd", "mgfld_te_bg");
  FLDMultigroupOperator op_ins(pmbp, pin, econs, table, c_light, rho_bg, te_bg, 2.0,
                               -1.0);
  // total radiation ENERGY = sum_g sum_i (E_g,i * cell_volume): the conserved quantity.
  // Volume weighting matters on a multilevel mesh (fine cells are smaller), where
  // conservation across a c/f boundary relies on the CorrectFlux flux correction.
  auto interior_sum = [&]() -> Real {
    auto h_c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), econs);
    Real s = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      Real vol = size.h_view(m).dx1 * size.h_view(m).dx2 * size.h_view(m).dx3;
      for (int g = 0; g < NG; ++g) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) { s += h_c(m,g,k,j,i)*vol; }
          }
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
                 "insulated multigroup FLD conserves total radiation energy");

  if (global_variable::my_rank == 0) {
    std::cout << "### mgfld_marshak_multiblock_test: nmb_total=" << pmy_mesh_->nmb_total
              << " multilevel=" << pmy_mesh_->multilevel
              << " nranks=" << global_variable::nranks
              << " ngroups=" << NG << std::endl;
  }

  test.Finish();
  // All checks have run on the live state; the athinput sets nlim = tlim = 0, so we
  // return here to shut down cleanly through main (MPI_Finalize) with no integration.
  return;
}
