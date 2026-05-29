//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mgfld_regrid_test.cpp
//  \brief Unit test: the multigroup radiation group-energy arrays (MHD::erad_mg) are
//  correctly RESTRICTED and PROLONGED across a DYNAMIC AMR regrid, so the operator-split
//  multigroup FLD physics survives re-refinement (#111/[A4], ADR-0009).
//
//  This is the dynamic-regrid array-registration gate that the static-SMR Marshak test
//  (mgfld_marshak_multiblock on the AMR mesh) cannot reach: a static mesh never regrids,
//  so the standalone erad_mg array is never re-mapped.  #110/[A3] used static
//  SMR for exactly this reason; #111 registers erad_mg with the AMR regrid machinery
//  (mesh_refinement.cpp DerefineCCSameRank / CopyForRefinementCC / RefineCC and, across
//  ranks, load_balance.cpp Pack/UnpackAMRBuffersCC) so the per-group field is
//  conservatively restricted/prolonged whenever the mesh refines or de-refines.
//
//  DESIGN.  Unlike the nlim=0 Marshak tests (which step the operator by hand in
//  UserProblem), this test RUNS THE DRIVER (nlim>0) so the standard adaptive-mesh-
//  refinement cycle actually fires:
//    * a uniform, static gas (v=0, uniform rho/p, B=0) so the hydro does nothing and the
//      timestep is well-behaved -- the only physics that matters is the operator-split
//      multigroup FLD diffusing the radiation field every step on the changing mesh;
//    * an INSULATED multigroup FLD operator (mhd/mgfld_e_source < 0) seeded with a smooth
//      per-group profile, so the conserved quantity is the volume-weighted total
//      energy sum_g sum_i E_g,i*dV (per-substage diffusion conserves it; a conservative
//      regrid restrict/prolong conserves it across a mesh change);
//    * a user refinement criterion (method=user) that REFINES the central x1 blocks for
//      the first half of the run and DE-REFINES them for the second half, so the run
//      exercises BOTH prolongation (refine) and restriction (de-refine).
//
//  ORACLE (Layer 1, conservation).  A conservative restriction/prolongation preserves the
//  volume-weighted integral exactly (to round-off); the insulated diffusion preserves it
//  too.  So the volume-weighted total radiation energy at the end of the run must equal
//  its initial value.  Without erad_mg registered in the regrid path, the data on
//  refined / de-refined / migrated blocks is never re-mapped (stale/uninitialized), so
//  total is NOT conserved -- the check is RED without the registration and GREEN with it.
//  A separate check asserts the mesh actually refined at least once (so the regrid path
//  was genuinely exercised, not silently skipped).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_mgfld_regrid_cpu.py (single rank,
//  same-rank Derefine/Copy/Refine path) and ..._mpicpu.py (4 ranks: load-balancing
//  migrates blocks across ranks, exercising the MPI Pack/UnpackAMRBuffersCC path).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::fabs, std::cos, std::acos
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mhd/mhd.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// shared state between UserProblem (setup), the refinement hook, and the finalize check
Real g_init_total = 0.0;     //!< volume-weighted total radiation energy at setup
int  g_root_nmb   = 0;       //!< MeshBlock count before any refinement
int  g_peak_nmb   = 0;       //!< largest MeshBlock count seen (proves a refine fired)
int  g_refine_until = 0;     //!< refine for ncycle <= this, de-refine after
Real g_xlo = 0.0, g_xhi = 0.0;  //!< x1 band of blocks to refine (central third)

//! \brief MPI_Allreduce a scalar across all ranks (identity on a single rank).
Real GlobalSum(Real local) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return global;
}

//! \brief Global volume-weighted total per-group radiation energy sum_g sum_i E_g,i*dV.
Real TotalRadEnergy(MeshBlockPack *pmbp) {
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto erad = pmhd->erad_mg;
  const int NG = erad.extent_int(1);
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto h_e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), erad);
  Real s = 0.0;
  for (int m = 0; m <= nmb1; ++m) {
    Real vol = size.h_view(m).dx1 * size.h_view(m).dx2 * size.h_view(m).dx3;
    for (int g = 0; g < NG; ++g) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) { s += h_e(m,g,k,j,i)*vol; }
        }
      }
    }
  }
  return GlobalSum(s);
}

//! \brief User refinement hook: refine the central x1 blocks for the first half of the
//! run, de-refine for the second half -- driving a dynamic regrid that exercises both
//! prolongation and restriction of the registered erad_mg array.
void MgfldRegridRefine(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const bool refine_phase = (pmbp->pmesh->ncycle <= g_refine_until);
  int nmb_now = pmbp->pmesh->nmb_total;
  g_peak_nmb = (nmb_now > g_peak_nmb) ? nmb_now : g_peak_nmb;
  for (int m = 0; m < nmb; ++m) {
    int level = pmbp->pmesh->lloc_eachmb[m+mbs].level;
    Real xc = 0.5*(size.h_view(m).x1min + size.h_view(m).x1max);
    bool central = (xc > g_xlo && xc < g_xhi);
    int flag = 0;
    if (refine_phase) {
      if (central && level < pmbp->pmesh->max_level) { flag = 1; }
    } else {
      if (level > pmbp->pmesh->root_level) { flag = -1; }
    }
    refine_flag.h_view(m+mbs) = flag;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

//! \brief Finalize check (runs after the driver loop): the volume-weighted total
//! radiation energy is conserved across the regrids, and the mesh actually refined.
void MgfldRegridFinalize(ParameterInput *pin, Mesh *pm) {
  unit_test::UnitTest test("mgfld_regrid_test");
  Real final_total = TotalRadEnergy(pm->pmb_pack);
  // every rank reduced the same globals; reduce the peak nmb too for a consistent check
  int peak = g_peak_nmb;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &peak, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  test.CheckTrue(peak > g_root_nmb,
                 "the mesh refined at least once (dynamic regrid path was exercised)");
  test.CheckNear(final_total, g_init_total, 5.0e-9, 1.0e-12,
                 "multigroup group-energy arrays conserved across the AMR regrid "
                 "(restricted/prolonged correctly)");
  if (global_variable::my_rank == 0) {
    std::cout << "### mgfld_regrid_test: root_nmb=" << g_root_nmb
              << " peak_nmb=" << peak
              << " init_total=" << g_init_total << " final_total=" << final_total
              << " rel_err=" << std::fabs(final_total-g_init_total)/g_init_total
              << std::endl;
  }
  test.Finish();
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Dynamic-AMR multigroup-FLD regrid setup: seed a conserved per-group field on a
//! static gas, enroll the refinement + finalize hooks, then let the driver run.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->pmhd->pmg_op == nullptr) {
    std::cout << "### mgfld_regrid_test FAILED: multigroup FLD operator not built "
              << "(set mhd/mgfld_operator_split=true)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmhd = pmbp->pmhd;
  const int NG = pmhd->pmg_op->ngroups();

  // ---- seed a smooth, conserved per-group radiation field (insulated => conserved) ----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  auto erad = pmhd->erad_mg;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n3 = erad.extent_int(2);
  const int n2 = erad.extent_int(3);
  const int n1 = erad.extent_int(4);
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real L = x1max - x1min;
  const Real PI = std::acos(-1.0);
  const Real kx = 2.0*PI/L;   // one smooth period across the global domain
  par_for("mgreg_ic", DevExeSpace(), 0, nmb1, 0, NG-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    erad(m,g,k,j,i) = (1.0 + 0.2*static_cast<Real>(g)) + 0.3*Kokkos::cos(kx*(x1 - x1min));
  });

  // ---- trivial uniform static gas so the hydro does nothing (B=0, v=0, uniform p) ----
  auto u0 = pmhd->u0;
  auto b0 = pmhd->b0;
  const int nvaru = u0.extent_int(1);
  par_for("mgreg_u0", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
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

  // ---- record the conserved baseline + enroll the refinement / finalize hooks ----
  g_root_nmb = pmy_mesh_->nmb_total;
  g_peak_nmb = g_root_nmb;
  g_refine_until = pin->GetInteger("time", "nlim")/2;   // refine 1st half, derefine 2nd
  g_xlo = x1min + 0.30*L;
  g_xhi = x1min + 0.70*L;
  g_init_total = TotalRadEnergy(pmbp);
  // NOTE: enroll the hooks as bare members of `this` (the ProblemGenerator being
  // constructed) -- pmy_mesh_->pgen is not assigned until this constructor RETURNS, so
  // dereferencing it here would segfault.
  user_ref_func  = MgfldRegridRefine;
  pgen_final_func = MgfldRegridFinalize;
  // return: the driver now runs nlim cycles, regridding via the user hook; the finalize
  // hook checks conservation of the registered erad_mg across those regrids.
  return;
}
