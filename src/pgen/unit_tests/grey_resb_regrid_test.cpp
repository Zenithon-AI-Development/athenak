//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grey_resb_regrid_test.cpp
//  \brief Unit test: the standalone GREY radiation-energy array (MHD::erad) and the
//  standalone resb B_phi array (MHD::bphi) are correctly RESTRICTED and PROLONGED across
//  a DYNAMIC AMR regrid, so the operator-split grey-FLD and resistive-B_phi physics
//  survive re-refinement (#248, the grey/resb port of the #111/[A4] erad_mg
//  registration).
//
//  This is the dynamic-regrid array-registration gate the static-SMR multiblock tests
//  (fld_marshak_multiblock / resb_bphi_multiblock on SMR meshes) cannot reach: a static
//  mesh never regrids, so the standalone arrays are never re-mapped.  The AMR regrid
//  machinery (mesh_refinement.cpp DerefineCCSameRank / CopyCC / CopyForRefinementCC /
//  RefineCC and, across ranks, load_balance.cpp Pack/UnpackAMRBuffersCC) packed ONLY
//  erad_mg before #248; the grey erad and resb bphi were silently dropped on any
//  refinement change -- which is exactly what this test demonstrates (RED before the
//  registration, GREEN with it).
//
//  DESIGN (mirrors mgfld_regrid_test):
//    * the driver RUNS (nlim>0) so the standard adaptive-mesh-refinement cycle fires on
//      a uniform, static gas (v=0, uniform rho/p, B=0);
//    * the grey FLD operator is INSULATED (mhd/fld_e_source < 0) and diffuses a smooth
//      seeded erad every step, so its conserved quantity is the volume-weighted total
//      radiation energy sum_i E_i*dV (insulated diffusion conserves it; a conservative
//      regrid restrict/prolong conserves it across a mesh change);
//    * the resb operator runs with eta = 0 (a frozen B_phi): its ApplyBoundary is
//      antisymmetric (Dirichlet-0) at the x1 domain faces, so a diffusing B_phi decays
//      by construction and cannot furnish a conservation oracle -- with eta = 0 the
//      operator action is exactly zero and the ONLY thing that can change the
//      volume-weighted total sum_i B_phi,i*dV is the regrid re-mapping under test;
//    * a user refinement criterion (method=user) REFINES the central x1 blocks for the
//      first half of the run and DE-REFINES them for the second half, exercising BOTH
//      prolongation (refine) and restriction (de-refine).
//
//  ORACLE (Layer 1, conservation).  A conservative restriction/prolongation preserves
//  the volume-weighted integral exactly (to round-off); the insulated grey diffusion
//  (and the eta=0 frozen bphi) preserve it too.  So the volume-weighted totals of erad
//  and bphi must equal their initial values AT EVERY AMR CHECK, not just at the end:
//  the refinement hook tracks the running maximum relative deviation of both totals.
//  The mid-run tracking is essential -- with a symmetric refine-then-derefine cycle the
//  final mesh returns to the initial root layout, so a FROZEN field that was never
//  re-mapped ends up back over its original (stale-but-correct) array slots and an
//  end-state-only check is blind to the drop; while the mesh is refined mid-run the
//  stale mapping is exposed directly.  Without erad/bphi registered in the regrid path
//  the data on refined / de-refined / migrated blocks is never re-mapped, so the totals
//  deviate mid-run (and, for the diffusing erad, at the end too).  A separate check
//  asserts the mesh actually refined at least once (the regrid path was exercised).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_grey_resb_regrid_cpu.py (single
//  rank, same-rank Derefine/Copy/Refine path) and ..._mpicpu.py (4 ranks: load balancing
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
#include "radiation_fld/fld_grey_operator.hpp"
#include "diffusion/resistive_bphi_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// shared state between UserProblem (setup), the refinement hook, and the finalize check
Real g_init_erad = 0.0;      //!< volume-weighted total grey radiation energy at setup
Real g_init_bphi = 0.0;      //!< volume-weighted total B_phi at setup
Real g_maxdev_erad = 0.0;    //!< max relative deviation of the erad total (per AMR check)
Real g_maxdev_bphi = 0.0;    //!< max relative deviation of the bphi total (per AMR check)
int  g_root_nmb  = 0;        //!< MeshBlock count before any refinement
int  g_peak_nmb  = 0;        //!< largest MeshBlock count seen (proves a refine fired)
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

//! \brief Global volume-weighted total of a standalone single-variable CC array.
Real TotalCC(MeshBlockPack *pmbp, const DvceArray5D<Real> &arr) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int NV = arr.extent_int(1);
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto h_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), arr);
  Real s = 0.0;
  for (int m = 0; m <= nmb1; ++m) {
    Real vol = size.h_view(m).dx1 * size.h_view(m).dx2 * size.h_view(m).dx3;
    for (int v = 0; v < NV; ++v) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) { s += h_a(m,v,k,j,i)*vol; }
        }
      }
    }
  }
  return GlobalSum(s);
}

//! \brief User refinement hook: refine the central x1 blocks for the first half of the
//! run, de-refine for the second half -- driving a dynamic regrid that exercises both
//! prolongation and restriction of the registered erad and bphi arrays.  Also tracks
//! the running maximum relative deviation of the conserved totals (see the oracle note
//! at the top of this file: the mid-run tracking is what catches a dropped FROZEN field
//! that an end-state-only check would miss).  Every rank calls this hook in lockstep at
//! each AMR check, so the collective reduce inside TotalCC is safe.
void GreyResbRegridRefine(MeshBlockPack *pmbp) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  const bool refine_phase = (pmbp->pmesh->ncycle <= g_refine_until);
  int nmb_now = pmbp->pmesh->nmb_total;
  g_peak_nmb = (nmb_now > g_peak_nmb) ? nmb_now : g_peak_nmb;
  Real dev_e = std::fabs(TotalCC(pmbp, pmbp->pmhd->erad) - g_init_erad)
             / std::fabs(g_init_erad);
  Real dev_b = std::fabs(TotalCC(pmbp, pmbp->pmhd->bphi) - g_init_bphi)
             / std::fabs(g_init_bphi);
  g_maxdev_erad = (dev_e > g_maxdev_erad) ? dev_e : g_maxdev_erad;
  g_maxdev_bphi = (dev_b > g_maxdev_bphi) ? dev_b : g_maxdev_bphi;
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

//! \brief Finalize check (runs after the driver loop): the volume-weighted totals of the
//! grey erad AND the resb bphi are conserved across the regrids, and the mesh refined.
void GreyResbRegridFinalize(ParameterInput *pin, Mesh *pm) {
  unit_test::UnitTest test("grey_resb_regrid_test");
  auto *pmhd = pm->pmb_pack->pmhd;
  Real final_erad = TotalCC(pm->pmb_pack, pmhd->erad);
  Real final_bphi = TotalCC(pm->pmb_pack, pmhd->bphi);
  // every rank reduced the same totals; reduce the peak nmb too for a consistent check
  // (g_maxdev_* are already identical on every rank: built from GlobalSum'd totals)
  int peak = g_peak_nmb;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &peak, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  test.CheckTrue(peak > g_root_nmb,
                 "the mesh refined at least once (dynamic regrid path was exercised)");
  test.CheckNear(final_erad, g_init_erad, 5.0e-9, 1.0e-12,
                 "grey radiation-energy array (erad) conserved across the AMR regrid "
                 "(restricted/prolonged correctly)");
  test.CheckNear(final_bphi, g_init_bphi, 5.0e-9, 1.0e-12,
                 "resb B_phi array (bphi) conserved across the AMR regrid "
                 "(restricted/prolonged correctly)");
  test.CheckTrue(g_maxdev_erad < 5.0e-9,
                 "grey erad total conserved at EVERY AMR check (mid-run, refined mesh)");
  test.CheckTrue(g_maxdev_bphi < 5.0e-9,
                 "resb bphi total conserved at EVERY AMR check (mid-run, refined mesh)");
  if (global_variable::my_rank == 0) {
    std::cout << "### grey_resb_regrid_test: root_nmb=" << g_root_nmb
              << " peak_nmb=" << peak
              << " init_erad=" << g_init_erad << " final_erad=" << final_erad
              << " init_bphi=" << g_init_bphi << " final_bphi=" << final_bphi
              << " maxdev_erad=" << g_maxdev_erad
              << " maxdev_bphi=" << g_maxdev_bphi
              << std::endl;
  }
  test.Finish();
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Dynamic-AMR grey-FLD + resb regrid setup: seed conserved erad and bphi fields
//! on a static gas, enroll the refinement + finalize hooks, then let the driver run.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr || pmbp->pmhd->pfld_op == nullptr) {
    std::cout << "### grey_resb_regrid_test FAILED: grey FLD operator not built "
              << "(set mhd/fld_operator_split=true)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->presb_op == nullptr) {
    std::cout << "### grey_resb_regrid_test FAILED: resb B_phi operator not built "
              << "(set mhd/resb_operator_split=true)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmhd = pmbp->pmhd;

  // ---- seed smooth, conserved erad and bphi fields (insulated / eta=0 => conserved) ---
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  auto erad = pmhd->erad;
  auto bphi = pmhd->bphi;
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
  par_for("greyresb_ic", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    erad(m,0,k,j,i) = 1.0 + 0.3*Kokkos::cos(kx*(x1 - x1min));
    bphi(m,0,k,j,i) = 2.0 + 0.5*Kokkos::cos(kx*(x1 - x1min) + 0.5);
  });

  // ---- trivial uniform static gas so the hydro does nothing (B=0, v=0, uniform p) ----
  auto u0 = pmhd->u0;
  auto b0 = pmhd->b0;
  const int nvaru = u0.extent_int(1);
  par_for("greyresb_u0", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
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

  // ---- record the conserved baselines + enroll the refinement / finalize hooks ----
  g_root_nmb = pmy_mesh_->nmb_total;
  g_peak_nmb = g_root_nmb;
  g_refine_until = pin->GetInteger("time", "nlim")/2;   // refine 1st half, derefine 2nd
  g_xlo = x1min + 0.30*L;
  g_xhi = x1min + 0.70*L;
  g_init_erad = TotalCC(pmbp, pmhd->erad);
  g_init_bphi = TotalCC(pmbp, pmhd->bphi);
  // NOTE: enroll the hooks as bare members of `this` (the ProblemGenerator being
  // constructed) -- pmy_mesh_->pgen is not assigned until this constructor RETURNS, so
  // dereferencing it here would segfault.
  user_ref_func  = GreyResbRegridRefine;
  pgen_final_func = GreyResbRegridFinalize;
  // return: the driver now runs nlim cycles, regridding via the user hook; the finalize
  // hook checks conservation of the registered erad and bphi across those regrids.
  return;
}
