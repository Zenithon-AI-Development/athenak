//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file strang_coupling_test.cpp
//  \brief Integrated test of the Strang-split coupled timestep (#115/[B2], ADR-0009):
//  the stiff operator-split parabolic block (here grey FLD radiation diffusion) is run as
//  ONE RKL2 super-step over a per-field CompositeParabolicOperator, Strang-wrapped around
//  the hyperbolic MHD update as  (1/2 super-step) . hydro . (1/2 super-step), with the
//  per-cell point-implicit matter-radiation emission/absorption coupling wrapped OUTSIDE
//  the super-step (a half coupling step on each Strang side).  This is where the full
//  physics is finally integrated, so it RUNS THE DRIVER (nlim>0): the orchestration
//  fires from MHD's "before_timeintegrator" and "after_timeintegrator" task slots exactly
//  as it does in production.  Enable with  mhd/strang_split=true + mhd/fld_operator_split
//  + mhd/mrad_coupling.
//
//  TWO modes (selected by <problem> mode), one pgen serving two athinputs:
//
//  (A) mode=uniform -- AC: "point-implicit coupling runs OUTSIDE the super-step, NOT
//      inflated by the substage count".  Uniform radiation E_r and uniform gas at rest
//      (v=0, B=0) => the FLD spatial diffusion has zero gradient and is the IDENTITY on
//      E_r, and the hydro update is inert, yet the RKL2 super-step still runs s>2
//      substages.  After ONE full timestep the (E_r, e_gas) state must therefore equal
//      EXACTLY two sequential backward-Euler coupling half-steps of dt/2 each (a host
//      oracle), regardless of s.  Were the coupling (wrongly) evaluated inside the RKL2
//      substage loop it would be applied 2*s times with dt/(2*s) and would NOT match the
//      2*(dt/2) oracle to round-off.  We also assert s>2 (the diffusion super-step really
//      substages) and that the coupling depth beta=c*chi_a*(dt/2) is O(1) (so the
//      two-vs-2s applications genuinely differ), and that E_r + e_gas is conserved.
//
//  (B) mode=bump -- AC: "the fully-coupled stack conserves total energy across the
//      split".  A smooth radiation bump on a uniform gas, INSULATED FLD (mhd/fld_e_source
//      < 0 => closed box, E_r integral conserved by the per-substage diffusion + flux
//      correction), periodic gas BCs (hydro conserves the total gas energy), and the
//      point-implicit coupling exchanging energy between the radiation and the gas every
//      step.  The volume-weighted TOTAL energy sum_i (E_r,i + E_gas,i) dV (radiation +
//      gas, the two "species") must be conserved to round-off across many fully-coupled
//      steps -- diffusion conserves the radiation integral, the coupling conserves
//      E_r+e_gas per cell, and the conservative hydro conserves the gas energy integral.
//      Runs multi-block so the half-step exercises the SyncParabolicGhosts exchange.
//
//  ORACLE (Layer 1).  (A) closed-form backward-Euler quartic (the #23 PointImplicitGrey-
//  Coupling on the host); (B) exact conservation of a conserved scalar.  Both are
//  method-independent analytic anchors (ADR-0008).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_strang_coupling_{cpu,mpicpu}.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <cmath>     // std::fabs, std::cos, std::acos, std::isfinite
#include <iomanip>   // std::setprecision
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// shared state between UserProblem (setup) and the finalize check (after the driver loop)
int  g_mode = 0;             //!< 0 = uniform (oracle), 1 = bump (conservation)
Real g_er0 = 0.0;            //!< uniform-mode initial radiation energy density
Real g_eg0 = 0.0;            //!< uniform-mode initial gas internal energy density
Real g_init_total = 0.0;     //!< bump-mode volume-weighted initial total (E_r + E_gas)

//! \brief MPI_Allreduce a scalar sum across all ranks (identity on a single rank).
Real GlobalSum(Real local) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return global;
}

//! \brief Global volume-weighted total of (radiation E_r + gas internal energy) over all
//! active cells.  Gas is at rest with B=0 in both modes, so the gas internal energy ==
//! u0(IEN).  This is the conserved "species split" total for mode B.
Real TotalEnergy(MeshBlockPack *pmbp) {
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto h_er = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pmhd->erad);
  auto h_u  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pmhd->u0);
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  Real s = 0.0;
  for (int m = 0; m <= nmb1; ++m) {
    Real vol = size.h_view(m).dx1 * size.h_view(m).dx2 * size.h_view(m).dx3;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          s += (h_er(m,0,k,j,i) + h_u(m,IEN,k,j,i))*vol;
        }
      }
    }
  }
  return GlobalSum(s);
}

//! \brief Finalize check, runs after the driver loop completes.
void StrangCouplingFinalize(ParameterInput *pin, Mesh *pm) {
  unit_test::UnitTest test("strang_coupling_test");
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;

  if (g_mode == 0) {
    // ---- (A) uniform: coupling-outside-the-super-step exact-oracle check ----
    auto &indcs = pm->mb_indcs;
    auto h_er = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pmhd->erad);
    auto h_u  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pmhd->u0);
    Real er_fin = h_er(0,0,indcs.ks,indcs.js,indcs.is);
    Real eg_fin = h_u(0,IEN,indcs.ks,indcs.js,indcs.is);   // v=0,B=0 => internal == IEN

    // the dt actually used this cycle (constant across the two Strang half-steps)
    const Real dt = pm->dt_last_completed;
    const Real cv = pmhd->mrad_cv, chia = pmhd->mrad_chi_a;
    const Real cl = pmhd->mrad_clight, arad = pmhd->mrad_arad;

    // host oracle: TWO sequential backward-Euler coupling half-steps of dt/2 (the FLD
    // diffusion and the hydro update are both identity on this uniform/at-rest state).
    Real er1, eg1, er2, eg2;
    radiationfld::PointImplicitGreyCoupling(g_er0, g_eg0, cv, chia, cl, arad, 0.5*dt,
                                            er1, eg1);
    radiationfld::PointImplicitGreyCoupling(er1, eg1, cv, chia, cl, arad, 0.5*dt,
                                            er2, eg2);

    // the FLD super-step really substages (s>2): coupling is "outside" something real
    const Real dt_exp = pmhd->pfld_op->ExplicitStableDt();
    const int s = parabolic::RKL2NumStages(0.5*dt, dt_exp);
    const Real beta = cl*chia*0.5*dt;   // backward-Euler coupling depth per half-step

    test.CheckTrue(s > 2,
                   "(A) the FLD RKL2 super-step genuinely substages (s>2) over dt/2");
    test.CheckTrue(beta > 0.1 && beta < 50.0,
                   "(A) coupling depth beta=c*chi_a*dt/2 is O(1) (two-vs-2s differ)");
    test.CheckNear(er_fin, er2, 1.0e-11, 1.0e-12,
                   "(A) E_r == two dt/2 coupling steps (outside super-step)");
    test.CheckNear(eg_fin, eg2, 1.0e-11, 1.0e-12,
                   "(A) e_gas == two dt/2 coupling steps (outside super-step)");
    test.CheckNear(er_fin + eg_fin, g_er0 + g_eg0, 0.0, 1.0e-11,
                   "(A) E_r + e_gas conserved across the species split (point coupling)");
    if (global_variable::my_rank == 0) {
      std::cout << std::setprecision(12)
                << "### strang_coupling_test[uniform]: dt=" << dt << " dt_exp=" << dt_exp
                << " s=" << s << " beta=" << beta
                << " er_fin=" << er_fin << " oracle=" << er2 << std::endl;
    }
  } else {
    // ---- (B) bump: fully-coupled energy conservation across the species split ----
    Real final_total = TotalEnergy(pmbp);
    test.CheckTrue(std::isfinite(final_total),
                   "(B) fully-coupled run stays finite (no blow-up)");
    test.CheckNear(final_total, g_init_total, 5.0e-10, 1.0e-12,
                   "(B) total energy (E_r + E_gas) conserved across the species split");
    if (global_variable::my_rank == 0) {
      std::cout << std::setprecision(12)
                << "### strang_coupling_test[bump]: init_total=" << g_init_total
                << " final_total=" << final_total
                << " rel_err=" << std::fabs(final_total-g_init_total)/g_init_total
                << std::endl;
    }
  }
  test.Finish();
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Set up the coupled grey-FLD + matter-radiation Strang test on a static/at-rest
//! gas, enroll the finalize check, then let the driver run the orchestrated steps.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### strang_coupling_test FAILED: MHD not enabled" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmhd = pmbp->pmhd;
  // the orchestration must be wired: Strang split + grey FLD operator + matter coupling
  if (!pmhd->strang_split || pmhd->pfld_op == nullptr || !pmhd->mrad_coupling) {
    std::cout << "### strang_coupling_test FAILED: requires mhd/strang_split=true, "
              << "mhd/fld_operator_split=true, mhd/mrad_coupling=true" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::string mode = pin->GetOrAddString("problem", "mode", "uniform");
  g_mode = (mode == "bump") ? 1 : 0;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is;
  auto erad = pmhd->erad;
  auto u0 = pmhd->u0;
  auto b0 = pmhd->b0;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const int n3 = erad.extent_int(2);
  const int n2 = erad.extent_int(3);
  const int n1 = erad.extent_int(4);
  const int nvaru = u0.extent_int(1);

  const Real rho0  = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real e_gas0 = pin->GetOrAddReal("problem", "e_gas0", 1.0);  // internal energy
  const Real e_rad0 = pin->GetOrAddReal("problem", "e_rad0", 1.0);
  const Real amp    = pin->GetOrAddReal("problem", "amp", 0.3);     // bump amplitude
  g_er0 = e_rad0;
  g_eg0 = e_gas0;

  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real L = x1max - x1min;
  const Real PI = std::acos(-1.0);
  const Real kx = 2.0*PI/L;              // one smooth period across the global domain
  const int mode_i = g_mode;

  // ---- radiation field: uniform (mode A) or a smooth cosine bump (mode B) ----
  par_for("strang_erad", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real er = e_rad0;
    if (mode_i == 1) {
      Real x1 = size.d_view(m).x1min + (static_cast<Real>(i-is) + 0.5)*size.d_view(m).dx1;
      er = e_rad0 + amp*Kokkos::cos(kx*(x1 - x1min));
    }
    erad(m,0,k,j,i) = er;
  });

  // ---- uniform gas at rest (v=0, B=0, uniform internal energy) ----
  par_for("strang_u0", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int n = 0; n < nvaru; ++n) { u0(m,n,k,j,i) = 0.0; }
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IEN,k,j,i) = e_gas0;   // v=0,B=0 => total == internal energy density
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = 0.0;
    if (i==n1-1) { b0.x1f(m,k,j,i+1) = 0.0; }
    if (j==n2-1) { b0.x2f(m,k,j+1,i) = 0.0; }
    if (k==n3-1) { b0.x3f(m,k+1,j,i) = 0.0; }
  });

  // ---- record baseline + enroll the finalize hook ----
  if (g_mode == 1) { g_init_total = TotalEnergy(pmbp); }
  // enroll as a bare member of `this` (pmy_mesh_->pgen is not assigned until this ctor
  // returns -- dereferencing it here would segfault).
  pgen_final_func = StrangCouplingFinalize;
  return;
}
