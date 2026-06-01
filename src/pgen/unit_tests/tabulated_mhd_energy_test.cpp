//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tabulated_mhd_energy_test.cpp
//  \brief Integrated test: the tabulated 3T (IONMIX) EOS drives the LIVE MHD energy
//  update through the table during the hyperbolic step (issue [P2]/#162, ADR-0002/0007).
//
//  Slice [P1]/#159 made eos=tabulated_3t a selectable EOS whose cons->prim produces
//  table-consistent temperatures; this slice routes the live hyperbolic pressure/energy
//  update through that tabulated inversion (the LLF Riemann solver + newdt now close the
//  gas pressure as p_gas = p_ele(T_e) + p_ion(T_i) from the table, replacing the
//  ideal-gamma value).  Unlike the #159 cons->prim unit test (which runs no integration),
//  this test RUNS THE DRIVER (nlim>0) and verifies the integrated result, mirroring the
//  strang_coupling_test pattern: UserProblem seeds the state + enrolls a finalize hook,
//  the driver integrates with rsolver=llf, then the hook checks the result.
//
//  Problem: a 1D periodic, field-free (B=0), initially-at-rest aluminum plasma with a
//  uniform background (rho0, T0) and a smooth central electron+ion TEMPERATURE bump.  The
//  bump is a tabulated PRESSURE perturbation, so the tabulated gas pressure -- and only
//  the tabulated gas pressure -- launches gas motion.
//
//  Checks (finalize, after the loop):
//   (AC#2 energy)  The volume-integrated conserved total energy is conserved to round-off
//                  (periodic + conservative LLF), and the passive electron-energy scalar
//                  integral is conserved (passive advection).
//   (AC#2 T match) On the EVOLVED state, the cached derived temperatures satisfy the
//                  table forward relation rho*EnergyEle(rho,T_e) == e_ele (and the ion
//                  analogue) to the table's log-interp round-trip tolerance -- i.e. the
//                  temperatures used in the live step are the table's T(rho,e), not
//                  ideal-gamma.
//   (feedback live) The tabulated pressure actually drove flow: max|v_x| grows to an O(1)
//                  fraction of the initial sound speed.  This is the RED->GREEN
//                  discriminator -- without the [P2] routing the tabulated EOS hits the
//                  isothermal LLF branch (iso_cs=0), exerts no pressure force, and the
//                  gas stays exactly at rest.
//
//  ORACLE (ADR-0008): method-independent analytic anchors -- exact conservation of two
//  conserved integrals, and the table's own forward/inverse round-trip on the final
//  state.  No fabricated values.
//
//  Built/run by tst/test_suite/unit_tests/test_verify_tabulated_mhd_energy_cpu.py.

#include <cmath>     // std::sqrt, std::log10, std::pow, std::fabs, std::isfinite
#include <iomanip>   // std::setprecision
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "eos/eos_table_3t.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// shared state between UserProblem (setup) and the finalize check (after the driver loop)
Real g_e0     = 0.0;   //!< initial volume-integrated conserved total energy
Real g_s0     = 0.0;   //!< initial volume-integrated electron-energy scalar (e_ele)
Real g_cs0    = 0.0;   //!< initial background sound speed sqrt(gamma*p0/rho0) (tabulated)

//! \brief MPI_Allreduce a scalar sum across all ranks (identity on a single rank).
Real GlobalSum(Real local) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return global;
}

//! \brief Global volume-weighted integral of conserved variable index `nidx` over all
//! active cells (host reduction; trivially the value itself on a single rank).
Real ConservedIntegral(MeshBlockPack *pmbp, int nidx) {
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto h_u  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pmhd->u0);
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  Real s = 0.0;
  for (int m = 0; m <= nmb1; ++m) {
    Real vol = size.h_view(m).dx1 * size.h_view(m).dx2 * size.h_view(m).dx3;
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is; i <= ie; ++i) {
          s += h_u(m,nidx,k,j,i)*vol;
        }
      }
    }
  }
  return GlobalSum(s);
}

//----------------------------------------------------------------------------------------
//! \fn TabulatedEnergyFinalize
//! \brief Finalize check, runs after the driver loop completes.
void TabulatedEnergyFinalize(ParameterInput *pin, Mesh *pm) {
  unit_test::UnitTest test("tabulated_mhd_energy_test");
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  // ---- (AC#2) conservation of the two conserved integrals across the live update ----
  const Real e_fin = ConservedIntegral(pmbp, IEN);
  const Real s_fin = ConservedIntegral(pmbp, nmhd);   // e_ele scalar (density) integral
  test.CheckTrue(std::isfinite(e_fin) && std::isfinite(s_fin),
                 "integrated run stays finite (no blow-up)");
  test.CheckNear(e_fin, g_e0, 1.0e-9, 0.0,
                 "conserved total energy conserved across the live tabulated update");
  test.CheckNear(s_fin, g_s0, 1.0e-9, 0.0,
                 "electron-energy scalar (e_ele) conserved by passive advection");

  // ---- run cons->prim once more so derived_te/ti reflect the final conserved state ----
  pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0, false,
                         is, ie, js, je, ks, ke);

  // ---- (AC#2 T match) + (feedback live), reduced on device over active cells ----
  auto u0 = pmhd->u0;
  auto te = pmhd->derived_te;
  auto ti = pmhd->derived_ti;
  eos_table_3t::EosTable3T tbl = pmhd->eos_tbl;   // shallow View copy into the kernel
  Real max_te_res = 0.0, max_ti_res = 0.0, max_vx = 0.0;
  Kokkos::parallel_reduce("tab_energy_check",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,ks,js,is},{nmb1+1,ke+1,je+1,ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                  Real &lte, Real &lti, Real &lvx) {
      Real rho   = u0(m,IDN,k,j,i);
      Real e_ele = u0(m,nmhd,k,j,i);                          // electron energy density
      Real ke_   = 0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i))
                        + SQR(u0(m,IM3,k,j,i)))/rho;          // B=0 -> no magnetic energy
      Real e_ion = u0(m,IEN,k,j,i) - ke_ - e_ele;            // ion energy by subtraction
      // table forward lookup of the cached temperatures must recover each species energy
      // density (the inverse of the inversion ConsToPrim2T did) -> table-consistent T.
      Real e_ele_fwd = rho*tbl.EnergyEle(rho, te(m,0,k,j,i));
      Real e_ion_fwd = rho*tbl.EnergyIon(rho, ti(m,0,k,j,i));
      Real rte = Kokkos::fabs(e_ele_fwd - e_ele)/Kokkos::fabs(e_ele);
      Real rti = Kokkos::fabs(e_ion_fwd - e_ion)/Kokkos::fabs(e_ion);
      Real vx  = Kokkos::fabs(u0(m,IM1,k,j,i))/rho;
      if (rte > lte) lte = rte;
      if (rti > lti) lti = rti;
      if (vx  > lvx) lvx = vx;
    }, Kokkos::Max<Real>(max_te_res), Kokkos::Max<Real>(max_ti_res),
       Kokkos::Max<Real>(max_vx));

  // table log-log interpolation round-trip tolerance (matches the #159 c2p test)
  const Real rt = 1.0e-6;
  test.CheckTrue(max_te_res < rt,
                 "evolved T_e matches the table: rho*EnergyEle(rho,T_e) == e_ele");
  test.CheckTrue(max_ti_res < rt,
                 "evolved T_i matches the table: rho*EnergyIon(rho,T_i) == e_ion");
  // the tabulated pressure actually drove flow (RED->GREEN discriminator): without the
  // [P2] routing the gas would stay exactly at rest (max_vx == 0).
  test.CheckTrue(max_vx > 1.0e-2*g_cs0,
                 "tabulated gas pressure drove flow (max|v_x| > 1e-2 cs0)");

  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(12)
              << "### tabulated_mhd_energy_test: E0=" << g_e0 << " Efin=" << e_fin
              << " relE=" << std::fabs(e_fin-g_e0)/std::fabs(g_e0)
              << " relS=" << std::fabs(s_fin-g_s0)/std::fabs(g_s0)
              << " max_vx/cs0=" << max_vx/g_cs0
              << " maxTres=" << max_te_res << " maxIres=" << max_ti_res << std::endl;
  }
  test.Finish();
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Seed the field-free, at-rest tabulated-EOS plasma with a central temperature
//! bump, record the conserved baselines, and enroll the finalize check; the driver then
//! integrates the live tabulated hyperbolic update (rsolver=llf).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### tabulated_mhd_energy_test FAILED: requires a <mhd> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmhd = pmbp->pmhd;
  if (!pmhd->peos->eos_data.is_tabulated) {
    std::cout << "### tabulated_mhd_energy_test FAILED: requires <mhd> eos=tabulated_3t."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmhd->nscalars < 1) {
    std::cout << "### tabulated_mhd_energy_test FAILED: requires <mhd> nscalars>=1 for"
              << " the electron-energy scalar." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  // background state at the table log-midpoints (interior -> clean monotone round trip),
  // and a central electron+ion temperature bump amplitude (in T, multiplicative).
  const eos_table_3t::EosTable3T &tbl = pmhd->eos_tbl;
  const Real rho0 = std::sqrt(tbl.RhoMin()*tbl.RhoMax());
  const Real t0   = std::pow(10.0, 0.5*std::log10(tbl.TempMin())
                                   + 0.5*std::log10(tbl.TempMax()));
  const Real amp  = pin->GetOrAddReal("problem", "amp", 0.5);   // bump amplitude in T
  const Real xc   = pin->GetReal("mesh","x1min")
                  + 0.5*(pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min"));
  const Real wbump = 0.08*(pin->GetReal("mesh","x1max") - pin->GetReal("mesh","x1min"));

  // field-free: zero all face-centered B (CT keeps it zero with v x B = 0)
  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  // seed the conserved state on device from (rho0, T(x)) via the table forward lookups
  auto u0 = pmhd->u0;
  auto size = pmbp->pmb->mb_size;
  eos_table_3t::EosTable3T tbl_d = pmhd->eos_tbl;   // shallow View copy into the kernel
  par_for("tab_energy_seed", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min
            + (static_cast<Real>(i-is) + 0.5)*size.d_view(m).dx1;
    Real g  = Kokkos::exp(-SQR((x1 - xc)/wbump));     // smooth central bump in [0,1]
    Real tt = t0*(1.0 + amp*g);                       // same bump for both species
    Real e_ele = rho0*tbl_d.EnergyEle(rho0, tt);
    Real e_ion = rho0*tbl_d.EnergyIon(rho0, tt);
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = 0.0;                            // at rest
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = e_ele + e_ion;                 // v=0,B=0 -> total == internal
    u0(m,nmhd,k,j,i) = e_ele;                         // first passive scalar: e_ele
  });

  // ---- record conserved baselines + the background sound speed ----
  g_e0 = ConservedIntegral(pmbp, IEN);
  g_s0 = ConservedIntegral(pmbp, nmhd);
  // tabulated background pressure -> sound speed (gamma is the wave-speed bookkeeping)
  const Real gamma = pmhd->peos->eos_data.gamma;
  {
    eos_table_3t::GasState2T s0 = eos_table_3t::ConsToPrim2T(
        tbl, rho0, rho0*tbl.EnergyEle(rho0, t0), rho0*tbl.EnergyIon(rho0, t0));
    g_cs0 = std::sqrt(gamma*s0.p_gas/rho0);
  }

  // enroll the finalize hook (pmy_mesh_->pgen is not yet assigned in this ctor).
  pgen_final_func = TabulatedEnergyFinalize;
  return;
}
