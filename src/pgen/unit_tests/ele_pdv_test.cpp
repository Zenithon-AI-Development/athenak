//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ele_pdv_test.cpp
//  \brief Driver-integrated unit test for the gated electron PdV (compression-heating)
//  source (issue #231, 3T consistency 1/4).
//
//  WHY.  `three_temp::ElectronPdVRate` has existed since ADR-0002 but was never wired
//  into the MHD driver: in MagLIF runs the evolved electron internal-energy scalar
//  `e_ele` (rides passive scalar 0 = u0(nmhd)) changes by ADVECTION ONLY, so the
//  electrons receive no compression heating and T_e stays cold through the implosion
//  while every T_e consumer (opacity lookups, EOS-aware conduction/coupling, the
//  e_ion = E - e_ele subtraction) reads a wrong temperature.  #231 wires the electron
//  PdV work term `d e_ele/dt = -p_ele div(v)` as a gated per-stage driver source
//  (<mhd> ele_pdv, default false => byte-identical).
//
//  WHAT THIS DOES (RUNS THE DRIVER, mirroring tabulated_mhd_energy_test): a 1D periodic,
//  field-free ideal-gas box with uniform (rho0, p0) and a sinusoidal velocity
//  v_x(x) = -v0 sin(2 pi x) that compresses the gas around x = 0.  The electron energy
//  scalar is seeded as a uniform fraction f_ele of the internal energy.  UserProblem
//  seeds the state and enrolls a finalize hook; the driver integrates; the hook checks,
//  branching on the active <mhd>/ele_pdv gate:
//
//  gate ON  (#231): every cell obeys the electron ADIABAT -- the flow is smooth (run
//      ends well before shock formation) and isentropic, and all fluid elements start on
//      the same adiabat, so at any time the specific electron energy satisfies
//          e_ele/rho == (e_ele0/rho0) * (rho/rho0)^(gamma-1)
//      pointwise (the analytic T_e(rho) anchor: T_e proportional to rho^(gamma-1) for an
//      ideal electron fluid with constant c_v).  Also: TOTAL energy is conserved to
//      round-off (the source touches ONLY the e_ele partition scalar, never u0(IEN)).
//
//  gate OFF (baseline, today's behaviour): the specific scalar e_ele/rho stays exactly
//      the seeded constant (advection preserves a uniform specific scalar to round-off)
//      even though the gas compresses -- the compression-blind e_ele this issue fixes.
//      Total energy conserved as before.
//
//  Both legs also require a real compression signal (max rho/rho0 well above 1), so the
//  adiabat check cannot pass vacuously on a quiescent state.  RED evidence: before the
//  #231 wiring the gate-ON leg fails the adiabat check by exactly the compression signal
//  (e_ele/rho stays constant while rho^(gamma-1) grows ~13%).
//
//  Two further DIRECT-CALL stencil legs (problem/check overrides, no time integration)
//  pin the geometry and EOS branches of the source itself:
//    cyl_stencil: in cylindrical coordinates div(v) must carry the radial metric term
//        (1/r) d(r v_r)/dr -- for v_r = -a*r the increment is exactly 2*a*p_ele*dt (a
//        cartesian-naive stencil lands at half);
//    tab_stencil: with eos=tabulated_3t the source must close p_ele from the table at
//        the cached derived T_e (PressureEle(rho, T_e)), not the ideal-gamma fallback.
//
//  ORACLE (ADR-0008): method-independent analytic anchors -- the isentropic-flow adiabat
//  T_e(rho), exact conservation of the periodic-box total energy, and exact central
//  differences of linear/quadratic velocity profiles.
//
//  Built/run by tst/test_suite/unit_tests/test_unit_ele_pdv_cpu.py.

#include <cmath>     // std::isfinite, std::pow, std::sin, std::sqrt, std::log10
#include <iomanip>   // std::setprecision
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
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
bool g_gate  = false;  //!< the <mhd>/ele_pdv gate the run was configured with
Real g_e0    = 0.0;    //!< initial volume-integrated conserved total energy
Real g_rho0  = 0.0;    //!< uniform initial density
Real g_sele0 = 0.0;    //!< uniform initial SPECIFIC electron energy e_ele0/rho0
Real g_gamma = 0.0;    //!< ideal-gas gamma (the electron adiabat index)

//! \brief MPI_Allreduce a scalar across all ranks (identity on a single rank).
Real GlobalReduce(Real local, bool take_max) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL,
                take_max ? MPI_MAX : MPI_SUM, MPI_COMM_WORLD);
#endif
  return global;
}

//! \brief Global volume-weighted integral of conserved variable index `nidx` over all
//! active cells (device reduction).
Real ConservedIntegral(MeshBlockPack *pmbp, int nidx) {
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  auto u0 = pmhd->u0;
  auto size = pmbp->pmb->mb_size;
  const int nmb1 = pmbp->nmb_thispack - 1;
  Real s = 0.0;
  Kokkos::parallel_reduce("elepdv_integral",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,ks,js,is},{nmb1+1,ke+1,je+1,ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &ls) {
      Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
      ls += u0(m,nidx,k,j,i)*vol;
    }, Kokkos::Sum<Real>(s));
  return GlobalReduce(s, false);
}

//----------------------------------------------------------------------------------------
//! \fn ElePdVFinalize
//! \brief Finalize check, runs after the driver loop completes.
void ElePdVFinalize(ParameterInput *pin, Mesh *pm) {
  unit_test::UnitTest test("ele_pdv_test");
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pmhd = pmbp->pmhd;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  // ---- conservation: the PdV source repartitions e_ele/e_ion but NEVER touches the
  // conserved total, so E_tot must be conserved to round-off with the gate on OR off.
  const Real e_fin = ConservedIntegral(pmbp, IEN);
  test.CheckTrue(std::isfinite(e_fin), "integrated run stays finite (no blow-up)");
  test.CheckNear(e_fin, g_e0, 1.0e-10, 0.0,
                 "conserved TOTAL energy conserved to round-off (partition only)");

  // ---- per-cell adiabat + compression-signal reductions on device ----
  auto u0 = pmhd->u0;
  const Real rho0 = g_rho0, sele0 = g_sele0, gm1 = g_gamma - 1.0;
  Real max_dev = 0.0, max_rho = 0.0, max_sdev = 0.0;
  Kokkos::parallel_reduce("elepdv_check",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,ks,js,is},{nmb1+1,ke+1,je+1,ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                  Real &ldev, Real &lrho, Real &lsdev) {
      Real rho  = u0(m,IDN,k,j,i);
      Real sele = u0(m,nmhd,k,j,i)/rho;   // specific electron energy e_ele/rho
      // analytic electron adiabat through the seeded state: s(rho) = s0*(rho/rho0)^(g-1)
      Real spred = sele0*Kokkos::pow(rho/rho0, gm1);
      Real dev   = Kokkos::fabs(sele/spred - 1.0);
      Real sdev  = Kokkos::fabs(sele/sele0 - 1.0);   // deviation from the SEEDED value
      Real rr    = rho/rho0;
      if (dev  > ldev)  ldev  = dev;
      if (rr   > lrho)  lrho  = rr;
      if (sdev > lsdev) lsdev = sdev;
    }, Kokkos::Max<Real>(max_dev), Kokkos::Max<Real>(max_rho),
       Kokkos::Max<Real>(max_sdev));
  max_dev  = GlobalReduce(max_dev, true);
  max_rho  = GlobalReduce(max_rho, true);
  max_sdev = GlobalReduce(max_sdev, true);

  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(12)
              << "### ele_pdv_test: gate=" << (g_gate ? 1 : 0)
              << " relE=" << std::fabs(e_fin - g_e0)/std::fabs(g_e0)
              << " max_rho/rho0=" << max_rho
              << " max_adiabat_dev=" << max_dev
              << " max_seed_dev=" << max_sdev << std::endl;
  }

  // ---- a real compression must have developed, else the adiabat check is vacuous ----
  test.CheckTrue(max_rho > 1.06,
                 "compression signal present (max rho/rho0 > 1.06)");

  if (g_gate) {
    // GREEN (#231): every cell tracks the analytic electron adiabat T_e(rho).  RED
    // before the wiring: e_ele/rho stays the seeded constant, so the deviation equals
    // the full compression signal (rho/rho0)^(gamma-1)-1 ~ 13% >> 2%.
    test.CheckTrue(max_dev < 0.02,
        "gate ON: e_ele tracks the analytic adiabat e_ele/rho = s0*(rho/rho0)^(g-1)");
  } else {
    // Baseline (gate off, default): e_ele is advection-only -- the specific scalar
    // stays exactly the seeded constant despite the compression (round-off only).
    test.CheckTrue(max_sdev < 1.0e-11,
        "gate OFF: specific e_ele unchanged by compression (advection-only baseline)");
  }

  test.Finish();
}

//----------------------------------------------------------------------------------------
//! \fn CylStencilCheck
//! \brief Direct-call stencil leg (no time integration): in CYLINDRICAL coordinates the
//! velocity divergence carries the radial metric term, div(v) = (1/r) d(r v_r)/dr.  Seed
//! the primitives with the pure radial flow v_r = -a*r (so div(v) = -2a EXACTLY, and the
//! central difference of the quadratic r*v_r = -a*r^2 is also exact), call the REAL
//! MHD::AddElectronPdVSource once, and check the per-cell e_ele increment against the
//! analytic -p_ele*div(v)*dt = +2*a*p_ele*dt.  A cartesian-naive stencil (dv_r/dr = -a)
//! produces exactly HALF the increment -- the RED discriminator for the metric term.
void CylStencilCheck(ParameterInput *pin, MeshBlockPack *pmbp) {
  unit_test::UnitTest test("ele_pdv_test[cyl_stencil]");
  auto *pmhd = pmbp->pmhd;
  test.CheckTrue(pmbp->pcoord->coord_system == CoordSystem::cylindrical,
                 "cylindrical coordinate system");

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  const Real aa    = pin->GetOrAddReal("problem","a",0.3);      // v_r = -a*r
  const Real rho0  = pin->GetOrAddReal("problem","rho0",1.0);
  const Real eele0 = pin->GetOrAddReal("problem","eele0",0.7);  // uniform e_ele density
  const Real dt    = pin->GetOrAddReal("problem","dt",0.01);
  const Real gamma = pmhd->peos->eos_data.gamma;
  const Real eint0 = pin->GetOrAddReal("problem","p0",0.6)/(gamma - 1.0);

  // seed primitives AND conserved over the FULL array extents (the stencil reads i+/-1
  // ghosts; the conserved state also keeps the post-check driver init well-posed).
  auto w0 = pmhd->w0;
  auto u0 = pmhd->u0;
  auto size = pmbp->pmb->mb_size;
  const int n3 = w0.extent_int(2), n2 = w0.extent_int(3), n1 = w0.extent_int(4);
  par_for("elepdv_cyl_seed", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real r  = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real vr = -aa*r;
    w0(m,IDN,k,j,i) = rho0;
    w0(m,IVX,k,j,i) = vr;
    w0(m,IVY,k,j,i) = 0.0;
    w0(m,IVZ,k,j,i) = 0.0;
    w0(m,IEN,k,j,i) = eint0;
    w0(m,nmhd,k,j,i) = eele0/rho0;   // primitive scalar is SPECIFIC, e_ele/rho
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*vr;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eint0 + 0.5*rho0*vr*vr;
    u0(m,nmhd,k,j,i) = eele0;
  });
  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  // ---- code under test: one direct application of the per-stage source ----
  pmhd->AddElectronPdVSource(dt);

  // analytic increment: -p_ele*div(v)*dt with div(v) = (1/r) d(r*(-a r))/dr = -2a
  const Real pele = (gamma - 1.0)*eele0;
  const Real inc  = 2.0*aa*pele*dt;
  Real max_err = 0.0;
  Kokkos::parallel_reduce("elepdv_cyl_check",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,ks,js,is},{nmb1+1,ke+1,je+1,ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lerr) {
      Real got = u0(m,nmhd,k,j,i) - eele0;
      Real err = Kokkos::fabs(got - inc)/inc;
      if (err > lerr) lerr = err;
    }, Kokkos::Max<Real>(max_err));
  max_err = GlobalReduce(max_err, true);

  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(12)
              << "### ele_pdv_test[cyl_stencil]: expected_inc=" << inc
              << " max_rel_err=" << max_err << std::endl;
  }
  // the quadratic r*v_r makes the central difference exact -> round-off only.  The
  // cartesian-naive stencil misses the metric term and lands at HALF the increment
  // (rel err 0.5), so this cleanly discriminates the cylindrical divergence.
  test.CheckTrue(max_err < 1.0e-12,
      "cylindrical div(v): e_ele increment matches -p_ele*(-2a)*dt in every cell");
  test.Finish();
}

//----------------------------------------------------------------------------------------
//! \fn TabStencilCheck
//! \brief Direct-call stencil leg for the TABULATED 3T EOS branch: with
//! eos=tabulated_3t the source must close p_ele from the table at the cached derived
//! temperature -- p_ele = PressureEle(rho, T_e) with T_e = derived_te (the ADR-0002
//! derived/cached field the cons->prim closure fills) -- NOT from the ideal-gamma
//! fallback (gamma-1)*e_ele.  The synthetic fixture's p_ele = 0.8*rho*T differs from
//! the fallback by ~13% at the table midpoint, so the wiring is cleanly discriminated.
//! Seed a uniform (rho0, T0) state with the linear flow v_x = -a*x (div(v) = -a exactly,
//! central difference exact), run the REAL tabulated ConsToPrim to fill w0 + derived_te,
//! call the REAL MHD::AddElectronPdVSource once, and check the per-cell e_ele increment
//! against dt*a*PressureEle(rho, T_e_cached) -- the same table lookup at the same cached
//! temperature the source must consume.  (The T_e ~ T0 sanity check breaks circularity:
//! it pins the cached temperature to the seeded one first.)
void TabStencilCheck(ParameterInput *pin, MeshBlockPack *pmbp) {
  unit_test::UnitTest test("ele_pdv_test[tab_stencil]");
  auto *pmhd = pmbp->pmhd;
  test.CheckTrue(pmhd->peos->eos_data.is_tabulated,
                 "tabulated 3T EOS selected (eos=tabulated_3t)");
  if (!pmhd->peos->eos_data.is_tabulated) { test.Finish(); return; }

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int nx1 = indcs.nx1;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  const Real aa = pin->GetOrAddReal("problem","a",0.3);   // v_x = -a*x
  const Real dt = pin->GetOrAddReal("problem","dt",0.01);

  // background at the table log-midpoints (clean monotone round trip, mirroring
  // tabulated_mhd_energy_test).
  const eos_table_3t::EosTable3T &tbl = pmhd->eos_tbl;
  const Real rho0 = std::sqrt(tbl.RhoMin()*tbl.RhoMax());
  const Real t0   = std::pow(10.0, 0.5*std::log10(tbl.TempMin())
                                   + 0.5*std::log10(tbl.TempMax()));
  const Real eele0 = rho0*tbl.EnergyEle(rho0, t0);
  const Real eion0 = rho0*tbl.EnergyIon(rho0, t0);

  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  // seed the conserved state over the FULL array extents (stencil reads i+/-1 ghosts)
  auto u0 = pmhd->u0;
  auto size = pmbp->pmb->mb_size;
  eos_table_3t::EosTable3T tbl_d = pmhd->eos_tbl;   // shallow View copy into the kernel
  const int n3 = u0.extent_int(2), n2 = u0.extent_int(3), n1 = u0.extent_int(4);
  par_for("elepdv_tab_seed", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real vx = -aa*x1;
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eele0 + eion0 + 0.5*rho0*vx*vx;
    u0(m,nmhd,k,j,i) = eele0;
  });

  // run the REAL tabulated cons->prim over the full extents (incl. ghosts) so w0 and
  // the derived/cached T_e/T_i fields carry exactly what the driver source consumes.
  const int n1m1 = nx1 + 2*ng - 1;
  const int n2m1 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng - 1) : 0;
  const int n3m1 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng - 1) : 0;
  pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0, false,
                         0, n1m1, 0, n2m1, 0, n3m1);

  // circularity breaker: the cached T_e must be the seeded T0 (table round trip)
  auto te = pmhd->derived_te;
  Real max_te_err = 0.0;
  Kokkos::parallel_reduce("elepdv_tab_te",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,ks,js,is},{nmb1+1,ke+1,je+1,ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, Real &lerr) {
      Real err = Kokkos::fabs(te(m,0,k,j,i)/t0 - 1.0);
      if (err > lerr) lerr = err;
    }, Kokkos::Max<Real>(max_te_err));
  max_te_err = GlobalReduce(max_te_err, true);
  test.CheckTrue(max_te_err < 1.0e-4,
                 "cached derived T_e recovers the seeded T0 (table round trip)");

  // ---- code under test: one direct application of the per-stage source ----
  pmhd->AddElectronPdVSource(dt);

  // per-cell wiring oracle: increment == dt*a*PressureEle(rho, T_e_cached) -- the SAME
  // table lookup at the SAME cached temperature the tabulated branch must consume.
  const Real dta = dt*aa;
  Real max_err = 0.0, max_got = 0.0, max_exp = 0.0;
  Kokkos::parallel_reduce("elepdv_tab_check",
    Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0,ks,js,is},{nmb1+1,ke+1,je+1,ie+1}),
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                  Real &lerr, Real &lgot, Real &lexp) {
      Real expect = dta*tbl_d.PressureEle(u0(m,IDN,k,j,i), te(m,0,k,j,i));
      Real got    = u0(m,nmhd,k,j,i) - eele0;
      Real err    = Kokkos::fabs(got - expect)/expect;
      if (err > lerr) lerr = err;
      if (Kokkos::fabs(got) > lgot) lgot = Kokkos::fabs(got);
      if (Kokkos::fabs(expect) > lexp) lexp = Kokkos::fabs(expect);
    }, Kokkos::Max<Real>(max_err), Kokkos::Max<Real>(max_got),
       Kokkos::Max<Real>(max_exp));
  max_err = GlobalReduce(max_err, true);

  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(12)
              << "### ele_pdv_test[tab_stencil]: rho0=" << rho0 << " T0=" << t0
              << " max_te_err=" << max_te_err
              << " eele0=" << eele0
              << " max_got=" << max_got << " max_exp=" << max_exp
              << " max_rel_err=" << max_err << std::endl;
  }
  // exact arithmetic (linear v_x -> exact central difference; identical lookups), but
  // the increment is recovered by subtracting the CGS-magnitude eele0, so allow the
  // subtraction's round-off amplification.  The ideal-gamma fallback (gamma-1)*e_ele is
  // off by orders of magnitude on a real table -- the RED discriminator for the
  // tabulated p_ele branch.
  test.CheckTrue(max_err < 1.0e-10,
      "tabulated p_ele: e_ele increment matches dt*a*PressureEle(rho, T_e_cached)");
  test.Finish();
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Seed the periodic, field-free ideal-gas box with a sinusoidal compressive
//! velocity and a uniform-fraction electron energy scalar, record baselines, and enroll
//! the finalize check; the driver then integrates the live MHD update.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### ele_pdv_test FAILED: requires a <mhd> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmhd = pmbp->pmhd;
  if (pmhd->nscalars < 1) {
    std::cout << "### ele_pdv_test FAILED: requires <mhd> nscalars>=1 for the"
              << " electron-energy scalar." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // ---- direct-call stencil leg (problem/check=cyl_stencil): checks the cylindrical
  // divergence metric term with one direct source application, no time integration
  // (the wrapper passes time/nlim=0).  Default "adiabat" runs the driver leg below.
  const std::string check = pin->GetOrAddString("problem","check","adiabat");
  if (check == "cyl_stencil") {
    CylStencilCheck(pin, pmbp);
    return;
  }
  if (check == "tab_stencil") {
    TabStencilCheck(pin, pmbp);
    return;
  }

  if (!pmhd->peos->eos_data.use_e || pmhd->peos->eos_data.is_tabulated) {
    std::cout << "### ele_pdv_test FAILED: the adiabat leg requires the ideal"
              << " (use_e) EOS." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // the #231 gate this run is configured with (read for the finalize branch; the same
  // parameter gates the driver source itself).
  g_gate = pin->GetOrAddBoolean("mhd","ele_pdv",false);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  const Real p0   = pin->GetOrAddReal("problem","p0",0.6);
  const Real v0   = pin->GetOrAddReal("problem","v0",0.35);
  const Real fele = pin->GetOrAddReal("problem","fele",0.4);
  const Real gamma = pmhd->peos->eos_data.gamma;
  const Real eint0 = p0/(gamma - 1.0);

  // field-free: zero all face-centered B (CT keeps it zero with v x B = 0)
  Kokkos::deep_copy(pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  auto u0 = pmhd->u0;
  auto size = pmbp->pmb->mb_size;
  par_for("elepdv_seed", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x1 = size.d_view(m).x1min
            + (static_cast<Real>(i-is) + 0.5)*size.d_view(m).dx1;
    Real vx = -v0*Kokkos::sin(2.0*M_PI*x1);   // compresses the gas around x = 0
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = eint0 + 0.5*rho0*vx*vx;
    u0(m,nmhd,k,j,i) = fele*eint0;             // first passive scalar: e_ele
  });

  // ---- record baselines for the finalize check ----
  g_e0    = ConservedIntegral(pmbp, IEN);
  g_rho0  = rho0;
  g_sele0 = fele*eint0/rho0;
  g_gamma = gamma;

  // enroll the finalize hook (pmy_mesh_->pgen is not yet assigned in this ctor).
  pgen_final_func = ElePdVFinalize;
  return;
}
