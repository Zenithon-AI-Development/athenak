//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tabulated_mhd_c2p_test.cpp
//  \brief Unit test: the tabulated 3T (IONMIX) EOS wired into the live MHD solver
//  (issue [P1]/#159, ADR-0002/0007).
//
//  Exercises the new end-to-end path on a real device kernel inside a fully-initialized
//  AthenaK MHD package:
//   (AC#1) The mesh constructs the MHD package with <mhd> eos=tabulated_3t reading the
//          real-material aluminum .cn4 table.  Merely reaching UserProblem proves the
//          mhd.cpp EOS selector accepted "tabulated_3t" and ionmix_eos_reader populated
//          the EosTable3T held on the package (else construction would have FATAL'd).  We
//          additionally assert the table is populated and the selected EOS is non-ideal.
//   (AC#2) Seed a uniform conserved state from KNOWN (rho, T_e, T_i) inside the table
//          (the electron internal energy density rides the first passive scalar; the ion
//          energy is recovered by subtraction), then run the package's own
//          peos->ConsToPrim once -- the per-step cons->prim work -- which routes through
//          eos_table_3t::ConsToPrim2T.  The derived, cached temperatures it stores must
//          match the seeded T_e/T_i (table-consistent inversion, the same round-trip the
//          cons_to_prim_2t unit test verifies, now with the real Al table).  A
//          cons->prim->cons round-trip (peos->PrimToCons) must preserve the conserved
//          total energy and the electron-energy scalar.
//
//  The table bounds are queried at runtime and the probe (rho, T_e, T_i) chosen at the
//  log-midpoints, so no table value is hard-coded / fabricated (ADR-0008).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/tabulated_mhd_c2p_test -B build_unit/... && make
//    ./athena -i inputs/unit_tests/tabulated_mhd_c2p_test.athinput \
//        mhd/eos_table=<abspath to a real .cn4>
//  Auto-run by tst/test_suite/unit_tests/test_unit_tabulated_mhd_c2p_cpu.py.

#include <cmath>     // std::sqrt, std::log10, std::pow
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "eos/eos_table_3t.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Tabulated 3T (IONMIX) EOS live-MHD cons->prim wiring unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("tabulated_mhd_c2p_test");

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR: tabulated_mhd_c2p_test requires a <mhd> block."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *pmhd = pmbp->pmhd;

  // ---------- (AC#1) selector accepted tabulated_3t + table loaded on the package ----
  test.CheckTrue(pmhd->eos_tbl.ntemp > 1 && pmhd->eos_tbl.nrho > 1,
                 "IONMIX (.cn4) EOS table loaded into the MHD package (ntemp,nrho > 1)");
  test.CheckTrue(!pmhd->peos->eos_data.is_ideal,
                 "selected EOS is the tabulated (non-ideal) MHD closure");
  test.CheckTrue(pmhd->nmhd == 5,
                 "tabulated_3t MHD uses 5 conserved variables");

  // ---------- choose a known uniform state inside the table (no hard-coded values) ----
  // Host-accessible axis scalars; the table Views live on the device.
  const eos_table_3t::EosTable3T &tbl = pmhd->eos_tbl;
  const Real rho     = std::sqrt(tbl.RhoMin()*tbl.RhoMax());   // log-midpoint density
  const Real lt_min  = std::log10(tbl.TempMin());
  const Real lt_max  = std::log10(tbl.TempMax());
  const Real te_true = std::pow(10.0, 0.5*lt_min + 0.5*lt_max);  // mid-log T_e
  const Real ti_true = std::pow(10.0, 0.6*lt_min + 0.4*lt_max);  // distinct (lower) T_i
  const Real bx0     = 1.0;   // uniform Bx (exercises the magnetic-energy subtraction)

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmhd = pmhd->nmhd;
  const int nmb1 = pmbp->nmb_thispack - 1;

  // uniform face-centered B = (bx0, 0, 0) -> cell-centered average bx0 everywhere
  Kokkos::deep_copy(pmhd->b0.x1f, bx0);
  Kokkos::deep_copy(pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pmhd->b0.x3f, 0.0);

  // ---------- seed the conserved state on device using the table forward lookups ----
  auto u0 = pmhd->u0;
  eos_table_3t::EosTable3T tbl_d = pmhd->eos_tbl;   // shallow View copy into the kernel
  par_for("tab_mhd_seed", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // specific -> density: e = rho * e_specific(rho, T)
    Real e_ele = rho*tbl_d.EnergyEle(rho, te_true);
    Real e_ion = rho*tbl_d.EnergyIon(rho, ti_true);
    Real me    = 0.5*(bx0*bx0);                 // magnetic energy density (v=0 -> KE=0)
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    u0(m,IEN,k,j,i) = e_ele + e_ion + me;       // conserved total energy
    u0(m,nmhd,k,j,i) = e_ele;                    // electron energy density (first scalar)
  });

  // ---------- (AC#2) run the package's own cons->prim; verify table-consistent T ----
  pmhd->peos->ConsToPrim(pmhd->u0, pmhd->b0, pmhd->w0, pmhd->bcc0, false,
                         is, ie, js, je, ks, ke);

  auto te_h = Kokkos::create_mirror_view(pmhd->derived_te);
  auto ti_h = Kokkos::create_mirror_view(pmhd->derived_ti);
  Kokkos::deep_copy(te_h, pmhd->derived_te);
  Kokkos::deep_copy(ti_h, pmhd->derived_ti);
  const Real rt = 1.0e-7;   // round-trip inversion tolerance (real-table log-interp)
  test.CheckNear(te_h(0,0,ks,js,is), te_true, rt, 0.0,
                 "T_e recovered from e_ele/rho via ConsToPrim2T matches the table value");
  test.CheckNear(ti_h(0,0,ks,js,is), ti_true, rt, 0.0,
                 "T_i recovered from e_ion/rho via ConsToPrim2T matches the table value");
  test.CheckTrue(te_h(0,0,ks,js,is) > ti_h(0,0,ks,js,is),
                 "genuine 2T state recovered (T_e > T_i as seeded)");

  // ---------- cons->prim->cons round-trip conserves total energy + electron scalar ----
  auto u_h = Kokkos::create_mirror_view(pmhd->u0);
  Kokkos::deep_copy(u_h, pmhd->u0);
  const Real etot_pre = u_h(0,IEN,ks,js,is);
  const Real eele_pre = u_h(0,nmhd,ks,js,is);
  pmhd->peos->PrimToCons(pmhd->w0, pmhd->bcc0, pmhd->u0, is, ie, js, je, ks, ke);
  Kokkos::deep_copy(u_h, pmhd->u0);
  test.CheckNear(u_h(0,IEN,ks,js,is), etot_pre, 1.0e-12, 0.0,
                 "cons->prim->cons preserves the conserved total energy");
  test.CheckNear(u_h(0,nmhd,ks,js,is), eele_pre, 1.0e-12, 0.0,
                 "cons->prim->cons preserves the electron internal-energy scalar");

  test.Finish();
  return;
}
