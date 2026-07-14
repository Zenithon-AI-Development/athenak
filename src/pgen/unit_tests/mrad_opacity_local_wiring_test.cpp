//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mrad_opacity_local_wiring_test.cpp
//  \brief Unit test: the <mhd> mrad_opacity_local wiring (issue #204).  The grey
//  matter-radiation coupling's Planck absorption coefficient chi_a, frozen at the
//  solid-liner reference value since #184, becomes a per-cell lookup
//  chi_a(rho,Te) = kappa_P(rho,Te)*rho_cgs*L (the grey Planck mean of the IONMIX
//  multigroup table) -- so the tenuous vacuum gap stops absorbing/emitting at solid
//  opacity (spurious equilibrium-locking of near-massless material) and the coupling
//  becomes consistent with the transparent-gap FLD transport (fld_opacity_local).
//
//  Setup: a fully-initialized tabulated_3t MHD package with fld_operator_split +
//  mrad_coupling + mrad_eos_aware + fld_opacity_local + mrad_opacity_local, a two-zone
//  liner|vacuum conserved state at 10 eV, uniform erad, one production
//  MatterRadCouplingHalf step.  Oracle (device, public kernels): the post-step
//  (erad, IEN) per cell equals PointImplicitGreyCoupling evaluated with the LOCAL
//  chi_a(rho,Te) chain -- a constant-chi_a coupling diverges from this strongly in the
//  vacuum cell -- and the vacuum cell's exchanged energy is <<< the liner cell's.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/mrad_opacity_local_wiring_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_mrad_opacity_local_wiring_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "diffusion/operator_si_calibration.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "radiation_fld/grey_opacity_mean.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief <mhd> mrad_opacity_local wiring unit test (#204).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("mrad_opacity_local_wiring_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  mhd::MHD *pmhd = pmbp->pmhd;

  test.CheckTrue(pmhd->mrad_opacity_local, "mrad_opacity_local knob parsed true");

  // --- two-zone conserved state at rest, B=0, Te=Ti=10 eV; uniform erad ---
  auto &indcs = pmy_mesh_->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int imid = n1/2;
  const int nmb1 = pmbp->nmb_thispack - 1;
  const Real erad0 = 1.0e-3;
  auto u0 = pmhd->u0;
  auto er = pmhd->erad;
  auto bcc = pmhd->bcc0;
  auto eos = pmhd->eos_tbl;
  const int eidx = pmhd->nmhd;
  Kokkos::deep_copy(bcc, 0.0);
  Kokkos::deep_copy(er, erad0);
  par_for("fill_u0", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho = (i < imid) ? 1.0 : 1.0e-4;
    u0(m, IDN, k, j, i) = rho;
    u0(m, IM1, k, j, i) = 0.0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
    Real e_ele = rho*eos.EnergyEle(rho, 10.0);
    u0(m, eidx, k, j, i) = e_ele;
    u0(m, IEN, k, j, i) = e_ele + rho*eos.EnergyIon(rho, 10.0);
  });

  // --- device oracle BEFORE the step: expected post-step (erad, IEN) per zone,
  //     through the same public kernels with the LOCAL chi_a(rho,Te) chain ---
  const Real dt_step = 1.0e-3;
  pmy_mesh_->dt = dt_step;
  const Real half_dt = 0.5*dt_step;
  const Real cl = pmhd->mrad_clight, arad = pmhd->mrad_arad;
  const Real chia_const = pmhd->mrad_chi_a;
  const Real dens_cgs = pmhd->fld_opac_dens_cgs, len_cgs = pmhd->fld_opac_len_cgs;
  auto opac = pmhd->fld_opac_tbl;
  test.CheckTrue(opac.ngroups > 0, "opacity table loaded on the package");

  enum { IER_LIN=0, IEN_LIN, IER_VAC, IEN_VAC, ICHIA_LIN, ICHIA_VAC, NVAL };
  DvceArray1D<Real> d_exp("exp", NVAL);
  DvceArray1D<int> d_cells("cells", 2);
  auto h_cells = Kokkos::create_mirror_view(d_cells);
  h_cells(0) = indcs.is; h_cells(1) = indcs.ie;
  Kokkos::deep_copy(d_cells, h_cells);
  const int ks = indcs.ks, js = indcs.js;
  par_for("oracle", DevExeSpace(), 0, 1, KOKKOS_LAMBDA(const int z) {
    const int i = d_cells(z);
    Real rho = u0(0, IDN, ks, js, i);
    Real e_ele = u0(0, eidx, ks, js, i);
    Real e_gas = u0(0, IEN, ks, js, i);      // at rest, B=0: all internal
    Real te = eos.Te(rho, e_ele/rho);
    Real ti = eos.Ti(rho, (e_gas - e_ele)/rho);
    Real cv_cell = rho*(eos.CvEle(rho, te) + eos.CvIon(rho, ti));
    Real rho_cgs = rho*dens_cgs;
    Real kap = radiationfld::GreyPlanckMean(opac, rho_cgs, te, 1.0);
    Real chia = op_si_calib::OpacityCodeFromCgs(kap, rho_cgs, len_cgs);
    Real er_new, eg_new;
    radiationfld::PointImplicitGreyCoupling(erad0, e_gas, cv_cell, chia, cl, arad,
                                            half_dt, er_new, eg_new);
    d_exp(2*z + 0) = er_new;
    d_exp(2*z + 1) = eg_new;
    d_exp(ICHIA_LIN + z) = chia;
  });
  auto h_exp = Kokkos::create_mirror_view(d_exp);
  Kokkos::deep_copy(h_exp, d_exp);

  // sanity: the local chain itself is transparent in the gap (else the oracle proves
  // nothing) and differs from the frozen constant there.
  test.CheckTrue(h_exp(ICHIA_LIN) >= 1.0e3*h_exp(ICHIA_VAC),
                 "local chi_a: liner >= 1e3 x vacuum");
  test.CheckTrue(h_exp(ICHIA_VAC) < 0.5*chia_const,
                 "local vacuum chi_a well below the frozen constant");

  // --- the production step under test ---
  pmhd->MatterRadCouplingHalf(nullptr, 0);

  auto h_u = Kokkos::create_mirror_view(u0);
  auto h_e = Kokkos::create_mirror_view(er);
  Kokkos::deep_copy(h_u, u0);
  Kokkos::deep_copy(h_e, er);

  test.CheckNear(h_e(0, 0, ks, js, indcs.is), h_exp(IER_LIN), 1.0e-12, 0.0,
                 "liner erad matches the local-chi_a point-implicit oracle");
  test.CheckNear(h_u(0, IEN, ks, js, indcs.is), h_exp(IEN_LIN), 1.0e-12, 0.0,
                 "liner IEN matches the local-chi_a point-implicit oracle");
  test.CheckNear(h_e(0, 0, ks, js, indcs.ie), h_exp(IER_VAC), 1.0e-12, 0.0,
                 "vacuum erad matches the local-chi_a point-implicit oracle");
  test.CheckNear(h_u(0, IEN, ks, js, indcs.ie), h_exp(IEN_VAC), 1.0e-12, 0.0,
                 "vacuum IEN matches the local-chi_a point-implicit oracle");

  // physics: the vacuum cell exchanges far less energy than the liner cell.
  const Real dlin = std::fabs(h_e(0, 0, ks, js, indcs.is) - erad0);
  const Real dvac = std::fabs(h_e(0, 0, ks, js, indcs.ie) - erad0);
  test.CheckTrue(dlin > 0.0, "liner cell exchanges energy");
  test.CheckTrue(dvac < 1.0e-2*dlin,
                 "vacuum cell exchange suppressed vs liner (transparent coupling)");

  test.Finish();
}
