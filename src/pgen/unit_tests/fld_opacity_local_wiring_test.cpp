//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_opacity_local_wiring_test.cpp
//  \brief Unit test: the <mhd> fld_opacity_local wiring (issue #204).  Selecting
//  eos=tabulated_3t + fld_operator_split + fld_opacity_local on a fully-initialized MHD
//  package must (a) parse the knob, (b) read the multigroup opacity table (defaulting to
//  the SAME cn4 file as the EOS, with the ion-number-density axis rescaled to g/cc via
//  eos_mass_per_ion), (c) allocate the per-cell chi field and register it with the grey
//  FLD operator (EnableLocalChi), and (d) expose MHD::RefreshFldLocalChi(), which fills
//  the field from the live u0 through the tabulated Te closure -- so a two-zone
//  liner|vacuum conserved state refreshes to an opaque-liner / transparent-gap chi
//  (the #204 acceptance property, through the production member path).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_opacity_local_wiring_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_opacity_local_wiring_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mhd/mhd.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief <mhd> fld_opacity_local wiring unit test (#204).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_opacity_local_wiring_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  mhd::MHD *pmhd = pmbp->pmhd;

  // (a,b,c) the knob parsed, the operator exists and is in local-chi mode.
  test.CheckTrue(pmhd != nullptr, "MHD package constructed");
  test.CheckTrue(pmhd->fld_opacity_local, "fld_opacity_local knob parsed true");
  test.CheckTrue(pmhd->pfld_op != nullptr, "grey FLD operator constructed");
  test.CheckTrue(pmhd->pfld_op->local_chi(),
                 "grey FLD operator registered the per-cell chi field");

  // two-zone conserved state: liner (rho=1 code) left half, vacuum gap (1e-4) right
  // half, both at Te = 10 eV through the package's own code-scaled EOS closure.
  auto &indcs = pmy_mesh_->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const int imid = n1/2;
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto u0 = pmhd->u0;
  auto eos = pmhd->eos_tbl;
  const int eidx = pmhd->nmhd;
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

  // (d) the production refresh path fills the member field from u0.
  pmhd->RefreshFldLocalChi();

  auto h_chi = Kokkos::create_mirror_view(pmhd->fld_chi_cell);
  Kokkos::deep_copy(h_chi, pmhd->fld_chi_cell);
  const Real chi_liner = h_chi(0, 0, indcs.ks, indcs.js, indcs.is);
  const Real chi_vac   = h_chi(0, 0, indcs.ks, indcs.js, indcs.ie);
  test.CheckTrue(chi_liner > 0.0 && chi_vac > 0.0, "refreshed chi positive");
  test.CheckTrue(chi_liner >= 1.0e3*chi_vac,
                 "liner chi >= 1e3 x vacuum chi through the production member path");
  // ghosts are refreshed too (the flux kernels read one cell into the ghost region).
  test.CheckTrue(h_chi(0, 0, indcs.ks, indcs.js, 0) > 0.0 &&
                 h_chi(0, 0, indcs.ks, indcs.js, n1-1) > 0.0,
                 "ghost-cell chi refreshed");

  test.Finish();
}
