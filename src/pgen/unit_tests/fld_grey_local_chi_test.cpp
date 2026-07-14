//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_grey_local_chi_test.cpp
//  \brief Unit test: FLDGreyOperator with a PER-CELL extinction field chi(m,k,j,i)
//  (EnableLocalChi, issue #204) in place of the frozen scalar constant.  The per-cell
//  chi is what lets the MagLIF grey FLD treat the dense liner as optically thick and the
//  tenuous vacuum gap as transparent -- with the frozen solid-density constant the gap
//  is opaque and radiation cannot free-stream across it (the rank-1 B1 faithfulness gap).
//
//  Batteries (all through the public ParabolicOperator interface):
//   (A) EQUIVALENCE: a chi field UNIFORMLY filled with the scalar value reproduces the
//       constant-chi operator's M(u) and ExplicitStableDt exactly (same arithmetic) --
//       the local mode degrades gracefully to the legacy behavior.
//   (B) TRANSPARENT GAP: on a hot-slab/cold-slab step profile, a thick|thin two-zone chi
//       field transports orders of magnitude more energy out of the interface cells than
//       the all-thick constant operator (the flux-limited flux rises toward the
//       free-streaming cap c*E instead of the diffusion-suppressed c/(3 chi) dE/dx).
//   (C) LOCALITY of the stability dt: making half the domain transparent (huge D in its
//       flat interior) tightens ExplicitStableDt by orders of magnitude relative to the
//       all-thick operator -- the dt bound reads the LOCAL chi, not a global constant.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_grey_local_chi_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_grey_local_chi_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Per-cell (local) chi field for the grey FLD operator: unit test (#204).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_grey_local_chi_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int ng = indcs.ng;
  const int N = indcs.nx1;
  const int nmb = pmbp->nmb_thispack;
  const int n1 = N + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;

  const Real c_light  = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real chi_thick = pin->GetOrAddReal("problem", "chi_thick", 1.0e6);
  const Real chi_thin  = pin->GetOrAddReal("problem", "chi_thin", 1.0e-6);
  const Real nl = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real efloor = pin->GetOrAddReal("problem", "fld_efloor", 1.0e-10);

  // hot-slab | cold-vacuum step: E = 1 on the left half, efloor on the right half.
  const int imid = is + N/2;
  DvceArray5D<Real> erad("erad", nmb, 1, n3, n2, n1);
  auto h_e = Kokkos::create_mirror_view(erad);
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_e(m, 0, k, j, i) = (i < imid) ? 1.0 : efloor;
        }
      }
    }
  }
  Kokkos::deep_copy(erad, h_e);

  // per-cell chi fields: uniform-thick (equivalence battery) and thick|thin two-zone
  // (transparent-gap battery; thin on the cold right half).
  DvceArray5D<Real> chi_uniform("chi_uni", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> chi_twozone("chi_two", nmb, 1, n3, n2, n1);
  auto h_c = Kokkos::create_mirror_view(chi_twozone);
  Kokkos::deep_copy(chi_uniform, chi_thick);
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_c(m, 0, k, j, i) = (i < imid) ? chi_thick : chi_thin;
        }
      }
    }
  }
  Kokkos::deep_copy(chi_twozone, h_c);

  DvceArray5D<Real> rhs_const("rhs_c", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> rhs_local("rhs_l", nmb, 1, n3, n2, n1);

  // insulated operators (esrc < 0): identical constant-chi baseline vs local-chi mode.
  FLDGreyOperator op_const(pmbp, pin, erad, c_light, chi_thick, nl, -1.0, efloor);
  FLDGreyOperator op_local(pmbp, pin, erad, c_light, chi_thick, nl, -1.0, efloor);

  const Real dt_const = op_const.ExplicitStableDt();
  op_const.ApplyBoundary(erad);
  op_const.OperatorAction(erad, rhs_const);

  // ===== (A) uniform local chi == scalar chi (M(u) and dt) =====
  op_local.EnableLocalChi(chi_uniform);
  const Real dt_uni = op_local.ExplicitStableDt();
  op_local.ApplyBoundary(erad);
  op_local.OperatorAction(erad, rhs_local);

  auto h_rc = Kokkos::create_mirror_view(rhs_const);
  auto h_rl = Kokkos::create_mirror_view(rhs_local);
  Kokkos::deep_copy(h_rc, rhs_const);
  Kokkos::deep_copy(h_rl, rhs_local);
  Real max_dev = 0.0, max_mag = 0.0;
  for (int m = 0; m < nmb; ++m) {
    for (int i = is; i <= ie; ++i) {
      Real dev = std::fabs(h_rl(m, 0, indcs.ks, indcs.js, i)
                           - h_rc(m, 0, indcs.ks, indcs.js, i));
      Real mag = std::fabs(h_rc(m, 0, indcs.ks, indcs.js, i));
      if (dev > max_dev) { max_dev = dev; }
      if (mag > max_mag) { max_mag = mag; }
    }
  }
  test.CheckTrue(max_mag > 0.0, "constant-chi baseline produces a nonzero M(u)");
  test.CheckNear(max_dev, 0.0, 0.0, 1.0e-14*max_mag,
                 "uniform per-cell chi reproduces the constant-chi M(u)");
  test.CheckNear(dt_uni, dt_const, 1.0e-14, 0.0,
                 "uniform per-cell chi reproduces the constant-chi ExplicitStableDt");

  // ===== (B) thick|thin two-zone chi: the transparent gap transports =====
  op_local.EnableLocalChi(chi_twozone);
  op_local.ApplyBoundary(erad);
  op_local.OperatorAction(erad, rhs_local);
  Kokkos::deep_copy(h_rl, rhs_local);

  // energy drain out of the hot interface cell (i = imid-1): local-thin vs const-thick.
  Real drain_local = 0.0, drain_const = 0.0;
  for (int m = 0; m < nmb; ++m) {
    drain_local += -h_rl(m, 0, indcs.ks, indcs.js, imid-1);
    drain_const += -h_rc(m, 0, indcs.ks, indcs.js, imid-1);
  }
  test.CheckTrue(drain_local > 0.0,
                 "transparent gap drains the hot interface cell (M < 0 there)");
  test.CheckTrue(drain_local > 1.0e3*std::fabs(drain_const),
                 "transparent gap transports >>1e3x the all-thick constant operator");

  // ===== (C) the stability dt reads the local chi =====
  const Real dt_local = op_local.ExplicitStableDt();
  test.CheckTrue(dt_local > 0.0 && dt_local < 1.0e30,
                 "local-chi ExplicitStableDt finite and > 0");
  test.CheckTrue(dt_local < 1.0e-3*dt_const,
                 "transparent half tightens the stability dt by >1e3 (local D honored)");

  test.Finish();
}
