//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_multigroup_operator_test.cpp
//  \brief Unit test: N-group flux-limited radiation diffusion (FLDMultigroupOperator)
//  with a per-group Larsen flux limiter and a per-group Rosseland diffusion coefficient,
//  exercised as a parabolic::ParabolicOperator on real device kernels
//  (issue [17a]/#24, ADR-0001/0007).
//
//  Batteries:
//   (A) Group structure FROM THE OPACITY TABLE: ngroups read from the table; the operator
//       precomputes the per-group extinction chi_g = rho*kappa_R,g(rho,te) from the
//       tabulated Rosseland transport opacity, matching the fixture's documented values.
//   (B) Per-group operator action on an insulated 1D cosine eigenmode: with chi_g large
//       enough that R<<3, M(E_g) returns the group's discrete diffusion eigenvalue
//       lambda_g*(E_g-E0), lambda_g = -(2 D_g/dx^2)(1-cos theta), D_g = c/(3 chi_g): the
//       per-group Rosseland D drives a distinct decay rate for each group.
//   (C) Per-group conservation: each group's interior M sums to ~0 (insulated box).
//   (D) ExplicitStableDt is positive and finite (min over groups).
//
//  The fixture path (the optically-thick multigroup-opacity table) is passed on the
//  command line as `problem/opacity_file=<abspath>` by the Python wrapper
//  (tst/test_suite/unit_tests/test_unit_fld_multigroup_operator_cpu.py).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_multigroup_operator_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_multigroup_operator_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::fabs, std::acos
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Multigroup FLD operator (per-group Larsen limiter, per-group Rosseland D) test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_multigroup_operator_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // --- read the optically-thick multigroup-opacity fixture into the representation ---
  std::string fname = pin->GetString("problem", "opacity_file");
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixOpacity(fname, table);

  // ===== (A) group structure read from the table =====
  test.CheckTrue(table.ngroups == 3, "ngroups read from the opacity table == 3");
  const int NG = table.ngroups;

  // Documented fixture: at the background (rho_bg, te_bg) the per-group Rosseland
  // *extinction* chi_g = rho_bg*kappa_R,g is {3000, 1500, 1000} 1/length.
  const Real rho_bg = pin->GetOrAddReal("problem", "rho_bg", 1.0e-2);
  const Real te_bg  = pin->GetOrAddReal("problem", "te_bg", 1.0e2);
  const Real chi_expect[3] = {3000.0, 1500.0, 1000.0};

  // ===== 1D insulated cosine eigenmode (shared by every group) =====
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int ng = indcs.ng;
  const int N = indcs.nx1;
  const int nmb = pmbp->nmb_thispack;
  const int n1 = N + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;

  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real dx = (x1max - x1min)/static_cast<Real>(N);

  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real nl = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real E0 = 1.0;
  const Real amp = 1.0e-2;
  const int m_mode = 4;
  const Real PI = std::acos(-1.0);
  const Real theta = m_mode*PI/static_cast<Real>(N);

  DvceArray5D<Real> erad("erad", nmb, NG, n3, n2, n1);
  DvceArray5D<Real> rhs("rhs", nmb, NG, n3, n2, n1);
  auto h_e = Kokkos::create_mirror_view(erad);
  auto cosmode = [&](int i) -> Real {
    return std::cos(theta*(static_cast<Real>(i - is) + 0.5));
  };
  for (int m = 0; m < nmb; ++m) {
    for (int g = 0; g < NG; ++g) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            h_e(m, g, k, j, i) = E0 + amp*cosmode(i);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(erad, h_e);

  // insulated operator (e_source < 0 => inner-x1 zero-gradient too): the conserving
  // cosine-eigenmode oracle.
  FLDMultigroupOperator op(pmbp, pin, erad, table, c_light, rho_bg, te_bg, nl, -1.0);

  // ===== (A, cont.) per-group extinction matches the documented fixture values =====
  auto d_chi = op.chi();
  auto h_chi = Kokkos::create_mirror_view(d_chi);
  Kokkos::deep_copy(h_chi, d_chi);
  for (int g = 0; g < NG; ++g) {
    test.CheckNear(h_chi(g), chi_expect[g], 1.0e-6, 0.0,
                   "per-group chi_g = rho*kappa_R,g from Rosseland transport opacity");
  }

  // ===== (D) explicit dt positive/finite =====
  const Real dt_exp = op.ExplicitStableDt();
  test.CheckTrue(dt_exp > 0.0 && dt_exp < 1.0e30,
                 "multigroup FLD explicit stable dt finite and >0");

  // ===== (B) per-group operator action == per-group diffusion eigenvalue =====
  op.ApplyBoundary(erad);
  op.OperatorAction(erad, rhs);
  auto h_rhs = Kokkos::create_mirror_view(rhs);
  Kokkos::deep_copy(h_rhs, rhs);
  for (int g = 0; g < NG; ++g) {
    const Real D_g = c_light/(3.0*chi_expect[g]);   // thick-limit per-group diffusivity
    const Real lambda_g = -(2.0*D_g/(dx*dx))*(1.0 - std::cos(theta));
    Real maxerr = 0.0;
    for (int i = is; i <= ie; ++i) {
      Real expected = lambda_g*(amp*cosmode(i));    // lambda_g*(E_i - E0)
      maxerr = std::fmax(maxerr, std::fabs(h_rhs(0, g, 0, 0, i) - expected));
    }
    test.CheckNear(maxerr, 0.0, 0.0, 1.0e-5*std::fabs(lambda_g)*amp,
                   "M(E_g) == lambda_g*(E_g-E0): per-group D_g = c/(3 chi_g)");
  }

  // ===== (B, cont.) groups diffuse at DISTINCT rates (group resolution) =====
  // The thinnest group (smallest chi) has the largest |M| amplitude; the thickest the
  // smallest.  Require a strict ordering so a bug collapsing all groups to one D fails.
  Real amp_thick = 0.0, amp_thin = 0.0;   // group 0 (chi=3000) vs group NG-1 (chi=1000)
  for (int i = is; i <= ie; ++i) {
    amp_thick = std::fmax(amp_thick, std::fabs(h_rhs(0, 0, 0, 0, i)));
    amp_thin  = std::fmax(amp_thin,  std::fabs(h_rhs(0, NG-1, 0, 0, i)));
  }
  test.CheckTrue(amp_thin > 1.5*amp_thick,
                 "thinner group diffuses faster (distinct per-group D, not collapsed)");

  // ===== (C) each group's insulated operator conserves: interior M sums to ~0 =====
  for (int g = 0; g < NG; ++g) {
    Real sum_m = 0.0, scale = 0.0;
    for (int i = is; i <= ie; ++i) {
      sum_m += h_rhs(0, g, 0, 0, i);
      scale = std::fmax(scale, std::fabs(h_rhs(0, g, 0, 0, i)));
    }
    test.CheckNear(sum_m, 0.0, 0.0, 1.0e-10*(scale*static_cast<Real>(N) + 1.0e-30),
                   "insulated multigroup FLD conserves energy per group (M sums to 0)");
  }

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
