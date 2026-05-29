//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_grey_operator_test.cpp
//  \brief Unit test: grey flux-limited radiation diffusion (FLDGreyOperator) with the
//  Larsen flux limiter, exercised as a parabolic::ParabolicOperator on real device
//  kernels (issue [8a]/#17, ADR-0001).
//
//  Batteries:
//   (A) Larsen limiter limits (device): lambda(0)=1/3 (diffusion), lambda*R<1 with
//       lambda*R->1 as R->inf (the free-streaming flux cap |F|=c E lambda R <= c E), and
//       lambda(R) monotonically decreasing.
//   (B) Operator action in the optically-thick limit: on an insulated 1D cosine eigenmode
//       with chi so large that R<<3, M(E) returns the discrete diffusion eigenvalue
//       lambda_eig*(E-E0), lambda_eig = -(2D/dx^2)(1-cos theta), D = c/(3 chi) -- pins
//       the FLD flux divergence AND the thick-limit limiter/coefficient D=c lambda/chi.
//   (C) Conservation: the insulated (zero-gradient) operator's M(u) sums to ~0 over the
//       interior (no flux through the domain boundary).
//   (D) ExplicitStableDt is positive and finite.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_grey_operator_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_grey_operator_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::fabs, std::acos
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Grey FLD operator (Larsen limiter) unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_grey_operator_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // ===== (A) Larsen flux limiter limits (computed on device) =====
  const int nR = 7;
  DvceArray1D<Real> d_lam("lam", nR), d_R("R", nR);
  auto h_R = Kokkos::create_mirror_view(d_R);
  h_R(0) = 0.0; h_R(1) = 0.1; h_R(2) = 1.0; h_R(3) = 3.0;
  h_R(4) = 10.0; h_R(5) = 1.0e2; h_R(6) = 1.0e4;
  Kokkos::deep_copy(d_R, h_R);
  const Real nl = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  par_for("fld_lam", DevExeSpace(), 0, nR-1, KOKKOS_LAMBDA(const int n) {
    d_lam(n) = radiationfld::LarsenLimiter(d_R(n), nl);
  });
  auto h_lam = Kokkos::create_mirror_view(d_lam);
  Kokkos::deep_copy(h_lam, d_lam);

  test.CheckNear(h_lam(0), 1.0/3.0, 0.0, 1.0e-12,
                 "Larsen limiter -> 1/3 in the optically-thick limit R=0");
  // free-streaming cap: |F| = c E lambda R <= c E, i.e. lambda*R < 1 for all finite R,
  // and lambda*R -> 1 as R -> inf.
  bool cap_ok = true, mono_ok = true;
  for (int n = 0; n < nR; ++n) {
    if (h_lam(n)*h_R(n) > 1.0) { cap_ok = false; }
    if (n > 0 && h_lam(n) >= h_lam(n-1)) { mono_ok = false; }
  }
  test.CheckTrue(cap_ok, "Larsen lambda*R <= 1 (radiative flux <= free-streaming c*E)");
  test.CheckTrue(mono_ok, "Larsen limiter decreases monotonically with R");
  test.CheckNear(h_lam(nR-1)*h_R(nR-1), 1.0, 0.0, 1.0e-3,
                 "Larsen lambda -> 1/R (free-streaming) as R -> inf");

  // ===== set up a 1D insulated cosine eigenmode for the operator action =====
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
  const Real chi = pin->GetOrAddReal("problem", "chi", 1.0e6);  // huge => thick limit
  const Real E0 = 1.0;
  const Real amp = 1.0e-2;
  const int m_mode = 4;
  const Real PI = std::acos(-1.0);
  const Real theta = m_mode*PI/static_cast<Real>(N);
  const Real D = c_light/(3.0*chi);                        // thick-limit diffusivity
  const Real lambda_eig = -(2.0*D/(dx*dx))*(1.0 - std::cos(theta));

  DvceArray5D<Real> erad("erad", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> rhs("rhs", nmb, 1, n3, n2, n1);
  auto h_e = Kokkos::create_mirror_view(erad);
  auto cosmode = [&](int i) -> Real {
    return std::cos(theta*(static_cast<Real>(i - is) + 0.5));
  };
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_e(m, 0, k, j, i) = E0 + amp*cosmode(i);
        }
      }
    }
  }
  Kokkos::deep_copy(erad, h_e);

  // insulated operator (esrc < 0 => inner-x1 zero-gradient too): exercises the conserving
  // cosine eigenmode oracle.
  FLDGreyOperator op(pmbp, pin, erad, c_light, chi, nl, -1.0);

  // ===== (D) explicit dt positive/finite =====
  const Real dt_exp = op.ExplicitStableDt();
  test.CheckTrue(dt_exp > 0.0 && dt_exp < 1.0e30, "FLD explicit stable dt finite and >0");

  // ===== (B) operator action == thick-limit diffusion eigenvalue =====
  op.ApplyBoundary(erad);
  op.OperatorAction(erad, rhs);
  auto h_rhs = Kokkos::create_mirror_view(rhs);
  Kokkos::deep_copy(h_rhs, rhs);
  Real maxerrB = 0.0;
  for (int i = is; i <= ie; ++i) {
    Real expected = lambda_eig*(amp*cosmode(i));         // lambda_eig*(E_i - E0)
    maxerrB = std::fmax(maxerrB, std::fabs(h_rhs(0, 0, 0, 0, i) - expected));
  }
  test.CheckNear(maxerrB, 0.0, 0.0, 1.0e-6*std::fabs(lambda_eig)*amp,
                 "M(E) == lambda_eig*(E-E0): FLD operator is diffusion with D=c/(3 chi)");

  // ===== (C) insulated operator conserves: interior M sums to ~0 =====
  Real sum_m = 0.0, scale = 0.0;
  for (int i = is; i <= ie; ++i) {
    sum_m += h_rhs(0, 0, 0, 0, i);
    scale = std::fmax(scale, std::fabs(h_rhs(0, 0, 0, 0, i)));
  }
  test.CheckNear(sum_m, 0.0, 0.0, 1.0e-10*(scale*static_cast<Real>(N) + 1.0e-30),
                 "insulated FLD operator conserves energy (interior M sums to 0)");

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
