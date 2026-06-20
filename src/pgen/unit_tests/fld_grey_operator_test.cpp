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
#include "driver/parabolic_integrator.hpp"
#include "driver/composite_parabolic_operator.hpp"
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
  const Real efloor = pin->GetOrAddReal("problem", "fld_efloor", 1.0e-10);
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
  FLDGreyOperator op(pmbp, pin, erad, c_light, chi, nl, -1.0, efloor);

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

  // ===== (E) free-streaming super-step positivity (issue #194) =====
  // In the optically-thin / free-streaming regime (small chi + steep gradient) the Larsen
  // limiter makes the FLD operator HYPERBOLIC (advective at speed c).  RKL2's parabolic
  // stability polynomial damps only the negative-real axis, so the c-advective (off-axis)
  // spectrum is UNDAMPED: a high-wavenumber (checkerboard) mode in a sustained
  // free-streaming gap is amplified each super-step, overshoots erad NEGATIVE, and with
  // no positivity floor -- the 1e-30 divide-guard then makes R = |grad E|/(chi E) and the
  // flux blow up to NaN (#194; observed in the coupled run at t~33 ns).  Reproduce that
  // accumulation faithfully: a HOT Dirichlet wall (esrc>0) driving radiation into a
  // checkerboard-seeded near-vacuum gap, advanced many real RKL2 super-steps
  // (parabolic::OperatorSplitStep -- the wired OperatorSplitFLD path).  Require erad to
  // stay FINITE and >= 0 throughout.  RED on the unfloored operator, GREEN once the
  // positivity floor + post-super-step projection land.  All knobs are deck-tunable, so
  // trigger can be sharpened without a rebuild.
  {
    const Real chi_thin = pin->GetOrAddReal("problem", "chi_thin", 1.0);    // free-stream
    const Real esrc_thin = pin->GetOrAddReal("problem", "esrc_thin", 1.0);  // hot wall
    const Real e_bg = pin->GetOrAddReal("problem", "e_bg", 1.0e-6);         // vacuum bg
    const Real seed = pin->GetOrAddReal("problem", "seed_amp", 1.0e-7);     // checkrbd
    const Real dt_fac = pin->GetOrAddReal("problem", "dt_fac", 5.0e1);      // dt/dt_exp
    const int nstep = static_cast<int>(pin->GetOrAddReal("problem", "nstep", 1.0e2));

    DvceArray5D<Real> et("erad_thin", nmb, 1, n3, n2, n1);
    auto h_et = Kokkos::create_mirror_view(et);
    for (int m = 0; m < nmb; ++m) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            // near-vacuum gap with a tiny checkerboard seed (the most-amplified mode).
            h_et(m, 0, k, j, i) = e_bg + seed*(((i % 2) == 0) ? 1.0 : -1.0);
          }
        }
      }
    }
    Kokkos::deep_copy(et, h_et);
    FLDGreyOperator opt(pmbp, pin, et, c_light, chi_thin, nl, esrc_thin, efloor);  // wall
    parabolic::ParabolicIntegrator integ(parabolic::ParabolicMethod::sts);
    for (int step = 0; step < nstep; ++step) {
      const Real dt_exp = opt.ExplicitStableDt();
      parabolic::OperatorSplitStep(integ, opt, et, dt_fac*dt_exp);
    }
    auto h_out = Kokkos::create_mirror_view(et);
    Kokkos::deep_copy(h_out, et);
    bool finite = true;
    Real min_e = 1.0e300;
    for (int i = is; i <= ie; ++i) {
      const Real e = h_out(0, 0, 0, 0, i);
      if (!std::isfinite(e)) { finite = false; }
      min_e = std::fmin(min_e, e);
    }
    std::cout << "[fld_grey_operator_test] (E) free-streaming drive: min(erad)=" << min_e
              << " finite=" << finite << " (nstep=" << nstep << ", dt_fac=" << dt_fac
              << ", chi=" << chi_thin << ")" << std::endl;
    test.CheckTrue(finite,
                   "free-streaming FLD super-step drive stays finite (no NaN) (#194)");
    test.CheckTrue(min_e >= -1.0e-12,
                   "free-streaming FLD super-step drive keeps erad >= 0 (#194)");
  }

  // ===== (F) PostSuperstepProject floors negatives to efloor + accounts injection ======
  // Battery E reaches green via the read-floor alone (min(erad) stays positive), so the
  // projection + its energy accounting would otherwise be untested.  Seed a field with
  // sub-floor and supra-floor cells; the projection must raise every active cell to >=
  // efloor, leave supra-floor cells untouched, and report injected energy = sum over
  // floored cells of (efloor - e)*cell_volume.  (op was built with this deck's efloor.)
  {
    DvceArray5D<Real> ep("erad_proj", nmb, 1, n3, n2, n1);
    auto h_ep = Kokkos::create_mirror_view(ep);
    Real expect_inj = 0.0;
    const Real dvol = dx;  // 1D Cartesian unit-test cell volume (dx2=dx3=1)
    for (int m = 0; m < nmb; ++m) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            Real e = ((i % 2) == 0) ? -2.0 : 5.0;  // sub/supra-floor
            h_ep(m, 0, k, j, i) = e;
            if (i >= is && i <= ie && (efloor - e) > 0.0) {
              expect_inj += (efloor - e)*dvol;
            }
          }
        }
      }
    }
    Kokkos::deep_copy(ep, h_ep);
    op.PostSuperstepProject(ep);
    auto h_po = Kokkos::create_mirror_view(ep);
    Kokkos::deep_copy(h_po, ep);
    Real min_p = 1.0e300;
    bool supra_ok = true;
    for (int i = is; i <= ie; ++i) {
      min_p = std::fmin(min_p, h_po(0, 0, 0, 0, i));
      if ((i % 2) != 0 && std::fabs(h_po(0, 0, 0, 0, i) - 5.0) > 1.0e-14) {
        supra_ok = false;
      }
    }
    test.CheckTrue(min_p >= efloor - 1.0e-15,
                   "PostSuperstepProject floors active cells to efloor (#194)");
    test.CheckTrue(supra_ok,
                   "PostSuperstepProject leaves supra-floor cells unchanged (#194)");
    test.CheckNear(op.injected_energy(), expect_inj, 0.0,
                   1.0e-10*(std::fabs(expect_inj) + 1.0e-30),
                   "PostSuperstepProject injected-energy = sum deficit*vol (#194)");
  }

  // ===== (G) CompositeParabolicOperator forwards PostSuperstepProject ==================
  // The coupled (strang-split) path advances each FLD operator INSIDE a composite, so
  // OperatorSplitStep calls PostSuperstepProject on the COMPOSITE.  Unless the composite
  // forwards it to its sub-operators, the positivity projection silently never runs in
  // the coupled run (the t~33ns NaN #194 targets).  RED if the composite inherits the
  // base no-op; GREEN once it forwards.
  {
    parabolic::CompositeParabolicOperator comp;
    comp.AddOperator(&op);
    DvceArray5D<Real> eg("erad_comp", nmb, 1, n3, n2, n1);
    auto h_eg = Kokkos::create_mirror_view(eg);
    for (int m = 0; m < nmb; ++m) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            h_eg(m, 0, k, j, i) = -3.0;  // all sub-floor: must be raised by the forward
          }
        }
      }
    }
    Kokkos::deep_copy(eg, h_eg);
    comp.PostSuperstepProject(eg);
    auto h_go = Kokkos::create_mirror_view(eg);
    Kokkos::deep_copy(h_go, eg);
    Real min_g = 1.0e300;
    for (int i = is; i <= ie; ++i) {
      min_g = std::fmin(min_g, h_go(0, 0, 0, 0, i));
    }
    test.CheckTrue(min_g >= efloor - 1.0e-15,
                   "CompositeParabolicOperator forwards PostSuperstepProject (#194)");
  }

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
