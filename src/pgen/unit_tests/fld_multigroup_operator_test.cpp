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
#include "driver/parabolic_integrator.hpp"
#include "driver/composite_parabolic_operator.hpp"
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
  const Real efloor = pin->GetOrAddReal("problem", "mgfld_efloor", 1.0e-10);
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
  FLDMultigroupOperator op(pmbp, pin, erad, table, c_light, rho_bg, te_bg, nl, -1.0,
                           efloor);

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

  // ===== (E) free-streaming super-step positivity, PER GROUP (issue #197) =====
  // Per-group mirror of grey battery (E) (#194).  In the optically-thin / free-streaming
  // regime (small chi_g + steep gradient) the Larsen limiter makes each group's FLD
  // operator HYPERBOLIC (advective at speed c).  RKL2's parabolic stability polynomial
  // damps only the negative-real axis, so the c-advective (off-axis) spectrum is UNDAMPED
  // a high-wavenumber (checkerboard) mode in a sustained free-streaming gap is amplified
  // each super-step, overshoots E_g NEGATIVE, and with no positivity floor the 1e-30
  // divide-guard then makes R_g = |grad E_g|/(chi_g E_g) and the flux blow up to NaN
  // (#194; seen in the coupled run at t~33 ns).  Reproduce that accumulation per group: a
  // HOT Dirichlet wall (esrc>0) driving radiation into a checkerboard-seeded near-vacuum
  // gap for EVERY group, advanced many real RKL2 super-steps (the wired
  // parabolic::OperatorSplitStep path).  Require EVERY group's E_g to stay FINITE and
  // >= 0 throughout.  RED on the unfloored operator (a thin group overshoots negative),
  // GREEN once the per-group positivity floor + projection land.  Knobs are deck-tunable.
  {
    const Real chi_thin = pin->GetOrAddReal("problem", "chi_thin_mg", 1.0);   // free-strm
    const Real esrc_thin = pin->GetOrAddReal("problem", "esrc_thin_mg", 1.0); // hot wall
    const Real e_bg = pin->GetOrAddReal("problem", "e_bg_mg", 1.0e-6);        // vacuum bg
    const Real seed = pin->GetOrAddReal("problem", "seed_amp_mg", 1.0e-7);    // checkrbd
    const Real dt_fac = pin->GetOrAddReal("problem", "dt_fac_mg", 5.0e1);     // dt/dt_exp
    const int nstep = static_cast<int>(pin->GetOrAddReal("problem", "nstep_mg", 1.0e2));

    // a free-streaming table (allocated fresh, not the shared fixture View) whose every
    // group has the same small extinction chi_thin so each group sits in the transition
    // band: chi_g = rho_bg*kappa_R,g => kappa_R,g = chi_thin/rho_bg (constant in rho,T).
    opacity::MultigroupOpacity tab_thin;
    tab_thin.Allocate(NG, 2, 2, 1.0, 1.0e3, 1.0e-4, 1.0e0);
    {
      const Real kr = chi_thin/rho_bg;
      auto h_kr = Kokkos::create_mirror_view(tab_thin.rosseland);
      for (int g = 0; g < NG; ++g) {
        for (int it = 0; it < tab_thin.ntemp; ++it) {
          for (int id = 0; id < tab_thin.ndens; ++id) {
            h_kr(g, it, id) = kr;
          }
        }
      }
      Kokkos::deep_copy(tab_thin.rosseland, h_kr);
    }

    DvceArray5D<Real> et("erad_thin", nmb, NG, n3, n2, n1);
    auto h_et = Kokkos::create_mirror_view(et);
    for (int m = 0; m < nmb; ++m) {
      for (int g = 0; g < NG; ++g) {
        for (int k = 0; k < n3; ++k) {
          for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
              // near-vacuum gap with a tiny checkerboard seed (the most-amplified mode).
              h_et(m, g, k, j, i) = e_bg + seed*(((i % 2) == 0) ? 1.0 : -1.0);
            }
          }
        }
      }
    }
    Kokkos::deep_copy(et, h_et);
    FLDMultigroupOperator opt(pmbp, pin, et, tab_thin, c_light, rho_bg, te_bg, nl,
                              esrc_thin, efloor);  // hot wall
    parabolic::ParabolicIntegrator integ(parabolic::ParabolicMethod::sts);
    for (int step = 0; step < nstep; ++step) {
      const Real dt_exp = opt.ExplicitStableDt();
      parabolic::OperatorSplitStep(integ, opt, et, dt_fac*dt_exp);
    }
    auto h_out = Kokkos::create_mirror_view(et);
    Kokkos::deep_copy(h_out, et);
    bool finite = true;
    Real min_e = 1.0e300;
    for (int g = 0; g < NG; ++g) {
      for (int i = is; i <= ie; ++i) {
        const Real e = h_out(0, g, 0, 0, i);
        if (!std::isfinite(e)) { finite = false; }
        min_e = std::fmin(min_e, e);
      }
    }
    std::cout << "[fld_multigroup_operator_test] (E) per-group free-streaming drive: "
              << "min(erad)=" << min_e << " finite=" << finite << " (nstep=" << nstep
              << ", dt_fac=" << dt_fac << ", chi=" << chi_thin << ")" << std::endl;
    test.CheckTrue(finite,
                   "per-group free-streaming super-step stays finite (no NaN) (#197)");
    test.CheckTrue(min_e >= -1.0e-12,
                   "per-group free-streaming super-step keeps every E_g >= 0 (#197)");
  }

  // ===== (F) PostSuperstepProject floors negatives per group + accounts injection ======
  // Per-group mirror of grey battery (F).  Battery E reaches green via the read-floor
  // alone (min(E_g) stays positive), so the projection + its energy accounting would
  // otherwise be untested.  Seed EVERY group with sub-floor and supra-floor cells; the
  // projection must raise every active cell of every group to >= efloor, leave
  // supra-floor cells untouched, and report injected energy = sum over ALL groups and
  // floored cells of (efloor - E_g)*cell_volume.  (op was built with this deck's efloor.)
  {
    DvceArray5D<Real> ep("erad_proj", nmb, NG, n3, n2, n1);
    auto h_ep = Kokkos::create_mirror_view(ep);
    Real expect_inj = 0.0;
    const Real dvol = dx;  // 1D Cartesian unit-test cell volume (dx2=dx3=1)
    for (int m = 0; m < nmb; ++m) {
      for (int g = 0; g < NG; ++g) {
        for (int k = 0; k < n3; ++k) {
          for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
              // distinct sub/supra-floor per group so a group-collapsing bug is caught.
              Real e = ((i % 2) == 0) ? -(2.0 + g) : (5.0 + g);
              h_ep(m, g, k, j, i) = e;
              if (i >= is && i <= ie && (efloor - e) > 0.0) {
                expect_inj += (efloor - e)*dvol;
              }
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
    for (int g = 0; g < NG; ++g) {
      for (int i = is; i <= ie; ++i) {
        min_p = std::fmin(min_p, h_po(0, g, 0, 0, i));
        if ((i % 2) != 0 && std::fabs(h_po(0, g, 0, 0, i) - (5.0 + g)) > 1.0e-14) {
          supra_ok = false;
        }
      }
    }
    test.CheckTrue(min_p >= efloor - 1.0e-15,
                   "PostSuperstepProject floors active cells per group to efloor (#197)");
    test.CheckTrue(supra_ok,
                   "PostSuperstepProject leaves supra-floor cells unchanged (#197)");
    test.CheckNear(op.injected_energy(), expect_inj, 0.0,
                   1.0e-10*(std::fabs(expect_inj) + 1.0e-30),
                   "PostSuperstepProject injected = sum deficit*vol over groups (#197)");
  }

  // ===== (G) CompositeParabolicOperator forwards PostSuperstepProject (per group) ======
  // Per-group mirror of grey battery (G).  The coupled (strang-split) path advances each
  // FLD operator INSIDE a composite, so OperatorSplitStep calls PostSuperstepProject on
  // the COMPOSITE.  Unless the composite forwards it to its sub-operators, the per-group
  // positivity projection silently never runs in the coupled run (the t~33ns NaN #194
  // targets).  The composite forward already lands on main (5b4c457e); GREEN here once
  // the multigroup override exists (RED if the multigroup sub-op keeps the base no-op).
  {
    parabolic::CompositeParabolicOperator comp;
    comp.AddOperator(&op);
    DvceArray5D<Real> eg("erad_comp", nmb, NG, n3, n2, n1);
    auto h_eg = Kokkos::create_mirror_view(eg);
    for (int m = 0; m < nmb; ++m) {
      for (int g = 0; g < NG; ++g) {
        for (int k = 0; k < n3; ++k) {
          for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
              h_eg(m, g, k, j, i) = -3.0;  // all sub-floor: must be raised by the forward
            }
          }
        }
      }
    }
    Kokkos::deep_copy(eg, h_eg);
    comp.PostSuperstepProject(eg);
    auto h_go = Kokkos::create_mirror_view(eg);
    Kokkos::deep_copy(h_go, eg);
    Real min_g = 1.0e300;
    for (int g = 0; g < NG; ++g) {
      for (int i = is; i <= ie; ++i) {
        min_g = std::fmin(min_g, h_go(0, g, 0, 0, i));
      }
    }
    test.CheckTrue(min_g >= efloor - 1.0e-15,
                   "CompositeParabolicOperator forwards PostSuperstepProject (#197)");
  }

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
