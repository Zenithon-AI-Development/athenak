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
#include "radiation_fld/fld_grey_operator.hpp"
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

  // ===== (E2) per-group monotone-front streaming runaway (issue #215 <- #194) ==========
  // Battery (E) seeds a CHECKERBOARD, whose Nyquist mode a central difference sees as
  // ZERO gradient -- marginal, and reachable-green by the positivity floor alone.  It
  // does NOT exercise what NaNs a real coupled run: a SUSTAINED MONOTONE STEEP FRONT.  In
  // the streaming limit each group's Larsen-limited flux F_g = -D_g dE_g/dx collapses to
  // F_g = -c*eface*sign(dE_g) -- the CENTERED discretization of advection, whose spatial
  // eigenvalues are PURE IMAGINARY.  RKL2 is a parabolic STS scheme whose stability
  // region
  // hugs the negative-real axis, so it AMPLIFIES every imaginary mode at any stage count
  // and a ~1-cell front grows E_g,max every super-step.  The floor (#197) is structurally
  // blind to this HIGH-side runaway -- exactly the finding of #194 on the grey path.
  //
  // The grey operator took the fix (a Lax-Friedrichs streaming-dissipation term
  // -0.5*upwind*alf*(er-el) with signal speed alf = c*lambda*R gated by (1-3 lambda)_+);
  // the multigroup operator never did.  RED here on the unstabilized centered per-group
  // flux (E_g,max -> NaN); GREEN once the per-group LF term lands.
  {
    const Real chi_fr  = pin->GetOrAddReal("problem", "chi_front_mg", 1.0);   // streaming
    const Real esrc_fr = pin->GetOrAddReal("problem", "esrc_front_mg", 1.0);  // hot wall
    const Real e_hi    = pin->GetOrAddReal("problem", "e_hi_mg", 1.0);        // bright
    const Real e_lo    = pin->GetOrAddReal("problem", "e_lo_mg", 1.0e-6);     // cold
    const Real dtf = pin->GetOrAddReal("problem", "dt_fac_front_mg", 5.0e1);  // dt/dt_exp
    const int nstp =
        static_cast<int>(pin->GetOrAddReal("problem", "nstep_front_mg", 2.0e2));

    // free-streaming table: every group at the same small extinction chi_fr, so all NG
    // groups sit in the streaming band together (a group-collapsing bug still fails (B)).
    opacity::MultigroupOpacity tab_fr;
    tab_fr.Allocate(NG, 2, 2, 1.0, 1.0e3, 1.0e-4, 1.0e0);
    {
      const Real kr = chi_fr/rho_bg;
      auto h_kr = Kokkos::create_mirror_view(tab_fr.rosseland);
      for (int g = 0; g < NG; ++g) {
        for (int it = 0; it < tab_fr.ntemp; ++it) {
          for (int id = 0; id < tab_fr.ndens; ++id) {
            h_kr(g, it, id) = kr;
          }
        }
      }
      Kokkos::deep_copy(tab_fr.rosseland, h_kr);
    }

    DvceArray5D<Real> ef("erad_front_mg", nmb, NG, n3, n2, n1);
    auto h_ef = Kokkos::create_mirror_view(ef);
    const int imid = is + N/2;
    for (int m = 0; m < nmb; ++m) {
      for (int g = 0; g < NG; ++g) {
        for (int k = 0; k < n3; ++k) {
          for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
              // sharp monotone front: hot (e_hi) inner half, cold (e_lo) outer half; the
              // hot inner-x1 Dirichlet wall (esrc_fr>0) sustains it through the drive.
              h_ef(m, g, k, j, i) = (i < imid) ? e_hi : e_lo;
            }
          }
        }
      }
    }
    Kokkos::deep_copy(ef, h_ef);
    FLDMultigroupOperator opf(pmbp, pin, ef, tab_fr, c_light, rho_bg, te_bg, nl,
                              esrc_fr, efloor);
    parabolic::ParabolicIntegrator integ2(parabolic::ParabolicMethod::sts);
    auto efv = ef;
    const Real e_lo_c = e_lo;
    const int ng1t = NG - 1;
    for (int step = 0; step < nstp; ++step) {
      const Real dtx = opf.ExplicitStableDt();
      parabolic::OperatorSplitStep(integ2, opf, ef, dtf*dtx);
      // RE-SHARPEN every step (as the hydro continually re-steepens the liner front): a
      // static front just diffuses to the wall value and never exposes the instability.
      par_for("e2mg_resharpen", DevExeSpace(), 0, nmb-1, 0, ng1t, 0, n3-1, 0, n2-1,
              imid, ie,
      KOKKOS_LAMBDA(const int m, const int g, const int k, const int j, const int i) {
        efv(m, g, k, j, i) = e_lo_c;
      });
    }
    auto h_of = Kokkos::create_mirror_view(ef);
    Kokkos::deep_copy(h_of, ef);
    bool finite_f = true;
    Real max_e = -1.0e300, min_e2 = 1.0e300;
    for (int g = 0; g < NG; ++g) {
      for (int i = is; i <= ie; ++i) {
        const Real e = h_of(0, g, 0, 0, i);
        if (!std::isfinite(e)) { finite_f = false; }
        max_e = std::fmax(max_e, e);
        min_e2 = std::fmin(min_e2, e);
      }
    }
    std::cout << "[fld_multigroup_operator_test] (E2) per-group monotone front: max(E_g)="
              << max_e << " min(E_g)=" << min_e2 << " finite=" << finite_f
              << " (nstep=" << nstp << ", dt_fac=" << dtf << ", chi=" << chi_fr << ")"
              << std::endl;
    test.CheckTrue(finite_f,
                   "per-group monotone-front super-step stays finite (no NaN) (#215)");
    test.CheckTrue(max_e <= 1.0e1*e_hi,
                   "per-group monotone-front super-step keeps E_g,max BOUNDED (#215)");
    test.CheckTrue(min_e2 >= -1.0e-12,
                   "per-group monotone-front super-step keeps E_g >= 0 (#215)");
  }

  // ===== (D2) ExplicitStableDt is FACE-CONSISTENT with the flux kernel (#215/#194) ====
  // The dt the operator reports must bound the scheme the flux kernel actually RUNS.  The
  // cell-centred estimate builds R from a CENTRAL difference over e0, so a DARK cell
  // beside a BRIGHT neighbour reports R = |grad|/(chi*e0) -> huge, hence lambda ~ 1/R,
  // D ~ 0, and
  // an effectively INFINITE dt for that cell.  The face-consistent bound instead builds
  // D_eff (flux-limited D + the LF streaming viscosity) per face from the SAME
  // el/er/eface
  // the flux kernel uses, where eface = (el+er)/2 stays bright and D_eff stays finite.
  //
  // A single front does not expose the gap: the flat interior cells report the thick
  // limit and the min-reduction is rescued by them.  A PERIOD-4 SQUARE WAVE does -- every
  // cell then has a nonzero central difference, so none reports the thick limit, and dark
  // cells' near-zero D lifts the reported dt above the bound the flux kernel needs.
  //
  // Oracle: the GREY operator, which already carries the face-consistent bound (#194).
  // On a SINGLE-group table with chi_g == the grey chi, over the same field, the two
  // operators
  // discretize the identical problem and MUST report the identical dt.  RED while the
  // multigroup bound is cell-centred (it reports ~5x the grey value on this field).
  {
    const Real chi_d2 = pin->GetOrAddReal("problem", "chi_dt_mg", 1.0);
    const Real e_hi   = pin->GetOrAddReal("problem", "e_hi_mg", 1.0);
    const Real e_lo   = pin->GetOrAddReal("problem", "e_lo_mg", 1.0e-6);
    opacity::MultigroupOpacity tab_d2;
    tab_d2.Allocate(1, 2, 2, 1.0, 1.0e3, 1.0e-4, 1.0e0);   // SINGLE group == grey
    {
      const Real kr = chi_d2/rho_bg;
      auto h_kr = Kokkos::create_mirror_view(tab_d2.rosseland);
      for (int it = 0; it < tab_d2.ntemp; ++it) {
        for (int id = 0; id < tab_d2.ndens; ++id) { h_kr(0, it, id) = kr; }
      }
      Kokkos::deep_copy(tab_d2.rosseland, h_kr);
    }
    // period-4 square wave [hi, hi, lo, lo], seeded identically into both operators.
    auto fill_square = [&](DvceArray5D<Real> &a) {
      auto h_a = Kokkos::create_mirror_view(a);
      const int nv = a.extent_int(1);
      for (int m = 0; m < nmb; ++m) {
        for (int v = 0; v < nv; ++v) {
          for (int k = 0; k < n3; ++k) {
            for (int j = 0; j < n2; ++j) {
              for (int i = 0; i < n1; ++i) {
                h_a(m, v, k, j, i) = (((i - is) & 2) == 0) ? e_hi : e_lo;
              }
            }
          }
        }
      }
      Kokkos::deep_copy(a, h_a);
    };
    DvceArray5D<Real> ed("erad_dt_mg", nmb, 1, n3, n2, n1);
    DvceArray5D<Real> eg2("erad_dt_grey", nmb, 1, n3, n2, n1);
    fill_square(ed);
    fill_square(eg2);
    FLDMultigroupOperator opd(pmbp, pin, ed, tab_d2, c_light, rho_bg, te_bg, nl,
                              -1.0, efloor);
    FLDGreyOperator opg(pmbp, pin, eg2, c_light, chi_d2, nl, -1.0, efloor);
    const Real dt_mg = opd.ExplicitStableDt();
    const Real dt_grey = opg.ExplicitStableDt();
    std::cout << "[fld_multigroup_operator_test] (D2) dt_multigroup=" << dt_mg
              << " dt_grey=" << dt_grey << " ratio=" << dt_mg/dt_grey << std::endl;
    test.CheckNear(dt_mg, dt_grey, 1.0e-12, 0.0,
                   "ExplicitStableDt matches the grey face-consistent bound (#215)");
  }

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
