//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parabolic_conduction_test.cpp
//  \brief Unit test: AthenaK's isotropic thermal conduction, re-expressed as a
//  parabolic::ParabolicOperator (ConductionOperator) and advanced operator-split by the
//  RKL2 super-time-stepping integrator, matches the explicit-conduction result and the
//  analytic diffusion decay (issue [4b]/#13, ADR-0001).
//
//  This exercises the WHOLE operator-split parabolic path end-to-end on real device
//  kernels: the ParabolicOperator interface, the 5D ParabolicIntegrator superstep, the
//  ConductionOperator (the existing isotropic heat-flux divergence), and the
//  parabolic::OperatorSplitStep tasklist-task body (the same call the production
//  Hydro::OperatorSplitConduction task makes).
//
//  ORACLE.  A static (v = 0), uniform-density (rho = 1) gas carries a temperature profile
//  T = (gamma-1) E.  Isotropic conduction then diffuses the total energy with diffusivity
//  D = kappa (gamma-1).  On the insulated box realized by ConductionOperator::ApplyBound-
//  ary (zero heat flux through the domain boundary), the cosine mode
//      E_i = E0 + A cos(theta (i+1/2)),   theta = m pi / N
//  is an EXACT eigenvector of the discrete conduction operator with eigenvalue
//      lambda = -(2 D / dx^2)(1 - cos theta),
//  so the semi-discrete solution is E_i(t) = E0 + A cos(theta(i+1/2)) exp(lambda t).  It
//  isolates the temporal integration error, exactly as the RKL2 (#9) test does, but now
//  through the real conduction operator.
//
//  Batteries:
//   (A) Operator action: one M(u) on the eigenmode IC returns lambda*(E_i - E0) to
//       machine precision -- pins the conduction flux divergence AND coeff D=k(g-1).
//   (B) STS vs analytic: an eigenmode advanced a fixed time by STS matches exp(lambda T).
//   (C) STS vs EXPLICIT conduction: STS matches forward-Euler sub-cycling of the SAME
//       operator (the existing explicit-conduction result) -- the literal acceptance.
//   (D) 2nd-order temporal convergence of STS vs analytic under superstep refinement.
//   (E) Conservation: the insulated operator conserves total energy to machine precision.
//   (F) STS stability beyond the explicit limit (dt_super = 256 dt_exp, many substages).
//   (G) Selector: <time> parabolic_integrator parses to sts.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/parabolic_conduction_test -B build_unit
//    (cd build_unit/src && make) then run ./athena on the athinput below.
//  Auto-run by tst/test_suite/unit_tests/test_unit_parabolic_conduction_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::exp, std::log2, std::fabs, std::acos
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/conduction.hpp"
#include "diffusion/conduction_operator.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Operator-split conduction (ConductionOperator + RKL2 STS) unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("parabolic_conduction_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // require the conduction-enabled hydro block this test drives
  test.CheckTrue(pmbp->phydro != nullptr, "hydro block constructed");
  test.CheckTrue(pmbp->phydro->pcond != nullptr,
                 "conduction enabled (<hydro> conductivity set)");
  if (pmbp->phydro == nullptr || pmbp->phydro->pcond == nullptr) {
    test.Finish();
    return;
  }
  auto *phydro = pmbp->phydro;

  // (G) selector wiring: <time> parabolic_integrator parses to the sts enum.
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- problem constants ----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int N = indcs.nx1;
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real kappa = phydro->pcond->kappa;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real dx = (x1max - x1min)/static_cast<Real>(N);

  const Real rho0 = 1.0;
  const Real E0 = 1.0;                       // background energy (so T0 = gm1 E0 > 0)
  const Real amp = 1.0e-2;                   // eigenmode amplitude (small => positive E)
  const int m_mode = 4;                      // moderate cosine mode
  const Real PI = std::acos(-1.0);
  const Real theta = m_mode*PI/static_cast<Real>(N);
  const Real D = kappa*gm1/rho0;             // energy diffusivity
  const Real lambda = -(2.0*D/(dx*dx))*(1.0 - std::cos(theta));   // discrete eigenvalue

  // ---- conserved-field work arrays (shaped like hydro u0) ----
  const int nmb  = phydro->u0.extent_int(0);
  const int nvar = phydro->u0.extent_int(1);
  const int n3   = phydro->u0.extent_int(2);
  const int n2   = phydro->u0.extent_int(3);
  const int n1   = phydro->u0.extent_int(4);
  DvceArray5D<Real> u_ic("u_ic", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> u_work("u_work", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> rhs("rhs", nmb, nvar, n3, n2, n1);
  auto h_ic = Kokkos::create_mirror_view(u_ic);
  auto h_w  = Kokkos::create_mirror_view(u_work);

  // analytic eigenmode IC (full array incl. ghosts; all states valid for later C2P)
  auto cosmode = [&](int i) -> Real {
    return std::cos(theta*(static_cast<Real>(i - is) + 0.5));
  };
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_ic(m, IDN, k, j, i) = rho0;
          h_ic(m, IM1, k, j, i) = 0.0;
          h_ic(m, IM2, k, j, i) = 0.0;
          h_ic(m, IM3, k, j, i) = 0.0;
          h_ic(m, IEN, k, j, i) = E0 + amp*cosmode(i);
        }
      }
    }
  }
  Kokkos::deep_copy(u_ic, h_ic);
  // also seed the live hydro state with this valid static medium, so the trivial driver
  // run after UserProblem (ConsToPrim over the mesh, nlim=0) does not touch uninitialized
  // memory.  The test itself works only on the local copies above.
  Kokkos::deep_copy(phydro->u0, u_ic);

  // the operator under test (diffuses u_work; density read from u_ic)
  ConductionOperator op(pmbp, u_ic, kappa, gamma);
  const Real dt_exp = op.ExplicitStableDt();
  test.CheckTrue(dt_exp > 0.0, "explicit conduction dt is positive");

  // ===== (A) operator action == discrete eigenvalue * perturbation (machine prec.) =====
  Kokkos::deep_copy(u_work, u_ic);
  op.ApplyBoundary(u_work);
  op.OperatorAction(u_work, rhs);
  auto h_rhs = Kokkos::create_mirror_view(rhs);
  Kokkos::deep_copy(h_rhs, rhs);
  Real maxerrA = 0.0;
  for (int i = is; i <= ie; ++i) {
    Real expected = lambda*(amp*cosmode(i));    // lambda*(E_i - E0)
    maxerrA = std::fmax(maxerrA, std::fabs(h_rhs(0, IEN, 0, 0, i) - expected));
  }
  test.CheckNear(maxerrA, 0.0, 0.0, 1.0e-10*std::fabs(lambda)*amp + 1.0e-14,
                 "M(u) == lambda*(E-E0): operator IS isotropic conduction (D=k(g-1))");
  // the non-evolved components carry M = 0 (frozen background)
  Real maxrho_rhs = 0.0;
  for (int i = is; i <= ie; ++i) {
    maxrho_rhs = std::fmax(maxrho_rhs, std::fabs(h_rhs(0, IDN, 0, 0, i)));
    maxrho_rhs = std::fmax(maxrho_rhs, std::fabs(h_rhs(0, IM1, 0, 0, i)));
  }
  test.CheckNear(maxrho_rhs, 0.0, 0.0, 1.0e-30,
                 "M(u) writes 0 into non-energy components (frozen background)");

  // ===== shared: total interior energy of a conserved field (host) =====
  auto total_energy = [&](DvceArray5D<Real> u) -> Real {
    auto hu = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(hu, u);
    Real s = 0.0;
    for (int i = is; i <= ie; ++i) { s += hu(0, IEN, 0, 0, i); }
    return s;
  };
  const Real e_init = total_energy(u_ic);

  // ===== (B,C,E) advance a fixed time by STS and by explicit conduction; compare =====
  const Real Tfinal = 0.7/std::fabs(lambda);     // mode decays to ~exp(-0.7) ~ 0.5
  const Real decay = std::exp(lambda*Tfinal);

  // -- STS: nsuper adaptive supersteps via the production task body OperatorSplitStep --
  const int nsuper = 4;
  const Real dt_super = Tfinal/static_cast<Real>(nsuper);
  Kokkos::deep_copy(u_work, u_ic);
  for (int n = 0; n < nsuper; ++n) {
    parabolic::OperatorSplitStep(integ, op, u_work, dt_super);
  }
  Kokkos::deep_copy(h_w, u_work);
  const Real e_sts = total_energy(u_work);
  // STS uses many substages (super-time-stepping, not plain RK)
  test.CheckTrue(integ.NumStages(dt_super, dt_exp) > 2,
                 "STS superstep uses >2 RKL2 substages (dt_super >> dt_exp)");

  // -- explicit conduction: forward-Euler sub-cycling of the SAME operator --
  const Real dt_fe = 0.25*dt_exp;                // accurate explicit reference
  const int nfe = static_cast<int>(std::ceil(Tfinal/dt_fe));
  const Real dt_fe_use = Tfinal/static_cast<Real>(nfe);
  DvceArray5D<Real> u_ref("u_ref", nmb, nvar, n3, n2, n1);
  Kokkos::deep_copy(u_ref, u_ic);
  for (int n = 0; n < nfe; ++n) {
    op.ApplyBoundary(u_ref);
    op.OperatorAction(u_ref, rhs);
    const Real dt_fe_l = dt_fe_use;
    par_for("cond_fe_step", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int v, const int k, const int j, const int i) {
      u_ref(m, v, k, j, i) += dt_fe_l*rhs(m, v, k, j, i);
    });
  }
  auto h_ref = Kokkos::create_mirror_view(u_ref);
  Kokkos::deep_copy(h_ref, u_ref);

  // (B) STS vs analytic decay; (C) STS vs explicit conduction
  Real maxerr_sts_an = 0.0, maxerr_sts_fe = 0.0, maxerr_fe_an = 0.0;
  for (int i = is; i <= ie; ++i) {
    Real analytic = E0 + amp*cosmode(i)*decay;
    maxerr_sts_an = std::fmax(maxerr_sts_an, std::fabs(h_w(0, IEN, 0, 0, i) - analytic));
    maxerr_fe_an  = std::fmax(maxerr_fe_an, std::fabs(h_ref(0, IEN, 0, 0, i) - analytic));
    maxerr_sts_fe = std::fmax(maxerr_sts_fe,
                              std::fabs(h_w(0, IEN, 0, 0, i) - h_ref(0, IEN, 0, 0, i)));
  }
  test.CheckTrue(maxerr_sts_an < 1.0e-2*amp,
                 "STS conduction matches analytic exp(lambda T) decay");
  test.CheckTrue(maxerr_fe_an < 1.0e-2*amp,
                 "explicit (forward-Euler) conduction matches analytic decay");
  test.CheckTrue(maxerr_sts_fe < 2.0e-2*amp,
                 "STS conduction matches the explicit-conduction result (acceptance)");

  // diffusion actually happened (perturbation decayed toward the mean)
  Real peak_ic = amp, peak_sts = 0.0;
  for (int i = is; i <= ie; ++i) {
    peak_sts = std::fmax(peak_sts, std::fabs(h_w(0, IEN, 0, 0, i) - E0));
  }
  test.CheckTrue(peak_sts < 0.9*peak_ic, "STS conduction diffuses (perturbation decays)");

  // (E) the insulated operator conserves total energy to machine precision
  test.CheckNear(e_sts, e_init, 0.0, 1.0e-12*std::fabs(e_init),
                 "STS conduction conserves total energy (insulated box)");
  test.CheckNear(total_energy(u_ref), e_init, 0.0, 1.0e-10*std::fabs(e_init),
                 "explicit conduction conserves total energy (insulated box)");

  // ===== (D) 2nd-order temporal convergence of STS vs analytic =====
  Real errD[3];
  const int ns_list[3] = {2, 4, 8};
  for (int r = 0; r < 3; ++r) {
    const int ns = ns_list[r];
    const Real tau = Tfinal/static_cast<Real>(ns);
    Kokkos::deep_copy(u_work, u_ic);
    for (int n = 0; n < ns; ++n) {
      parabolic::OperatorSplitStep(integ, op, u_work, tau);
    }
    Kokkos::deep_copy(h_w, u_work);
    Real linf = 0.0;
    for (int i = is; i <= ie; ++i) {
      Real analytic = E0 + amp*cosmode(i)*decay;
      linf = std::fmax(linf, std::fabs(h_w(0, IEN, 0, 0, i) - analytic));
    }
    errD[r] = linf;
  }
  test.CheckTrue(errD[0] > errD[1] && errD[1] > errD[2],
                 "STS temporal error decreases monotonically under superstep refinement");
  const Real order1 = std::log2(errD[0]/errD[1]);
  const Real order2 = std::log2(errD[1]/errD[2]);
  test.CheckTrue(order1 > 1.6 && order1 < 2.6,
                 "STS temporal convergence order ~2 (coarse->mid)");
  test.CheckTrue(order2 > 1.6 && order2 < 2.6,
                 "STS temporal convergence order ~2 (mid->fine)");

  // ===== (F) stability far beyond the explicit limit =====
  const Real ratio_big = 256.0;
  const Real tau_big = ratio_big*dt_exp;
  const int s_big = integ.NumStages(tau_big, dt_exp);
  test.CheckTrue(s_big > 20, "large dt ratio -> many RKL2 substages");
  Kokkos::deep_copy(u_work, u_ic);
  parabolic::OperatorSplitStep(integ, op, u_work, tau_big);
  Kokkos::deep_copy(h_w, u_work);
  bool finite = true;
  Real peak_big = 0.0;
  for (int i = is; i <= ie; ++i) {
    Real e = h_w(0, IEN, 0, 0, i);
    if (!(std::fabs(e) < 1.0e30)) { finite = false; }   // catches NaN/Inf
    peak_big = std::fmax(peak_big, std::fabs(e - E0));
  }
  test.CheckTrue(finite, "STS stays finite at dt = 256 dt_exp (no blow-up)");
  test.CheckTrue(peak_big < peak_ic,
                 "STS decays the mode at dt = 256 dt_exp (forward Euler would diverge)");
  test.CheckNear(total_energy(u_work), e_init, 0.0, 1.0e-12*std::fabs(e_init),
                 "STS conserves total energy even at dt = 256 dt_exp");

  test.Finish();
  // All checks live in UserProblem on local arrays; this is a pure test harness with no
  // meaningful time integration (nlim = tlim = 0).  Exit cleanly on success rather than
  // proceed into the trivial driver run, which would needlessly drive the full hydro
  // setup/boundary machinery on the static seed state.
  std::exit(EXIT_SUCCESS);
  return;
}
