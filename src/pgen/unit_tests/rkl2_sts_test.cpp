//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rkl2_sts_test.cpp
//  \brief Unit test: the RKL2 super-time-stepping integrator
//  (driver/parabolic_integrator.hpp) advances a scalar diffusion problem with the correct
//  2nd-order temporal accuracy and is stable far beyond the explicit parabolic limit
//  (issue [4a]/#9, ADR-0001).
//
//  PHYSICS-AGNOSTIC oracle. The integrator's only contract is to advance U over a
//  superstep given the operator action M(.). We exercise it with the 1D periodic heat
//  equation  u_t = D u_xx,  discretized by the central difference
//  M(u)_i = D (u_{i-1} - 2 u_i + u_{i+1})/dx^2 on N periodic points. The Fourier mode
//  sin(k x_i), k = 2 pi m / L, is an exact eigenvector of this discrete operator with
//  eigenvalue  lambda = -(2 D/dx^2)(1 - cos(k dx)),  so the semi-discrete ODE has the
//  EXACT solution u_i(t) = sin(k x_i) exp(lambda t). This isolates the temporal error:
//  comparing the numerical result to exp(lambda t) measures the integrator alone, with no
//  spatial-discretization contamination.
//
//  The four batteries:
//   (A) Coefficient consistency: one RKL2 superstep applied to the scalar ODE
//       u' = lambda u returns the amplification R_s(z), z = lambda*dt. RKL2 is 2nd order,
//       so R_s(z) = 1 + z + z^2/2 + O(z^3); for s = 2 this is EXACT. The residual scales
//       as z^3 (order >= 3 => method >= 2nd order). A 1st-order bug (wrong z^2 weight) is
//       rejected.
//   (B) Temporal convergence: advance the eigenmode a fixed time T at FIXED stage count
//       while refining the superstep dt; the L-infinity error vs exp(lambda T) drops at
//       2nd order.
//   (C) Adaptive stage count: NumStages(dt_super, dt_exp) returns the minimal s>=2 whose
//       stability interval (s^2+s-2)/4 covers the requested dt ratio.
//   (D) Stability at large dt ratio: a Gaussian (broadband, includes the stiff Nyquist
//       mode) advanced with dt_super = 256 dt_exp (s ~ 32 substages) stays bounded -- the
//       discrete L2 norm is non-increasing, the mean (sum) is conserved, and the field
//       decays. Forward Euler would blow up catastrophically here; RKL2 does not.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/rkl2_sts_test -B build_unit
//    (cd build_unit/src && make) && ./build_unit/src/athena \
//        -i inputs/unit_tests/rkl2_sts_test.athinput
//  Auto-run by tst/test_suite/unit_tests/test_unit_rkl2_sts_cpu.py.

#include <cmath>     // std::sin, std::cos, std::exp, std::log2, std::acos, std::fabs
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief RKL2 super-time-stepping integrator unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("rkl2_sts_test");
  // default backend; the RKL2 math below is method-agnostic
  const parabolic::ParabolicIntegrator sts;

  // ===== (E) selector wiring: <time> parabolic_integrator maps to the enum =====
  {
    parabolic::ParabolicIntegrator sel;
    sel.SetFromInput(pin);
    test.CheckTrue(sel.method() == parabolic::ParabolicMethod::sts,
                   "<time> parabolic_integrator selector parsed as sts");
  }

  // ===== (A) RKL2 amplification consistency on the scalar ODE u' = lambda u =====
  // scalar_op evaluates M(u) = -u (lambda = -1); one superstep of size dt = |z| then
  // produces R_s(z) with z = lambda*dt = -|z|.
  auto scalar_op = [](DvceArray1D<Real> in, DvceArray1D<Real> out) {
    par_for("rkl2_scalar_op", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int i) {
      out(i) = -in(i);
    });
  };
  DvceArray1D<Real> u1("u1", 1);
  auto h_u1 = Kokkos::create_mirror_view(u1);

  // Amplification factor R_s(z) for z = -zmag (one superstep, exactly s stages).
  auto amp = [&](int s, Real zmag) -> Real {
    h_u1(0) = 1.0;
    Kokkos::deep_copy(u1, h_u1);
    sts.SuperstepStages(u1, zmag, s, scalar_op);
    Kokkos::deep_copy(h_u1, u1);
    return h_u1(0);
  };

  const Real zmag = 0.02;                              // |z|; z = -0.02
  const Real z = -zmag;
  const Real second_order = 1.0 + z + 0.5*z*z;         // exp(z) to 2nd order
  // s = 2 reproduces 1 + z + z^2/2 *exactly* (the minimal RKL2 polynomial is degree 2).
  test.CheckNear(amp(2, zmag), second_order, 0.0, 1.0e-12,
                 "R_2(z) == 1 + z + z^2/2 exactly");
  // Higher s: residual is O(z^3), comfortably below 1e-4; a 1st-order weight
  // (~z^2/2 ~ 2e-4) would fail this bound, so it pins the 2nd-order term.
  test.CheckTrue(std::fabs(amp(3, zmag)  - second_order) < 1.0e-4,
                 "R_3(z) consistent with 2nd order");
  test.CheckTrue(std::fabs(amp(5, zmag)  - second_order) < 1.0e-4,
                 "R_5(z) consistent with 2nd order");
  test.CheckTrue(std::fabs(amp(10, zmag) - second_order) < 1.0e-4,
                 "R_10(z) consistent with 2nd order");

  // Residual order: halving z must shrink the residual by ~2^3 (the error is O(z^3) <=>
  // the scheme is 2nd-order accurate). Use log2 so the assertion is constant-free.
  for (int s : {4, 6}) {
    Real r_full = std::fabs(amp(s, 0.04) - (1.0 - 0.04 + 0.5*0.04*0.04));
    Real r_half = std::fabs(amp(s, 0.02) - (1.0 - 0.02 + 0.5*0.02*0.02));
    Real order = std::log2(r_full/r_half);
    test.CheckTrue(order > 2.5 && order < 4.5,
                   "R_s(z) - (1+z+z^2/2) is O(z^3) (2nd-order accurate)");
  }

  // ===== shared scalar-diffusion grid (used by B and D) =====
  const int  N  = 64;
  const Real PI = std::acos(-1.0);
  const Real L  = 2.0*PI;
  const Real dx = L/static_cast<Real>(N);
  const Real dx2 = dx*dx;
  const Real D  = 0.1;
  const Real dt_exp = dx2/(2.0*D);          // forward-Euler parabolic limit dx^2/(2D)

  // M(u)_i = D (u_{i-1} - 2 u_i + u_{i+1})/dx^2, periodic.
  auto diffusion_op = [=](DvceArray1D<Real> in, DvceArray1D<Real> out) {
    const Real Dl = D, dx2l = dx2;
    const int Nl = N;
    par_for("rkl2_diff_op", DevExeSpace(), 0, Nl-1, KOKKOS_LAMBDA(const int i) {
      int im = (i - 1 + Nl) % Nl;
      int ip = (i + 1) % Nl;
      out(i) = Dl*(in(im) - 2.0*in(i) + in(ip))/dx2l;
    });
  };

  DvceArray1D<Real> u("rkl2_u", N);
  auto h_u = Kokkos::create_mirror_view(u);

  // ===== (B) 2nd-order temporal convergence on the eigenmode =====
  const int  m_mode = 2;
  const Real kmode  = 2.0*PI*static_cast<Real>(m_mode)/L;   // = m_mode (L = 2 pi)
  const Real lambda = -(2.0*D/dx2)*(1.0 - std::cos(kmode*dx));
  const Real Tfinal = 2.0;
  const int  s_fixed = 8;                  // stable for the coarsest dt below (ratio ~10)
  const int  nsuper_list[3] = {4, 8, 16};
  Real errB[3];
  for (int r = 0; r < 3; ++r) {
    const int nsuper = nsuper_list[r];
    const Real tau = Tfinal/static_cast<Real>(nsuper);
    for (int i = 0; i < N; ++i) { h_u(i) = std::sin(kmode*i*dx); }
    Kokkos::deep_copy(u, h_u);
    for (int n = 0; n < nsuper; ++n) {
      sts.SuperstepStages(u, tau, s_fixed, diffusion_op);
    }
    Kokkos::deep_copy(h_u, u);
    const Real decay = std::exp(lambda*Tfinal);
    Real linf = 0.0;
    for (int i = 0; i < N; ++i) {
      linf = std::fmax(linf, std::fabs(h_u(i) - std::sin(kmode*i*dx)*decay));
    }
    errB[r] = linf;
  }
  test.CheckTrue(errB[0] > errB[1] && errB[1] > errB[2],
                 "RKL2 temporal error decreases monotonically under dt refinement");
  const Real orderB1 = std::log2(errB[0]/errB[1]);
  const Real orderB2 = std::log2(errB[1]/errB[2]);
  test.CheckTrue(orderB1 > 1.8 && orderB1 < 2.2,
                 "RKL2 temporal convergence order ~2 (coarse->mid)");
  test.CheckTrue(orderB2 > 1.8 && orderB2 < 2.2,
                 "RKL2 temporal convergence order ~2 (mid->fine)");

  // ===== (C) adaptive stage count: minimal s>=2 covering the dt ratio =====
  // Small ratios floor at s = 2.
  test.CheckTrue(sts.NumStages(0.1, 1.0) == 2, "tiny dt ratio -> s == 2 (floor)");
  test.CheckTrue(sts.NumStages(1.0, 1.0) == 2, "ratio 1 -> s == 2");
  for (Real ratio : {5.0, 20.0, 100.0, 500.0}) {
    const int s = sts.NumStages(ratio, 1.0);
    const Real cover    = (s*s + s - 2)/4.0;
    const Real cover_m1 = ((s-1)*(s-1) + (s-1) - 2)/4.0;
    test.CheckTrue(s >= 2, "adaptive stage count s >= 2");
    test.CheckTrue(cover >= ratio, "stage count covers the dt ratio");
    test.CheckTrue(cover_m1 < ratio, "stage count is minimal (s-1 would not cover)");
  }

  // ===== (D) stability at a large dt ratio (Gaussian, broadband) =====
  const Real ratio_big = 256.0;
  const Real tau_big = ratio_big*dt_exp;
  const int  s_big = sts.NumStages(tau_big, dt_exp);
  test.CheckTrue(s_big > 20, "large dt ratio -> many RKL2 substages");

  const Real xc = 0.5*L, sigma = L/16.0;
  for (int i = 0; i < N; ++i) {
    const Real dxc = i*dx - xc;
    h_u(i) = std::exp(-0.5*dxc*dxc/(sigma*sigma));
  }
  Kokkos::deep_copy(u, h_u);
  // Initial diagnostics.
  auto l2_of = [&](void) -> Real {
    Kokkos::deep_copy(h_u, u);
    Real s2 = 0.0;
    for (int i = 0; i < N; ++i) { s2 += h_u(i)*h_u(i); }
    return std::sqrt(s2);
  };
  auto sum_of = [&](void) -> Real {
    Kokkos::deep_copy(h_u, u);
    Real s = 0.0;
    for (int i = 0; i < N; ++i) { s += h_u(i); }
    return s;
  };
  const Real l2_init = l2_of();
  const Real sum_init = sum_of();
  bool nonincreasing = true, finite = true;
  Real l2_prev = l2_init;
  const int nsuper_big = 6;
  for (int n = 0; n < nsuper_big; ++n) {
    sts.Superstep(u, tau_big, dt_exp, diffusion_op);   // adaptive s internally
    const Real l2 = l2_of();
    if (!(l2 < 1.0e30)) { finite = false; }            // catches NaN/Inf (NaN fails <)
    if (l2 > l2_prev*(1.0 + 1.0e-9)) { nonincreasing = false; }
    l2_prev = l2;
  }
  const Real l2_final = l2_prev;
  const Real sum_final = sum_of();
  test.CheckTrue(finite, "RKL2 stays finite at dt = 256 dt_exp (no blow-up)");
  test.CheckTrue(nonincreasing,
                 "discrete L2 norm is non-increasing every superstep (stable)");
  test.CheckTrue(l2_final < 0.99*l2_init, "broadband field actually diffuses (decays)");
  test.CheckNear(sum_final, sum_init, 1.0e-10, 0.0,
                 "RKL2 conserves the mean (sum) of a periodic diffusion field");

  test.Finish();
  return;
}
