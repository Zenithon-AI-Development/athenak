#ifndef DRIVER_PARABOLIC_INTEGRATOR_HPP_
#define DRIVER_PARABOLIC_INTEGRATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file parabolic_integrator.hpp
//  \brief RKL2 super-time-stepping integrator behind a pluggable ParabolicIntegrator
//  interface (issue [4a]/#9, ADR-0001).
//
// The MagLIF/Z-pinch target advances stiff parabolic operators (multigroup FLD radiation,
// electron/ion conduction, resistive B) by OPERATOR SPLITTING in the driver's
// once-per-step `before/after_timeintegrator` tasklist slots, rather than collapsing the
// hyperbolic timestep to the diffusive limit `dt ~ dx^2/D`.  ADR-0001 chooses an explicit
// second-order Runge-Kutta-Legendre (RKL2) super-time-stepping scheme (Meyer, Balsara &
// Aslam 2014) as the default backend: it needs no linear solver, is GPU- and AMR-local,
// and covers a superstep `dt` with `s` substages stable for `dt/dt_exp` up to
// `(s^2+s-2)/4` while staying 2nd-order accurate in `dt`.  The integrator is selected at
// runtime by `<time> parabolic_integrator = sts | implicit` (default `sts`); the
// `implicit` (matrix-free Krylov/multigrid) backend is deferred behind the same interface
// so it can drop in later without rewriting any physics kernel.
//
// This module is PHYSICS-AGNOSTIC.  It owns only the RKL2 recursion and its adaptive
// stage-count; the operator action `M(.)` (the parabolic flux divergence) is supplied by
// the caller as a functor, exactly the shape a ParabolicOperator (issue #13) provides.
// The RKL2 recursion, with `Y_0 = U^n`:
//
//   Y_1 = Y_0 + mu_tilde_1 dt M(Y_0)
//   Y_j = mu_j Y_{j-1} + nu_j Y_{j-2} + (1 - mu_j - nu_j) Y_0
//          + mu_tilde_j dt M(Y_{j-1}) + gamma_tilde_j dt M(Y_0),   j = 2..s
//   U^{n+1} = Y_s
//
// where the stage coefficients are derived from the shifted Chebyshev recursion (see the
// b_j / w_1 helpers below).  All coefficient helpers are KOKKOS_INLINE_FUNCTION so the
// integrator and any device recursion share a single definition.

#include <cstdlib>     // exit, EXIT_FAILURE
#include <iostream>
#include <string>

#include "athena.hpp"            // Real, DvceArray1D, par_for, DevExeSpace, Kokkos
#include "parameter_input.hpp"   // ParameterInput
#include "driver/parabolic_operator.hpp"  // ParabolicOperator (operator-split task body)

namespace parabolic {

//----------------------------------------------------------------------------------------
//! \enum ParabolicMethod
//! \brief Selector for the operator-split parabolic integrator backend (ADR-0001).
//! `sts` = RKL2 super-time-stepping (shipped, the default).  `none` = no parabolic
//! integration (the integrator is inert until a ParabolicOperator is registered, #13).
//! The `implicit` Krylov/multigrid backend is deferred and FATALs if requested.
enum class ParabolicMethod {none, sts};

//----------------------------------------------------------------------------------------
//! \fn RKL2_b
//! \brief Shifted-Chebyshev b_j coefficients for RKL2 (Meyer, Balsara & Aslam 2014).
//! b_0 = b_1 = b_2 = 1/3;  b_j = (j^2 + j - 2)/(2 j (j+1)) for j >= 2.
KOKKOS_INLINE_FUNCTION
Real RKL2_b(int j) {
  if (j < 2) { return 1.0/3.0; }
  Real jr = static_cast<Real>(j);
  return (jr*jr + jr - 2.0)/(2.0*jr*(jr + 1.0));
}

//----------------------------------------------------------------------------------------
//! \fn RKL2_w1
//! \brief RKL2 superstep-scaling factor w_1 = 4/(s^2 + s - 2) for a degree-s scheme.
//! Requires s >= 2 (the denominator vanishes at s = 1).
KOKKOS_INLINE_FUNCTION
Real RKL2_w1(int s) {
  Real sr = static_cast<Real>(s);
  return 4.0/(sr*sr + sr - 2.0);
}

//----------------------------------------------------------------------------------------
//! \fn RKL2_MuTilde1
//! \brief Stage-1 coefficient: Y_1 = Y_0 + mu_tilde_1 dt M(Y_0), mu_tilde_1 = b_1 w_1.
KOKKOS_INLINE_FUNCTION
Real RKL2_MuTilde1(int s) { return RKL2_b(1)*RKL2_w1(s); }

//----------------------------------------------------------------------------------------
//! \struct RKL2Stage
//! \brief The four per-stage coefficients of the RKL2 recursion for stage j (2<=j<=s).
struct RKL2Stage {
  Real mu;           //!< weight on Y_{j-1}
  Real nu;           //!< weight on Y_{j-2}
  Real mu_tilde;     //!< weight on dt M(Y_{j-1})
  Real gamma_tilde;  //!< weight on dt M(Y_0)
};

//----------------------------------------------------------------------------------------
//! \fn RKL2Coeffs
//! \brief Build the RKL2 stage-j coefficients for a degree-s scheme (Meyer et al. 2014):
//!   mu_j         = (2j-1)/j * b_j/b_{j-1}
//!   nu_j         = -(j-1)/j * b_j/b_{j-2}
//!   mu_tilde_j   = mu_j * w_1
//!   gamma_tilde_j= -(1 - b_{j-1}) * mu_tilde_j      (a_{j-1} = 1 - b_{j-1})
KOKKOS_INLINE_FUNCTION
RKL2Stage RKL2Coeffs(int j, int s) {
  Real w1   = RKL2_w1(s);
  Real bj   = RKL2_b(j);
  Real bjm1 = RKL2_b(j-1);
  Real bjm2 = RKL2_b(j-2);
  Real jr   = static_cast<Real>(j);
  RKL2Stage c;
  c.mu          = (2.0*jr - 1.0)/jr * (bj/bjm1);
  c.nu          = -(jr - 1.0)/jr * (bj/bjm2);
  c.mu_tilde    = c.mu*w1;
  c.gamma_tilde = -(1.0 - bjm1)*c.mu_tilde;
  return c;
}

//----------------------------------------------------------------------------------------
//! \fn RKL2NumStages
//! \brief Smallest RKL2 stage count s >= 2 whose stability interval covers the requested
//! ratio of the parabolic superstep `dt_super` to the explicit stability limit `dt_exp`.
//! RKL2 is stable for dt_super/dt_exp <= (s^2 + s - 2)/4, hence
//!   s = ceil( (sqrt(9 + 16 r) - 1)/2 ),   r = dt_super/dt_exp,
//! floored at 2 (RKL2 needs at least two stages).  This is the adaptive stage-count
//! "derived from the diffusive/superstep dt ratio".
KOKKOS_INLINE_FUNCTION
int RKL2NumStages(Real dt_super, Real dt_exp) {
  Real ratio = (dt_exp > 0.0) ? (dt_super/dt_exp) : 0.0;
  if (ratio < 0.0) { ratio = 0.0; }
  Real sr = 0.5*(Kokkos::sqrt(9.0 + 16.0*ratio) - 1.0);
  int s = static_cast<int>(Kokkos::ceil(sr));
  return (s < 2) ? 2 : s;
}

//----------------------------------------------------------------------------------------
//! \fn ParseMethod
//! \brief Map the `<time> parabolic_integrator` input string to a ParabolicMethod.
//! Uses GetOrAddString (default `sts`) so it is order-independent of the Driver's own
//! parse.  FATALs on the deferred `implicit` backend and on any unrecognized value.
inline ParabolicMethod ParseMethod(ParameterInput *pin) {
  std::string sel = pin->GetOrAddString("time", "parabolic_integrator", "sts");
  if (sel == "sts") {
    return ParabolicMethod::sts;
  } else if (sel == "none") {
    return ParabolicMethod::none;
  } else if (sel == "implicit") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<time> parabolic_integrator = 'implicit' is deferred (ADR-0001); the "
              << "matrix-free Krylov/multigrid backend is not yet implemented. Use 'sts'."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<time> parabolic_integrator = '" << sel << "' not implemented. "
            << "Valid choices are [sts, none]." << std::endl;
  std::exit(EXIT_FAILURE);
  return ParabolicMethod::none;  // unreachable; silences -Wreturn-type
}

//----------------------------------------------------------------------------------------
//! \class ParabolicIntegrator
//! \brief Pluggable interface advancing a parabolic operator over an operator-split
//! superstep.  The shipped backend is RKL2 super-time-stepping; the selection is made
//! once from `<time> parabolic_integrator` via SetFromInput().  The operator action is
//! supplied per-superstep as a callable `op(in, out)` that writes M(in) -> out, so the
//! integrator never depends on any particular physics (#13 supplies the flux divergence).

class ParabolicIntegrator {
 public:
  ParabolicIntegrator() : method_(ParabolicMethod::none) {}
  explicit ParabolicIntegrator(ParabolicMethod m) : method_(m) {}

  //! \brief Select the backend from `<time> parabolic_integrator` (wires the selector).
  void SetFromInput(ParameterInput *pin) { method_ = ParseMethod(pin); }

  ParabolicMethod method() const { return method_; }

  //! \brief Adaptive RKL2 stage count for a superstep dt_super vs explicit limit dt_exp.
  int NumStages(Real dt_super, Real dt_exp) const {
    return RKL2NumStages(dt_super, dt_exp);
  }

  //--------------------------------------------------------------------------------------
  //! \fn SuperstepStages
  //! \brief Advance the scalar field `u` over the superstep `dt_super` with exactly `s`
  //! RKL2 substages (s is clamped to >= 2).  `op(in,out)` evaluates the operator action
  //! M(in) -> out (a DvceArray1D -> DvceArray1D map; the caller launches its own kernel).
  //! Register management is local: production callers (#13) own their registers, but the
  //! recursion form is identical.
  template <class OpFunc>
  void SuperstepStages(DvceArray1D<Real> u, Real dt_super, int s, OpFunc op) const {
    if (s < 2) { s = 2; }
    const int n = u.extent_int(0);
    const Real dt = dt_super;

    // RKL2 work registers (Y_0, M(Y_0), Y_{j-1}, Y_{j-2}, M(Y_{j-1}), Y_j).
    DvceArray1D<Real> y0("rkl2_y0", n), my0("rkl2_my0", n);
    DvceArray1D<Real> ym1("rkl2_ym1", n), ym2("rkl2_ym2", n);
    DvceArray1D<Real> mym1("rkl2_mym1", n), ynew("rkl2_ynew", n);

    Kokkos::deep_copy(y0, u);
    op(y0, my0);                                   // M(Y_0), reused every stage

    // Stage 1: Y_1 = Y_0 + mu_tilde_1 dt M(Y_0).
    const Real mt1 = RKL2_MuTilde1(s);
    par_for("rkl2_stage1", DevExeSpace(), 0, n-1, KOKKOS_LAMBDA(const int i) {
      ym1(i) = y0(i) + mt1*dt*my0(i);
    });
    Kokkos::deep_copy(ym2, y0);                    // Y_{j-2} for j=2 is Y_0

    // Stages 2..s.
    for (int j = 2; j <= s; ++j) {
      op(ym1, mym1);                               // M(Y_{j-1})
      const RKL2Stage c = RKL2Coeffs(j, s);
      const Real mu = c.mu, nu = c.nu, mut = c.mu_tilde, gt = c.gamma_tilde;
      const Real one_m = 1.0 - mu - nu;
      par_for("rkl2_stagej", DevExeSpace(), 0, n-1, KOKKOS_LAMBDA(const int i) {
        ynew(i) = mu*ym1(i) + nu*ym2(i) + one_m*y0(i) + mut*dt*mym1(i) + gt*dt*my0(i);
      });
      Kokkos::deep_copy(ym2, ym1);
      Kokkos::deep_copy(ym1, ynew);
    }
    Kokkos::deep_copy(u, ym1);                     // U^{n+1} = Y_s
  }

  //--------------------------------------------------------------------------------------
  //! \fn Superstep
  //! \brief Advance `u` over `dt_super`, choosing the RKL2 stage count adaptively
  //! from the explicit parabolic stability limit `dt_exp`.
  template <class OpFunc>
  void Superstep(DvceArray1D<Real> u, Real dt_super, Real dt_exp, OpFunc op) const {
    SuperstepStages(u, dt_super, NumStages(dt_super, dt_exp), op);
  }

  //--------------------------------------------------------------------------------------
  //! \fn SuperstepStages (5D overload)
  //! \brief Advance the conserved field `u` (DvceArray5D, indexed (m,var,k,j,i)) over the
  //! superstep `dt_super` with exactly `s` RKL2 substages (clamped to >= 2).  This is the
  //! production shape a ParabolicOperator (#13) drives: `op(in,out)` writes M(in) -> out
  //! over the SAME 5D layout.  The recursion is identical to the 1D version; it runs over
  //! every (m,var,k,j,i), so variables the operator does not evolve (M = 0 there) stay
  //! frozen at Y_0 by construction (mu + nu + (1 - mu - nu) = 1), realizing the
  //! operator-split "frozen background".  Ghost zones are refreshed by `op` itself (the
  //! ParabolicOperator::ApplyBoundary contract), so any stale ghost values from the
  //! recursion are overwritten before the next M(.) evaluation.
  template <class OpFunc>
  void SuperstepStages(DvceArray5D<Real> u, Real dt_super, int s, OpFunc op) const {
    if (s < 2) { s = 2; }
    const int nmb  = u.extent_int(0);
    const int nvar = u.extent_int(1);
    const int n3   = u.extent_int(2);
    const int n2   = u.extent_int(3);
    const int n1   = u.extent_int(4);
    const Real dt = dt_super;

    // RKL2 work registers (Y_0, M(Y_0), Y_{j-1}, Y_{j-2}, M(Y_{j-1}), Y_j).
    DvceArray5D<Real> y0("rkl2_y0", nmb, nvar, n3, n2, n1);
    DvceArray5D<Real> my0("rkl2_my0", nmb, nvar, n3, n2, n1);
    DvceArray5D<Real> ym1("rkl2_ym1", nmb, nvar, n3, n2, n1);
    DvceArray5D<Real> ym2("rkl2_ym2", nmb, nvar, n3, n2, n1);
    DvceArray5D<Real> mym1("rkl2_mym1", nmb, nvar, n3, n2, n1);
    DvceArray5D<Real> ynew("rkl2_ynew", nmb, nvar, n3, n2, n1);

    Kokkos::deep_copy(y0, u);
    op(y0, my0);                                   // M(Y_0), reused every stage

    // Stage 1: Y_1 = Y_0 + mu_tilde_1 dt M(Y_0).
    const Real mt1 = RKL2_MuTilde1(s);
    par_for("rkl2_5d_stage1", DevExeSpace(),
    0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      ym1(m,n,k,j,i) = y0(m,n,k,j,i) + mt1*dt*my0(m,n,k,j,i);
    });
    Kokkos::deep_copy(ym2, y0);                    // Y_{j-2} for j=2 is Y_0

    // Stages 2..s.
    for (int jstg = 2; jstg <= s; ++jstg) {
      op(ym1, mym1);                               // M(Y_{j-1})
      const RKL2Stage c = RKL2Coeffs(jstg, s);
      const Real mu = c.mu, nu = c.nu, mut = c.mu_tilde, gt = c.gamma_tilde;
      const Real one_m = 1.0 - mu - nu;
      par_for("rkl2_5d_stagej", DevExeSpace(),
      0, nmb-1, 0, nvar-1, 0, n3-1, 0, n2-1, 0, n1-1,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        ynew(m,n,k,j,i) = mu*ym1(m,n,k,j,i) + nu*ym2(m,n,k,j,i) + one_m*y0(m,n,k,j,i)
                          + mut*dt*mym1(m,n,k,j,i) + gt*dt*my0(m,n,k,j,i);
      });
      Kokkos::deep_copy(ym2, ym1);
      Kokkos::deep_copy(ym1, ynew);
    }
    Kokkos::deep_copy(u, ym1);                     // U^{n+1} = Y_s
  }

  //--------------------------------------------------------------------------------------
  //! \fn Superstep (5D overload)
  //! \brief Advance the conserved field `u` over `dt_super`, picking the RKL2 stage count
  //! adaptively from the explicit parabolic stability limit `dt_exp`.
  template <class OpFunc>
  void Superstep(DvceArray5D<Real> u, Real dt_super, Real dt_exp, OpFunc op) const {
    SuperstepStages(u, dt_super, NumStages(dt_super, dt_exp), op);
  }

 private:
  ParabolicMethod method_;
};

//----------------------------------------------------------------------------------------
//! \fn OperatorSplitStep
//! \brief Advance one ParabolicOperator's `field` over the operator-split step `dt` with
//! a single adaptive RKL2 superstep.  This is the BODY of the operator-split task
//! (#13): the operator's ghost zones are refreshed (ApplyBoundary) up front and again
//! before every RKL2 stage evaluation, so each M(.) sees a boundary-consistent stencil.
//! The production task (Hydro::OperatorSplitConduction) and the unit test call this same
//! function, so the wired path is exactly what the test exercises.
inline void OperatorSplitStep(const ParabolicIntegrator &integ, ParabolicOperator &op,
                              DvceArray5D<Real> field, Real dt) {
  op.ApplyBoundary(field);
  const Real dt_exp = op.ExplicitStableDt();
  auto opfunc = [&op](DvceArray5D<Real> in, DvceArray5D<Real> out) {
    op.ApplyBoundary(in);
    op.OperatorAction(in, out);
  };
  integ.Superstep(field, dt, dt_exp, opfunc);
}

}  // namespace parabolic

#endif  // DRIVER_PARABOLIC_INTEGRATOR_HPP_
