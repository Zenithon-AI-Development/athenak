#ifndef RADIATION_FLD_MATTER_RADIATION_COUPLING_HPP_
#define RADIATION_FLD_MATTER_RADIATION_COUPLING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file matter_radiation_coupling.hpp
//! \brief Per-cell point-implicit (backward-Euler) grey matter-radiation emission/
//! absorption exchange (issue [8b]/#23, ADR-0001/0002, "local coupling").
//
// This is the LOCAL, point-stiff partner of the grey FLD spatial diffusion operator
// (FLDGreyOperator, #17): the diffusion operator transports the radiation energy in
// space;
// this module exchanges energy between the radiation field and the matter, per cell, with
// no spatial dependence.  ADR-0001 prescribes a per-cell point-implicit solve for the
// stiff local exchange (vs. the STS spatial path), reusing AthenaK's existing implicit
// radiation-fluid coupling pattern (Radiation::RadFluidCoupling / radiation_source.cpp).
//
// In the 1T stage the grey radiation energy density E_r couples to a SINGLE matter
// internal energy density e_mat with temperature T (the 2T re-targeting to T_e is #30):
//
//     dE_r/dt  = c chi_a (a T^4 - E_r)      (emission a T^4, absorption E_r)
//     de_mat/dt= c chi_a (E_r - a T^4) = -dE_r/dt,
//
// where chi_a [1/length] is the Planck-mean absorption coefficient (= kappa_P rho), c is
// the code-unit speed of light, a is the radiation constant, and the matter caloric
// relation is e_mat = c_v T (c_v the volumetric heat capacity dE/dT).  The pair conserves
// E_r + e_mat exactly and relaxes to radiative equilibrium a T^4 = E_r.
//
// BACKWARD-EULER REDUCTION.  Evaluating the emission/absorption implicitly at the new
// state and eliminating E_r^{n+1} via the radiation equation, the matter temperature
// update collapses to a single quartic in T^{n+1} (beta = dt c chi_a):
//
//     E_r^{n+1} = (E_r^n + beta a T^4)/(1+beta),
//     c_v T - e_mat^n = -beta (a T^4 - E_r^n)/(1+beta)
//   =>  coef4 T^4 + T + tconst = 0,
//       coef4  =  beta a / ((1+beta) c_v)   (>= 0),
//       tconst = -e_mat^n/c_v - beta E_r^n/((1+beta) c_v)   (<= 0).
//
// This is EXACTLY the quartic AthenaK's GR radiation source solves for the new gas
// temperature (radiation_source.cpp: coef[1] T^4 + T + coef[0] = 0, coef[1] ~ dt c
// sigma_a a (gm1/rho), coef[0] = -T_gas - absorbed), so we reuse the same closed-form
// quartic root FourthPolyRoot.  Because coef4 >= 0 and tconst <= 0 the quartic has a
// unique positive root; backward Euler makes it unconditionally stable, so a stiff cell
// (beta >> 1) lands at equilibrium (a T^4 + c_v T = E_r^n + e_mat^n) in a single step.

#include "athena.hpp"

namespace radiationfld {

//----------------------------------------------------------------------------------------
//! \fn bool FourthPolyRoot
//! \brief Exact positive real root of the quartic coef4 x^4 + x + tconst = 0; returns
//! false (root unchanged) if no admissible positive root is found.  This is the same
//! solver AthenaK's GR radiation-fluid coupling uses for the implicit gas-temperature
//! update (radiation::FourthPolyRoot in radiation_source.cpp); duplicated in the
//! radiationfld namespace so the FLD coupling stays self-contained (no dependence on the
//! GR M1 radiation module).  Valid regime: coef4 >= 0, tconst <= 0 (one positive root).
KOKKOS_INLINE_FUNCTION
bool FourthPolyRoot(const Real coef4, const Real tconst, Real &root) {
  // Calculate real root of z^3 - 4*tconst/coef4 * z - 1/coef4^2 = 0
  Real ccubic = tconst * tconst * tconst;
  Real delta1 = 0.25 - 64.0 * ccubic * coef4 / 27.0;
  if (delta1 < 0.0) {
    return false;
  }
  delta1 = Kokkos::sqrt(delta1);
  if (delta1 < 0.5) {
    return false;
  }
  Real zroot;
  if (delta1 > 1.0e11) {  // to avoid small number cancellation
    zroot = Kokkos::pow(delta1, -2.0/3.0) / 3.0;
  } else {
    zroot = Kokkos::pow(0.5 + delta1, 1.0/3.0) - Kokkos::pow(-0.5 + delta1, 1.0/3.0);
  }
  if (zroot < 0.0) {
    return false;
  }
  zroot *= Kokkos::pow(coef4, -2.0/3.0);

  // Calculate quartic root using cubic root
  Real rcoef = Kokkos::sqrt(zroot);
  Real delta2 = -zroot + 2.0 / (coef4 * rcoef);
  if (delta2 < 0.0) {
    return false;
  }
  delta2 = Kokkos::sqrt(delta2);
  root = 0.5 * (delta2 - rcoef);
  if (root < 0.0) {
    return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool PointImplicitGreyCoupling
//! \brief One per-cell backward-Euler step of the grey matter-radiation emission/
//! absorption exchange.  Given the old radiation energy density `e_rad_old`, matter
//! internal energy density `e_mat_old`, volumetric heat capacity `cv` (e_mat = cv T),
//! absorption coefficient `chi_a` [1/length], light speed `c_light`, radiation constant
//! `a_rad`, and timestep `dt`, writes the implicitly-updated `e_rad_new`/`e_mat_new`.
//!
//! Conserves E_r + e_mat to machine precision (the matter update is the exact negative of
//! the radiation exchange) and relaxes to a T^4 = E_r.  Returns true on success; on a
//! failed root solve (or beta <= 0) it leaves the state unchanged and returns false (no
//! exchange) -- mirroring the "badcell" fallback in radiation_source.cpp.  All math is
//! KOKKOS_INLINE_FUNCTION so it runs on host and device and is shared with the unit test.
KOKKOS_INLINE_FUNCTION
bool PointImplicitGreyCoupling(Real e_rad_old, Real e_mat_old, Real cv,
    Real chi_a, Real c_light, Real a_rad, Real dt,
    Real &e_rad_new, Real &e_mat_new) {
  Real beta = dt*c_light*chi_a;          // dimensionless absorption depth c kappa dt
  if (!(beta > 0.0) || !(cv > 0.0)) {    // no coupling (or ill-posed): leave state alone
    e_rad_new = e_rad_old;
    e_mat_new = e_mat_old;
    return (beta == 0.0);
  }
  Real inv_cv = 1.0/cv;
  Real opb = 1.0 + beta;
  // backward-Euler quartic for the new matter temperature: coef4 T^4 + T + tconst = 0.
  Real coef4 = beta*a_rad*inv_cv/opb;
  Real tconst = -(e_mat_old + beta*e_rad_old/opb)*inv_cv;
  Real t_new;
  if (coef4 > 1.0e-20) {
    bool ok = FourthPolyRoot(coef4, tconst, t_new);
    if (!ok || !Kokkos::isfinite(t_new)) {  // badcell: skip the exchange this step
      e_rad_new = e_rad_old;
      e_mat_new = e_mat_old;
      return false;
    }
  } else {
    t_new = -tconst;                     // negligible coupling: T ~ unchanged
  }
  // Net energy moved from matter into radiation over the step (= dE_r = -de_mat).  Apply
  // it once, equal-and-opposite, so E_r + e_mat is conserved regardless of any round-off
  // in the quartic root.
  Real t4 = t_new*t_new*t_new*t_new;
  Real exchange = beta*(a_rad*t4 - e_rad_old)/opb;
  e_rad_new = e_rad_old + exchange;
  e_mat_new = e_mat_old - exchange;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool PointImplicitElectronRadiationCoupling
//! \brief 2T re-targeting of the point-implicit matter-radiation coupling to the ELECTRON
//! temperature T_e (issue [14c]/#30, ADR-0002).  In the 2T stage radiation exchanges
//! energy with the electrons -- absorption/emission heat the electrons, not a single
//! matter fluid -- so the "matter" reservoir is the electron internal energy density
//! `e_ele` with the electron volumetric heat capacity `cv_ele` (T_e = e_ele/cv_ele).  The
//! backward-Euler quartic is identical to the grey 1T form; only the energy slot and heat
//! capacity change, so this forwards to PointImplicitGreyCoupling with the electron
//! quantities.  The ions are untouched by this exchange (they couple to the electrons
//! separately, e.g. via collisional equilibration -- out of scope here).  Conserves
//! `E_r + e_ele` to machine precision and relaxes to `a T_e^4 = E_r`.
KOKKOS_INLINE_FUNCTION
bool PointImplicitElectronRadiationCoupling(Real e_rad_old, Real e_ele_old, Real cv_ele,
    Real chi_a, Real c_light, Real a_rad, Real dt,
    Real &e_rad_new, Real &e_ele_new) {
  return PointImplicitGreyCoupling(e_rad_old, e_ele_old, cv_ele, chi_a, c_light, a_rad,
                                   dt, e_rad_new, e_ele_new);
}

}  // namespace radiationfld

#endif  // RADIATION_FLD_MATTER_RADIATION_COUPLING_HPP_
