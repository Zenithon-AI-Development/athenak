#ifndef RADIATION_FLD_MULTIGROUP_COUPLING_HPP_
#define RADIATION_FLD_MULTIGROUP_COUPLING_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file multigroup_coupling.hpp
//! \brief Per-cell point-implicit (backward-Euler) MULTIGROUP matter-radiation
//! emission/absorption coupling solved as an ARROWHEAD system over the photon-energy
//! groups + the electron temperature, plus the electron-ion exchange (issue [17b]/#35,
//! ADR-0001/0002).
//
// This is the multigroup generalisation of the grey per-cell point-implicit coupling
// (radiationfld::PointImplicitGreyCoupling, #23, and its electron re-target #30).  Where
// the grey module exchanged energy between one radiation energy density and one matter
// reservoir, here the radiation is resolved into the N photon-energy GROUPS of the FLD
// model (CONTEXT.md "Group"), each with its own extinction chi_g, all coupling
// to a SHARED electron temperature T_e -- and the electrons additionally exchange energy
// with the ions.  Per cell, backward Euler (emission/absorption + e-i evaluated at
// the new state):
//
//     dE_g/dt   = c chi_g (B_g(T_e) - E_g)            (g = 0..N-1; emission - absorption)
//     de_ele/dt = -sum_g dE_g/dt + c_ei (T_i - T_e)   (radiation absorbed + e-i exchange)
//     de_ion/dt =                 - c_ei (T_i - T_e)
//
// with e_ele = cv_ele T_e, e_ion = cv_ion T_i, and the group-integrated Planck function
//
//     B_g(T_e) = a T_e^4 [ F(x_{g+1}) - F(x_g) ],     x_g = eps_g / (k_B T_e),
//
// where F(x) is the fractional Planck (blackbody) function (PlanckFraction below) and
// eps_g are the group photon-energy boundaries.  sum_g B_g = a T_e^4 when the groups span
// [0,inf), so the radiation relaxes to the PLANCK SPECTRUM at the electron temperature
// (each group equilibrates to its own slice a T_e^4 [F(x_{g+1})-F(x_g)]).  In LTE the
// emission and absorption opacities coincide (Kirchhoff), so a single per-group chi_g is
// used for both -- exactly as the grey module uses one chi_a for emission (a T^4) and
// absorption (E_r).
//
// ARROWHEAD STRUCTURE (ADR-0001 "groups couple only through electron temperature ->
// O(N_group) Schur elimination + Newton").  The backward-Euler Jacobian over the unknowns
// (E_0..E_{N-1}, T_e) is an arrowhead matrix: each E_g couples only to itself and to the
// shared node T_e (the spine), no group-group coupling.  Each group equation is linear
// in E_g given T_e, so E_g is eliminated ANALYTICALLY (the Schur step, O(N)):
//
//     E_g(T_e) = (E_g^n + beta_g B_g(T_e)) / (1 + beta_g),     beta_g = dt c chi_g,
//
// and substituted into the electron balance.  The electron-ion exchange is linear in
// (T_e,T_i), so T_i is eliminated analytically too, leaving a SCALAR equation in
// T_e (nonlinear through the Planck term B_g(T_e)) solved by a safeguarded Newton
// (rtsafe); each residual/derivative evaluation is O(N) (one sweep over groups).  This
// is precisely the "O(N_group) Schur elimination wrapped in Newton" of ADR-0001.
//
// CONSERVATION.  After Newton finds T_e the new group energies E_g(T_e) are formed,
// and the electron/ion energies are set from the EXACT balances (electron gets minus
// the radiation gain plus the ion exchange; ion gets minus the exchange).  Hence
// sum_g E_g + e_ele + e_ion is conserved to machine precision regardless of any Newton
// residual -- the same equal-and-opposite trick the grey module uses.
//
// REDUCES TO GREY.  For a single group (N=1) spanning the whole spectrum (F -> 1) with no
// electron-ion exchange (c_ei = 0), B_0(T_e) = a T_e^4 and the scalar equation is exactly
// the grey backward-Euler quartic, so the result matches PointImplicitGreyCoupling.

#include "athena.hpp"

namespace radiationfld {

//----------------------------------------------------------------------------------------
//! \fn Real PlanckFraction
//! \brief Fractional Planck (blackbody) function
//!   F(x) = (15/pi^4) \int_0^x t^3/(e^t - 1) dt,
//! the fraction of blackbody power radiated below dimensionless photon energy
//! x = eps/(k_B T).  F(0) = 0, F(+inf) = 1, monotone increasing.  Computed by a small-x
//! integral power series for x < 1 and the exact exponential (large-x) series for x >= 1;
//! both converge to machine precision in the overlap, so F is continuous.  All math is
//! KOKKOS_INLINE_FUNCTION (host + device) so the unit-test oracle can call it on host.
KOKKOS_INLINE_FUNCTION
Real PlanckFraction(Real x) {
  constexpr Real c15pi4 = 0.15398973382026614;  // 15/pi^4
  if (x <= 0.0) { return 0.0; }
  if (x < 1.0) {
    // small-x: integral of the Bernoulli series of t^3/(e^t-1)
    //   \int_0^x = x^3/3 - x^4/8 + x^5/60 - x^7/5040 + x^9/272160 - x^11/13305600
    //             + x^13/622702080 - c15 x^15 + c17 x^17 (error ~1e-14 at x=1).
    Real x2 = x*x;
    Real x3 = x2*x;
    Real x4 = x3*x;
    Real x5 = x4*x;
    Real s = x3/3.0 - x4/8.0 + x5/60.0;
    Real x7 = x5*x2; s -= x7/5040.0;
    Real x9 = x7*x2; s += x9/272160.0;
    Real x11 = x9*x2; s -= x11/13305600.0;
    Real x13 = x11*x2; s += x13/622702080.0;
    Real x15 = x13*x2; s -= 3.5228353354650344e-11*x15;
    Real x17 = x15*x2; s += 7.872080910521232e-13*x17;
    return c15pi4*s;
  }
  // large-x: F(x) = 1 - (15/pi^4) \int_x^inf t^3/(e^t-1) dt, with the exact series
  //   \int_x^inf = sum_{n>=1} e^{-nx} (x^3/n + 3x^2/n^2 + 6x/n^3 + 6/n^4).
  Real x2 = x*x;
  Real x3 = x2*x;
  Real sum = 0.0;
  for (int n = 1; n <= 64; ++n) {
    Real rn = 1.0/static_cast<Real>(n);
    Real term = Kokkos::exp(-static_cast<Real>(n)*x)
                *rn*(x3 + rn*(3.0*x2 + rn*(6.0*x + rn*6.0)));
    sum += term;
    if (term < 1.0e-17*(sum + 1.0e-300)) { break; }
  }
  return 1.0 - c15pi4*sum;
}

//----------------------------------------------------------------------------------------
//! \fn Real PlanckFractionXDeriv
//! \brief G(x) = x F'(x) = (15/pi^4) x^4/(e^x - 1).  Used for the temp derivative of
//! the group Planck function via dB_g/dT_e = a T_e^3 [4 dF - (G(x_hi)-G(x_lo))].
//! G(0) = 0, G(+inf) = 0; expm1 keeps it accurate near x=0 and overflow-safe at large x.
KOKKOS_INLINE_FUNCTION
Real PlanckFractionXDeriv(Real x) {
  constexpr Real c15pi4 = 0.15398973382026614;  // 15/pi^4
  if (x <= 0.0) { return 0.0; }
  Real em1 = Kokkos::expm1(x);     // e^x - 1, accurate near 0, -> +inf at large x
  if (!(em1 > 0.0)) { return 0.0; }
  Real x4 = x*x*x*x;
  return c15pi4*x4/em1;            // x4/inf -> 0 at large x (no overflow)
}

namespace internal {
//! \brief Residual F(T_e) and derivative F'(T_e) of the arrowhead electron energy balance
//! (groups + ion analytically eliminated).  F is monotone increasing with F(0) <= 0, so a
//! safeguarded Newton (rtsafe) on [0, T_hi] converges.  ngroups groups with old energies
//! e_rad_old[g], boundaries group_bounds[0..ngroups] (photon energy), per-group chi[g];
//! gamma = dt c_ei, Ti0 = e_ion_old/cv_ion (0 if no ions).
KOKKOS_INLINE_FUNCTION
void ArrowheadResidual(Real te, int ngroups, const Real *e_rad_old,
    const Real *group_bounds, const Real *chi, Real e_ele_old, Real cv_ele, Real cv_ion,
    Real c_light, Real a_rad, Real kboltz, Real gamma, Real dt, Real Ti0,
    Real *fval, Real *dfval) {
  Real beta_c = dt*c_light;
  Real inv_kt = 1.0/(kboltz*te);
  Real te3 = te*te*te;
  Real te4 = te3*te;
  Real sum_de = 0.0;        // sum_g (E_g(te) - E_g^old)
  Real sum_dedt = 0.0;      // sum_g dE_g/dte
  for (int g = 0; g < ngroups; ++g) {
    Real beta_g = beta_c*chi[g];
    Real xlo = group_bounds[g]*inv_kt;
    Real xhi = group_bounds[g+1]*inv_kt;
    Real dfr = PlanckFraction(xhi) - PlanckFraction(xlo);   // group Planck fraction
    Real bg = a_rad*te4*dfr;                                // B_g(te)
    Real eg = (e_rad_old[g] + beta_g*bg)/(1.0 + beta_g);    // Schur-eliminated E_g(te)
    sum_de += eg - e_rad_old[g];
    Real dg = PlanckFractionXDeriv(xhi) - PlanckFractionXDeriv(xlo);
    Real dbg = a_rad*te3*(4.0*dfr - dg);                    // dB_g/dte
    sum_dedt += (beta_g/(1.0 + beta_g))*dbg;
  }
  // electron-ion exchange (electrons gain gamma (T_i - T_e); T_i eliminated):
  //   gamma (T_i - T_e) = gamma cv_ion (Ti0 - te)/(cv_ion + gamma).
  Real ei_gain = 0.0, dei = 0.0;
  if (gamma > 0.0 && (cv_ion + gamma) > 0.0) {
    Real denom = cv_ion + gamma;
    ei_gain = gamma*cv_ion*(Ti0 - te)/denom;
    dei = -gamma*cv_ion/denom;
  }
  *fval = cv_ele*te - e_ele_old + sum_de - ei_gain;
  *dfval = cv_ele + sum_dedt - dei;
}
}  // namespace internal

//----------------------------------------------------------------------------------------
//! \fn bool MultigroupArrowheadCoupling
//! \brief One per-cell backward-Euler step of the multigroup matter-radiation coupling +
//! electron-ion exchange, by O(N_group) Schur elimination over the groups wrapped in
//! a safeguarded Newton iteration for the electron temperature.
//!
//! Inputs: number of groups `ngroups`; old per-group radiation energy densities
//! `e_rad_old[0..ngroups-1]`; ascending photon-energy group boundaries
//! `group_bounds[0..ngroups]`; per-group extinction `chi[g]` [1/length]; old electron/ion
//! internal energy densities `e_ele_old`/`e_ion_old` with (volumetric) heat capacities
//! `cv_ele`/`cv_ion` (T = e/cv); code-unit light speed `c_light`, radiation constant
//! `a_rad`, Boltzmann constant `kboltz` (so x = eps/(kboltz T_e)); electron-ion exchange
//! coefficient `c_ei` (de_ele/dt += c_ei (T_i - T_e); 0 disables and leaves ions
//! untouched); timestep `dt`.  Writes the implicitly-updated `e_rad_new[0..ngroups-1]`,
//! `e_ele_new`, `e_ion_new`.
//!
//! Conserves sum_g E_g + e_ele + e_ion to roundoff; relaxes to the Planck spectrum
//! E_g = a T_e^4 [F(x_{g+1}) - F(x_g)] with electrons (and, if c_ei>0, ions) at the
//! common equilibrium temperature.  Returns true on success; on a bad temperature
//! guess or failed bracket it leaves the state unchanged and returns false.  All math is
//! KOKKOS_INLINE_FUNCTION (host + device), shared with the unit test.
KOKKOS_INLINE_FUNCTION
bool MultigroupArrowheadCoupling(int ngroups, const Real *e_rad_old,
    const Real *group_bounds, const Real *chi, Real e_ele_old, Real e_ion_old,
    Real cv_ele, Real cv_ion, Real c_light, Real a_rad, Real kboltz, Real c_ei, Real dt,
    Real *e_rad_new, Real &e_ele_new, Real &e_ion_new) {
  // Degenerate / ill-posed: no electron heat capacity -> cannot define T_e.  Leave alone.
  if (!(cv_ele > 0.0) || !(dt > 0.0) || ngroups < 1) {
    for (int g = 0; g < ngroups; ++g) { e_rad_new[g] = e_rad_old[g]; }
    e_ele_new = e_ele_old;
    e_ion_new = e_ion_old;
    return false;
  }
  const Real gamma = dt*c_ei;
  const Real Ti0 = (cv_ion > 0.0) ? e_ion_old/cv_ion : 0.0;

  // ---- bracket the (unique, monotone) root of F(T_e) = 0 on [Tlo, Thi] ----
  // F(0) <= 0; grow Thi until F(Thi) > 0 (radiation/ions only remove energy from a hot
  // electron, so a finite Thi exists).
  Real tlo = 0.0;
  Real thi = e_ele_old/cv_ele;                 // T_e if no exchange occurred
  if (gamma > 0.0) { thi = Kokkos::fmax(thi, Ti0); }
  thi = Kokkos::fmax(thi, 1.0e-100);
  Real fhi, dfhi;
  bool bracketed = false;
  for (int it = 0; it < 200; ++it) {
    internal::ArrowheadResidual(thi, ngroups, e_rad_old, group_bounds, chi, e_ele_old,
                                cv_ele, cv_ion, c_light, a_rad, kboltz, gamma, dt, Ti0,
                                &fhi, &dfhi);
    if (fhi > 0.0) { bracketed = true; break; }
    thi *= 2.0;
  }
  if (!bracketed) {                            // could not bracket -> skip the exchange
    for (int g = 0; g < ngroups; ++g) { e_rad_new[g] = e_rad_old[g]; }
    e_ele_new = e_ele_old;
    e_ion_new = e_ion_old;
    return false;
  }

  // ---- safeguarded Newton (rtsafe) for T_e on [tlo, thi] ----
  Real te = 0.5*(tlo + thi);
  Real dxold = thi - tlo;
  Real dx = dxold;
  Real fval, dfval;
  internal::ArrowheadResidual(te, ngroups, e_rad_old, group_bounds, chi, e_ele_old,
                              cv_ele, cv_ion, c_light, a_rad, kboltz, gamma, dt, Ti0,
                              &fval, &dfval);
  for (int it = 0; it < 100; ++it) {
    // bisect if Newton would leave the bracket or is converging too slowly
    bool out = (((te - thi)*dfval - fval)*((te - tlo)*dfval - fval) > 0.0);
    if (out || (Kokkos::fabs(2.0*fval) > Kokkos::fabs(dxold*dfval))) {
      dxold = dx;
      dx = 0.5*(thi - tlo);
      te = tlo + dx;
    } else {
      dxold = dx;
      dx = fval/dfval;
      te -= dx;
    }
    if (Kokkos::fabs(dx) <= 1.0e-14*Kokkos::fabs(te)) { break; }
    internal::ArrowheadResidual(te, ngroups, e_rad_old, group_bounds, chi, e_ele_old,
                                cv_ele, cv_ion, c_light, a_rad, kboltz, gamma, dt, Ti0,
                                &fval, &dfval);
    if (fval < 0.0) { tlo = te; } else { thi = te; }
  }

  // ---- form the new state from the converged T_e; conserve energy by construction ----
  Real beta_c = dt*c_light;
  Real inv_kt = 1.0/(kboltz*te);
  Real te4 = te*te*te*te;
  Real rad_gain = 0.0;                         // sum_g (E_g_new - E_g_old)
  for (int g = 0; g < ngroups; ++g) {
    Real beta_g = beta_c*chi[g];
    Real xlo = group_bounds[g]*inv_kt;
    Real xhi = group_bounds[g+1]*inv_kt;
    Real bg = a_rad*te4*(PlanckFraction(xhi) - PlanckFraction(xlo));
    Real eg = (e_rad_old[g] + beta_g*bg)/(1.0 + beta_g);
    e_rad_new[g] = eg;
    rad_gain += eg - e_rad_old[g];
  }
  Real ei_to_ele = 0.0;                        // energy ions -> electrons over the step
  if (gamma > 0.0 && (cv_ion + gamma) > 0.0) {
    ei_to_ele = gamma*cv_ion*(Ti0 - te)/(cv_ion + gamma);
    e_ion_new = e_ion_old - ei_to_ele;
  } else {
    e_ion_new = e_ion_old;                     // ions untouched when c_ei = 0
  }
  e_ele_new = e_ele_old - rad_gain + ei_to_ele;
  return true;
}

}  // namespace radiationfld

#endif  // RADIATION_FLD_MULTIGROUP_COUPLING_HPP_
