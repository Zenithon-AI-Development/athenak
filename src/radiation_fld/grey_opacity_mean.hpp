#ifndef RADIATION_FLD_GREY_OPACITY_MEAN_HPP_
#define RADIATION_FLD_GREY_OPACITY_MEAN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grey_opacity_mean.hpp
//! \brief Grey (frequency-integrated) reductions of a tabulated multigroup opacity
//! (issue #204).  The grey FLD operator and the grey matter-radiation coupling need a
//! LOCAL per-cell opacity chi(rho,Te) = rho*kappa(rho,Te) in place of the frozen
//! reference-state constant of ADR-0014 (#184) -- the constant treats the tenuous vacuum
//! gap at solid-liner opacity, so radiation cannot free-stream across it (the rank-1 B1
//! faithfulness gap).  The IONMIX tables are ingested per photon-energy GROUP
//! (opacity::MultigroupOpacity); these helpers collapse the group structure to the two
//! standard grey means at the local state:
//!
//!   Rosseland (transport, for the FLD diffusion coefficient D = c lambda/chi):
//!     the dB/dT-weighted HARMONIC mean -- transparent windows dominate transport,
//!       1/kappa_R = sum_g w_g/kappa_R,g / sum_g w_g,
//!       w_g = int_{x_g}^{x_{g+1}} x^4 e^x/(e^x-1)^2 dx = [4F - G]_{x_g}^{x_{g+1}},
//!   Planck (absorption, for the emission/absorption coupling c chi_a (aT^4 - E)):
//!     the B-weighted ARITHMETIC mean -- strong lines dominate absorption,
//!       kappa_P = sum_g b_g kappa_P,g / sum_g b_g,   b_g = [F]_{x_g}^{x_{g+1}},
//!
//! with x = eps/(k_B Te) and F/G the fractional Planck function and its x-derivative
//! helper already shared by the multigroup point-implicit coupling
//! (radiation_fld/multigroup_coupling.hpp).  The 15/pi^4 normalization cancels in both
//! ratios.  When the Planck spectrum lies entirely outside the tabulated group range
//! (all weights underflow to 0), the mean collapses to the nearest edge group -- the
//! physically-adjacent band -- rather than dividing 0/0.  Both helpers are
//! KOKKOS_INLINE_FUNCTION (host + device) and O(ngroups) per call; the table lookups
//! clamp (rho,Te) to the tabulated range exactly as the per-group accessors do.

#include "athena.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "radiation_fld/multigroup_coupling.hpp"  // PlanckFraction(+XDeriv)

namespace radiationfld {

//----------------------------------------------------------------------------------------
//! \fn Real GreyRosselandMean
//! \brief dB/dT-weighted harmonic mean of the per-group Rosseland transport opacity at
//! (rho, te): the grey transport opacity [same mass-opacity units as the table].
//! `kboltz` converts the table's photon-energy group bounds to the temperature unit
//! (x = eps/(kboltz*te); 1.0 when both are eV, as for IONMIX + MagLIF code units).
KOKKOS_INLINE_FUNCTION
Real GreyRosselandMean(const opacity::MultigroupOpacity &tab, Real rho, Real te,
                       Real kboltz) {
  const int ng = tab.ngroups;
  if (ng == 1 || !(te > 0.0)) { return tab.RosselandTransport(0, rho, te); }
  const Real inv_kt = 1.0/(kboltz*te);
  Real wsum = 0.0, hsum = 0.0;
  for (int g = 0; g < ng; ++g) {
    Real xlo = tab.GroupBound(g)*inv_kt;
    Real xhi = tab.GroupBound(g + 1)*inv_kt;
    Real w = 4.0*(PlanckFraction(xhi) - PlanckFraction(xlo))
             - (PlanckFractionXDeriv(xhi) - PlanckFractionXDeriv(xlo));
    if (w > 0.0) {
      wsum += w;
      hsum += w/tab.RosselandTransport(g, rho, te);
    }
  }
  if (!(wsum > 0.0) || !(hsum > 0.0)) {
    // spectrum entirely off the tabulated group range: nearest edge group
    int g_edge = (tab.GroupBound(ng)*inv_kt < 1.0) ? (ng - 1) : 0;
    return tab.RosselandTransport(g_edge, rho, te);
  }
  return wsum/hsum;
}

//----------------------------------------------------------------------------------------
//! \fn Real GreyPlanckMean
//! \brief Planck-fraction-weighted arithmetic mean of the per-group Planck absorption
//! opacity at (rho, te): the grey absorption opacity [table mass-opacity units].
KOKKOS_INLINE_FUNCTION
Real GreyPlanckMean(const opacity::MultigroupOpacity &tab, Real rho, Real te,
                    Real kboltz) {
  const int ng = tab.ngroups;
  if (ng == 1 || !(te > 0.0)) { return tab.PlanckAbsorption(0, rho, te); }
  const Real inv_kt = 1.0/(kboltz*te);
  Real bsum = 0.0, ksum = 0.0;
  for (int g = 0; g < ng; ++g) {
    Real xlo = tab.GroupBound(g)*inv_kt;
    Real xhi = tab.GroupBound(g + 1)*inv_kt;
    Real b = PlanckFraction(xhi) - PlanckFraction(xlo);
    if (b > 0.0) {
      bsum += b;
      ksum += b*tab.PlanckAbsorption(g, rho, te);
    }
  }
  if (!(bsum > 0.0)) {
    int g_edge = (tab.GroupBound(ng)*inv_kt < 1.0) ? (ng - 1) : 0;
    return tab.PlanckAbsorption(g_edge, rho, te);
  }
  return ksum/bsum;
}

}  // namespace radiationfld

#endif  // RADIATION_FLD_GREY_OPACITY_MEAN_HPP_
