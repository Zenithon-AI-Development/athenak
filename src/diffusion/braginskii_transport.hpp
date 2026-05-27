#ifndef DIFFUSION_BRAGINSKII_TRANSPORT_HPP_
#define DIFFUSION_BRAGINSKII_TRANSPORT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file braginskii_transport.hpp
//! \brief Magnetized Braginskii transport coefficients (ADR-0003/0006), issue [5].
//!
//! Pure device functions returning the parallel/perpendicular Braginskii electron+ion
//! thermal conductivities and the parallel/perpendicular electrical resistivities as
//! functions of density, temperature, mean ionization `Z`, Coulomb logarithm `lnLambda`
//! and magnetization `x = omega_c * tau`, plus the Bohm-like anomalous cross-field term
//! and the vacuum-resistivity floor.  These are the shared coefficients consumed by the
//! anisotropic conduction operator (issue [6a]/#18), the resistive-B operator and Ohmic
//! heating (issue [7a]/#20).  The module is intentionally physics-only (no grid, no code
//! units): callers convert their code-unit state to the SI inputs documented below,
//! evaluate these, and convert the result back -- exactly the pattern the existing
//! temperature-dependent conduction (`KappaTemp`) uses with the CGS `Units` helpers.
//!
//! Unit convention (SI):
//!   density n     [m^-3]   (electron density n_e or ion density n_i)
//!   temperature T [K]      (q = -kappa * grad(T))
//!   magnetic field B [T]
//!   collision time tau [s]
//!   thermal conductivity kappa [W m^-1 K^-1]
//!   electrical resistivity eta [Ohm m]   (use MagneticDiffusivity() for the L^2/T form
//!                                          that the induction-equation eta_ohm expects)
//!   magnetization x = omega_c * tau [dimensionless]
//!
//! The dimensionless Braginskii coefficients are the canonical Z = 1 values (Braginskii
//! 1965; reproduced e.g. in Huba, NRL Plasma Formulary).  Higher-Z polynomial corrections
//! to the dimensionless coefficients are a documented future refinement; the dominant Z
//! dependence already enters through the collision time and the gyrofrequency.

#include "athena.hpp"  // Real, Kokkos, SQR

namespace braginskii {

//----------------------------------------------------------------------------------------
// Physical constants (SI, CODATA-ish).  constexpr so they live happily on the device.
constexpr Real kPi               = 3.14159265358979323846;
constexpr Real kBoltzmann        = 1.380649e-23;      // J/K
constexpr Real kElectronMass     = 9.1093837015e-31;  // kg
constexpr Real kProtonMass       = 1.67262192369e-27; // kg
constexpr Real kElementaryCharge = 1.602176634e-19;   // C
constexpr Real kVacuumEps0       = 8.8541878128e-12;  // F/m
constexpr Real kVacuumMu0        = 1.25663706212e-6;  // H/m
constexpr Real kElectronVolt     = 1.602176634e-19;   // J  (1 eV in joules)

//----------------------------------------------------------------------------------------
// Dimensionless Braginskii transport coefficients for Z = 1 (Braginskii 1965).
// Electron thermal conduction: kappa = coeff(x) * (n_e k_B^2 T_e tau_e / m_e),
//   coeff(x) = (gamma1p x^2 + gamma0p)/(x^4 + delta1 x^2 + delta0); coeff(0) = gamma0.
constexpr Real kGamma0e  = 3.1616;   // parallel electron conduction coefficient
constexpr Real kGamma0pe = 11.92;    // perpendicular numerator (x^2 term absent)
constexpr Real kGamma1pe = 4.664;    // perpendicular numerator (x^2 coefficient)
constexpr Real kDelta0e  = 3.7703;   // denominator constant
constexpr Real kDelta1e  = 14.79;    // denominator x^2 coefficient
// Electron resistivity: eta = coeff(x) * eta0, eta0 = m_e/(n_e e^2 tau_e),
//   coeff(x) = 1 - (alpha1p x^2 + alpha0p)/(x^4 + delta1 x^2 + delta0); ->alpha0 at x=0.
constexpr Real kAlpha0e  = 0.5129;   // parallel resistivity coefficient
constexpr Real kAlpha0pe = 1.837;    // perpendicular numerator (x^2 term absent)
constexpr Real kAlpha1pe = 0.7796;   // perpendicular numerator (x^2 coefficient)
// Ion thermal conduction (Z-independent like-particle coefficients).
constexpr Real kGamma0i  = 3.906;    // parallel ion conduction coefficient
constexpr Real kGamma0pi = 2.645;    // perpendicular numerator (x^2 term absent)
constexpr Real kGamma1pi = 2.0;      // perpendicular numerator (x^2 coefficient)
constexpr Real kDelta0i  = 0.677;    // denominator constant
constexpr Real kDelta1i  = 2.70;     // denominator x^2 coefficient

//----------------------------------------------------------------------------------------
//! \fn Real CoulombLog
//! \brief NRL electron-ion Coulomb logarithm ln(Lambda) from (n_e [m^-3], T_e [K], Z).
//! Two-branch fit (T_e below/above 10 Z^2 eV), floored at 1 to stay positive in
//! cold/dense cells.  A convenience helper; the coefficient functions take ln(Lambda)
//! explicitly so a caller can supply a tabulated or clamped value instead.
KOKKOS_INLINE_FUNCTION
Real CoulombLog(Real n_e, Real T_e, Real Z) {
  Real n_e_cm3 = n_e * 1.0e-6;                       // m^-3 -> cm^-3
  Real T_e_eV  = kBoltzmann * T_e / kElectronVolt;   // K -> eV
  Real ln_lambda;
  if (T_e_eV < 10.0 * Z * Z) {
    ln_lambda = 23.0 - log(sqrt(n_e_cm3) * Z * pow(T_e_eV, -1.5));
  } else {
    ln_lambda = 24.0 - log(sqrt(n_e_cm3) * pow(T_e_eV, -1.0));
  }
  return fmax(ln_lambda, 1.0);
}

//----------------------------------------------------------------------------------------
//! \fn Real ElectronCollisionTime
//! \brief Braginskii electron collision time tau_e [s] (SI).  Reproduces the NRL
//! practical value tau_e = 3.44e5 T_e[eV]^1.5 / (n_e[cm^-3] Z ln(Lambda)).  Inputs:
//! n_e [m^-3], T_e [K], mean ionization Z, Coulomb log ln(Lambda).
KOKKOS_INLINE_FUNCTION
Real ElectronCollisionTime(Real n_e, Real T_e, Real Z, Real ln_lambda) {
  Real kT = kBoltzmann * T_e;
  Real e4 = SQR(SQR(kElementaryCharge));
  Real fourpieps0_sq = SQR(4.0 * kPi * kVacuumEps0);
  return 3.0 * sqrt(kElectronMass) * pow(kT, 1.5) * fourpieps0_sq /
         (4.0 * sqrt(2.0 * kPi) * n_e * Z * e4 * ln_lambda);
}

//----------------------------------------------------------------------------------------
//! \fn Real IonCollisionTime
//! \brief Braginskii ion collision time tau_i [s] (SI).  Reproduces the NRL practical
//! value tau_i = 2.09e7 T_i[eV]^1.5 sqrt(mu) / (n_i[cm^-3] Z^4 ln(Lambda)), mu = m_i/m_p.
//! Inputs: n_i [m^-3], T_i [K], Z, ion mass m_i [kg], Coulomb log ln(Lambda).
KOKKOS_INLINE_FUNCTION
Real IonCollisionTime(Real n_i, Real T_i, Real Z, Real m_i, Real ln_lambda) {
  Real kT = kBoltzmann * T_i;
  Real e4 = SQR(SQR(kElementaryCharge));
  Real fourpieps0_sq = SQR(4.0 * kPi * kVacuumEps0);
  Real Z4 = SQR(SQR(Z));
  return 3.0 * sqrt(m_i) * pow(kT, 1.5) * fourpieps0_sq /
         (4.0 * sqrt(kPi) * n_i * Z4 * e4 * ln_lambda);
}

//----------------------------------------------------------------------------------------
//! \fn Real ElectronGyroFrequency / IonGyroFrequency
//! \brief Cyclotron frequency omega_c [s^-1] (SI): omega_ce = eB/m_e, omega_ci = ZeB/m_i.
KOKKOS_INLINE_FUNCTION
Real ElectronGyroFrequency(Real b_mag) {
  return kElementaryCharge * b_mag / kElectronMass;
}
KOKKOS_INLINE_FUNCTION
Real IonGyroFrequency(Real b_mag, Real Z, Real m_i) {
  return Z * kElementaryCharge * b_mag / m_i;
}

//----------------------------------------------------------------------------------------
// Dimensionless magnetized multipliers as functions of x = omega_c * tau.
//   *PerpCoeff* return the full multiplier so that <Coeff>(0) equals the parallel value
//   (the unmagnetized limit), and they roll off / saturate at large x.
KOKKOS_INLINE_FUNCTION
Real ConductionCoeffPerpElectron(Real x) {
  Real x2 = SQR(x);
  return (kGamma1pe * x2 + kGamma0pe) / (SQR(x2) + kDelta1e * x2 + kDelta0e);
}
KOKKOS_INLINE_FUNCTION
Real ConductionCoeffPerpIon(Real x) {
  Real x2 = SQR(x);
  return (kGamma1pi * x2 + kGamma0pi) / (SQR(x2) + kDelta1i * x2 + kDelta0i);
}
KOKKOS_INLINE_FUNCTION
Real ResistivityCoeffPerpElectron(Real x) {
  Real x2 = SQR(x);
  return 1.0 - (kAlpha1pe * x2 + kAlpha0pe) / (SQR(x2) + kDelta1e * x2 + kDelta0e);
}

//----------------------------------------------------------------------------------------
//! \fn Real ConductivityParElectron / ConductivityPerpElectron
//! \brief Braginskii electron thermal conductivity kappa [W m^-1 K^-1] (q=-kappa gradT).
//! Base scale n_e k_B^2 T_e tau_e / m_e; parallel uses gamma0, perpendicular the
//! magnetized multiplier of x_e = omega_ce tau_e (-> gamma0 as x_e->0).
KOKKOS_INLINE_FUNCTION
Real ConductivityParElectron(Real n_e, Real T_e, Real tau_e) {
  return kGamma0e * n_e * SQR(kBoltzmann) * T_e * tau_e / kElectronMass;
}
KOKKOS_INLINE_FUNCTION
Real ConductivityPerpElectron(Real n_e, Real T_e, Real tau_e, Real x_e) {
  Real base = n_e * SQR(kBoltzmann) * T_e * tau_e / kElectronMass;
  return ConductionCoeffPerpElectron(x_e) * base;
}

//----------------------------------------------------------------------------------------
//! \fn Real ConductivityParIon / ConductivityPerpIon
//! \brief Braginskii ion thermal conductivity kappa [W m^-1 K^-1].  Base scale
//! n_i k_B^2 T_i tau_i / m_i; magnetization x_i = omega_ci tau_i.
KOKKOS_INLINE_FUNCTION
Real ConductivityParIon(Real n_i, Real T_i, Real tau_i, Real m_i) {
  return kGamma0i * n_i * SQR(kBoltzmann) * T_i * tau_i / m_i;
}
KOKKOS_INLINE_FUNCTION
Real ConductivityPerpIon(Real n_i, Real T_i, Real tau_i, Real m_i, Real x_i) {
  Real base = n_i * SQR(kBoltzmann) * T_i * tau_i / m_i;
  return ConductionCoeffPerpIon(x_i) * base;
}

//----------------------------------------------------------------------------------------
//! \fn Real ResistivityParElectron / ResistivityPerpElectron
//! \brief Braginskii electrical resistivity eta [Ohm m].  Base scale eta0 = m_e/(n_e e^2
//! tau_e) (= the unmagnetized perpendicular / Lorentz value); parallel is alpha0 * eta0
//! (the Spitzer parallel resistivity ~ 5.2e-5 Z ln(Lambda) T_e[eV]^-1.5 Ohm m), and the
//! perpendicular multiplier rises from alpha0 (x->0) toward 1 (x->inf), so
//! eta_perp/eta_par -> 1/alpha0 ~ 1.95 in the strongly magnetized limit.
KOKKOS_INLINE_FUNCTION
Real ResistivityParElectron(Real n_e, Real tau_e) {
  Real eta0 = kElectronMass / (n_e * SQR(kElementaryCharge) * tau_e);
  return kAlpha0e * eta0;
}
KOKKOS_INLINE_FUNCTION
Real ResistivityPerpElectron(Real n_e, Real tau_e, Real x_e) {
  Real eta0 = kElectronMass / (n_e * SQR(kElementaryCharge) * tau_e);
  return ResistivityCoeffPerpElectron(x_e) * eta0;
}

//----------------------------------------------------------------------------------------
//! \fn Real MagneticDiffusivity
//! \brief Convert an electrical resistivity eta [Ohm m] to the magnetic diffusivity
//! eta/mu0 [m^2 s^-1] -- the form the induction-equation resistive operator (eta_ohm,
//! dE = eta J) uses, since d_t B = (eta/mu0) grad^2 B.
KOKKOS_INLINE_FUNCTION
Real MagneticDiffusivity(Real eta_si) {
  return eta_si / kVacuumMu0;
}

//----------------------------------------------------------------------------------------
// Bohm-like anomalous cross-field term.  The standard Bohm diffusion coefficient is
// D_Bohm = (1/16) k_B T_e / (e B), i.e. an anomalous collision freq nu = omega_ce/16.
// (ADR-0006 quotes "nu_bohm = 16 omega_ce"; that reads as the inverse of the standard
// Bohm relation -- the 16 belongs in the denominator -- so the physical 1/16 form is used
// here, with the factor exposed as a named constant for easy adjustment.)
constexpr Real kBohmFactor = 1.0 / 16.0;
KOKKOS_INLINE_FUNCTION
Real BohmCollisionFrequency(Real omega_ce) {
  return kBohmFactor * omega_ce;
}
KOKKOS_INLINE_FUNCTION
Real BohmDiffusivity(Real T_e, Real b_mag) {
  return kBohmFactor * kBoltzmann * T_e / (kElementaryCharge * b_mag);
}

//----------------------------------------------------------------------------------------
//! \fn Real ApplyVacuumResistivityFloor
//! \brief Vacuum-resistivity floor: in cells below the user density floor rho_floor, use
//! the large vacuum resistivity eta_vac (suppressing unphysical currents in near-vacuum);
//! otherwise return the classical eta unchanged.
KOKKOS_INLINE_FUNCTION
Real ApplyVacuumResistivityFloor(Real eta, Real rho, Real rho_floor, Real eta_vac) {
  return (rho < rho_floor) ? eta_vac : eta;
}

}  // namespace braginskii

#endif  // DIFFUSION_BRAGINSKII_TRANSPORT_HPP_
