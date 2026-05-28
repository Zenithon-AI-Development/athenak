#ifndef DIFFUSION_VARIABLE_RESISTIVITY_HPP_
#define DIFFUSION_VARIABLE_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file variable_resistivity.hpp
//! \brief Device-inline Spitzer/Braginskii variable magnetic diffusivity eta(rho,T_e)
//!  for the Ohmic resistive operator (ADR-0003).  Wraps the SIM-61 transport-coefficient
//!  module (braginskii_transport.hpp): converts a cell's code-unit (rho, temperature,
//!  |B|) to SI, evaluates the Braginskii electrical resistivity eta_par/eta_perp [Ohm m],
//!  converts to the magnetic diffusivity eta/mu0 [m^2/s] (the form the induction-equation
//!  resistive operator uses: E_resistive = eta J, d_t B = eta grad^2 B), and back to code
//!  units, then applies the vacuum-resistivity floor (large eta below a density
//!  threshold) to suppress unphysical vacuum currents.
//!
//!  The conversion factors are supplied as plain linear scales (code -> SI for the
//!  inputs, SI -> code for the output diffusivity) so this module is decoupled from the
//!  global <units> block, mirroring the anisotropic-conduction operator (SIM-78 / #18).

#include "athena.hpp"
#include "braginskii_transport.hpp"

namespace resistivity {

//----------------------------------------------------------------------------------------
//! \struct VarResistivityParams
//! \brief Linear code<->SI conversions + plasma parameters for the Braginskii/Spitzer
//!  magnetic diffusivity, plus the vacuum-resistivity floor.  Trivially copyable so it
//!  can be captured by value into a Kokkos par_for.
struct VarResistivityParams {
  Real dens_si = 1.0;     // code mass density -> SI [kg m^-3]
  Real temp_si = 1.0;     // code temperature  -> SI [K]
  Real bmag_si = 1.0;     // code |B|          -> SI [Tesla]
  Real eta_si  = 1.0;     // SI magnetic diffusivity [m^2 s^-1] -> code units
  Real zbar    = 1.0;     // mean ionization Z
  Real m_i     = braginskii::kProtonMass;  // ion mass [kg]
  Real ln_lambda = 0.0;   // Coulomb log; <= 0 => use NRL CoulombLog(n_e,T_e,Z)
  bool anisotropic = false;  // include omega_ce*tau_e magnetization (use eta_perp)
  Real rho_floor = -1.0;  // vacuum density threshold (code); < 0 disables the floor
  Real eta_vac   = 0.0;   // vacuum magnetic diffusivity (code) assigned below rho_floor
};

//----------------------------------------------------------------------------------------
//! \fn void SpitzerMagneticDiffusivity
//! \brief Fill the parallel and perpendicular magnetic diffusivities (CODE units) from a
//!  cell's code-unit (rho, temperature, |B|).  In the unmagnetized limit (|B| -> 0, or
//!  anisotropic == false) eta_perp == eta_par == the Spitzer parallel value.  The
//!  vacuum-resistivity floor overrides BOTH components below rho_floor.
KOKKOS_INLINE_FUNCTION
void SpitzerMagneticDiffusivity(const VarResistivityParams &p, Real rho_code,
                                Real temp_code, Real bmag_code,
                                Real &eta_par_code, Real &eta_perp_code) {
  Real n_i  = (rho_code * p.dens_si) / p.m_i;   // ion number density [m^-3]
  Real n_e  = p.zbar * n_i;                      // electron number density [m^-3]
  Real T_e  = temp_code * p.temp_si;             // electron temperature [K]
  Real B_si = bmag_code * p.bmag_si;             // field magnitude [Tesla]

  Real lnL = (p.ln_lambda > 0.0) ? p.ln_lambda
                                  : braginskii::CoulombLog(n_e, T_e, p.zbar);
  Real tau_e = braginskii::ElectronCollisionTime(n_e, T_e, p.zbar, lnL);

  Real eta_par_si  = braginskii::ResistivityParElectron(n_e, tau_e);   // Ohm m
  Real eta_perp_si = eta_par_si;
  if (p.anisotropic) {
    Real x_e = braginskii::ElectronGyroFrequency(B_si) * tau_e;        // omega_ce*tau_e
    eta_perp_si = braginskii::ResistivityPerpElectron(n_e, tau_e, x_e);
  }

  eta_par_code  = braginskii::MagneticDiffusivity(eta_par_si)  * p.eta_si;
  eta_perp_code = braginskii::MagneticDiffusivity(eta_perp_si) * p.eta_si;

  // vacuum-resistivity floor (SIM-61 helper): assign the large vacuum diffusivity in
  // near-vacuum cells (rho < rho_floor) to suppress unphysical currents there.
  if (p.rho_floor > 0.0) {
    eta_par_code  = braginskii::ApplyVacuumResistivityFloor(eta_par_code, rho_code,
                                                            p.rho_floor, p.eta_vac);
    eta_perp_code = braginskii::ApplyVacuumResistivityFloor(eta_perp_code, rho_code,
                                                            p.rho_floor, p.eta_vac);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real EffectiveDiffusivity
//! \brief The single scalar magnetic diffusivity fed into the (isotropic-EMF)
//!  constrained-transport resistive update.  Returns the cross-field eta_perp when
//!  anisotropic (the value that governs B diffusion across field lines, e.g. J_z x B_phi
//!  in a Z-pinch), else the Spitzer parallel eta_par.  Both reduce to the same Spitzer
//!  value when unmagnetized.
KOKKOS_INLINE_FUNCTION
Real EffectiveDiffusivity(const VarResistivityParams &p, Real rho_code,
                          Real temp_code, Real bmag_code) {
  Real eta_par, eta_perp;
  SpitzerMagneticDiffusivity(p, rho_code, temp_code, bmag_code, eta_par, eta_perp);
  return p.anisotropic ? eta_perp : eta_par;
}

}  // namespace resistivity

#endif  // DIFFUSION_VARIABLE_RESISTIVITY_HPP_
