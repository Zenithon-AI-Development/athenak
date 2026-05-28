#ifndef EOS_THREE_TEMPERATURE_HPP_
#define EOS_THREE_TEMPERATURE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file three_temperature.hpp
//! \brief Conservative 3T energy reconciliation: electron internal energy + ion-by-
//!        subtraction (ADR-0002, issue [14a]/#19).
//!
//! The 3T energy formulation (ADR-0002) keeps AthenaK's conservative total-energy MHD
//! update *unchanged* (so shock jumps and global energy conservation are guaranteed) and
//! additionally evolves the **electron internal energy density** `e_ele`.  `e_ele` is
//! transported by the existing **passive-scalar machinery** as the specific quantity
//! `e_ele/rho` (conserved scalar `s = rho * (e_ele/rho) = e_ele`, primitive `e_ele/rho`),
//! so no new advection code is needed -- the scalar's conserved value IS the electron
//! energy density and it rides the upwind scalar flux.  On top of that advection two
//! operator-split pieces are added per cell, both provided here as pure device functions:
//!
//!   1. **PdV (compression) heating split by pressure fraction.**  The reversible
//!      compression work rate on the gas is `-p_gas * div(v)` per unit volume; the
//!      electron share is `f_ele = p_ele/p_gas` of it, i.e.
//!      `d e_ele/dt|_PdV = -p_ele * div(v)`.
//!      Adding only the electron PdV term and recovering the ion energy by subtraction
//!      (below) gives the ion its complementary `-p_ion*div(v)` share automatically.
//!
//!   2. **Ion internal energy by subtraction:**
//!         `e_ion = E_tot - 1/2 rho v^2 - 1/2 B^2 - e_ele`.
//!      Because `E_tot` is the (unchanged) conserved total, total energy is conserved by
//!      construction and *irreversible shock dissipation lands on the ions* (the excess
//!      of the internal-energy increment over the reversible electron PdV ends up in
//!      `e_ion`).  This is the physically correct partition (CONTEXT.md: shock heating ->
//!      ions; T_e drives conduction/resistivity/opacity).
//!
//! This module is deliberately EOS-agnostic and per-cell: the pressures `p_ele`, `p_ion`
//! ride in as arguments.  A consumer obtains them from the tabulated 3T EOS (#6,
//! `eos/eos_table_3t.hpp`): invert `e_ele/rho -> T_e` and `e_ion/rho -> T_i`, then
//! `p_ele = PressureEle(rho,T_e)`, `p_ion = PressureIon(rho,T_i)`.  Wiring `e_ele`
//! through `ConsToPrim` with `p_gas = p_ele + p_ion` is issue [14b]/#26.  All functions
//! here are `KOKKOS_INLINE_FUNCTION` (host + device), captured into a `par_for`.
//!
//! On top of the advection + PdV split, issue [14c]/#30 (ADR-0003) adds the **Ohmic
//! (Joule) heating** source to `e_ele`.  AthenaK advances the resistive energy as a
//! conservative Poynting flux `dE/dt = -div(eta (J x B))` while the resistive EMF
//! `E_res = eta J` (the same edge current `J = curl B` that drives constrained-transport
//! B-diffusion) decays the field; the dissipated magnetic energy reappears in the
//! conserved `E_tot` as internal energy.  With `e_ion` recovered by subtraction that
//! implicit Joule heat lands on the *ions* by default -- physically backwards, since the
//! current is electron-carried.  ADR-0003's fix: deposit into `e_ele` the *local
//! (non-transport) magnetic-energy decrement* of the CT update, computed from the **same
//! discrete `eta` and `J`** (so electron gain == magnetic loss), then recover `e_ion` by
//! subtraction.  The local dissipation rate per unit volume is `E_res . J = eta |J|^2`;
//! depositing `eta |J|^2 dt` to `e_ele` leaves the ion residual unchanged and keeps total
//! energy conserved (`OhmicHeatingRate` / `ApplyOhmicElectronHeating` below).  Do **not**
//! re-discretize `eta J^2` at a different centering -- that mismatch is the energy leak
//! ADR-0003 warns about; pass the consumer's CT `J` and `eta` in directly.
//!
//! Heaviside-Lorentz code units: magnetic energy density is `B^2/2` (CONTEXT.md),
//! matching the rest of AthenaK's MHD (e.g. the energy assembled in PrimToCons).

#include "athena.hpp"  // Real, Kokkos, KOKKOS_INLINE_FUNCTION, SQR

namespace three_temp {

//----------------------------------------------------------------------------------------
//! \fn KineticEnergyDensity
//! \brief Bulk kinetic energy density 1/2 rho v^2 = 1/2 |m|^2 / rho from the momentum
//!  density components (m = rho v).  rho must be > 0.
KOKKOS_INLINE_FUNCTION
Real KineticEnergyDensity(Real rho, Real m1, Real m2, Real m3) {
  return 0.5*(SQR(m1) + SQR(m2) + SQR(m3))/rho;
}

//----------------------------------------------------------------------------------------
//! \fn MagneticEnergyDensity
//! \brief Magnetic energy density 1/2 B^2 (Heaviside-Lorentz code units).
KOKKOS_INLINE_FUNCTION
Real MagneticEnergyDensity(Real bx, Real by, Real bz) {
  return 0.5*(SQR(bx) + SQR(by) + SQR(bz));
}

//----------------------------------------------------------------------------------------
//! \fn IonInternalEnergyHyd
//! \brief Hydro ion internal energy density by subtraction:
//!         e_ion = E_tot - 1/2 rho v^2 - e_ele.
//!  (m1,m2,m3) are the momentum-density components.  By construction
//!  e_ele + e_ion + KE = E_tot, so total energy is conserved exactly.
KOKKOS_INLINE_FUNCTION
Real IonInternalEnergyHyd(Real etot, Real rho, Real m1, Real m2, Real m3, Real eele) {
  return etot - KineticEnergyDensity(rho, m1, m2, m3) - eele;
}

//----------------------------------------------------------------------------------------
//! \fn IonInternalEnergyMHD
//! \brief MHD ion internal energy density by subtraction:
//!         e_ion = E_tot - 1/2 rho v^2 - 1/2 B^2 - e_ele.
//!  (bx,by,bz) are the cell-centred magnetic-field components.  By construction
//!  e_ele + e_ion + KE + 1/2 B^2 = E_tot.
KOKKOS_INLINE_FUNCTION
Real IonInternalEnergyMHD(Real etot, Real rho, Real m1, Real m2, Real m3,
                          Real bx, Real by, Real bz, Real eele) {
  return etot - KineticEnergyDensity(rho, m1, m2, m3)
             - MagneticEnergyDensity(bx, by, bz) - eele;
}

//----------------------------------------------------------------------------------------
//! \fn ElectronPressureFraction
//! \brief Electron share of the gas pressure, f_ele = p_ele/(p_ele + p_ion).  This is the
//!  fraction of the compression (PdV) work that heats the electrons.  Guards a
//!  non-positive total gas pressure (cold/vacuum cell) by returning 0 (no electron
//!  share).
KOKKOS_INLINE_FUNCTION
Real ElectronPressureFraction(Real pele, Real pion) {
  Real pgas = pele + pion;
  return (pgas > 0.0) ? pele/pgas : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn ElectronPdVRate
//! \brief Electron PdV (compression) heating rate per unit volume,
//!         d e_ele/dt|_PdV = -p_ele * div(v).
//!  Equals the pressure-fraction share f_ele of the total reversible work -p_gas*div(v):
//!  compression (div v < 0) heats the electrons, expansion cools them.  The velocity
//!  divergence div(v) is supplied by the consumer (computed from the cell-centred
//!  velocities via the coordinate metric); this is the per-cell algebra only.
KOKKOS_INLINE_FUNCTION
Real ElectronPdVRate(Real pele, Real divv) {
  return -pele*divv;
}

//----------------------------------------------------------------------------------------
//! \fn ApplyElectronPdV
//! \brief Operator-split electron PdV update over a step dt:
//!         e_ele <- e_ele - p_ele * div(v) * dt.
//!  Applied after the passive-scalar advection of e_ele; the complementary ion PdV share
//!  is delivered automatically by recovering e_ion via subtraction.
KOKKOS_INLINE_FUNCTION
Real ApplyElectronPdV(Real eele, Real pele, Real divv, Real dt) {
  return eele + ElectronPdVRate(pele, divv)*dt;
}

//----------------------------------------------------------------------------------------
//! \fn OhmicHeatingRate
//! \brief Local Ohmic (Joule) dissipation rate per unit volume, `eta |J|^2`, where
//!  `(j1,j2,j3)` are the cell-centred current-density components `J = curl B` and `eta`
//!  is the magnetic diffusivity (ADR-0003, issue [14c]/#30).  This equals the resistive
//!  electric field dotted with the current, `E_res . J` with `E_res = eta J`, i.e. the
//!  rate at which the constrained-transport resistive update converts magnetic energy
//!  into internal energy locally.  Computing it from the SAME discrete `eta` and `J` that
//!  drive the CT B-diffusion is what keeps the electron heating consistent with the field
//!  decay (no energy leak).  `eta >= 0`.
KOKKOS_INLINE_FUNCTION
Real OhmicHeatingRate(Real eta, Real j1, Real j2, Real j3) {
  return eta*(SQR(j1) + SQR(j2) + SQR(j3));
}

//----------------------------------------------------------------------------------------
//! \fn ApplyOhmicElectronHeating
//! \brief Operator-split electron Ohmic-heating update over a step dt:
//!         e_ele <- e_ele + eta |J|^2 dt.
//!  The Joule dissipation is deposited entirely on the electrons (the current is
//!  electron-carried).  Because the conservative `E_tot` update has already returned the
//!  dissipated magnetic energy to the internal-energy budget, claiming exactly that local
//!  decrement (`eta |J|^2 dt`) for `e_ele` and recovering `e_ion` by subtraction
//!  (`IonInternalEnergyMHD`) leaves the ion energy unchanged and conserves total energy
//!  by construction (ADR-0003).
KOKKOS_INLINE_FUNCTION
Real ApplyOhmicElectronHeating(Real eele, Real eta, Real j1, Real j2, Real j3, Real dt) {
  return eele + OhmicHeatingRate(eta, j1, j2, j3)*dt;
}

}  // namespace three_temp

#endif  // EOS_THREE_TEMPERATURE_HPP_
