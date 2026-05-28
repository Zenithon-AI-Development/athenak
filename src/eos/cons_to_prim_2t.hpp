#ifndef EOS_CONS_TO_PRIM_2T_HPP_
#define EOS_CONS_TO_PRIM_2T_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cons_to_prim_2t.hpp
//! \brief Tabulated 2T EOS conserved->primitive closure (ADR-0002, issue [14b]/#26).
//!
//! In the conservative 3T energy formulation (ADR-0002) the matter carries two internal
//! energies -- the **electron internal energy density** `e_ele` (advected as the passive
//! scalar `e_ele/rho`, #19) and the **ion internal energy density** `e_ion` (recovered by
//! subtraction `E_tot - 1/2 rho v^2 - 1/2 B^2 - e_ele`, #19).  The gas pressure that
//! closes the MHD update is the sum of the two species pressures evaluated at their *own*
//! temperatures:
//!
//!     T_e = invert e_ele/rho via the tabulated EOS   (eos_table_3t::EosTable3T::Te)
//!     T_i = invert e_ion/rho via the tabulated EOS   (eos_table_3t::EosTable3T::Ti)
//!     p_gas = p_ele(rho, T_e) + p_ion(rho, T_i)
//!
//! This header provides that per-cell closure as a single pure device function,
//! `ConsToPrim2T`, the `ConsToPrim`-path piece ADR-0002 calls out ("a tabulated 2T EOS in
//! the ConsToPrim path, taking (rho, e_ele, e_ion)").  The temperatures `T_e`/`T_i` it
//! returns are the *derived, cached* temperature fields of the formulation (per ADR-0002,
//! "temperatures are derived, cached fields").  The function is EOS-format-agnostic: it
//! consumes the common `eos_table_3t::EosTable3T` representation (#6) that the IONMIX
//! (#10) and SESAME (#11) readers populate, captured by value into the kernel exactly
//! like AthenaK's `EOS_Data`.
//!
//! Conventions (matching #6 / #19):
//!   - `e_ele`, `e_ion` are internal energy *densities* (per unit volume), the same kind
//!     of quantity as `IEN`/`E_tot`; they are the conserved inputs.  The table lookups
//!     want *specific* (per-mass) energy, so we divide by `rho` before inverting (#6
//!     "divide the evolved energy density by rho before inverting").
//!   - `rho > 0` is assumed (density floor enforced upstream in `ConsToPrim`).
//!   - Out-of-range `(rho, e)` is handled by the table itself (edge-clamped inversion and
//!     lookups, #6), so no floors are needed here.
//!   - In the ideal-gas limit (a table with per-species `p = (gamma-1) rho e`) the
//!     closure recovers `p_gas = (gamma_e-1) e_ele + (gamma_i-1) e_ion`, i.e. the exact
//!     ideal-gas pressure `(gamma-1) (e_ele + e_ion)` when the species share one `gamma`.
//!
//! All functions are `KOKKOS_INLINE_FUNCTION` (host + device) and the struct is a trivial
//! POD, so the closure inlines into a `par_for` `ConsToPrim` kernel with no overhead.

#include "athena.hpp"             // Real, Kokkos, KOKKOS_INLINE_FUNCTION
#include "eos/eos_table_3t.hpp"   // eos_table_3t::EosTable3T

namespace eos_table_3t {

//----------------------------------------------------------------------------------------
//! \struct GasState2T
//! \brief Result of the tabulated 2T conserved->primitive closure for one cell: the
//!  derived electron/ion temperatures (cached fields) and the species + total pressures.
struct GasState2T {
  Real te;     //!< electron temperature T_e (derived, cached field)
  Real ti;     //!< ion temperature T_i (derived, cached field)
  Real p_ele;  //!< electron pressure p_ele(rho, T_e)
  Real p_ion;  //!< ion pressure p_ion(rho, T_i)
  Real p_gas;  //!< total gas pressure p_gas = p_ele + p_ion (closes the MHD update)
};

//----------------------------------------------------------------------------------------
//! \fn ConsToPrim2T
//! \brief Per-cell tabulated 2T EOS conserved->primitive closure.  Given the density and
//!  the electron/ion internal energy *densities* `(rho, e_ele, e_ion)`, invert each
//!  species energy to its temperature via the tabulated EOS and return the derived
//!  temperatures plus `p_gas = p_ele(rho,T_e) + p_ion(rho,T_i)`.
//!
//!  \param table  populated tabulated 3T EOS (captured by value into the kernel)
//!  \param rho    mass density (> 0)
//!  \param e_ele  electron internal energy density (per volume)
//!  \param e_ion  ion internal energy density (per volume)
KOKKOS_INLINE_FUNCTION
GasState2T ConsToPrim2T(const EosTable3T &table, Real rho, Real e_ele, Real e_ion) {
  GasState2T s;
  // densities -> specific (per-mass) energies for the table inversion
  Real e_ele_specific = e_ele/rho;
  Real e_ion_specific = e_ion/rho;
  // derived, cached temperatures from per-cell monotonic e->T inversion
  s.te = table.Te(rho, e_ele_specific);
  s.ti = table.Ti(rho, e_ion_specific);
  // species pressures at their own temperatures, summed to close the MHD pressure
  s.p_ele = table.PressureEle(rho, s.te);
  s.p_ion = table.PressureIon(rho, s.ti);
  s.p_gas = s.p_ele + s.p_ion;
  return s;
}

}  // namespace eos_table_3t

#endif  // EOS_CONS_TO_PRIM_2T_HPP_
