#ifndef OPACITY_MULTIGROUP_OPACITY_HPP_
#define OPACITY_MULTIGROUP_OPACITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file multigroup_opacity.hpp
//! \brief Common internal representation for tabulated multigroup opacities + device
//!        lookups (ADR-0007, issue [16a]/#7).
//!
//! The multigroup flux-limited-diffusion (FLD) radiation model (ADR-0001, CONTEXT.md)
//! needs, per photon-energy *group*, the Planck absorption, Planck emission and Rosseland
//! transport opacities as functions of `(rho, T_e)`.  Transport opacity feeds the FLD
//! diffusion coefficient `D = c/(3 kappa_R)`; Planck absorption/emission feed the
//! point-implicit matter-radiation coupling.  This header is the shared, format-agnostic
//! representation that the concrete file readers populate -- IONMIX now
//! (`ionmix_opacity_reader.hpp`), TOPS/PROPACEOS later target the same struct.  It never
//! touches the disk: a reader (or a test) allocates the table, fills the per-node arrays
//! through host mirrors, and copies them to the device.
//!
//! Layout and conventions (mirrors `eos/eos_table_3t.hpp`):
//!   - Two log-spaced axes: log10(T_e) [temperature index `it` in [0,ntemp)] and
//!     log10(rho) [density index `id` in [0,ndens)], plus a discrete group index
//!     `ig` in [0,ngroups).  Each opacity field is a `DvceArray3D` of shape
//!     (ngroups, ntemp, ndens) -- group-major, then temperature, then density.
//!   - `group_bounds` holds the `ngroups+1` ascending photon-energy boundaries (group
//!     `ig` spans `[group_bounds(ig), group_bounds(ig+1)]`); units are the reader's
//!     (eV for IONMIX).
//!   - Opacities are stored as *mass* (specific) opacities [length^2 / mass], the IONMIX
//!     convention; the FLD coefficient consumer multiplies by rho to get an inverse
//!     length.  The representation is otherwise unit-agnostic -- the reader sets units.
//!   - Opacities are positive-definite, so they are interpolated bilinearly on
//!     log10(value) in (log10 rho, log10 T_e).  Log-log interpolation reproduces any
//!     power-law opacity exactly, on and off grid.
//!   - Out-of-range queries are clamped to the table bounds (the interpolation clamps the
//!     (rho,T_e) fractional index), so no out-of-bounds memory access occurs and an
//!     out-of-range (rho,T_e) returns the nearest edge value.
//!
//! The struct holds Kokkos Views and PODs, so it is copied by value into a `par_for`
//! kernel exactly like AthenaK's `EOS_Data` / `eos_table_3t::EosTable3T` (shallow View
//! copy, device-valid pointers); all lookups are `const KOKKOS_INLINE_FUNCTION`.

#include <cmath>       // std::log10

#include "athena.hpp"  // Real, Kokkos, DvceArray1D, DvceArray3D

namespace opacity {

//----------------------------------------------------------------------------------------
//! \enum OpacityKind
//! \brief Selects which opacity field a generic lookup operates on.
enum OpacityKind { kPlanckAbsorption = 0, kPlanckEmission = 1, kRosseland = 2 };

//----------------------------------------------------------------------------------------
//! \struct MultigroupOpacity
//! \brief Device-friendly tabulated multigroup opacity: log-spaced (rho,T_e) grid +
//!        per-group Planck absorption / Planck emission / Rosseland transport opacities,
//!        with edge-clamped log-log bilinear device lookups.
struct MultigroupOpacity {
  // --- axis metadata (scalars; trivially copied to the device) ---
  int ntemp = 0;            //!< number of temperature nodes (>= 2)
  int ndens = 0;            //!< number of density nodes (>= 2)
  int ngroups = 0;          //!< number of photon-energy groups (>= 1)
  Real log_temp_min = 0.0;  //!< log10 of minimum tabulated temperature
  Real log_dens_min = 0.0;  //!< log10 of minimum tabulated density
  Real dlog_temp = 0.0;     //!< log10(T) node spacing
  Real dlog_dens = 0.0;     //!< log10(rho) node spacing
  Real inv_dlog_temp = 0.0; //!< 1 / dlog_temp (precomputed for the hot path)
  Real inv_dlog_dens = 0.0; //!< 1 / dlog_dens

  // --- group structure: ngroups+1 ascending photon-energy boundaries ---
  DvceArray1D<Real> group_bounds;

  // --- per-node opacity data, shape (ngroups, ntemp, ndens), mass opacity [L^2/M] ---
  DvceArray3D<Real> planck_absorb;  //!< Planck-mean absorption opacity kappa_pa(ig,T,rho)
  DvceArray3D<Real> planck_emiss;   //!< Planck-mean emission opacity   kappa_pe(ig,T,rho)
  DvceArray3D<Real> rosseland;      //!< Rosseland transport opacity    kappa_R (ig,T,rho)

  //--------------------------------------------------------------------------------------
  //! \fn Allocate
  //! \brief Host-side: size the table and set the log-spaced axes from the (min,max)
  //!  temperature and density bounds.  Allocates the group-boundary array and all
  //!  per-node opacity Views (uninitialized); a reader/test then fills them through host
  //!  mirrors and deep-copies to the device.
  void Allocate(int n_groups, int n_temp, int n_dens, Real temp_min, Real temp_max,
                Real dens_min, Real dens_max) {
    ngroups = n_groups;
    ntemp = n_temp;
    ndens = n_dens;
    log_temp_min = std::log10(temp_min);
    log_dens_min = std::log10(dens_min);
    dlog_temp = (std::log10(temp_max) - log_temp_min) / static_cast<Real>(ntemp - 1);
    dlog_dens = (std::log10(dens_max) - log_dens_min) / static_cast<Real>(ndens - 1);
    inv_dlog_temp = 1.0 / dlog_temp;
    inv_dlog_dens = 1.0 / dlog_dens;
    group_bounds = DvceArray1D<Real>("opac_group_bounds", ngroups + 1);
    planck_absorb = DvceArray3D<Real>("opac_planck_absorb", ngroups, ntemp, ndens);
    planck_emiss = DvceArray3D<Real>("opac_planck_emiss", ngroups, ntemp, ndens);
    rosseland = DvceArray3D<Real>("opac_rosseland", ngroups, ntemp, ndens);
  }

  // --- axis coordinate helpers (host + device) ---
  KOKKOS_INLINE_FUNCTION Real LogTempAt(int it) const {
    return log_temp_min + it*dlog_temp;
  }
  KOKKOS_INLINE_FUNCTION Real LogDensAt(int id) const {
    return log_dens_min + id*dlog_dens;
  }
  KOKKOS_INLINE_FUNCTION Real TempAt(int it) const {
    return Kokkos::pow(10.0, LogTempAt(it));
  }
  KOKKOS_INLINE_FUNCTION Real DensAt(int id) const {
    return Kokkos::pow(10.0, LogDensAt(id));
  }
  KOKKOS_INLINE_FUNCTION Real TempMin() const { return Kokkos::pow(10.0, log_temp_min); }
  KOKKOS_INLINE_FUNCTION Real TempMax() const { return TempAt(ntemp - 1); }
  KOKKOS_INLINE_FUNCTION Real DensMin() const { return Kokkos::pow(10.0, log_dens_min); }
  KOKKOS_INLINE_FUNCTION Real DensMax() const { return DensAt(ndens - 1); }

  //! \brief Photon-energy boundary `ig` in [0,ngroups]; group `ig` is
  //!  [GroupBound(ig), GroupBound(ig+1)].
  KOKKOS_INLINE_FUNCTION Real GroupBound(int ig) const { return group_bounds(ig); }

  //--------------------------------------------------------------------------------------
  //! \fn Interp
  //! \brief Log-log bilinear interpolation of opacity `field` for group `ig` at
  //!  (log10 rho, log10 T_e).  The four corner values are interpolated in log10 and
  //!  exponentiated back (exact for power-law opacity).  The fractional index is clamped
  //!  to [0,n-1] in each direction, so out-of-range queries return the nearest in-bounds
  //!  (edge) value and never read out of bounds.
  KOKKOS_INLINE_FUNCTION
  Real Interp(const DvceArray3D<Real> &field, int ig, Real ldens, Real lT) const {
    Real fit = (lT - log_temp_min)*inv_dlog_temp;
    Real fid = (ldens - log_dens_min)*inv_dlog_dens;
    fit = Kokkos::fmin(Kokkos::fmax(fit, 0.0), static_cast<Real>(ntemp - 1));
    fid = Kokkos::fmin(Kokkos::fmax(fid, 0.0), static_cast<Real>(ndens - 1));
    int it = static_cast<int>(fit);
    int id = static_cast<int>(fid);
    if (it > ntemp - 2) { it = ntemp - 2; }
    if (id > ndens - 2) { id = ndens - 2; }
    Real wt = fit - it;
    Real wd = fid - id;
    Real f00 = Kokkos::log10(field(ig, it,   id));
    Real f01 = Kokkos::log10(field(ig, it,   id+1));
    Real f10 = Kokkos::log10(field(ig, it+1, id));
    Real f11 = Kokkos::log10(field(ig, it+1, id+1));
    Real v = (1.0 - wt)*((1.0 - wd)*f00 + wd*f01)
           + wt*((1.0 - wd)*f10 + wd*f11);
    return Kokkos::pow(10.0, v);
  }

  // --- forward lookups (device): mass opacity for group `ig` at (rho, T_e) ---
  KOKKOS_INLINE_FUNCTION
  Real PlanckAbsorption(int ig, Real rho, Real te) const {
    return Interp(planck_absorb, ig, Kokkos::log10(rho), Kokkos::log10(te));
  }
  KOKKOS_INLINE_FUNCTION
  Real PlanckEmission(int ig, Real rho, Real te) const {
    return Interp(planck_emiss, ig, Kokkos::log10(rho), Kokkos::log10(te));
  }
  KOKKOS_INLINE_FUNCTION
  Real RosselandTransport(int ig, Real rho, Real te) const {
    return Interp(rosseland, ig, Kokkos::log10(rho), Kokkos::log10(te));
  }
  //! \brief Generic lookup dispatched on OpacityKind (for callers iterating over kinds).
  KOKKOS_INLINE_FUNCTION
  Real Opacity(int kind, int ig, Real rho, Real te) const {
    const DvceArray3D<Real> &field = (kind == kPlanckAbsorption) ? planck_absorb
                                   : (kind == kPlanckEmission)   ? planck_emiss
                                                                 : rosseland;
    return Interp(field, ig, Kokkos::log10(rho), Kokkos::log10(te));
  }
};

}  // namespace opacity

#endif  // OPACITY_MULTIGROUP_OPACITY_HPP_
