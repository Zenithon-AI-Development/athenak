#ifndef EOS_EOS_TABLE_3T_HPP_
#define EOS_EOS_TABLE_3T_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos_table_3t.hpp
//! \brief Common internal representation for a tabulated 3T EOS + per-cell e->T inversion
//!        (ADR-0002/0007, issue [13a]/#6).
//!
//! The 3T energy formulation (ADR-0002) evolves separate electron and ion internal
//! energies and *derives* temperatures by inverting the tabulated EOS per cell.  This
//! header provides the shared, format-agnostic table representation those readers
//! (IONMIX #10, SESAME #11) populate, plus the device-side lookups and the monotonic
//! `e -> T` inversion (bracketed bisection safeguarded by Newton using the table heat
//! capacity).  Concrete file readers live in separate slices; this module never touches
//! the disk -- a reader (or a test) allocates the table, fills the per-node arrays via
//! host mirrors, and copies them to the device.
//!
//! Layout and conventions:
//!   - Two log-spaced axes: log10(T) [temperature index `it` in [0,ntemp)] and
//!     log10(rho) [density index `ir` in [0,nrho)].  Every field is a `DvceArray2D`
//!     of shape (ntemp, nrho) -- temperature-major, matching the IONMIX nodal layout.
//!   - Internal energies and heat capacities are *specific* (per unit mass), as IONMIX
//!     stores them; the caller divides an energy *density* by rho before inverting.
//!   - The representation is unit-agnostic: a reader sets the units when it fills the
//!     table.  The only physics assumption is `e(T)` monotone increasing at fixed rho
//!     (guaranteed by `c_v > 0`), which makes the inversion well-posed inside bounds.
//!   - Positive-definite fields (e, p, c_v) are interpolated bilinearly on log10(value)
//!     in (log10 rho, log10 T); the mean ionization `Zbar` (which may be 0) is
//!     interpolated on its raw value.  Log-log interpolation reproduces any power-law /
//!     ideal-gas relation exactly, on and off grid.
//!   - Out-of-range queries are clamped to the table bounds (floors/ceilings): the
//!     interpolation clamps the (rho,T) fractional index, and the inversion clamps the
//!     target energy into `[e(Tmin), e(Tmax)]`.  No out-of-bounds memory access occurs.
//!
//! The struct holds Kokkos Views and PODs, so it is copied by value into a `par_for`
//! kernel exactly like AthenaK's `EOS_Data` (shallow View copy, device-valid pointers);
//! all lookups are `const KOKKOS_INLINE_FUNCTION`.

#include <cmath>       // std::log10

#include "athena.hpp"  // Real, Kokkos, DvceArray2D

namespace eos_table_3t {

//----------------------------------------------------------------------------------------
//! \enum Species
//! \brief Selects which species' table a lookup/inversion operates on.
enum Species { kElectron = 0, kIon = 1 };

//----------------------------------------------------------------------------------------
//! \struct EosTable3T
//! \brief Device-friendly tabulated 3T EOS: log-spaced (rho,T) grid + per-node species
//!        energies, pressures, mean ionization and heat capacities, with device lookups
//!        and a monotonic e->T inversion.
struct EosTable3T {
  // --- axis metadata (scalars; trivially copied to the device) ---
  int ntemp = 0;            //!< number of temperature nodes (>= 2)
  int nrho = 0;             //!< number of density nodes (>= 2)
  Real log_temp_min = 0.0;  //!< log10 of minimum tabulated temperature
  Real log_rho_min = 0.0;   //!< log10 of minimum tabulated density
  Real dlog_temp = 0.0;     //!< log10(T) node spacing
  Real dlog_rho = 0.0;      //!< log10(rho) node spacing
  Real inv_dlog_temp = 0.0; //!< 1 / dlog_temp (precomputed for the hot path)
  Real inv_dlog_rho = 0.0;  //!< 1 / dlog_rho

  // --- per-node table data, shape (ntemp, nrho) ---
  DvceArray2D<Real> e_ele;   //!< specific electron internal energy e_e(T,rho)
  DvceArray2D<Real> e_ion;   //!< specific ion internal energy e_i(T,rho)
  DvceArray2D<Real> p_ele;   //!< electron pressure p_e(T,rho)
  DvceArray2D<Real> p_ion;   //!< ion pressure p_i(T,rho)
  DvceArray2D<Real> zbar;    //!< mean ionization Zbar(T,rho)
  DvceArray2D<Real> cv_ele;  //!< specific electron heat capacity c_v,e = de_e/dT
  DvceArray2D<Real> cv_ion;  //!< specific ion heat capacity c_v,i = de_i/dT

  // Inversion controls.
  static constexpr int   kMaxIter = 100;    //!< max bisection/Newton iterations
  static constexpr Real  kTempRTol = 1.0e-12; //!< relative-in-T convergence tolerance

  //--------------------------------------------------------------------------------------
  //! \fn Allocate
  //! \brief Host-side: size the table and set the log-spaced axes from the (min,max)
  //!  temperature and density bounds.  Allocates all per-node Views (uninitialized);
  //!  a reader/test then fills them through host mirrors and deep-copies to the device.
  void Allocate(int n_temp, int n_rho, Real temp_min, Real temp_max,
                Real rho_min, Real rho_max) {
    ntemp = n_temp;
    nrho = n_rho;
    log_temp_min = std::log10(temp_min);
    log_rho_min = std::log10(rho_min);
    dlog_temp = (std::log10(temp_max) - log_temp_min) / static_cast<Real>(ntemp - 1);
    dlog_rho = (std::log10(rho_max) - log_rho_min) / static_cast<Real>(nrho - 1);
    inv_dlog_temp = 1.0 / dlog_temp;
    inv_dlog_rho = 1.0 / dlog_rho;
    e_ele = DvceArray2D<Real>("eos3t_e_ele", ntemp, nrho);
    e_ion = DvceArray2D<Real>("eos3t_e_ion", ntemp, nrho);
    p_ele = DvceArray2D<Real>("eos3t_p_ele", ntemp, nrho);
    p_ion = DvceArray2D<Real>("eos3t_p_ion", ntemp, nrho);
    zbar = DvceArray2D<Real>("eos3t_zbar", ntemp, nrho);
    cv_ele = DvceArray2D<Real>("eos3t_cv_ele", ntemp, nrho);
    cv_ion = DvceArray2D<Real>("eos3t_cv_ion", ntemp, nrho);
  }

  // --- axis coordinate helpers (host + device) ---
  KOKKOS_INLINE_FUNCTION Real LogTempAt(int it) const {
    return log_temp_min + it*dlog_temp;
  }
  KOKKOS_INLINE_FUNCTION Real LogRhoAt(int ir) const {
    return log_rho_min + ir*dlog_rho;
  }
  KOKKOS_INLINE_FUNCTION Real TempAt(int it) const {
    return Kokkos::pow(10.0, LogTempAt(it));
  }
  KOKKOS_INLINE_FUNCTION Real RhoAt(int ir) const {
    return Kokkos::pow(10.0, LogRhoAt(ir));
  }
  KOKKOS_INLINE_FUNCTION Real TempMin() const { return Kokkos::pow(10.0, log_temp_min); }
  KOKKOS_INLINE_FUNCTION Real TempMax() const { return TempAt(ntemp - 1); }
  KOKKOS_INLINE_FUNCTION Real RhoMin() const { return Kokkos::pow(10.0, log_rho_min); }
  KOKKOS_INLINE_FUNCTION Real RhoMax() const { return RhoAt(nrho - 1); }

  //--------------------------------------------------------------------------------------
  //! \fn Interp
  //! \brief Bilinear interpolation of `field` at (log10 rho, log10 T).  When `log_value`
  //!  is true the four corner values are interpolated in log10 and exponentiated back
  //!  (exact for power-law data).  The fractional index is clamped to [0,n-1] in each
  //!  direction, so out-of-range queries return the nearest in-bounds (edge) value and
  //!  never read out of bounds.
  KOKKOS_INLINE_FUNCTION
  Real Interp(const DvceArray2D<Real> &field, Real lrho, Real lT, bool log_value) const {
    Real fit = (lT - log_temp_min)*inv_dlog_temp;
    Real fir = (lrho - log_rho_min)*inv_dlog_rho;
    fit = Kokkos::fmin(Kokkos::fmax(fit, 0.0), static_cast<Real>(ntemp - 1));
    fir = Kokkos::fmin(Kokkos::fmax(fir, 0.0), static_cast<Real>(nrho - 1));
    int it = static_cast<int>(fit);
    int ir = static_cast<int>(fir);
    if (it > ntemp - 2) { it = ntemp - 2; }
    if (ir > nrho - 2) { ir = nrho - 2; }
    Real wt = fit - it;
    Real wr = fir - ir;
    Real f00 = field(it,   ir);
    Real f01 = field(it,   ir+1);
    Real f10 = field(it+1, ir);
    Real f11 = field(it+1, ir+1);
    if (log_value) {
      f00 = Kokkos::log10(f00); f01 = Kokkos::log10(f01);
      f10 = Kokkos::log10(f10); f11 = Kokkos::log10(f11);
    }
    Real v = (1.0 - wt)*((1.0 - wr)*f00 + wr*f01)
           + wt*((1.0 - wr)*f10 + wr*f11);
    return log_value ? Kokkos::pow(10.0, v) : v;
  }

  // --- forward lookups (device): value at (rho, T) for the given species ---
  KOKKOS_INLINE_FUNCTION
  Real Energy(int s, Real rho, Real temp) const {
    return Interp((s == kElectron) ? e_ele : e_ion,
                  Kokkos::log10(rho), Kokkos::log10(temp), true);
  }
  KOKKOS_INLINE_FUNCTION
  Real Pressure(int s, Real rho, Real temp) const {
    return Interp((s == kElectron) ? p_ele : p_ion,
                  Kokkos::log10(rho), Kokkos::log10(temp), true);
  }
  KOKKOS_INLINE_FUNCTION
  Real HeatCapacity(int s, Real rho, Real temp) const {
    return Interp((s == kElectron) ? cv_ele : cv_ion,
                  Kokkos::log10(rho), Kokkos::log10(temp), true);
  }
  KOKKOS_INLINE_FUNCTION
  Real Zbar(Real rho, Real temp) const {
    return Interp(zbar, Kokkos::log10(rho), Kokkos::log10(temp), false);
  }

  // Convenience named accessors (electron uses T_e, ion uses T_i).
  KOKKOS_INLINE_FUNCTION Real EnergyEle(Real rho, Real te) const {
    return Energy(kElectron, rho, te);
  }
  KOKKOS_INLINE_FUNCTION Real EnergyIon(Real rho, Real ti) const {
    return Energy(kIon, rho, ti);
  }
  KOKKOS_INLINE_FUNCTION Real PressureEle(Real rho, Real te) const {
    return Pressure(kElectron, rho, te);
  }
  KOKKOS_INLINE_FUNCTION Real PressureIon(Real rho, Real ti) const {
    return Pressure(kIon, rho, ti);
  }
  KOKKOS_INLINE_FUNCTION Real CvEle(Real rho, Real te) const {
    return HeatCapacity(kElectron, rho, te);
  }
  KOKKOS_INLINE_FUNCTION Real CvIon(Real rho, Real ti) const {
    return HeatCapacity(kIon, rho, ti);
  }

  //--------------------------------------------------------------------------------------
  //! \fn InvertTemp
  //! \brief Per-cell monotonic e->T inversion for species `s`: solve
  //!  `Energy(s,rho,T) = e_target` for T.  Because `e` is monotone increasing in T
  //!  (c_v>0), the root is bracketed by the table temperature range; we use a Newton
  //!  step (slope = table heat capacity c_v = de/dT) safeguarded by bisection, which
  //!  converges robustly within bounds.  `e_target` is clamped into
  //!  `[e(Tmin), e(Tmax)]` (table-edge clamping) so an out-of-range energy returns the
  //!  corresponding edge temperature rather than diverging.
  KOKKOS_INLINE_FUNCTION
  Real InvertTemp(int s, Real rho, Real e_target) const {
    Real xl = TempMin();   // f(xl) = e(xl) - e_target <= 0 after clamping
    Real xh = TempMax();   // f(xh) >= 0 after clamping
    Real e_lo = Energy(s, rho, xl);
    Real e_hi = Energy(s, rho, xh);
    if (e_target <= e_lo) { return xl; }
    if (e_target >= e_hi) { return xh; }

    // Safeguarded Newton-bisection (cf. Numerical Recipes "rtsafe").  xl brackets the
    // low (f<0) side, xh the high (f>0) side.
    Real rts = 0.5*(xl + xh);
    Real dxold = xh - xl;
    Real dx = dxold;
    Real f = Energy(s, rho, rts) - e_target;
    Real df = HeatCapacity(s, rho, rts);   // de/dT > 0
    for (int it = 0; it < kMaxIter; ++it) {
      // Take a bisection step if Newton would leave the bracket or is converging slowly.
      if ((((rts - xh)*df - f)*((rts - xl)*df - f) > 0.0)
          || (Kokkos::fabs(2.0*f) > Kokkos::fabs(dxold*df))) {
        dxold = dx;
        dx = 0.5*(xh - xl);
        rts = xl + dx;
      } else {
        dxold = dx;
        dx = f/df;
        rts = rts - dx;
      }
      if (Kokkos::fabs(dx) < kTempRTol*rts) { return rts; }
      f = Energy(s, rho, rts) - e_target;
      df = HeatCapacity(s, rho, rts);
      if (f < 0.0) { xl = rts; } else { xh = rts; }
    }
    return rts;
  }

  // Named inversions per the issue interface: T_e from e_ele, T_i from e_ion.
  KOKKOS_INLINE_FUNCTION Real Te(Real rho, Real e_ele_in) const {
    return InvertTemp(kElectron, rho, e_ele_in);
  }
  KOKKOS_INLINE_FUNCTION Real Ti(Real rho, Real e_ion_in) const {
    return InvertTemp(kIon, rho, e_ion_in);
  }
};

}  // namespace eos_table_3t

#endif  // EOS_EOS_TABLE_3T_HPP_
