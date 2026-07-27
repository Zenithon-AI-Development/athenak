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

  // --- degeneracy-pressure floor (#209; opt-in via SetDegeneracyFloor) ---
  // P_ele >= deg_k * max(0, rho^(5/3) - deg_rho053), all in CODE units.  Zero deg_k
  // (the default) disables the floor and keeps every closure byte-identical.
  Real deg_k = 0.0;       //!< K = (2/5)(hbar^2/2m_e)(3pi^2)^(2/3)(Z* rho_u/m_i)^(5/3)
                          //!<     / (rho_u v_u^2)
  Real deg_rho053 = 0.0;  //!< rho0^(5/3): solid reference where the floor vanishes

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

  //--------------------------------------------------------------------------------------
  //! \fn ScaleToCodeUnits
  //! \brief Host-side: rescale a table loaded in physical CGS into the AthenaK code-unit
  //!  system once at load (ADR-0010, the EOS-table unit boundary).  The density axis is
  //!  relabelled `rho_phys -> rho_phys/density_cgs` (a constant log10 shift preserving
  //!  the log-uniform grid), the specific energies and heat capacities are divided by
  //!  `velocity_cgs^2` (specific energy has units velocity^2), and the pressures by
  //!  `density_cgs*velocity_cgs^2` (the code pressure unit rho*v^2).  The TEMPERATURE
  //!  axis stays eV (T is a derived diagnostic, not an evolved conserved variable), and
  //!  mean ionization `Zbar` is dimensionless -- both unchanged.  After this call every
  //!  forward lookup / `e->T` inversion is natively code-unit (rho in code density, e/p
  //!  in code units, T in eV), so the solver `ConsToPrim2T` and the pgen initial
  //!  conditions share one code-unit-native table.  Identity for `density_cgs ==
  //!  velocity_cgs == 1` (the default), so a table consumed in physical units is
  //!  unchanged.  Host-only (operates on host mirrors, then deep-copies back); call once
  //!  after a reader fills the table, before it is captured into a device kernel.
  void ScaleToCodeUnits(Real density_cgs, Real velocity_cgs) {
    const Real inv_e = 1.0/(velocity_cgs*velocity_cgs);          // specific energy ~ v^2
    const Real inv_p = 1.0/(density_cgs*velocity_cgs*velocity_cgs);  // pressure ~ rho*v^2
    // density axis relabel: a constant divide is a log10 shift of the axis minimum only
    // (the spacing is preserved, so the grid stays log-uniform).
    log_rho_min -= std::log10(density_cgs);
    ScaleField(e_ele,  inv_e);
    ScaleField(e_ion,  inv_e);
    ScaleField(cv_ele, inv_e);   // c_v = de/dT, T in eV unchanged -> scales like energy
    ScaleField(cv_ion, inv_e);
    ScaleField(p_ele,  inv_p);
    ScaleField(p_ion,  inv_p);
    // zbar (dimensionless) and the temperature axis (eV) are unchanged.
  }

  //--------------------------------------------------------------------------------------
  //! \fn SetDegeneracyFloor
  //! \brief Host-side: enable the zero-temperature Fermi degeneracy-pressure floor on
  //!  the ELECTRON pressure (#209).  The IONMIX table is ideal-ion nkT with Zbar ~ 0
  //!  below ~1 eV and its density axis edge-clamps (dP/drho = 0 off-table), so a
  //!  magnetically driven cold shell is pressureless and its peak density diverges with
  //!  resolution.  The floor
  //!      P_deg(rho) = K * max(0, rho^(5/3) - rho0^(5/3)),
  //!      K = (2/5)(hbar^2/2 m_e)(3 pi^2)^(2/3) (Z* rho_u/m_ion)^(5/3) / (rho_u v_u^2)
  //!  is the ideal Fermi-gas pressure of Z* electrons per ion (Z* = COLD/valence
  //!  ionization -- the table's own near-zero cold Zbar would neuter it), referenced to
  //!  zero at the solid density rho0 (cold-curve binding cancels the Fermi pressure at
  //!  solid, so the quiescent pre-drive liner feels nothing).  rho^(5/3) keeps rising
  //!  through and beyond the table edge, restoring dP/drho > 0 everywhere.  Pressure-
  //!  only (the energy tables and the e->T inversion are untouched): compression work
  //!  against the floor lands in the gas energy through the ordinary conservative
  //!  update.  Call any time after the reader fills the table (independent of
  //!  ScaleToCodeUnits; the unit conversion is folded into K here).
  //!  \param zstar         cold (valence) ionization used for n_e = Z* rho/m_ion
  //!  \param mion_g        ion mass [g]
  //!  \param density_cgs   code->CGS density conversion (the <units> value)
  //!  \param velocity_cgs  code->CGS velocity conversion (the <units> value)
  //!  \param rho0_code     solid reference density [code] where the floor is zero
  void SetDegeneracyFloor(Real zstar, Real mion_g, Real density_cgs, Real velocity_cgs,
                          Real rho0_code) {
    const Real hbar = 1.054571817e-27;   // [erg s]
    const Real m_e  = 9.1093837015e-28;  // [g]
    const Real pi   = 3.14159265358979323846;
    // (2/5)(hbar^2/2m_e)(3pi^2)^(2/3): ideal Fermi P = C * n_e^(5/3)  [erg/cc, n_e cm^-3]
    const Real c_fermi = 0.4*(hbar*hbar/(2.0*m_e))*std::pow(3.0*pi*pi, 2.0/3.0);
    deg_k = c_fermi*std::pow(zstar*density_cgs/mion_g, 5.0/3.0)
            /(density_cgs*velocity_cgs*velocity_cgs);
    deg_rho053 = std::pow(rho0_code, 5.0/3.0);
  }

  //--------------------------------------------------------------------------------------
  //! \fn DegeneracyPressureFloor
  //! \brief Device: the #209 electron-pressure floor at a code density (0 when disabled
  //!  or at/below the solid reference).  Applied in ConsToPrim2T as
  //!  p_ele = max(p_ele_table, floor).
  KOKKOS_INLINE_FUNCTION
  Real DegeneracyPressureFloor(Real rho) const {
    if (!(deg_k > 0.0)) { return 0.0; }
    Real excess = Kokkos::pow(rho, 5.0/3.0) - deg_rho053;
    return (excess > 0.0) ? deg_k*excess : 0.0;
  }

  //--------------------------------------------------------------------------------------
  //! \fn ScaleField
  //! \brief Host helper for ScaleToCodeUnits: multiply every node of a (ntemp,nrho) field
  //!  by a constant, through a host mirror, copying back to the device.  Pure host code
  //!  (no device kernel) so it is ODR-safe in this multiply-included header.
  void ScaleField(DvceArray2D<Real> &field, Real factor) {
    auto h = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(h, field);
    for (int it = 0; it < ntemp; ++it) {
      for (int ir = 0; ir < nrho; ++ir) { h(it, ir) *= factor; }
    }
    Kokkos::deep_copy(field, h);
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
