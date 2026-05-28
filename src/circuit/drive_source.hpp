#ifndef CIRCUIT_DRIVE_SOURCE_HPP_
#define CIRCUIT_DRIVE_SOURCE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file drive_source.hpp
//! \brief Pluggable "drive source" behind the cylindrical B_phi outer-radial boundary
//!        (ADR-0005, issue [9a]/#21).
//!
//! Z-pinch / MagLIF is circuit-driven: a pulsed-power source delivers a load current
//! I(t) that sets the azimuthal field at the outer radius, B_phi(R)=mu0*I(t)/(2*pi*R).
//! ADR-0005 puts the origin of I(t) behind a *pluggable* drive source with three modes:
//!   (A) prescribed I(t)  -- tabulated/analytic waveform, no feedback (this issue);
//!   (B) prescribed voltage + fixed RLC (a later issue);
//!   (C) prescribed voltage + coupled circuit with load feedback (the default target,
//!       Faraday load voltage via a global reduction -- issues #31/#34).
//! Only mode A (prescribed current) is implemented here; the struct/enum leave room for
//! B and C without touching the boundary code below.
//!
//! This is the deep-module home for the circuit drive: the host-side waveform evaluation
//! (DriveSource::Current) and the device-inline boundary formulas (DrivenBphi /
//! NoCurrentBphi).  The reusable user-BC ghost fill that applies them at the outer-radial
//! boundary is the sibling circuit/drive_bphi_bc.hpp (which pulls in the Mesh/MHD types),
//! so a pgen wiring the boundary (#29 driven pinch, #32 MagLIF) only has to construct a
//! DriveSource and enroll a one-line user_bcs_func.  Because no file in the main build
//! includes either header (only the pgens that opt in via -D PROBLEM=<name> do), they
//! cost the main binary nothing and cannot perturb existing behaviour.
//!
//! Units: I(t), B_phi and mu0 are in code units.  AthenaK uses Heaviside-Lorentz units
//! (magnetic pressure B^2/2, Ampere's law curl(B)=J with no mu0), so the enclosed-current
//! relation is B_phi*(2*pi*r)=I and mu0=1 in code units -- the default.  The parameter is
//! exposed so a user running a different non-dimensionalization can match their system.

#include <algorithm>  // std::lower_bound
#include <cmath>      // std::sin
#include <cstdlib>    // exit, EXIT_FAILURE
#include <fstream>    // std::ifstream
#include <iostream>
#include <sstream>    // std::istringstream
#include <string>
#include <vector>

#include "athena.hpp"

namespace circuit {

//! 2*pi as a Real constant (avoids the host-only M_PI macro inside a device function).
constexpr Real kTwoPi = 6.2831853071795864769;

//----------------------------------------------------------------------------------------
//! \enum DriveMode
//! \brief Origin of the boundary current (ADR-0005).  Only prescribed_current (mode A) is
//!        implemented now; voltage_rlc (B) and coupled_circuit (C) are later issues.
enum class DriveMode { prescribed_current, voltage_rlc, coupled_circuit };

//----------------------------------------------------------------------------------------
//! \enum CurrentWaveform
//! \brief Shape of the prescribed I(t) for mode A: an analytic shape or a tabulated
//!        (t, I) waveform read from a file (e.g. a replayed measured Z-machine shot).
enum class CurrentWaveform { constant, linear_ramp, sin_squared, tabulated };

//----------------------------------------------------------------------------------------
//! \fn Real DrivenBphi
//! \brief The driven azimuthal field a load current I deposits at radius r:
//!        B_phi = mu0 * I / (2*pi*r)  (Ampere's law for an enclosed current I).
//!        Device-inline so the boundary kernel calls it directly.
KOKKOS_INLINE_FUNCTION
Real DrivenBphi(Real current, Real radius, Real mu0) {
  return mu0*current/(kTwoPi*radius);
}

//----------------------------------------------------------------------------------------
//! \fn Real NoCurrentBphi
//! \brief The "nocurrent" vacuum extrapolation outside the current-carrying load: with no
//!        enclosed current d_r(r*B_phi)=0, so r*B_phi is constant and B_phi falls as 1/r.
//!        Given the reference (r_ref, B_phi(r_ref)) just inside the boundary, the ghost
//!        value at radius r_ghost is B_phi(r_ref)*r_ref/r_ghost.  Device-inline.
KOKKOS_INLINE_FUNCTION
Real NoCurrentBphi(Real bphi_ref, Real r_ref, Real r_ghost) {
  return bphi_ref*r_ref/r_ghost;
}

//----------------------------------------------------------------------------------------
//! \class DriveSource
//! \brief Host-side prescribed-current drive (mode A).  Current(t) returns the load
//!        current I(t) from an analytic waveform or a tabulated (t, I) curve.  The value
//!        is a single host scalar evaluated once per step and captured into the boundary
//!        kernel, so this object never needs to be device-copyable.
class DriveSource {
 public:
  DriveMode mode = DriveMode::prescribed_current;
  CurrentWaveform waveform = CurrentWaveform::constant;
  Real i0 = 0.0;        //!< constant / peak current amplitude [code units]
  Real t_rise = 1.0;    //!< ramp time (linear_ramp) or pulse duration (sin_squared)
  Real mu0 = 1.0;       //!< code-unit permeability (Heaviside-Lorentz: 1)
  std::vector<Real> t_tab;  //!< tabulated time nodes (strictly ascending)
  std::vector<Real> i_tab;  //!< tabulated current values at t_tab

  //! \brief Evaluate the prescribed load current I(t) (host).
  Real Current(Real t) const {
    switch (waveform) {
      case CurrentWaveform::constant:
        return i0;
      case CurrentWaveform::linear_ramp:
        // 0 -> i0 linearly over [0, t_rise], then held at i0.
        if (t <= 0.0) return 0.0;
        if (t >= t_rise) return i0;
        return i0*(t/t_rise);
      case CurrentWaveform::sin_squared: {
        // A smooth sin^2 pulse over [0, t_rise]: 0 at the ends, peak i0 at t_rise/2.
        if (t <= 0.0 || t >= t_rise) return 0.0;
        Real s = std::sin(kTwoPi*0.5*t/t_rise);
        return i0*s*s;
      }
      case CurrentWaveform::tabulated:
        return InterpTable(t);
    }
    return 0.0;
  }

  //! \brief Piecewise-linear interpolation of the tabulated (t, I) waveform; the end
  //!        values are held (clamped) outside [t_tab.front(), t_tab.back()].
  Real InterpTable(Real t) const {
    const int n = static_cast<int>(t_tab.size());
    if (n == 0) return 0.0;
    if (t <= t_tab[0]) return i_tab[0];
    if (t >= t_tab[n-1]) return i_tab[n-1];
    // first node strictly greater than t (the bracket is [hi-1, hi]); t_tab is ascending.
    int hi = static_cast<int>(
        std::upper_bound(t_tab.begin(), t_tab.end(), t) - t_tab.begin());
    int lo = hi - 1;
    Real w = (t - t_tab[lo])/(t_tab[hi] - t_tab[lo]);
    return i_tab[lo] + w*(i_tab[hi] - i_tab[lo]);
  }
};

//----------------------------------------------------------------------------------------
//! \fn DriveSourceFatal
//! \brief Print a clear reader error and abort (consistent with AthenaK's FATAL style).
inline void DriveSourceFatal(const std::string &fname, const std::string &msg) {
  std::cout << "### FATAL ERROR in drive-source current waveform reader" << std::endl
            << "  file: " << fname << std::endl
            << "  " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn ReadCurrentWaveform
//! \brief Parse a tabulated current waveform file into `ds` (host I/O).
//!
//! File layout (whitespace-delimited ASCII; lines whose first non-blank character is `#`
//! are comments and are skipped, so a waveform file may be self-documenting):
//!   npts                          (>= 2, the number of (t, I) samples)
//!   t_0   I_0                      (time and current, code units)
//!   t_1   I_1
//!   ...                           (npts rows; time strictly ascending)
//! Numeric tokens are consumed in order, so the per-line grouping is irrelevant.  After
//! this call `ds.t_tab`/`ds.i_tab` are populated and `ds.waveform == tabulated`.
inline void ReadCurrentWaveform(const std::string &fname, DriveSource &ds) {
  std::ifstream fin(fname);
  if (!fin) {
    DriveSourceFatal(fname, "could not open current waveform file for reading.");
  }
  std::ostringstream filtered;
  std::string line;
  while (std::getline(fin, line)) {
    std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') { continue; }
    filtered << line << ' ';
  }
  std::istringstream toks(filtered.str());

  int npts = 0;
  if (!(toks >> npts)) {
    DriveSourceFatal(fname, "failed to read header (expected: npts).");
  }
  if (npts < 2) {
    DriveSourceFatal(fname, "invalid waveform (need npts >= 2).");
  }

  ds.t_tab.assign(npts, 0.0);
  ds.i_tab.assign(npts, 0.0);
  for (int p = 0; p < npts; ++p) {
    if (!(toks >> ds.t_tab[p] >> ds.i_tab[p])) {
      DriveSourceFatal(fname, "unexpected end of file while reading (t, I) samples.");
    }
    if (p > 0 && !(ds.t_tab[p] > ds.t_tab[p-1])) {
      DriveSourceFatal(fname, "time column must be strictly ascending.");
    }
  }
  ds.waveform = CurrentWaveform::tabulated;
}

//----------------------------------------------------------------------------------------
//! \fn ParseWaveform
//! \brief Map an input string to a CurrentWaveform (FATALs on an unknown name).
inline CurrentWaveform ParseWaveform(const std::string &name) {
  if (name == "constant")     return CurrentWaveform::constant;
  if (name == "linear_ramp")  return CurrentWaveform::linear_ramp;
  if (name == "sin_squared")  return CurrentWaveform::sin_squared;
  if (name == "tabulated")    return CurrentWaveform::tabulated;
  std::cout << "### FATAL ERROR: unknown drive-source current_waveform '" << name
            << "' (expected constant|linear_ramp|sin_squared|tabulated)." << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace circuit

#endif  // CIRCUIT_DRIVE_SOURCE_HPP_
