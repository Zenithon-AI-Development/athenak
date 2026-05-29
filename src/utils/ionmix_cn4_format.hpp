#ifndef UTILS_IONMIX_CN4_FORMAT_HPP_
#define UTILS_IONMIX_CN4_FORMAT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_cn4_format.hpp
//! \brief Parser for the real FLASH IONMIX `.cn4` table format (issue [C1]/#118).
//!
//! The synthetic IONMIX fixtures used by the [13b]/[16a] reader unit tests are
//! whitespace-delimited ASCII; the *real* developer-supplied FLASH IONMIX tables
//! (`ionmix_cn4_combined/*.cn4`, e.g. aluminum / beryllium / deuterium) are the packed,
//! fixed-width "cn4" format that FLASH itself reads at runtime.  This header decodes that
//! packed format into a flat field stream that the EOS (`eos/ionmix_eos_reader.hpp`) and
//! opacity (`opacity/ionmix_opacity_reader.hpp`) readers consume in record order; it is
//! the shared piece both real-material readers build on.  Host-only, header-only,
//! and included only by those readers' real-material code paths, so it costs the main
//! binary nothing (same additive shape as the synthetic readers).
//!
//! The format mirrors the FLASH-center `opacplot2` reference reader/writer
//! (`opg_ionmix.py`):
//!
//!   Header (4 lines, single-line forms — true for the Al/Be/D tables):
//!     line 1:  ntemp  ndens                         (two integers, 10-char fields)
//!     line 2:  " atomic #s of gases: " z1 z2 ...     (atomic numbers, 10-char fields)
//!     line 3:  " relative fractions: " f1 f2 ...     (number fractions, 10-char fields)
//!     line 4:  ngroups                              (one integer, 12-char field)
//!
//!   Data: every value is a 12-character fixed-width field, 4 per line, written by
//!   the IONMIX `convert()` as a mantissa in `[0.1,1)` with an explicit `E` and a 2-digit
//!   exponent (e.g. `0.200000E+05` == 2.0e4; note the leading `0.`, not `2.0`).  Reading
//!   is therefore: drop the 4 header lines, concatenate the rest (stripping line
//!   terminators), and consume the stream in 12-char chunks.  Records are in this order
//!   (every value 12-char-packed; 2-D arrays are density-major with **temperature varying
//!   fastest**, i.e. flat index = id*ntemp + it; 3-D opacity is group-major then density
//!   then temperature, flat index = ((ig*ndens + id)*ntemp + it)):
//!     temps[ntemp]                         (electron temperature grid, eV)
//!     numdens[ndens]                       (ion *number* density grid, cm^-3)
//!     zbar, dzdt,                          (each ndens*ntemp)
//!     pion, pele, dpidt, dpedt,
//!     eion, eele, cvion, cvele, deidn, deedn
//!     opac_bounds[ngroups+1]               (photon-energy group boundaries, eV)
//!     rosseland, planck_absorb, planck_emiss   (each ngroups*ndens*ntemp)
//!
//! Units are left exactly as stored (IONMIX SI-ish: energies/pressures in Joules-based
//! units, opacities in cm^2/g, temperatures in eV, density as number density) -- this
//! header is a faithful loader; downstream code-unit normalization is a separate concern.

#include <cmath>      // (unused here, kept for parity with the readers)
#include <cstddef>    // std::size_t
#include <cstdlib>    // std::exit, EXIT_FAILURE
#include <fstream>    // std::ifstream
#include <iostream>
#include <sstream>    // std::istringstream
#include <string>
#include <vector>

#include "athena.hpp"

namespace ionmix_cn4 {

//----------------------------------------------------------------------------------------
//! \fn Cn4Fatal
//! \brief Print a clear reader error and abort (consistent with AthenaK's FATAL style).
inline void Cn4Fatal(const std::string &fname, const std::string &msg) {
  std::cout << "### FATAL ERROR in IONMIX cn4 reader" << std::endl
            << "  file: " << fname << std::endl
            << "  " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn ParseField
//! \brief Parse one 12-character IONMIX field to a Real.  The normal form carries an
//!  explicit `E` (`0.200000E+05`) which `std::stod` handles directly.  As a robustness
//!  measure we also reconstruct the rare FORTRAN 3-digit-exponent form that drops the `E`
//!  (`0.100000+100`) by re-inserting an `E` before the exponent sign.  (Al/Be/D tables
//!  never need this, but it makes the loader total.)
inline Real ParseField(const std::string &fname, const std::string &fld) {
  std::string s = fld;
  if (s.find('E') == std::string::npos && s.find('e') == std::string::npos) {
    // No exponent marker: find the exponent sign (a +/- after the leading char) and
    // splice an 'E' in front of it so std::stod sees a well-formed exponent.
    for (std::size_t i = 1; i < s.size(); ++i) {
      if (s[i] == '+' || s[i] == '-') { s.insert(i, "E"); break; }
    }
  }
  try {
    std::size_t consumed = 0;
    Real v = static_cast<Real>(std::stod(s, &consumed));
    if (consumed == 0) { throw std::invalid_argument("empty"); }
    return v;
  } catch (...) {
    Cn4Fatal(fname, "could not parse numeric field \"" + fld + "\".");
  }
  return 0.0;  // unreachable (Cn4Fatal exits)
}

//----------------------------------------------------------------------------------------
//! \struct Cn4Reader
//! \brief Parses a real FLASH IONMIX `.cn4` file into its header scalars plus the flat,
//!  in-order stream of 12-char data fields, then hands them out sequentially via Next() /
//!  ReadVec() / Skip().  The EOS and opacity readers walk the stream in record order.
struct Cn4Reader {
  std::string fname;
  int ntemp = 0, ndens = 0, ngroups = 0;
  std::vector<int> atomic_numbers;     //!< per-element atomic numbers (header line 2)
  std::vector<Real> fractions;         //!< per-element number fractions (header line 3)
  std::vector<Real> fields;            //!< all data fields, in file order
  std::size_t cursor = 0;              //!< next unread field

  explicit Cn4Reader(const std::string &file) : fname(file) {
    std::ifstream fin(fname);
    if (!fin) { Cn4Fatal(fname, "could not open IONMIX cn4 file for reading."); }

    // --- header: 4 lines (single-line forms, true for the Al/Be/D tables) ---
    std::string l_dims, l_atom, l_frac, l_grp;
    if (!std::getline(fin, l_dims) || !std::getline(fin, l_atom)
        || !std::getline(fin, l_frac) || !std::getline(fin, l_grp)) {
      Cn4Fatal(fname, "file too short to contain the 4-line IONMIX cn4 header.");
    }
    {
      std::istringstream hs(l_dims);
      if (!(hs >> ntemp >> ndens)) {
        Cn4Fatal(fname, "failed to read header dims (expected: ntemp ndens).");
      }
    }
    // atomic numbers / fractions follow a "label:" prefix; parse tokens after the colon.
    auto after_colon = [](const std::string &line) -> std::string {
      std::size_t c = line.find(':');
      return (c == std::string::npos) ? line : line.substr(c + 1);
    };
    { std::istringstream as(after_colon(l_atom)); int z;
      while (as >> z) { atomic_numbers.push_back(z); } }
    { std::istringstream fs(after_colon(l_frac)); Real f;
      while (fs >> f) { fractions.push_back(f); } }
    { std::istringstream gs(l_grp);
      if (!(gs >> ngroups)) {
        Cn4Fatal(fname, "failed to read ngroups (header line 4).");
      }
    }
    if (ntemp < 2 || ndens < 2 || ngroups < 1) {
      Cn4Fatal(fname, "invalid dimensions (need ntemp>=2, ndens>=2, ngroups>=1).");
    }

    // --- data: concatenate the remaining lines (strip terminators/trailing space) and
    //     split into fixed 12-char fields ---
    std::string packed, line;
    while (std::getline(fin, line)) {
      std::size_t end = line.find_last_not_of(" \t\r\n");
      if (end != std::string::npos) { packed += line.substr(0, end + 1); }
    }
    if (packed.size() % 12 != 0) {
      Cn4Fatal(fname, "packed data length (" + std::to_string(packed.size())
               + " chars) is not a multiple of the 12-char IONMIX field width — the "
                 "file is not in the fixed-width cn4 format this reader expects.");
    }
    const std::size_t nfields = packed.size() / 12;
    fields.reserve(nfields);
    for (std::size_t i = 0; i < nfields; ++i) {
      fields.push_back(ParseField(fname, packed.substr(i*12, 12)));
    }
  }

  //! \brief Number of data fields still unread.
  std::size_t Remaining() const { return fields.size() - cursor; }

  //! \brief Consume and return the next field; FATAL on overrun.
  Real Next() {
    if (cursor >= fields.size()) {
      Cn4Fatal(fname, "unexpected end of cn4 data stream (more records expected than the "
                      "file holds — dimensions/record-order mismatch).");
    }
    return fields[cursor++];
  }

  //! \brief Read `n` fields into `v` (resized to `n`).
  void ReadVec(std::vector<Real> &v, int n) {
    v.resize(n);
    for (int i = 0; i < n; ++i) { v[i] = Next(); }
  }

  //! \brief Skip `n` fields (records this reader does not need, e.g. dzdt/dpidt/deidn).
  void Skip(int n) { for (int i = 0; i < n; ++i) { (void)Next(); } }

  //! \brief FATAL unless every field was consumed (guards against a record-order bug
  //!  silently leaving trailing data or over-reading).
  void RequireFullyConsumed() {
    if (cursor != fields.size()) {
      Cn4Fatal(fname, "cn4 record-order mismatch: consumed " + std::to_string(cursor)
               + " of " + std::to_string(fields.size()) + " data fields.");
    }
  }
};

}  // namespace ionmix_cn4

#endif  // UTILS_IONMIX_CN4_FORMAT_HPP_
