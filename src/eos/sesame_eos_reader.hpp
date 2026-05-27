#ifndef EOS_SESAME_EOS_READER_HPP_
#define EOS_SESAME_EOS_READER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sesame_eos_reader.hpp
//! \brief SESAME 3T-native EOS file reader populating the common 3T EOS representation
//!        (ADR-0007, issue [13c]/#11).
//!
//! Parses a SESAME-style ASCII EOS table into an `eos_table_3t::EosTable3T`
//! (eos_table_3t.hpp).  This is the SESAME plug-in for the shared representation; the
//! IONMIX reader (#10) is its sibling and they target the same struct (ADR-0007).
//! Host-only: it does the disk I/O on the host, fills the per-node arrays through host
//! mirrors, and deep-copies them to the device.  Because no file in the main build
//! includes this header (only its pgen unit test does), it costs the main binary nothing
//! and cannot perturb existing behaviour -- the same additive, header-only shape as the
//! IONMIX EOS/opacity readers (#10/#7).
//!
//! SESAME organises a material's data into numbered sub-tables; the 300-series carries
//! the EOS:
//!   301  total EOS (combined ion+electron+cold) -- the 1T material
//!   303  ion EOS (incl. cold curve)             }  the 3T "ion" data
//!   305  ion EOS (thermal, no cold curve)        }
//!   304  electron EOS (thermal electrons)           the 3T "electron" data
//!   601  mean ionization Zbar
//! Per ADR-0007 the 2T/3T path accepts a material ONLY if it carries separate electron
//! AND ion data, i.e. an electron table (304) together with an ion table (305 or 303).
//! A 1T-only material -- one that carries only the total table (301) and no separate
//! electron/ion tables -- is therefore rejected with a clear error (no 1T->2T split
//! modelling).  An optional ionization table (601) populates Zbar; if it is absent Zbar
//! is left at zero.
//!
//! File layout (whitespace-delimited ASCII; lines whose first non-blank character is `#`
//! are comments and are skipped, so a table may be self-documenting):
//!   matid  ntables                          (material id, number of tables that follow)
//!   then, for each of the `ntables` tables:
//!     table_id  nr  nt                       (SESAME table id + grid dimensions)
//!     density[nr]                            (g/cc, log-spaced ascending)
//!     temperature[nt]                        (eV, log-spaced ascending)
//!     <values>                               (EOS tables 301/303/304/305: pressure then
//!                                             specific energy, each nr*nt with the
//!                                             DENSITY index fastest; 601: Zbar, nr*nt)
//! Numeric tokens are consumed in order, so the per-line grouping is irrelevant -- any
//! whitespace layout parses.  The grids must be log-uniform (the representation stores
//! only (min,spacing)) and the electron/ion (and Zbar) tables must share one grid; both
//! are checked on read.  SESAME does NOT tabulate the heat capacity, so the reader
//! DERIVES the specific heat `c_v = de/dT` per species by differencing the parsed
//! specific energy along the temperature axis (central in the interior, one-sided at the
//! ends); a non-monotone energy (c_v <= 0) is rejected since it breaks the e->T
//! inversion's well-posedness.  Energies are SPECIFIC (per unit mass).  The reader is
//! unit-agnostic: it stores the table's own units (native SESAME is K / g-cc / GPa /
//! MJ-kg); the consumer (#26) is responsible for any unit mapping, exactly as for IONMIX.

#include <algorithm>  // std::find_if
#include <cmath>      // std::log10, std::fabs
#include <cstdlib>    // exit, EXIT_FAILURE
#include <fstream>    // std::ifstream
#include <iostream>
#include <sstream>    // std::istringstream
#include <string>
#include <utility>    // std::move
#include <vector>

#include "athena.hpp"
#include "eos/eos_table_3t.hpp"

namespace eos_table_3t {

//----------------------------------------------------------------------------------------
//! \fn SesameEosFatal
//! \brief Print a clear reader error and abort (consistent with AthenaK's FATAL style).
inline void SesameEosFatal(const std::string &fname, const std::string &msg) {
  std::cout << "### FATAL ERROR in SESAME EOS reader" << std::endl
            << "  file: " << fname << std::endl
            << "  " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn CheckSesameLogUniform
//! \brief Verify `nodes` is strictly increasing and log-uniform (constant ratio); the
//!  representation stores only (min, spacing) so a non-log-uniform grid would be
//!  misinterpreted.  FATALs with a helpful message on violation.  (Mirrors the IONMIX
//!  reader's CheckEosLogUniform; kept local so each reader is self-contained.)
inline void CheckSesameLogUniform(const std::string &fname, const std::string &axis,
                                  const std::vector<Real> &nodes) {
  const int n = static_cast<int>(nodes.size());
  for (int i = 0; i < n; ++i) {
    if (!(nodes[i] > 0.0)) {
      SesameEosFatal(fname, axis + " nodes must be positive (log-spaced axis).");
    }
  }
  Real dlog = (std::log10(nodes[n-1]) - std::log10(nodes[0]))/static_cast<Real>(n - 1);
  for (int i = 0; i < n; ++i) {
    Real expect = std::log10(nodes[0]) + i*dlog;
    if (std::fabs(std::log10(nodes[i]) - expect) > 1.0e-6*std::fabs(dlog) + 1.0e-12) {
      SesameEosFatal(fname, axis + " grid is not log-uniform (node " + std::to_string(i)
                     + "); the EOS representation requires a log-spaced grid.");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \struct SesameSubTable
//! \brief One parsed SESAME sub-table: its id, grid, and value arrays (host scratch).
//!  EOS tables (301/303/304/305) fill `p` and `e`; the ionization table (601) fills `z`.
//!  Value arrays are laid out [it*nr + ir] (density index fastest, matching the file).
struct SesameSubTable {
  int id = 0;
  int nr = 0, nt = 0;
  std::vector<Real> rho, temp, p, e, z;
};

//----------------------------------------------------------------------------------------
//! \fn SesameIsEosId / SesameIsIonizationId / SesameIsIonId
//! \brief Classify a SESAME table id.  EOS tables carry (pressure, energy); 601 carries
//!  Zbar; 304 is the electron table; 303/305 are the ion tables.
inline bool SesameIsEosId(int id) {
  return id == 301 || id == 303 || id == 304 || id == 305;
}
inline bool SesameIsIonizationId(int id) { return id == 601; }
inline bool SesameIsIonId(int id) { return id == 303 || id == 305; }

//----------------------------------------------------------------------------------------
//! \fn FindSesameTable
//! \brief Return a pointer to the first parsed sub-table whose id satisfies `pred`, or
//!  nullptr if none is present.
template <typename Pred>
inline const SesameSubTable *FindSesameTable(const std::vector<SesameSubTable> &tables,
                                             Pred pred) {
  auto it = std::find_if(tables.begin(), tables.end(),
                         [&](const SesameSubTable &t) { return pred(t.id); });
  return (it == tables.end()) ? nullptr : &(*it);
}

//----------------------------------------------------------------------------------------
//! \fn CheckSesameSameGrid
//! \brief FATAL unless sub-table `t` is on exactly the same (nr,nt) grid and node values
//!  as the reference electron table `ref` (the electron/ion/Zbar tables must share one
//!  grid to populate a single representation).
inline void CheckSesameSameGrid(const std::string &fname, const SesameSubTable &ref,
                                const SesameSubTable &t, const std::string &what) {
  if (t.nr != ref.nr || t.nt != ref.nt) {
    SesameEosFatal(fname, what + " grid dimensions differ from the electron table.");
  }
  for (int i = 0; i < ref.nr; ++i) {
    if (std::fabs(t.rho[i] - ref.rho[i]) > 1.0e-12*std::fabs(ref.rho[i])) {
      SesameEosFatal(fname, what + " density grid differs from the electron table.");
    }
  }
  for (int i = 0; i < ref.nt; ++i) {
    if (std::fabs(t.temp[i] - ref.temp[i]) > 1.0e-12*std::fabs(ref.temp[i])) {
      SesameEosFatal(fname, what + " temperature grid differs from the electron table.");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn ReadSesameEos
//! \brief Parse the SESAME-style 3T EOS file `fname` into `table` (host I/O + device
//!  copy).  After this call `table` is fully populated (separate electron/ion specific
//!  energies, pressures, derived heat capacities and -- if a 601 table is present --
//!  mean ionization) and ready to be captured by value into a device kernel for the
//!  forward lookups and the e->T inversion.  A 1T-only material (no separate electron +
//!  ion tables) is rejected with a clear FATAL (ADR-0007: no 1T->2T splitting).
inline void ReadSesameEos(const std::string &fname, EosTable3T &table) {
  std::ifstream fin(fname);
  if (!fin) {
    SesameEosFatal(fname, "could not open SESAME EOS file for reading.");
  }

  // Filter out comment (`#`) and blank lines, then stream tokens in order.  This keeps
  // the parser insensitive to the per-line value grouping of the SESAME record format.
  std::ostringstream filtered;
  std::string line;
  while (std::getline(fin, line)) {
    std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') { continue; }
    filtered << line << ' ';
  }
  std::istringstream toks(filtered.str());

  // --- material header: material id + number of sub-tables ---
  int matid = 0, ntables = 0;
  if (!(toks >> matid >> ntables)) {
    SesameEosFatal(fname, "failed to read header (expected: matid ntables).");
  }
  if (ntables < 1) {
    SesameEosFatal(fname, "invalid table count (need ntables >= 1).");
  }

  // --- parse each sub-table: header (id nr nt), grids, then its value arrays ---
  auto read_n = [&](std::vector<Real> &v, int n, const std::string &what) {
    v.resize(n);
    for (int i = 0; i < n; ++i) {
      if (!(toks >> v[i])) {
        SesameEosFatal(fname, "unexpected end of file while reading " + what + ".");
      }
    }
  };
  std::vector<SesameSubTable> tables;
  for (int k = 0; k < ntables; ++k) {
    SesameSubTable t;
    if (!(toks >> t.id >> t.nr >> t.nt)) {
      SesameEosFatal(fname, "failed to read a sub-table header (expected: id nr nt).");
    }
    if (!SesameIsEosId(t.id) && !SesameIsIonizationId(t.id)) {
      SesameEosFatal(fname, "unsupported SESAME table id " + std::to_string(t.id)
                     + " (supported: 301/303/304/305 EOS, 601 ionization).");
    }
    if (t.nr < 2 || t.nt < 2) {
      SesameEosFatal(fname, "sub-table " + std::to_string(t.id)
                     + " has invalid dimensions (need nr>=2, nt>=2).");
    }
    read_n(t.rho, t.nr, "density nodes");
    read_n(t.temp, t.nt, "temperature nodes");
    if (SesameIsEosId(t.id)) {
      read_n(t.p, t.nr*t.nt, "pressure block");
      read_n(t.e, t.nr*t.nt, "specific-energy block");
    } else {  // 601 ionization
      read_n(t.z, t.nr*t.nt, "Zbar block");
    }
    tables.push_back(std::move(t));
  }

  // --- require natively-3T data: an electron table AND an ion table must be present ---
  const SesameSubTable *elec = FindSesameTable(tables,
                                               [](int id) { return id == 304; });
  const SesameSubTable *ion = FindSesameTable(tables, SesameIsIonId);
  if (elec == nullptr || ion == nullptr) {
    SesameEosFatal(fname, "non-3T material: the 2T/3T EOS path requires separate "
                   "electron (table 304) AND ion (table 303/305) data. A 1T-only "
                   "material (total table 301 only) is not accepted -- no 1T->2T split "
                   "modelling (ADR-0007).");
  }
  const SesameSubTable *zt = FindSesameTable(tables, SesameIsIonizationId);

  // --- grids: ion (and Zbar) must match the electron grid; both axes log-uniform ---
  CheckSesameSameGrid(fname, *elec, *ion, "ion table");
  if (zt != nullptr) { CheckSesameSameGrid(fname, *elec, *zt, "ionization table"); }
  CheckSesameLogUniform(fname, "temperature", elec->temp);
  CheckSesameLogUniform(fname, "density", elec->rho);

  const int nt = elec->nt, nr = elec->nr;
  table.Allocate(nt, nr, elec->temp.front(), elec->temp.back(),
                 elec->rho.front(), elec->rho.back());

  // --- fill the representation (shape (ntemp,nrho); file is [it*nr+ir], ir fastest) ---
  auto h_zbar   = Kokkos::create_mirror_view(table.zbar);
  auto h_e_ele  = Kokkos::create_mirror_view(table.e_ele);
  auto h_e_ion  = Kokkos::create_mirror_view(table.e_ion);
  auto h_p_ele  = Kokkos::create_mirror_view(table.p_ele);
  auto h_p_ion  = Kokkos::create_mirror_view(table.p_ion);
  auto h_cv_ele = Kokkos::create_mirror_view(table.cv_ele);
  auto h_cv_ion = Kokkos::create_mirror_view(table.cv_ion);

  // Positivity check for a log-log interpolated field (e/p must be strictly positive).
  auto require_positive = [&](Real v, const std::string &what) {
    if (!(v > 0.0)) {
      SesameEosFatal(fname, what + " must be positive (log-interpolated field).");
    }
  };
  // Derive c_v = de/dT by differencing the specific energy along the (linear) T axis:
  // central in the interior, one-sided at the ends.  A non-monotone e(T) (c_v<=0) breaks
  // the e->T inversion's well-posedness, so it is rejected.
  auto cv_at = [&](const SesameSubTable &tab, int it, int ir, const std::string &what) {
    int lo = (it == 0) ? 0 : it - 1;
    int hi = (it == tab.nt - 1) ? tab.nt - 1 : it + 1;
    Real dedt = (tab.e[hi*tab.nr + ir] - tab.e[lo*tab.nr + ir])
              / (tab.temp[hi] - tab.temp[lo]);
    if (!(dedt > 0.0)) {
      SesameEosFatal(fname, what + " specific energy is not increasing in temperature "
                     "(derived c_v <= 0); the e->T inversion requires c_v > 0.");
    }
    return dedt;
  };
  for (int it = 0; it < nt; ++it) {
    for (int ir = 0; ir < nr; ++ir) {
      const int idx = it*nr + ir;
      require_positive(elec->e[idx], "electron specific energy");
      require_positive(ion->e[idx], "ion specific energy");
      require_positive(elec->p[idx], "electron pressure");
      require_positive(ion->p[idx], "ion pressure");
      h_e_ele(it, ir)  = elec->e[idx];
      h_e_ion(it, ir)  = ion->e[idx];
      h_p_ele(it, ir)  = elec->p[idx];
      h_p_ion(it, ir)  = ion->p[idx];
      h_cv_ele(it, ir) = cv_at(*elec, it, ir, "electron");
      h_cv_ion(it, ir) = cv_at(*ion, it, ir, "ion");
      // Zbar may be 0 (raw bilinear), so only require it to be non-negative.
      Real zval = (zt == nullptr) ? 0.0 : zt->z[idx];
      if (!(zval >= 0.0)) {
        SesameEosFatal(fname, "mean ionization Zbar must be non-negative.");
      }
      h_zbar(it, ir) = zval;
    }
  }

  Kokkos::deep_copy(table.zbar,   h_zbar);
  Kokkos::deep_copy(table.e_ele,  h_e_ele);
  Kokkos::deep_copy(table.e_ion,  h_e_ion);
  Kokkos::deep_copy(table.p_ele,  h_p_ele);
  Kokkos::deep_copy(table.p_ion,  h_p_ion);
  Kokkos::deep_copy(table.cv_ele, h_cv_ele);
  Kokkos::deep_copy(table.cv_ion, h_cv_ion);
}

}  // namespace eos_table_3t

#endif  // EOS_SESAME_EOS_READER_HPP_
