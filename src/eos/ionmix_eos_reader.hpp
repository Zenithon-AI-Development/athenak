#ifndef EOS_IONMIX_EOS_READER_HPP_
#define EOS_IONMIX_EOS_READER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_eos_reader.hpp
//! \brief IONMIX (.cn4) 3T-native EOS file reader populating the common 3T EOS
//!        representation (ADR-0007, issue [13b]/#10).
//!
//! Parses an IONMIX-style ASCII EOS table into an `eos_table_3t::EosTable3T`
//! (eos_table_3t.hpp).  The reader is the IONMIX plug-in for the shared representation;
//! a SESAME reader (#11) targets the same struct later (ADR-0007).  Host-only: it does
//! the disk I/O on the host, fills the per-node arrays through host mirrors, and
//! deep-copies them to the device.  Because no file in the main build includes this
//! header (only its pgen unit test does), it costs the main binary nothing and cannot
//! perturb existing behaviour -- the same additive, header-only shape as the IONMIX
//! opacity reader (#7).
//!
//! IONMIX is natively 3T (separate electron/ion data); per ADR-0007 the 2T/3T path
//! accepts a table ONLY if it carries separate electron and ion data.  A non-3T table
//! (a 1T-only table, marked `nspec=1`) is therefore rejected with a clear error -- no
//! 1T->2T split modelling.
//!
//! File layout (whitespace-delimited ASCII; lines whose first non-blank character is `#`
//! are comments and are skipped, so a table may be self-documenting):
//!   ntemp  ndens  nspec                          (3 integers; nspec must be 2)
//!   temps[ntemp]                                 (eV, log-spaced ascending)
//!   densities[ndens]                             (g/cc, log-spaced ascending)
//!   zbar  [it][id]                               (mean ionization; density idx fastest)
//!   e_ele [it][id]                               (specific electron internal energy)
//!   e_ion [it][id]                               (specific ion internal energy)
//!   p_ele [it][id]                               (electron pressure)
//!   p_ion [it][id]                               (ion pressure)
//!   cv_ele[it][id]                               (specific electron heat capacity)
//!   cv_ion[it][id]                               (specific ion heat capacity)
//! Numeric tokens are consumed in order, so the per-line grouping (the IONMIX `e12.6`
//! "values per line" convention) is irrelevant -- any whitespace layout parses.  The
//! grids must be log-uniform (the representation stores only (min,spacing)); checked on
//! read.  Energies and heat capacities are SPECIFIC (per unit mass), the IONMIX
//! convention -- the representation and the ConsToPrim consumer (#26) honour that.

#include <cmath>      // std::log10, std::fabs
#include <cstdlib>    // exit, EXIT_FAILURE
#include <fstream>    // std::ifstream
#include <iostream>
#include <sstream>    // std::istringstream
#include <string>
#include <vector>

#include "athena.hpp"
#include "eos/eos_table_3t.hpp"
#include "utils/ionmix_cn4_format.hpp"

namespace eos_table_3t {

//----------------------------------------------------------------------------------------
//! \fn IonmixEosFatal
//! \brief Print a clear reader error and abort (consistent with AthenaK's FATAL style).
inline void IonmixEosFatal(const std::string &fname, const std::string &msg) {
  std::cout << "### FATAL ERROR in IONMIX EOS reader" << std::endl
            << "  file: " << fname << std::endl
            << "  " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn CheckEosLogUniform
//! \brief Verify `nodes` is strictly increasing and log-uniform (constant ratio); the
//!  representation stores only (min, spacing) so a non-log-uniform grid would be
//!  misinterpreted.  FATALs with a helpful message on violation.
//! \param rtol relative-to-spacing tolerance for the log-uniformity test.  The default
//!  (1e-6) suits the high-precision synthetic fixtures; the cn4 readers pass a looser
//!  value because the FLASH tables store the grid to only ~6 significant figures, whose
//!  rounding (~1e-6 of the spacing) would otherwise trip the strict default.
inline void CheckEosLogUniform(const std::string &fname, const std::string &axis,
                               const std::vector<Real> &nodes, Real rtol = 1.0e-6) {
  const int n = static_cast<int>(nodes.size());
  for (int i = 0; i < n; ++i) {
    if (!(nodes[i] > 0.0)) {
      IonmixEosFatal(fname, axis + " nodes must be positive (log-spaced axis).");
    }
  }
  Real dlog = (std::log10(nodes[n-1]) - std::log10(nodes[0]))/static_cast<Real>(n - 1);
  for (int i = 0; i < n; ++i) {
    Real expect = std::log10(nodes[0]) + i*dlog;
    if (std::fabs(std::log10(nodes[i]) - expect) > rtol*std::fabs(dlog) + 1.0e-12) {
      IonmixEosFatal(fname, axis + " grid is not log-uniform (node " + std::to_string(i)
                     + "); the EOS representation requires a log-spaced grid.");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn ReadIonmixEos
//! \brief Parse the IONMIX-style 3T EOS file `fname` into `table` (host I/O + device
//!  copy).  After this call `table` is fully populated (separate electron/ion specific
//!  energies, pressures, heat capacities and mean ionization) and ready to be captured by
//!  value into a device kernel for the forward lookups and the e->T inversion.  A non-3T
//!  (`nspec != 2`) table is rejected with a clear FATAL (ADR-0007: no 1T->2T splitting).
inline void ReadIonmixEos(const std::string &fname, EosTable3T &table) {
  std::ifstream fin(fname);
  if (!fin) {
    IonmixEosFatal(fname, "could not open IONMIX EOS file for reading.");
  }

  // Filter out comment (`#`) and blank lines, then stream tokens in order.  This keeps
  // the parser insensitive to the per-line value grouping of the IONMIX record format.
  std::ostringstream filtered;
  std::string line;
  while (std::getline(fin, line)) {
    std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') { continue; }
    filtered << line << ' ';
  }
  std::istringstream toks(filtered.str());

  // --- header: dimensions + species count ---
  int ntemp = 0, ndens = 0, nspec = 0;
  if (!(toks >> ntemp >> ndens >> nspec)) {
    IonmixEosFatal(fname, "failed to read header (expected: ntemp ndens nspec).");
  }
  if (ntemp < 2 || ndens < 2) {
    IonmixEosFatal(fname, "invalid dimensions (need ntemp>=2, ndens>=2).");
  }
  // Require a natively-3T table: separate electron + ion data (nspec == 2).  A 1T-only
  // table (nspec == 1, total energy/pressure only) is rejected -- no 1T->2T split
  // modelling (ADR-0007).
  if (nspec != 2) {
    IonmixEosFatal(fname, "non-3T table (nspec=" + std::to_string(nspec)
                   + "): the 2T/3T EOS path requires a natively-3T table carrying "
                     "separate electron and ion data (nspec=2). 1T-only tables are not "
                     "accepted -- no 1T->2T split modelling (ADR-0007).");
  }

  // --- grids ---
  std::vector<Real> temps(ntemp), dens(ndens);
  auto read_vec = [&](std::vector<Real> &v, const std::string &what) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (!(toks >> v[i])) {
        IonmixEosFatal(fname, "unexpected end of file while reading " + what + ".");
      }
    }
  };
  read_vec(temps, "temperature nodes");
  read_vec(dens, "density nodes");
  CheckEosLogUniform(fname, "temperature", temps);
  CheckEosLogUniform(fname, "density", dens);

  // --- allocate the representation from the parsed grid extents ---
  table.Allocate(ntemp, ndens, temps.front(), temps.back(),
                 dens.front(), dens.back());

  // --- per-node EOS fields: stored [it][id] (density fastest); EosTable3T uses (it,ir) -
  auto h_zbar   = Kokkos::create_mirror_view(table.zbar);
  auto h_e_ele  = Kokkos::create_mirror_view(table.e_ele);
  auto h_e_ion  = Kokkos::create_mirror_view(table.e_ion);
  auto h_p_ele  = Kokkos::create_mirror_view(table.p_ele);
  auto h_p_ion  = Kokkos::create_mirror_view(table.p_ion);
  auto h_cv_ele = Kokkos::create_mirror_view(table.cv_ele);
  auto h_cv_ion = Kokkos::create_mirror_view(table.cv_ion);
  // `positive`: e/p/cv are log-log interpolated so must be strictly positive; zbar may be
  // 0 (raw bilinear), so only require it to be non-negative.
  auto read_block = [&](decltype(h_zbar) &h, const std::string &what, bool positive) {
    for (int it = 0; it < ntemp; ++it) {
      for (int id = 0; id < ndens; ++id) {
        Real val;
        if (!(toks >> val)) {
          IonmixEosFatal(fname, "unexpected end of file while reading " + what + ".");
        }
        if (positive ? !(val > 0.0) : !(val >= 0.0)) {
          IonmixEosFatal(fname, what + (positive
              ? " must be positive (log-interpolated field)."
              : " must be non-negative."));
        }
        h(it, id) = val;
      }
    }
  };
  read_block(h_zbar,   "mean ionization Zbar",          false);
  read_block(h_e_ele,  "electron specific energy",      true);
  read_block(h_e_ion,  "ion specific energy",           true);
  read_block(h_p_ele,  "electron pressure",             true);
  read_block(h_p_ion,  "ion pressure",                  true);
  read_block(h_cv_ele, "electron specific heat",        true);
  read_block(h_cv_ion, "ion specific heat",             true);

  Kokkos::deep_copy(table.zbar,   h_zbar);
  Kokkos::deep_copy(table.e_ele,  h_e_ele);
  Kokkos::deep_copy(table.e_ion,  h_e_ion);
  Kokkos::deep_copy(table.p_ele,  h_p_ele);
  Kokkos::deep_copy(table.p_ion,  h_p_ion);
  Kokkos::deep_copy(table.cv_ele, h_cv_ele);
  Kokkos::deep_copy(table.cv_ion, h_cv_ion);
}

//----------------------------------------------------------------------------------------
//! \var kCn4LogFloor
//! \brief Positive floor applied to the log-interpolated EOS fields (e, p, c_v) when
//!  ingesting a real IONMIX cn4 table.  The synthetic fixtures are strictly positive, but
//!  real tables carry a few cold/low-density-corner electron entries that underflow to 0
//!  (or to tiny negative numerical noise, e.g. c_v,e ~ -1e-9 in the deuterium table).
//!  Those would make `log10()` non-finite in the log-log interpolation, so they are
//!  clamped up to this floor — far below every physical entry (energies ~1e4..1e16,
//!  pressures, heat capacities), so legitimate values and T-monotonicity are untouched.
constexpr Real kCn4LogFloor = 1.0e-30;

//----------------------------------------------------------------------------------------
//! \fn ReadIonmixCn4Eos
//! \brief Parse a *real* FLASH IONMIX `.cn4` EOS table (e.g. aluminum / beryllium /
//!  deuterium) into `table` (host I/O + device copy).  Unlike
//!  `ReadIonmixEos` (which reads the whitespace-delimited synthetic fixtures), this walks
//!  the packed fixed-width cn4 record stream (utils/ionmix_cn4_format.hpp) in the FLASH
//!  record order, extracting the natively-3T per-species data the common representation
//!  needs: mean ionization `Zbar`, ion/electron specific energies, pressures and heat
//!  capacities.  The intermediate cn4 records this 2T/3T path does not use (the
//!  temperature/density derivatives dZdT, dP/dT, dE/dN) are skipped.
//!
//!  Values are loaded faithfully in the table's native units (energies/pressures as
//!  stored, eV temperatures); the only transform is the optional `mass_per_ion`
//!  (grams/ion): IONMIX tabulates the density axis as ion number density [cm^-3], so
//!  when `mass_per_ion > 0` the axis is rescaled to mass density [g/cc] (= n_ion*m_ion)
//!  for the mass-density consumers.  The default (`mass_per_ion = 1`) keeps the axis as
//!  number density, which is what the ingestion-verification test queries.
inline void ReadIonmixCn4Eos(const std::string &fname, EosTable3T &table,
                             Real mass_per_ion = 1.0) {
  ionmix_cn4::Cn4Reader r(fname);
  const int ntemp = r.ntemp, ndens = r.ndens;

  // --- grids: temperatures [eV] then ion number densities [cm^-3] ---
  std::vector<Real> temps, ndense;
  r.ReadVec(temps, ntemp);
  r.ReadVec(ndense, ndens);
  CheckEosLogUniform(fname, "temperature", temps, 1.0e-4);
  CheckEosLogUniform(fname, "number density", ndense, 1.0e-4);

  // Optionally relabel the (log-spaced) density axis from number density to mass density;
  // a constant multiplier preserves log-uniformity, so we only scale the (min,max) ends.
  const Real dscale = (mass_per_ion > 0.0) ? mass_per_ion : 1.0;
  table.Allocate(ntemp, ndens, temps.front(), temps.back(),
                 ndense.front()*dscale, ndense.back()*dscale);

  auto h_zbar   = Kokkos::create_mirror_view(table.zbar);
  auto h_e_ele  = Kokkos::create_mirror_view(table.e_ele);
  auto h_e_ion  = Kokkos::create_mirror_view(table.e_ion);
  auto h_p_ele  = Kokkos::create_mirror_view(table.p_ele);
  auto h_p_ion  = Kokkos::create_mirror_view(table.p_ion);
  auto h_cv_ele = Kokkos::create_mirror_view(table.cv_ele);
  auto h_cv_ion = Kokkos::create_mirror_view(table.cv_ion);

  // 2-D cn4 blocks are density-major with temperature varying fastest (flat index
  // id*ntemp + it); EosTable3T stores (it, id), so the inner loop is over `it`.  `floor`
  // clamps the log-interpolated fields up to kCn4LogFloor; `zbar` is stored raw (>= 0).
  auto read_block = [&](decltype(h_zbar) &h, bool floor) {
    for (int id = 0; id < ndens; ++id) {
      for (int it = 0; it < ntemp; ++it) {
        Real v = r.Next();
        h(it, id) = floor ? Kokkos::fmax(v, kCn4LogFloor) : Kokkos::fmax(v, 0.0);
      }
    }
  };
  read_block(h_zbar,   false);                  // zbar
  r.Skip(ndens*ntemp);                          // dzdt   (unused)
  read_block(h_p_ion,  true);                   // pion
  read_block(h_p_ele,  true);                   // pele
  r.Skip(ndens*ntemp);                          // dpidt  (unused)
  r.Skip(ndens*ntemp);                          // dpedt  (unused)
  read_block(h_e_ion,  true);                   // eion
  read_block(h_e_ele,  true);                   // eele
  read_block(h_cv_ion, true);                   // cvion
  read_block(h_cv_ele, true);                   // cvele
  // deidn, deedn and the opacity records that follow are not part of the EOS read.

  Kokkos::deep_copy(table.zbar,   h_zbar);
  Kokkos::deep_copy(table.e_ele,  h_e_ele);
  Kokkos::deep_copy(table.e_ion,  h_e_ion);
  Kokkos::deep_copy(table.p_ele,  h_p_ele);
  Kokkos::deep_copy(table.p_ion,  h_p_ion);
  Kokkos::deep_copy(table.cv_ele, h_cv_ele);
  Kokkos::deep_copy(table.cv_ion, h_cv_ion);
}

}  // namespace eos_table_3t

#endif  // EOS_IONMIX_EOS_READER_HPP_
