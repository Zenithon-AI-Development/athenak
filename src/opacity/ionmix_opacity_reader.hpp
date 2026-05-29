#ifndef OPACITY_IONMIX_OPACITY_READER_HPP_
#define OPACITY_IONMIX_OPACITY_READER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_opacity_reader.hpp
//! \brief IONMIX multigroup-opacity file reader populating the common opacity
//!        representation (ADR-0007, issue [16a]/#7).
//!
//! Parses an IONMIX-style ASCII multigroup-opacity table into a
//! `opacity::MultigroupOpacity` (multigroup_opacity.hpp).  The reader is the IONMIX
//! plug-in for the shared representation; TOPS / PROPACEOS readers would target the same
//! struct later (ADR-0007).  Host-only: it does the disk I/O on the host, fills the
//! per-node arrays through host mirrors, and deep-copies them to the device.  Because no
//! file in the main build includes this header (only its pgen unit test does), it costs
//! the main binary nothing and cannot perturb existing behaviour.
//!
//! File layout (whitespace-delimited ASCII; lines whose first non-blank character is `#`
//! are comments and are skipped, so a table may be self-documenting):
//!   ntemp  ndens  ngroups                         (3 integers)
//!   temps[ntemp]                                  (eV, log-spaced ascending)
//!   densities[ndens]                              (g/cc, log-spaced ascending)
//!   group_bounds[ngroups+1]                       (eV, ascending photon-energy bounds)
//!   planck_absorb[it][id][ig]                     (cm^2/g, group index fastest)
//!   planck_emiss [it][id][ig]
//!   rosseland    [it][id][ig]
//! Numeric tokens are consumed in order, so the per-line grouping (the IONMIX `e12.6`
//! "4 values per line" convention) is irrelevant -- any whitespace layout parses.  The
//! grids must be log-uniform (the representation assumes it); this is checked on read.
//!
//! Opacities are mass (specific) opacities [cm^2/g], the IONMIX convention; the FLD
//! consumer multiplies by rho.  Only the multigroup-opacity records are parsed here; the
//! EOS records an IONMIX `.cn4` file also carries are handled by the EOS readers
//! (#10/#11) into `eos/eos_table_3t.hpp`.

#include <cmath>      // std::log10, std::fabs
#include <cstdlib>    // exit, EXIT_FAILURE
#include <fstream>    // std::ifstream
#include <iostream>
#include <sstream>    // std::istringstream
#include <string>
#include <vector>

#include "athena.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "utils/ionmix_cn4_format.hpp"

namespace opacity {

//----------------------------------------------------------------------------------------
//! \fn IonmixFatal
//! \brief Print a clear reader error and abort (consistent with AthenaK's FATAL style).
inline void IonmixFatal(const std::string &fname, const std::string &msg) {
  std::cout << "### FATAL ERROR in IONMIX opacity reader" << std::endl
            << "  file: " << fname << std::endl
            << "  " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn CheckLogUniform
//! \brief Verify `nodes` is strictly increasing and log-uniform (constant ratio); the
//!  representation stores only (min, spacing) so a non-log-uniform grid would be
//!  misinterpreted.  Returns nothing; FATALs with a helpful message on violation.
//! \param rtol relative-to-spacing tolerance (default 1e-6 for synthetic fixtures; the
//!  cn4 reader passes a looser value because the real FLASH tables store the grid to ~6
//!  significant figures).
inline void CheckLogUniform(const std::string &fname, const std::string &axis,
                            const std::vector<Real> &nodes, Real rtol = 1.0e-6) {
  const int n = static_cast<int>(nodes.size());
  for (int i = 0; i < n; ++i) {
    if (!(nodes[i] > 0.0)) {
      IonmixFatal(fname, axis + " nodes must be positive (log-spaced axis).");
    }
  }
  Real dlog = (std::log10(nodes[n-1]) - std::log10(nodes[0]))/static_cast<Real>(n - 1);
  for (int i = 0; i < n; ++i) {
    Real expect = std::log10(nodes[0]) + i*dlog;
    if (std::fabs(std::log10(nodes[i]) - expect) > rtol*std::fabs(dlog) + 1.0e-12) {
      IonmixFatal(fname, axis + " grid is not log-uniform (node " + std::to_string(i)
                  + "); the opacity representation requires a log-spaced grid.");
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn ReadIonmixOpacity
//! \brief Parse the IONMIX-style opacity file `fname` into `table` (host I/O + device
//!  copy).  After this call `table` is fully populated and ready to be captured by value
//!  into a device kernel.
inline void ReadIonmixOpacity(const std::string &fname, MultigroupOpacity &table) {
  std::ifstream fin(fname);
  if (!fin) {
    IonmixFatal(fname, "could not open IONMIX opacity file for reading.");
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

  // --- header: dimensions ---
  int ntemp = 0, ndens = 0, ngroups = 0;
  if (!(toks >> ntemp >> ndens >> ngroups)) {
    IonmixFatal(fname, "failed to read header (expected: ntemp ndens ngroups).");
  }
  if (ntemp < 2 || ndens < 2 || ngroups < 1) {
    IonmixFatal(fname, "invalid dimensions (need ntemp>=2, ndens>=2, ngroups>=1).");
  }

  // --- grids and group boundaries ---
  std::vector<Real> temps(ntemp), dens(ndens), bounds(ngroups + 1);
  auto read_vec = [&](std::vector<Real> &v, const std::string &what) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (!(toks >> v[i])) {
        IonmixFatal(fname, "unexpected end of file while reading " + what + ".");
      }
    }
  };
  read_vec(temps, "temperature nodes");
  read_vec(dens, "density nodes");
  read_vec(bounds, "group boundaries");

  CheckLogUniform(fname, "temperature", temps);
  CheckLogUniform(fname, "density", dens);
  for (int ig = 0; ig < ngroups; ++ig) {
    if (!(bounds[ig+1] > bounds[ig])) {
      IonmixFatal(fname, "group energy boundaries must be strictly ascending.");
    }
  }

  // --- allocate the representation from the parsed grid extents ---
  table.Allocate(ngroups, ntemp, ndens, temps.front(), temps.back(),
                 dens.front(), dens.back());

  auto h_bounds = Kokkos::create_mirror_view(table.group_bounds);
  for (int ig = 0; ig <= ngroups; ++ig) { h_bounds(ig) = bounds[ig]; }
  Kokkos::deep_copy(table.group_bounds, h_bounds);

  // --- opacity blocks: stored [it][id][ig] (group fastest); store as (ig,it,id) ---
  auto h_pa = Kokkos::create_mirror_view(table.planck_absorb);
  auto h_pe = Kokkos::create_mirror_view(table.planck_emiss);
  auto h_ro = Kokkos::create_mirror_view(table.rosseland);
  auto read_block = [&](decltype(h_pa) &h, const std::string &what) {
    for (int it = 0; it < ntemp; ++it) {
      for (int id = 0; id < ndens; ++id) {
        for (int ig = 0; ig < ngroups; ++ig) {
          Real val;
          if (!(toks >> val)) {
            IonmixFatal(fname, "unexpected end of file while reading " + what + ".");
          }
          if (!(val > 0.0)) {
            IonmixFatal(fname, what + " must be positive (log-interpolated opacity).");
          }
          h(ig, it, id) = val;
        }
      }
    }
  };
  read_block(h_pa, "Planck absorption opacity");
  read_block(h_pe, "Planck emission opacity");
  read_block(h_ro, "Rosseland transport opacity");

  Kokkos::deep_copy(table.planck_absorb, h_pa);
  Kokkos::deep_copy(table.planck_emiss, h_pe);
  Kokkos::deep_copy(table.rosseland, h_ro);
}

//----------------------------------------------------------------------------------------
//! \fn ReadIonmixCn4Opacity
//! \brief Parse a *real* FLASH IONMIX `.cn4` multigroup-opacity table (e.g. aluminum /
//!  beryllium / deuterium) into `table` (host I/O + device copy).  Unlike
//!  `ReadIonmixOpacity` (which reads the whitespace synthetic fixtures), this walks the
//!  packed fixed-width cn4 record stream (utils/ionmix_cn4_format.hpp): it skips the EOS
//!  records that precede the opacity data in the combined cn4 file (those are the EOS
//!  reader's job), then reads the `ngroups+1` photon-energy group boundaries [eV] and the
//!  three opacity arrays in the FLASH order **Rosseland, Planck absorption, Planck
//!  emission** (note: the synthetic fixture's order is absorption/emission/Rosseland; the
//!  real cn4 order differs, and we map each to the right field here).  Opacities load
//!  faithfully in their native mass-opacity units [cm^2/g].  `mass_per_ion` (grams/ion)
//!  optionally rescales the density axis from ion number density [cm^-3] to mass density
//!  [g/cc], matching `ReadIonmixCn4Eos`; the default leaves it as number density.
inline void ReadIonmixCn4Opacity(const std::string &fname, MultigroupOpacity &table,
                                 Real mass_per_ion = 1.0) {
  ionmix_cn4::Cn4Reader r(fname);
  const int ntemp = r.ntemp, ndens = r.ndens, ngroups = r.ngroups;

  // --- grids: temperatures [eV] then ion number densities [cm^-3] ---
  std::vector<Real> temps, ndense;
  r.ReadVec(temps, ntemp);
  r.ReadVec(ndense, ndens);
  CheckLogUniform(fname, "temperature", temps, 1.0e-4);
  CheckLogUniform(fname, "number density", ndense, 1.0e-4);

  // --- skip the 12 EOS records (zbar..deedn), each ndens*ntemp, that precede opacity ---
  r.Skip(12 * ndens * ntemp);

  // --- group boundaries [eV], ascending ---
  std::vector<Real> bounds;
  r.ReadVec(bounds, ngroups + 1);
  for (int ig = 0; ig < ngroups; ++ig) {
    if (!(bounds[ig+1] > bounds[ig])) {
      IonmixFatal(fname, "group energy boundaries must be strictly ascending.");
    }
  }

  const Real dscale = (mass_per_ion > 0.0) ? mass_per_ion : 1.0;
  table.Allocate(ngroups, ntemp, ndens, temps.front(), temps.back(),
                 ndense.front()*dscale, ndense.back()*dscale);

  auto h_bounds = Kokkos::create_mirror_view(table.group_bounds);
  for (int ig = 0; ig <= ngroups; ++ig) { h_bounds(ig) = bounds[ig]; }
  Kokkos::deep_copy(table.group_bounds, h_bounds);

  // 3-D cn4 opacity blocks are group-major, then density, then temperature (flat index
  // ((ig*ndens + id)*ntemp + it)); MultigroupOpacity stores (ig, it, id), so the inner
  // loop runs over `it`.  All real opacities are strictly positive (log-interpolated).
  auto h_pa = Kokkos::create_mirror_view(table.planck_absorb);
  auto h_pe = Kokkos::create_mirror_view(table.planck_emiss);
  auto h_ro = Kokkos::create_mirror_view(table.rosseland);
  auto read_block = [&](decltype(h_pa) &h, const std::string &what) {
    for (int ig = 0; ig < ngroups; ++ig) {
      for (int id = 0; id < ndens; ++id) {
        for (int it = 0; it < ntemp; ++it) {
          Real v = r.Next();
          if (!(v > 0.0)) {
            IonmixFatal(fname, what + " must be positive (log-interpolated opacity).");
          }
          h(ig, it, id) = v;
        }
      }
    }
  };
  read_block(h_ro, "Rosseland transport opacity");   // cn4 order: Rosseland first
  read_block(h_pa, "Planck absorption opacity");
  read_block(h_pe, "Planck emission opacity");

  Kokkos::deep_copy(table.planck_absorb, h_pa);
  Kokkos::deep_copy(table.planck_emiss, h_pe);
  Kokkos::deep_copy(table.rosseland, h_ro);
}

}  // namespace opacity

#endif  // OPACITY_IONMIX_OPACITY_READER_HPP_
