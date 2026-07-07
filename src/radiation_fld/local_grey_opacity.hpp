#ifndef RADIATION_FLD_LOCAL_GREY_OPACITY_HPP_
#define RADIATION_FLD_LOCAL_GREY_OPACITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file local_grey_opacity.hpp
//! \brief Per-cell grey extinction refresh chi(rho,Te) for the FLD operator (#204).
//!
//! The frozen SI-calibrated constant (ADR-0014, #184) evaluates the Rosseland opacity
//! ONCE at the solid-liner reference state and applies it to every cell -- so the B1
//! tenuous vacuum gap (rho ~ 2.7e4 x below solid) is treated as optically thick and
//! radiation cannot free-stream across it (the rank-1 B1 faithfulness gap).  This
//! refresh replaces that constant with a LOCAL lookup, one cell at a time:
//!
//!   rho    = u0(IDN)                      [code units]
//!   Te     = eos.Te(rho, e_ele/rho)       [eV] -- the SAME tabulated_3t closure
//!            ConsToPrim2T inverts (the mrad_eos_aware precedent, #183)
//!   kappa  = GreyRosselandMean(opac, rho*dens_cgs, Te)   [cm^2/g, table units]
//!   chi    = kappa * rho_cgs * L          [1/code-length] (OpacityCodeFromCgs)
//!
//! The result lands in the erad-shaped per-cell field the grey FLDGreyOperator diffuses
//! with (EnableLocalChi): opaque liner, transparent gap.  The refresh covers GHOST cells
//! too (u0's ghosts are current at the operator-split call points), because the flux
//! kernels read the face-harmonic chi one cell into the ghost region.  chi is floored at
//! `chi_floor` (> 0), which both guards the operator's 1/chi divisions and catches
//! non-finite lookups from degenerate states; a cell with rho <= 0 floors directly.
//!
//! The state (and hence chi) is FROZEN over an RKL2 super-step like the rest of the
//! operator-split background, so the caller refreshes once per super-step (per Strang
//! half in the coupled MagLIF task list), not per substage.

#include "athena.hpp"
#include "eos/eos_table_3t.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "diffusion/operator_si_calibration.hpp"
#include "radiation_fld/grey_opacity_mean.hpp"

namespace radiationfld {

//----------------------------------------------------------------------------------------
//! \fn void RefreshGreyChiField
//! \brief Fill the erad-shaped per-cell extinction field chi_cell(m,0,k,j,i) from the
//! live conserved state u0 (density IDN + electron energy density at `eele_idx`),
//! through the tabulated_3t Te closure and the grey Rosseland mean of the multigroup
//! opacity table.  `dens_to_cgs` converts code density to the opacity table's density
//! unit (g/cc); `length_cgs` converts the cgs extinction to 1/code-length.  Covers the
//! full (ghost-inclusive) extent of chi_cell.
inline void RefreshGreyChiField(const DvceArray5D<Real> &u0, const int eele_idx,
    const eos_table_3t::EosTable3T &eos, const opacity::MultigroupOpacity &opac,
    const Real dens_to_cgs, const Real length_cgs, const Real chi_floor,
    DvceArray5D<Real> &chi_cell) {
  const int nmb1 = chi_cell.extent_int(0) - 1;
  const int n3 = chi_cell.extent_int(2);
  const int n2 = chi_cell.extent_int(3);
  const int n1 = chi_cell.extent_int(4);
  par_for("fld_chi_refresh", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho = u0(m, IDN, k, j, i);
    Real chi = chi_floor;
    if (rho > 0.0) {
      Real te = eos.Te(rho, u0(m, eele_idx, k, j, i)/rho);
      Real rho_cgs = rho*dens_to_cgs;
      Real kap = GreyRosselandMean(opac, rho_cgs, te, 1.0);
      chi = op_si_calib::OpacityCodeFromCgs(kap, rho_cgs, length_cgs);
      if (!(chi > chi_floor)) { chi = chi_floor; }   // floor; catches NaN too
    }
    chi_cell(m, 0, k, j, i) = chi;
  });
}

}  // namespace radiationfld

#endif  // RADIATION_FLD_LOCAL_GREY_OPACITY_HPP_
