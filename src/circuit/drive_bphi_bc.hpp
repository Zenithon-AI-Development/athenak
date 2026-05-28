#ifndef CIRCUIT_DRIVE_BPHI_BC_HPP_
#define CIRCUIT_DRIVE_BPHI_BC_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file drive_bphi_bc.hpp
//! \brief Reusable cylindrical B_phi outer-radial user boundary fill driven by the
//!        circuit drive source (ADR-0005, issue [9a]/#21).
//!
//! Implements the time-dependent B_phi boundary the pulsed-power drive imposes at the
//! outer radius via AthenaK's existing user-BC hook (UserBoundaryFnPtr / BoundaryFlag::
//! user, dispatched in MHD::ApplyPhysicalBCs).  In AthenaK-cylindrical (ADR-0004) the
//! coordinates are (x1,x2,x3)=(r,phi,z), so B_phi is the x2-face field b0.x2f, centred at
//! the cell centre in r.  Two modes for the outer-x1 (R_out) ghosts:
//!   * driven    -- B_phi = mu0*I(t)/(2*pi*r)         (current I(t) flows in the load);
//!   * nocurrent -- d_r(r*B_phi)=0 => B_phi ~ 1/r     (vacuum outside the load).
//! B_r (x1f) and B_z (x3f) use a zero-gradient (outflow) ghost, the standard choice at an
//! open radial boundary.  The kernel only touches MeshBlocks whose outer_x1 flag is
//! `user`, so it is safe to call unconditionally from a user_bcs_func.  Hydro/MHD
//! cell-centred ghosts at the same face are the pgen's responsibility (a standard outflow
//! copy), keeping this helper focused on the field the circuit drives.
//!
//! A pgen wires the boundary in two lines: construct a circuit::DriveSource, and enroll a
//! user_bcs_func that evaluates I = ds.Current(pm->time) and calls ApplyDriveBphiBC.  It
//! is only included by such pgens (never the main build), so it is additive.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "bvals/bvals.hpp"
#include "circuit/drive_source.hpp"

namespace circuit {

//----------------------------------------------------------------------------------------
//! \fn ApplyDriveBphiBC
//! \brief Fill the outer-x1 B-field ghosts on every MeshBlock flagged `user`: B_phi from
//!        the driven (mu0*I/2*pi*r) or nocurrent (1/r) formula, B_r and B_z by outflow.
//! \param pm        the Mesh (gives mb_bcs, mb_size, mb_indcs, time, and pmhd->b0)
//! \param current   the load current I(t) already evaluated on the host this step
//! \param mu0       code-unit permeability (Heaviside-Lorentz: 1)
//! \param nocurrent if true use the 1/r vacuum extrapolation rather than the driven form
inline void ApplyDriveBphiBC(Mesh *pm, Real current, Real mu0, bool nocurrent) {
  if (pm->pmb_pack->pmhd == nullptr) return;  // a B_phi BC is meaningful only for MHD

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int &ie = indcs.ie;
  int nx1 = indcs.nx1;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pm->pmb_pack->nmb_thispack;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &b0 = pm->pmb_pack->pmhd->b0;

  par_for("drive_bphi_ox1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::user) { return; }
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    // last interior cell-centre radius + its B_phi: the nocurrent extrapolation anchor.
    Real r_ref = CellCenterX(nx1-1, nx1, x1min, x1max);
    for (int i=0; i<ng; ++i) {
      // ghost cell-centre radius (B_phi=x2f is cell-centred in r).
      Real r_g = CellCenterX(nx1+i, nx1, x1min, x1max);
      Real bphi = nocurrent ? NoCurrentBphi(b0.x2f(m,k,j,ie), r_ref, r_g)
                            : DrivenBphi(current, r_g, mu0);
      b0.x2f(m,k,j,ie+i+1) = bphi;
      if (j == n2-1) { b0.x2f(m,k,j+1,ie+i+1) = bphi; }

      // B_r (x1f) and B_z (x3f): zero-gradient (outflow) at the open radial boundary.
      b0.x1f(m,k,j,ie+i+2) = b0.x1f(m,k,j,ie+1);
      b0.x3f(m,k,j,ie+i+1) = b0.x3f(m,k,j,ie);
      if (k == n3-1) { b0.x3f(m,k+1,j,ie+i+1) = b0.x3f(m,k+1,j,ie); }
    }
  });
}

}  // namespace circuit

#endif  // CIRCUIT_DRIVE_BPHI_BC_HPP_
