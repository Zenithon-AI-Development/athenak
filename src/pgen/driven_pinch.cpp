//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file driven_pinch.cpp
//! \brief Problem generator for a current-driven cylindrical Z-pinch column, wiring the
//!  circuit drive source to the B_phi outer-radial boundary (ADR-0005, issue [9a]/#21).
//!
//! Lays down a static plasma column (v=0) with an optional axial seed field B_z
//! (premagnetization) on a cylindrical grid (x1,x2,x3)=(r,phi,z); the r=0 axis uses the
//! `axis` BC (issue #16) and the outer radius R_out uses the new circuit-driven `user`
//! B_phi BC.  A pulsed-power drive source supplies the load current I(t); the enrolled
//! user_bcs_func evaluates I(pm->time) each step and sets the ghost azimuthal field via
//! circuit::ApplyDriveBphiBC -- either driven (B_phi=mu0*I/2*pi*r) or the nocurrent
//! vacuum extrapolation (d_r(r*B_phi)=0).  The initial B_phi is seeded consistently from
//! I(0) so the field is continuous at t=0.
//!
//! Two initial-condition profiles (problem/profile):
//!   * `uniform` (default) -- a uniform static column (rho=d0) threaded everywhere by the
//!     driven 1/r field; the minimal Phase-1 driver exercising the drive/boundary plumb.
//!   * `liner`   -- a dense liner shell (fuel | liner | vacuum, by radius) w/ the driven
//!     field laid down ONLY in the vacuum OUTSIDE the liner (B_phi=0 in the fuel+liner).
//!     The magnetic piston then compresses the liner radially inward -- the bulk
//!     implosion verification of issue [9b]/#29 (trajectory vs the analytic thin-shell).
//!
//! USER pgen: build with -D PROBLEM=driven_pinch (not part of the default suite binary).

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "circuit/drive_source.hpp"
#include "circuit/drive_bphi_bc.hpp"
#include "pgen.hpp"

namespace {
// File-scope drive state shared with the (stateless-signature) user_bcs_func.
circuit::DriveSource drive_source_;
bool nocurrent_mode_ = false;

//----------------------------------------------------------------------------------------
//! \fn DrivenPinchBCs
//! \brief Outer-x1 user BC: fill the cell-centred MHD ghosts by outflow and the B_phi
//!  ghosts from the circuit drive (driven mu0*I(t)/2*pi*r, or the nocurrent 1/r form).
void DrivenPinchBCs(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int &ie = indcs.ie;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  int nvar = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = pmbp->pmhd->u0;

  // (1) cell-centred conserved variables: zero-gradient (outflow) at the outer-x1 ghosts.
  par_for("pinch_ccbc_ox1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::user) { return; }
    for (int i=0; i<ng; ++i) {
      for (int n=0; n<nvar; ++n) { u0(m,n,k,j,ie+i+1) = u0(m,n,k,j,ie); }
    }
  });

  // (2) face-centred B: driven / nocurrent B_phi + outflow B_r, B_z (shared helper).
  Real current = drive_source_.Current(pm->time);
  circuit::ApplyDriveBphiBC(pm, current, drive_source_.mu0, nocurrent_mode_);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize the driven Z-pinch column and enroll the circuit-driven B_phi BC.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Enroll the user BC (must happen on restart too, before pgen.cpp's enrollment check).
  user_bcs_func = DrivenPinchBCs;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "driven_pinch requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_system == CoordSystem::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "driven_pinch requires <coord> system = cylindrical" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- drive-source configuration (<problem> block) ----
  std::string mode = pin->GetOrAddString("problem", "current_mode", "driven");
  nocurrent_mode_ = (mode == "nocurrent");
  std::string wf = pin->GetOrAddString("problem", "current_waveform", "linear_ramp");
  drive_source_.waveform = circuit::ParseWaveform(wf);
  drive_source_.i0     = pin->GetOrAddReal("problem", "i0", 1.0);
  drive_source_.t_rise = pin->GetOrAddReal("problem", "t_rise", 1.0);
  drive_source_.mu0    = pin->GetOrAddReal("problem", "mu0", 1.0);
  if (drive_source_.waveform == circuit::CurrentWaveform::tabulated) {
    std::string cfile = pin->GetOrAddString("problem", "current_file", "unset");
    circuit::ReadCurrentWaveform(cfile, drive_source_);
  }

  if (restart) return;

  // ---- static plasma column + optional axial seed field; B_phi from I(0) ----
  std::string profile = pin->GetOrAddString("problem", "profile", "uniform");
  bool is_liner = (profile == "liner");
  Real d0  = pin->GetOrAddReal("problem", "d0", 1.0);
  Real p0  = pin->GetOrAddReal("problem", "p0", 1.0);
  Real bz0 = pin->GetOrAddReal("problem", "bz0", 0.0);
  Real i_init = drive_source_.Current(0.0);
  Real mu0 = drive_source_.mu0;

  // Liner-shell geometry (only read for profile=liner): three radial regions
  //   fuel   [0, r_fuel)      -- low-density gas inside the liner (B_phi = 0)
  //   liner  [r_fuel, r_liner)-- dense conducting shell           (B_phi = 0)
  //   vacuum [r_liner, R_out) -- low-density gap carrying the driven 1/r field
  // The current sheet sits at the liner outer edge r_liner: the magnetic pressure of the
  // vacuum field just outside it drives the implosion (the magnetic piston).
  Real r_fuel  = pin->GetOrAddReal("problem", "r_fuel", 0.4);
  Real r_liner = pin->GetOrAddReal("problem", "r_liner", 0.6);
  Real d_fuel  = pin->GetOrAddReal("problem", "d_fuel", 0.02);
  Real d_liner = pin->GetOrAddReal("problem", "d_liner", 1.0);
  Real d_vac   = pin->GetOrAddReal("problem", "d_vac", 0.01);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1;
  auto &size = pmbp->pmb->mb_size;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  bool is_ideal = eos.is_ideal;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  par_for("pgen_driven_pinch", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    // Density profile: uniform column, or the liner shell (fuel | liner | vacuum).
    Real dens = d0;
    if (is_liner) {
      dens = (x1v < r_fuel) ? d_fuel : ((x1v < r_liner) ? d_liner : d_vac);
    }
    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // B_z (premag) is the x3-face field; B_r=0; B_phi (x2-face, cell-centred in r) is the
    // initial driven profile mu0*I(0)/(2*pi*r) so it matches the boundary value at t=0.
    // For the liner profile the field exists ONLY in the vacuum outside the liner (the
    // load current is enclosed by the liner), so B_phi=0 for r < r_liner.
    Real bphi = circuit::DrivenBphi(i_init, x1v, mu0);
    if (is_liner && x1v < r_liner) { bphi = 0.0; }
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = bphi;
    b0.x3f(m,k,j,i) = bz0;
    if (i==ie) { b0.x1f(m,k,j,i+1) = 0.0; }
    if (j==je) { b0.x2f(m,k,j+1,i) = bphi; }
    if (k==ke) { b0.x3f(m,k+1,j,i) = bz0; }

    if (is_ideal) {
      u0(m,IEN,k,j,i) = p0/gm1 + 0.5*(bphi*bphi + bz0*bz0);
    }
  });

  return;
}
