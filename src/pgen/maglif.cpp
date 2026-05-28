//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file maglif.cpp
//! \brief MagLIF / Z-pinch problem generator (issue [10a]/#32): the reusable integrated
//!  setup for the Phase-1 verification benchmarks 1-4 (single-/multi-mode MRT, RM, ICF).
//!
//! Lays down the canonical MagLIF target on a cylindrical grid -- ADR-0004 fixes
//! (x1,x2,x3)=(r,phi,z), so the AXIAL direction is x3 -- with three radial regions:
//!   * fuel   [0, r_fuel)        -- low-density gas pre-loaded with the axial seed field;
//!   * liner  [r_fuel, r_liner)  -- the dense conducting shell that implodes;
//!   * vacuum [r_liner, R_out)   -- the low-density gap carrying the driven 1/r B_phi.
//! The fuel + liner are premagnetized by a uniform axial B_z (the "premag" field that is
//! adiabatically compressed and magneto-insulates the hot fuel); the load current I(t)
//! enclosed by the liner produces the azimuthal drive field B_phi=mu0*I/(2*pi*r) laid
//! down only in the vacuum OUTSIDE the liner, so the magnetic piston compresses the shell
//! radially inward (the same physics verified in 1-D by issue [9b]/#29, here generalized
//! to a seeded, axially-resolved target).
//!
//! Perturbation seeding (problem/perturbation) displaces the liner interface radii by a
//! z-dependent amount dr(z) -- both interfaces move rigidly so the shell thickness (and
//! hence its areal mass) is preserved -- to seed the magneto-Rayleigh-Taylor (MRT) feed:
//!   * none         -- unperturbed (clean bulk implosion);
//!   * single_mode  -- one axial mode, dr(z)=A cos(2*pi*n (z-z0)/Lz + phi0);
//!   * roughness    -- a band of modes [n_min,n_max] with deterministic random phases
//!                     (a manufactured-surface-roughness spectrum, reproducible by seed).
//! The axial perturbation is only applied when z (x3) is resolved (a 3-D r-phi-z run);
//! for a 1-D radial or 2-D r-phi study it is a no-op.
//!
//! Drive wiring (ADR-0005): reuses the circuit::DriveSource + the reusable circuit-driven
//! B_phi outer-radial user boundary (circuit/drive_bphi_bc.hpp, issue [9a]/#21), just as
//! driven_pinch.cpp does -- the enrolled user_bcs_func evaluates I(pm->time) each step.
//!
//! Coordinate note: ADR-0004 ships 2-D (r,z) as the production MagLIF geometry, but
//! AthenaK's mesh forbids the (nx2=1, nx3>1) topology ("2-D in the x1-x3 plane"), so an
//! axially-resolved run uses the supported 3-D (r,phi,z) grid with a thin azimuthal
//! extent (the IC is axisymmetric -> the phi direction simply replicates the r-z state).
//!
//! USER pgen: build with -D PROBLEM=maglif (not part of the default suite binary).

#include <cmath>
#include <iostream>
#include <random>
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
//! 2*pi as a Real (device-safe; avoids the host-only M_PI macro in a device kernel).
constexpr Real kTwoPiM = 6.2831853071795864769;

// File-scope drive state shared with the (stateless-signature) user_bcs_func.
circuit::DriveSource drive_source_;
bool nocurrent_mode_ = false;

//----------------------------------------------------------------------------------------
//! \fn MagLIFBCs
//! \brief Outer-x1 user BC: fill the cell-centred MHD ghosts by outflow and the B_phi
//!  ghosts from the circuit drive (driven mu0*I(t)/2*pi*r, or the nocurrent 1/r form).
//!  Identical structure to driven_pinch.cpp's boundary (the shared ADR-0005 plumbing).
void MagLIFBCs(Mesh *pm) {
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
  par_for("maglif_ccbc_ox1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
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
//! \brief Initialize the MagLIF liner/fuel/vacuum target (+ B_z premag + perturbation
//!  seeding) and enroll the circuit-driven B_phi outer boundary.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Enroll the user BC (must happen on restart too, before pgen.cpp's enrollment check).
  user_bcs_func = MagLIFBCs;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "maglif requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_system == CoordSystem::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "maglif requires <coord> system = cylindrical" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- drive-source configuration (<problem> block; ADR-0005 mode A) ----
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

  // ---- target geometry + state (<problem> block) ----
  Real r_fuel  = pin->GetOrAddReal("problem", "r_fuel", 0.4);
  Real r_liner = pin->GetOrAddReal("problem", "r_liner", 0.6);
  Real d_fuel  = pin->GetOrAddReal("problem", "d_fuel", 0.05);
  Real d_liner = pin->GetOrAddReal("problem", "d_liner", 1.0);
  Real d_vac   = pin->GetOrAddReal("problem", "d_vac", 0.01);
  Real p0      = pin->GetOrAddReal("problem", "p0", 1.0e-3);
  Real bz0     = pin->GetOrAddReal("problem", "bz0", 0.0);   // axial premag field
  Real i_init  = drive_source_.Current(0.0);
  Real mu0     = drive_source_.mu0;

  // ---- perturbation seeding (axial, requires z resolved) ----
  // Build a unified per-mode (k_n, amplitude, phase) table; dr(z) = sum over modes of
  //   amp_n cos(2*pi*k_n (z-z0)/Lz + phase_n).
  // none -> 0 modes; single_mode -> 1 mode; roughness -> a band [n_min,n_max].
  std::string pert = pin->GetOrAddString("problem", "perturbation", "none");
  Real pert_amp    = pin->GetOrAddReal("problem", "pert_amp", 0.0);
  int  pert_mode   = pin->GetOrAddInteger("problem", "pert_mode", 1);
  Real pert_phase  = pin->GetOrAddReal("problem", "pert_phase", 0.0);
  int  pert_nmin   = pin->GetOrAddInteger("problem", "pert_nmin", 1);
  int  pert_nmax   = pin->GetOrAddInteger("problem", "pert_nmax", 8);
  int  pert_seed   = pin->GetOrAddInteger("problem", "pert_seed", 12345);

  int nmodes = 0;
  if (pmy_mesh_->three_d) {           // axial perturbation only when z (x3) is resolved
    if (pert == "single_mode") {
      nmodes = 1;
    } else if (pert == "roughness") {
      nmodes = (pert_nmax >= pert_nmin) ? (pert_nmax - pert_nmin + 1) : 0;
    } else if (pert != "none") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "unknown problem/perturbation '" << pert
                << "' (expected none|single_mode|roughness)" << std::endl;
      exit(EXIT_FAILURE);
    }
  } else if (pert != "none" && pert_amp != 0.0) {
    std::cout << "### WARNING: maglif problem/perturbation='" << pert << "' ignored: "
              << "axial direction x3 is not resolved (nx3=1); run a 3-D (r,phi,z) grid."
              << std::endl;
  }

  // Per-mode tables filled on the host then mirrored to the device for the IC kernel.
  DualArray1D<Real> mode_k("maglif_mode_k", (nmodes > 0) ? nmodes : 1);
  DualArray1D<Real> mode_amp("maglif_mode_amp", (nmodes > 0) ? nmodes : 1);
  DualArray1D<Real> mode_phase("maglif_mode_phase", (nmodes > 0) ? nmodes : 1);
  if (nmodes == 1 && pert == "single_mode") {
    mode_k.h_view(0)     = static_cast<Real>(pert_mode);
    mode_amp.h_view(0)   = pert_amp;
    mode_phase.h_view(0) = pert_phase;
  } else if (nmodes > 0) {                 // roughness: deterministic random phases
    std::mt19937 rng(static_cast<unsigned>(pert_seed));
    std::uniform_real_distribution<Real> uni(0.0, kTwoPiM);
    Real amp_per = pert_amp/static_cast<Real>(nmodes);   // aggregate -> per-mode share
    for (int n = 0; n < nmodes; ++n) {
      mode_k.h_view(n)     = static_cast<Real>(pert_nmin + n);
      mode_amp.h_view(n)   = amp_per;
      mode_phase.h_view(n) = uni(rng);
    }
  } else {                                  // none / unresolved: a single zero-amp mode
    mode_k.h_view(0)     = 0.0;
    mode_amp.h_view(0)   = 0.0;
    mode_phase.h_view(0) = 0.0;
  }
  mode_k.template modify<HostMemSpace>();
  mode_amp.template modify<HostMemSpace>();
  mode_phase.template modify<HostMemSpace>();
  mode_k.template sync<DevExeSpace>();
  mode_amp.template sync<DevExeSpace>();
  mode_phase.template sync<DevExeSpace>();
  auto k_d   = mode_k.d_view;
  auto amp_d = mode_amp.d_view;
  auto phs_d = mode_phase.d_view;
  int nm = (nmodes > 0) ? nmodes : 1;       // loop bound (dummy mode is zero-amplitude)

  // Global axial extent for the perturbation wavelength (full mesh, not the MeshBlock).
  Real z0 = pmy_mesh_->mesh_size.x3min;
  Real lz = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  bool is_ideal = eos.is_ideal;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  par_for("pgen_maglif", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    // Axial perturbation displacement dr(z) (0 when z unresolved / perturbation=none).
    Real dr = 0.0;
    if (lz > 0.0) {
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      for (int n = 0; n < nm; ++n) {
        dr += amp_d(n)*cos(kTwoPiM*k_d(n)*(x3v - z0)/lz + phs_d(n));
      }
    }
    Real rf = r_fuel + dr;
    Real rl = r_liner + dr;

    // Density by (perturbed) radial region.
    Real dens = (x1v < rf) ? d_fuel : ((x1v < rl) ? d_liner : d_vac);
    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // B_z premag (x3-face, uniform); B_r=0; driven B_phi (x2-face) only in the vacuum
    // OUTSIDE the (perturbed) liner -- the load current is enclosed by the liner.
    Real bphi = (x1v >= rl) ? circuit::DrivenBphi(i_init, x1v, mu0) : 0.0;
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
