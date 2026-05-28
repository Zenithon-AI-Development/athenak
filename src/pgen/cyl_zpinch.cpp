//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyl_zpinch.cpp
//! \brief Static cylindrical Z-pinch magnetohydrostatic equilibrium (ADR-0004
//!  verification, issue [3b]/#22 -- the "magnoh / Z-pinch equilibrium" slice).
//!
//! A purely azimuthal field B_phi(r) confined by a gas-pressure gradient, in radial force
//! balance:   dp/dr = -d(B_phi^2/2)/dr - B_phi^2/r
//! The second term is the magnetic HOOP STRESS (the -B_phi^2 piece of the cylindrical
//! radial geometric source added in issue #16) -- this equilibrium is the discriminating
//! test of that term: with the wrong-sign or missing hoop stress the column is no
//! longer in balance and drifts radially at O(1) instead of staying static.
//!
//! We use a Gaussian-current channel whose force balance integrates in closed form:
//!     B_phi(r) = B0 (r/a) exp(-(r/a)^2 / 2)            (regular on the axis, B_phi(0)=0)
//!     p(r)     = p0 - B_phi(r)^2/2 + (B0^2/2)(e^{-(r/a)^2} - e^{-(rmax/a)^2})
//! The enclosed current ~ r*B_phi is finite; B_phi rises ~linearly off the axis, peaks at
//! r=a, then decays, so at the outer radius (rmax >> a) the field is ~0 and the
//! pressure is ~p0 with ~zero gradient -- consistent with an outflow outer BC.  The r=0
//! axis uses the `axis` BC (issue #16: B_phi gets the antisymmetric ghost).
//!
//! Run 1-D radial (nx2=nx3=1): B_phi on the x2-face is the only field, B_r=0 (x1f),
//! B_z=bz0 (x3f, uniform, default 0) -- div(B)=0 by construction.  A correct cylindrical
//! MHD update holds this static to ~truncation (small bounded v_r), verified in
//! tst/test_suite/cylindrical/test_verify_cyl_zpinch_cpu.py against a golden baseline.
//!
//! USER pgen: build with -D PROBLEM=cyl_zpinch (not part of the default suite binary).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize the static cylindrical Z-pinch magnetohydrostatic equilibrium.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cyl_zpinch requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_system == CoordSystem::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cyl_zpinch requires <coord> system = cylindrical" << std::endl;
    exit(EXIT_FAILURE);
  }

  // equilibrium parameters (<problem> block)
  Real d0   = pin->GetOrAddReal("problem", "d0", 1.0);     // uniform density
  Real p0   = pin->GetOrAddReal("problem", "p0", 1.0);     // on-axis reference pressure
  Real b0   = pin->GetOrAddReal("problem", "b0", 1.0);     // peak B_phi scale
  Real arad = pin->GetOrAddReal("problem", "a", 1.0);      // radial scale of the channel
  Real bz0  = pin->GetOrAddReal("problem", "bz0", 0.0);    // uniform axial seed field

  Real rmax = pmy_mesh_->mesh_size.x1max;
  Real smax = (rmax/arad)*(rmax/arad);
  Real expsmax = std::exp(-smax);

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
  auto &b0f = pmbp->pmhd->b0;

  par_for("pgen_cyl_zpinch", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    // Gaussian-current azimuthal field + the closed-form force-balanced pressure.
    Real s = (x1v/arad)*(x1v/arad);
    Real bphi = b0*(x1v/arad)*std::exp(-0.5*s);
    Real pres = p0 - 0.5*bphi*bphi + 0.5*b0*b0*(std::exp(-s) - expsmax);

    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // B_r = 0 (x1f); B_phi (x2f, at cell-centre radius); B_z = bz0 (x3f, uniform).
    b0f.x1f(m,k,j,i) = 0.0;
    b0f.x2f(m,k,j,i) = bphi;
    b0f.x3f(m,k,j,i) = bz0;
    if (i==ie) { b0f.x1f(m,k,j,i+1) = 0.0; }
    if (j==je) { b0f.x2f(m,k,j+1,i) = bphi; }
    if (k==ke) { b0f.x3f(m,k+1,j,i) = bz0; }

    if (is_ideal) {
      u0(m,IEN,k,j,i) = pres/gm1 + 0.5*(bphi*bphi + bz0*bz0);
    }
  });

  return;
}
