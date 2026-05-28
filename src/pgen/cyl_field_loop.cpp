//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyl_field_loop.cpp
//! \brief Advected cylindrical field loop in the (r,phi) plane (ADR-0004 verification,
//!  issue [3b]/#22 -- the field-loop div(B) slice).
//!
//! A weak magnetic field loop, a circle in PHYSICAL (x,y) space centred off the axis, is
//! laid down from a z-directed vector potential A_z(r,phi) and advected by a rigid
//! azimuthal rotation v_phi = Omega*r.  The verification
//! (test_verify_cyl_field_loop_cpu.py) checks that the constrained-transport scheme keeps
//! div(B) at round-off as the loop is advected -- i.e. the cylindrical CT / r-weighted
//! EMF (issue #16) is divergence-preserving.
//!
//! KEY (cylindrical CT consistency): the face field must be the *cylindrical* discrete
//! curl of A so that the finite-volume div(B) operator annihilates it identically.  For a
//! loop in the r-phi plane (only A_z != 0) the only term that differs from the Cartesian
//! curl is the radial face field, which is the azimuthal derivative of A_z divided by the
//! ARC LENGTH r_face*dphi (Edge2Length at the face radius), not dphi:
//!     B_r   (x1f) = [A_z(j+1) - A_z(j)] / (r_face * dphi)      <-- r-weighted
//!     B_phi (x2f) = -[A_z(i+1) - A_z(i)] / dr                  <-- same as Cartesian
//! This mirrors mhd_ct.cpp's cylindrical B1-from-E3 term (divide the azimuthal E_z
//! difference by Edge2Length at the face radius).  With this div(B)=0 to round-off and CT
//! preserves it.  Run 2-D (r,phi): nx1>1, nx2>1, nx3=1.
//!
//! USER pgen: build with -D PROBLEM=cyl_field_loop (not the default suite binary).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coord_geometry.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize the advected cylindrical (r,phi) field loop.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cyl_field_loop requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_system == CoordSystem::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cyl_field_loop requires <coord> system = cylindrical" << std::endl;
    exit(EXIT_FAILURE);
  }

  // problem parameters (<problem> block)
  Real d0    = pin->GetOrAddReal("problem", "d0", 1.0);      // uniform density
  Real press = pin->GetOrAddReal("problem", "press", 1.0);   // uniform gas pressure
  Real amp   = pin->GetOrAddReal("problem", "amp", 1.0e-3);  // vector-potential amplitude
  Real rad   = pin->GetOrAddReal("problem", "rad", 0.3);     // loop radius (physical)
  Real rc    = pin->GetOrAddReal("problem", "rc", 1.0);      // loop centre radius
  Real phic  = pin->GetOrAddReal("problem", "phic", 0.0);    // loop centre azimuth
  Real omega = pin->GetOrAddReal("problem", "omega", 0.5);   // rigid rotation rate

  Real xc = rc*std::cos(phic);
  Real yc = rc*std::sin(phic);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto csys = pmbp->pcoord->coord_system;

  // z-directed vector potential A_z at cell corners (x1f, x2f).
  DvceArray4D<Real> az;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells3 = indcs.nx3 + 2*indcs.ng;
  Kokkos::realloc(az, nmb, ncells3, ncells2, ncells1);

  par_for("cylfl_az", DevExeSpace(), 0,(nmb-1),ks,ke+1,js,je+1,is,ie+1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1f = LeftEdgeX(i-is, nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2f = LeftEdgeX(j-js, nx2, x2min, x2max);
    // physical (x,y) of the (r,phi) corner; distance from the loop centre.
    Real xx = x1f*std::cos(x2f);
    Real yy = x1f*std::sin(x2f);
    Real dist = std::sqrt((xx-xc)*(xx-xc) + (yy-yc)*(yy-yc));
    az(m,k,j,i) = (dist < rad) ? amp*(rad - dist) : 0.0;
  });

  // cylindrical discrete curl -> face fields (div(B)=0 to round-off by construction).
  auto &b0 = pmbp->pmhd->b0;
  par_for("cylfl_b", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real rf  = LeftEdgeX(i-is,   nx1, x1min, x1max);   // radius of the x1-face i
    Real rfp = LeftEdgeX(i+1-is, nx1, x1min, x1max);   // radius of the x1-face i+1

    // B_r = (1/r) dA_z/dphi : divide the azimuthal A_z difference by the arc length.
    b0.x1f(m,k,j,i) = (az(m,k,j+1,i) - az(m,k,j,i))/Edge2Length(csys, dx2, rf);
    // B_phi = -dA_z/dr (no r-weighting).
    b0.x2f(m,k,j,i) = -(az(m,k,j,i+1) - az(m,k,j,i))/Edge1Length(csys, dx1);
    b0.x3f(m,k,j,i) = 0.0;

    if (i==ie) {
      b0.x1f(m,k,j,i+1) = (az(m,k,j+1,i+1) - az(m,k,j,i+1))/Edge2Length(csys, dx2, rfp);
    }
    if (j==je) {
      b0.x2f(m,k,j+1,i) = -(az(m,k,j+1,i+1) - az(m,k,j+1,i))/Edge1Length(csys, dx1);
    }
    if (k==ke) {
      b0.x3f(m,k+1,j,i) = 0.0;
    }
  });

  // conserved variables: uniform rho/p, rigid azimuthal rotation v_phi = omega*r.
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  bool is_ideal = eos.is_ideal;
  auto &u0 = pmbp->pmhd->u0;
  par_for("cylfl_cons", DevExeSpace(), 0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real bcc1 = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
    Real bcc2 = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
    Real bcc3 = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));

    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = d0*omega*x1v;   // rho * v_phi, v_phi = omega*r
    u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) {
      u0(m,IEN,k,j,i) = press/gm1
                      + 0.5*(SQR(u0(m,IM1,k,j,i)) + SQR(u0(m,IM2,k,j,i))
                             + SQR(u0(m,IM3,k,j,i)))/d0
                      + 0.5*(bcc1*bcc1 + bcc2*bcc2 + bcc3*bcc3);
    }
  });

  return;
}
