//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_decay.cpp
//  \brief Problem generator for the resistive magnetic-field-decay verification (issue
//  [7a], ADR-0003).  Lays down a uniform, static medium (rho, p, v=0) threaded by a small
//  transverse single-mode field B_y(x) = by0 sin(k x) (with optional guide field bx0/bz0)
//  on a periodic 1-D box.  Under the induction equation with a uniform magnetic
//  diffusivity eta, B_y diffuses as
//        B_y(x,t) = by0 sin(k x) exp(-eta k^2 t),
//  so the antinode amplitude decays at the analytic rate eta k^2.  Run with the variable
//  Spitzer/Braginskii resistivity (<mhd> resistivity = spitzer) the diffusivity eta is
//  T-dependent Spitzer value eta(rho,T_e); the verification test
//  (tst/test_suite/verification/test_verify_resistive_decay_cpu.py) measures the decay
//  rate and checks it against that analytic eta.
//
//  Because rho and p are uniform and the perturbation is linear (by0 << background), eta
//  is uniform in space, so the decay is the clean single-mode exponential above.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "driver/driver.hpp"
#include "pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for the resistive decay of a single-mode B_y(x).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "resistive_decay requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read parameters
  Real d0   = pin->GetOrAddReal("problem", "d0", 1.0);
  Real p0   = pin->GetOrAddReal("problem", "p0", 1.0);
  Real by0  = pin->GetOrAddReal("problem", "by0", 1.0e-2);
  Real bx0  = pin->GetOrAddReal("problem", "bx0", 0.0);
  Real bz0  = pin->GetOrAddReal("problem", "bz0", 0.0);
  Real nwave = pin->GetOrAddReal("problem", "nwave", 1.0);

  Real x1size = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real kx = 2.0*M_PI*nwave/x1size;   // wavenumber of the seeded transverse mode

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  par_for("pgen_resist_decay", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    // cell-centered conserved variables (static medium)
    u0(m,IDN,k,j,i) = d0;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // face-centered fields: B_x = bx0, B_y = by0 sin(k x), B_z = bz0.  B_y on an x2-face
    // is centered at the cell center in x1, so it uses x1v -> div(B) = 0 exactly in 1-D.
    Real byc = by0*sin(kx*x1v);
    b0.x1f(m,k,j,i) = bx0;
    b0.x2f(m,k,j,i) = byc;
    b0.x3f(m,k,j,i) = bz0;
    if (i==ie) {b0.x1f(m,k,j,i+1) = bx0;}
    if (j==je) {b0.x2f(m,k,j+1,i) = byc;}
    if (k==ke) {b0.x3f(m,k+1,j,i) = bz0;}

    if (eos.is_ideal) {
      // total energy = internal + kinetic (0) + magnetic (cell-centered |B|^2/2)
      u0(m,IEN,k,j,i) = p0/gm1 + 0.5*(bx0*bx0 + byc*byc + bz0*bz0);
    }
  });

  return;
}
