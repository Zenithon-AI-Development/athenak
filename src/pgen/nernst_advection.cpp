//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nernst_advection.cpp
//  \brief Problem generator for the Nernst B_z-advection verification (RED-first, #238,
//  ADR-0017).  Lays down a static, uniform-pressure 1-D medium carrying a linear
//  electron-temperature ramp T(x) = (p0/d0)*(1 - eps*(x - xt0)) -- imposed through
//  rho(x) = p0/T(x) so that with v = 0 and uniform p the state is an exact hydro
//  equilibrium -- threaded by a passive Gaussian axial-field bump
//        B_z(x) = bz0 * exp(-((x - xc)/w)^2),
//  with bz0 small enough that its magnetic pressure never stirs the gas.  Under the
//  (future, #238-gated) Nernst operator the bump advects down the temperature gradient
//  at the Nernst velocity v_N = -(beta_wedge(x_e)/x_e)*(tau_e/m_e)*grad(k_B T_e); the
//  verification test (tst/test_suite/verification/test_verify_nernst_advection_cpu.py)
//  measures the bump-centroid displacement over tlim against that analytic velocity,
//  evaluated independently from the same Braginskii chain
//  (diffusion/braginskii_transport.hpp).  Until the operator exists nothing moves the
//  bump, so the test is RED for exactly the right reason.
//
//  Built with -D PROBLEM=nernst_advection.

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
//! \brief Sets initial conditions: static T_e ramp + passive Gaussian B_z bump.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "nernst_advection requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }

  // read parameters
  Real d0  = pin->GetOrAddReal("problem", "d0", 1.0);    // density at the ramp anchor
  Real p0  = pin->GetOrAddReal("problem", "p0", 1.0);    // uniform pressure
  Real eps = pin->GetOrAddReal("problem", "eps", 0.2);   // fractional T drop per unit x
  Real xt0 = pin->GetOrAddReal("problem", "xt0", 0.5);   // ramp anchor: T(xt0) = p0/d0
  Real bz0 = pin->GetOrAddReal("problem", "bz0", 1.0e-3);  // B_z bump amplitude
  Real xc  = pin->GetOrAddReal("problem", "xc", 0.4);    // B_z bump center
  Real w   = pin->GetOrAddReal("problem", "w", 0.05);    // B_z bump Gaussian width

  Real t0 = p0/d0;   // code temperature at the ramp anchor (ideal gas T = p/rho)

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  par_for("pgen_nernst_adv", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    // static medium: uniform p, linear T ramp carried by rho = p0/T (v = 0)
    Real tcode = t0*(1.0 - eps*(x1v - xt0));
    u0(m,IDN,k,j,i) = p0/tcode;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // face-centered fields: passive Gaussian B_z bump.  B_z on an x3-face is centered at
    // the cell center in x1, so it uses x1v; d_z B_z = 0 -> div(B) = 0 exactly in 1-D.
    Real bzc = bz0*exp(-SQR((x1v - xc)/w));
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = bzc;
    if (i==ie) {b0.x1f(m,k,j,i+1) = 0.0;}
    if (j==je) {b0.x2f(m,k,j+1,i) = 0.0;}
    if (k==ke) {b0.x3f(m,k+1,j,i) = bzc;}

    if (eos.is_ideal) {
      // total energy = internal + kinetic (0) + magnetic (cell-centered |B|^2/2)
      u0(m,IEN,k,j,i) = p0/gm1 + 0.5*bzc*bzc;
    }
  });

  return;
}
