//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyl_aniso_ring.cpp
//  \brief Verification driver for the anisotropic (magnetized) Braginskii conduction
//  operator (AnisotropicConductionOperator) in CYLINDRICAL geometry (issue [6b]/#25,
//  ADR-0006/ADR-0004): the field-aligned ring test under the cylindrical (r,phi) metric.
//
//  This is a USER pgen (built with -D PROBLEM=cyl_aniso_ring, driven from a test via
//  testutils.run_pgen).  The operator is not yet wired into the driver tasklist (the
//  Ohmic->electron coupling slice #30 does that), so the verification advances it
//  operator-split entirely inside UserProblem -- the same parabolic::OperatorSplitStep
//  task body the production conduction task uses -- on LOCAL work arrays, and writes the
//  diagnostic temperature profiles the Python verification test reads.
//
//  SETUP (the cylindrical analog of the Parrish-Stone ring test).  A 2D r-phi annulus
//  carries a PURELY AZIMUTHAL frozen field B = B0 e_phi (stored in the cell-centred B as
//  the x2/phi component), so the magnetic field lines are concentric circles of constant
//  radius -- a "ring" by construction in cylindrical coordinates.  A small smooth
//  Gaussian temperature blob sits on the mid-annulus ring at (r0, phi=0).  With the
//  strongly magnetized Braginskii coefficients kappa_par >> kappa_perp, heat must conduct
//  AZIMUTHALLY around the ring (ALONG the field, at constant r) while the cross-field
//  RADIAL leakage stays a tiny fraction -- "heat follows field lines under the cyl
//  metric".  The cylindrical metric enters through the operator's geometry accessors: the
//  azimuthal flux divergence is r-weighted (FluxDivX2 = (1/r)dF/dphi) and the physical
//  phi-gradient is the arc-length derivative (1/r)dT/dphi (CenterWidth2/Edge2Length).
//
//  OUTPUT.  Two ASCII profiles for the harness:
//    * "cyl_aniso_ring_phi.dat": T vs phi along the ring radius r=r0 (azimuthal spread);
//    * "cyl_aniso_ring_r.dat":   T vs r at phi=0 (radial confinement / cross-field leak).
//  Each lists (coord, T_final, T_init).  The Python test (test_verify_cyl_aniso_ring_cpu)
//  asserts the azimuthal far-field heated >> radial far-field leakage and baselines both
//  profiles through the shared verification harness.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::exp, std::sqrt
#include <fstream>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Cylindrical field-aligned anisotropic-conduction ring-test verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: cyl_aniso_ring requires a <hydro> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *phydro = pmbp->phydro;
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;

  // ---- IC / run parameters ----
  const Real t_cold   = pin->GetOrAddReal("problem", "t_cold", 1.0);
  const Real dt_peak  = pin->GetOrAddReal("problem", "dt_peak", 0.1);
  const Real r0       = pin->GetOrAddReal("problem", "r0", 1.0);
  const Real sigma_r  = pin->GetOrAddReal("problem", "sigma_r", 0.05);
  const Real sigma_p  = pin->GetOrAddReal("problem", "sigma_phi", 0.19634954);  // pi/16
  const Real b0       = pin->GetOrAddReal("problem", "b0", 0.6);
  const Real rho0     = pin->GetOrAddReal("problem", "rho0", 1.0);
  const int  nsuper   = pin->GetOrAddInteger("problem", "n_super", 80);
  const Real dt_fac   = pin->GetOrAddReal("problem", "dt_fac", 50.0);

  // ---- Braginskii physical + unit-conversion parameters (strongly magnetized HED) ----
  anisocond::AnisoCondParams par;
  par.zbar       = pin->GetOrAddReal("problem", "zbar", 1.0);
  par.mi_si      = pin->GetOrAddReal("problem", "mi_si", braginskii::kProtonMass);
  par.lnlam      = pin->GetOrAddReal("problem", "lnlam", 10.0);
  par.dens_conv  = pin->GetOrAddReal("problem", "dens_conv", 1.0e20);
  par.temp_conv  = pin->GetOrAddReal("problem", "temp_conv", 1.0e6);
  par.bmag_conv  = pin->GetOrAddReal("problem", "bmag_conv", 1.0e2);
  par.kappa_conv = pin->GetOrAddReal("problem", "kappa_conv", 1.0e-4);
  par.nlarsen    = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  par.vfs        = pin->GetOrAddReal("problem", "vfs", -1.0);   // Larsen cap off if <=0
  par.incl_e     = true;
  par.incl_i     = true;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks;
  const int N1 = indcs.nx1, N2 = indcs.nx2;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real x2min = pin->GetReal("mesh", "x2min");
  const Real x2max = pin->GetReal("mesh", "x2max");
  const Real dx1 = (x1max - x1min)/static_cast<Real>(N1);
  const Real dx2 = (x2max - x2min)/static_cast<Real>(N2);

  // cell-centre (r,phi) of a cell index (uniform grid; valid in ghosts too).
  auto rc = [&](int i) -> Real {
    return x1min + (static_cast<Real>(i - is) + 0.5)*dx1;
  };
  auto pc = [&](int j) -> Real {
    return x2min + (static_cast<Real>(j - js) + 0.5)*dx2;
  };

  // ---- local conserved-field + cell-centred B work arrays (sized like hydro u0) ----
  const int nmb  = phydro->u0.extent_int(0);
  const int nvar = phydro->u0.extent_int(1);
  const int n3   = phydro->u0.extent_int(2);
  const int n2   = phydro->u0.extent_int(3);
  const int n1   = phydro->u0.extent_int(4);
  DvceArray5D<Real> cons("cons", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> bcc("bcc",  nmb, 3,    n3, n2, n1);
  DvceArray5D<Real> work("work", nmb, nvar, n3, n2, n1);
  auto h_cons = Kokkos::create_mirror_view(cons);
  auto h_bcc  = Kokkos::create_mirror_view(bcc);
  auto h_work = Kokkos::create_mirror_view(work);

  // smooth Gaussian temperature blob on the ring (r0, phi=0); azimuthal frozen field.
  auto t_init = [&](Real r, Real phi) -> Real {
    Real kr = std::exp(-SQR((r - r0)/sigma_r));
    Real kp = std::exp(-SQR(phi/sigma_p));
    return t_cold + dt_peak*kr*kp;
  };
  const Real me = 0.5*b0*b0;            // magnetic energy density (uniform |B| = b0)
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          // purely azimuthal field: B = b0 e_phi (x2/phi component is IBY in cyl).
          h_bcc(m,IBX,k,j,i) = 0.0;     // B_r
          h_bcc(m,IBY,k,j,i) = b0;      // B_phi
          h_bcc(m,IBZ,k,j,i) = 0.0;     // B_z
          Real eint = t_init(rc(i), pc(j))*rho0/gm1;
          h_cons(m,IDN,k,j,i) = rho0;
          h_cons(m,IM1,k,j,i) = 0.0;
          h_cons(m,IM2,k,j,i) = 0.0;
          h_cons(m,IM3,k,j,i) = 0.0;
          h_cons(m,IEN,k,j,i) = eint + me;   // total energy incl. magnetic
        }
      }
    }
  }
  Kokkos::deep_copy(cons, h_cons);
  Kokkos::deep_copy(bcc, h_bcc);

  // ---- advance the operator operator-split by RKL2 super-time-stepping ----
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  AnisotropicConductionOperator op(pmbp, nullptr, cons, bcc, gamma, par);
  const Real dt_exp = op.ExplicitStableDt();
  const Real dt_super = dt_fac*dt_exp;
  Kokkos::deep_copy(work, cons);
  for (int s = 0; s < nsuper; ++s) {
    parabolic::OperatorSplitStep(integ, op, work, dt_super);
  }
  Kokkos::deep_copy(h_work, work);

  // recover the gas temperature from the evolved conserved state (v=0, |B|=b0 uniform).
  auto temp = [&](int j, int i) -> Real {
    return gm1*(h_work(0,IEN,ks,j,i) - me)/h_work(0,IDN,ks,j,i);
  };

  // ring radial index (closest to r0) and the phi=0 column index.
  int i_ring = is;
  Real dmin = 1.0e30;
  for (int i = is; i <= ie; ++i) {
    Real d = std::fabs(rc(i) - r0);
    if (d < dmin) { dmin = d; i_ring = i; }
  }
  int j_phi0 = js;
  dmin = 1.0e30;
  for (int j = js; j <= je; ++j) {
    Real d = std::fabs(pc(j));
    if (d < dmin) { dmin = d; j_phi0 = j; }
  }

  // ---- write the diagnostic profiles (rank 0; single-block 2D verification) ----
  if (global_variable::my_rank == 0) {
    std::ofstream fphi("cyl_aniso_ring_phi.dat");
    fphi << "# cylindrical anisotropic-conduction ring test (issue #25): T vs phi at the "
         << "ring radius\n";
    fphi << "# r0=" << r0 << " r_ring=" << rc(i_ring) << " b0=" << b0
         << " dt_exp=" << dt_exp << " dt_super=" << dt_super << " n_super=" << nsuper
         << " t_cold=" << t_cold << " dt_peak=" << dt_peak << "\n";
    fphi << "# phi  T_final  T_init\n";
    for (int j = js; j <= je; ++j) {
      fphi << pc(j) << "  " << temp(j, i_ring) << "  "
           << t_init(rc(i_ring), pc(j)) << "\n";
    }
    fphi.close();

    std::ofstream fr("cyl_aniso_ring_r.dat");
    fr << "# cylindrical anisotropic-conduction ring test (issue #25): T vs r at phi=0\n";
    fr << "# phi0=" << pc(j_phi0) << " r0=" << r0 << " sigma_r=" << sigma_r << "\n";
    fr << "# r  T_final  T_init\n";
    for (int i = is; i <= ie; ++i) {
      fr << rc(i) << "  " << temp(j_phi0, i) << "  "
         << t_init(rc(i), pc(j_phi0)) << "\n";
    }
    fr.close();

    std::cout << "### cyl_aniso_ring: wrote cyl_aniso_ring_phi.dat (" << (je - js + 1)
              << " phi cells at r=" << rc(i_ring) << ") and cyl_aniso_ring_r.dat ("
              << (ie - is + 1) << " r cells at phi=" << pc(j_phi0) << ")." << std::endl;
  }

  // The verification advances the operator entirely here on local arrays; exit cleanly
  // (the athinput sets nlim = tlim = 0 so the driver does no time integration).
  std::exit(EXIT_SUCCESS);
  return;
}
