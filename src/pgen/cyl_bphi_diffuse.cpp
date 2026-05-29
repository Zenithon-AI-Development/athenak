//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file cyl_bphi_diffuse.cpp
//  \brief Verification driver for the special cylindrical resistive B_phi diffusion
//  operator (ResistiveBphiOperator) -- the -eta B_phi/r^2 curl-curl term, conservative
//  (1/r)d_r(r*flux) radial form, antisymmetric axis ghost (issue [7b]/#27, ADR-0004).
//
//  This is a USER pgen (built with -D PROBLEM=cyl_bphi_diffuse, driven from a test via
//  testutils.run_pgen).  The operator is not yet wired into the driver tasklist, so the
//  verification advances it operator-split entirely inside UserProblem -- the same
//  parabolic::OperatorSplitStep task body the production parabolic tasks use -- on a
//  LOCAL single-component B_phi work array, and writes the radial profile the test reads.
//
//  ANALYTIC ORACLE (a Bessel J_1 eigenmode, regular on AND away from the axis).  The
//  axisymmetric resistive B_phi diffusion equation
//      dB_phi/dt = eta [ d^2B_phi/dr^2 + (1/r) dB_phi/dr - B_phi/r^2 ]
//  has the EXACT decaying-eigenmode solution
//      B_phi(r,t) = A J_1(k r) exp(-eta k^2 t),
//  because J_1 satisfies Bessel's equation of order 1: J_1'' + (1/r)J_1' + (1 - 1/r^2)J_1
//  = 0.  The -B_phi/r^2 curl-curl term is ESSENTIAL: drop it (naive scalar diffusion) and
//  J_1 is no longer an eigenmode, so the near-axis shape distorts and the decay rate is
//  wrong.  We run on r in [0, R] with k = j_{1,1}/R (j_{1,1} = first zero of J_1), so the
//  mode is zero on the axis (J_1(0)=0) AND at the outer radius (J_1(j_{1,1})=0) -- the
//  antisymmetric ghost at both radial ends keeps the mode exact.  The operator should
//  preserve the J_1 SHAPE and decay it at the analytic rate eta k^2, NEAR and AWAY from
//  the axis alike.
//
//  OUTPUT.  One ASCII profile "cyl_bphi_diffuse.dat" listing
//  (r, B_phi_numerical, B_phi_initial, B_phi_analytic), plus the decay factor / run
//  parameters in the header comment.  The Python test (test_verify_cyl_bphi_diffuse_cpu)
//  asserts the numerical profile matches the analytic J_1-decay profile near and away
//  from the axis, the eigenmode shape is preserved, and baselines the profile through the
//  shared verification harness.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::exp, std::fabs
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
#include "diffusion/resistive_bphi_operator.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"

namespace {
// First positive zero of the Bessel function J_1 (so J_1(j11) = 0).
constexpr double kJ11 = 3.8317059702075123;

// J_1(x) by its power series J_1(x) = sum_{m>=0} (-1)^m/(m!(m+1)!) (x/2)^{2m+1};
// converges rapidly for |x| <= j11 ~ 3.83 used here (host-only, IC + analytic oracle).
double BesselJ1(double x) {
  double half = 0.5*x;
  double term = half;          // m = 0 term
  double sum = term;
  for (int m = 1; m < 60; ++m) {
    term *= -(half*half)/(static_cast<double>(m)*(m + 1));
    sum += term;
    if (std::fabs(term) <= 1.0e-18*std::fabs(sum)) break;
  }
  return sum;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Cylindrical resistive B_phi diffusion test (Bessel J_1 eigenmode decay).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: cyl_bphi_diffuse requires a <hydro> block (for grid "
              << "sizing)." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---- run / physics parameters ----
  const Real eta     = pin->GetOrAddReal("problem", "eta", 0.01);
  const Real amp     = pin->GetOrAddReal("problem", "amp", 1.0);
  const int  nsuper  = pin->GetOrAddInteger("problem", "n_super", 60);
  const Real dt_fac  = pin->GetOrAddReal("problem", "dt_fac", 200.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int N1 = indcs.nx1;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real dx1 = (x1max - x1min)/static_cast<Real>(N1);
  const Real kwave = kJ11/(x1max - x1min);   // J_1 node at the outer radius

  // cell-centre radius of radial index i (uniform grid; valid in ghosts too).
  auto rc = [&](int i) -> Real {
    return x1min + (static_cast<Real>(i - is) + 0.5)*dx1;
  };
  auto b_init = [&](Real r) -> Real {
    return amp*static_cast<Real>(BesselJ1(static_cast<double>(kwave*r)));
  };

  // ---- single-component B_phi work field + cell-centred eta field (constant) ----
  auto &phydro = pmbp->phydro;
  const int nmb = phydro->u0.extent_int(0);
  const int n3  = phydro->u0.extent_int(2);
  const int n2  = phydro->u0.extent_int(3);
  const int n1  = phydro->u0.extent_int(4);
  DvceArray5D<Real> bphi("bphi", nmb, 1, n3, n2, n1);
  DvceArray4D<Real> etafld("eta", nmb, n3, n2, n1);
  auto h_bphi = Kokkos::create_mirror_view(bphi);
  auto h_eta  = Kokkos::create_mirror_view(etafld);

  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_bphi(m,ResistiveBphiOperator::IBPHI,k,j,i) = b_init(rc(i));
          h_eta(m,k,j,i) = eta;
        }
      }
    }
  }
  Kokkos::deep_copy(bphi, h_bphi);
  Kokkos::deep_copy(etafld, h_eta);

  // ---- advance the operator operator-split by RKL2 super-time-stepping ----
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  // single-block 1-D radial verification: pass nullptr so no neighbour exchange is built
  // (the operator applies only its antisymmetric/zero-gradient physical fill -- the
  // multi-block/MPI/AMR exchange is verified by unit_tests/resb_bphi_multiblock_test).
  ResistiveBphiOperator op(pmbp, nullptr, bphi, etafld);
  const Real dt_exp = op.ExplicitStableDt();
  const Real dt_super = dt_fac*dt_exp;
  for (int s = 0; s < nsuper; ++s) {
    parabolic::OperatorSplitStep(integ, op, bphi, dt_super);
  }
  Kokkos::deep_copy(h_bphi, bphi);

  // analytic eigenmode decay over the elapsed physical time t = n_super * dt_super.
  const Real t_total = static_cast<Real>(nsuper)*dt_super;
  const Real decay = std::exp(-eta*kwave*kwave*t_total);

  // ---- write the diagnostic radial profile (rank 0; single-block 1D verification) ----
  if (global_variable::my_rank == 0) {
    std::ofstream f("cyl_bphi_diffuse.dat");
    f << "# cylindrical resistive B_phi diffusion (issue #27): J_1 eigenmode decay\n";
    f << "# eta=" << eta << " kwave=" << kwave << " amp=" << amp
      << " dt_exp=" << dt_exp << " dt_super=" << dt_super << " n_super=" << nsuper
      << " t_total=" << t_total << " decay=" << decay << "\n";
    f << "# r  B_phi_numerical  B_phi_initial  B_phi_analytic\n";
    for (int i = is; i <= ie; ++i) {
      Real r = rc(i);
      f << r << "  " << h_bphi(0,ResistiveBphiOperator::IBPHI,ks,js,i) << "  "
        << b_init(r) << "  " << decay*b_init(r) << "\n";
    }
    f.close();
    std::cout << "### cyl_bphi_diffuse: wrote cyl_bphi_diffuse.dat (" << (ie - is + 1)
              << " radial cells, eta=" << eta << ", analytic decay=" << decay << ")."
              << std::endl;
  }

  // The verification advances the operator entirely here on local arrays; exit cleanly
  // (the athinput sets nlim = tlim = 0 so the driver does no time integration).
  std::exit(EXIT_SUCCESS);
  return;
}
