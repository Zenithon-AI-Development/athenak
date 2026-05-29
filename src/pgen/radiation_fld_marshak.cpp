//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fld_marshak.cpp
//  \brief Verification driver for the grey flux-limited radiation diffusion operator
//  (FLDGreyOperator) advanced by RKL2 super-time-stepping (issue [8a]/#17, ADR-0001).
//
//  This is a USER pgen (built with -D PROBLEM=radiation_fld_marshak, driven from a test
//  via testutils.run_pgen).  The FLDGreyOperator is not yet wired into the driver's
//  tasklist (the matter coupling #23 and multigroup #24 slices do that), so the
//  verification advances it operator-split entirely inside UserProblem -- the same
//  parabolic::OperatorSplitStep task body the production Hydro conduction task uses --
//  and writes its results to ASCII files the Python verification test reads.
//
//  PART 1 -- analytic Marshak wave.  A 1D, optically-thick (chi large, so the Larsen
//  limiter sits in its diffusion limit lambda=1/3) cold slab E_r = e_floor is heated from
//  the inner-x1 face held at E_r(x1min) = e_source (Dirichlet).  In the
//  equilibrium-diffusion limit the grey FLD reduces to linear diffusion with
//  D = c/(3 chi), whose half-space solution into a cold medium is the error-function
//  profile
//      E_r(x,t) = e_floor + (e_source - e_floor) * erfc( (x - x1min) / (2 sqrt(D t)) ).
//  The final E_r(x) profile is written to "marshak_fld_profile.dat" (x, E_numeric).
//
//  PART 2 -- STS substage-count sweep (the ADR-0001 "measure-first" go/no-go).  On a
//  fixed representative radiation front, the per-cell optical depth tau = chi*dx is swept
//  optically-thick to optically-thin; for each, the operator's flux-limited explicit
//  diffusion stability dt is measured and the adaptive RKL2 substage count for a fixed
//  reference superstep is reported to "marshak_fld_substages.dat" (tau_cell, dt_exp,
//  nstages).  As the medium thins the diffusive dt collapses (dt_exp ~ tau * dx/c), so
//  the substage count grows -- quantifying the optically-thin c-CFL cost ADR-0001 flags
//  for the explicit approach.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::erfc, std::sqrt
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Grey FLD Marshak-wave verification + STS substage-count sweep.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // ---- parameters ----
  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real chi     = pin->GetOrAddReal("problem", "chi", 1.0e3);
  const Real nl      = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real e_src   = pin->GetOrAddReal("problem", "e_source", 1.0);
  const Real e_floor = pin->GetOrAddReal("problem", "e_floor", 0.1);
  const Real tlim    = pin->GetOrAddReal("problem", "tlim_fld", 50.0);
  const int  nsuper  = pin->GetOrAddInteger("problem", "n_super", 20);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int N  = indcs.nx1;
  const int nmb = pmbp->nmb_thispack;
  const int n1 = N + 2*ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real dx = (x1max - x1min)/static_cast<Real>(N);

  // cell-centre radial coordinate (uniform 1D grid)
  auto xc = [&](int i) -> Real {
    return x1min + (static_cast<Real>(i - is) + 0.5)*dx;
  };

  // ===== PART 1: Marshak wave =====================================================
  DvceArray5D<Real> erad("erad", nmb, 1, n3, n2, n1);
  auto h_e = Kokkos::create_mirror_view(erad);
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_e(m, 0, k, j, i) = e_floor;        // cold slab
        }
      }
    }
  }
  Kokkos::deep_copy(erad, h_e);

  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);

  // Dirichlet inner-x1 source = e_src, zero-gradient elsewhere.
  FLDGreyOperator op(pmbp, pin, erad, c_light, chi, nl, e_src);

  // advance nsuper operator-split RKL2 supersteps to tlim, recording substage counts.
  const Real dt_super = tlim/static_cast<Real>(nsuper);
  std::vector<int> run_stages;
  for (int n = 0; n < nsuper; ++n) {
    op.ApplyBoundary(erad);
    int s = integ.NumStages(dt_super, op.ExplicitStableDt());
    run_stages.push_back(s);
    parabolic::OperatorSplitStep(integ, op, erad, dt_super);
  }
  Kokkos::deep_copy(h_e, erad);

  // ===== PART 2: STS substage-count sweep over optical depth ======================
  // Fixed representative radiation front (independent of chi): erfc of half-width w.
  const Real w_front = pin->GetOrAddReal("problem", "w_front", 0.1);
  DvceArray5D<Real> efront("efront", nmb, 1, n3, n2, n1);
  auto h_f = Kokkos::create_mirror_view(efront);
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          Real arg = (xc(i) - x1min)/(2.0*w_front);
          h_f(m, 0, k, j, i) = e_floor + (e_src - e_floor)*std::erfc(arg);
        }
      }
    }
  }
  Kokkos::deep_copy(efront, h_f);

  // sweep chi optically-thick -> optically-thin (per-cell optical depth tau = chi*dx)
  std::vector<Real> chi_list = {3.0e3, 1.0e3, 3.0e2, 1.0e2, 3.0e1, 1.0e1, 3.0, 1.0};
  std::vector<Real> tau_list, dtexp_list;
  std::vector<int> stage_list;
  Real dt_super_ref = -1.0;   // set from the thickest case (first in the list)
  for (size_t q = 0; q < chi_list.size(); ++q) {
    FLDGreyOperator op_q(pmbp, pin, efront, c_light, chi_list[q], nl, -1.0);
    op_q.ApplyBoundary(efront);
    Real dt_exp_q = op_q.ExplicitStableDt();
    if (dt_super_ref < 0.0) { dt_super_ref = dt_exp_q; }   // thickest: ~2 stages
    int s = integ.NumStages(dt_super_ref, dt_exp_q);
    tau_list.push_back(chi_list[q]*dx);
    dtexp_list.push_back(dt_exp_q);
    stage_list.push_back(s);
  }

  // ===== write results (rank 0 only; single-block 1D verification) ================
  if (global_variable::my_rank == 0) {
    std::ofstream fp("marshak_fld_profile.dat");
    fp << "# grey FLD Marshak wave (issue #17): x  E_numeric\n";
    fp << "# c=" << c_light << " chi=" << chi << " n_larsen=" << nl
       << " e_source=" << e_src << " e_floor=" << e_floor
       << " tlim=" << tlim << " D=" << c_light/(3.0*chi) << "\n";
    for (int i = is; i <= ie; ++i) {
      fp << xc(i) << "  " << h_e(0, 0, ks, js, i) << "\n";
    }
    fp.close();

    std::ofstream fs("marshak_fld_substages.dat");
    fs << "# grey FLD STS substage sweep (issue #17): tau_cell  dt_exp  nstages\n";
    fs << "# dt_super_ref=" << dt_super_ref << " (= dt_exp of the thickest case)\n";
    for (size_t q = 0; q < chi_list.size(); ++q) {
      fs << tau_list[q] << "  " << dtexp_list[q] << "  " << stage_list[q] << "\n";
    }
    fs.close();

    std::cout << "### radiation_fld_marshak: wrote marshak_fld_profile.dat ("
              << (ie - is + 1) << " cells) and marshak_fld_substages.dat ("
              << chi_list.size() << " optical depths)." << std::endl;
    std::cout << "### Marshak run substage counts per superstep:";
    for (size_t n = 0; n < run_stages.size(); ++n) { std::cout << " " << run_stages[n]; }
    std::cout << std::endl;
    std::cout << "### STS substage sweep (tau_cell -> nstages, thick->thin):";
    for (size_t q = 0; q < stage_list.size(); ++q) {
      std::cout << " " << tau_list[q] << ":" << stage_list[q];
    }
    std::cout << std::endl;
  }

  // The verification advances the operator entirely here on local arrays; exit cleanly
  // (the athinput sets nlim = tlim = 0 so the driver does no time integration).
  std::exit(EXIT_SUCCESS);
  return;
}
