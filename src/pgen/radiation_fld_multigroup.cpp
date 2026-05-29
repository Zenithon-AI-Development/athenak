//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fld_multigroup.cpp
//  \brief Verification driver for the N-group flux-limited radiation diffusion operator
//  (FLDMultigroupOperator) advanced by RKL2 super-time-stepping (issue [17a]/#24,
//  ADR-0001/0007).
//
//  This is a USER pgen (built with -D PROBLEM=radiation_fld_multigroup, driven from a
//  test via testutils.run_pgen).  Like the grey Marshak driver (#17), the operator is not
//  yet wired into the driver tasklist (the matter coupling #35 / radiation-hydro #41
//  slices do that), so the verification advances it operator-split entirely inside
//  UserProblem -- the same parabolic::OperatorSplitStep task body the production Hydro
//  conduction task uses -- and writes its results to ASCII files the Python test reads.
//
//  MULTIGROUP MARSHAK WAVE.  A 1D optically-thick cold slab E_g = e_floor (all groups) is
//  heated from the inner-x1 face held at E_g(x1min) = e_source (Dirichlet, all groups).
//  Each group g has its OWN extinction chi_g = rho*kappa_R,g(rho,T_e) read from the
//  tabulated Rosseland transport opacity (#7), so in the equilibrium-diffusion limit each
//  group reduces to linear diffusion with its own coefficient D_g = c/(3 chi_g) and its
//  own error-function half-space profile
//      E_g(x,t) = e_floor + (e_source - e_floor) * erfc( (x - x1min) / (2 sqrt(D_g t)) ).
//  The groups therefore penetrate to DIFFERENT depths -- a group-resolved diffusion test.
//  The final per-group profiles are written to "multigroup_fld_profile.dat"
//  (x  E_0 ... E_{N-1}) and the per-group (chi_g, D_g) to "multigroup_fld_groups.dat"
//  (ig  chi_g  D_g), so the Python test can build the per-group analytic erfc oracle.

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
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Multigroup FLD Marshak-wave verification (per-group Rosseland D).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // ---- parameters ----
  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real rho_bg  = pin->GetOrAddReal("problem", "rho_bg", 1.0e-2);
  const Real te_bg   = pin->GetOrAddReal("problem", "te_bg", 1.0e2);
  const Real nl      = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real e_src   = pin->GetOrAddReal("problem", "e_source", 1.0);
  const Real e_floor = pin->GetOrAddReal("problem", "e_floor", 0.1);
  const Real tlim    = pin->GetOrAddReal("problem", "tlim_fld", 50.0);
  const int  nsuper  = pin->GetOrAddInteger("problem", "n_super", 20);
  const std::string fname = pin->GetString("problem", "opacity_file");

  // ---- read the multigroup opacity table (group structure + per-group Rosseland D) ----
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixOpacity(fname, table);
  const int NG = table.ngroups;

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

  // ===== multigroup cold slab heated from the inner-x1 Dirichlet source =====
  DvceArray5D<Real> erad("erad", nmb, NG, n3, n2, n1);
  auto h_e = Kokkos::create_mirror_view(erad);
  for (int m = 0; m < nmb; ++m) {
    for (int g = 0; g < NG; ++g) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            h_e(m, g, k, j, i) = e_floor;        // cold slab, every group
          }
        }
      }
    }
  }
  Kokkos::deep_copy(erad, h_e);

  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);

  // Dirichlet inner-x1 source = e_src (all groups), zero-gradient elsewhere.
  FLDMultigroupOperator op(pmbp, pin, erad, table, c_light, rho_bg, te_bg, nl, e_src);

  // advance nsuper operator-split RKL2 supersteps to tlim.
  const Real dt_super = tlim/static_cast<Real>(nsuper);
  for (int n = 0; n < nsuper; ++n) {
    parabolic::OperatorSplitStep(integ, op, erad, dt_super);
  }
  Kokkos::deep_copy(h_e, erad);

  // per-group extinction chi_g (precomputed in the operator from the Rosseland opacity).
  auto d_chi = op.chi();
  auto h_chi = Kokkos::create_mirror_view(d_chi);
  Kokkos::deep_copy(h_chi, d_chi);

  // ===== write results (rank 0 only; single-block 1D verification) ================
  if (global_variable::my_rank == 0) {
    std::ofstream fp("multigroup_fld_profile.dat");
    fp << "# multigroup FLD Marshak wave (issue #24): x  E_0 ... E_" << (NG-1) << "\n";
    fp << "# c=" << c_light << " rho_bg=" << rho_bg << " te_bg=" << te_bg
       << " n_larsen=" << nl << " e_source=" << e_src << " e_floor=" << e_floor
       << " tlim=" << tlim << " ngroups=" << NG << "\n";
    for (int i = is; i <= ie; ++i) {
      fp << xc(i);
      for (int g = 0; g < NG; ++g) { fp << "  " << h_e(0, g, ks, js, i); }
      fp << "\n";
    }
    fp.close();

    std::ofstream fg("multigroup_fld_groups.dat");
    fg << "# multigroup FLD per-group coefficients (issue #24): ig  chi_g  D_g\n";
    fg << "# chi_g = rho*kappa_R,g(rho,te) [1/length]; D_g = c/(3 chi_g)\n";
    for (int g = 0; g < NG; ++g) {
      Real D_g = c_light/(3.0*h_chi(g));
      fg << g << "  " << h_chi(g) << "  " << D_g << "\n";
    }
    fg.close();

    std::cout << "### radiation_fld_multigroup: wrote multigroup_fld_profile.dat ("
              << (ie - is + 1) << " cells, " << NG << " groups) and "
              << "multigroup_fld_groups.dat." << std::endl;
    std::cout << "### per-group (chi_g -> D_g):";
    for (int g = 0; g < NG; ++g) {
      std::cout << " " << h_chi(g) << "->" << c_light/(3.0*h_chi(g));
    }
    std::cout << std::endl;
  }

  // The verification advances the operator entirely here on local arrays; exit cleanly
  // (the athinput sets nlim = tlim = 0 so the driver does no time integration).
  std::exit(EXIT_SUCCESS);
  return;
}
