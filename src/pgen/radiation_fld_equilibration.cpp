//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fld_equilibration.cpp
//  \brief Verification driver for COUPLED grey radiation-hydro (issue [8c]/#28,
//  ADR-0001): the grey flux-limited radiation diffusion operator (FLDGreyOperator, #17)
//  together with the per-cell point-implicit matter-radiation emission/absorption
//  coupling (radiationfld::PointImplicitGreyCoupling, #23).
//
//  This is a USER pgen (built with -D PROBLEM=radiation_fld_equilibration, driven from a
//  test via testutils.run_pgen).  Neither the FLD operator nor the coupling is wired into
//  the driver tasklist yet (that is the MagLIF integration #30/#32), so the verification
//  advances them operator-split entirely inside ProblemGenerator::UserProblem -- the FLD
//  spatial diffusion via the same parabolic::OperatorSplitStep task body the production
//  conduction task uses, then the local matter-radiation exchange as a per-cell
//  point-implicit (backward-Euler) substep -- and writes ASCII profiles the Python
//  verification test reads.  The matter energy is the hydro internal energy density: with
//  the ideal-gas closure e_mat = c_v T and c_v = rho/(gamma-1) (T = (gamma-1) e_mat/rho),
//  matching radiation_source's gm1/rho volumetric heat capacity.
//
//  PART 1 -- matter-radiation EQUILIBRATION (the analytic/reference equilibration test).
//  A spatially-uniform box (FLD diffusion is a no-op on a flat field) is initialised out
//  of thermal equilibrium and relaxed by the point-implicit coupling alone:
//      dE_r/dt = c chi_a (a T^4 - E_r),   de_mat/dt = -dE_r/dt,   e_mat = c_v T.
//  Two cases are recorded -- (A) hot radiation preheats cold matter, (B) hot matter cools
//  into cold radiation -- each as a time history (t, E_r, e_mat, T) to
//  "rad_equil_history.dat".  The pair conserves E_r + e_mat exactly and relaxes to
//  radiative equilibrium a T_eq^4 = E_r,eq with a T_eq^4 + c_v T_eq = E_r^0 + e_mat^0.
//  The Python test compares the recorded trajectory to a high-accuracy RK4 integration of
//  the coupled ODE (the reference solution) and the final state to the analytic
//  equilibrium.
//
//  PART 2 -- COUPLED radiation diffusion + matter equilibration ("radiating absorption
//  front").  A 1-D optically-thick (chi large) cold absorbing slab is heated from an
//  inner-x1 Dirichlet radiation source; the radiation DIFFUSES into the slab (FLD
//  operator) while the strong local coupling (chi_a large) keeps the matter LOCKED to the
//  radiation field, a T^4 = E_r.  With a tenuous matter heat capacity (small rho ->
//  radiation-dominated) the radiation profile stays close to the equilibrium-diffusion
//  erfc Marshak wave E_r(x) = e_floor + (e_source-e_floor) erfc(x/(2 sqrt(D t))),
//  D = c/(3 chi), while the matter is heated to local radiative equilibrium behind the
//  front.  The final E_r(x), e_mat(x), T(x) and a T^4(x) profiles are written to
//  "rad_equil_front.dat".

#include <cstdlib>   // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <cmath>     // std::pow, std::sqrt
#include <fstream>
#include <iomanip>   // std::setprecision
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"

namespace {
//! \brief Apply one per-cell point-implicit matter-radiation coupling substep over the
//! active cells of (erad, emat) with uniform heat capacity cv and absorption chi_a.
void CoupleStep(MeshBlockPack *pmbp, DvceArray5D<Real> erad, DvceArray5D<Real> emat,
                Real cv, Real chi_a, Real c_light, Real a_rad, Real dt) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  par_for("rad_equil_couple", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real er_new, em_new;
    radiationfld::PointImplicitGreyCoupling(erad(m,0,k,j,i), emat(m,0,k,j,i), cv,
                                            chi_a, c_light, a_rad, dt, er_new, em_new);
    erad(m,0,k,j,i) = er_new;
    emat(m,0,k,j,i) = em_new;
  });
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Coupled grey radiation-hydro verification: matter-radiation equilibration
//! (PART 1) + a coupled FLD-diffusion / point-implicit coupling absorption front (PART2).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: radiation_fld_equilibration requires a <hydro> block "
              << "(for the ideal-gas gamma => matter heat capacity)." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *phydro = pmbp->phydro;
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;

  // ---- shared radiation constants (code units) ----
  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real a_rad   = pin->GetOrAddReal("problem", "a_rad", 1.0);

  // ---- local single-component work arrays sized like the hydro conserved field ----
  const int nmb = phydro->u0.extent_int(0);
  const int n3  = phydro->u0.extent_int(2);
  const int n2  = phydro->u0.extent_int(3);
  const int n1  = phydro->u0.extent_int(4);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, ks = indcs.ks;
  const int ng = indcs.ng;
  const int N  = indcs.nx1;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real dx = (x1max - x1min)/static_cast<Real>(N);
  auto xc = [&](int i) -> Real {
    return x1min + (static_cast<Real>(i - is) + 0.5)*dx;
  };

  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);

  // ===== PART 1: uniform matter-radiation equilibration ===========================
  // Two uniform cases relaxed by the point-implicit coupling alone (FLD is a no-op on a
  // flat field): (A) hot radiation preheats cold matter; (B) hot matter cools.
  const Real eq_rho   = pin->GetOrAddReal("problem", "eq_rho", 1.0);
  const Real eq_chi_a = pin->GetOrAddReal("problem", "eq_chi_a", 1.0);
  const Real eq_dt    = pin->GetOrAddReal("problem", "eq_dt", 0.02);
  const int  eq_nstep = pin->GetOrAddInteger("problem", "eq_nstep", 300);
  const Real cv_eq    = eq_rho/gm1;          // matter volumetric heat capacity rho/(g-1)
  // case A: cold matter (T) + hot radiation; case B: hot matter + cold radiation.
  const Real tmat_a = pin->GetOrAddReal("problem", "eq_tmat_a", 0.5);
  const Real trad_a = pin->GetOrAddReal("problem", "eq_trad_a", 1.5);
  const Real tmat_b = pin->GetOrAddReal("problem", "eq_tmat_b", 1.5);
  const Real trad_b = pin->GetOrAddReal("problem", "eq_trad_b", 0.5);

  DvceArray5D<Real> erA("erA", nmb, 1, n3, n2, n1), emA("emA", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> erB("erB", nmb, 1, n3, n2, n1), emB("emB", nmb, 1, n3, n2, n1);
  Kokkos::deep_copy(erA, a_rad*trad_a*trad_a*trad_a*trad_a);
  Kokkos::deep_copy(emA, cv_eq*tmat_a);
  Kokkos::deep_copy(erB, a_rad*trad_b*trad_b*trad_b*trad_b);
  Kokkos::deep_copy(emB, cv_eq*tmat_b);

  // sample one (uniform) active cell into the history each step.
  auto erA_h = Kokkos::create_mirror_view(erA);
  auto emA_h = Kokkos::create_mirror_view(emA);
  auto erB_h = Kokkos::create_mirror_view(erB);
  auto emB_h = Kokkos::create_mirror_view(emB);
  std::vector<Real> th, erA_t, emA_t, erB_t, emB_t;
  auto record = [&](Real t) {
    Kokkos::deep_copy(erA_h, erA); Kokkos::deep_copy(emA_h, emA);
    Kokkos::deep_copy(erB_h, erB); Kokkos::deep_copy(emB_h, emB);
    th.push_back(t);
    erA_t.push_back(erA_h(0,0,ks,js,is)); emA_t.push_back(emA_h(0,0,ks,js,is));
    erB_t.push_back(erB_h(0,0,ks,js,is)); emB_t.push_back(emB_h(0,0,ks,js,is));
  };
  record(0.0);
  for (int s = 0; s < eq_nstep; ++s) {
    CoupleStep(pmbp, erA, emA, cv_eq, eq_chi_a, c_light, a_rad, eq_dt);
    CoupleStep(pmbp, erB, emB, cv_eq, eq_chi_a, c_light, a_rad, eq_dt);
    record(static_cast<Real>(s + 1)*eq_dt);
  }

  // ===== PART 2: coupled FLD diffusion + matter equilibration (absorption front) ==
  const Real chi      = pin->GetOrAddReal("problem", "chi", 1.0e3);   // FLD Rosseland
  const Real chi_a    = pin->GetOrAddReal("problem", "chi_a", 1.0e3); // Planck absorption
  const Real nl       = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real e_source = pin->GetOrAddReal("problem", "e_source", 1.0);
  const Real e_floor  = pin->GetOrAddReal("problem", "e_floor", 1.0e-3);
  const Real t_floor  = pin->GetOrAddReal("problem", "t_floor", 0.1);
  const Real diff_rho = pin->GetOrAddReal("problem", "diff_rho", 1.0e-2);
  const Real tlim_fld = pin->GetOrAddReal("problem", "tlim_fld", 50.0);
  const int  n_super  = pin->GetOrAddInteger("problem", "n_super", 40);
  const Real cv_diff  = diff_rho/gm1;        // tenuous matter -> radiation-dominated

  DvceArray5D<Real> erad("erad", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> emat("emat", nmb, 1, n3, n2, n1);
  Kokkos::deep_copy(erad, e_floor);
  Kokkos::deep_copy(emat, cv_diff*t_floor);

  // Dirichlet inner-x1 source = e_source, zero-gradient elsewhere.
  FLDGreyOperator op(pmbp, erad, c_light, chi, nl, e_source);
  const Real dt_super = tlim_fld/static_cast<Real>(n_super);
  for (int s = 0; s < n_super; ++s) {
    // 1) FLD spatial diffusion of the radiation energy (operator-split RKL2 superstep).
    parabolic::OperatorSplitStep(integ, op, erad, dt_super);
    // 2) local point-implicit matter-radiation exchange (re-locks matter to a T^4 = E_r).
    CoupleStep(pmbp, erad, emat, cv_diff, chi_a, c_light, a_rad, dt_super);
  }
  auto erad_h = Kokkos::create_mirror_view(erad);
  auto emat_h = Kokkos::create_mirror_view(emat);
  Kokkos::deep_copy(erad_h, erad);
  Kokkos::deep_copy(emat_h, emat);

  // ===== write results (rank 0 only; single-block 1D verification) ================
  if (global_variable::my_rank == 0) {
    std::ofstream fh("rad_equil_history.dat");
    fh << std::setprecision(12);
    fh << "# coupled grey radiation-hydro equilibration (issue #28): matter-radiation "
       << "relaxation\n";
    fh << "# c=" << c_light << " a=" << a_rad << " cv=" << cv_eq << " chi_a=" << eq_chi_a
       << " dt=" << eq_dt << " nstep=" << eq_nstep << "\n";
    fh << "# A: cold matter (T=" << tmat_a << ") + hot radiation (T=" << trad_a << "); "
       << "B: hot matter (T=" << tmat_b << ") + cold radiation (T=" << trad_b << ")\n";
    fh << "# t  ErA  emA  ErB  emB\n";
    for (size_t q = 0; q < th.size(); ++q) {
      fh << th[q] << "  " << erA_t[q] << "  " << emA_t[q] << "  "
         << erB_t[q] << "  " << emB_t[q] << "\n";
    }
    fh.close();

    std::ofstream ff("rad_equil_front.dat");
    ff << std::setprecision(10);
    ff << "# coupled grey radiation-hydro absorption front (issue #28): FLD diffusion + "
       << "point-implicit coupling\n";
    ff << "# c=" << c_light << " a=" << a_rad << " chi=" << chi << " chi_a=" << chi_a
       << " cv=" << cv_diff << " e_source=" << e_source << " e_floor=" << e_floor
       << " tlim=" << tlim_fld << " D=" << c_light/(3.0*chi) << "\n";
    ff << "# x  E_r  e_mat  T  aT4\n";
    for (int i = is; i <= ie; ++i) {
      Real er = erad_h(0,0,ks,js,i);
      Real em = emat_h(0,0,ks,js,i);
      Real tt = em/cv_diff;
      Real at4 = a_rad*tt*tt*tt*tt;
      ff << xc(i) << "  " << er << "  " << em << "  " << tt << "  " << at4 << "\n";
    }
    ff.close();

    std::cout << "### radiation_fld_equilibration: wrote rad_equil_history.dat ("
              << th.size() << " samples) and rad_equil_front.dat ("
              << (ie - is + 1) << " cells)." << std::endl;
  }

  // The verification advances both operators entirely here on local arrays; exit cleanly
  // (the athinput sets nlim = tlim = 0 so the driver does no time integration).
  std::exit(EXIT_SUCCESS);
  return;
}
