//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_fld_multigroup_equilibration.cpp
//  \brief Verification driver for COUPLED MULTIGROUP radiation-hydro (issue [17c]/#41,
//  ADR-0001): the N-group flux-limited radiation diffusion operator
//  (FLDMultigroupOperator, #24) together with the per-cell point-implicit MULTIGROUP
//  matter-radiation coupling (radiationfld::MultigroupArrowheadCoupling, #35) and the
//  ideal-gas matter heat capacity.
//
//  This is a USER pgen (built with -D PROBLEM=radiation_fld_multigroup_equilibration,
//  driven from a test via testutils.run_pgen).  Neither the FLD operator nor the coupling
//  is wired into the driver tasklist yet (that is the MagLIF integration), so the
//  verification advances them operator-split entirely inside UserProblem -- the FLD
//  spatial diffusion via the same parabolic::OperatorSplitStep task body the production
//  conduction task uses, then the local multigroup matter-radiation exchange as a
//  per-cell point-implicit (backward-Euler) substep -- and writes ASCII profiles the
//  Python verification test reads.  It is the multigroup analogue of the grey
//  radiation_fld_equilibration driver (#28).
//
//  The matter is split into electron and ion reservoirs with volumetric heat capacities
//  cv_ele = rho_ele/(gamma-1), cv_ion = rho_ion/(gamma-1) (ideal-gas closure, T = e/cv);
//  the photon energy is resolved into the N groups of the tabulated multigroup opacity.
//  Per-group extinctions come from that table: chi_P,g = rho*kappa_{P,g} (Planck
//  absorption -> emission/absorption coupling) and chi_R,g = rho*kappa_{R,g} (Rosseland
//  transport -> FLD diffusion D_g = c/(3 chi_R,g)).  Photon energy eps_g (group bounds)
//  enters the group Planck function through x = eps_g/(kboltz T_e).
//
//  PART 1 -- multigroup NON-EQUILIBRIUM spectral relaxation (the reference test).  A
//  spatially-uniform box (FLD diffusion is a no-op on a flat field) is initialised with a
//  NON-Planck radiation spectrum and matter out of thermal equilibrium, then relaxed by
//  the multigroup point-implicit coupling alone:
//      dE_g/dt   = c chi_P,g (B_g(T_e) - E_g),   B_g = a T_e^4 [F(x_{g+1}) - F(x_g)],
//      de_ele/dt = -sum_g dE_g/dt + c_ei (T_i - T_e),   de_ion/dt = -c_ei (T_i - T_e).
//  Two cases are recorded as time histories -- (A) hot radiation, all energy initially in
//  the LOWEST group (far from a Planck spectrum) preheating cold electrons; and
//  (B) hot electrons cooling into cold radiation -- each with the electrons exchanging
//  with hotter/colder ions (c_ei > 0).  The pair conserves sum_g E_g + e_ele +
//  e_ion exactly and relaxes to the PLANCK SPECTRUM E_g = a T_eq^4 [F(x_{g+1}) - F(x_g)]
//  at the common equilibrium temperature.  The Python test compares the recorded
//  trajectory to a high-accuracy RK4 integration of the coupled multigroup ODE (the
//  reference solution) and the final per-group state to the analytic Planck-spectrum
//  equilibrium -- the SPECTRAL / per-group verification.
//
//  PART 2 -- COUPLED multigroup FLD diffusion + matter coupling (multigroup Marshak).  A
//  1-D optically-thick cold slab is heated from an inner-x1 radiation source (all
//  groups); the per-group radiation DIFFUSES (FLD, each group with its own D_g) while the
//  strong local coupling (chi_P,g large -> beta_g >> 1) THERMALISES the spectrum to the
//  local Planck spectrum at the matter temperature, E_g(x) = a T_e(x)^4 [F(x_{g+1}(x)) -
//  F(x_g(x))].  The matter is heated behind the front.  The final per-group E_g(x), the
//  matter temperature T_e(x) and the total radiation energy are written; the Python test
//  verifies the local Planck-spectrum lock (the spectral reference), the monotone front
//  and the matter heating.

#include <cstdlib>   // std::exit, EXIT_SUCCESS, EXIT_FAILURE
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
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "radiation_fld/multigroup_coupling.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"

namespace {
// Compile-time maximum group count for per-cell device stack scratch (the table's runtime
// ngroups must not exceed this); 16 comfortably covers the verification fixtures.
constexpr int MAXG = 16;

//! \brief Apply one per-cell multigroup point-implicit coupling substep over the active
//! cells of (erad[ng], eele, eion) with uniform per-group Planck extinction chi_p[ng] and
//! group boundaries gbnd[ng+1].
void CoupleStepMG(MeshBlockPack *pmbp, DvceArray5D<Real> erad, DvceArray5D<Real> eele,
                  DvceArray5D<Real> eion, DvceArray1D<Real> chi_p,
                  DvceArray1D<Real> gbnd, int ng, Real cv_ele, Real cv_ion, Real c_light,
                  Real a_rad, Real kboltz, Real c_ei, Real dt) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  par_for("mg_rad_couple", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real eold[MAXG], enew[MAXG];
    for (int g = 0; g < ng; ++g) { eold[g] = erad(m,g,k,j,i); }
    Real ee_new, ei_new;
    radiationfld::MultigroupArrowheadCoupling(ng, eold, gbnd.data(), chi_p.data(),
        eele(m,0,k,j,i), eion(m,0,k,j,i), cv_ele, cv_ion, c_light, a_rad, kboltz,
        c_ei, dt, enew, ee_new, ei_new);
    for (int g = 0; g < ng; ++g) { erad(m,g,k,j,i) = enew[g]; }
    eele(m,0,k,j,i) = ee_new;
    eion(m,0,k,j,i) = ei_new;
  });
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Coupled multigroup radiation-hydro verification: multigroup non-equilibrium
//! spectral relaxation (PART 1) + a coupled multigroup FLD Marshak front (PART 2).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: radiation_fld_multigroup_equilibration requires a "
              << "<hydro> block (for the ideal-gas gamma => matter heat capacity)."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;

  // ---- shared radiation constants (code units) ----
  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real a_rad   = pin->GetOrAddReal("problem", "a_rad", 1.0);
  const Real kboltz  = pin->GetOrAddReal("problem", "kboltz", 30.0);

  // ---- read the multigroup opacity table (group structure + per-group opacities) ----
  const std::string fname = pin->GetString("problem", "opacity_file");
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixOpacity(fname, table);
  const int NG = table.ngroups;
  if (NG > MAXG) {
    std::cout << "### FATAL ERROR: ngroups=" << NG << " exceeds MAXG=" << MAXG
              << " (per-cell stack scratch)." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---- per-group extinctions + group boundaries (device kernel -> host mirror) ----
  // PART 1 coupling background (eq_rho, eq_te) -> chi_P,g (thin: smooth relaxation);
  // PART 2 backgrounds (diff_rho, te_bg) -> chi_P,g (thick: strong coupling lock) and the
  // FLD operator computes chi_R,g itself from the same table/background.
  const Real eq_rho  = pin->GetOrAddReal("problem", "eq_rho", 1.0e-2);
  const Real eq_te   = pin->GetOrAddReal("problem", "eq_te", 1.0e3);
  const Real diff_rho = pin->GetOrAddReal("problem", "diff_rho", 1.0e-2);
  const Real te_bg   = pin->GetOrAddReal("problem", "te_bg", 1.0e2);
  DvceArray1D<Real> chip1("chip1", NG), chip2("chip2", NG), gbnd("gbnd", NG+1);
  {
    opacity::MultigroupOpacity tab = table;   // shallow View copy -> device-valid
    auto c1 = chip1; auto c2 = chip2; auto gb = gbnd;
    const Real er1 = eq_rho, et1 = eq_te, dr2 = diff_rho, te2 = te_bg;
    const int ng_ = NG;
    par_for("mg_rad_chi", DevExeSpace(), 0, ng_-1, KOKKOS_LAMBDA(const int ig) {
      c1(ig) = er1*tab.PlanckAbsorption(ig, er1, et1);
      c2(ig) = dr2*tab.PlanckAbsorption(ig, dr2, te2);
    });
    par_for("mg_rad_gb", DevExeSpace(), 0, ng_, KOKKOS_LAMBDA(const int ig) {
      gb(ig) = tab.GroupBound(ig);
    });
  }
  auto h_chip1 = Kokkos::create_mirror_view(chip1);
  auto h_gbnd  = Kokkos::create_mirror_view(gbnd);
  Kokkos::deep_copy(h_chip1, chip1);
  Kokkos::deep_copy(h_gbnd, gbnd);

  // ---- matter heat capacities (ideal gas, volumetric: T = e/cv) ----
  const Real rho_ele = pin->GetOrAddReal("problem", "rho_ele", 1.0);
  const Real rho_ion = pin->GetOrAddReal("problem", "rho_ion", 1.0);
  const Real cv_ele = rho_ele/gm1;
  const Real cv_ion = rho_ion/gm1;

  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);

  // ===== PART 1: 0-D multigroup non-equilibrium spectral relaxation (host scalars) =====
  // The box is spatially uniform (FLD a no-op), so the relaxation is a per-cell ODE on
  // the host with the (host-callable) arrowhead coupling -- the opacity lookups already
  // done above on the device.  Two cases, each electrons + ions + N radiation groups.
  const Real c_ei1   = pin->GetOrAddReal("problem", "c_ei", 2.0);
  const Real eq_dt   = pin->GetOrAddReal("problem", "eq_dt", 0.01);
  const int  eq_nstep = pin->GetOrAddInteger("problem", "eq_nstep", 400);
  // case A: cold electrons + hotter ions, hot radiation ALL in the lowest group.
  const Real te_a = pin->GetOrAddReal("problem", "eq_te_a", 0.3);
  const Real ti_a = pin->GetOrAddReal("problem", "eq_ti_a", 0.8);
  const Real erad_a = pin->GetOrAddReal("problem", "eq_erad_a", 1.0);   // lowest group
  // case B: hot electrons + colder ions, cold radiation (all groups at floor).
  const Real te_b = pin->GetOrAddReal("problem", "eq_te_b", 1.5);
  const Real ti_b = pin->GetOrAddReal("problem", "eq_ti_b", 0.5);
  const Real erad_floor = pin->GetOrAddReal("problem", "eq_erad_floor", 1.0e-3);

  Real egA[MAXG], egB[MAXG];
  for (int g = 0; g < NG; ++g) {
    egA[g] = (g == 0) ? erad_a : erad_floor;   // non-Planck: energy in lowest group only
    egB[g] = erad_floor;                        // cold radiation
  }
  Real eeleA = cv_ele*te_a, eionA = cv_ion*ti_a;
  Real eeleB = cv_ele*te_b, eionB = cv_ion*ti_b;

  std::vector<Real> th;
  std::vector<std::vector<Real>> egA_t(NG), egB_t(NG);
  std::vector<Real> eeleA_t, eionA_t, eeleB_t, eionB_t;
  auto record = [&](Real t) {
    th.push_back(t);
    for (int g = 0; g < NG; ++g) {
      egA_t[g].push_back(egA[g]);
      egB_t[g].push_back(egB[g]);
    }
    eeleA_t.push_back(eeleA); eionA_t.push_back(eionA);
    eeleB_t.push_back(eeleB); eionB_t.push_back(eionB);
  };
  record(0.0);
  for (int s = 0; s < eq_nstep; ++s) {
    Real ngA[MAXG], ngB[MAXG], eeA, eiA, eeB, eiB;
    radiationfld::MultigroupArrowheadCoupling(NG, egA, h_gbnd.data(), h_chip1.data(),
        eeleA, eionA, cv_ele, cv_ion, c_light, a_rad, kboltz, c_ei1, eq_dt,
        ngA, eeA, eiA);
    radiationfld::MultigroupArrowheadCoupling(NG, egB, h_gbnd.data(), h_chip1.data(),
        eeleB, eionB, cv_ele, cv_ion, c_light, a_rad, kboltz, c_ei1, eq_dt,
        ngB, eeB, eiB);
    for (int g = 0; g < NG; ++g) { egA[g] = ngA[g]; egB[g] = ngB[g]; }
    eeleA = eeA; eionA = eiA; eeleB = eeB; eionB = eiB;
    record(static_cast<Real>(s + 1)*eq_dt);
  }

  // ===== PART 2: coupled multigroup FLD diffusion + coupling (Marshak front) =====
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
  auto xc = [&](int i) -> Real {
    return x1min + (static_cast<Real>(i - is) + 0.5)*dx;
  };

  const Real nl       = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real e_source = pin->GetOrAddReal("problem", "e_source", 1.0);
  const Real e_floor  = pin->GetOrAddReal("problem", "e_floor", 1.0e-3);
  const Real t_floor  = pin->GetOrAddReal("problem", "t_floor", 0.1);
  const Real tlim_fld = pin->GetOrAddReal("problem", "tlim_fld", 20.0);
  const int  n_super  = pin->GetOrAddInteger("problem", "n_super", 40);
  const Real cv_diff  = diff_rho/gm1;   // tenuous matter (radiation-dominated)

  DvceArray5D<Real> erad("erad", nmb, NG, n3, n2, n1);
  DvceArray5D<Real> eele("eele", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> eion("eion", nmb, 1, n3, n2, n1);
  Kokkos::deep_copy(erad, e_floor);          // cold slab, every group
  Kokkos::deep_copy(eele, cv_diff*t_floor);  // cold matter
  Kokkos::deep_copy(eion, 0.0);              // ions unused in PART 2 (c_ei = 0)

  // Dirichlet inner-x1 source = e_source (all groups), zero-gradient elsewhere; chi_R,g
  // from the table at (diff_rho, te_bg).
  FLDMultigroupOperator op(pmbp, erad, table, c_light, diff_rho, te_bg, nl, e_source);
  const Real dt_super = tlim_fld/static_cast<Real>(n_super);
  for (int s = 0; s < n_super; ++s) {
    // 1) per-group FLD spatial diffusion (operator-split RKL2 superstep).
    parabolic::OperatorSplitStep(integ, op, erad, dt_super);
    // 2) local multigroup point-implicit coupling: thermalise groups -> Planck spectrum
    //    at the local T_e and heat the matter (electron-only here: c_ei = 0).
    CoupleStepMG(pmbp, erad, eele, eion, chip2, gbnd, NG, cv_diff, 0.0, c_light, a_rad,
                 kboltz, 0.0, dt_super);
  }

  auto erad_h = Kokkos::create_mirror_view(erad);
  auto eele_h = Kokkos::create_mirror_view(eele);
  Kokkos::deep_copy(erad_h, erad);
  Kokkos::deep_copy(eele_h, eele);

  // per-group Rosseland extinction chi_R,g from the FLD operator (the diffusion D_g).
  auto d_chir = op.chi();
  auto h_chir = Kokkos::create_mirror_view(d_chir);
  Kokkos::deep_copy(h_chir, d_chir);
  auto h_chip2 = Kokkos::create_mirror_view(chip2);
  Kokkos::deep_copy(h_chip2, chip2);

  // ===== write results (rank 0 only; single-block 1D verification) ================
  if (global_variable::my_rank == 0) {
    // -- PART 1 history: t  EgA_0..EgA_{N-1} eeleA eionA  EgB_0..EgB_{N-1} eeleB eionB --
    std::ofstream fh("multigroup_rad_equil.dat");
    fh << std::setprecision(12);
    fh << "# coupled multigroup radiation-hydro spectral relaxation (issue #41)\n";
    fh << "# c=" << c_light << " a=" << a_rad << " kboltz=" << kboltz
       << " cv_ele=" << cv_ele << " cv_ion=" << cv_ion << " c_ei=" << c_ei1
       << " dt=" << eq_dt << " nstep=" << eq_nstep << " ngroups=" << NG << "\n";
    fh << "# A: cold e (Te=" << te_a << ") + hot ion (Ti=" << ti_a << ") + radiation in "
       << "lowest group (E0=" << erad_a << "); B: hot e (Te=" << te_b
       << ") + cold ion (Ti=" << ti_b << ") + cold radiation\n";
    fh << "# t  EgA_0..EgA_" << (NG-1) << "  eeleA eionA  EgB_0..EgB_" << (NG-1)
       << "  eeleB eionB\n";
    for (size_t q = 0; q < th.size(); ++q) {
      fh << th[q];
      for (int g = 0; g < NG; ++g) { fh << "  " << egA_t[g][q]; }
      fh << "  " << eeleA_t[q] << "  " << eionA_t[q];
      for (int g = 0; g < NG; ++g) { fh << "  " << egB_t[g][q]; }
      fh << "  " << eeleB_t[q] << "  " << eionB_t[q] << "\n";
    }
    fh.close();

    // -- per-group coefficients: ig gb_lo gb_hi chi_P1 chi_P2 chi_R2 D2 --
    std::ofstream fg("multigroup_rad_groups.dat");
    fg << std::setprecision(12);
    fg << "# per-group coeffs (issue #41): ig gb_lo gb_hi chi_P1 chi_P2 chi_R2 D2\n";
    fg << "# chi_P1 = PART1 coupling extinction (rho_eq,Te_eq); chi_P2/chi_R2 = PART2 "
       << "Planck/Rosseland extinction (diff_rho,te_bg); D2 = c/(3 chi_R2)\n";
    for (int g = 0; g < NG; ++g) {
      Real D2 = c_light/(3.0*h_chir(g));
      fg << g << "  " << h_gbnd(g) << "  " << h_gbnd(g+1) << "  " << h_chip1(g)
         << "  " << h_chip2(g) << "  " << h_chir(g) << "  " << D2 << "\n";
    }
    fg.close();

    // -- PART 2 front profiles: x  E_g_0..E_g_{N-1}  e_ele  Te  E_tot  aTe4 --
    std::ofstream ff("multigroup_rad_front.dat");
    ff << std::setprecision(10);
    ff << "# coupled multigroup FLD Marshak front (#41): FLD diffusion + multigroup "
       << "point-implicit coupling\n";
    ff << "# c=" << c_light << " a=" << a_rad << " kboltz=" << kboltz << " cv=" << cv_diff
       << " e_source=" << e_source << " e_floor=" << e_floor << " tlim=" << tlim_fld
       << " ngroups=" << NG << "\n";
    ff << "# x  E_g_0..E_g_" << (NG-1) << "  e_ele  Te  E_tot  aTe4\n";
    for (int i = is; i <= ie; ++i) {
      Real etot = 0.0;
      ff << xc(i);
      for (int g = 0; g < NG; ++g) {
        Real eg = erad_h(0,g,ks,js,i);
        etot += eg;
        ff << "  " << eg;
      }
      Real em = eele_h(0,0,ks,js,i);
      Real te = em/cv_diff;
      Real ate4 = a_rad*te*te*te*te;
      ff << "  " << em << "  " << te << "  " << etot << "  " << ate4 << "\n";
    }
    ff.close();

    std::cout << "### radiation_fld_multigroup_equilibration: wrote "
              << "multigroup_rad_equil.dat (" << th.size()
              << " samples), multigroup_rad_groups.dat (" << NG
              << " groups), multigroup_rad_front.dat (" << (ie - is + 1) << " cells)."
              << std::endl;
  }

  // The verification advances both operators entirely here on local arrays; exit cleanly
  // (the athinput sets nlim = tlim = 0 so the driver does no time integration).
  std::exit(EXIT_SUCCESS);
  return;
}
