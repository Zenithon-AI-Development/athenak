//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ohmic_2t_chain.cpp
//  \brief Verification driver for the 2T equilibration chain Ohmic -> electron ->
//  radiation/ion (issue [14c]/#30, ADR-0002/0003).
//
//  This is a USER pgen (built with -D PROBLEM=ohmic_2t_chain, driven from a test via
//  testutils.run_pgen).  The Ohmic->electron heating source (three_temp::
//  ApplyOhmicElectronHeating) and the 2T point-implicit radiation coupling
//  (radiationfld::PointImplicitElectronRadiationCoupling) are not yet wired into the
//  driver tasklist (that is the MagLIF integration #32/#37), so the verification advances
//  them operator-split entirely inside ProblemGenerator::UserProblem and writes an ASCII
//  history the Python verification test reads.
//
//  A spatially-uniform cell carries four energy reservoirs (code units, c = a = 1):
//    - magnetic energy   me   (1/2 B^2),
//    - electron internal e_ele (T_e = e_ele/cv_ele),
//    - ion internal      e_ion (by subtraction, e_ion = E_tot - KE - me - e_ele),
//    - radiation energy  E_r  (kept OUTSIDE the conserved MHD total E_tot, ADR-0002).
//  Each macro-step applies the chain:
//    1. OHMIC -> ELECTRON.  The constrained-transport resistive update dissipates the
//       local magnetic-energy decrement Q = eta|J|^2 dt; E_tot is held fixed,
//       so me -= Q while the dissipated energy reappears as internal energy.  ADR-0003
//       deposits exactly that decrement on the electrons: e_ele += Q (E_tot held fixed),
//       which leaves the ion residual unchanged.
//    2. ELECTRON <-> RADIATION.  The point-implicit backward-Euler coupling exchanges
//       energy between E_r and e_ele targeting T_e (a T_e^4 = E_r at equilibrium).
//       Whatever the electrons lose to (or gain from) the radiation is mirrored in E_tot
//       (the radiation reservoir lives outside E_tot), so the ion energy recovered by
//       subtraction is again unchanged -- the radiation never heats the ions either.
//  Net: the ions are UNTOUCHED by the whole chain (e_ion constant to machine precision),
//  and the closed system me + e_ele + e_ion + E_r is conserved exactly.  The (E_r, e_ele)
//  trajectory is the verification observable, compared by the Python test to a
//  high-accuracy RK4 integration of the coupled ODE
//       de_ele/dt = Q - c chi_a (a T_e^4 - E_r),   dE_r/dt = c chi_a (a T_e^4 - E_r).

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
#include "eos/three_temperature.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief 2T equilibration-chain verification (Ohmic -> electron -> radiation/ion).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR: ohmic_2t_chain requires a <hydro> block (sizing)."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *phydro = pmbp->phydro;

  // ---- radiation + plasma parameters (code units) ----
  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real a_rad   = pin->GetOrAddReal("problem", "a_rad", 1.0);
  const Real cv_ele  = pin->GetOrAddReal("problem", "cv_ele", 1.0);
  const Real chi_a   = pin->GetOrAddReal("problem", "chi_a", 1.0);
  // resistive substep inputs: eta and the (constant) discrete current J = curl B.
  const Real eta = pin->GetOrAddReal("problem", "eta", 0.1);
  const Real jx  = pin->GetOrAddReal("problem", "jx", 1.0);
  const Real jy  = pin->GetOrAddReal("problem", "jy", 0.0);
  const Real jz  = pin->GetOrAddReal("problem", "jz", 0.0);
  // initial reservoirs.
  const Real me0   = pin->GetOrAddReal("problem", "me0", 2.0);    // 1/2 B^2 (replenished)
  const Real ke    = pin->GetOrAddReal("problem", "ke", 0.0);     // bulk kinetic energy
  const Real te0   = pin->GetOrAddReal("problem", "te0", 0.5);    // initial T_e
  const Real eion0 = pin->GetOrAddReal("problem", "eion0", 1.0);  // ion internal energy
  const Real erad0 = pin->GetOrAddReal("problem", "erad0", 0.0);  // initial radiation
  const Real dt    = pin->GetOrAddReal("problem", "dt", 0.02);
  const int  nstep = pin->GetOrAddInteger("problem", "nstep", 300);

  const Real eele0 = cv_ele*te0;
  const Real etot0 = ke + me0 + eele0 + eion0;       // conserved MHD total (rad outside)
  const Real qstep = three_temp::OhmicHeatingRate(eta, jx, jy, jz)*dt;  // Joule heat/step

  // ---- uniform single-component reservoir fields, sized like the hydro conserved array
  const int nmb = phydro->u0.extent_int(0);
  const int n3  = phydro->u0.extent_int(2);
  const int n2  = phydro->u0.extent_int(3);
  const int n1  = phydro->u0.extent_int(4);
  DvceArray5D<Real> me("me", nmb, 1, n3, n2, n1);     // magnetic energy 1/2 B^2
  DvceArray5D<Real> eele("eele", nmb, 1, n3, n2, n1); // electron internal energy
  DvceArray5D<Real> etot("etot", nmb, 1, n3, n2, n1); // conserved MHD total energy
  DvceArray5D<Real> erad("erad", nmb, 1, n3, n2, n1); // radiation energy (outside etot)
  Kokkos::deep_copy(me,   me0);
  Kokkos::deep_copy(eele, eele0);
  Kokkos::deep_copy(etot, etot0);
  Kokkos::deep_copy(erad, erad0);

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke_ = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;

  // history sampling of one (uniform) active cell.
  auto me_h   = Kokkos::create_mirror_view(me);
  auto eele_h = Kokkos::create_mirror_view(eele);
  auto etot_h = Kokkos::create_mirror_view(etot);
  auto erad_h = Kokkos::create_mirror_view(erad);
  std::vector<Real> th, er_t, ee_t, ei_t, sys_t;
  auto record = [&](Real t) {
    Kokkos::deep_copy(me_h, me);     Kokkos::deep_copy(eele_h, eele);
    Kokkos::deep_copy(etot_h, etot); Kokkos::deep_copy(erad_h, erad);
    Real m  = me_h(0,0,ks,js,is);
    Real ee = eele_h(0,0,ks,js,is);
    Real et = etot_h(0,0,ks,js,is);
    Real er = erad_h(0,0,ks,js,is);
    Real ei = et - ke - m - ee;                       // ion energy by subtraction
    th.push_back(t);
    er_t.push_back(er);
    ee_t.push_back(ee);
    ei_t.push_back(ei);
    sys_t.push_back(m + ee + ei + er);                // closed-system total energy
  };
  record(0.0);

  for (int s = 0; s < nstep; ++s) {
    par_for("ohmic_2t_chain", DevExeSpace(), 0, nmb1, ks, ke_, js, je, is, ie,
    KOKKOS_LAMBDA(const int m_, const int k, const int j, const int i) {
      // 1) Ohmic (Joule) heating -> electrons: dissipate the magnetic-energy decrement
      //    and deposit it on e_ele; the conserved MHD total E_tot is held fixed (so e_ion
      //    by subtraction is unchanged by this step).
      Real q = three_temp::OhmicHeatingRate(eta, jx, jy, jz)*dt;
      me(m_,0,k,j,i)   -= q;
      eele(m_,0,k,j,i)  = three_temp::ApplyOhmicElectronHeating(eele(m_,0,k,j,i), eta,
                                                                jx, jy, jz, dt);
      // 2) electron <-> radiation point-implicit coupling targeting T_e.  Whatever leaves
      //    the electrons goes to the radiation reservoir (outside E_tot), so mirror the
      //    electron-energy change in E_tot -> e_ion (by subtraction) stays unchanged.
      Real er_new, ee_new;
      Real ee_pre = eele(m_,0,k,j,i);
      radiationfld::PointImplicitElectronRadiationCoupling(erad(m_,0,k,j,i), ee_pre,
          cv_ele, chi_a, c_light, a_rad, dt, er_new, ee_new);
      erad(m_,0,k,j,i) = er_new;
      eele(m_,0,k,j,i) = ee_new;
      etot(m_,0,k,j,i) += (ee_new - ee_pre);
    });
    record(static_cast<Real>(s + 1)*dt);
  }

  // ---- write the trajectory (rank 0; single-block uniform verification) ----
  if (global_variable::my_rank == 0) {
    std::ofstream fh("ohmic_2t_chain.dat");
    fh << std::setprecision(12);
    fh << "# 2T equilibration chain (issue #30): Ohmic -> electron -> radiation/ion\n";
    fh << "# c=" << c_light << " a=" << a_rad << " cv_ele=" << cv_ele
       << " chi_a=" << chi_a << " eta=" << eta << " |J|^2=" << (jx*jx + jy*jy + jz*jz)
       << " Q/step=" << qstep << " dt=" << dt << " nstep=" << nstep << "\n";
    fh << "# me0=" << me0 << " te0=" << te0 << " eele0=" << eele0 << " eion0=" << eion0
       << " erad0=" << erad0 << " etot0=" << etot0 << "\n";
    fh << "# t  E_r  e_ele  e_ion  E_system\n";
    for (size_t q = 0; q < th.size(); ++q) {
      fh << th[q] << "  " << er_t[q] << "  " << ee_t[q] << "  " << ei_t[q] << "  "
         << sys_t[q] << "\n";
    }
    fh.close();
    std::cout << "### ohmic_2t_chain: wrote ohmic_2t_chain.dat (" << th.size()
              << " samples)." << std::endl;
  }

  // The chain is advanced entirely here on local arrays; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  return;
}
