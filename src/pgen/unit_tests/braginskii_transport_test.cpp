//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file braginskii_transport_test.cpp
//  \brief Unit test: the magnetized Braginskii transport coefficients
//  (diffusion/braginskii_transport.hpp) evaluated on the device reproduce the Braginskii
//  reference values and the correct unmagnetized / strongly-magnetized limits, and the
//  Bohm-like anomalous term + vacuum-resistivity floor activate as specified (issue [5],
//  ADR-0003/0006).
//
// Built/run by
//   cmake -D PROBLEM=unit_tests/braginskii_transport_test -B build_unit
//   (cd build_unit/src && make) && ./build_unit/src/athena \
//       -i inputs/unit_tests/braginskii_transport_test.athinput
// Auto-run by tst/test_suite/unit_tests/test_unit_braginskii_transport_cpu.py.

#include <cmath>     // std::log, std::sqrt, std::pow
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace bt = braginskii;

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Braginskii transport-coefficient unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("braginskii_transport_test");

  // Reference dense-plasma state (SI): n_e = 1e26 m^-3 (1e20 cm^-3), T = 100 eV, Z = 1,
  // ln(Lambda) = 5, hydrogenic ions (m_i = m_p), B = 100 T for the gyro/Bohm checks.
  const Real n_e   = 1.0e26;
  const Real Zbar  = 1.0;
  const Real lnL   = 5.0;
  const Real T_eV  = 100.0;
  const Real T_K   = T_eV * bt::kElectronVolt / bt::kBoltzmann;   // 100 eV in Kelvin
  const Real m_i   = bt::kProtonMass;
  const Real b_mag = 100.0;
  // Vacuum-floor parameters.
  const Real rho_floor = 1.0e-3, eta_vac = 1.0e10, eta_in = 5.0;
  // Cold state for the Coulomb-log floor branch (1 eV).
  const Real T_cold_K = 1.0 * bt::kElectronVolt / bt::kBoltzmann;

  enum {
    ITAU_E=0, ITAU_I, IKPAR_E, IKPERP_E0, IKPERP_E100, IKPERP_E200, IKPAR_I, IKPERP_I0,
    IETA_PAR, IETA_PERP0, IETA_PERP1000, IBASE_E, IBASE_I, IETA0, IBOHM_D, IBOHM_NU,
    IOMEGA_CE, IMAGDIFF, IFLOOR_BELOW, IFLOOR_ABOVE, ICLOG_HOT, ICLOG_COLD, NVAL
  };
  DvceArray1D<Real> d_vals("brag_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  // Evaluate every coefficient inside a real device kernel and copy results to the host.
  par_for("brag_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real tau_e = bt::ElectronCollisionTime(n_e, T_K, Zbar, lnL);
    Real tau_i = bt::IonCollisionTime(n_e, T_K, Zbar, m_i, lnL);
    d_vals(ITAU_E) = tau_e;
    d_vals(ITAU_I) = tau_i;

    d_vals(IKPAR_E)     = bt::ConductivityParElectron(n_e, T_K, tau_e);
    d_vals(IKPERP_E0)   = bt::ConductivityPerpElectron(n_e, T_K, tau_e, 0.0);
    d_vals(IKPERP_E100) = bt::ConductivityPerpElectron(n_e, T_K, tau_e, 100.0);
    d_vals(IKPERP_E200) = bt::ConductivityPerpElectron(n_e, T_K, tau_e, 200.0);
    d_vals(IKPAR_I)     = bt::ConductivityParIon(n_e, T_K, tau_i, m_i);
    d_vals(IKPERP_I0)   = bt::ConductivityPerpIon(n_e, T_K, tau_i, m_i, 0.0);

    d_vals(IETA_PAR)      = bt::ResistivityParElectron(n_e, tau_e);
    d_vals(IETA_PERP0)    = bt::ResistivityPerpElectron(n_e, tau_e, 0.0);
    d_vals(IETA_PERP1000) = bt::ResistivityPerpElectron(n_e, tau_e, 1000.0);

    // Base scales for the dimensionless-coefficient checks.
    d_vals(IBASE_E) = n_e * SQR(bt::kBoltzmann) * T_K * tau_e / bt::kElectronMass;
    d_vals(IBASE_I) = n_e * SQR(bt::kBoltzmann) * T_K * tau_i / m_i;
    d_vals(IETA0)   = bt::kElectronMass / (n_e * SQR(bt::kElementaryCharge) * tau_e);

    Real omega_ce = bt::ElectronGyroFrequency(b_mag);
    d_vals(IBOHM_D)   = bt::BohmDiffusivity(T_K, b_mag);
    d_vals(IBOHM_NU)  = bt::BohmCollisionFrequency(omega_ce);
    d_vals(IOMEGA_CE) = omega_ce;
    d_vals(IMAGDIFF)  = bt::MagneticDiffusivity(d_vals(IETA_PAR));

    d_vals(IFLOOR_BELOW) = bt::ApplyVacuumResistivityFloor(eta_in, 0.999*rho_floor,
                                                           rho_floor, eta_vac);
    d_vals(IFLOOR_ABOVE) = bt::ApplyVacuumResistivityFloor(eta_in, 1.001*rho_floor,
                                                           rho_floor, eta_vac);
    d_vals(ICLOG_HOT)  = bt::CoulombLog(n_e, T_K, Zbar);
    d_vals(ICLOG_COLD) = bt::CoulombLog(n_e, T_cold_K, Zbar);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // (1) Collision times vs the NRL practical Braginskii values:
  //     tau_e = 3.44e5 T[eV]^1.5/(n_e[cm^-3] Z lnL), tau_i = 2.09e7 T^1.5 sqrt(mu)/(...).
  Real n_e_cm3 = n_e * 1.0e-6;
  Real tau_e_ref = 3.44e5 * std::pow(T_eV, 1.5) / (n_e_cm3 * Zbar * lnL);
  Real mu = m_i / bt::kProtonMass;
  Real tau_i_ref = 2.09e7 * std::pow(T_eV, 1.5) * std::sqrt(mu) /
                   (n_e_cm3 * std::pow(Zbar, 4.0) * lnL);
  test.CheckNear(h_vals(ITAU_E), tau_e_ref, 2.0e-2, 0.0, "tau_e == NRL Braginskii value");
  test.CheckNear(h_vals(ITAU_I), tau_i_ref, 2.0e-2, 0.0, "tau_i == NRL Braginskii value");

  // (2) Parallel coefficients equal the Braginskii Z=1 dimensionless constants * base.
  test.CheckNear(h_vals(IKPAR_E), 3.1616*h_vals(IBASE_E), 1.0e-10, 0.0,
                 "kappa_par_e == gamma0_e (3.1616) * base");
  test.CheckNear(h_vals(IKPAR_I), 3.906*h_vals(IBASE_I), 1.0e-10, 0.0,
                 "kappa_par_i == gamma0_i (3.906) * base");
  test.CheckNear(h_vals(IETA_PAR), 0.5129*h_vals(IETA0), 1.0e-10, 0.0,
                 "eta_par == alpha0 (0.5129) * eta0");

  // (3) Parallel resistivity vs the practical Spitzer value 0.51*1.03e-4 Z lnL/T[eV]^1.5.
  Real eta_par_ref = 0.5129 * 1.03e-4 * Zbar * lnL / std::pow(T_eV, 1.5);
  test.CheckNear(h_vals(IETA_PAR), eta_par_ref, 2.0e-2, 0.0,
                 "eta_par == Spitzer parallel resistivity");

  // (4) Unmagnetized limit (x->0): the perpendicular fit reduces to the parallel value.
  //     The agreement is to ~2.5e-4, not machine precision, because the published
  //     Braginskii parallel constant (e.g. gamma0_e=3.1616) and the perpendicular fit's
  //     x->0 limit (gamma0'_e/delta0_e = 11.92/3.7703 = 3.16155) are rounded apart.
  test.CheckNear(h_vals(IKPERP_E0), h_vals(IKPAR_E), 1.0e-3, 0.0,
                 "kappa_perp_e(x=0) -> kappa_par_e");
  test.CheckNear(h_vals(IKPERP_I0), h_vals(IKPAR_I), 1.0e-3, 0.0,
                 "kappa_perp_i(x=0) -> kappa_par_i");
  test.CheckNear(h_vals(IETA_PERP0), h_vals(IETA_PAR), 1.0e-3, 0.0,
                 "eta_perp(x=0) -> eta_par");

  // (5) Strongly-magnetized conduction: kappa_par/kappa_perp ~ x^2, so doubling x
  //     (100->200) quadruples the ratio; and kappa_perp << kappa_par.
  Real ratio100 = h_vals(IKPAR_E)/h_vals(IKPERP_E100);
  Real ratio200 = h_vals(IKPAR_E)/h_vals(IKPERP_E200);
  test.CheckNear(ratio200/ratio100, 4.0, 1.0e-2, 0.0,
                 "kappa_par/kappa_perp scales as (omega_c tau)^2");
  test.CheckTrue(h_vals(IKPERP_E100) < 1.0e-2*h_vals(IKPAR_E),
                 "kappa_perp_e strongly suppressed at x=100");

  // (6) Strongly-magnetized resistivity: eta_perp -> eta0, so eta_perp/eta_par->1/alpha0.
  test.CheckNear(h_vals(IETA_PERP1000)/h_vals(IETA_PAR), 1.0/0.5129, 5.0e-3, 0.0,
                 "eta_perp/eta_par -> 1/alpha0 (~1.95) at large x");

  // (7) Magnetic diffusivity is the electrical resistivity divided by mu0.
  test.CheckNear(h_vals(IMAGDIFF), h_vals(IETA_PAR)/bt::kVacuumMu0, 1.0e-12, 0.0,
                 "MagneticDiffusivity == eta/mu0");

  // (8) Bohm-like anomalous term: D = (1/16) k_B T/(e B), nu = omega_ce/16.
  Real bohm_d_ref = bt::kBoltzmann*T_K/(16.0*bt::kElementaryCharge*b_mag);
  test.CheckNear(h_vals(IBOHM_D), bohm_d_ref, 1.0e-12, 0.0,
                 "BohmDiffusivity == k_B T/(16 e B)");
  test.CheckNear(h_vals(IBOHM_NU), h_vals(IOMEGA_CE)/16.0, 1.0e-12, 0.0,
                 "BohmCollisionFrequency == omega_ce/16");

  // (9) Vacuum-resistivity floor activates below rho_floor and is inert above it.
  test.CheckNear(h_vals(IFLOOR_BELOW), eta_vac, 0.0, 0.0,
                 "vacuum floor: rho<rho_floor -> eta_vac");
  test.CheckNear(h_vals(IFLOOR_ABOVE), eta_in, 0.0, 0.0,
                 "vacuum floor: rho>rho_floor -> classical eta");

  // (10) Coulomb log: NRL high-T branch value, and the >=1 floor in the cold/dense limit.
  Real clog_ref = 24.0 - std::log(std::sqrt(n_e_cm3) / T_eV);
  test.CheckNear(h_vals(ICLOG_HOT), clog_ref, 1.0e-2, 0.0,
                 "CoulombLog matches NRL high-T branch");
  test.CheckNear(h_vals(ICLOG_COLD), 1.0, 1.0e-12, 0.0,
                 "CoulombLog floored at 1 in cold/dense limit");

  test.Finish();
  return;
}
