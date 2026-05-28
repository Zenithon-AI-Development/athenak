//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file variable_resistivity_test.cpp
//  \brief Unit test: the Spitzer/Braginskii variable magnetic-diffusivity helper
//  (diffusion/variable_resistivity.hpp) evaluated on the device composes the SIM-61
//  transport chain (rho,T_e -> n_e,tau_e -> eta_par/eta_perp [Ohm m] -> eta/mu0 ->
//  code units) correctly, applies the linear code<->SI conversions, reproduces the
//  density-independence and T^-1.5 scaling of Spitzer resistivity, distinguishes
//  eta_perp >= eta_par under magnetization, and activates the vacuum-resistivity floor
//  below a density threshold (issue [7a], ADR-0003).
//
// Built/run by
//   cmake -D PROBLEM=unit_tests/variable_resistivity_test -B build_unit
//   (cd build_unit/src && make) && ./build_unit/src/athena \
//       -i inputs/unit_tests/variable_resistivity_test.athinput
// Auto-run by tst/test_suite/unit_tests/test_unit_variable_resistivity_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "diffusion/variable_resistivity.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace bt = braginskii;

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Variable Spitzer/Braginskii magnetic-diffusivity unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("variable_resistivity_test");

  // ---- Linear code<->SI conversions + plasma parameters (arbitrary but fixed) --------
  // Fixed ln(Lambda) (>0) so the chain is deterministic and the density-independence of
  // Spitzer resistivity is exact (CoulombLog is bypassed).
  resistivity::VarResistivityParams p;
  p.dens_si = 2.0;      // code density -> SI [kg m^-3]
  p.temp_si = 1.0e4;    // code temperature -> SI [K]
  p.bmag_si = 1.0;      // code |B| -> SI [Tesla]
  p.eta_si  = 7.0;      // SI magnetic diffusivity -> code
  p.zbar    = 1.0;
  p.m_i     = bt::kProtonMass;
  p.ln_lambda = 5.0;
  p.anisotropic = true;
  p.rho_floor = -1.0;   // floor disabled for the base state
  p.eta_vac   = 0.0;

  // Reference cell state (code units).  bmag_code large -> omega_ce*tau_e ~ O(1) so the
  // magnetized eta_perp clearly exceeds eta_par.
  const Real rho0  = 4.0;
  const Real temp0 = 100.0;
  const Real bmag0 = 1000.0;

  // Isotropic-flag variant of the same parameters (eta_perp must collapse onto eta_par).
  resistivity::VarResistivityParams p_iso = p;
  p_iso.anisotropic = false;

  // Vacuum-floor variant.
  resistivity::VarResistivityParams p_floor = p;
  p_floor.rho_floor = 0.5;     // code density threshold
  p_floor.eta_vac   = 1.0e6;   // code magnetic diffusivity assigned in vacuum

  enum {
    IPAR=0, IPERP, IEFF_ANISO, IEFF_ISO, IISO_PAR, IISO_PERP,
    IPAR_T1, IPAR_T2, IPAR_R1, IPAR_R2,
    IFLOOR_LO_PAR, IFLOOR_LO_PERP, IFLOOR_HI_PAR, NVAL
  };
  DvceArray1D<Real> d_vals("resist_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("resist_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    Real epar, eperp;
    resistivity::SpitzerMagneticDiffusivity(p, rho0, temp0, bmag0, epar, eperp);
    d_vals(IPAR)  = epar;
    d_vals(IPERP) = eperp;
    d_vals(IEFF_ANISO) = resistivity::EffectiveDiffusivity(p,     rho0, temp0, bmag0);
    d_vals(IEFF_ISO)   = resistivity::EffectiveDiffusivity(p_iso, rho0, temp0, bmag0);

    Real ipar, iperp;
    resistivity::SpitzerMagneticDiffusivity(p_iso, rho0, temp0, bmag0, ipar, iperp);
    d_vals(IISO_PAR)  = ipar;
    d_vals(IISO_PERP) = iperp;

    // T^-1.5 scaling: quadruple the temperature -> eta_par scales by 4^-1.5 = 1/8.
    Real a, b;
    resistivity::SpitzerMagneticDiffusivity(p, rho0, temp0,       bmag0, a, b);
    d_vals(IPAR_T1) = a;
    resistivity::SpitzerMagneticDiffusivity(p, rho0, 4.0*temp0,   bmag0, a, b);
    d_vals(IPAR_T2) = a;

    // density independence (fixed ln Lambda): eta_par is unchanged when rho changes.
    resistivity::SpitzerMagneticDiffusivity(p, rho0,     temp0, bmag0, a, b);
    d_vals(IPAR_R1) = a;
    resistivity::SpitzerMagneticDiffusivity(p, 9.0*rho0, temp0, bmag0, a, b);
    d_vals(IPAR_R2) = a;

    // vacuum floor: below threshold both components are eta_vac; above, classical
    Real fpar, fperp;
    resistivity::SpitzerMagneticDiffusivity(p_floor, 0.4, temp0, bmag0, fpar, fperp);
    d_vals(IFLOOR_LO_PAR)  = fpar;
    d_vals(IFLOOR_LO_PERP) = fperp;
    resistivity::SpitzerMagneticDiffusivity(p_floor, 4.0, temp0, bmag0, fpar, fperp);
    d_vals(IFLOOR_HI_PAR)  = fpar;
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // ---- Host reference: replay the exact SIM-61 chain (bt:: functions are host+device)
  Real n_i = (rho0 * p.dens_si) / p.m_i;
  Real n_e = p.zbar * n_i;
  Real T_e = temp0 * p.temp_si;
  Real B_si = bmag0 * p.bmag_si;
  Real tau_e = bt::ElectronCollisionTime(n_e, T_e, p.zbar, p.ln_lambda);
  Real eta_par_si  = bt::ResistivityParElectron(n_e, tau_e);
  Real x_e = bt::ElectronGyroFrequency(B_si) * tau_e;
  Real eta_perp_si = bt::ResistivityPerpElectron(n_e, tau_e, x_e);
  Real eta_par_ref  = bt::MagneticDiffusivity(eta_par_si)  * p.eta_si;
  Real eta_perp_ref = bt::MagneticDiffusivity(eta_perp_si) * p.eta_si;

  // (1) The helper reproduces the full chain to machine precision.
  test.CheckNear(h_vals(IPAR),  eta_par_ref,  1.0e-12, 0.0,
                 "eta_par  == MagneticDiffusivity(ResistivityParElectron)*eta_si");
  test.CheckNear(h_vals(IPERP), eta_perp_ref, 1.0e-12, 0.0,
                 "eta_perp == MagneticDiffusivity(ResistivityPerpElectron)*eta_si");

  // (2) Magnetization: eta_perp > eta_par (here x_e ~ O(1) so clearly resolved).
  test.CheckTrue(h_vals(IPERP) > 1.2*h_vals(IPAR),
                 "eta_perp > eta_par under magnetization (omega_ce*tau_e ~ O(1))");

  // (3) The isotropic flag collapses eta_perp onto eta_par (par is flag-independent).
  test.CheckNear(h_vals(IISO_PERP), h_vals(IISO_PAR), 1.0e-12, 0.0,
                 "anisotropic=false: eta_perp == eta_par");
  test.CheckNear(h_vals(IISO_PAR), h_vals(IPAR), 1.0e-12, 0.0,
                 "eta_par independent of the anisotropic flag");

  // (4) EffectiveDiffusivity selects eta_perp (anisotropic) / eta_par (isotropic).
  test.CheckNear(h_vals(IEFF_ANISO), h_vals(IPERP), 1.0e-12, 0.0,
                 "EffectiveDiffusivity(anisotropic) == eta_perp");
  test.CheckNear(h_vals(IEFF_ISO), h_vals(IISO_PAR), 1.0e-12, 0.0,
                 "EffectiveDiffusivity(isotropic) == eta_par");

  // (5) Spitzer T^-1.5 scaling: eta_par(4T)/eta_par(T) == 4^-1.5 = 0.125.
  test.CheckNear(h_vals(IPAR_T2)/h_vals(IPAR_T1), std::pow(4.0, -1.5), 1.0e-10, 0.0,
                 "eta_par scales as T^-1.5 (Spitzer)");

  // (6) Density independence of Spitzer resistivity at fixed ln(Lambda).
  test.CheckNear(h_vals(IPAR_R2), h_vals(IPAR_R1), 1.0e-12, 0.0,
                 "eta_par independent of density at fixed ln(Lambda)");

  // (7) Vacuum-resistivity floor: below rho_floor both components are eta_vac; above it,
  //     the classical Spitzer value is recovered.
  test.CheckNear(h_vals(IFLOOR_LO_PAR),  p_floor.eta_vac, 0.0, 0.0,
                 "vacuum floor: rho<rho_floor -> eta_par = eta_vac");
  test.CheckNear(h_vals(IFLOOR_LO_PERP), p_floor.eta_vac, 0.0, 0.0,
                 "vacuum floor: rho<rho_floor -> eta_perp = eta_vac");
  test.CheckNear(h_vals(IFLOOR_HI_PAR), eta_par_ref, 1.0e-12, 0.0,
                 "vacuum floor inert above rho_floor (classical eta_par)");

  test.Finish();
  return;
}
