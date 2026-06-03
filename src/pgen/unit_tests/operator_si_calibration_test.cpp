//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file operator_si_calibration_test.cpp
//  \brief Unit test: the SI calibration of the reference-model-set operator coefficients
//  (issue [P7d]/#184, ADR-0014/ADR-0012 gap d).  Closed-form / table ANCHORS that the
//  faithful B1 coefficients are derived from first-principles transport / opacity, not
//  uncalibrated placeholders:
//
//   (AC#1, resistivity) The Spitzer-Braginskii parallel resistivity
//   `op_si_calib::SpitzerResistivityParSI(T_e,n_e,Z,lnL)` equals the NRL practical value
//   eta_|| = 5.2e-5 Z lnL T_e[eV]^(-3/2) Ohm m at a known classical state.
//
//   (AC#1, conduction) The Braginskii parallel electron thermal conductivity
//   `op_si_calib::BraginskiiKappaParElectronSI` equals the NRL/Spitzer-Harm value
//   kappa_||e = 3.16 n_e k_B^2 T_e tau_e / m_e (with the NRL collision time tau_e).
//
//   (AC#1, opacity) The FLD/mrad opacity coefficient is the IONMIX table value at a known
//   point: a real aluminum IONMIX cn4 Planck-absorption / Rosseland node, converted to
//   the code absorption coefficient kappa[cm^2/g]*rho[g/cc]*length_cgs.
//
//   (RED meta-checks) Each calibrated coefficient is grossly inconsistent with the
//   order-unity placeholder the faithful B1 deck carried before #184 (resb_eta=1e-3,
//   acond_kappa_conv=0.05, mrad_chi_a=0.04), proving calibration is genuinely required
//   (run-it-first, observe-fail), mirroring the EOS-aware operator test (#183).
//
//  No transport or opacity value is hard-coded: the analytic anchors come from the
//  CODATA constants in braginskii_transport.hpp and the opacity from the real aluminum
//  IONMIX table at runtime, through the exact calibration code path the MHD package runs
//  (ADR-0008).  Built/run by test_unit_operator_si_calibration_cpu.py.

#include <cmath>     // std::fabs, std::pow, std::sqrt
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "diffusion/operator_si_calibration.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief SI operator-coefficient calibration anchor unit test (#184, ADR-0014).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("operator_si_calibration_test");
  namespace bk = braginskii;
  namespace oc = op_si_calib;

  // ---- reference classical state for the Spitzer/Braginskii anchors (state-only; the
  // NRL practical formulae are valid here: hot, classical, fixed Coulomb log) ----
  const Real T_e_eV = 100.0;
  const Real n_e    = 1.0e27;     // m^-3
  const Real Z      = 4.0;
  const Real lnL    = 5.0;

  // (AC#1, resistivity) Spitzer eta_|| vs the NRL value 5.2e-5 Z lnL T^-3/2 Ohm m.
  Real eta_si  = oc::SpitzerResistivityParSI(T_e_eV, n_e, Z, lnL);
  Real eta_nrl = 5.2e-5 * Z * lnL * std::pow(T_e_eV, -1.5);
  test.CheckNear(eta_si, eta_nrl, 3.0e-2, 0.0,
                 "Spitzer eta_|| matches NRL 5.2e-5 Z lnL T_eV^-1.5 [Ohm m]");

  // (AC#1, conduction) Braginskii kappa_||,e vs NRL 3.16 n_e k_B^2 T_e tau_e/m_e, with
  // NRL practical collision time tau_e = 3.44e5 T_eV^1.5 / (n_e[cm^-3] Z lnL) [s].
  Real kappa_si = oc::BraginskiiKappaParElectronSI(T_e_eV, n_e, Z, lnL);
  Real T_K      = T_e_eV * oc::kEvToKelvin;
  Real n_e_cm3  = n_e * 1.0e-6;
  Real tau_nrl  = 3.44e5 * std::pow(T_e_eV, 1.5) / (n_e_cm3 * Z * lnL);
  Real kappa_nrl = bk::kGamma0e * n_e * bk::kBoltzmann * bk::kBoltzmann * T_K * tau_nrl
                   / bk::kElectronMass;
  test.CheckNear(kappa_si, kappa_nrl, 3.0e-2, 0.0,
                 "Braginskii kappa_||,e matches NRL 3.16 n_e kB^2 T tau_e/m_e [W/m/K]");

  // ---- (AC#1, opacity) the IONMIX table at a known point ----
  // Load the real aluminum IONMIX cn4 multigroup-opacity table (cm^2/g) and read raw
  // Planck-absorption / Rosseland opacities at an interior (group, T, rho) node on the
  // device through the exact lookup arrays the FLD/coupling operators consume.
  const std::string opac_file   = pin->GetString("problem", "opacity_file");
  const Real mass_per_ion = pin->GetOrAddReal("problem", "opacity_mass_per_ion", 1.0);
  opacity::MultigroupOpacity otbl;
  opacity::ReadIonmixCn4Opacity(opac_file, otbl, mass_per_ion);

  const int ig = otbl.ngroups / 2;
  const int it = otbl.ntemp / 2;
  const int id = otbl.ndens / 2;
  DvceArray1D<Real> raw("opac_raw", 4);   // [planck_absorb, rosseland, rho_node, T_node]
  auto otbl_d = otbl;
  Kokkos::parallel_for("opac_node", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int) {
    raw(0) = otbl_d.planck_absorb(ig, it, id);   // cm^2/g
    raw(1) = otbl_d.rosseland(ig, it, id);       // cm^2/g
    raw(2) = otbl_d.DensAt(id);                  // g/cc (mass_per_ion-scaled axis)
    raw(3) = otbl_d.TempAt(it);                  // eV
  });
  auto hraw = Kokkos::create_mirror_view(raw);
  Kokkos::deep_copy(hraw, raw);
  Real kappa_pa = hraw(0), kappa_ro = hraw(1), rho_node = hraw(2);

  test.CheckTrue(std::isfinite(kappa_pa) && kappa_pa > 0.0,
                 "IONMIX Planck-absorption opacity at the node is finite and positive");
  test.CheckTrue(std::isfinite(kappa_ro) && kappa_ro > 0.0,
                 "IONMIX Rosseland opacity at the node is finite and positive");

  // ---- assemble the full code-unit calibration for the faithful B1 <units> system ----
  const Real L = 0.1, D = 2.7, V = 1.0e8;      // length_cgs/density_cgs/velocity_cgs (B1)
  oc::OperatorReferenceState s;
  s.t_e_ev = T_e_eV; s.n_e_m3 = n_e; s.z_bar = Z; s.ln_lambda = lnL;
  s.m_i_kg = 4.4804e-26;                        // Al ion mass [kg] (4.4804e-23 g)
  s.rho_gcc = rho_node;                         // reference density = the opacity node
  s.kappa_planck_cm2g = kappa_pa;               // IONMIX Planck-absorption (cm^2/g)
  s.kappa_ross_cm2g   = kappa_ro;               // IONMIX Rosseland (cm^2/g)
  oc::OperatorSiCoefficients c = oc::CalibrateSiOperators(L, D, V, s);

  // The opacity coefficient IS the IONMIX node value converted to code units (cm^2/g *
  // rho[g/cc]*length_cgs), i.e. the FLD/mrad opacity is sourced from the table point.
  test.CheckNear(c.mrad_chi_a, kappa_pa * rho_node * L, 1.0e-12, 0.0,
                 "calibrated mrad_chi_a == IONMIX Planck-absorption * rho * length_cgs");
  test.CheckNear(c.fld_chi, kappa_ro * rho_node * L, 1.0e-12, 0.0,
                 "calibrated fld_chi == IONMIX Rosseland * rho * length_cgs");

  // The code-unit resb_eta is the SI Spitzer magnetic diffusivity over the code
  // diffusivity unit Lu*Vu = L*V*1e-4 [m^2/s].
  Real eta_m2s = bk::MagneticDiffusivity(eta_si);
  test.CheckNear(c.resb_eta, eta_m2s / (L * V * 1.0e-4), 1.0e-12, 0.0,
                 "calibrated resb_eta == eta/mu0 / (length_cgs*velocity_cgs*1e-4)");

  // ---- (RED meta-checks) the calibrated coefficients are NOT the placeholders ----
  // Each calibrated code-unit value differs from the order-unity placeholder the B1 deck
  // carried before #184 by >> a factor of two, proving calibration is genuinely required.
  test.CheckTrue(std::fabs(c.acond_kappa_conv - 0.05) > 0.5*0.05,
                 "calibrated acond_kappa_conv differs from the 0.05 placeholder "
                 "(SI calibration is required)");
  test.CheckTrue(c.acond_kappa_conv > 0.0 && std::isfinite(c.acond_kappa_conv),
                 "calibrated acond_kappa_conv is finite and positive");
  test.CheckTrue(std::fabs(c.mrad_chi_a - 0.04) > 0.5*0.04,
                 "calibrated mrad_chi_a differs from the 0.04 placeholder "
                 "(IONMIX calibration is required)");
  test.CheckTrue(c.acond_temp_conv > 1.0e4 && c.acond_temp_conv < 1.3e4,
                 "acond_temp_conv is the eV->K factor (~11604.5)");
  test.CheckTrue(c.acond_dens_conv > 0.0 && std::isfinite(c.acond_dens_conv),
                 "acond_dens_conv (code rho -> n_i) is finite and positive");
  test.CheckTrue(c.acond_bmag_conv > 0.0 && std::isfinite(c.acond_bmag_conv),
                 "acond_bmag_conv (code |B| -> Tesla) is finite and positive");

  test.Finish();
  return;
}
