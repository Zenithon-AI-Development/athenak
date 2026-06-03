#ifndef DIFFUSION_OPERATOR_SI_CALIBRATION_HPP_
#define DIFFUSION_OPERATOR_SI_CALIBRATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file operator_si_calibration.hpp
//! \brief SI calibration of reference-model-set operator coefficients for the faithful
//!  MagLIF run (issue [P7d]/#184, ADR-0014 -- the operator analogue of the ADR-0010 drive
//!  calibration, closing the ADR-0012 coefficient-calibration gap).
//!
//!  ADR-0012 closed the three operator *wiring* gaps (resb<->b0, FLD erad sourcing, the
//!  EOS-aware acond/mrad closures), but left every operator *coefficient* an uncalibrated
//!  placeholder in code units (`resb_eta`, `acond_*_conv`, `mrad_chi_a`, `fld_chi` were
//!  order-unity guesses).  This header derives each coefficient from first-principles
//!  transport / opacity and converts it into the AthenaK code-unit system fixed by the
//!  `<units>` block (`length_cgs` [cm], `density_cgs` [g/cc], `velocity_cgs` [cm/s],
//!  Heaviside-Lorentz `mu0=1`), exactly as `circuit::CalibrateSiDrive` (ADR-0010) does
//!  for the drive trace.
//!
//!  Like `braginskii_transport.hpp` and `CalibrateSiDrive`, this is intentionally pure --
//!  closed-form physics + linear unit factors, no grid, no I/O -- so the anchor unit test
//!  (`operator_si_calibration_test`) can recompute every value through the same code path
//!  and assert it against the analytic Spitzer/Braginskii formulae and the IONMIX opacity
//!  table (ADR-0008 Layer-1 independent oracle).
//!
//!  CODE-UNIT SYSTEM (the same one ADR-0010 fixes).  With `length_cgs` L [cm],
//!  `density_cgs` D [g/cc], `velocity_cgs` V [cm/s]:
//!    length   unit  Lu = L*1e-2          [m]
//!    density  unit  Du = D*1e3           [kg/m^3]
//!    velocity unit  Vu = V*1e-2          [m/s]
//!    time     unit  Tu = Lu/Vu = L/V     [s]
//!    pressure unit  Pu = Du*Vu^2 = 0.1*D*V^2   [Pa]   (= the ADR-0010 `p_unit_pa`)
//!  Every conversion below is built from these four, so a unit error in any breaks the
//!  corresponding anchor (mirroring the single magnetic-pressure anchor of ADR-0010).

#include "athena.hpp"                       // Real, SQR
#include "diffusion/braginskii_transport.hpp"

namespace op_si_calib {

//----------------------------------------------------------------------------------------
// eV <-> Kelvin (CODATA): the tabulated_3t temperature axis stays in eV (ADR-0010), so
// the EOS-aware conduction operator conducts that eV temperature; the `temp_conv`
// factor is exactly this eV->K conversion (k_B T = eV).
constexpr Real kEvToKelvin = braginskii::kElectronVolt/braginskii::kBoltzmann;  // ~11604

//----------------------------------------------------------------------------------------
//! \fn Real SpitzerResistivityParSI
//! \brief Spitzer-Braginskii parallel electrical resistivity eta_|| [Ohm m] at electron
//!  temperature `T_e_eV` [eV], electron density `n_e` [m^-3], mean ionization `Z` and
//!  Coulomb log `lnL`.  Reproduces the NRL practical value
//!  eta_|| ~= 5.2e-5 Z lnL T_e[eV]^(-3/2) Ohm m (density-independent, as Spitzer is).
//!  Pure wrapper over braginskii::{ElectronCollisionTime, ResistivityParElectron}.
KOKKOS_INLINE_FUNCTION
Real SpitzerResistivityParSI(Real T_e_eV, Real n_e, Real Z, Real lnL) {
  Real T_K   = T_e_eV * kEvToKelvin;
  Real tau_e = braginskii::ElectronCollisionTime(n_e, T_K, Z, lnL);
  return braginskii::ResistivityParElectron(n_e, tau_e);   // Ohm m
}

//----------------------------------------------------------------------------------------
//! \fn Real SpitzerMagneticDiffusivitySI
//! \brief Spitzer magnetic diffusivity eta/mu0 [m^2/s] -- the form the induction-equation
//!  resistive operator (`resb`) diffuses (d_t B = (eta/mu0) grad^2 B) -- at the reference
//!  state.  This is the SI value the code-unit `resb_eta` is converted FROM.
KOKKOS_INLINE_FUNCTION
Real SpitzerMagneticDiffusivitySI(Real T_e_eV, Real n_e, Real Z, Real lnL) {
  return braginskii::MagneticDiffusivity(SpitzerResistivityParSI(T_e_eV, n_e, Z, lnL));
}

//----------------------------------------------------------------------------------------
//! \fn Real BraginskiiKappaParElectronSI
//! \brief Braginskii parallel electron thermal conductivity kappa_||,e [W/m/K] at the
//!  reference state -- the dominant conduction channel (electron) the `acond` operator
//!  carries.  kappa = 3.16 n_e k_B^2 T_e tau_e / m_e (NRL/Spitzer-Harm), via
//!  braginskii::ConductivityParElectron.
KOKKOS_INLINE_FUNCTION
Real BraginskiiKappaParElectronSI(Real T_e_eV, Real n_e, Real Z, Real lnL) {
  Real T_K   = T_e_eV * kEvToKelvin;
  Real tau_e = braginskii::ElectronCollisionTime(n_e, T_K, Z, lnL);
  return braginskii::ConductivityParElectron(n_e, T_K, tau_e);  // W/m/K
}

//----------------------------------------------------------------------------------------
// ---- SI -> code unit conversions for each operator coefficient ----
// Each takes the (length_cgs, density_cgs, velocity_cgs) of the run's <units> block.

//! \fn Real EtaCodeFromSI
//! \brief Convert a magnetic diffusivity eta [m^2/s] to the code diffusivity unit
//!  Lu*Vu = L*V*1e-4 [m^2/s], i.e. the code-unit `resb_eta`.
KOKKOS_INLINE_FUNCTION
Real EtaCodeFromSI(Real eta_m2s, Real length_cgs, Real velocity_cgs) {
  return eta_m2s / (length_cgs * velocity_cgs * 1.0e-4);
}

//! \fn Real AcondDensConv
//! \brief code mass density -> ion number density n_i [m^-3] (the `acond_dens_conv`):
//!  n_i = rho_code * Du / m_i, so the factor is Du/m_i = D*1e3 / m_i[kg].
KOKKOS_INLINE_FUNCTION
Real AcondDensConv(Real density_cgs, Real m_i_kg) {
  return (density_cgs * 1.0e3) / m_i_kg;
}

//! \fn Real AcondTempConv
//! \brief code (eos-aware) temperature [eV] -> [K] (the `acond_temp_conv`): the
//!  tabulated_3t conducted temperature is in eV (ADR-0010), so this is exactly eV->K.
KOKKOS_INLINE_FUNCTION
Real AcondTempConv() { return kEvToKelvin; }

//! \fn Real AcondBmagConv
//! \brief code |B| -> [Tesla] (the `acond_bmag_conv`).  In Heaviside-Lorentz code units
//!  magnetic pressure is B_code^2/2 over the code pressure unit Pu, and the SI magnetic
//!  pressure is B_SI^2/(2 mu0_SI), so B_SI = B_code*sqrt(mu0_SI*Pu) -- the same relation
//!  ADR-0010 drive current calibration uses (consistency of the field/current units).
KOKKOS_INLINE_FUNCTION
Real AcondBmagConv(Real density_cgs, Real velocity_cgs) {
  Real p_unit_pa = 0.1 * density_cgs * velocity_cgs * velocity_cgs;     // = Du*Vu^2 [Pa]
  return Kokkos::sqrt(braginskii::kVacuumMu0 * p_unit_pa);
}

//! \fn Real AcondKappaConv
//! \brief SI thermal conductivity [W m^-1 K^-1] -> code conduction coefficient (the
//!  `acond_kappa_conv`).  The operator forms q = -kappa dT/dx with T in eV, x/t/energy
//!  in code units; matching the SI energy-density rate d/dx(kappa dT/dx) gives
//!  kappa_code = kappa_SI * (eV->K) / (Lu*Vu*Pu).  Derivation in ADR-0014.
KOKKOS_INLINE_FUNCTION
Real AcondKappaConv(Real length_cgs, Real density_cgs, Real velocity_cgs) {
  Real lu_m      = length_cgs * 1.0e-2;                                 // Lu [m]
  Real vu_ms     = velocity_cgs * 1.0e-2;                               // Vu [m/s]
  Real p_unit_pa = 0.1 * density_cgs * velocity_cgs * velocity_cgs;     // Pu [Pa]
  return kEvToKelvin / (lu_m * vu_ms * p_unit_pa);
}

//! \fn Real OpacityCodeFromCgs
//! \brief Convert an IONMIX mass opacity kappa [cm^2/g] at density rho [g/cc] to the code
//!  absorption coefficient [1/code-length] (the `mrad_chi_a` / `fld_chi`).  The CGS
//!  absorption coefficient is chi = kappa*rho [1/cm]; one code length is L cm, so
//!  chi_code = kappa*rho*L.
KOKKOS_INLINE_FUNCTION
Real OpacityCodeFromCgs(Real kappa_cm2g, Real rho_gcc, Real length_cgs) {
  return kappa_cm2g * rho_gcc * length_cgs;
}

//----------------------------------------------------------------------------------------
//! \struct OperatorSiCoefficients
//! \brief The SI-calibrated reference-model-set operator coefficients in code units,
//!  to drop into the MHD operator constructors in place of the ADR-0012 placeholders.
struct OperatorSiCoefficients {
  Real resb_eta;          //!< Spitzer magnetic diffusivity (code) at the reference state
  Real acond_dens_conv;   //!< code density -> n_i [m^-3]
  Real acond_temp_conv;   //!< code temperature [eV] -> [K]
  Real acond_bmag_conv;   //!< code |B| -> [Tesla]
  Real acond_kappa_conv;  //!< SI kappa [W/m/K] -> code conduction coefficient
  Real mrad_chi_a;        //!< code absorption coefficient (IONMIX Planck-absorption)
  Real fld_chi;           //!< code absorption coefficient (IONMIX Rosseland transport)
};

//----------------------------------------------------------------------------------------
//! \struct OperatorReferenceState
//! \brief The reference plasma state the (state-dependent) coefficients are evaluated at,
//!  plus the representative IONMIX mass opacities [cm^2/g] for the run material.
//!  Resistivity and conduction depend on (T_e, Z, lnL) (density-independent for Spitzer
//!  eta); the opacity coefficients scale with the reference mass density.
struct OperatorReferenceState {
  Real t_e_ev;            //!< reference electron temperature [eV]
  Real n_e_m3;            //!< reference electron number density [m^-3]
  Real z_bar;             //!< mean ionization Z
  Real ln_lambda;         //!< Coulomb logarithm
  Real m_i_kg;            //!< ion mass [kg]
  Real rho_gcc;           //!< reference mass density [g/cc] (opacity-coefficient scale)
  Real kappa_planck_cm2g; //!< representative IONMIX Planck-absorption opacity [cm^2/g]
  Real kappa_ross_cm2g;   //!< representative IONMIX Rosseland transport opacity [cm^2/g]
};

//----------------------------------------------------------------------------------------
//! \fn OperatorSiCoefficients CalibrateSiOperators
//! \brief Compute all SI-calibrated operator coefficients in code units from the run's
//!  `<units>` system (length_cgs, density_cgs, velocity_cgs) and the reference plasma
//!  state.  This is the single boundary where physical transport/opacity values cross
//!  the code unit system, mirroring `circuit::CalibrateSiDrive` (ADR-0010/0014).
inline OperatorSiCoefficients CalibrateSiOperators(Real length_cgs, Real density_cgs,
                                                   Real velocity_cgs,
                                                   const OperatorReferenceState &s) {
  OperatorSiCoefficients c;
  Real eta_m2s = SpitzerMagneticDiffusivitySI(s.t_e_ev, s.n_e_m3, s.z_bar, s.ln_lambda);
  c.resb_eta         = EtaCodeFromSI(eta_m2s, length_cgs, velocity_cgs);
  c.acond_dens_conv  = AcondDensConv(density_cgs, s.m_i_kg);
  c.acond_temp_conv  = AcondTempConv();
  c.acond_bmag_conv  = AcondBmagConv(density_cgs, velocity_cgs);
  c.acond_kappa_conv = AcondKappaConv(length_cgs, density_cgs, velocity_cgs);
  c.mrad_chi_a       = OpacityCodeFromCgs(s.kappa_planck_cm2g, s.rho_gcc, length_cgs);
  c.fld_chi          = OpacityCodeFromCgs(s.kappa_ross_cm2g,   s.rho_gcc, length_cgs);
  return c;
}

}  // namespace op_si_calib

#endif  // DIFFUSION_OPERATOR_SI_CALIBRATION_HPP_
