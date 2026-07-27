//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.cpp
//! \brief implementation of MHD class constructor and assorted functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "eos/ionmix_eos_reader.hpp"  // tabulated 3T (IONMIX) EOS reader (#159)
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "radiation_fld/local_grey_opacity.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "diffusion/operator_si_calibration.hpp"
#include "diffusion/resistive_bphi_operator.hpp"
#include "driver/composite_parabolic_operator.hpp"
#include "mhd/mhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

MHD::MHD(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack),
    u0("cons",1,1,1,1,1),
    w0("prim",1,1,1,1,1),
    b0("B_fc",1,1,1,1),
    bcc0("B_cc",1,1,1,1,1),
    coarse_u0("ccons",1,1,1,1,1),
    coarse_w0("cprim",1,1,1,1,1),
    coarse_b0("cB_fc",1,1,1,1),
    u1("cons1",1,1,1,1,1),
    b1("B_fc1",1,1,1,1),
    uflx("uflx",1,1,1,1,1),
    efld("efld",1,1,1,1),
    wsaved("wsaved",1,1,1,1,1),
    bccsaved("bccsaved",1,1,1,1,1),
    e3x1("e3x1",1,1,1,1),
    e2x1("e2x1",1,1,1,1),
    e1x2("e1x2",1,1,1,1),
    e3x2("e3x2",1,1,1,1),
    e2x3("e2x3",1,1,1,1),
    e1x3("e1x3",1,1,1,1),
    e1_cc("e1_cc",1,1,1,1),
    e2_cc("e2_cc",1,1,1,1),
    e3_cc("e3_cc",1,1,1,1),
    utest("utest",1,1,1,1,1),
    bcctest("bcctest",1,1,1,1,1),
    erad("erad",1,1,1,1,1),
    erad_mg("erad_mg",1,1,1,1,1),
    bphi("bphi",1,1,1,1,1),
    eta_resb("eta_resb",1,1,1,1),
    fofc("fofc",1,1,1,1) {
  // Total number of MeshBlocks on this rank to be used in array dimensioning
  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));

  // (1) construct EOS object (no default)
  std::string eqn_of_state = pin->GetString("mhd","eos");
  // ideal gas EOS
  if (eqn_of_state.compare("ideal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic) {
      peos = new IdealSRMHD(ppack, pin);
    } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
      // DynGRMHD uses PrimitiveSolver instead, so use a no-op here.
      peos = new NoOpDynGRMHD(ppack, pin);
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      peos = new IdealGRMHD(ppack, pin);
    } else {
      peos = new IdealMHD(ppack, pin);
    }
    nmhd = 5;

  // isothermal EOS
  } else if (eqn_of_state.compare("isothermal") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
                <<"<mhd> eos = isothermal cannot be used with SR/GR"<< std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      peos = new IsothermalMHD(ppack, pin);
      nmhd = 4;
    }

  // tabulated 3T (IONMIX) real-material EOS (issue [P1]/#159, ADR-0002/0007).  Reads the
  // .cn4 table path from <mhd>/eos_table into the EosTable3T held on the package via
  // ionmix_eos_reader; the live cons->prim closure (eos/tabulated_mhd.cpp) inverts the
  // per-species energies to T_e/T_i.  Nonrelativistic only (HED/MagLIF code path).
  } else if (eqn_of_state.compare("tabulated_3t") == 0) {
    if (pmy_pack->pcoord->is_special_relativistic ||
        pmy_pack->pcoord->is_general_relativistic) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
                <<"<mhd> eos = tabulated_3t cannot be used with SR/GR"<< std::endl;
      std::exit(EXIT_FAILURE);
    }
    // .cn4 table path (required) + optional ion mass [g] to relabel the IONMIX number-
    // density axis as mass density (default 1.0 keeps the native number-density axis).
    std::string eos_table_file = pin->GetString("mhd","eos_table");
    Real eos_mass_per_ion = pin->GetOrAddReal("mhd","eos_mass_per_ion",1.0);
    eos_table_3t::ReadIonmixCn4Eos(eos_table_file, eos_tbl, eos_mass_per_ion);
    // Scale the table into the code-unit system fixed by <units> ONCE at load (ADR-0010,
    // the EOS-table unit boundary): specific energy /velocity_cgs^2, pressure
    // /(density_cgs*velocity_cgs^2), density axis /density_cgs; temp axis stays eV.
    // After this every lookup -- the live ConsToPrim2T AND the pgen initial conditions --
    // is code-unit native.  Defaults (1,1) are an identity, so a table consumed in
    // physical units (no <units> block) is unchanged.
    Real u_rho = pin->GetOrAddReal("units","density_cgs",1.0);
    Real u_vel = pin->GetOrAddReal("units","velocity_cgs",1.0);
    eos_tbl.ScaleToCodeUnits(u_rho, u_vel);
    // #209: opt-in zero-temperature Fermi degeneracy-pressure floor on the electron
    // pressure.  The cn4 table is ideal-ion nkT with a near-zero cold Zbar AND its
    // density axis edge-clamps (dP/drho = 0 off-table), so a magnetically driven cold
    // shell is pressureless and its peak density diverges with resolution.  Z* is the
    // COLD (valence) ionization for n_e = Z* rho/m_ion (Al: 3); the floor vanishes at
    // and below eos_deg_rho0 (the solid reference) so the quiescent pre-drive liner is
    // untouched.  Enabled BEFORE the eos_data.eos_tbl copy below so the cons->prim
    // closure and the Riemann/newdt interface pressure carry the same floor.
    Real deg_zstar = pin->GetOrAddReal("mhd","eos_deg_zstar",0.0);
    if (deg_zstar > 0.0) {
      Real deg_rho0 = pin->GetOrAddReal("mhd","eos_deg_rho0",1.0);
      eos_tbl.SetDegeneracyFloor(deg_zstar, eos_mass_per_ion, u_rho, u_vel, deg_rho0);
    }
    peos = new TabulatedMHD(ppack, pin);
    // hand the populated table to the EOS data so the live hyperbolic pressure/wave-speed
    // (LLF Riemann solver + newdt) can close p_gas from the table (#162); the cons->prim
    // closure reads the package member pmhd->eos_tbl directly.
    peos->eos_data.eos_tbl = eos_tbl;
    nmhd = 5;

  // EOS string not recognized
  } else {
    std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line "<< __LINE__ << std::endl
              <<"<mhd> eos = '"<< eqn_of_state <<"' not implemented"<< std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) Initialize scalars, diffusion, source terms
  nscalars = pin->GetOrAddInteger("mhd","nscalars",0);

  // Viscosity (only constructed if needed)
  if (pin->DoesParameterExist("mhd","viscosity")) {
    pvisc = new Viscosity("mhd", ppack, pin);
  } else {
    pvisc = nullptr;
  }

  // Resistivity (only constructed if needed): legacy constant "ohmic_resistivity" or the
  // Spitzer/Braginskii variable eta(rho,T_e) selected via "resistivity" (ADR-0003).
  if (pin->DoesParameterExist("mhd","ohmic_resistivity") ||
      pin->DoesParameterExist("mhd","resistivity")) {
    presist = new Resistivity(ppack, pin);
  } else {
    presist = nullptr;
  }

  // Thermal conduction (only constructed if needed)
  if (pin->DoesParameterExist("mhd","conductivity") ||
      pin->DoesParameterExist("mhd","tdep_conductivity")) {
    pcond = new Conduction("mhd", ppack, pin);
  } else {
    pcond = nullptr;
  }

  // Source terms (if needed)
  if (pin->DoesBlockExist("mhd_srcterms")) {
    psrc = new SourceTerms("mhd_srcterms", ppack, pin);
  }

  // (3) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  // allocate memory for conserved and primitive variables
  // With AMR, maximum size of Views are limited by total device memory through an input
  // parameter, which in turn limits max number of MBs that can be created.
  {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(u0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
    Kokkos::realloc(w0,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);

    // allocate memory for face-centered and cell-centered magnetic fields
    Kokkos::realloc(bcc0,   nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(b0.x1f, nmb, ncells3, ncells2, ncells1+1);
    Kokkos::realloc(b0.x2f, nmb, ncells3, ncells2+1, ncells1);
    Kokkos::realloc(b0.x3f, nmb, ncells3+1, ncells2, ncells1);

    // Derived, cached electron/ion temperature fields for the tabulated 3T EOS, allocated
    // ONLY when eos=tabulated_3t so the ideal/isothermal paths stay byte-identical.
    if (eqn_of_state.compare("tabulated_3t") == 0) {
      Kokkos::realloc(derived_te, nmb, 1, ncells3, ncells2, ncells1);
      Kokkos::realloc(derived_ti, nmb, 1, ncells3, ncells2, ncells1);
    }
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int n_ccells1 = indcs.cnx1 + 2*(indcs.ng);
    int n_ccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
    int n_ccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(coarse_u0, nmb, (nmhd+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_w0, nmb, (nmhd+nscalars), n_ccells3, n_ccells2, n_ccells1);
    Kokkos::realloc(coarse_b0.x1f, nmb, n_ccells3, n_ccells2, n_ccells1+1);
    Kokkos::realloc(coarse_b0.x2f, nmb, n_ccells3, n_ccells2+1, n_ccells1);
    Kokkos::realloc(coarse_b0.x3f, nmb, n_ccells3+1, n_ccells2, n_ccells1);
  }

  // allocate boundary buffers for conserved (cell-centered) and face-centered variables
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, false);
  pbval_u->InitializeBuffers((nmhd+nscalars));
  pbval_b = new MeshBoundaryValuesFC(ppack, pin);
  pbval_b->InitializeBuffers(3);

  // SI calibration of the reference-model-set operator coefficients (#184/[P7d],
  // closing the ADR-0012 coefficient-calibration gap d).  Default OFF => every operator
  // reads its placeholder coefficient via GetOrAddReal below, byte-identical to pre-#184.
  // When `operators_si_calibrate` is on (gated to the faithful tabulated_3t path, as
  // EOS-aware knobs of #183), each placeholder is REPLACED by the value derived from
  // first-principles Spitzer/Braginskii transport + the IONMIX opacity and converted into
  // the code-unit system fixed by <units> (the operator analogue of the ADR-0010 drive
  // calibration).  The reference plasma state (T_e, Z, lnLambda, ion mass, representative
  // IONMIX opacities) is read from <mhd> si_calib_* params.  This single boundary is
  // physical transport/opacity values cross into code units, like CalibrateSiDrive.
  bool operators_si_calibrate =
      pin->GetOrAddBoolean("mhd","operators_si_calibrate",false);
  if (operators_si_calibrate && (eqn_of_state.compare("tabulated_3t") != 0)) {
    std::cout << "### WARNING in " << __FILE__ << " line " << __LINE__
              << ": <mhd> operators_si_calibrate=true ignored (needs eos=tabulated_3t); "
              << "operators use the uncalibrated placeholder coefficients." << std::endl;
    operators_si_calibrate = false;
  }
  op_si_calib::OperatorSiCoefficients si_coeff{};
  if (operators_si_calibrate) {
    Real u_len = pin->GetOrAddReal("units","length_cgs",1.0);
    Real u_rho = pin->GetOrAddReal("units","density_cgs",1.0);
    Real u_vel = pin->GetOrAddReal("units","velocity_cgs",1.0);
    Real m_i_g = pin->GetOrAddReal("mhd","eos_mass_per_ion",1.0);  // ion mass [g]
    op_si_calib::OperatorReferenceState s;
    s.t_e_ev    = pin->GetOrAddReal("mhd","si_calib_te_ev", 100.0);
    s.z_bar     = pin->GetOrAddReal("mhd","si_calib_zbar", 1.0);
    s.ln_lambda = pin->GetOrAddReal("mhd","si_calib_lnlam", 5.0);
    s.m_i_kg    = m_i_g*1.0e-3;
    // reference electron density: the liner solid density (code density 1.0 = rho unit).
    Real n_i    = (u_rho*1.0e3)/s.m_i_kg;             // ion number density [m^-3]
    s.n_e_m3    = s.z_bar*n_i;
    s.rho_gcc   = u_rho;                                           // reference rho [g/cc]
    // representative IONMIX Rosseland / Planck-absorption mass opacities [cm^2/g] for the
    // run material (traceable to the cn4 table; see ADR-0014 + the B1 deck comments).
    s.kappa_planck_cm2g = pin->GetOrAddReal("mhd","si_calib_kappa_planck_cm2g", 1.0);
    s.kappa_ross_cm2g   = pin->GetOrAddReal("mhd","si_calib_kappa_ross_cm2g", 1.0);
    si_coeff = op_si_calib::CalibrateSiOperators(u_len, u_rho, u_vel, s);
  }

  // Grey flux-limited radiation diffusion (FLD) wired operator-split into the MHD step
  // (#110/[A3], ADR-0001).  Only built when explicitly enabled, so default MHD runs add
  // nothing and stay byte-identical.  `erad` is a standalone single-variable radiation
  // energy field; the FLDGreyOperator owns its own ghost-exchange boundary object (built
  // from `pin`) so it runs multi-block/MPI/AMR, mirroring hydro's operator-split
  // conduction (#108).  Routing decision (#109): FLD is stiff => super-time-stepped, so
  // it does not limit the hyperbolic dt (the RKL2 super-step covers the full step).
  fld_operator_split = pin->GetOrAddBoolean("mhd","fld_operator_split",false);
  if (fld_operator_split) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(erad, nmb, 1, ncells3, ncells2, ncells1);
    Real fld_c   = pin->GetOrAddReal("mhd","fld_c_light", 1.0);
    // SI-calibrated Rosseland opacity (#184) replaces the placeholder when calibrating.
    Real fld_chi = operators_si_calibrate ? si_coeff.fld_chi
                                          : pin->GetOrAddReal("mhd","fld_chi", 1.0);
    Real fld_nl  = pin->GetOrAddReal("mhd","fld_n_larsen", 2.0);
    Real fld_es  = pin->GetOrAddReal("mhd","fld_e_source", -1.0);
    // positivity floor for erad (#194): bounds the free-streaming read in the operator
    // and is the level the post-super-step projection floors negative cells to.
    Real fld_ef  = pin->GetOrAddReal("mhd","fld_efloor", 1.0e-30);
    // Lax-Friedrichs streaming-dissipation gate (#194): 1 => streaming-limit advective
    // mode (which RKL2 cannot damp) is stabilized; 0 => the bare centered flux (unstable
    // at imploding fronts).  Default 1 (the fix on); thick-limit byte-identical to tol.
    Real fld_up  = pin->GetOrAddReal("mhd","fld_upwind", 1.0);
    pfld_op = new FLDGreyOperator(ppack, pin, erad, fld_c, fld_chi, fld_nl, fld_es,
                                  fld_ef, fld_up);
    // #204: per-cell chi(rho,Te) lookup replacing the frozen reference-state constant
    // (which treated the tenuous vacuum gap at solid-liner opacity).  Requires the
    // tabulated_3t closure for the per-cell Te inversion -- warn + ignore otherwise
    // (the mrad_eos_aware precedent, #183).
    fld_opacity_local = pin->GetOrAddBoolean("mhd","fld_opacity_local",false);
    if (fld_opacity_local && (eqn_of_state.compare("tabulated_3t") != 0)) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                << ": <mhd> fld_opacity_local=true ignored (requires eos=tabulated_3t);"
                << " grey FLD uses the constant fld_chi." << std::endl;
      fld_opacity_local = false;
    }
    if (fld_opacity_local) {
      // the opacity records live in the SAME cn4 file as the EOS unless overridden;
      // eos_mass_per_ion rescales the cn4 ion-number-density axis to g/cc (as the EOS
      // reader does), so lookups take mass density in g/cc.
      std::string ofile = pin->GetOrAddString("mhd","fld_opacity_file",
                                              pin->GetString("mhd","eos_table"));
      Real mion = pin->GetOrAddReal("mhd","eos_mass_per_ion",1.0);
      opacity::ReadIonmixCn4Opacity(ofile, fld_opac_tbl, mion);
      fld_opac_dens_cgs = pin->GetOrAddReal("units","density_cgs",1.0);
      fld_opac_len_cgs  = pin->GetOrAddReal("units","length_cgs",1.0);
      fld_chi_floor     = pin->GetOrAddReal("mhd","fld_chi_floor",1.0e-10);
      Kokkos::realloc(fld_chi_cell, nmb, 1, ncells3, ncells2, ncells1);
      // defined (constant) values until the first per-super-step refresh
      Kokkos::deep_copy(fld_chi_cell, fld_chi);
      pfld_op->EnableLocalChi(fld_chi_cell);
    }
  }

  // Multigroup flux-limited radiation diffusion (FLD) wired operator-split into the MHD
  // step (#111/[A4], ADR-0001/ADR-0007).  Only built when explicitly enabled, so default
  // MHD runs add nothing and stay byte-identical.  `erad_mg` is a standalone per-group
  // radiation-energy field (var extent == the opacity table's group count), sized to
  // nmb_maxperrank so it survives an AMR regrid; the operator owns its own ghost-exchange
  // boundary object (built from `pin`) plus the coarse scratch reused by the regrid
  // restrict/prolong.  Stiff (#109) => super-time-stepped, so it does not limit the
  // hyperbolic dt (the RKL2 super-step covers the full step).
  mgfld_operator_split = pin->GetOrAddBoolean("mhd","mgfld_operator_split",false);
  if (mgfld_operator_split) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    // read the multigroup opacity table (group count + per-group Rosseland D)
    std::string mg_fname = pin->GetString("mhd","mgfld_opacity_file");
    opacity::MultigroupOpacity mg_table;
    opacity::ReadIonmixOpacity(mg_fname, mg_table);
    Kokkos::realloc(erad_mg, nmb, mg_table.ngroups, ncells3, ncells2, ncells1);
    Real mg_c   = pin->GetOrAddReal("mhd","mgfld_c_light", 1.0);
    Real mg_rho = pin->GetOrAddReal("mhd","mgfld_rho_bg", 1.0);
    Real mg_te  = pin->GetOrAddReal("mhd","mgfld_te_bg", 1.0);
    Real mg_nl  = pin->GetOrAddReal("mhd","mgfld_n_larsen", 2.0);
    Real mg_es  = pin->GetOrAddReal("mhd","mgfld_e_source", -1.0);
    // per-group positivity floor for erad (#197): bounds the free-streaming read in the
    // operator and is the level the post-super-step projection floors negative cells to.
    Real mg_ef  = pin->GetOrAddReal("mhd","mgfld_efloor", 1.0e-30);
    pmg_op = new FLDMultigroupOperator(ppack, pin, erad_mg, mg_table, mg_c, mg_rho, mg_te,
                                       mg_nl, mg_es, mg_ef);
  }

  // Anisotropic (magnetized) Braginskii electron+ion thermal conduction wired operator-
  // split into the MHD step (#112/[A5], ADR-0006/ADR-0001).  Only built when explicitly
  // enabled, so default MHD runs add nothing and stay byte-identical.  Unlike the FLD
  // operators it diffuses the LIVE conserved field u0 (IEN) field-aligned along the
  // frozen cell-centred B (bcc0), so it needs no standalone array.  The operator owns its
  // own ghost-exchange boundary object (built from `pin`) plus coarse scratch, so it runs
  // multi-block/MPI/AMR via SyncParabolicGhosts.  Stiff (#109) => super-time-stepped,
  // so it does not limit the hyperbolic dt (the RKL2 super-step covers the full step).
  acond_operator_split = pin->GetOrAddBoolean("mhd","acond_operator_split",false);
  if (acond_operator_split) {
    anisocond::AnisoCondParams apar;
    // When calibrating (#184), the local Braginskii state (Z, lnLambda, ion mass) is
    // reference state so the operator's kappa(rho,T) is consistent with the conv factors;
    // otherwise the per-param deck values (byte-identical to pre-#184).
    apar.zbar       = operators_si_calibrate
                          ? pin->GetOrAddReal("mhd","si_calib_zbar", 1.0)
                          : pin->GetOrAddReal("mhd","acond_zbar", 1.0);
    apar.mi_si      = operators_si_calibrate
                      ? pin->GetOrAddReal("mhd","eos_mass_per_ion",1.0)*1.0e-3
                      : pin->GetOrAddReal("mhd","acond_mi_si", braginskii::kProtonMass);
    apar.lnlam      = operators_si_calibrate
                          ? pin->GetOrAddReal("mhd","si_calib_lnlam", 5.0)
                          : pin->GetOrAddReal("mhd","acond_lnlam", 10.0);
    // SI-calibrated Braginskii code<->SI conversion factors (#184) replace placeholders
    // when calibrating; otherwise the per-param values (byte-identical to pre-#184).
    apar.dens_conv  = operators_si_calibrate ? si_coeff.acond_dens_conv
                                  : pin->GetOrAddReal("mhd","acond_dens_conv", 1.0);
    apar.temp_conv  = operators_si_calibrate ? si_coeff.acond_temp_conv
                                  : pin->GetOrAddReal("mhd","acond_temp_conv", 1.0);
    apar.bmag_conv  = operators_si_calibrate ? si_coeff.acond_bmag_conv
                                  : pin->GetOrAddReal("mhd","acond_bmag_conv", 1.0);
    apar.kappa_conv = operators_si_calibrate ? si_coeff.acond_kappa_conv
                                  : pin->GetOrAddReal("mhd","acond_kappa_conv", 1.0);
    apar.nlarsen    = pin->GetOrAddReal("mhd","acond_n_larsen", 2.0);
    apar.vfs        = pin->GetOrAddReal("mhd","acond_vfs", -1.0);
    apar.incl_e     = pin->GetOrAddBoolean("mhd","acond_incl_e", true);
    apar.incl_i     = pin->GetOrAddBoolean("mhd","acond_incl_i", true);
    Real agamma = peos->eos_data.gamma;
    // EOS-aware temperature recovery (#183/[P7c], ADR-0012 gap c).  Default off => the
    // operator recovers the ideal-gamma bookkeeping temperature T=(gamma-1)eint/rho, so
    // every existing acond run stays byte-identical.  On the faithful tabulated path the
    // `acond_eos_aware` knob makes conduction invert the conducted (electron) temperature
    // from the tabulated EOS closure (e_ele scalar -> T_e), thermodynamically consistent
    // with the faithful EOS rather than ideal-gamma.  Gated to tabulated: an ideal/
    // isothermal run leaves it off (the table is empty) and warns if it was requested.
    anisocond::EosAwareTemp etemp;
    bool acond_eos_aware = pin->GetOrAddBoolean("mhd","acond_eos_aware",false);
    bool tab3t = (eqn_of_state.compare("tabulated_3t") == 0);
    if (acond_eos_aware && !tab3t) {
      std::cout << "### WARNING in " << __FILE__ << " line " << __LINE__
                << ": <mhd> acond_eos_aware=true ignored (requires eos=tabulated_3t); "
                << "conduction uses the ideal-gamma temperature closure." << std::endl;
    }
    etemp.eos_aware = acond_eos_aware && tab3t;
    etemp.eele_idx  = nmhd;          // electron internal energy density rides scalar 0
    etemp.gm1       = agamma - 1.0;
    etemp.table     = eos_tbl;       // shallow View handle; only read when eos_aware
    pacond_op = new AnisotropicConductionOperator(ppack, pin, u0, bcc0, agamma, apar,
                                                  etemp);
  }

  // Cylindrical resistive B_phi diffusion (the -eta B_phi/r^2 curl-curl operator) wired
  // operator-split into the MHD step (#113/[A6], ADR-0004/ADR-0001).  Only built when
  // explicitly enabled, so default MHD runs add nothing and stay byte-identical.  `bphi`
  // is a standalone single-component field (B_phi) the operator diffuses, with a cell-
  // centred magnetic diffusivity `eta_resb` (here a constant fill -- the SIM-76/#20
  // Resistivity field is substituted in a fully-coupled run).  The operator owns its own
  // ghost-exchange object (built from `pin`) plus coarse scratch, so it runs multi-block/
  // MPI/AMR via SyncParabolicGhosts (#108) with the antisymmetric axis ghost preserved
  // only at the true r=0 face.  Stiff (#109) => super-time-stepped, so it does not limit
  // the hyperbolic dt (the RKL2 super-step covers the full step).
  resb_operator_split = pin->GetOrAddBoolean("mhd","resb_operator_split",false);
  if (resb_operator_split) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(bphi, nmb, 1, ncells3, ncells2, ncells1);
    Kokkos::realloc(eta_resb, nmb, ncells3, ncells2, ncells1);
    // SI-calibrated Spitzer magnetic diffusivity (#184) replaces the placeholder when on.
    Real resb_eta = operators_si_calibrate ? si_coeff.resb_eta
                                           : pin->GetOrAddReal("mhd","resb_eta", 1.0);
    Kokkos::deep_copy(eta_resb, resb_eta);
    presb_op = new ResistiveBphiOperator(ppack, pin, bphi, eta_resb);
    // Couple the standalone bphi to the live driven b0.x2f (#181/[P7a], ADR-0012 gap a).
    // Default off => the operator diffuses the decoupled scratch field as before (the
    // resb unit/verification pgens fill their own bphi IC, so they stay byte-identical);
    // the maglif faithful-B1 setup turns it on so resistivity acts on the driving field.
    resb_couple_b0 = pin->GetOrAddBoolean("mhd","resb_couple_b0",false);
    // Deposit the resistively-dissipated B_phi field energy as Ohmic heat on electrons
    // (#193).  Default off => the write-back drops the dissipated heat (#192 byte-
    // identical); the maglif faithful-B1 setup turns it on to close the energy budget
    // (field energy lost == electron heat gained).
    resb_deposit_heat = pin->GetOrAddBoolean("mhd","resb_deposit_heat",false);
  }

  // Strang-split orchestration of the coupled timestep (#115/[B2], ADR-0009).  Group the
  // active stiff operator-split operators into one CompositeParabolicOperator PER EVOLVED
  // FIELD (operators that share a field sum into one super-step; distinct fields get
  // distinct composites), then Strang-wrap the composite block around the hyperbolic
  // update -- a half super-step before the integrator and the other half after.  The
  // composite (ADR-0009) is what owns the GLOBAL min-dt MPI all-reduce, so every rank
  // derives the same RKL2 stage count (removing the per-operator deadlock risk, #114).
  // Default off => the individual full-step operator tasks above run unchanged
  // (byte-identical).  The per-field composites are heap-owned (deleted in the dtor).
  strang_split = pin->GetOrAddBoolean("mhd","strang_split",false);
  if (strang_split) {
    // erad (grey FLD) -- one composite per distinct evolved field.
    if (pfld_op != nullptr) {
      auto *c = new parabolic::CompositeParabolicOperator();
      c->AddOperator(pfld_op);
      strang_comps.push_back(c);
      strang_field_ids.push_back(SF_ERAD);
    }
    // erad_mg (multigroup FLD)
    if (pmg_op != nullptr) {
      auto *c = new parabolic::CompositeParabolicOperator();
      c->AddOperator(pmg_op);
      strang_comps.push_back(c);
      strang_field_ids.push_back(SF_ERADMG);
    }
    // u0 (anisotropic Braginskii conduction acts on the live conserved energy)
    if (pacond_op != nullptr) {
      auto *c = new parabolic::CompositeParabolicOperator();
      c->AddOperator(pacond_op);
      strang_comps.push_back(c);
      strang_field_ids.push_back(SF_U0);
    }
    // bphi (cylindrical resistive B_phi)
    if (presb_op != nullptr) {
      auto *c = new parabolic::CompositeParabolicOperator();
      c->AddOperator(presb_op);
      strang_comps.push_back(c);
      strang_field_ids.push_back(SF_BPHI);
    }
  }

  // Per-cell point-implicit grey matter-radiation coupling (#23), Strang-wrapped OUTSIDE
  // the super-step by the orchestration above.  Requires grey FLD (couples erad <-> gas
  // IEN).  Constant code-unit coefficients here; real opacity/EOS values come with
  // the IONMIX tables (Phase C, #118).  Default off => byte-identical.
  mrad_coupling = pin->GetOrAddBoolean("mhd","mrad_coupling",false);
  if (mrad_coupling) {
    // SI-calibrated IONMIX Planck-absorption opacity (#184) replaces placeholder when on.
    mrad_chi_a   = operators_si_calibrate ? si_coeff.mrad_chi_a
                                          : pin->GetOrAddReal("mhd","mrad_chi_a", 0.0);
    mrad_cv      = pin->GetOrAddReal("mhd","mrad_cv", 1.0);
    mrad_arad    = pin->GetOrAddReal("mhd","mrad_arad", 1.0);
    mrad_clight  = pin->GetOrAddReal("mhd","mrad_clight",
                                     (pfld_op != nullptr) ? pfld_op->c_light() : 1.0);
    // EOS-aware heat capacity (#183/[P7c], ADR-0012 gap c).  Default off => the coupling
    // uses the constant `mrad_cv` placeholder, byte-identical for existing mrad runs.
    // On the faithful tabulated_3t path the `mrad_eos_aware` knob makes the per-cell
    // volumetric heat capacity in the e_gas <-> erad exchange the tabulated closure value
    // rho*(c_v,e(rho,T_e) + c_v,i(rho,T_i)) instead of the constant -- the same EOS the
    // ConsToPrim2T closure uses.  Gated to the tabulated path (the table is empty
    // otherwise); warns if requested off the tabulated EOS.
    mrad_eos_aware = pin->GetOrAddBoolean("mhd","mrad_eos_aware",false);
    if (mrad_eos_aware && (eqn_of_state.compare("tabulated_3t") != 0)) {
      std::cout << "### WARNING in " << __FILE__ << " line " << __LINE__
                << ": <mhd> mrad_eos_aware=true ignored (requires eos=tabulated_3t); "
                << "coupling uses the constant mrad_cv." << std::endl;
      mrad_eos_aware = false;
    }
    // #204: per-cell Planck absorption chi_a(rho,Te) in place of the frozen constant
    // (which locked the near-massless vacuum gap to radiative equilibrium at solid
    // opacity).  Requires the tabulated_3t closure for the per-cell Te; shares the
    // opacity table + unit conversions with fld_opacity_local, loading them here when
    // the FLD knob did not.
    mrad_opacity_local = pin->GetOrAddBoolean("mhd","mrad_opacity_local",false);
    if (mrad_opacity_local && (eqn_of_state.compare("tabulated_3t") != 0)) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__
                << ": <mhd> mrad_opacity_local=true ignored (requires eos=tabulated_3t);"
                << " coupling uses the constant mrad_chi_a." << std::endl;
      mrad_opacity_local = false;
    }
    if (mrad_opacity_local && fld_opac_tbl.ngroups == 0) {
      std::string ofile = pin->GetOrAddString("mhd","fld_opacity_file",
                                              pin->GetString("mhd","eos_table"));
      Real mion = pin->GetOrAddReal("mhd","eos_mass_per_ion",1.0);
      opacity::ReadIonmixCn4Opacity(ofile, fld_opac_tbl, mion);
      fld_opac_dens_cgs = pin->GetOrAddReal("units","density_cgs",1.0);
      fld_opac_len_cgs  = pin->GetOrAddReal("units","length_cgs",1.0);
    }
  }

  // Orbital advection and shearing box BCs (if requested in input file)
  if (pin->DoesBlockExist("shearing_box")) {
    porb_u = new OrbitalAdvectionCC(ppack, pin, (nmhd+nscalars));
    porb_b = new OrbitalAdvectionFC(ppack, pin);
    psbox_u = new ShearingBoxCC(ppack, pin, (nmhd+nscalars));
    psbox_b = new ShearingBoxFC(ppack, pin);
  } else {
    porb_u = nullptr;
    porb_b = nullptr;
    psbox_u = nullptr;
    psbox_b = nullptr;
  }

  // for time-evolving problems, continue to construct methods, allocate arrays
  if (evolution_t.compare("stationary") != 0) {
    // determine if FOFC is enabled
    use_fofc = pin->GetOrAddBoolean("mhd","fofc",false);
    // FOFC's first-order LLF fallback does not yet carry the tabulated 3T electron-energy
    // scalar into the single-state solver, so it would not close the tabulated gas
    // pressure (#162).  FATAL rather than silently fall back to an ideal-gamma pressure.
    if (use_fofc && peos->eos_data.is_tabulated) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "<mhd> fofc=true not yet supported with eos=tabulated_3t (#162)"
        << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select reconstruction method (default PLM)
    std::string xorder = pin->GetOrAddString("mhd","reconstruct","plm");
    if (xorder.compare("dc") == 0) {
      recon_method = ReconstructionMethod::dc;
    } else if (xorder.compare("plm") == 0) {
      recon_method = ReconstructionMethod::plm;
      // check that nghost > 2 with PLM+FOFC
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (use_fofc && indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 3 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (xorder.compare("ppm4") == 0 ||
               xorder.compare("ppmx") == 0 ||
               xorder.compare("wenoz") == 0) {
      // check that nghost > 2
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      if (indcs.ng < 3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << xorder << " reconstruction requires at least 3 ghost zones, "
          << "but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // check that nghost > 3 with PPM4(or PPMX or WENOZ)+FOFC
      if (use_fofc && indcs.ng < 4) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "FOFC and " << xorder << " reconstruction requires at "
          << "least 4 ghost zones, but <mesh>/nghost=" << indcs.ng << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (xorder.compare("ppm4") == 0) {
        recon_method = ReconstructionMethod::ppm4;
      } else if (xorder.compare("ppmx") == 0) {
        recon_method = ReconstructionMethod::ppmx;
      } else if (xorder.compare("wenoz") == 0) {
        recon_method = ReconstructionMethod::wenoz;
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<mhd>/recon = '" << xorder << "' not implemented"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // select Riemann solver (no default).  Test for compatibility of options
    std::string rsolver = pin->GetString("mhd","rsolver");
    // Special relativistic solvers
    if (pmy_pack->pcoord->is_special_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_sr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_sr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for SR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for SR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // General relativistic solvers
    } else if (pmy_pack->pcoord->is_general_relativistic) {
      if (evolution_t.compare("dynamic") == 0) {
        if (rsolver.compare("llf") == 0) {
          rsolver_method = MHD_RSolver::llf_gr;
        } else if (rsolver.compare("hlle") == 0) {
          rsolver_method = MHD_RSolver::hlle_gr;
        // Error for anything else
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "<mhd> rsolver = '" << rsolver << "' not implemented"
                    << " for GR dynamics" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "kinematic dynamics not implemented for GR" <<std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic dynamic solvers
    } else if (evolution_t.compare("dynamic") == 0) {
      // The tabulated 3T EOS (#162) routes the hyperbolic gas pressure through the table
      // ONLY in the LLF solver (the one NR-MHD solver that takes the internal energy and
      // the pressure independently); HLLE/HLLD bake the ideal-gamma relation
      // e=p/(gamma-1) into their Roe-averaged wave structure, so they would silently fall
      // back to an ideal-gamma (here zero, iso_cs=0) pressure.  Warn rather than FATAL so
      // a nlim=0 construction/inspection run (no Riemann solver call) still works.
      if (peos->eos_data.is_tabulated && rsolver.compare("llf") != 0 &&
          global_variable::my_rank == 0) {
        std::cout << "### WARNING in " << __FILE__ << ": <mhd> eos=tabulated_3t routes"
                  << " the tabulated gas pressure through the hyperbolic update only with"
                  << " rsolver=llf; rsolver='" << rsolver << "' will use an ideal-gamma"
                  << " (iso_cs=0) pressure in the flux." << std::endl;
      }
      // LLF solver
      if (rsolver.compare("llf") == 0) {
        rsolver_method = MHD_RSolver::llf;
      // HLLE solver
      } else if (rsolver.compare("hlle") == 0) {
        rsolver_method = MHD_RSolver::hlle;
      // HLLD solver
      } else if (rsolver.compare("hlld") == 0) {
        rsolver_method = MHD_RSolver::hlld;
      // Roe solver
      // } else if (rsolver.compare("roe") == 0) {
      //   rsolver_method = MHD_RSolver::roe;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for dynamic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }

    // Non-relativistic kinematic solver
    } else {
      // Advect solver
      if (rsolver.compare("advect") == 0) {
        rsolver_method = MHD_RSolver::advect;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<mhd>/rsolver = '" << rsolver << "' not implemented"
                  << " for kinematic problems" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }

    // Final memory allocations
    {
      // allocate second registers
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      int ncells1 = indcs.nx1 + 2*(indcs.ng);
      int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
      int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(u1,     nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
      Kokkos::realloc(b1.x1f, nmb, ncells3, ncells2, ncells1+1);
      Kokkos::realloc(b1.x2f, nmb, ncells3, ncells2+1, ncells1);
      Kokkos::realloc(b1.x3f, nmb, ncells3+1, ncells2, ncells1);

      // allocate fluxes, electric fields
      Kokkos::realloc(uflx.x1f, nmb, (nmhd+nscalars), ncells3, ncells2, ncells1+1);
      Kokkos::realloc(uflx.x2f, nmb, (nmhd+nscalars), ncells3, ncells2+1, ncells1);
      Kokkos::realloc(uflx.x3f, nmb, (nmhd+nscalars), ncells3+1, ncells2, ncells1);
      Kokkos::realloc(efld.x1e, nmb, ncells3+1, ncells2+1, ncells1);
      Kokkos::realloc(efld.x2e, nmb, ncells3+1, ncells2, ncells1+1);
      Kokkos::realloc(efld.x3e, nmb, ncells3, ncells2+1, ncells1+1);

      // allocate scratch arrays for face- and cell-centered E used in CornerE
      Kokkos::realloc(e3x1, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2x1, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1x2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e3x2, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2x3, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1x3, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e1_cc, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e2_cc, nmb, ncells3, ncells2, ncells1);
      Kokkos::realloc(e3_cc, nmb, ncells3, ncells2, ncells1);

      // allocate array of flags used with FOFC
      if (use_fofc) {
        int nvars = (pmy_pack->pcoord->is_dynamical_relativistic) ? nmhd+nscalars : nmhd;
        Kokkos::realloc(fofc,    nmb, ncells3, ncells2, ncells1);
        Kokkos::realloc(utest,   nmb, nvars, ncells3, ncells2, ncells1);
        Kokkos::realloc(bcctest, nmb, 3,    ncells3, ncells2, ncells1);
        Kokkos::deep_copy(fofc, false);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
// destructor

MHD::~MHD() {
  if (psbox_b != nullptr) {delete psbox_b;}
  if (psbox_u != nullptr) {delete psbox_u;}
  if (porb_b != nullptr) {delete porb_b;}
  if (porb_u != nullptr) {delete porb_u;}
  delete pbval_b;
  delete pbval_u;
  if (pfld_op != nullptr) {delete pfld_op;}
  if (pmg_op != nullptr) {delete pmg_op;}
  if (pacond_op != nullptr) {delete pacond_op;}
  if (presb_op != nullptr) {delete presb_op;}
  for (auto *c : strang_comps) {delete c;}
  if (psrc!= nullptr) {delete psrc;}
  if (pcond != nullptr) {delete pcond;}
  if (presist!= nullptr) {delete presist;}
  if (pvisc != nullptr) {delete pvisc;}
  delete peos;
}

//----------------------------------------------------------------------------------------
// SetSaveWBcc:  set flag to save primitives and cell-centered B field, e.g., for jcon

void MHD::SetSaveWBcc() {
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  // allocated saved arrays for time derivatives
  Kokkos::realloc(wsaved,   nmb, (nmhd+nscalars), ncells3, ncells2, ncells1);
  Kokkos::realloc(bccsaved, nmb, 3,               ncells3, ncells2, ncells1);

  wbcc_saved = true;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::RefreshFldLocalChi()
//! \brief Re-fill the per-cell grey extinction field chi(rho,Te) from the live conserved
//! state (#204): rho = u0(IDN), Te from the tabulated_3t closure on the electron-energy
//! scalar, grey Rosseland mean of the multigroup opacity at the local state.  Called
//! once per RKL2 super-step at the operator-split call sites (the operator-split
//! background, hence chi, is frozen across the substages).  Ghost cells included -- the
//! flux kernels read the face-harmonic chi one cell into the ghost region.

void MHD::RefreshFldLocalChi() {
  radiationfld::RefreshGreyChiField(u0, nmhd, eos_tbl, fld_opac_tbl,
                                    fld_opac_dens_cgs, fld_opac_len_cgs,
                                    fld_chi_floor, fld_chi_cell);
}

} // namespace mhd
