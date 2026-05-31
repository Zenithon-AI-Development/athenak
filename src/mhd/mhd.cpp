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
#include "radiation_fld/fld_grey_operator.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "diffusion/braginskii_transport.hpp"
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
    peos = new TabulatedMHD(ppack, pin);
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
    Real fld_chi = pin->GetOrAddReal("mhd","fld_chi", 1.0);
    Real fld_nl  = pin->GetOrAddReal("mhd","fld_n_larsen", 2.0);
    Real fld_es  = pin->GetOrAddReal("mhd","fld_e_source", -1.0);
    pfld_op = new FLDGreyOperator(ppack, pin, erad, fld_c, fld_chi, fld_nl, fld_es);
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
    pmg_op = new FLDMultigroupOperator(ppack, pin, erad_mg, mg_table, mg_c, mg_rho, mg_te,
                                       mg_nl, mg_es);
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
    apar.zbar       = pin->GetOrAddReal("mhd","acond_zbar", 1.0);
    apar.mi_si      = pin->GetOrAddReal("mhd","acond_mi_si", braginskii::kProtonMass);
    apar.lnlam      = pin->GetOrAddReal("mhd","acond_lnlam", 10.0);
    apar.dens_conv  = pin->GetOrAddReal("mhd","acond_dens_conv", 1.0);
    apar.temp_conv  = pin->GetOrAddReal("mhd","acond_temp_conv", 1.0);
    apar.bmag_conv  = pin->GetOrAddReal("mhd","acond_bmag_conv", 1.0);
    apar.kappa_conv = pin->GetOrAddReal("mhd","acond_kappa_conv", 1.0);
    apar.nlarsen    = pin->GetOrAddReal("mhd","acond_n_larsen", 2.0);
    apar.vfs        = pin->GetOrAddReal("mhd","acond_vfs", -1.0);
    apar.incl_e     = pin->GetOrAddBoolean("mhd","acond_incl_e", true);
    apar.incl_i     = pin->GetOrAddBoolean("mhd","acond_incl_i", true);
    Real agamma = peos->eos_data.gamma;
    pacond_op = new AnisotropicConductionOperator(ppack, pin, u0, bcc0, agamma, apar);
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
    Real resb_eta = pin->GetOrAddReal("mhd","resb_eta", 1.0);
    Kokkos::deep_copy(eta_resb, resb_eta);
    presb_op = new ResistiveBphiOperator(ppack, pin, bphi, eta_resb);
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
    mrad_chi_a   = pin->GetOrAddReal("mhd","mrad_chi_a", 0.0);
    mrad_cv      = pin->GetOrAddReal("mhd","mrad_cv", 1.0);
    mrad_arad    = pin->GetOrAddReal("mhd","mrad_arad", 1.0);
    mrad_clight  = pin->GetOrAddReal("mhd","mrad_clight",
                                     (pfld_op != nullptr) ? pfld_op->c_light() : 1.0);
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

} // namespace mhd
