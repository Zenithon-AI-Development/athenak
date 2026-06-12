//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd_tasks.cpp
//! \brief functions that control MHD tasks stored in tasklists in MeshBlockPack

#include <map>
#include <memory>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/resistivity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "driver/driver.hpp"
#include "driver/parabolic_integrator.hpp"
#include "driver/composite_parabolic_operator.hpp"
#include "radiation_fld/matter_radiation_coupling.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "diffusion/resistive_bphi_operator.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"

namespace mhd {
//----------------------------------------------------------------------------------------
//! \fn void MHD::AssembleMHDTasks
//! \brief Adds mhd tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after MHD constructor
//! See comments Hydro::AssembleHydroTasks() function for more details.

void MHD::AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);

  // assemble "before_timeintegrator" task list
  id.savest = tl["before_timeintegrator"]->AddTask(&MHD::SaveMHDState, this, none);

  if (!strang_split) {
    // ---- DEFAULT (Lie-split) path: each operator advances a FULL super-step once per
    // step in "before_timeintegrator" (#110-#113).  Unchanged from before #115; default
    // runs add no task and are byte-identical. ----

    // operator-split grey FLD radiation (RKL2 STS), once per step.  pfld_op is built
    // in the MHD constructor (needs `pin` for its own ghost-exchange object, #108);
    // non-null only when grey FLD operator-split is enabled, so default runs add no
    // task and are byte-identical (ADR-0001, #110/[A3]).
    if (pfld_op != nullptr) {
      id.opsfld = tl["before_timeintegrator"]->AddTask(&MHD::OperatorSplitFLD, this,
                                                       none);
    }

    // operator-split MULTIGROUP FLD radiation (RKL2 STS), once per step.  pmg_op is built
    // in the MHD ctor (it needs `pin` for its ghost-exchange object + opacity table);
    // non-null only when multigroup FLD operator-split is enabled, so default runs add no
    // task and are byte-identical (ADR-0001/ADR-0007, #111/[A4]).
    if (pmg_op != nullptr) {
      id.opsmgfld = tl["before_timeintegrator"]->AddTask(&MHD::OperatorSplitMultigroupFLD,
                                                         this, none);
    }

    // operator-split ANISOTROPIC Braginskii conduction (RKL2 STS), once per step.
    // pacond_op is built in the MHD ctor (needs `pin` for its own ghost-exchange boundary
    // object, #108); non-null only when aniso conduction operator-split is enabled, so
    // default runs add no task and are byte-identical (ADR-0006/ADR-0001, #112/[A5]).
    if (pacond_op != nullptr) {
      id.opsacond = tl["before_timeintegrator"]->AddTask(
                        &MHD::OperatorSplitAnisoConduction, this, none);
    }

    // operator-split CYLINDRICAL resistive B_phi diffusion (RKL2 STS), once per step.
    // presb_op is built in the MHD ctor (needs `pin` for its own ghost-exchange boundary
    // object, #108); non-null only when resistive-B_phi operator-split is enabled, so
    // default runs add no task and are byte-identical (ADR-0004/ADR-0001, #113/[A6]).
    if (presb_op != nullptr) {
      id.opsresb = tl["before_timeintegrator"]->AddTask(&MHD::OperatorSplitResistiveBphi,
                                                        this, none);
    }
  } else {
    // ---- STRANG-split path (#115/[B2], ADR-0009): the parabolic composite block is
    // Strang-wrapped around the hyperbolic update as  (1/2 STS) . hydro . (1/2 STS),
    // with the per-cell point-implicit matter-radiation coupling wrapped OUTSIDE the
    // super-step on each side (a half coupling step each), so it is NOT inflated
    // by the RKL2 substage count.  "before_timeintegrator": (1/2 coupling) -> (1/2 STS);
    // "after_timeintegrator": (1/2 STS) -> (1/2 coupling).  Symmetric => globally 2nd
    // order.  The half-step tasks are no-ops unless their physics is active. ----
    id.cplb    = tl["before_timeintegrator"]->AddTask(&MHD::MatterRadCouplingHalf, this,
                                                      id.savest);
    id.strangb = tl["before_timeintegrator"]->AddTask(&MHD::StrangParabolicHalf, this,
                                                      id.cplb);
    id.stranga = tl["after_timeintegrator"]->AddTask(&MHD::StrangParabolicHalf, this,
                                                     none);
    id.cpla    = tl["after_timeintegrator"]->AddTask(&MHD::MatterRadCouplingHalf, this,
                                                     id.stranga);
  }

  // assemble "before_stagen" task list
  id.irecv = tl["before_stagen"]->AddTask(&MHD::InitRecv, this, none);

  // assemble "stagen" task list
  id.copyu     = tl["stagen"]->AddTask(&MHD::CopyCons, this, none);
  id.flux      = tl["stagen"]->AddTask(&MHD::Fluxes, this, id.copyu);
  id.sendf     = tl["stagen"]->AddTask(&MHD::SendFlux, this, id.flux);
  id.recvf     = tl["stagen"]->AddTask(&MHD::RecvFlux, this, id.sendf);
  id.rkupdt    = tl["stagen"]->AddTask(&MHD::RKUpdate, this, id.recvf);
  id.srctrms   = tl["stagen"]->AddTask(&MHD::MHDSrcTerms, this, id.rkupdt);
  id.sendu_oa  = tl["stagen"]->AddTask(&MHD::SendU_OA, this, id.srctrms);
  id.recvu_oa  = tl["stagen"]->AddTask(&MHD::RecvU_OA, this, id.sendu_oa);
  id.restu     = tl["stagen"]->AddTask(&MHD::RestrictU, this, id.recvu_oa);
  id.sendu     = tl["stagen"]->AddTask(&MHD::SendU, this, id.restu);
  id.recvu     = tl["stagen"]->AddTask(&MHD::RecvU, this, id.sendu);
  id.sendu_shr = tl["stagen"]->AddTask(&MHD::SendU_Shr, this, id.recvu);
  id.recvu_shr = tl["stagen"]->AddTask(&MHD::RecvU_Shr, this, id.sendu_shr);
  id.efld      = tl["stagen"]->AddTask(&MHD::CornerE, this, id.recvu_shr);
  id.efldsrc   = tl["stagen"]->AddTask(&MHD::EFieldSrc, this, id.efld);
  id.sende     = tl["stagen"]->AddTask(&MHD::SendE, this, id.efldsrc);
  id.recve     = tl["stagen"]->AddTask(&MHD::RecvE, this, id.sende);
  id.ct        = tl["stagen"]->AddTask(&MHD::CT, this, id.recve);
  id.sendb_oa  = tl["stagen"]->AddTask(&MHD::SendB_OA, this, id.ct);
  id.recvb_oa  = tl["stagen"]->AddTask(&MHD::RecvB_OA, this, id.sendb_oa);
  id.restb     = tl["stagen"]->AddTask(&MHD::RestrictB, this, id.recvb_oa);
  id.sendb     = tl["stagen"]->AddTask(&MHD::SendB, this, id.restb);
  id.recvb     = tl["stagen"]->AddTask(&MHD::RecvB, this, id.sendb);
  id.sendb_shr = tl["stagen"]->AddTask(&MHD::SendB_Shr, this, id.recvb);
  id.recvb_shr = tl["stagen"]->AddTask(&MHD::RecvB_Shr, this, id.sendb_shr);
  id.bcs       = tl["stagen"]->AddTask(&MHD::ApplyPhysicalBCs, this, id.recvb_shr);
  id.prol      = tl["stagen"]->AddTask(&MHD::Prolongate, this, id.bcs);
  id.c2p       = tl["stagen"]->AddTask(&MHD::ConToPrim, this, id.prol);
  id.newdt     = tl["stagen"]->AddTask(&MHD::NewTimeStep, this, id.c2p);

  // assemble "after_stagen" task list
  id.csend = tl["after_stagen"]->AddTask(&MHD::ClearSend, this, none);
  // although RecvFlux/U/E/B functions check that all recvs complete, add ClearRecv to
  // task list anyways to catch potential bugs in MPI communication logic
  id.crecv = tl["after_stagen"]->AddTask(&MHD::ClearRecv, this, id.csend);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SaveMHDState
//! \brief Copy primitives and bcc before step to enable computation of time derivatives,
//! for example to compute jcon in GRMHD.

TaskStatus MHD::SaveMHDState(Driver *pdrive, int stage) {
  if (wbcc_saved) {
    Kokkos::deep_copy(DevExeSpace(), wsaved, w0);
    Kokkos::deep_copy(DevExeSpace(), bccsaved, bcc0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::OperatorSplitFLD
//! \brief Advance the grey FLD radiation energy `erad` one operator-split RKL2 super-step
//! over the full hyperbolic dt (#110/[A3]).  This is the radiation analog of hydro's
//! OperatorSplitConduction: parabolic::OperatorSplitStep wraps the FLDGreyOperator, whose
//! per-substage ApplyBoundary delegates the cross-block/rank (and AMR coarse/fine) ghost
//! refresh to MeshBoundaryValuesCC::SyncParabolicGhosts (#108/[A1]), so it runs on >1
//! MeshBlock, >1 MPI rank, and under block-AMR.  Registered once per step in the
//! "before_timeintegrator" list, so it runs coupled with the hyperbolic MHD update.

TaskStatus MHD::OperatorSplitFLD(Driver *pdrive, int stage) {
  if (pfld_op == nullptr) { return TaskStatus::complete; }
  parabolic::OperatorSplitStep(pdrive->pparabolic, *pfld_op, erad,
                               pmy_pack->pmesh->dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::OperatorSplitMultigroupFLD
//! \brief Advance the MULTIGROUP FLD per-group radiation energy `erad_mg` one
//! operator-split RKL2 super-step over the full hyperbolic dt (#111/[A4]).  Identical
//! structure to OperatorSplitFLD: parabolic::OperatorSplitStep wraps the
//! FLDMultigroupOperator, whose per-substage ApplyBoundary delegates the cross-block/rank
//! (and AMR coarse/fine) ghost refresh of ALL groups (one batched exchange) to
//! SyncParabolicGhosts, and whose OperatorAction runs the conservative refine-boundary
//! flux correction, so it runs on >1 MeshBlock, >1 MPI rank, and under block-AMR
//! (incl. dynamic regrid, #111/[A4]).

TaskStatus MHD::OperatorSplitMultigroupFLD(Driver *pdrive, int stage) {
  if (pmg_op == nullptr) { return TaskStatus::complete; }
  parabolic::OperatorSplitStep(pdrive->pparabolic, *pmg_op, erad_mg,
                               pmy_pack->pmesh->dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::OperatorSplitAnisoConduction
//! \brief Advance the LIVE conserved field `u0` one operator-split RKL2 super-step under
//! anisotropic (magnetized) Braginskii thermal conduction over the full hyperbolic dt
//! (#112/[A5]).  Like the FLD operators, parabolic::OperatorSplitStep wraps the
//! AnisotropicConductionOperator, whose per-substage ApplyBoundary delegates the
//! cross-block/rank (and AMR c/f) ghost refresh to SyncParabolicGhosts (#108/[A1])
//! and whose OperatorAction runs the conservative refine-boundary flux correction, so it
//! runs on >1 MeshBlock, >1 MPI rank, and under block-AMR.  The flux is aligned to the
//! frozen cell-centred B (bcc0) captured at construction; u0 is already AMR-regrid-
//! registered as the MHD conserved field, so no standalone-array registration is needed.

TaskStatus MHD::OperatorSplitAnisoConduction(Driver *pdrive, int stage) {
  if (pacond_op == nullptr) { return TaskStatus::complete; }
  parabolic::OperatorSplitStep(pdrive->pparabolic, *pacond_op, u0,
                               pmy_pack->pmesh->dt);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::OperatorSplitResistiveBphi
//! \brief Advance the standalone B_phi field `bphi` one operator-split RKL2 super-step
//! under the cylindrical resistive B_phi diffusion (the -eta B_phi/r^2 curl-curl op) over
//! the full hyperbolic dt (#113/[A6]).  Like the FLD operators, OperatorSplitStep wraps
//! the ResistiveBphiOperator, whose per-substage ApplyBoundary delegates the cross-block/
//! rank (and AMR c/f) ghost refresh to SyncParabolicGhosts (#108/[A1]) -- keeping the
//! antisymmetric axis ghost only at the true r=0 domain face -- and whose OperatorAction
//! runs the conservative refine-boundary flux correction, so it runs on >1 MeshBlock,
//! >1 MPI rank, and under block-AMR.

TaskStatus MHD::OperatorSplitResistiveBphi(Driver *pdrive, int stage) {
  if (presb_op == nullptr) { return TaskStatus::complete; }
  // #181/[P7a], ADR-0012 gap a: when coupled, bracket the super-step with the copy-in /
  // write-back so resistivity diffuses the *live driven* B_phi (b0.x2f) rather than a
  // decoupled scratch field; off by default => byte-identical (the standalone bphi path).
  if (resb_couple_b0) { CoupleResbBphiFromB0(); }
  parabolic::OperatorSplitStep(pdrive->pparabolic, *presb_op, bphi,
                               pmy_pack->pmesh->dt);
  if (resb_couple_b0) { CoupleResbBphiToB0(); }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::CoupleResbBphiFromB0
//! \brief Copy the live driven azimuthal face field `b0.x2f` into the resb operator's
//! standalone `bphi` (component 0) over the active cells (#181/[P7a], ADR-0012 gap a).
//! In the cylindrical (r,phi,z) maglif column B_phi is the x2-face field and is uniform
//! in phi (axisymmetric drive), so the lower x2-face value equals the cell-centred B_phi
//! the operator diffuses.  Ghost zones of `bphi` are refilled by the operator's own
//! per-substage ApplyBoundary (antisymmetric axis + SyncParabolicGhosts), so only the
//! active cells are seeded here.

void MHD::CoupleResbBphiFromB0() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto bph = bphi;
  auto b2f = b0.x2f;
  par_for("resb_b0_in", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    bph(m,0,k,j,i) = b2f(m,k,j,i);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void MHD::CoupleResbBphiToB0
//! \brief Write the resistively-diffused `bphi` (component 0) back into the live driven
//! face field `b0.x2f` over the active cells (#181/[P7a], ADR-0012 gap a).  Mirrors the
//! maglif IC face fill: the lower x2-face of every active cell, plus the top x2-face of
//! the last cell in phi (j==je) -- B_phi is uniform in phi so that face equals the je
//! cell value.
//!
//! ENERGY (#192/[P8]): the total energy u0(IEN) is updated by the per-cell magnetic-
//! energy change of the swap, 0.5*(bphi_new^2 - bphi_old^2), so the gas internal energy
//! ConsToPrim recovers (e_int = E - KE - B^2/2) is INVARIANT under the write-back.
//! Leaving IEN unchanged (the pre-#192 behaviour) silently fed the field-energy change
//! into e_int with the WRONG SIGN -- cells the field diffuses INTO are cooled by
//! -dB^2/2, and in the cold liner skin / tenuous vacuum gap (B^2/2 >> e_int) that drove
//! e_int negative every step, so the pressure floor re-filled it and FABRICATED energy:
//! on the paper-resolution faithful B1 deck this floor pump injected ~4 orders of
//! magnitude more energy than the drive had delivered by t=24 ns and ended in the
//! exponential mass/density runaway documented on #192 (the A100 bisect attributes the
//! runaway to resb alone; fld+mrad and acond stay bounded).  With the swap made
//! energy-consistent the field energy a cell gains arrives via the operator's transport
//! (sourced at the driven boundary, i.e. the circuit does the work) instead of being
//! billed to the local gas.  The net field energy the super-step DISSIPATES is dropped
//! (not deposited as heat): Ohmic/Joule heating of the skin is a one-signed, stabilising
//! omission deferred to a follow-up (it needs the eta J^2 split the RKL2 composite does
//! not expose); see ADR-0015.

void MHD::CoupleResbBphiToB0() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto bph = bphi;
  auto b2f = b0.x2f;
  auto u = u0;
  par_for("resb_b0_out", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real b_old = b2f(m,k,j,i);
    const Real b_new = bph(m,0,k,j,i);
    u(m,IEN,k,j,i) += 0.5*(b_new*b_new - b_old*b_old);
    b2f(m,k,j,i) = b_new;
    if (j == je) { b2f(m,k,j+1,i) = b_new; }
  });
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::StrangParabolicHalf
//! \brief Advance every per-field composite parabolic operator (#114) by ONE RKL2 super-
//! step over HALF the hyperbolic dt (#115/[B2], ADR-0009).  Registered in BOTH the
//! "before_timeintegrator" and "after_timeintegrator" slots, so the halves Strang-wrap
//! the hyperbolic update: (1/2 STS) . hydro . (1/2 STS).  Each composite owns the global
//! min-dt MPI all-reduce, so every rank derives the same stage count per field.  Runs
//! only when `<mhd> strang_split` is on (else the individual full-step operator tasks
//! run instead); a no-op when no composite is registered.

TaskStatus MHD::StrangParabolicHalf(Driver *pdrive, int stage) {
  const Real half_dt = 0.5*pmy_pack->pmesh->dt;
  const int ncomp = static_cast<int>(strang_comps.size());
  for (int n = 0; n < ncomp; ++n) {
    DvceArray5D<Real> *field = nullptr;
    switch (strang_field_ids[n]) {
      case SF_ERAD:   field = &erad;    break;
      case SF_ERADMG: field = &erad_mg; break;
      case SF_U0:     field = &u0;      break;
      case SF_BPHI:   field = &bphi;    break;
      default: break;
    }
    if (field != nullptr) {
      // #181/[P7a], ADR-0012 gap a: the resb B_phi composite optionally diffuses the live
      // driven b0.x2f (copy-in before / write-back after this HALF super-step), so each
      // Strang half acts on the *driving* field.  Off by default => byte-identical.
      const bool couple = (strang_field_ids[n] == SF_BPHI) && resb_couple_b0;
      if (couple) { CoupleResbBphiFromB0(); }
      parabolic::OperatorSplitStep(pdrive->pparabolic, *strang_comps[n], *field, half_dt);
      if (couple) { CoupleResbBphiToB0(); }
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::MatterRadCouplingHalf
//! \brief One per-cell point-implicit (backward-Euler) matter-radiation coupling step
//! over HALF the hyperbolic dt (#115/[B2], reusing the #23 kernel).  Exchanges energy
//! between the grey radiation field `erad` and the gas internal energy (u0 IEN), driving
//! each cell toward radiative equilibrium a*T^4 = E_r, conserving E_r + e_gas exactly
//! (the "species split").  This is the local, point-stiff partner of the FLD spatial
//! diffusion; it is wrapped OUTSIDE the RKL2 super-step (half a step each Strang side),
//! so it is applied exactly twice per timestep -- NOT inflated by the substage count `s`.
//! No-op unless `<mhd> mrad_coupling` is on (and grey FLD is active).

TaskStatus MHD::MatterRadCouplingHalf(Driver *pdrive, int stage) {
  if (!mrad_coupling || pfld_op == nullptr) { return TaskStatus::complete; }
  const Real half_dt = 0.5*pmy_pack->pmesh->dt;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto u   = u0;
  auto er  = erad;
  auto bcc = bcc0;
  const Real chia = mrad_chi_a, cv = mrad_cv, arad = mrad_arad, cl = mrad_clight;
  // EOS-aware heat capacity (#183/[P7c], ADR-0012 gap c): on the faithful tabulated path
  // read c_v from the same tabulated closure ConsToPrim2T uses, instead of the constant
  // placeholder.  The electron internal energy density rides scalar 0 (index nmhd).
  const bool eos_aware = mrad_eos_aware;
  const int eele_idx = nmhd;
  auto tbl = pmy_pack->pmhd->eos_tbl;
  par_for("mrad_cpl", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // recover the gas INTERNAL energy density (subtract kinetic + magnetic) so coupling
    // exchanges only thermal energy; momentum/B are untouched by the local exchange.
    Real rho = u(m,IDN,k,j,i);
    Real mx = u(m,IM1,k,j,i), my = u(m,IM2,k,j,i), mz = u(m,IM3,k,j,i);
    Real ke_d = (rho > 0.0) ? 0.5*(mx*mx + my*my + mz*mz)/rho : 0.0;
    Real bx = bcc(m,IBX,k,j,i), by = bcc(m,IBY,k,j,i), bz = bcc(m,IBZ,k,j,i);
    Real me_d = 0.5*(bx*bx + by*by + bz*bz);
    Real e_gas = u(m,IEN,k,j,i) - ke_d - me_d;
    // per-cell heat capacity: tabulated closure (volumetric c_v = rho*(c_v,e + c_v,i) at
    // inverted electron/ion temperatures) when EOS-aware, else the constant placeholder.
    Real cv_cell = cv;
    if (eos_aware && rho > 0.0) {
      Real e_ele = u(m,eele_idx,k,j,i);          // electron internal energy density
      Real e_ion = e_gas - e_ele;                // ion energy by subtraction (ADR-0002)
      Real te = tbl.Te(rho, e_ele/rho);
      Real ti = tbl.Ti(rho, e_ion/rho);
      cv_cell = rho*(tbl.CvEle(rho, te) + tbl.CvIon(rho, ti));   // specific -> volumetric
    }
    Real e_rad_new, e_gas_new;
    radiationfld::PointImplicitGreyCoupling(er(m,0,k,j,i), e_gas, cv_cell, chia, cl, arad,
                                            half_dt, e_rad_new, e_gas_new);
    er(m,0,k,j,i) = e_rad_new;
    u(m,IEN,k,j,i) = e_gas_new + ke_d + me_d;
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::InitRecv
//! \brief Wrapper task list function to post non-blocking receives (with MPI), and
//! initialize all boundary receive status flags to waiting (with or without MPI).  Note
//! this must be done for communication of BOTH conserved (cell-centered) and
//! face-centered fields AND their fluxes (with SMR/AMR).

TaskStatus MHD::InitRecv(Driver *pdrive, int stage) {
  // post receives for U
  TaskStatus tstat = pbval_u->InitRecv(nmhd+nscalars);
  if (tstat != TaskStatus::complete) return tstat;
  // post receives for B
  tstat = pbval_b->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;

  // with SMR/AMR post receives for fluxes of U, always post receives for fluxes of B
  // do not post receives for fluxes when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR, post receives for fluxes of U
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->InitFluxRecv(nmhd+nscalars);
      if (tstat != TaskStatus::complete) return tstat;
    }
    // post receives for fluxes of B, which are used even with uniform grids
    tstat = pbval_b->InitFluxRecv(3);
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection post receives for U and B
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->InitRecv();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = porb_b->InitRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries calculate x2-distance x1-boundaries have sheared and
  // with MPI post receives for U and B
  if (psbox_u != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      Real time = pmy_pack->pmesh->time;
      if (stage == pdrive->nexp_stages) {
        time += pmy_pack->pmesh->dt;
      }
      tstat = psbox_u->InitRecv(time);
      if (tstat != TaskStatus::complete) return tstat;
      tstat = psbox_b->InitRecv(time);
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::CopyCons
//! \brief Simple task list function that copies u0 --> u1, and b0 --> b1 in first stage

TaskStatus MHD::CopyCons(Driver *pdrive, int stage) {
  if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
    Kokkos::deep_copy(DevExeSpace(), b1.x1f, b0.x1f);
    Kokkos::deep_copy(DevExeSpace(), b1.x2f, b0.x2f);
    Kokkos::deep_copy(DevExeSpace(), b1.x3f, b0.x3f);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::Fluxes
//! \brief Wrapper task list function that calls everything necessary to compute fluxes
//! of conserved variables

TaskStatus MHD::Fluxes(Driver *pdrive, int stage) {
  // select which calculate_flux function to call based on rsolver_method
  if (rsolver_method == MHD_RSolver::advect) {
    CalculateFluxes<MHD_RSolver::advect>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf) {
    CalculateFluxes<MHD_RSolver::llf>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle) {
    CalculateFluxes<MHD_RSolver::hlle>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlld) {
    CalculateFluxes<MHD_RSolver::hlld>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_sr) {
    CalculateFluxes<MHD_RSolver::llf_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_sr) {
    CalculateFluxes<MHD_RSolver::hlle_sr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::llf_gr) {
    CalculateFluxes<MHD_RSolver::llf_gr>(pdrive, stage);
  } else if (rsolver_method == MHD_RSolver::hlle_gr) {
    CalculateFluxes<MHD_RSolver::hlle_gr>(pdrive, stage);
  }

  // Add viscous, resistive, heat-flux, etc fluxes
  if (pvisc != nullptr) {
    pvisc->IsotropicViscousFlux(w0, pvisc->nu_iso, peos->eos_data, uflx);
  }
  if ((presist != nullptr) && (peos->eos_data.is_ideal)) {
    presist->OhmicEnergyFlux(b0, uflx);
  }
  if (pcond != nullptr) {
    pcond->AddHeatFlux(w0, peos->eos_data, uflx);
  }

  // call FOFC if necessary
  if (use_fofc) {
    FOFC(pdrive, stage);
  } else if (pmy_pack->pcoord->is_general_relativistic) {
    if (pmy_pack->pcoord->coord_data.bh_excise) {
      FOFC(pdrive, stage);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendFlux
//! \brief Wrapper task list function to pack/send restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::SendFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel)  {
    tstat = pbval_u->PackAndSendFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvFlux
//! \brief Wrapper task list function to recv/unpack restricted values of fluxes of
//! conserved variables at fine/coarse boundaries

TaskStatus MHD::RecvFlux(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  // Only execute BoundaryValues function with SMR/SMR
  if (pmy_pack->pmesh->multilevel) {
    tstat = pbval_u->RecvAndUnpackFluxCC(uflx);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::MHDSrcTerms
//! \brief Wrapper task list function to apply source terms to conservative vars
//! Note source terms must be computed using only primitives (w0), as the conserved
//! variables (u0) have already been partially updated when this fn called.

TaskStatus MHD::MHDSrcTerms(Driver *pdrive, int stage) {
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Add physics source terms (must be computed from primitives)
  if (psrc != nullptr) psrc->ApplySrcTerms(w0, peos->eos_data,  beta_dt, u0);

  // Add shearing box source terms for CC MHD variables
  if (psbox_u != nullptr) psbox_u->SourceTermsCC(w0, bcc0, peos->eos_data, beta_dt, u0);

  // Add coordinate source terms in GR.  Again, must be computed with only primitives.
  if (pmy_pack->pcoord->is_general_relativistic &&
      !pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pcoord->CoordSrcTerms(w0, bcc0, peos->eos_data, beta_dt, u0);
  } else if (pmy_pack->pcoord->is_dynamical_relativistic) {
    pmy_pack->pdyngr->AddCoordTerms(w0, bcc0, beta_dt, u0, pmy_pack->pmesh->mb_indcs.ng);
  }

  // Add curvilinear (cylindrical) geometric source terms (ADR-0004, #16): the radial
  // centrifugal+pressure+magnetic-hoop-stress source and the angular-momentum-conserving
  // azimuthal source, using the cell-centered field and the x1-flux from the Fluxes task.
  if (pmy_pack->pcoord->coord_system != CoordSystem::cartesian) {
    pmy_pack->pcoord->CoordSrcTermsMHDCyl(w0, bcc0, uflx.x1f, peos->eos_data,
                                          beta_dt, u0);
  }

  // Add user source terms
  if (pmy_pack->pmesh->pgen->user_srcs) {
    (pmy_pack->pmesh->pgen->user_srcs_func)(pmy_pack->pmesh, beta_dt);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->PackAndSendCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvU_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->RecvAndUnpackCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictU
//! \brief Wrapper task list function to restrict conserved vars

TaskStatus MHD::RestrictU(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendU
//! \brief Wrapper task list function to pack/send cell-centered conserved variables

TaskStatus MHD::SendU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->PackAndSendCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvU
//! \brief Wrapper task list function to receive/unpack cell-centered conserved variables

TaskStatus MHD::RecvU(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_u->RecvAndUnpackCC(u0, coarse_u0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendU_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->PackAndSendCC(u0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvU_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvU_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_u != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi) {
      tstat = psbox_u->RecvAndUnpackCC(u0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::EFieldSrc
//! \brief Wrapper task list function to apply source terms to electric field

TaskStatus MHD::EFieldSrc(Driver *pdrive, int stage) {
  if (psbox_b != nullptr) {
    // only execute when (2D)
    if (pmy_pack->pmesh->two_d) {
      psbox_b->SourceTermsFC(b0, efld);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendE
//! \brief Wrapper task list function to pack/send fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries. This is performed both
//! at MeshBlock boundaries at the same level (to keep magnetic flux in-sync on different
//! MeshBlocks), and at fine/coarse boundaries with SMR/AMR using restricted values of E.

TaskStatus MHD::SendE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->PackAndSendFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvE
//! \brief Wrapper task list function to recv/unpack fluxes of magnetic fields
//! (i.e. edge-centered electric field E) at MeshBlock boundaries

TaskStatus MHD::RecvE(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  tstat = pbval_b->RecvAndUnpackFluxFC(efld);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_OA
//! \brief Wrapper task list function to pack/send data for orbital advection

TaskStatus MHD::SendB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_b != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_b->shearing_box_r_phi)) {
      tstat = porb_b->PackAndSendFC(b0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_OA
//! \brief Wrapper task list function to recv/unpack data for orbital advection

TaskStatus MHD::RecvB_OA(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (porb_b != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_b->shearing_box_r_phi)) {
      tstat = porb_b->RecvAndUnpackFC(b0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::SendB
//! \brief Wrapper task list function to pack/send face-centered magnetic fields

TaskStatus MHD::SendB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->PackAndSendFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RecvB
//! \brief Wrapper task list function to recv/unpack face-centered magnetic fields

TaskStatus MHD::RecvB(Driver *pdrive, int stage) {
  TaskStatus tstat = pbval_b->RecvAndUnpackFC(b0, coarse_b0);
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::SendB_Shr
//! \brief Wrapper task list function to pack/send data for shearing box boundaries

TaskStatus MHD::SendB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_b != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_b->shearing_box_r_phi) {
      tstat = psbox_b->PackAndSendFC(b0, recon_method);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::RecvB_Shr
//! \brief Wrapper task list function to recv/unpack data for shearing box boundaries
//! Orbital remap is performed in this step.

TaskStatus MHD::RecvB_Shr(Driver *pdrive, int stage) {
  TaskStatus tstat = TaskStatus::complete;
  if (psbox_b != nullptr) {
    // only execute when (3D OR 2d_r_phi)
    if (pmy_pack->pmesh->three_d || psbox_b->shearing_box_r_phi) {
      tstat = psbox_b->RecvAndUnpackFC(b0);
    }
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ApplyPhysicalBCs
//! \brief Wrapper task list function to call functions that set physical and user BCs

TaskStatus MHD::ApplyPhysicalBCs(Driver *pdrive, int stage) {
  // do not apply BCs if domain is strictly periodic
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;

  // physical BCs
  pbval_u->HydroBCs((pmy_pack), (pbval_u->u_in), u0);
  pbval_b->BFieldBCs((pmy_pack), (pbval_b->b_in), b0);

  // user BCs
  if (pmy_pack->pmesh->pgen->user_bcs) {
    (pmy_pack->pmesh->pgen->user_bcs_func)(pmy_pack->pmesh);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskList MHD::Prolongate
//! \brief Wrapper task list function to prolongate conserved (or primitive) variables
//! at fine/coarse bundaries with SMR/AMR

TaskStatus MHD::Prolongate(Driver *pdrive, int stage) {
  if (pmy_pack->pmesh->multilevel) {  // only prolongate with SMR/AMR
    pbval_u->FillCoarseInBndryCC(u0, coarse_u0);
    pbval_b->FillCoarseInBndryFC(b0, coarse_b0);
    if (pmy_pack->pmesh->pmr->prolong_prims) {
      pbval_u->ConsToPrimCoarseBndry(coarse_u0, coarse_b0, coarse_w0);
      pbval_u->ProlongateCC(w0, coarse_w0);
      pbval_b->ProlongateFC(b0, coarse_b0);
      pbval_u->PrimToConsFineBndry(w0, b0, u0);
    } else {
      pbval_u->ProlongateCC(u0, coarse_u0);
      pbval_b->ProlongateFC(b0, coarse_b0);
    }
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ConToPrim
//! \brief Wrapper task list function to call ConsToPrim over entire mesh (including gz)

TaskStatus MHD::ConToPrim(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &ng = indcs.ng;
  int n1m1 = indcs.nx1 + 2*ng - 1;
  int n2m1 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng - 1) : 0;
  int n3m1 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng - 1) : 0;
  peos->ConsToPrim(u0, b0, w0, bcc0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearSend
//! \brief Wrapper task list function that checks all MPI sends have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearSend(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check sends of U complete
    TaskStatus tstat = pbval_u->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    // check sends of B complete
    tstat = pbval_b->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check sends for fluxes of U complete.  Always check sends of E complete
  // do not check flux send for ICs (stage < 0)
  if (stage >= 0) {
    // with SMR/AMR check sends of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // check sends of restricted fluxes of B complete even for uniform grids
    tstat = pbval_b->ClearFluxSend();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check sends for U and B complete
  // only execute when (shearing box defined) AND (last stage) AND (3D OR 2d_r_phi)
  if (porb_u != nullptr) {
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = porb_b->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check sends of U and B complete
  if (psbox_u != nullptr) {
    // only execute when (stage>=0 or -4) AND (3D OR 2d_r_phi)
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = psbox_b->ClearSend();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::ClearRecv
//! \brief Wrapper task list function that checks all MPI receives have completed. Used in
//! TaskList and in Driver::InitBoundaryValuesAndPrimitives()
//! If stage=(last stage):      clears U, B, Flx_U, Flx_B, U_OA, B_OA, U_Shr, BShr
//! If (last stage)>stage>=(0): clears U, B, Flx_U, Flx_B,             U_Shr, B_Shr
//! If stage=(-1):              clears U, B
//! If stage=(-4):              clears                                 U_Shr, B_Shr

TaskStatus MHD::ClearRecv(Driver *pdrive, int stage) {
  TaskStatus tstat;
  if ((stage >= 0) || (stage == -1)) {
    // check receives of U complete
    tstat = pbval_u->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
    // check receives of B complete
    tstat = pbval_b->ClearRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with SMR/AMR check recvs for fluxes of U complete.  Always check recvs of E complete
  // do not check flux receives when stage < 0 (i.e. ICs)
  if (stage >= 0) {
    // with SMR/AMR check receives of restricted fluxes of U complete
    if (pmy_pack->pmesh->multilevel) {
      tstat = pbval_u->ClearFluxRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
    // with SMR/AMR check receives of restricted fluxes of B complete
    tstat = pbval_b->ClearFluxRecv();
    if (tstat != TaskStatus::complete) return tstat;
  }

  // with orbital advection check receives of U and B are complete
  if (porb_u != nullptr) {
    // only execute when (last stage) AND (3D OR 2d_r_phi)
    if ((stage == pdrive->nexp_stages) &&
        (pmy_pack->pmesh->three_d || porb_u->shearing_box_r_phi)) {
      tstat = porb_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = porb_b->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  // with shearing box boundaries check receives of U and B complete
  if (psbox_u != nullptr) {
    // only execute when (stage>=0 or -4) AND (3D OR 2d_r_phi)
    if (((stage >= 0) || (stage == -4)) &&
        (pmy_pack->pmesh->three_d || psbox_u->shearing_box_r_phi)) {
      tstat = psbox_u->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
      tstat = psbox_b->ClearRecv();
      if (tstat != TaskStatus::complete) return tstat;
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MHD::RestrictB
//! \brief Wrapper function that restricts face-centered variables (magnetic field)

TaskStatus MHD::RestrictB(Driver *pdrive, int stage) {
  // Only execute Mesh function with SMR/AMR
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictFC(b0, coarse_b0);
  }
  return TaskStatus::complete;
}

} // namespace mhd
