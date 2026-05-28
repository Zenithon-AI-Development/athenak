//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coupled_liner.cpp
//! \brief Problem generator for an end-to-end COUPLED-CIRCUIT (ADR-0005 mode C)
//!  cylindrical liner implosion: a voltage-driven lumped-element circuit ODE integrated
//!  alongside the MHD, with the Faraday load voltage V_load = d(Phi)/dt fed back into the
//!  circuit, sets the outer-radial B_phi boundary current (issue [18c]/#40).
//!
//! This is the consumer that wires the three ADR-0005 circuit pieces into one run:
//!   * circuit::LumpedCircuit (#34) -- the series-loop ODE
//!     L dI/dt + (Z0+R_loss) I + V_C + V_load = V_oc(t), dV_C/dt = I/C, advanced once per
//!     MHD step by RK4 over host scalars;
//!   * circuit::FaradayVoltage / PoloidalFluxGlobal (#31) -- the global reduction of the
//!     poloidal flux Phi = int int B_phi dr dz and its time derivative V_load (the mode-C
//!     feedback);
//!   * circuit::ApplyDriveBphiBC (#21) -- the reusable outer-x1 B_phi ghost fill that
//!     turns the integrated current I(t) into B_phi = mu0*I/(2*pi*r) at the load radius.
//!
//! The initial condition is the driven-pinch `liner` profile (#29): a dense conducting
//! shell (fuel | liner | vacuum, by radius).  The circuit starts at rest (I=0, V_C=0) so
//! B_phi=0 everywhere at t=0; the open-circuit voltage source then ramps the current up,
//! the magnetic pressure just outside the liner drives it inward (the magnetic piston),
//! and as the liner implodes the load inductance L_load(R)=(mu0 H/2pi) ln(R_out/R) grows
//! -- the resulting back-EMF V_load = d(Phi)/dt feeds back through the circuit (current
//! loss + the stagnation voltage spike).  The run is 1-D radial (axisymmetric, no RT
//! modes): a clean bulk implosion whose current/voltage waveforms the verification (#40)
//! compares to the coupled-circuit + thin-shell reference (the anchor circuit, ADR-0005).
//!
//! Diagnostics: a user history output ("<basename>.user.hst") exposes the circuit current
//! I, capacitor voltage V_C, the Faraday load voltage V_load, and the reduced poloidal
//! flux Phi each output step -- the current/voltage waveforms the verification baselines.
//!
//! USER pgen: build with -D PROBLEM=coupled_liner (not part of the default suite binary).

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "circuit/drive_source.hpp"
#include "circuit/drive_bphi_bc.hpp"
#include "circuit/faraday_voltage.hpp"
#include "circuit/lumped_circuit.hpp"
#include "pgen.hpp"

namespace {
// File-scope circuit state shared with the (stateless-signature) user_bcs_func and the
// user_hist_func.  The lumped-element circuit ODE is advanced on the host once per step.
circuit::LumpedCircuit circuit_;     // the coupled-circuit ODE (ADR-0005 mode C, #34)
circuit::FaradayVoltage faraday_;    // d(Phi)/dt monitor -> the mode-C load voltage (#31)
Real circ_t_ = 0.0;                  // circuit internal time (tracks the MHD time)
Real v_load_last_ = 0.0;             // most recent load voltage fed to the circuit
Real r_loss_ = 0.0;                  // (constant) loss resistor R_loss for this run

//----------------------------------------------------------------------------------------
//! \fn CoupledLinerBCs
//! \brief Outer-x1 user BC.  Advances the coupled circuit one MHD step (once per cycle,
//!  guarded by the circuit's own clock so the per-stage re-dispatch does not re-step),
//!  feeding the Faraday load voltage V_load = d(Phi)/dt back into the ODE, then fills the
//!  cell-centred ghosts by outflow and the B_phi ghosts from the circuit current I(t).
void CoupledLinerBCs(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  // --- advance the circuit from circ_t_ up to the current MHD time (once per step) ---
  // pm->time is constant across integrator stages and advances once per cycle, so the
  // strict ">" guard advances the circuit exactly once per MHD step (and never on the
  // pre-loop ghost fill at t=0, where pm->time == circ_t_ == 0).
  if (pm->time > circ_t_) {
    Real flux = circuit::PoloidalFluxGlobal(pm);          // Phi at the current MHD state
    Real v_load = faraday_.Update(flux, pm->time);        // (Phi - Phi_prev)/(t - t_prev)
    circuit_.SetLossResistance(r_loss_);
    circuit_.Step(pm->time - circ_t_, circ_t_, v_load);   // RK4 over [circ_t_, pm->time]
    v_load_last_ = v_load;
    circ_t_ = pm->time;
  }

  auto &indcs = pm->mb_indcs;
  int &ng = indcs.ng;
  int &ie = indcs.ie;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  int nvar = pmbp->pmhd->nmhd + pmbp->pmhd->nscalars;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &u0 = pmbp->pmhd->u0;

  // (1) cell-centred conserved variables: zero-gradient (outflow) at the outer-x1 ghosts.
  par_for("coupled_ccbc_ox1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::user) { return; }
    for (int i=0; i<ng; ++i) {
      for (int n=0; n<nvar; ++n) { u0(m,n,k,j,ie+i+1) = u0(m,n,k,j,ie); }
    }
  });

  // (2) face-centred B: driven B_phi from the circuit current + outflow B_r, B_z.
  circuit::ApplyDriveBphiBC(pm, circuit_.Current(), circuit_.mu0, false);
}

//----------------------------------------------------------------------------------------
//! \fn CoupledLinerHistory
//! \brief User history output: the circuit current/voltage waveforms + the reduced flux.
//!  Columns: I_circuit, V_cap, V_load, Bphi_flux.  The circuit scalars are global host
//!  values (identical on every rank); written on rank 0 only (0 elsewhere) so the
//!  in-place MPI_Reduce(SUM) in history.cpp yields the global value.  The flux is the
//!  per-rank LOCAL partial (PoloidalFluxLocal), so the same SUM gives the global flux.
void CoupledLinerHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 4;
  pdata->label[0] = "I_circuit";
  pdata->label[1] = "V_cap";
  pdata->label[2] = "V_load";
  pdata->label[3] = "Bphi_flux";

  bool root = (global_variable::my_rank == 0);
  pdata->hdata[0] = root ? circuit_.Current()    : 0.0;
  pdata->hdata[1] = root ? circuit_.CapVoltage() : 0.0;
  pdata->hdata[2] = root ? v_load_last_          : 0.0;
  pdata->hdata[3] = circuit::PoloidalFluxLocal(pm);
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) { pdata->hdata[n] = 0.0; }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize the liner column and wire the coupled-circuit B_phi BC + history.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Enroll the user hooks (must happen on restart too, before the enrollment check).
  user_bcs_func = CoupledLinerBCs;
  user_hist_func = CoupledLinerHistory;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "coupled_liner requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_system == CoordSystem::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "coupled_liner requires <coord> system = cylindrical" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- coupled-circuit (ADR-0005 mode C) configuration (<problem> block) ----
  // The open-circuit voltage source V_oc(t) reuses the prescribed-source waveform
  // machinery (constant/linear_ramp/sin_squared/tabulated); its i0 is the PEAK VOLTAGE.
  circuit_.mode = circuit::DriveMode::coupled_circuit;
  std::string wf = pin->GetOrAddString("problem", "voltage_waveform", "constant");
  circuit_.voltage_source.waveform = circuit::ParseWaveform(wf);
  circuit_.voltage_source.i0     = pin->GetOrAddReal("problem", "v_oc", 1.0);
  circuit_.voltage_source.t_rise = pin->GetOrAddReal("problem", "t_rise", 1.0);
  circuit_.L   = pin->GetOrAddReal("problem", "circuit_L", 1.0);
  circuit_.cap = pin->GetOrAddReal("problem", "circuit_C", 0.0);   // <=0 -> pure RL loop
  circuit_.Z0  = pin->GetOrAddReal("problem", "circuit_Z0", 0.0);
  circuit_.mu0 = pin->GetOrAddReal("problem", "mu0", 1.0);
  r_loss_      = pin->GetOrAddReal("problem", "r_loss", 0.0);
  if (circuit_.voltage_source.waveform == circuit::CurrentWaveform::tabulated) {
    std::string vfile = pin->GetOrAddString("problem", "voltage_file", "unset");
    circuit::ReadCurrentWaveform(vfile, circuit_.voltage_source);
  }
  // Start the circuit (and its clock + flux monitor) from rest: I=0, V_C=0, Phi(0)=0.
  circuit_.Reset(0.0, 0.0);
  faraday_.Reset();
  circ_t_ = 0.0;
  v_load_last_ = 0.0;

  if (restart) return;

  // ---- dense liner shell (fuel | liner | vacuum, by radius); B_phi=0 at t=0 (I0=0) ----
  Real p0  = pin->GetOrAddReal("problem", "p0", 1.0e-3);
  Real bz0 = pin->GetOrAddReal("problem", "bz0", 0.0);
  Real r_fuel  = pin->GetOrAddReal("problem", "r_fuel", 0.4);
  Real r_liner = pin->GetOrAddReal("problem", "r_liner", 0.6);
  Real d_fuel  = pin->GetOrAddReal("problem", "d_fuel", 0.02);
  Real d_liner = pin->GetOrAddReal("problem", "d_liner", 1.0);
  Real d_vac   = pin->GetOrAddReal("problem", "d_vac", 0.01);

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1;
  auto &size = pmbp->pmb->mb_size;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  bool is_ideal = eos.is_ideal;
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;

  par_for("pgen_coupled_liner", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real dens = (x1v < r_fuel) ? d_fuel : ((x1v < r_liner) ? d_liner : d_vac);
    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // B_z (premag) is the x3-face field; B_r=0; B_phi (x2-face) starts at 0 (rest).
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = 0.0;
    b0.x3f(m,k,j,i) = bz0;
    if (i==ie) { b0.x1f(m,k,j,i+1) = 0.0; }
    if (j==je) { b0.x2f(m,k,j+1,i) = 0.0; }
    if (k==ke) { b0.x3f(m,k+1,j,i) = bz0; }

    if (is_ideal) {
      u0(m,IEN,k,j,i) = p0/gm1 + 0.5*(bz0*bz0);
    }
  });

  return;
}
