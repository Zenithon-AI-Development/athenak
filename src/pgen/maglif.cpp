//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file maglif.cpp
//! \brief MagLIF / Z-pinch problem generator (issue [10a]/#32): the reusable integrated
//!  setup for the Phase-1 verification benchmarks 1-4 (single-/multi-mode MRT, RM, ICF).
//!
//! Lays down the canonical MagLIF target on a cylindrical grid -- ADR-0004 fixes
//! (x1,x2,x3)=(r,phi,z), so the AXIAL direction is x3 -- with three radial regions:
//!   * fuel   [0, r_fuel)        -- low-density gas pre-loaded with the axial seed field;
//!   * liner  [r_fuel, r_liner)  -- the dense conducting shell that implodes;
//!   * vacuum [r_liner, R_out)   -- the low-density gap carrying the driven 1/r B_phi.
//! The fuel + liner are premagnetized by a uniform axial B_z (the "premag" field that is
//! adiabatically compressed and magneto-insulates the hot fuel); the load current I(t)
//! enclosed by the liner produces the azimuthal drive field B_phi=mu0*I/(2*pi*r) laid
//! down only in the vacuum OUTSIDE the liner, so the magnetic piston compresses the shell
//! radially inward (the same physics verified in 1-D by issue [9b]/#29, here generalized
//! to a seeded, axially-resolved target).
//!
//! Perturbation seeding (problem/perturbation) seeds the magneto-Rayleigh-Taylor (MRT)
//! feed by one of two mechanisms.  The GEOMETRIC modes displace the liner interface radii
//! by a z-dependent amount dr(z) -- both interfaces move rigidly so the shell thickness
//! (and hence its areal mass) is preserved:
//!   * none         -- unperturbed (clean bulk implosion);
//!   * single_mode  -- one axial mode, dr(z)=A cos(2*pi*n (z-z0)/Lz + phi0);
//!   * roughness    -- a band of modes [n_min,n_max] with deterministic random phases
//!                     (a manufactured-surface-roughness spectrum, reproducible by seed).
//! The geometric (axial) perturbation is only applied when z (x3) is resolved (a 3-D
//! r-phi-z run); for a 1-D radial or 2-D r-phi study it is a no-op.  The TEMPERATURE mode
//! instead leaves the geometry clean and seeds a random zone-by-zone temperature (Ellison
//! et al. 2025, arXiv:2504.10760, Eq.16), see pgen/maglif_temperature_seed.hpp:
//!   * temperature  -- T(r,z)=max(T_min,T_0+exp((r-r_o)/lambda)*N_dT) on the target,
//!                     applied as a per-zone pressure factor T/T_0 at fixed density (mass
//!                     conserving; unperturbed at dT=0).  Reproducible by seed.
//!
//! Drive wiring (ADR-0005): `<problem> current_mode` selects the origin of the boundary
//! current (issue #236 promotes modes B/C from the coupled_liner USER pgen into this
//! suite path):
//!   * driven / nocurrent (A) -- prescribed I(t) via circuit::DriveSource + the reusable
//!     circuit-driven B_phi outer-radial user boundary (circuit/drive_bphi_bc.hpp, #21),
//!     just as driven_pinch.cpp does (user_bcs_func evaluates I(pm->time) each step);
//!   * voltage_rlc (B) -- open-circuit voltage source + fixed series RLC: the lumped-
//!     element circuit ODE (circuit/lumped_circuit.hpp, #34) is advanced host-side once
//!     per MHD step by RK4 and its integrated current I(t) drives the same boundary;
//!   * coupled_circuit (C) -- identical, plus the Faraday load voltage V_load = d(Phi)/dt
//!     (the global poloidal-flux reduction, circuit/faraday_voltage.hpp, #31) fed back
//!     into the loop -- current loss + the stagnation voltage spike.
//! Modes B/C expose the circuit waveforms (I, V_C, V_load, Phi) as user history columns;
//! all circuit parameters are deck-driven (tabulated/analytic V_oc(t) + R/L/C elements +
//! the r_loss loss-resistor hook the z3018 calibration [B5] will drive).
//!
//! Coordinate note: ADR-0004 ships 2-D (r,z) as the production MagLIF geometry, but
//! AthenaK's mesh forbids the (nx2=1, nx3>1) topology ("2-D in the x1-x3 plane"), so an
//! axially-resolved run uses the supported 3-D (r,phi,z) grid with a thin azimuthal
//! extent (the IC is axisymmetric -> the phi direction simply replicates the r-z state).
//!
//! USER pgen: build with -D PROBLEM=maglif (not part of the default suite binary).

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coord_geometry.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"
#include "eos/eos_table_3t.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "circuit/drive_source.hpp"
#include "circuit/drive_bphi_bc.hpp"
#include "circuit/faraday_voltage.hpp"
#include "circuit/lumped_circuit.hpp"
#include "pgen/maglif_temperature_seed.hpp"
#include "pgen.hpp"

// Forward declaration (defined below): the cons/div(B) history filler, reused by the
// circuit-mode combined history (#236).
void MagLIFConsHistory(HistoryData *pdata, Mesh *pm);

namespace {
//! 2*pi as a Real (device-safe; avoids the host-only M_PI macro in a device kernel).
constexpr Real kTwoPiM = 6.2831853071795864769;

// File-scope drive state shared with the (stateless-signature) user_bcs_func.
circuit::DriveSource drive_source_;
bool nocurrent_mode_ = false;

// #236 circuit-drive state (ADR-0005 modes B/C promoted into the maglif suite path):
// the voltage-driven lumped-element circuit ODE advanced host-side once per MHD step by
// the user BC, plus the Faraday d(Phi)/dt monitor supplying the mode-C feedback.
bool circuit_mode_ = false;          // current_mode is voltage_rlc / coupled_circuit
circuit::LumpedCircuit circuit_;     // series-loop ODE (ADR-0005 modes B/C, #34)
circuit::FaradayVoltage faraday_;    // d(Phi)/dt monitor -> mode-C load voltage (#31)
Real circ_t_ = 0.0;                  // circuit internal clock (tracks the MHD time)
Real v_load_last_ = 0.0;             // most recent load voltage fed to the circuit
Real r_loss_ = 0.0;                  // loss resistor R_loss (constant; z3018 hook, B5)

// #192 outer-x1 inflow-guard state (set from <problem> at pgen time; read by MagLIFBCs).
bool ox1_inflow_guard_ = false;
Real ox1_vac_dn_ = 0.0;    // vacuum ghost density (code units)
Real ox1_vac_ei_ = 0.0;    // vacuum ghost internal-energy density (code units)

//----------------------------------------------------------------------------------------
//! \fn MagLIFBCs
//! \brief Outer-x1 user BC: fill the cell-centred MHD ghosts by outflow and the B_phi
//!  ghosts from the circuit drive (driven mu0*I(t)/2*pi*r, or the nocurrent 1/r form).
//!  Identical structure to driven_pinch.cpp's boundary (the shared ADR-0005 plumbing).
void MagLIFBCs(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

  // --- #236 circuit drive (modes B/C): advance the circuit ODE once per MHD step ---
  // pm->time is constant across integrator stages and advances once per cycle, so the
  // strict ">" guard advances the circuit exactly once per MHD step (and never on the
  // pre-loop ghost fill, where pm->time == circ_t_).  Mode C first reduces the global
  // poloidal flux and differences it into the Faraday load voltage V_load = d(Phi)/dt
  // (the coupled feedback); mode B has no load feedback and skips the reduction.
  if (circuit_mode_ && pm->time > circ_t_) {
    Real v_load = 0.0;
    if (circuit_.mode == circuit::DriveMode::coupled_circuit) {
      Real flux = circuit::PoloidalFluxGlobal(pm);        // Phi at the current MHD state
      v_load = faraday_.Update(flux, pm->time);           // (Phi - Phi_prev)/(t - t_prev)
    }
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
  par_for("maglif_ccbc_ox1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
  KOKKOS_LAMBDA(int m, int k, int j) {
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::user) { return; }
    for (int i=0; i<ng; ++i) {
      for (int n=0; n<nvar; ++n) { u0(m,n,k,j,ie+i+1) = u0(m,n,k,j,ie); }
    }
  });

  // (2) face-centred B: driven / nocurrent B_phi + outflow B_r, B_z (shared helper).
  // The boundary current is the prescribed waveform (mode A) or the circuit ODE's
  // integrated current (modes B/C, #236).
  Real current = circuit_mode_ ? circuit_.Current() : drive_source_.Current(pm->time);
  Real mu0 = circuit_mode_ ? circuit_.mu0 : drive_source_.mu0;
  circuit::ApplyDriveBphiBC(pm, current, mu0, nocurrent_mode_);

  // (3) #192 inflow guard (<problem> ox1_inflow_guard, default off): wherever the last
  // interior cell moves INWARD, replace the zero-gradient hydro ghosts of that column
  // with a static vacuum state (rho=d_vac, v=0, e_int=p0/(gamma-1), magnetic energy
  // from the just-filled drive faces).  The plain copy mirrors an inward wind back as
  // fresh ghost mass and momentum -- an unbounded reservoir: under the resb-diffused
  // drive field the ghost density tracks the rising outer cell, the wind self-amplifies
  // and the snowplow crushes the liner (mass x77,000 by 70 ns at paper resolution).
  // Outflowing columns keep the zero-gradient copy, so the open boundary still vents.
  // Scalars are zeroed; in the tabulated_3t runs scalar 0 (e_ele) then edge-clamps at
  // the table floor, exactly how the below-floor vacuum gap is already handled.
  if (ox1_inflow_guard_) {
    auto &b0 = pmbp->pmhd->b0;
    int nmhd = pmbp->pmhd->nmhd;
    Real dvac = ox1_vac_dn_;
    Real evac = ox1_vac_ei_;
    par_for("maglif_diode_ox1", DevExeSpace(), 0,(nmb-1),0,(n3-1),0,(n2-1),
    KOKKOS_LAMBDA(int m, int k, int j) {
      if (mb_bcs.d_view(m,BoundaryFace::outer_x1) != BoundaryFlag::user) { return; }
      if (u0(m,IM1,k,j,ie) >= 0.0) { return; }
      for (int i=0; i<ng; ++i) {
        int ig = ie+i+1;
        Real br   = 0.5*(b0.x1f(m,k,j,ig) + b0.x1f(m,k,j,ig+1));
        Real bphi = 0.5*(b0.x2f(m,k,j,ig) + b0.x2f(m,k,j+1,ig));
        Real bz   = 0.5*(b0.x3f(m,k,j,ig) + b0.x3f(m,k+1,j,ig));
        u0(m,IDN,k,j,ig) = dvac;
        u0(m,IM1,k,j,ig) = 0.0;
        u0(m,IM2,k,j,ig) = 0.0;
        u0(m,IM3,k,j,ig) = 0.0;
        u0(m,IEN,k,j,ig) = evac + 0.5*(br*br + bphi*bphi + bz*bz);
        for (int n=nmhd; n<nvar; ++n) { u0(m,n,k,j,ig) = 0.0; }
      }
    });
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void MagLIFConsHistory()
//! \brief User history output: AMR-safe global diagnostics of the MagLIF implosion over
//! all ACTIVE cells of all local MeshBlocks.  Enrolled only when <problem> user_hist=true
//! (the AMR-vs-uniform verification inputs maglif_amr / maglif_uniform.athinput); the
//! plain integrated-target run (#32) leaves it off, so that setup is unchanged.
//!
//! All quantities are CYLINDRICAL finite-volume integrals/reductions (cell volume
//! 1/2(rp^2-rm^2)*dphi*dz, NOT the Cartesian dx1*dx2*dx3 the built-in mhd history uses),
//! so they are the genuinely-conserved integrals AthenaK's conservative AMR
//! restriction/prolongation preserves -- hence directly comparable between an AMR run and
//! a uniform run, and continuous across a refinement event (issue #43).  Columns:
//!   mass   = sum rho dV               (total mass; extensive -> rank SUM is correct)
//!   etot   = sum E_mhd dV             (total MHD energy)
//!   massr  = sum rho*r dV             (mass-weighted-radius numerator; <r>=massr/mass)
//!   maxdivb= max |div(B)| (cyl FV)    (the #38 div-free-prolongation invariant)
//!   maxdens= max rho                  (peak compression)
//!   nmb    = local MeshBlock count    (rises when AMR refines)
//!   ncells = active cell count        (AMR work proxy for the speedup vs uniform)
//! History reduces by SUM across ranks (history.cpp); in a serial (_cpu) run that is the
//! identity, so the two MAX columns report true global maxima -- this probe is _cpu-only
//! by construction, exactly like cyl_field_loop's div(B) history (issue #38).

void MagLIFConsHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  // When the grey-FLD operator-split radiation field is live (coupled stack, #116/[B3])
  // append an extra "erad" column = sum E_r dV.  Together with "etot" (MHD total energy)
  // this closes the species-split energy budget of the coupled run; with FLD off (the
  // ideal-MHD benchmarks) the column is absent so the history is byte-identical.
  bool have_rad = (pmbp->pmhd != nullptr && pmbp->pmhd->pfld_op != nullptr);
  pdata->nhist = have_rad ? 8 : 7;
  pdata->label[0] = "mass";
  pdata->label[1] = "etot";
  pdata->label[2] = "massr";
  pdata->label[3] = "maxdivb";
  pdata->label[4] = "maxdens";
  pdata->label[5] = "nmb";
  pdata->label[6] = "ncells";
  if (have_rad) { pdata->label[7] = "erad"; }

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1;
  bool multi_d = pm->multi_d;
  bool three_d = pm->three_d;
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto u0 = pmbp->pmhd->u0;
  auto b0 = pmbp->pmhd->b0;
  auto csys = pmbp->pcoord->coord_system;

  const int ni   = (ie - is + 1);
  const int nji  = (je - js + 1)*ni;
  const int nkji = (ke - ks + 1)*nji;

  // (1) extensive cylindrical-volume integrals: mass, total energy, mass*radius.
  Real mass = 0.0, etot = 0.0, massr = 0.0;
  Kokkos::parallel_reduce("maglif_hist_sum",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &lmass, Real &letot, Real &lmassr) {
    int m = idx/nkji;
    int kji = idx - m*nkji;
    int k = kji/nji;
    int ji = kji - k*nji;
    int j = ji/ni;
    int i = (ji - j*ni) + is;
    j += js;
    k += ks;
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real rm = LeftEdgeX(i  -is, nx1, x1min, x1max);
    Real rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
    Real vol = CellVolume(csys, dx1, dx2, dx3, rm, rp);
    Real dn = u0(m,IDN,k,j,i);
    lmass  += vol*dn;
    letot  += vol*u0(m,IEN,k,j,i);
    lmassr += vol*dn*x1v;
  }, Kokkos::Sum<Real>(mass), Kokkos::Sum<Real>(etot), Kokkos::Sum<Real>(massr));

  // (2) intensive maxima: peak density and the cyl finite-volume max|div(B)|.  The div(B)
  // formula mirrors cyl_field_loop's history probe / the mhd_divb derived variable.
  Real maxdens = 0.0, maxdivb = 0.0;
  Kokkos::parallel_reduce("maglif_hist_max",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
  KOKKOS_LAMBDA(const int idx, Real &ldmax, Real &lbmax) {
    int m = idx/nkji;
    int kji = idx - m*nkji;
    int k = kji/nji;
    int ji = kji - k*nji;
    int j = ji/ni;
    int i = (ji - j*ni) + is;
    j += js;
    k += ks;
    Real dx1 = size.d_view(m).dx1;
    Real dx2 = size.d_view(m).dx2;
    Real dx3 = size.d_view(m).dx3;
    Real x1min = size.d_view(m).x1min;
    Real x1max = size.d_view(m).x1max;
    Real rm = LeftEdgeX(i  -is, nx1, x1min, x1max);
    Real rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
    Real vol = CellVolume(csys, dx1, dx2, dx3, rm, rp);
    Real divb = (Face1Area(csys, dx2, dx3, rp)*b0.x1f(m,k,j,i+1)
               - Face1Area(csys, dx2, dx3, rm)*b0.x1f(m,k,j,i))/vol;
    if (multi_d) {
      divb += Face2Area(csys, dx1, dx3)*(b0.x2f(m,k,j+1,i) - b0.x2f(m,k,j,i))/vol;
    }
    if (three_d) {
      divb += Face3Area(csys, dx1, dx2, rm, rp)
              *(b0.x3f(m,k+1,j,i) - b0.x3f(m,k,j,i))/vol;
    }
    ldmax = fmax(ldmax, u0(m,IDN,k,j,i));
    lbmax = fmax(lbmax, fabs(divb));
  }, Kokkos::Max<Real>(maxdens), Kokkos::Max<Real>(maxdivb));

  // (3) radiation energy integral E_rad = sum E_r dV (coupled stack only, #116).  Same
  // cylindrical finite-volume weight as the mass/etot integrals, so etot+erad is the
  // genuinely-conserved combined energy the matter-radiation coupling exchanges between.
  Real erad_tot = 0.0;
  if (have_rad) {
    auto erad = pmbp->pmhd->erad;
    Kokkos::parallel_reduce("maglif_hist_erad",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nkji),
    KOKKOS_LAMBDA(const int idx, Real &lerad) {
      int m = idx/nkji;
      int kji = idx - m*nkji;
      int k = kji/nji;
      int ji = kji - k*nji;
      int j = ji/ni;
      int i = (ji - j*ni) + is;
      j += js;
      k += ks;
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real x1min = size.d_view(m).x1min;
      Real x1max = size.d_view(m).x1max;
      Real rm = LeftEdgeX(i  -is, nx1, x1min, x1max);
      Real rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      Real vol = CellVolume(csys, dx1, dx2, dx3, rm, rp);
      lerad += vol*erad(m,0,k,j,i);
    }, Kokkos::Sum<Real>(erad_tot));
  }

  pdata->hdata[0] = mass;
  pdata->hdata[1] = etot;
  pdata->hdata[2] = massr;
  pdata->hdata[3] = maxdivb;
  pdata->hdata[4] = maxdens;
  pdata->hdata[5] = static_cast<Real>(nmb);
  pdata->hdata[6] = static_cast<Real>(nmb*nkji);
  if (have_rad) { pdata->hdata[7] = erad_tot; }
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) { pdata->hdata[n] = 0.0; }
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn void MagLIFAppendCircuitHistory()
//! \brief Append the four circuit-drive columns (I_circuit, V_cap, V_load, Bphi_flux)
//! after the pdata->nhist columns already filled (#236, ADR-0005 modes B/C).  MPI-safe
//! by construction (the coupled_liner pattern): the circuit scalars are global host
//! values (identical on every rank) written on rank 0 only, and the flux column is the
//! per-rank LOCAL partial, so history.cpp's in-place MPI_Reduce(SUM) yields the global
//! values on the root rank.

void MagLIFAppendCircuitHistory(HistoryData *pdata, Mesh *pm) {
  int n0 = pdata->nhist;
  pdata->nhist = n0 + 4;
  pdata->label[n0+0] = "I_circuit";
  pdata->label[n0+1] = "V_cap";
  pdata->label[n0+2] = "V_load";
  pdata->label[n0+3] = "Bphi_flux";

  bool root = (global_variable::my_rank == 0);
  pdata->hdata[n0+0] = root ? circuit_.Current()    : 0.0;
  pdata->hdata[n0+1] = root ? circuit_.CapVoltage() : 0.0;
  pdata->hdata[n0+2] = root ? v_load_last_          : 0.0;
  pdata->hdata[n0+3] = circuit::PoloidalFluxLocal(pm);
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) { pdata->hdata[n] = 0.0; }
}

//----------------------------------------------------------------------------------------
//! \fn void MagLIFCircuitHistory()
//! \brief User history for the circuit-driven modes B/C: the circuit waveforms only --
//! the observables the circuit verification (#236) baselines.

void MagLIFCircuitHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 0;
  MagLIFAppendCircuitHistory(pdata, pm);
}

//----------------------------------------------------------------------------------------
//! \fn void MagLIFConsCircuitHistory()
//! \brief Combined user history for a circuit-driven run that also asks for the
//! conservation/div(B) diagnostics (problem/user_hist=true): the MagLIFConsHistory
//! columns first, then the four circuit columns.

void MagLIFConsCircuitHistory(HistoryData *pdata, Mesh *pm) {
  MagLIFConsHistory(pdata, pm);
  MagLIFAppendCircuitHistory(pdata, pm);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Initialize the MagLIF liner/fuel/vacuum target (+ B_z premag + perturbation
//!  seeding) and enroll the circuit-driven B_phi outer boundary.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Enroll the user BC (must happen on restart too, before pgen.cpp's enrollment check).
  user_bcs_func = MagLIFBCs;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "maglif requires an <mhd> block" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pmbp->pcoord->coord_system == CoordSystem::cartesian) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "maglif requires <coord> system = cylindrical" << std::endl;
    exit(EXIT_FAILURE);
  }

  // ---- drive-source configuration (<problem> block; ADR-0005 modes A/B/C) ----
  // Mode A (driven / nocurrent) keeps the prescribed-current source; modes B/C (#236)
  // instead configure the voltage-driven lumped-element circuit (the coupled_liner
  // wiring promoted into this suite path).  Read before the restart return so a
  // restarted run keeps the same drive; the circuit state itself restarts from rest at
  // the restart time (ODE-state continuity across restarts is out of scope, as in
  // coupled_liner).  Every mode-A deck takes the else branch below unchanged
  // (byte-identical reads in the same order).
  std::string mode = pin->GetOrAddString("problem", "current_mode", "driven");
  nocurrent_mode_ = (mode == "nocurrent");
  circuit_mode_ = (mode == "voltage_rlc" || mode == "coupled_circuit");
  if (!circuit_mode_ && !nocurrent_mode_ && mode != "driven") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "unknown problem/current_mode '" << mode
              << "' (expected driven|nocurrent|voltage_rlc|coupled_circuit)"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (circuit_mode_) {
    // ADR-0005 modes B/C: the open-circuit voltage source V_oc(t) reuses the
    // prescribed-source waveform machinery (constant/linear_ramp/sin_squared/tabulated;
    // its i0 is the PEAK VOLTAGE) + deck-driven series elements.  r_loss is the
    // loss-resistor hook (constant here; the z3018 calibration [B5] will drive it).
    circuit_.mode = (mode == "coupled_circuit") ? circuit::DriveMode::coupled_circuit
                                                : circuit::DriveMode::voltage_rlc;
    std::string vwf = pin->GetOrAddString("problem", "voltage_waveform", "constant");
    circuit_.voltage_source.waveform = circuit::ParseWaveform(vwf);
    circuit_.voltage_source.i0     = pin->GetOrAddReal("problem", "v_oc", 1.0);
    circuit_.voltage_source.t_rise = pin->GetOrAddReal("problem", "t_rise", 1.0);
    circuit_.L   = pin->GetOrAddReal("problem", "circuit_L", 1.0);
    circuit_.cap = pin->GetOrAddReal("problem", "circuit_C", 0.0);  // <=0 -> pure RL
    circuit_.Z0  = pin->GetOrAddReal("problem", "circuit_Z0", 0.0);
    circuit_.mu0 = pin->GetOrAddReal("problem", "mu0", 1.0);
    r_loss_      = pin->GetOrAddReal("problem", "r_loss", 0.0);
    if (circuit_.voltage_source.waveform == circuit::CurrentWaveform::tabulated) {
      std::string vfile = pin->GetOrAddString("problem", "voltage_file", "unset");
      circuit::ReadCurrentWaveform(vfile, circuit_.voltage_source);
    }
    // Start the circuit (and its clock + flux monitor) from rest at the mesh time.
    circuit_.Reset(0.0, 0.0);
    faraday_.Reset();
    circ_t_ = pmy_mesh_->time;
    v_load_last_ = 0.0;
  } else {
    std::string wf = pin->GetOrAddString("problem", "current_waveform", "linear_ramp");
    drive_source_.waveform = circuit::ParseWaveform(wf);
    drive_source_.i0     = pin->GetOrAddReal("problem", "i0", 1.0);
    drive_source_.t_rise = pin->GetOrAddReal("problem", "t_rise", 1.0);
    drive_source_.mu0    = pin->GetOrAddReal("problem", "mu0", 1.0);
    if (drive_source_.waveform == circuit::CurrentWaveform::tabulated) {
      std::string cfile = pin->GetOrAddString("problem", "current_file", "unset");
      circuit::ReadCurrentWaveform(cfile, drive_source_);
    }
  }

  // ---- #192 outer-x1 inflow guard (<problem> ox1_inflow_guard; default off) ----
  // Read BEFORE the restart return (like the drive config above) so a restarted run
  // keeps the guarded boundary.  The vacuum ghost state reuses the IC's <problem> and
  // <units> values; GetOrAdd* is idempotent, so the duplicate reads below see exactly
  // the numbers the IC normalisation reads later.  e_int uses the ideal p0/(gamma-1):
  // in the tabulated_3t runs the vacuum sits below the table floor and edge-clamps
  // regardless, so the exact tiny value is immaterial -- what the guard must pin is
  // the ghost mass and momentum reservoir.
  ox1_inflow_guard_ = pin->GetOrAddBoolean("problem", "ox1_inflow_guard", false);
  if (ox1_inflow_guard_) {
    Real g_rho = pin->GetOrAddReal("units", "density_cgs", 1.0);
    Real g_vel = pin->GetOrAddReal("units", "velocity_cgs", 1.0);
    Real g_prs = g_rho*g_vel*g_vel;
    Real g_gm1 = pin->GetOrAddReal("mhd", "gamma", 5.0/3.0) - 1.0;
    ox1_vac_dn_ = pin->GetOrAddReal("problem", "d_vac", 0.01) / g_rho;
    ox1_vac_ei_ = (pin->GetOrAddReal("problem", "p0", 1.0e-3) / g_prs) / g_gm1;
  }

  // Optional: enroll the conservation/div(B) history output (AMR-vs-uniform check, #43).
  // Default off, so the plain integrated-target run (#32) is unchanged.  Set on restart
  // too (before the restart return) so a resumed run keeps writing the history columns.
  // The circuit-driven modes B/C (#236) ALWAYS expose the circuit waveforms (I, V_C,
  // V_load, Phi) -- the verification observables; with user_hist=true the cons columns
  // come first and the circuit columns are appended.
  bool cons_hist = pin->GetOrAddBoolean("problem", "user_hist", false);
  if (circuit_mode_) {
    user_hist = true;
    user_hist_func = cons_hist ? MagLIFConsCircuitHistory : MagLIFCircuitHistory;
  } else if (cons_hist) {
    user_hist = true;
    user_hist_func = MagLIFConsHistory;
  }

  if (restart) return;

  // ---- dimensional (SI/CGS) unit system (<units> block; #119/[C2]) ----
  // The replication benchmarks (Phase C) are set up against the published SI geometry, so
  // the <problem> lengths/densities/pressure are expressed in physical units and the pgen
  // normalises them to code units here.  Three independent CGS scales fix the system:
  //   length_cgs   -- cm per code length (so a code radius r maps to r*length_cgs cm)
  //   density_cgs  -- g/cm^3 per code density
  //   velocity_cgs -- cm/s per code velocity (-> time=length/vel, pressure=density*vel^2)
  // ALL DEFAULT to 1.0, so a benchmark with no <units> block (every pre-#119 input) sees
  // an identity normalisation and stays byte-identical.  The <mesh> extent is in code
  // units; an SI input sets it to (physical extent)/length_cgs to stay consistent here.
  Real u_len = pin->GetOrAddReal("units", "length_cgs", 1.0);
  Real u_rho = pin->GetOrAddReal("units", "density_cgs", 1.0);
  Real u_vel = pin->GetOrAddReal("units", "velocity_cgs", 1.0);
  Real u_prs = u_rho*u_vel*u_vel;                 // code pressure unit = rho*v^2

  // ---- SI drive calibration (ADR-0010) ----
  // When problem/current_si=true, convert the tabulated load trace from its faithful SI
  // columns (time [ns], current [MA]) into code units via the factors fixed by <units>.
  // Default false -> the trace is consumed in code units, so every pre-#174 tabulated
  // input (and the analytic waveforms, whose i0/t_rise are code-unit) is unchanged.
  // Converting here (after <units>, before i_init) means both the boundary drive and the
  // IC-time current see the calibrated trace; mu0=1 is left untouched.
  if (pin->GetOrAddBoolean("problem", "current_si", false) &&
      drive_source_.waveform == circuit::CurrentWaveform::tabulated) {
    circuit::SiDriveUnits du = circuit::CalibrateSiDrive(u_len, u_rho, u_vel);
    int ntab = static_cast<int>(drive_source_.t_tab.size());
    for (int p = 0; p < ntab; ++p) {
      drive_source_.t_tab[p] *= du.time_per_ns;
      drive_source_.i_tab[p] *= du.current_per_ma;
    }
  }

  // ---- target geometry + state (<problem>; physical units, normalised by <units>) ----
  // r_rod > 0 selects the converging Richtmyer-Meshkov geometry (#119/[C2]): a dense
  // on-axis ROD carrying the seeded perturbation, a working-fluid gap out to r_fuel, then
  // the driven liner [r_fuel, r_liner) and the vacuum [r_liner, R).  r_rod <= 0 (default)
  // is the original 3-region fuel/liner/vacuum target (MRT/RM/ICF), byte-identical.
  Real r_rod   = pin->GetOrAddReal("problem", "r_rod", 0.0) / u_len;
  Real d_rod   = pin->GetOrAddReal("problem", "d_rod", 1.0) / u_rho;
  Real r_fuel  = pin->GetOrAddReal("problem", "r_fuel", 0.4) / u_len;
  Real r_liner = pin->GetOrAddReal("problem", "r_liner", 0.6) / u_len;
  Real d_fuel  = pin->GetOrAddReal("problem", "d_fuel", 0.05) / u_rho;
  Real d_liner = pin->GetOrAddReal("problem", "d_liner", 1.0) / u_rho;
  Real d_vac   = pin->GetOrAddReal("problem", "d_vac", 0.01) / u_rho;
  Real p0      = pin->GetOrAddReal("problem", "p0", 1.0e-3) / u_prs;
  Real bz0     = pin->GetOrAddReal("problem", "bz0", 0.0);   // axial premag field (code)
  bool rod_target = (r_rod > 0.0);
  // Modes B/C (#236) start from the circuit's rest state (I=0 -> no IC drive field).
  Real i_init  = circuit_mode_ ? circuit_.Current() : drive_source_.Current(0.0);
  Real mu0     = circuit_mode_ ? circuit_.mu0 : drive_source_.mu0;

  // ---- perturbation seeding (axial, requires z resolved) ----
  // Build a unified per-mode (k_n, amplitude, phase) table; dr(z) = sum over modes of
  //   amp_n cos(2*pi*k_n (z-z0)/Lz + phase_n).
  // none -> 0 modes; single_mode -> 1 mode; roughness -> a band [n_min,n_max].
  std::string pert = pin->GetOrAddString("problem", "perturbation", "none");
  Real pert_amp    = pin->GetOrAddReal("problem", "pert_amp", 0.0) / u_len;
  int  pert_mode   = pin->GetOrAddInteger("problem", "pert_mode", 1);
  Real pert_phase  = pin->GetOrAddReal("problem", "pert_phase", 0.0);
  int  pert_nmin   = pin->GetOrAddInteger("problem", "pert_nmin", 1);
  int  pert_nmax   = pin->GetOrAddInteger("problem", "pert_nmax", 8);
  int  pert_seed   = pin->GetOrAddInteger("problem", "pert_seed", 12345);

  // Ellison Eq.16 temperature-seed parameters (perturbation=temperature; #161).  T0/Tmin/
  // dT are KELVIN and enter only through the dimensionless ratio T/T0, so they are NOT
  // unit-normalised; the decay length is a physical length and IS normalised by <units>.
  // The decay reference r_o defaults to the outer liner radius r_liner (Eq.16's r_o).
  bool temp_seed   = (pert == "temperature");
  Real pert_T0     = pin->GetOrAddReal("problem", "pert_T0", 293.0);
  Real pert_Tmin   = pin->GetOrAddReal("problem", "pert_Tmin", 273.0);
  Real pert_dT     = pin->GetOrAddReal("problem", "pert_dT", 100.0);
  Real pert_decay  = pin->GetOrAddReal("problem", "pert_decay", 0.05) / u_len;
  Real pert_r_o    = pin->GetOrAddReal("problem", "pert_decay_r0", -1.0);
  pert_r_o = (pert_r_o > 0.0) ? (pert_r_o / u_len) : r_liner;   // default = outer liner

  int nmodes = 0;
  if (pmy_mesh_->three_d) {           // axial perturbation only when z (x3) is resolved
    if (pert == "single_mode") {
      nmodes = 1;
    } else if (pert == "roughness") {
      nmodes = (pert_nmax >= pert_nmin) ? (pert_nmax - pert_nmin + 1) : 0;
    } else if (pert != "none" && pert != "temperature") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "unknown problem/perturbation '" << pert
                << "' (expected none|single_mode|roughness|temperature)" << std::endl;
      exit(EXIT_FAILURE);
    }
  } else if (pert != "none" && pert != "temperature" && pert_amp != 0.0) {
    std::cout << "### WARNING: maglif problem/perturbation='" << pert << "' ignored: "
              << "axial direction x3 is not resolved (nx3=1); run a 3-D (r,phi,z) grid."
              << std::endl;
  }

  // Per-mode tables filled on the host then mirrored to the device for the IC kernel.
  DualArray1D<Real> mode_k("maglif_mode_k", (nmodes > 0) ? nmodes : 1);
  DualArray1D<Real> mode_amp("maglif_mode_amp", (nmodes > 0) ? nmodes : 1);
  DualArray1D<Real> mode_phase("maglif_mode_phase", (nmodes > 0) ? nmodes : 1);
  if (nmodes == 1 && pert == "single_mode") {
    mode_k.h_view(0)     = static_cast<Real>(pert_mode);
    mode_amp.h_view(0)   = pert_amp;
    mode_phase.h_view(0) = pert_phase;
  } else if (nmodes > 0) {                 // roughness: deterministic random phases
    std::mt19937 rng(static_cast<unsigned>(pert_seed));
    std::uniform_real_distribution<Real> uni(0.0, kTwoPiM);
    Real amp_per = pert_amp/static_cast<Real>(nmodes);   // aggregate -> per-mode share
    for (int n = 0; n < nmodes; ++n) {
      mode_k.h_view(n)     = static_cast<Real>(pert_nmin + n);
      mode_amp.h_view(n)   = amp_per;
      mode_phase.h_view(n) = uni(rng);
    }
  } else {                                  // none / unresolved: a single zero-amp mode
    mode_k.h_view(0)     = 0.0;
    mode_amp.h_view(0)   = 0.0;
    mode_phase.h_view(0) = 0.0;
  }
  mode_k.template modify<HostMemSpace>();
  mode_amp.template modify<HostMemSpace>();
  mode_phase.template modify<HostMemSpace>();
  mode_k.template sync<DevExeSpace>();
  mode_amp.template sync<DevExeSpace>();
  mode_phase.template sync<DevExeSpace>();
  auto k_d   = mode_k.d_view;
  auto amp_d = mode_amp.d_view;
  auto phs_d = mode_phase.d_view;
  int nm = (nmodes > 0) ? nmodes : 1;       // loop bound (dummy mode is zero-amplitude)

  // Global axial extent for the perturbation wavelength (full mesh, not the MeshBlock).
  Real z0 = pmy_mesh_->mesh_size.x3min;
  Real lz = pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min;

  // Global mesh mins + root-level cell counts, used to map each cell to a GLOBAL cell id
  // for the temperature seed's per-zone random draw (so the seeded field is reproducible
  // by pert_seed and identical under any domain decomposition; see the header).
  Real mx1min = pmy_mesh_->mesh_size.x1min;
  Real mx2min = pmy_mesh_->mesh_size.x2min;
  Real mx3min = pmy_mesh_->mesh_size.x3min;
  std::uint64_t ngx = static_cast<std::uint64_t>(pmy_mesh_->mesh_indcs.nx1);
  std::uint64_t ngy = static_cast<std::uint64_t>(pmy_mesh_->mesh_indcs.nx2);
  std::uint64_t tseed = static_cast<std::uint64_t>(
      static_cast<unsigned>(pert_seed));

  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &size = pmbp->pmb->mb_size;
  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  Real gm1 = eos.gamma - 1.0;
  bool is_ideal = eos.is_ideal;
  // Tabulated 3T (IONMIX) initial-condition state (ADR-0010 / ADR-0002).  The code-unit
  // scaled table held on the MHD package (mhd.cpp) is captured by value into the kernel
  // (shallow View copy, device-valid, like the live ConsToPrim path); the per-cell
  // cold-LTE temperature `t_init` (eV) is clamped up to the table floor.  Only used in
  // the tabulated branch -- for the ideal EOS the table is empty and unused.
  eos_table_3t::EosTable3T tbl_d = pmbp->pmhd->eos_tbl;
  Real t_init  = pin->GetOrAddReal("problem", "t_init_ev", 1.0);
  Real t_floor = is_ideal ? 0.0 : tbl_d.TempMin();
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  // Passive scalar(s): in the converging-RM (rod) target, tag the on-axis ROD material
  // (s=1 in the rod, 0 elsewhere) so the rod/working-fluid interface can be tracked by
  // composition -- like the experiment's material-contrast radiography -- robustly
  // through convergence, where a density threshold cannot separate the rod from the
  // equally dense (same-material) liner once shocked working fluid piles up between them.
  int nmhd = pmbp->pmhd->nmhd;
  int nscal = pmbp->pmhd->nscalars;

  // #182/[P7b], ADR-0012 gap b: optionally SOURCE the operator-split grey-FLD `erad` from
  // the local gas state inside the IC kernel below (the LTE grey energy a*T^4 at the cell
  // temperature, partitioned OUT of the gas internal energy), instead of the uniform
  // radiative-equilibrium seed further down.  Read ONLY when grey FLD is live
  // (pfld_op != nullptr) and gated on a knob defaulting OFF, so FLD-off runs -- and
  // existing FLD runs that leave the knob off (e.g. the coupled smoke/B3 stack) -- stay
  // byte-identical (they keep the uniform seed).  The radiation constant `a` and heat
  // capacity `c_v` come from the same <mhd> mrad_* params the coupling uses (present in
  // the athinput regardless of mrad_coupling), so the `fld` (no-coupling) and `fld+mrad`
  // runs source the SAME field -- both then differ from the operators-off baseline.
  bool fld_on = (pmbp->pmhd->pfld_op != nullptr);
  bool source_erad = fld_on &&
      pin->GetOrAddBoolean("mhd", "fld_source_erad_from_gas", false);
  Real src_arad = source_erad ? pin->GetOrAddReal("mhd", "mrad_arad", 1.0) : 0.0;
  Real src_cv   = source_erad ? pin->GetOrAddReal("mhd", "mrad_cv",   1.0) : 0.0;
  auto erad_d   = pmbp->pmhd->erad;   // valid handle when fld_on; written iff source_erad

  par_for("pgen_maglif", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    // Axial perturbation displacement dr(z) (0 when z unresolved / perturbation=none).
    Real dr = 0.0;
    if (lz > 0.0) {
      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      for (int n = 0; n < nm; ++n) {
        dr += amp_d(n)*cos(kTwoPiM*k_d(n)*(x3v - z0)/lz + phs_d(n));
      }
    }
    // Density by radial region.  In the converging-RM (rod) target the seed displaces the
    // on-axis ROD interface (r_rod+dr) and the liner interfaces stay clean; otherwise the
    // seed rigidly displaces both liner interfaces (the original MRT/RM target).
    Real rr = rod_target ? (r_rod + dr) : 0.0;
    Real rf = rod_target ? r_fuel : (r_fuel + dr);
    Real rl = rod_target ? r_liner : (r_liner + dr);
    Real dens;
    if (rod_target && x1v < rr) {
      dens = d_rod;                       // dense on-axis rod (perturbed interface)
    } else if (x1v < rf) {
      dens = d_fuel;                       // fuel / working fluid
    } else if (x1v < rl) {
      dens = d_liner;                      // driven liner (dense shell)
    } else {
      dens = d_vac;                        // vacuum gap (carries the drive B_phi)
    }
    u0(m,IDN,k,j,i) = dens;
    u0(m,IM1,k,j,i) = 0.0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;

    // Passive-scalar rod tag (conserved s*d): s=1 in the rod, 0 elsewhere.  Advected with
    // the flow, so the rod/working-fluid interface is the s=0.5 contour at all times.
    Real s_rod = (rod_target && x1v < rr) ? 1.0 : 0.0;
    for (int n = 0; n < nscal; ++n) { u0(m, nmhd+n, k, j, i) = s_rod*dens; }

    // B_z premag (x3-face, uniform); B_r=0; driven B_phi (x2-face) only in the vacuum
    // OUTSIDE the (perturbed) liner -- the load current is enclosed by the liner.
    Real bphi = (x1v >= rl) ? circuit::DrivenBphi(i_init, x1v, mu0) : 0.0;
    b0.x1f(m,k,j,i) = 0.0;
    b0.x2f(m,k,j,i) = bphi;
    b0.x3f(m,k,j,i) = bz0;
    if (i==ie) { b0.x1f(m,k,j,i+1) = 0.0; }
    if (j==je) { b0.x2f(m,k,j+1,i) = bphi; }
    if (k==ke) { b0.x3f(m,k+1,j,i) = bz0; }

    // Ellison Eq.16 temperature seed (perturbation=temperature): a per-zone random
    // electron+ion temperature applied at FIXED density (mass conserving) as a pressure
    // factor T(r,z)/T0.  Seeded only on the target material (x1v < r_liner); the envelope
    // exp((r-r_o)/lambda) decays inward from the outer liner surface, so the vacuum is
    // unperturbed and the factor is positive (>= Tmin/T0).  dT=0 -> T=T0 -> factor 1 ->
    // identical to perturbation=none.  This sets a single-temperature (ideal-gas) state;
    // the per-species e_ele/e_ion split is the tabulated-EOS path ([P1]/[P2]).
    Real tfac = 1.0;
    if (temp_seed && x1v < rl) {
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;
      Real x2v = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      Real x3v = CellCenterX(k-ks, nx3, size.d_view(m).x3min, size.d_view(m).x3max);
      std::uint64_t gi = static_cast<std::uint64_t>((x1v - mx1min)/dx1);
      std::uint64_t gj = static_cast<std::uint64_t>((x2v - mx2min)/dx2);
      std::uint64_t gk = static_cast<std::uint64_t>((x3v - mx3min)/dx3);
      std::uint64_t gid = (gk*ngy + gj)*ngx + gi;
      Real g = maglif_tseed::CellGaussian(gid, tseed);
      Real tcell = maglif_tseed::EllisonTemperature(x1v, pert_r_o, pert_decay,
                                                    pert_T0, pert_Tmin, pert_dT, g);
      tfac = tcell/pert_T0;
    }

    if (is_ideal) {
      u0(m,IEN,k,j,i) = (p0*tfac)/gm1 + 0.5*(bphi*bphi + bz0*bz0);
    } else {
      // Tabulated 3T IC (ADR-0010): a single cold-LTE temperature per cell
      // (T_e=T_i=t_init, optionally modulated by the Ellison temperature seed via tfac),
      // clamped up to the table floor.  Look up the per-species SPECIFIC energies on the
      // code-unit-scaled table and fill BOTH the conserved total energy and the e_ele
      // passive scalar (the electron internal energy density that rides scalar 0; the ion
      // energy is recovered by subtraction in ConsToPrim2T).  This is the else branch the
      // faithful dimensional run needs -- previously total energy was filled only when
      // is_ideal, leaving the tabulated energy uninitialised (#174).
      Real t_cell  = fmax(t_init*tfac, t_floor);
      Real e_ele_d = dens*tbl_d.EnergyEle(dens, t_cell);
      Real e_ion_d = dens*tbl_d.EnergyIon(dens, t_cell);
      u0(m,IEN,k,j,i)  = e_ele_d + e_ion_d + 0.5*(bphi*bphi + bz0*bz0);
      u0(m,nmhd,k,j,i) = e_ele_d;   // electron internal energy density rides scalar 0
    }

    // #182/[P7b], ADR-0012 gap b: source the grey-FLD radiation energy from the local gas
    // state.  Partition the LTE grey energy a*T^4 (at the cell temperature) OUT of the
    // gas internal energy, conserving E_gas + E_rad, so the radiation field is physically
    // fed and mrad has a real, spatially-structured field to exchange against.  The
    // temperature is the faithful tabulated_3t cell temperature (eV) in the tabulated
    // path, else the coupling's gas temperature T = e_int/c_v.  Capped to half the local
    // gas thermal energy so the gas stays positive (the coefficients are uncalibrated
    // placeholders; SI calibration is #P7d/#184); the cap also makes erad track the gas
    // energy into the low-density vacuum, giving FLD a real gradient at the liner edge to
    // diffuse.  Off => the uniform equilibrium seed below runs (byte-identical FLD runs).
    if (source_erad) {
      Real e_int = u0(m,IEN,k,j,i) - 0.5*(bphi*bphi + bz0*bz0);
      Real t_src = is_ideal ? ((src_cv > 0.0) ? e_int/src_cv : 0.0)
                            : fmax(t_init*tfac, t_floor);
      Real erad_cell = src_arad*t_src*t_src*t_src*t_src;
      erad_cell = fmin(erad_cell, 0.5*fmax(e_int, 0.0));
      u0(m,IEN,k,j,i) -= erad_cell;
      erad_d(m,0,k,j,i) = erad_cell;
    }
  });

  // ---- seed the operator-split grey-FLD radiation field to radiative equilibrium ----
  // The coupled radiation-MHD stack (#116/[B3]) needs the standalone `erad` array (live
  // only when <mhd> fld_operator_split=true) initialised; the MHD ctor allocates it
  // uninitialised.  The target IC is uniform-pressure gas (e_gas = p0/(gamma-1)), so the
  // point-implicit matter-radiation coupling's equilibrium E_r = a*T^4 with T = e_gas/c_v
  // (c_v = <mhd> mrad_cv, a = <mhd> mrad_arad) is a single uniform value.  Seeding to
  // equilibrium means the coupling starts balanced (no spurious initial energy kick) and
  // the FLD diffusion + coupling then redistribute energy conservatively.  Guarded on
  // pfld_op so the ideal-MHD benchmarks (FLD off) stay byte-identical: no array, no fill.
  // Skipped when `fld_source_erad_from_gas` is on (#182/[P7b]): the kernel above already
  // filled `erad` per cell from the gas state and partitioned it out of the gas energy.
  if (fld_on && !source_erad) {
    Real e_gas = p0/gm1;
    Real cv    = pmbp->pmhd->mrad_cv;
    Real arad  = pmbp->pmhd->mrad_arad;
    Real teq   = (cv > 0.0) ? e_gas/cv : 0.0;
    Real erad_eq = arad*teq*teq*teq*teq;
    Kokkos::deep_copy(pmbp->pmhd->erad, erad_eq);
  }

  return;
}
