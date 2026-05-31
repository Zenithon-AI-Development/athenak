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
//! Drive wiring (ADR-0005): reuses the circuit::DriveSource + the reusable circuit-driven
//! B_phi outer-radial user boundary (circuit/drive_bphi_bc.hpp, issue [9a]/#21), just as
//! driven_pinch.cpp does -- the enrolled user_bcs_func evaluates I(pm->time) each step.
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
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coord_geometry.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "circuit/drive_source.hpp"
#include "circuit/drive_bphi_bc.hpp"
#include "pgen/maglif_temperature_seed.hpp"
#include "pgen.hpp"

namespace {
//! 2*pi as a Real (device-safe; avoids the host-only M_PI macro in a device kernel).
constexpr Real kTwoPiM = 6.2831853071795864769;

// File-scope drive state shared with the (stateless-signature) user_bcs_func.
circuit::DriveSource drive_source_;
bool nocurrent_mode_ = false;

//----------------------------------------------------------------------------------------
//! \fn MagLIFBCs
//! \brief Outer-x1 user BC: fill the cell-centred MHD ghosts by outflow and the B_phi
//!  ghosts from the circuit drive (driven mu0*I(t)/2*pi*r, or the nocurrent 1/r form).
//!  Identical structure to driven_pinch.cpp's boundary (the shared ADR-0005 plumbing).
void MagLIFBCs(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->pmhd == nullptr) return;

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
  Real current = drive_source_.Current(pm->time);
  circuit::ApplyDriveBphiBC(pm, current, drive_source_.mu0, nocurrent_mode_);
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

  // ---- drive-source configuration (<problem> block; ADR-0005 mode A) ----
  std::string mode = pin->GetOrAddString("problem", "current_mode", "driven");
  nocurrent_mode_ = (mode == "nocurrent");
  std::string wf = pin->GetOrAddString("problem", "current_waveform", "linear_ramp");
  drive_source_.waveform = circuit::ParseWaveform(wf);
  drive_source_.i0     = pin->GetOrAddReal("problem", "i0", 1.0);
  drive_source_.t_rise = pin->GetOrAddReal("problem", "t_rise", 1.0);
  drive_source_.mu0    = pin->GetOrAddReal("problem", "mu0", 1.0);
  if (drive_source_.waveform == circuit::CurrentWaveform::tabulated) {
    std::string cfile = pin->GetOrAddString("problem", "current_file", "unset");
    circuit::ReadCurrentWaveform(cfile, drive_source_);
  }

  // Optional: enroll the conservation/div(B) history output (AMR-vs-uniform check, #43).
  // Default off, so the plain integrated-target run (#32) is unchanged.  Set on restart
  // too (before the restart return) so a resumed run keeps writing the history columns.
  if (pin->GetOrAddBoolean("problem", "user_hist", false)) {
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
  Real i_init  = drive_source_.Current(0.0);
  Real mu0     = drive_source_.mu0;

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
  auto &u0 = pmbp->pmhd->u0;
  auto &b0 = pmbp->pmhd->b0;
  // Passive scalar(s): in the converging-RM (rod) target, tag the on-axis ROD material
  // (s=1 in the rod, 0 elsewhere) so the rod/working-fluid interface can be tracked by
  // composition -- like the experiment's material-contrast radiography -- robustly
  // through convergence, where a density threshold cannot separate the rod from the
  // equally dense (same-material) liner once shocked working fluid piles up between them.
  int nmhd = pmbp->pmhd->nmhd;
  int nscal = pmbp->pmhd->nscalars;

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
  if (pmbp->pmhd->pfld_op != nullptr) {
    Real e_gas = p0/gm1;
    Real cv    = pmbp->pmhd->mrad_cv;
    Real arad  = pmbp->pmhd->mrad_arad;
    Real teq   = (cv > 0.0) ? e_gas/cv : 0.0;
    Real erad_eq = arad*teq*teq*teq*teq;
    Kokkos::deep_copy(pmbp->pmhd->erad, erad_eq);
  }

  return;
}
