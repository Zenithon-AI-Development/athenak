#ifndef MHD_MHD_HPP_
#define MHD_MHD_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file mhd.hpp
//  \brief definitions for MHD class

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"
#include "eos/eos_table_3t.hpp"  // tabulated 3T (IONMIX) EOS state on the package (#159)

// forward declarations
namespace parabolic {
class CompositeParabolicOperator;
}
class EquationOfState;
class Coordinates;
class Viscosity;
class Resistivity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class OrbitalAdvectionFC;
class ShearingBoxCC;
class ShearingBoxFC;
class Driver;
class FLDGreyOperator;
class FLDMultigroupOperator;
class AnisotropicConductionOperator;
class ResistiveBphiOperator;

// function ptr for user-defined MHD boundary functions enrolled in problem generator
namespace mhd {
using MHDBoundaryFnPtr = void (*)(int m, Mesh* pm, MHD* pmhd, DvceArray5D<Real> &u);
}

// constants that enumerate MHD Riemann Solver options
enum class MHD_RSolver {advect, llf, hlle, hlld, roe,   // non-relativistic
                        llf_sr, hlle_sr,                // SR
                        llf_gr, hlle_gr};                       // GR

//----------------------------------------------------------------------------------------
//! \struct MHDTaskIDs
//  \brief container to hold TaskIDs of all mhd tasks

struct MHDTaskIDs {
  TaskID savest;
  TaskID opsfld;
  TaskID opsmgfld;
  TaskID opsacond;
  TaskID opsresb;
  TaskID cplb;       // matter-radiation coupling, before (Strang) -- outside super-step
  TaskID strangb;    // half parabolic super-step, before (Strang)
  TaskID stranga;    // half parabolic super-step, after  (Strang)
  TaskID cpla;       // matter-radiation coupling, after  (Strang) -- outside super-step
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID rkupdt;
  TaskID srctrms;
  TaskID sendu_oa;
  TaskID recvu_oa;
  TaskID restu;
  TaskID sendu;
  TaskID recvu;
  TaskID sendu_shr;
  TaskID recvu_shr;
  TaskID efld;
  TaskID efldsrc;
  TaskID sende;
  TaskID recve;
  TaskID ct;
  TaskID sendb_oa;
  TaskID recvb_oa;
  TaskID restb;
  TaskID sendb;
  TaskID recvb;
  TaskID sendb_shr;
  TaskID recvb_shr;
  TaskID bcs;
  TaskID prol;
  TaskID c2p;
  TaskID newdt;
  TaskID csend;
  TaskID crecv;
};

namespace mhd {

//----------------------------------------------------------------------------------------
//! \class MHD

class MHD {
 public:
  MHD(MeshBlockPack *ppack, ParameterInput *pin);
  ~MHD();

  // data
  ReconstructionMethod recon_method;
  MHD_RSolver rsolver_method;
  EquationOfState *peos;   // chosen EOS

  // Tabulated 3T (IONMIX) real-material EOS state, populated ONLY when eos=tabulated_3t
  // (issue [P1]/#159, ADR-0002/0007).  `eos_tbl` is the real-material EOS table read from
  // the .cn4 file (held on the MHD package); `derived_te`/`derived_ti` are the derived,
  // cached electron/ion temperature fields (ADR-0002 "temperatures are derived, cached
  // fields") the tabulated cons->prim closure fills.  Empty (and never touched) for the
  // ideal/isothermal EOS paths, so those stay byte-identical.
  eos_table_3t::EosTable3T eos_tbl;  // tabulated 3T EOS table (e->T inversion + lookups)
  DvceArray5D<Real> derived_te;      // derived cached electron temperature T_e(rho,e_ele)
  DvceArray5D<Real> derived_ti;      // derived cached ion temperature T_i(rho,e_ion)

  int nmhd;                // number of mhd variables (5/4 for ideal/isothermal EOS)
  int nscalars;            // number of passive scalars
  DvceArray5D<Real> u0;    // conserved variables
  DvceArray5D<Real> w0;    // primitive variables
  DvceFaceFld4D<Real> b0;  // face-centered magnetic fields
  DvceArray5D<Real> bcc0;  // cell-centered magnetic fields

  DvceArray5D<Real> coarse_u0;    // conserved variables on 2x coarser grid (for SMR/AMR)
  DvceArray5D<Real> coarse_w0;    // primitive variables on 2x coarser grid (for SMR/AMR)
  DvceFaceFld4D<Real> coarse_b0;  // face-centered B-field on 2x coarser grid

  // Objects containing boundary communication buffers and routines for u and b
  MeshBoundaryValuesCC *pbval_u;
  MeshBoundaryValuesFC *pbval_b;
  MHDBoundaryFnPtr MHDBoundaryFunc[6];

  // Orbital advection and shearing box BCs
  OrbitalAdvectionCC *porb_u = nullptr;
  OrbitalAdvectionFC *porb_b = nullptr;
  ShearingBoxCC *psbox_u = nullptr;
  ShearingBoxFC *psbox_b = nullptr;

  // Object(s) for extra physics (viscosity, resistivity, thermal conduction, srcterms)
  Viscosity *pvisc = nullptr;
  Resistivity *presist = nullptr;
  Conduction *pcond = nullptr;
  SourceTerms *psrc = nullptr;

  // Grey flux-limited radiation diffusion (FLD), advanced operator-split (RKL2 STS) in
  // the MHD once-per-step slot (#110/[A3], ADR-0001).  `erad` is the standalone radiation
  // energy density E_r the operator diffuses; `pfld_op` is non-null only when grey FLD is
  // on AND operator-split is enabled, so default runs allocate nothing and stay
  // byte-identical (mirrors hydro's operator-split conduction).
  bool fld_operator_split = false;
  DvceArray5D<Real> erad;
  FLDGreyOperator *pfld_op = nullptr;

  // Multigroup flux-limited radiation diffusion (FLD), advanced operator-split (RKL2 STS)
  // in the same MHD once-per-step slot (#111/[A4], ADR-0001/ADR-0007).  `erad_mg` is the
  // standalone per-group radiation-energy field (var extent == number of photon-energy
  // groups) the multigroup operator diffuses; `pmg_op` is non-null only when multigroup
  // FLD is enabled, so default runs allocate nothing and stay byte-identical.  The
  // per-group arrays ARE registered with the AMR regrid machinery (mesh_refinement.cpp /
  // load_balance.cpp), so they survive a re-refinement; the operator owns the coarse
  // scratch used both per-substage (SyncParabolicGhosts) and by the regrid (RefineCC).
  bool mgfld_operator_split = false;
  DvceArray5D<Real> erad_mg;
  FLDMultigroupOperator *pmg_op = nullptr;

  // Anisotropic (magnetized) Braginskii electron+ion thermal conduction, advanced
  // operator-split (RKL2 STS) in the MHD step (#112/[A5], ADR-0006/ADR-0001).
  // Unlike the FLD operators it diffuses the LIVE conserved energy (u0 IEN) field-aligned
  // along the frozen cell-centred B (bcc0), so no standalone array; `pacond_op` is
  // non-null only when explicitly enabled, so default MHD runs allocate nothing and stay
  // byte-identical.  Stiff (#109) => super-time-stepped, so it does not limit the
  // hyperbolic dt.  The operator owns its own ghost-exchange boundary object so it runs
  // multi-block/MPI/AMR via SyncParabolicGhosts (#108/[A1]).
  bool acond_operator_split = false;
  AnisotropicConductionOperator *pacond_op = nullptr;

  // Cylindrical resistive B_phi diffusion (the -eta B_phi/r^2 curl-curl operator, the
  // "one true cylindrical exception" of ADR-0004), advanced operator-split (RKL2 STS) in
  // the MHD step (#113/[A6], ADR-0004/ADR-0001).  `bphi` is a standalone single-component
  // field (B_phi) the operator diffuses with the cell-centred magnetic diffusivity
  // `eta_resb`; `presb_op` is non-null only when explicitly enabled, so default MHD runs
  // allocate nothing and stay byte-identical.  Stiff (#109) => super-time-stepped, so it
  // does not limit the hyperbolic dt.  The operator owns its own ghost-exchange boundary
  // object so it runs multi-block/MPI/AMR via SyncParabolicGhosts (#108/[A1]), with the
  // antisymmetric axis ghost kept only at the true r=0 domain face.
  bool resb_operator_split = false;
  DvceArray5D<Real> bphi;
  DvceArray4D<Real> eta_resb;
  ResistiveBphiOperator *presb_op = nullptr;

  // Couple the standalone resb `bphi` field to the live driven MHD face field `b0.x2f`
  // (#181/[P7a], ADR-0012 gap a).  Without this the `bphi` the operator diffuses is a
  // decoupled scratch field the production `maglif` pgen never fills from -- nor writes
  // back to -- the driven azimuthal `b0.x2f`, so resistivity diffuses a zero field and
  // the implosion is unchanged.  When on, the resb super-step is bracketed by a copy-in
  // (`b0.x2f` -> `bphi`) and a write-back (`bphi` -> `b0.x2f`), so resistivity diffuses
  // the *driving* B_phi (Ohmic dissipation flows to the gas via the unchanged total
  // energy on the next ConsToPrim).  Gated off by default => byte-identical; only the
  // `maglif` faithful-B1 setup enables it (`resb_eta` SI calibration is #P7d).
  bool resb_couple_b0 = false;

  // Strang-split orchestration of the coupled timestep (#115/[B2], ADR-0009).  The active
  // stiff operator-split parabolic operators (FLD radiation, anisotropic conduction,
  // resistive B_phi -- built above) are grouped into one CompositeParabolicOperator PER
  // EVOLVED FIELD (operators sharing a field sum their action into one RKL2 super-step;
  // distinct fields get distinct composites), and that composite block is Strang-wrapped
  // around the hyperbolic update as  (1/2 super-step) . hydro . (1/2 super-step): a half
  // super-step in "before_timeintegrator" and the other half in "after_timeintegrator".
  // The per-cell point-implicit matter-radiation emission/absorption coupling is wrapped
  // OUTSIDE the super-step (a half coupling step on each side), so it is NOT inflated by
  // the RKL2 substage count.  Gated on `<mhd> strang_split` (default off => individual
  // full-step operator tasks run as before, byte-identical).  `strang_comps` are owned by
  // MHD; `strang_field_ids` selects which evolved field each composite acts on.
  enum StrangField {SF_ERAD = 0, SF_ERADMG = 1, SF_U0 = 2, SF_BPHI = 3};
  bool strang_split = false;
  std::vector<parabolic::CompositeParabolicOperator*> strang_comps;
  std::vector<int> strang_field_ids;

  // Per-cell point-implicit (backward-Euler) grey matter-radiation coupling (#23 kernel),
  // wrapped outside the super-step by the Strang orchestration above.  Exchanges energy
  // between the grey radiation field `erad` and the gas internal energy (u0 IEN), driving
  // each cell toward radiative equilibrium a*T^4 = E_r while conserving E_r + e_gas
  // exactly (the "species split").  Constant code-unit coefficients (real opacity/EOS-
  // derived coefficients arrive with the IONMIX tables in Phase C); gated on
  // `<mhd> mrad_coupling` (and requires grey FLD).  Default off.
  bool mrad_coupling = false;
  Real mrad_chi_a = 0.0;     // Planck-mean absorption coefficient chi_a [1/length]
  Real mrad_cv = 1.0;        // volumetric heat capacity c_v (e_gas = c_v T)
  Real mrad_arad = 1.0;      // radiation constant a
  Real mrad_clight = 1.0;    // code-unit speed of light c
  bool mrad_eos_aware = false;  // c_v from the tabulated_3t closure (#183/[P7c]) vs const

  // following only used for time-evolving flow
  DvceArray5D<Real> u1;       // conserved variables, second register
  DvceFaceFld4D<Real> b1;     // face-centered magnetic fields, second register
  DvceFaceFld5D<Real> uflx;   // fluxes of conserved quantities on cell faces
  DvceEdgeFld4D<Real> efld;   // edge-centered electric fields (fluxes of B)
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e3x1, e2x1;
  DvceArray4D<Real> e1x2, e3x2;
  DvceArray4D<Real> e2x3, e1x3;
  Real dtnew;

  // following used for time derivatives in computation of jcon
  bool wbcc_saved = false;
  DvceArray5D<Real> wsaved;
  DvceArray5D<Real> bccsaved;

  // following used for FOFC algorithm
  DvceArray4D<bool> fofc;  // flag for each cell to indicate if FOFC is needed
  bool use_fofc = false;   // flag to enable FOFC

  // container to hold names of TaskIDs
  MHDTaskIDs id;

  // functions...
  void SetSaveWBcc();
  void AssembleMHDTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // ...in "before_timeintegrator" task list
  TaskStatus SaveMHDState(Driver *d, int stage);
  // ...in "before_timeintegrator" list (operator-split grey FLD radiation, #110/[A3])
  TaskStatus OperatorSplitFLD(Driver *d, int stage);
  // ...in "before_timeintegrator" list (operator-split multigroup FLD, #111/[A4])
  TaskStatus OperatorSplitMultigroupFLD(Driver *d, int stage);
  // ...in "before_timeintegrator" list (operator-split anisotropic conduction, #112/[A5])
  TaskStatus OperatorSplitAnisoConduction(Driver *d, int stage);
  // ...in "before_timeintegrator" list (operator-split cylindrical resistive B_phi, #113)
  TaskStatus OperatorSplitResistiveBphi(Driver *d, int stage);
  // Copy-in / write-back coupling the resb `bphi` to the driven `b0.x2f` (#181/[P7a],
  // ADR-0012 gap a).  No-ops unless `resb_couple_b0` is on (only the maglif faithful-B1
  // setup enables it); bracket every resb super-step so resistivity diffuses the live
  // driven azimuthal field rather than a decoupled scratch field.
  void CoupleResbBphiFromB0();   // b0.x2f -> bphi  (copy-in,   before the super-step)
  void CoupleResbBphiToB0();     // bphi   -> b0.x2f (write-back, after the super-step)
  // Strang orchestration (#115/[B2]): a HALF parabolic super-step over each per-field
  // composite (in BOTH "before_" and "after_timeintegrator"), and the matter-radiation
  // point-implicit coupling HALF step (outside the super-step).
  TaskStatus StrangParabolicHalf(Driver *d, int stage);
  TaskStatus MatterRadCouplingHalf(Driver *d, int stage);
  // ...in "before_stagen_tl" task list
  TaskStatus InitRecv(Driver *d, int stage);
  // ...in "stagen_tl" task list
  TaskStatus CopyCons(Driver *d, int stage);
  TaskStatus Fluxes(Driver *d, int stage);
  TaskStatus SendFlux(Driver *d, int stage);
  TaskStatus RecvFlux(Driver *d, int stage);
  TaskStatus RKUpdate(Driver *d, int stage);
  TaskStatus MHDSrcTerms(Driver *d, int stage);
  TaskStatus SendU_OA(Driver *d, int stage);
  TaskStatus RecvU_OA(Driver *d, int stage);
  TaskStatus RestrictU(Driver *d, int stage);
  TaskStatus SendU(Driver *d, int stage);
  TaskStatus RecvU(Driver *d, int stage);
  TaskStatus SendU_Shr(Driver *d, int stage);
  TaskStatus RecvU_Shr(Driver *d, int stage);
  TaskStatus CornerE(Driver *d, int stage);
  TaskStatus EFieldSrc(Driver *d, int stage);
  TaskStatus SendE(Driver *d, int stage);
  TaskStatus RecvE(Driver *d, int stage);
  TaskStatus CT(Driver *d, int stage);
  TaskStatus SendB_OA(Driver *d, int stage);
  TaskStatus RecvB_OA(Driver *d, int stage);
  TaskStatus RestrictB(Driver *d, int stage);
  TaskStatus SendB(Driver *d, int stage);
  TaskStatus RecvB(Driver *d, int stage);
  TaskStatus SendB_Shr(Driver *d, int stage);
  TaskStatus RecvB_Shr(Driver *d, int stage);
  TaskStatus ApplyPhysicalBCs(Driver* pdrive, int stage);
  TaskStatus Prolongate(Driver* pdrive, int stage);
  TaskStatus ConToPrim(Driver *d, int stage);
  TaskStatus NewTimeStep(Driver *d, int stage);
  // ...in "after_stagen_tl" task list
  TaskStatus ClearSend(Driver *d, int stage);
  TaskStatus ClearRecv(Driver *d, int stage);  // also in Driver::Initialize

  // CalculateFluxes function templated over Riemann Solvers
  template <MHD_RSolver T>
  void CalculateFluxes(Driver *d, int stage);

  // first-order flux correction
  void FOFC(Driver *d, int stage);

  DvceArray5D<Real> utest, bcctest;  // scratch arrays for FOFC

 private:
  MeshBlockPack* pmy_pack;   // ptr to MeshBlockPack containing this MHD
  // temporary variables used to store face-centered electric fields returned by RS
  DvceArray4D<Real> e1_cc, e2_cc, e3_cc;
};

} // namespace mhd
#endif // MHD_MHD_HPP_
