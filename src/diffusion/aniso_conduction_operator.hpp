#ifndef DIFFUSION_ANISO_CONDUCTION_OPERATOR_HPP_
#define DIFFUSION_ANISO_CONDUCTION_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file aniso_conduction_operator.hpp
//! \brief Anisotropic (magnetized) Braginskii electron+ion thermal conduction expressed
//! as a parabolic::ParabolicOperator, so it is advanced operator-split by the RKL2
//! super-time-stepping integrator (issue [6a]/#18, ADR-0006/ADR-0001).
//
// In MagLIF the axial premagnetization thermally insulates the hot fuel from the cold
// liner ACROSS field lines.  Isotropic conduction unphysically bleeds that heat radially.
// This operator implements the field-aligned Braginskii heat flux (ADR-0006)
//
//     q = -kappa_par bhat (bhat . grad T) - kappa_perp (grad T - bhat (bhat . grad T))
//       = -kappa_perp grad T - (kappa_par - kappa_perp) bhat (bhat . grad T),
//
// where bhat = B/|B| is the local field direction and kappa_par/kappa_perp ~ (omega_c
// tau)^2 are the parallel/perpendicular Braginskii conductivities (electron + ion) from
// the shared transport-coefficient module (braginskii_transport.hpp, issue [5]/#5).
//
// SHAPE.  Mirrors the existing ConductionOperator (#13) and FLDGreyOperator (#17): a
// face-centred diffusive flux differenced into the evolved energy component through the
// shared geometry accessors (coord_geometry.hpp FluxDivX*), so it is coordinate-agnostic.
// Splitting freezes the background (density, momentum, magnetic field), so the operator
// evolves ONLY the total energy IEN and writes 0 into every other conserved component
// (the RKL2 recursion then leaves them frozen).  The magnetic field is a frozen
// background read from a cell-centred B array; the operator does NOT evolve it.
//
// LIMITERS (the two ADR-0006 monotonicity safeguards):
//   * Sharma-Hammett (2007).  The face-normal temperature derivative is the direct
//     two-point difference; the TRANSVERSE derivatives are reconstructed at the face with
//     a van-Leer slope limiter (the harmonic mean VanLeer/VL4 below).  This is the
//     Sharma-Hammett monotonic limiter: it bounds the limited transverse slope by the
//     neighbouring one-sided slopes, so heat cannot flow from cold to hot across field
//     lines and no new temperature extrema are created (no unphysical cross-field leakage
//     / negative undershoot on the grid).
//   * Larsen flux limiter.  The FIELD-ALIGNED (parallel) flux is capped at the
//     free-streaming heat flux q_fs = v_fs * (energy density) via the SAME Larsen form
//     reused for radiation FLD (CONTEXT.md): the effective parallel conductivity is
//     kappa_par_eff = 3 kappa_par lambda(R), where
//     R = 3 kappa_par |bhat . grad T| / (v_fs T_face) and lambda(R) = (3^n + R^n)^(-1/n).
//     In the unsaturated (small-R) limit lambda->1/3 so kappa_par_eff->kappa_par
//     (Spitzer); as R->inf the parallel flux saturates at q_fs.  v_fs <= 0 disables the
//     limiter (kappa_par_eff = kappa_par exactly).
//
// UNITS.  The Braginskii coefficients are SI (n [m^-3], T [K], B [T], kappa [W/m/K]).
// The operator is intentionally decoupled from the Units machinery: the caller supplies
// linear code->SI conversion factors (dens_conv, temp_conv, bmag_conv) and an SI->code
// factor for kappa (kappa_conv) in AnisoCondParams, exactly the punit-derived factors
// the existing temperature-dependent conduction uses.  This keeps the operator
// self-contained and its unit test fully deterministic (the test chooses the factors
// and recomputes the expected coefficients from the same braginskii functions).
//
// BOUNDARY.  The RKL2 substeps run inside the integrator, so ghosts are refreshed via the
// ParabolicOperator::ApplyBoundary hook.  Two steps, in order (mirrors ConductionOperator
// #108/[A1] and FLDGreyOperator #110/[A3]): (1) a zero-gradient / insulated PHYSICAL fill
// of all ghost zones, then (2) the shared synchronous multi-block/MPI/AMR neighbour
// exchange MeshBoundaryValuesCC::SyncParabolicGhosts, which OVERWRITES ghosts on every
// internal block (and coarse/fine) face with neighbour data, leaving the insulated fill
// only at true domain boundaries.  This lets the operator span >1 MeshBlock, >1 MPI rank,
// and block-AMR while staying a single insulated box at its outer edges (#112/[A5]).
// The cell-centred B background is filled once (incl. ghosts) by the caller, NOT
// refreshed here.  At true domain boundaries OperatorAction zeroes the boundary
// face heat flux (the anisotropic parallel flux has a normal component even with a
// zero-gradient fill); internal/periodic block faces are NOT zeroed, so heat conducts
// across them.

#include "athena.hpp"
#include "driver/parabolic_operator.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "eos/eos_table_3t.hpp"

// forward declarations
class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace anisocond {

//----------------------------------------------------------------------------------------
//! \struct AnisoCondParams
//! \brief Physical + unit-conversion parameters for the anisotropic conduction operator.
//! All KOKKOS-trivially-copyable so the struct is captured BY VALUE into the device
//! kernels.  See the file comment for the unit convention.
struct AnisoCondParams {
  Real zbar = 1.0;        //!< mean ionization Z (n_e = Z n_i)
  Real mi_si = 1.67262192369e-27;  //!< ion mass [kg] (default proton)
  Real lnlam = 10.0;      //!< Coulomb log; <= 0 => compute via braginskii::CoulombLog
  Real dens_conv = 1.0;   //!< code density -> ion number density n_i [m^-3]
  Real temp_conv = 1.0;   //!< code temperature (gm1 eint/rho) -> T [K]
  Real bmag_conv = 1.0;   //!< code |B| -> B [T]
  Real kappa_conv = 1.0;  //!< SI conductivity [W/m/K] -> code conductivity
  Real nlarsen = 2.0;     //!< Larsen flux-limiter exponent n
  Real vfs = -1.0;        //!< free-streaming speed [code] for Larsen cap; <=0 disables
  bool incl_e = true;     //!< include the electron conduction contribution
  bool incl_i = true;     //!< include the ion conduction contribution
};

//----------------------------------------------------------------------------------------
//! \fn Real VanLeer
//! \brief van Leer harmonic-mean slope limiter: 2ab/(a+b) when a,b have the same sign,
//! else 0.  The Sharma-Hammett (2007) monotonic limiter building block -- the limited
//! slope never exceeds the smaller of the two one-sided slopes, so no new extrema form.
KOKKOS_INLINE_FUNCTION
Real VanLeer(Real a, Real b) {
  return (a*b > 0.0) ? (2.0*a*b/(a + b)) : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn Real VL4
//! \brief Face-centred van-Leer transverse slope (Sharma-Hammett): VanLeer of the limited
//! slopes of the two cells either side of the face.  Pass the forward/backward one-sided
//! differences of the "right" cell (fr,br) and the "left" cell (fl,bl) across the face.
KOKKOS_INLINE_FUNCTION
Real VL4(Real fr, Real br, Real fl, Real bl) {
  return VanLeer(VanLeer(fr, br), VanLeer(fl, bl));
}

//----------------------------------------------------------------------------------------
//! \fn Real LarsenLimiter
//! \brief Larsen flux limiter lambda(R) = (3^n + R^n)^(-1/n) (the same smooth form reused
//! for radiation FLD, CONTEXT.md).  lambda(0)=1/3, lambda(R->inf)->1/R.  KOKKOS_INLINE so
//! it is host- and device-callable and shared by the operator and its unit test.
KOKKOS_INLINE_FUNCTION
Real LarsenLimiter(Real R, Real n) {
  Real Rc = (R > 0.0) ? R : 0.0;
  Real three_n = Kokkos::pow(static_cast<Real>(3.0), n);
  Real Rn = Kokkos::pow(Rc, n);
  return Kokkos::pow(three_n + Rn, -1.0/n);
}

//----------------------------------------------------------------------------------------
//! \fn void Conductivities
//! \brief Parallel and perpendicular Braginskii thermal conductivities (electron + ion)
//! in CODE units for a cell/face with code density `rho`, code temperature `tcode`
//! (= (gamma-1) eint/rho), and code field magnitude `bmag`.  Converts to SI via the
//! AnisoCondParams factors, sums the requested species' SIM-61 coefficients, and
//! converts back to code units.  KOKKOS_INLINE_FUNCTION (host+device) so the unit test
//! can recompute the expected values from the same code path.
KOKKOS_INLINE_FUNCTION
void Conductivities(const AnisoCondParams &p, Real rho, Real tcode, Real bmag,
                    Real &kpar, Real &kperp) {
  Real n_i = p.dens_conv*rho;             // ion number density [m^-3]
  Real n_e = p.zbar*n_i;                  // electron number density [m^-3]
  Real T   = p.temp_conv*tcode;           // temperature [K] (single-temperature plasma)
  Real B   = p.bmag_conv*bmag;            // field magnitude [T]
  Real lnL = (p.lnlam > 0.0) ? p.lnlam : braginskii::CoulombLog(n_e, T, p.zbar);

  Real kp = 0.0, kt = 0.0;                // SI [W/m/K]
  if (p.incl_e) {
    Real tau_e = braginskii::ElectronCollisionTime(n_e, T, p.zbar, lnL);
    Real x_e   = braginskii::ElectronGyroFrequency(B)*tau_e;
    kp += braginskii::ConductivityParElectron(n_e, T, tau_e);
    kt += braginskii::ConductivityPerpElectron(n_e, T, tau_e, x_e);
  }
  if (p.incl_i) {
    Real tau_i = braginskii::IonCollisionTime(n_i, T, p.zbar, p.mi_si, lnL);
    Real x_i   = braginskii::IonGyroFrequency(B, p.zbar, p.mi_si)*tau_i;
    kp += braginskii::ConductivityParIon(n_i, T, tau_i, p.mi_si);
    kt += braginskii::ConductivityPerpIon(n_i, T, tau_i, p.mi_si, x_i);
  }
  kpar  = p.kappa_conv*kp;                // -> code units
  kperp = p.kappa_conv*kt;
}

//----------------------------------------------------------------------------------------
//! \struct EosAwareTemp
//! \brief Per-cell gas-temperature recovery for the conduction operator, EOS-aware
//!  (#183/[P7c], ADR-0012 gap c).  The flux kernels need a temperature T(rho, e) to build
//!  the conducted heat-flux gradient.  By default (`eos_aware=false`) this is the
//!  ideal-gamma bookkeeping value `T = (gamma-1) eint/rho` -- the original `TempMHD`
//!  closure, byte-identical for every existing run.  On the faithful tabulated_3t path
//!  (`eos_aware=true`) the temperature is instead inverted from the tabulated closure:
//!  the electron internal energy density rides the first passive scalar (`e_ele` at index
//!  `eele_idx`, ADR-0002), and the conducted temperature is the *electron* temperature
//!  `T_e = table.Te(rho, e_ele/rho)` -- the same `e->T` inversion ConsToPrim2T caches
//!  as the derived field, so conduction is consistent with the faithful EOS
//!  rather than the ideal-gamma relation.  Electron conduction is the dominant Braginskii
//!  channel, so `T_e` is the relevant conducted temperature (the operator keeps
//!  its single-temperature flux structure).  Trivially copyable (the table is a shallow
//!  View handle), so it is captured BY VALUE into the kernels like AnisoCondParams.
struct EosAwareTemp {
  bool eos_aware = false;             //!< false => ideal-gamma TempMHD (byte-identical)
  int eele_idx = 0;                   //!< index of the e_ele passive scalar in `cons`
  Real gm1 = 0.0;                     //!< gamma-1 for the ideal-gamma closure
  eos_table_3t::EosTable3T table;     //!< tabulated 3T EOS (only read when eos_aware)

  //! \brief Recover the conducted gas temperature for cell (m,k,j,i) from the conserved
  //!  state `u` and cell-centred field `bcc`.  Both closures subtract the kinetic and
  //!  (cell-centred) magnetic energy from the total energy to form the gas internal
  //!  energy, exactly as the original TempMHD did.
  KOKKOS_INLINE_FUNCTION
  Real Temp(const DvceArray5D<Real> &u, const DvceArray5D<Real> &bcc,
            int m, int k, int j, int i) const {
    Real rho = u(m,IDN,k,j,i);
    Real ke = 0.5*(SQR(u(m,IM1,k,j,i)) + SQR(u(m,IM2,k,j,i)) + SQR(u(m,IM3,k,j,i)))/rho;
    Real me = 0.5*(SQR(bcc(m,IBX,k,j,i)) + SQR(bcc(m,IBY,k,j,i)) + SQR(bcc(m,IBZ,k,j,i)));
    Real eint = u(m,IEN,k,j,i) - ke - me;
    if (!eos_aware) {
      return gm1*eint/rho;             // original ideal-gamma bookkeeping temperature
    }
    // faithful tabulated_3t closure: invert the electron energy scalar to T_e
    Real e_ele = u(m,eele_idx,k,j,i);
    return table.Te(rho, e_ele/rho);
  }
};

}  // namespace anisocond

//----------------------------------------------------------------------------------------
//! \class AnisotropicConductionOperator
//! \brief Anisotropic Braginskii electron+ion thermal conduction (field-aligned heat flux
//! with Sharma-Hammett transverse limiting and a Larsen field-aligned flux cap) as a
//! ParabolicOperator.

class AnisotropicConductionOperator : public parabolic::ParabolicOperator {
 public:
  //! \brief Build the operator over the conserved field `cons` it diffuses (rho/mom/E;
  //! also read for the explicit-dt density) and the frozen cell-centred magnetic field
  //! `bcc` (3 components IBX/IBY/IBZ) it aligns the flux to, with adiabatic index `gamma`
  //! and the physical/unit parameters `par`.  `pin` is used to build the operator's own
  //! cell-centred boundary-values object (unique MPI communicator) for the per-substage
  //! multi-block/MPI ghost exchange (#112); pass nullptr for a single-block-only use.
  AnisotropicConductionOperator(MeshBlockPack *pp, ParameterInput *pin,
                                const DvceArray5D<Real> &cons,
                                const DvceArray5D<Real> &bcc, Real gamma,
                                const anisocond::AnisoCondParams &par,
                                const anisocond::EosAwareTemp &etemp =
                                    anisocond::EosAwareTemp());
  ~AnisotropicConductionOperator();

  //! \brief M(u): anisotropic heat-flux divergence into rhs_out(IEN); 0 in all other
  //! conserved components (frozen background).  Reads ghost zones of u_in and bcc.
  void OperatorAction(const DvceArray5D<Real> &u_in,
                      DvceArray5D<Real> &rhs_out) override;

  //! \brief Forward-Euler stability dt for the (parallel) anisotropic conduction of the
  //! current state (kappa_par is the most restrictive coefficient).
  Real ExplicitStableDt() override;

  //! \brief Zero-gradient (insulated) ghost-zone refresh for the operator-split substeps.
  void ApplyBoundary(DvceArray5D<Real> &u) override;

 private:
  MeshBlockPack *pmy_pack;
  DvceArray5D<Real> cons_;       // the live conserved field (for the explicit-dt density)
  DvceArray5D<Real> bcc_;        // frozen cell-centred magnetic field (3 components)
  Real gamma_;                   // adiabatic index, for T = (gamma-1) eint/rho
  anisocond::AnisoCondParams par_;  // physical + unit-conversion parameters
  anisocond::EosAwareTemp etemp_;   // EOS-aware temperature recovery (#183/[P7c])
  DvceFaceFld5D<Real> hflx_;     // scratch face-centred heat flux
  // per-substage multi-block/MPI/AMR neighbour ghost exchange for the diffused field
  // (#112/[A5]); owns a unique MPI communicator.  nullptr only if built with pin==nullptr
  // (legacy single-block-only construction) -> SyncParabolicGhosts skipped.
  MeshBoundaryValuesCC *pbval_;
  // coarse-mesh scratch for the restriction/prolongation SyncParabolicGhosts does at
  // fine/coarse boundaries under SMR/AMR (1^5 and untouched on a uniform grid).
  DvceArray5D<Real> coarse_;
  // conservative fine->coarse flux correction at AMR/SMR level boundaries (#33); built
  // only on a multilevel mesh, nullptr otherwise -> no-op on a uniform grid.
  MeshBoundaryValuesCC *pbval_flux_;
};

#endif  // DIFFUSION_ANISO_CONDUCTION_OPERATOR_HPP_
