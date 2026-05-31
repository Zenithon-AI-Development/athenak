//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file tabulated_mhd.cpp
//! \brief derived class implementing the tabulated 3T (IONMIX) real-material EOS in
//!  nonrelativistic MHD (issue [P1]/#159, ADR-0002/0007).
//!
//! This is the live-solver wiring of the tabulated 3T machinery that previously existed
//! only in unit tests: the common EOS representation (eos/eos_table_3t.hpp), the per-cell
//! cons->prim closure (eos/cons_to_prim_2t.hpp) and the 3T energy reconciliation
//! (eos/three_temperature.hpp).  The real-material `.cn4` table is read into the
//! `EosTable3T` held on the MHD package (mhd.cpp selector); this file implements the
//! per-step cons<->prim conversions on top of it.
//!
//! Conserved/primitive layout (ADR-0002):
//!   - The conservative total-energy MHD update is UNCHANGED (so shock jumps and global
//!     energy conservation are guaranteed).
//!   - The electron internal energy density `e_ele` rides the FIRST passive scalar as the
//!     specific quantity `e_ele/rho` (conserved scalar value == `e_ele`); the ion
//!     internal energy is recovered by subtraction
//!     `e_ion = E_tot - 1/2 rho v^2 - 1/2 B^2 - e_ele` (three_temp::IonInternalEnergyMHD)
//!   - ConsToPrim inverts `(rho, e_ele, e_ion)` to the derived/cached temperatures
//!     T_e/T_i and the species/total pressures via eos_table_3t::ConsToPrim2T, storing
//!     T_e/T_i in the package's `derived_te`/`derived_ti` fields.  The primitive energy
//!     is the internal energy density `e_ele + e_ion` (the `use_e` convention), so
//!     PrimToCons re-assembles `E_tot` EOS-independently and the round-trip is exact.
//!
//! Routing the live hyperbolic pressure/energy update through the tabulated inversion
//! (rather than the ideal-gamma closure the reconstruction/Riemann path still uses for
//! wave speeds) is the follow-on slice [P2]/#162; this slice makes `eos=tabulated_3t` a
//! real, selectable EOS whose cons->prim produces table-consistent temperatures.

#include <float.h>   // FLT_MAX

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/cons_to_prim_2t.hpp"
#include "eos/three_temperature.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

TabulatedMHD::TabulatedMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = false;
  // gamma is bookkeeping for the ideal-gamma wave-speed estimate the hyperbolic solver
  // still uses (the tabulated pressure feedback is slice [P2]/#162); it does not enter
  // the tabulated cons->prim closure below.
  eos_data.gamma = pin->GetOrAddReal("mhd","gamma",(5.0/3.0));
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;   // evolve internal energy density
  eos_data.use_t = false;
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
}

//----------------------------------------------------------------------------------------
//! \!fn void ConsToPrim()
//! \brief Tabulated 3T conserved->primitive closure.  Recovers the ion internal energy by
//! subtraction, inverts (rho, e_ele, e_ion) to T_e/T_i via the tabulated EOS, and stores
//! the primitive state (internal energy density) plus the derived/cached temperatures.

void TabulatedMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                              DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                              const bool only_testfloors,
                              const int il, const int iu, const int jl, const int ju,
                              const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &eos = eos_data;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  // tabulated EOS table + derived cached temperature fields held on the MHD package
  // (shallow View copies captured by value into the kernel).
  eos_table_3t::EosTable3T tbl = pmy_pack->pmhd->eos_tbl;
  DvceArray5D<Real> derived_te = pmy_pack->pmhd->derived_te;
  DvceArray5D<Real> derived_ti = pmy_pack->pmhd->derived_ti;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0;
  Kokkos::parallel_reduce("mhd_c2p_tab",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load conserved variables
    Real u_d  = cons(m,IDN,k,j,i);
    Real u_m1 = cons(m,IM1,k,j,i);
    Real u_m2 = cons(m,IM2,k,j,i);
    Real u_m3 = cons(m,IM3,k,j,i);
    Real u_e  = cons(m,IEN,k,j,i);

    // cell-centered magnetic field: input CC fields when only testing floors with FOFC,
    // else simple linear average of the face-centered fields.
    Real bx, by, bz;
    if (only_testfloors) {
      bx = bcc(m,IBX,k,j,i);
      by = bcc(m,IBY,k,j,i);
      bz = bcc(m,IBZ,k,j,i);
    } else {
      bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // density floor
    bool dfloor_used = false;
    Real rho = u_d;
    if (rho < eos.dfloor) { rho = eos.dfloor; dfloor_used = true; }

    // electron internal energy density rides the first passive scalar (conserved value);
    // ion internal energy by subtraction (three-temperature formulation, ADR-0002).
    Real e_ele = cons(m,nmhd,k,j,i);
    Real e_ion = three_temp::IonInternalEnergyMHD(u_e, rho, u_m1, u_m2, u_m3,
                                                  bx, by, bz, e_ele);

    // tabulated 2T cons->prim closure: derived T_e/T_i + species/total pressures.
    eos_table_3t::GasState2T s = eos_table_3t::ConsToPrim2T(tbl, rho, e_ele, e_ion);

    if (only_testfloors) {
      if (dfloor_used) {
        fofc_(m,k,j,i) = true;
        sumd++;
      }
    } else {
      // reset conserved density if floor was hit
      if (dfloor_used) {
        cons(m,IDN,k,j,i) = rho;
        sumd++;
      }
      // store primitive state: internal energy density e_ele + e_ion (use_e convention)
      prim(m,IDN,k,j,i) = rho;
      prim(m,IVX,k,j,i) = u_m1/rho;
      prim(m,IVY,k,j,i) = u_m2/rho;
      prim(m,IVZ,k,j,i) = u_m3/rho;
      prim(m,IEN,k,j,i) = e_ele + e_ion;
      // store cell-centered fields
      bcc(m,IBX,k,j,i) = bx;
      bcc(m,IBY,k,j,i) = by;
      bcc(m,IBZ,k,j,i) = bz;
      // derived, cached temperatures (ADR-0002)
      derived_te(m,0,k,j,i) = s.te;
      derived_ti(m,0,k,j,i) = s.ti;
      // convert scalars (incl. e_ele/rho), always stored at end of cons and prim arrays.
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        if (cons(m,n,k,j,i) < 0.0) {
          cons(m,n,k,j,i) = 0.0;
        }
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/rho;
      }
    }
  }, Kokkos::Sum<int>(nfloord_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \!fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  EOS-independent re-assembly of
//! the conserved total energy from the primitive internal energy density (use_e), so the
//! cons->prim->cons round-trip is exact.  Does not change cell- or face-centered fields.

void TabulatedMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                              DvceArray5D<Real> &cons, const int il, const int iu,
                              const int jl, const int ju, const int kl, const int ku) {
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;

  par_for("mhd_p2c_tab", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real wd  = prim(m,IDN,k,j,i);
    Real vx  = prim(m,IVX,k,j,i);
    Real vy  = prim(m,IVY,k,j,i);
    Real vz  = prim(m,IVZ,k,j,i);
    Real we  = prim(m,IEN,k,j,i);   // internal energy density
    Real bx  = bcc(m,IBX,k,j,i);
    Real by  = bcc(m,IBY,k,j,i);
    Real bz  = bcc(m,IBZ,k,j,i);

    // store conserved state: total energy = internal + kinetic + magnetic
    cons(m,IDN,k,j,i) = wd;
    cons(m,IM1,k,j,i) = wd*vx;
    cons(m,IM2,k,j,i) = wd*vy;
    cons(m,IM3,k,j,i) = wd*vz;
    cons(m,IEN,k,j,i) = we + 0.5*(wd*(SQR(vx) + SQR(vy) + SQR(vz))
                                  + (SQR(bx) + SQR(by) + SQR(bz)));

    // convert scalars (if any), always stored at end of cons and prim arrays.
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = wd*prim(m,n,k,j,i);
    }
  });

  return;
}
