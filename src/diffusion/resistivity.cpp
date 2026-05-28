//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.cpp
//  \brief Implements functions for Resistivity class.

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "eos/eos.hpp"
#include "resistivity.hpp"
#include "current_density.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls Resistivity base class constructor

Resistivity::Resistivity(MeshBlockPack *pp, ParameterInput *pin) :
  pmy_pack(pp) {
  // Resistivity model: "constant" (legacy, scalar eta_ohm) or Spitzer/Braginskii
  // variable eta(rho,T_e) from the SIM-61 transport module (ADR-0003).
  std::string model = pin->GetOrAddString("mhd","resistivity","constant");

  if (model == "constant") {
    // Legacy constant Ohmic diffusivity -- byte-identical to the original path.
    eta_ohm = pin->GetReal("mhd","ohmic_resistivity");
    variable_eta = false;

    // resistive timestep on MeshBlock(s) in this pack
    dtnew = std::numeric_limits<float>::max();
    auto size = pmy_pack->pmb->mb_size;
    Real fac;
    if (pp->pmesh->three_d) {
      fac = 1.0/6.0;
    } else if (pp->pmesh->two_d) {
      fac = 0.25;
    } else {
      fac = 0.5;
    }
    for (int m=0; m<(pp->nmb_thispack); ++m) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/eta_ohm);
      if (pp->pmesh->multi_d) {dtnew=std::min(dtnew,fac*SQR(size.h_view(m).dx2)/eta_ohm);}
      if (pp->pmesh->three_d) {dtnew=std::min(dtnew,fac*SQR(size.h_view(m).dx3)/eta_ohm);}
    }
  } else if (model == "spitzer" || model == "braginskii") {
    // Variable Spitzer/Braginskii eta(rho,T_e), eta_par/eta_perp from braginskii module.
    variable_eta = true;
    eta_ohm = 0.0;  // unused; the operator gates on variable_eta / IsActive()
    rparams.dens_si     = pin->GetOrAddReal("mhd","resist_dens_si",1.0);
    rparams.temp_si     = pin->GetOrAddReal("mhd","resist_temp_si",1.0);
    rparams.bmag_si     = pin->GetOrAddReal("mhd","resist_bmag_si",1.0);
    rparams.eta_si      = pin->GetOrAddReal("mhd","resist_eta_si",1.0);
    rparams.zbar        = pin->GetOrAddReal("mhd","resist_zbar",1.0);
    rparams.m_i         = pin->GetOrAddReal("mhd","resist_m_i",braginskii::kProtonMass);
    rparams.ln_lambda   = pin->GetOrAddReal("mhd","resist_lnlambda",0.0);
    rparams.anisotropic = pin->GetOrAddBoolean("mhd","resist_anisotropic",false);
    rparams.rho_floor   = pin->GetOrAddReal("mhd","vacuum_density",-1.0);
    rparams.eta_vac     = pin->GetOrAddReal("mhd","vacuum_resistivity",0.0);

    // allocate the cell-centered eta field over the full cell extent (incl. ghosts);
    // it is (re)filled from the current MHD state by ComputeEta() each substep.
    auto &indcs = pp->pmesh->mb_indcs;
    int nc1 = indcs.nx1 + 2*indcs.ng;
    int nc2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*indcs.ng) : 1;
    int nc3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*indcs.ng) : 1;
    Kokkos::realloc(eta, pp->nmb_thispack, nc3, nc2, nc1);

    // dt is set from max(eta) by NewTimeStep() (called in Driver::Initialize); start big
    dtnew = std::numeric_limits<float>::max();
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Unknown <mhd> resistivity = '" << model << "' (expected 'constant', "
              << "'spitzer', or 'braginskii')" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// Resistivity destructor

Resistivity::~Resistivity() {
}

//----------------------------------------------------------------------------------------
//! \fn void Resistivity::ComputeEta()
//! \brief Fill the cell-centered magnetic-diffusivity field eta(m,k,j,i) (code units)
//!  from the current MHD primitive state (rho, T_e = (gamma-1) e/rho) and cell-centered
//!  |B|, using the Spitzer/Braginskii model + vacuum-resistivity floor.  No-op in the
//!  constant-eta path.  Covers the full cell extent (incl. ghosts) so the OhmicEField /
//!  OhmicEnergyFlux edge/face averages are valid up to the domain boundary.

void Resistivity::ComputeEta() {
  if (!variable_eta) {return;}
  EOS_Data &eos = pmy_pack->pmhd->peos->eos_data;
  if (!eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Variable Spitzer/Braginskii resistivity requires an ideal-gas EOS "
              << "(temperature from internal energy)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real gm1 = eos.gamma - 1.0;
  auto rp = rparams;
  auto eta_ = eta;
  auto w0_ = pmy_pack->pmhd->w0;
  auto bcc_ = pmy_pack->pmhd->bcc0;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int nc1 = eta.extent_int(3);
  int nc2 = eta.extent_int(2);
  int nc3 = eta.extent_int(1);

  par_for("resist_eta", DevExeSpace(), 0, nmb1, 0, nc3-1, 0, nc2-1, 0, nc1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real rho  = w0_(m,IDN,k,j,i);
    Real temp = gm1*w0_(m,IEN,k,j,i)/rho;
    Real bx = bcc_(m,IBX,k,j,i);
    Real by = bcc_(m,IBY,k,j,i);
    Real bz = bcc_(m,IBZ,k,j,i);
    Real bmag = sqrt(bx*bx + by*by + bz*bz);
    eta_(m,k,j,i) = resistivity::EffectiveDiffusivity(rp, rho, temp, bmag);
  });
}

//----------------------------------------------------------------------------------------
//! \fn void Resistivity::NewTimeStep()
//! \brief Recompute the explicit resistive-diffusion timestep from the current max(eta)
//!  (variable-eta path).  No-op in the constant path (dtnew set once in the ctor).

void Resistivity::NewTimeStep() {
  if (!variable_eta) {return;}
  ComputeEta();

  // global max of eta over the interior cells of this pack
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  auto eta_ = eta;
  const int nmkji = (pmy_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  Real max_eta = 0.0;
  Kokkos::parallel_reduce("resist_dt", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &mx) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    mx = fmax(mx, eta_(m,k,j,i));
  }, Kokkos::Max<Real>(max_eta));

  dtnew = std::numeric_limits<float>::max();
  if (max_eta <= 0.0) {return;}
  auto size = pmy_pack->pmb->mb_size;
  Real fac;
  if (pmy_pack->pmesh->three_d) {
    fac = 1.0/6.0;
  } else if (pmy_pack->pmesh->two_d) {
    fac = 0.25;
  } else {
    fac = 0.5;
  }
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx1)/max_eta);
    if (pmy_pack->pmesh->multi_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx2)/max_eta);
    }
    if (pmy_pack->pmesh->three_d) {
      dtnew = std::min(dtnew, fac*SQR(size.h_view(m).dx3)/max_eta);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn OhmicEField()
//  \brief Adds electric field from Ohmic resistivity to corner-centered electric field
//  Using Ohm's Law to compute the electric field:  E + (v x B) = \eta J, then
//    E_{inductive} = - (v x B)  [computed in the MHD Riemann solver]
//    E_{resistive} = \eta J     [computed in this function]

void Resistivity::OhmicEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // refresh the cell-centered eta(rho,T_e) field for the variable-eta path
  if (variable_eta) {ComputeEta();}
  auto var = variable_eta;
  auto eta_fld = eta;

  //---- 1-D problem:
  //  copy face-centered E-fields to edges and return.
  //  Note e2[is:ie+1,js:je,  ks:ke+1]
  //       e3[is:ie+1,js:je+1,ks:ke  ]

  if (pmy_pack->pmesh->one_d) {
    // capture class variables for the kernels
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm1", DevExeSpace(), scr_size, scr_level, 0, nmb1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, js, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real eta_i = var ? 0.5*(eta_fld(m,ks,js,i-1) + eta_fld(m,ks,js,i)) : eta_o;
        e2(m,ks,  js  ,i) += eta_i*j2(i);
        e2(m,ke+1,js  ,i) += eta_i*j2(i);
        e3(m,ks  ,js  ,i) += eta_i*j3(i);
        e3(m,ks  ,je+1,i) += eta_i*j3(i);
      });
    });
    return;
  }

  //---- 2-D problem:
  if (pmy_pack->pmesh->two_d) {
    // capture class variables for the kernels
    auto e1 = efld.x1e;
    auto e2 = efld.x2e;
    auto e3 = efld.x3e;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto eta_o = eta_ohm;

    int scr_level = 0;
    size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

    par_for_outer("ohm2", DevExeSpace(), scr_size, scr_level, 0, nmb1, js, je+1,
    KOKKOS_LAMBDA(TeamMember_t member, const int m, const int j) {
      ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
      ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

      CurrentDensity(member, m, ks, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

      // Add E_{resistive} = \eta J to corner-centered electric fields
      par_for_inner(member, is, ie+1, [&](const int i) {
        Real eta_e1 = var ? 0.5*(eta_fld(m,ks,j-1,i) + eta_fld(m,ks,j,i)) : eta_o;
        Real eta_e2 = var ? 0.5*(eta_fld(m,ks,j,i-1) + eta_fld(m,ks,j,i)) : eta_o;
        Real eta_e3 = var ? 0.25*(eta_fld(m,ks,j-1,i-1) + eta_fld(m,ks,j-1,i) +
                                  eta_fld(m,ks,j  ,i-1) + eta_fld(m,ks,j  ,i)) : eta_o;
        e1(m,ks,  j,i) += eta_e1*j1(i);
        e1(m,ke+1,j,i) += eta_e1*j1(i);
        e2(m,ks,  j,i) += eta_e2*j2(i);
        e2(m,ke+1,j,i) += eta_e2*j2(i);
        e3(m,ks  ,j,i) += eta_e3*j3(i);
      });
    });
    return;
  }

  //---- 3-D problem:

  // capture class variables for the kernels
  auto e1 = efld.x1e;
  auto e2 = efld.x2e;
  auto e3 = efld.x3e;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto eta_o = eta_ohm;

  int scr_level = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells1) * 3;

  par_for_outer("ohm3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> j1(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j2(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> j3(member.team_scratch(scr_level), ncells1);

    CurrentDensity(member, m, k, j, is, ie+1, b0, mbsize.d_view(m), j1, j2, j3);

    // Add E_{resistive} = \eta J to corner-centered electric fields
    par_for_inner(member, is, ie+1, [&](const int i) {
      Real eta_e1 = var ? 0.25*(eta_fld(m,k-1,j-1,i) + eta_fld(m,k-1,j,i) +
                                eta_fld(m,k  ,j-1,i) + eta_fld(m,k  ,j,i)) : eta_o;
      Real eta_e2 = var ? 0.25*(eta_fld(m,k-1,j,i-1) + eta_fld(m,k-1,j,i) +
                                eta_fld(m,k  ,j,i-1) + eta_fld(m,k  ,j,i)) : eta_o;
      Real eta_e3 = var ? 0.25*(eta_fld(m,k,j-1,i-1) + eta_fld(m,k,j-1,i) +
                                eta_fld(m,k,j  ,i-1) + eta_fld(m,k,j  ,i)) : eta_o;
      e1(m,k,j,i) += eta_e1*j1(i);
      e2(m,k,j,i) += eta_e2*j2(i);
      e3(m,k,j,i) += eta_e3*j3(i);
    });
  });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn OhmicEnergyFlux()
//  \brief Adds Poynting flux from Ohmic resistivity to energy flux
//  Total energy equation is dE/dt = - Div(F) where F = (E X B) = \eta (J X B)


void Resistivity::OhmicEnergyFlux(const DvceFaceFld4D<Real> &b,
                                  DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  Real qa = 0.25*eta_ohm;  // constant-eta prefactor (byte-identical legacy path)

  // refresh the cell-centered eta(rho,T_e) field for the variable-eta path; the Poynting
  // flux must use the same eta as the resistive EMF for energy consistency (ADR-0003)
  if (variable_eta) {ComputeEta();}
  auto var = variable_eta;
  auto eta_fld = eta;

  //------------------------------
  // energy fluxes in x1-direction

  auto &flx1 = flx.x1f;
  par_for("ohm_heat1", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie+1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j2k   = -(b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1;
    Real j2kp1 = -(b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j,i-1))/size.d_view(m).dx1;

    Real j3j   = (b.x2f(m,k,j  ,i) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1;
    Real j3jp1 = (b.x2f(m,k,j+1,i) - b.x2f(m,k,j+1,i-1))/size.d_view(m).dx1;

    if (multi_d) {
      j3j   -= (b.x1f(m,k,j  ,i) - b.x1f(m,k,j-1,i))/size.d_view(m).dx2;
      j3jp1 -= (b.x1f(m,k,j+1,i) - b.x1f(m,k,j  ,i))/size.d_view(m).dx2;
    }
    if (three_d) {
      j2k   += (b.x1f(m,k  ,j,i) - b.x1f(m,k-1,j,i))/size.d_view(m).dx3;
      j2kp1 += (b.x1f(m,k+1,j,i) - b.x1f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // flx1 = (E X B)_{1} =  ((\eta J) X B)_{1} = \eta (J2*B3 - J3*B2)
    Real qf = var ? 0.125*(eta_fld(m,k,j,i-1) + eta_fld(m,k,j,i)) : qa;
    flx1(m,IEN,k,j,i) += qf*(j2k  *(b.x3f(m,k  ,j  ,i) + b.x3f(m,k  ,j  ,i-1)) +
                             j2kp1*(b.x3f(m,k+1,j  ,i) + b.x3f(m,k+1,j  ,i-1)) -
                             j3j  *(b.x2f(m,k  ,j  ,i) + b.x2f(m,k  ,j  ,i-1)) -
                             j3jp1*(b.x2f(m,k  ,j+1,i) + b.x2f(m,k  ,j+1,i-1)));
  });
  if (pmy_pack->pmesh->one_d) {return;}

  //------------------------------
  // energy fluxes in x2-direction

  auto &flx2 = flx.x2f;
  par_for("ohm_heat2", DevExeSpace(), 0, nmb1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1k   = (b.x3f(m,k  ,j,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2;
    Real j1kp1 = (b.x3f(m,k+1,j,i) - b.x3f(m,k+1,j-1,i))/size.d_view(m).dx2;

    Real j3i   = (b.x2f(m,k,j,i  ) - b.x2f(m,k,j  ,i-1))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i  ) - b.x1f(m,k,j-1,i  ))/size.d_view(m).dx2;
    Real j3ip1 = (b.x2f(m,k,j,i+1) - b.x2f(m,k,j  ,i  ))/size.d_view(m).dx1
               - (b.x1f(m,k,j,i+1) - b.x1f(m,k,j-1,i+1))/size.d_view(m).dx2;

    if (three_d) {
      j1k   -= (b.x2f(m,k  ,j,i) - b.x2f(m,k-1,j,i))/size.d_view(m).dx3;
      j1kp1 -= (b.x2f(m,k+1,j,i) - b.x2f(m,k  ,j,i))/size.d_view(m).dx3;
    }

    // E2 = \eta (J X B)_{2} = \eta (J3*B1 - J1*B3)
    Real qf = var ? 0.125*(eta_fld(m,k,j-1,i) + eta_fld(m,k,j,i)) : qa;
    flx2(m,IEN,k,j,i) += qf*(j3i  *(b.x1f(m,k  ,j,i  ) + b.x1f(m,k  ,j-1,i  )) +
                             j3ip1*(b.x1f(m,k  ,j,i+1) + b.x1f(m,k  ,j-1,i+1)) -
                             j1k  *(b.x3f(m,k  ,j,i  ) + b.x3f(m,k  ,j-1,i  )) -
                             j1kp1*(b.x3f(m,k+1,j,i  ) + b.x3f(m,k+1,j-1,i  )));
  });
  if (pmy_pack->pmesh->two_d) {return;}

  //------------------------------
  // energy fluxes in x3-direction

  auto &flx3 = flx.x3f;
  par_for("ohm_heat3", DevExeSpace(), 0, nmb1, ks, ke+1, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real j1j   = (b.x3f(m,k,j  ,i) - b.x3f(m,k  ,j-1,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j  ,i) - b.x2f(m,k-1,j  ,i))/size.d_view(m).dx3;
    Real j1jp1 = (b.x3f(m,k,j+1,i) - b.x3f(m,k  ,j  ,i))/size.d_view(m).dx2
               - (b.x2f(m,k,j+1,i) - b.x2f(m,k-1,j+1,i))/size.d_view(m).dx3;

    Real j2i   = -(b.x3f(m,k,j,i  ) - b.x3f(m,k  ,j,i-1))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i  ) - b.x1f(m,k-1,j,i  ))/size.d_view(m).dx3;
    Real j2ip1 = -(b.x3f(m,k,j,i+1) - b.x3f(m,k  ,j,i  ))/size.d_view(m).dx1
                + (b.x1f(m,k,j,i+1) - b.x1f(m,k-1,j,i+1))/size.d_view(m).dx3;

    // E3 = \eta (J X B)_{3} = \eta (J1*B2 - J2*B1)
    Real qf = var ? 0.125*(eta_fld(m,k-1,j,i) + eta_fld(m,k,j,i)) : qa;
    flx3(m,IEN,k,j,i) += qf*(j1j  *(b.x2f(m,k,j  ,i  ) + b.x2f(m,k-1,j  ,i  )) +
                             j1jp1*(b.x2f(m,k,j+1,i  ) + b.x2f(m,k-1,j+1,i  )) -
                             j2i  *(b.x1f(m,k,j  ,i  ) + b.x1f(m,k-1,j  ,i  )) -
                             j2ip1*(b.x1f(m,k,j  ,i+1) + b.x1f(m,k-1,j  ,i+1)));
  });

  return;
}
