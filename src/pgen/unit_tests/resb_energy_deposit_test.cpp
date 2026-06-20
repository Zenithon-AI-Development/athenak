//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resb_energy_deposit_test.cpp
//  \brief Unit test: MHD::CoupleResbBphiToB0() conserves TOTAL energy when the Ohmic-heat
//  deposit is on, depositing the dissipated field energy on the ELECTRON internal-energy
//  scalar (leaving the ions unchanged) -- issue #193.
//
//  WHY.  The #192 fix made the bphi->b0.x2f representation swap energy-consistent: it
//  bills 0.5*(b_new^2-b_old^2) to u0(IEN) so e_int is invariant under the swap.  But it
//  DROPS the resistively-dissipated field energy (the Ohmic eta*J^2 heat) instead of
//  depositing it.  So on current main TOTAL energy DECREASES by exactly the field energy
//  the resb diffusion removed -- a one-signed, stabilising omission.  #193 adds a gated
//  deposit (<mhd> resb_deposit_heat, default false => #192 byte-identical) that adds it
//  back, to the electrons, closing the budget: field energy lost == electron heat gained,
//  E_tot invariant, ions unchanged.
//
//  CONSERVATION FORMULATION.  The deposit equals the field energy the SAME resb stencil
//  removed -- 0.5*(b_old^2 - b_new^2) per cell, b_old the pre-write phi-mean face (the
//  seed the operator diffused), b_new the post-diffusion bphi.  That is the operator's
//  own magnetic-energy decrease, so it matches the field-energy drop by construction (no
//  centering mismatch can leak energy) and never exceeds the dissipation (no #192 pump).
//
//  WHAT THIS DOES (no time integration; all in UserProblem):
//   - seed a div-free axisymmetric b0 (B_r=B_z=0, B_phi(r) phi-uniform), uniform rho,
//     known E_tot, known e_ele (scalar 0 = u0(nmhd)), zero velocity;
//   - populate resb `bphi` with the same phi-mean seed shrunk by a uniform decay factor
//     (a clean one-step resb stand-in that removes a known, positive field energy);
//   - snapshot E_tot, e_ele, e_ion (=E_tot-KE-B^2/2-e_ele) and the field-energy decrease;
//   - call the REAL CoupleResbBphiToB0() and, branching on the active resb_deposit_heat:
//       deposit OFF (#192): E_tot drops by exactly dE_field; e_ele fixed; e_ion fixed.
//       deposit ON  (#193): E_tot conserved to round-off; e_ele += dE_field; e_ion fixed.
//
//  Built/run by tst/test_suite/unit_tests/test_unit_resb_energy_deposit_cpu.py (runs the
//  binary twice: resb_deposit_heat=false baseline then =true).

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "coordinates/cell_locations.hpp"
#include "mhd/mhd.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
// Axisymmetric B_phi(r) profile on the x2-faces (phi-uniform => div-free seed).
KOKKOS_INLINE_FUNCTION
Real BphiProfile(Real r) { return r; }  // B0=1, linear in r; any phi-uniform f(r) works
constexpr Real RHO0  = 2.0;     // uniform density
constexpr Real ETOT0 = 5.0;     // uniform seeded total energy density (>> B^2/2)
constexpr Real EELE0 = 0.7;     // uniform seeded electron internal energy (scalar 0)
constexpr Real DECAY = 0.20;    // bphi := b_seed*(1-DECAY): a resb step that loses energy
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief resb B_phi write-back energy-budget unit test (#193).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("resb_energy_deposit_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  mhd::MHD *pmhd = pmbp->pmhd;
  test.CheckTrue(pmhd != nullptr, "MHD package constructed");
  if (pmhd == nullptr) { test.Finish(); return; }
  test.CheckTrue(pmbp->pcoord->coord_system == CoordSystem::cylindrical,
                 "cylindrical coordinate system");
  test.CheckTrue(pmhd->resb_operator_split,
                 "resb operator-split enabled (standalone bphi field allocated)");
  test.CheckTrue(pmhd->resb_couple_b0,
                 "resb_couple_b0 enabled (the coupled write-back path under test)");

  const bool deposit = pmhd->resb_deposit_heat;  // the #193 gate (read from <mhd>)

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1;
  const int nphi = je - js + 1;
  test.CheckTrue(nphi >= 2, "more than one phi cell (phi structure representable)");

  const int nmb1 = pmbp->nmb_thispack - 1;
  const int eele_idx = pmhd->nmhd;     // electron energy rides scalar 0 = u0(nmhd)
  test.CheckTrue(pmhd->nscalars >= 1,
                 "at least one passive scalar (the e_ele electron slot)");
  auto size = pmbp->pmb->mb_size;

  auto b1f = pmhd->b0.x1f;
  auto b2f = pmhd->b0.x2f;
  auto b3f = pmhd->b0.x3f;
  auto bph = pmhd->bphi;
  auto u   = pmhd->u0;

  // ---- seed a div-free axisymmetric b0 (B_r=B_z=0, B_phi(r) phi-uniform).  Fill the
  // full face extents so every face the write-back touches carries the right value.
  Kokkos::deep_copy(b1f, 0.0);
  Kokkos::deep_copy(b3f, 0.0);
  {
    const int n3 = b2f.extent_int(1);
    const int n2 = b2f.extent_int(2);
    const int n1 = b2f.extent_int(3);
    par_for("seed_b2f", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real r = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      b2f(m,k,j,i) = BphiProfile(r);
    });
  }
  // uniform conserved state: rho, zero momentum, known E_tot, known e_ele on scalar 0.
  Kokkos::deep_copy(u, 0.0);
  par_for("seed_u", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u(m,IDN,k,j,i) = RHO0;
    u(m,IM1,k,j,i) = 0.0;
    u(m,IM2,k,j,i) = 0.0;
    u(m,IM3,k,j,i) = 0.0;
    u(m,IEN,k,j,i) = ETOT0;
    u(m,eele_idx,k,j,i) = EELE0*RHO0;   // scalars are density-weighted (rho*s)
  });

  // ---- populate resb bphi with the phi-mean seed shrunk by (1-DECAY): a clean stand-in
  // for one resb super-step that removes a known POSITIVE field energy.  Phi-uniform =>
  // the write-back hits every face and div(B) is preserved (covered by the #192 test).
  par_for("seed_bphi", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real r = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    bph(m,0,k,j,i) = BphiProfile(r)*(1.0 - DECAY);
  });

  // ---- per-cell budget reduction over the active cells: sum E_tot, e_ele, e_ion, and
  // the field-energy decrease.  e_ion = E_tot - KE - B^2/2 - e_ele, B^2 the cell-centred
  // energy from the surrounding x2-faces (b1f=b3f=0 here).  The write-back shifts every
  // x2-face uniformly, so the phi-uniform field makes the cell-centred b2f well-posed.
  auto budget = [&](Real &sum_etot, Real &sum_eele, Real &sum_eion, Real &sum_dEfield) {
    Real le = 0.0, lee = 0.0, lei = 0.0, ldf = 0.0;
    const int nkji = (ke-ks+1)*(je-js+1)*(ie-is+1);
    const int nji  = (je-js+1)*(ie-is+1);
    const int ni   = (ie-is+1);
    Kokkos::parallel_reduce("budget",
    Kokkos::RangePolicy<>(DevExeSpace(),0,(nmb1+1)*nkji),
    KOKKOS_LAMBDA(const int idx, Real &me, Real &mee, Real &mei, Real &mdf) {
      int m = idx/nkji;
      int kji = idx - m*nkji;
      int k = kji/nji + ks;
      int ji = kji - (kji/nji)*nji;
      int j = ji/ni + js;
      int i = ji - (ji/ni)*ni + is;
      Real etot = u(m,IEN,k,j,i);
      Real rho  = u(m,IDN,k,j,i);
      Real ke_c = 0.5*(u(m,IM1,k,j,i)*u(m,IM1,k,j,i)
                     + u(m,IM2,k,j,i)*u(m,IM2,k,j,i)
                     + u(m,IM3,k,j,i)*u(m,IM3,k,j,i))/rho;
      Real eele = u(m,eele_idx,k,j,i);  // density-weighted (rho*e_ele)
      // cell-centred magnetic energy (b1f=b3f=0, so only the x2-face pair contributes).
      Real b2  = 0.5*(b2f(m,k,j,i) + b2f(m,k,j+1,i));
      Real eb  = 0.5*b2*b2;
      Real eion = etot - ke_c - eb - eele;
      // field-energy decrease for THIS cell's lower x2-face under the (1-DECAY) shrink:
      // 0.5*(b_old^2 - b_new^2) (the per-cell face the column owns).
      Real b_old = b2f(m,k,j,i);
      Real b_new = bph(m,0,k,j,i);
      Real df = 0.5*(b_old*b_old - b_new*b_new);
      me += etot; mee += eele; mei += eion; mdf += df;
    }, le, lee, lei, ldf);
    sum_etot = le; sum_eele = lee; sum_eion = lei; sum_dEfield = ldf;
  };

  Real etot_pre = 0.0, eele_pre = 0.0, eion_pre = 0.0, dEfield = 0.0;
  budget(etot_pre, eele_pre, eion_pre, dEfield);
  test.CheckTrue(dEfield > 0.0,
                 "the seeded resb step removes positive field energy (dissipative)");

  // ---- code under test: write the diffused bphi back onto b0.x2f (+ optional deposit).
  pmhd->CoupleResbBphiToB0();

  Real etot_post = 0.0, eele_post = 0.0, eion_post = 0.0, dEfield_post = 0.0;
  budget(etot_post, eele_post, eion_post, dEfield_post);

  const Real dEtot = etot_post - etot_pre;
  const Real dEele = eele_post - eele_pre;
  const Real dEion = eion_post - eion_pre;
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(12)
              << "### resb_energy_deposit: deposit=" << (deposit ? 1 : 0)
              << " dEfield=" << dEfield
              << " dEtot=" << dEtot << " dEele=" << dEele << " dEion=" << dEion
              << std::endl;
  }

  // tolerances: exact arithmetic on uniform fields, so round-off only.
  const Real rtol = 1.0e-12, atol = 1.0e-12;
  if (deposit) {
    // GREEN (#193): total energy conserved; electrons gained exactly the dissipated heat;
    // ions unchanged.
    test.CheckNear(dEtot, 0.0, 0.0, atol,
        "deposit ON: TOTAL energy conserved to round-off (field lost == heat gained)");
    test.CheckNear(dEele, dEfield, rtol, atol,
        "deposit ON: electrons gained exactly the dissipated field energy");
    test.CheckNear(dEion, 0.0, 0.0, atol,
        "deposit ON: ion internal energy unchanged");
  } else {
    // RED baseline (#192, current main): the field energy is dropped, so TOTAL energy
    // DROPS by exactly the field-energy decrease; electrons did NOT gain the heat; ions
    // unchanged.  (These mirror the deposit==true GREEN checks; with the deposit forced
    // off the conservation check FAILS by exactly dEfield -- the RED evidence.)
    test.CheckNear(dEtot, -dEfield, rtol, atol,
        "deposit OFF (#192): TOTAL energy drops by exactly the field-energy decrease");
    test.CheckNear(dEele, 0.0, 0.0, atol,
        "deposit OFF (#192): electrons did NOT gain the dissipated heat");
    test.CheckNear(dEion, 0.0, 0.0, atol,
        "deposit OFF (#192): ion internal energy unchanged");
  }

  test.Finish();
  return;
}
