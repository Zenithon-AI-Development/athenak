//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resb_divb_couple_test.cpp
//  \brief Unit test: the resb B_phi write-back MHD::CoupleResbBphiToB0() preserves the
//  cylindrical finite-volume div(B) even when the diffused `bphi` field carries
//  azimuthal (phi) structure (issue #192).
//
//  WHY THIS TEST.  The operator-split cylindrical resistive-B_phi step (#113, ADR-0004)
//  diffuses a standalone `bphi` field and -- when coupled to the live MHD drive
//  (resb_couple_b0, #181/ADR-0012) -- writes the result onto the staggered face field
//  b0.x2f via CoupleResbBphiToB0().  That write touches ONLY the x2 (phi) faces, OUTSIDE
//  the constrained-transport curl that keeps div(B)=0.  The cylindrical FV divergence
//      div(B) = (1/V)[ A1(i+1)B1(i+1) - A1(i)B1(i)
//                    + A2( B2(j+1) - B2(j) )            <- the phi term, B2 = b0.x2f
//                    + A3( B3(k+1) - B3(k) ) ]
//  changes under the write-back by EXACTLY the phi term, i.e. by A2/V * d(dB_phi)/dphi.
//  So the write is divergence-free IFF the APPLIED change in B_phi is phi-uniform.  At
//  paper resolution b0.x2f stays phi-uniform and div(B) holds; at the 1.5x-refined grid a
//  sub-machine phi asymmetry creeps into the diffused `bphi`, the per-phi-slice write
//  injects div(B), and constrained transport then faithfully PRESERVES and grows the
//  seeded constraint violation into a density runaway -> NaN (#192).
//
//  This test pins the fix's invariant at the unit level, fast and on the CPU: seed a
//  divergence-free axisymmetric b0 (B_r=B_z=0, B_phi(r) phi-uniform => div(B)=0 exactly),
//  populate the resb `bphi` with a real axisymmetric increment PLUS a zero-mean phi
//  perturbation (the refined-grid failure mode), call the REAL CoupleResbBphiToB0(), and
//  assert:
//    (1) the seeded b0 is divergence-free (sanity);
//    (2) AFTER the write-back div(B) is still ~0 -- phi structure did NOT leak into the
//        solenoidal constraint (RED with per-slice write b2f(j)=bphi(j); GREEN with the
//        phi-uniform-delta write);
//    (3) the write did real work -- it applied the axisymmetric increment to every face
//        (so a "fix" that stabilizes by writing nothing is rejected); and
//    (4) every face collapsed to the intended axisymmetric target f(r)*(1+d_ax), i.e. the
//        write keeps the physical (phi-mean) diffusion while dropping only the spurious
//        phi noise.
//
//  Built/run by tst/test_suite/unit_tests/test_unit_resb_divb_couple_cpu.py.

#include <cmath>
#include <iostream>
#include <limits>

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
// Axisymmetric B_phi(r) profile written onto the x2-faces (phi-uniform => div-free seed).
KOKKOS_INLINE_FUNCTION
Real BphiProfile(Real r) { return r; }  // B0=1, linear in r; any phi-uniform f(r) works
constexpr Real D_AX = 0.10;  // axisymmetric increment the diffusion applies (real work)
constexpr Real EPS  = 0.05;   // amplitude of the spurious zero-mean phi perturbation
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief resb B_phi write-back divergence-preservation unit test (#192).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("resb_divb_couple_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  mhd::MHD *pmhd = pmbp->pmhd;
  test.CheckTrue(pmhd != nullptr, "MHD package constructed");
  if (pmhd == nullptr) { test.Finish(); return; }
  test.CheckTrue(pmbp->pcoord->coord_system == CoordSystem::cylindrical,
                 "cylindrical coordinate system (the phi-divergence term is active)");
  test.CheckTrue(pmhd->resb_operator_split,
                 "resb operator-split enabled (standalone bphi field is allocated)");
  // CoupleResbBphiToB0 reads bphi(0,..) into b0.x2f; the coupled path must be wired.
  test.CheckTrue(pmhd->resb_couple_b0,
                 "resb_couple_b0 enabled (the coupled write-back path under test)");

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nx1 = indcs.nx1;
  const int nphi = je - js + 1;
  test.CheckTrue(nphi >= 2, "more than one phi cell (phi structure is representable)");

  const int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;
  auto csys = pmbp->pcoord->coord_system;
  const bool multi_d = pmy_mesh_->multi_d;
  const bool three_d = pmy_mesh_->three_d;

  auto b1f = pmhd->b0.x1f;
  auto b2f = pmhd->b0.x2f;
  auto b3f = pmhd->b0.x3f;
  auto bph = pmhd->bphi;
  auto u   = pmhd->u0;

  // ---- (A) seed a divergence-free axisymmetric b0: B_r = B_z = 0, B_phi(r) phi-uniform.
  // Fill the full face extents so every face the divergence stencil and the write-back
  // touch (incl. the i+1 / j+1 / k+1 upper faces) carries the right value.
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
  // sane conserved state so IEN write-back bill is well-defined (div(B) is independent
  // of it, but keep u0 finite).
  Kokkos::deep_copy(u, 0.0);
  par_for("seed_u", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u(m,IDN,k,j,i) = 1.0;
    u(m,IEN,k,j,i) = 1.0;
  });

  // ---- div(B) reduction (cylindrical FV; mirrors src/pgen/maglif.cpp's history probe).
  auto max_divb = [&]() -> Real {
    Real lmax = 0.0;
    const int nkji = (ke-ks+1)*(je-js+1)*(ie-is+1);
    const int nji  = (je-js+1)*(ie-is+1);
    const int ni   = (ie-is+1);
    Kokkos::parallel_reduce("divb", Kokkos::RangePolicy<>(DevExeSpace(),0,(nmb1+1)*nkji),
    KOKKOS_LAMBDA(const int idx, Real &mx) {
      int m = idx/nkji;
      int kji = idx - m*nkji;
      int k = kji/nji + ks;
      int ji = kji - (kji/nji)*nji;
      int j = ji/ni + js;
      int i = ji - (ji/ni)*ni + is;
      Real dx1 = size.d_view(m).dx1, dx2 = size.d_view(m).dx2, dx3 = size.d_view(m).dx3;
      Real x1min = size.d_view(m).x1min, x1max = size.d_view(m).x1max;
      Real rm = LeftEdgeX(i  -is, nx1, x1min, x1max);
      Real rp = LeftEdgeX(i+1-is, nx1, x1min, x1max);
      Real vol = CellVolume(csys, dx1, dx2, dx3, rm, rp);
      Real divb = (Face1Area(csys, dx2, dx3, rp)*b1f(m,k,j,i+1)
                 - Face1Area(csys, dx2, dx3, rm)*b1f(m,k,j,i))/vol;
      if (multi_d) {
        divb += Face2Area(csys, dx1, dx3)*(b2f(m,k,j+1,i) - b2f(m,k,j,i))/vol;
      }
      if (three_d) {
        divb += Face3Area(csys, dx1, dx2, rm, rp)*(b3f(m,k+1,j,i) - b3f(m,k,j,i))/vol;
      }
      mx = fmax(mx, fabs(divb));
    }, Kokkos::Max<Real>(lmax));
    return lmax;
  };

  const Real divb_seed = max_divb();
  test.CheckTrue(divb_seed < 1.0e-12,
                 "seeded b0 is divergence-free (axisymmetric B_phi, B_r=B_z=0)");

  // ---- (B) populate resb bphi: a REAL axisymmetric increment (D_AX) PLUS a zero-mean
  // phi perturbation (EPS) -- the refined-grid failure mode the write-back must not leak.
  par_for("seed_bphi", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real r = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real f = BphiProfile(r);
    Real phase = 2.0*M_PI*static_cast<Real>(j-js)/static_cast<Real>(nphi);
    bph(m,0,k,j,i) = f*(1.0 + D_AX + EPS*Kokkos::cos(phase));
  });

  // snapshot the pre-write face field to measure the applied change.
  DvceArray4D<Real> b2f0("b2f0", b2f.extent_int(0), b2f.extent_int(1),
                         b2f.extent_int(2), b2f.extent_int(3));
  Kokkos::deep_copy(b2f0, b2f);

  // ---- (C) the code under test: write the diffused bphi back onto b0.x2f.
  pmhd->CoupleResbBphiToB0();

  // ---- (2) the #192 invariant: the write-back did NOT inject div(B).
  const Real divb_post = max_divb();
  if (global_variable::my_rank == 0) {
    std::cout << "### resb_divb_couple: divb_seed=" << divb_seed
              << " divb_post=" << divb_post << " (nphi=" << nphi << ")" << std::endl;
  }
  test.CheckTrue(divb_post < 1.0e-10,
      "CoupleResbBphiToB0 preserves div(B) under a phi-structured bphi (the #192 fix)");

  // ---- (3) the write applied the axisymmetric increment (rejects a no-op "fix"), and
  // ---- (4) every face collapsed to the intended phi-uniform target f(r)*(1+D_AX).
  Real max_change = 0.0, max_target_err = 0.0;
  {
    Real lchg = 0.0, lerr = 0.0;
    const int nkji = (ke-ks+1)*(je-js+1)*(ie-is+1);
    const int nji  = (je-js+1)*(ie-is+1);
    const int ni   = (ie-is+1);
    Kokkos::parallel_reduce("postcheck",
    Kokkos::RangePolicy<>(DevExeSpace(),0,(nmb1+1)*nkji),
    KOKKOS_LAMBDA(const int idx, Real &mchg, Real &merr) {
      int m = idx/nkji;
      int kji = idx - m*nkji;
      int k = kji/nji + ks;
      int ji = kji - (kji/nji)*nji;
      int j = ji/ni + js;
      int i = ji - (ji/ni)*ni + is;
      Real r = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      Real target = BphiProfile(r)*(1.0 + D_AX);
      mchg = fmax(mchg, fabs(b2f(m,k,j,i) - b2f0(m,k,j,i)));
      merr = fmax(merr, fabs(b2f(m,k,j,i) - target));
    }, Kokkos::Max<Real>(lchg), Kokkos::Max<Real>(lerr));
    max_change = lchg; max_target_err = lerr;
  }
  // the smallest-radius interior cell has the smallest axisymmetric increment ~ r*D_AX;
  // require the write changed the field by at least half of the min expected increment.
  Real rmin_cell = CellCenterX(0, nx1, size.h_view(0).x1min, size.h_view(0).x1max);
  test.CheckTrue(max_change > 0.5*rmin_cell*D_AX,
      "write-back applied the axisymmetric increment (not a no-op stabilization)");
  test.CheckTrue(max_target_err < 1.0e-10,
      "every x2-face collapsed to the phi-uniform target f(r)*(1+D_AX) (noise dropped)");

  test.Finish();
  return;
}
