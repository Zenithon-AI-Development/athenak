//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_amr_flux_correction_test.cpp
//  \brief Unit test: the operator-split diffusive operators (here the grey
//  FLDGreyOperator, #17) conserve the diffused quantity across a STATIC refinement (SMR)
//  boundary once their face flux is registered for AthenaK's conservative fine->coarse
//  flux correction (issue [19a]/#33).  The same registration is wired into anisotropic
//  conduction (#18) and resistive B_phi (#27) operators; FLD is the operator whose
//  block-boundary face flux is NOT zeroed, so it genuinely exercises the correction.
//
//  IDEA (the discrete conservation identity).  For an insulated/periodic closed domain,
//  total diffused quantity changes at rate  d/dt SUM(E dV) = SUM(dV * M(E)) , which must
//  be ZERO: the per-cell flux divergence telescopes to the net flux through the domain
//  boundary (= 0).  This telescoping requires the flux through every INTERNAL face to
//  cancel between the two cells that share it.  At a 2:1 refinement face the lone coarse
//  flux must equal the area-weighted sum of the fine fluxes -- which is exactly what the
//  flux correction enforces (it overwrites the coarse-side face flux with the restricted,
//  area-averaged fine flux).  WITHOUT the correction the coarse one-sided flux differs
//  from the average of the fine fluxes, the refinement-boundary terms do NOT cancel, and
//  SUM(dV*M) is O(1).  WITH it they cancel to round-off.  So |SUM(dV*M)| / SUM|dV*M| is
//  the red->green discriminator.
//
//  SETUP.  A 2D periodic Cartesian mesh (48x48, 16x16 MeshBlocks -> 3x3 root blocks) with
//  a single INTERIOR root block statically refined to level 1 (region [0.34,0.66]^2), so
//  the fine quadrant is surrounded on all four faces by coarse blocks and no refinement
//  face touches the (periodic) domain edge.  The radiation field is filled (active cells
//  AND all ghost zones, across block boundaries) from a periodic analytic profile
//  E = E0 + amp*(cos 2pi x + cos 2pi y); this gives a single-valued, consistent value at
//  every same-level face (so those cancel) and a non-trivial gradient across the
//  refinement faces (so the correction matters).  OperatorAction is then called DIRECTLY
//  (NOT via ApplyBoundary, which would overwrite the analytic cross-block ghosts with a
//  zero-gradient fill), and the conservation residual is reduced on the device with the
//  per-block cell volume dV.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_amr_flux_correction_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_amr_flux_correction_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::acos
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief AMR/SMR flux-correction conservation unit test for the diffusive operators.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_amr_flux_correction_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int ng = indcs.ng;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const int n1 = nx1 + 2*ng;
  const int n2 = (nx2 > 1) ? nx2 + 2*ng : 1;
  const int n3 = (nx3 > 1) ? nx3 + 2*ng : 1;
  const bool multi_d = pmy_mesh_->multi_d;
  const bool three_d = pmy_mesh_->three_d;

  // The whole point of the test is a genuinely multilevel, multi-block mesh.
  test.CheckTrue(pmy_mesh_->multilevel, "mesh is multilevel (static refinement present)");
  test.CheckTrue(nmb > 1, "more than one MeshBlock present (refinement boundary exists)");

  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real chi = pin->GetOrAddReal("problem", "chi", 1.0);
  const Real nl = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real E0 = 1.0;
  const Real amp = 0.1;
  const Real PI = std::acos(-1.0);

  DvceArray5D<Real> erad("erad", nmb, 1, n3, n2, n1);
  DvceArray5D<Real> rhs("rhs", nmb, 1, n3, n2, n1);
  auto size = pmbp->pmb->mb_size;

  // Fill EVERY cell (active + ghosts) of every block from the periodic analytic profile,
  // using each block's own metric so the fine blocks get the fine spacing.  Filling the
  // ghosts (rather than calling ApplyBoundary) makes the value single-valued across block
  // boundaries -> same-level fluxes match and cancel; only the refinement faces mismatch.
  par_for("fld_amr_fill", DevExeSpace(), 0, nmb-1, 0, n3-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real x = size.d_view(m).x1min + (static_cast<Real>(i - is) + 0.5)*size.d_view(m).dx1;
    Real val = E0 + amp*Kokkos::cos(2.0*PI*x);
    if (multi_d) {
      Real y = size.d_view(m).x2min
             + (static_cast<Real>(j - js) + 0.5)*size.d_view(m).dx2;
      val += amp*Kokkos::cos(2.0*PI*y);
    }
    erad(m,0,k,j,i) = val;
  });

  // Build the grey FLD operator (on a multilevel mesh it registers rflx for flux
  // correction) and evaluate M(E) = div(D grad E) directly (no ApplyBoundary: keep the
  // analytic cross-block ghosts).  The CorrectFlux call inside OperatorAction overwrites
  // the coarse-side refinement-boundary face flux with the restricted fine flux.
  FLDGreyOperator op(pmbp, pin, erad, c_light, chi, nl, -1.0);
  op.OperatorAction(erad, rhs);

  // Conservation residual SUM(dV*M) and its scale SUM|dV*M| over the ACTIVE cells of all
  // blocks (per-block dV, so fine blocks carry their smaller volume).
  const int nmkji = nmb*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  Real ssum = 0.0;
  Kokkos::parallel_reduce("fld_amr_consv", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &lsum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real dV = size.d_view(m).dx1;
    if (multi_d) { dV *= size.d_view(m).dx2; }
    if (three_d) { dV *= size.d_view(m).dx3; }
    lsum += dV*rhs(m,0,k,j,i);
  }, ssum);
  Real sabs = 0.0;
  Kokkos::parallel_reduce("fld_amr_scale", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &lsum) {
    int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks; j += js;
    Real dV = size.d_view(m).dx1;
    if (multi_d) { dV *= size.d_view(m).dx2; }
    if (three_d) { dV *= size.d_view(m).dx3; }
    lsum += Kokkos::fabs(dV*rhs(m,0,k,j,i));
  }, sabs);

  // The diffusion is non-trivial (a sanity guard so the conservation check is meaningful:
  // if every flux were zero the residual would be trivially zero).
  test.CheckTrue(sabs > 1.0e-8, "operator action is non-trivial (some diffusive flux)");

  // The conservation discriminator: with the registered flux correction the refinement-
  // boundary contributions cancel and the residual is at round-off; without it the
  // residual is O(1) vs the scale (red->green by disabling the CorrectFlux call).
  test.CheckNear(ssum, 0.0, 0.0, 1.0e-11*(sabs + 1.0e-30),
                 "FLD operator conserves the diffused energy across the SMR boundary");

  test.Finish();
  // All checks run on local arrays in UserProblem; exit cleanly (nlim = tlim = 0).
  std::exit(EXIT_SUCCESS);
  (void)restart;
  return;
}
