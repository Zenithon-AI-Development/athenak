//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_cyl_phi_metric_test.cpp
//  \brief Unit test: the AZIMUTHAL metric of the FLD operators on a cylindrical (r, phi)
//  mesh (issue #215, consuming the #116 geometry accessors).
//
//  On a cylindrical mesh `dx2` is an ANGLE, not a length.  The azimuthal gradient of the
//  radiation energy is therefore dE/(r dphi), and the physical face width is the arc
//  length CenterWidth2(cylindrical, dx2, x1v) = x1v*dx2 -- so the phi contribution to the
//  Laplacian carries 1/r^2.  An operator that divides by the raw `dx2` is off by r^2 (and
//  is dimensionally inconsistent with its own x1 term, so the diffusion becomes
//  anisotropic in a way nothing physical asked for).  On a Cartesian mesh
//  CenterWidth2 returns dx2 and every expression below is bit-identical, which is why the
//  existing Cartesian Marshak tests never covered this.
//
//  Batteries (both on the SAME field and the SAME analytic oracle):
//   (A) GREY guard: FLDGreyOperator already takes the arc length (#116); pin it.
//   (B) MULTIGROUP: FLDMultigroupOperator must agree.  RED while its x2 kernel divides by
//       the raw dx2 -- on this mesh (r ~ 2.5) that is an O(r^2) ~ 6x error.
//
//  The field is a pure azimuthal cosine, uniform in r:
//      E(r, phi) = E0 + a cos(m phi)
//  so the radial flux vanishes identically and the operator action is the discrete
//  azimuthal diffusion eigenvalue
//      M(E) = -(2 D / (r_i dphi)^2) (1 - cos(m dphi)) * a cos(m phi_j),   D = c/(3 chi)
//  in the optically-THICK limit (chi large => R << 1 => lambda -> 1/3), which also makes
//  the #194 Lax-Friedrichs streaming term vanish identically ((1-3 lambda)_+ = 0) so this
//  test isolates the metric alone.  Cells adjacent to the x2 boundaries are excluded (the
//  operators fill those ghosts zero-gradient, not periodically).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_cyl_phi_metric_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_cyl_phi_metric_cpu.py.

#include <cmath>     // std::cos, std::fabs, std::acos
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "radiation_fld/fld_grey_operator.hpp"
#include "radiation_fld/fld_multigroup_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Azimuthal-metric test for the grey and multigroup FLD operators on (r, phi).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_cyl_phi_metric_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ng = indcs.ng;
  const int nmb = pmbp->nmb_thispack;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = indcs.nx2 + 2*ng;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*ng : 1;

  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real x2min = pin->GetReal("mesh", "x2min");
  const Real x2max = pin->GetReal("mesh", "x2max");
  const Real dr = (x1max - x1min)/static_cast<Real>(indcs.nx1);
  const Real dphi = (x2max - x2min)/static_cast<Real>(indcs.nx2);

  // The mesh must actually be cylindrical and 2-D in (r, phi), or this test is vacuous.
  test.CheckTrue(pmbp->pcoord->coord_system == CoordSystem::cylindrical,
                 "test mesh is cylindrical (else the azimuthal metric is untested)");
  test.CheckTrue(indcs.nx2 > 1, "test mesh resolves phi (nx2 > 1)");
  // r must be far from 1 or the r^2 factor is invisible.
  test.CheckTrue(x1min >= 2.0, "test mesh sits at r >> 1 so the r^2 metric factor bites");

  const Real c_light = pin->GetOrAddReal("problem", "c_light", 1.0);
  const Real nl = pin->GetOrAddReal("problem", "n_larsen", 2.0);
  const Real efloor = pin->GetOrAddReal("problem", "efloor", 1.0e-30);
  // optically THICK: R << 1 => lambda -> 1/3 => D -> c/(3 chi) and the LF term is gated
  // off, so any grey/multigroup difference here is the metric and nothing else.
  const Real chi = pin->GetOrAddReal("problem", "chi_thick", 1.0e4);
  const Real E0 = 1.0;
  const Real amp = 1.0e-3;
  const int m_mode = 2;

  // phi at the centre of cell j (x2min + (j - js + 0.5) dphi)
  auto phic = [&](int j) -> Real {
    return x2min + dphi*(static_cast<Real>(j - js) + 0.5);
  };
  auto rc = [&](int i) -> Real {
    return x1min + dr*(static_cast<Real>(i - is) + 0.5);
  };

  auto fill = [&](DvceArray5D<Real> &a) {
    auto h_a = Kokkos::create_mirror_view(a);
    const int nv = a.extent_int(1);
    for (int m = 0; m < nmb; ++m) {
      for (int v = 0; v < nv; ++v) {
        for (int k = 0; k < n3; ++k) {
          for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
              h_a(m, v, k, j, i) = E0 + amp*std::cos(m_mode*phic(j));
            }
          }
        }
      }
    }
    Kokkos::deep_copy(a, h_a);
  };

  // Discrete azimuthal diffusion eigenvalue at cell (i, j), with the ARC-LENGTH face
  // width w2 = r_i dphi:  M = -(2 D/w2^2)(1 - cos(m dphi)) * a cos(m phi_j).
  const Real D = c_light/(3.0*chi);
  auto expected = [&](int i, int j) -> Real {
    const Real w2 = rc(i)*dphi;
    return -(2.0*D/(w2*w2))*(1.0 - std::cos(m_mode*dphi))*amp*std::cos(m_mode*phic(j));
  };
  // scale for the tolerance: the eigenvalue magnitude at the innermost radius
  const Real escale =
      (2.0*D/((rc(is)*dphi)*(rc(is)*dphi)))*(1.0 - std::cos(m_mode*dphi))*amp;

  // Reduce |M - analytic| over the interior AND report finiteness separately.  This must
  // NOT use a bare std::fmax accumulator: IEEE fmax(a, NaN) returns a, so a kernel that
  // produces NaN at every cell would reduce to 0 and read as a perfect match.  That is
  // precisely how the missing cylindrical geometry in the multigroup flux divergence
  // (0/0 -> NaN at every cell) first hid from this test.
  auto reduce = [&](const DvceArray5D<Real> &rhs, Real *maxerr, bool *finite) {
    auto h_r = Kokkos::create_mirror_view(rhs);
    Kokkos::deep_copy(h_r, rhs);
    *maxerr = 0.0;
    *finite = true;
    for (int i = is; i <= ie; ++i) {
      for (int j = js + 1; j <= je - 1; ++j) {     // skip the zero-gradient x2 ghosts
        const Real got = h_r(0, 0, 0, j, i);
        if (!std::isfinite(got)) { *finite = false; continue; }
        const Real err = std::fabs(got - expected(i, j));
        if (err > *maxerr) { *maxerr = err; }
      }
    }
  };

  // ===== (A) GREY guard: the grey operator already carries the arc length (#116) =====
  {
    DvceArray5D<Real> e("erad_grey", nmb, 1, n3, n2, n1);
    DvceArray5D<Real> r("rhs_grey", nmb, 1, n3, n2, n1);
    fill(e);
    FLDGreyOperator opg(pmbp, pin, e, c_light, chi, nl, -1.0, efloor);
    opg.ApplyBoundary(e);
    opg.OperatorAction(e, r);
    Real maxerr = 0.0;
    bool finite = true;
    reduce(r, &maxerr, &finite);
    std::cout << "[fld_cyl_phi_metric_test] (A) grey max|M - analytic| = " << maxerr
              << " finite=" << finite << " (scale " << escale << ")" << std::endl;
    test.CheckTrue(finite, "grey FLD operator action is finite on a cylindrical mesh");
    test.CheckNear(maxerr, 0.0, 0.0, 1.0e-6*escale,
                   "grey FLD azimuthal flux uses the arc length r*dphi (#116)");
  }

  // ===== (B) MULTIGROUP must agree: same field, same oracle (#215) =====
  {
    opacity::MultigroupOpacity tab;
    tab.Allocate(1, 2, 2, 1.0, 1.0e3, 1.0e-4, 1.0e0);
    const Real rho_bg = pin->GetOrAddReal("problem", "rho_bg", 1.0);
    const Real te_bg = pin->GetOrAddReal("problem", "te_bg", 1.0e2);
    {
      const Real kr = chi/rho_bg;                  // chi_0 = rho_bg*kappa == grey chi
      auto h_kr = Kokkos::create_mirror_view(tab.rosseland);
      for (int it = 0; it < tab.ntemp; ++it) {
        for (int id = 0; id < tab.ndens; ++id) { h_kr(0, it, id) = kr; }
      }
      Kokkos::deep_copy(tab.rosseland, h_kr);
    }
    DvceArray5D<Real> e("erad_mg", nmb, 1, n3, n2, n1);
    DvceArray5D<Real> r("rhs_mg", nmb, 1, n3, n2, n1);
    fill(e);
    FLDMultigroupOperator opm(pmbp, pin, e, tab, c_light, rho_bg, te_bg, nl, -1.0,
                              efloor);
    opm.ApplyBoundary(e);
    opm.OperatorAction(e, r);
    Real maxerr = 0.0;
    bool finite = true;
    reduce(r, &maxerr, &finite);
    std::cout << "[fld_cyl_phi_metric_test] (B) multigroup max|M - analytic| = " << maxerr
              << " finite=" << finite << " (scale " << escale << ")" << std::endl;
    test.CheckTrue(finite,
                   "multigroup FLD operator action is FINITE on a cylindrical mesh: the "
                   "flux divergence must carry rm/rp/x1v or it is 0/0 everywhere (#215)");
    test.CheckNear(maxerr, 0.0, 0.0, 1.0e-6*escale,
                   "multigroup FLD azimuthal flux uses the arc length r*dphi (#215)");
  }

  test.Finish();
}
