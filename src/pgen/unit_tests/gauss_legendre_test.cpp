//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gauss_legendre_test.cpp
//  \brief Unit test: cross integrals of spin-weighted spherical harmonics on a
//  GaussLegendreGrid should be delta functions (orthonormality).  Converted to the shared
//  unit_test.hpp assert/report helpers (see that header and sample_unit_test.cpp).

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>
#include <utility>    // std::pair
#include <cctype>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "utils/spherical_harm.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Gauss-Legendre spherical-harmonic orthonormality unit test.
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;
  (void)indcs;

  unit_test::UnitTest test("gauss_legendre_test");

  int ntheta = pin->GetOrAddInteger("problem", "ntheta", 16);

  GaussLegendreGrid grid(pmbp, ntheta, 1);

  // --------------------------------------------------------------------------------
  // Guard 1 (TIGHT, precision-safe): the quadrature weights must integrate the constant
  // 1 over the unit sphere to 4*pi.  This directly exercises the Gauss-Legendre roots and
  // weights (the legendre_roots.hpp root-finding) WITHOUT touching the spherical-harmonic
  // evaluator, so its tolerance can stay at quadrature precision (~1e-12).  Any real
  // regression in the root-finding/weights (e.g. #629) trips HERE, where the bound is
  // tight -- it is NOT masked by the harmonic-evaluation floor relaxed for Guard 2 below.
  double weight_sum = 0.0;
  for (int ip = 0; ip < grid.nangles; ++ip) {
    weight_sum += grid.int_weights.h_view(ip);
  }
  test.CheckNear(weight_sum, 4.0*M_PI, 0.0, 1e-12, "sum of quadrature weights = 4*pi");

  // --------------------------------------------------------------------------------
  // Guard 2: cross integrals of spin-weighted spherical harmonics are delta functions
  // (orthonormality).  A DETERMINISTIC set of (l,m) pairs spanning low->high l (was a
  // non-deterministic std::random_device draw, which made this CI test flaky; see #178).
  // The pairs include m=0 / same-m-different-l (theta orthogonality at high polynomial
  // degree), different-m (phi orthogonality), and m=l (a single-term, well-conditioned
  // Wigner-d) so the grid is exercised across its full degree range reproducibly.
  const int lmax = ntheta - 1;
  const std::vector<std::pair<int, int>> pairs = {
    {1, 0}, {lmax/2, 0}, {lmax, 0},                       // m=0: theta-orthonormality
    {std::max(1, lmax/4), 4}, {std::max(4, lmax-4), 4},   // same m, different l
    {std::max(3, lmax/3), -3}, {lmax, -3},                // same m, different l
    {std::max(1, lmax-3), std::max(1, lmax-3)},           // m=l (single-term Wigner-d)
    {std::max(7, lmax/2), -7}, {std::max(11, 3*lmax/4), 11}  // mixed
  };
  const int npairs = static_cast<int>(pairs.size());

  // Tolerance rationale (see #178): in exact arithmetic these integrals are delta
  // functions to machine precision -- verified independently that (a) the GL roots are
  // accurate to ~1e-16 / weights to ~1e-15, and (b) with high-precision harmonics the
  // grid quadrature error is ~1e-15.  The observed deviation is therefore NOT a
  // grid/root-finding regression but the finite-precision floor of the double-precision
  // Wigner-d/factorial SWSphericalHarm evaluation; its cancellation grows steeply with l
  // (measured max error at ntheta=25: ~2e-14 @ l=10, ~8e-13 @ l=15, ~2e-10 @ l=24, i.e.
  // ~ eps * 10^(0.295 l)).  The bound below envelopes that floor with a ~5x margin while
  // staying tight (1e-13) at low l, so a genuine high-degree quadrature regression still
  // trips it.  Do NOT flat-loosen this to hide a root-finding regression -- Guard 1 above
  // keeps the directly-testable quadrature pinned at quadrature precision.
  double ylmR1, ylmI1, ylmR2, ylmI2;
  double int_r, int_i;
  double max_err = 0;
  double max_rel = 0;  // worst error relative to its own l-aware tolerance

  // outer loop over pairs of spherical harmonics
  for (int n1 = 0; n1 < npairs; ++n1)
  for (int n2 = n1; n2 < npairs; ++n2) {
    // reset doubles to store integration value
    int_r = 0;
    int_i = 0;

    // iterate over the angles
    for (int ip = 0; ip < grid.nangles; ++ip) {
      Real theta = grid.polar_pos.h_view(ip,0);
      Real phi = grid.polar_pos.h_view(ip,1);
      Real weight = grid.int_weights.h_view(ip);
      SWSphericalHarm(&ylmR1,&ylmI1, pairs[n1].first, pairs[n1].second, 0, theta, phi);
      SWSphericalHarm(&ylmR2,&ylmI2, pairs[n2].first, pairs[n2].second, 0, theta, phi);
      // complex conjugate
      ylmI2 *= -1;
      int_r += weight*(ylmR1*ylmR2 - ylmI1*ylmI2);
      int_i += weight*(ylmR1*ylmI2 + ylmR2*ylmI1);
    }

    // expected: real part is 1 on the diagonal (same l,m) and 0 otherwise; imag part is
    // always 0 (the harmonics are orthonormal under the conjugated inner product).
    bool diagonal = (pairs[n1].first == pairs[n2].first &&
                     pairs[n1].second == pairs[n2].second);
    double expected_r = diagonal ? 1.0 : 0.0;
    max_err = std::max(max_err, std::abs(int_r - expected_r));
    max_err = std::max(max_err, std::abs(int_i));

    // l-aware finite-precision floor of the double-precision harmonic evaluation
    int lp = std::max(pairs[n1].first, pairs[n2].first);
    double tol = std::max(1.0e-13, 1.0e-16 * std::pow(10.0, 0.295*lp));
    max_rel = std::max(max_rel, std::abs(int_r - expected_r)/tol);
    max_rel = std::max(max_rel, std::abs(int_i)/tol);

    std::ostringstream label;
    label << "<Y(l=" << pairs[n1].first << ",m=" << pairs[n1].second << ")|Y(l="
          << pairs[n2].first << ",m=" << pairs[n2].second << ")>";
    test.CheckNear(int_r, expected_r, 0.0, tol, label.str() + " real part");
    test.CheckNear(int_i, 0.0, 0.0, tol, label.str() + " imag part");
  }

  std::cout << "[gauss_legendre_test] maximum error = " << max_err
            << " (worst error/tolerance = " << max_rel << ")" << std::endl;
  test.Finish();

  return;
}
