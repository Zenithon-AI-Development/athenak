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
#include <cctype>
#include <random>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "geodesic-grid/gauss_legendre.hpp"
#include "utils/spherical_harm.hpp"
#include "pgen/unit_tests/unit_test.hpp"

using u32    = uint_least32_t;
using s32    = int_least32_t;
using engine = std::mt19937;

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

  // test that the cross integral of spherical harmonics are delta functions.
  // First initialize 10 random pairs of l and m, with 0 <= l <=ntheta.

  std::random_device os_seed;
  const u32 seed = os_seed();

  engine generator( seed );
  std::uniform_int_distribution< u32 > distribute_l( 1, ntheta-1);

  std::vector<int> ls;
  std::vector<int> ms;

  for (int repetition = 0; repetition < 10; ++repetition) {
    int l = distribute_l(generator);
    std::uniform_int_distribution< s32 > distribute_m( -l, l);
    int m = distribute_m(generator);
    ls.push_back(l);
    ms.push_back(m);
  }

  double ylmR1, ylmI1, ylmR2, ylmI2;
  double int_r, int_i;
  double max_err = 0;
  const double tol = 1e-10;

  // outer loop over pairs of spherical harmonics
  for (int n1 = 0; n1 < 10; ++n1)
  for (int n2 = n1; n2 < 10; ++n2) {
    // reset doubles to store integration value
    int_r = 0;
    int_i = 0;

    // iterate over the angles
    for (int ip = 0; ip < grid.nangles; ++ip) {
      Real theta = grid.polar_pos.h_view(ip,0);
      Real phi = grid.polar_pos.h_view(ip,1);
      Real weight = grid.int_weights.h_view(ip);
      SWSphericalHarm(&ylmR1,&ylmI1, ls[n1], ms[n1], 0, theta, phi);
      SWSphericalHarm(&ylmR2,&ylmI2, ls[n2], ms[n2], 0, theta, phi);
      // complex conjugate
      ylmI2 *= -1;
      int_r += weight*(ylmR1*ylmR2 - ylmI1*ylmI2);
      int_i += weight*(ylmR1*ylmI2 + ylmR2*ylmI1);
    }

    // expected: real part is 1 on the diagonal (same l,m) and 0 otherwise; imag part is
    // always 0 (the harmonics are orthonormal under the conjugated inner product).
    bool diagonal = (ls[n1] == ls[n2] && ms[n1] == ms[n2]);
    double expected_r = diagonal ? 1.0 : 0.0;
    max_err = std::max(max_err, std::abs(int_r - expected_r));
    max_err = std::max(max_err, std::abs(int_i));

    std::ostringstream label;
    label << "<Y(l=" << ls[n1] << ",m=" << ms[n1] << ")|Y(l=" << ls[n2]
          << ",m=" << ms[n2] << ")>";
    test.CheckNear(int_r, expected_r, 0.0, tol, label.str() + " real part");
    test.CheckNear(int_i, 0.0, 0.0, tol, label.str() + " imag part");
  }

  std::cout << "[gauss_legendre_test] maximum error = " << max_err << std::endl;
  test.Finish();

  return;
}
