//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file grey_opacity_mean_test.cpp
//  \brief Unit test: the grey (frequency-integrated) reduction of a tabulated multigroup
//  opacity (radiation_fld/grey_opacity_mean.hpp), issue #204.  The grey Rosseland mean
//  is the dB/dT-weighted harmonic mean over the photon-energy groups,
//      1/kappa_R(rho,Te) = [ sum_g w_g / kappa_R,g(rho,Te) ] / [ sum_g w_g ],
//      w_g = int_{x_g}^{x_{g+1}} x^4 e^x/(e^x-1)^2 dx,   x = eps/(k_B Te),
//  i.e. the exact continuum Rosseland mean when kappa is piecewise-constant per group.
//  It is the per-cell chi(rho,Te) lookup the grey FLD operator needs to stop treating
//  the tenuous vacuum gap at solid-liner opacity (#204: opaque liner, transparent gap).
//
//  Fixture: the committed IONMIX-style power-law table of the reader test
//  (inputs/unit_tests/ionmix_opacity_test.cn4):
//      rosseland(ig,rho,T) = 1.5*(ig+1)*rho^0.6*T^-2.0,  4 groups, bounds 1..1e4 eV.
//  Oracles (Layer 1, analytic / independent quadrature):
//   (1) matches an INDEPENDENT host Simpson quadrature of the weighted harmonic mean
//       (the weights are integrated numerically here, not via the code's own
//       PlanckFraction helpers -- an independent oracle);
//   (2) bounded by the min/max per-group kappa at the same (rho,Te);
//   (3) spectrum far below the group range -> the lowest group's kappa; far above ->
//       the highest group's kappa;
//   (4) rho-scaling passthrough: weights are rho-independent, so the mean inherits the
//       fixture's exact rho^0.6 power law;
//   (5) the group-weighting factor (mean divided by the common rho/T law) rises
//       monotonically with Te as the Planck spectrum sweeps up through groups of
//       increasing kappa.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/grey_opacity_mean_test -B build_unit
//    (cd build_unit/src && make) && ./build_unit/src/athena \
//        -i inputs/unit_tests/grey_opacity_mean_test.athinput \
//        problem/opacity_file=$PWD/inputs/unit_tests/ionmix_opacity_test.cn4

#include <cmath>     // std::pow, std::exp, std::expm1
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "radiation_fld/grey_opacity_mean.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {

// Synthetic power-law Rosseland / Planck-absorption mass-opacity laws of the fixture
// (host oracles).
Real Kro(int ig, Real rho, Real T) {
  return 1.5*(ig + 1)*std::pow(rho, 0.6)*std::pow(T, -2.0);
}
Real Kpa(int ig, Real rho, Real T) {
  return 2.0*(ig + 1)*std::pow(rho, 0.5)*std::pow(T, -1.5);
}

// Rosseland weight integrand x^4 e^x/(e^x-1)^2, evaluated overflow-safely.  This is the
// (unnormalized) dB/dT spectral weight; the normalization cancels in the weighted mean,
// so the oracle needs no 15/pi^4 factors.
Real WRoss(Real x) {
  if (x <= 0.0) { return 0.0; }
  if (x > 500.0) { return 0.0; }               // e^-x underflows the weight anyway
  Real emx = std::exp(-x);
  Real om = -std::expm1(-x);                   // 1 - e^-x, accurate near 0
  Real x2 = x*x;
  return x2*x2*emx/(om*om);
}

// Planck weight integrand x^3/(e^x-1) (unnormalized B_g spectral weight).
Real WPlanck(Real x) {
  if (x <= 0.0) { return 0.0; }
  if (x > 500.0) { return 0.0; }
  Real emx = std::exp(-x);
  Real om = -std::expm1(-x);                   // 1 - e^-x
  return x*x*x*emx/om;
}

// Composite-Simpson integral of f on [a,b] (independent quadrature oracle).
template <typename F>
Real Simpson(F f, Real a, Real b, int n) {     // n even
  if (b <= a) { return 0.0; }
  Real h = (b - a)/n;
  Real s = f(a) + f(b);
  for (int i = 1; i < n; ++i) {
    s += f(a + i*h)*((i % 2 == 1) ? 4.0 : 2.0);
  }
  return s*h/3.0;
}

// Independent host oracle: dB/dT-weighted harmonic mean of the fixture's power-law
// per-group kappas at (rho, te), group bounds in eV, te in eV (k_B = 1 in eV units).
Real OracleGreyRosseland(const Real *bounds, int ngroups, Real rho, Real te) {
  Real wsum = 0.0, hsum = 0.0;
  for (int g = 0; g < ngroups; ++g) {
    Real xlo = bounds[g]/te;
    Real xhi = bounds[g+1]/te;
    // clip the quadrature to where the weight is non-negligible
    Real lo = (xlo < 80.0) ? xlo : 80.0;
    Real hi = (xhi < 80.0) ? xhi : 80.0;
    Real w = Simpson(WRoss, lo, hi, 20000);
    wsum += w;
    hsum += w/Kro(g, rho, te);
  }
  return wsum/hsum;
}

// Independent host oracle: Planck-fraction-weighted arithmetic mean of the fixture's
// per-group Planck-absorption kappas at (rho, te).
Real OracleGreyPlanck(const Real *bounds, int ngroups, Real rho, Real te) {
  Real bsum = 0.0, ksum = 0.0;
  for (int g = 0; g < ngroups; ++g) {
    Real xlo = bounds[g]/te;
    Real xhi = bounds[g+1]/te;
    Real lo = (xlo < 80.0) ? xlo : 80.0;
    Real hi = (xhi < 80.0) ? xhi : 80.0;
    Real b = Simpson(WPlanck, lo, hi, 20000);
    bsum += b;
    ksum += b*Kpa(g, rho, te);
  }
  return ksum/bsum;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Grey Rosseland mean over a tabulated multigroup opacity: unit test (#204).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("grey_opacity_mean_test");

  std::string fname = pin->GetString("problem", "opacity_file");
  opacity::MultigroupOpacity table;
  opacity::ReadIonmixOpacity(fname, table);
  test.CheckTrue(table.ngroups == 4, "fixture parsed with 4 groups");

  const Real bounds[5] = {1.0, 10.0, 100.0, 1000.0, 10000.0};
  const Real kboltz = 1.0;                     // group bounds [eV], Te [eV]

  // query points: interior (rho, te); rho pair for the scaling check; te sweep for the
  // limit + monotonicity checks (all te inside the table's [1,1e4] eV temperature range
  // so the table lookup and the analytic law agree without clamping).
  const Real rho_a = 5.0e-2, te_a = 50.0;      // interior: weights span groups 1-2
  const Real rho_b = 5.0e-3;                   // rho-scaling partner (x10 below rho_a)
  // te_lo sits far enough below the group range that ~all in-range dB/dT weight lands
  // in group 0 (at te=1 eV ~10% still spills into group 1); the table lookup clamps
  // te_lo -> TempMin=1 eV, so the expected kappa is the law at the clamp edge.
  const Real te_lo = 0.3;                      // Planck peak ~1.2 eV << group-1 bound
  const Real te_hi = 1.0e4;                    // Planck peak ~2.8e4 eV: highest group
  const Real te_sweep[4] = {2.0, 20.0, 200.0, 2000.0};

  enum { IMEAN_A=0, IMEAN_B, IMEAN_LO, IMEAN_HI, ISW0, ISW1, ISW2, ISW3,
         IPMEAN_A, IPMEAN_B, IPMEAN_LO, IPMEAN_HI, NVAL };
  DvceArray1D<Real> d_vals("grey_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("grey_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    d_vals(IMEAN_A)  = radiationfld::GreyRosselandMean(table, rho_a, te_a, kboltz);
    d_vals(IMEAN_B)  = radiationfld::GreyRosselandMean(table, rho_b, te_a, kboltz);
    d_vals(IMEAN_LO) = radiationfld::GreyRosselandMean(table, rho_a, te_lo, kboltz);
    d_vals(IMEAN_HI) = radiationfld::GreyRosselandMean(table, rho_a, te_hi, kboltz);
    d_vals(ISW0) = radiationfld::GreyRosselandMean(table, rho_a, te_sweep[0], kboltz);
    d_vals(ISW1) = radiationfld::GreyRosselandMean(table, rho_a, te_sweep[1], kboltz);
    d_vals(ISW2) = radiationfld::GreyRosselandMean(table, rho_a, te_sweep[2], kboltz);
    d_vals(ISW3) = radiationfld::GreyRosselandMean(table, rho_a, te_sweep[3], kboltz);
    d_vals(IPMEAN_A)  = radiationfld::GreyPlanckMean(table, rho_a, te_a, kboltz);
    d_vals(IPMEAN_B)  = radiationfld::GreyPlanckMean(table, rho_b, te_a, kboltz);
    d_vals(IPMEAN_LO) = radiationfld::GreyPlanckMean(table, rho_a, te_lo, kboltz);
    d_vals(IPMEAN_HI) = radiationfld::GreyPlanckMean(table, rho_a, te_hi, kboltz);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // (1) independent-quadrature oracle at the interior point.
  test.CheckNear(h_vals(IMEAN_A), OracleGreyRosseland(bounds, 4, rho_a, te_a),
                 1.0e-6, 0.0, "grey Rosseland mean matches independent quadrature");

  // (2) bounded by the per-group extremes at the same (rho, te).
  test.CheckTrue(h_vals(IMEAN_A) >= Kro(0, rho_a, te_a) &&
                 h_vals(IMEAN_A) <= Kro(3, rho_a, te_a),
                 "grey mean lies between the min and max per-group kappa");

  // (3) spectrum-limit behavior: Planck peak below group 0 -> group-0 kappa; peak above
  //     the top group -> top-group kappa.
  test.CheckNear(h_vals(IMEAN_LO), Kro(0, rho_a, 1.0), 1.0e-9, 0.0,
                 "cold spectrum collapses to the lowest group's kappa (clamped te)");
  test.CheckNear(h_vals(IMEAN_HI), Kro(3, rho_a, te_hi), 5.0e-3, 0.0,
                 "hot spectrum collapses to the highest group's kappa");

  // (4) exact rho-scaling passthrough: weights are rho-independent, the fixture law is
  //     rho^0.6, and log-log bilinear interpolation is exact for power laws.
  test.CheckNear(h_vals(IMEAN_A)/h_vals(IMEAN_B), std::pow(10.0, 0.6), 1.0e-9, 0.0,
                 "grey mean inherits the fixture's exact rho^0.6 scaling");

  // (5) the group-weighting factor rises monotonically with te (spectrum sweeps up
  //     through groups of increasing kappa) and stays inside the per-group factor range
  //     [1.5, 6.0] of the fixture law.
  Real fac[4];
  for (int s = 0; s < 4; ++s) {
    fac[s] = h_vals(ISW0 + s)/(std::pow(rho_a, 0.6)*std::pow(te_sweep[s], -2.0));
    test.CheckTrue(fac[s] >= 1.5 && fac[s] <= 6.0,
                   "group-weighting factor within the fixture's per-group range");
  }
  test.CheckTrue(fac[0] < fac[1] && fac[1] < fac[2] && fac[2] < fac[3],
                 "group-weighting factor increases monotonically with te");

  // --- grey Planck (absorption) mean: B-weighted ARITHMETIC mean of PlanckAbsorption ---
  // (6) independent-quadrature oracle at the interior point.
  test.CheckNear(h_vals(IPMEAN_A), OracleGreyPlanck(bounds, 4, rho_a, te_a),
                 1.0e-6, 0.0, "grey Planck mean matches independent quadrature");
  // (7) bounded by the per-group extremes.
  test.CheckTrue(h_vals(IPMEAN_A) >= Kpa(0, rho_a, te_a) &&
                 h_vals(IPMEAN_A) <= Kpa(3, rho_a, te_a),
                 "grey Planck mean lies between the min and max per-group kappa");
  // (8) spectrum-limit collapse to the edge groups (te_lo clamps the lookup to
  //     TempMin=1; the hot end mixes ~1e-3 of group 2, hence the looser tolerance).
  test.CheckNear(h_vals(IPMEAN_LO), Kpa(0, rho_a, 1.0), 1.0e-9, 0.0,
                 "cold spectrum collapses to the lowest group's Planck kappa");
  test.CheckNear(h_vals(IPMEAN_HI), Kpa(3, rho_a, te_hi), 5.0e-3, 0.0,
                 "hot spectrum collapses to the highest group's Planck kappa");
  // (9) exact rho-scaling passthrough of the fixture's rho^0.5 Planck law.
  test.CheckNear(h_vals(IPMEAN_A)/h_vals(IPMEAN_B), std::pow(10.0, 0.5), 1.0e-9, 0.0,
                 "grey Planck mean inherits the fixture's exact rho^0.5 scaling");

  test.Finish();
}
