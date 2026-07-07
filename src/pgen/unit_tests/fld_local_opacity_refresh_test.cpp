//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file fld_local_opacity_refresh_test.cpp
//  \brief Unit test: the per-cell grey extinction refresh chi(rho,Te) (issue #204,
//  radiationfld::RefreshGreyChiField).  The refresh reads the LIVE conserved state
//  (rho = u0(IDN), e_ele = the electron-energy scalar), inverts Te through the SAME
//  tabulated_3t closure ConsToPrim2T uses, looks up the grey Rosseland mean of the real
//  aluminum IONMIX multigroup opacity at the LOCAL (rho,Te), and writes the code-unit
//  extinction chi = kappa_R(rho,Te) * rho_cgs * L into the per-cell field the grey FLD
//  operator diffuses with (EnableLocalChi).  This is what replaces the frozen
//  solid-density constant that made the B1 vacuum gap optically thick (#204).
//
//  Uses the REAL committed aluminum table (inputs/ionmix/al-imx-004.cn4) for BOTH the
//  EOS closure and the opacity, at the B1 deck's <units> (L=0.1 cm, rho_u=2.7 g/cc).
//
//  Batteries:
//   (A) TRANSPARENT GAP (the #204 acceptance property, on real data): a solid-density
//       liner cell (rho=1 code = 2.7 g/cc) gets an extinction >= 1e3 x the tenuous
//       vacuum-gap cell (rho=1e-4 code) at comparable temperature -- the local rho
//       factor alone guarantees the gap is transparent even where the table clamps.
//   (B) PLUMBING: each refreshed chi equals the same public-lookup chain evaluated
//       independently (Te closure -> grey Rosseland mean -> OpacityCodeFromCgs).
//   (C) LIVENESS: heating a cell (raising e_ele) changes its refreshed chi -- the field
//       tracks the evolving state rather than a frozen reference.
//   (D) FLOOR: chi is floored at the caller's chi_floor (positivity for the operator's
//       1/chi), and every refreshed value (ghosts included) is >= the floor.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/fld_local_opacity_refresh_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_fld_local_opacity_refresh_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "opacity/multigroup_opacity.hpp"
#include "opacity/ionmix_opacity_reader.hpp"
#include "diffusion/operator_si_calibration.hpp"
#include "radiation_fld/local_grey_opacity.hpp"
#include "radiation_fld/grey_opacity_mean.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
constexpr int kNmhd = 5;   // electron-energy scalar rides index nmhd (= IEN+1)
}

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Per-cell grey chi(rho,Te) refresh unit test (#204).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("fld_local_opacity_refresh_test");

  // --- the real aluminum IONMIX table, read for BOTH closures, B1 <units> ---
  const std::string tbl_file = pin->GetString("problem", "eos_table");
  const Real mass_per_ion = pin->GetOrAddReal("problem", "eos_mass_per_ion", 4.4804e-23);
  const Real u_len = pin->GetOrAddReal("units", "length_cgs", 0.1);
  const Real u_rho = pin->GetOrAddReal("units", "density_cgs", 2.7);
  const Real u_vel = pin->GetOrAddReal("units", "velocity_cgs", 1.0e8);

  eos_table_3t::EosTable3T eos;
  eos_table_3t::ReadIonmixCn4Eos(tbl_file, eos, mass_per_ion);
  eos.ScaleToCodeUnits(u_rho, u_vel);          // exactly what mhd.cpp does at load

  opacity::MultigroupOpacity opac;
  // mass_per_ion rescales the cn4 ion-number-density axis to g/cc (as the EOS reader
  // does), so lookups take mass density in g/cc.
  opacity::ReadIonmixCn4Opacity(tbl_file, opac, mass_per_ion);

  // --- a 4-cell conserved mini-state: [liner, vacuum-gap, heated-vacuum, near-empty] ---
  const int NC = 4;
  const Real rho_c[NC]  = {1.0, 1.0e-4, 1.0e-4, 1.0e-12};   // code densities
  const Real te_c[NC]   = {10.0, 10.0, 1000.0, 10.0};       // target Te [eV]
  const Real chi_floor  = 1.0e-8;

  DvceArray5D<Real> u0("u0", 1, kNmhd+1, 1, 1, NC);
  auto hu = Kokkos::create_mirror_view(u0);
  Kokkos::deep_copy(hu, 0.0);
  // host-side EnergyEle through host mirrors is unavailable (device Views), so fill
  // e_ele on the device from the table's own closure at the target (rho, Te).
  DvceArray1D<Real> d_rho("rho", NC), d_te("te", NC);
  auto h_rho = Kokkos::create_mirror_view(d_rho);
  auto h_te = Kokkos::create_mirror_view(d_te);
  for (int i = 0; i < NC; ++i) { h_rho(i) = rho_c[i]; h_te(i) = te_c[i]; }
  Kokkos::deep_copy(d_rho, h_rho);
  Kokkos::deep_copy(d_te, h_te);
  par_for("fill_u0", DevExeSpace(), 0, NC-1, KOKKOS_LAMBDA(const int i) {
    u0(0, IDN, 0, 0, i) = d_rho(i);
    u0(0, kNmhd, 0, 0, i) = d_rho(i)*eos.EnergyEle(d_rho(i), d_te(i));
  });

  // --- the refresh under test ---
  DvceArray5D<Real> chi_cell("chi_cell", 1, 1, 1, 1, NC);
  radiationfld::RefreshGreyChiField(u0, kNmhd, eos, opac, u_rho, u_len, chi_floor,
                                    chi_cell);

  auto h_chi = Kokkos::create_mirror_view(chi_cell);
  Kokkos::deep_copy(h_chi, chi_cell);

  // (B) plumbing oracle: the same public-lookup chain, evaluated independently per cell.
  DvceArray1D<Real> d_exp("exp", NC);
  par_for("oracle", DevExeSpace(), 0, NC-1, KOKKOS_LAMBDA(const int i) {
    Real rho = u0(0, IDN, 0, 0, i);
    Real te = eos.Te(rho, u0(0, kNmhd, 0, 0, i)/rho);
    Real rho_gcc = rho*u_rho;
    Real kap = radiationfld::GreyRosselandMean(opac, rho_gcc, te, 1.0);
    Real chi = op_si_calib::OpacityCodeFromCgs(kap, rho_gcc, u_len);
    d_exp(i) = (chi > chi_floor) ? chi : chi_floor;
  });
  auto h_exp = Kokkos::create_mirror_view(d_exp);
  Kokkos::deep_copy(h_exp, d_exp);
  for (int i = 0; i < NC; ++i) {
    test.CheckNear(h_chi(0, 0, 0, 0, i), h_exp(i), 1.0e-12, 0.0,
                   "refreshed chi equals the public-lookup chain (cell "
                   + std::to_string(i) + ")");
  }

  // (A) the #204 acceptance property on real data: opaque liner, transparent gap.
  test.CheckTrue(h_chi(0, 0, 0, 0, 0) >= 1.0e3*h_chi(0, 0, 0, 0, 1),
                 "liner chi >= 1e3 x vacuum-gap chi (transparent gap)");
  // sanity anchor: liner-cell chi within a broad physical band around the frozen
  // reference value the deck used until #204 (fld_chi ~ 1447 from kappa_ross 5358).
  test.CheckTrue(h_chi(0, 0, 0, 0, 0) > 1.0e1 && h_chi(0, 0, 0, 0, 0) < 1.0e6,
                 "liner chi in a physically-plausible band");

  // (C) liveness: the heated vacuum cell's chi differs from the cold vacuum cell's.
  test.CheckTrue(h_chi(0, 0, 0, 0, 2) != h_chi(0, 0, 0, 0, 1),
                 "chi responds to the local temperature (heated vs cold gap)");

  // (D) floor: the near-empty cell floors; every value >= floor.
  bool floored_ok = true;
  for (int i = 0; i < NC; ++i) {
    if (!(h_chi(0, 0, 0, 0, i) >= chi_floor)) { floored_ok = false; }
  }
  test.CheckTrue(floored_ok, "all refreshed chi >= chi_floor");
  test.CheckNear(h_chi(0, 0, 0, 0, 3), chi_floor, 1.0e-14, 0.0,
                 "near-empty cell collapses to the chi floor");

  test.Finish();
}
