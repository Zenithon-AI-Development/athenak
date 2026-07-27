//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ionmix_cn4_units_test.cpp
//  \brief Unit test: the DIMENSIONAL anchor of the IONMIX cn4 EOS ingestion (issue
//  #209/#210).  The cn4 EOS blocks are JOULE-based (specific energies / heat capacities
//  in J/g, pressures in J/cm^3 -- the FLASH loadIonmix convention), while the code-unit
//  boundary (EosTable3T::ScaleToCodeUnits, ADR-0010) divides by CGS scales
//  (velocity_cgs^2, density_cgs*velocity_cgs^2).  The reader must therefore convert
//  J -> erg (x 1e7) at load; without it every tabulated_3t pressure/energy/cv is 1e7x
//  too small in code units and a magnetically driven liner is thermally pressureless at
//  ALL temperatures (the #209 resolution-divergent collapse).
//
//  No prior test pinned this scale: the ingestion-fidelity test compares reader output
//  against the RAW file records (units-agnostic by construction) and the representation
//  tests use synthetic power laws.  This test pins the ABSOLUTE dimensional scale twice
//  over at an exact table node (rho = 4.4804 g/cc, T = 1000 eV) of the committed
//  aluminum table:
//   (A) FIRST-PRINCIPLES: the tabulated ion pressure at the node is the ideal-gas
//       n_i k_B T to table precision (the cn4 ion channel IS ideal nkT there), computed
//       from CODATA constants and the B1 <units> -- an anchor independent of both the
//       reader and the file's unit convention.
//   (B) FILE ORACLE: pressures/energies/heat capacities at the node equal the decoded
//       raw file records converted J->erg->code (values hardcoded from an independent
//       python decode of the committed table).
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/ionmix_cn4_units_test -B build_unit
//  Auto-run by tst/test_suite/unit_tests/test_unit_ionmix_cn4_units_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief cn4 EOS dimensional-anchor unit test (#209/#210).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("ionmix_cn4_units_test");

  const std::string tbl_file = pin->GetString("problem", "eos_table");
  const Real mion  = pin->GetOrAddReal("problem", "eos_mass_per_ion", 4.4804e-23);
  const Real u_rho = pin->GetOrAddReal("units", "density_cgs", 2.7);
  const Real u_vel = pin->GetOrAddReal("units", "velocity_cgs", 1.0e8);

  eos_table_3t::EosTable3T eos;
  eos_table_3t::ReadIonmixCn4Eos(tbl_file, eos, mion);
  eos.ScaleToCodeUnits(u_rho, u_vel);

  // Exact table node (independently decoded): axis node 21 = 4.4804 g/cc after the
  // mass_per_ion relabel, temperature node 12 = 1000 eV exactly.
  const Real rho_node = 4.4804/2.7;      // = 1.6594074074 code
  const Real te_node  = 1000.0;

  enum { IPI=0, IPE, IEE, IEI, ICVE, NVAL };
  DvceArray1D<Real> d_vals("cn4units_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);
  par_for("cn4units", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    d_vals(IPI)  = eos.PressureIon(rho_node, te_node);
    d_vals(IPE)  = eos.PressureEle(rho_node, te_node);
    d_vals(IEE)  = eos.EnergyEle(rho_node, te_node);
    d_vals(IEI)  = eos.EnergyIon(rho_node, te_node);
    d_vals(ICVE) = eos.CvEle(rho_node, te_node);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // (A) first-principles anchor: p_ion(node) == n_i k_B T in code units.
  //     n_i = rho_cgs/m_ion; k_B T = 1000 eV = 1000*1.602176634e-12 erg.
  const Real n_i = 4.4804/4.4804e-23;                       // = 1e23 cm^-3
  const Real p_ion_phys = n_i*1000.0*1.602176634e-12;       // erg/cc
  const Real p_ion_code = p_ion_phys/(u_rho*u_vel*u_vel);
  test.CheckNear(h_vals(IPI), p_ion_code, 2.0e-4, 0.0,
                 "p_ion(4.48 g/cc, 1 keV) == ideal n_i k_B T from CODATA constants "
                 "(the absolute dimensional anchor; table records nkT to ~4 digits)");

  // (B) file oracle: raw cn4 records (J-based) x 1e7 -> erg -> code units.
  test.CheckNear(h_vals(IPI), 5.9333333333e-3, 1.0e-9, 0.0,
                 "p_ion == raw 1.6020e7 J/cc * 1e7 / (rho_u v_u^2)");
  test.CheckNear(h_vals(IPE), 7.2052962963e-2, 1.0e-9, 0.0,
                 "p_ele == raw 1.94543e8 J/cc * 1e7 / (rho_u v_u^2)");
  test.CheckNear(h_vals(IEE), 8.1743800000e-2, 1.0e-9, 0.0,
                 "e_ele == raw 8.17438e7 J/g * 1e7 / v_u^2");
  test.CheckNear(h_vals(IEI), 1.1331900000e-2, 1.0e-9, 0.0,
                 "e_ion == raw 1.13319e7 J/g * 1e7 / v_u^2");
  test.CheckNear(h_vals(ICVE), 7.1144200000e-5, 1.0e-9, 0.0,
                 "cv_ele == raw 7.11442e4 J/g/eV * 1e7 / v_u^2");

  test.Finish();
  std::exit(0);
}
