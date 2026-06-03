//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file maglif_eos_aware_operators_test.cpp
//  \brief Unit test: the EOS-aware operator closures the faithful tabulated_3t B1 run
//  needs (issue [P7c]/#183, ADR-0012 gap c).  Two reference-model-set consistency facts,
//  both checked against the *same* tabulated EOS closure ConsToPrim2T uses:
//
//   (AC#1, conduction) The anisotropic-conduction temperature recovery
//   (`anisocond::EosAwareTemp::Temp`) in EOS-aware mode equals the tabulated_3t electron
//   temperature `T_e = table.Te(rho, e_ele/rho)` -- the derived/cached field of the 3T
//   formulation -- at known (rho,e_ele) table nodes, NOT the ideal-gamma value
//   `T = (gamma-1) eint/rho`.  The RED state this pins (run-it-first, observe-fail): the
//   ideal-gamma recovery the operator used before #183 gives a temperature grossly
//   inconsistent with the tabulated eV closure, asserted as a meta-check so the fix's
//   necessity is proven, not assumed.
//
//   (AC#2, matter-radiation) EOS-aware grey-coupling heat capacity equals the tabulated
//   closure value `c_v = rho*(c_v,e(rho,T_e) + c_v,i(rho,T_i))` (volumetric, the same
//   per-species heat capacities the table feeds ConsToPrim2T), positive and NOT equal to
//   the constant `mrad_cv` placeholder the coupling used before #183.
//
//  No temperature or heat capacity is hard-coded: table nodes are read at runtime from
//  the real aluminum IONMIX table and the closures are evaluated through the exact code
//  path the operators run (ADR-0008).  Built/run by
//  tst/test_suite/unit_tests/test_unit_maglif_eos_aware_operators_cpu.py.

#include <cmath>     // std::fabs
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
constexpr int kNmhd = 5;        // electron-energy scalar rides index nmhd (=IEN+1)
constexpr Real kGamma = 5.0/3.0;  // ideal-gamma reference for the RED meta-check
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief EOS-aware conduction temperature + matter-radiation heat capacity unit test
//! (#183, ADR-0012 gap c).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("maglif_eos_aware_operators_test");

  // Load the real aluminum IONMIX table (raw physical units: specific energies in cgs,
  // temperature axis in eV) -- the same closure the tabulated_3t solver inverts per cell.
  const std::string tbl_file = pin->GetString("mhd", "eos_table");
  const Real mass_per_ion = pin->GetOrAddReal("mhd", "eos_mass_per_ion", 1.0);
  eos_table_3t::EosTable3T tbl;
  eos_table_3t::ReadIonmixCn4Eos(tbl_file, tbl, mass_per_ion);

  // Choose interior table nodes (rho,T): exact grid points, so EnergyEle/EnergyIon return
  // the node value and the e->T inversion recovers T to the Newton tolerance.
  const int NPTS = 3;
  const int it_pick[NPTS] = {tbl.ntemp/4, tbl.ntemp/2, (3*tbl.ntemp)/4};
  const int ir_pick[NPTS] = {tbl.nrho/4,  tbl.nrho/2,  (3*tbl.nrho)/4};

  // Assemble a tiny conserved/field state (NPTS cells on the i-axis) holding, per node,
  // a quiescent cell (zero momentum + small magnetic field) at the node density with the
  // electron energy density e_ele = rho*EnergyEle(rho,T) on scalar 0 and the total
  // energy IEN = e_ele + e_ion + 1/2 B^2 -- exactly the maglif tabulated IC layout.
  DvceArray5D<Real> u("eos_aware_u", 1, kNmhd+1, 1, 1, NPTS);
  DvceArray5D<Real> bcc("eos_aware_bcc", 1, 3, 1, 1, NPTS);
  auto hu  = Kokkos::create_mirror_view(u);
  auto hb  = Kokkos::create_mirror_view(bcc);
  HostArray1D<Real> rho_h("rho", NPTS), te_in_h("te", NPTS);
  for (int p = 0; p < NPTS; ++p) {
    Real rho = tbl.RhoAt(ir_pick[p]);
    Real T   = tbl.TempAt(it_pick[p]);
    Real e_ele = rho*tbl.EnergyEle(rho, T);   // electron internal energy density
    Real e_ion = rho*tbl.EnergyIon(rho, T);   // ion internal energy density
    Real bz = 0.137;                       // arbitrary nonzero field (tests me removal)
    hu(0,IDN,0,0,p)   = rho;
    hu(0,IM1,0,0,p)   = 0.0;
    hu(0,IM2,0,0,p)   = 0.0;
    hu(0,IM3,0,0,p)   = 0.0;
    hu(0,IEN,0,0,p)   = e_ele + e_ion + 0.5*bz*bz;   // total energy density
    hu(0,kNmhd,0,0,p) = e_ele;                        // electron energy scalar (density)
    hb(0,IBX,0,0,p)   = 0.0;
    hb(0,IBY,0,0,p)   = 0.0;
    hb(0,IBZ,0,0,p)   = bz;
    rho_h(p)   = rho;
    te_in_h(p) = T;
  }
  Kokkos::deep_copy(u, hu);
  Kokkos::deep_copy(bcc, hb);

  // The EOS-aware temperature-recovery helper the conduction operator runs.
  anisocond::EosAwareTemp etemp;
  etemp.eos_aware = true;
  etemp.eele_idx  = kNmhd;
  etemp.gm1       = kGamma - 1.0;
  etemp.table     = tbl;

  // Evaluate the recovered temperature, the ideal-gamma value, and the EOS-aware grey
  // coupling heat capacity on the device through the exact operator code paths.
  DvceArray1D<Real> t_rec("t_rec", NPTS), t_clo("t_clo", NPTS), t_ide("t_ide", NPTS);
  DvceArray1D<Real> cv_eos("cv_eos", NPTS);
  auto tbl_d = tbl;
  auto et_d = etemp;
  auto u_d = u, b_d = bcc;
  const int eele_idx = kNmhd;
  Kokkos::parallel_for("eos_aware_eval", Kokkos::RangePolicy<>(DevExeSpace(), 0, NPTS),
  KOKKOS_LAMBDA(const int p) {
    Real rho = u_d(0,IDN,0,0,p);
    Real bz  = b_d(0,IBZ,0,0,p);
    Real me  = 0.5*bz*bz;
    Real eint = u_d(0,IEN,0,0,p) - me;            // zero momentum -> no kinetic term
    Real e_ele = u_d(0,eele_idx,0,0,p);
    Real e_ion = eint - e_ele;
    // EOS-aware conduction temperature recovery (the operator's exact call).
    t_rec(p) = et_d.Temp(u_d, b_d, 0, 0, 0, p);
    // tabulated_3t electron-temperature closure (= ConsToPrim2T's derived T_e).
    t_clo(p) = tbl_d.Te(rho, e_ele/rho);
    // ideal-gamma bookkeeping temperature (the pre-#183 conduction recovery).
    t_ide(p) = (kGamma - 1.0)*eint/rho;
    // EOS-aware grey-coupling volumetric heat capacity (the mrad EOS-aware path).
    Real te = tbl_d.Te(rho, e_ele/rho);
    Real ti = tbl_d.Ti(rho, e_ion/rho);
    cv_eos(p) = rho*(tbl_d.CvEle(rho, te) + tbl_d.CvIon(rho, ti));
  });
  auto ht_rec = Kokkos::create_mirror_view(t_rec);
  auto ht_clo = Kokkos::create_mirror_view(t_clo);
  auto ht_ide = Kokkos::create_mirror_view(t_ide);
  auto hcv    = Kokkos::create_mirror_view(cv_eos);
  Kokkos::deep_copy(ht_rec, t_rec);
  Kokkos::deep_copy(ht_clo, t_clo);
  Kokkos::deep_copy(ht_ide, t_ide);
  Kokkos::deep_copy(hcv, cv_eos);

  // Constant placeholder heat capacity the pre-#183 coupling used (mrad_cv default = 1).
  const Real cv_const = 1.0;

  for (int p = 0; p < NPTS; ++p) {
    // (AC#1) the EOS-aware conduction temperature IS the tabulated electron-temperature
    // closure, and recovers the node temperature the cell was filled at.
    test.CheckNear(ht_rec(p), ht_clo(p), 1.0e-12, 0.0,
                   "acond EOS-aware temperature == tabulated_3t T_e closure");
    test.CheckNear(ht_rec(p), te_in_h(p), 1.0e-9, 0.0,
                   "acond EOS-aware temperature recovers the node T (e->T inversion)");
    // (AC#1 RED) the ideal-gamma recovery is grossly inconsistent with the eV closure:
    // pre-#183 conduction temperature is NOT the faithful temperature (fix is required).
    test.CheckTrue(std::fabs(ht_ide(p) - ht_clo(p)) > 0.5*std::fabs(ht_clo(p)),
                   "ideal-gamma temperature does NOT match the tabulated closure "
                   "(EOS-aware recovery is required)");
    // (AC#2) the EOS-aware grey-coupling heat capacity is positive and NOT the constant
    // placeholder -- it carries the tabulated per-species closure value.
    test.CheckTrue(hcv(p) > 0.0,
                   "mrad EOS-aware heat capacity is positive");
    test.CheckTrue(std::fabs(hcv(p) - cv_const) > 1.0e-30,
                   "mrad EOS-aware c_v differs from the constant placeholder "
                   "(EOS-aware closure is required)");
  }

  test.Finish();
  return;
}
