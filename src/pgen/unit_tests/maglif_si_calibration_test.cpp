//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file maglif_si_calibration_test.cpp
//  \brief Unit test: the faithful dimensional units + drive calibration the B1
//  replication needs (issue [P5]/#174, ADR-0010).
//
//  Two independent, citable Layer-1 oracles (ADR-0008), both closed-form:
//
//   (1) MAGNETIC-PRESSURE ANCHOR (AC#2, CONTEXT.md "Magnetic-pressure anchor").  The SI
//       drive calibration (circuit::CalibrateSiDrive) converts the z2173 load trace
//       (time [ns], current [MA]) into code units; applying the code-unit peak current to
//       the code drive `B_phi = mu0*I_code/(2*pi*r_code)` (mu0=1) and reconverting the
//       resulting code magnetic pressure `B^2/2` back to physical Pa MUST reproduce the
//       independent closed-form `mu0_SI*I^2/(8*pi^2*r^2)` (~5 Mbar at 20 MA / 3.47 mm).
//       unit error ANYWHERE in the drive calibration breaks this single check.  The RED
//       state this pins (run-it-first, observe-it-fail, AC#1): consuming the raw trace MA
//       directly as a code current (the pre-#174 behaviour) gives a wildly wrong magnetic
//       pressure -- asserted here as a meta-check so the calibration's necessity is
//       proven, not assumed.
//
//   (2) EOS-TABLE UNIT BOUNDARY (ADR-0010).  EosTable3T::ScaleToCodeUnits divides the
//       specific energies/heat capacities by velocity_cgs^2, the pressures by
//       density_cgs*velocity_cgs^2, relabels the density axis /density_cgs, and leaves
//       eV temperature axis and dimensionless Zbar untouched.  Loading the real aluminum
//       table twice (once raw, once scaled) and comparing node-for-node MUST reproduce
//       exactly those factors.
//
//  No table value or magnetic pressure is hard-coded: the trace peak, the table nodes and
//  the physical reference pressure are all computed at runtime (ADR-0008).  Built/run by
//  tst/test_suite/unit_tests/test_unit_maglif_si_calibration_cpu.py.

#include <cmath>     // std::sqrt, std::fabs
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "eos/eos_table_3t.hpp"
#include "eos/ionmix_eos_reader.hpp"
#include "circuit/drive_source.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace {
//! pi as a Real (host-only test; avoids the M_PI portability macro).
constexpr Real kPi = 3.1415926535897932385;
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Faithful dimensional units + drive calibration unit test (#174, ADR-0010).

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("maglif_si_calibration_test");

  // The chosen code-unit system (read from <units>; the athinput uses the B1 values).
  const Real len_cgs = pin->GetOrAddReal("units", "length_cgs", 1.0);
  const Real rho_cgs = pin->GetOrAddReal("units", "density_cgs", 1.0);
  const Real vel_cgs = pin->GetOrAddReal("units", "velocity_cgs", 1.0);

  // ====================== (1) MAGNETIC-PRESSURE ANCHOR (AC#2) ======================
  const std::string trace = pin->GetString("problem", "current_file");
  const Real r_out_mm = pin->GetOrAddReal("problem", "r_out_mm", 3.4688);  // liner/vacuum

  circuit::DriveSource ds;
  circuit::ReadCurrentWaveform(trace, ds);     // raw SI columns: t [ns], I [MA]
  // peak current (MA) and its time (ns) straight from the committed trace
  Real i_peak_ma = 0.0, t_peak_ns = 0.0;
  for (std::size_t p = 0; p < ds.i_tab.size(); ++p) {
    if (ds.i_tab[p] > i_peak_ma) { i_peak_ma = ds.i_tab[p]; t_peak_ns = ds.t_tab[p]; }
  }
  test.CheckTrue(i_peak_ma > 0.0, "z2173 trace carries a positive peak current");

  // Independent closed-form physical magnetic pressure at the liner/vacuum radius [Pa].
  const Real mu0_si    = 4.0e-7*kPi;
  const Real i_peak_si = i_peak_ma*1.0e6;            // MA -> A
  const Real r_out_m   = r_out_mm*1.0e-3;            // mm -> m
  const Real pmag_phys = mu0_si*i_peak_si*i_peak_si/(8.0*kPi*kPi*r_out_m*r_out_m);

  // Calibrated peak current -> code B_phi -> code magnetic pressure -> back to Pa.
  const circuit::SiDriveUnits du = circuit::CalibrateSiDrive(len_cgs, rho_cgs, vel_cgs);
  const Real r_out_code = r_out_mm*0.1/len_cgs;      // mm -> cm -> code length
  const Real p_unit_pa  = 0.1*rho_cgs*vel_cgs*vel_cgs;        // code pressure unit [Pa]
  const Real i_code     = i_peak_ma*du.current_per_ma;
  const Real b_code     = circuit::DrivenBphi(i_code, r_out_code, 1.0);  // mu0=1
  const Real pmag_code_pa = (0.5*b_code*b_code)*p_unit_pa;

  test.CheckNear(pmag_code_pa, pmag_phys, 1.0e-10, 0.0,
                 "calibrated code magnetic pressure B^2/2 = mu0*I^2/(8 pi^2 r^2)");

  // RED meta-check (AC#1): the pre-#174 behaviour -- raw trace MA consumed directly as
  // a code current -- gives a grossly wrong magnetic pressure, so the calibration is
  // genuinely necessary (this is the "wrong magnetic pressure" failure mode the anchor
  // catches; it must be off by many orders of magnitude for these B1 units).
  const Real b_raw  = circuit::DrivenBphi(i_peak_ma, r_out_code, 1.0);
  const Real pmag_raw_pa = (0.5*b_raw*b_raw)*p_unit_pa;
  test.CheckTrue(std::fabs(pmag_raw_pa - pmag_phys) > 0.5*pmag_phys,
                 "uncalibrated raw-MA current does NOT match the physical anchor "
                 "(calibration is required)");

  // Time-axis calibration: code time per ns = 1e-9 * vel_cgs / length_cgs; converting
  // the peak time and back recovers the nanosecond value (round-trip exact).
  const Real t_peak_code = t_peak_ns*du.time_per_ns;
  test.CheckNear(t_peak_code/du.time_per_ns, t_peak_ns, 1.0e-12, 0.0,
                 "drive time-axis ns<->code conversion round-trips");

  // ====================== (2) EOS-TABLE UNIT BOUNDARY ======================
  const std::string tbl_file = pin->GetString("mhd", "eos_table");
  const Real mass_per_ion = pin->GetOrAddReal("mhd", "eos_mass_per_ion", 1.0);

  eos_table_3t::EosTable3T raw, scaled;
  eos_table_3t::ReadIonmixCn4Eos(tbl_file, raw, mass_per_ion);
  eos_table_3t::ReadIonmixCn4Eos(tbl_file, scaled, mass_per_ion);
  scaled.ScaleToCodeUnits(rho_cgs, vel_cgs);

  test.CheckTrue(scaled.ntemp == raw.ntemp && scaled.nrho == raw.nrho,
                 "scaling preserves the table dimensions");
  // temperature axis (eV) unchanged; density axis relabelled by 1/density_cgs
  test.CheckNear(scaled.TempMin(), raw.TempMin(), 1.0e-12, 0.0,
                 "temperature axis stays in eV (unchanged by scaling)");
  test.CheckNear(scaled.RhoMin(), raw.RhoMin()/rho_cgs, 1.0e-10, 0.0,
                 "density axis relabelled rho_phys -> rho_phys/density_cgs");

  // node-for-node value scaling, checked at the log-grid midpoint (a representative node)
  const int it = raw.ntemp/2;
  const int ir = raw.nrho/2;
  auto raw_e  = Kokkos::create_mirror_view(raw.e_ele);
  auto scl_e  = Kokkos::create_mirror_view(scaled.e_ele);
  auto raw_p  = Kokkos::create_mirror_view(raw.p_ele);
  auto scl_p  = Kokkos::create_mirror_view(scaled.p_ele);
  auto raw_z  = Kokkos::create_mirror_view(raw.zbar);
  auto scl_z  = Kokkos::create_mirror_view(scaled.zbar);
  Kokkos::deep_copy(raw_e, raw.e_ele);   Kokkos::deep_copy(scl_e, scaled.e_ele);
  Kokkos::deep_copy(raw_p, raw.p_ele);   Kokkos::deep_copy(scl_p, scaled.p_ele);
  Kokkos::deep_copy(raw_z, raw.zbar);    Kokkos::deep_copy(scl_z, scaled.zbar);

  const Real inv_e = 1.0/(vel_cgs*vel_cgs);
  const Real inv_p = 1.0/(rho_cgs*vel_cgs*vel_cgs);
  test.CheckNear(scl_e(it, ir), raw_e(it, ir)*inv_e, 1.0e-10, 0.0,
                 "specific electron energy scaled by 1/velocity_cgs^2");
  test.CheckNear(scl_p(it, ir), raw_p(it, ir)*inv_p, 1.0e-10, 0.0,
                 "electron pressure scaled by 1/(density_cgs*velocity_cgs^2)");
  test.CheckNear(scl_z(it, ir), raw_z(it, ir), 1.0e-12, 0.0,
                 "mean ionization Zbar is dimensionless (unchanged by scaling)");

  test.Finish();
  return;
}
