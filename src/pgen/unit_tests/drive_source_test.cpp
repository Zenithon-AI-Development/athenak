//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file drive_source_test.cpp
//! \brief Unit test for the circuit drive source + driven B_phi boundary formulas
//!  (circuit/drive_source.hpp, issue [9a]/#21, ADR-0005).
//!
//! Two parts.  (1) Host: the prescribed-I(t) drive source -- the analytic waveforms
//! (constant / linear_ramp / sin_squared) at known sample points, and a tabulated (t, I)
//! waveform read from a committed fixture (node values, piecewise-linear interpolation,
//! and end clamping).  (2) Device: the boundary formulas the user-BC writes into the
//! ghosts, exercised at radii from the same CellCenterX helper the real fill uses --
//!   * driven    B_phi == mu0*I/(2*pi*r), so r*B_phi is constant (the enclosed current);
//!   * nocurrent r*B_phi is preserved (d_r(r*B_phi)=0 -> B_phi ~ 1/r decay);
//!   * consistency: extrapolating the driven interior profile by nocurrent reproduces the
//!     driven 1/r profile exactly (the two boundary modes agree on a vacuum 1/r field).
//!
//! Built with -D PROBLEM=unit_tests/drive_source_test and run with the athinput
//! inputs/unit_tests/drive_source_test.athinput plus a problem/current_file=<abspath>
//! override pointing at the tabulated-waveform fixture.  Auto-run by
//! tst/test_suite/unit_tests/test_unit_drive_source_cpu.py.

#include <cmath>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "circuit/drive_source.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace cir = circuit;

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Drive-source + driven-B_phi-boundary unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("drive_source_test");

  // ======================================================================================
  // Part 1 -- host: prescribed I(t) waveforms.
  // ======================================================================================
  const Real i0 = 5.0;
  const Real t_rise = 4.0;

  // (a) constant
  cir::DriveSource ds_const;
  ds_const.waveform = cir::CurrentWaveform::constant;
  ds_const.i0 = i0;
  test.CheckNear(ds_const.Current(0.0), i0, 1.0e-15, 0.0, "constant I(0) == i0");
  test.CheckNear(ds_const.Current(7.3), i0, 1.0e-15, 0.0, "constant I(7.3) == i0");

  // (b) linear ramp 0 -> i0 over [0, t_rise], then held
  cir::DriveSource ds_ramp;
  ds_ramp.waveform = cir::CurrentWaveform::linear_ramp;
  ds_ramp.i0 = i0;
  ds_ramp.t_rise = t_rise;
  test.CheckNear(ds_ramp.Current(-1.0), 0.0, 1.0e-15, 0.0, "ramp I(t<0) == 0");
  test.CheckNear(ds_ramp.Current(0.0), 0.0, 1.0e-15, 0.0, "ramp I(0) == 0");
  test.CheckNear(ds_ramp.Current(0.5*t_rise), 0.5*i0, 1.0e-14, 0.0,
                 "ramp I(t_rise/2) == i0/2");
  test.CheckNear(ds_ramp.Current(t_rise), i0, 1.0e-15, 0.0, "ramp I(t_rise) == i0");
  test.CheckNear(ds_ramp.Current(2.0*t_rise), i0, 1.0e-15, 0.0,
                 "ramp held at i0 for t > t_rise");

  // (c) sin^2 pulse over [0, t_rise]: 0 at ends, peak i0 at t_rise/2
  cir::DriveSource ds_sin;
  ds_sin.waveform = cir::CurrentWaveform::sin_squared;
  ds_sin.i0 = i0;
  ds_sin.t_rise = t_rise;
  test.CheckNear(ds_sin.Current(0.0), 0.0, 1.0e-15, 0.0, "sin^2 I(0) == 0");
  test.CheckNear(ds_sin.Current(0.5*t_rise), i0, 1.0e-14, 0.0,
                 "sin^2 peak I(t_rise/2) == i0");
  test.CheckNear(ds_sin.Current(0.25*t_rise), 0.5*i0, 1.0e-14, 0.0,
                 "sin^2 I(t_rise/4) == i0/2 (sin^2(pi/4))");
  test.CheckNear(ds_sin.Current(t_rise), 0.0, 1.0e-15, 0.0, "sin^2 I(t_rise) == 0");

  // (d) tabulated waveform read from the committed fixture (linear segments)
  cir::DriveSource ds_tab;
  std::string wfile = pin->GetOrAddString("problem", "current_file", "unset");
  cir::ReadCurrentWaveform(wfile, ds_tab);
  test.CheckTrue(ds_tab.waveform == cir::CurrentWaveform::tabulated,
                 "ReadCurrentWaveform sets waveform == tabulated");
  // fixture nodes: (0,0) (1,10) (2,30) (4,30)
  test.CheckNear(ds_tab.Current(0.0), 0.0, 1.0e-14, 0.0, "tab node I(0) == 0");
  test.CheckNear(ds_tab.Current(1.0), 10.0, 1.0e-14, 0.0, "tab node I(1) == 10");
  test.CheckNear(ds_tab.Current(2.0), 30.0, 1.0e-14, 0.0, "tab node I(2) == 30");
  test.CheckNear(ds_tab.Current(4.0), 30.0, 1.0e-14, 0.0, "tab node I(4) == 30");
  test.CheckNear(ds_tab.Current(0.5), 5.0, 1.0e-14, 0.0,
                 "tab interp I(0.5) == 5 (linear (0,0)-(1,10))");
  test.CheckNear(ds_tab.Current(1.5), 20.0, 1.0e-14, 0.0,
                 "tab interp I(1.5) == 20 (linear (1,10)-(2,30))");
  test.CheckNear(ds_tab.Current(-1.0), 0.0, 1.0e-14, 0.0, "tab clamps below to I[0]");
  test.CheckNear(ds_tab.Current(99.0), 30.0, 1.0e-14, 0.0, "tab clamps above to I[last]");

  // (d') committed z2173 load-current trace (issue #160/[P3]): the faithful Be-liner
  //      drive tst/inputs/z2173_current.dat, read through the SAME ReadCurrentWaveform
  //      path the maglif pgen uses, must reproduce the PUBLISHED scalar characteristics
  //      (M. R. Gomez / R. D. McBride et al., PRL 109, 135004, 2012, Fig. 1(d), z2173):
  //      peak load current ~20 MA at a rise time ~100 ns.  File units: t [ns], I [MA].
  cir::DriveSource ds_z2173;
  std::string zfile = pin->GetOrAddString("problem", "z2173_file", "unset");
  cir::ReadCurrentWaveform(zfile, ds_z2173);
  test.CheckTrue(ds_z2173.waveform == cir::CurrentWaveform::tabulated,
                 "z2173 trace loads as a tabulated waveform");
  const int z_n = static_cast<int>(ds_z2173.i_tab.size());
  test.CheckTrue(z_n >= 2 && static_cast<int>(ds_z2173.t_tab.size()) == z_n,
                 "z2173 trace has >= 2 matched (t, I) samples");
  // peak current = max over the tabulated samples; rise time = its time (onset at t=0).
  Real z_peak = ds_z2173.i_tab[0];
  Real z_tpeak = ds_z2173.t_tab[0];
  for (int p = 0; p < z_n; ++p) {
    if (ds_z2173.i_tab[p] > z_peak) {
      z_peak = ds_z2173.i_tab[p];
      z_tpeak = ds_z2173.t_tab[p];
    }
  }
  // Published anchors with an ABSOLUTE tolerance band (rtol=0, atol set) reflecting the
  // issue's "~20 MA / ~100 ns" and the QC-pending detailed shape (peak +/- 2 MA; rise
  // +/- 15 ns).  CheckNear arg order is (actual, expected, rtol, atol, label).
  test.CheckNear(z_peak, 20.0, 0.0, 2.0, "z2173 peak current ~20 MA (McBride 2012)");
  test.CheckNear(z_tpeak, 100.0, 0.0, 15.0, "z2173 rise time ~100 ns (McBride 2012)");
  test.CheckTrue(ds_z2173.t_tab[0] <= 0.0 && ds_z2173.i_tab[0] <= 1.0e-6,
                 "z2173 trace starts at t=0 with ~zero current (current onset)");

  // ======================================================================================
  // Part 2 -- device: the driven / nocurrent B_phi boundary formulas, evaluated at the
  // outer-x1 ghost radii (cylindrical r) from the same CellCenterX helper the real
  // user-BC fill (circuit/drive_bphi_bc.hpp) uses.
  // ======================================================================================
  const int nx1   = 8;
  const Real rmin = 1.0, rmax = 2.0;     // r in [1,2]; ghosts lie at r > rmax
  const Real mu0  = 1.0;
  const Real mu0b = 2.0;                 // a second permeability to check the mu0 scaling
  const Real cur  = 7.0;                 // load current I

  // reference (last interior cell-centre) radius + a chosen interior B_phi for nocurrent
  const Real r_ref    = CellCenterX(nx1-1, nx1, rmin, rmax);
  const Real bphi_ref = 0.5;
  // two ghost cell-centre radii (i = 0,1 beyond the last interior cell)
  const Real r_g0 = CellCenterX(nx1+0, nx1, rmin, rmax);
  const Real r_g1 = CellCenterX(nx1+1, nx1, rmin, rmax);

  enum {
    IDRV0=0, IDRV1, IDRVB0,        // driven B_phi at ghost0/ghost1 (mu0), ghost0 (mu0b)
    INC0, INC1,                    // nocurrent B_phi at ghost0/ghost1
    ICONS0, ICONS1,                // nocurrent extrapolation of the driven interior
    NVAL
  };
  DvceArray1D<Real> d_vals("drive_vals", NVAL);
  auto h_vals = Kokkos::create_mirror_view(d_vals);

  par_for("drive_fill", DevExeSpace(), 0, 0, KOKKOS_LAMBDA(const int) {
    d_vals(IDRV0)  = cir::DrivenBphi(cur, r_g0, mu0);
    d_vals(IDRV1)  = cir::DrivenBphi(cur, r_g1, mu0);
    d_vals(IDRVB0) = cir::DrivenBphi(cur, r_g0, mu0b);
    d_vals(INC0)   = cir::NoCurrentBphi(bphi_ref, r_ref, r_g0);
    d_vals(INC1)   = cir::NoCurrentBphi(bphi_ref, r_ref, r_g1);
    // interior profile set by the driven BC just inside R_out, then extrapolated outward
    // by the nocurrent rule -- should reproduce the driven 1/r profile at the ghosts.
    Real bphi_in = cir::DrivenBphi(cur, r_ref, mu0);
    d_vals(ICONS0) = cir::NoCurrentBphi(bphi_in, r_ref, r_g0);
    d_vals(ICONS1) = cir::NoCurrentBphi(bphi_in, r_ref, r_g1);
  });
  Kokkos::deep_copy(h_vals, d_vals);

  // (e) driven: B_phi == mu0*I/(2*pi*r) exactly at each ghost radius.
  test.CheckNear(h_vals(IDRV0), mu0*cur/(cir::kTwoPi*r_g0), 1.0e-13, 0.0,
                 "driven B_phi(ghost0) == mu0*I/(2*pi*r)");
  test.CheckNear(h_vals(IDRV1), mu0*cur/(cir::kTwoPi*r_g1), 1.0e-13, 0.0,
                 "driven B_phi(ghost1) == mu0*I/(2*pi*r)");

  // (f) driven boundary => r*B_phi is constant (the enclosed current mu0*I/(2*pi)).
  test.CheckNear(r_g0*h_vals(IDRV0), mu0*cur/cir::kTwoPi, 1.0e-13, 0.0,
                 "driven r*B_phi(ghost0) == mu0*I/(2*pi) (enclosed current)");
  test.CheckNear(r_g1*h_vals(IDRV1), r_g0*h_vals(IDRV0), 1.0e-13, 0.0,
                 "driven r*B_phi same at both ghosts (uniform enclosed current)");

  // (g) mu0 scaling: doubling mu0 doubles B_phi.
  test.CheckNear(h_vals(IDRVB0), 2.0*h_vals(IDRV0), 1.0e-13, 0.0,
                 "driven B_phi scales linearly with mu0");

  // (h) nocurrent: r*B_phi preserved across the boundary (d_r(r*B_phi)=0 -> 1/r decay).
  test.CheckNear(r_g0*h_vals(INC0), r_ref*bphi_ref, 1.0e-13, 0.0,
                 "nocurrent r*B_phi(ghost0) == r_ref*B_phi_ref");
  test.CheckNear(r_g1*h_vals(INC1), r_ref*bphi_ref, 1.0e-13, 0.0,
                 "nocurrent r*B_phi(ghost1) == r_ref*B_phi_ref");
  // explicit 1/r form
  test.CheckNear(h_vals(INC0), bphi_ref*r_ref/r_g0, 1.0e-13, 0.0,
                 "nocurrent B_phi(ghost0) == B_phi_ref*r_ref/r");
  test.CheckTrue(h_vals(INC1) < h_vals(INC0),
                 "nocurrent B_phi decreases outward (1/r decay)");

  // (i) consistency: nocurrent extrapolation of the driven interior profile reproduces
  //     the driven 1/r profile exactly (both modes agree on a vacuum 1/r field).
  test.CheckNear(h_vals(ICONS0), h_vals(IDRV0), 1.0e-13, 0.0,
                 "nocurrent(driven interior) == driven B_phi at ghost0");
  test.CheckNear(h_vals(ICONS1), h_vals(IDRV1), 1.0e-13, 0.0,
                 "nocurrent(driven interior) == driven B_phi at ghost1");

  test.Finish();
  return;
}
