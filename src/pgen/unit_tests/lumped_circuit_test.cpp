//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lumped_circuit_test.cpp
//! \brief Unit test for the coupled lumped-element circuit ODE
//!  (circuit/lumped_circuit.hpp, issue [18b]/#34, ADR-0005 modes B/C).
//!
//! Host-side test of the once-per-step RK4 circuit advance.  The discriminating oracle is
//! the textbook series-RLC step response: with a constant open-circuit voltage and
//! constant series resistance R = Z0 + R_loss and zero initial conditions, mode B
//! reproduces the analytic underdamped / overdamped current (and capacitor voltage), and
//! the RL limit (no capacitor) reproduces I = (V0/R)(1-exp(-Rt/L)).  Mode C is checked by
//! a constant Faraday load voltage V_load, which simply lowers the effective source to
//! V_oc-V_load (so the same analytic RLC with the reduced source), while mode B is shown
//! to IGNORE the same V_load (fixed RLC).  Finally the integrated current is shown to
//! feed the driven B_phi boundary via BoundaryBphi == mu0*I/(2*pi*r).
//!
//! Built with -D PROBLEM=unit_tests/lumped_circuit_test and run with
//! inputs/unit_tests/lumped_circuit_test.athinput (nlim = tlim = 0; a tiny <z4c> block
//! satisfies AddPhysics -- the circuit is pure host scalars needing no matter/EOS/MHD).
//! Auto-run by tst/test_suite/unit_tests/test_unit_lumped_circuit_cpu.py.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "circuit/drive_source.hpp"
#include "circuit/lumped_circuit.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

namespace cir = circuit;

namespace {
// Integrate a copy of `proto` from t=0 with `nstep` RK4 steps of size T/nstep, feeding
// the constant load voltage `v_load` each step (mode B ignores it); return final state.
cir::LumpedCircuit IntegrateTo(cir::LumpedCircuit proto, Real T, int nstep, Real v_load) {
  proto.Reset(0.0, 0.0);
  Real dt = T/static_cast<Real>(nstep);
  for (int n = 0; n < nstep; ++n) {
    proto.Step(dt, n*dt, v_load);
  }
  return proto;
}

// Analytic series-RLC step response: constant source V0, zero initial conditions.
Real IunderAnalytic(Real V0, Real L, Real R, Real Cp, Real t) {
  Real a = R/(2.0*L);
  Real w0 = 1.0/std::sqrt(L*Cp);
  Real wd = std::sqrt(w0*w0 - a*a);
  return (V0/(L*wd))*std::exp(-a*t)*std::sin(wd*t);
}
Real VcunderAnalytic(Real V0, Real L, Real R, Real Cp, Real t) {
  Real a = R/(2.0*L);
  Real w0 = 1.0/std::sqrt(L*Cp);
  Real wd = std::sqrt(w0*w0 - a*a);
  return V0*(1.0 - std::exp(-a*t)*(std::cos(wd*t) + (a/wd)*std::sin(wd*t)));
}
Real IoverAnalytic(Real V0, Real L, Real R, Real Cp, Real t) {
  Real a = R/(2.0*L);
  Real w0 = 1.0/std::sqrt(L*Cp);
  Real b = std::sqrt(a*a - w0*w0);
  return (V0/(L*b))*std::exp(-a*t)*std::sinh(b*t);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Lumped-element circuit ODE unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  (void)pin;  // no input parameters needed; the circuit is configured in-test
  unit_test::UnitTest test("lumped_circuit_test");

  const Real rtol = 1.0e-6;            // RK4 with dt=1e-3 is well below this
  const int steps_per_time = 1000;     // dt = 1/steps_per_time

  // ======================================================================================
  // Part 1 -- mode B, UNDERDAMPED series RLC step response (the analytic oracle).
  //   L=2, C=0.5 => w0 = 1/sqrt(LC) = 1; R = Z0 = 0.4 => a = R/2L = 0.1 < w0 (underdamp).
  // ======================================================================================
  const Real V0u = 3.0, Lu = 2.0, Cu = 0.5, Ru = 0.4;
  cir::LumpedCircuit cb;
  cb.mode = cir::DriveMode::voltage_rlc;
  cb.voltage_source.waveform = cir::CurrentWaveform::constant;
  cb.voltage_source.i0 = V0u;
  cb.L = Lu;
  cb.cap = Cu;
  cb.Z0 = Ru;
  cb.r_loss = 0.0;

  const Real tsamp[3] = {1.5, 3.0, 5.0};
  for (int s = 0; s < 3; ++s) {
    Real T = tsamp[s];
    int N = static_cast<int>(T*steps_per_time);
    cir::LumpedCircuit c = IntegrateTo(cb, T, N, 0.0);
    test.CheckNear(c.Current(), IunderAnalytic(V0u, Lu, Ru, Cu, T), rtol, 1.0e-10,
                   "mode B underdamped I(t) matches analytic RLC");
    test.CheckNear(c.CapVoltage(), VcunderAnalytic(V0u, Lu, Ru, Cu, T), rtol, 1.0e-10,
                   "mode B underdamped V_C(t) matches analytic RLC");
  }
  // I(0)=0, V_C(0)=0 (zero initial conditions): a freshly reset circuit starts at rest.
  cb.Reset(0.0, 0.0);
  test.CheckNear(cb.Current(), 0.0, 1.0e-15, 1.0e-15, "I(0) == 0 (zero IC)");
  test.CheckNear(cb.CapVoltage(), 0.0, 1.0e-15, 1.0e-15, "V_C(0) == 0 (zero IC)");

  // ======================================================================================
  // Part 2 -- mode B, OVERDAMPED series RLC step response.
  //   L=1, C=0.25 => w0 = 2; R = Z0 = 5 => a = 2.5 > w0 (overdamped); b = sqrt(a^2-w0^2).
  // ======================================================================================
  const Real V0o = 4.0, Lo = 1.0, Co = 0.25, Ro = 5.0;
  cir::LumpedCircuit co;
  co.mode = cir::DriveMode::voltage_rlc;
  co.voltage_source.waveform = cir::CurrentWaveform::constant;
  co.voltage_source.i0 = V0o;
  co.L = Lo;
  co.cap = Co;
  co.Z0 = Ro;

  const Real tsamp_o[2] = {0.5, 1.0};
  for (int s = 0; s < 2; ++s) {
    Real T = tsamp_o[s];
    int N = static_cast<int>(T*steps_per_time);
    cir::LumpedCircuit c = IntegrateTo(co, T, N, 0.0);
    test.CheckNear(c.Current(), IoverAnalytic(V0o, Lo, Ro, Co, T), rtol, 1.0e-10,
                   "mode B overdamped I(t) matches analytic RLC");
  }

  // ======================================================================================
  // Part 3 -- RL limit (no capacitor, cap <= 0): I(t) = (V0/R)(1-exp(-Rt/L)).
  // ======================================================================================
  const Real V0r = 4.0, Lr = 1.0, Rr = 2.0;
  cir::LumpedCircuit crl;
  crl.mode = cir::DriveMode::voltage_rlc;
  crl.voltage_source.waveform = cir::CurrentWaveform::constant;
  crl.voltage_source.i0 = V0r;
  crl.L = Lr;
  crl.cap = 0.0;        // capacitor dropped -> pure RL loop
  crl.Z0 = Rr;
  {
    Real T = 1.0;
    int N = static_cast<int>(T*steps_per_time);
    cir::LumpedCircuit c = IntegrateTo(crl, T, N, 0.0);
    Real ianl = (V0r/Rr)*(1.0 - std::exp(-(Rr/Lr)*T));
    test.CheckNear(c.Current(), ianl, rtol, 1.0e-10,
                   "RL limit (no capacitor) I(t) == (V0/R)(1-exp(-Rt/L))");
    test.CheckNear(c.CapVoltage(), 0.0, 1.0e-13, 1.0e-13,
                   "RL limit: capacitor voltage stays 0 (branch dropped)");
  }

  // ======================================================================================
  // Part 4 -- mode C consumes the Faraday load voltage; mode B ignores it.
  //   A constant V_load just lowers the effective source to V0-V_load, so the coupled
  //   trajectory equals the analytic RLC with the reduced source.  The SAME V_load fed to
  //   mode B leaves the fixed-RLC trajectory unchanged (response of the full V0).
  // ======================================================================================
  const Real vload = 0.5;
  cir::LumpedCircuit cc = cb;                 // same underdamped RLC parameters
  cc.mode = cir::DriveMode::coupled_circuit;  // mode C
  {
    Real T = 3.0;
    int N = static_cast<int>(T*steps_per_time);
    cir::LumpedCircuit c = IntegrateTo(cc, T, N, vload);
    test.CheckNear(c.Current(), IunderAnalytic(V0u - vload, Lu, Ru, Cu, T), rtol, 1.0e-10,
                   "mode C: constant V_load lowers effective source to V0-V_load");

    // mode B with the same V_load passed must be unchanged (fixed RLC, no feedback).
    cir::LumpedCircuit cbfb = IntegrateTo(cb, T, N, vload);   // cb is mode B
    cir::LumpedCircuit cbno = IntegrateTo(cb, T, N, 0.0);
    test.CheckNear(cbfb.Current(), cbno.Current(), 1.0e-14, 1.0e-14,
                   "mode B ignores the supplied V_load (fixed RLC)");
    test.CheckNear(cbfb.Current(), IunderAnalytic(V0u, Lu, Ru, Cu, T), rtol, 1.0e-10,
                   "mode B response uses the full source V0 regardless of V_load");
    // mode C with V_load actually changed the current (the feedback is live).
    test.CheckTrue(std::fabs(c.Current() - cbno.Current()) > 1.0e-3,
                   "mode C current differs from mode B (feedback is consumed)");
  }

  // ======================================================================================
  // Part 5 -- the integrated current drives the B_phi boundary: BoundaryBphi==DrivenBphi
  //   and the enclosed-current relation r*B_phi = mu0*I/(2*pi).
  // ======================================================================================
  cir::LumpedCircuit cbnd = IntegrateTo(cb, 3.0, 3*steps_per_time, 0.0);
  const Real mu0 = 1.0, rload = 1.5;
  cbnd.mu0 = mu0;
  Real bphi = cbnd.BoundaryBphi(rload);
  test.CheckNear(bphi, cir::DrivenBphi(cbnd.Current(), rload, mu0), 1.0e-14, 0.0,
                 "BoundaryBphi == DrivenBphi(I, r, mu0)");
  test.CheckNear(bphi, mu0*cbnd.Current()/(cir::kTwoPi*rload), 1.0e-13, 0.0,
                 "boundary B_phi == mu0*I/(2*pi*r) (current drives the boundary)");
  test.CheckNear(rload*bphi, mu0*cbnd.Current()/cir::kTwoPi, 1.0e-13, 0.0,
                 "r*B_phi == mu0*I/(2*pi) (enclosed load current)");

  test.Finish();
  return;
}
