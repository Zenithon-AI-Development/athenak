#ifndef CIRCUIT_LUMPED_CIRCUIT_HPP_
#define CIRCUIT_LUMPED_CIRCUIT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file lumped_circuit.hpp
//! \brief Coupled lumped-element circuit ODE for the voltage-driven boundary
//!        (ADR-0005 modes B and C, issue [18b]/#34).
//!
//! Z-pinch / MagLIF is circuit-driven (ADR-0005).  Mode A (prescribed I(t),
//! circuit/drive_source.hpp, #21) covers benchmarks 1-4.  Modes B/C drive the load with a
//! *voltage* source instead, integrating a small series lumped-element circuit ODE once
//! per step to obtain the current I(t) that sets the outer-radial B_phi boundary:
//!   * B -- voltage + fixed RLC: a series circuit (open-circuit voltage V_oc(t), series
//!          inductance L, capacitance C, source/line resistance Z0, loss resistor R_loss)
//!          with NO load feedback;
//!   * C -- voltage + coupled circuit: identical, plus the load voltage V_load=d(Phi)/dt
//!          (the Faraday reduction circuit/faraday_voltage.hpp, #31) fed back as an extra
//!          series voltage drop -- this captures current loss and the stagnation voltage
//!          spike as the liner's inductance changes during implosion.
//!
//! Circuit model (Kirchhoff's voltage law around the series loop):
//!     L dI/dt + (Z0 + R_loss) I + V_C + V_load = V_oc(t),   dV_C/dt = I / C,
//! with state (I, V_C) = (load current, capacitor voltage).  V_load is 0 for mode B and
//! the supplied Faraday load voltage for mode C.  C <= 0 selects a pure RL loop (the
//! capacitor branch is dropped, V_C stays 0) for an upstream-cap drive modelled purely by
//! V_oc(t).  The ODE is integrated once per MHD step with a single classic RK4 step over
//! host-side scalars (the operator-split, host-side advance ADR-0005 prescribes); R_loss
//! and V_load are the per-step inputs, held fixed across the RK4 substeps.
//!
//! For a constant V_oc and constant R = Z0 + R_loss with zero initial conditions, mode B
//! reproduces the textbook series-RLC step response exactly (underdamped:
//! I(t) = (V_oc/(L*wd)) exp(-a t) sin(wd t), a = R/2L, w0 = 1/sqrt(LC),
//! wd = sqrt(w0^2-a^2)), which is the unit-test oracle.
//!
//! The open-circuit voltage waveform reuses the prescribed-source machinery of
//! circuit/drive_source.hpp (the same constant/linear_ramp/sin_squared/tabulated shapes
//! and file reader): a held DriveSource whose scalar output is interpreted as the voltage
//! V_oc(t) (its i0 = peak open-circuit voltage), so no waveform code is duplicated.
//!
//! Units: code units throughout (Heaviside-Lorentz, mu0 = 1 by default).  BoundaryBphi()
//! turns the integrated current into the driven azimuthal field at the load radius via
//! the shared DrivenBphi formula, so a pgen (#40 coupled implosion) wires the boundary by
//! Step()-ing the circuit each step and feeding Current()/BoundaryBphi() to the B_phi BC.
//!
//! Deep, header-only, additive module: no file in the main build includes it (only pgens
//! that opt in via -D PROBLEM=<name>), so it costs the main binary nothing and cannot
//! perturb existing behaviour -- the same shape as the sibling circuit/drive_source.hpp.

#include "athena.hpp"
#include "circuit/drive_source.hpp"

namespace circuit {

//----------------------------------------------------------------------------------------
//! \class LumpedCircuit
//! \brief Series lumped-element circuit (open-circuit voltage source, L, C, Z0, R_loss)
//!        advanced once per step by RK4 (ADR-0005 modes B/C).  Host-side scalar state.
class LumpedCircuit {
 public:
  //! Drive mode: voltage_rlc (B, no feedback) or coupled_circuit (C, consumes V_load).
  DriveMode mode = DriveMode::voltage_rlc;

  //! Open-circuit voltage source V_oc(t): a DriveSource reused as a voltage waveform (its
  //! i0 = peak voltage; supports constant/linear_ramp/sin_squared/tabulated shapes).
  DriveSource voltage_source;

  // Lumped elements (code units).
  Real L = 1.0;        //!< series inductance
  Real cap = 0.0;      //!< series capacitance C (<= 0 => pure RL loop, capacitor dropped)
  Real Z0 = 0.0;       //!< fixed source/line resistance (characteristic impedance)
  Real r_loss = 0.0;   //!< time-varying loss resistor R_loss(t) value for THIS step
  Real mu0 = 1.0;      //!< code-unit permeability (for BoundaryBphi)

  //! \brief Evaluate the open-circuit voltage V_oc(t) (host).
  Real Voltage(Real t) const { return voltage_source.Current(t); }

  //! \brief Set the loss resistor for the upcoming step (a real run updates it from the
  //!        resistive load state; the unit test holds it constant).
  void SetLossResistance(Real r) { r_loss = r; }

  //! \brief Reset the dynamic state (current, capacitor voltage).
  void Reset(Real current0 = 0.0, Real vcap0 = 0.0) {
    current_ = current0;
    vcap_ = vcap0;
  }

  //! \brief Advance the circuit one step from time t to t+dt with a single RK4 step.
  //! \param dt      step size (code units)
  //! \param t       current time (V_oc is sampled at t, t+dt/2, t+dt)
  //! \param v_load  Faraday load voltage d(Phi)/dt; used only in mode C (coupled),
  //!                ignored (treated as 0) in mode B (voltage_rlc, fixed RLC).
  void Step(Real dt, Real t, Real v_load) {
    const Real vl = (mode == DriveMode::coupled_circuit) ? v_load : 0.0;

    Real i0 = current_, v0 = vcap_;
    Real ki1, kv1, ki2, kv2, ki3, kv3, ki4, kv4;
    Derivs(t,          i0,                 v0,                 vl, ki1, kv1);
    Derivs(t + 0.5*dt, i0 + 0.5*dt*ki1,    v0 + 0.5*dt*kv1,    vl, ki2, kv2);
    Derivs(t + 0.5*dt, i0 + 0.5*dt*ki2,    v0 + 0.5*dt*kv2,    vl, ki3, kv3);
    Derivs(t + dt,     i0 + dt*ki3,        v0 + dt*kv3,        vl, ki4, kv4);

    current_ = i0 + (dt/6.0)*(ki1 + 2.0*ki2 + 2.0*ki3 + ki4);
    vcap_    = v0 + (dt/6.0)*(kv1 + 2.0*kv2 + 2.0*kv3 + kv4);
  }

  //! \brief Convenience overload for mode B (no load feedback).
  void Step(Real dt, Real t) { Step(dt, t, 0.0); }

  //! \brief The integrated load current I(t).
  Real Current() const { return current_; }

  //! \brief The capacitor voltage V_C(t).
  Real CapVoltage() const { return vcap_; }

  //! \brief The driven azimuthal field the present current deposits at radius `radius`:
  //!        B_phi = mu0*I/(2*pi*r).  This is the one-line bridge from the circuit current
  //!        to the outer-radial B_phi boundary (ADR-0005), reusing the shared formula.
  Real BoundaryBphi(Real radius) const { return DrivenBphi(current_, radius, mu0); }

 private:
  //! \brief Series-loop derivatives (dI/dt, dV_C/dt) at (t, I, V_C) with feedback v_load.
  void Derivs(Real t, Real ii, Real vc, Real v_load, Real &didt, Real &dvdt) const {
    const Real r_tot = Z0 + r_loss;
    didt = (Voltage(t) - r_tot*ii - vc - v_load)/L;
    dvdt = (cap > 0.0) ? (ii/cap) : 0.0;
  }

  Real current_ = 0.0;  //!< load current I
  Real vcap_ = 0.0;     //!< capacitor voltage V_C
};

}  // namespace circuit

#endif  // CIRCUIT_LUMPED_CIRCUIT_HPP_
