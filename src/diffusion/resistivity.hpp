#ifndef DIFFUSION_RESISTIVITY_HPP_
#define DIFFUSION_RESISTIVITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistivity.hpp
//  \brief Contains data and functions that implement various non-ideal MHD (resistive)
//  processes, such as Ohmic diffusion. TODO(@user): add ambipolar diffusion, Hall effect

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/meshblock.hpp"
#include "variable_resistivity.hpp"

//----------------------------------------------------------------------------------------
//! \class Resistivity
//  \brief data and functions that implement various resistive physics

class Resistivity {
 public:
  Resistivity(MeshBlockPack *pp, ParameterInput *pin);
  ~Resistivity();

  // data
  Real dtnew;
  Real eta_ohm;

  // variable (Spitzer/Braginskii) resistivity eta(rho,T_e) [ADR-0003]; when false the
  // constant scalar eta_ohm is used (byte-identical to the legacy Ohmic path).
  bool variable_eta = false;
  resistivity::VarResistivityParams rparams;  // code<->SI conversions + plasma + floor
  DvceArray4D<Real> eta;  // cell-centered magnetic diffusivity (code), variable mode only

  // is the resistive operator active at all (constant or variable)?
  bool IsActive() const { return (variable_eta || eta_ohm > 0.0); }

  // fill the cell-centered eta field from the current MHD state (variable mode)
  void ComputeEta();
  // recompute the diffusion timestep from the current max(eta) (variable mode)
  void NewTimeStep();

  // functions to add resistive E-Field and energy flux
  void OhmicEField(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
  void OhmicEnergyFlux(const DvceFaceFld4D<Real> &b, DvceFaceFld5D<Real> &flx);

 private:
  MeshBlockPack* pmy_pack;
};

#endif // DIFFUSION_RESISTIVITY_HPP_
