#ifndef DIFFUSION_RESISTIVE_BPHI_OPERATOR_HPP_
#define DIFFUSION_RESISTIVE_BPHI_OPERATOR_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resistive_bphi_operator.hpp
//! \brief The special CYLINDRICAL resistive diffusion of B_phi as a
//! parabolic::ParabolicOperator (issue [7b]/#27, ADR-0004/ADR-0001).
//
// ADR-0004 calls the resistive B_phi diffusion "the one true cylindrical exception": the
// phi-component of the resistive (curl-curl) operator is NOT the cylindrical scalar
// Laplacian -- it carries a genuine -eta*B_phi/r^2 term:
//
//   dB_phi/dt = d/dr[ eta (1/r) d(r B_phi)/dr ] + d/dz[ eta dB_phi/dz ]
//             = (1/r) d/dr( r eta dB_phi/dr ) - eta B_phi/r^2 + eta d^2B_phi/dz^2 ,
//
// i.e. the vector-Laplacian phi-component (for an axisymmetric / out-of-plane B_phi).
// The radial part is carried in the CONSERVATIVE (1/r) d_r(r * flux) finite-volume form
// (the shared FluxDivX1 accessor = exactly (1/r) d(r F)/dr in cylindrical), so the r=0
// axis face -- whose area vanishes -- naturally carries zero radial flux; the curl-curl
// -eta B_phi/r^2 term is then added explicitly at cell centres (r = x1v).  The axis is
// closed with an ANTISYMMETRIC ghost B_phi(i-1) = -B_phi(i) (B_phi is odd across r=0:
// B_phi ~ r near the axis, like the J_1 Bessel eigenmode the verification test uses).
//
// This is the resistive-B ParabolicOperator of ADR-0001's parabolic stack: it implements
// the same OperatorAction / ExplicitStableDt / ApplyBoundary contract the FLD (#17) and
// conduction (#13/#18) operators do, and is advanced operator-split by the RKL2
// super-time-stepping integrator (#9).  The magnetic diffusivity eta(rho,T_e) is supplied
// by the caller as a cell-centred field -- exactly the field the variable Spitzer/
// Braginskii resistivity module (SIM-76 / #20, Resistivity::ComputeEta) produces -- so
// the operator stays a pure geometry/diffusion kernel and the eta physics lives in #20.
//
// FIELD LAYOUT.  The operator evolves a single-component DvceArray5D `bphi` holding B_phi
// in component 0 (so the RKL2 5D Superstep advances exactly that field).  B_phi does NOT
// enter div(B) for an axisymmetric column (d_phi B_phi = 0), so diffusing it on its own
// never violates the solenoidal constraint -- which is why it gets this dedicated scalar-
// like operator rather than riding the constrained-transport EMF (#20's Cartesian path).
//
// SCOPE.  Radial (x1, the special 1/r^2 operator) + axial (x3 = z, plain diffusion) +
// azimuthal (x2 = phi, the cylindrical scalar phi-Laplacian via the r-weighted accessors)
// -- the MagLIF/Z-pinch r-z (and r-phi) target.  The 3D B_r<->B_phi cross-coupling term
// (2/r^2) d_phi B_r of the full vector Laplacian is deferred (it vanishes for the
// axisymmetric column and the radial verification mode).  The Cartesian path drops the
// -eta B_phi/r^2 term (flat space has no curl-curl geometric term) and reduces to plain
// constant-eta scalar diffusion of one field component.

#include "athena.hpp"
#include "driver/parabolic_operator.hpp"

// forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class ResistiveBphiOperator
//! \brief Cylindrical resistive B_phi diffusion (the -eta B_phi/r^2 curl-curl operator,
//! conservative (1/r)d_r(r*flux) radial form, antisymmetric axis ghost) as a
//! parabolic::ParabolicOperator.

class ResistiveBphiOperator : public parabolic::ParabolicOperator {
 public:
  //! \brief Build the operator over the single-component field `bphi` (component 0 holds
  //! B_phi) it diffuses, with the cell-centred magnetic diffusivity field `eta`
  //! (nmb,n3,n2,n1) -- the SIM-76 / #20 Resistivity::ComputeEta output (or a constant
  //! fill for the verification oracle).
  ResistiveBphiOperator(MeshBlockPack *pp, const DvceArray5D<Real> &bphi,
                        const DvceArray4D<Real> &eta);
  ~ResistiveBphiOperator() = default;

  //! \brief M(u): cylindrical resistive B_phi diffusion into rhs_out(0); 0 in any other
  //! component (frozen background).  Reads ghost zones of u_in.
  void OperatorAction(const DvceArray5D<Real> &u_in,
                      DvceArray5D<Real> &rhs_out) override;

  //! \brief Forward-Euler resistive stability dt, including the near-axis 1/r^2 term.
  Real ExplicitStableDt() override;

  //! \brief Ghost-zone refresh: antisymmetric ghost B_phi(i-1) = -B_phi(i) at both radial
  //! ends (the r=0 axis, and the outer Dirichlet B_phi=0 node of the verification mode);
  //! zero-gradient in x2/x3.  Called before every M(.) so the RKL2 substeps see a
  //! boundary-consistent stencil.
  void ApplyBoundary(DvceArray5D<Real> &u) override;

  static constexpr int IBPHI = 0;   // component slot holding B_phi in the field array

 private:
  MeshBlockPack *pmy_pack;
  DvceArray5D<Real> bphi_;      // the live B_phi field (for the explicit-dt loop)
  DvceArray4D<Real> eta_;       // cell-centred magnetic diffusivity (SIM-76 / #20)
  DvceFaceFld5D<Real> bflx_;    // scratch face-centred resistive B_phi flux
};

#endif  // DIFFUSION_RESISTIVE_BPHI_OPERATOR_HPP_
