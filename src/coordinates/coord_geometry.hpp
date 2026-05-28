#ifndef COORDINATES_COORD_GEOMETRY_HPP_
#define COORDINATES_COORD_GEOMETRY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coord_geometry.hpp
//  \brief coordinate-system enum + inline device geometry accessors.
//
// These are the geometry-agnostic building blocks that let the finite-volume kernels
// (flux divergence, constrained transport, the CFL timestep, and -- in later issues --
// the diffusion operators) run without hard-coding uniform-Cartesian arithmetic.  A run
// selects its coordinate system with the input parameter `<coord> system` (default
// `cartesian`, stored as the CoordSystem enum on the Coordinates class), and the kernels
// call the inline accessors below to obtain cell volumes, face areas, edge lengths,
// cell-center widths, geometric source coefficients, and the conservative flux div.
//
// The constrained-transport curl (mhd_ct.cpp) divides the EMF difference by the cell-edge
// length in the differentiation direction (Edge?Length), and newdt divides the cell-
// center width (CenterWidth?) by the signal speed; both reduce to the legacy `/dx` on the
// Cartesian path so the update stays byte-identical with zero extra FLOPs.
//
// Every accessor is a KOKKOS_INLINE_FUNCTION so it inlines into device kernels with zero
// call overhead.  The Cartesian specialization returns the analytic uniform-grid
// constants (constant area/volume, zero coordinate-source), and the FluxDiv* helpers
// return the legacy closed-form `(fr - fl)/dx` so the Cartesian update stays
// byte-identical with -- and free of the extra area/volume FLOPs of -- the previous
// uniform-grid code (ADR-0004).
//
// CYLINDRICAL specialization (issue #14, ADR-0004): coordinates are (x1,x2,x3)=(r,phi,z).
// The metric enters through the radial face radii (rm = r at the i-1/2 face, rp = r at
// the i+1/2 face) and the cell-center radius (x1v); the curvilinear accessors take these
// extra arguments that DEFAULT to 0 so every existing Cartesian call site -- which never
// supplies them -- compiles unchanged and the `cartesian` branch (which ignores them)
// stays bit-for-bit identical.  Spherical (issue #16) adds further cases the same way.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \enum CoordSystem
//! \brief selects the coordinate system; chosen from the `<coord> system` input.
//! `cartesian` and `cylindrical` are implemented; `spherical` is reserved for a later
//! curvilinear-geometry issue and is currently rejected by the Coordinates constructor.

enum class CoordSystem {
  cartesian,
  cylindrical,
  spherical
};

//----------------------------------------------------------------------------------------
//! \fn Real CellVolume()
//! \brief volume of cell (i,j,k).  Cartesian: dx1*dx2*dx3.  Cylindrical (per unit dphi*dz
//! the radial factor is 1/2(rp^2-rm^2)): V = 1/2(rp^2-rm^2)*dx2*dx3.

KOKKOS_INLINE_FUNCTION
static Real CellVolume(CoordSystem csys, Real dx1, Real dx2, Real dx3,
                       Real rm=0.0, Real rp=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return 0.5*(rp*rp - rm*rm)*dx2*dx3;
    case CoordSystem::cartesian:
    default:
      return dx1*dx2*dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Face1Area()
//! \brief area of the x1-normal face (at radius `rface` in cylindrical).  Cartesian:
//! dx2*dx3.  Cylindrical: rface*dx2*dx3 (the area vanishes on the axis rface=0).

KOKKOS_INLINE_FUNCTION
static Real Face1Area(CoordSystem csys, Real dx2, Real dx3, Real rface=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return rface*dx2*dx3;
    case CoordSystem::cartesian:
    default:
      return dx2*dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Face2Area()
//! \brief area of the x2-normal face.  Cartesian: dx1*dx3.  Cylindrical: dr*dz = dx1*dx3
//! (the phi-normal face is a radial-axial plane, independent of radius), so identical.

KOKKOS_INLINE_FUNCTION
static Real Face2Area(CoordSystem csys, Real dx1, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return dx1*dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Face3Area()
//! \brief area of the x3-normal face.  Cartesian: dx1*dx2.  Cylindrical (annular sector):
//! 1/2(rp^2-rm^2)*dx2.

KOKKOS_INLINE_FUNCTION
static Real Face3Area(CoordSystem csys, Real dx1, Real dx2,
                      Real rm=0.0, Real rp=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return 0.5*(rp*rp - rm*rm)*dx2;
    case CoordSystem::cartesian:
    default:
      return dx1*dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Edge1Length()
//! \brief length of the x1-aligned cell edge.  Cartesian: dx1.  Cylindrical: dr = dx1.

KOKKOS_INLINE_FUNCTION
static Real Edge1Length(CoordSystem csys, Real dx1) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return dx1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Edge2Length()
//! \brief length of the x2-aligned cell edge.  Cartesian: dx2.  Cylindrical azimuthal
//! edge at radius `rface`: rface*dx2 (the arc length r*dphi).

KOKKOS_INLINE_FUNCTION
static Real Edge2Length(CoordSystem csys, Real dx2, Real rface=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return rface*dx2;
    case CoordSystem::cartesian:
    default:
      return dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Edge3Length()
//! \brief length of the x3-aligned cell edge.  Cartesian: dx3.  Cylindrical: dz = dx3.

KOKKOS_INLINE_FUNCTION
static Real Edge3Length(CoordSystem csys, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CenterWidth1()
//! \brief physical width of cell (i,j,k) through its center along x1, used by the CFL
//! timestep (newdt) as dt ~ CenterWidth/signal-speed.  Cartesian: dx1.  Cylindrical: dx1.
//!
//! This is the cell-CENTER width, distinct from Edge1Length (a cell-edge length): in
//! curvilinear systems they differ (e.g. cylindrical CenterWidth2 = x1v*dx2 evaluated at
//! the cell-center radius x1v, whereas Edge2Length = r*dx2 at a face radius).  Cartesian
//! collapses both to dx, so newdt routed through this accessor stays byte-identical.

KOKKOS_INLINE_FUNCTION
static Real CenterWidth1(CoordSystem csys, Real dx1) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return dx1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CenterWidth2()
//! \brief physical width of cell (i,j,k) through its center along x2 (see CenterWidth1).
//! Cartesian: dx2.  Cylindrical: x1v*dx2, the near-axis-aware azimuthal CFL width (the
//! arc length r*dphi evaluated at the cell-center radius x1v).

KOKKOS_INLINE_FUNCTION
static Real CenterWidth2(CoordSystem csys, Real dx2, Real x1v=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return x1v*dx2;
    case CoordSystem::cartesian:
    default:
      return dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CenterWidth3()
//! \brief physical width of cell (i,j,k) through its center along x3 (see CenterWidth1).
//! Cartesian: dx3.  Cylindrical: dz = dx3.

KOKKOS_INLINE_FUNCTION
static Real CenterWidth3(CoordSystem csys, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CoordSrc1Coeff()
//! \brief geometric (coordinate) source coefficient for the x1-momentum equation.
//! Cartesian space is flat, so there is no coordinate-source term: returns 0.
//! Cylindrical: the volume-averaged <1/r> = (rp-rm)/[1/2(rp^2-rm^2)].  The radial
//! momentum source is S_1 = <1/r>(rho v_phi^2 + p), the centrifugal + pressure terms.
//! Computed with the SAME 1/2(rp^2-rm^2) denominator as FluxDivX1 so a uniform-pressure
//! static state is held to round-off (the pressure flux divergence and S_1 cancel).

KOKKOS_INLINE_FUNCTION
static Real CoordSrc1Coeff(CoordSystem csys, Real rm=0.0, Real rp=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return (rp - rm)/(0.5*(rp*rp - rm*rm));
    case CoordSystem::cartesian:
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CoordSrc2Coeff()
//! \brief geometric (coordinate) source coefficient for the x2-momentum equation.
//! Cartesian space is flat, so there is no coordinate-source term: returns 0.
//! Cylindrical: 2/(rp+rm)^2.  The azimuthal momentum source uses the re-symmetrized,
//! face-r-weighted x1-flux of rho v_phi: S_2 = -CoordSrc2Coeff*(rm*F_l + rp*F_r), which
//! makes the discrete scheme conserve angular momentum r*(rho v_phi) to round-off.

KOKKOS_INLINE_FUNCTION
static Real CoordSrc2Coeff(CoordSystem csys, Real rm=0.0, Real rp=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return 2.0/((rp + rm)*(rp + rm));
    case CoordSystem::cartesian:
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CoordSrc3Coeff()
//! \brief geometric (coordinate) source coefficient for the x3-momentum equation.
//! Both Cartesian and the cylindrical axial (z) direction are flat: returns 0.

KOKKOS_INLINE_FUNCTION
static Real CoordSrc3Coeff(CoordSystem csys) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FluxDivX1()
//! \brief contribution of the x1 fluxes to the conservative flux divergence at a cell,
//! i.e. the x1 part of (1/V)[A_{i+1/2}F_{i+1/2} - A_{i-1/2}F_{i-1/2}].
//!
//! `fl`/`fr` are the fluxes through the left (i-1/2) / right (i+1/2) x1-faces, `dx1` the
//! cell width, and (in cylindrical) `rm`/`rp` the left/right face radii.  Cartesian: the
//! constant face area cancels the volume, leaving the legacy `(fr - fl)/dx1` (closed
//! form, byte-identical, no area/volume FLOPs).  Cylindrical: the shared dphi*dz cancels,
//! leaving (rp*fr - rm*fl)/[1/2(rp^2-rm^2)] = (1/r) d(r F)/dr; the rm=0 axis face carries
//! zero area, so flux through the axis vanishes automatically.

KOKKOS_INLINE_FUNCTION
static Real FluxDivX1(CoordSystem csys, Real fl, Real fr, Real dx1,
                      Real rm=0.0, Real rp=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return (rp*fr - rm*fl)/(0.5*(rp*rp - rm*rm));
    case CoordSystem::cartesian:
    default:
      return (fr - fl)/dx1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FluxDivX2()
//! \brief contribution of the x2 fluxes to the conservative flux divergence at a cell.
//! Cartesian: (fr - fl)/dx2 (see FluxDivX1).  Cylindrical (azimuthal): the phi-face area
//! dr*dz divided by the cell volume leaves (fr - fl)/(x1v*dx2) = (1/r) dF/dphi.

KOKKOS_INLINE_FUNCTION
static Real FluxDivX2(CoordSystem csys, Real fl, Real fr, Real dx2, Real x1v=0.0) {
  switch (csys) {
    case CoordSystem::cylindrical:
      return (fr - fl)/(x1v*dx2);
    case CoordSystem::cartesian:
    default:
      return (fr - fl)/dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FluxDivX3()
//! \brief contribution of the x3 fluxes to the conservative flux divergence at a cell.
//! Cartesian: (fr - fl)/dx3 (see FluxDivX1).  Cylindrical (axial): the z-face area equals
//! the cell volume per unit dz, so it is again (fr - fl)/dx3 -- identical to Cartesian.

KOKKOS_INLINE_FUNCTION
static Real FluxDivX3(CoordSystem csys, Real fl, Real fr, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    case CoordSystem::cylindrical:
    default:
      return (fr - fl)/dx3;
  }
}

#endif // COORDINATES_COORD_GEOMETRY_HPP_
