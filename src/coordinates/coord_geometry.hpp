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
// (flux divergence, and -- in later issues -- CT, newdt, and the diffusion operators)
// run without hard-coding uniform-Cartesian arithmetic.  A run selects its coordinate
// system with the input parameter `<coord> system` (default `cartesian`, stored as the
// CoordSystem enum on the Coordinates class), and the kernels call the inline accessors
// below to obtain cell volumes, face areas, edge lengths, geometric source coefficients,
// and the conservative flux divergence.
//
// Every accessor is a KOKKOS_INLINE_FUNCTION so it inlines into device kernels with zero
// call overhead.  The Cartesian specialization returns the analytic uniform-grid
// constants (constant area/volume, zero coordinate-source), and the FluxDiv* helpers
// return the legacy closed-form `(fr - fl)/dx` so the Cartesian update stays
// byte-identical with -- and free of the extra area/volume FLOPs of -- the previous
// uniform-grid code (ADR-0004).
//
// The curvilinear specializations (cylindrical #14, spherical #16) add cases to the
// switch statements below; until then the Coordinates constructor accepts only
// `cartesian`, so the `default` branches are never reached at run time.

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \enum CoordSystem
//! \brief selects the coordinate system; chosen from the `<coord> system` input.
//! Only `cartesian` is implemented today; `cylindrical`/`spherical` are reserved for the
//! curvilinear-geometry issues and currently rejected by the Coordinates constructor.

enum class CoordSystem {
  cartesian,
  cylindrical,
  spherical
};

//----------------------------------------------------------------------------------------
//! \fn Real CellVolume()
//! \brief volume of cell (i,j,k).  Cartesian: dx1*dx2*dx3.

KOKKOS_INLINE_FUNCTION
static Real CellVolume(CoordSystem csys, Real dx1, Real dx2, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx1*dx2*dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Face1Area()
//! \brief area of the x1-normal face.  Cartesian: dx2*dx3.

KOKKOS_INLINE_FUNCTION
static Real Face1Area(CoordSystem csys, Real dx2, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx2*dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Face2Area()
//! \brief area of the x2-normal face.  Cartesian: dx1*dx3.

KOKKOS_INLINE_FUNCTION
static Real Face2Area(CoordSystem csys, Real dx1, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx1*dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Face3Area()
//! \brief area of the x3-normal face.  Cartesian: dx1*dx2.

KOKKOS_INLINE_FUNCTION
static Real Face3Area(CoordSystem csys, Real dx1, Real dx2) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx1*dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Edge1Length()
//! \brief length of the x1-aligned cell edge.  Cartesian: dx1.

KOKKOS_INLINE_FUNCTION
static Real Edge1Length(CoordSystem csys, Real dx1) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Edge2Length()
//! \brief length of the x2-aligned cell edge.  Cartesian: dx2.

KOKKOS_INLINE_FUNCTION
static Real Edge2Length(CoordSystem csys, Real dx2) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real Edge3Length()
//! \brief length of the x3-aligned cell edge.  Cartesian: dx3.

KOKKOS_INLINE_FUNCTION
static Real Edge3Length(CoordSystem csys, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return dx3;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CoordSrc1Coeff()
//! \brief geometric (coordinate) source coefficient for the x1-momentum equation.
//! Cartesian space is flat, so there is no coordinate-source term: returns 0.

KOKKOS_INLINE_FUNCTION
static Real CoordSrc1Coeff(CoordSystem csys) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CoordSrc2Coeff()
//! \brief geometric (coordinate) source coefficient for the x2-momentum equation.
//! Cartesian space is flat, so there is no coordinate-source term: returns 0.

KOKKOS_INLINE_FUNCTION
static Real CoordSrc2Coeff(CoordSystem csys) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real CoordSrc3Coeff()
//! \brief geometric (coordinate) source coefficient for the x3-momentum equation.
//! Cartesian space is flat, so there is no coordinate-source term: returns 0.

KOKKOS_INLINE_FUNCTION
static Real CoordSrc3Coeff(CoordSystem csys) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return 0.0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FluxDivX1()
//! \brief contribution of the x1 fluxes to the conservative flux divergence at a cell,
//! i.e. the x1 part of (1/V)[A_{i+1/2}F_{i+1/2} - A_{i-1/2}F_{i-1/2}].
//!
//! `fl`/`fr` are the fluxes through the left (i-1/2) / right (i+1/2) x1-faces and `dx1`
//! the cell width.  Cartesian: the constant face area cancels the volume, leaving the
//! legacy `(fr - fl)/dx1` (returned in closed form so the uniform-grid path is
//! byte-identical and carries no area/volume FLOPs).  Curvilinear specializations will
//! use Face1Area()/CellVolume().

KOKKOS_INLINE_FUNCTION
static Real FluxDivX1(CoordSystem csys, Real fl, Real fr, Real dx1) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return (fr - fl)/dx1;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FluxDivX2()
//! \brief contribution of the x2 fluxes to the conservative flux divergence at a cell.
//! Cartesian: (fr - fl)/dx2 (see FluxDivX1).

KOKKOS_INLINE_FUNCTION
static Real FluxDivX2(CoordSystem csys, Real fl, Real fr, Real dx2) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return (fr - fl)/dx2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real FluxDivX3()
//! \brief contribution of the x3 fluxes to the conservative flux divergence at a cell.
//! Cartesian: (fr - fl)/dx3 (see FluxDivX1).

KOKKOS_INLINE_FUNCTION
static Real FluxDivX3(CoordSystem csys, Real fl, Real fr, Real dx3) {
  switch (csys) {
    case CoordSystem::cartesian:
    default:
      return (fr - fl)/dx3;
  }
}

#endif // COORDINATES_COORD_GEOMETRY_HPP_
