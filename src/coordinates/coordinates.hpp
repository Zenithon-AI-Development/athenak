#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of light-weight coordinates class.  Provides data structure that
//! stores array of RegionSizes over (# of MeshBlocks), and inline functions for
//! computing positions.  In GR, also provides inline metric functions (currently only
//! Cartesian Kerr-Schild)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coord_geometry.hpp"  // CoordSystem + inline geometry accessors

// forward declarations
struct EOS_Data;

// Enumerator for the excision method
enum class ExcisionScheme {
  fixed,
  lapse
};

//----------------------------------------------------------------------------------------
//! \struct CoordData
//! \brief container for Coordinate variables and functions needed inside kernels. Storing
//! everything in a container makes them easier to capture, and pass to inline functions,
//! inside kernels.

struct CoordData {
  // following data is only used in GR calculations to compute metric
  bool is_minkowski;               // flag to specify Minkowski (flat) space
  Real bh_spin;                    // needed for GR metric
  bool bh_excise;                  // flag to specify excision
  Real rexcise;                    // excision radius (SKS)
  Real dexcise;                    // rest-mass density inside excised region
  Real pexcise;                    // pressure inside excised region
  Real flux_excise_r;              // reduce to first-order inside this radius
  ExcisionScheme excision_scheme;  // excision method
  Real excise_lapse;               // if excision_scheme = lapse, excise under this lapse
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates {
 public:
  explicit Coordinates(ParameterInput *pin, MeshBlockPack *ppack);
  ~Coordinates() {}

  // coordinate system selected from `<coord> system` (default cartesian); curvilinear
  // kernels dispatch on this via the inline accessors in coord_geometry.hpp
  CoordSystem coord_system = CoordSystem::cartesian;

  // flags to denote relativistic dynamics in these coordinates
  bool is_special_relativistic = false;
  bool is_general_relativistic = false;
  bool is_dynamical_relativistic = false;

  // data needed to compute metric in GR
  CoordData coord_data;

  // excision masks
  DvceArray4D<bool> excision_floor;  // cell-centered mask for C2P flooring about horizon
  DvceArray4D<bool> excision_flux;   // cell-centered mask for FOFC about horizon

  // functions
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void CoordSrcTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);
  // cylindrical (curvilinear) hydro geometric source terms (ADR-0004, issue #14): the
  // radial centrifugal+pressure source and the angular-momentum-conserving azimuthal
  // source (re-symmetrized x1-flux of rho v_phi).  `flx1` is the x1-flux of the conserved
  // hydro variables (uflx.x1f), used for the angular-momentum-conserving azimuthal term.
  void CoordSrcTermsHydroCyl(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &flx1,
                             const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);
  // cylindrical (curvilinear) MHD geometric source terms (ADR-0004, issue #16): the
  // radial centrifugal+pressure source augmented with the magnetic stress
  // 1/2(B_r^2 - B_phi^2 + B_z^2) (the -B_phi^2 hoop stress), and the angular-momentum-
  // conserving azimuthal source.  `bcc` is the cell-centered field; `flx1` is the x1-flux
  // of the conserved MHD variables (uflx.x1f); its IM2 entry carries the Maxwell stress.
  void CoordSrcTermsMHDCyl(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                           const DvceArray5D<Real> &flx1, const EOS_Data &eos,
                           const Real dt, DvceArray5D<Real> &u0);
  void SetExcisionMasks(DvceArray4D<bool> &floor, DvceArray4D<bool> &flux);

  void UpdateExcisionMasks();

 private:
  MeshBlockPack* pmy_pack;
};

#endif // COORDINATES_COORDINATES_HPP_
