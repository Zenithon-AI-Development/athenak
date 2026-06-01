#ifndef PGEN_MAGLIF_TEMPERATURE_SEED_HPP_
#define PGEN_MAGLIF_TEMPERATURE_SEED_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file maglif_temperature_seed.hpp
//! \brief Ellison Eq.16 random zone-by-zone temperature perturbation for MRT seeding
//!  (issue [P4]/#161).
//!
//! The MagLIF multi-mode MRT benchmark (Ellison et al. 2025, arXiv:2504.10760, Eq.16)
//! seeds the magneto-Rayleigh-Taylor feed not with a geometric interface displacement
//! (the `perturbation=roughness` mode in maglif.cpp) but with a RANDOM, zone-by-zone
//! electron+ion temperature perturbation:
//!
//!   T_ele(r,z) = T_ion(r,z) = max( T_min, T_0 + exp((r - r_o)/lambda) * N_dT(i,j) )
//!
//! where T_0 = 293 K (reference), T_min = 273 K (floor), r_o is the OUTER liner radius,
//! lambda the radial decay length (the perturbation is full at the outer liner surface
//! r=r_o and decays exponentially INWARD toward smaller radii), and N_dT(i,j) is an
//! independent draw from a normal distribution with mean 0 and standard deviation dT
//! (the swept amplitude, e.g. 100 K or 1000 K).
//!
//! This header factors the two non-trivial, device-inline pieces so the maglif pgen and
//! its unit test share ONE implementation:
//!  * CellGaussian(gid, seed): a deterministic, counter-based standard-normal draw keyed
//!    by a GLOBAL cell id + user seed.  Keying on the global cell id (not a per-MeshBlock
//!    or per-rank index) makes the seeded field reproducible by `pert_seed` and INVARIANT
//!    under domain decomposition -- the same physical zone draws the same value on 1 rank
//!    or N.  No host RNG array / scan is needed (unlike the roughness per-mode phases).
//!  * EllisonTemperature(...): Eq.16 itself, given the per-zone Gaussian draw.

#include <cstdint>

#include "athena.hpp"   // Real, Kokkos

namespace maglif_tseed {

//! 2*pi as a Real (device-safe; avoids the host-only M_PI macro in a device kernel).
constexpr Real kTwoPi = 6.2831853071795864769;

//----------------------------------------------------------------------------------------
//! \fn uint64_t SplitMix64
//! \brief SplitMix64 finalizing hash (Steele et al. 2014): a fast, well-mixed bijection
//!  uint64->uint64.  Used as a stateless counter-based PRNG: hashing distinct inputs
//!  yields statistically-independent outputs, so a per-cell draw needs no shared RNG
//!  state and is trivially reproducible (a pure function of its argument).
KOKKOS_INLINE_FUNCTION
uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

//----------------------------------------------------------------------------------------
//! \fn Real HashToUnit
//! \brief Map a 64-bit hash to a Real in the half-open interval (0, 1].  Uses the top 53
//!  bits (the double mantissa) and shifts off zero so the result is never exactly 0 --
//!  required because it feeds log() in the Box-Muller transform below.
KOKKOS_INLINE_FUNCTION
Real HashToUnit(uint64_t h) {
  return (static_cast<Real>(h >> 11) + 1.0) * (1.0/9007199254740992.0);  // (0,1]
}

//----------------------------------------------------------------------------------------
//! \fn Real CellGaussian
//! \brief Deterministic standard-normal N(0,1) draw for a global cell id `gid` under user
//!  `seed`.  Two independent SplitMix64 hashes give two uniforms in (0,1]; one Box-Muller
//!  transform returns a single standard-normal sample.  Pure function of (gid, seed):
//!  same arguments -> identical sample on host or device, on any number of ranks.
KOKKOS_INLINE_FUNCTION
Real CellGaussian(uint64_t gid, uint64_t seed) {
  uint64_t h1 = SplitMix64(gid ^ (seed * 0x9E3779B97F4A7C15ULL));
  uint64_t h2 = SplitMix64(h1);
  Real u1 = HashToUnit(h1);
  Real u2 = HashToUnit(h2);
  return Kokkos::sqrt(-2.0*Kokkos::log(u1)) * Kokkos::cos(kTwoPi*u2);
}

//----------------------------------------------------------------------------------------
//! \fn Real EllisonTemperature
//! \brief Ellison Eq.16 per-zone temperature given the standard-normal draw `g`:
//!    T = max( Tmin, T0 + exp((r - r_o)/lambda) * dT * g ).
//!  `r_o` is the outer liner radius (decay reference); the envelope is 1 at r=r_o and
//!  decays inward.  dT*g is the N(0,dT) perturbation; the floor clips it at Tmin.  At
//!  dT=0 this returns T0 (since T0 > Tmin), i.e. the unperturbed state.
KOKKOS_INLINE_FUNCTION
Real EllisonTemperature(Real r, Real r_o, Real lambda, Real T0, Real Tmin,
                        Real dT, Real g) {
  Real env = Kokkos::exp((r - r_o)/lambda);
  Real T = T0 + env*dT*g;
  return (T < Tmin) ? Tmin : T;
}

}  // namespace maglif_tseed

#endif  // PGEN_MAGLIF_TEMPERATURE_SEED_HPP_
