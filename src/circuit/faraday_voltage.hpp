#ifndef CIRCUIT_FARADAY_VOLTAGE_HPP_
#define CIRCUIT_FARADAY_VOLTAGE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file faraday_voltage.hpp
//! \brief Faraday load-voltage global reduction for the circuit-driven boundary
//!        (ADR-0005 mode C feedback term, issue [18a]/#31).
//!
//! Z-pinch / MagLIF is circuit-driven (ADR-0005).  In the coupled-circuit mode the load
//! voltage feeds back into the lumped-element circuit ODE and is, by Faraday's law, the
//! rate of change of the magnetic flux linking the circuit:  V_load = d(Phi)/dt.
//!
//! In AthenaK-cylindrical (ADR-0004) coordinates are (x1,x2,x3)=(r,phi,z).  The axial
//! load current I(t) flows up the column (J_z) and returns at the outer radius; that
//! loop lives in the poloidal (r-z) plane and is threaded by the azimuthal field B_phi.
//! The flux linking it is the integral of B_phi over the poloidal plane,
//!     Phi = \int\int B_phi dr dz,
//! averaged over phi-planes (it reduces to the single-plane flux for an axisymmetric
//! nx2=1 run, which is the MagLIF case).  B_phi is the x2-face field b0.x2f, and the
//! phi-normal face it lives on is a radial-axial surface element with area
//! Face2Area = dr*dz (= dx1*dx3, geometry-independent -- see coord_geometry.hpp), so the
//! reduction integrand is simply b0.x2f * Face2Area.
//!
//! This is a deep module:
//!   * PoloidalFluxLocal  -- the per-rank partial of Phi via a Kokkos parallel_reduce
//!                           (the per-cell sum AthenaK's history machinery uses);
//!   * PoloidalFluxGlobal -- adds the MPI_Allreduce so a standalone caller (the coupled
//!                           circuit, #34) gets the global flux mid-step;
//!   * FaradayVoltage     -- a tiny stateful monitor that differences successive fluxes
//!                           in time, V = (Phi - Phi_prev)/(t - t_prev);
//!   * FillFaradayHistory -- exposes (flux, V_load) as a two-column history diagnostic by
//!                           filling a HistoryData (reusing history.cpp's per-rank-then-
//!                           MPI_Reduce machinery), to be enrolled via user_hist_func.
//!
//! Because no file in the main build includes this header (only the pgens that opt in via
//! -D PROBLEM=<name> do), it costs the main binary nothing and cannot perturb existing
//! behaviour -- the same additive, byte-identical-by-construction shape as the sibling
//! circuit/drive_source.hpp and the eos/opacity deep modules.

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/coord_geometry.hpp"
#include "outputs/outputs.hpp"

namespace circuit {

//----------------------------------------------------------------------------------------
//! \fn Real PoloidalFluxLocal
//! \brief Per-rank partial of the poloidal magnetic flux Phi = \int\int B_phi dr dz,
//!        averaged over phi-planes.  Sum across MPI ranks (history.cpp's MPI_Reduce on
//!        hdata, or the MPI_Allreduce in PoloidalFluxGlobal) to obtain the global flux.
//!        Returns 0 if the MHD module is absent (B_phi flux is meaningful only for MHD).
inline Real PoloidalFluxLocal(Mesh *pm) {
  if (pm->pmb_pack->pmhd == nullptr) { return 0.0; }

  auto &bx2f = pm->pmb_pack->pmhd->b0.x2f;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto csys = pm->pmb_pack->pcoord->coord_system;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  // global phi-plane count: divide the all-face sum by it to get the per-plane average
  // (exact single-plane flux for the axisymmetric nx2=1 MagLIF run).
  int nphi = pm->mesh_indcs.nx2;

  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;

  Real flux_sum = 0.0;
  Kokkos::parallel_reduce("FaradayFlux", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    // decompose the flat index into (m,k,j,i)
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    // flux of B_phi (the x2-face field) through this cell's phi-normal face.
    Real da = Face2Area(csys, size.d_view(m).dx1, size.d_view(m).dx3);
    sum += bx2f(m,k,j,i)*da;
  }, Kokkos::Sum<Real>(flux_sum));
  Kokkos::fence();

  return flux_sum/static_cast<Real>(nphi);
}

//----------------------------------------------------------------------------------------
//! \fn Real PoloidalFluxGlobal
//! \brief The global poloidal flux: PoloidalFluxLocal plus an MPI_Allreduce.  Use this
//!        for a standalone reduction (the coupled-circuit feedback in #34 needs the value
//!        every step); the history diagnostic uses the *local* partial and lets
//!        history.cpp do the cross-rank sum.
inline Real PoloidalFluxGlobal(Mesh *pm) {
  Real local = PoloidalFluxLocal(pm);
#if MPI_PARALLEL_ENABLED
  Real global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  return global;
#else
  return local;
#endif
}

//----------------------------------------------------------------------------------------
//! \class FaradayVoltage
//! \brief Stateful d(flux)/dt monitor implementing Faraday's law V_load = d(Phi)/dt as a
//!        backward difference of successive flux samples:
//!            V = (Phi - Phi_prev)/(t - t_prev).
//!        Update() stores the new sample and returns the voltage (0 on the first call,
//!        when there is no previous sample to difference against, and when t == t_prev).
//!        Differencing is linear, so feeding it per-rank *local* fluxes and summing the
//!        resulting per-rank voltages across ranks yields exactly the global voltage --
//!        which is why FillFaradayHistory can rely on history.cpp's MPI_Reduce(SUM).
class FaradayVoltage {
 public:
  //! \brief Record a new flux sample at time `time`; return V = d(flux)/dt.
  Real Update(Real flux, Real time) {
    Real volt = 0.0;
    if (initialized_ && time > time_prev_) {
      volt = (flux - flux_prev_)/(time - time_prev_);
    }
    flux_prev_ = flux;
    time_prev_ = time;
    voltage_ = volt;
    initialized_ = true;
    return volt;
  }
  Real flux() const { return flux_prev_; }
  Real voltage() const { return voltage_; }
  Real time() const { return time_prev_; }
  bool initialized() const { return initialized_; }
  void Reset() {
    flux_prev_ = 0.0;
    time_prev_ = 0.0;
    voltage_ = 0.0;
    initialized_ = false;
  }

 private:
  Real flux_prev_ = 0.0;   //!< last flux sample
  Real time_prev_ = 0.0;   //!< time of the last sample
  Real voltage_ = 0.0;     //!< last computed voltage
  bool initialized_ = false;
};

//----------------------------------------------------------------------------------------
//! \fn void FillFaradayHistory
//! \brief Expose the poloidal flux and the Faraday load voltage as two history columns
//!        ("Bphi_flux", "V_load").  Stores the PER-RANK LOCAL partials, so history.cpp's
//!        in-place MPI_Reduce(SUM) in WriteOutputFile yields the global flux and (because
//!        the time difference is linear) the global voltage.  A pgen wires the diagnostic
//!        in two steps: keep a file-scope FaradayVoltage monitor, set user_hist=true, and
//!        enroll a user_hist_func that calls this with that monitor.
inline void FillFaradayHistory(HistoryData *pdata, Mesh *pm, FaradayVoltage &mon) {
  pdata->nhist = 2;
  pdata->label[0] = "Bphi_flux";
  pdata->label[1] = "V_load";

  Real flux_local = PoloidalFluxLocal(pm);
  Real volt_local = mon.Update(flux_local, pm->time);

  pdata->hdata[0] = flux_local;
  pdata->hdata[1] = volt_local;
  for (int n=pdata->nhist; n<NHISTORY_VARIABLES; ++n) {
    pdata->hdata[n] = 0.0;
  }
}

}  // namespace circuit

#endif  // CIRCUIT_FARADAY_VOLTAGE_HPP_
