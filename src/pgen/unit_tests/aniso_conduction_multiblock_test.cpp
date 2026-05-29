//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file aniso_conduction_multiblock_test.cpp
//  \brief Unit test: the operator-split anisotropic (magnetized) Braginskii conduction
//  operator (AnisotropicConductionOperator + RKL2 STS) keeps heat ON field lines while
//  running across MeshBlock / MPI-rank / AMR coarse-fine boundaries, via the shared
//  per-substage ghost-exchange helper MeshBoundaryValuesCC::SyncParabolicGhosts
//  (issue [A5]/#112, ADR-0006/ADR-0001).
//
//  This is the multi-block / multi-rank / AMR verification of the wired operator.  The
//  single-block physics (SIM-61 Braginskii coefficients, field-aligned projection, the
//  Parrish-Stone ring, the Larsen cap) is pinned by unit_tests/aniso_conduction_test;
//  we add ONLY the thing #112 builds: the inter-substage neighbour ghost exchange + the
//  internal-face-only insulating BC that let the operator span >1 MeshBlock, >1 MPI rank,
//  and block-AMR without changing the field-aligned physics.
//
//  ORACLES (all Layer 1, analytic; reductions MPI_Allreduce'd so every rank checks the
//  SAME globals and Finish() exits consistently everywhere):
//   (1) Multi-block: nmb_total > 1, so the cross-block ghost exchange is exercised.
//   (2) FIELD-ALIGNED decay (uniform B = B0 xhat): a tiny cosine mode along xhat is an
//       exact eigenvector of the discrete PARALLEL conduction operator with eigenvalue
//       lambda_par = -(2 D_par/dx^2)(1 - cos theta), D_par = kpar(gamma-1)/rho.  The mode
//       is GLOBAL, so only the correct cross-block exchange reproduces exp(lambda_par t)
//       decay.  Built WITH exchange it matches the analytic decay across blocks (GREEN);
//       built WITHOUT it (pin==nullptr) each block decays as its own insulated box and
//       the global Linf error is large (RED discriminator -- proves the exchange does the
//       work, not the insulated fill).
//   (3) CROSS-FIELD insulation (uniform B = B0 xhat): a cosine mode along yhat
//       (perpendicular to B) diffuses at kperp(gamma-1)/rho << kpar, over the SAME time
//       it barely decays -- heat stays on the field lines even across the y-block
//       boundaries the exchange refreshes.  decay_perp ~ 1 >> the parallel decay.
//   (4) RING (Parrish-Stone, heat follows field lines, multi-block): a circular frozen
//       field bhat = (-y, x)/r with a hot Gaussian patch on the ring at (r0, phi=0); the
//       ring crosses every block.  The AZIMUTHAL heating (heat carried ALONG the
//       ring, across block boundaries) far exceeds the RADIAL far-field leakage.
//   (5) Cross-field leakage vs RKL2 SUBSTAGE count: the perpendicular-mode leakage is
//       reported and asserted bounded and converged across a sweep of super-step sizes
//       (=> different RKL2 stage counts), informing the stagnation/implicit-fallback
//       trigger (#114/#115).
//   (6) Conservation: the insulated global box conserves the VOLUME-WEIGHTED total energy
//       across the decomposition (volume weighting matters under SMR/AMR: fine cells are
//       smaller; conservative pbval_flux_ correction carries energy across c/f faces).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_aniso_conduction_multiblock_cpu.py
//  (4 MeshBlocks, uniform, one rank), ..._mpicpu.py (4 MeshBlocks across MPI ranks), and
//  ..._amr_cpu.py (static-SMR, exercises the SyncParabolicGhosts coarse/fine path).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::exp, std::fabs, std::acos, std::sqrt
#include <algorithm> // std::max
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
//! \brief MPI_Allreduce a scalar across all ranks (identity on a single rank), so every
//! rank records the same global value and Finish() exits consistently everywhere.
Real GlobalReduce(Real local, bool want_max) {
  Real global = local;
#if MPI_PARALLEL_ENABLED
  MPI_Op op = want_max ? MPI_MAX : MPI_SUM;
  MPI_Allreduce(&local, &global, 1, MPI_ATHENA_REAL, op, MPI_COMM_WORLD);
#else
  (void)want_max;
#endif
  return global;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Multi-block / MPI / AMR anisotropic-conduction field-alignment verification.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("aniso_conduction_multiblock_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // (1) requires hydro (for array sizing) running on >1 MeshBlock
  bool have_hydro = (pmbp->phydro != nullptr);
  test.CheckTrue(have_hydro, "hydro block constructed (array sizing)");
  if (!have_hydro) { test.Finish(); return; }
  auto *phydro = pmbp->phydro;
  test.CheckTrue(pmy_mesh_->nmb_total > 1,
                 "more than one MeshBlock (cross-block ghost exchange is exercised)");

  // selector: <time> parabolic_integrator parses to the RKL2 STS backend
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- problem constants (GLOBAL mesh) ----
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real x2min = pin->GetReal("mesh", "x2min");
  const Real x2max = pin->GetReal("mesh", "x2max");
  const int N1 = pin->GetInteger("mesh", "nx1");
  const int N2 = pin->GetInteger("mesh", "nx2");
  const Real L1 = x1max - x1min;
  const Real L2 = x2max - x2min;
  const Real dx1 = L1/static_cast<Real>(N1);
  const Real dx2 = L2/static_cast<Real>(N2);
  const Real PI = std::acos(-1.0);
  const Real rho0 = 1.0;
  const bool multilevel = pmy_mesh_->multilevel;

  // ---- Braginskii physical + unit-conversion parameters (strongly magnetized HED) ----
  anisocond::AnisoCondParams par;
  par.zbar = 1.0;
  par.mi_si = braginskii::kProtonMass;
  par.lnlam = 10.0;
  par.dens_conv = 1.0e20;      // code density -> n_i [m^-3]
  par.temp_conv = 1.0e6;       // code temperature -> T [K]
  par.bmag_conv = 1.0e2;       // code |B| -> B [T]
  par.kappa_conv = 1.0e-4;     // SI conductivity -> code (keeps kappa_par ~ O(1))
  par.nlarsen = 2.0;
  par.vfs = -1.0;              // Larsen cap off
  par.incl_e = true;
  par.incl_i = true;

  // ---- conserved-field work arrays (shaped like hydro u0) + cell-centred B field ----
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb  = phydro->u0.extent_int(0);
  const int nvar = phydro->u0.extent_int(1);
  const int n3   = phydro->u0.extent_int(2);
  const int n2   = phydro->u0.extent_int(3);
  const int n1   = phydro->u0.extent_int(4);
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;

  DvceArray5D<Real> bcc("bcc",  nmb, 3,    n3, n2, n1);
  DvceArray5D<Real> work("work", nmb, nvar, n3, n2, n1);

  // background field magnitude + magnetic energy density (uniform |B| = B0)
  const Real B0 = 0.6;
  const Real me = 0.5*B0*B0;
  const Real T0 = 1.0;

  // representative kpar / kperp at the background state (host, for the analytic oracle)
  Real kpar, kperp;
  anisocond::Conductivities(par, rho0, T0, B0, kpar, kperp);
  test.CheckTrue(kpar > 1.0e3*kperp, "strongly magnetized: kpar >> kperp");
  const Real Dpar = kpar*gm1/rho0;     // parallel energy diffusivity
  const Real Dperp = kperp*gm1/rho0;   // perpendicular energy diffusivity

  // ----- volume-weighted interior energy sum (multilevel-correct conservation) -----
  auto interior_vol_energy = [&]() -> Real {
    auto h_w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work);
    Real s = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      Real dv = size.h_view(m).dx1*size.h_view(m).dx2*size.h_view(m).dx3;
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) { s += h_w(m,IEN,k,j,i)*dv; }
        }
      }
    }
    return s;
  };

  // ========= (2) FIELD-ALIGNED cosine decay reproduced across blocks =========
  // uniform B = B0 xhat, tiny cosine mode along xhat -> exact parallel eigenmode.
  // m_mode = 1 over the symmetric domain [-x,x] puts the single internal x1 block face
  // (domain centre) on the mode's steep ZERO-CROSSING, not an extremum -- so a broken /
  // missing exchange is maximally visible (the #108 gotcha: a face on an extremum, where
  // dT/dx=0, makes the insulated-only fill accidentally correct and the test blind).
  const int m_mode = 1;
  const Real amp = 1.0e-6;
  const Real theta_par = m_mode*PI/static_cast<Real>(N1);
  const Real kwave_par = m_mode*PI/L1;
  const Real lambda_par = -(2.0*Dpar/(dx1*dx1))*(1.0 - std::cos(theta_par));
  const int nsuper = 8;
  const Real Tfinal = 0.7/std::fabs(lambda_par);     // decays to ~exp(-0.7)
  const Real dt_super = Tfinal/static_cast<Real>(nsuper);

  // seed a field-aligned cosine mode into `work` from each cell's GLOBAL physical x1.
  auto seed_par_mode = [&]() {
    par_for("acmb_seed_par", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x1 = size.d_view(m).x1min + (static_cast<Real>(i-is)+0.5)*size.d_view(m).dx1;
      bcc(m,IBX,k,j,i) = B0;     // bhat = xhat
      bcc(m,IBY,k,j,i) = 0.0;
      bcc(m,IBZ,k,j,i) = 0.0;
      Real tcell = T0 + amp*Kokkos::cos(kwave_par*(x1 - x1min));
      work(m,IDN,k,j,i) = rho0;
      work(m,IM1,k,j,i) = 0.0;
      work(m,IM2,k,j,i) = 0.0;
      work(m,IM3,k,j,i) = 0.0;
      work(m,IEN,k,j,i) = tcell*rho0/gm1 + me;
    });
  };

  // global L-inf temperature error vs the analytic exp(lambda_par t) decay.
  auto par_mode_error = [&](Real tphys) -> Real {
    auto h_w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work);
    Real decay = std::exp(lambda_par*tphys);
    Real e = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real x1 = size.h_view(m).x1min + (static_cast<Real>(i-is)+0.5)
                      *size.h_view(m).dx1;
            Real tcell = gm1*(h_w(m,IEN,k,j,i) - me)/h_w(m,IDN,k,j,i);
            Real analytic = T0 + amp*std::cos(kwave_par*(x1 - x1min))*decay;
            e = std::fmax(e, std::fabs(tcell - analytic));
          }
        }
      }
    }
    return e;
  };

  // (2a) WITH the per-substage exchange (pin != nullptr): reproduces the global decay.
  {
    seed_par_mode();
    const Real e_init = GlobalReduce(interior_vol_energy(), false);
    AnisotropicConductionOperator op(pmbp, pin, work, bcc, gamma, par);
    for (int n = 0; n < nsuper; ++n) {
      parabolic::OperatorSplitStep(integ, op, work, dt_super);
    }
    const Real max_err = GlobalReduce(par_mode_error(Tfinal), true);
    // uniform mesh: exact discrete eigenmode (tight); multilevel: the c/f prolongation is
    // only O(dx^2)-consistent so loosen, but it must still track the global decay.
    const Real tol = (multilevel ? 1.5e-1 : 2.0e-2)*amp;
    test.CheckTrue(max_err < tol,
        "field-aligned cosine decay reproduced across blocks (exchange ON)");
    // (6) conservation across the decomposition (volume-weighted, insulated global box).
    const Real e_final = GlobalReduce(interior_vol_energy(), false);
    test.CheckNear(e_final, e_init, 0.0, 1.0e-9*std::fabs(e_init),
        "insulated box conserves volume-weighted total energy across decomposition");
    if (global_variable::my_rank == 0) {
      std::cout << "### aniso_mb: nmb_total=" << pmy_mesh_->nmb_total
                << " multilevel=" << multilevel
                << " par-mode max_err/amp(ON)=" << (max_err/amp) << std::endl;
    }
  }

  // (2b) WITHOUT the exchange (pin == nullptr): each block self-insulates -> large error.
  // Only a meaningful discriminator when block faces are interior to the mode (uniform
  // mesh, mode crossing a block boundary); skip the assertion under SMR where the static
  // refinement geometry makes the "no-exchange" baseline ill-defined.
  if (!multilevel) {
    seed_par_mode();
    AnisotropicConductionOperator op_noex(pmbp, nullptr, work, bcc, gamma, par);
    for (int n = 0; n < nsuper; ++n) {
      parabolic::OperatorSplitStep(integ, op_noex, work, dt_super);
    }
    const Real max_err_noex = GlobalReduce(par_mode_error(Tfinal), true);
    test.CheckTrue(max_err_noex > 5.0*2.0e-2*amp,
        "RED: without the exchange each block self-insulates (decay NOT reproduced)");
    if (global_variable::my_rank == 0) {
      std::cout << "### aniso_mb: par-mode max_err/amp(OFF)=" << (max_err_noex/amp)
                << std::endl;
    }
  }

  // ======== (3) CROSS-FIELD insulation: perpendicular mode barely decays ========
  // uniform B = B0 xhat, cosine mode along yhat -> D_perp << D_par; over the SAME Tfinal
  // the perpendicular peak retains nearly all its amplitude (heat stays on field lines).
  Real perp_decay_measured = 1.0;
  {
    const Real kwave_perp = m_mode*PI/L2;
    par_for("acmb_seed_perp", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real x2 = size.d_view(m).x2min + (static_cast<Real>(j-js)+0.5)*size.d_view(m).dx2;
      bcc(m,IBX,k,j,i) = B0;
      bcc(m,IBY,k,j,i) = 0.0;
      bcc(m,IBZ,k,j,i) = 0.0;
      Real tcell = T0 + amp*Kokkos::cos(kwave_perp*(x2 - x2min));
      work(m,IDN,k,j,i) = rho0;
      work(m,IM1,k,j,i) = 0.0;
      work(m,IM2,k,j,i) = 0.0;
      work(m,IM3,k,j,i) = 0.0;
      work(m,IEN,k,j,i) = tcell*rho0/gm1 + me;
    });
    auto perp_peak = [&]() -> Real {
      auto h_w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work);
      Real p = 0.0;
      for (int m = 0; m <= nmb1; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
              Real tcell = gm1*(h_w(m,IEN,k,j,i) - me)/h_w(m,IDN,k,j,i);
              p = std::fmax(p, std::fabs(tcell - T0));
            }
          }
        }
      }
      return p;
    };
    const Real peak_init = GlobalReduce(perp_peak(), true);
    AnisotropicConductionOperator op(pmbp, pin, work, bcc, gamma, par);
    for (int n = 0; n < nsuper; ++n) {
      parabolic::OperatorSplitStep(integ, op, work, dt_super);
    }
    const Real peak_final = GlobalReduce(perp_peak(), true);
    perp_decay_measured = peak_final/peak_init;
    // analytic perpendicular decay over Tfinal (for reference / the substage sweep below)
    const Real theta_perp = m_mode*PI/static_cast<Real>(N2);
    const Real lambda_perp = -(2.0*Dperp/(dx2*dx2))*(1.0 - std::cos(theta_perp));
    const Real analytic_perp = std::exp(lambda_perp*Tfinal);
    test.CheckTrue(perp_decay_measured > 0.99,
        "cross-field (perpendicular) mode barely decays -- heat stays on field lines");
    if (global_variable::my_rank == 0) {
      std::cout << "### aniso_mb: perp decay measured=" << perp_decay_measured
                << " analytic=" << analytic_perp
                << " (par decay=" << std::exp(lambda_par*Tfinal) << ")" << std::endl;
    }
  }

  // ========= (4) RING: heat follows circular field lines across blocks =========
  // circular frozen field bhat = (-y, x)/r; hot Gaussian patch on the ring at phi=0.
  // The ring crosses every block, so azimuthal spread requires the cross-block exchange.
  {
    const Real r0 = 0.5*std::fmin(L1, L2)*0.6;   // a ring well inside the domain
    const Real xc = 0.5*(x1min + x1max);
    const Real yc = 0.5*(x2min + x2max);
    const Real sigma_r = 2.5*std::fmax(dx1, dx2);
    const Real sigma_a = r0*0.30;                // azimuthal Gaussian width (arc length)
    const Real dT_peak = 1.0e-2;
    auto tring = [&](Real x, Real y) -> Real {
      Real dx = x - xc, dy = y - yc;
      Real r = std::sqrt(dx*dx + dy*dy);
      Real phi = std::atan2(dy, dx);             // patch centred at phi=0 (+x axis)
      Real kr = std::exp(-SQR((r - r0)/sigma_r));
      Real ka = std::exp(-SQR((phi*r0)/sigma_a));
      return T0 + dT_peak*kr*ka;
    };
    auto h_b = Kokkos::create_mirror_view(bcc);
    auto h_w = Kokkos::create_mirror_view(work);
    for (int m = 0; m <= nmb1; ++m) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          Real y = size.h_view(m).x2min + (static_cast<Real>(j-js)+0.5)
                   *size.h_view(m).dx2 - yc;
          for (int i = 0; i < n1; ++i) {
            Real x = size.h_view(m).x1min + (static_cast<Real>(i-is)+0.5)
                     *size.h_view(m).dx1 - xc;
            Real r = std::sqrt(x*x + y*y);
            Real inv = (r > 1.0e-12) ? 1.0/r : 0.0;
            h_b(m,IBX,k,j,i) = -B0*y*inv;        // bhat = (-y, x)/r (azimuthal)
            h_b(m,IBY,k,j,i) =  B0*x*inv;
            h_b(m,IBZ,k,j,i) = 0.0;
            Real tcell = tring(x + xc, y + yc);
            h_w(m,IDN,k,j,i) = rho0;
            h_w(m,IM1,k,j,i) = 0.0;
            h_w(m,IM2,k,j,i) = 0.0;
            h_w(m,IM3,k,j,i) = 0.0;
            h_w(m,IEN,k,j,i) = tcell*rho0/gm1 + me;
          }
        }
      }
    }
    Kokkos::deep_copy(bcc, h_b);
    Kokkos::deep_copy(work, h_w);

    AnisotropicConductionOperator op(pmbp, pin, work, bcc, gamma, par);
    // Run long enough for clear AZIMUTHAL spread along the ring but short enough
    // that the (kperp/kpar ~ 1e-3) cross-field RADIAL leak stays sub-cell.  The
    // parallel spread after time t is ~ sqrt(Dpar t); reaching phi ~ 0.5 rad at
    // r0 needs t ~ (0.5 r0)^2/Dpar.  Express in explicit-dt units (dt_exp ~ fac dx^2/
    // Dpar): t ~ (0.5 r0)^2/(fac dx^2) * dt_exp.  Step it in stable RKL2 super-steps.
    const Real dt_exp_ring = op.ExplicitStableDt();
    const Real arc = 0.5*r0;
    const Real fac2d = 0.25;
    const Real t_ring = (arc*arc)/(fac2d*std::fmax(dx1, dx2)*std::fmax(dx1, dx2))
                        *dt_exp_ring;
    const int nring = 12;
    const Real dt_ring = t_ring/static_cast<Real>(nring);
    for (int n = 0; n < nring; ++n) {
      parabolic::OperatorSplitStep(integ, op, work, dt_ring);
    }

    // measure heated cells AZIMUTHALLY far from the patch (|phi| large, r ~ r0) vs the
    // RADIAL far-field leak (|r - r0| large, phi ~ 0).  band: |r - r0| < 1.5 sigma_r.
    auto azim_radial = [&](Real &dT_azim, Real &dT_rad) {
      auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work);
      dT_azim = 0.0; dT_rad = 0.0;
      for (int m = 0; m <= nmb1; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            Real y = size.h_view(m).x2min + (static_cast<Real>(j-js)+0.5)
                     *size.h_view(m).dx2 - yc;
            for (int i = is; i <= ie; ++i) {
              Real x = size.h_view(m).x1min + (static_cast<Real>(i-is)+0.5)
                       *size.h_view(m).dx1 - xc;
              Real r = std::sqrt(x*x + y*y);
              Real phi = std::atan2(y, x);
              Real tcell = gm1*(hh(m,IEN,k,j,i) - me)/hh(m,IDN,k,j,i);
              Real dT = tcell - T0;
              bool on_ring = std::fabs(r - r0) < 1.5*sigma_r;
              if (on_ring && std::fabs(phi) > 0.4) {       // azimuthally far along ring
                dT_azim = std::fmax(dT_azim, dT);
              }
              if (std::fabs(phi) < 0.15 && std::fabs(r - r0) > 3.0*sigma_r) {  // radial
                dT_rad = std::fmax(dT_rad, std::fabs(dT));
              }
            }
          }
        }
      }
    };
    Real dT_azim, dT_rad;
    azim_radial(dT_azim, dT_rad);
    dT_azim = GlobalReduce(dT_azim, true);
    dT_rad  = GlobalReduce(dT_rad, true);
    test.CheckTrue(dT_azim > 0.02*dT_peak,
        "ring: heat spreads AZIMUTHALLY along the field across block boundaries");
    test.CheckTrue(dT_rad < 0.5*dT_azim,
        "ring: cross-field RADIAL leak stays << azimuthal spread (heat on field lines)");
    if (global_variable::my_rank == 0) {
      std::cout << "### aniso_mb ring: r0=" << r0 << " dT_azim=" << dT_azim
                << " dT_rad=" << dT_rad << " ratio=" << (dT_rad/dT_azim) << std::endl;
    }
  }

  // ======= (5) cross-field leakage vs RKL2 SUBSTAGE count (instrumentation) =======
  // Sweep the super-step size (=> different RKL2 stage counts) on the perpendicular mode
  // and report the cross-field leakage 1 - decay.  The STS super-step is unconditionally
  // stable, so the leakage must stay BOUNDED and roughly converged as the stage count
  // grows -- this informs the implicit-fallback trigger (#114/#115):
  // a leakage that blew up with stage count would signal STS mis-resolves the operator.
  {
    const Real kwave_perp = m_mode*PI/L2;
    auto seed_perp = [&]() {
      par_for("acmb_seed_perp2", DevExeSpace(), 0, nmb1, 0, n3-1, 0, n2-1, 0, n1-1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real x2 = size.d_view(m).x2min + (static_cast<Real>(j-js)+0.5)*size.d_view(m).dx2;
        bcc(m,IBX,k,j,i) = B0;
        bcc(m,IBY,k,j,i) = 0.0;
        bcc(m,IBZ,k,j,i) = 0.0;
        Real tcell = T0 + amp*Kokkos::cos(kwave_perp*(x2 - x2min));
        work(m,IDN,k,j,i) = rho0;
        work(m,IM1,k,j,i) = 0.0;
        work(m,IM2,k,j,i) = 0.0;
        work(m,IM3,k,j,i) = 0.0;
        work(m,IEN,k,j,i) = tcell*rho0/gm1 + me;
      });
    };
    auto perp_peak = [&]() -> Real {
      auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), work);
      Real p = 0.0;
      for (int m = 0; m <= nmb1; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            for (int i = is; i <= ie; ++i) {
              Real tcell = gm1*(hh(m,IEN,k,j,i) - me)/hh(m,IDN,k,j,i);
              p = std::fmax(p, std::fabs(tcell - T0));
            }
          }
        }
      }
      return p;
    };
    AnisotropicConductionOperator op(pmbp, pin, work, bcc, gamma, par);
    const Real dt_exp = op.ExplicitStableDt();
    test.CheckTrue(dt_exp > 0.0, "anisotropic conduction explicit dt is positive");
    const int factors[3] = {10, 40, 160};
    Real leak[3];
    int stages[3];
    bool bounded = true, converged = true;
    for (int s = 0; s < 3; ++s) {
      seed_perp();
      const Real peak_init = GlobalReduce(perp_peak(), true);
      const Real dts = factors[s]*dt_exp;
      stages[s] = integ.NumStages(dts, dt_exp);
      // advance the SAME total time in steps of dts (=> same physical leakage target)
      const int nstep = std::max(1, static_cast<int>(std::ceil(Tfinal/dts)));
      for (int n = 0; n < nstep; ++n) {
        parabolic::OperatorSplitStep(integ, op, work, dts);
      }
      const Real peak_final = GlobalReduce(perp_peak(), true);
      leak[s] = 1.0 - peak_final/peak_init;       // cross-field leakage fraction
      if (!(leak[s] >= -1.0e-6 && leak[s] < 0.05)) { bounded = false; }
    }
    test.CheckTrue(stages[2] > stages[0],
        "RKL2 substage count grows with the super-step size (STS sweep is meaningful)");
    test.CheckTrue(bounded,
        "cross-field leakage stays small + bounded across the RKL2 substage sweep");
    // leakage of the largest-stage run should not exceed the smallest-stage run by much
    if (std::fabs(leak[2] - leak[0]) > 0.02) { converged = false; }
    test.CheckTrue(converged,
        "cross-field leakage is converged across substage sweep (implicit-trigger ok)");
    if (global_variable::my_rank == 0) {
      std::cout << "### aniso_mb leakage-vs-substages:";
      for (int s = 0; s < 3; ++s) {
        std::cout << " [s=" << stages[s] << " leak=" << leak[s] << "]";
      }
      std::cout << std::endl;
    }
  }

  test.Finish();
  // All checks ran on local arrays; the athinput sets nlim=tlim=0, so returning lets the
  // program shut down through main (MPI_Finalize).  Finish() std::exit'd nonzero on
  // any failed check.
  return;
}
