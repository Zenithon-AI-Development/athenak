//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file resb_bphi_multiblock_test.cpp
//  \brief Unit test: the operator-split CYLINDRICAL resistive B_phi diffusion operator
//  (ResistiveBphiOperator + RKL2 STS) reproduces the analytic Bessel-J_1 eigenmode decay
//  while running across MeshBlock / MPI-rank / AMR coarse-fine boundaries, via the shared
//  per-substage ghost-exchange helper MeshBoundaryValuesCC::SyncParabolicGhosts
//  (issue [A6]/#113, ADR-0004/ADR-0001).
//
//  This is the multi-block / multi-rank / AMR verification of the wired operator.  The
//  single-block physics (the -eta B_phi/r^2 curl-curl term, the conservative
//  (1/r)d_r(r*flux) radial form, the antisymmetric axis ghost) is pinned by the user pgen
//  cyl_bphi_diffuse / test_verify_cyl_bphi_diffuse; here we add ONLY the thing #113
//  builds: the inter-substage neighbour ghost exchange that lets the operator span >1
//  MeshBlock, >1 MPI rank, and block-AMR without changing the cylindrical physics.
//
//  ANALYTIC ORACLE (Layer 1).  The axisymmetric resistive B_phi equation
//      dB_phi/dt = eta [ d^2B_phi/dr^2 + (1/r) dB_phi/dr - B_phi/r^2 ]
//  has the EXACT decaying eigenmode  B_phi(r,t) = A J_1(kr) exp(-eta kr^2 t)  (J_1 solves
//  Bessel's equation of order 1, so the -B_phi/r^2 curl-curl term is ESSENTIAL: drop it
//  and J_1 is no longer an eigenmode -> the shape distorts).  We choose the radial
//  wavenumber kr (from the athinput) so that BOTH radial domain ends sit on zeros of J_1,
//  so the operator's antisymmetric/Dirichlet-0 ghost is exact at the true domain faces:
//   * full-disk mesh r in [0, R] (the _cpu / _amr meshes): kr = j_{1,1}/R, so J_1(kr*0)=0
//     (the axis) and J_1(kr*R)=J_1(j_{1,1})=0 (the outer node).  This exercises the axis
//     antisymmetric ghost and the near-axis 1/r^2 stiffness.
//   * annular mesh r in [r0, R] (the _mpicpu mesh): kr = j_{1,2}/R with r0 = j_{1,1}/kr,
//     so J_1(kr*r0)=J_1(j_{1,1})=0 and J_1(kr*R)=J_1(j_{1,2})=0.  The annulus EXCLUDES
//     the axis, so the near-axis 1/r^2 term no longer dominates the explicit dt and the
//     per-rank ExplicitStableDt (no global min-dt reduction yet -- that is #114/[B1]) is
//     uniform across the radial blocks -> identical RKL2 stage counts -> the synchronous
//     cross-rank SyncParabolicGhosts cannot deadlock.  The curl-curl term is still
//     verified through the J_1 SHAPE preservation.
//
//  CHECKS (reductions MPI_Allreduce'd so every rank checks the SAME globals and Finish()
//  exits consistently everywhere):
//   (1) multi-block: nmb_total > 1, so the cross-block ghost exchange is exercised.
//   (2) WITH the per-substage exchange (pin != nullptr): the numerical B_phi matches the
//       analytic A J_1(kr) exp(-eta kr^2 t) profile to a tight Linf tolerance, NEAR the
//       inner radial end and AWAY from it alike -- only the correct cross-block exchange
//       reproduces the GLOBAL eigenmode.
//   (3) eigenmode SHAPE preserved: the numerical field is a near-constant multiple of the
//       initial J_1 profile (projection residual small).
//   (4) decay RATE matches exp(-eta kr^2 t): the J_1-projected decay factor matches it.
//   (5) WITHOUT the exchange (pin == nullptr, uniform mesh only): each block self-applies
//       its antisymmetric ghost at the INTERNAL block faces (sign-flipping B_phi where it
//       is large) -> the global Linf error is a large fraction of the amplitude (RED --
//       proves the exchange does the work, not the per-block physical fill).
//
//  Built/run by tst/test_suite/unit_tests/test_verify_resb_bphi_multiblock_cpu.py
//  (4 radial MeshBlocks, full disk, one rank), ..._mpicpu.py (4 annular radial MeshBlocks
//  across MPI ranks), and ..._amr_cpu.py (static-SMR, exercises the c/f exchange path).

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::exp, std::fabs, std::sqrt
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/resistive_bphi_operator.hpp"
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

// J_1(x) by its power series J_1(x) = sum_{m>=0} (-1)^m/(m!(m+1)!) (x/2)^{2m+1};
// converges rapidly for the |x| <= j_{1,2} ~ 7.02 used here (host-only IC + oracle).
double BesselJ1(double x) {
  double half = 0.5*x;
  double term = half;          // m = 0 term
  double sum = term;
  for (int m = 1; m < 80; ++m) {
    term *= -(half*half)/(static_cast<double>(m)*(m + 1));
    sum += term;
    if (std::fabs(term) <= 1.0e-18*std::fabs(sum)) break;
  }
  return sum;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Multi-block / MPI / AMR cylindrical resistive-B_phi eigenmode decay test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("resb_bphi_multiblock_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // (1) requires hydro (for array sizing) running on >1 MeshBlock, in cylindrical coords.
  bool have_hydro = (pmbp->phydro != nullptr);
  test.CheckTrue(have_hydro, "hydro block constructed (array sizing)");
  if (!have_hydro) { test.Finish(); return; }
  auto *phydro = pmbp->phydro;
  test.CheckTrue(pmy_mesh_->nmb_total > 1,
                 "more than one MeshBlock (cross-block ghost exchange is exercised)");
  test.CheckTrue(pmbp->pcoord->coord_system == CoordSystem::cylindrical,
                 "cylindrical coordinate system (the -eta B_phi/r^2 curl-curl operator)");

  // selector: <time> parabolic_integrator parses to the RKL2 STS backend
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  // ---- run / physics parameters (GLOBAL mesh) ----
  const Real eta    = pin->GetOrAddReal("problem", "eta", 0.01);
  const Real amp    = pin->GetOrAddReal("problem", "amp", 1.0);
  const Real kr     = pin->GetReal("problem", "kr");          // radial wavenumber
  const int  nsuper = pin->GetOrAddInteger("problem", "n_super", 24);
  const Real dt_fac = pin->GetOrAddReal("problem", "dt_fac", 200.0);
  const Real x1min  = pin->GetReal("mesh", "x1min");
  const Real x1max  = pin->GetReal("mesh", "x1max");
  const bool multilevel = pmy_mesh_->multilevel;

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb = phydro->u0.extent_int(0);
  const int n3  = phydro->u0.extent_int(2);
  const int n2  = phydro->u0.extent_int(3);
  const int n1  = phydro->u0.extent_int(4);
  const int nmb1 = pmbp->nmb_thispack - 1;
  auto size = pmbp->pmb->mb_size;

  // ---- single-component B_phi work field + cell-centred eta field (constant) ----
  DvceArray5D<Real> bphi("bphi", nmb, 1, n3, n2, n1);
  DvceArray4D<Real> etafld("eta", nmb, n3, n2, n1);

  const int IB = ResistiveBphiOperator::IBPHI;

  // seed the J_1(kr*r) eigenmode from each cell's GLOBAL physical radius (host fill: the
  // Bessel power series stays on the host; valid in the ghost zones too).
  auto seed = [&]() {
    auto h_bphi = Kokkos::create_mirror_view(bphi);
    auto h_eta  = Kokkos::create_mirror_view(etafld);
    for (int m = 0; m <= nmb1; ++m) {
      Real bx1min = size.h_view(m).x1min;
      Real bdx1   = size.h_view(m).dx1;
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            Real r = bx1min + (static_cast<Real>(i - is) + 0.5)*bdx1;
            h_bphi(m,IB,k,j,i) =
                amp*static_cast<Real>(BesselJ1(static_cast<double>(kr*r)));
            h_eta(m,k,j,i) = eta;
          }
        }
      }
    }
    Kokkos::deep_copy(bphi, h_bphi);
    Kokkos::deep_copy(etafld, h_eta);
  };

  // Linf error of the numerical B_phi vs the analytic A J_1(kr) exp(-eta kr^2 t) profile,
  // separately over the inner-radial band (r within 25% of x1min) and the rest.
  auto profile_error = [&](Real tphys, Real &e_inner, Real &e_outer) {
    auto h_bphi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bphi);
    Real decay = std::exp(-eta*kr*kr*tphys);
    Real r_band = x1min + 0.25*(x1max - x1min);
    e_inner = 0.0; e_outer = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      Real bx1min = size.h_view(m).x1min;
      Real bdx1   = size.h_view(m).dx1;
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real r = bx1min + (static_cast<Real>(i - is) + 0.5)*bdx1;
            Real analytic =
                amp*static_cast<Real>(BesselJ1(static_cast<double>(kr*r)))*decay;
            Real err = std::fabs(h_bphi(m,IB,k,j,i) - analytic);
            if (r < r_band) {
              e_inner = std::fmax(e_inner, err);
            } else {
              e_outer = std::fmax(e_outer, err);
            }
          }
        }
      }
    }
  };

  // J_1-projected decay factor  <B_num, B_init>/<B_init, B_init>  and the shape residual
  // (Linf of B_num - proj*B_init) -- a clean shape+rate diagnostic vs the mesh layout.
  auto project_decay = [&](Real &proj, Real &shape_resid) {
    auto h_bphi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bphi);
    Real num = 0.0, den = 0.0, peak = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      Real bx1min = size.h_view(m).x1min;
      Real bdx1   = size.h_view(m).dx1;
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real r = bx1min + (static_cast<Real>(i - is) + 0.5)*bdx1;
            Real binit = amp*static_cast<Real>(BesselJ1(static_cast<double>(kr*r)));
            num += h_bphi(m,IB,k,j,i)*binit;
            den += binit*binit;
            peak = std::fmax(peak, std::fabs(binit));
          }
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    Real gnum = num, gden = den;
    MPI_Allreduce(&num, &gnum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&den, &gden, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    num = gnum; den = gden;
#endif
    proj = (den > 0.0) ? num/den : 0.0;
    shape_resid = 0.0;
    for (int m = 0; m <= nmb1; ++m) {
      Real bx1min = size.h_view(m).x1min;
      Real bdx1   = size.h_view(m).dx1;
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          for (int i = is; i <= ie; ++i) {
            Real r = bx1min + (static_cast<Real>(i - is) + 0.5)*bdx1;
            Real binit = amp*static_cast<Real>(BesselJ1(static_cast<double>(kr*r)));
            shape_resid = std::fmax(shape_resid,
                std::fabs(h_bphi(m,IB,k,j,i) - proj*binit));
          }
        }
      }
    }
    shape_resid = GlobalReduce(shape_resid, true)/std::fmax(peak, 1.0e-30);
  };

  // ========= (2)-(4) WITH the per-substage exchange: reproduce the eigenmode =========
  Real t_total = 0.0, decay = 1.0;
  {
    seed();
    ResistiveBphiOperator op(pmbp, pin, bphi, etafld);
    const Real dt_exp = op.ExplicitStableDt();
    test.CheckTrue(dt_exp > 0.0, "resistive B_phi explicit-stable dt is positive");
    const Real dt_super = dt_fac*dt_exp;
    for (int s = 0; s < nsuper; ++s) {
      parabolic::OperatorSplitStep(integ, op, bphi, dt_super);
    }
    t_total = static_cast<Real>(nsuper)*dt_super;
    decay = std::exp(-eta*kr*kr*t_total);

    Real e_inner, e_outer;
    profile_error(t_total, e_inner, e_outer);
    e_inner = GlobalReduce(e_inner, true);
    e_outer = GlobalReduce(e_outer, true);
    // uniform mesh: the cross-block exchange makes the interior stencil identical to a
    // lone block, so it matches the analytic decay tightly; multilevel: c/f prolong is
    // only O(dx^2)-consistent so loosen, but it must still track the global decay.
    const Real tol = (multilevel ? 1.0e-1 : 3.0e-2)*amp;
    test.CheckTrue(e_outer < tol,
        "analytic J_1 decay reproduced away from the inner radial end (exchange ON)");
    test.CheckTrue(e_inner < tol,
        "analytic J_1 decay reproduced near the inner radial end (1/r^2, exch ON)");

    Real proj, shape_resid;
    project_decay(proj, shape_resid);
    const Real shape_tol = (multilevel ? 1.5e-1 : 5.0e-2);
    test.CheckTrue(shape_resid < shape_tol,
        "J_1 eigenmode SHAPE preserved across blocks (numerical ~ proj * J_1)");
    const Real rate_tol = (multilevel ? 8.0e-2 : 3.0e-2);
    test.CheckNear(proj, decay, rate_tol, 0.0,
        "J_1-projected decay factor matches analytic exp(-eta kr^2 t)");

    if (global_variable::my_rank == 0) {
      std::cout << "### resb_bphi_mb: nmb_total=" << pmy_mesh_->nmb_total
                << " multilevel=" << multilevel << " kr=" << kr
                << " dt_exp=" << dt_exp << " t=" << t_total << " decay=" << decay
                << " proj=" << proj << " e_in/amp=" << (e_inner/amp)
                << " e_out/amp=" << (e_outer/amp)
                << " shape_resid=" << shape_resid << std::endl;
    }
  }

  // ========= (5) WITHOUT the exchange (pin == nullptr): RED discriminator =========
  // Each block self-applies its antisymmetric ghost at the INTERNAL block faces, sign-
  // flipping B_phi where it is large -> the global decay is NOT reproduced.  Skipped on
  // SMR (the no-exchange baseline is ill-defined with a static refinement geometry).  The
  // no-exchange path never calls SyncParabolicGhosts, so it issues no MPI and cannot
  // deadlock even if per-rank stage counts were to differ.
  if (!multilevel) {
    seed();
    ResistiveBphiOperator op_noex(pmbp, nullptr, bphi, etafld);
    const Real dt_super = dt_fac*op_noex.ExplicitStableDt();
    for (int s = 0; s < nsuper; ++s) {
      parabolic::OperatorSplitStep(integ, op_noex, bphi, dt_super);
    }
    Real e_inner, e_outer;
    profile_error(t_total, e_inner, e_outer);
    Real e_noex = GlobalReduce(std::fmax(e_inner, e_outer), true);
    test.CheckTrue(e_noex > 5.0*3.0e-2*amp,
        "RED: without the exchange the blocks self-insulate (J_1 decay NOT reproduced)");
    if (global_variable::my_rank == 0) {
      std::cout << "### resb_bphi_mb: no-exchange max_err/amp=" << (e_noex/amp)
                << std::endl;
    }
  }

  test.Finish();
  // All checks ran on local arrays; the athinput sets nlim=tlim=0, so returning lets the
  // program shut down through main (MPI_Finalize).  Finish() std::exit'd nonzero on any
  // failed check.
  return;
}
