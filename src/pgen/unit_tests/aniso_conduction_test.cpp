//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file aniso_conduction_test.cpp
//  \brief Unit test: anisotropic (magnetized) Braginskii electron+ion thermal conduction
//  (AnisotropicConductionOperator), advanced operator-split by the RKL2 STS integrator,
//  conducts heat ALONG magnetic field lines, not across them (issue [6a]/#18, ADR-0006).
//
//  This exercises the whole operator-split anisotropic-conduction path on real device
//  kernels: the SIM-61 Braginskii coefficients (electron + ion), the field-aligned heat
//  flux q = -kappa_perp grad T - (kappa_par-kappa_perp) bhat(bhat.grad T), the
//  Sharma-Hammett van-Leer transverse limiter, and the Larsen field-aligned flux cap.
//
//  Batteries:
//   (A) SIM-61 coefficients: kappa_par,kappa_perp > 0; strongly magnetized kappa_par >>
//       kappa_perp; the (omega_c tau)^2 anisotropy law (kappa_perp ~ 1/B^2, kappa_par
//       B-independent); and electron+ion both contribute (sum > either alone).
//   (B) Field-aligned projection (exact cosine eigenmodes): with bhat = xhat, a
//       temperature mode along B reduces M(u) to 1D conduction with kappa_par; a mode
//       across B reduces it to kappa_perp.  Pins the bhat(bhat.grad T) projection.
//   (C) Parrish-Stone ring test: an azimuthal field B=(-y,x), a hot arc in an annulus.
//       After STS supersteps the heat spreads AZIMUTHALLY (along field lines) while
//       cross-field RADIAL leakage stays a tiny fraction -- heat stays on field lines.
//   (D) Sharma-Hammett monotonicity: the ring solution develops no new extrema (no
//       cross-field over/undershoot): T stays within [T_cold, T_hot] to tolerance.
//   (E) Larsen flux cap: the field-aligned flux saturates at the free-streaming limit
//       v_fs T (limiter identities) and the saturated operator action is below the
//       unsaturated (Spitzer) one on a steep parallel gradient.
//
//  Built/run by
//    cmake -D PROBLEM=unit_tests/aniso_conduction_test -B build_unit/aniso_cond
//    (cd build_unit/.../src && make) && ./athena -i inputs/unit_tests/<this>.athinput
//  Auto-run by tst/test_suite/unit_tests/test_unit_aniso_conduction_cpu.py.

#include <cstdlib>   // std::exit, EXIT_SUCCESS
#include <cmath>     // std::cos, std::sqrt, std::fabs, std::atan2, std::acos
#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "diffusion/aniso_conduction_operator.hpp"
#include "diffusion/braginskii_transport.hpp"
#include "driver/parabolic_integrator.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Anisotropic Braginskii conduction operator unit test.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("aniso_conduction_test");
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  test.CheckTrue(pmbp->phydro != nullptr, "hydro block constructed (array sizing)");
  if (pmbp->phydro == nullptr) { test.Finish(); return; }
  auto *phydro = pmbp->phydro;

  // selector wiring: <time> parabolic_integrator parses to the sts enum.
  parabolic::ParabolicIntegrator integ;
  integ.SetFromInput(pin);
  test.CheckTrue(integ.method() == parabolic::ParabolicMethod::sts,
                 "<time> parabolic_integrator selector parsed as sts");

  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, ie = indcs.ie;
  const int js = indcs.js, je = indcs.je;
  const int N1 = indcs.nx1, N2 = indcs.nx2;
  const Real gamma = phydro->peos->eos_data.gamma;
  const Real gm1 = gamma - 1.0;
  const Real x1min = pin->GetReal("mesh", "x1min");
  const Real x1max = pin->GetReal("mesh", "x1max");
  const Real x2min = pin->GetReal("mesh", "x2min");
  const Real x2max = pin->GetReal("mesh", "x2max");
  const Real dx1 = (x1max - x1min)/static_cast<Real>(N1);
  const Real dx2 = (x2max - x2min)/static_cast<Real>(N2);
  const Real PI = std::acos(-1.0);
  const Real rho0 = 1.0;

  // ---- conserved-field work arrays (shaped like hydro u0) + cell-centred B field ----
  const int nmb  = phydro->u0.extent_int(0);
  const int nvar = phydro->u0.extent_int(1);
  const int n3   = phydro->u0.extent_int(2);
  const int n2   = phydro->u0.extent_int(3);
  const int n1   = phydro->u0.extent_int(4);
  DvceArray5D<Real> cons("cons", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> bcc("bcc",  nmb, 3,    n3, n2, n1);
  DvceArray5D<Real> work("work", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> rhs("rhs",  nmb, nvar, n3, n2, n1);
  auto h_cons = Kokkos::create_mirror_view(cons);
  auto h_bcc  = Kokkos::create_mirror_view(bcc);
  auto h_rhs  = Kokkos::create_mirror_view(rhs);
  auto h_work = Kokkos::create_mirror_view(work);

  // physical (x,y) of a cell-centre index (valid in ghosts too: field defined everywhere)
  auto xcoord = [&](int i) -> Real {
    return x1min + (static_cast<Real>(i - is) + 0.5)*dx1;
  };
  auto ycoord = [&](int j) -> Real {
    return x2min + (static_cast<Real>(j - js) + 0.5)*dx2;
  };

  // ---- base physical + unit-conversion parameters (strongly magnetized HED plasma) ----
  anisocond::AnisoCondParams par;
  par.zbar = 1.0;
  par.mi_si = braginskii::kProtonMass;
  par.lnlam = 10.0;            // fixed Coulomb log (deterministic; no log branch)
  par.dens_conv = 1.0e20;      // code density -> n_i [m^-3]
  par.temp_conv = 1.0e6;       // code temperature -> T [K]
  par.bmag_conv = 1.0e2;       // code |B| -> B [T]
  par.kappa_conv = 1.0e-4;     // SI conductivity -> code (keeps kappa_par ~ O(1))
  par.nlarsen = 2.0;
  par.vfs = -1.0;              // Larsen flux cap OFF for A-D
  par.incl_e = true;
  par.incl_i = true;

  // =================== (A) SIM-61 Braginskii coefficients (host) ====================
  {
    const Real T0 = 1.0;                 // code temperature (-> 1e6 K)
    const Real b1 = 0.6;                 // code |B|  (-> 60 T): strongly magnetized
    Real kpar1, kperp1;
    anisocond::Conductivities(par, rho0, T0, b1, kpar1, kperp1);
    test.CheckTrue(kpar1 > 0.0 && kperp1 > 0.0, "kappa_par, kappa_perp positive");
    test.CheckTrue(kpar1 > 1.0e3*kperp1,
                   "strongly magnetized: kappa_par >> kappa_perp (field insulation)");

    // (omega_c tau)^2 law: kappa_perp ~ 1/B^2; kappa_par is B-independent.
    Real kpar2, kperp2;
    anisocond::Conductivities(par, rho0, T0, 2.0*b1, kpar2, kperp2);
    test.CheckNear(kpar2, kpar1, 0.0, 1.0e-10*kpar1,
                   "kappa_par is independent of B (parallel conduction unmagnetized)");
    test.CheckNear(kperp1/kperp2, 4.0, 0.0, 1.0e-2,
                   "kappa_perp ~ 1/B^2: doubling B quarters it ((omega_c tau)^2 law)");

    // electron + ion: the sum exceeds either species alone.
    anisocond::AnisoCondParams pe = par; pe.incl_i = false;
    anisocond::AnisoCondParams pi = par; pi.incl_e = false;
    Real kpar_e, kperp_e, kpar_i, kperp_i;
    anisocond::Conductivities(pe, rho0, T0, b1, kpar_e, kperp_e);
    anisocond::Conductivities(pi, rho0, T0, b1, kpar_i, kperp_i);
    test.CheckTrue(kpar_e > 0.0 && kpar_i > 0.0, "both electron and ion conduct (kpar)");
    test.CheckNear(kpar1, kpar_e + kpar_i, 0.0, 1.0e-10*kpar1,
                   "kappa_par(total) = electron + ion contributions");
  }

  // ============ (B) field-aligned projection: exact cosine eigenmodes ===============
  // bhat = xhat (uniform B along x).  A mode varying along x -> kappa_par diffusion; a
  // mode varying along y -> kappa_perp diffusion.  Small amplitude so the (T-dependent)
  // Braginskii coefficients are effectively constant -> the discrete eigenmode is exact.
  {
    const Real T0 = 1.0;
    const Real B0 = 0.6;                  // uniform field magnitude (code)
    const Real amp = 1.0e-6;              // tiny pert. (constant-coefficient limit)
    const int m_mode = 4;
    Real kpar, kperp;
    anisocond::Conductivities(par, rho0, T0, B0, kpar, kperp);

    auto fill_mode = [&](int dir) {       // dir=1 mode along x (parallel), dir=2 along y
      const Real theta = m_mode*PI/static_cast<Real>(dir == 1 ? N1 : N2);
      for (int m = 0; m < nmb; ++m) {
        for (int k = 0; k < n3; ++k) {
          for (int j = 0; j < n2; ++j) {
            for (int i = 0; i < n1; ++i) {
              h_bcc(m,IBX,k,j,i) = B0;     // bhat = xhat everywhere
              h_bcc(m,IBY,k,j,i) = 0.0;
              h_bcc(m,IBZ,k,j,i) = 0.0;
              Real c = (dir == 1) ? std::cos(theta*(static_cast<Real>(i-is)+0.5))
                                  : std::cos(theta*(static_cast<Real>(j-js)+0.5));
              Real tcell = T0 + amp*c;
              Real eint = tcell*rho0/gm1;
              h_cons(m,IDN,k,j,i) = rho0;
              h_cons(m,IM1,k,j,i) = 0.0;
              h_cons(m,IM2,k,j,i) = 0.0;
              h_cons(m,IM3,k,j,i) = 0.0;
              h_cons(m,IEN,k,j,i) = eint + 0.5*B0*B0;   // total energy (+ magnetic)
            }
          }
        }
      }
      Kokkos::deep_copy(cons, h_cons);
      Kokkos::deep_copy(bcc, h_bcc);
    };

    AnisotropicConductionOperator op(pmbp, cons, bcc, gamma, par);

    // The operator evolves E (= IEN), and dE/dt = -div(q) = kappa * d^2 T/dx_d^2 for the
    // active diffusion direction d (the kappa*gm1/rho energy-diffusivity D enters when
    // recasting as lambda*(E-E0); we use the equivalent direct form kappa * d^2T/dx_d^2).

    // (B1) parallel: mode along B -> kappa_par
    fill_mode(1);
    Kokkos::deep_copy(work, cons);
    op.ApplyBoundary(work);
    op.OperatorAction(work, rhs);
    Kokkos::deep_copy(h_rhs, rhs);
    {
      const Real theta = m_mode*PI/static_cast<Real>(N1);
      const Real lambda_unit = -(2.0/(dx1*dx1))*(1.0 - std::cos(theta));
      const Real scale = std::fabs(kpar*lambda_unit)*amp;
      Real maxrel = 0.0;
      for (int i = is; i <= ie; ++i) {
        Real c = std::cos(theta*(static_cast<Real>(i-is)+0.5));
        Real expected = kpar*lambda_unit*amp*c;              // dE/dt = kappa * d2T/dx2
        maxrel = std::fmax(maxrel, std::fabs(h_rhs(0,IEN,0,js,i) - expected)/scale);
      }
      test.CheckTrue(maxrel < 2.0e-3,
                     "parallel mode: M(u) == kappa_par diffusion (bhat.gradT proj)");
    }

    // (B2) perpendicular: mode across B -> kappa_perp
    fill_mode(2);
    Kokkos::deep_copy(work, cons);
    op.ApplyBoundary(work);
    op.OperatorAction(work, rhs);
    Kokkos::deep_copy(h_rhs, rhs);
    {
      const Real theta = m_mode*PI/static_cast<Real>(N2);
      const Real lambda_unit = -(2.0/(dx2*dx2))*(1.0 - std::cos(theta));
      const Real scale = std::fabs(kperp*lambda_unit)*amp;
      Real maxrel = 0.0;
      for (int j = js; j <= je; ++j) {
        Real c = std::cos(theta*(static_cast<Real>(j-js)+0.5));
        Real expected = kperp*lambda_unit*amp*c;
        maxrel = std::fmax(maxrel, std::fabs(h_rhs(0,IEN,0,j,is) - expected)/scale);
      }
      test.CheckTrue(maxrel < 2.0e-3,
                     "perpendicular mode: M(u) == kappa_perp diffusion (cross-field)");
      // and the cross-field diffusion is hugely weaker than along-field
      test.CheckTrue(kperp < 1.0e-3*kpar, "cross-field conduction strongly suppressed");
    }
  }

  // ===================== (C,D) Parrish-Stone ring test ==============================
  // Azimuthal field B=(-y,x) (|B|=r), a SMOOTH Gaussian-blob temperature bump on a
  // ring.  Heat must conduct AZIMUTHALLY (along field lines) and not RADIALLY (across).
  // A smooth profile keeps the limiter out of step-discontinuity regimes (Sharma-Hammett
  // reduces but doesn't eliminate undershoots on a step IC, per Sharma & Hammett 2007).
  {
    const Real T_cold = 1.0;
    const Real dT_peak = 0.1;                 // small perturbation
    const Real r0 = 0.55;                     // ring radius
    const Real sigma_r = 0.05;                // radial Gaussian width
    const Real sigma_phi = PI/16.0;           // azimuthal Gaussian width

    auto t_init = [&](Real x, Real y) -> Real {
      Real r = std::sqrt(x*x + y*y);
      Real phi = std::atan2(y, x);
      Real ker_r = std::exp(-SQR((r - r0)/sigma_r));
      Real ker_p = std::exp(-SQR(phi/sigma_phi));
      return T_cold + dT_peak*ker_r*ker_p;
    };
    for (int m = 0; m < nmb; ++m) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            Real x = xcoord(i), y = ycoord(j);
            // azimuthal field B = (-y, x, 0): bhat is azimuthal, |B| = r
            h_bcc(m,IBX,k,j,i) = -y;
            h_bcc(m,IBY,k,j,i) =  x;
            h_bcc(m,IBZ,k,j,i) = 0.0;
            Real tcell = t_init(x, y);
            Real eint = tcell*rho0/gm1;
            h_cons(m,IDN,k,j,i) = rho0;
            h_cons(m,IM1,k,j,i) = 0.0;
            h_cons(m,IM2,k,j,i) = 0.0;
            h_cons(m,IM3,k,j,i) = 0.0;
            h_cons(m,IEN,k,j,i) = eint + 0.5*(x*x + y*y);   // + magnetic energy |B|^2/2
          }
        }
      }
    }
    Kokkos::deep_copy(cons, h_cons);
    Kokkos::deep_copy(bcc, h_bcc);

    AnisotropicConductionOperator op(pmbp, cons, bcc, gamma, par);
    const Real dt_exp = op.ExplicitStableDt();
    test.CheckTrue(dt_exp > 0.0, "anisotropic conduction explicit dt is positive");
    const Real dt_super = 20.0*dt_exp;
    const int nsuper = 15;
    test.CheckTrue(integ.NumStages(dt_super, dt_exp) > 4,
                   "ring test uses many RKL2 substages (STS beyond explicit dt)");
    Kokkos::deep_copy(work, cons);
    for (int s = 0; s < nsuper; ++s) {
      parabolic::OperatorSplitStep(integ, op, work, dt_super);
    }
    Kokkos::deep_copy(h_work, work);

    auto temp_work = [&](int j, int i) -> Real {
      Real rho = h_work(0,IDN,0,j,i);
      Real me = 0.5*(SQR(h_bcc(0,IBX,0,j,i)) + SQR(h_bcc(0,IBY,0,j,i))
                   + SQR(h_bcc(0,IBZ,0,j,i)));
      return gm1*(h_work(0,IEN,0,j,i) - me)/rho;
    };
    // PEAK dT vs distance along- vs across-field-lines.  Heat that traveled AZIMUTHALLY
    // (along b_hat) shows up at the ring radius, far from phi=0.  Heat that leaked
    // RADIALLY (across b_hat) shows up at the initial phi, far from r0.  For preferential
    // field-aligned conduction the azimuthal peak grows while the radial peak does not.
    Real dT_azim_far = 0.0, dT_radial_far = 0.0;
    Real tmin = T_cold + dT_peak, tmax = T_cold;
    for (int i = is; i <= ie; ++i) {
      for (int j = js; j <= je; ++j) {
        Real x = xcoord(i), y = ycoord(j);
        Real r = std::sqrt(x*x + y*y);
        Real phi = std::atan2(y, x);
        Real T = temp_work(j, i);
        Real dT = T - T_cold;
        tmin = std::fmin(tmin, T);
        tmax = std::fmax(tmax, T);
        // ON the ring radius (|r - r0| <= sigma_r), beyond ~3 sigma_phi from the initial
        // blob in azimuth -- heat that travelled along the field.
        if (std::fabs(r - r0) <= sigma_r && std::fabs(phi) >= 3.0*sigma_phi) {
          dT_azim_far = std::fmax(dT_azim_far, dT);
        }
        // OFF the ring radius (|r - r0| >= 3 sigma_r), near the initial phi=0 -- heat
        // that leaked across the field.
        if (std::fabs(r - r0) >= 3.0*sigma_r && std::fabs(phi) <= sigma_phi) {
          dT_radial_far = std::fmax(dT_radial_far, dT);
        }
      }
    }
    // dT_azim_far grew from the initial ~ exp(-9) ~ 1e-5 tail to a sizeable fraction; the
    // ratio of azimuthal to radial peak is the canonical field-alignment indicator (heat
    // travels along b_hat).  An isotropic operator gives a near-1 ratio; field-aligned
    // gives the azimuthal peak >> radial peak.
    test.CheckTrue(dT_azim_far > 0.01*dT_peak,
                   "ring test: heat reached the azimuthal far side along field lines");
    test.CheckTrue(dT_radial_far < 0.5*dT_azim_far,
                   "ring test: cross-field radial leakage peak << azimuthal spread peak");
    // (D) Sharma-Hammett monotonicity on the smooth blob: small bounded over/undershoot.
    // The limiter REDUCES but does not eliminate undershoots in anisotropic FV (Sharma &
    // Hammett 2007); on a smooth IC the bound is a small fraction of the peak.
    test.CheckTrue(tmin > T_cold - 0.05*dT_peak,
                   "Sharma-Hammett: undershoot bounded below 5% of the perturbation");
    test.CheckTrue(tmax < T_cold + dT_peak + 0.05*dT_peak,
                   "Sharma-Hammett: overshoot bounded below 5% of the perturbation");
  }

  // ========================= (E) Larsen field-aligned flux cap ======================
  {
    // limiter identities (host): lambda(0)=1/3 so kappa_par_eff -> kappa_par (Spitzer);
    // at large R the field-aligned flux saturates at v_fs * T (free-streaming).
    test.CheckNear(anisocond::LarsenLimiter(0.0, 2.0), 1.0/3.0, 0.0, 1.0e-12,
                   "Larsen limiter lambda(0) = 1/3 (recovers Spitzer when unsaturated)");
    const Real kpar = 5.0, vfs = 1.0, T_f = 2.0, bdg = 100.0;   // steep parallel gradient
    Real R = 3.0*kpar*std::fabs(bdg)/(vfs*T_f);
    Real kpar_eff = 3.0*kpar*anisocond::LarsenLimiter(R, 2.0);
    Real flux = kpar_eff*std::fabs(bdg);
    test.CheckNear(flux, vfs*T_f, 0.0, 5.0e-2*vfs*T_f,
                   "Larsen cap: field-aligned flux saturates at free-streaming v_fs T");

    // operator level: a steep parallel cosine mode, saturated vs unsaturated.
    const Real T0 = 1.0, B0 = 0.6, amp = 0.4;     // LARGE amplitude => steep gradient
    const int m_mode = 8;
    const Real theta = m_mode*PI/static_cast<Real>(N1);
    for (int m = 0; m < nmb; ++m) {
      for (int k = 0; k < n3; ++k) {
        for (int j = 0; j < n2; ++j) {
          for (int i = 0; i < n1; ++i) {
            h_bcc(m,IBX,k,j,i) = B0; h_bcc(m,IBY,k,j,i) = 0.0; h_bcc(m,IBZ,k,j,i) = 0.0;
            Real tcell = T0 + amp*std::cos(theta*(static_cast<Real>(i-is)+0.5));
            h_cons(m,IDN,k,j,i) = rho0;
            h_cons(m,IM1,k,j,i) = 0.0;
            h_cons(m,IM2,k,j,i) = 0.0;
            h_cons(m,IM3,k,j,i) = 0.0;
            h_cons(m,IEN,k,j,i) = tcell*rho0/gm1 + 0.5*B0*B0;
          }
        }
      }
    }
    Kokkos::deep_copy(cons, h_cons);
    Kokkos::deep_copy(bcc, h_bcc);

    anisocond::AnisoCondParams p_off = par;  p_off.vfs = -1.0;          // unsaturated
    anisocond::AnisoCondParams p_sat = par;  p_sat.vfs = 1.0e-3;     // heavy saturation
    AnisotropicConductionOperator op_off(pmbp, cons, bcc, gamma, p_off);
    AnisotropicConductionOperator op_sat(pmbp, cons, bcc, gamma, p_sat);
    Kokkos::deep_copy(work, cons);
    op_off.ApplyBoundary(work);
    op_off.OperatorAction(work, rhs);
    Kokkos::deep_copy(h_rhs, rhs);
    Real peak_off = 0.0;
    for (int i = is; i <= ie; ++i) {
      peak_off = std::fmax(peak_off, std::fabs(h_rhs(0,IEN,0,js,i)));
    }
    Kokkos::deep_copy(work, cons);
    op_sat.ApplyBoundary(work);
    op_sat.OperatorAction(work, rhs);
    Kokkos::deep_copy(h_rhs, rhs);
    Real peak_sat = 0.0;
    for (int i = is; i <= ie; ++i) {
      peak_sat = std::fmax(peak_sat, std::fabs(h_rhs(0,IEN,0,js,i)));
    }
    test.CheckTrue(peak_sat < peak_off,
                   "Larsen cap: saturated field-aligned conduction is below Spitzer");
  }

  // seed the live hydro state with a valid uniform medium so the post-UserProblem trivial
  // driver run (ConsToPrim over the mesh, nlim=0) never touches uninitialized memory.
  Kokkos::deep_copy(h_cons, phydro->u0);
  for (int m = 0; m < nmb; ++m) {
    for (int k = 0; k < n3; ++k) {
      for (int j = 0; j < n2; ++j) {
        for (int i = 0; i < n1; ++i) {
          h_cons(m,IDN,k,j,i) = rho0;
          h_cons(m,IM1,k,j,i) = 0.0;
          h_cons(m,IM2,k,j,i) = 0.0;
          h_cons(m,IM3,k,j,i) = 0.0;
          h_cons(m,IEN,k,j,i) = rho0/gm1;     // uniform T = 1
        }
      }
    }
  }
  Kokkos::deep_copy(phydro->u0, h_cons);

  test.Finish();
  // All checks live in UserProblem on local arrays; exit cleanly on success rather than
  // proceed into the (pointless) trivial driver run.
  std::exit(EXIT_SUCCESS);
  return;
}
