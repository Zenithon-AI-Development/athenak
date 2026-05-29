"""
Single-mode magneto-Rayleigh-Taylor verification (FLASH MagLIF benchmark 1, #39).

Runs the integrated MagLIF/Z-pinch problem generator (issue [10a]/#32) as a SUSTAINEDLY
accelerated liner implosion -- a dense cylindrical liner driven radially inward by a
prescribed constant load current, with a single axial mode seeded on the liner interface.
The magnetic pressure of the enclosed B_phi field (the light fluid) continuously
accelerates the dense liner (the heavy fluid) inward: the outer (liner/vacuum) interface
is therefore magneto-Rayleigh-Taylor (MRT) unstable, and the seeded single mode grows.
This is benchmark 1 of the FLASH MagLIF validation ladder -- the Phase-1 convergence point
(Ellison et al. 2025, arXiv:2504.10760).

Physics scope (Phase B, #116/[B3], ADR-0009): this benchmark now runs the FULL COUPLED
radiation-conduction-MHD stack -- cylindrical ideal-MHD (ADR-0004) + the prescribed-I(t)
circuit drive (ADR-0005 mode A) + the operator-split parabolic block (grey flux-limited
radiation diffusion + anisotropic Braginskii conduction on the live gas energy) Strang-
wrapped around the hyperbolic update, with the point-implicit matter-radiation coupling
outside the super-step (see inputs/maglif_mrt.athinput's <mhd> block).  Coefficients are
nondimensional code units chosen so the coupled physics is a stable, signature-preserving
perturbation; real opacity/EOS-derived values arrive with the IONMIX tables in Phase C
(#118).  Resistive B_phi and multigroup FLD stay off here (the former is wired as a
standalone array not fed back into the live implosion B and is axis-stiff; the latter
needs an IONMIX opacity file).  The regression baseline reflects this coupled stack.

Discriminating quantities (see inputs/maglif_mrt.athinput):
  * R_liner(t)  -- mass-weighted mean liner radius: the bulk convergence trajectory.
  * R_front(t)  -- inner (fuel-side) half-max interface: the leading imploding front.
  * a_outer(t)  -- single-mode amplitude on the driven (liner/vacuum) interface: the
                   MRT-unstable surface -- it grows under the sustained acceleration.
  * a_inner(t)  -- single-mode amplitude on the fuel-side interface: RT-stabilised during
                   the run-in (it does NOT run away), the expected MagLIF asymmetry.
  * rad_mod(t)  -- single-mode contrast of a SYNTHETIC SIDE-ON RADIOGRAPH (the Abel
                   forward projection of the axisymmetric density along the sight line):
                   the experimentally observable MRT signature on the imploding limb.

The test runs the implosion, checks that (1) the liner converges, (2) the leading front
moves inward, (3) the seeded single mode grows on the driven surface and stays dominant
(no spurious mode cascade), (4) the synthetic-radiograph limb modulation grows, then plots
the trajectories + a synthetic radiograph and saves/diffs a golden regression baseline.
Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import glob
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import test_suite.testutils as testutils  # noqa: E402
import test_suite.verification.harness as harness  # noqa: E402
import test_suite.cylindrical.maglif_coupled_energy as cenergy  # noqa: E402

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_mrt.athinput.
PERT_MODE = 3          # seeded axial mode number (wavelength = Lz/pert_mode)
PERT_AMP = 0.05        # seeded radial interface amplitude
D_LINER = 1.0          # liner (dense shell) density
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

# Verification thresholds.
CONV_FRAC = 0.75       # require >= 25% bulk convergence (R_liner shrinks)
# >= 1.3x single-mode growth on the driven surface.  Re-anchored from the ideal-MHD 1.5x
# for the coupled regime (#116/[B3]): anisotropic conduction stabilises the shorter-
# wavelength axial mode, so the coupled single-mode growth (~1.5x) is below the ideal
# (~1.9x); 1.3x keeps the same ~80% relative margin the original bar had vs its ideal run.
GROWTH_MIN = 1.3
DOM_FRAC = 0.6         # seeded mode must hold >= 60% of the interface z-variance
RAD_GROWTH = 1.3       # require the synthetic-radiograph modulation to grow >= 1.3x

input_file = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif_mrt.athinput")
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")


def _cross(r, p, i, j):
    """Sub-cell radius where the density profile p crosses HALF between cells i and j."""
    f = (HALF - p[i]) / (p[j] - p[i])
    return r[i] + f * (r[j] - r[i])


def _interfaces(data):
    """Per-z liner interfaces (phi-averaged; the axisymmetric IC makes phi a replica).

    For each axial height z, reduce to the radial density profile and return the inner
    and outer half-max crossings of the dense-liner slab (continuous, FP-robust) plus the
    mass-weighted mean liner radius (rho*r dr weighting).  Each is an array of length nz.
    """
    r = np.asarray(data["x1v"], dtype=float)            # (nr,)
    dens = np.asarray(data["dens"], dtype=float)        # (nz, nphi, nr)
    nz = dens.shape[0]
    r_in = np.full(nz, np.nan)
    r_out = np.full(nz, np.nan)
    r_mw = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)                       # (nr,) phi-averaged
        dense = p >= HALF
        idx = np.where(dense)[0]
        if len(idx) == 0:
            continue
        w = np.where(dense, p, 0.0)
        r_mw[kz] = np.sum(w * r * r) / np.sum(w * r)
        lo, hi = idx[0], idx[-1]
        r_in[kz] = r[lo] if lo == 0 else _cross(r, p, lo - 1, lo)
        r_out[kz] = r[hi] if hi == len(r) - 1 else _cross(r, p, hi, hi + 1)
    return r_in, r_out, r_mw


def _mode_amp(rz, z, k):
    """Magnitude of the discrete cos/sin projection of r(z) onto axial mode k.

    Magnitude (not signed cosine) so the measure is robust to any phase drift of the
    growing mode.  z is the axial coordinate on [0, Lz=1].
    """
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    zz = z[good]
    a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz))
    b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz))
    return float(np.hypot(a, b))


def _synthetic_radiograph(data):
    """Synthetic side-on radiograph = Abel forward projection of the axisymmetric density.

    A parallel X-ray beam crosses the cylinder perpendicular to its axis (z).  For an
    axisymmetric density rho(r,z) the line-of-sight areal density at transverse impact
    parameter x is the forward Abel transform
        Sigma(x,z) = 2 * integral_{r=x}^{R} rho(r,z) r / sqrt(r^2 - x^2) dr,
    evaluated here by the exact per-annulus chord length L(x,r) = 2[sqrt(r_out^2 - x^2) -
    sqrt(r_in^2 - x^2)] (singularity-free), so Sigma(x,z) = sum_r rho(r,z) L(x,r).
    Returns (x, Sigma) with Sigma shaped (nz, nx); the dense liner is a bright limb whose
    axial position is modulated by the seeded MRT mode.
    """
    r = np.asarray(data["x1v"], dtype=float)            # (nr,) uniform cell centres
    dens = np.asarray(data["dens"], dtype=float)        # (nz, nphi, nr)
    rho_rz = dens.mean(axis=1)                          # (nz, nr) phi-averaged
    dr = r[1] - r[0]
    r_in = np.clip(r - 0.5 * dr, 0.0, None)             # annulus inner faces (axis -> 0)
    r_out = r + 0.5 * dr                                # annulus outer faces
    x = r.copy()                                        # impact parameters = cell radii
    x2 = x[:, None] ** 2                                # (nx, 1)
    a = np.sqrt(np.clip(r_out[None, :] ** 2 - x2, 0.0, None))   # (nx, nr)
    b = np.sqrt(np.clip(r_in[None, :] ** 2 - x2, 0.0, None))
    weight = 2.0 * (a - b)                              # chord length L(x_i, r_j)
    sigma = rho_rz @ weight.T                           # (nz, nx)
    return x, sigma


def _radiograph_mod(x, sigma, z, k):
    """Single-mode axial contrast of the synthetic radiograph at the bright limb.

    The limb impact parameter is the x maximising the z-averaged areal density (the
    tangent ray grazing the dense shell).  At that fixed x, the areal density modulates
    along z as the perturbed liner moves through the sight line; return the seeded-mode
    amplitude normalised by the mean limb brightness (dimensionless radiograph contrast).
    """
    s_mean = sigma.mean(axis=0)                          # (nx,)
    ilimb = int(np.argmax(s_mean))
    col = sigma[:, ilimb]                                # Sigma(x_limb, z)
    base = float(np.mean(col))
    cc = col - base
    a = 2.0 * np.mean(cc * np.cos(2.0 * np.pi * k * z))
    b = 2.0 * np.mean(cc * np.sin(2.0 * np.pi * k * z))
    return float(np.hypot(a, b)) / (abs(base) + 1.0e-30)


def _plot_radiograph(x, z, sig0, sigN, t0, tN):
    """Save the initial vs final synthetic radiograph (Abel projection) as a PNG."""
    harness._ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    ext = [x[0], x[-1], z[0], z[-1]]
    for ax, sig, t, ttl in (
        (axes[0], sig0, t0, "initial"),
        (axes[1], sigN, tN, "final"),
    ):
        im = ax.imshow(sig, origin="lower", aspect="auto", extent=ext, cmap="inferno")
        ax.set_title(f"{ttl}  (t={t:.3f})")
        ax.set_xlabel("impact parameter x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\Sigma$")
    axes[0].set_ylabel("axial z")
    fig.suptitle("Synthetic side-on radiograph (MagLIF benchmark 1, single-mode MRT)")
    fig.tight_layout()
    out = os.path.join(harness.PLOT_DIR, "maglif_mrt_radiograph.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def test_verify_maglif_mrt():
    """Run single-mode MRT implosion; verify convergence + MRT growth + radiograph."""
    try:
        assert testutils.run_pgen("maglif", input_file), "MagLIF MRT run failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_mrt.prim.*.bin")))
        assert len(files) > 8, f"too few MRT snapshots: {len(files)}"

        z = None
        times, r_liner, r_front, a_outer, a_inner, rad_mod = [], [], [], [], [], []
        rad0 = radN = None
        for fp in files:
            d = bin_convert.read_binary_as_athdf(fp)
            if z is None:
                z = np.asarray(d["x3v"], dtype=float)
            r_in, r_out, r_mw = _interfaces(d)
            xrad, sigma = _synthetic_radiograph(d)
            times.append(float(d["Time"]))
            r_liner.append(float(np.nanmean(r_mw)))
            r_front.append(float(np.nanmean(r_in)))
            a_outer.append(_mode_amp(r_out, z, PERT_MODE))
            a_inner.append(_mode_amp(r_in, z, PERT_MODE))
            rad_mod.append(_radiograph_mod(xrad, sigma, z, PERT_MODE))
            if rad0 is None:
                rad0 = sigma
            radN = sigma
        times = np.array(times)
        r_liner = np.array(r_liner)
        r_front = np.array(r_front)
        a_outer = np.array(a_outer)
        a_inner = np.array(a_inner)
        rad_mod = np.array(rad_mod)

        # Final snapshot is finite and stable (the implosion ran in cleanly, no blow-up).
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # (1) The liner actually converges (bulk implosion, not a static/expanding shell).
        assert r_liner[-1] < CONV_FRAC * r_liner[0], (
            f"liner did not converge: R0={r_liner[0]:.4f} -> R={r_liner[-1]:.4f}"
        )

        # (2) The leading (fuel-side) front moves inward as the liner accelerates inward.
        assert r_front[-1] < r_front[0], (
            f"leading front did not move inward: {r_front[0]:.4f} -> {r_front[-1]:.4f}"
        )

        # (3) The seeded single mode grows on the magnetically-driven (outer) interface --
        # the MRT growth signature -- starting from ~the seeded amplitude.
        assert abs(a_outer[0] - PERT_AMP) < 0.3 * PERT_AMP, (
            f"initial driven-surface amplitude {a_outer[0]:.4f} != seed {PERT_AMP}"
        )
        assert a_outer[-1] > GROWTH_MIN * a_outer[0], (
            f"single-mode did not grow: a_out {a_outer[0]:.4f} -> {a_outer[-1]:.4f} "
            f"(need > {GROWTH_MIN}x)"
        )

        # (4) The seeded mode stays dominant on the driven interface (no spurious
        # cascade): its modal power is the majority of the z-variance at stagnation.
        r_out_fin = _interfaces(fin)[1]
        ro = r_out_fin[np.isfinite(r_out_fin)]
        var_tot = float(np.mean((ro - ro.mean()) ** 2))
        seed_pow = 0.5 * a_outer[-1] ** 2
        dom = seed_pow / var_tot
        assert dom > DOM_FRAC, (
            f"seeded mode not dominant at stagnation: {dom:.2f} < {DOM_FRAC}"
        )

        # (5) The synthetic-radiograph limb modulation (the observable MRT signal) grows.
        assert rad_mod[-1] > RAD_GROWTH * rad_mod[0], (
            f"radiograph modulation did not grow: {rad_mod[0]:.4e} -> {rad_mod[-1]:.4e} "
            f"(need > {RAD_GROWTH}x)"
        )

        # (6) The coupled radiation-conduction-MHD stack ran and kept the species-split
        # energy budget physically sane (erad finite, non-neg, bounded; #116/[B3]).
        cenergy.assert_coupled_energy_sane("maglif_mrt")

        # Plot the synthetic radiograph (initial vs final); trajectories + baseline below.
        _plot_radiograph(xrad, z, rad0, radN, times[0], times[-1])

        harness.verify(
            "maglif_mrt",
            times,
            {
                "R_liner": r_liner,
                "R_front": r_front,
                "a_outer": a_outer,
                "a_inner": a_inner,
                "rad_mod": rad_mod,
            },
            coord_label="t",
            xlabel="t",
            title="Single-mode MRT (MagLIF benchmark 1): convergence + growth",
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
