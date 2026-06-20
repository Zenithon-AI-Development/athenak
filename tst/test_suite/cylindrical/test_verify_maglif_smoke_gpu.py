"""
GPU coupled-stack SMOKE TEST -- the operator-split parabolic GPU regression guard (#139).

This is the GPU sibling of ``test_verify_maglif_smoke_cpu.py``.  It runs the SAME tiny,
multi-block, FULL COUPLED radiation-conduction-MHD MagLIF config
(``inputs/maglif_smoke.athinput``: 32x4x24, 2 MeshBlocks in z, tlim=0.2) but BUILT AND RUN
WITH CUDA, so it exercises the operator-split parabolic super-step (RKL2 STS, ADR-0009) on
the GPU end to end: cylindrical ideal-MHD + the constant-I(t) circuit drive + the
Strang-split operator-split parabolic block (grey flux-limited radiation diffusion +
anisotropic Braginskii conduction on the live gas energy) + the point-implicit
matter-radiation coupling, with the per-substage cross-block ghost exchange
(``SyncParabolicGhosts``, #108) running inside the super-step.

Why it exists: the operator-split parabolic GPU path used to SEGFAULT at runtime on the
very first super-step (#139) -- a header-template ``par_for<...>`` instantiated across
multiple translation units emitted colliding per-TU host launch-stubs that resolved to a
null kernel pointer (relocatable device code is OFF by default).  It was fixed in #153 by
hoisting the four RKL2 stage kernels into named namespace-scope functors launched directly
via ``Kokkos::parallel_for`` (see ``src/driver/parabolic_integrator.hpp``).  Before #153
EVERY GPU MagLIF run was forced to ideal-MHD only; this test is the regression guard that
would have caught the original fault and now keeps the coupled GPU path green.

It is the GPU analog of the CPU per-PR guard: NOT pinned to a golden baseline, asserting
only the QUALITATIVE coupled signature so it stays fast and robust.  It asserts:

  1. the run completes on the GPU and the final state is finite (no operator-split
     blow-up, and -- the point of #139 -- no first-super-step segfault);
  2. the coupled radiation stack actually ran on the GPU -- the species-split ``erad``
     history column is present and etot+erad stays finite/positive/bounded (with the
     coupled stack off there is no ``erad`` column, so this fails);
  3. the liner converges (bulk inward implosion has begun); and
  4. the seeded single MRT mode is present and grows on the magnetically-driven interface.

Oracle: Layer-2 self-consistency / qualitative-signature (ADR-0008).  GPU-only (``_gpu``
suffix): built+run with CUDA, auto-collected by ``run_test_suite.py --gpu`` (excluded from
CPU CI).  The heavy self-hosted-runner GPU CI harness that automates it is #123/[C6]; this
test is what that harness runs to keep the coupled GPU path from silently rotting again.
"""

# Modules
import glob
import os
import shutil
import sys

import numpy as np
import test_suite.testutils as testutils  # noqa: E402
import test_suite.cylindrical.maglif_coupled_energy as cenergy  # noqa: E402

# bin_convert lives next to athena_read in vis/python; add it by absolute path so the
# import is robust to the test's working directory (tests run from tst/build/src).
sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

# Values that must match inputs/maglif_smoke.athinput.
PERT_MODE = 3          # seeded axial mode number (wavelength = Lz/pert_mode)
PERT_AMP = 0.05        # seeded radial interface amplitude
D_LINER = 1.0          # liner (dense shell) density
HALF = 0.5 * D_LINER   # half-max density level isolating the dense liner

# Smoke thresholds -- deliberately lenient (a qualitative GPU guard on a tiny/short run,
# identical to the CPU smoke): require that convergence and single-mode growth are
# *detected*, not that they reach the benchmark magnitudes.
CONV_FRAC = 0.97       # require at least a few % bulk convergence (R_liner shrinks)
GROWTH_MIN = 1.0       # require the seeded mode to grow at all on the driven surface

# GPU build flags (athenakdev / A100); overridable for other runners (#123 generalizes).
GPU_ARCH = os.environ.get("ATHENAK_GPU_ARCH", "Kokkos_ARCH_AMPERE80")
NVCC_WRAPPER = os.environ.get(
    "ATHENAK_NVCC_WRAPPER",
    os.path.join(testutils._repo_root(), "kokkos", "bin", "nvcc_wrapper"),
)
GPU_FLAGS = [
    "-D", "Kokkos_ENABLE_CUDA=On",
    "-D", f"{GPU_ARCH}=On",
    "-D", f"CMAKE_CXX_COMPILER={NVCC_WRAPPER}",
]

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_smoke.athinput")
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")
# The maglif _cpu tests build into the SAME build_pgen/maglif dir as a Serial-Kokkos (CPU)
# binary; a parent-level reconfigure does NOT switch Kokkos's backend from Serial to Cuda
# (the kokkos sub-config is sticky), so a GPU run in a CPU-contaminated dir would silently
# run on the host.  Force a CLEAN build dir so the CUDA backend is configured fresh.
build_dir = os.path.join(testutils._repo_root(), "tst", "build_pgen", "maglif")


def _build_dir_is_cuda():
    """True iff build_pgen/maglif is already configured with the Kokkos CUDA backend."""
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, "r") as f:
        return "KOKKOS_ENABLE_CUDA:BOOL=ON" in f.read().upper()


def _cross(r, p, i, j):
    """Sub-cell radius where the density profile p crosses HALF between cells i and j."""
    f = (HALF - p[i]) / (p[j] - p[i])
    return r[i] + f * (r[j] - r[i])


def _interfaces(data):
    """Per-z (phi-averaged) outer interface and mass-weighted mean liner radius."""
    r = np.asarray(data["x1v"], dtype=float)            # (nr,)
    dens = np.asarray(data["dens"], dtype=float)        # (nz, nphi, nr)
    nz = dens.shape[0]
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
        hi = idx[-1]
        r_out[kz] = r[hi] if hi == len(r) - 1 else _cross(r, p, hi, hi + 1)
    return r_out, r_mw


def _mode_amp(rz, z, k):
    """Magnitude of the discrete cos/sin projection of r(z) onto axial mode k."""
    good = np.isfinite(rz)
    rr = rz[good] - np.mean(rz[good])
    zz = z[good]
    a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz))
    b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz))
    return float(np.hypot(a, b))


def test_verify_maglif_smoke_gpu():
    """Coupled-stack GPU guard: runs (no #139 segfault), energy-sane, converges, grows."""
    # Clean build dir so Kokkos is (re)configured with the CUDA backend, not a stale
    # Serial backend left by a prior maglif _cpu build sharing build_pgen/maglif.
    if not _build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    # AthenaK APPENDS to an existing .user.hst, so a stale history from a prior run (maybe
    # with a different column count) would corrupt the parse.  Clear our own outputs first
    # so the history holds exactly this run -- the guard must be deterministic on re-runs.
    shutil.rmtree(bin_dir, ignore_errors=True)
    if os.path.exists(cenergy.user_hist_path("maglif_smoke")):
        os.remove(cenergy.user_hist_path("maglif_smoke"))
    try:
        assert testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS), \
            "MagLIF coupled-stack GPU smoke run failed."

        files = sorted(glob.glob(os.path.join(bin_dir, "maglif_smoke.prim.*.bin")))
        assert len(files) > 4, f"too few smoke snapshots: {len(files)}"

        z = None
        times, r_liner, a_outer = [], [], []
        for fp in files:
            d = bin_convert.read_binary_as_athdf(fp)
            if z is None:
                z = np.asarray(d["x3v"], dtype=float)
            r_out, r_mw = _interfaces(d)
            times.append(float(d["Time"]))
            r_liner.append(float(np.nanmean(r_mw)))
            a_outer.append(_mode_amp(r_out, z, PERT_MODE))
        times = np.array(times)
        r_liner = np.array(r_liner)
        a_outer = np.array(a_outer)

        # (1) The run completed on the GPU and the final state is finite -- i.e. the
        # operator-split parabolic super-step did NOT segfault on the first step (#139)
        # and did not blow up.
        fin = bin_convert.read_binary_as_athdf(files[-1])
        for v in ("dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"):
            assert np.all(np.isfinite(np.asarray(fin[v]))), f"{v} non-finite (unstable)"

        # (2) The coupled radiation stack actually ran on the GPU and kept the
        # species-split energy budget physically sane (erad present + finite/non-neg +
        # etot+erad bounded).  It is the integration-rot discriminator: with the coupled
        # stack off the maglif pgen writes no 'erad' column, so the assert raises.
        cenergy.assert_coupled_energy_sane("maglif_smoke")

        # (3) The liner converges (bulk inward implosion has begun).
        assert r_liner[-1] < CONV_FRAC * r_liner[0], (
            f"liner did not converge: R0={r_liner[0]:.4f} -> R={r_liner[-1]:.4f}"
        )

        # (4) The seeded single MRT mode is present and grows on the driven interface.
        assert abs(a_outer[0] - PERT_AMP) < 0.4 * PERT_AMP, (
            f"initial driven-surface amplitude {a_outer[0]:.4f} != seed {PERT_AMP}"
        )
        assert a_outer[-1] > GROWTH_MIN * a_outer[0], (
            f"single-mode did not grow: a_out {a_outer[0]:.4f} -> {a_outer[-1]:.4f}"
        )
    finally:
        shutil.rmtree(bin_dir, ignore_errors=True)
