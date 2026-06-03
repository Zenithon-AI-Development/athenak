"""
Faithful B1 per-operator ATTRIBUTION probe (issue #175/[P6], ADR-0011/ADR-0012).

Issue #175 asks the ADR-0011 follow-up question: enable the reference (Ellison/FLASH,
arXiv:2504.10760) model-set operators -- electron/ion conduction, resistivity, gray
radiation diffusion -- in the faithful tabulated-3T single-mode-MRT B1 run
(inputs/maglif_b1_sinars.athinput), ONE OPERATOR AT A TIME, and see whether the
amplitude(t) growth gap closes (which would flip the #120 AC#2 comparison to binding) or
survives (in which case the residual is documented per-operator, still NOT strength).

FINDING (ADR-0012), confirmed here on the GPU: with the faithful ``eos = tabulated_3t``
closure, NONE of the operators yet alters the implosion FAITHFULLY, for code-structural
reasons --

  * resistivity (``resb_operator_split``) advances a STANDALONE ``bphi`` array that the
    maglif pgen never couples to the live MHD face field ``b0.x2f`` (where the driven
    B_phi lives) -- so it is INERT.
  * gray FLD (``fld_operator_split``) diffuses a STANDALONE ``erad`` the maglif pgen never
    sources, and without ``mrad_coupling`` never touches the gas -- so it is INERT.
  * gray FLD + matter-radiation coupling (``mrad_coupling``) couples erad<->IEN with a
    CONSTANT heat capacity ``mrad_cv`` (not the tabulated closure), but has nothing to
    exchange against because ``erad`` is never sourced (stays 0) -- so it is INERT in
    practice (the EOS-inconsistent c_v only bites once erad is sourced).
  * conduction (``acond_operator_split``) DOES act on the live gas energy, but recovers
    temperature as the ideal-gamma T = (gamma-1) eint/rho (aniso_conduction_operator.cpp
    TempMHD), inconsistent with the tabulated_3t electron/ion-split closure -- so it
    CHANGES the amplitude(t) curve, but through a non-faithful (ideal-gamma) closure.

Measured on the GPU (reduced grid): resb, gray-FLD and gray-FLD+mrad leave the seeded-mode
amplitude(t) BITWISE identical to the operators-off baseline; only acond perturbs it
(max ~1% over the run-in), and EOS-inconsistently.

This test LOCKS IN that attribution as a regression: it enables each operator one at a
time on the faithful setup (at a cheap REDUCED resolution -- the inert/active facts are
code-structural and resolution-independent, and operator-on is compared against
operator-off at the SAME grid so the seed resolution cancels), and asserts
  * resb, gray-FLD and gray-FLD+mrad leave the seeded-mode amplitude(t) IDENTICAL to
    baseline (INERT -- decoupled/unsourced standalone fields), and
  * acond CHANGES it while staying finite (ACTIVE, EOS-inconsistent), and its pert_amp=0
    control stays exactly 1-D (the AC#1 "1-D-stays-1-D" anchor still holds with it ON).
Each per-operator verdict is recorded in the scorecard (binding=False, ADR-0012 note).

CONCLUSION (recorded, NOT strength; ADR-0011 holds): the faithful B1 amplitude(t) residual
is bounded by THREE reference-model-set engineering gaps -- (a) couple resb's bphi to the
live b0.x2f, (b) source the FLD erad from the gas, (c) make acond's T and mrad's c_v
EOS-aware -- all filed as a follow-up.  The paper-resolution amplitude(t) oracle verdict
itself stays the job of test_verify_maglif_b1_sinars_gpu.py (the faithful baseline run).

GPU-only (``_gpu`` suffix): built+run with CUDA; auto-collected by ``run_test_suite.py
--gpu`` (excluded from CPU CI).  Heavy GPU CI harness is #123/[C6].
"""

# Modules
import glob
import os
import shutil
import sys

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.scorecard as scorecard

sys.path.insert(0, os.path.join(testutils._repo_root(), "vis", "python"))
import bin_convert  # noqa: E402

PERT_MODE = 4          # seeded axial mode number (must match the athinput)
LZ = 1.6               # axial domain extent [mm]
HALF = 0.5             # half-max density level isolating the dense liner (code units)

# Reduced GPU resolution: cheap, deterministic, and split into 2 axial MeshBlocks so the
# coupled operators' per-substage cross-block ghost exchange (SyncParabolicGhosts, #108)
# is exercised on the GPU.  The inert/active attribution is code-structural (resolution-
# independent), and every operator-on run is compared against the operator-off run at the
# SAME grid, so the (sub-cell) seed resolution cancels out of the comparison.
NX1, NX2, NX3 = 72, 4, 32
MBX1, MBX2, MBX3 = 72, 4, 16
TLIM = 80.0            # ns; long enough for conduction/coupling to act on the run-in

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
    testutils._repo_root(), "tst", "inputs", "maglif_b1_sinars.athinput"
)
table_file = os.path.join(
    testutils._repo_root(), "inputs", "ionmix", "al-imx-004.cn4"
)
trace_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "z2173_current.dat"
)
bin_dir = os.path.join(testutils.pgen_run_dir("maglif"), "bin")
build_dir = os.path.join(testutils._repo_root(), "tst", "build_pgen", "maglif")

# The four reference-model-set operators, enabled ONE AT A TIME on the faithful setup.
# ``inert`` is the ADR-0012 structural prediction this test verifies on the GPU.
OPERATORS = [
    {"key": "resb", "args": ["mhd/strang_split=true", "mhd/resb_operator_split=true"],
     "inert": True,  "label": "resistivity (resb)",
     "why": "standalone bphi, never coupled to live b0.x2f in maglif"},
    {"key": "fld", "args": ["mhd/strang_split=true", "mhd/fld_operator_split=true"],
     "inert": True,  "label": "gray FLD (fld, no coupling)",
     "why": "standalone erad, never sourced; no mrad coupling -> never touches the gas"},
    {"key": "fldmrad",
     "args": ["mhd/strang_split=true", "mhd/fld_operator_split=true",
              "mhd/mrad_coupling=true"],
     "inert": True,  "label": "gray FLD + matter-rad coupling (fld+mrad)",
     "why": "erad never sourced in maglif (stays 0) -> coupling inert; constant-c_v "
            "closure also EOS-inconsistent (latent until erad is sourced)"},
    {"key": "acond", "args": ["mhd/strang_split=true", "mhd/acond_operator_split=true"],
     "inert": False, "label": "conduction (acond)",
     "why": "acts on live IEN via ideal-gamma T=(g-1)e/rho, EOS-inconsistent"},
]


def _build_dir_is_cuda():
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return False
    with open(cache, "r") as f:
        return "Kokkos_ENABLE_CUDA:BOOL=ON" in f.read().upper()


def _run(basename, extra_args=None, pert_amp=None):
    """Run the faithful B1 (reduced GPU) with optional operator/seed overrides."""
    args = [
        f"mhd/eos_table={table_file}",
        f"problem/current_file={trace_file}",
        f"job/basename={basename}",
        f"mesh/nx1={NX1}", f"mesh/nx2={NX2}", f"mesh/nx3={NX3}",
        f"meshblock/nx1={MBX1}", f"meshblock/nx2={MBX2}", f"meshblock/nx3={MBX3}",
        f"time/tlim={TLIM}", "output1/dt=5.0", "output2/dt=1000",
    ]
    if extra_args:
        args += extra_args
    if pert_amp is not None:
        args.append(f"problem/pert_amp={pert_amp}")
    ok = testutils.run_pgen("maglif", input_file, flags=GPU_FLAGS, args=args)
    assert ok, f"faithful B1 operator run failed (basename={basename})"
    files = sorted(glob.glob(os.path.join(bin_dir, f"{basename}.prim.*.bin")))
    assert len(files) >= 3, f"too few snapshots for {basename}: {len(files)}"
    return files


def _cross(r, p, i, j):
    return r[i] + (HALF - p[i]) / (p[j] - p[i]) * (r[j] - r[i])


def _outer_interface(d):
    """Per-z outer half-max interface r_out(z) (length nz)."""
    r = np.asarray(d["x1v"], dtype=float)
    dens = np.asarray(d["dens"], dtype=float)            # (nz, nphi, nr)
    nz = dens.shape[0]
    r_out = np.full(nz, np.nan)
    for kz in range(nz):
        p = dens[kz].mean(axis=0)
        idx = np.where(p >= HALF)[0]
        if len(idx) == 0:
            continue
        hi = idx[-1]
        r_out[kz] = r[hi] if hi == len(r) - 1 else _cross(r, p, hi, hi + 1)
    return r_out


def _mode_amp(rz, zn, k):
    good = np.isfinite(rz)
    if good.sum() < 2:
        return 0.0
    rr = rz[good] - np.mean(rz[good])
    zz = zn[good]
    a = 2.0 * np.mean(rr * np.cos(2.0 * np.pi * k * zz))
    b = 2.0 * np.mean(rr * np.sin(2.0 * np.pi * k * zz))
    return float(np.hypot(a, b))


def _amp_trajectory(files):
    """Reduce snapshots to the seeded-mode outer-interface amplitude a_outer(t)."""
    times, a_outer = [], []
    zn = None
    for fp in files:
        d = bin_convert.read_binary_as_athdf(fp)
        if zn is None:
            zn = np.asarray(d["x3v"], dtype=float) / LZ
        times.append(float(d["Time"]))
        a_outer.append(_mode_amp(_outer_interface(d), zn, PERT_MODE))
    return np.array(times), np.array(a_outer)


def _finite_final(files):
    """True iff every primitive field in the final snapshot is finite (stable run)."""
    d = bin_convert.read_binary_as_athdf(files[-1])
    return all(np.all(np.isfinite(np.asarray(d[v])))
               for v in ("dens", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3"))


def test_verify_maglif_b1_operators_gpu():
    """Enable conduction/resistivity/gray-FLD one at a time; attribute the B1 residual."""
    if not _build_dir_is_cuda():
        shutil.rmtree(build_dir, ignore_errors=True)
    pats = ["b1op_base.*"] + [f"b1op_{o['key']}.*" for o in OPERATORS] \
        + [f"b1op_{o['key']}_ctl.*" for o in OPERATORS if not o["inert"]]
    try:
        # Baseline: faithful ideal-MHD core (operators OFF), same reduced grid.
        bfiles = _run("b1op_base")
        assert _finite_final(bfiles), "baseline final state non-finite"
        _, a_base = _amp_trajectory(bfiles)

        for op in OPERATORS:
            files = _run(f"b1op_{op['key']}", op["args"])
            assert _finite_final(files), f"{op['key']} final state non-finite (unstable)"
            _, a_op = _amp_trajectory(files)
            n = min(len(a_base), len(a_op))
            same = np.allclose(a_op[:n], a_base[:n], rtol=1.0e-7, atol=1.0e-9)
            denom = np.maximum(np.abs(a_base[:n]), 1.0e-12)
            max_rel = float(np.max(np.abs(a_op[:n] - a_base[:n]) / denom))

            if op["inert"]:
                # ADR-0012: decoupled/unsourced standalone field -> the live MHD state,
                # hence the seeded-mode amplitude(t), is IDENTICAL to baseline.
                assert same, (
                    f"{op['label']} was expected INERT ({op['why']}) but changed the "
                    f"seeded-mode amplitude(t): max rel deviation {max_rel:.3e} > tol"
                )
                verdict = "INERT"
                note = (f"{op['label']}: {op['why']} -> seeded-mode amplitude(t) "
                        f"identical to baseline (rel dev {max_rel:.1e}); no effect on "
                        f"the faithful B1 implosion (ADR-0012, NOT strength)")
            else:
                # The operator acts on the live gas energy -> it changes the curve;
                # ADR-0012: but via an ideal-gamma closure inconsistent with tabulated_3t.
                assert not same, (
                    f"{op['label']} expected ACTIVE on the gas ({op['why']}) but left "
                    f"the seeded-mode amplitude(t) unchanged (max rel dev {max_rel:.3e})"
                )
                # AC#1 anchor: the operator must not spuriously seed axial structure --
                # its pert_amp=0 control stays exactly 1-D.
                cfiles = _run(f"b1op_{op['key']}_ctl", op["args"], pert_amp=0.0)
                assert _finite_final(cfiles), f"{op['key']} control non-finite"
                _, ca = _amp_trajectory(cfiles)
                assert float(np.max(ca)) < 1.0e-6, (
                    f"{op['label']} pert_amp=0 control developed axial structure "
                    f"(max a_outer={float(np.max(ca)):.3e}); breaks 1-D-stays-1-D"
                )
                verdict = "ACTIVE/EOS-INCONSISTENT"
                note = (f"{op['label']}: changes amplitude(t) (rel {max_rel:.1e}) but "
                        f"{op['why']}; 1-D control stays 1-D; not faithful on "
                        f"tabulated_3t (ADR-0012, NOT strength)")

            # sim/experiment kept None: the scorecard formats numeric cells with ':g', so
            # the string verdict lives in the `verdict` field and details in `note`.
            scorecard.record(
                "B1", f"single_mode_growth_op_{op['key']}",
                sim=None, experiment=None, band=None, unit=None,
                verdict=verdict, binding=False, note=note,
            )
            print(f"[B1][#175] {op['label']:42s} -> {verdict}  "
                  f"(max rel dev vs baseline {max_rel:.3e})")

        print("[B1][#175] residual bounded by 3 model-set gaps (resb<->b0 coupling, erad "
              "sourcing, EOS-aware acond/mrad T & c_v); NOT strength (ADR-0012).")
    finally:
        for pat in pats:
            for f in glob.glob(os.path.join(bin_dir, pat)):
                os.remove(f)
        shutil.rmtree(bin_dir, ignore_errors=True)
