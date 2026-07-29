"""
Multi-RANK (MPI) companion of the maglif coupled-circuit verification (#236).

The mode-C feedback V_load = d(Phi)/dt is a GLOBAL reduction: PoloidalFluxGlobal must
MPI_Allreduce the per-rank flux partials before the circuit ODE consumes them, and the
circuit history must write the (host-global) circuit scalars on rank 0 only while the
flux column stays a per-rank LOCAL partial (history.cpp's MPI_Reduce(SUM) then yields
the global value).  Any violation -- a rank-local flux fed to the circuit, or circuit
scalars written on every rank -- breaks the exact coupled-circuit identity

    L*I(t) + Phi(t) = V_oc * t        (Z0 = C = R_loss = 0, started from rest),

so re-checking that identity on a 2-rank / 2-MeshBlock run of the SAME deck
(inputs/maglif_circuit.athinput, radially split so the poloidal-flux integrand spans
both ranks) is the discriminating MPI-safety pin the single-rank _cpu test cannot give.

Builds the maglif pgen with -D Athena_ENABLE_MPI=ON into its own isolated build
directory (tst/build_pgen/maglif_mpi) and runs it under ``mpirun -np 2`` (2 CPUs).

Oracle: Layer 1 -- analytic (the exact flux-conservation integral of the coupled loop).
Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os
import shutil

import numpy as np
import test_suite.testutils as testutils

# Values that must match inputs/maglif_circuit.athinput.
V_OC = 2.0          # open-circuit (constant) voltage source [code units]
L_CIRCUIT = 1.0     # series inductance (pure L + coupled load)
TLIM = 1.25         # run end time
NRANKS = 2          # 2 radial MeshBlocks / 2 ranks: the flux reduction spans ranks

input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_circuit.athinput"
)
BASENAME = "maglif_circ_mpi"


def test_maglif_coupled_circuit_mpi():
    """Mode-C flux conservation on 2 ranks: the Faraday reduction is MPI-safe."""
    repo_root = testutils._repo_root()
    build_dir = os.path.join(repo_root, "tst", "build_pgen", "maglif_mpi")
    run_dir = os.path.join(build_dir, "src")
    hst_file = os.path.join(run_dir, f"{BASENAME}.user.hst")

    original_dir = os.getcwd()
    try:
        os.chdir(repo_root)
        config = [
            "cmake",
            "-D", "Athena_ENABLE_MPI=ON",
            "-D", "PROBLEM=maglif",
            "-B", build_dir,
        ]
        assert testutils.run_command(config), "CMake (MPI) configuration failed"
        os.chdir(run_dir)
        assert testutils.run_command(
            ["make", "-j", f"{os.cpu_count()}"]
        ), "MPI build of the maglif pgen failed"

        # AthenaK APPENDS to an existing .user.hst; clear ours so the parse is this run.
        if os.path.exists(hst_file):
            os.remove(hst_file)

        # 2 radial MeshBlocks (64 zones each) -> one per rank: every flux partial and
        # the circuit feedback cross the rank boundary.  The circuit-only history
        # (problem/user_hist=false) keeps the MPI run off the rank-unsafe MAX cons
        # columns (MagLIFConsHistory is documented single-rank-only).
        passed = testutils.run_command(
            [
                "mpirun", "-np", str(NRANKS), "./athena", "-i", input_file,
                f"job/basename={BASENAME}",
                "meshblock/nx1=64",
                "problem/user_hist=false",
            ]
        )
        assert passed, "maglif coupled_circuit (mode C) MPI run failed."

        import athena_read

        hst = athena_read.hst(hst_file)
        assert "I_circuit" in hst and "Bphi_flux" in hst, (
            "maglif history lacks the circuit columns under MPI: drive mode C is not "
            "wired into the maglif path (#236)"
        )
        th = hst["time"]
        cur = hst["I_circuit"]
        flux = hst["Bphi_flux"]
        assert len(th) > 50, f"too few history samples: {len(th)}"

        # The exact coupled-loop integral must survive the cross-rank reduction: a
        # rank-local flux in the feedback (or double-written circuit scalars in the
        # history) shifts Phi (or I) by O(1) and breaks this identity.
        resid = np.abs(L_CIRCUIT * cur + flux - V_OC * th)
        assert resid.max() < 0.02, (
            f"flux conservation L*I+Phi=V_oc*t violated on {NRANKS} ranks: max abs "
            f"residual {resid.max():.4f} (>= 0.02) -- Faraday reduction not MPI-safe"
        )
        late = th > 0.3
        rel_resid = (resid[late] / (V_OC * th[late])).max()
        assert rel_resid < 0.02, (
            f"flux conservation relative residual too large past startup on "
            f"{NRANKS} ranks: {rel_resid:.4f}"
        )

        # Faraday tally under MPI: V_load column (rank-0 scalar + SUM) still equals
        # d(Phi)/dt of the (rank-summed) flux column.
        dphidt = np.gradient(flux, th)
        win = (th > 0.1) & (th < 0.98 * TLIM)
        rel_far = np.abs(hst["V_load"][win] - dphidt[win]) / (
            np.abs(dphidt[win]) + 1.0e-3
        )
        assert np.median(rel_far) < 0.03, (
            f"Faraday V_load != d(Phi)/dt on {NRANKS} ranks: median rel error "
            f"{np.median(rel_far):.4f}"
        )
    finally:
        os.chdir(original_dir)
        shutil.rmtree(os.path.join(run_dir, "tab"), ignore_errors=True)
        if os.path.exists(hst_file):
            os.remove(hst_file)
