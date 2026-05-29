"""
Multi-RANK (MPI) Strang-split coupled-timestep integrated test ([B2]/#115, ADR-0009).

The MPI companion to test_verify_strang_coupling_cpu (bump mode): builds the same pgen
(src/pgen/unit_tests/strang_coupling_test.cpp) with -D Athena_ENABLE_MPI=ON into its own
isolated build directory and runs the bump conservation case under ``mpirun -np 2`` so the
TWO MeshBlocks land one-per-rank -- making the internal block boundary a cross-RANK MPI
ghost exchange for the parabolic half-step (MeshBoundaryValuesCC::SyncParabolicGhosts).

This also exercises the #114 global min-dt reduction that makes the orchestration safe:
the bump's flux-limited FLD diffusivity (and hence its explicit dt) varies in space, so a
PER-RANK dt would give ranks different RKL2 substage counts and DEADLOCK the synchronous
per-substage ghost exchange.  Because the Strang half-step routes through the per-field
CompositeParabolicOperator -- whose ExplicitStableDt does the global MPI_Allreduce-MIN --
every rank derives the SAME stage count, so the run does not deadlock and still conserves
the volume-weighted total energy (E_r + E_gas) across the cross-rank exchange.

The pgen MPI_Allreduce's the conserved total so every rank checks the same value and
exits consistently; pass iff the run exits 0.

Oracle: Layer 1 -- exact conservation of a conserved scalar (see _cpu for details).

Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "strang_coupling_test"
NRANKS = 2  # 2 MeshBlocks / 2 ranks => the internal block face is a cross-rank exchange


def test_run_mpi():
    """Build (MPI) + mpirun the Strang-coupling bump conservation case; pass if exit 0."""
    repo_root = testutils._repo_root()
    build_dir = os.path.join(repo_root, "tst", "build_unit", NAME + "_mpi")
    input_file = os.path.abspath(
        os.path.join(
            repo_root, "inputs", "unit_tests", "strang_coupling_bump_test.athinput"
        )
    )
    original_dir = os.getcwd()
    try:
        os.chdir(repo_root)
        config = [
            "cmake",
            "-D", "Athena_ENABLE_MPI=ON",
            "-D", f"PROBLEM=unit_tests/{NAME}",
            "-B", build_dir,
        ]
        assert testutils.run_command(config), "CMake (MPI) configuration failed"
        os.chdir(os.path.join(build_dir, "src"))
        assert testutils.run_command(
            ["make", "-j", f"{os.cpu_count()}"]
        ), "MPI build of strang_coupling_test failed"
        passed = testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", input_file]
        )
        assert passed, (
            "strang_coupling_test[bump] failed under MPI: the fully-coupled Strang stack "
            "either deadlocked (per-rank stage-count mismatch -- the #114 global min-dt "
            "reduction did not run) or did not conserve total energy across the "
            "SyncParabolicGhosts exchange (#115)"
        )
    finally:
        os.chdir(original_dir)
