"""
Multi-RANK (MPI) operator-split grey-FLD Marshak verification ([A3]/#110, ADR-0001/0009).

The MPI companion to test_verify_fld_marshak_multiblock_cpu: builds the same pgen
(src/pgen/unit_tests/fld_marshak_multiblock_test.cpp) with -D Athena_ENABLE_MPI=ON into
its own isolated build directory and runs it under ``mpirun -np 4`` so the FOUR MeshBlocks
land one-per-rank -- making every internal block boundary a cross-RANK MPI ghost exchange.
This exercises the MPI_Isend/Irecv path of MeshBoundaryValuesCC::SyncParabolicGhosts (the
single-rank _cpu test only exercises the same-rank direct-buffer-copy path), proving the
operator-split grey FLD runs on >1 MPI rank and still reproduces the analytic erfc Marshak
wave across rank boundaries.

The pgen MPI_Allreduce's every comparison so all ranks check identical global values and
exit consistently; pass iff the run exits 0.

Oracle: Layer 1 -- analytic (see test_verify_fld_marshak_multiblock_cpu for the wave).

Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "fld_marshak_multiblock_test"
NRANKS = 4  # 4 MeshBlocks / 4 ranks => every internal block face is a cross-rank exchange


def test_run_mpi():
    """Build (MPI) + mpirun the multi-block grey-FLD Marshak test; pass iff it exits 0."""
    repo_root = testutils._repo_root()
    build_dir = os.path.join(repo_root, "tst", "build_unit", NAME + "_mpi")
    input_file = os.path.abspath(
        os.path.join(repo_root, "inputs", "unit_tests", NAME + ".athinput")
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
        ), "MPI build of fld_marshak_multiblock_test failed"
        passed = testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", input_file]
        )
        assert passed, (
            "fld_marshak_multiblock_test failed under MPI: operator-split grey FLD did "
            "not reproduce the analytic Marshak wave across MPI-rank boundaries "
            "(SyncParabolicGhosts MPI exchange)"
        )
    finally:
        os.chdir(original_dir)
