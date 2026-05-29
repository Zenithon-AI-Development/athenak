"""
Multi-RANK (MPI) operator-split anisotropic (Braginskii) conduction verification
([A5]/#112, ADR-0006/ADR-0001).

The MPI companion to test_verify_aniso_conduction_multiblock_cpu: builds the same pgen
(src/pgen/unit_tests/aniso_conduction_multiblock_test.cpp) with -D Athena_ENABLE_MPI=ON
into its own isolated build directory and runs it under ``mpirun -np 4`` so the FOUR
MeshBlocks land one-per-rank -- making every block boundary a cross-RANK MPI ghost
exchange. This exercises the MPI_Isend/Irecv path of
MeshBoundaryValuesCC::SyncParabolicGhosts (the single-rank _cpu test only exercises the
same-rank direct-buffer-copy path), proving the operator-split anisotropic conduction
keeps heat field-aligned across MPI-rank boundaries.

The pgen MPI_Allreduce's every comparison so all ranks check identical global values and
exit consistently; pass iff the run exits 0.

Oracle: Layer 1 -- analytic (see test_verify_aniso_conduction_multiblock_cpu for the
field-aligned eigenmode + ring oracle).

Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "aniso_conduction_multiblock_test"
NRANKS = 4  # 4 MeshBlocks / 4 ranks => every block face is a cross-rank MPI exchange


def test_run_mpi():
    """Build (MPI) + mpirun the multi-block aniso-conduction test; pass iff it exits 0."""
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
        ), "MPI build of aniso_conduction_multiblock_test failed"
        passed = testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", input_file]
        )
        assert passed, (
            "aniso_conduction_multiblock_test failed under MPI: operator-split aniso "
            "conduction did not keep heat field-aligned across MPI-rank boundaries "
            "(SyncParabolicGhosts MPI exchange)"
        )
    finally:
        os.chdir(original_dir)
