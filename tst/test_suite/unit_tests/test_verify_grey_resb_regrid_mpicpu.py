"""
Multi-RANK (MPI) dynamic-AMR GREY-FLD + resb-B_phi regrid verification (#248).

The MPI companion to test_verify_grey_resb_regrid_cpu: builds the same pgen
(src/pgen/unit_tests/grey_resb_regrid_test.cpp) with -D Athena_ENABLE_MPI=ON and runs it
under ``mpirun -np 4``. With four ranks the load balancer MIGRATES the refined/de-refined
MeshBlocks across ranks during each regrid, exercising the MPI AMR load-balance path
(load_balance.cpp Pack/UnpackAMRBuffersCC) for the registered standalone grey
radiation-energy array MHD::erad and resb B_phi array MHD::bphi -- the single-rank _cpu
test only exercises the same-rank Derefine/Copy/Refine path. It proves both standalone
arrays are conserved across a regrid that also moves blocks between ranks.

The finalize hook MPI_Allreduce's the conserved totals and the peak MeshBlock count so
all ranks check identical globals and exit consistently; pass iff the run exits 0.

Oracle: Layer 1 -- conservation (see test_verify_grey_resb_regrid_cpu).

Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "grey_resb_regrid_test"
NRANKS = 4  # 4 root MeshBlocks; refinement + load balancing migrates blocks across ranks


def test_run_mpi():
    """Build (MPI) + mpirun the dynamic-AMR grey-FLD + resb regrid test (exit 0)."""
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
        ), "MPI build of grey_resb_regrid_test failed"
        passed = testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", input_file]
        )
        assert passed, (
            "grey_resb_regrid_test failed under MPI: the grey erad and/or resb bphi "
            "arrays were not conserved across a dynamic AMR regrid that moved blocks "
            "across ranks (not packed/unpacked in the MPI AMR load-balance path)"
        )
    finally:
        os.chdir(original_dir)
