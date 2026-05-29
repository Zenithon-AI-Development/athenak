"""
Multi-RANK (MPI) operator-split CYLINDRICAL resistive B_phi diffusion verification
([A6]/#113, ADR-0004/ADR-0001).

The MPI companion to test_verify_resb_bphi_multiblock_cpu: builds the same pgen
(src/pgen/unit_tests/resb_bphi_multiblock_test.cpp) with -D Athena_ENABLE_MPI=ON into its
own isolated build directory and runs it under ``mpirun -np 4`` so the FOUR radial
MeshBlocks land one-per-rank -- making every internal radial block face a cross-RANK MPI
ghost exchange. This exercises the MPI_Isend/Irecv path of
MeshBoundaryValuesCC::SyncParabolicGhosts (the single-rank _cpu test only exercises the
same-rank direct-buffer-copy path), proving the operator-split cylindrical resistive B_phi
diffusion reproduces the analytic J_1 eigenmode decay across MPI-rank boundaries.

It runs on a radial ANNULUS r in [r0, R] (input file resb_bphi_mpi_test.athinput) with
kr = j_{1,2}/R and r0 = j_{1,1}/kr, so J_1 vanishes at both annulus ends and the
antisymmetric ghost is exact at the true radial faces. The annulus EXCLUDES the axis on
purpose: the operator's explicit-stable dt is currently reduced per rank (the global
min-dt reduction is #114/[B1]), so the near-axis -eta B_phi/r^2 stiffness of a full-disk
mesh would give different ranks different RKL2 stage counts and the synchronous cross-rank
SyncParabolicGhosts would deadlock. On the annulus the radial diffusion dominates and the
per-rank dt is uniform to ~0.005% -> every rank uses the same stage count. The
-eta B_phi/r^2 curl-curl term is still verified through the J_1 shape preservation; the
near-axis 1/r^2 stiffness is covered by the single-rank _cpu / _amr full-disk meshes.

The pgen MPI_Allreduce's every comparison so all ranks check identical global values and
exit consistently; pass iff the run exits 0.

Oracle: Layer 1 -- analytic (see test_verify_resb_bphi_multiblock_cpu for the J_1
eigenmode oracle).

Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "resb_bphi_multiblock_test"
INPUT = "resb_bphi_mpi_test"  # radial annulus (no axis) so per-rank dt is uniform
NRANKS = 4  # 4 MeshBlocks / 4 ranks => every block face is a cross-rank MPI exchange


def test_run_mpi():
    """Build (MPI) + mpirun the multi-block resistive-B_phi test; pass iff it exits 0."""
    repo_root = testutils._repo_root()
    build_dir = os.path.join(repo_root, "tst", "build_unit", NAME + "_mpi")
    input_file = os.path.abspath(
        os.path.join(repo_root, "inputs", "unit_tests", INPUT + ".athinput")
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
        ), "MPI build of resb_bphi_multiblock_test failed"
        passed = testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", input_file]
        )
        assert passed, (
            "resb_bphi_multiblock_test failed under MPI: operator-split cylindrical "
            "resistive B_phi diffusion did not reproduce the analytic J_1 eigenmode "
            "across MPI-rank boundaries (SyncParabolicGhosts MPI exchange)"
        )
    finally:
        os.chdir(original_dir)
