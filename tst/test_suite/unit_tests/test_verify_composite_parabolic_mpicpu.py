"""
Multi-RANK (MPI) composite parabolic operator verification ([B1]/#114, ADR-0009).

The MPI companion to test_verify_composite_parabolic_cpu: builds the same pgen
(src/pgen/unit_tests/composite_parabolic_test.cpp) with -D Athena_ENABLE_MPI=ON into its
own isolated build directory and runs it under ``mpirun -np 4`` so the FOUR MeshBlocks
land one-per-rank. This exercises the two MPI paths #114 adds/relies on:

  - the GLOBAL min-dt MPI all-reduce behind
    CompositeParabolicOperator::ExplicitStableDt(). The pgen builds a sub-operator pair
    over a PER-RANK-VARYING density so each rank's local dt_exp differs (rank r has
    rho = rho0*(1+r) => dt_exp grows with r); the composite must return the SAME global
    minimum (rank 0's) on every rank, <= each rank's local min, and strictly below the
    cross-rank maximum -- proving the all-reduce genuinely combines ranks (single-rank
    runs can only verify the identity reduction).
  - the summed RKL2 super-step running across MPI-rank boundaries: the combined cosine
    mode reproduces the analytic exp((lambda1+lambda2) t) decay across rank boundaries
    via each sub-operator's SyncParabolicGhosts exchange, with a single
    globally-consistent stage count (the global min-dt fixes the spatially-varying-dt
    deadlock risk, cf. #113).

The pgen MPI_Allreduce's every comparison so all ranks check identical global values and
exit consistently; pass iff the run exits 0.

Oracle: Layer 1 -- analytic (see test_verify_composite_parabolic_cpu for the eigenmode).

Auto-collected by run_test_suite.py under --mpicpu (module name contains ``_mpicpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "composite_parabolic_test"
NRANKS = 4  # 4 MeshBlocks / 4 ranks => each rank holds one block (cross-rank exchange)


def test_run_mpi():
    """Build (MPI) + mpirun the composite parabolic operator test; pass iff it exits 0."""
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
        ), "MPI build of composite_parabolic_test failed"
        passed = testutils.run_command(
            ["mpirun", "-np", str(NRANKS), "./athena", "-i", input_file]
        )
        assert passed, (
            "composite_parabolic_test failed under MPI: the composite ExplicitStableDt "
            "global min-dt all-reduce or the summed STS evolution did not reproduce the "
            "expected cross-rank result"
        )
    finally:
        os.chdir(original_dir)
