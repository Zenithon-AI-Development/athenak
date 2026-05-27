"""
End-to-end demonstration of the Tier-2 verification harness on an existing problem
(Sod's shock tube). Running this:
  - runs Sod once,
  - emits a standard diagnostic plot of the density / velocity / internal-energy
    profiles at t = 0.25 to ``verification/plots/sod.png``,
  - captures a versioned golden baseline (``verification/baselines/sod.json``) on first
    use, and
  - on subsequent runs diffs the profiles against that baseline within tolerance.

This is auto-collected by ``run_test_suite.py`` (no runner edits) because the module
name contains ``_cpu``.
"""

# Modules
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

input_file = "inputs/sod.athinput"


def test_verify_sod():
    """Run Sod and verify the t=0.25 profiles against the golden baseline."""
    try:
        # Fixed, deterministic configuration so the baseline is reproducible.
        results = testutils.run(input_file, ["job/basename=sod"])
        assert results, "Sod verification run failed."
        data = athena_read.tab("tab/sod.hydro_w.00001.tab")
        harness.verify(
            "sod",
            data["x1v"],
            {
                "dens": data["dens"],
                "velx": data["velx"],
                "eint": data["eint"],
            },
            coord_label="x1v",
            xlabel="x",
            title="Sod shock tube (t=0.25)",
        )
    finally:
        testutils.cleanup()
