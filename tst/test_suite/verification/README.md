# Tier-2 verification harness

Shared substrate for the MagLIF / Z-pinch verification slices (the `Verify …` issues).
Each slice runs a problem, emits a standard diagnostic plot, captures a versioned golden
baseline once, and diffs future runs against it within a tolerance.

## Add a new verification slice

1. Create `test_verify_<name>_<device>.py` in any `test_suite` subdirectory. The module
   name must contain `_cpu`, `_mpicpu`, or `_gpu` so `run_test_suite.py` selects it for
   the right device. **No edit to the runner is needed** — pytest auto-collects it.
2. In the test: run the problem with `testutils.run(...)`, read the output with
   `athena_read`, then call `harness.verify(name, coord, fields, ...)`.

```python
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

def test_verify_mything():
    try:
        assert testutils.run("inputs/mything.athinput", ["job/basename=mything"])
        data = athena_read.tab("tab/mything.hydro_w.00001.tab")
        harness.verify("mything", data["x1v"],
                       {"dens": data["dens"]}, coord_label="x1v")
    finally:
        testutils.cleanup()
```

## How baselines work

* First run (or `ATHENAK_UPDATE_BASELINES=1`): the run is captured as the golden
  baseline at `baselines/<name>.json` (committed, human-readable, review-friendly).
* Later runs: every field and the coordinate are diffed against the baseline with
  `numpy.allclose(rtol, atol)`; a regression raises `AssertionError` naming the
  offending quantities.
* Tolerances are stored in the baseline so captures and diffs stay consistent. Tune
  `rtol`/`atol` per slice via the `harness.verify(...)` arguments.

## Output locations

* Baselines: `tst/test_suite/verification/baselines/<name>.json` (committed).
* Plots: `tst/test_suite/verification/plots/<name>.png` (git-ignored build artifact).

Both are anchored to the harness module, so they land here regardless of the build
directory the test binary runs from.

See `test_verify_sod_cpu.py` for the worked Sod example.

## Experiment-anchoring substrate (PRD #138, ADR-0008)

`harness.verify` is *self-regression* (diff vs the run's own golden baseline). For the
MagLIF benchmarks the **binding** oracle is the experiment, served by `ground_truth_oracle`
(`GroundTruthOracle`, committed datums under `ground_truth/`). Three reusable pieces turn a
benchmark into an experiment-validated one — drop them into any `test_verify_*` slice:

* **`ground_truth_oracle.GroundTruthOracle`** — `oracle.compare(benchmark, datum_id, sim)`
  returns a `ComparisonResult` (pass iff `|sim-exp| <= max(exp_err, dig_err)`). A
  not-yet-digitized datum raises `PendingDatumError`; a datum missing provenance raises
  `ProvenanceError`. The experiment is always the oracle — FLASH/LASNEX values live in a
  datum's optional `secondary_reference` and are never binding.
* **`scorecard`** — record each verdict (`record_result(res)` / `record_pending(bench,
  obs, issue=NNN)`); the suite emits one verdict table at the end of the session (via
  `conftest.py`'s `pytest_terminal_summary`, written to `plots/scorecard.txt`). Pass
  `binding=False` for a quantity the reduced run only *reports* (e.g. an absolute SI
  scalar or EOS-sensitive ratio an ideal-gamma run cannot reproduce — the B3
  velocity-ratio policy). Pending anchors are reported as `PENDING (#NNN)`, **never** a
  pass.
* **`experiment_overlay.overlay_scalars`** — emits `plots/<name>_overlay.png` overlaying
  each scalar's experimental value + tolerance band (and any FLASH secondary, dashed) on
  the simulation series, in physical units.

Worked example: `cylindrical/test_verify_maglif_icf_cpu.py` (B4 ICF, the first slice using
this substrate). The curve benchmarks (#142/#143) plug a `kind: curve` oracle comparison
into the same scorecard + overlay path.
