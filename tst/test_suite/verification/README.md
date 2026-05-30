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
  * **Curve datums** (`kind: "curve"`, for the growth-curve benchmarks B1 #120 / B2 #122):
    the datum carries a `points` list (`{x, y, experimental_error, digitization_error}`)
    plus a `unit` (y) and `x_unit` (abscissa). Pass the **simulation curve** as
    `sim = (x_sim, y_sim)`; the oracle interpolates the sim onto the experiment's abscissa
    over the **overlapping support** `[max(min x), min(max x)]` (no extrapolation), tests
    each compared point within its own `max(exp_err, dig_err)` band, and returns a
    `CurveComparisonResult` (`.point_results`, `.n_compared`, `.n_failed`, `.passed`,
    `.worst_point`). The aggregate passes iff every compared point is in band **and** at
    least one point overlapped (an empty-support curve is a FAIL, not a vacuous pass).
    `.worst_point` is a plain `ComparisonResult` you can hand to `scorecard.record_result`.
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
  * **`experiment_overlay.overlay_curve`** (#142 [VA3]) — the curve analogue, emitting
    `plots/<name>_growth_overlay.png`: the simulation growth curve as a marked line with
    the digitized experimental points + per-point band overlaid. Pass `exp_points` as a
    list of `{x, y, band}` (the absolute per-point band, e.g. from the oracle's
    `CurveComparisonResult.point_results`); pass `exp_points=None` (with `pending_issue`)
    for a not-yet-digitized curve — the sim curve is still plotted, annotated
    `pending digitization (#NNN)`, never a fabricated band.

Worked examples: `cylindrical/test_verify_maglif_icf_cpu.py` (B4 ICF scalars, #140 [VA1]),
`cylindrical/test_verify_maglif_mrt_cpu.py` (B1 single-mode MRT growth *curve*, #142 [VA3])
and `cylindrical/test_verify_maglif_mmrt_cpu.py` (B2 multi-mode MRT growth *curve*, #143
[VA4]; its curve-anchor dispatch is covered red-first by `test_verify_b2_curve_anchor_cpu.py`).
The curve-comparison engine itself is unit-tested in isolation in
`test_verify_ground_truth_oracle_cpu.py` (#141 [VA2]); the B1/B2 growth-curve benchmarks
(#142/#143) plug a `kind: curve` oracle comparison into the same scorecard + overlay path.
A reduced nondimensional `_cpu` surrogate emits its curve in code units and *reports* the
verdict (`binding=False`); the absolute experimental comparison is the paper-resolution SI
run (#120/#122). An un-digitized curve datum stays `pending_digitization` and is reported
as `PENDING`, never a pass (ADR-0008 — no fabricated points/tolerances).

The B3 converging-RM anchor follows the same report-vs-assert split on *scalar* observables:
`cylindrical/test_verify_maglif_rm_anchor_cpu.py` (#144 [VA5]) runs a compact code-unit
converging RM (`inputs/maglif_rm_anchor.athinput`, coupled stack off) on the per-PR CPU path,
reduces it to the dimensionless `rm_growth_factor` with the **same definition** as the GPU
anchor `cylindrical/test_verify_maglif_rm_si_gpu.py` (#119), and *reports* both that growth
factor and the EOS-sensitive shock/interface velocity ratio against the B3 oracle
(`binding=False`) — the compact CPU surrogate can't develop the experimental RM growth that
the paper-resolution GPU SI run (nx1=512, tlim=4.5) does, so the binding magnitude hard-assert
stays on the GPU run while the binding CPU gate is the cheap qualitative convergence signature
(converge + rebound, seeded mode dominant, finiteness). The GPU `_gpu` anchor is untouched.
