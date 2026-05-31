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
`cylindrical/test_verify_maglif_mrt_cpu.py` (B1 single-mode MRT growth *curve*, #142 [VA3];
the Sinars-2011 curve is digitized + committed in #154 [VA6], so its dispatch is covered
red-first by `test_verify_b1_curve_anchor_cpu.py`) and
`cylindrical/test_verify_maglif_mmrt_cpu.py` (B2 multi-mode MRT growth *curve*, #143 [VA4];
its curve-anchor dispatch is covered red-first by `test_verify_b2_curve_anchor_cpu.py`).
The curve-comparison engine itself is unit-tested in isolation in
`test_verify_ground_truth_oracle_cpu.py` (#141 [VA2]); the B1/B2 growth-curve benchmarks
(#142/#143) plug a `kind: curve` oracle comparison into the same scorecard + overlay path.
A reduced nondimensional `_cpu` surrogate emits its curve in code units; per PRD #138
("reduce to the dimensionless observable") it is reduced to its dimensionless growth-factor
trajectory `G(t)=a/a0` mapped onto the experiment's seed amplitude + observation window
(only the *sim* observable is normalized — no experimental point/tolerance is altered) and
*reports* the verdict (`binding=False`); the absolute experimental comparison is the
paper-resolution SI run (#120/#122). An un-digitized curve datum stays
`pending_digitization` and is reported as `PENDING`, never a pass (ADR-0008 — no fabricated
points/tolerances).

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

The **B2 multi-mode MRT** benchmark has a second, *qualitative-tier* arm on the GPU:
`cylindrical/test_verify_maglif_mmrt_gpu.py` (#122 [C5]) runs the multi-mode surface-roughness
implosion at paper resolution (`inputs/maglif_mmrt_si.athinput`, nx1=512 × nx3=128, coupled
stack **off** — ideal-MHD only, since the operator-split parabolic GPU path segfaults at
runtime, #139) and validates the **qualitative** limb-modulation behaviour: a synthetic
side-on radiograph (Abel projection) whose bright-limb modulation is broadband (spectral
participation ≥ 2 modes, no single-mode collapse), driven-surface-dominant (the MagLIF
asymmetry: the magnetically-driven outer interface roughens far more than the RT-stabilised
fuel side), and *grows* under the sustained acceleration. Per the tiered bar (multi-mode MRT
from roughness is inherently stochastic) the verdict is recorded as `QUALITATIVE-MATCH`
(`binding=False`) via `scorecard.record` — it is **not** a quantitative curve match; the
McBride-2012 growth *curve* stays the reduced-`_cpu` reported anchor (#143). Same
report-vs-assert discipline, on a qualitative observable.

### Faithful single-mode B1 (Sinars): paper-resolution GPU replication (#120)

`cylindrical/test_verify_maglif_b1_sinars_gpu.py` (#120 [C3]) is the *faithful* single-mode
MRT replication — the dimensional AR=6 aluminum liner (#160 geometry, solid-Al density unit),
the measured z2173 drive (#160), a single seeded axial sinusoid at λ=400 µm (within the
published Sinars 25–400 µm range) at 20 µm seed amplitude, ideal-MHD core, run at the paper
resolution (12.5 µm finest) on the GPU. It runs the committed `inputs/maglif_b1_sinars.athinput`,
reduces to the seeded-mode amplitude history, and **reports** (`binding=False`) the
growth-factor `G(t)=a/a0` against the committed Sinars-2011 curve via the curve oracle. The
**binding** gate is the qualitative single-mode MRT signature — the liner converges, the seeded
mode e-folds (calibrated ~3× peak) then is crushed by deep convergence, it carries a sizeable
share of the interface structure at its peak, the synthetic radiograph limb modulates, and a
`pert_amp=0` control stays exactly 1-D. The faithful run measures a peak growth factor ~3× vs
the experiment's ~46×: the quantitative *match* (AC#2) is **reported, not asserted**, because an
ideal-MHD converging liner e-folds only order-unity times before stagnation. Closing AC#2 needs
the IONMIX tabulated EOS wired into the multi-material maglif IC (the solver path exists, #162,
but the vacuum/fill fall below the table's valid density range), an SI current/time calibration,
and a material-strength model (#P6). Same report-vs-assert discipline as the reduced `_cpu`
arm (`test_verify_maglif_mrt_cpu.py`, #142), now on the faithful paper-resolution GPU run.

### Faithful Ellison B2: qualitative Stage 1 (#163) + soft quantitative anchor (#155)

`cylindrical/test_verify_maglif_b2_ellison_cpu.py` (#163 [B2-S1]) is the *faithful* Ellison
benchmark-2 replication — every knob is the published value (AR=6 Be liner #160, measured
z2173 drive #160, Ellison Eq.16 random-**temperature** seed `perturbation=temperature`
dT=100 K #161, radiation off). It runs the committed `inputs/maglif_b2_ellison.athinput`
(the 12.5 µm GPU artifact) at reduced CPU resolution and **hard-asserts** only the
qualitative MRT morphology signature (multi-mode structure develops for dT>0, stays 1-D for
the dT=0 control, grows toward stagnation, dominant resolved sub-mm mode, multi-mode
participation). That qualitative gate is the binding contract — Ellison B2 has no
amplitude-vs-time curve to digitize and the authors disclaim absolute-time comparison.

The **Stage-2 soft quantitative anchor** (#155 [B2-S2]) rides on that same faithful run and
is strictly **report-only** (it never asserts, so it cannot fail the test):

* `mcbride_b2_anchor.py` holds the pure reductions: `outer_surface_extrema` (R_spike=max,
  R_bubble=min of the driven interface), `growth_fraction(R_spike,R_bubble,R0)`,
  `convergence_x(R_bubble,R0)`, and `linear_law`/`fit_laws` for the McBride printed fits.
* `growth_fraction = (R_spike-R_bubble)/(R0-R_bubble)` at deepest convergence is compared
  via the scalar oracle against the committed McBride band **[0.05, 0.15]** (datum
  `growth_fraction` in `b2_multimode_mrt_mcbride_2012.json`, encoded 0.10 ± 0.05) and
  recorded `binding=False`; a gf(t) overlay is drawn against the band. A persistent miss is
  **escalated as a "needs investigation (#155 Stage 3)" note**, never a hard fail (FLASH
  matched B2, so a miss is a diagnostic signal about our reduced setup, not a regression).
* The amplitude(µm) `= 450·x − 90` and wavelength(µm) `= 750·x` Fig. 7a/b fit laws are
  committed in the `fit_laws` block (reachable via `oracle.meta("B2")`, ignored by the
  datum loader) and the reduced-run values are overlaid against them
  (`plots/maglif_b2_fit_laws.png`); the *quantitative* law match is reported `PENDING` the
  paper-resolution dimensional SI run (the laws are only physical for x > 0.2, and the
  coarse CPU gate under-resolves µm-scale amplitudes). These are *stated published laws*,
  not figure-digitized points (ADR-0008). Dispatch is covered red-first by
  `test_verify_b2_growth_fraction_cpu.py`.

### Corrected v3 radius-vs-time trajectory anchors (#156 [VA8])

The figure-traced **trajectory** point clouds for B3 (Knapp 2020 converging-RM) and B4
(Knapp 2017 ICF) were human-QC-corrected to **v3** (B3 shock: spurious upper strand culled
→ 30 monotonic pts; B4 inner-radius: the 2 rebound points at r≈0.54 recovered and the
legend box excluded → 6 circles) and committed as `kind: curve` datums:
`rm_liner_trajectory` / `rm_shock_trajectory` (`b3_*.json`) and `inner_radius_trajectory`
(`b4_*.json`). B4's per-point experimental error is the **digitized published error bar**
(`kind: absolute`); B3's is a documented relative placeholder. The corrected v3 points live
under `ground_truth/digitization_review/extracted/extracted_points_v3.json` with the
reproducible `fix_b3.py` / `fix_b4.py` extraction and `*_overlay_v3.png` verification plots.

Both benchmarks (`test_verify_maglif_rm_anchor_cpu.py`, `test_verify_maglif_icf_cpu.py`)
*report* each trajectory against the oracle (`binding=False`) and overlay it via
`overlay_curve`, reproducing the v3 diagnostic figure. Because a reduced code-unit
trajectory is incommensurate with the experiment's mm/ns axis, each test maps the **sim
abscissa** onto the digitized window (`_map_to_window`) so every experimental point overlaps
for a point-by-point comparison **without extrapolation**; the **ordinate stays physically
calibrated** (a provisional length scale per test — `R_FUEL_MM`, `R_MM_PER_CODE`), so the
verdict is an honest reported deviation, not a forced fit. Only the *sim* curve is mapped —
no experimental point or tolerance is altered (ADR-0008). B4's `confinement_time` (a stated
text scalar in Knapp 2017: 14 ns measured vs 16 ns 1D) is likewise promoted out of
`pending_digitization` and *reported* via the scalar oracle (mapped to ns with the
provisional `NS_PER_CODE` time calibration). The dispatch is covered red-first by
`test_verify_b3b4_trajectory_anchor_cpu.py`. The absolute-SI binding asserts stay on the
GPU/SI runs (#119/#120/#121).

> Baseline note (#156): `baselines/maglif_rm_anchor.json` was stale — captured at
> `tlim=1.0` (101 pts) while the committed input is `tlim=0.85` (→ 86 pts), so the B3 anchor
> Layer-2 guard failed `101→86` on `main` regardless of this change. It was regenerated to
> the committed-input length; the new baseline is **byte-identical** to the old on the
> overlapping first 86 points (no physics drift — only the 15 phantom trailing points were
> trimmed).
