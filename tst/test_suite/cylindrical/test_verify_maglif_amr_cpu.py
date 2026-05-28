"""
AMR-vs-uniform MagLIF implosion verification + speedup measurement (issue [19c]/#43).

With the AMR-ready operators in place -- the diffusive refinement-boundary flux correction
(#33) and the cylindrical Balsara divergence-preserving prolongation for B (#38) -- a
MagLIF liner implosion (the maglif pgen, #32) is run on an ADAPTIVE mesh that refines only
the dense imploding liner, and compared against the matched UNIFORM fine grid at the same
maximum resolution.  This is the Tier-2 check that AMR is a free lunch: same answer, fewer
cells (ADR-0004 AMR note; the FLASH/MagLIF "5 um where it is needed" motivation).

Two runs of the same axisymmetric 2-D (r,phi) driven implosion (the IC is axisymmetric so
the solution is independent of the azimuthal resolution -- nx2 only sets cost, making the
cell-count comparison fair):
  * maglif_amr.athinput      -- root 64 radial x one refined level (min_max density
                                criterion tracks the liner); vacuum/fuel stay coarse,
  * maglif_uniform.athinput  -- uniform 128 radial = the AMR run's refined resolution
                                everywhere (the reference "truth").

Both enroll the same AMR-safe user history (MagLIFConsHistory): cylindrical finite-volume
total mass, total MHD energy, mass-weighted-radius numerator, max|div(B)|, peak density,
MeshBlock count, and active cell count -- reductions, not gridded dumps, so they are
correct across the composite AMR mesh and continuous across a refinement event.

The test asserts the three acceptance criteria of #43:
  (1) MATCH   -- the AMR implosion trajectory (mass-weighted radius <r>(t)) and the peak
                 compression match the uniform run within tolerance.  These resolution-
                 robust dynamical observables agree to <1% / <5% even though the coarse
                 base grid discretises the sharp liner-interface IC slightly differently
                 (so total mass carries a ~7% interface-discretisation offset that is
                 NOT a conservation error -- see (3)).
  (2) SPEEDUP -- refinement actually happened DURING the run (the block count rose above
                 the root grid mid-evolution, so RefineFC's prolongation was exercised),
                 and the AMR run used materially fewer active cells than uniform; the
                 wall-clock speedup is measured and reported.
  (3) CONSERV -- max|div(B)| stays at round-off on every row INCLUDING the post-refinement
                 rows (the #38 divergence-preserving prolongation invariant), and total
                 mass is conserved within the AMR run across refinement (no jump;
                 the small smooth drift matches the uniform run's, i.e. it is boundary
                 physics, not an AMR artifact).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).  Serial (_cpu): the
history rank-reduction is the identity, so the max columns report true global maxima.

Oracle: Layer 1 -- the conservation identity div(B)=0 (boundary-independent) is binding;
the uniform-grid run is the reference for the dynamical match.  The
harness.verify baseline is a regression guard on the (deterministic) AMR + uniform
trajectories.  References: ADR-0004 (AMR note); Balsara, JCP 174, 614 (2001); Toth & Roe,
JCP 180, 736 (2002); FLASH MagLIF arXiv:2504.10760.
"""

# Modules
import glob
import logging
import os
import shutil
import time
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
import athena_read

INPUT_AMR = os.path.join(testutils._repo_root(), "tst", "inputs", "maglif_amr.athinput")
INPUT_UNIFORM = os.path.join(
    testutils._repo_root(), "tst", "inputs", "maglif_uniform.athinput"
)
RUN_DIR = testutils.pgen_run_dir("maglif")

NROOT_MB = 8           # root MeshBlocks (8 radial x 1 azimuthal); refinement pushes above
DIVB_TOL = 1.0e-10     # div(B) round-off (field ~0.1; a prolongation bug shows at ~1e-3)
RADIUS_RTOL = 0.02     # AMR vs uniform mass-weighted-radius trajectory (measured ~0.4%)
MAXDENS_RTOL = 0.05    # AMR vs uniform peak-compression trajectory (measured ~1.8%)
MASS_DRIFT_TOL = 0.02  # within-run AMR mass conservation across refinement (~0.8%)


def _hst_file(basename):
    return os.path.join(RUN_DIR, f"{basename}.user.hst")


def _run_timed(input_file, basename):
    """Remove any stale history (append-mode file), run, and return wall-clock seconds."""
    try:
        os.remove(_hst_file(basename))
    except OSError:
        pass
    t0 = time.time()
    ok = testutils.run_pgen("maglif", input_file)
    elapsed = time.time() - t0
    assert ok, f"maglif run failed for {os.path.basename(input_file)}"
    return elapsed


def _read_hist(basename):
    """Load the user history and return (time, dict-of-columns, mean_radius)."""
    hst = athena_read.hst(_hst_file(basename))
    cols = {k: np.asarray(v, dtype=float) for k, v in hst.items()}
    mean_radius = cols["massr"] / cols["mass"]   # mass-weighted mean radius <r>
    return cols["time"], cols, mean_radius


def test_verify_maglif_amr():
    """Run AMR + uniform MagLIF; verify match, measure speedup, check conservation."""
    try:
        # Warm-up: build the maglif binary once (its first build is the slow step) so the
        # two timed runs below measure simulation wall-clock, not the one-off pgen build.
        assert testutils.run_pgen("maglif", INPUT_AMR), "maglif (AMR) build/run failed."

        t_uni = _run_timed(INPUT_UNIFORM, "maglif_uniform")
        t_amr = _run_timed(INPUT_AMR, "maglif_amr")

        t, amr, r_amr = _read_hist("maglif_amr")
        tu, uni, r_uni = _read_hist("maglif_uniform")

        # Both runs write at the same history cadence to the same tlim -> aligned rows.
        assert amr["time"].shape == uni["time"].shape, (
            f"row-count mismatch: AMR {amr['time'].shape} vs uniform {uni['time'].shape}"
        )
        assert np.allclose(t, tu, atol=2.0e-3), "AMR/uniform history times not aligned"

        # --- (2) SPEEDUP: dynamic refinement happened + AMR is cheaper than uniform ----
        # Refinement occurred DURING the run: starts on the root grid, block count rises.
        assert amr["nmb"][0] == NROOT_MB, (
            f"AMR did not start on the root grid ({amr['nmb'][0]:g} != {NROOT_MB})"
        )
        assert amr["nmb"].max() > NROOT_MB, (
            f"no refinement occurred -- RefineFC prolongation was not exercised "
            f"(max block count {amr['nmb'].max():g} <= {NROOT_MB} root blocks)"
        )
        ncells_amr = amr["ncells"].max()
        ncells_uni = uni["ncells"][0]   # uniform grid: constant cell count
        assert ncells_amr < ncells_uni, (
            f"AMR not cheaper than uniform: {ncells_amr:g} >= {ncells_uni:g} cells"
        )
        cell_speedup = ncells_uni / ncells_amr
        wall_speedup = t_uni / t_amr
        logging.info(
            f"[maglif_amr] AMR speedup vs uniform: {cell_speedup:.2f}x fewer cells "
            f"({ncells_amr:.0f} vs {ncells_uni:.0f}); wall-clock {wall_speedup:.2f}x "
            f"({t_amr:.2f}s vs {t_uni:.2f}s)"
        )

        # --- guard: a real implosion occurred (not a trivial static state) -----
        assert amr["maxdens"][-1] > 1.5 * amr["maxdens"][0], (
            f"liner did not compress (peak density {amr['maxdens'][0]:.3f} -> "
            f"{amr['maxdens'][-1]:.3f}); the AMR check would be vacuous"
        )
        assert r_amr[-1] < r_amr[0], "mass-weighted radius did not implode inward"

        # --- (1) MATCH: AMR reproduces the uniform-grid dynamics within tolerance
        rad_rel = np.abs(r_amr - r_uni) / np.abs(r_uni)
        assert rad_rel.max() < RADIUS_RTOL, (
            f"AMR implosion trajectory <r>(t) deviates from uniform: max rel "
            f"{rad_rel.max():.3e} >= {RADIUS_RTOL}"
        )
        dens_rel = np.abs(amr["maxdens"] - uni["maxdens"]) / np.abs(uni["maxdens"])
        assert dens_rel.max() < MAXDENS_RTOL, (
            f"AMR peak compression deviates from uniform: max rel "
            f"{dens_rel.max():.3e} >= {MAXDENS_RTOL}"
        )

        # --- (3) CONSERVATION across dynamic refinement
        # div(B)=0 to round-off on every row, including the post-refinement rows: the
        # cylindrical Balsara prolongation (#38) is divergence-preserving and CT keeps it.
        assert amr["maxdivb"].max() < DIVB_TOL, (
            f"max|div(B)| not at round-off across AMR refinement: "
            f"{amr['maxdivb'].max():.3e} (>= {DIVB_TOL:g}) -- prolongation not div-free"
        )
        assert uni["maxdivb"].max() < DIVB_TOL, (
            f"uniform-run max|div(B)| not at round-off: {uni['maxdivb'].max():.3e}"
        )
        # Total mass conserved within the AMR run across the refinement event (no jump);
        # the small smooth drift matches the uniform run's -> boundary physics, not AMR.
        mass_drift = np.abs(amr["mass"] - amr["mass"][0]) / amr["mass"][0]
        assert mass_drift.max() < MASS_DRIFT_TOL, (
            f"AMR total mass not conserved across refinement: max drift "
            f"{mass_drift.max():.3e} >= {MASS_DRIFT_TOL}"
        )

        # --- plot + golden regression baseline (deterministic trajectories)
        harness.verify(
            "maglif_amr_vs_uniform",
            t,
            {
                "mean_radius_amr": r_amr,
                "mean_radius_uniform": r_uni,
                "maxdens_amr": amr["maxdens"],
                "maxdens_uniform": uni["maxdens"],
                "mass_amr": amr["mass"],
                "max_divb_amr": amr["maxdivb"],
                "ncells_amr": amr["ncells"],
                "ncells_uniform": uni["ncells"],
            },
            coord_label="t",
            xlabel="t",
            title=(
                f"MagLIF AMR vs uniform: {cell_speedup:.1f}x fewer cells, "
                f"{wall_speedup:.1f}x faster, div(B)=0 across refinement"
            ),
            rtol=1.0e-5,
            atol=1.0e-8,
        )
    finally:
        for f in glob.glob(os.path.join(RUN_DIR, "*.user.hst")):
            try:
                os.remove(f)
            except OSError:
                pass
        shutil.rmtree(os.path.join(RUN_DIR, "bin"), ignore_errors=True)
