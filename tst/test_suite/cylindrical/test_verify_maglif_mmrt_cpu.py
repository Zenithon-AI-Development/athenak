"""MagLIF Benchmark 2: multi-mode magneto-Rayleigh-Taylor (MRT) — CPU verification.

Reduced nondimensional surrogate of the McBride-2012 multi-mode MRT experiment, wired
through the experiment-anchoring substrate (PRD #138, ADR-0008), mirroring the B1
single-mode wiring in the sibling test_verify_maglif_mrt_cpu.py (#142):

* the integrated multi-mode amplitude-growth history A(t) is compared against the
  committed McBride-2012 B2 curve datum via GroundTruthOracle's `kind: curve`
  comparison (#141), the verdict recorded in the suite scorecard and an overlay
  plot emitted;
* because the McBride-2012 figure is not yet digitized (paywalled; see #143), the
  curve datum is committed as `status: pending_digitization` and the comparison
  records a PENDING row (no fabricated points, per ADR-0008);
* the binding pass/fail gate stays the cheap qualitative signature (finiteness,
  net integrated-amplitude growth, the seeded-mode energy fraction) plus a
  radiograph contrast (feedthrough) sign check.

Run on CPU via `run_test_suite.py --cpu`.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from verification import harness  # noqa: E402
from verification import ground_truth_oracle as gto  # noqa: E402
from verification import scorecard  # noqa: E402
from verification import experiment_overlay as eo  # noqa: E402

# ---------------------------------------------------------------------------
# Reduced-surrogate parameters (toy, nondimensional). NOT paper-resolution.
# ---------------------------------------------------------------------------
NMODES = 8             # number of seeded azimuthal modes
SEED_AMP = 0.02        # per-mode seed perturbation amplitude
GROWTH_SIGN_MIN = 1.0  # integrated amplitude must show net growth (sign, not magnitude)


def _run():
    """Run the reduced multi-mode MRT surrogate and return diagnostics."""
    pin = harness.load_input("maglif_mmrt")
    pin["mesh"]["nx1"] = 64
    pin["mesh"]["nx2"] = 128
    pin["time"]["tlim"] = 0.6
    pin["problem"]["mmrt_nmodes"] = NMODES
    pin["problem"]["mmrt_seed_amp"] = SEED_AMP
    out = harness.run_sim("maglif_mmrt", pin)
    return out


def _amp_history(out):
    """Integrated multi-mode amplitude history A(t) (sum over seeded modes)."""
    hist = out["history"]
    times = np.asarray([row["time"] for row in hist])
    amp = np.asarray([row["mmrt_amp_integrated"] for row in hist])
    return times, amp


def test_maglif_mmrt_growth_cpu():
    """B2 multi-mode MRT: net integrated-amplitude growth is the binding
    qualitative signature; the McBride-2012 growth curve is reported against the
    oracle (pending until the paywalled figure is digitized, #143)."""
    out = _run()
    times, amp = _amp_history(out)

    # --- Binding qualitative-signature gate (cheap, self-anchored sanity) ---
    assert np.all(np.isfinite(amp)), "amplitude history must be finite"
    assert amp[0] > 0.0, "seed amplitude must be positive"
    growth = float(amp[-1] / (amp[0] + 1e-30))
    assert growth > GROWTH_SIGN_MIN, (
        f"integrated amplitude must show net MRT growth; ratio "
        f"{growth:.3f} <= {GROWTH_SIGN_MIN}")

    # --- Reported (non-binding): multi-mode growth vs McBride-2012 curve ---
    oracle = gto.GroundTruthOracle()
    datum = oracle.get_datum("b2_multimode_mrt", "mmrt_growth_curve")
    exp_points = None
    if datum.is_pending:
        scorecard.record_pending(
            "b2_multimode_mrt", "mmrt_growth_curve", issue=143)
    else:
        res = oracle.compare(
            "b2_multimode_mrt", "mmrt_growth_curve", (times, amp))
        scorecard.record_result(res.worst_point, binding=False)
        exp_points = [{"x": p.x, "y": p.exp_value, "band": p.band}
                      for p in res.point_results]

    # Overlay plot (sim growth curve; exp band when digitized).
    eo.overlay_curve(
        "maglif_mmrt", times, amp, exp_points=exp_points,
        pending_issue=(143 if datum.is_pending else None),
        xlabel="time (code units)", ylabel="integrated multi-mode amplitude")


def test_maglif_mmrt_spectrum_cpu():
    """Seeded modes must hold most of the perturbation energy (no grid noise
    blowup); spectrum must stay finite."""
    out = _run()
    spec = np.asarray(out["final"]["mode_amplitudes"])
    assert np.all(np.isfinite(spec)), "mode spectrum must be finite"
    seeded = float(np.sum(spec[1:NMODES + 1]))
    total = float(np.sum(spec)) + 1e-30
    seeded_frac = float(seeded / total)
    assert seeded_frac >= 0.30, (
        f"seeded-mode fraction {seeded_frac:.3f} < 0.30")


def test_maglif_mmrt_radiograph_growth_cpu():
    """Radiograph contrast must increase (qualitative feedthrough signature)."""
    out = _run()
    series = out["radiograph"]
    early = float(np.mean(series[: len(series) // 4]))
    late = float(np.mean(series[-len(series) // 4:]))
    assert np.isfinite(early) and np.isfinite(late)
    assert late > early, "radiograph contrast should grow (feedthrough)"


# (no module-level execution; pytest collects the test_* functions above)
