"""
Grid-convergence band logic for the faithful B1 coupled benchmark (issues #195 / #192).

Pure-CPU (no GPU, no build) red->green anchor for the BINDING grid-convergence verdict the
coupled GPU gate (test_verify_maglif_b1_coupled_gpu) asserts.  It exercises
``maglif_grid_convergence.grid_convergence_verdict`` against the recorded SAME-BUILD #195
bracket data (athenakdev:~/p192/runs/bracket195/bracket.json) so the band -- and the claim
that the implosion is grid-converged between the refined (432x4x192) and 2x (576x4x256)
legs while the paper (288x4x128) leg is the UNDER-RESOLVED outlier -- can be checked
offline, without a GPU.

The #192 last open AC was "grid convergence is REPORTED, not binding".  #195 resolved it:
with the div(B) runaway fixed, the refined and 2x legs converge to 0.11% on peak density
(16.106 vs 16.089) while the paper leg under-resolves the implosion (rho 2.89).  This test
pins the verdict that lets the gate flip that AC to BINDING:
  * refined <-> 2x  -> CONVERGED  (green: every binding metric within band), and
  * paper  <-> refined -> NOT converged (the under-resolved signal the gate must reject).
"""

# Modules
from test_suite.cylindrical.maglif_grid_convergence import (
    CONV_AMP_RTOL, CONV_RHO_RTOL, CONV_TPEAK_TOL, grid_convergence_verdict,
)

# Same-build #195 bracket legs (athenakdev:~/p192/runs/bracket195/bracket.json):
# {rho_max [code], amp_peak [mm] = seeded-mode peak amplitude, t_peak_ns [ns]}.
PAPER = {"rho_max": 2.892040, "amp_peak": 0.0399927, "t_peak_ns": 70.000}
REFINED = {"rho_max": 16.106453, "amp_peak": 0.0274593, "t_peak_ns": 57.000}
TWOX = {"rho_max": 16.088915, "amp_peak": 0.0254803, "t_peak_ns": 54.000}


def test_refined_2x_is_grid_converged():
    """The converged pair (refined 432 -> 2x 576) passes every binding metric."""
    v = grid_convergence_verdict(coarse=REFINED, fine=TWOX)
    assert v["converged"], f"refined<->2x should be converged; failures={v['failures']}"
    assert not v["failures"]


def test_paper_refined_is_rejected_as_under_resolved():
    """The under-resolved paper leg (288) must FAIL convergence vs the refined leg --
    and specifically on the implosion-strength (rho_max) metric, the #195 headline."""
    v = grid_convergence_verdict(coarse=PAPER, fine=REFINED)
    assert not v["converged"], "paper<->refined must NOT pass (paper-288 under-resolved)"
    assert any("rho_max" in f for f in v["failures"]), (
        f"the under-resolved paper rho (2.89 vs 16) must trip the rho_max band; "
        f"failures={v['failures']}"
    )


def test_band_cleanly_separates_converged_from_under_resolved():
    """The rho_max band sits between the converged grid-pair dev and the paper signal it
    must reject -- a clean separation, not a knife-edge."""
    conv = grid_convergence_verdict(coarse=REFINED, fine=TWOX)["rho_dev"]
    under = grid_convergence_verdict(coarse=PAPER, fine=REFINED)["rho_dev"]
    assert conv < CONV_RHO_RTOL < under, (
        f"converged dev {conv:.4f} < band {CONV_RHO_RTOL} < paper dev {under:.4f}"
    )
    # measured grid-pair convergence is ~0.11%, well inside the 5% band (>= 40x margin).
    assert conv < 0.01


def test_secondary_amp_and_timing_bands_hold_on_converged_pair():
    """The tightened seeded-mode amplitude band (0.15, down from the a-priori 0.30) and
    the timing band (one extra 3-ns snapshot) hold on the converged refined<->2x pair."""
    v = grid_convergence_verdict(coarse=REFINED, fine=TWOX)
    assert v["amp_dev"] < CONV_AMP_RTOL, f"amp dev {v['amp_dev']:.4f} >= {CONV_AMP_RTOL}"
    assert v["tpk_dev"] <= CONV_TPEAK_TOL, f"t_peak dev {v['tpk_dev']:.3f} ns"
