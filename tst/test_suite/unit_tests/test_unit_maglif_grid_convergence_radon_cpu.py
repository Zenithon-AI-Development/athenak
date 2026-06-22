"""
Radiation-ON grid-convergence band logic for the faithful B1 coupled benchmark (#199).

Pure-CPU (no GPU, no build) red->green anchor for the BINDING radiation-ON convergence
verdict the GPU gate (test_verify_maglif_b1_coupled_radon_conv_gpu) asserts.  It exercises
``maglif_grid_convergence.grid_convergence_verdict`` against the recorded SAME-BUILD #199
RADIATION-ON bracket (athenakdev:~/p192_logs/conv199_gate.log) with the RE-DERIVED
radiation-ON bands (RADON_CONV_*), NOT the radiation-OFF #195 bands -- so the band and
claim that the implosion is grid-converged between the refined (432x4x192) and 2x
(576x4x256) legs while paper (288x4x128) under-resolves, can be checked offline.

#194 made the full-physics stack NaN-clean; #199 supplies the missing CONVERGENCE verdict
(NaN-clean at two grids is stability, not convergence).  The verdict that lets the gate
bind:
  * refined <-> 2x  -> CONVERGED  (green: every binding metric within the radiation-ON
    band), and
  * paper  <-> refined -> NOT converged (the under-resolved signal the gate must reject).

NOTE (#199, pre-GPU-bracket): the leg values below are PLACEHOLDERS seeded from the
radiation-OFF #195 same-build bracket -- radiation is nearly inert at the constant FLD
opacity, so the radiation-ON bracket tracks them closely.  They are FINALIZED from the
recorded radiation-ON bracket once the #199 GPU run lands, with
the RADON_CONV_* values in maglif_grid_convergence and ADR-0015 addendum 3.
"""

# Modules
from test_suite.cylindrical.maglif_grid_convergence import (
    RADON_CONV_AMP_RTOL, RADON_CONV_RHO_RTOL, RADON_CONV_TPEAK_TOL,
    grid_convergence_verdict,
)

# Radiation-ON #199 bracket legs (PLACEHOLDER = #195 priors; finalize from the GPU run):
# {rho_max [code], amp_peak [mm] = seeded-mode peak amplitude, t_peak_ns [ns]}.
PAPER = {"rho_max": 2.892040, "amp_peak": 0.0399927, "t_peak_ns": 70.000}
REFINED = {"rho_max": 16.106453, "amp_peak": 0.0274593, "t_peak_ns": 57.000}
TWOX = {"rho_max": 16.088915, "amp_peak": 0.0254803, "t_peak_ns": 54.000}

_BANDS = dict(rho_rtol=RADON_CONV_RHO_RTOL, amp_rtol=RADON_CONV_AMP_RTOL,
              tpk_tol=RADON_CONV_TPEAK_TOL)


def test_refined_2x_is_grid_converged_radon():
    """The converged pair (refined 432 -> 2x 576) passes every radiation-ON band."""
    v = grid_convergence_verdict(coarse=REFINED, fine=TWOX, **_BANDS)
    assert v["converged"], f"refined<->2x should be converged; failures={v['failures']}"
    assert not v["failures"]


def test_paper_refined_is_rejected_as_under_resolved_radon():
    """The under-resolved paper leg (288) must FAIL convergence vs the refined leg --
    specifically on the implosion-strength (rho_max) metric."""
    v = grid_convergence_verdict(coarse=PAPER, fine=REFINED, **_BANDS)
    assert not v["converged"], "paper<->refined must NOT pass (paper-288 under-resolved)"
    assert any("rho_max" in f for f in v["failures"]), (
        f"the under-resolved paper rho must trip the rho_max band; "
        f"failures={v['failures']}"
    )


def test_radon_band_cleanly_separates_converged_from_under_resolved():
    """The radiation-ON rho_max band sits between the converged grid-pair dev and the
    paper signal it must reject -- a clean separation, not a knife-edge."""
    conv = grid_convergence_verdict(coarse=REFINED, fine=TWOX, **_BANDS)["rho_dev"]
    under = grid_convergence_verdict(coarse=PAPER, fine=REFINED, **_BANDS)["rho_dev"]
    assert conv < RADON_CONV_RHO_RTOL < under, (
        f"converged dev {conv:.4f} < band {RADON_CONV_RHO_RTOL} < paper dev {under:.4f}"
    )
