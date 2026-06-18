"""
Grid-convergence verdict for the faithful B1 coupled benchmark (issues #195 / #192).

PURE: no I/O, no GPU, no build deps (stdlib only).  Imported by the GPU integration gate
(test_verify_maglif_b1_coupled_gpu) so the gate's BINDING convergence assert and the CPU
unit test (test_unit_maglif_grid_convergence_cpu) share one definition of "converged" and
can never drift apart -- and so the band logic is exercised offline against the recorded
same-build bracket data without a GPU run.

WHY THESE BANDS (issue #195).  #192's last open AC was "grid convergence REPORTED, not
binding".  With the resb div(B) write-back runaway fixed, the same-build #195 bracket
(athenakdev:~/p192/runs/bracket195/bracket.json) showed the implosion is grid-converged
between the refined (432x4x192) and 2x (576x4x256) legs, while the paper (288x4x128) leg
under-resolves it:

  metric              refined 432   2x 576    dev (vs 2x)   paper 288   paper-vs-ref dev
  rho_max  [code]       16.1065      16.0889     0.11%         2.892          82%
  amp_peak [mm]          0.02746      0.02548    7.8%          0.03999        46%
  t_peak   [ns]          57.0         54.0        3.0 ns        70.0           13.0 ns

Each band sits comfortably ABOVE the converged grid-pair deviation (and any same-grid GPU
run-to-run / 3-ns-snapshot-quantization noise) and far BELOW the under-resolved paper-288
signal it must reject.  Paper-288 is the UNDER-RESOLVED outlier and is excluded from the
binding pair (it is kept as a separate stability/red->green anchor in the gate, and its
under-resolution is REPORTED there).

  * CONV_RHO_RTOL = 0.05  -- implosion strength (peak density).  PRIMARY metric: the #195
      headline and the cleanest-converged quantity (0.11% measured, >= 40x margin).  It is
      what discriminates "resolved implosion" (~16) from "under-resolved" (paper ~2.9).
  * CONV_AMP_RTOL = 0.15  -- seeded-mode peak amplitude [mm].  TIGHTENED from the a-priori
      0.30 now that a converged pair exists (7.8% measured, ~2x margin); a delicate
      interface/mode-projection observable, so not tightened to the rho_max level.
  * CONV_TPEAK_TOL = 6.0  -- seeded-mode peak timing [ns].  = TWO 3-ns snapshot intervals;
      not tightened below the snapshot cadence (measured 3.0 ns = one interval).

Deviations are taken relative to the FINER (best-estimate) leg.

SCOPE / HONESTY.  This is a two-grid (Cauchy-style) agreement check between successive
refinements -- a standard production convergence gate -- NOT a formal Richardson
order-of-accuracy proof (no convergence order p is fit; that needs >=3 grids).
The bands are calibrated on the single #195 same-build bracket and the radiation-OFF
stack; they will need re-derivation when the physics stack changes (notably when FLD+mrad
re-enable under #194).  See ADR-0015 addendum 2.
"""

CONV_RHO_RTOL = 0.05   # implosion strength |rho_fine - rho_coarse| / rho_fine (primary)
CONV_AMP_RTOL = 0.15   # seeded-mode peak amplitude (tightened from the a-priori 0.30)
CONV_TPEAK_TOL = 6.0   # ns; seeded-mode peak timing (= 2 x 3-ns snapshot interval)


def grid_convergence_verdict(coarse, fine):
    """Verdict on whether the coarse->fine refinement has grid-converged.

    ``coarse`` / ``fine`` are dicts carrying the leg observables:
      ``rho_max``   peak density [code units],
      ``amp_peak``  seeded-mode peak amplitude [mm],
      ``t_peak_ns`` time of that peak [ns].
    ``coarse`` is the less-resolved leg, ``fine`` the more-resolved (e.g. refined 432 ->
    2x 576).  Deviations are relative to the finer (best-estimate) leg.

    Returns a report dict: the per-metric deviations (``rho_dev``, ``amp_dev``,
    ``tpk_dev``), the bands in force, a human-readable ``failures`` list, and
    ``converged`` (True iff every binding metric is within band).
    """
    rho_dev = abs(fine["rho_max"] - coarse["rho_max"]) / abs(fine["rho_max"])
    amp_dev = abs(fine["amp_peak"] - coarse["amp_peak"]) / abs(fine["amp_peak"])
    tpk_dev = abs(fine["t_peak_ns"] - coarse["t_peak_ns"])
    failures = []
    if rho_dev > CONV_RHO_RTOL:
        failures.append(
            f"rho_max dev {rho_dev:.4f} > {CONV_RHO_RTOL} (implosion not converged)"
        )
    if amp_dev > CONV_AMP_RTOL:
        failures.append(
            f"seeded-mode peak amp dev {amp_dev:.4f} > {CONV_AMP_RTOL}"
        )
    if tpk_dev > CONV_TPEAK_TOL:
        failures.append(
            f"seeded-mode peak timing dev {tpk_dev:.2f} ns > {CONV_TPEAK_TOL} ns"
        )
    return {
        "rho_dev": rho_dev,
        "amp_dev": amp_dev,
        "tpk_dev": tpk_dev,
        "rho_rtol": CONV_RHO_RTOL,
        "amp_rtol": CONV_AMP_RTOL,
        "tpk_tol": CONV_TPEAK_TOL,
        "failures": failures,
        "converged": not failures,
    }
