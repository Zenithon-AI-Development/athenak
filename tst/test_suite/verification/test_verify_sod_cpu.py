"""
Planar Sod shock-tube verification (Sod 1978).

Runs AthenaK's built-in ``shock_tube`` problem generator with the canonical Sod initial
states (rho, u, p) = (1, 0, 1) | (0.125, 0, 0.1), gamma=1.4, interface at x=0, to t=0.25.
The test:

  * runs Sod once,
  * compares the numerical density and pressure profiles at t=0.25 to the EXACT Sod
    Riemann solution (shared exact solver ``verification/exact_riemann.py``) and asserts
    the relative L1 and the max (Linf) errors are within documented tolerances,
  * emits a standard diagnostic plot of the profiles to ``verification/plots/sod.png``,
  * captures/diffs a versioned golden baseline (``verification/baselines/sod.json``) as a
    secondary regression guard.

Oracle: Layer 1 -- analytic.  Exact Sod (1978) Riemann solution at t=0.25, computed by the
standard Toro (2009, ch. 4) exact Riemann solver (shared with ``test_verify_cyl_sod_cpu``:
Newton solve for the star-region pressure p*=0.30313, u*=0.92745, then self-similar
sampling of the rarefaction/contact/shock structure).  The binding assertion is that the
relative L1 error of the density and pressure profiles is within tolerance; the Linf error
is bounded by the ~1-cell smearing of the contact/shock (a shock-capturing scheme is only
first order at discontinuities, so Linf does not vanish).  The harness.verify baseline is
a regression guard only.

This is auto-collected by ``run_test_suite.py`` (no runner edits) because the module name
contains ``_cpu``.
"""

# Modules
import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness
from test_suite.verification.exact_riemann import exact_riemann
import athena_read

input_file = "inputs/sod.athinput"

# Sod initial states (must match inputs/sod.athinput) and run parameters.
GAMMA = 1.4
WL = (1.0, 0.0, 1.0)     # (rho, u, p) left
WR = (0.125, 0.0, 0.1)   # (rho, u, p) right
XSHOCK = 0.0
TFINAL = 0.25


def test_verify_sod():
    """Run Sod; verify the t=0.25 profiles vs the exact Riemann solution + baseline."""
    try:
        # Fixed, deterministic configuration so the baseline is reproducible.
        results = testutils.run(input_file, ["job/basename=sod"])
        assert results, "Sod verification run failed."
        data = athena_read.tab("tab/sod.hydro_w.00001.tab")
        x1v = np.asarray(data["x1v"])
        dens = np.asarray(data["dens"])
        velx = np.asarray(data["velx"])
        pres = (GAMMA - 1.0) * np.asarray(data["eint"])  # w(IEN) = p/(gamma-1)

        # Exact planar Sod solution at the same cell centres.
        xi = (x1v - XSHOCK) / TFINAL
        rho_ex, _, p_ex = exact_riemann(xi, GAMMA, WL, WR)

        # Layer-1 oracle.  Relative L1 (the binding convergence metric) is ~0.6% (rho) and
        # ~0.5% (p) at 256 cells; the max-norm Linf is set by the ~1-cell contact/shock
        # smearing of the LLF+PLM scheme (first order at discontinuities -> ~0.07 in rho,
        # ~0.10 in p).  Tolerances carry ~2-3x headroom over that measured residual.
        l1_dens = np.mean(np.abs(dens - rho_ex)) / np.mean(np.abs(rho_ex))
        l1_pres = np.mean(np.abs(pres - p_ex)) / np.mean(np.abs(p_ex))
        linf_dens = np.max(np.abs(dens - rho_ex))
        linf_pres = np.max(np.abs(pres - p_ex))
        assert l1_dens < 0.02, f"Sod density L1 vs analytic too large: {l1_dens:g}"
        assert l1_pres < 0.02, f"Sod pressure L1 vs analytic too large: {l1_pres:g}"
        assert linf_dens < 0.15, f"Sod density Linf vs analytic too large: {linf_dens:g}"
        assert linf_pres < 0.22, f"Sod pressure Linf vs analytic too large: {linf_pres:g}"

        harness.verify(
            "sod",
            x1v,
            {
                "dens": dens,
                "velx": velx,
                "eint": data["eint"],
            },
            coord_label="x1v",
            xlabel="x",
            title="Sod shock tube (t=0.25)",
        )
    finally:
        testutils.cleanup()
