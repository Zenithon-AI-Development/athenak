"""
Grey flux-limited radiation diffusion (FLD) verification (ADR-0001, issue #17).

Drives the grey FLD operator (FLDGreyOperator, Larsen flux limiter) with the RKL2
super-time-stepping integrator via the ``radiation_fld_marshak`` user pgen (built/run with
testutils.run_pgen), then checks two things and stores golden baselines + plots through
the shared verification harness:

  1. Analytic Marshak wave.  An optically-thick cold slab heated from a Dirichlet inner-x1
     radiation source reproduces, in the equilibrium-diffusion limit (Larsen lambda ->
     1/3, D = c/(3 chi)), the error-function half-space solution
         E_r(x,t) = e_floor + (e_source - e_floor) * erfc( (x - x1min) / (2 sqrt(D t)) ).
     The test compares the run profile to this analytic solution over the well-resolved
     interior (L1 / Linf tolerance) and baselines E_numeric vs E_analytic.

  2. STS substage-count sweep (the ADR-0001 "measure-first" go/no-go).  The per-cell
     optical depth tau = chi*dx is swept thick->thin; the adaptive RKL2 substage count
     must grow as the medium thins (the diffusive dt collapses ~ tau*dx/c -- the
     optically-thin c-CFL cost STS cannot accelerate).  The test asserts the count rises
     monotonically toward the thin end and baselines the (tau_cell -> nstages) curve.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

_erfc = np.vectorize(math.erfc)

# Problem (must match inputs/radiation_fld/marshak.athinput).
PROBLEM = "radiation_fld_marshak"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "marshak_fld.athinput"
)
C_LIGHT, CHI = 1.0, 1.0e3
E_SOURCE, E_FLOOR = 1.0, 0.1
TLIM, X1MIN = 50.0, 0.0
D_DIFF = C_LIGHT / (3.0 * CHI)            # equilibrium-diffusion coefficient
# Interior window: clear of the immediate boundary cell and inside the cold tail.
WIN_LO, WIN_HI = 0.01, 0.5


def _marshak_exact(x):
    """Analytic erfc Marshak wave in the equilibrium-diffusion limit."""
    z = (x - X1MIN) / (2.0 * np.sqrt(D_DIFF * TLIM))
    return E_FLOOR + (E_SOURCE - E_FLOOR) * _erfc(z)


def test_verify_marshak_fld():
    """Run the grey FLD driver; verify the Marshak wave + the STS substage sweep."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    prof_path = os.path.join(run_dir, "marshak_fld_profile.dat")
    sub_path = os.path.join(run_dir, "marshak_fld_substages.dat")
    try:
        assert testutils.run_pgen(PROBLEM, input_file), "grey FLD Marshak run failed"

        # -- Part 1: analytic Marshak wave --
        prof = np.loadtxt(prof_path)
        x, e_num = prof[:, 0], prof[:, 1]
        e_an = _marshak_exact(x)
        mask = (x > WIN_LO) & (x < WIN_HI)
        l1 = float(np.mean(np.abs(e_num[mask] - e_an[mask])))
        linf = float(np.max(np.abs(e_num[mask] - e_an[mask])))
        # FLD in the diffusion limit reproduces the analytic erfc wave to ~1e-4 of the
        # source-to-floor contrast; require well under 1% for margin yet catch a break.
        contrast = E_SOURCE - E_FLOOR
        assert l1 < 5.0e-3 * contrast, f"Marshak L1 too large: {l1:.3e}"
        assert linf < 1.0e-2 * contrast, f"Marshak Linf too large: {linf:.3e}"

        harness.verify(
            "marshak_fld",
            x[mask].tolist(),
            {
                "E_numeric": e_num[mask].tolist(),
                "E_analytic": e_an[mask].tolist(),
            },
            coord_label="x1v",
            xlabel="x",
            title="Grey FLD Marshak wave vs analytic erfc (D = c/3chi)",
            rtol=1.0e-4,
            atol=1.0e-6,
        )

        # -- Part 2: STS substage-count sweep over optical depth (thick -> thin) --
        sweep = np.loadtxt(sub_path)
        tau, nstages = sweep[:, 0], sweep[:, 2]
        # tau decreases down the table (thick -> thin); substage count must rise.
        assert all(
            nstages[i + 1] >= nstages[i] for i in range(len(nstages) - 1)
        ), f"substage count not rising as medium thins: {nstages}"
        assert nstages[-1] > 5 * nstages[0], (
            f"optically-thin end should need many more substages: {nstages}"
        )

        harness.verify(
            "marshak_fld_substages",
            tau.tolist(),
            {"nstages": nstages.tolist()},
            coord_label="tau_cell",
            xlabel="per-cell optical depth tau = chi*dx",
            title="Grey FLD: RKL2 substage count vs optical depth (thick->thin)",
            rtol=1.0e-9,
            atol=0.5,
        )
    finally:
        for p in (prof_path, sub_path):
            try:
                os.remove(p)
            except OSError:
                pass
        testutils.cleanup()
