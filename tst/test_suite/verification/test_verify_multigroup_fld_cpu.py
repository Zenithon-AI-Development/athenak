"""
Multigroup flux-limited radiation diffusion (FLD) verification (ADR-0001/0007, issue #24).

Drives the N-group FLD operator (FLDMultigroupOperator, per-group Larsen flux limiter,
per-group Rosseland diffusion coefficient) with the RKL2 super-time-stepping integrator
via the ``radiation_fld_multigroup`` user pgen (built/run with testutils.run_pgen), then
checks the group-resolved Marshak wave and stores a golden baseline + plot through the
shared verification harness.

Multigroup Marshak wave.  An optically-thick cold slab heated from a Dirichlet inner-x1
radiation source (all groups) reproduces, in the equilibrium-diffusion limit (Larsen
lambda -> 1/3), a PER-GROUP error-function half-space solution
    E_g(x,t) = e_floor + (e_source - e_floor) * erfc( (x - x1min) / (2 sqrt(D_g t)) ),
with a distinct coefficient D_g = c/(3 chi_g) for each group, where chi_g is the
extinction read from the tabulated Rosseland transport opacity.  The groups therefore
penetrate to different depths (a group-resolved diffusion test).  D_g is read from the
pgen's groups file (so the oracle is built from the actual table lookups, not
re-implemented here); each group's run profile is compared to its analytic erfc (L1 /
Linf) and all per-group profiles are baselined.

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import math
import os

import numpy as np
import test_suite.testutils as testutils
import test_suite.verification.harness as harness

_erfc = np.vectorize(math.erfc)

# Problem (must match inputs/radiation_fld/multigroup.athinput).
PROBLEM = "radiation_fld_multigroup"
input_file = os.path.join(
    testutils._repo_root(), "tst", "inputs", "multigroup_fld.athinput"
)
fixture = os.path.join(
    testutils._repo_root(), "inputs", "radiation_fld", "multigroup_opacity.cn4"
)
E_SOURCE, E_FLOOR = 1.0, 0.1
TLIM, X1MIN = 50.0, 0.0
# Interior window: clear of the immediate boundary cell and inside the cold tail.
WIN_LO, WIN_HI = 0.01, 0.6


def _marshak_exact(x, d_diff):
    """Analytic erfc Marshak wave for diffusion coefficient ``d_diff``."""
    z = (x - X1MIN) / (2.0 * np.sqrt(d_diff * TLIM))
    return E_FLOOR + (E_SOURCE - E_FLOOR) * _erfc(z)


def test_verify_multigroup_fld():
    """Run the multigroup FLD driver; verify the per-group Marshak waves."""
    run_dir = testutils.pgen_run_dir(PROBLEM)
    prof_path = os.path.join(run_dir, "multigroup_fld_profile.dat")
    grp_path = os.path.join(run_dir, "multigroup_fld_groups.dat")
    try:
        assert testutils.run_pgen(
            PROBLEM, input_file, args=[f"problem/opacity_file={fixture}"]
        ), "multigroup FLD Marshak run failed"

        # per-group (chi_g, D_g) from the run (oracle built from the actual table lookups)
        grp = np.atleast_2d(np.loadtxt(grp_path))
        chi_g, d_g = grp[:, 1], grp[:, 2]
        ngroups = grp.shape[0]

        # per-group numeric profiles: columns x, E_0, ..., E_{ngroups-1}
        prof = np.loadtxt(prof_path)
        x = prof[:, 0]
        mask = (x > WIN_LO) & (x < WIN_HI)
        contrast = E_SOURCE - E_FLOOR

        fields = {}
        for g in range(ngroups):
            e_num = prof[:, 1 + g]
            e_an = _marshak_exact(x, d_g[g])
            l1 = float(np.mean(np.abs(e_num[mask] - e_an[mask])))
            linf = float(np.max(np.abs(e_num[mask] - e_an[mask])))
            # Each group reduces to the analytic erfc to ~1e-4 of the contrast (the grey
            # FLD result); require well under 1% for margin yet catch a break.
            assert l1 < 5.0e-3 * contrast, f"group {g} Marshak L1 too large: {l1:.3e}"
            assert linf < 1.0e-2 * contrast, (
                f"group {g} Marshak Linf too large: {linf:.3e}"
            )
            fields[f"E_g{g}_numeric"] = e_num[mask].tolist()
            fields[f"E_g{g}_analytic"] = e_an[mask].tolist()

        # Group resolution: the groups must have distinct, monotonically ordered
        # diffusion coefficients (thicker chi -> smaller D -> shallower penetration).
        assert ngroups >= 2, f"expected multiple groups, got {ngroups}"
        assert all(
            d_g[i + 1] > d_g[i] * 1.2 for i in range(ngroups - 1)
        ), f"per-group D not distinct/ordered: D={d_g}, chi={chi_g}"
        # Deeper-penetrating (larger-D) group sits above the shallower one in the front.
        front = mask & (x > 0.2)
        assert np.all(
            prof[front, 1 + (ngroups - 1)] >= prof[front, 1] - 1.0e-9
        ), "largest-D group should penetrate deepest"

        harness.verify(
            "multigroup_fld",
            x[mask].tolist(),
            fields,
            coord_label="x1v",
            xlabel="x",
            title="Multigroup FLD Marshak waves vs per-group analytic erfc",
            rtol=1.0e-4,
            atol=1.0e-6,
        )
    finally:
        for p in (prof_path, grp_path):
            try:
                os.remove(p)
            except OSError:
                pass
        testutils.cleanup()
