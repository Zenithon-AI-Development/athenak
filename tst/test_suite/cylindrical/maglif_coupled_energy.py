"""
Shared coupled-stack energy/stability helpers for the MagLIF benchmarks (#116/[B3]).

Phase B re-runs benchmarks 1-4 with the full coupled radiation-conduction-MHD stack
(``<mhd> strang_split`` + ``fld_operator_split`` + ``acond_operator_split`` +
``mrad_coupling``; see the ``inputs/maglif_*.athinput`` files).  The MagLIF problem
generator writes a per-step conservation history (``<basename>.user.hst``) carrying the
MHD total energy column ``etot`` and -- only when the grey-FLD radiation field is live --
the radiation energy column ``erad`` (sum E_r dV over the cylindrical finite volume).

This module reads that history and asserts the *species-split* energy budget is physically
well-behaved.  The four benchmarks are magnetically DRIVEN implosions with an outflow
outer boundary, so the combined energy is NOT globally conserved (the constant-current
drive does work and the boundary leaks) -- the rigorous closed-box conservation of the
point-implicit matter-radiation coupling is verified separately by the #115 unit test.
What we assert here is that the coupled stack (a) actually ran (the ``erad`` column is
present), (b) kept the radiation field finite and non-negative, and (c) kept the combined
``etot + erad`` budget finite, positive, and physically bounded (no spurious blow-up or
collapse through the operator split).  These are the benchmark-level "energy conserved"
checks; the trajectory itself is pinned by the re-captured golden baseline.

Helper (not a ``test_*`` module, so pytest does not collect it directly).
"""

import os

import numpy as np

import test_suite.testutils as testutils


def user_hist_path(basename):
    """Absolute path to the MagLIF conservation history for run ``basename``."""
    return os.path.join(testutils.pgen_run_dir("maglif"), f"{basename}.user.hst")


def read_user_hist(basename):
    """Read the MagLIF ``user.hst`` into a dict of column-name -> 1-D float array.

    Column names come from the ``# [n]=name`` header AthenaK writes; the radiation column
    ``erad`` is present only for a coupled-stack run (grey FLD active).
    """
    path = user_hist_path(basename)
    assert os.path.exists(path), f"coupled-stack history not written: {path}"
    names = None
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                # the last commented line of the form "[1]=time  [2]=dt ..." is the header
                if "=" in s and "[" in s:
                    names = [tok.split("=", 1)[1]
                             for tok in s.lstrip("#").split() if "=" in tok]
                continue
            rows.append([float(v) for v in s.split()])
    assert names is not None, f"no column header found in {path}"
    arr = np.asarray(rows, dtype=float)
    # AthenaK appends to an existing history file, so a re-run (the suite reuses the build
    # dir) concatenates several runs.  Keep only the LAST run: the rows after the final
    # time reset (time column is column 0, monotonically increasing within one run).
    t = arr[:, 0]
    resets = np.where(np.diff(t) < 0.0)[0]
    start = (resets[-1] + 1) if len(resets) else 0
    arr = arr[start:]
    return {name: arr[:, i] for i, name in enumerate(names)}


def assert_coupled_energy_sane(basename):
    """Assert the coupled-stack species-split energy budget is physically well-behaved.

    Returns ``(t, etot, erad)`` for optional downstream use (e.g. baselining).
    """
    cols = read_user_hist(basename)
    assert "erad" in cols, (
        f"'erad' column absent in {basename}.user.hst -- the coupled radiation stack "
        f"did not run (grey FLD operator-split off?)"
    )
    t = cols["time"]
    etot = cols["etot"]
    erad = cols["erad"]
    assert len(t) > 4, f"too few history rows for {basename}: {len(t)}"

    # (a) the radiation field is finite and non-negative throughout (a*T^4 >= 0).
    assert np.all(np.isfinite(erad)), f"{basename}: erad went non-finite (unstable)"
    assert np.all(erad >= -1.0e-12), (
        f"{basename}: erad went negative (min {erad.min():.3e})"
    )

    # (b) the MHD total energy stays finite.
    assert np.all(np.isfinite(etot)), f"{basename}: etot went non-finite (unstable)"

    # (c) the combined species-split budget is finite, positive, and physically bounded --
    # no spurious creation (blow-up) or collapse through the operator split.  A driven
    # implosion injects work, so a generous upper bound (relative to the initial budget)
    # is the right sanity gate, not strict global conservation.
    esys = etot + erad
    assert np.all(np.isfinite(esys)), f"{basename}: etot+erad non-finite"
    assert np.all(esys > 0.0), f"{basename}: combined energy went non-positive"
    assert esys.max() < 1.0e4 * abs(esys[0]), (
        f"{basename}: combined energy blew up: {esys[0]:.3e} -> {esys.max():.3e}"
    )
    return t, etot, erad
