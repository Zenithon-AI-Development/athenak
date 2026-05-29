"""
Strang-split coupled-timestep integrated test ([B2]/#115, ADR-0009).

Builds src/pgen/unit_tests/strang_coupling_test.cpp
(-D PROBLEM=unit_tests/strang_coupling_test) into the isolated tst/build_unit directory
and RUNS THE DRIVER (nlim>0) in two modes, asserting each exits 0 (all checks passed):

* mode=uniform -- the per-cell point-implicit matter-radiation coupling runs OUTSIDE the
  RKL2 super-step and is NOT inflated by the substage count.  Uniform radiation on a
  uniform gas at rest: the grey-FLD spatial diffusion is the identity on E_r and the hydro
  update is inert, yet the FLD super-step still runs s>2 substages (the pgen reports
  s=25).  After ONE Strang-orchestrated step the (E_r, e_gas) state equals EXACTLY two
  backward-Euler coupling half-steps of dt/2 (a host oracle) -- which it would NOT if the
  coupling were evaluated inside the substage loop (2*s applications of dt/(2*s)).  This
  is RED if the coupling is mis-placed inside the super-step and GREEN with the Strang
  orchestration of #115 (verified by a controlled break of the half-step dt).

* mode=bump -- the fully-coupled stack conserves total energy across the species split.
  A smooth radiation bump on a uniform gas, advanced by the full orchestration (insulated
  grey-FLD half super-steps + point-implicit matter-radiation coupling + conservative
  hydro) on TWO MeshBlocks.  The volume-weighted total energy sum_i (E_r,i + E_gas,i) dV
  (radiation + gas, the two species) is conserved to round-off across the run.

Oracle: Layer 1 -- (uniform) closed-form backward-Euler quartic (the #23 point-implicit
coupling, applied on the host); (bump) exact conservation of a conserved scalar.  Both are
method-independent analytic anchors (ADR-0008).

Auto-collected by run_test_suite.py (module name contains ``_cpu``).
"""

# Modules
import os

import test_suite.testutils as testutils

NAME = "strang_coupling_test"


def test_run():
    """Build + run both Strang-coupling modes on a single rank; pass iff each exits 0."""
    inputs_dir = os.path.join(testutils._repo_root(), "inputs", "unit_tests")

    uniform = testutils.run_unit_test(
        NAME,
        input_file=os.path.join(inputs_dir, "strang_coupling_uniform_test.athinput"),
    )
    assert uniform, (
        "strang_coupling_test[uniform] failed: the point-implicit matter-radiation "
        "coupling result did not match exactly two dt/2 backward-Euler steps -- the "
        "coupling is not being applied outside the RKL2 super-step (#115)"
    )

    bump = testutils.run_unit_test(
        NAME,
        input_file=os.path.join(inputs_dir, "strang_coupling_bump_test.athinput"),
    )
    assert bump, (
        "strang_coupling_test[bump] failed: the fully-coupled stack (FLD diffusion + "
        "matter-radiation coupling + hydro) did not conserve total energy across the "
        "species split (#115)"
    )
