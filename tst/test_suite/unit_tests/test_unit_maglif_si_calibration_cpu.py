"""
Faithful dimensional units + drive calibration unit test (issue [P5]/#174, ADR-0010).

Builds src/pgen/unit_tests/maglif_si_calibration_test.cpp (-D
PROBLEM=unit_tests/maglif_si_calibration_test) into the isolated tst/build_unit directory
and runs it against the committed z2173 current trace and the real aluminum IONMIX table,
asserting it exited 0 (all checks passed).

Two closed-form Layer-1 oracles (ADR-0008), the last-mile machinery the faithful B1
replication needs:

  (AC#2) Magnetic-pressure anchor (CONTEXT.md "Magnetic-pressure anchor").  The SI drive
  calibration (circuit::CalibrateSiDrive) converts the z2173 load trace (time [ns],
  current [MA]) into code units; applying the calibrated peak current to the code drive
  ``B_phi = mu0*I_code/(2*pi*r_code)`` (mu0=1) and reconverting ``B^2/2`` back to physical
  Pa must reproduce the independent closed-form ``mu0_SI*I^2/(8*pi^2*r^2)`` (~5 Mbar at
  20 MA / 3.47 mm).  The test ALSO asserts (AC#1 red-first) that consuming the raw trace
  MA directly as a code current -- the pre-#174 behaviour -- gives a grossly wrong
  magnetic pressure, so the calibration is genuinely necessary.

  EOS-table unit boundary (ADR-0010).  EosTable3T::ScaleToCodeUnits divides specific
  energies/heat capacities by velocity_cgs^2, pressures by density_cgs*velocity_cgs^2, and
  relabels the density axis /density_cgs, leaving the eV temperature axis and Zbar
  untouched.  Loading the Al table twice (raw vs scaled) and comparing node-for-node
  reproduces exactly those factors.

The trace and table paths are passed as absolute-path command-line overrides because the
unit-test binary runs from tst/build_unit/<name>/src.
"""

# Modules
import os

import test_suite.testutils as testutils


def test_run():
    """Build + run the SI units/drive calibration unit test; pass iff it exits 0."""
    repo = testutils._repo_root()
    table = os.path.join(repo, "inputs", "ionmix", "al-imx-004.cn4")
    trace = os.path.join(repo, "tst", "inputs", "z2173_current.dat")
    passed = testutils.run_unit_test(
        "maglif_si_calibration_test",
        args=[f"mhd/eos_table={table}", f"problem/current_file={trace}"],
    )
    assert passed, "maglif_si_calibration_test reported a failing check (nonzero exit)"
