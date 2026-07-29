"""
Auto-collected wrapper for the FLD azimuthal-metric pgen-based unit test (#215).

Builds src/pgen/unit_tests/fld_cyl_phi_metric_test.cpp
(-D PROBLEM=unit_tests/fld_cyl_phi_metric_test) into the isolated tst/build_unit
directory, runs it on a CYLINDRICAL (r, phi) mesh, and asserts it exited 0.

On a cylindrical mesh ``dx2`` is an ANGLE, so the azimuthal gradient of the radiation
energy is ``dE/(r dphi)`` and the physical face width is the arc length
``CenterWidth2(cylindrical, dx2, x1v) = x1v*dx2``.  The grey operator has taken that since
#116; the multigroup operator divided by the raw ``dx2``, making its phi diffusion off by
``r^2`` (and dimensionally inconsistent with its own x1 term).  On a Cartesian mesh
``CenterWidth2`` returns ``dx2`` and both are bit-identical, which is why the existing
Cartesian Marshak tests never covered this -- hence a dedicated cylindrical mesh here.

Both operators are driven with the SAME pure-azimuthal cosine field (uniform in r, so the
radial flux vanishes) in the optically-THICK limit, where ``lambda -> 1/3`` and the #194
Lax-Friedrichs streaming term is gated off -- so the batteries isolate the metric alone.

Oracle: Layer 1 -- analytic.  The discrete azimuthal diffusion eigenvalue
``M(E) = -(2 D/(r dphi)^2)(1 - cos(m dphi)) a cos(m phi)`` with ``D = c/(3 chi)``, checked
for the grey operator (guard) and the multigroup operator (the #215 fix).
"""

# Modules
import test_suite.testutils as testutils


def test_run():
    """Build + run the FLD azimuthal-metric unit test; pass iff it exits 0."""
    passed = testutils.run_unit_test("fld_cyl_phi_metric_test")
    assert passed, "fld_cyl_phi_metric_test reported a failing check (nonzero exit)"
