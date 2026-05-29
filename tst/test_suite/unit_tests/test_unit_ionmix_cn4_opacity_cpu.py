"""
Real-material IONMIX cn4 multigroup-opacity ingestion + verification (issue [C1]/#118).

Builds src/pgen/unit_tests/ionmix_cn4_opacity_test.cpp, run once per developer-supplied
FLASH IONMIX table -- aluminum, beryllium, deuterium (committed under inputs/ionmix/) --
exercising the real-material opacity reader ReadIonmixCn4Opacity
(src/opacity/ionmix_opacity_reader.hpp) + the common multigroup representation
(src/opacity/multigroup_opacity.hpp), per ADR-0007.

For each material the independent Python decoder (ionmix_cn4_decode) re-reads the
same cn4 file and supplies, as `problem/...` overrides, the dimensions, the first/last
photon-energy group boundaries, and the Planck-absorption / Planck-emission / Rosseland
opacities at two interior probe (group, T, rho) nodes; the pgen asserts the reader
reproduces them (the real cn4 opacity order is Rosseland, Planck-absorption,
Planck-emission -- reverse of the synthetic fixture, mapped correctly), that group
bounds are ascending, every stored opacity is positive, and out-of-range queries clamp to
table edges.  The binary exits 0 iff every check passes.

Oracle: Layer 1 -- independent re-derivation (ionmix_cn4_decode, the opacplot2 reference
format), cross-checked against the C++ reader and anchored by physical invariants
(ascending group bounds, positive opacities, edge clamping).  References: opacplot2 cn4
format; tables Al/Be/D supplied by the developer.
"""

# Modules
import os

import pytest

import test_suite.testutils as testutils
import test_suite.unit_tests.ionmix_cn4_decode as cn4


MATERIALS = [
    ("aluminum", "al-imx-004.cn4"),
    ("beryllium", "Be-006-imx.cn4"),
    ("deuterium", "DD-006-imx.cn4"),
]


def _table_path(fname):
    return os.path.join(testutils._repo_root(), "inputs", "ionmix", fname)


def _probe_args(dec):
    """Two interior probe (group, T, rho) nodes and expected opacities + group bounds."""
    ntemp, ndens, ngroups = dec["ntemp"], dec["ndens"], dec["ngroups"]
    ig0, it0, id0 = ngroups // 2, ntemp // 2, ndens // 2
    ig1, it1, id1 = ngroups // 3, (3 * ntemp) // 4, ndens // 3
    args = [
        f"problem/exp_ntemp={ntemp}",
        f"problem/exp_ndens={ndens}",
        f"problem/exp_ngroups={ngroups}",
        f"problem/exp_gb_lo={float(dec['opac_bounds'][0])!r}",
        f"problem/exp_gb_hi={float(dec['opac_bounds'][ngroups])!r}",
        f"problem/pg0={ig0}", f"problem/pit0={it0}", f"problem/pid0={id0}",
        f"problem/pg1={ig1}", f"problem/pit1={it1}", f"problem/pid1={id1}",
    ]
    for tag, ig, it, idx in (("0", ig0, it0, id0), ("1", ig1, it1, id1)):
        args += [
            f"problem/exp_pa{tag}={float(dec['planck_abs'][ig][idx][it])!r}",
            f"problem/exp_pe{tag}={float(dec['planck_emi'][ig][idx][it])!r}",
            f"problem/exp_ro{tag}={float(dec['rosseland'][ig][idx][it])!r}",
        ]
    return args


@pytest.mark.parametrize("label,fname", MATERIALS)
def test_run(label, fname):
    """Ingest the real Al/Be/D IONMIX cn4 opacity table; reader reproduces it (exit 0)."""
    path = _table_path(fname)
    dec = cn4.decode(path)
    args = [f"problem/opacity_file={path}"] + _probe_args(dec)
    passed = testutils.run_unit_test("ionmix_cn4_opacity_test", args=args)
    assert passed, f"ionmix_cn4_opacity_test failed for {label} ({fname})"
