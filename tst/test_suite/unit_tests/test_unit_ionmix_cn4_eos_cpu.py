"""
Real-material IONMIX cn4 EOS ingestion + lookup-verification unit test (issue [C1]/#118).

Builds src/pgen/unit_tests/ionmix_cn4_eos_test.cpp and runs it once per developer-supplied
FLASH IONMIX table -- aluminum, beryllium and deuterium (under inputs/ionmix/) --
exercising the real-material EOS reader
ReadIonmixCn4Eos (src/eos/ionmix_eos_reader.hpp) + the common 3T representation
(src/eos/eos_table_3t.hpp), per ADR-0007.

For each material the independent Python decoder (ionmix_cn4_decode) re-reads the same
packed cn4 file and supplies, as `problem/...` command-line overrides, the reference
dimensions, the material charge state Z (= sum z*frac), and the per-species energies /
pressures / ionization / heat-capacities at two interior probe nodes; the pgen asserts
its reader reproduces them, that Zbar stays in [0,Z], that electron specific energy is
monotone in T (so e->T inverts), that the inversion round-trips, and that out-of-range
queries clamp to the table edges.  The binary exits 0 iff every check passes.

Oracle: Layer 1 -- independent re-derivation.  The reference values come from a second,
independent decode of the same cn4 bytes (ionmix_cn4_decode, the opacplot2 reference
format), cross-checked against the C++ reader; physical invariants (Zbar in [0,Z], e(T)
monotone, round-trip, edge clamping) anchor it.  References: FLASH IONMIX / opacplot2 cn4
format; tables Al/Be/D supplied by the developer.
"""

# Modules
import os

import pytest

import test_suite.testutils as testutils
import test_suite.unit_tests.ionmix_cn4_decode as cn4


# (material label, committed cn4 filename under inputs/ionmix/).
MATERIALS = [
    ("aluminum", "al-imx-004.cn4"),
    ("beryllium", "Be-006-imx.cn4"),
    ("deuterium", "DD-006-imx.cn4"),
]


def _table_path(fname):
    return os.path.join(testutils._repo_root(), "inputs", "ionmix", fname)


def _probe_args(dec):
    """Two interior probe nodes (well away from the cold/low-density floored corner) and
    the expected per-species table values there, as problem/... override strings."""
    ntemp, ndens = dec["ntemp"], dec["ndens"]
    it0, id0 = ntemp // 2, ndens // 2
    it1, id1 = (3 * ntemp) // 4, ndens // 3
    args = [
        f"problem/exp_ntemp={ntemp}",
        f"problem/exp_ndens={ndens}",
        f"problem/exp_zsum={dec['zsum']!r}",
        f"problem/pit0={it0}", f"problem/pid0={id0}",
        f"problem/pit1={it1}", f"problem/pid1={id1}",
    ]
    for tag, it, idx in (("0", it0, id0), ("1", it1, id1)):
        args += [
            f"problem/exp_eele{tag}={float(dec['eele'][idx][it])!r}",
            f"problem/exp_eion{tag}={float(dec['eion'][idx][it])!r}",
            f"problem/exp_pele{tag}={float(dec['pele'][idx][it])!r}",
            f"problem/exp_pion{tag}={float(dec['pion'][idx][it])!r}",
            f"problem/exp_zbar{tag}={float(dec['zbar'][idx][it])!r}",
            f"problem/exp_cvele{tag}={float(dec['cvele'][idx][it])!r}",
            f"problem/exp_cvion{tag}={float(dec['cvion'][idx][it])!r}",
        ]
    return args


@pytest.mark.parametrize("label,fname", MATERIALS)
def test_run(label, fname):
    """Ingest the real Al/Be/D IONMIX cn4 EOS table; reader must reproduce it (exit 0)."""
    path = _table_path(fname)
    dec = cn4.decode(path)
    args = [f"problem/eos_file={path}"] + _probe_args(dec)
    passed = testutils.run_unit_test("ionmix_cn4_eos_test", args=args)
    assert passed, f"ionmix_cn4_eos_test failed for {label} ({fname})"
