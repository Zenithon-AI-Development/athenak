"""
Independent decoder for the real FLASH IONMIX `.cn4` table format (issue [C1]/#118).

This is the *oracle* for the real-material IONMIX ingestion tests
(test_unit_ionmix_cn4_eos_cpu.py / test_unit_ionmix_cn4_opacity_cpu.py): it re-implements
the packed fixed-width cn4 decode (the FLASH-center `opacplot2` reference format)
independently of the C++ reader (src/utils/ionmix_cn4_format.hpp + the IONMIX EOS/opacity
readers), so agreement between the two is a genuine cross-implementation check rather than
a self-snapshot (ADR-0008, Layer-1 independent re-derivation).  It is a plain helper
(no `_cpu`/`_mpicpu`/`_gpu` suffix), so the suite runner does not collect it as a test.

Format (mirrors `src/utils/ionmix_cn4_format.hpp`):
  Header: 4 lines -- "ntemp ndens", "atomic #s of gases: ...", "relative fractions: ...",
  "ngroups".  Data: 12-char fixed-width fields, concatenated across the data lines and
  read in record order: temps[ntemp], numdens[ndens], then the 12 EOS blocks
  (zbar, dzdt, pion, pele, dpidt, dpedt, eion, eele, cvion, cvele, deidn, deedn -- each
  ndens*ntemp, temperature varying fastest), then opac_bounds[ngroups+1], then the three
  opacity arrays Rosseland, Planck-absorption, Planck-emission (each ngroups*ndens*ntemp,
  group-major then density then temperature).  2-D arrays are returned shaped [id][it];
  3-D opacity arrays [ig][id][it].  Units are left exactly as stored (the C++ reader is a
  faithful loader too).
"""

# Modules
import numpy as np


def decode(fname):
    """Decode a FLASH IONMIX .cn4 file into a dict of header scalars + data arrays."""
    with open(fname) as fobj:
        lines = fobj.read().splitlines()
    if len(lines) < 4:
        raise ValueError(f"{fname}: too short for the 4-line IONMIX cn4 header")

    dims = lines[0].split()
    ntemp, ndens = int(dims[0]), int(dims[1])
    zvals = [int(x) for x in lines[1].split(":", 1)[1].split()]
    fracs = [float(x) for x in lines[2].split(":", 1)[1].split()]
    ngroups = int(lines[3].split()[0])

    packed = "".join(line.rstrip() for line in lines[4:])
    if len(packed) % 12 != 0:
        raise ValueError(f"{fname}: packed length {len(packed)} not a multiple of 12")
    nfields = len(packed) // 12
    vals = np.array([float(packed[i * 12:(i + 1) * 12]) for i in range(nfields)])

    cursor = [0]

    def block(count):
        start = cursor[0]
        cursor[0] += count
        return vals[start:start + count]

    def block2d():
        # flat order: density outer, temperature inner -> (ndens, ntemp) = [id][it]
        return block(ndens * ntemp).reshape(ndens, ntemp)

    def block3d():
        # flat order: group, density, temperature (T fastest) -> [ig][id][it]
        return block(ngroups * ndens * ntemp).reshape(ngroups, ndens, ntemp)

    temps = block(ntemp)
    numdens = block(ndens)
    zbar = block2d()
    block2d()                    # dzdt   (unused)
    pion = block2d()
    pele = block2d()
    block2d()                    # dpidt  (unused)
    block2d()                    # dpedt  (unused)
    eion = block2d()
    eele = block2d()
    cvion = block2d()
    cvele = block2d()
    block2d()                    # deidn  (unused)
    block2d()                    # deedn  (unused)
    opac_bounds = block(ngroups + 1)
    rosseland = block3d()
    planck_abs = block3d()
    planck_emi = block3d()
    if cursor[0] != nfields:
        raise ValueError(f"{fname}: consumed {cursor[0]} of {nfields} fields "
                         "(record-order mismatch)")

    return {
        "ntemp": ntemp, "ndens": ndens, "ngroups": ngroups,
        "zvals": zvals, "fracs": fracs,
        "zsum": float(sum(z * f for z, f in zip(zvals, fracs))),
        "temps": temps, "numdens": numdens,
        "zbar": zbar, "pion": pion, "pele": pele,
        "eion": eion, "eele": eele, "cvion": cvion, "cvele": cvele,
        "opac_bounds": opac_bounds,
        "rosseland": rosseland, "planck_abs": planck_abs, "planck_emi": planck_emi,
    }
