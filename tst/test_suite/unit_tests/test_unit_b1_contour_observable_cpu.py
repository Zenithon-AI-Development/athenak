"""Paper density-contour spike-bubble observable for the B1 harness (#222).

Ellison arXiv:2504.10760 SS3.1 (Fig. 3) defines the B1 observable as the
(max - min) radius of the ABSOLUTE rho = 0.1 g/cc density contour at the
vacuum/conductor interface -- a criterion the paper states is "not very
sensitive to the exact choice of density threshold".  Our existing reduction
(a_sb, #208/#212) is the synthetic-radiograph half-max limb: a RELATIVE
criterion that moves as the shell decompresses and can switch features when a
brighter structure appears in the projection (the suspected 2.1x single-step
jump at 51->54 ns on the 432 leg).  The paper's contour is the independent
second observable:

  _contour_edge(d, rho_gcc)  = per-z OUTERMOST radial crossing of the absolute
                               density level rho_gcc [g/cc] (code units via
                               DENSITY_CGS)
  _contour_amp(rz)           = FULL max - min excursion over z [mm] (paper
                               convention; note a_sb is the HALF excursion)

Pure-python metric tests on synthetic density fields: no build, no run.
"""

import numpy as np

import test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu as b1

NR = 400
RMAX = 4.0                       # [mm]; dr = 10 um, matches the amp-metrics tests
DR = RMAX / NR
DELTA = 0.06                     # seeded outer-radius modulation [mm]
NZ = 64


def _r_grid():
    return (np.arange(NR) + 0.5) * DR


def _r_out(nz=NZ, delta=DELTA):
    zn = np.linspace(0.0, 1.0, nz, endpoint=False)
    return zn, 3.0 + delta * np.cos(2.0 * np.pi * 4 * zn)


def _shell_dict(r_out, peak_code, r_in=2.2, column=None, time=0.0):
    """Synthetic snapshot dict: top-hat shell [r_in, r_out(z)] at peak_code density.

    column=(r_col, d_col) adds an unmodulated on-axis dense column (r < r_col).
    """
    r = _r_grid()
    nz = r_out.size
    dens = np.zeros((nz, 1, NR))
    for kz in range(nz):
        dens[kz, 0, (r >= r_in) & (r <= r_out[kz])] = peak_code
        if column is not None:
            dens[kz, 0, r < column[0]] = column[1]
    return {"x1v": r, "dens": dens, "Time": time}


def test_contour_amp_is_full_max_minus_min():
    """The paper convention: a_contour = max - min (NOT halved like a_sb)."""
    rz = np.array([3.0, 3.1, 2.9, np.nan])
    a = b1._contour_amp(rz)
    assert abs(a - 0.2) < 1.0e-12
    assert abs(a - 2.0 * b1._spike_bubble_amp(rz)) < 1.0e-12


def test_contour_edge_tracks_seeded_outer_radius():
    """The 0.1 g/cc contour follows the seeded modulation to sub-cell accuracy."""
    _, r_out = _r_out()
    d = _shell_dict(r_out, peak_code=1.0)          # solid liner: 2.7 g/cc
    edge = b1._contour_edge(d)
    assert np.all(np.isfinite(edge))
    assert np.max(np.abs(edge - r_out)) < DR, (
        f"contour edge missed the seeded surface by {np.max(np.abs(edge - r_out)):.4f}"
    )
    a = b1._contour_amp(edge)
    assert abs(a - 2.0 * DELTA) < DR, (
        f"a_contour {a:.4f} != seeded excursion {2.0 * DELTA:.4f}"
    )


def test_contour_amp_threshold_insensitive_mirroring_paper_claim():
    """Paper claim (SS3.1): the observable is 'not very sensitive to the exact
    choice of density threshold' -- <10% over 0.05-0.2 g/cc, even with an edge
    whose ramp width varies along z (a decompressing, non-uniform interface)."""
    zn, r_out = _r_out()
    width = 0.02 + 0.01 * np.cos(2.0 * np.pi * 8 * zn)   # z-varying edge ramp [mm]
    r = _r_grid()
    dens = np.zeros((NZ, 1, NR))
    for kz in range(NZ):
        ramp = (r_out[kz] + width[kz] - r) / width[kz]   # 1 at r_out, 0 at r_out+w
        p = np.clip(ramp, 0.0, 1.0)
        p[r < 2.2] = 0.0
        dens[kz, 0, :] = p
    d = {"x1v": r, "dens": dens, "Time": 0.0}
    amps, spread = b1._contour_threshold_spread(d)
    for thr in b1.CONTOUR_THRESHOLDS_GCC:
        assert abs(amps[thr] - 2.0 * DELTA) < 2.0 * DR, (
            f"a_contour({thr} g/cc) = {amps[thr]:.4f} lost the seeded excursion"
        )
    assert spread < 0.10, (
        f"a_contour varies {spread:.1%} over 0.05-0.2 g/cc (paper claims <10%)"
    )


def test_contour_survives_decompressing_thin_shell():
    """The absolute 0.1 g/cc criterion keeps tracking the SAME material edge as
    the shell decompresses -- exactly where a fixed half-max criterion loses it."""
    _, r_out = _r_out()
    dense = _shell_dict(r_out, peak_code=1.0)            # 2.7 g/cc
    thin = _shell_dict(r_out, peak_code=0.06)            # 0.162 g/cc < HALF, > 0.1
    # premise: the half-max interface finder loses the decompressed shell entirely
    assert not np.any(np.isfinite(b1._interfaces(thin)[1])), (
        "test premise: HALF-level interfaces must fail on the decompressed shell"
    )
    a_dense = b1._contour_amp(b1._contour_edge(dense))
    a_thin = b1._contour_amp(b1._contour_edge(thin))
    assert np.isfinite(a_thin)
    assert abs(a_thin - a_dense) < 1.0e-9, (
        f"a_contour moved under pure decompression: {a_dense:.4f} -> {a_thin:.4f}"
    )


def test_contour_ignores_bright_interior_feature_where_limb_switches():
    """A bright on-axis column (stagnating fuel) re-keys the radiograph's relative
    half-max and the limb edge jumps inboard to the column -- feature switching.
    The absolute contour stays on the shell's outermost surface."""
    _, r_out = _r_out()
    d = _shell_dict(r_out, peak_code=0.15, column=(0.5, 2.0))
    x, sigma = b1._synthetic_radiograph(d)
    limb_amp = b1._spike_bubble_amp(b1._limb_edge(x, sigma))
    assert limb_amp < 0.2 * DELTA, (
        "test premise: the relative half-max limb must switch to the bright "
        f"column (limb amp {limb_amp:.4f} vs seeded {DELTA})"
    )
    a = b1._contour_amp(b1._contour_edge(d))
    assert abs(a - 2.0 * DELTA) < DR, (
        f"a_contour {a:.4f} lost the shell to the interior feature"
    )


def test_contour_edge_nan_where_slice_never_reaches_threshold():
    """Slices with no material above 0.1 g/cc give NaN; the amp skips them."""
    _, r_out = _r_out()
    d = _shell_dict(r_out, peak_code=0.2)
    d["dens"][NZ // 2:, :, :] = 0.02                     # 0.054 g/cc < 0.1 everywhere
    edge = b1._contour_edge(d)
    assert np.all(np.isfinite(edge[:NZ // 2]))
    assert not np.any(np.isfinite(edge[NZ // 2:]))
    a = b1._contour_amp(edge)
    assert np.isfinite(a)
    ref = b1._contour_amp(edge[:NZ // 2])
    assert abs(a - ref) < 1.0e-12
    assert np.isnan(b1._contour_amp(np.full(4, np.nan)))


def test_postprocess_reduces_snapshot_dicts():
    """The CPU post-processing core reduces saved snapshots (as dicts) to
    a_contour(t) per threshold next to a_sb(t)/a_sb_raw(t), sorted by time."""
    import test_suite.cylindrical.b1_contour_postprocess as pp
    _, r_out = _r_out()
    # decompressed leg stays above the full sweep band (0.09 code = 0.243 g/cc)
    later = _shell_dict(r_out, peak_code=0.09, time=57.0)
    early = _shell_dict(r_out, peak_code=1.0, time=54.0)    # dense
    out = pp.reduce_snapshot_dicts([later, early])          # unsorted on purpose
    assert list(out["times"]) == [54.0, 57.0]
    for thr in pp.THRESHOLDS_GCC:
        a = np.asarray(out["a_contour"][thr], dtype=float)
        assert a.shape == (2,)
        assert np.all(np.abs(a - 2.0 * DELTA) < 2.0 * DR)
    assert np.asarray(out["a_sb"]).shape == (2,)
    assert np.asarray(out["a_sb_raw"]).shape == (2,)
