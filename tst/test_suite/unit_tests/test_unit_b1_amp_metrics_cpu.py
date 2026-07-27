"""Sinars-observable amplitude reductions for the B1 harness (#120).

The b1_window120_204 acceptance run (refined 432x4x192, radiation-ON, local
opacity) exposed a metric mismatch: the mode-4 FOURIER amplitude of the outer
interface saturates near lambda/2pi and phase-rotates once the seeded MRT goes
nonlinear (1.3x "growth", then decay), while the physical spike-bubble
amplitude keeps growing (0.021 -> 0.094 mm, 4.5x, by 84 ns).  Sinars 2011
measures SPIKE-TO-BUBBLE amplitude on radiograph limbs, so the committed
B1 curve must be compared against that observable:

  _spike_bubble_amp(rz)   = (max - min)/2 of an interface trace (B2 convention:
                            spike = max radius, bubble = min, run-in MRT)
  _limb_edge(x, sigma)    = per-z outermost half-max crossing of the synthetic
                            radiograph Sigma(z, x) (the bright-limb edge)

The post-#211 three-resolution acceptance then exposed the MIRROR failure (#212):
the unfiltered (max - min) admits every wavelength the grid supports, MRT growth
goes like sqrt(k), and so a_sb roughly DOUBLED per refinement (0.0162 / 0.0403 /
0.0798 mm at 60 ns for 288 / 432 / 864) -- grid-seeded modes, not the seeded one.
The Fourier projection was too narrow, the raw extremum too wide; the observable
needs a band:

  _band_limit(rz, k_max)  = real-FFT truncation of an axial trace to k <= k_max
                            (NaN rows interpolated first so the transform is
                            well-posed)
  _spike_bubble_amp(rz, k_max=...) = the same excursion measured on that band

Pure-python metric tests: no build, no run.
"""

import numpy as np

import test_suite.cylindrical.test_verify_maglif_b1_sinars_gpu as b1


def test_spike_bubble_equals_fourier_for_pure_cosine():
    """Linear regime: a clean single mode gives p2v == Fourier amp == seed."""
    zn = np.linspace(0.0, 1.0, 256, endpoint=False)
    seed = 0.02
    rz = 3.0 + seed * np.cos(2.0 * np.pi * 4 * zn)
    assert abs(b1._spike_bubble_amp(rz) - seed) < 1.0e-3 * seed
    assert abs(b1._mode_amp(rz, zn, 4) - seed) < 1.0e-2 * seed


def test_spike_bubble_sees_nonlinear_spikes_the_fundamental_misses():
    """Nonlinear regime: narrow spikes riding the mode carry the amplitude the
    fundamental cannot represent -- the exact failure seen in b1_window120_204."""
    zn = np.linspace(0.0, 1.0, 512, endpoint=False)
    rz = 3.0 + 0.01 * np.cos(2.0 * np.pi * 4 * zn)
    for zc in (0.0, 0.25, 0.5, 0.75):                      # spikes at the mode crests
        dz = np.minimum(np.abs(zn - zc), 1.0 - np.abs(zn - zc))
        rz += 0.08 * np.exp(-0.5 * (dz / 0.008) ** 2)
    a_true = 0.5 * (rz.max() - rz.min())
    a_sb = b1._spike_bubble_amp(rz)
    a_f = b1._mode_amp(rz, zn, 4)
    assert abs(a_sb - a_true) < 1.0e-12
    assert a_sb > 2.0 * a_f, (
        f"spike-bubble {a_sb:.4f} should dwarf the saturated fundamental {a_f:.4f}"
    )


def test_limb_edge_tracks_modulated_shell_outer_radius():
    """The radiograph limb edge follows the seeded outer-radius modulation."""
    nr, nz = 400, 64
    r = (np.arange(nr) + 0.5) * 0.01                       # dr = 10 um over [0, 4] mm
    zn = np.linspace(0.0, 1.0, nz, endpoint=False)
    delta = 0.06
    r_out = 3.0 + delta * np.cos(2.0 * np.pi * 4 * zn)
    dens = np.zeros((nz, 1, nr))
    for kz in range(nz):
        dens[kz, 0, (r >= 2.2) & (r <= r_out[kz])] = 1.0
    d = {"x1v": r, "dens": dens}
    x, sigma = b1._synthetic_radiograph(d)
    le = b1._limb_edge(x, sigma)
    assert np.all(np.isfinite(le))
    a_sb = b1._spike_bubble_amp(le)
    assert abs(a_sb - delta) < delta / 3.0, (
        f"limb-edge spike-bubble amp {a_sb:.4f} does not track the seeded {delta}"
    )


def test_spike_bubble_amp_nan_robust():
    """Rows where the interface finder failed (NaN) must not poison the metric."""
    rz = np.array([3.0, 3.1, np.nan, 2.9, np.nan, 3.05])
    a = b1._spike_bubble_amp(rz)
    assert np.isfinite(a)
    assert abs(a - 0.1) < 1.0e-12


# --- #212: band-limiting the observable -------------------------------------
#
# The cut is an INSTRUMENT RESPONSE, not a tuned harmonic count: a real radiograph
# cannot resolve structure below its own spatial resolution, so neither may the
# synthetic one we compare against it.  b1.K_MAX_BAND is that cutoff expressed in
# cycles across the axial domain; PERT_MODE = 4 is the seeded fundamental, so the
# band spans the fundamental plus its resolvable harmonics.


def _spiky_profile(zn, seed=0.01, spike=0.08, width=0.008):
    """Nonlinear single-mode profile: mode-4 carrier + narrow spikes on its crests."""
    rz = 3.0 + seed * np.cos(2.0 * np.pi * 4 * zn)
    for zc in (0.0, 0.25, 0.5, 0.75):
        dz = np.minimum(np.abs(zn - zc), 1.0 - np.abs(zn - zc))
        rz = rz + spike * np.exp(-0.5 * (dz / width) ** 2)
    return rz


def test_band_cutoff_sits_between_the_seeded_harmonics_and_the_grid():
    """The cutoff must admit the seeded mode's nonlinear harmonics and nothing the
    instrument could not have resolved."""
    assert b1.K_MAX_BAND > 4 * b1.PERT_MODE, (
        "band must keep at least the first few harmonics of the seeded mode"
    )
    # the 864 leg carries nx3 = 384 axial cells -> Nyquist 192 cycles/domain
    assert b1.K_MAX_BAND < 192, "band must cut BELOW the finest grid's Nyquist"


def test_band_limit_preserves_a_pure_fundamental():
    """Linear regime must be untouched: band-limiting a clean mode-4 cosine is a
    no-op, so the banded amplitude still recovers the seed exactly."""
    zn = np.linspace(0.0, 1.0, 256, endpoint=False)
    seed = 0.02
    rz = 3.0 + seed * np.cos(2.0 * np.pi * 4 * zn)
    banded = b1._band_limit(rz, b1.K_MAX_BAND)
    assert np.max(np.abs(banded - rz)) < 1.0e-9 * seed
    assert abs(b1._spike_bubble_amp(rz, k_max=b1.K_MAX_BAND) - seed) < 1.0e-3 * seed


def test_banded_amp_still_beats_the_saturated_fundamental():
    """The #208 property must survive: nonlinear spike structure built from
    harmonics of the seeded mode is physical and must still be counted."""
    zn = np.linspace(0.0, 1.0, 512, endpoint=False)
    rz = _spiky_profile(zn)
    a_band = b1._spike_bubble_amp(rz, k_max=b1.K_MAX_BAND)
    a_f = b1._mode_amp(rz, zn, 4)
    assert a_band > 1.5 * a_f, (
        f"banded spike-bubble {a_band:.4f} must still exceed the saturated "
        f"fundamental {a_f:.4f}"
    )


def test_banded_amp_rejects_grid_scale_noise():
    """Grid-seeded short-wavelength MRT must not enter the observable.

    Adding a near-Nyquist ripple inflates the raw extremum but must leave the
    banded amplitude essentially unchanged.
    """
    zn = np.linspace(0.0, 1.0, 512, endpoint=False)
    clean = _spiky_profile(zn)
    noisy = clean + 0.03 * np.cos(2.0 * np.pi * 200 * zn)     # k = 50 k_0, past the cut
    raw_clean = b1._spike_bubble_amp(clean)
    raw_noisy = b1._spike_bubble_amp(noisy)
    assert raw_noisy > 1.4 * raw_clean, (
        "test premise: the ripple must visibly inflate the UNFILTERED metric "
        f"({raw_noisy:.4f} vs {raw_clean:.4f})"
    )
    band_clean = b1._spike_bubble_amp(clean, k_max=b1.K_MAX_BAND)
    band_noisy = b1._spike_bubble_amp(noisy, k_max=b1.K_MAX_BAND)
    assert abs(band_noisy - band_clean) < 0.05 * band_clean, (
        f"banded metric moved {band_clean:.4f} -> {band_noisy:.4f} under pure "
        "grid-scale noise"
    )


def test_banded_amp_is_resolution_insensitive():
    """The #212 acceptance property in miniature.

    The same analytic interface, sampled at nz and 2 nz with grid-scale structure
    riding on it whose amplitude grows with the wavenumber the grid supports, must
    give the same banded amplitude -- while the unfiltered metric diverges, as
    432 -> 864 did (a_sb 0.0403 -> 0.0798 mm at 60 ns).
    """
    def sample(nz):
        zn = np.linspace(0.0, 1.0, nz, endpoint=False)
        k_grid = nz // 4                        # shortest mode the grid resolves well
        amp = 0.02 * (k_grid / 128.0)           # faster-growing the finer the grid
        return _spiky_profile(zn) + amp * np.cos(2.0 * np.pi * k_grid * zn)

    lo, hi = sample(512), sample(1024)
    raw_lo, raw_hi = b1._spike_bubble_amp(lo), b1._spike_bubble_amp(hi)
    assert raw_hi > 1.15 * raw_lo, (
        f"test premise: unfiltered metric must diverge ({raw_lo:.4f} -> {raw_hi:.4f})"
    )
    band_lo = b1._spike_bubble_amp(lo, k_max=b1.K_MAX_BAND)
    band_hi = b1._spike_bubble_amp(hi, k_max=b1.K_MAX_BAND)
    assert abs(band_hi - band_lo) < 0.05 * band_lo, (
        f"banded metric is resolution-dependent: {band_lo:.4f} -> {band_hi:.4f}"
    )


def test_band_limit_is_nan_robust():
    """NaN rows (failed interface finds) must be filled, not propagated."""
    zn = np.linspace(0.0, 1.0, 256, endpoint=False)
    rz = 3.0 + 0.02 * np.cos(2.0 * np.pi * 4 * zn)
    holed = rz.copy()
    holed[[7, 8, 100, 201]] = np.nan
    banded = b1._band_limit(holed, b1.K_MAX_BAND)
    assert np.all(np.isfinite(banded))
    a = b1._spike_bubble_amp(holed, k_max=b1.K_MAX_BAND)
    assert abs(a - 0.02) < 5.0e-2 * 0.02
