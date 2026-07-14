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
