# B2 digitization-QC evidence (McBride 2012 fit laws)

Provenance trail for the **Stage-2 soft quantitative anchors** committed in
[`../../b2_multimode_mrt_mcbride_2012.json`](../../b2_multimode_mrt_mcbride_2012.json)
(the `fit_laws` block and the `growth_fraction` / `multimode_amplitude_law` /
`multimode_wavelength_law` datums). Collected during the Stage-2 transcription QC
(2026-06-02, previously untracked as repo-root `b2_qc/`); committed here per issue #229
AC#2 so the provenance trail is reproducible from the repo (ADR-0008).

## Primary source

R. D. McBride et al., *Penetrating Radiography of Imploding and Stagnating Beryllium
Liners on the Z Accelerator*, Phys. Rev. Lett. **109**, 135004 (2012),
DOI [10.1103/PhysRevLett.109.135004](https://doi.org/10.1103/PhysRevLett.109.135004).

The full paper PDF is **deliberately not committed** (APS-copyrighted, paywalled
redistribution). The QC was performed against the copy identified by

```
sha256(mcbride2012.pdf) = cda52a3290055a06617023f45ae928a312ae8b0a54577386e3677f093f0aa3d2
```

obtainable via the DOI above. Only figure excerpts needed to verify the transcriptions
are committed (fair-use QC evidence, same convention as `../figures/`).

## What was QC'd, and how

The committed Stage-2 anchors are **verbatim transcriptions of printed annotations and
text**, not pixel-traced points (ADR-0008: no fabrication). Each file below is a render
of the relevant figure region from the source PDF, kept as the evidence that the
transcription matches the print:

| File | Evidence for |
|------|--------------|
| `fig7a_fitlaw.png` | Fig. 7(a) crop: the printed amplitude fit line with the experiment / GORGON 3D / LASNEX 2D points — verifies the transcribed law `amplitude_um = 450*x - 90` and its x-intercept at x = 0.2. |
| `fig7b_fitlaw.png` | Fig. 7(b) crop: the printed annotation `750*[1-R(t)/R(0)]` — verifies the transcribed law `wavelength_um = 750*x`. |
| `mcbride2012_fig7a_amplitude.png` | Fig. 7(a)+(b) column render (both printed laws in one frame, with legends). |
| `mcbride2012_fig6_7_column.png` | Full Fig. 6/7 column render — context for the inner / bubble / spike radius definitions feeding the `fig6_radii_vs_time_supporting` datum (which stays `machine_extracted_needs_human_qc`). |
| `mcbride2012_fig6_radii_vs_time.png` | Fig. 6 render: inner / bubble / spike radii vs time (supporting datum only, NOT a binding gate). |

## The growth-fraction band and its validity window (#229)

The `growth_fraction` band **[0.05, 0.15]** is transcribed verbatim from the paper text
accompanying Fig. 7(a):

> "Expressed as a fraction of the distance moved, this growth is therefore nearly
> constant, and is in the range of 0.05–0.15, which is consistent with results from
> classical hydrodynamic Rayleigh-Taylor experiments in the nonlinear regime."

The Fig. 7(a) crops above are also the evidence for the committed
`fit_laws.x_valid_range = [0.4, 0.95]`: the measured points span x ~ 0.4–0.95 and the
printed amplitude line is negative below x = 0.2, so the band is a published claim only
over that nonlinear-regime domain. Comparisons at shallower convergence are recorded
PENDING, not FAIL (#229; enforced by `mcbride_b2_anchor.in_validity_window` and gated in
`cylindrical/test_verify_maglif_b2_ellison_cpu.py`).

Spike/bubble definitions (paper text, for reduction faithfulness): the radiography has
**15 um spatial resolution**; the *bubble* radius is the steep density gradient of the
dominant MRT bubbles in the Abel-inverted volume-density images, and the *spike* radius
is the **center-of-mass of the trailing MRT spike structure** — not raw per-zone
extrema. The synthetic reduction band-limits the interface trace to the instrument band
(k <= 53 across the 1.6 mm axial extent) before taking extrema (#229, mirroring the
#212 B1 fix).
