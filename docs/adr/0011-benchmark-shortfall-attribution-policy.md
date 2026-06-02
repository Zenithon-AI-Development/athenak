# Attribution policy: a benchmark shortfall is blamed on a missing physics model only with evidence, and only from the reference's model set

**Context.** The autonomous backlog loop ([[ralph-loop-issue-selection]]) repeatedly declared the B1/B4
benchmarks blocked because *"AthenaK has no material-strength model (0 grep hits for deviatoric/yield)"*
and demanded a human file a strength issue ([[b1b4-faithful-run-still-ideal-eos]]). This is
attribution-by-absence: "the code lacks model X" is **not** evidence that the *experimental discrepancy*
is caused by the lack of X. It was also never checked against the golden reference. The Ellison FLASH
validation paper (arXiv:2504.10760, [[maglif-verification-anchor]]) **includes no material-strength
model** — the liner is a strengthless hydrodynamic fluid — and still matches single-mode MRT (B1) on
implosion timing *and* mode-amplitude growth. So strength was never the gap.

**Decision.** When a faithful benchmark run misses its experimental anchor, the residual is attributed
to a missing physics model **only when both** hold:
1. **Bounded by the reference's actual model set.** Candidate physics is restricted to what the
   reference simulation (Ellison/FLASH) actually ran for that benchmark: tabulated 3T EOS, magnetized
   electron/ion conduction, resistivity, gray (B1-4) / multigroup (B5-6) radiation diffusion, Nernst &
   alpha (B5-6). **Material strength is explicitly excluded** — FLASH matched B1-4 without it. A model
   absent from the reference is not a permissible explanation for a reference-defined benchmark.
2. **Isolated by independent anchors, pragmatically.** Upstream causes are ruled out first — the
   closed-form magnetic-pressure anchor (calibration), implosion *timing* on the observable's absolute
   time axis (bulk dynamics), 1D-stays-1D + a coarse two-grid convergence (numerics). Only a residual
   surviving these, consistent with the candidate model's operative regime/sign/magnitude, may be filed
   — and the filed issue must carry that evidence. The gate forces evidence; it does **not** require the
   gap be closed before moving on (file the follow-up with receipts, close the current issue, revisit).

**Consequences.** The loop cannot terminate a benchmark as "blocked, needs model X" by code-grep alone;
it must produce anchors and cite the reference. Keeps the loop moving on honest engineering residuals
(EOS-IC wiring, drive calibration) instead of stalling on invented physics gaps. Complements the
provenance policy (ADR-0008) and the faithful-units convention (ADR-0010).
