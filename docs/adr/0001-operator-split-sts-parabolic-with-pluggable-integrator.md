# Radiation/diffusion advanced by operator-split super-time-stepping, behind a pluggable parabolic-integrator interface

**Context.** The MagLIF/Z-pinch target needs multi-group flux-limited radiation diffusion (FLD),
flux-limited electron/ion thermal conduction, and Spitzer resistivity, at 5 µm / 200 ns on a single
A100 (scaling to a few). These are stiff parabolic operators; naive explicit integration collapses the
timestep (`dt ~ Δx²/D`). AthenaK has **no** implicit/elliptic solver infrastructure (no multigrid, CG,
HYPRE/PETSc/AMGX); building one is a large, GPU-scaling-sensitive effort. FLASH/CASTRO/RAMSES solve
multigroup FLD fully implicitly; RKL2 super-time-stepping (Meyer, Balsara & Aslam 2014) is a proven
explicit alternative for parabolic terms — including, in that paper, grey flux-limited radiation
diffusion — and scales far better in parallel (Vaidya 2017; Caplan 2017 found RKL competitive with or
better than implicit PCG, with some accuracy limitations).

**Decision.** Advance all stiff diffusion via **operator splitting**, in AthenaK's existing
`before/after_timeintegrator` tasklist slots (once per hydro step, wrapping the RK `stagen` loop):
- **Spatial diffusion** (FLD groups, e/i conduction, resistive B): a **`ParabolicIntegrator`** strategy
  selected at runtime (`<time> parabolic_integrator = sts | implicit`). `sts` = RKL2; `implicit` is a
  deferred matrix-free Krylov/multigrid (or AMGX) backend. Both call the *same* `ParabolicOperator`
  action (flux-divergence), shaped like the existing `Conduction::AddHeatFlux` /
  `Resistivity::OhmicEnergyFlux`. No physics kernel is rewritten when the backend is swapped.
- **Local matter–radiation & electron–ion exchange** (point-stiff, not spatial): a per-cell
  **point-implicit** solve (arrowhead system, groups couple only through electron temperature →
  O(N_group) Schur elimination + Newton), reusing the `RadFluidCoupling` / `ion-neutral` IMEX pattern.

We ship **STS first** (it also serves conduction/resistivity, which need it regardless) and do **not**
build the implicit backend until a real problem demonstrates STS cannot scale.

**Why (the hedge).** Fully explicit STS maximizes GPU scaling and avoids a large solver build; the
operator/integrator split means choosing STS is *not* a strong bet — the implicit backend can drop in
behind the same interface later. This matches the stated preference: explicit ideally, implicit only if
accuracy/scaling later demands it.

**Known consequence (thin-region c-CFL).** In optically-thin cells the flux limiter makes FLD
free-stream at `c` — a hyperbolic term STS cannot accelerate (`dt ≲ Δx/c`). Mitigation: cap the
radiation transport speed (local reduced-c) in flux-limiter-saturated cells, accepting reduced
thin-region radiation fidelity (acceptable for ICF, where energy lives in the dense liner/fuel). If this
proves unacceptable on real problems, that is the trigger to enable the implicit backend.

**Considered and rejected (for now).** (a) Implicit FLD up front — most robust, but large build and not
needed to start. (b) Multigroup M1 + reduced-speed-of-light — reuses in-tree M1 but distorts
radiation–matter equilibration and costs 4× radiation memory.

**Update — AMR reinforces this.** AthenaK's native block-AMR is in scope (5 µm is the finest level, not
uniform). Explicit STS is AMR-*local* (same prolongation/restriction as the hyperbolic update), whereas
a fully-implicit FLD on AMR would require a multilevel linear solver — a further argument for STS-first
and against premature implicit.
