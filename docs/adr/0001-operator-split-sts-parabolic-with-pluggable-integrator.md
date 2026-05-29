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

---

## Addendum (issue [A2]/#109) — per-operator STS-vs-flux-fuse routing decision

**Why this addendum.** ADR-0001 above advances *stiff* spatial diffusion by operator-split STS, but not
every parabolic operator is stiff enough to earn the per-substage multi-block ghost exchange that STS
costs (the `SyncParabolicGhosts` helper, #108). The alternative is the **flux-fused** path: the diffusive
flux rides the hyperbolic flux update and the global timestep simply becomes `min(dt_hyp, dt_exp)` (see
`Hydro::NewTimeStep`, `src/hydro/hydro_newdt.cpp`, the `cond_operator_split == false` branch). Flux-fusion
adds **no** per-substage exchange and inherits AthenaK's hyperbolic scaling for free, but if `dt_exp <
dt_hyp` it slows the *whole* simulation by the stiffness ratio `r = dt_hyp/dt_exp`. STS instead covers the
hyperbolic step in `s ≈ ceil((sqrt(9+16 r)−1)/2)` substages (Meyer, Balsara & Aslam 2014), i.e. cost
`~sqrt(r)` rather than `~r`. So the trade is: **flux-fuse a mild operator** (`r ≲ 1`, dt_exp does not
bind — free); **super-time-step a stiff one** (`r ≫ 1`, sqrt(r) ≪ r). The pragmatic cutoff is
`FLUX_FUSE_RATIO = 2.0` (flux-fuse only operators whose explicit dt is within ~2× of the hyperbolic dt);
it is a tunable constant in the diagnostic, not a hard physical boundary.

**The diagnostic.** `tst/test_suite/verification/stiffness_diagnostic.py` reports, per operator,
`dt_exp` (mirroring each operator's C++ `ExplicitStableDt` closed form) relative to `dt_hyp` (the
hyperbolic CFL limit) on a **representative-difficult** MagLIF stagnation config — 5 µm resolution, hot
(~2 keV) DD fuel, cold (~10 eV) Al liner edge, optically-thick and optically-thin radiation — built from
standard Spitzer/Braginskii transport closed forms (NRL Plasma Formulary). It also reports the resulting
RKL2 stage count and the routing label. It is unit-tested against hand-computed known ratios on a
controlled dimensionless config (`test_verify_stiffness_diagnostic_cpu.py`). The measured ratios on the
difficult config (engineering estimates, robust to O(1) constant uncertainty):

| Operator | dt_exp/dt_hyp ratio `r` | RKL2 stages `~sqrt(r)` | Decision |
|----------|-------------------------|------------------------|----------|
| Electron/ion thermal conduction (iso + Braginskii aniso, κ∥) | ~3·10³ | ~110 | **STS** |
| Resistive B (Spitzer η, cold-liner / vacuum floor) | ~2·10³ | ~90 | **STS** |
| Multigroup FLD radiation — optically-thick (diffusive) | ~9·10³ | ~190 | **STS** |
| Multigroup FLD radiation — optically-thin (free-streaming) | ~1·10⁴ | ~230 | **STS** *(see caveat)* |

**Decision.** All three stiff HED operators — thermal conduction, resistive B, and multigroup FLD
radiation — route to **STS**. None is remotely mild on the difficult config (every `r` ≫ the
FLUX_FUSE_RATIO cutoff by 3+ orders of magnitude), which is exactly the stiffness ADR-0001 chose STS to
absorb; flux-fusing any of them would collapse the global timestep by ~10³–10⁴×. The flux-fused path is
retained as the documented home for any operator that *is* mild (e.g. ordinary isotropic viscosity, which
is out of the MagLIF stack), and such operators add no per-substage exchange by construction.

**Caveat — thin-cell radiation (restates ADR-0001's "known consequence").** The optically-thin radiation
"stiffness" is **not** diffusive: the Larsen limiter free-streams at `c`, so `dt_exp` collapses to the
`c`-CFL floor `dt ~ dx/c`, and STS **cannot** accelerate a hyperbolic light-speed term (the large stage
count there is the symptom, not a win). The mitigation is the **local reduced transport speed** in
flux-limiter-saturated cells — built *only if* measurement on a real run shows it sets the global
timestep (ADR-0001) — **not** flux-fusion (which would slow everything else equally) and **not** a global
reduced speed of light (which distorts thick-interior equilibration). This keeps "absence of a thin-cell
mitigation" a falsifiable, measured decision rather than a silent assumption.
