# Single Strang-split RKL2 super-step over a composite parabolic operator

**Context.** ADR-0001 chose operator-split RKL2 super-time-stepping behind the pluggable
`ParabolicIntegrator`/`ParabolicOperator` interface, but deliberately left two questions
open: (1) when several stiff spatial operators are active (anisotropic Braginskii e/i
conduction, resistive `B_φ` diffusion, multi-group FLD radiation), do we run *one* STS
super-step over their sum or *one sweep per operator*, and (2) Lie (1st-order) vs Strang
(2nd-order) splitting between the parabolic block and the explicit hyperbolic MHD update.
The integration epic must settle both before wiring the operators into the production MHD
timestep. The per-cell point-implicit matter–radiation / electron–ion coupling is a
separate, point-stiff (zero-stencil) solve and is *not* a `ParabolicOperator`.

**Decision.**
- **Composite operator, one super-step.** A `CompositeParabolicOperator : ParabolicOperator`
  sums the sub-operators' actions, `M(u) = Σ_i M_i(u)`, returns `min_i ExplicitStableDt_i`
  (so the RKL2 stage count is set by the single stiffest term), and runs as **one** RKL2
  super-step with **one ghost exchange per substage shared across all physics**. This is
  correct because every operator already obeys the frozen-background rule (writes 0 into
  components it does not evolve); operators that touch the *same* component (isotropic +
  anisotropic conduction → energy) must **accumulate**, not overwrite.
- **Strang-wrap the parabolic block** around the hyperbolic update
  (`½·STS · hydro · ½·STS`) for global 2nd-order accuracy. The point-implicit coupling is
  Strang-wrapped *outside* the STS block (it needs no halos; do not inflate it by the STS
  substage count `s`).
- **Keep RKL2** (not RKL1/RKC) and **keep the deferred implicit backend** behind the same
  interface as the documented stagnation fallback (see Known consequence).

**Why.** This is the proven design in every production FV code that uses STS — Athena++,
AthenaPK (the Kokkos/Parthenon descendant closest to AthenaK), PLUTO (Vaidya 2017), and
MHDSTS (Nóbrega-Siverio 2018) all sum the parabolic operators into one super-step with `s`
from the minimum diffusive `dt`. RKL2's stability polynomial only needs to bound the
largest eigenvalue of the *summed* operator, which `dt_diff = min` guarantees, so summing
carries no stability penalty. Summing also dissolves the finicky Lie/Strang ordering
*among* the parabolic terms (they are added, never sequenced). **Strang is nearly free:**
because `s ∝ √(dt/dt_par)`, two half-super-steps of `s/√2` stages each total the same
substage count as one full super-step of `s` stages — the "2×" is two boundary/startup
cycles, not 2× the flux work — so we get 2nd order without paying for it, which matters
because MagLIF's strong stagnation/MRT gradients make 1st-order Lie splitting error
pollute the energetics. RKL2 over RKC because RKC needs a hand-tuned damping parameter
(`ν≈0.25/N²`) that is easy to destabilize; RKL2 has none and is monotonicity-preserving
and robust with saturated conduction (Vaidya 2017).

**Performance follow-ons (tuning, not architecture).** Fuse the summed RHS into one kernel
per substage (read the shared halo once); async-overlap the per-substage halo with interior
compute and exchange only the parabolic variables. Cap `s` and subcycle the stiffest
operator when `dt_hyp/dt_par` blows up (cf. Athena++ `sts_max_dt_ratio ~100`). The
optically-thin FLD free-streaming term saturates at `|F|→cE`, a c-CFL term STS cannot
accelerate (`dt≲Δx/c`); handle it — *only if measurement on a representative-difficult case
shows it sets the global step* — with a **local reduced-c in flux-limiter-saturated cells**
(ADR-0001's stated mitigation), NOT a global reduced-speed-of-light, which would distort the
thick-interior equilibration timescale. Do not build reduced-c preemptively.

**Known consequence — the stagnation fallback.** The validation anchor itself (FLASH,
Ellison et al. 2025, arXiv:2504.10760) does **not** use STS — it solves all diffusion fully
implicitly (HYPRE backward-Euler). The reason is the regime where STS is least trustworthy:
RKL's monotonicity guarantee is proven only for *linear, constant-coefficient* diffusion,
whereas Spitzer `κ∝T^{5/2}` is strongly solution-dependent, and Vaidya 2017 shows the
nonlinear Sharma-Hammett transverse limiter leaks *excess cross-field heat as `s` grows* —
catastrophic for MagLIF, whose entire premise is suppressing cross-field conduction. The
named trigger to enable the implicit backend (which this composite design keeps intact — the
backend integrates the same `CompositeParabolicOperator`): cross-field leakage growing with
`s`, or `s` capping out, in the dense stagnation zones. Apply Sharma-Hammett only as a
floor-triggered safety and instrument cross-field leakage vs `s`.

**Considered and rejected.**
- **Separate STS sweep per operator** — N× ghost exchanges per step plus a Lie splitting
  error *between* operators; the literature unanimously avoids it.
- **Lie split around the hyperbolic update** — 1st-order; the splitting error at MagLIF
  stagnation looks like a physics bug in the energetics, and Strang costs almost nothing
  here.
- **Global reduced-speed-of-light up front** — distorts equilibration in the optically-thick
  interior; the local saturated-cell cap is surgical and conditional on measurement.
- **RKC / RKL1** — RKC's damping parameter is fragile; RKL1 is only 1st-order.
