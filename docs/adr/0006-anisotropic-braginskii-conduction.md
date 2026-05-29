# Anisotropic (magnetized) Braginskii thermal conduction

**Context.** In MagLIF the axial `B_z` (premagnetization, compressed to ~10³ T) thermally insulates the
hot fuel from the cold liner *across* field lines — this directional insulation is the operating
principle of magneto-inertial fusion. AthenaK has only isotropic conduction, which would unphysically
bleed fuel heat radially. The verification anchor ([[maglif-verification-anchor]]) uses magnetized
Braginskii electron+ion conduction with a Bohm-like anomalous term.

**Decision.** Implement **anisotropic Braginskii electron and ion thermal conduction** with parallel and
perpendicular components: `q = −κ∥ b̂(b̂·∇T) − κ⊥(∇T − b̂(b̂·∇T))`, `κ∥/κ⊥ ~ (ω_cτ)²`. Coefficients from
**analytic Braginskii formulae** (functions of `T`, `n`, `Z`, `lnΛ`, magnetization `ω_cτ`) plus the
**Bohm-like anomalous cross-field term** (`ν_bohm = 16 ω_ce`). Apply the **Larsen flux limiter** on the
field-aligned flux and **Sharma–Hammett (2007) monotonic slope limiters** to prevent unphysical
cross-field heat leakage on the grid. Integrate via **RKL super-time-stepping** (ADR-0001) — anisotropic
diffusion is RKL's canonical use case (Vaidya 2017). Port the operator + limiters from Athena++/PLUTO.

**Deferred.** The antisymmetric **Righi-Leduc `b̂×∇T` cross-term** — small for heat flux; its more
important thermoelectric cousin **Nernst** is extended-MHD (scoped out in [[adr-0003]], needed for
verification benchmarks 5–6). Both come together when extended MHD is added.

**Rejected.** Isotropic conduction with a scalar magnetization cap — misses the directional field-line
insulation that defines MagLIF.
