# AthenaK — HED / Fusion Plasma Physics

Glossary for the radiation-resistive-MHD capabilities being added to AthenaK for
MagLIF and Z-pinch (set-pitch) fusion simulations. AthenaK's pre-existing domain is
astrophysical (GR)MHD; this context covers the high-energy-density (HED) extensions
and disambiguates terms that mean different things in the astro vs. HED-lab-plasma worlds.

## Language

### Radiation

**Multi-group flux-limited diffusion (FLD)**:
The chosen radiation model. Radiation energy is split into discrete photon-energy
**groups**; within each group radiation is treated as a diffusing energy field whose
flux is capped by a **flux limiter** so it never exceeds free-streaming (c·E). Parabolic,
solved implicitly per group. This is the FLASH MagLIF approach.
_Avoid_: "radiation transport" (ambiguous — implies angular/Sₙ transport, which we are NOT doing).

**Group**:
A photon-energy bin with fixed energy bounds. The radiation field carries one energy
density per group per cell. (Distinct from a Kokkos thread group or an MPI group.)

**Flux limiter**:
A closure that interpolates the radiative flux between the diffusion limit (optically
thick) and the free-streaming limit (optically thin). **Decision: use the Larsen
flux limiter** (smooth, well-behaved across the optically thick/thin transition).
The same Larsen form is reused for flux-limited thermal conduction.

**Three distinct "limiters/coefficients" in magnetized transport (do not conflate)**:
These three answer different questions and are used *together*; the grilling for the
Phase-integration work showed they are easy to confuse.
- **Braginskii coefficients** — the *physics*: `κ∥`, `κ⊥` (and `η∥`, `η⊥`) as functions of
  magnetization `ω_cτ`. How much heat/current flows along vs. across **B**. (ADR-0006.)
- **Larsen flux limiter** — *physical saturation*: caps a flux at its free-streaming limit
  (radiation `|F|→cE`; field-aligned heat flux at the free-streaming electron flux). The
  radiation form uses the tabulated Rosseland opacity, `D = cλ(R)/χ`, `R=|∇E|/(χE)`.
  (CONTEXT "Flux limiter" above.)
- **Sharma–Hammett (2007) limiter** — *numerical monotonicity*, NOT saturation and NOT
  Braginskii: a van-Leer slope limiter on the **transverse** temperature gradient in the
  anisotropic-conduction stencil, so the *discretization* cannot leak heat cold→hot across
  field lines. Matters wherever `∇T` is oblique to **B** (the `B_z`-insulated fuel/liner
  interface — the MagLIF mechanism itself). (ADR-0006; src/diffusion/aniso_conduction_operator.*)
_Avoid_: calling Sharma–Hammett a "flux limiter" (it is a slope limiter) or associating it
with Braginskii (it is purely a grid-discretization safeguard).

## Flagged ambiguities

**"Radiation" (module-name collision)**:
AthenaK already has a `Radiation` class (`pmbp->prad`) — **grey M1 closure, hard-locked to general
relativity** — serving the astrophysical GRMHD path. The HED work adds a *different* radiation model:
**multi-group flux-limited diffusion** for non-relativistic MagLIF/Z-pinch. These must not share a name.
Resolution: the existing GR module keeps **Radiation / M1**; the new module is **Radiation-FLD**
(`prad_fld` or similar). When someone says "radiation" unqualified, ask which.

### Plasma energetics

**3T**:
This project's three-temperature model: **ion temperature** `T_i`, **electron temperature** `T_e`,
and the **radiation field** (resolved spectrally as multigroup FLD energies, not a single `T_rad`).
"3T" elsewhere can mean ion/electron/radiation as three scalar temperatures — here the radiation
component is the multigroup field. When precision matters, say `T_i`, `T_e`, and "radiation groups".

**Internal energy (the evolved quantity)**:
We evolve *energies*, not temperatures. `E_tot` (conservative MHD total) and `e_ele` (electron
internal energy) are evolved; `e_ion` is recovered by subtraction. Temperatures `T_e(e_ele,ρ)`,
`T_i(e_ion,ρ)` are **derived** by inverting the tabulated EOS per cell. "Temperature" is always a
derived, cached field — never a primary variable.

**Gas pressure vs magnetic pressure**:
`p_gas = p_ele + p_ion` is the thermal pressure from the EOS. `p_mag = B²/2` (Heaviside-Lorentz
code units) is the Maxwell pressure, supplied by the MHD field solver, **not** the EOS. "Pressure"
unqualified is ambiguous in a pinch where `p_mag` dominates — name which one.

### Circuit drive

**Drive source**:
The pluggable origin of the boundary current `I(t)` that sets `B_φ(R_out)=μ₀I(t)/2πR_out`. Three modes:
(A) **prescribed current** — tabulated `I(t)`, no feedback (FLASH benchmarks 1–4); (B) prescribed
**open-circuit voltage** + fixed RLC; (C) prescribed voltage + **coupled circuit with load feedback**
(default target). "Driven" defaults to a *voltage* source feeding a circuit, not a current.

**Load voltage**:
The voltage the imploding load presents back to the circuit, `V_load = d(L_load·I)/dt + I·R_load`.
Computed by **Faraday's law as the rate of change of magnetic flux** via a global reduction over the
domain — the feedback term closing the coupled circuit (mode C).

**nocurrent boundary**:
The vacuum radial BC outside the current-carrying load enforcing `∂_r(r·B_φ)=0` → constant enclosed
current → `B_φ ∝ 1/r` decay. Distinct from the **circuit** BC, which *drives* `B_φ` from `I(t)`.

### Units & calibration

**Code units**:
AthenaK's single internal unit system. Every evolved quantity, EOS lookup, and the magnetic
drive live here. Set by `length_cgs`, `density_cgs`, `velocity_cgs` in `<units>`. Physical CGS
appears only at two boundaries: the EOS-table lookup and post-run reporting (ADR-0010).
_Avoid_: "physical units" for in-code quantities — say which side of the boundary you mean.

**`velocity_cgs`**:
A numerical-conditioning knob, **not** a measurement. With `length_cgs`, `density_cgs` and the
Heaviside-Lorentz `mu0=1` fixed, it is the one remaining free scale; any consistent value gives
identical physics. Pick it so code numbers are O(1).
_Avoid_: treating it as a measured/derived velocity, or "the absolute scale" — there is none to set.

**EOS-table unit boundary**:
The one place physical CGS meets code units. The IONMIX table is scaled into code units once at
load (energy `/velocity_cgs^2`, pressure `/(density_cgs*velocity_cgs^2)`, density axis
`/density_cgs`; the **temperature axis stays in eV**). Every lookup afterwards is code-unit native.
_Avoid_: per-cell unit conversion in the solver hot path.

**Magnetic-pressure anchor**:
The closed-form calibration check `B^2/2 = mu0*I^2/(8*pi^2*r^2)` at the outer radius (~5 Mbar at
20 MA / 3.47 mm). A unit error anywhere in the drive/EOS calibration breaks it; it is the Layer-1
independent oracle for "the dimensional setup is self-consistent" (ADR-0008, ADR-0010).

## Verification

Every HED verification and unit test declares an independent, citable **ground-truth oracle**. The
policy (Layer-1 independent oracle vs. Layer-2 self-captured snapshot, and what counts as "grounded")
is [ADR-0008](docs/adr/0008-verification-provenance-policy.md); the per-test
**[verification provenance index](docs/verification.md)** records each test's oracle type, citation,
and — for snapshot-only tests — the Layer-1 test that anchors the underlying method's correctness.

**Benchmark shortfall attribution**:
A faithful-benchmark miss is blamed on a missing physics model only with evidence and only from the
*reference's* (Ellison/FLASH) model set — tabulated 3T EOS, conduction, resistivity, gray/multigroup
radiation — after the upstream anchors (magnetic-pressure calibration, implosion timing, 1D-stays-1D +
two-grid convergence) are cleared. **Material strength is excluded** (FLASH matched B1–4 strengthless).
([ADR-0011](docs/adr/0011-benchmark-shortfall-attribution-policy.md).)
_Avoid_: "the code lacks model X, so X is the gap" (attribution-by-absence).

**Operator-coupling status (faithful B1)**:
The B1 growth residual is bounded by three model-set **wiring/consistency** gaps (not strength),
closed one at a time: **(a) [CLOSED, #181/[P7a]]** `resb` (resistivity) is now coupled to the live
driven face field — the resb super-step is bracketed by a copy-in (`b0.x2f→bphi`) / write-back
(`bphi→b0.x2f`) gated on `resb_couple_b0`, so resistivity diffuses the *driving* `B_phi` and is
**active**; **(b) [#182/[P7b]]** the gray-FLD `erad` is still a standalone field the `maglif` pgen
never sources (inert); **(c) [#183/[P7c]]** `acond` (conduction) / `mrad` (coupling) still bake
**ideal-gamma** thermodynamics (`T=(γ-1)e/ρ`, constant `c_v`) inconsistent with the tabulated closure.
([ADR-0012](docs/adr/0012-b1-operator-coupling-attribution.md).)
