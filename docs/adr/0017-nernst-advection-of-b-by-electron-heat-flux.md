# Nernst advection of B by the electron heat flux (upwinded, magnetization-reduced)

**Status. PROPOSED — human decision gate (#238).** This ADR reverses a documented scope-out
(ADR-0003 "scoped out (deliberate v1 approximation)"; ADR-0006 "deferred ... both come
together when extended MHD is added") and is **not accepted** until the human signs off on
the open questions below. What is committed with this draft is the red-first verification
contract only (two xfail-pinned red tests); the operator itself is gated on acceptance.
(ADR numbering note: 0016 is reserved by the in-flight AMR r-z draft.)

**Context.** Nernst advection of the magnetic field by the electron heat flux does not
exist in AthenaK. B5 (integrated shot z2977) and B6 (60 MA current scaling) both run it
(Ellison et al. 2025, arXiv:2504.10760, §2.1.2), and the paper's §3.5 shows why it is
load-bearing: laser preheat erects steep `∇T_e` in the fuel, Nernst advects the axial
magnetic flux from the hotspot toward the cold liner, and only the FLASH run *with* Nernst
reproduces the LASNEX + data-based `BR` (fuel magnetization) inference for z2977 (Fig. 15).
Without it the compressed `B_z` — hence magnetic insulation, `T_ion`, and yield — is
overestimated. The paper's form: Braginskii thermoelectric coefficient,
**magnetization-reduced**, and **upwinded** ("to mitigate numerical instabilities
occasionally observed when compressional waves converge on axis in a magnetized plasma,
leading to the creation of spots with negative axial magnetic field values").

**Decision (proposed).**

- **Operator form.** The induction equation gains the thermoelectric EMF
  `E_N = -v_N x B`, i.e. `dB/dt += curl(v_N x B)`: the field is *advected* at the Nernst
  velocity

  `v_N = -(beta_wedge(x_e)/x_e) * (tau_e/m_e) * grad(k_B T_e)`, `x_e = omega_ce tau_e`,

  with the Braginskii Z=1 wedge-thermoelectric fit
  `beta_wedge(x) = x (1.5 x^2 + 3.053) / (x^4 + 14.79 x^2 + 3.7703)` — the denominator is
  the same electron `Delta(x)` already carried by `diffusion/braginskii_transport.hpp`.
  The **magnetization reduction is intrinsic and bounded both ways**:
  `beta_wedge/x -> 3.053/3.7703 ~= 0.810` unmagnetized, `-> 1.5/x^2` strongly magnetized
  (no runaway at either limit). Equivalent statement (the issue title's phrasing):
  `v_N ~= -(2/5) q_perp_e / p_e` — flux rides the perpendicular electron heat flux
  (Braginskii 1965; Haines, PPCF 28, 1705, 1986; Nishiguchi et al., PRL 53, 262, 1984).
  New **pure-physics device functions** (`NernstCoefficient(x)` = `beta_wedge/x`, plus a
  velocity helper) join `braginskii_transport.hpp` under its existing physics-only SI
  contract; `tau_e`, `omega_ce`, `CoulombLog` are reused as-is.

- **Upwinding.** Donor-cell: the Nernst EMF is assembled at the CT edges with `v_N`
  interpolated to the edge and the advected `B` taken from the **upwind** side per the
  sign of `v_N` — the paper's cure for the on-axis negative-`B_z` spots, and the same
  lesson #194 taught for the FLD streaming front (a centered advective stencil is
  neutrally stable and RKL2 cannot damp it). First-order donor-cell is v1; MC-limited
  second-order reconstruction of the advected `B` is a named follow-up *only if* the
  upwind diffusion measurably pollutes the `BR(t)` observable.

- **Placement — NOT in the RKL2 parabolic composite.** Nernst is hyperbolic (advective),
  not parabolic; ADR-0009/#194 already established that advection-shaped operators sit
  outside RKL2's stability domain. Proposed placement: the Nernst EMF is added into the
  **corner EMFs of the constrained-transport update inside the MHD stage loop** (the
  Athena++ idiom for non-ideal EMFs), so it (a) rides the RK integrator at the hyperbolic
  dt with full temporal order, and (b) preserves `div(B) = 0 to machine precision by
  construction` (the CT circulation identity — the acceptance criterion). When gated on,
  `|v_N|` joins the MHD `NewTimeStep` signal-speed reduction (`dt <= CFL * dx / |v_N|`).
  If measurement shows Nernst-limited dt collapsing the step in a few preheated cells,
  the named fallback is subcycling the Nernst EMF within the step — do not build it
  preemptively (the ADR-0009 discipline).

- **Gating.** `<mhd> nernst = true`, default **false => byte-identical** (acceptance
  criterion; same discipline as `resb_couple_b0` / `strang_split` / every HED gate).
  Guards: `v_N = 0` in cells below the vacuum density floor (the vacuum-resistivity-floor
  philosophy — near-vacuum `tau_e/T_e` are meaningless there) and wherever the EOS
  closure yields no valid `T_e`. `T_e` comes from the tabulated electron closure when
  `eos = tabulated_3t` (the cached `ConsToPrim2T` inversion, ADR-0012(c)); the ideal-gas
  fallback is `T = p/rho` with per-operator `nernst_*_si` code<->SI conversions (the
  proven `resist_*_si` pattern of the Spitzer resistivity), to be folded into
  `operators_si_calibrate` (ADR-0014) for the faithful decks.

- **Interaction with resb and CT.** Nernst acts **only on the live face field `b0`
  through the CT EMFs — never on the standalone `bphi`** — so there is no double
  advection: the resb copy-in/write-back bracket (`resb_couple_b0`, ADR-0012(a)/
  ADR-0015) is unchanged, resb keeps *diffusing* `B_phi` in the parabolic slot while
  Nernst *advects* the same flux (via `b0.x2f`) in the hyperbolic stage loop. In the
  MagLIF r-z plane the physically decisive action is on `B_z` (`b0.x3f`), the compressed
  premagnetization flux that sets `BR`.

- **Energy bookkeeping.** The Nernst EMF changes magnetic energy; the conservative form
  carries the corresponding Poynting flux `E_N x B` in the total-energy flux (exactly how
  the Ohmic EMF is booked), so `E_tot` is conserved by construction and the field-energy
  change is billed against the gas internal energy — the ADR-0015 lesson, applied
  up-front instead of retrofitted. The thermoelectric *heat-flux* counterpart
  (Ettingshausen) is **scoped out as a documented v1 approximation** (ADR-0003
  precedent); flagged as open question (3).

- **Verification (red-first, committed with this draft — both xfail-pinned to #238).**
  1. `tst/test_suite/verification/test_verify_nernst_advection_cpu.py` — Layer-1
     analytic oracle (ADR-0008): static uniform-pressure 1-D medium with a linear `T_e`
     ramp (uniform `grad T_e`, imposed through `rho = p0/T` so the state is a hydro
     equilibrium) and a passive Gaussian `B_z` bump (`src/pgen/nernst_advection.cpp`,
     `tst/inputs/nernst_advection.athinput`); the bump centroid must translate by
     `v_N * tlim` with `v_N` from the same Braginskii chain evaluated independently in
     Python. RED today: measured displacement ~ 0.
  2. `tst/test_suite/unit_tests/test_unit_nernst_operator_cpu.py` — wrapper for the
     future pgen unit test `nernst_operator_test` (coefficient limits + magnetization
     roll-off, upwind switch, gate-off => zero action, `div(B)` at round-off). RED today:
     the pgen does not exist, the build fails.

  Green (post-acceptance) must flip both (strict xfail turns XPASS into a suite failure,
  forcing marker removal), add the harness baseline for the advected profile, and close
  with the GPU-clean smoke on athenakdev (acceptance criterion).

**Rejected.**

- **Advancing Nernst inside the RKL2 composite super-step** — advection has no place in
  an STS integrator built for real-negative spectra (#194's lesson); it would need
  permanent LF dissipation to fake stability and would inflate the stage count.
- **A once-per-step operator-split update on `b0` in the before/after slot** (the HED
  parabolic-operator pattern) — needs its own CT-consistent circulation machinery plus a
  first-order splitting error, duplicating what the stage-loop EMF assembly gives for
  free. Kept only as the fallback if the stage-loop wiring proves invasive (question 1).
- **Folding `v_N` into the Riemann-solver velocity** — corrupts mass/momentum/energy
  fluxes; Nernst advects only `B`.
- **A cell-centered (non-CT) update of `bcc`** — violates the machine-precision `div(B)`
  acceptance criterion.
- **Implicit treatment** — FLASH solves its *diffusion* implicitly, but Nernst is
  explicit upwinded advection there too; it is CFL-comparable, not stiff.

**Open questions for the human gate.**

1. Stage-loop CT-EMF placement (proposed) vs the operator-split before/after slot
   (rejected-but-fallback): the former touches the production MHD EMF assembly (gated,
   default byte-identical), the latter stays in the HED sandbox at the cost of splitting
   error and duplicated circulation machinery.
2. Whether to co-limit `v_N` with the Larsen-limited electron heat flux (Nernst-velocity
   saturation, consistent with flux-limited conduction — cf. the `v_N ~ q_e/p_e`
   identity) or accept the unlimited Braginskii form in v1 and pay the dt cost in
   preheated cells.
3. Is the Ettingshausen omission acceptable for the B5/B6 fidelity claims (it is the
   energy-equation twin of Nernst; FLASH's §2.1.2 model statement names only Nernst)?
4. Z-dependence: keep the module's documented Z=1 fit constants (dominant Z dependence
   enters via `tau_e`, matching `braginskii_transport.hpp`'s convention) vs adopting
   Epperlein-Haines Z-corrected thermoelectric fits now.

**Consequences.** Once **accepted**, this ADR supersedes the Nernst scope-outs in
ADR-0003 and ADR-0006 (Righi-Leduc remains deferred), unblocks the B5/B6 capability
ladder (#200/#201), and adds the first hyperbolic member to the extended-MHD stack —
establishing the CT-EMF pattern any future Hall/Biermann term would reuse. Until
accepted, ADR-0003/0006 stand and the red tests document the contract.
