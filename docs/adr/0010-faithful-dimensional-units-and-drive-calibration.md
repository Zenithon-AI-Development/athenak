# Faithful dimensional benchmarks: one consistent code-unit system, conversion only at the EOS-table boundary

**Context.** The Ellison benchmarks ([[maglif-verification-anchor]]) are run as *faithful dimensional*
setups (real geometry, the measured z2173 drive, tabulated 3T EOS), not code-unit surrogates
([[benchmarks-run-as-surrogates-not-faithful]]). Three dimensional inputs must coexist consistently:
the IONMIX EOS table (physical CGS: specific energy in erg/g, T in eV, rho in g/cc), the load-current
trace (`tst/inputs/z2173_current.dat`, columns `time [ns]`, `current [MA]`), and the magnetic drive
`B_phi = mu0*I/(2*pi*r)` (Heaviside-Lorentz, `mu0=1`). With `length_cgs` (1 mm) and `density_cgs`
(material solid density) fixed and `mu0=1`, **`velocity_cgs` is the only remaining free knob**, and the
EOS table currently loads in raw physical units with nothing bridging it to the code-unit evolved
quantities ([[ionmix-eos-built-not-wired]], [[b1b4-faithful-run-still-ideal-eos]]).

**Decision.** Everything internal is in **one AthenaK code-unit system**; physical CGS appears *only* at
two boundaries — the EOS-table lookup and post-run reporting.
- **`velocity_cgs` is a numerical-conditioning knob, not a measurement.** Any consistent choice yields
  identical physics; pick it so code numbers are O(1) (e.g. code time unit ~ 1 ns over a ~100 ns run).
- **Convert at the EOS-table boundary, both directions.** Scale the loaded IONMIX table into code units
  **once at load** (per `eos_table_3t.hpp` "a reader sets the units when it fills the table"): specific
  energy `/ velocity_cgs^2`, pressure `/ (density_cgs*velocity_cgs^2)`, density axis `/ density_cgs`.
  The **temperature axis stays in eV** (T is a derived diagnostic, not an evolved conserved variable;
  k_B conversions live only where T meets energy). Every lookup — solver `ConsToPrim2T` *and* the pgen
  initial conditions — is then natively code-unit. One conversion point, consistent everywhere.
- **Initial conditions use the table, not an ad-hoc `p0`.** Set a physical initial temperature per
  material (`T_e=T_i=T_init`, a cold LTE start, clamped up to the table T floor), look up per-species
  specific energies, and fill both total energy and the `e_ele` passive scalar.
- **Drive trace converts on load:** `time [ns]` -> code time, `current [MA]` -> code current via the
  unit factors derived from the chosen system (`mu0=1` is left untouched).
- **Validate the whole chain with one closed-form anchor:** peak drive magnetic pressure `B^2/2` at
  `r_out` must match `mu0*I^2/(8*pi^2*r^2)` (~5 Mbar at 20 MA / 3.47 mm). Any unit error in *any*
  conversion breaks this single check (Layer-1 independent oracle, [[verification-anchoring-gap]],
  ADR-0008).

**Rejected.** Pinning `velocity_cgs` to a physical reference (a measured implosion velocity, or pressure
unit = exactly 1 Mbar) — rejected as false precision; the degrees of freedom are honestly "one free
conditioning knob + one physical anchor test". Per-lookup unit conversion in the hot c2p path — rejected
for a single load-time scaling pass instead.
