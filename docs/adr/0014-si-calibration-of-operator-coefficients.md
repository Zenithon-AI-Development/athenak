# SI calibration of the reference-model-set operator coefficients (the operator analogue of the ADR-0010 drive calibration)

**Context.** ADR-0010 fixed the faithful *drive* units: one consistent code-unit system
(`length_cgs`, `density_cgs`, `velocity_cgs`, Heaviside-Lorentz `mu0=1`), with physical CGS
crossing into code units only at the EOS-table and drive-trace boundaries, validated by the
closed-form magnetic-pressure anchor. ADR-0012 then closed the three reference-model-set
operator *wiring/consistency* gaps for the faithful B1 run — resistivity coupled to the live
`b0.x2f` (#181), the FLD `erad` sourced from the gas (#182), and the EOS-aware `acond`
temperature / `mrad` heat capacity (#183) — but explicitly left **every operator coefficient an
uncalibrated placeholder in code units**: `resb_eta`, the `acond_*_conv` factors, `mrad_chi_a`
and `fld_chi` were all order-unity guesses ([[b1-operator-coupling-gaps]]). With the operators
wired but running at arbitrary strength, they cannot faithfully move the B1 amplitude(t) verdict
(#120 AC#2). This is ADR-0012 **gap (d)**, filed as #184/[P7d].

**Decision.** Derive each operator coefficient from **first-principles transport / opacity** and
convert it into the code-unit system fixed by `<units>` **at a single calibration boundary**,
exactly as `circuit::CalibrateSiDrive` does for the drive trace. The physics primitives already
exist (`diffusion/braginskii_transport.hpp` for Spitzer/Braginskii, `opacity/` for the IONMIX
multigroup opacity); the new `diffusion/operator_si_calibration.hpp` is a pure (no grid, no I/O)
header that wraps them and applies the unit factors. It is opt-in via `<mhd>
operators_si_calibrate` (default OFF, gated to `eos=tabulated_3t`), so every existing run — and
the per-operator attribution test, which overrides it OFF — stays **byte-identical**.

The code-unit system (the ADR-0010 one). With `length_cgs` L [cm], `density_cgs` D [g/cc],
`velocity_cgs` V [cm/s]: length unit `Lu = L*1e-2` [m], density unit `Du = D*1e3` [kg/m^3],
velocity unit `Vu = V*1e-2` [m/s], time unit `Tu = Lu/Vu = L/V` [s], pressure unit
`Pu = Du*Vu^2 = 0.1*D*V^2` [Pa] (the ADR-0010 `p_unit_pa`).

The conversions (each one its own anchor, like ADR-0010's single magnetic-pressure check):

- **Resistivity `resb_eta`** (a magnetic *diffusivity*, length^2/time). The Spitzer-Braginskii
  parallel resistivity `eta_|| (T_e, Z, lnLambda)` [Ohm m] (NRL practical value
  `5.2e-5 Z lnLambda T_e[eV]^(-3/2)`, density-independent) is divided by `mu0` to the magnetic
  diffusivity `eta/mu0` [m^2/s] (the form `d_t B = (eta/mu0) grad^2 B` uses), then over the code
  diffusivity unit `Lu*Vu = L*V*1e-4` [m^2/s]:
  `resb_eta = (eta_||/mu0_SI) / (L*V*1e-4)`.

- **Conduction `acond_*_conv`** (the anisotropic-conduction operator computes Braginskii
  `kappa(rho,T,B)` in SI from code-unit inputs, so calibration is its code<->SI factors):
  - `dens_conv` (code rho -> ion number density `n_i` [m^-3]) `= Du/m_i = D*1e3 / m_i[kg]`.
  - `temp_conv` (code temperature -> [K]). On the faithful EOS-aware path the conducted
    temperature is the tabulated electron temperature in **eV** (ADR-0010 keeps the table T axis
    in eV), so `temp_conv` is exactly the eV->K factor `kEvToKelvin = e/k_B ~= 11604.5`.
  - `bmag_conv` (code |B| -> [Tesla]). In Heaviside-Lorentz code units magnetic pressure is
    `B_code^2/2` over `Pu`, and SI magnetic pressure is `B_SI^2/(2 mu0_SI)`, so
    `B_SI = B_code*sqrt(mu0_SI*Pu)` — the same field/current relation the ADR-0010 drive current
    conversion uses. Hence `bmag_conv = sqrt(mu0_SI * Pu)`.
  - `kappa_conv` (SI `kappa` [W/m/K] -> code conduction coefficient). The operator forms
    `q = -kappa dT/dx` with T in eV and x, t, energy in code units; matching the SI
    energy-density rate `d/dx(kappa dT/dx)` to the code rate (`Pu/Tu`) gives
    `kappa_code = kappa_SI * kEvToKelvin / (Lu*Vu*Pu)`. (Derivation: `rate_SI = rate_code*Pu/Tu`;
    `d^2T_K/dx_m^2 = (kEvToKelvin/Lu^2) d^2T_eV/dx_code^2`; equate and use `Tu = Lu/Vu`.)

- **FLD / matter-radiation opacity `mrad_chi_a` / `fld_chi`** (absorption coefficients, 1/length).
  An IONMIX mass opacity `kappa` [cm^2/g] at density `rho` [g/cc] is the CGS absorption
  coefficient `chi = kappa*rho` [1/cm]; one code length is L cm, so `chi_code = kappa*rho*L`.
  `mrad_chi_a` uses the Planck-absorption opacity, `fld_chi` the Rosseland transport opacity,
  both read from the run material's IONMIX cn4 table at the reference state.

**Reference state.** Resistivity and conduction depend on `(T_e, Z, lnLambda)` (Spitzer eta is
density-independent); the opacity coefficients scale with the reference density. The faithful B1
deck supplies the aluminum reference (`si_calib_te_ev = 100`, `si_calib_zbar = 6.65` and the
Rosseland/Planck opacities are the `al-imx-004.cn4` values at the ~100 eV / solid-density node).
The grey FLD/coupling operators use a single representative opacity (not a per-cell table lookup),
so the representative IONMIX value at the reference node is the right calibration target.

**Validation (ADR-0008 Layer-1 independent oracle).** `test_unit_operator_si_calibration_cpu`
asserts, RED-first against the placeholders: the Spitzer `eta_||` matches the NRL closed form, the
Braginskii `kappa_||e` matches the NRL/Spitzer-Harm value, and the FLD/mrad opacity is the real
aluminum IONMIX cn4 value at a known node converted by `kappa*rho*L`; plus meta-checks that each
calibrated code-unit coefficient is grossly inconsistent with the order-unity placeholder the deck
carried (proving the calibration is necessary). The faithful B1 deck enables
`operators_si_calibrate` so the coupled #120 paper-resolution run consumes the calibrated
coefficients; the per-operator attribution test (`test_verify_maglif_b1_operators_gpu`) overrides
it OFF to keep pinning the #181-183 structural facts with the known placeholders.

**Rejected.** Per-cell IONMIX opacity lookups inside the FLD/coupling operators — rejected for
this issue: the grey operators carry a single constant opacity by design, so a representative
table value at the reference state is the correct, minimal calibration. Pinning the reference
temperature to a measured implosion temperature — rejected as false precision (the reference state
is an honest single conditioning choice, documented in the deck). Calibrating the radiation
constants `a_rad` / `c_light` here — out of scope for #184 (named coefficients are eta / kappa /
opacity); they are radiation-unit constants, ADR-0010-adjacent.

**Consequences.** All four ADR-0012 gaps (a)/(b)/(c)/(d) are now closed: the faithful B1 model set
is wired *and* calibrated. The amplitude(t) oracle re-attribution at paper resolution (#120 AC#2)
can now be re-run on a fully consistent, SI-calibrated model set. Complements ADR-0010 (faithful
units / drive calibration), ADR-0012 (operator coupling), ADR-0006 (Braginskii conduction),
ADR-0007 (tabulated EOS/opacity readers).
