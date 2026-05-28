# Verification provenance index

This index records the **ground-truth oracle** of every HED verification and unit test, per the
policy in [ADR-0008](adr/0008-verification-provenance-policy.md). Each row states:

- **Test** — the test module under `tst/test_suite/` (directory given per section).
- **Layer** — **1** (independent oracle) or **2** (self-captured `harness.verify` snapshot). See ADR-0008.
- **Oracle type** — `analytic`, `literature`, `other-code`, or `self-snapshot`.
- **Ground-truth oracle** — what the test actually compares against.
- **Citation** — the published reference or canonical method; closed-form oracles cite the solution itself.
- **Method anchor** — for Layer-2 (`self-snapshot`) rows, the Layer-1 test that establishes the
  underlying method's correctness. Layer-1 rows are their own anchor (`self`).

Most Layer-1 tests *also* keep a `harness.verify` baseline as a regression guard; that does not change
their layer — the binding assertion is the independent oracle.

**Scope.** This index covers the HED verification + unit tests in
`tst/test_suite/{verification,unit_tests,cylindrical}/`. AthenaK's upstream astrophysical regression
tiers (`nr/`, `sr/`, `gr/`, `z4c/`, `dyngrmhd/`, `rad/`, `ion-neutral/`, `sbox/`) are out of scope —
they validate the unmodified astro solvers against AthenaK's own published test problems.

---

## Hydrodynamics

Directories: `cylindrical/`, `verification/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_verify_cyl_sod_cpu.py` | 1 | analytic | Exact Sod Riemann solution at t=0.2 (radial run at large r, geometric source ≪ discretization error); L1 error within tolerance | Toro 2009; Sod 1978 | self |
| `test_verify_cyl_mignone_cpu.py` | 1 | analytic | Closed-form geometric-dilution solution ρ(r,t)=(r₀/r)ρ₀(r₀), r₀=r−vt; PLM 2nd-order convergence (L1 ~ N⁻²) | closed form; cf. Mignone 2014 cyl advection | self |
| `test_cyl_equilibrium_cpu.py` | 1 | analytic | Uniform-pressure cylindrical static equilibrium: geometric source S₁=⟨1/r⟩(ρv_φ²+p) cancels flux divergence, held to round-off | closed form (steady state) | self |
| `test_verify_cyl_blast_cpu.py` | 2 | self-snapshot | Shock-front sphericity (azimuthal distortion (r_max−r_min)/r_ave < tol) + harness baseline; **no quantitative blast-wave oracle** | — | `test_verify_cyl_sod_cpu.py`, `test_verify_cyl_mignone_cpu.py` — **TODO(#82)** |
| `test_verify_sod_cpu.py` | 2 | self-snapshot | harness.verify baseline of the t=0.25 profiles only; **no exact-Riemann comparison** | — | `test_verify_cyl_sod_cpu.py` — **TODO(#79)** |

## Magnetohydrodynamics

Directories: `cylindrical/`, `verification/`, `unit_tests/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_verify_cyl_lwave_cpu.py` | 1 | analytic | Fast-magnetosonic linear wave on a div-free B_z equilibrium; 2nd-order convergence (L1 ratio → ¼ per doubling) | closed form; Stone et al. 2008 (convergence) | self |
| `test_verify_cyl_zpinch_cpu.py` | 1 | analytic | Gaussian-current Z-pinch magnetohydrostatic equilibrium with closed-form p(r) balancing the −B_φ²/r hoop stress; max\|v_r\| at truncation | closed form; Freidberg 2014 (Z-pinch MHS) | self |
| `test_verify_cyl_bphi_diffuse_cpu.py` | 1 | analytic | Bessel-J₁ resistive eigenmode decay B_φ(r,t)=A·J₁(kr)·exp(−η k² t) | closed form (Bessel-J₁ eigenmode); Abramowitz & Stegun 1972 | self |
| `test_verify_resistive_decay_cpu.py` | 1 | analytic | Transverse Fourier mode B_y(x,t)=b₀ sin(kx)·exp(−η k² t) with analytic Spitzer η(ρ,T_e); measured decay rate vs analytic | closed form; Spitzer 1962, Braginskii 1965 | self |
| `test_unit_cyl_mhd_ct_divb_cpu.py` | 1 | analytic | Cylindrical constrained-transport curl-B divergence vanishes to machine precision (div(B)=0) | closed form (CT identity) | self |
| `test_verify_cyl_field_loop_cpu.py` | 2 | self-snapshot | max\|div(B)\| at round-off (initial + advected) + harness baseline; **no advected-loop shape oracle** | — | `test_unit_cyl_mhd_ct_divb_cpu.py` — **TODO(#83)** |

## Radiation — flux-limited diffusion and matter coupling

Directories: `verification/`, `unit_tests/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_verify_marshak_fld_cpu.py` | 1 | analytic | Error-function Marshak wave E_r(x,t) with equilibrium diffusion D=c/(3χ) and Larsen flux limiter | closed form (erfc); Marshak 1958, Zel'dovich & Raizer 1967; FLD anchor arXiv:2504.10760 | self |
| `test_verify_multigroup_fld_cpu.py` | 1 | analytic | Per-group erfc Marshak wave with group-specific D_g=c/(3χ_g) from tabulated Rosseland opacity | closed form (per-group erfc); Marshak 1958, Zel'dovich & Raizer 1967; arXiv:2504.10760 | self |
| `test_verify_rad_equilibration_cpu.py` | 1 | analytic | Radiative equilibrium a·T⁴=E_r and erfc Marshak front with D=c/(3χ) | closed form | self |
| `test_verify_multigroup_rad_hydro_cpu.py` | 1 | analytic | Planck-spectrum equilibrium E_g=a·T_eq⁴[F(x_{g+1})−F(x_g)] + local spectral lock; RK4 ODE reference; independent quadrature for F(x) | closed form (Planck spectrum) + RK4 | self |
| `test_verify_ohmic_2t_chain_cpu.py` | 1 | analytic | RK4 reference integration of the coupled Ohmic→electron→radiation ODE; backward-Euler truncation + energy conservation | closed form (RK4 ODE reference) | self |
| `test_unit_fld_grey_operator_cpu.py` | 1 | analytic | Larsen limiter limits (λ(0)=⅓ diffusion limit, free-streaming cap, monotonicity); eigenvalue with D=c/(3χ); conservation | closed form; Larsen limiter | self |
| `test_unit_fld_multigroup_operator_cpu.py` | 1 | analytic | Per-group Larsen limiter + per-group eigenvalue λ_g with D_g=c/(3χ_g) from tabulated Rosseland opacity; conservation | closed form (per-group) | self |
| `test_unit_matter_radiation_coupling_cpu.py` | 1 | analytic | Point-implicit grey coupling: relaxation to a·T⁴=E_r, exact E_r+e_mat conservation, unconditional stability for stiff absorption | closed form (equilibrium + conservation) | self |
| `test_unit_multigroup_coupling_cpu.py` | 1 | analytic | Independent host-quadrature fractional Planck function F(x); Planck-spectrum equilibrium E_g=a·T_e⁴[F(x_{g+1})−F(x_g)]; exact conservation | closed form (Planck fraction) | self |
| `test_unit_ionmix_opacity_cpu.py` | 1 | analytic | IONMIX multigroup opacity reader: group structure, per-group Planck/Rosseland reproduction, (ρ,T_e) bilinear interpolation, out-of-range clamping | closed form (log-log bilinear on synthetic table) | self |

## Thermal transport, resistivity, and equation of state

Directories: `unit_tests/`, `cylindrical/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_unit_braginskii_transport_cpu.py` | 1 | analytic | Braginskii reference coefficients (collision times, parallel κ/η, Spitzer resistivity); unmagnetized & strongly-magnetized limits; Bohm anomalous term | Braginskii 1965; Spitzer 1962 | self |
| `test_unit_aniso_conduction_cpu.py` | 1 | analytic | Field-aligned heat-flux projection (κ∥≫κ⊥); Parrish–Stone ring test; Sharma–Hammett monotonicity; Larsen free-streaming cap | Parrish & Stone 2005; Sharma & Hammett 2007 | self |
| `test_unit_variable_resistivity_cpu.py` | 1 | analytic | Spitzer/Braginskii η composition: T⁻¹·⁵ scaling, η_⊥≥η_∥ magnetization anisotropy, vacuum floor | Spitzer 1962; Braginskii 1965 | self |
| `test_unit_parabolic_conduction_cpu.py` | 1 | analytic | Cosine diffusion eigenmode: operator returns λ(E−E₀), exp(λt) decay, 2nd-order temporal convergence, energy conservation | closed form (diffusion eigenmode) | self |
| `test_unit_eos_table_3t_cpu.py` | 1 | analytic | Synthetic ideal-gas 3T table: analytic energies/pressures/c_v/Z̄ under log-log bilinear, T→e→T round-trip, monotonic inversion | closed form (ideal-gas table) | self |
| `test_unit_ionmix_eos_cpu.py` | 1 | analytic | IONMIX 3T EOS reader: per-species e/p/c_v/Z̄ reproduction, (ρ,T) bilinear, e→T inversion, out-of-range clamping | closed form (synthetic IONMIX table) | self |
| `test_unit_sesame_eos_cpu.py` | 1 | analytic | SESAME reader (tables 304/305 electron/ion + 601 ionization): (ρ,T) laws, e→T inversion, c_v derivation | closed form (synthetic SESAME table) | self |
| `test_unit_cons_to_prim_2t_cpu.py` | 1 | analytic | Tabulated 2T EOS inversion (e→T) + pressure composition for distinct γ; edge clamping; ideal-gas limit | closed form | self |
| `test_unit_three_temp_cpu.py` | 1 | analytic | 3T energy reconciliation: e_tot conservation, PdV split by pressure fraction, shock dissipation to ions, 1T ideal-gas recovery | closed form (energy identities) | self |
| `test_verify_cyl_aniso_ring_cpu.py` | 2 | self-snapshot | Azimuthal-vs-radial broadening ratio (field alignment), cross-field leakage < tol, Sharma–Hammett monotonicity + harness baseline; **no documented ring-test oracle** | — | `test_unit_aniso_conduction_cpu.py` — **TODO(#81)** |

## Circuit drive (ADR-0005)

Directories: `unit_tests/`, `cylindrical/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_unit_drive_source_cpu.py` | 1 | analytic | Prescribed I(t) waveforms (constant/linear_ramp/sin_squared), tabulated interpolation, boundary formula B_φ=μ₀I/(2πr) | closed form (Ampère's law) | self |
| `test_unit_lumped_circuit_cpu.py` | 1 | analytic | Series-RLC ODE analytic step response (underdamped/overdamped current, capacitor voltage, RL limit) + B_φ boundary formula | closed form (RLC step response) | self |
| `test_unit_faraday_voltage_cpu.py` | 1 | analytic | Poloidal-flux reduction Φ=∬B_φ dr dz vs discrete sum; Faraday voltage V_load=d(Φ)/dt vs analytic rate | closed form (Faraday's law) | self |
| `test_verify_driven_liner_cpu.py` | 1 | analytic | Thin-shell slug model m_l·d²R/dt²=−μ₀²I(t)²/(4πR); simulated trajectory matches the ODE | closed form (thin-shell ODE) | self |
| `test_verify_coupled_liner_cpu.py` | 1 | analytic | Pure-inductor coupled loop flux conservation L·I+Φ=V_oc·t; Faraday feedback V_load=d(Φ)/dt; thin-shell trajectory from measured I(t) | closed form (flux conservation); arXiv:2504.10760 | self |

## Geometry, integrators, and scaffolding

Directory: `unit_tests/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_unit_coord_geometry_cpu.py` | 1 | analytic | Cartesian coordinate accessors return analytic uniform-grid constants; flux divergence reduces to (F_{i+1}−F_i)/dx bit-for-bit | closed form (Cartesian metric) | self |
| `test_unit_rkl2_sts_cpu.py` | 1 | analytic | RKL2 amplification factor R_s(z)=1+z+z²/2+O(z³); 2nd-order convergence vs exp(λt); L2 stability at 256× explicit dt | Meyer, Balsara & Aslam 2014 | self |
| `test_unit_fld_amr_flux_correction_cpu.py` | 1 | analytic | Operator-split diffusion conserves the evolved quantity across an AMR fine/coarse boundary (discrete conservation residual at round-off) | closed form (conservation identity) | self |
| `test_unit_gauss_legendre_cpu.py` | 1 | analytic | Cross-integrals of spin-weighted spherical harmonics on the Gauss–Legendre grid are δ functions (orthonormality) | closed form (orthonormality) | self |
| `test_unit_ohmic_electron_heating_cpu.py` | 1 | analytic | Ohmic η\|J\|² deposited on electrons consistent with the EMF; 2T backward-Euler coupling to radiation; ions untouched; total energy conserved | closed form (energy identities) | self |
| `test_unit_sample_cpu.py` | — | n/a | Demonstration scaffold for the auto-collection contract; `ApproxEqual` self-checks against hardcoded values — not a physics oracle | — | — |

## MagLIF integrated benchmarks

Directory: `cylindrical/`. The validation ladder follows the FLASH MagLIF benchmarks of
**Ellison et al. 2025 (arXiv:2504.10760)** — the project's verification anchor (ADR-0005). The
benchmark *configuration and expected qualitative growth signature* are the published ground truth;
each test also keeps a `harness.verify` regression baseline.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_maglif_cpu.py` | — | analytic (IC) / smoke | Layered liner/fuel/vacuum IC + uniform B_z + B_φ=μ₀I/(2πr) drive IC sanity; driven run launches and stays finite (stability smoke, no evolution oracle) | closed form (IC formulas) | self |
| `test_verify_maglif_mrt_cpu.py` | 1 | literature | FLASH benchmark 1 — single-mode MRT: seeded-mode amplitude growth on the magnetically-driven liner interface under sustained acceleration | Ellison et al. 2025 (arXiv:2504.10760) | self |
| `test_verify_maglif_mmrt_cpu.py` | 1 | literature | FLASH benchmark 2 — multi-mode MRT from surface roughness: broadband RMS growth + spectral participation (stays multi-mode) | Ellison et al. 2025 (arXiv:2504.10760) | self |
| `test_verify_maglif_rm_cpu.py` | 1 | literature | FLASH benchmark 3 — converging single-mode Richtmyer–Meshkov: single-mode amplitude growth during compression | Ellison et al. 2025 (arXiv:2504.10760) | self |
| `test_verify_maglif_icf_cpu.py` | 1 | literature | FLASH benchmark 4 — ICF confinement-time signature: bulk stagnation via enclosed-mass tracking (R_if, ρ_fuel) | Ellison et al. 2025 (arXiv:2504.10760) | self |

---

## Layer-2 self-snapshot gaps (TODO)

The four tests below assert only against a self-captured snapshot (plus qualitative sanity
properties) and have no documented Layer-1 oracle yet. Each is grounded for now by the listed
method-correctness anchor and is scheduled to be upgraded to an independent oracle by a follow-up slice
(per ADR-0008).

| Test | Method anchor (current) | Grounding slice |
|------|-------------------------|-----------------|
| `test_verify_sod_cpu.py` (planar Sod) | `test_verify_cyl_sod_cpu.py` (exact Riemann) | **V4 / #79** — exact-Riemann oracle |
| `test_verify_cyl_aniso_ring_cpu.py` | `test_unit_aniso_conduction_cpu.py` (Parrish–Stone ring) | **V6 / #81** — documented ring-test oracle |
| `test_verify_cyl_blast_cpu.py` | `test_verify_cyl_sod_cpu.py`, `test_verify_cyl_mignone_cpu.py` | **V7 / #82** — documented oracle |
| `test_verify_cyl_field_loop_cpu.py` | `test_unit_cyl_mhd_ct_divb_cpu.py` (div(B)=0) | **V8 / #83** — documented advected-loop oracle |
