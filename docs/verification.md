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
| `test_verify_cyl_blast_cpu.py` | 1 | analytic | Circular symmetry (isotropy) of the self-similar blast front: on a deliberately misaligned (r,φ) polar grid the azimuthal distortion (r_max−r_min)/r_ave < tol measures the curvilinear scheme's geometric error; harness baseline is the regression guard | Sedov 1959, Taylor 1950 (self-similar blast); Londrillo & Del Zanna 2000, Stone et al. 2008 (Athena blast test) | self |
| `test_verify_sod_cpu.py` | 1 | analytic | Exact Sod Riemann solution at t=0.25 (planar shock tube); relative L1 (density+pressure) within tolerance, Linf bounded by the ~1-cell contact/shock smearing | Toro 2009; Sod 1978 | self |

## Magnetohydrodynamics

Directories: `cylindrical/`, `verification/`, `unit_tests/`.

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_verify_cyl_lwave_cpu.py` | 1 | analytic | Fast-magnetosonic linear wave on a div-free B_z equilibrium; 2nd-order convergence (L1 ratio → ¼ per doubling) | closed form; Stone et al. 2008 (convergence) | self |
| `test_verify_cyl_zpinch_cpu.py` | 1 | analytic | Gaussian-current Z-pinch magnetohydrostatic equilibrium with closed-form p(r) balancing the −B_φ²/r hoop stress; max\|v_r\| at truncation | closed form; Freidberg 2014 (Z-pinch MHS) | self |
| `test_verify_cyl_bphi_diffuse_cpu.py` | 1 | analytic | Bessel-J₁ resistive eigenmode decay B_φ(r,t)=A·J₁(kr)·exp(−η k² t) | closed form (Bessel-J₁ eigenmode); Abramowitz & Stegun 1972 | self |
| `test_verify_resistive_decay_cpu.py` | 1 | analytic | Transverse Fourier mode B_y(x,t)=b₀ sin(kx)·exp(−η k² t) with analytic Spitzer η(ρ,T_e); measured decay rate vs analytic | closed form; Spitzer 1962, Braginskii 1965 | self |
| `test_unit_cyl_mhd_ct_divb_cpu.py` | 1 | analytic | Cylindrical constrained-transport curl-B divergence vanishes to machine precision (div(B)=0) | closed form (CT identity) | self |
| `test_verify_cyl_field_loop_cpu.py` | 2 | self-snapshot | max\|div(B)\| at round-off (initial + advected) is the binding physics oracle; azimuthally-averaged \|B\|_rms(r) harness baseline is a cited-method regression guard for the expected slow numerical diffusion of the advected loop | Gardiner & Stone 2005 (field-loop advection test) | `test_unit_cyl_mhd_ct_divb_cpu.py` (div(B)=0) |

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
| `test_verify_cyl_aniso_ring_cpu.py` | 2 | self-snapshot | Field-aligned conduction reproducing the published ring test: azimuthal-vs-radial broadening ratio, cross-field leakage < tol, Sharma–Hammett monotonicity; harness baseline is the regression guard | Parrish & Stone 2005 (ring test); Sharma & Hammett 2007 (limiter) | `test_unit_aniso_conduction_cpu.py`, `test_unit_parabolic_conduction_cpu.py` |
| `test_verify_conduction_multiblock_cpu.py` / `_mpicpu.py` | 1 | analytic | Same cosine diffusion eigenmode as `parabolic_conduction`, but the GLOBAL mode split over 4 MeshBlocks (cpu) / 4 MPI ranks (mpicpu): exp(λt) decay reproduced across block/rank boundaries via `SyncParabolicGhosts`, energy conserved across the decomposition (#108/[A1]) | closed form (diffusion eigenmode) | `test_unit_parabolic_conduction_cpu.py` |
| `test_verify_fld_marshak_multiblock_cpu.py` / `_amr_cpu.py` / `_mpicpu.py` | 1 | analytic | Grey FLD wired operator-split into the MHD timestep (`mhd/fld_operator_split`, #110/[A3]): the erfc Marshak wave E_r(x,t)=e_floor+(e_source−e_floor)·erfc((x−x1min)/(2√(Dt))), D=c/(3χ), reproduced across 4 MeshBlocks (cpu), a static-SMR coarse/fine boundary (amr_cpu), and 4 MPI ranks (mpicpu) via `SyncParabolicGhosts`; insulated diffusion conserves volume-weighted radiation energy across the decomposition | closed form (erfc Marshak wave); Marshak 1958, Zel'dovich & Raizer 1967 | `test_verify_marshak_fld_cpu.py` |
| `test_verify_mgfld_marshak_multiblock_cpu.py` / `_amr_cpu.py` / `_mpicpu.py` | 1 | analytic | Multigroup FLD wired operator-split into the MHD timestep (`mhd/mgfld_operator_split`, #111/[A4]): EVERY group's erfc Marshak wave E_g(x,t)=e_floor+(e_source−e_floor)·erfc((x−x1min)/(2√(D_g t))), D_g=c/(3χ_g) from the tabulated Rosseland opacity, reproduced across 4 MeshBlocks (cpu), a static-SMR coarse/fine boundary (amr_cpu), and 4 MPI ranks (mpicpu) via the batched `SyncParabolicGhosts` + `CorrectFlux`; insulated diffusion conserves volume-weighted radiation energy across the decomposition | closed form (per-group erfc Marshak wave); Marshak 1958, Zel'dovich & Raizer 1967 | `test_verify_multigroup_fld_cpu.py` |
| `test_verify_mgfld_regrid_cpu.py` / `_mpicpu.py` | 1 | analytic (conservation) | The standalone multigroup group-energy arrays (`MHD::erad_mg`) are conservatively restricted/prolonged across a DYNAMIC AMR regrid (#111/[A4]): a static gas carries a smooth per-group field diffused by an insulated operator while a user criterion refines then de-refines the central blocks; the volume-weighted total radiation energy is invariant to round-off across the regrids (same-rank Derefine/Copy/Refine on cpu; cross-rank load-balance Pack/Unpack on mpicpu) | conservation (conservative restrict/prolong preserves the volume integral) | `test_verify_mgfld_marshak_multiblock_cpu.py` |
| `test_verify_aniso_conduction_multiblock_cpu.py` / `_amr_cpu.py` / `_mpicpu.py` | 1 | analytic | Anisotropic Braginskii conduction wired operator-split into the MHD timestep (`mhd/acond_operator_split`, #112/[A5]): a field-aligned cosine mode (bhat=xhat) reproduces its analytic parallel decay exp(λ∥t), λ∥=−(2D∥/dx²)(1−cos θ), D∥=κ∥(γ−1)/ρ, across 4 MeshBlocks (cpu), a static-SMR coarse/fine boundary (amr_cpu), and 4 MPI ranks (mpicpu) via `SyncParabolicGhosts`; the perpendicular mode (D⊥≪D∥) stays frozen and the Parrish–Stone ring keeps heat on its circular field lines (azimuthal ≫ radial leak) — heat follows field lines across the decomposition; cross-field leakage is bounded+converged across an RKL2 substage sweep; volume-weighted energy conserved. Embedded RED→GREEN: the same mode built WITHOUT the exchange (`pin==nullptr`) self-insulates each block (~60% Linf error) | closed form (anisotropic diffusion eigenmode); Parrish & Stone 2005 (ring); Sharma & Hammett 2007 (limiter) | `test_unit_aniso_conduction_cpu.py`, `test_verify_conduction_multiblock_cpu.py` |
| `test_verify_resb_bphi_multiblock_cpu.py` / `_amr_cpu.py` / `_mpicpu.py` | 1 | analytic | Cylindrical resistive B_φ diffusion (the −η B_φ/r² curl-curl operator) wired operator-split into the MHD timestep (`mhd/resb_operator_split`, #113/[A6]): the Bessel-J₁ eigenmode B_φ(r,t)=A·J₁(kr)·exp(−η k² t) reproduced across 4 radial MeshBlocks on the full disk r∈[0,R], k=j₁,₁/R (cpu — exercises the antisymmetric axis ghost + near-axis 1/r² stiffness), a static-SMR radial coarse/fine boundary (amr_cpu), and 4 MPI ranks on a radial annulus r∈[r₀,R] with k=j₁,₂/R, r₀=j₁,₁/k (mpicpu — the annulus excludes the axis so the per-rank explicit dt is uniform and the synchronous cross-rank `SyncParabolicGhosts` cannot deadlock pending the #114 global min-dt reduction) via `SyncParabolicGhosts`; the J₁ shape (hence the −B_φ/r² term) is preserved and the J₁-projected decay matches analytic. Embedded RED→GREEN: the same mode built WITHOUT the exchange (`pin==nullptr`) sign-flips B_φ at the internal radial faces (~28% Linf error) | closed form (Bessel-J₁ resistive eigenmode); Abramowitz & Stegun 1972; Meyer, Balsara & Aslam 2014 (RKL2 STS) | `test_verify_cyl_bphi_diffuse_cpu.py` |
| `test_verify_composite_parabolic_cpu.py` / `_mpicpu.py` | 1 | analytic | `parabolic::CompositeParabolicOperator` (summed action, min-dt, global min-dt MPI all-reduce; #114/[B1], ADR-0009): TWO isotropic ConductionOperators with different κ over the SAME energy field, both eigenoperators of the global cosine mode with λ_a=−(2D_a/dx²)(1−cos θ), D_a=κ_a(γ−1)/ρ. The composite action equals the SUMMED eigenvalue (λ₁+λ₂)·(E−E0) to machine precision (accumulate, not overwrite — materially distinct from a single operator's λ₂·(E−E0)) with 0 in the non-energy components; `ExplicitStableDt` returns min over sub-operators (the stiffer larger-κ operator); a single RKL2 super-step reproduces exp((λ₁+λ₂)t) across 4 MeshBlocks (cpu) / 4 MPI ranks (mpicpu) and conserves energy; the empty composite is inert (M=0, dt_exp→∞). The GLOBAL min-dt all-reduce is exercised cross-rank by a sub-operator pair over a PER-RANK-VARYING density (rank r: ρ=ρ0(1+r)): the composite returns the same global minimum (rank 0's) on every rank, ≤ each rank's local min and strictly below the cross-rank max — also the fix for the spatially-varying-dt deadlock (#113). Embedded RED→GREEN: an overwrite (non-accumulating) composite fails the summed-action and exp((λ₁+λ₂)t) checks | closed form (diffusion eigenmode); Meyer, Balsara & Aslam 2014 (RKL2); ADR-0009 (one super-step over the summed operator, dt=min over operators) | `test_verify_conduction_multiblock_cpu.py`, `test_unit_parabolic_conduction_cpu.py` |

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
| `test_verify_stiffness_diagnostic_cpu.py` | 1 | analytic | Per-operator stiffness diagnostic (#109): each operator's forward-Euler dt_exp (mirroring its C++ `ExplicitStableDt`), the hyperbolic CFL dt, the ratio→RKL2 stage count, and the STS-vs-flux-fuse routing — reproduced exactly on a controlled config; the difficult-MagLIF config routes conduction/resistive-B/thick-radiation to STS (ADR-0001 routing addendum) | closed form (FE diffusion limit; CFL; RKL2 stage count, Meyer et al. 2014); Spitzer 1962 / NRL Plasma Formulary | self |
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

## Experimental ground-truth oracle — Ellison benchmarks 1–4 (ADR-0008)

Directory: `verification/`. The **Layer-1 experimental oracle** the Phase-C benchmark
replications (#119–#122) compare their simulation observables against. The binding ground
truth is the **experiment** (digitized/stated data from the primary Sandia papers), never
another code — FLASH/LASNEX/HYDRA values, where stored, are secondary reference only. The
committed, provenance-tagged data lives in `verification/ground_truth/` (one JSON file per
benchmark, each datum carrying source paper + figure + DOI + extraction method); the oracle
(`verification/ground_truth_oracle.py`) loads it, validates the provenance, and compares
within a tolerance band = max(experimental error, digitization error). The test below
exercises that load/provenance/tolerance-band/compare logic (a component test, no sim).

| Test | Layer | Oracle type | Ground-truth oracle | Citation | Method anchor |
|------|-------|-------------|---------------------|----------|---------------|
| `test_verify_ground_truth_oracle_cpu.py` | 1 | literature (experiment) | Committed experimental data for Ellison 1–4: B1 single-mode MRT (Sinars 2011), B2 multi-mode MRT (McBride 2012/2013), B3 convergent RM (Knapp 2020), B4 ICF confinement (Knapp 2017, min-radius 0.45 mm, peak density ~10 g/cc); provenance-validated load + tolerance-band compare | Sinars PoP 18,056301 (2011) [10.1063/1.3560911]; McBride PRL 109,135004 (2012) [10.1103/PhysRevLett.109.135004] / PoP 20,056309 (2013) [10.1063/1.4803079]; Knapp PoP 27,092707 (2020) [10.1063/5.0013194]; Knapp PoP 24,042708 (2017) [10.1063/1.4981206] | self |

---

## Layer-2 self-snapshot gaps — all closed

Every Layer-2 (self-snapshot) test now carries a documented oracle/citation and a Layer-1
method-correctness anchor, so there are **no remaining verification gaps** under this index (per
ADR-0008). The snapshot-only tests flagged at adoption were each grounded by a V-series grounding slice:

- planar `sod` (`test_verify_sod_cpu.py`) — grounded in V4 / #79 against the exact Riemann solution
  (promoted to Layer 1).
- `cyl_aniso_ring` (`test_verify_cyl_aniso_ring_cpu.py`) — grounded in V6 / #81 against the published
  Parrish–Stone ring-test behavior plus the analytic `aniso_conduction` + `parabolic_conduction`
  unit-test method anchors (stays Layer 2 with anchor).
- `cyl_blast` (`test_verify_cyl_blast_cpu.py`) — grounded in V7 / #82 against the circular symmetry of
  the self-similar blast front, the Athena/Athena++ blast benchmark's symmetry oracle (promoted to
  Layer 1).
- `cyl_field_loop` (`test_verify_cyl_field_loop_cpu.py`) — grounded in V8 / #83: the machine-precision
  div(B) check is the binding physics oracle, the advected-loop configuration cites Gardiner & Stone
  2005 (field-loop advection test), and the analytic `test_unit_cyl_mhd_ct_divb` unit test (div(B)=0)
  is the method-correctness anchor; the `|B|_rms(r)` baseline is a cited-method regression guard for
  the loop's expected slow numerical diffusion (stays Layer 2 with anchor).
