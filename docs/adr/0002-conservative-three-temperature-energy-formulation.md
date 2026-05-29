# Conservative three-temperature (3T) energy formulation

**Context.** MagLIF/Z-pinch needs separate ion and electron temperatures (`T_i` → fusion yield;
`T_e` → conduction, resistivity, opacity, radiation coupling) plus a multigroup radiation field.
AthenaK is single-temperature and evolves a conservative total energy. We need 2T-matter + radiation
without losing the conservative shock capturing that makes the converging-shock/stagnation physics
correct, and without a temperature formulation (temperature is not conserved, and the tabulated EOS
relation `e(T,ρ)` is nonlinear).

**Decision.** Use the CASTRO/FLASH-style **total-energy + species-internal-energy** formulation:
- Keep AthenaK's **conservative `E_tot`** update untouched (includes `½B²`); guarantees shock jumps and
  exact global energy conservation.
- Additionally evolve **electron internal energy `e_ele`** (advected via the passive-scalar machinery
  as `e_ele/ρ`; PdV work and physics sources added operator-split) and the **multigroup radiation
  energies `E_g`** (kept *outside* `E_tot`; coupled by a source term).
- Recover **`e_ion = E_tot − ½ρv² − ½B² − e_ele`** by subtraction → total energy conserved by
  construction; irreversible **shock dissipation lands on ions** (physically correct).
- **Compression (PdV) heating** splits between species by pressure fraction (`p_ele/p_gas`).
- **Temperatures are derived, cached fields:** per-cell monotonic inversion of the tabulated EOS
  (`e` monotone in `T` since `c_v>0` → bracketed binary-search + Newton using table `c_v`). The MHD
  pressure is `p_gas = p_ele(ρ,T_e) + p_ion(ρ,T_i)`; magnetic pressure/tension come from the field solver.
- Requires a **tabulated 2T EOS** (IONMIX) in the `ConsToPrim` path, taking `(ρ, e_ele, e_ion)`.

Delivered **staged**: 1T-matter (just `E_tot`) radiation/conduction/resistivity first, then add `e_ele`
+ subtraction to reach 2T-matter (re-targeting the radiation coupling from the single matter `T` to `T_e`).

**Known risk (dual energy).** `e_ion` by subtraction is a difference of large numbers in cold,
high-kinetic-energy regions → possible precision loss (Bryan et al. dual-energy problem). Mitigation if
it bites: switch to directly evolving `e_ion` in those cells. Not pre-built.

**Rejected.** Temperature formulation (non-conservative, fights EOS nonlinearity); evolving all species
internal energies without a conservative total (mis-partitions/loses shock dissipation).
