# Tabulated EOS and opacity via pluggable readers into a common 3T-native representation

**Context.** The 2T/3T formulation (ADR-0002) needs separate electron and ion internal energies and
derives temperatures by per-cell `e→T` table inversion. The verification anchor
([[maglif-verification-anchor]]) uses SESAME + PROPACEOS EOS and TOPS opacities; the user requires
**SESAME + IONMIX** EOS now and **IONMIX** opacity now (TOPS possibly later). IONMIX is natively 3T
(separate `e_ele`, `e_ion`, `Z̄`); SESAME is usually 1T-total.

**Decision.** A **common internal table representation** with **pluggable format readers**, keeping the
physics format-agnostic:
- **EOS readers: IONMIX and SESAME** populate the shared representation; the per-cell monotonic `e→T`
  inversion + forward lookups (`p_ele`, `p_ion`, `Z̄`, `c_v`) operate only on that representation.
- **Require natively-3T tables only.** The reader accepts a table for the 2T path *only* if it carries
  separate electron/ion data (IONMIX always; SESAME only its 3T variants). **No 1T→2T split modeling** —
  cleanest physics, no ad-hoc electron/ion partitioning; the cost is that 1T-only SESAME materials are
  not usable for 2T runs (acceptable).
- **Opacity readers: IONMIX** now (multigroup Planck absorption, Planck emission, Rosseland transport),
  interface ready for **TOPS** / **PROPACEOS** later. Opacity feeds two consumers: transport opacity →
  FLD diffusion coefficient `D=c/3κ_R`; Planck absorption/emission → the point-implicit coupling.
- **Table-edge handling**: clamp to table bounds (floors/ceilings); monotonicity of `e(T)` (from
  `c_v>0`) guarantees the inversion is well-posed inside bounds.

**Rejected.** A 1T→2T split model for SESAME (degeneracy/ionization-modeled electron/ion partition) —
rejected to avoid ad-hoc physics; use 3T-native tables instead. Hardwiring a single format — the
verification suite spans IONMIX/SESAME/TOPS, so readers must be pluggable behind one representation.
