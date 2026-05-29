# Ohmic (Joule) heating routed to electrons, consistently with constrained transport

**Context.** AthenaK advances resistive energy as a *conservative Poynting flux* (`OhmicEnergyFlux`:
`dE/dt = −∇·(η(J×B))`), with the resistive EMF `ηJ` (edge-centered, `OhmicEField`) diffusing B via
constrained transport. Joule heating is implicit: as B decays, conserved `E_tot` returns the energy to
internal. In our 3T subtraction (`e_ion = E_tot − KE − ME − e_ele`), that implicit heat lands on the
**ions** by default — physically backwards, since the current is electron-carried. This is the classic
3T MagLIF Ohmic-heating bug.

**Decision.** Keep the conservative resistive flux (total energy conserved, B diffuses correctly) and
add an **explicit electron Joule-heating source** to `e_ele`, computed from the **same discrete edge
current/EMF (`J=∇×B`, `ηJ`) that drives the CT B-diffusion** — i.e. deposit into `e_ele` the *local
(non-transport) magnetic-energy decrement* of the CT update, so electron gain = magnetic loss, the ion
residual is ~0, and total energy stays conserved. Do **not** independently re-discretize `ηJ²` at a
different centering — that mismatch is the energy leak. Resistivity `η` becomes Spitzer/tabulated
`η(ρ,T_e)`, used identically for B-diffusion and electron heating.

**Scoped out (deliberate v1 approximation).** Nernst (B advected by electron heat flux) and
Ettingshausen effects — extended-MHD thermoelectric cross terms — are omitted under the resistive-only
scope. They are known to matter at strongly Ohmic-heated Z-pinch edges (B transport, suppression of e-i
temperature separation); their absence is the most likely "resistive-only isn't enough" surprise and is
the trigger to revisit extended MHD.

**Rejected.** GORGON/HYDRA-style direct internal-energy/temperature equations with explicit `ηj²→e`
and no strict total-energy conservation — simpler partition, but incompatible with the conservative
shock-capturing core chosen in ADR-0002.

**Resistivity transport model (folded in).** `η` is **anisotropic Braginskii `η∥/η⊥`**, from the same
transport-coefficient set as the anisotropic conduction (ADR-0006) — consistent magnetized transport.
Add the **vacuum-resistivity floor** (assign large `η` to cells below a user-set density) to suppress
unphysical currents in low-density/vacuum regions, per the verification anchor
([[maglif-verification-anchor]]). The anisotropic `B_φ` resistive diffusion still uses the special
`1/r²` cylindrical operator + antisymmetric axis ghost (ADR-0004).
