# The faithful B1 growth residual is bounded by three reference-model-set wiring/consistency gaps, not material strength

**Context.** ADR-0011 fixed the *attribution* of a faithful-benchmark shortfall: the residual is
bounded by the reference (Ellison/FLASH, arXiv:2504.10760) model set — tabulated 3T EOS, electron/ion
conduction, resistivity, gray radiation diffusion — and **material strength is excluded** (FLASH
matched B1 strengthless). Issue #175/[P6] then asked the next question: *enable that model set in the
faithful tabulated-3T B1 run (`inputs/maglif_b1_sinars.athinput`), one operator at a time, and see
whether the amplitude(t) growth gap closes.* The operators exist as operator-split modules (#108/#110/
#116, ADR-0001/0006/0009) and the coupled stack runs cleanly on GPU since #139. The investigation found
that **none of them yet alters the tabulated B1 implosion faithfully**, for code-structural reasons —
not because the physics is wrong, but because the production `maglif` pgen does not yet *couple* each
operator to the live, tabulated MHD state:

1. **Resistivity (`resb_operator_split`) is inert.** The operator advances a *standalone* `bphi`
   array (`mhd.cpp` realloc; `mhd_tasks.cpp` `OperatorSplitResistiveBphi`/`SF_BPHI`) that the `maglif`
   pgen never fills from — nor writes back to — the live MHD face field `b0.x2f` (where `maglif.cpp`
   sets the driven `B_phi`). Only the dedicated resb diffusion/unit-test pgens populate it. So with
   `eos=tabulated_3t` the resb step diffuses a zero field and the implosion is unchanged.
2. **Gray FLD (`fld_operator_split`) is inert** on the gas. It diffuses a *standalone* `erad` the
   `maglif` pgen never sources; without `mrad_coupling` it never touches the gas energy at all.
3. **Conduction (`acond_operator_split`) is EOS-inconsistent.** It acts on the live `u0` IEN but
   recovers temperature as `T = (gamma-1) eint/rho` (`aniso_conduction_operator.cpp` `TempMHD`,
   `gamma` = the ideal-gamma bookkeeping value), which is **not** the tabulated_3t electron/ion-split
   temperature. It changes the gas, but via an ideal-gamma closure inconsistent with the faithful EOS.
4. **Matter-radiation coupling (`mrad_coupling`) is inert in practice, and EOS-inconsistent.** It
   exchanges `erad <-> IEN` with a **constant** heat capacity `mrad_cv` (`e_gas = c_v T`,
   `mhd_tasks.cpp`), not the tabulated closure — but it has nothing to exchange against because
   `erad` is never sourced (gap 2), so `fld+mrad` leaves the implosion **bitwise identical** to
   baseline (measured). The EOS-inconsistent `c_v` only bites once `erad` is sourced.

Additionally every operator coefficient (`resb_eta`, `acond_kappa_conv`, the `fld_*`/`mrad_*` set) is
an **uncalibrated placeholder** in code units — there is no Spitzer/Braginskii/IONMIX-opacity SI
calibration analogous to the #174/ADR-0010 drive calibration.

**Decision.** Per the ADR-0011 gate (upstream anchors cleared: the closed-form magnetic-pressure
calibration anchor, implosion timing, 1D-stays-1D + two-grid convergence — all operator-independent or
verified to still hold), the **faithful B1 amplitude(t) growth residual is attributed to three
reference-model-set engineering gaps, all of which must be closed before any operator can faithfully
move the verdict**:

- **(a) Couple `resb`'s `bphi` to the live `b0.x2f`** in the `maglif` pgen (copy-in before the
  super-step, write-back after), so resistivity diffuses the *driving* azimuthal field.
- **(b) Source the FLD `erad`** from the gas state in the `maglif` IC / per-step, so radiation is fed.
- **(c) Make `acond` temperature and `mrad` heat capacity EOS-aware** — read `T` and `c_v` from the
  tabulated_3t closure rather than the ideal-gamma relation — so conduction and matter-radiation
  coupling are thermodynamically consistent with the faithful EOS.

All three are **within the reference's model set** (conduction / resistivity / radiation); **none is
material strength** (ADR-0011 still holds). They are filed as a follow-up with this evidence. Until they
land, the faithful B1 amplitude(t) comparison stays **reported (binding=False)**: enabling the operators
on `tabulated_3t` either leaves the curve unchanged (resb, fld, fld+mrad — inert) or shifts it
through an ideal-gamma closure (acond) that is not the faithful physics, so neither can flip the
gate to binding.

**Consequences.** The per-operator experiment is wired as a permanent regression
(`test_verify_maglif_b1_operators_gpu.py`): it enables each operator one at a time on the faithful B1
setup and asserts the structural attribution facts above (resb, fld and fld+mrad leave the
seeded-mode amplitude *bitwise identical* to baseline; only acond changes it, while staying finite),
recording each amplitude(t) oracle
verdict in the scorecard. The faithful baseline benchmark (`test_verify_maglif_b1_sinars_gpu.py`) is
unchanged. Closing the B1 gap (#120 AC#2) requires the three gaps above, not strength. Complements
ADR-0011 (attribution policy), ADR-0010 (faithful units), ADR-0009 (composite Strang/RKL2 super-step).

---

## Update (2026-06-03, #181/[P7a]): gap (a) closed — `resb` coupled to the live `b0.x2f`

The first of the three gaps is now closed. The `maglif` faithful-B1 setup couples the resb
operator's standalone `bphi` to the live driven azimuthal face field `b0.x2f`: when the new
`<mhd> resb_couple_b0` knob is on, every resb RKL2 super-step (both the full-step
`OperatorSplitResistiveBphi` task and the Strang-half `StrangParabolicHalf` `SF_BPHI` branch) is
bracketed by a **copy-in** (`b0.x2f → bphi`, `MHD::CoupleResbBphiFromB0`) before and a **write-back**
(`bphi → b0.x2f`, `MHD::CoupleResbBphiToB0`) after, so resistivity diffuses the *driving* `B_phi`
rather than a decoupled zero field. The copy-in/write-back mirror the maglif IC face layout (B_phi is
the x2-face field, uniform in phi for the axisymmetric column; the top x2-face of the last phi cell
is set at `j==je`). The total energy `u0(IEN)` is left unchanged across the super-step, so the next
`ConsToPrim` recovers the magnetic-energy decrement as gas internal energy (Ohmic dissipation);
making that flow EOS-consistent for `tabulated_3t` is gap (c)/#183 and the SI `η` calibration is
#184/[P7d].

The knob is **gated off by default** (`resb_couple_b0=false`), read only inside the
`resb_operator_split` setup block, so every existing run — including the resb unit/verification pgens
that fill their own `bphi` IC (`resb_bphi_multiblock`, `cyl_bphi_diffuse`) and the operators-off
maglif baseline — stays **byte-identical**. `test_verify_maglif_b1_operators_gpu.py` flips the `resb`
case from `inert:True` to **ACTIVE** (seeded-mode amplitude(t) ≠ baseline, while its `pert_amp=0`
control stays exactly 1-D — radial `B_phi` diffusion preserves axisymmetry). Gaps (b) #182 and (c)
#183 remain; the paper-resolution amplitude(t) oracle re-attribution stays #120 AC#2.
