# The B1 paper-res runaway: resb energy bookkeeping + the ox1 ghost-fed mass reservoir (not radiation)

**Context.** With the P7 chain merged (ADR-0012 wiring, ADR-0014 SI calibration), the full
coupled operator stack (`strang_split + resb + acond + fld + mrad`) on the faithful B1 deck
(`tst/inputs/maglif_b1_sinars.athinput`) ran away at paper resolution 288x4x128: peak density
1.59e9 code (solid Al = 1.0) by t=70 ns, collapse onset t~36-39 ns. Issue #192 hypothesised
radiative collapse (FLD/mrad over-cooling). The red test
(`tst/test_suite/cylindrical/test_verify_maglif_b1_coupled_gpu.py`) asserts bounded peak
density at paper res and gates a 1.5x-refinement consistency leg behind it (ADR-0011: a
runaway-vs-bounded flip is an orders-of-magnitude attribution question, so the convergence
band is deliberately loose: peak amplitude +/-30%, peak timing +/-6 ns).

**Evidence (bisect at paper res, tlim=70 ns, A100; raw artifacts `athenakdev:~/p192/runs/`).**

| config                      | peak rho (code) | verdict                  |
| --------------------------- | --------------- | ------------------------ |
| fld+mrad only               | 15.0            | bounded, gentle          |
| acond only                  | 15.1            | bounded, gentle          |
| resb only                   | 1.0e9           | RUNAWAY — the driver     |
| resb+acond+fld (no mrad)    | 1.0e9           | == resb-only to 4-5 digits |
| resb only + energy fix      | 1.0e9           | STILL runaway, same onset |
| resb only + energy fix + ox1 guard | 2.89        | bounded, gentle — GREEN  |

The radiative-collapse hypothesis is **wrong**: radiation (and conduction) alone are bounded
and gentle, and removing mrad from the resb stack changes nothing to 4-5 digits. The
runaway is two stacked bookkeeping defects in the resb coupled path, each independently
fatal at paper resolution:

**Defect 1 — the write-back energy pump.** The #181 coupling `MHD::CoupleResbBphiToB0()`
(`src/mhd/mhd_tasks.cpp`) replaced `b0.x2f` with the RKL2-diffused `bphi` while leaving the
total energy `u0(IEN)` unchanged, so the next ConsToPrim recomputed
`e_int = E - KE - B^2/2` with the field-energy change billed to the gas with the WRONG sign:
cells the driven field diffuses INTO are cooled by -dB^2/2. In the cold liner skin and the
tenuous vacuum gap (B^2/2 >> e_int) this drove `e_int` negative every step; the pressure
floor refilled it, fabricating energy continuously. Signals: etot(resb-only) = 1.5e-2 at
t=24 ns vs 2.0e-6 for rad-only — ~1e4 x more than the drive had delivered (drive scale
~1e-4 at peak current); interior B_phi amplified 25-44x above the boundary drive value
while the boundary value itself tracked mu0*I/2*pi*r correctly.

**Defect 2 — the ox1 ghost-fed mass reservoir.** With the energy pump fixed (commit
`53fd9f11`), the fixed resb-only probe STILL ran away with the same t~36-39 ns onset
(rho_max 1.0e9, mass x108 by t=51 ns) — but with etot now bleeding negative (the dropped
Ohmic heat, see trade-off below) instead of pumping positive, isolating the second defect.
Radial profiles show the mechanism: the boundary B_phi is correct, but magnetic pressure
raises an inward wind in the vacuum gap (outer-cell v_r reaches -1.05 by t=24 ns, -1.23 by
t=36 ns) and the zero-gradient hydro ghosts (`MagLIFBCs`, `src/pgen/maglif.cpp`) mirror that
wind back into the domain as fresh mass and momentum. The ghost density tracks the rising
outer cell (x600 by t=36 ns), the wind self-amplifies, flux-compresses interior B_phi to
~5x above even the full-penetration Ampere bound, and snowplows the liner: total mass grew
x77,000 by 70 ns pre-fix. A zero-gradient ghost at an open boundary is an unbounded
reservoir the moment the boundary-adjacent flow turns inward.

**Guarded probe (both fixes, resb-only, paper res, 70 ns):** rho_max 2.89, total mass
+0.016% (vs +0.1% for rad-only's own creep), etot +3.0e-4 — positive and at the drive
scale — R_liner 3.146 at 70 ns vs rad-only's 3.119 (comparable implosion pace), div(B)
3.5e-11. Both defects were individually fatal; both fixes together are bounded and
physical.

**Decision (two gated changes, committed defaults byte-identical).**

1. **Energy-consistent write-back** (commit `53fd9f11`): `CoupleResbBphiToB0()` now updates
   `u0(IEN) += 0.5*(b_new^2 - b_old^2)` per cell, making `e_int` invariant under the swap.
   Field energy a cell gains arrives via the operator's transport — the circuit does the
   work at the driven boundary — not billed to the local gas. Gated inside `resb_couple_b0`,
   so the operators-OFF baseline is untouched.
2. **ox1 inflow guard** (commit `01abd6cf`, `<problem> ox1_inflow_guard`, default off): a
   boundary column whose last interior cell moves inward gets static vacuum ghosts
   (rho = d_vac, v = 0, e_int = p0/(gamma-1), magnetic energy from the just-filled drive
   faces); outflowing columns keep the zero-gradient copy, so the open boundary still
   vents. Scalars are zeroed; in tabulated_3t runs scalar 0 (e_ele) edge-clamps at the
   table floor, exactly how the below-floor vacuum gap is already handled. Enabled via
   args in the coupled test, which defines the #192 configuration.

**Trade-off: dropped Ohmic heat.** The energy-consistent write-back conserves
gas-internal energy but DROPS the net resistively-dissipated field energy instead of
depositing it as eta*J^2 heat: the ADR-0009 RKL2 composite superstep exposes only the
end-to-end bphi update, not the transport/dissipation split. The omission is one-signed
(loses energy — stabilising), visible as the negative etot drift in the fixed probe. The
faithful deposit (to electrons, ADR-0003; the non-split path's `OhmicEnergyFlux` at
`src/mhd/mhd_tasks.cpp:507` is the conservative reference, itself currently gated on
ideal EOS) is follow-up #193.

**Non-findings (recorded so they are not re-investigated).**

- **Radiation is innocent.** fld+mrad alone is bounded and gentle (peak rho 15.0); the #192
  radiative-collapse hypothesis is dead.
- **RKL2 stability was never the problem.** "Feed the parabolic dt into NewTimeStep" is not
  needed: the super-time-stepper is unconditionally stable and converged; the runaway was
  energy/mass bookkeeping around it, not in it.

**Consequences.** The pre-fix coupled-stack results at reduced grid — including the
celebrated "75x growth / 1.58 mm" amplitude — are energy-pump contaminated and must not be
defended; the honest coupled growth number is re-derived from the post-fix green run (the
#120 gate's job). The guard is opt-in per run; decks whose outer boundary is meant to feed
mass (none today) simply leave it off.
