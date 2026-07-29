# 2-D axisymmetric (r,z) cylindrical mode with divergence-preserving (r,z) AMR

**Status.** Proposed (decision pending, #245). DRAFT — decision gate on the ADR-0004
interaction (below) must be confirmed before implementation begins.

**Context.** The faithful MagLIF benchmarks (B1 Sinars, B2 Ellison) seed an *axial*
perturbation, so the axial direction z must be resolved. AthenaK fixes the axial direction
to x3 (ADR-0004), but it has no 3-D cylindrical SMR/AMR — the issue-#38 Balsara
divergence-preserving prolongation of the face B-field is implemented only for the 2-D
(r,phi) plane (`src/mesh/prolongation.hpp:245-269`, gated in
`src/coordinates/coordinates.cpp:53-67`). Two structural guards forbid an (r,z) plane
outright:

- `src/coordinates/coordinates.cpp:61` — refuses `multilevel && three_d`
  (`three_d = nx3>1`); a refined (r,z) run has `nx3>1` and is rejected as "3-D cylindrical".
- `src/mesh/mesh.cpp:221-226` — fatally rejects `nx2==1 && nx3>1` ("2D problems in x1-x3
  plane not supported"); this blocks a uniform (r,z) grid before refinement even enters.

Consequently the faithful B1/B2 decks (`tst/inputs/maglif_b1_sinars.athinput:45-51`
RESOLUTION NOTE) carry the axisymmetric initial condition on a **uniform 3-D thin-phi-wedge**
grid (288×4×128, dr=dz=12.5 µm) — a φ-wedge that is azimuthally degenerate, wastes the φ
dimension, and (most damaging, per ADR-0015 addendum / #195) **under-resolves the stagnation
region** with a uniform grid where the converged answer demands refinement-on-stagnation
(ρ_max 2.89 uniform-paper vs 16.1 refined). AMR on an (r,z) plane is the lever that unblocks
faithful, grid-converged MagLIF without a 1024² uniform grid.

The physics machinery for (r,z) MHD already exists and is metric-correct: the cylindrical CT
update (`src/mhd/mhd_ct.cpp`) already has explicit cylindrical branches for the **B_z** face
field on the x3-face (lines 122-134, `FluxDivX1`/`CenterWidth2`) and the radial **B_r**
x1-face (lines 78-88), and the finite-volume div(B) operator
(`src/outputs/derived_variables.cpp:1079-1090`) already r-weights the **Face1Area** (radial)
and **Face3Area** (axial) faces. What is missing is purely the **AMR plumbing**: the
dimension-flag derivation, the refinement topology/restriction (hard-wired to refine
x1→x2→x3), and the divergence-preserving *internal-face* prolongation analog operating in the
(x1,x3) plane instead of (x1,x2).

**Decision.** Add a native 2-D axisymmetric **(r,z)** cylindrical mode — `nx1=r (>1)`,
`nx2=1` (ignorable azimuth φ), `nx3=z (>1)` — and an **(r,z) Balsara divergence-preserving
internal-face prolongation** (the (x1,x3)-plane analog of the existing #38 (r,phi) operator),
so SMR/AMR can refine the (r,z) plane while keeping div(B) at round-off across fine/coarse
boundaries. This is **Route A** below.

The axial direction remains x3 (ADR-0004 preserved); the new mode makes x2 (φ) the ignorable
symmetry direction for an axisymmetric (∂/∂φ = 0) run. The φ-wedge degeneracy of today's
faithful decks is removed: the (r,z) plane is the physically correct 2-D reduction of
axisymmetric MagLIF (and matches the FLASH production MagLIF setup, which is 2-D r-z).

### Options considered

**Route A — native 2-D axisymmetric (r,z), (r,z) Balsara prolongation (CHOSEN).**
Generalize the dimension flags so an `nx2==1, nx3>1` cylindrical grid is a legal 2-D mode
(remove the `mesh.cpp:221` guard for cylindrical; relax the `coordinates.cpp:61` guard to
`nx2>1 && nx3>1` true 3-D); add a 2-D (r,z) branch to `ProlongFCInternal` that does the
Toth-&-Roe / Balsara construction in flux variables on the (x1,x3) faces with radial
flux-weighting Φ₁ = r·dφ·B_r and area-invariant axial flux Φ₃ = dr·dφ·B_z; teach the
refinement topology and 2-D restriction (`src/mesh/mesh_refinement.cpp`,
`src/mesh/meshblock.cpp`) that the active second plane direction can be x3.

- *Pros.* The CT/div(B)/source machinery is already (r,z)-correct; the only new numerics is a
  small, well-understood prolongation kernel that is the index-mirror of the proven #38 (r,phi)
  kernel; matches FLASH production MagLIF and the physically correct axisymmetric reduction;
  AMR-on-stagnation is exactly what #195 says is needed for grid convergence; full-φ 3-D
  remains untouched and supported.
- *Cons.* The dimension-flag generalization is internal-API churn touching mesh, meshblock,
  bvals, mesh_refinement (every consumer that currently equates "multi_d ⇒ x2 active");
  off-by-one risk in the i↔k face-staggering swap; **no existing r-z test of any kind** to
  build on (all cylindrical tests are r-phi), so a correctness floor must be established first.

**Route B — full 3-D (r,φ,z) Balsara prolongation (the general Toth & Roe / Balsara 3-D
flux-form), used with a thin-φ wedge + AMR.** Keep today's 3-D thin-φ-wedge decks but make
them refinable by implementing the full 3-D divergence-preserving prolongation that #38
deferred.

- *Pros.* Unblocks AMR for *all* 3-D cylindrical (r,φ,z) runs at once, not just the
  axisymmetric reduction; no dimension-flag churn (the topology already supports 3-D refine).
- *Cons.* Much larger and riskier numerics: the full 3-D cylindrical flux-form Balsara
  construction (27-neighbor stencil, all three internal-edge families, all r-weighted) is
  precisely what #38/ADR-0004 deferred as "3D cylindrical CTU is hard"; it carries the φ-wedge
  waste forward (refining a degenerate azimuth); and it does **not** give the physically clean
  axisymmetric reduction MagLIF wants. Higher effort, higher risk, weaker payoff for the
  driving use case.

**Recommendation: Route A.** It is the smallest faithful path to grid-converged MagLIF: it
reuses the metric-correct (r,z) physics already in the tree, adds one prolongation kernel that
is a disciplined index-mirror of a *merged, tested* kernel (#38), and produces the physically
correct 2-D axisymmetric model. Route B's extra generality is not needed by the driving
benchmarks and front-loads the hardest deferred numerics. Route B can be revisited later if a
genuinely 3-D (non-axisymmetric, full-φ) refined run is ever required.

### ADR-0004 interaction (the decision gate)

ADR-0004 fixes (x1,x2,x3) = (r,φ,z) and states "Ship **2D (r,z) first** (FLASH production
MagLIF is 2D r-z); 3D (r,φ,z) later with a φ-wedge / reduced-φ"
(`docs/adr/0004-...md:46-47`). Route A makes x1=r, x2=1 (ignorable φ), x3=z — the axial
direction is **still x3**, so ADR-0004's coordinate convention is **preserved, not violated**.
The 2-D (r,z) mode is the very thing ADR-0004 named as the intended first deliverable; the
historical implementation order happened to land 2-D (r,φ) AMR (#38) first and left a guard
(`mesh.cpp:221`) that incidentally forbids the (r,z) plane that ADR-0004 had prioritized.

**Decision question (stated crisply).** *Does enabling a 2-D axisymmetric (r,z) cylindrical
mode require a new coordinate ADR, or is it an amendment to ADR-0004?* — Because x3 remains the
axial direction and the (x1,x2,x3)=(r,φ,z) convention is unchanged, this ADR-0016 records a
**refinement of ADR-0004, not a reversal**: it realizes ADR-0004's explicitly-deferred "2-D
(r,z) first" path and supersedes only the *incidental* `mesh.cpp:221` "x1-x3 plane not
supported" guard (which was a 3-D-AMR-era safety stop, never a coordinate-convention decision).
ADR-0004 stays in force; this ADR is the operative decision for the (r,z) axisymmetric mode and
its (r,z) AMR. No reversal of any ADR-0004 invariant is proposed.

### TDD / validation (red-first)

Two correctness floors must precede the AMR kernel, because there is **no existing r-z test
anywhere** (every cylindrical test in `tst/test_suite/cylindrical/` is r-phi):

1. **Non-AMR (r,z) cylindrical MHD is correct** (prerequisite). A uniform (r,z) toy with a
   known answer — e.g. an (r,z)-plane analog of `cyl_field_loop` (div-free loop in the (r,z)
   plane laid from an azimuthal vector potential A_φ, so B_r and B_z are the cylindrical curl)
   run to a few steps and asserting max|div(B)| stays at round-off under the existing
   cylindrical CT. This proves the (r,z) physics path (CT, div(B), sources) before any
   refinement is added. Red = the `mesh.cpp:221`/`coordinates.cpp:61` guards reject the deck;
   green once the flags admit the (r,z) mode.

2. **div(B)-clean (r,z) AMR** (the headline). The smallest analog of
   `test_verify_cyl_field_loop_amr` (`tst/test_suite/cylindrical/`,
   `tst/inputs/cyl_field_loop_amr.athinput`): a div-free (r,z) loop on an adaptive 2×2 (r,z)
   root grid, a `location` criterion refining one block, asserting (a) refinement actually
   happened (nmb > 4) and (b) max|div(B)| < 1e-10 on every history row including post-refine.
   Red = the legacy Cartesian internal-face prolongation leaves an O(B·dr/r) residual when the
   fine (r,z) block appears; green = the new (r,z) Balsara branch makes every fine cell div-free
   to round-off. This is the byte-for-byte mirror of the #38 acceptance test, swapping the
   refined plane from (r,φ) to (r,z).

3. **Unit pin (CPU).** Mirror `cyl_mhd_ct_divb_test` /
   `resb_divb_couple_test`: a small-grid unit test that seeds a div-free (r,z) coarse cell,
   calls the real `ProlongFCInternal` (r,z) branch directly, and asserts the four prolonged
   fine cells are div-free under the cylindrical FV operator to machine precision (1e-12).
   This pins the kernel independent of the full run and catches the i↔k staggering swap.

4. **Backwards compat.** Cartesian and existing (r,φ) AMR paths must be byte-identical (the new
   branch is selected only for `nx2==1 && nx3>1` cylindrical); the full (r,φ,z) 3-D guard stays.

**Rejected.** Continuing the uniform thin-φ-wedge workaround (under-resolves stagnation per
#195; the converged regime needs refinement, not a finer uniform grid). A φ-wedge + 3-D Balsara
(Route B) for the axisymmetric use case (front-loads the hardest deferred numerics, keeps the
azimuthal-degeneracy waste, no clean axisymmetric reduction).

**Reference map.** Existing #38 (r,phi) prolongation: `src/mesh/prolongation.hpp:245-269`
(`ProlongFCInternal` cylindrical branch) and its dispatch in
`src/mesh/mesh_refinement.cpp` (`RefineFC`, `csys`/radius plumbing). Guards:
`src/coordinates/coordinates.cpp:53-67`, `src/mesh/mesh.cpp:221-226`. Dimension flags:
`src/mesh/mesh.cpp:70-79`; refinement topology `src/mesh/meshblock.cpp:145-147,164-168`;
2-D restriction `src/mesh/mesh_refinement.cpp` (`restrictCC-2D`/`restrictFC-2D`, `two_d`,
`cks`-pinned). (r,z)-correct physics already present: `src/mhd/mhd_ct.cpp:78-88,122-134`;
`src/outputs/derived_variables.cpp:1079-1090`. Acceptance-test template:
`tst/test_suite/cylindrical/test_verify_cyl_field_loop_amr_cpu.py`,
`tst/inputs/cyl_field_loop_amr.athinput`. Athena++/FLASH reference: ADR-0004 reference map
(FLASH production MagLIF is 2-D r-z).
