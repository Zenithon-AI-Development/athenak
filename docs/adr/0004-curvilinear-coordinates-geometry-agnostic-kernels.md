# Cylindrical (and future spherical) coordinates via geometry-agnostic kernels with 1D-in-r metric

**Context.** AthenaK is uniform-Cartesian: geometry is one scalar `dx1/dx2/dx3` per block
(`RegionSize`), with no `CellVolume`/`FaceArea` API. Z-pinch/MagLIF is cylindrical (the circuit drives
`B_φ` at outer `r`; the `B_φ²/r` hoop stress is the pinch). Athena++ (AthenaK's ancestor) has a proven
cylindrical implementation; FLASH does cylindrical 3T rad-MHD for MagLIF. We port their *software*
approach rather than reinvent it, preserve AthenaK's Kokkos/GPU scaling, stay backwards compatible
(Cartesian regression unchanged), and keep the door open to spherical.

**Decision.**

*Performance-preserving core (Athena++ pattern).* Store cylindrical geometry as **precomputed 1D device
arrays in the radial index `i`** (cell-volume factor `½(r_p²−r_m²)`, radial/axial face-area factors,
azimuthal edge length `r·dφ`, the source coefficients `coord_src1_i = dx1f/dV`, `coord_src2_i`),
multiplied by trivial `dx2·dx3` in-kernel. No 3D geometry storage, no per-cell transcendentals.

*Geometry-agnostic kernels.* Make the flux divergence, CT, diffusion operators, and `newdt`
coordinate-agnostic — they call inline `KOKKOS_INLINE_FUNCTION` `CellVolume`/`Face?Area`/`Edge?Length`.
A coordinate-system enum (input `<coord> system = cartesian|cylindrical`, default `cartesian`) selects
behavior; the **Cartesian specialization returns constants so the compiler folds to today's `/dx`
arithmetic — byte-identical, zero overhead.** Reuse the existing GR insertion pattern (rsolvers already
plumb `size`/`coord`; a `CoordSrcTerms` hook exists). FV update becomes
`du/dt = −(1/V)[A_{i+½}F_{i+½} − A_{i−½}F_{i−½}] + S_geo`.

*MHD geometric source terms (Athena++ / Skinner-Ostriker).* Radial:
`S_r = <1/r>(ρv_φ² + p + ½(B_r² − B_φ² + B_z²))` (centrifugal + pressure + **−B_φ² hoop stress**).
Azimuthal: the **re-symmetrized face-r-weighted X1-flux of ρv_φ** (`coord_src2·(r_m F_i + r_p F_{i+1})`).
Evolve `ρv_φ` (not `r·ρv_φ`); this discretization conserves angular momentum to machine precision.

*Resistive `B_φ` is the one true exception (FLASH lesson).* It carries a real `−ηB_φ/r²` curl-curl term;
implement it via the conservative `(1/r)∂_r(r·flux)` form with the FLASH `2/(r_i²−r_{i-1}|r_{i-1}|)`
stencil and an **antisymmetric axis ghost `B_φ(i−1)=−B_φ(i)`** — not naive scalar diffusion. Carry `B_φ`
and angular momentum in conservative area-weighted form in cylindrical resistive runs.

*Diffusion operators stay geometry-agnostic (FLASH lesson).* FLD and conduction pass cell-centered
diffusion coefficients; the `r`-weighting enters only through the shared `Area/Volume` factors — the same
machinery as the hyperbolic divergence. This is precisely ADR-0001's `ParabolicOperator` consuming the
coordinate metric.

*Axis `r=0`.* `A_r(0)=0` zeroes radial flux naturally (natural Neumann for diffusion); use a
reflecting/axisymmetric BC with `v_r`,`B_r` sign-flip and `B_φ` antisymmetric ghost; `r_min>0` is the
simplest start. Near-axis `r·dφ` CFL is handled automatically by `CenterWidth2 = x1v·dx2`.

*Deferred / sequencing.* Curvilinear reconstruction corrections (Mignone 2014: radial PLM limiter, PPM
`h±=3±Δr/2r`, volume-centroid `x1v`) are **deferred** — start with cell-centered reconstruction. AMR
needs **Balsara** prolongation on curvilinear faces. Ship **2D (r,z) first** (FLASH production MagLIF is
2D r-z; 3D cylindrical CTU is hard/blocked there); 3D (r,φ,z) later with a φ-wedge / reduced-φ. The
`Area/Volume/Edge/CoordSrc` interface generalizes to **spherical** as a later specialization.

**TDD / validation.**
- *Backwards compat:* snapshot `tst/run_test_suite.py` before; Cartesian path is byte-identical after →
  all existing tests pass. Add a "cylindrical-reduces-to-Cartesian along z" bit-identical guard.
- *New `tst/test_suite/cylindrical/`* (auto-collected via `_cpu` suffix, no harness change): port from
  Athena++ — `mignone_radial` (rigorous reconstruction convergence), `blast_cyl` (sphericity), `magnoh`
  (MHD Z-pinch/Noh equilibrium), `field_loop` cyl (CT divB), `disk_cyl` (centrifugal); from FLASH —
  `NohCylindrical`, a cylindrical Marshak/conduction test, and the paired **2D Biermann cyl-vs-cart**
  geometry-correctness check.
- *Caveat:* `OutputErrors`/`history` L1 metric uses Cartesian cell volume — either make it volume-aware
  (`r·dr·dφ·dz`) or rely on the metric-insensitive **convergence ratio** as the primary assertion.

**Rejected.** Runtime metric arrays for all kernels (adds memory traffic to the Cartesian hot path →
regression risk); body-fitted/unstructured meshes (destroy the structured-grid GPU performance that is
the entire point of "logically Cartesian").

**Reference map.** Athena++: `src/coordinates/cylindrical.cpp`, `src/hydro/add_flux_divergence.cpp`,
`src/field/ct.cpp`, `src/reconstruct/reconstruction.cpp` (Mignone 2014), pgen `blast/magnoh/
mignone_advection/disk/field_loop`. FLASH 4.8: `Grid/GridSolvers/HYPRE/.../gr_hypreCreateMatrix1Blk.F90`,
`Grid_advanceDiffusion.F90`, `GridMain/.../gr_getCellFaceArea.F90`, `Hydro/.../unsplit/
hy_uhd_unsplitUpdate.F90` (lines ~540-975), `hy_uhd_staggeredDivb.F90`, `.../HYPRE/MHD/.../
gr_hypreCreateMatrix1BlkMag.F90` (B_φ 1/r² stencil), `GridBoundaryConditions/Grid_bcApplyToRegion.F90`
(axis/nocurrent/circuit BCs), `Simulation/SimulationMain/magnetoHD/{MagLIF,ZPinchRuiz,NohCylindrical,
2DCylindricalBiermannTest}`.
