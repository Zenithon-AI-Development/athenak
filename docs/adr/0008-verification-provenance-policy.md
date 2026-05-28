# Every verification and unit test declares an independent, citable ground-truth oracle

**Context.** The HED/MagLIF work added two test tiers on top of AthenaK's upstream astrophysical
regression suite: **pgen-based unit tests** (`tst/test_suite/unit_tests/`, which build a
`unit_tests/<name>` problem generator that self-checks and `std::exit`s nonzero on failure) and
**verification-harness tests** (`tst/test_suite/verification/` plus the cylindrical `test_verify_*`,
which run a problem and call `harness.verify` to plot and diff a committed JSON baseline). The
2026-05-28 verification audit (PRD #1) found that some of these tests assert only against a
**self-captured snapshot** of their own output. A snapshot is a regression guard, not a proof of
correctness: it locks in whatever the code produced on the day it was captured — bugs included. A
green snapshot test cannot be distinguished from one that faithfully reproduces a stale wrong answer.

**Decision.** Every verification and unit test declares a **ground-truth oracle**, classified into two
layers, and records it in the provenance index (`docs/verification.md`).

- **Layer 1 — independent oracle.** The test compares against a ground truth derived *independently of
  AthenaK's own output*: a **closed-form analytic** solution (exact Riemann solution, eigenmode decay
  rate, manufactured/equilibrium solution, Planck spectrum, conserved-quantity identity), a
  **published-literature** benchmark result, or **another code's** solution. A committed
  `harness.verify` baseline, if present, is a *regression convenience* layered on top — the binding
  assertion is the comparison to the independent oracle.
- **Layer 2 — self-captured snapshot.** The test's binding assertion is a `harness.verify` diff against
  a baseline captured from AthenaK itself (optionally plus qualitative sanity properties — a symmetry,
  a sign, a conserved quantity at round-off). This is permitted **only if** the underlying numerical
  method is itself established by a Layer-1 test, recorded in the index as that row's
  **method-correctness anchor**. A Layer-2 row with no anchor is a verification *gap* and must carry a
  `TODO` pointing at the slice that will ground it.

**"Grounded" means** a reader can name the oracle and follow the citation (Layer 1) or the
method-correctness anchor (Layer 2) to an independent correctness argument — not merely "the numbers
have not changed since we captured them."

**Docstring-header convention.** Every test module's docstring names its oracle and citation in a
stable form, so the index can be audited against (or regenerated from) the sources. Add an
`Oracle:` line:

```
"""
<one-line description of the problem the test runs>

Oracle: <Layer 1 | Layer 2> -- <analytic | literature | other-code | self-snapshot>.
<the closed form / published result that grounds it, with citation;
 for Layer 2, name the method-correctness anchor test and a TODO for the grounding slice>.
"""
```

Example — Layer 1, analytic:

```
Oracle: Layer 1 -- analytic.  Exact Sod Riemann solution at t=0.2 (Toro 2009); the L1
error over the profile must be within tolerance.  The harness.verify baseline is a
regression guard only.
```

Example — Layer 2, self-snapshot with anchor:

```
Oracle: Layer 2 -- self-snapshot.  harness.verify baseline of the advected div(B).  The
cylindrical CT method is grounded by test_unit_cyl_mhd_ct_divb (div(B)=0 to machine
precision).  TODO(#83): replace with a documented advected-loop oracle.
```

**Known gaps at adoption (Layer 2, to be closed by follow-up slices).** Four tests are snapshot-only
with no documented Layer-1 oracle and are flagged `TODO` in the index:

- planar `sod` (`test_verify_sod_cpu.py`) → exact-Riemann oracle (V4 / #79);
- `cyl_aniso_ring` (`test_verify_cyl_aniso_ring_cpu.py`) → documented ring-test oracle (V6 / #81);
- `cyl_blast` (`test_verify_cyl_blast_cpu.py`) → documented oracle (V7 / #82);
- `cyl_field_loop` (`test_verify_cyl_field_loop_cpu.py`) → documented advected-loop oracle (V8 / #83).

**Scope.** This policy governs the HED verification and unit tests under
`tst/test_suite/{verification,unit_tests,cylindrical}/`. AthenaK's pre-existing upstream regression
tiers (`nr/`, `sr/`, `gr/`, `z4c/`, `dyngrmhd/`, `rad/`, `ion-neutral/`, `sbox/`) validate the
unmodified astrophysical solvers against AthenaK's own published test problems and are out of scope
for this index.

**Rejected.** (a) Treating every committed baseline as sufficient — a snapshot proves stability, not
correctness, and silently blesses a wrong answer. (b) Deleting all snapshot tests — the
`harness.verify` regression guard is valuable layered on a Layer-1 oracle, and as a documented stopgap
for a not-yet-grounded method, provided the gap is recorded with a method-correctness anchor and a
`TODO`.
