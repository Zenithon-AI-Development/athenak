# The maglif_rm_anchor eta baseline is bit-stable on main; #176's byte-identical claim holds; the green-CI blocker was a flaky suite-build decode crash

**Status.** Accepted (issue #185).

**Context.** Issue #185 reported a *second* CPU regression red (separate from the
`gauss_legendre` flaky test, #178): the converging-RM B3 CPU anchor
`tst/test_suite/cylindrical/test_verify_maglif_rm_anchor_cpu.py` was said to fail its
**Layer-2 self-regression baseline guard** (`harness.verify`) with the seeded-mode amplitude
projection `eta` drifting `max rel = 9.78e-4` (> `rtol = 1e-4`; `abs = 2.6e-6`), while `R_rod`
and `R_shock` stayed bit-stable and every qualitative physics gate passed. The report flagged a
**contradiction** with #176 (P5), whose PR claimed *"every pre-#174 tabulated input … is
unchanged / byte-identical."* `maglif_rm_anchor.athinput` is a pre-#174 `eos=ideal` /
`perturbation=single_mode` input, so in exact arithmetic its IC + integration path are unchanged
by the gated tabulated/SI branches added in #167/#176/#179 — yet `eta` was reported to have
drifted. The issue asked to bisect the introducing commit and classify
**legitimate-evolution vs regression** *before* any re-baseline, explicitly warning **not to
blindly recapture** (which would mask a real regression).

**Investigation (empirical, on the A100 box).**

1. **The eta drift does not reproduce on a clean checkout of `main` (6288c17).** A fresh
   default (Serial-backend) CPU build of the `maglif` pgen, run on `maglif_rm_anchor.athinput`,
   reduces to `(R_rod, eta, R_shock)` trajectories that are **bit-identical** to the committed
   baseline (`verification/baselines/maglif_rm_anchor.json`, captured #158 / `342c286`):
   `max|Δ| = 0`, `max rel = 0` for **all three** fields. Verified three independent ways —
   two standalone build+run+reduce passes (the serial sim is run-to-run deterministic) **and**
   the real `pytest` test module (`1 passed`, the `harness.verify` guard green).
2. **#176's byte-identical claim is upheld, not violated.** The shared-path changes across
   `342c286..6288c17` that could in principle touch the ideal/`hlld` path are all either
   `is_tabulated`-guarded ternaries that select the *identical* ideal expression when
   `is_tabulated == false` (the LLF rsolver + `mhd_newdt` changes in #162; `rm_anchor` uses
   `hlld`, not `llf`, so the LLF changes never execute) or the opt-in `<time>/dt_max` ceiling
   (#176), which is `if (dt_max > 0.0) …` and defaults to `0.0` (`rm_anchor` sets no `dt_max`,
   so the branch is skipped). Hence no FP reassociation reaches the ideal/`hlld` `rm_anchor`
   run, consistent with the observed bit-identity.
3. **The actual green-CI blocker encountered was a flaky suite-build decode crash**, unrelated
   to `rm_anchor`. `run_test_suite.py --cpu` aborted in `testutils.run_command`'s
   `process.communicate()` with `UnicodeDecodeError: 'utf-8' codec can't decode bytes …` while
   reading the `make -j 12` output — **before any test ran**. Cause: parallel `make`
   interleaves child stdout/stderr and can split a multibyte char in a compiler diagnostic
   across two pipe writes, leaving an invalid UTF-8 byte sequence in the merged stream; the
   default strict decoding then raises and takes down the entire suite. This is nondeterministic
   (it depends on `-j` interleaving timing), which is consistent with the prior iteration's
   `--cpu` run completing while this one did not.

**Verdict / classification.** The `eta` reported in #185 is **neither legitimate-evolution
drift nor a regression — it is a non-event on clean `main`** (bit-identical to baseline). The
most likely origin of the prior single red is the documented box hazard (fabricated/garbled tool
output) or a one-off contaminated/partial build state, not a code change. Therefore, per the
issue's own decision tree, **no baseline recapture and no `maglif.cpp`/EOS code fix is warranted**
(recapturing would have been a no-op write that falsely implied a drift existed).

**Decision.**

1. **Do not recapture the `maglif_rm_anchor` baseline and do not change the sim code.** The
   baseline (`#158`) is exactly reproduced by current `main`; the `rtol = 1e-4` / `atol = 1e-8`
   Layer-2 guard stays as-is. (Leaving the guard tight preserves its ability to catch a *real*
   future regression — `R_rod`/`R_shock` carry the physical convergence signal and `eta` rides
   the same rod-interface contour.)
2. **Harden `testutils.run_command` against undecodable build output** — decode subprocess
   pipes with `errors="replace"`. The output is only logged (the boolean return is driven by the
   process return code), so replacing undecodable bytes is lossless and cannot mask a build
   failure; it removes a spurious green-CI blocker that aborts the suite at the build step.

**Consequences.** The CPU regression suite no longer aborts on a flaky `make`-output decode.
`maglif_rm_anchor` is confirmed green and bit-stable. #185 is resolved as
*root-caused — no regression*; this unblocks #178 (whose full-suite-green AC was gated on #185).
Future iterations: a single observed `harness.verify` red on this box should be **reproduced on
a clean build before any recapture** (ADR-0008's anti-masking policy), since the box can emit
garbled/fabricated tool output.
