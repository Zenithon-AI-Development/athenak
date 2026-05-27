# Pgen-based unit-test harness

Formalizes AthenaK's `src/pgen/unit_tests/` convention into a TDD-friendly,
auto-collected harness. A unit test is an ordinary problem generator built with
`-D PROBLEM=unit_tests/<name>`: because it runs inside a fully-initialized AthenaK
(Kokkos device backend live, `MeshBlockPack` constructed) it exercises the **real device
kernels** — not a host-only mock. The C++ helpers live in
`src/pgen/unit_tests/unit_test.hpp`; the Python build/run glue is
`testutils.run_unit_test(...)`.

## Add a new unit test

1. **Write the pgen** `src/pgen/unit_tests/<name>.cpp`. In `UserProblem`, construct a
   `unit_test::UnitTest`, run your kernels, record checks, and call `Finish()`:

   ```cpp
   #include "pgen/unit_tests/unit_test.hpp"

   void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
     unit_test::UnitTest test("my_test");
     // ... run device kernels (parallel_for / parallel_reduce), reduce to host ...
     test.CheckTrue(condition, "what should hold");
     test.CheckNear(value, expected, /*rtol=*/1e-12, /*atol=*/1e-14, "value matches");
     test.Finish();  // prints a summary; exits nonzero if any check failed
   }
   ```

2. **Add a minimal input** `inputs/unit_tests/<name>.athinput`. The test logic runs in
   `UserProblem`, so use `nlim = tlim = 0` (no time integration). Exactly one physics
   `<block>` is required (`MeshBlockPack::AddPhysics`); an empty `<z4c>` block is the
   lightest choice that needs no matter ICs, EOS, or external data.

3. **Add an auto-collected wrapper** `test_unit_<name>_<device>.py` in this directory.
   The module name must contain `_cpu`, `_mpicpu`, or `_gpu` so `run_test_suite.py`
   selects it for the right device. **No edit to the runner is needed.**

   ```python
   import test_suite.testutils as testutils

   def test_run():
       assert testutils.run_unit_test("my_test"), "my_test reported a failing check"
   ```

   For a `_gpu` variant, pass the CUDA cmake flags:
   `testutils.run_unit_test("my_test", flags=["-D", "Kokkos_ENABLE_CUDA=On"])`.

## Pass / fail contract (CI-friendly)

* `Finish()` prints `[<name>] P/N checks passed.` and either `TEST PASSED` (exit 0) or
  `TEST FAILED` (`std::exit(EXIT_FAILURE)`) after reporting each failing check on its own
  `[FAIL] …` line — including the got/expected values for `CheckNear`.
* `run_unit_test` returns the binary's success (exit 0), which the wrapper asserts. A
  build/cmake failure raises instead (a broken build is an error, not a test failure).

## Build isolation

`run_unit_test` builds into `tst/build_unit/<name>/` — **separate** from the main suite's
`tst/build/` — so a unit-test `PROBLEM` never clobbers the default binary the system tests
share. Each test gets its own directory (with its `PROBLEM` fixed), so re-running an
unchanged test is an incremental no-op build, not a full reconfigure+rebuild. All of
`build_unit/` is git-ignored (`build*/`).

See `src/pgen/unit_tests/sample_unit_test.cpp` (device-reduction template) and
`gauss_legendre_test.cpp` (a real test converted to the helpers) for worked examples.
