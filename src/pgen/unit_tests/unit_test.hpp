#ifndef PGEN_UNIT_TESTS_UNIT_TEST_HPP_
#define PGEN_UNIT_TESTS_UNIT_TEST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file unit_test.hpp
//  \brief Shared assert/report helpers for pgen-based unit tests.
//
// A pgen-based unit test is an ordinary problem generator built with
//   -D PROBLEM=unit_tests/<name>
// Because it runs inside a fully-initialized AthenaK (Kokkos device backend live,
// MeshBlockPack constructed) it exercises the real device kernels rather than a host-only
// mock -- the thing we actually need to test.  These helpers standardize the bookkeeping:
// tally pass/fail checks, print a clear per-check failure report, and exit nonzero so the
// test is CI-friendly.  The Python wrapper in tst/test_suite/unit_tests/ builds the pgen
// in an isolated build directory, runs it, and asserts on the (zero) exit code.
//
// Usage inside ProblemGenerator::UserProblem():
//   #include "pgen/unit_tests/unit_test.hpp"
//   ...
//   unit_test::UnitTest test("my_test");
//   // ... run device kernels, compute results on the host ...
//   test.CheckTrue(some_condition, "condition should hold");
//   test.CheckNear(value, expected, 1.0e-12, 1.0e-14, "value matches analytic");
//   test.Finish();   // prints the summary; exits nonzero if any check failed
//
// See sample_unit_test.cpp for a complete, runnable template.

#include <cstdlib>    // exit, EXIT_FAILURE
#include <iomanip>    // setprecision
#include <iostream>
#include <string>

#include "athena.hpp"   // Real, Kokkos

namespace unit_test {

//----------------------------------------------------------------------------------------
//! \fn bool ApproxEqual
//! \brief Mixed absolute/relative floating-point comparison: |a-b| <= atol + rtol*|b|.
//! Declared KOKKOS_INLINE_FUNCTION so it can be used both on the host (for CheckNear
//! below) and inside device kernels when reducing per-cell pass/fail flags.
KOKKOS_INLINE_FUNCTION
bool ApproxEqual(Real a, Real b, Real rtol, Real atol) {
  return Kokkos::fabs(a - b) <= (atol + rtol*Kokkos::fabs(b));
}

//----------------------------------------------------------------------------------------
//! \class UnitTest
//! \brief Host-side pass/fail tally + reporting for a single pgen-based unit test.
//!
//! Construct one at the top of UserProblem, record checks as the test runs, then call
//! Finish() exactly once at the end.  A failing check is reported on a single line; if
//! any check failed, Finish() prints "TEST FAILED" and exits with a nonzero code so the
//! test runner registers a failure.

class UnitTest {
 public:
  explicit UnitTest(const std::string &name) : name_(name) {}

  //! \brief Record a boolean check; prints a one-line failure report when it fails.
  void CheckTrue(bool passed, const std::string &what) {
    ++n_checks_;
    if (passed) {
      ++n_passed_;
    } else {
      std::cout << "  [FAIL] " << what << std::endl;
    }
  }

  //! \brief Record that |actual-expected| is within tolerance (see ApproxEqual).
  //! Reports both values and the achieved difference on failure.
  void CheckNear(Real actual, Real expected, Real rtol, Real atol,
                 const std::string &what) {
    bool passed = ApproxEqual(actual, expected, rtol, atol);
    ++n_checks_;
    if (passed) {
      ++n_passed_;
    } else {
      std::cout << std::setprecision(17)
                << "  [FAIL] " << what << ": got " << actual
                << ", expected " << expected
                << " (|diff|=" << Kokkos::fabs(actual - expected)
                << ", rtol=" << rtol << ", atol=" << atol << ")" << std::endl;
    }
  }

  //! \brief Print the summary and exit nonzero if any check failed.  Call once, last.
  void Finish() {
    int n_failed = n_checks_ - n_passed_;
    std::cout << "[" << name_ << "] " << n_passed_ << "/" << n_checks_
              << " checks passed." << std::endl;
    if (n_failed > 0) {
      std::cout << "[" << name_ << "] TEST FAILED (" << n_failed
                << " check(s) failed)." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::cout << "[" << name_ << "] TEST PASSED." << std::endl;
  }

  int num_checks() const { return n_checks_; }
  int num_passed() const { return n_passed_; }

 private:
  std::string name_;
  int n_checks_ = 0;
  int n_passed_ = 0;
};

}  // namespace unit_test

#endif  // PGEN_UNIT_TESTS_UNIT_TEST_HPP_
