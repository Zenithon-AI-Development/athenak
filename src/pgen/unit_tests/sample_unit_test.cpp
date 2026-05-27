//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file sample_unit_test.cpp
//  \brief Sample pgen-based unit test and copy-paste template for unit_test.hpp.
//
// Demonstrates the full pattern: run a real Kokkos device kernel (here a parallel_reduce
// over a RangePolicy on DevExeSpace, so it runs on whichever backend AthenaK was built
// with -- SERIAL on CPU, CUDA on GPU), bring the result back to the host, and check it
// against a closed-form value with the unit_test::UnitTest helpers.  Build/run with
//   cmake -D PROBLEM=unit_tests/sample_unit_test -B build_unit
//   (cd build_unit/src && make) && ./build_unit/src/athena \
//       -i inputs/unit_tests/sample_unit_test.athinput
// Auto-run by tst/test_suite/unit_tests/test_unit_sample_cpu.py.

#include <iostream>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "pgen/unit_tests/unit_test.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sample unit test exercising a device reduction + the tolerance helpers.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  unit_test::UnitTest test("sample_unit_test");

  // Number of elements to reduce over (overridable from the <problem> block).
  const int n = pin->GetOrAddInteger("problem", "n", 10000);

  // (1) Device kernel: sum_{i=0}^{n-1} i, exercising a real Kokkos parallel_reduce.
  Real sum = 0.0;
  Kokkos::parallel_reduce("sample_unit_test_sum",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
    KOKKOS_LAMBDA(const int i, Real &partial) {
      partial += static_cast<Real>(i);
    }, sum);

  // Closed-form result (integers up to ~1e7 are exactly representable in double).
  const Real expected = 0.5*static_cast<Real>(n)*static_cast<Real>(n - 1);
  test.CheckNear(sum, expected, 1.0e-12, 0.0,
                 "device parallel_reduce sum equals N(N-1)/2");

  // (2) Host-side tolerance checks demonstrating the helper API.
  const Real root2 = Kokkos::sqrt(static_cast<Real>(2.0));
  test.CheckNear(root2*root2, 2.0, 1.0e-12, 1.0e-14, "sqrt(2)^2 recovers 2");
  test.CheckTrue(unit_test::ApproxEqual(1.0/3.0, 0.3333333333333333, 1.0e-12, 0.0),
                 "1/3 matches its decimal expansion");

  // Print the summary and exit nonzero if any check failed.
  test.Finish();
  return;
}
