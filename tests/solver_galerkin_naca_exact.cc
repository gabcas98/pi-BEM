// The mixed problem on the NACA wing with the exact parametric geometry,
// p = 2, level 1, multipole method and Calderon preconditioner: folded
// pairs at the trailing edge, skewed cells on the tips, kinked pairs along
// their edges, the Dirichlet datum by its L2 projection
// (naca_mixed_exact_p2_fmm of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run(
    "naca_mixed.prm",
    {{"Exact geometry", "true"},
     {"Exact geometry map", "parametric"},
     {"Galerkin singular quadrature order for kinked pairs", "10"},
     {"Galerkin near singular distance ratio for kinked pairs", "1.0"}},
    {"mixed_flux", "mixed_potential", "iterations"});
}
