// A Robin part in the mixed Galerkin problem, with the multipole method:
// the flux -(a/b) u of the unknowns in V and K^T, its identity term in the
// potential rows (robin_galerkin_fma of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run(
    "sphere_real_robin.prm",
    solver_tests::unrefined_in_debug(
      {{"Boundary integral formulation", "galerkin"},
       {"Linear solver", "CG"},
       {"Preconditioner", "BlockJacobi"}}),
    {"mixed_flux", "mixed_potential", "mixed_robin_flux", "iterations"});
}
