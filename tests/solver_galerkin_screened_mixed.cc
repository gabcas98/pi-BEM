// The screened kernel in the mixed Galerkin problem, with Dirichlet and
// Robin parts: the weak hypersingular operator with its term
// kappa^2 (n_x . n_y) G, dense, with the Calderon preconditioner
// (screened_mixed_robin of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run(
    "sphere_screened_robin.prm",
    solver_tests::unrefined_in_debug(
      {{"Boundary integral formulation", "galerkin"},
       {"Linear solver", "CG"},
       {"Preconditioner", "BlockJacobi"},
       {"Solution method", "Direct"},
       {"Mixed Galerkin preconditioner", "calderon"}}),
    {"mixed_flux", "mixed_potential", "mixed_robin_flux", "iterations"});
}
