// The screened kernel in the collocation problem with Dirichlet and Robin
// parts, through the screened multipole method
// (screened_collocation_robin_fma of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_screened_robin.prm",
                    solver_tests::unrefined_in_debug({}),
                    {"phi_nodes", "robin_residual", "iterations"});
}
