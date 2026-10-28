// The collocation problem with Dirichlet and Robin parts through the
// multipole method: the far field of the quadrature points of all cells at
// the support points, the singular and quasi-singular rules of the matrices
// as near corrections, and the preconditioner on the near band
// (robin_real_fma of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_real_robin.prm",
                    solver_tests::unrefined_in_debug({}),
                    {"phi_nodes", "robin_residual", "iterations"});
}
