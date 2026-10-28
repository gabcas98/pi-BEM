// The complex problem in the collocation formulation, real and imaginary
// part as two blocks with the matrices, and the errors of both components
// against the exact solution (complex_collocation_errors of the regression
// suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_complex_galerkin.prm",
                    solver_tests::unrefined_in_debug(
                      {{"Errors", "exact solution"}}),
                    {"phi_nodes", "flux", "gradient", "steps"});
}
