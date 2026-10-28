// Two components with the same operators in the mixed Galerkin problem,
// dense matrices: the second reuses the operators of the first; the errors
// of the second (components_mixed_dense of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_components_mixed.prm",
                    {},
                    {"mixed_flux", "mixed_potential", "iterations"});
}
