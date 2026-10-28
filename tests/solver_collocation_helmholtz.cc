// The Helmholtz kernel exp(i k r) / (4 pi r), k = 2, in the collocation
// problem with Dirichlet and impedance parts: the errors of the plane wave
// (helmholtz_collocation_robin of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_helmholtz_robin.prm",
                    solver_tests::unrefined_in_debug({}),
                    {"phi_nodes", "flux", "gradient", "steps"});
}
