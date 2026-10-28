// The Helmholtz kernel through the multipole method in the collocation
// problem with Dirichlet and impedance parts, complex sources: the near
// corrections of the real part of the kernel
// (helmholtz_collocation_robin_fma of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_helmholtz_robin.prm",
                    solver_tests::unrefined_in_debug(
                      {{"Solution method", "FMA"},
                       {"FMA Truncation Order", "10"}}),
                    {"phi_nodes", "flux", "gradient", "steps"});
}
