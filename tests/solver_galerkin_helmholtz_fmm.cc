// The Helmholtz kernel through the multipole method in the mixed Galerkin
// problem with Dirichlet and impedance parts, on the unrefined sphere: the
// complex rows, the near field of both parts of the kernel and the
// Calderon preconditioner of the real part (helmholtz_galerkin_robin_fma of
// the regression suite, coarser). Release only.

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_helmholtz_robin.prm",
                    {{"Boundary integral formulation", "galerkin"},
                     {"Linear solver", "CG"},
                     {"Preconditioner", "BlockJacobi"},
                     {"Mixed Galerkin preconditioner", "calderon"},
                     {"Solution method", "FMA"},
                     {"FMA Truncation Order", "8"},
                     {"Number of cycles", "0"}},
                    {"mixed_flux_real",
                     "mixed_potential_real",
                     "mixed_flux",
                     "mixed_potential",
                     "mixed_robin_flux",
                     "iterations"});
}
