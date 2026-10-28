// The Helmholtz kernel exp(i k r) / (4 pi r), k = 2, in the mixed Galerkin
// problem with Dirichlet and impedance parts: V, K and W of the real and of
// the imaginary part of the kernel couple the parts; Calderon
// preconditioner (helmholtz_galerkin_robin of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_helmholtz_robin.prm",
                    solver_tests::unrefined_in_debug(
                      {{"Boundary integral formulation", "galerkin"},
                       {"Linear solver", "CG"},
                       {"Preconditioner", "BlockJacobi"},
                       {"Mixed Galerkin preconditioner", "calderon"}}),
                    {"mixed_flux_real",
                     "mixed_potential_real",
                     "mixed_flux",
                     "mixed_potential",
                     "mixed_robin_flux",
                     "iterations"});
}
