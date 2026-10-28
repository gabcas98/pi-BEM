// The complex problem in the Galerkin formulation, different real and
// imaginary parts: one mixed system coupled by the complex a/b of the Robin
// part, with the multipole method, the Calderon preconditioner part by part
// and relaxed products (complex_galerkin_fma of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_complex_galerkin.prm",
                    solver_tests::unrefined_in_debug(
                      {{"Boundary integral formulation", "galerkin"},
                       {"Linear solver", "CG"},
                       {"Preconditioner", "BlockJacobi"},
                       {"Solution method", "FMA"},
                       {"Mixed Galerkin preconditioner", "calderon"},
                       {"Mixed Galerkin relaxed products below", "1e-3"}}),
                    {"mixed_flux_real",
                     "mixed_potential_real",
                     "mixed_flux",
                     "mixed_potential",
                     "mixed_robin_flux",
                     "iterations"});
}
