// Two slots, each exact on its manifold and wrong on the other: the
// Galerkin data of every cell from the slot of its manifold, in the mixed
// problem with the multipole method (slots_mixed_fma of the regression
// suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("sphere_slots_mixed.prm",
                    {},
                    {"mixed_flux", "mixed_potential", "iterations"});
}
