// The screened kernel exp(-kappa r)/(4 pi r) in the Dirichlet problem,
// discontinuous p = 2, with the multipole method and the patch Schwarz
// preconditioner (dg_screened_p2_fmm of the regression suite).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("screened_p2_bal0_weak.prm",
                    {{"Preconditioner", "PatchSchwarz"},
                     {"Solution method", "FMA"},
                     {"FMA Truncation Order", "10"}},
                    {"rhs_norm", "flux", "gradient", "iterations"});
}
