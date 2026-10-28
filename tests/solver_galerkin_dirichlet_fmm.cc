// The Dirichlet problem in the Galerkin formulation, discontinuous p = 1 on
// the sphere in a box, with the multipole method: the load vector, the
// errors of the flux and of the recovered gradient, and the ratio to the
// best approximation (dg_laplace_p1_fmm of the regression suite). The steps
// of CG are left out: the stopping test falls between two steps, so that
// the multipole products of another compiler take one more (70 and 71).

#include "solver_tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  solver_tests::run("g_p1_bal0_weak.prm",
                    {{"Preconditioner", "BlockJacobi"},
                     {"Solution method", "FMA"},
                     {"FMA Truncation Order", "10"}},
                    {"rhs_norm", "flux", "gradient", "best_ratio"});
}
