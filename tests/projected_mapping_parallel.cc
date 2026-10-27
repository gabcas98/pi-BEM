// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the pi-BEM authors.
//
// This file is part of the pi-BEM library.
//
// The pi-BEM is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License version 2.1 as published by the Free Software Foundation.
// The full text of the license can be found in the file LICENSE at
// the top level of the pi-BEM distribution.
//
// ---------------------------------------------------------------------

// The interpolation of the exact geometry built by the processes of a
// communicator together, each projecting a share of the cells and checking
// a share of the active cells, equals the one built by a single process:
// the same support points on the cells of every level and the same largest
// deviations.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "projected_mapping.h"
#include "projected_quadrature.h"
#include "tests.h"

using namespace dealii;
using namespace ExactGeometry;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  mpi_initlog();

  // A flat mesh of the sphere with two levels.
  const Point<3>      center(0.1, -0.2, 0.3);
  const double        R = 1.3;
  Triangulation<2, 3> curved, tria;
  GridGenerator::hyper_sphere(curved, center, R);
  curved.refine_global(2);
  GridGenerator::flatten_triangulation(curved, tria);
  tria.refine_global(1);
  const SphericalProjector<2, 3> projector(center, R);
  const ProjectedMapping<2, 3>   exact(1, projector);

  const unsigned int n_processes =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  deallog << n_processes << " processes, " << tria.n_cells() << " cells on "
          << tria.n_levels() << " levels" << std::endl;
  for (const unsigned int degree : {4u, 7u})
    {
      MappingQCache<2, 3> together(degree), alone(degree);
      const auto          deviation_together =
        interpolate_exact_geometry(together, exact, tria, MPI_COMM_WORLD);
      const auto deviation_alone =
        interpolate_exact_geometry(alone, exact, tria, MPI_COMM_SELF);

      const FE_Nothing<2, 3> nothing;
      const QGauss<2>        quadrature(degree + 1);
      FEValues<2, 3> a(together, nothing, quadrature, update_quadrature_points);
      FEValues<2, 3> b(alone, nothing, quadrature, update_quadrature_points);
      double         difference = 0;
      for (const auto &cell : tria.cell_iterators())
        {
          a.reinit(cell);
          b.reinit(cell);
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            difference =
              std::max(difference,
                       a.quadrature_point(q).distance(b.quadrature_point(q)));
        }
      difference = Utilities::MPI::max(difference, MPI_COMM_WORLD);
      deallog << "degree " << degree << ": largest difference of the points "
              << difference << ", deviations "
              << (deviation_together == deviation_alone ? "equal" : "different")
              << std::endl;
    }
}
