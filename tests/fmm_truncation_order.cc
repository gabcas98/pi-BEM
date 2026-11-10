// The truncation order of the multipole method that grows with the mesh
// (BEMFMA::mesh_order()), on the sphere of the campaign of the report: the
// cube sphere with 96 cells refined twice to six times. Degree 2 needed
// order 8 with 1536 cells, 10 with 6144 and 12 with 24576; degree 1 order 8
// with 98304 cells, the mean cell size of 24576 halved.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "bem_fma.h"
#include "tests.h"

int
main()
{
  initlog();

  Triangulation<2, 3> tria;
  GridGenerator::hyper_sphere(tria);
  tria.refine_global(2);

  // With the defaults (order 6, from the mesh) and without the mesh.
  const BEMFMA<3> fma;

  for (unsigned int level = 0; level <= 4; ++level)
    {
      if (level > 0)
        tria.refine_global(1);
      const auto corners =
        GridTools::compute_bounding_box(tria).get_boundary_points();
      const double diameter = corners.first.distance(corners.second);
      const double cell_size =
        std::sqrt(GridTools::volume(tria) / tria.n_active_cells());
      deallog << tria.n_active_cells() << " cells, D/h "
              << static_cast<int>(diameter / cell_size + 0.5)
              << ": degree 1 order "
              << BEMFMA<3>::mesh_order(diameter, cell_size, 1)
              << ", degree 2 order "
              << BEMFMA<3>::mesh_order(diameter, cell_size, 2)
              << ", with the default order "
              << fma.truncation_order(diameter, cell_size, 2) << std::endl;
      if (level == 4)
        deallog << 4 * tria.n_active_cells() << " cells: degree 1 order "
                << BEMFMA<3>::mesh_order(diameter, cell_size / 2, 1)
                << std::endl;
    }

  // Degenerate input: no mesh, or a single cell as large as the surface.
  deallog << "degenerate " << BEMFMA<3>::mesh_order(0., 0., 2) << ' '
          << BEMFMA<3>::mesh_order(1., 1., 2) << std::endl;
}
