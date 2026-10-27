#ifndef planar_mesh_quality_h
#define planar_mesh_quality_h

#include <deal.II/base/types.h>

#include <deal.II/grid/tria.h>

#include <set>

/** Quality optimization of planar quadrilateral patches. Boundary vertices,
 * material interfaces and the vertices of nonconforming faces stay fixed.
 * The signed corner mean ratio controls both skew and elongation. A move
 * must increase the sum of incident cell qualities without decreasing any
 * cell's minimum corner quality or reversing a cell.
 */
namespace PlanarMeshQuality
{
  struct Statistics
  {
    unsigned int sweeps         = 0;
    unsigned int accepted_moves = 0;
    double       minimum_before = 1.;
    double       minimum_after  = 1.;
    double       mean_before    = 0.;
    double       mean_after     = 0.;
  };

  Statistics
  smooth(dealii::Triangulation<2, 3>                &tria,
         const std::set<dealii::types::material_id> &material_ids,
         const unsigned int                          maximum_sweeps);
} // namespace PlanarMeshQuality

#endif
