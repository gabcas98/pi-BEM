#include "planar_mesh_quality.h"

#include <deal.II/grid/grid_generator.h>

#include "tests.h"

int
main()
{
  initlog();
  Triangulation<2, 3> tria;
  GridGenerator::subdivided_hyper_rectangle(tria,
                                            {2, 2},
                                            Point<2>(0, 0),
                                            Point<2>(2, 2));
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto v : cell->vertex_indices())
      if (cell->vertex(v).distance(Point<3>(1, 1, 0)) < 1e-12)
        {
          cell->vertex(v) = Point<3>(1.5, 1.4, 0);
          break;
        }
  const auto             old_vertices = tria.get_vertices();
  std::set<unsigned int> boundary;
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->at_boundary(f))
        for (const auto v : cell->face(f)->vertex_indices())
          boundary.insert(cell->face(f)->vertex_index(v));
  const auto result = PlanarMeshQuality::smooth(tria, {0}, 20);
  AssertThrow(result.minimum_after > 0.99 &&
                result.minimum_after > result.minimum_before,
              ExcMessage("The skew interior vertex was not repaired."));
  for (const auto v : boundary)
    AssertThrow(tria.get_vertices()[v] == old_vertices[v],
                ExcMessage("Smoothing moved the patch boundary."));
  for (const auto &point : tria.get_vertices())
    AssertThrow(point[2] == 0., ExcMessage("Smoothing left the plane."));

  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  const auto             before_hanging = tria.get_vertices();
  std::set<unsigned int> hanging_face_vertices;
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->face(f)->has_children())
        {
          const auto face = cell->face(f);
          for (const auto v : face->vertex_indices())
            hanging_face_vertices.insert(face->vertex_index(v));
          for (unsigned int c = 0; c < face->n_children(); ++c)
            for (const auto v : face->child(c)->vertex_indices())
              hanging_face_vertices.insert(face->child(c)->vertex_index(v));
        }
  AssertThrow(!hanging_face_vertices.empty(),
              ExcMessage("Missing hanging face."));
  PlanarMeshQuality::smooth(tria, {0}, 5);
  for (const auto v : hanging_face_vertices)
    AssertThrow(tria.get_vertices()[v] == before_hanging[v],
                ExcMessage("Smoothing changed a nonconforming contact."));

  // Curved material patches must fail before any vertex is moved.
  tria.begin_active()->vertex(0)[2] = 0.1;
  const auto before_curved          = tria.get_vertices();
  bool       rejected               = false;
  try
    {
      PlanarMeshQuality::smooth(tria, {0}, 5);
    }
  catch (const ExceptionBase &)
    {
      rejected = true;
    }
  AssertThrow(rejected && tria.get_vertices() == before_curved,
              ExcMessage("A curved patch was silently modified."));
  deallog << "Planar quality, fixed boundary and hanging contacts: OK"
          << std::endl;
}
