#include "tests.h"
#include "computational_domain.h"

#include <deal.II/grid/grid_generator.h>

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  ComputationalDomain<3> domain;
  ParameterAcceptor::initialize();
  Triangulation<3> volume;
  GridGenerator::hyper_cube(volume, -1., 1.);
  GridGenerator::extract_boundary_mesh(volume, domain.tria);
  for (const auto &cell : domain.tria.active_cell_iterators())
    cell->set_material_id(2);
  auto selected = domain.tria.begin_active();
  ++selected;
  selected->set_material_id(1);
  const auto focus = selected->center();
  auto cut = domain.tria.begin();
  for (unsigned int step = 0; step < 3; ++step)
    {
      cut->set_refine_flag(RefinementCase<2>::cut_x);
      domain.tria.execute_coarsening_and_refinement();
      cut = cut->child(0);
    }
  const auto count_unselected = [&]() {
    unsigned int count = 0;
    for (const auto &cell : domain.tria.active_cell_iterators())
      if (cell->material_id() == 2)
        ++count;
    return count;
  };
  const auto before = count_unselected();
  const auto flags = domain.tria.get_mesh_smoothing();
  domain.max_element_aspect_ratio = 1000.;
  domain.graded_refinement_material_ids = {1};
  domain.graded_refinement_focus = focus;
  domain.graded_refinement_size_ratio = 0.2;
  domain.graded_refinement_minimum_size = 0.1;
  domain.graded_refinement_maximum_cycles = 1;
  domain.refine_and_resize(0);
  domain.update_triangulation();
  AssertThrow(count_unselected() == before,
              ExcMessage("Local grading refined unrelated anisotropic materials."));
  AssertThrow(domain.tria.get_mesh_smoothing() == flags,
              ExcMessage("Local grading leaked temporary mesh-smoothing flags."));
  domain.graded_refinement_maximum_cycles = 0;
  domain.exponential_refinement_decay_vector = {0., 0., 1.};
  domain.exponential_refinement_minimum_size = 10.;
  domain.exponential_refinement_maximum_cycles = 1;
  domain.refine_and_resize(0);
  AssertThrow(domain.tria.get_mesh_smoothing() == flags,
              ExcMessage("Exponential grading leaked temporary smoothing flags."));
  deallog << "Local grading preserves anisotropic locality and smoothing flags: OK"
          << std::endl;
}
