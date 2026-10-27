#include "mapping_hanging_constraints.h"

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_tools.h>

#include "tests.h"

int
main()
{
  initlog();
  const std::vector<Point<3>> vertices = {Point<3>(0, 0, 0),
                                          Point<3>(1, 0, 0),
                                          Point<3>(2, 0, 0),
                                          Point<3>(0, 1, .3),
                                          Point<3>(1, 1, .3),
                                          Point<3>(2, 1, .3)};
  std::vector<CellData<2>>    cells(2);
  cells[0].vertices = {{0, 1, 3, 4}};
  cells[1].vertices = {{1, 2, 4, 5}};
  GridTools::consistently_order_cells(cells);
  Triangulation<2, 3> tria;
  tria.create_triangulation(vertices, cells, SubCellData());
  std::next(tria.begin_active())->set_refine_flag(RefinementCase<2>::cut_y);
  tria.execute_coarsening_and_refinement();
  // Simulate a CAD midpoint chosen at parameter .4 instead of .5.
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto v : cell->vertex_indices())
      if (std::abs(cell->vertex(v)[1] - .5) < 1e-12)
        cell->vertex(v) = Point<3>(cell->vertex(v)[0], .4, .3 * .4 * .4);
  const FE_Q<2, 3>      geometry(2);
  const MappingQ1<2, 3> linear;
  MappingQCache<2, 3>   mapping(2);
  mapping.initialize(tria, [&](const Triangulation<2, 3>::cell_iterator &cell) {
    std::vector<Point<3>> points;
    for (const auto &ref : geometry.get_unit_support_points())
      {
        Point<3> p = linear.transform_unit_to_real_cell(cell, ref);
        p[2]       = .3 * p[1] * p[1];
        points.push_back(p);
      }
    return points;
  });
  for (const unsigned int degree : {1u, 2u, 3u})
    for (const unsigned int components : {1u, 3u})
      {
        const FESystem<2, 3> fe(FE_Q<2, 3>(degree), components);
        DoFHandler<2, 3>     dh(tria);
        dh.distribute_dofs(fe);
        AffineConstraints<double> standard, corrected;
        DoFTools::make_hanging_node_constraints(dh, standard);
        standard.close();
        const auto stats =
          ExactGeometry::make_mapping_hanging_node_constraints(mapping,
                                                               dh,
                                                               corrected);
        corrected.close();
        AssertThrow(stats.corrected_lines > 0 && stats.off_edge_points == 0,
                    ExcMessage(
                      "The shifted curved hanging points were not corrected."));
        std::vector<Point<3>> points(dh.n_dofs());
        DoFTools::map_dofs_to_support_points(mapping, dh, points);
        Vector<double>                       data(dh.n_dofs());
        std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
        for (const auto &cell : dh.active_cell_iterators())
          {
            cell->get_dof_indices(indices);
            for (unsigned int i = 0; i < indices.size(); ++i)
              {
                const unsigned int d = fe.system_to_component_index(i).first;
                const auto        &p = points[indices[i]];
                data[indices[i]] =
                  (d + 1) *
                    (1 + 2 * p[1] + (degree > 1 ? 3 * p[1] * p[1] : 0.)) +
                  d * p[0];
              }
          }
        Vector<double> old(data), fixed(data), ones(dh.n_dofs());
        standard.distribute(old);
        corrected.distribute(fixed);
        old -= data;
        fixed -= data;
        AssertThrow(old.linfty_norm() > .1 && fixed.linfty_norm() < 1e-11,
                    ExcMessage(
                      "Physical polynomial traces are not continuous."));
        ones = 1.;
        corrected.distribute(ones);
        ones.add(-1.);
        AssertThrow(ones.linfty_norm() < 1e-12,
                    ExcMessage("Constant trace was changed."));
      }
  // On an affine mesh, retain the ordinary weights exactly.
  Triangulation<2, 3> flat_tria;
  flat_tria.create_triangulation(vertices, cells, SubCellData());
  std::next(flat_tria.begin_active())
    ->set_refine_flag(RefinementCase<2>::cut_y);
  flat_tria.execute_coarsening_and_refinement();
  DoFHandler<2, 3> flat_dh(flat_tria);
  const FE_Q<2, 3> scalar(2);
  flat_dh.distribute_dofs(scalar);
  AffineConstraints<double> flat;
  const auto                unchanged =
    ExactGeometry::make_mapping_hanging_node_constraints(linear, flat_dh, flat);
  AssertThrow(unchanged.corrected_lines == 0,
              ExcMessage("Affine hanging weights changed."));
  deallog
    << "Physical hanging traces, scalar/vector components and constants: OK"
    << std::endl;
}
