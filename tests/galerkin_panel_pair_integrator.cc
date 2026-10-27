#include "galerkin_panel_pair_integrator.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include "tests.h"

namespace
{
  class LinearBoundaryFunction : public Function<3>
  {
  public:
    double
    value(const Point<3>    &point,
          const unsigned int component = 0) const override
    {
      (void)component;
      return 1.0 + 0.2 * point[0] - 0.3 * point[1] + 0.4 * point[2];
    }
  };

  double
  matrix_norm(const FullMatrix<double> &matrix)
  {
    double norm_square = 0.0;
    for (unsigned int i = 0; i < matrix.m(); ++i)
      for (unsigned int j = 0; j < matrix.n(); ++j)
        norm_square += matrix(i, j) * matrix(i, j);
    return std::sqrt(norm_square);
  }

  double
  relative_difference(const FullMatrix<double> &left,
                      const FullMatrix<double> &right,
                      const bool                transpose_right = false)
  {
    FullMatrix<double> difference(left);
    for (unsigned int i = 0; i < left.m(); ++i)
      for (unsigned int j = 0; j < left.n(); ++j)
        difference(i, j) -= transpose_right ? right(j, i) : right(i, j);
    return matrix_norm(difference) /
           std::max(matrix_norm(left), std::numeric_limits<double>::min());
  }

  FullMatrix<double>
  dense_copy(const SparseMatrix<double> &matrix)
  {
    FullMatrix<double> dense(matrix.m(), matrix.n());
    for (unsigned int i = 0; i < matrix.m(); ++i)
      for (unsigned int j = 0; j < matrix.n(); ++j)
        dense(i, j) = matrix.el(i, j);
    return dense;
  }
} // namespace


int
main()
{
  initlog();

  std::vector<Point<3>>    vertices = {Point<3>(0, 0, 0),
                                       Point<3>(1, 0, 0),
                                       Point<3>(0, 1, 0),
                                       Point<3>(1, 1, 0),
                                       Point<3>(2, 0, 0),
                                       Point<3>(2, 1, 0),
                                       Point<3>(1, 2, 0),
                                       Point<3>(2, 2, 0),
                                       Point<3>(0, 0, 0.3),
                                       Point<3>(1, 0, 0.3),
                                       Point<3>(0, 1, 0.3),
                                       Point<3>(1, 1, 0.3)};
  std::vector<CellData<2>> cells(4);
  cells[0].vertices    = {{0, 1, 2, 3}};
  cells[1].vertices    = {{1, 4, 3, 5}};
  cells[2].vertices    = {{3, 5, 6, 7}};
  cells[3].vertices    = {{8, 9, 10, 11}};
  cells[0].material_id = 1;
  cells[1].material_id = 1;
  cells[2].material_id = 2;
  cells[3].material_id = 2;

  SubCellData subcell_data;
  GridTools::consistently_order_cells(cells);
  Triangulation<2, 3> triangulation;
  triangulation.create_triangulation(vertices, cells, subcell_data);

  FE_DGQ<2, 3>     finite_element(1);
  DoFHandler<2, 3> dof_handler(triangulation);
  dof_handler.distribute_dofs(finite_element);
  MappingQ1<2, 3>                mapping;
  GalerkinPanelPairIntegrator<3> integrator(mapping, finite_element);

  std::vector<DoFHandler<2, 3>::active_cell_iterator> active_cells;
  for (const auto &cell : dof_handler.active_cell_iterators())
    active_cells.push_back(cell);

  AssertThrow(integrator.classify(active_cells[0], active_cells[0]) ==
                GalerkinPanelPairType::same_panel,
              ExcInternalError());
  AssertThrow(integrator.classify(active_cells[0], active_cells[1]) ==
                GalerkinPanelPairType::common_edge,
              ExcInternalError());
  AssertThrow(integrator.classify(active_cells[0], active_cells[2]) ==
                GalerkinPanelPairType::common_vertex,
              ExcInternalError());
  AssertThrow(integrator.classify(active_cells[0], active_cells[3]) ==
                GalerkinPanelPairType::regular,
              ExcInternalError());

  const FullMatrix<double> same_order_5 =
    integrator.integrate_single_layer(active_cells[0], active_cells[0], 5);
  const FullMatrix<double> same_order_7 =
    integrator.integrate_single_layer(active_cells[0], active_cells[0], 7);
  const FullMatrix<double> same_order_9 =
    integrator.integrate_single_layer(active_cells[0], active_cells[0], 9);
  AssertThrow(relative_difference(same_order_7, same_order_9) <
                relative_difference(same_order_5, same_order_9),
              ExcMessage("Same-panel Sauter quadrature is not converging."));
  AssertThrow(relative_difference(same_order_7, same_order_9) < 1e-7,
              ExcMessage("Same-panel Sauter quadrature is not accurate."));
  AssertThrow(relative_difference(same_order_9, same_order_9, true) < 1e-13,
              ExcMessage("Same-panel single-layer block is not symmetric."));

  Vector<double> coefficients(finite_element.dofs_per_cell);
  coefficients[0] = 1.0;
  coefficients[1] = -0.3;
  coefficients[2] = 0.2;
  coefficients[3] = 0.7;
  Vector<double> product(coefficients.size());
  same_order_9.vmult(product, coefficients);
  AssertThrow(coefficients * product > 0.0,
              ExcMessage("Same-panel single-layer block is not positive."));

  GalerkinPanelPairIntegrator<3> duffy_integrator(
    mapping, finite_element, GalerkinSingularQuadrature::iterated_duffy);
  const FullMatrix<double> duffy_same_order_7 =
    duffy_integrator.integrate_single_layer(active_cells[0],
                                            active_cells[0],
                                            7);
  const double duffy_same_difference_7 =
    relative_difference(duffy_same_order_7, same_order_9);
  AssertThrow(duffy_same_difference_7 < 1e-3,
              ExcMessage("Iterated Duffy same-panel quadrature is not "
                         "accurate; relative difference = " +
                         std::to_string(duffy_same_difference_7)));
  AssertThrow(relative_difference(duffy_same_order_7,
                                  duffy_same_order_7,
                                  true) < 1e-13,
              ExcMessage("Iterated Duffy same-panel block is not symmetric."));

  const FullMatrix<double> duffy_edge_order_7 =
    duffy_integrator.integrate_single_layer(active_cells[0],
                                            active_cells[1],
                                            7);
  const FullMatrix<double> reference_edge_order_9 =
    integrator.integrate_single_layer(active_cells[0], active_cells[1], 9);
  const double duffy_edge_difference_7 =
    relative_difference(duffy_edge_order_7, reference_edge_order_9);
  AssertThrow(duffy_edge_difference_7 < 2e-3,
              ExcMessage("Projected Duffy common-edge quadrature is not "
                         "accurate; relative difference = " +
                         std::to_string(duffy_edge_difference_7)));

  for (const unsigned int other_cell : {1U, 2U, 3U})
    {
      const FullMatrix<double> forward =
        integrator.integrate_single_layer(active_cells[0],
                                          active_cells[other_cell],
                                          7);
      const FullMatrix<double> backward =
        integrator.integrate_single_layer(active_cells[other_cell],
                                          active_cells[0],
                                          7);
      AssertThrow(relative_difference(forward, backward, true) < 1e-12,
                  ExcMessage("Panel-pair single-layer blocks are not mutual "
                             "transposes."));
    }

  GalerkinPanelPairIntegrator<3> cross_integrator(
    mapping,
    finite_element,
    GalerkinSingularQuadrature::iterated_duffy,
    7,
    4,
    0.5,
    2,
    5);
  const auto cross_layers =
    cross_integrator.integrate_cross_layers(active_cells[0],
                                            active_cells[3],
                                            5);
  const FullMatrix<double> separate_cross_v =
    cross_integrator.integrate_single_layer(active_cells[0],
                                            active_cells[3],
                                            5);
  const FullMatrix<double> reverse_cross_k =
    cross_integrator.integrate_double_layer(active_cells[3],
                                            active_cells[0],
                                            5);
  AssertThrow(relative_difference(cross_layers.single_layer, separate_cross_v) <
                1e-12,
              ExcMessage("Combined cross-layer V differs from the separate "
                         "single-layer integration."));
  AssertThrow(relative_difference(cross_layers.adjoint_double_layer,
                                  reverse_cross_k,
                                  true) < 1e-12,
              ExcMessage("Combined cross-layer K' differs from the transpose "
                         "of the reverse double-layer integration."));

  Vector<double> constant_coefficients(finite_element.dofs_per_cell);
  constant_coefficients = 1.0;
  class CoordinateGradient : public Function<3>
  {
  public:
    CoordinateGradient()
      : Function<3>(3)
    {}
    double
    value(const Point<3> &, const unsigned int component = 0) const override
    {
      return component == 0 ? 1.0 : 0.0;
    }
  } coordinate_gradient;
  for (const unsigned int trial_cell : {0U, 1U, 2U, 3U})
    {
      const FullMatrix<double> weak_hypersingular =
        integrator.integrate_weak_hypersingular(active_cells[0],
                                                active_cells[trial_cell],
                                                7);
      Vector<double> constant_image(finite_element.dofs_per_cell);
      weak_hypersingular.vmult(constant_image, constant_coefficients);
      Vector<double> coordinate_coefficients(finite_element.dofs_per_cell);
      for (unsigned int j = 0; j < finite_element.dofs_per_cell; ++j)
        coordinate_coefficients[j] = mapping.transform_unit_to_real_cell(
          active_cells[trial_cell], finite_element.unit_support_point(j))[0];
      Vector<double> coordinate_image(finite_element.dofs_per_cell);
      weak_hypersingular.vmult(coordinate_image, coordinate_coefficients);
      coordinate_image -= integrator.integrate_hypersingular_rhs(
        active_cells[0], active_cells[trial_cell], 7, coordinate_gradient);
      AssertThrow(coordinate_image.l2_norm() < 1e-11,
                  ExcMessage("Analytical W RHS fails coordinate patch test."));
      AssertThrow(constant_image.l2_norm() < 1e-12,
                  ExcMessage("Weak hypersingular block does not annihilate "
                             "the constant trial trace."));

      const FullMatrix<double> reverse_weak_hypersingular =
        integrator.integrate_weak_hypersingular(active_cells[trial_cell],
                                                active_cells[0],
                                                7);
      AssertThrow(relative_difference(weak_hypersingular,
                                      reverse_weak_hypersingular,
                                      true) < 1e-12,
                  ExcMessage("Weak hypersingular panel blocks are not mutual "
                             "transposes."));
    }

  const LinearBoundaryFunction linear_function;
  const FullMatrix<double>     separated_double_layer =
    integrator.integrate_double_layer(active_cells[0], active_cells[3], 7);
  Vector<double> trial_coefficients(finite_element.dofs_per_cell);
  for (unsigned int j = 0; j < finite_element.dofs_per_cell; ++j)
    trial_coefficients[j] =
      linear_function.value(mapping.transform_unit_to_real_cell(
        active_cells[3], finite_element.unit_support_point(j)));
  Vector<double> interpolated_rhs(finite_element.dofs_per_cell);
  separated_double_layer.vmult(interpolated_rhs, trial_coefficients);
  const Vector<double> analytical_rhs = integrator.integrate_double_layer(
    active_cells[0], active_cells[3], 7, linear_function);
  Vector<double> rhs_difference(analytical_rhs);
  rhs_difference -= interpolated_rhs;
  AssertThrow(rhs_difference.l2_norm() /
                  std::max(analytical_rhs.l2_norm(),
                           std::numeric_limits<double>::min()) <
                1e-12,
              ExcMessage("Analytical Galerkin double-layer RHS is "
                         "inconsistent with an exactly represented trace."));

  const Functions::ConstantFunction<3> constant_function(1.0);
  for (const unsigned int trial_cell : {0U, 1U, 2U, 3U})
    {
      const Vector<double> regularized_constant_rhs =
        integrator.integrate_double_layer(active_cells[0],
                                          active_cells[trial_cell],
                                          7,
                                          constant_function,
                                          true);
      AssertThrow(regularized_constant_rhs.l2_norm() == 0.0,
                  ExcMessage("Regularized Galerkin double-layer RHS does not "
                             "annihilate a constant trace."));
    }

  DynamicSparsityPattern filtered_dynamic_sparsity(dof_handler.n_dofs(),
                                                   dof_handler.n_dofs());
  for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < dof_handler.n_dofs(); ++j)
      filtered_dynamic_sparsity.add(i, j);
  SparsityPattern filtered_sparsity;
  filtered_sparsity.copy_from(filtered_dynamic_sparsity);
  AffineConstraints<double> no_panel_constraints;
  no_panel_constraints.close();
  SparseMatrix<double> complete_panel_matrix(filtered_sparsity);
  SparseMatrix<double> filtered_panel_matrix(filtered_sparsity);
  integrator.assemble_single_layer(dof_handler,
                                   no_panel_constraints,
                                   complete_panel_matrix,
                                   5);
  integrator.assemble_single_layer(dof_handler,
                                   no_panel_constraints,
                                   filtered_panel_matrix,
                                   5,
                                   numbers::invalid_subdomain_id,
                                   1,
                                   2);
  std::vector<types::material_id>      dof_material(dof_handler.n_dofs());
  std::vector<types::global_dof_index> material_dofs(
    finite_element.dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(material_dofs);
      for (const auto dof : material_dofs)
        dof_material[dof] = cell->material_id();
    }
  for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < dof_handler.n_dofs(); ++j)
      {
        const double expected = dof_material[i] == 1 && dof_material[j] == 2 ?
                                  complete_panel_matrix.el(i, j) :
                                  0.0;
        AssertThrow(std::abs(filtered_panel_matrix.el(i, j) - expected) < 1e-14,
                    ExcMessage("Material-filtered Galerkin assembly does not "
                               "match the corresponding complete block."));
      }

  Triangulation<2, 3> adaptive_triangulation;
  GridGenerator::subdivided_hyper_rectangle(adaptive_triangulation,
                                            std::vector<unsigned int>{2, 1},
                                            Point<2>(0.0, 0.0),
                                            Point<2>(2.0, 1.0));
  adaptive_triangulation.begin_active()->set_refine_flag();
  adaptive_triangulation.execute_coarsening_and_refinement();

  FE_Q<2, 3>       adaptive_element(1);
  DoFHandler<2, 3> adaptive_dof_handler(adaptive_triangulation);
  adaptive_dof_handler.distribute_dofs(adaptive_element);
  AffineConstraints<double> hanging_node_constraints;
  DoFTools::make_hanging_node_constraints(adaptive_dof_handler,
                                          hanging_node_constraints);
  hanging_node_constraints.close();
  AssertThrow(hanging_node_constraints.n_constraints() > 0,
              ExcMessage("The adaptive test mesh has no hanging nodes."));

  GalerkinPanelPairIntegrator<3> adaptive_integrator(mapping, adaptive_element);
  std::vector<DoFHandler<2, 3>::active_cell_iterator> adaptive_cells;
  for (const auto &cell : adaptive_dof_handler.active_cell_iterators())
    adaptive_cells.push_back(cell);

  unsigned int nonconforming_test  = numbers::invalid_unsigned_int;
  unsigned int nonconforming_trial = numbers::invalid_unsigned_int;
  for (unsigned int i = 0; i < adaptive_cells.size(); ++i)
    for (unsigned int j = i + 1; j < adaptive_cells.size(); ++j)
      if (adaptive_integrator.classify(adaptive_cells[i], adaptive_cells[j]) ==
          GalerkinPanelPairType::nonconforming_edge)
        {
          nonconforming_test  = i;
          nonconforming_trial = j;
          break;
        }
  AssertThrow(nonconforming_test != numbers::invalid_unsigned_int,
              ExcMessage("No nonconforming panel pair was detected."));

  const FullMatrix<double> nonconforming_order_4 =
    adaptive_integrator.integrate_single_layer(
      adaptive_cells[nonconforming_test],
      adaptive_cells[nonconforming_trial],
      4);
  const FullMatrix<double> nonconforming_forward =
    adaptive_integrator.integrate_single_layer(
      adaptive_cells[nonconforming_test],
      adaptive_cells[nonconforming_trial],
      6);
  const FullMatrix<double> nonconforming_order_8 =
    adaptive_integrator.integrate_single_layer(
      adaptive_cells[nonconforming_test],
      adaptive_cells[nonconforming_trial],
      8);
  const FullMatrix<double> nonconforming_backward =
    adaptive_integrator.integrate_single_layer(
      adaptive_cells[nonconforming_trial],
      adaptive_cells[nonconforming_test],
      6);
  AssertThrow(relative_difference(nonconforming_forward,
                                  nonconforming_backward,
                                  true) < 1e-11,
              ExcMessage("Nonconforming panel blocks are not reciprocal."));
  AssertThrow(
    relative_difference(nonconforming_forward, nonconforming_order_8) <
      relative_difference(nonconforming_order_4, nonconforming_order_8),
    ExcMessage("Nonconforming panel quadrature is not converging."));
  AssertThrow(relative_difference(nonconforming_forward,
                                  nonconforming_order_8) < 1e-5,
              ExcMessage("Nonconforming panel quadrature is not accurate."));

  DynamicSparsityPattern dynamic_sparsity(adaptive_dof_handler.n_dofs(),
                                          adaptive_dof_handler.n_dofs());
  for (types::global_dof_index i = 0; i < adaptive_dof_handler.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < adaptive_dof_handler.n_dofs(); ++j)
      dynamic_sparsity.add(i, j);
  SparsityPattern sparsity;
  sparsity.copy_from(dynamic_sparsity);
  SparseMatrix<double> global_single_layer(sparsity);
  adaptive_integrator.assemble_single_layer(adaptive_dof_handler,
                                            hanging_node_constraints,
                                            global_single_layer,
                                            5);

  const FullMatrix<double> dense_single_layer = dense_copy(global_single_layer);
  AssertThrow(relative_difference(dense_single_layer,
                                  dense_single_layer,
                                  true) < 1e-11,
              ExcMessage("Constrained global single-layer matrix is not "
                         "symmetric."));
  for (types::global_dof_index i = 0; i < adaptive_dof_handler.n_dofs(); ++i)
    if (hanging_node_constraints.is_constrained(i))
      for (types::global_dof_index j = 0; j < adaptive_dof_handler.n_dofs();
           ++j)
        AssertThrow(std::abs(dense_single_layer(i, j)) < 1e-14 &&
                      std::abs(dense_single_layer(j, i)) < 1e-14,
                    ExcMessage("A hanging-node row or column was not "
                               "eliminated."));

  Vector<double> adaptive_coefficients(adaptive_dof_handler.n_dofs());
  for (types::global_dof_index i = 0; i < adaptive_dof_handler.n_dofs(); ++i)
    if (!hanging_node_constraints.is_constrained(i))
      adaptive_coefficients[i] = 0.2 + 0.1 * static_cast<double>(i);
  Vector<double> adaptive_product(adaptive_coefficients.size());
  global_single_layer.vmult(adaptive_product, adaptive_coefficients);
  AssertThrow(adaptive_coefficients * adaptive_product > 0.0,
              ExcMessage("Constrained global single-layer matrix is not "
                         "positive on the unconstrained space."));

  Triangulation<3> volume_mesh;
  GridGenerator::hyper_cube(volume_mesh, -1.0, 1.0);
  Triangulation<2, 3> cube_surface;
  GridGenerator::extract_boundary_mesh(volume_mesh, cube_surface);
  FE_DGQ<2, 3>     constant_element(0);
  DoFHandler<2, 3> cube_dof_handler(cube_surface);
  cube_dof_handler.distribute_dofs(constant_element);
  AffineConstraints<double> no_constraints;
  no_constraints.close();

  DynamicSparsityPattern cube_dynamic_sparsity(cube_dof_handler.n_dofs(),
                                               cube_dof_handler.n_dofs());
  for (types::global_dof_index i = 0; i < cube_dof_handler.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < cube_dof_handler.n_dofs(); ++j)
      cube_dynamic_sparsity.add(i, j);
  SparsityPattern cube_sparsity;
  cube_sparsity.copy_from(cube_dynamic_sparsity);
  SparseMatrix<double>           double_layer(cube_sparsity);
  SparseMatrix<double>           adjoint_double_layer(cube_sparsity);
  SparseMatrix<double>           mass_matrix(cube_sparsity);
  SparseMatrix<double>           weak_hypersingular(cube_sparsity);
  GalerkinPanelPairIntegrator<3> cube_integrator(mapping, constant_element);
  cube_integrator.assemble_double_layer(cube_dof_handler,
                                        no_constraints,
                                        double_layer,
                                        7);
  cube_integrator.assemble_adjoint_double_layer(cube_dof_handler,
                                                no_constraints,
                                                adjoint_double_layer,
                                                7);
  cube_integrator.assemble_mass_matrix(cube_dof_handler,
                                       no_constraints,
                                       mass_matrix,
                                       2);
  cube_integrator.assemble_weak_hypersingular(cube_dof_handler,
                                              no_constraints,
                                              weak_hypersingular,
                                              7);
  for (types::global_dof_index i = 0; i < cube_dof_handler.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < cube_dof_handler.n_dofs(); ++j)
      AssertThrow(std::abs(adjoint_double_layer(i, j) - double_layer(j, i)) <
                    1e-12,
                  ExcMessage("The assembled K' matrix is not the transpose "
                             "of K."));

  Vector<double> constant_trace(cube_dof_handler.n_dofs());
  constant_trace = 1.0;
  Vector<double> constant_residual(cube_dof_handler.n_dofs());
  Vector<double> mass_action(cube_dof_handler.n_dofs());
  Vector<double> hypersingular_constant(cube_dof_handler.n_dofs());
  double_layer.vmult(constant_residual, constant_trace);
  mass_matrix.vmult(mass_action, constant_trace);
  weak_hypersingular.vmult(hypersingular_constant, constant_trace);
  constant_residual.add(0.5, mass_action);
  AssertThrow(constant_residual.l2_norm() / mass_action.l2_norm() < 1e-6,
              ExcMessage("The Galerkin K + 0.5 M constant-patch identity "
                         "failed."));
  AssertThrow(hypersingular_constant.l2_norm() < 1e-12,
              ExcMessage("The assembled weak hypersingular matrix does not "
                         "annihilate a constant trace."));
  for (types::global_dof_index i = 0; i < cube_dof_handler.n_dofs(); ++i)
    for (types::global_dof_index j = 0; j < cube_dof_handler.n_dofs(); ++j)
      AssertThrow(std::abs(weak_hypersingular(i, j) -
                           weak_hypersingular(j, i)) < 1e-12,
                  ExcMessage("The assembled weak hypersingular matrix is "
                             "not symmetric."));

  deallog << "Galerkin panel-pair quadrature invariants: OK" << std::endl;
}
