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

// The Galerkin formulation of BEMProblem, apart from the rest of the class
// for its size: the dual space of the CalderonDual preconditioner, the
// assembly of the single layer and of the Calderon matrices, the Galerkin
// FMM of the Dirichlet problem, the load vector of the Dirichlet datum and
// the energy norm of the flux error. The mixed problem is in
// bem_problem_mixed.cc.

#include "../include/bem_problem.h"
#include "../include/cellwise_function.h"
#include "../include/galerkin_calderon_fmm_operator.h"
#include "../include/galerkin_panel_pair_integrator.h"
#include "../include/laplace_kernel.h"
#include "../include/screened_kernel.h"
#include "../include/quasi_singular_kernel_integral.h"
#include "../include/singular_kernel_integral.h"
#include "../include/surface_potential_evaluator.h"
#include "bem_problem_helpers.h"
#include "mapping_hanging_constraints.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

using namespace std;
using bem_problem_internal::createFiniteElementPointer;

template <int dim>
void
BEMProblem<dim>::initialize_calderon_dual_space()
{
  AssertThrow(dim == 3, ExcNotImplemented());
  const unsigned int auxiliary_order = 1;
  calderon_fe =
    createFiniteElementPointer<dim - 1, dim, false>("FEQ",
                                                     auxiliary_order);
  calderon_dh.distribute_dofs(*calderon_fe);
  DoFRenumbering::component_wise(calderon_dh);
  DoFRenumbering::subdomain_wise(calderon_dh);
  pcout << "Initializing the Calderon dual space: dofs distributed"
        << std::endl;

  calderon_cpu_set.clear();
  calderon_cpu_set.set_size(calderon_dh.n_dofs());
  std::vector<types::subdomain_id> dof_subdomains(calderon_dh.n_dofs());
  DoFTools::get_subdomain_association(calderon_dh, dof_subdomains);
  for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
    if (dof_subdomains[i] == this_mpi_process)
      calderon_cpu_set.add_index(i);
  calderon_cpu_set.compress();

  calderon_constraints.clear();
  if (exact_geometry)
    ExactGeometry::make_mapping_hanging_node_constraints(*mapping, calderon_dh, calderon_constraints);
  else
    DoFTools::make_hanging_node_constraints(calderon_dh, calderon_constraints);
  if (preconditioner_type == "CalderonDual")
    AssertThrow(calderon_constraints.n_constraints() == 0,
                ExcMessage("CalderonDual currently requires a conforming "
                           "active mesh without hanging nodes."));

  std::vector<Point<dim>> support_points(calderon_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     calderon_dh,
                                                     support_points);
  pcout << "Initializing the Calderon dual space: support points mapped"
        << std::endl;
  std::vector<types::global_dof_index> master(calderon_dh.n_dofs());
  std::iota(master.begin(), master.end(), 0);
  const auto root = [&master](types::global_dof_index index) {
    while (master[index] != index)
      index = master[index];
    return index;
  };
  constexpr double duplicate_tolerance = 1e-10;
  for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
    for (types::global_dof_index j = i + 1; j < calderon_dh.n_dofs(); ++j)
      if (support_points[i].distance(support_points[j]) <
          duplicate_tolerance)
        {
          const types::global_dof_index root_i = root(i);
          const types::global_dof_index root_j = root(j);
          if (root_i != root_j)
            {
              const types::global_dof_index canonical =
                std::min(root_i, root_j);
              master[std::max(root_i, root_j)] = canonical;
            }
        }
  std::vector<types::global_dof_index> class_root(master.size());
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    class_root[i] = root(i);

  std::vector<types::global_dof_index> class_representative(
    master.size(), numbers::invalid_dof_index);
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    if (!calderon_constraints.is_constrained(i))
      {
        auto &representative = class_representative[class_root[i]];
        if (representative == numbers::invalid_dof_index ||
            i < representative)
          representative = i;
      }
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    {
      const auto representative = class_representative[class_root[i]];
      master[i] = representative != numbers::invalid_dof_index ?
                    representative :
                    class_root[i];
    }
  pcout << "Initializing the Calderon dual space: duplicate points matched"
        << std::endl;
  unsigned int merged_vertices = 0;
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    if (master[i] != i && !calderon_constraints.is_constrained(i))
      {
        calderon_constraints.add_line(i);
        calderon_constraints.add_entry(i, master[i], 1.0);
        ++merged_vertices;
      }
  calderon_constraints.close();

  calderon_dual_cell_vertices.assign(dh.n_dofs(), {});
  calderon_dual_vertex_valence.assign(calderon_dh.n_dofs(), 0);
  calderon_dual_cell_areas.assign(dh.n_dofs(), 0.0);
  std::vector<types::global_dof_index> primary_indices(fe->dofs_per_cell);
  std::vector<types::global_dof_index> vertex_indices(
    calderon_fe->dofs_per_cell);
  const QGauss<dim - 1> area_quadrature(
    std::max(2U, mapping_degree + 1));
  FEValues<dim - 1, dim> area_values(*mapping,
                                     *calderon_fe,
                                     area_quadrature,
                                     update_JxW_values);
  auto primary_cell = dh.begin_active();
  auto vertex_cell  = calderon_dh.begin_active();
  for (; primary_cell != dh.end(); ++primary_cell, ++vertex_cell)
    {
      AssertThrow(vertex_cell != calderon_dh.end(), ExcInternalError());
      primary_cell->get_dof_indices(primary_indices);
      vertex_cell->get_dof_indices(vertex_indices);
      AssertDimension(primary_indices.size(), 1);
      const types::global_dof_index cell_dof = primary_indices[0];
      auto &cell_vertices = calderon_dual_cell_vertices[cell_dof];
      cell_vertices.reserve(vertex_indices.size());
      for (const auto vertex : vertex_indices)
        {
          const types::global_dof_index canonical_vertex = master[vertex];
          if (std::find(cell_vertices.begin(),
                        cell_vertices.end(),
                        canonical_vertex) == cell_vertices.end())
            cell_vertices.push_back(canonical_vertex);
        }
      AssertDimension(cell_vertices.size(), vertex_indices.size());
      for (const auto vertex : cell_vertices)
        ++calderon_dual_vertex_valence[vertex];

      area_values.reinit(vertex_cell);
      for (unsigned int q = 0; q < area_quadrature.size(); ++q)
        calderon_dual_cell_areas[cell_dof] += area_values.JxW(q);
      AssertThrow(calderon_dual_cell_areas[cell_dof] > 0.0,
                  ExcMessage("Non-positive CalderonDual cell area."));
    }
  AssertThrow(vertex_cell == calderon_dh.end(), ExcInternalError());

  calderon_sparsity_pattern.reinit(calderon_cpu_set, mpi_communicator);
  for (const auto i : calderon_cpu_set)
    for (types::global_dof_index j = 0; j < calderon_dh.n_dofs(); ++j)
      calderon_sparsity_pattern.add(i, j);
  calderon_sparsity_pattern.compress();
  calderon_hypersingular_matrix.reinit(calderon_sparsity_pattern);
  calderon_mass_matrix.reinit(calderon_sparsity_pattern);

  pcout << "CalderonDual auxiliary FEQ(1) space: "
        << calderon_dh.n_dofs() << " algebraic vertices, "
        << merged_vertices << " duplicate CAD vertices constrained"
        << std::endl;
}

template <int dim>
void
BEMProblem<dim>::assemble_galerkin_system()
{
  AssertThrow(boundary_integral_formulation == "galerkin",
              ExcInternalError());
  if constexpr (dim == 3)
    {
      pcout << "Assembling Galerkin boundary operators with order "
            << galerkin_quadrature_order << ", singular order "
            << galerkin_singular_quadrature_order << " using "
            << galerkin_singular_quadrature;
      if (galerkin_singular_quadrature == "iterated_duffy")
        pcout << " with " << galerkin_duffy_outer_subdivisions
              << " outer subdivisions";
      pcout << std::endl;
      dirichlet_matrix = 0.0;

      AffineConstraints<double> hanging_node_constraints;
      if (exact_geometry)
        ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, hanging_node_constraints);
      else
        DoFTools::make_hanging_node_constraints(dh, hanging_node_constraints);
      hanging_node_constraints.close();

      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_far_field_tolerance(galerkin_far_field_tolerance);
      const bool light = integrator.set_light_geometry(galerkin_light_geometry);
      integrator.set_folded_singular_order(
        galerkin_folded_singular_quadrature_order);
      integrator.set_material_singular_order(
        galerkin_singular_quadrature_material_ids,
        galerkin_material_singular_quadrature_order);
      integrator.set_skewed_singular_order(
        galerkin_skewed_singular_quadrature_order,
        galerkin_skewed_minimum_angle);
      integrator.set_kinked_singular_order(
        galerkin_kinked_singular_quadrature_order,
        galerkin_kinked_minimum_angle);
      integrator.set_kinked_distance_ratio(
        galerkin_kinked_distance_ratio);
      galerkin_rhs_from_fused_assembly =
        galerkin_fused_rhs && galerkin_fused_dirichlet_function != nullptr;
      if (galerkin_rhs_from_fused_assembly)
        {
          exact_regularized_dirichlet_rhs = 0.0;
          integrator.assemble_single_layer_and_double_layer_rhs(
            dh,
            hanging_node_constraints,
            dirichlet_matrix,
            exact_regularized_dirichlet_rhs,
            galerkin_quadrature_order,
            *galerkin_fused_dirichlet_function,
            this_mpi_process,
            dirichlet_rhs_treatment != "nodal");
        }
      else
        integrator.assemble_single_layer(dh,
                                         hanging_node_constraints,
                                         dirichlet_matrix,
                                         galerkin_quadrature_order,
                                         this_mpi_process);
      dirichlet_matrix.compress(VectorOperation::add);
      if (light)
        pcout << "Near-field rules evaluated from the reconstructed cell "
                 "geometry; largest deviation from FEValues "
              << Utilities::MPI::max(integrator.light_geometry_deviation(),
                                     mpi_communicator)
              << "; element evaluated as "
              << (integrator.tensor_shape_functions() ?
                    "a tensor Lagrange basis" :
                    "a general finite element")
              << std::endl;
      if (galerkin_far_field_tolerance > 0.0)
        {
          // Every rank takes part in every sum, whether or not it used the
          // order: a sum called on some ranks only desynchronises the
          // collective calls that follow.
          pcout << "Far-field quadrature orders used:";
          const auto &counts = integrator.far_field_order_counts();
          for (unsigned int o = 0; o < counts.size(); ++o)
            {
              const unsigned long long total =
                Utilities::MPI::sum(counts[o], mpi_communicator);
              if (total > 0)
                pcout << ' ' << o << '=' << total;
            }
          pcout << std::endl;
        }
      {
        const auto &profile = integrator.rhs_profile();
        pcout << (galerkin_rhs_from_fused_assembly ?
                    "Galerkin V and RHS" :
                    "Galerkin V")
              << " time on the slowest process: far pairs "
              << Utilities::MPI::max(profile.far_seconds, mpi_communicator)
              << " s, near and singular pairs "
              << Utilities::MPI::max(profile.near_seconds, mpi_communicator)
              << " s, of which evaluating the Dirichlet datum "
              << Utilities::MPI::max(profile.function_seconds,
                                     mpi_communicator)
              << " s" << std::endl
              << "  rule construction for uncached pairs "
              << Utilities::MPI::max(profile.setup_seconds, mpi_communicator)
              << " s (" << Utilities::MPI::sum(double(profile.uncached_pairs),
                                               mpi_communicator)
              << " pairs), kernel loops "
              << Utilities::MPI::max(profile.kernel_seconds, mpi_communicator)
              << " s" << std::endl;
        if (light)
          pcout << "  direct path: "
                << Utilities::MPI::sum(double(profile.singular_leaves),
                                       mpi_communicator)
                << " singular leaves ("
                << Utilities::MPI::sum(double(profile.singular_points),
                                       mpi_communicator)
                << " points) "
                << Utilities::MPI::max(profile.singular_seconds,
                                       mpi_communicator)
                << " s, "
                << Utilities::MPI::sum(double(profile.regular_leaves),
                                       mpi_communicator)
                << " regular leaves "
                << Utilities::MPI::max(profile.regular_seconds,
                                       mpi_communicator)
                << " s" << std::endl
                << "  cell geometry at "
                << Utilities::MPI::sum(double(profile.geometry_points),
                                       mpi_communicator)
                << " points "
                << Utilities::MPI::max(profile.geometry_seconds,
                                       mpi_communicator)
                << " s, shape values "
                << Utilities::MPI::max(profile.shape_seconds,
                                       mpi_communicator)
                << " s" << std::endl;
      }
      pcout << "Galerkin V near-singular adaptive pairs/leaves: "
            << Utilities::MPI::sum(integrator.n_near_singular_pairs(),
                                   mpi_communicator)
            << '/'
            << Utilities::MPI::sum(integrator.n_near_singular_leaves(),
                                   mpi_communicator)
            << std::endl;
      if (preconditioner_type == "Calderon")
        {
          calderon_hypersingular_matrix = 0.0;
          calderon_mass_matrix          = 0.0;
          integrator.assemble_weak_hypersingular(
            dh,
            hanging_node_constraints,
            calderon_hypersingular_matrix,
            galerkin_quadrature_order,
            this_mpi_process);
          integrator.assemble_mass_matrix(dh,
                                          hanging_node_constraints,
                                          calderon_mass_matrix,
                                          galerkin_quadrature_order,
                                          this_mpi_process);
          calderon_hypersingular_matrix.compress(VectorOperation::add);
          calderon_mass_matrix.compress(VectorOperation::add);
          TrilinosWrappers::MPI::Vector constant_trace(this_cpu_set,
                                                        mpi_communicator);
          TrilinosWrappers::MPI::Vector constant_image(this_cpu_set,
                                                        mpi_communicator);
          for (const auto i : this_cpu_set)
            constant_trace[i] = 1.0;
          constant_trace.compress(VectorOperation::insert);
          calderon_hypersingular_matrix.vmult(constant_image,
                                              constant_trace);
          pcout << "Assembled weak hypersingular W and coupling mass M for "
                   "Calderon operator preconditioning"
                << std::endl;
          pcout << "Weak hypersingular constant-patch residual: "
                << constant_image.l2_norm() << std::endl;
        }
      else if (preconditioner_type == "CalderonDual")
        {
          calderon_hypersingular_matrix = 0.0;
          calderon_mass_matrix          = 0.0;
          GalerkinPanelPairIntegrator<3> auxiliary_integrator(
            *mapping,
            *calderon_fe,
            singular_strategy,
            galerkin_singular_quadrature_order,
            galerkin_duffy_outer_subdivisions);
          auxiliary_integrator.set_screening(galerkin_screening());
          auxiliary_integrator.set_folded_singular_order(
            galerkin_folded_singular_quadrature_order);
          auxiliary_integrator.set_material_singular_order(
            galerkin_singular_quadrature_material_ids,
            galerkin_material_singular_quadrature_order);
          auxiliary_integrator.set_skewed_singular_order(
            galerkin_skewed_singular_quadrature_order,
            galerkin_skewed_minimum_angle);
          auxiliary_integrator.set_kinked_singular_order(
            galerkin_kinked_singular_quadrature_order,
            galerkin_kinked_minimum_angle);
          auxiliary_integrator.set_kinked_distance_ratio(
            galerkin_kinked_distance_ratio);
          auxiliary_integrator.assemble_weak_hypersingular(
            calderon_dh,
            calderon_constraints,
            calderon_hypersingular_matrix,
            galerkin_quadrature_order,
            this_mpi_process);
          auxiliary_integrator.assemble_mass_matrix(
            calderon_dh,
            calderon_constraints,
            calderon_mass_matrix,
            galerkin_quadrature_order,
            this_mpi_process);
          calderon_hypersingular_matrix.compress(VectorOperation::add);
          calderon_mass_matrix.compress(VectorOperation::add);

          TrilinosWrappers::MPI::Vector constant_trace(calderon_cpu_set,
                                                        mpi_communicator);
          TrilinosWrappers::MPI::Vector constant_image(calderon_cpu_set,
                                                        mpi_communicator);
          for (const auto i : calderon_cpu_set)
            if (!calderon_constraints.is_constrained(i))
              constant_trace[i] = 1.0;
          constant_trace.compress(VectorOperation::insert);
          calderon_hypersingular_matrix.vmult(constant_image,
                                              constant_trace);
          pcout << "Assembled auxiliary weak hypersingular W and mass M for "
                   "P0/P1 CalderonDual"
                << std::endl;
          pcout << "Auxiliary W constant-patch residual: "
                << constant_image.l2_norm() << std::endl;
        }
      pcout << (galerkin_rhs_from_fused_assembly ?
                  "Done assembling Galerkin V together with the regularised "
                  "Dirichlet RHS" :
                  "Done assembling Galerkin V; the analytical Dirichlet RHS "
                  "is assembled separately")
            << std::endl;
    }
  else
    AssertThrow(false,
                ExcMessage("Galerkin assembly currently supports 3D only."));
}

template <int dim>
void
BEMProblem<dim>::setup_galerkin_fmm(const Function<dim> *dirichlet_function,
                                    const Function<dim> *exact_gradient)
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  if constexpr (dim == 3)
    {
      pcout << "Setting up the Galerkin FMM: order "
            << fma.truncation_order() << ", opening criterion "
            << fma.opening_criterion() << ", at most "
            << fma.points_per_leaf()
            << " points per leaf; quadrature order "
            << galerkin_quadrature_order << ", near field with singular order "
            << galerkin_singular_quadrature_order << " using "
            << galerkin_singular_quadrature << std::endl;
      AffineConstraints<double> hanging_node_constraints;
      if (exact_geometry)
        ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, hanging_node_constraints);
      else
        DoFTools::make_hanging_node_constraints(dh, hanging_node_constraints);
      hanging_node_constraints.close();
      // The integrator of the dense path decides which pairs need special
      // rules, and integrates them.
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_light_geometry(galerkin_light_geometry);
      integrator.set_folded_singular_order(
        galerkin_folded_singular_quadrature_order);
      integrator.set_material_singular_order(
        galerkin_singular_quadrature_material_ids,
        galerkin_material_singular_quadrature_order);
      integrator.set_skewed_singular_order(
        galerkin_skewed_singular_quadrature_order,
        galerkin_skewed_minimum_angle);
      integrator.set_kinked_singular_order(
        galerkin_kinked_singular_quadrature_order,
        galerkin_kinked_minimum_angle);
      integrator.set_kinked_distance_ratio(
        galerkin_kinked_distance_ratio);

      GalerkinFMMOperator::AdditionalData data;
      data.quadrature_order         = galerkin_quadrature_order;
      data.near_distance_ratio      = std::max(galerkin_near_singular_distance_ratio,
                                               galerkin_kinked_distance_ratio);
      data.fmm.order                = fma.truncation_order();
      data.fmm.theta                = fma.opening_criterion();
      data.fmm.max_points_per_leaf  = fma.points_per_leaf();
      // The near field of the right-hand sides is integrated together with
      // that of the operator: the double layer of the Dirichlet datum, and
      // the manufactured single layer when it is used.
      GalerkinFMMOperator::RightHandSideData rhs;
      rhs.dirichlet  = dirichlet_function;
      rhs.regularize = dirichlet_rhs_treatment != "nodal";
      if (dirichlet_rhs_treatment == "manufactured_flux")
        rhs.gradient = exact_gradient;
      // The consistent right-hand side is integrated in the traversal of
      // the single layer and the double layer.
      if (rhs.gradient != nullptr && rhs.dirichlet != nullptr)
        integrator.set_flux_function(rhs.gradient);
      galerkin_fmm = std::make_unique<GalerkinFMMOperator>(mpi_communicator);
      galerkin_fmm->reinit(dh,
                           *mapping,
                           hanging_node_constraints,
                           integrator,
                           this_cpu_set,
                           data,
                           rhs);
      galerkin_fmm->print_statistics(pcout.get_stream());
      // Where the near field spends its time, as for the dense assembly.
      const auto &profile = integrator.rhs_profile();
      pcout << "  near field on the slowest process: rule construction "
            << Utilities::MPI::max(profile.setup_seconds, mpi_communicator)
            << " s (" << Utilities::MPI::sum(double(profile.uncached_pairs),
                                             mpi_communicator)
            << " pairs), kernel loops "
            << Utilities::MPI::max(profile.kernel_seconds, mpi_communicator)
            << " s, Dirichlet datum "
            << Utilities::MPI::max(profile.function_seconds, mpi_communicator)
            << " s (" << Utilities::MPI::sum(double(profile.function_calls),
                                             mpi_communicator)
            << " evaluations)" << std::endl
            << "  singular leaves "
            << Utilities::MPI::sum(double(profile.singular_leaves),
                                   mpi_communicator)
            << " ("
            << Utilities::MPI::sum(double(profile.singular_points),
                                   mpi_communicator)
            << " points) "
            << Utilities::MPI::max(profile.singular_seconds, mpi_communicator)
            << " s, regular leaves "
            << Utilities::MPI::sum(double(profile.regular_leaves),
                                   mpi_communicator)
            << ' '
            << Utilities::MPI::max(profile.regular_seconds, mpi_communicator)
            << " s; near-singular adaptive pairs/leaves "
            << Utilities::MPI::sum(integrator.n_near_singular_pairs(),
                                   mpi_communicator)
            << '/'
            << Utilities::MPI::sum(integrator.n_near_singular_leaves(),
                                   mpi_communicator)
            << std::endl;
    }
  else
    AssertThrow(false, ExcMessage("The Galerkin FMM supports 3D only."));
}

template <int dim>
void
BEMProblem<dim>::assemble_exact_galerkin_dirichlet_rhs(
  const Function<dim> &dirichlet_function,
  const Function<dim> *exact_gradient)
{
  AssertThrow(boundary_integral_formulation == "galerkin",
              ExcInternalError());
  if constexpr (dim == 3)
    {
      // The fused assembly has already added the double-layer part.
      const bool from_fused = galerkin_rhs_from_fused_assembly;
      galerkin_rhs_from_fused_assembly = false;
      if (!from_fused)
        exact_regularized_dirichlet_rhs = 0.0;
      AffineConstraints<double> hanging_node_constraints;
      if (exact_geometry)
        ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, hanging_node_constraints);
      else
        DoFTools::make_hanging_node_constraints(dh, hanging_node_constraints);
      hanging_node_constraints.close();

      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_light_geometry(galerkin_light_geometry);
      integrator.set_folded_singular_order(
        galerkin_folded_singular_quadrature_order);
      integrator.set_material_singular_order(
        galerkin_singular_quadrature_material_ids,
        galerkin_material_singular_quadrature_order);
      integrator.set_skewed_singular_order(
        galerkin_skewed_singular_quadrature_order,
        galerkin_skewed_minimum_angle);
      integrator.set_kinked_singular_order(
        galerkin_kinked_singular_quadrature_order,
        galerkin_kinked_minimum_angle);
      integrator.set_kinked_distance_ratio(
        galerkin_kinked_distance_ratio);
      const QGauss<2> mass_quadrature(galerkin_quadrature_order);
      FEValues<2, 3> mass_values(*mapping,
                                 *fe,
                                 mass_quadrature,
                                 update_values | update_quadrature_points |
                                   update_JxW_values);
      if (galerkin_fmm)
        galerkin_fmm->double_layer_rhs(exact_regularized_dirichlet_rhs);
      else if (!from_fused)
      {
      integrator.assemble_double_layer_rhs(dh,
                                           hanging_node_constraints,
                                           exact_regularized_dirichlet_rhs,
                                           galerkin_quadrature_order,
                                           dirichlet_function,
                                           this_mpi_process,
                                           dirichlet_rhs_treatment !=
                                             "nodal");
      {
        const auto &profile = integrator.rhs_profile();
        pcout << "Galerkin RHS time on the slowest process: far pairs "
              << Utilities::MPI::max(profile.far_seconds, mpi_communicator)
              << " s, near and singular pairs "
              << Utilities::MPI::max(profile.near_seconds, mpi_communicator)
              << " s, of which evaluating the Dirichlet datum "
              << Utilities::MPI::max(profile.function_seconds,
                                     mpi_communicator)
              << " s (" << Utilities::MPI::sum(double(profile.function_calls),
                                               mpi_communicator)
              << " evaluations)" << std::endl
              << "  rule construction for uncached pairs "
              << Utilities::MPI::max(profile.setup_seconds, mpi_communicator)
              << " s (" << Utilities::MPI::sum(double(profile.uncached_pairs),
                                               mpi_communicator)
              << " pairs), kernel loops "
              << Utilities::MPI::max(profile.kernel_seconds, mpi_communicator)
              << " s" << std::endl;
      }
      pcout << "Galerkin RHS near-singular adaptive pairs/leaves: "
            << Utilities::MPI::sum(integrator.n_near_singular_pairs(),
                                   mpi_communicator)
            << '/'
            << Utilities::MPI::sum(integrator.n_near_singular_leaves(),
                                   mpi_communicator)
            << std::endl;
      }

      if (dirichlet_rhs_treatment == "nodal")
        {
          std::vector<types::global_dof_index> test_indices(
            fe->dofs_per_cell);
          for (const auto &test_cell : dh.active_cell_iterators())
            {
              if (test_cell->subdomain_id() != this_mpi_process)
                continue;

              Vector<double> local_rhs(fe->dofs_per_cell);
              mass_values.reinit(test_cell);
              for (unsigned int q = 0; q < mass_quadrature.size(); ++q)
                for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                  local_rhs[i] +=
                    0.5 * mass_values.shape_value(i, q) *
                    on_cell(dirichlet_function, test_cell)
                      .value(mass_values.quadrature_point(q)) *
                    mass_values.JxW(q);

              local_rhs *= -1.0;
              test_cell->get_dof_indices(test_indices);
              hanging_node_constraints.distribute_local_to_global(
                local_rhs, test_indices, exact_regularized_dirichlet_rhs);
            }
        }

      exact_regularized_dirichlet_rhs.compress(VectorOperation::add);
      {
        // Sixteen digits, so that two assemblies can be compared to rounding.
        // With the FMM the matrix is not formed.
        const auto   old_precision = pcout.get_stream().precision();
        const double rhs_norm      = exact_regularized_dirichlet_rhs.l2_norm();
        pcout << std::setprecision(16)
              << "Galerkin assembly check: |b| " << rhs_norm << ", |V|_F ";
        if (galerkin_fmm)
          pcout << "not formed (FMM)";
        else
          pcout << dirichlet_matrix.frobenius_norm();
        pcout << std::setprecision(old_precision) << std::endl;
      }

      // With manufactured_flux the manufactured right-hand side is assembled
      // next to the regularised one, so the two constructions are compared
      // in the same run: their difference is precisely the inconsistency
      // that breaks Galerkin orthogonality, measured rather than inferred.
      const bool compare_rhs_constructions =
        dirichlet_rhs_treatment == "manufactured_flux";
      if (exact_gradient != nullptr && compare_rhs_constructions)
        {
          TrilinosWrappers::MPI::Vector manufactured_rhs(this_cpu_set,
                                                         mpi_communicator);
          manufactured_rhs = 0.0;
          if (galerkin_fmm)
            galerkin_fmm->manufactured_rhs(manufactured_rhs);
          else
            integrator.assemble_single_layer_flux_rhs(
              dh,
              hanging_node_constraints,
              manufactured_rhs,
              galerkin_quadrature_order,
              *exact_gradient,
              this_mpi_process);
          manufactured_rhs.compress(VectorOperation::add);
          // The regularized assembly carries an overall sign, so compare the
          // two on the same footing before reporting their distance.
          TrilinosWrappers::MPI::Vector difference(this_cpu_set,
                                                   mpi_communicator);
          difference = manufactured_rhs;
          difference.add(1.0, exact_regularized_dirichlet_rhs);
          const double reference = manufactured_rhs.l2_norm();
          pcout << "Dirichlet RHS comparison: |b_manufactured| " << reference
                << ", |b_manufactured + b_regularized| " << difference.l2_norm()
                << ", relative "
                << (reference > 0.0 ? difference.l2_norm() / reference : 0.0)
                << std::endl;
          if (dirichlet_rhs_treatment == "manufactured_flux")
            {
              exact_regularized_dirichlet_rhs = manufactured_rhs;
              exact_regularized_dirichlet_rhs *= -1.0;
              pcout << "Using the manufactured single-layer Dirichlet RHS"
                    << std::endl;
            }
        }
      else
        AssertThrow(dirichlet_rhs_treatment != "manufactured_flux" ||
                      exact_gradient != nullptr,
                    ExcMessage("The manufactured Dirichlet RHS needs the "
                               "exact gradient."));


      pcout << "Assembled "
            << (dirichlet_rhs_treatment == "exact_regularized" ?
                  "regularized Galerkin Dirichlet RHS from phi(y)-phi(x)" :
                  "legacy Galerkin Dirichlet RHS from K phi and 0.5 M phi")
            << std::endl;
    }
  else
    AssertThrow(false,
                ExcMessage("Galerkin assembly currently supports 3D only."));
}

template <int dim>
double
BEMProblem<dim>::compute_energy_norm_error(
  const TrilinosWrappers::MPI::Vector &dphi_dn,
  const Function<dim>                 &exact_gradient) const
{
  AssertThrow(boundary_integral_formulation == "galerkin",
              ExcMessage("The energy norm is defined here for the Galerkin "
                         "single-layer formulation."));
  if constexpr (dim == 3)
    {
      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_light_geometry(galerkin_light_geometry);
      integrator.set_folded_singular_order(
        galerkin_folded_singular_quadrature_order);
      integrator.set_material_singular_order(
        galerkin_singular_quadrature_material_ids,
        galerkin_material_singular_quadrature_order);
      integrator.set_skewed_singular_order(
        galerkin_skewed_singular_quadrature_order,
        galerkin_skewed_minimum_angle);
      integrator.set_kinked_singular_order(
        galerkin_kinked_singular_quadrature_order,
        galerkin_kinked_minimum_angle);
      integrator.set_kinked_distance_ratio(
        galerkin_kinked_distance_ratio);
      const double kappa = galerkin_screening();

      // Per-cell quadrature data, computed once. The overwhelming majority of
      // pairs are far apart and regular, and for those the double integral is
      // a plain tensor sum over this cached data; only the singular and
      // near-singular pairs need the panel-pair machinery. This mirrors how
      // the operator assembly is organised, and is the difference between the
      // energy norm costing about one assembly and costing many.
      const QGauss<dim - 1>  quadrature(galerkin_quadrature_order);
      FEValues<dim - 1, dim> values(*mapping,
                                    *fe,
                                    quadrature,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
      const Vector<double>                 localized(dphi_dn);
      std::vector<types::global_dof_index> indices(fe->dofs_per_cell);

      std::vector<cell_it>                      cells;
      std::vector<std::vector<double>>          coefficients;
      std::vector<std::vector<Point<dim>>>      points;
      std::vector<std::vector<double>>          jxw;
      std::vector<std::vector<double>>          error_values;
      std::vector<std::vector<double>>          discrete_values;
      Vector<double>                            gradient(dim);
      for (const auto &cell : dh.active_cell_iterators())
        {
          values.reinit(cell);
          cell->get_dof_indices(indices);
          std::vector<double> local(fe->dofs_per_cell);
          for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
            local[i] = localized[indices[i]];

          std::vector<double>     cell_error(quadrature.size());
          std::vector<double>     cell_discrete(quadrature.size());
          std::vector<double>     cell_jxw(quadrature.size());
          std::vector<Point<dim>> cell_points(quadrature.size());
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              cell_points[q] = values.quadrature_point(q);
              cell_jxw[q]    = values.JxW(q);
              on_cell(exact_gradient, cell).vector_value(cell_points[q],
                                                         gradient);
              Tensor<1, dim> normal = values.normal_vector(q);
              const double   norm   = normal.norm();
              AssertThrow(norm > 1e-14,
                          ExcMessage("Cannot evaluate the exact flux with a "
                                     "zero normal."));
              normal /= norm;
              double exact = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                exact += gradient[d] * normal[d];
              double discrete = 0.0;
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                discrete += local[i] * values.shape_value(i, q);
              cell_error[q]    = exact - discrete;
              cell_discrete[q] = discrete;
            }

          cells.push_back(cell);
          coefficients.push_back(std::move(local));
          points.push_back(std::move(cell_points));
          jxw.push_back(std::move(cell_jxw));
          error_values.push_back(std::move(cell_error));
          discrete_values.push_back(std::move(cell_discrete));
        }

      // The form is symmetric, so only the upper triangle is visited and the
      // off-diagonal contributions are doubled. Summed over ranks this covers
      // every ordered pair exactly once. The field is a parameter so that the
      // same machinery can be exercised on a known quantity: with a zero
      // exact flux the result must be (V lambda_h, lambda_h), which the
      // assembled matrix gives independently.
      const unsigned int n_q = quadrature.size();
      // The three classes of pair are accumulated separately: a negative
      // total has to come from one of them, and knowing which one is the
      // difference between guessing and knowing.
      const auto energy_of =
        [&](const Function<dim>                    &field,
            const std::vector<std::vector<double>> &values_at_q,
            double                                 &diagonal,
            double                                 &far,
            double                                 &near) {
          double local_diagonal = 0.0;
          double local_far      = 0.0;
          double local_near     = 0.0;
          for (std::size_t i = 0; i < cells.size(); ++i)
            {
              if (cells[i]->subdomain_id() != this_mpi_process)
                continue;
              local_diagonal += integrator.integrate_error_energy(
                cells[i], cells[i], galerkin_quadrature_order, field,
                coefficients[i], coefficients[i]);
              for (std::size_t j = i + 1; j < cells.size(); ++j)
                {
                  if (integrator.is_far_regular_pair(cells[i], cells[j]))
                    {
                      double pair_sum = 0.0;
                      for (unsigned int qx = 0; qx < n_q; ++qx)
                        {
                          const double left =
                            values_at_q[i][qx] * jxw[i][qx];
                          if (left == 0.0)
                            continue;
                          for (unsigned int qy = 0; qy < n_q; ++qy)
                            {
                              const double distance =
                                points[i][qx].distance(points[j][qy]);
                              pair_sum +=
                                left * values_at_q[j][qy] * jxw[j][qy] *
                                (kappa == 0.0 ? 1.0 :
                                                std::exp(-kappa * distance)) /
                                (4.0 * numbers::PI * distance);
                            }
                        }
                      local_far += 2.0 * pair_sum;
                    }
                  else
                    local_near += 2.0 * integrator.integrate_error_energy(
                      cells[i], cells[j], galerkin_quadrature_order, field,
                      coefficients[i], coefficients[j]);
                }
            }
          diagonal = Utilities::MPI::sum(local_diagonal, mpi_communicator);
          far      = Utilities::MPI::sum(local_far, mpi_communicator);
          near     = Utilities::MPI::sum(local_near, mpi_communicator);
          return diagonal + far + near;
        };

      double       diagonal_part = 0.0;
      double       far_part      = 0.0;
      double       near_part     = 0.0;
      const double total         = energy_of(
        exact_gradient, error_values, diagonal_part, far_part, near_part);
      AssertThrow(total > -1e-10 * std::max(1.0, std::abs(total)),
                  ExcMessage("The energy of the flux error came out negative, "
                             "which the single-layer operator forbids."));
      return std::sqrt(std::max(total, 0.0));
    }
  else
    {
      (void)dphi_dn;
      (void)exact_gradient;
      AssertThrow(false, ExcMessage("Energy norm supports 3D only."));
      return 0.0;
    }
}


template void
BEMProblem<2>::initialize_calderon_dual_space();
template void
BEMProblem<3>::initialize_calderon_dual_space();
template void
BEMProblem<2>::assemble_galerkin_system();
template void
BEMProblem<3>::assemble_galerkin_system();
template void
BEMProblem<2>::setup_galerkin_fmm(const Function<2> *dirichlet_function,
                                    const Function<2> *exact_gradient);
template void
BEMProblem<3>::setup_galerkin_fmm(const Function<3> *dirichlet_function,
                                    const Function<3> *exact_gradient);
template void
BEMProblem<2>::assemble_exact_galerkin_dirichlet_rhs(
  const Function<2> &dirichlet_function,
  const Function<2> *exact_gradient);
template void
BEMProblem<3>::assemble_exact_galerkin_dirichlet_rhs(
  const Function<3> &dirichlet_function,
  const Function<3> *exact_gradient);
template double
BEMProblem<2>::compute_energy_norm_error(
  const TrilinosWrappers::MPI::Vector &dphi_dn,
  const Function<2>                 &exact_gradient) const;
template double
BEMProblem<3>::compute_energy_norm_error(
  const TrilinosWrappers::MPI::Vector &dphi_dn,
  const Function<3>                 &exact_gradient) const;
