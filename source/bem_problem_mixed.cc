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

// The mixed Galerkin problem of BEMProblem (solve_galerkin_mixed()), apart
// from the rest of the class for its size: the unknowns and their
// constraints, the datum, the operators through the FMM or dense matrices,
// the Calderon preconditioner, GMRES, the errors and the indicators of the
// adaptive refinement.

#include "../include/bem_problem.h"
#include "../include/galerkin_calderon_fmm_operator.h"
#include "../include/galerkin_panel_pair_integrator.h"
#include "../include/inexact_gmres.h"
#include "../include/mixed_calderon_preconditioner.h"
#include "mapping_hanging_constraints.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

using namespace std;

template <int dim>
void
BEMProblem<dim>::solve_galerkin_mixed(TrilinosWrappers::MPI::Vector &phi,
                                      TrilinosWrappers::MPI::Vector &dphi_dn,
                                      const Function<dim>           &potential,
                                      const Function<dim>           &gradient)
{
  if constexpr (dim == 3)
    {
      AssertThrow(solution_method == "Direct" || solution_method == "FMA",
                  ExcMessage("The mixed Galerkin problem needs Solution "
                             "method = Direct or FMA."));
      AssertThrow(galerkin_screening() == 0.0,
                  ExcMessage("The mixed Galerkin problem is implemented for "
                             "the Laplace kernel, whose weak hypersingular "
                             "operator is available."));
      AssertThrow((dynamic_cast<const FE_Q<dim - 1, dim> *>(fe.get()) !=
                   nullptr),
                  ExcMessage("The mixed Galerkin problem needs continuous "
                             "elements, Finite Element type = FEQ."));
      const auto start = std::chrono::steady_clock::now();
      const types::global_dof_index n_original = dh.n_dofs();
      const unsigned int            k          = fe->dofs_per_cell;

      AffineConstraints<double> hanging_node_constraints;
      if (exact_geometry)
        ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, hanging_node_constraints);
      else
        DoFTools::make_hanging_node_constraints(dh, hanging_node_constraints);
      hanging_node_constraints.close();

      const auto is_dirichlet_cell = [&](const auto &cell) {
        return std::find(comp_dom.dirichlet_boundary_ids.begin(),
                         comp_dom.dirichlet_boundary_ids.end(),
                         cell->material_id()) !=
               comp_dom.dirichlet_boundary_ids.end();
      };

      // A degree of freedom on cells of both parts gets a second index for
      // the Neumann cells: there the flux of the Dirichlet side and the
      // potential of the Neumann side are different functions, as at a
      // double node of the mesh, and the copy is treated as one.
      std::vector<char> original_on_dirichlet(n_original, 0),
        original_on_neumann(n_original, 0);
      std::vector<types::global_dof_index> cell_dofs(k);
      for (const auto &cell : dh.active_cell_iterators())
        {
          cell->get_dof_indices(cell_dofs);
          for (const auto i : cell_dofs)
            (is_dirichlet_cell(cell) ? original_on_dirichlet :
                                       original_on_neumann)[i] = 1;
        }
      std::vector<types::global_dof_index> copy_of(n_original,
                                                   numbers::invalid_dof_index);
      types::global_dof_index              n = n_original;
      for (types::global_dof_index i = 0; i < n_original; ++i)
        if (original_on_dirichlet[i] && original_on_neumann[i])
          copy_of[i] = n++;
      std::vector<types::global_dof_index> original(n);
      for (types::global_dof_index e = 0; e < n; ++e)
        original[e] = e;
      for (types::global_dof_index i = 0; i < n_original; ++i)
        if (copy_of[i] != numbers::invalid_dof_index)
          original[copy_of[i]] = i;
      const auto neumann_index = [&](const types::global_dof_index i) {
        return copy_of[i] != numbers::invalid_dof_index ? copy_of[i] : i;
      };
      std::vector<std::vector<types::global_dof_index>> cell_indices(
        dh.get_triangulation().n_active_cells());
      for (const auto &cell : dh.active_cell_iterators())
        {
          auto &indices = cell_indices[cell->active_cell_index()];
          indices.resize(k);
          cell->get_dof_indices(indices);
          if (!is_dirichlet_cell(cell))
            for (auto &i : indices)
              i = neumann_index(i);
        }
      const auto indices_of = [&](const auto &cell) -> const auto & {
        return cell_indices[cell->active_cell_index()];
      };
      std::vector<char> on_dirichlet(n, 0), on_neumann(n, 0);
      for (const auto &cell : dh.active_cell_iterators())
        for (const auto e : indices_of(cell))
          (is_dirichlet_cell(cell) ? on_dirichlet : on_neumann)[e] = 1;

      // The Robin part: on its cells the potential is unknown, as on the
      // Neumann part, and the flux is g/b - (a/b) u, from the values at the
      // nodes of BoundaryConditions::prepare_robin_datastructs(). A node of
      // Neumann and Robin cells is a Neumann node.
      const auto is_robin_cell = [&](const auto &cell) {
        return std::find(comp_dom.robin_boundary_ids.begin(),
                         comp_dom.robin_boundary_ids.end(),
                         cell->material_id()) !=
               comp_dom.robin_boundary_ids.end();
      };
      std::vector<char> on_neumann_cell(n, 0), on_robin_cell(n, 0);
      for (const auto &cell : dh.active_cell_iterators())
        if (!is_dirichlet_cell(cell))
          for (const auto e : indices_of(cell))
            (is_robin_cell(cell) ? on_robin_cell : on_neumann_cell)[e] = 1;

      // The hanging nodes, on each side in the indices of that side. A
      // hanging node on the interface whose coarse cell lies on the other
      // part is not hanging for its own side: on the Dirichlet side the flux
      // there is a free unknown, since the flux is continuous on the
      // Dirichlet part only; on the Neumann side the potential is fixed by
      // the Dirichlet datum, the trace of the coarse Dirichlet cell at the
      // node, with the weights of the constraint.
      AffineConstraints<double> constraints;
      std::vector<std::pair<
        types::global_dof_index,
        std::vector<std::pair<types::global_dof_index, double>>>>
                        interface_known;
      std::vector<char> forced_known(n, 0);
      // The part of the coarse cell of every hanging node: 1 Dirichlet,
      // 0 Neumann.
      std::vector<signed char> coarse_part(n_original, -1);
      {
        std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);
        for (const auto &cell : dh.active_cell_iterators())
          for (const unsigned int f : cell->face_indices())
            if (!cell->at_boundary(f) && cell->face(f)->has_children())
              for (unsigned int c = 0; c < cell->face(f)->n_children(); ++c)
                {
                  cell->face(f)->child(c)->get_dof_indices(face_dofs);
                  for (const auto i : face_dofs)
                    if (hanging_node_constraints.is_constrained(i))
                      coarse_part[i] = is_dirichlet_cell(cell) ? 1 : 0;
                }
      }
      // The degrees of freedom of the Neumann side on the interface: on the
      // edges of Neumann cells to Dirichlet cells, also where the Dirichlet
      // cells are finer and a node of the coarse edge belongs to no
      // Dirichlet cell. The potential there is fixed by the datum.
      {
        std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);
        for (const auto &cell : dh.active_cell_iterators())
          if (!is_dirichlet_cell(cell))
            for (const unsigned int f : cell->face_indices())
              {
                if (cell->at_boundary(f))
                  continue;
                bool touches = false;
                if (cell->face(f)->has_children())
                  {
                    for (unsigned int c = 0; c < cell->face(f)->n_children();
                         ++c)
                      if (is_dirichlet_cell(
                            cell->neighbor_child_on_subface(f, c)))
                        touches = true;
                  }
                else if (is_dirichlet_cell(cell->neighbor(f)))
                  touches = true;
                if (!touches)
                  continue;
                cell->face(f)->get_dof_indices(face_dofs);
                for (const auto i : face_dofs)
                  forced_known[neumann_index(i)] = 1;
              }
      }
      for (types::global_dof_index i = 0; i < n_original; ++i)
        if (hanging_node_constraints.is_constrained(i))
          for (const bool neumann_side : {false, true})
            {
              if (!(neumann_side ? original_on_neumann[i] :
                                   original_on_dirichlet[i]))
                continue;
              const types::global_dof_index e =
                neumann_side ? neumann_index(i) : i;
              const bool crosses =
                coarse_part[i] >= 0 && coarse_part[i] != (neumann_side ? 0 : 1);
              if (crosses)
                {
                  if (neumann_side)
                    {
                      interface_known.emplace_back(
                        e,
                        *hanging_node_constraints.get_constraint_entries(i));
                      forced_known[e] = 1;
                    }
                  continue;
                }
              constraints.add_line(e);
              for (const auto &[m, c] :
                   *hanging_node_constraints.get_constraint_entries(i))
                constraints.add_entry(e, neumann_side ? neumann_index(m) : m, c);
              constraints.set_inhomogeneity(
                e, hanging_node_constraints.get_inhomogeneity(i));
            }
      constraints.close();

      // The role of every index: the flux on the Dirichlet part, the
      // potential on the Neumann part. Where copies meet, at double nodes of
      // the mesh or at the copies above, the potential is continuous: equal
      // to the Dirichlet datum next to the Dirichlet part, the same on all
      // copies elsewhere; the flux may differ on the two sides.
      enum Role
      {
        flux,
        free_potential,
        tied_potential,
        known_potential,
        hanging
      };
      std::vector<Role>                    role(n);
      std::vector<types::global_dof_index> master(n);
      unsigned int                         counts[5] = {0, 0, 0, 0, 0};
      for (types::global_dof_index e = 0; e < n; ++e)
        {
          master[e] = e;
          if (on_dirichlet[e])
            role[e] = flux;
          else
            {
              bool next_to_dirichlet = false;
              for (const auto o : double_nodes_set[original[e]])
                for (const auto m : {o, copy_of[o]})
                  if (m != numbers::invalid_dof_index)
                    {
                      if (on_dirichlet[m])
                        next_to_dirichlet = true;
                      else
                        master[e] = std::min(master[e], m);
                    }
              role[e] = next_to_dirichlet || forced_known[e] ? known_potential :
                        master[e] == e ? free_potential :
                                         tied_potential;
              if (role[e] == known_potential)
                master[e] = e;
            }
          // Hanging nodes follow from the others: the matrices are condensed
          // and their values distributed at the end.
          if (constraints.is_constrained(e))
            {
              role[e]   = hanging;
              master[e] = e;
            }
          ++counts[role[e]];
        }
      pcout << "Mixed Galerkin problem: " << counts[flux]
            << " flux unknowns on the Dirichlet part, "
            << counts[free_potential] << " potential unknowns on the Neumann "
            << "part, " << counts[tied_potential]
            << " tied double nodes, " << counts[known_potential]
            << " potential values fixed by the Dirichlet datum, of which "
            << n - n_original << " on degrees of freedom of both parts, "
            << counts[hanging] << " hanging nodes" << std::endl;

      // The nodes of the Robin part, with a/b and g/b there.
      std::vector<char> robin(n, 0);
      unsigned int      n_robin_unknowns = 0;
      for (types::global_dof_index e = 0; e < n; ++e)
        if (on_robin_cell[e] && !on_neumann_cell[e] &&
            (role[e] == free_potential || role[e] == tied_potential ||
             role[e] == known_potential))
          {
            robin[e] = 1;
            n_robin_unknowns += role[e] == free_potential;
          }
      const bool     has_robin = std::find(robin.begin(), robin.end(), 1) !=
                             robin.end();
      Vector<double> robin_coefficient(n), robin_datum(n);
      if (has_robin)
        {
          AssertThrow(robin_matrix_diagonal.size() == n_original &&
                        robin_rhs.size() == n_original,
                      ExcMessage("The Robin data are not prepared."));
          const Vector<double> coefficient(robin_matrix_diagonal),
            datum(robin_rhs);
          for (types::global_dof_index e = 0; e < n; ++e)
            if (robin[e])
              {
                robin_coefficient[e] = coefficient[original[e]];
                robin_datum[e]       = datum[original[e]];
              }
          pcout << "Mixed Galerkin problem: " << n_robin_unknowns
                << " of the potential unknowns on the Robin part" << std::endl;
        }

      // The data: the Dirichlet datum interpolated at the nodes of the
      // Dirichlet part and at the copies next to it, the Neumann datum
      // projected in L2 on the Neumann part. The mass matrix of the whole
      // surface serves the identity terms.
      // With the data of the exact surface on the mesh, data and errors are
      // taken at the points, and with the normals, of the projected mapping.
      std::shared_ptr<Mapping<dim - 1, dim>> data_mapping_owner;
      const Mapping<dim - 1, dim>           *data_mapping = &*mapping;
      if (boundary_data_on_exact_surface && !exact_geometry)
        {
          data_mapping_owner = make_exact_geometry_mapping();
          data_mapping       = data_mapping_owner.get();
          pcout << "Mixed Galerkin problem: boundary data and errors at the "
                   "points of the exact surface"
                << std::endl;
        }
      std::vector<Point<dim>> original_points(n_original);
      DoFTools::map_dofs_to_support_points<dim - 1, dim>(*data_mapping,
                                                         dh,
                                                         original_points);
      const auto point = [&](const types::global_dof_index e) {
        return original_points[original[e]];
      };
      Vector<double> u_data(n), g_data(n);
      for (types::global_dof_index e = 0; e < n; ++e)
        if (role[e] == flux || role[e] == known_potential)
          u_data[e] = potential.value(point(e));
      constraints.distribute(u_data);

      DynamicSparsityPattern dsp(n, n);
      for (const auto &cell : dh.active_cell_iterators())
        constraints.add_entries_local_to_global(indices_of(cell), dsp, false);
      SparsityPattern pattern;
      pattern.copy_from(dsp);
      const QGauss<dim - 1>  quadrature(fe->degree + 3);
      FEValues<dim - 1, dim> values(*mapping,
                                    *fe,
                                    quadrature,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
      FEValues<dim - 1, dim> data_values(*data_mapping,
                                         *fe,
                                         quadrature,
                                         update_normal_vectors |
                                           update_quadrature_points);
      const bool             separate_data = data_mapping != &*mapping;
      // The mass matrix of the cells of one part, and the projection of the
      // normal trace of the gradient on them.
      const auto project_flux = [&](const bool on_dirichlet_part,
                                    SparseMatrix<double> *whole_mass,
                                    const bool potential_values = false) {
        SparseMatrix<double> part_mass(pattern);
        Vector<double>       part_rhs(n), result(n);
        FullMatrix<double>   local(k, k);
        Vector<double>       local_rhs(k);
        Vector<double>       g(dim);
        for (const auto &cell : dh.active_cell_iterators())
          {
            values.reinit(cell);
            const FEValues<dim - 1, dim> &data =
              separate_data ? (data_values.reinit(cell), data_values) : values;
            local     = 0;
            local_rhs = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                double flux_value = 0;
                if (potential_values)
                  flux_value = potential.value(data.quadrature_point(q));
                else
                  {
                    gradient.vector_value(data.quadrature_point(q), g);
                    for (unsigned int d = 0; d < dim; ++d)
                      flux_value += g[d] * data.normal_vector(q)[d];
                  }
                for (unsigned int i = 0; i < k; ++i)
                  {
                    for (unsigned int j = 0; j < k; ++j)
                      local(i, j) += values.shape_value(i, q) *
                                     values.shape_value(j, q) * values.JxW(q);
                    local_rhs(i) +=
                      values.shape_value(i, q) * flux_value * values.JxW(q);
                  }
              }
            if (whole_mass != nullptr)
              constraints.distribute_local_to_global(local,
                                                     indices_of(cell),
                                                     *whole_mass);
            if (on_dirichlet_part ?
                  is_dirichlet_cell(cell) :
                  !is_dirichlet_cell(cell) && !is_robin_cell(cell))
              constraints.distribute_local_to_global(
                local, local_rhs, indices_of(cell), part_mass, part_rhs);
          }
        for (types::global_dof_index e = 0; e < n; ++e)
          if (!(on_dirichlet_part ? on_dirichlet[e] : on_neumann_cell[e]))
            part_mass.set(e, e, 1.0);
        SolverControl control(1000, 1e-14 * part_rhs.l2_norm() + 1e-300);
        SolverCG<Vector<double>>               cg(control);
        PreconditionSSOR<SparseMatrix<double>> ssor;
        ssor.initialize(part_mass, 1.2);
        cg.solve(part_mass, result, part_rhs, ssor);
        constraints.distribute(result);
        return result;
      };
      SparseMatrix<double> mass(pattern);
      g_data = project_flux(false, &mass);
      // The Dirichlet datum on the Dirichlet part by its L2 projection
      // instead of its interpolant; the copies on the Neumann side of the
      // interface take the value of the node of the Dirichlet part at the
      // same point, so that the potential stays continuous across it.
      if (mixed_galerkin_dirichlet_datum == "projection")
        {
          const Vector<double> projected = project_flux(true, nullptr, true);
          std::map<std::array<long long, dim>, double> at_point;
          const auto key = [&](const Point<dim> &x) {
            std::array<long long, dim> k;
            for (unsigned int d = 0; d < dim; ++d)
              k[d] = std::llround(x[d] * 1e9);
            return k;
          };
          for (types::global_dof_index e = 0; e < n; ++e)
            if (role[e] == flux)
              {
                u_data[e] = projected[e];
                at_point[key(point(e))] = projected[e];
              }
          for (types::global_dof_index e = 0; e < n; ++e)
            if (role[e] == known_potential)
              {
                const auto twin = at_point.find(key(point(e)));
                if (twin != at_point.end())
                  u_data[e] = twin->second;
              }
          constraints.distribute(u_data);
          pcout << "Mixed Galerkin problem: Dirichlet datum by its L2 "
                   "projection"
                << std::endl;
        }
      // The potential at the hanging nodes of the Neumann side of the
      // interface: the datum of the coarse Dirichlet cell at the node.
      for (const auto &[e, entries] : interface_known)
        {
          double value = 0;
          for (const auto &[m, c] : entries)
            value += c * u_data[m];
          u_data[e] = value;
        }
      if (!interface_known.empty())
        pcout << "Mixed Galerkin problem: " << interface_known.size()
              << " hanging nodes of the Neumann side on the interface, fixed "
                 "by the datum"
              << std::endl;
      // The flux of the Robin part in the data: g/b, and where the potential
      // is fixed by the Dirichlet datum the whole g/b - (a/b) u. The term
      // -(a/b) u of the unknowns is in the block operator.
      if (has_robin)
        {
          for (types::global_dof_index e = 0; e < n; ++e)
            if (robin[e])
              g_data[e] = robin_datum[e] - (role[e] == known_potential ?
                                              robin_coefficient[e] * u_data[e] :
                                              0.0);
          constraints.distribute(g_data);
        }

      // The operators of the whole surface, on rows and columns of all
      // indices, each process holding the rows of its degrees of freedom and
      // of their copies.
      IndexSet owned(n);
      owned.add_indices(this_cpu_set);
      for (const auto i : this_cpu_set)
        if (copy_of[i] != numbers::invalid_dof_index)
          owned.add_index(copy_of[i]);
      owned.compress();
      // The dense single layer and its full pattern of the Dirichlet path are
      // not used here: freed before the three dense matrices of this
      // problem, and rebuilt by the next reinit().
      dirichlet_matrix.clear();
      full_sparsity_pattern.clear();
      // With Solution method = FMA the operators are applied by the fast
      // multipole method (GalerkinCalderonFMMOperator) instead of dense
      // matrices.
      const bool through_fmm = solution_method == "FMA";
      TrilinosWrappers::SparseMatrix single_layer, double_layer, hypersingular;
      if (!through_fmm)
        {
          pcout << "Mixed Galerkin problem: three dense matrices of " << n
                << " x " << n << ", about "
                << 3.0 * double(n) * double(n) * 16.0 / 1e9
                << " GB with the column indices" << std::endl;
          TrilinosWrappers::SparsityPattern full(owned, mpi_communicator);
          for (const auto e : owned)
            for (types::global_dof_index f = 0; f < n; ++f)
              full.add(e, f);
          full.compress();
          single_layer.reinit(full);
          double_layer.reinit(full);
          hypersingular.reinit(full);
        }

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
      integrator.set_far_field_tolerance(galerkin_far_field_tolerance);
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
      integrator.set_dof_index_map(
        [&](const typename DoFHandler<dim - 1, dim>::active_cell_iterator
                                                  &cell,
            std::vector<types::global_dof_index> &indices) {
          indices = indices_of(cell);
        });
      const auto assembly_start = std::chrono::steady_clock::now();
      const auto seconds_since  = [](const auto &since) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             since)
          .count();
      };
      std::unique_ptr<GalerkinCalderonFMMOperator> calderon;
      if (through_fmm)
        {
          GalerkinCalderonFMMOperator::AdditionalData data;
          data.quadrature_order        = galerkin_quadrature_order;
          data.near_distance_ratio     = std::max(galerkin_near_singular_distance_ratio,
                                              galerkin_kinked_distance_ratio);
          data.fmm.order               = fma.truncation_order();
          data.fmm.theta               = fma.opening_criterion();
          data.fmm.max_points_per_leaf = fma.points_per_leaf();
          data.dipole_order_increase   = fma.dipole_order_increase();
          data.preconditioner_order    = fma.preconditioner_order();
          data.preconditioner_theta    = fma.preconditioner_theta();
          data.preconditioner_points_per_leaf =
            fma.preconditioner_points_per_leaf();
          data.relaxed_order = fma.relaxed_order();
          data.relaxed_theta = fma.relaxed_theta();
          pcout << "Mixed Galerkin problem: V, K, K^T and W through the FMM, "
                   "order "
                << data.fmm.order << " (dipoles "
                << data.fmm.order + data.dipole_order_increase
                << "), opening criterion " << data.fmm.theta
                << ", quadrature order " << data.quadrature_order
                << std::endl;
          calderon =
            std::make_unique<GalerkinCalderonFMMOperator>(mpi_communicator);
          // The near-field blocks of the previous mesh, for adaptive
          // refinement, if the element, the mapping and the rules are the
          // same.
          if (reuse_near_field)
            {
              const std::vector<double> signature = {
                double(fe->degree),
                double(exact_geometry ? geometry_degree_in_use : mapping_degree),
                double(data.quadrature_order),
                double(galerkin_singular_quadrature_order),
                double(galerkin_folded_singular_quadrature_order),
                double(galerkin_skewed_singular_quadrature_order),
                galerkin_skewed_minimum_angle,
                double(galerkin_kinked_singular_quadrature_order),
                galerkin_kinked_minimum_angle,
                galerkin_kinked_distance_ratio,
                double(galerkin_material_singular_quadrature_order),
                data.near_distance_ratio};
              if (signature != near_field_signature)
                {
                  near_field_cache.blocks.clear();
                  near_field_signature = signature;
                }
            }
          calderon->reinit(dh,
                           *mapping,
                           cell_indices,
                           n,
                           constraints,
                           integrator,
                           data,
                           reuse_near_field ? &near_field_cache : nullptr);
          pcout << "Mixed Galerkin problem: FMM set up in "
                << seconds_since(assembly_start) << " s" << std::endl;
          calderon->print_statistics(pcout.get_stream());
        }
      else
        {
          integrator.assemble_single_double_hypersingular(
            dh,
            constraints,
            single_layer,
            double_layer,
            hypersingular,
            galerkin_quadrature_order,
            this_mpi_process);
          single_layer.compress(VectorOperation::add);
          double_layer.compress(VectorOperation::add);
          hypersingular.compress(VectorOperation::add);
        }
      if (!through_fmm)
      {
        const auto &profile = integrator.rhs_profile();
        pcout << "Mixed Galerkin problem: V, K and W assembled in "
              << seconds_since(assembly_start)
              << " s; on the slowest process far pairs "
              << Utilities::MPI::max(profile.far_seconds, mpi_communicator)
              << " s, near and singular pairs "
              << Utilities::MPI::max(profile.near_seconds, mpi_communicator)
              << " s" << std::endl
              << "  rule construction "
              << Utilities::MPI::max(profile.setup_seconds, mpi_communicator)
              << " s, of which geometry "
              << Utilities::MPI::max(profile.geometry_seconds,
                                     mpi_communicator)
              << " s and shape values "
              << Utilities::MPI::max(profile.shape_seconds, mpi_communicator)
              << " s; kernel loops "
              << Utilities::MPI::max(profile.kernel_seconds, mpi_communicator)
              << " s; "
              << Utilities::MPI::sum(double(profile.singular_leaves),
                                     mpi_communicator)
              << " singular leaves ("
              << Utilities::MPI::sum(double(profile.singular_points),
                                     mpi_communicator)
              << " points), "
              << Utilities::MPI::sum(double(profile.regular_leaves),
                                     mpi_communicator)
              << " regular leaves" << std::endl;
      }

      // Products of the distributed matrices with vectors that every process
      // holds whole.
      TrilinosWrappers::MPI::Vector in(owned, mpi_communicator),
        out(owned, mpi_communicator);
      using CalderonOperator = GalerkinCalderonFMMOperator::Operator;
      const auto product = [&](const TrilinosWrappers::SparseMatrix &matrix,
                               const Vector<double>                 &x,
                               const bool transpose) {
        if (through_fmm)
          {
            const CalderonOperator op =
              &matrix == &single_layer ? CalderonOperator::single_layer :
              &matrix == &hypersingular ?
                                         CalderonOperator::hypersingular :
              transpose ? CalderonOperator::double_layer_transpose :
                          CalderonOperator::double_layer;
            Vector<double> y(n);
            calderon->vmult(op, y, x);
            return y;
          }
        for (const auto e : owned)
          in[e] = x[e];
        in.compress(VectorOperation::insert);
        if (transpose)
          matrix.Tvmult(out, in);
        else
          matrix.vmult(out, in);
        return Vector<double>(out);
      };

      // The sign of K in the convention of the Calderon identities, with the
      // normals out of the domain: (1/2 I + K) 1 = 0.
      Vector<double> ones(n), mass_ones(n);
      ones = 1.0;
      mass.vmult(mass_ones, ones);
      const Vector<double> double_layer_ones =
        product(double_layer, ones, false);
      Vector<double> plus(mass_ones), minus(mass_ones);
      plus *= 0.5;
      minus *= 0.5;
      plus += double_layer_ones;
      minus -= double_layer_ones;
      // The rows of hanging nodes hold only the diagonal of the condensed
      // matrices.
      for (types::global_dof_index i = 0; i < n; ++i)
        if (role[i] == hanging)
          plus[i] = minus[i] = mass_ones[i] = 0;
      const double k_sign = plus.l2_norm() < minus.l2_norm() ? 1.0 : -1.0;
      // PIBEM_MIXED_DIAGNOSTICS: the checks that cost products, |W 1|, the
      // residual of the exact solution and the error of the relaxed
      // products.
      const bool diagnostics =
        std::getenv("PIBEM_MIXED_DIAGNOSTICS") != nullptr;
      pcout << "Mixed Galerkin problem: |(1/2 M + K) 1| / |1/2 M 1| = "
            << plus.l2_norm() / (0.5 * mass_ones.l2_norm())
            << ", with K of the opposite sign "
            << minus.l2_norm() / (0.5 * mass_ones.l2_norm());
      if (diagnostics)
        pcout << "; |W 1| = "
              << product(hypersingular, ones, false).l2_norm();
      pcout << std::endl;

      // The block operator on vectors of all degrees of freedom: rows and
      // columns of the flux unknowns, of the free potentials, whose rows
      // collect those of their tied copies, and the identity elsewhere.
      const auto split = [&](const Vector<double> &x,
                             Vector<double>       &t,
                             Vector<double>       &u) {
        t.reinit(n);
        u.reinit(n);
        for (types::global_dof_index i = 0; i < n; ++i)
          if (role[i] == flux)
            t[i] = x[i];
          else if (role[i] == free_potential || role[i] == tied_potential)
            u[i] = x[master[i]];
      };
      const auto combine = [&](const Vector<double> &dirichlet_rows,
                               const Vector<double> &neumann_rows,
                               const Vector<double> &identity,
                               Vector<double>       &y) {
        y.reinit(n);
        for (types::global_dof_index i = 0; i < n; ++i)
          if (role[i] == flux)
            y[i] = dirichlet_rows[i];
          else if (role[i] == free_potential || role[i] == tied_potential)
            y[master[i]] += neumann_rows[i];
        for (types::global_dof_index i = 0; i < n; ++i)
          if (role[i] == tied_potential || role[i] == known_potential ||
              role[i] == hanging)
            y[i] = identity[i];
      };
      struct BlockOperator
      {
        std::function<void(Vector<double> &, const Vector<double> &)> apply;
        void
        vmult(Vector<double> &y, const Vector<double> &x) const
        {
          apply(y, x);
        }
      } block;
      // With relaxed_rows, the products of the FMM through the coarser
      // expansions of the preconditioner (the last steps of GMRES).
      bool relaxed_rows = false;
      // On the Robin part the flux -(a/b) u of the unknowns joins the flux
      // of the Dirichlet part in V and K^T, and its identity term 1/2 M
      // goes to the potential rows.
      const auto robin_flux = [&](const Vector<double> &u) {
        Vector<double> r(n);
        for (types::global_dof_index e = 0; e < n; ++e)
          if (robin[e] &&
              (role[e] == free_potential || role[e] == tied_potential))
            r[e] = robin_coefficient[e] * u[e];
        return r;
      };
      block.apply = [&](Vector<double> &y, const Vector<double> &x) {
        Vector<double> t, u, r;
        split(x, t, u);
        if (has_robin)
          {
            r = robin_flux(u);
            t -= r;
          }
        Vector<double> dirichlet_rows, neumann_rows;
        if (through_fmm)
          calderon->vmult_rows(
            t, u, k_sign, dirichlet_rows, neumann_rows, relaxed_rows);
        else
          {
            dirichlet_rows = product(single_layer, t, false);
            dirichlet_rows.add(-k_sign, product(double_layer, u, false));
            neumann_rows = product(hypersingular, u, false);
            neumann_rows.add(k_sign, product(double_layer, t, true));
          }
        if (has_robin)
          {
            Vector<double> mass_r(n);
            mass.vmult(mass_r, r);
            neumann_rows.add(0.5, mass_r);
          }
        combine(dirichlet_rows, neumann_rows, x, y);
      };

      // The right-hand side: 1/2 M u + k K u - V g on the flux rows and
      // 1/2 M g - k K^T g - W u on the potential rows, the mass terms minus
      // the rows of the operator at (g, u), through the FMM in one product.
      Vector<double> rhs;
      {
        Vector<double> dirichlet_rows(n), neumann_rows(n), zero(n);
        mass.vmult(dirichlet_rows, u_data);
        dirichlet_rows *= 0.5;
        mass.vmult(neumann_rows, g_data);
        neumann_rows *= 0.5;
        if (through_fmm)
          {
            Vector<double> operator_dirichlet, operator_neumann;
            calderon->vmult_rows(
              g_data, u_data, k_sign, operator_dirichlet, operator_neumann);
            dirichlet_rows -= operator_dirichlet;
            neumann_rows -= operator_neumann;
          }
        else
          {
            dirichlet_rows.add(k_sign, product(double_layer, u_data, false));
            dirichlet_rows.add(-1.0, product(single_layer, g_data, false));
            neumann_rows.add(-k_sign, product(double_layer, g_data, true));
            neumann_rows.add(-1.0, product(hypersingular, u_data, false));
          }
        combine(dirichlet_rows, neumann_rows, zero, rhs);
      }

      // Diagonal preconditioner.
      Vector<double> diagonal(n);
      {
        Vector<double> v_diagonal(n), w_diagonal(n);
        if (through_fmm)
          {
            v_diagonal = calderon->diagonal(CalderonOperator::single_layer);
            w_diagonal = calderon->diagonal(CalderonOperator::hypersingular);
          }
        else
          {
            TrilinosWrappers::MPI::Vector v_local(owned, mpi_communicator),
              w_local(owned, mpi_communicator);
            for (const auto e : owned)
              {
                v_local[e] = single_layer.diag_element(e);
                w_local[e] = hypersingular.diag_element(e);
              }
            v_local.compress(VectorOperation::insert);
            w_local.compress(VectorOperation::insert);
            v_diagonal = Vector<double>(v_local);
            w_diagonal = Vector<double>(w_local);
          }
        for (types::global_dof_index i = 0; i < n; ++i)
          if (role[i] == flux)
            diagonal[i] = v_diagonal[i];
          else if (role[i] == free_potential || role[i] == tied_potential)
            diagonal[master[i]] +=
              w_diagonal[i] +
              (robin[i] ? 0.5 * robin_coefficient[i] * mass.diag_element(i) :
                          0.0);
        for (types::global_dof_index i = 0; i < n; ++i)
          if (role[i] == tied_potential || role[i] == known_potential ||
              role[i] == hanging)
            diagonal[i] = 1.0;
      }

      // The consistency of the discrete system: the residual of the exact
      // solution, the flux projected in L2 on the Dirichlet part and the
      // potential interpolated on the Neumann part.
      if (diagnostics || std::getenv("PIBEM_RESIDUAL_DUMP") != nullptr)
      {
        Vector<double> exact = project_flux(true, nullptr);
        for (types::global_dof_index e = 0; e < n; ++e)
          if (role[e] == free_potential)
            exact[e] = potential.value(point(e));
          else if (role[e] != flux)
            exact[e] = 0;
        Vector<double> residual;
        block.vmult(residual, exact);
        residual -= rhs;
        // PIBEM_RESIDUAL_DUMP=file: for every flux and free potential
        // unknown, its support point, whether it is a flux, its residual and
        // its right-hand side.
        if (const char *name = std::getenv("PIBEM_RESIDUAL_DUMP");
            name != nullptr && this_mpi_process == 0)
          {
            std::ofstream out(name);
            out << std::setprecision(10);
            for (types::global_dof_index e = 0; e < n; ++e)
              if (role[e] == flux || role[e] == free_potential)
                out << point(e) << ' ' << (role[e] == flux) << ' '
                    << residual[e] << ' ' << rhs[e] << '\n';
          }
        pcout << "Mixed Galerkin problem: relative residual of the exact "
                 "solution "
              << residual.l2_norm() / rhs.l2_norm() << std::endl;
      }

      // GMRES, every process running the same iteration on whole vectors.
      // The flux and the potential rows have different scales, about h^3
      // and h: the residual is reduced relative to its preconditioned start.
      Vector<double> x(n);
      // On a refined mesh, start from the traces of the previous one: each
      // cell takes the polynomial of its ancestor among the cells of that
      // mesh, at its support points mapped into the reference cell of the
      // ancestor.
      if (reuse_near_field && !transfer_values.empty() &&
          transfer_degree == fe->degree)
        {
          const auto &unit_points = fe->get_unit_support_points();
          unsigned int transferred = 0;
          for (const auto &cell : dh.active_cell_iterators())
            {
              const auto &cell_dofs = indices_of(cell);
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                {
                  const types::global_dof_index e = cell_dofs[i];
                  const bool flux_dof = role[e] == flux;
                  const bool potential_dof =
                    role[e] == free_potential || role[e] == tied_potential;
                  if (!flux_dof && !potential_dof)
                    continue;
                  Point<dim - 1> point    = unit_points[i];
                  auto           ancestor = cell;
                  auto found = transfer_values.find({{ancestor->level(),
                                                      ancestor->index()}});
                  while (found == transfer_values.end() &&
                         ancestor->level() > 0)
                    {
                      const auto   parent = ancestor->parent();
                      unsigned int child  = 0;
                      for (unsigned int k = 0; k < parent->n_children(); ++k)
                        if (parent->child(k) == ancestor)
                          child = k;
                      point = GeometryInfo<dim - 1>::child_to_cell_coordinates(
                        point, child, parent->refinement_case());
                      ancestor = parent;
                      found    = transfer_values.find(
                        {{ancestor->level(), ancestor->index()}});
                    }
                  if (found == transfer_values.end())
                    continue;
                  double value = 0;
                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                    value += found->second[j] * fe->shape_value(j, point);
                  x[flux_dof ? e : master[e]] = value;
                  ++transferred;
                }
            }
          pcout << "Mixed Galerkin problem: initial guess from the previous "
                   "mesh at "
                << transferred << " cell nodes" << std::endl;
        }
      // GMRES stops when the preconditioned residual falls below the given
      // fraction of the preconditioned right-hand side: from zero, its
      // reduction relative to the start.
      SolverControl control(std::max(solver_control.max_steps(), 5000u),
                            1e-300);
      const auto set_tolerance = [&](const auto &preconditioner) {
        Vector<double> preconditioned_rhs(n);
        preconditioner.vmult(preconditioned_rhs, rhs);
        control.set_tolerance(mixed_galerkin_solver_reduction *
                              preconditioned_rhs.l2_norm());
      };
      SolverGMRES<Vector<double>> gmres(
        control, SolverGMRES<Vector<double>>::AdditionalData(400));
      unsigned int gmres_steps = 0;
      struct Diagonal
      {
        Vector<double> inverse;
        void
        vmult(Vector<double> &y, const Vector<double> &x) const
        {
          y = x;
          y.scale(inverse);
        }
      } preconditioner;
      preconditioner.inverse.reinit(n);
      for (types::global_dof_index i = 0; i < n; ++i)
        preconditioner.inverse[i] =
          diagonal[i] != 0.0 ? 1.0 / diagonal[i] : 1.0;
      if (mixed_galerkin_preconditioner == "calderon")
        {
          // The Calderon preconditioner (MixedCalderonPreconditioner), with
          // a rank-one term for each piece of the Dirichlet part: its flux
          // unknowns joined through the cells.
          std::vector<types::global_dof_index> piece_of(n);
          for (types::global_dof_index e = 0; e < n; ++e)
            piece_of[e] = e;
          const auto find = [&](types::global_dof_index e) {
            while (piece_of[e] != e)
              e = piece_of[e] = piece_of[piece_of[e]];
            return e;
          };
          for (const auto &cell : dh.active_cell_iterators())
            if (is_dirichlet_cell(cell))
              {
                types::global_dof_index first = numbers::invalid_dof_index;
                for (const auto e : indices_of(cell))
                  if (role[e] == flux)
                    {
                      if (first == numbers::invalid_dof_index)
                        first = find(e);
                      else
                        piece_of[find(e)] = first;
                    }
              }
          std::map<types::global_dof_index, unsigned int> piece_number;
          std::vector<unsigned int> piece(n, numbers::invalid_unsigned_int);
          std::vector<MixedCalderonPreconditioner::Kind> kind(
            n, MixedCalderonPreconditioner::other);
          for (types::global_dof_index e = 0; e < n; ++e)
            if (role[e] == flux)
              {
                kind[e] = MixedCalderonPreconditioner::flux;
                piece[e] =
                  piece_number.try_emplace(find(e), piece_number.size())
                    .first->second;
              }
            else if (role[e] == free_potential)
              kind[e] = MixedCalderonPreconditioner::free_potential;
            else if (role[e] == tied_potential)
              kind[e] = MixedCalderonPreconditioner::tied_potential;
          // PIBEM_CALDERON_LUMPED: the lumped masses.
          const MixedCalderonPreconditioner calderon_preconditioner(
            kind,
            master,
            piece,
            mass,
            [&](const Vector<double> &v) {
              return product(single_layer, v, false);
            },
            [&](const Vector<double> &v_src,
                const Vector<double> &w_src,
                Vector<double>       &v,
                Vector<double>       &w) {
              if (through_fmm)
                calderon->vmult_preconditioner_pair(v_src, w_src, v, w);
              else
                {
                  v = product(single_layer, v_src, false);
                  w = product(hypersingular, w_src, false);
                }
            },
            std::getenv("PIBEM_CALDERON_LUMPED") != nullptr);
          if (calderon_preconditioner.n_pieces() > 1)
            pcout << "Mixed Galerkin problem: the Dirichlet part in "
                  << calderon_preconditioner.n_pieces() << " pieces"
                  << std::endl;
          pcout << "Mixed Galerkin problem: Calderon preconditioner"
                << std::endl;
          set_tolerance(calderon_preconditioner);
          if (through_fmm && mixed_galerkin_relaxation > 0.)
            {
              // Inexact GMRES (Simoncini and Szyld, Bouras and Fraysse): a
              // product may err by about the tolerance over the residual
              // reached, relative to the start, so the steps after the
              // residual has fallen below the given fraction of the
              // preconditioned right-hand side use the relaxed products;
              // the fraction should be about the reduction over the error
              // of those products, which PIBEM_MIXED_DIAGNOSTICS measures.
              Vector<double> preconditioned_rhs(n);
              calderon_preconditioner.vmult(preconditioned_rhs, rhs);
              const double rhs_norm = preconditioned_rhs.l2_norm();
              if (diagnostics)
                {
                  Vector<double> probe(n), full, coarse;
                  for (types::global_dof_index e = 0; e < n; ++e)
                    if (role[e] == flux || role[e] == free_potential)
                      probe[e] = std::sin(1.0 + 7.3 * e) + 0.5;
                  block.vmult(full, probe);
                  relaxed_rows = true;
                  block.vmult(coarse, probe);
                  relaxed_rows = false;
                  coarse -= full;
                  pcout << "Mixed Galerkin problem: relaxed products below "
                        << mixed_galerkin_relaxation
                        << " of the right-hand side, relative error "
                        << coarse.l2_norm() / full.l2_norm() << std::endl;
                }
              unsigned int relaxed_steps = 0;
              const auto   apply         = [&](Vector<double>       &y,
                                       const Vector<double> &v,
                                       const double          residual) {
                relaxed_rows = residual < mixed_galerkin_relaxation * rhs_norm;
                relaxed_steps += relaxed_rows;
                Vector<double> w(n);
                block.vmult(w, v);
                relaxed_rows = false;
                calderon_preconditioner.vmult(y, w);
              };
              gmres_steps = InexactGMRES::solve(apply,
                                                preconditioned_rhs,
                                                x,
                                                control.tolerance(),
                                                control.max_steps());
              pcout << "Mixed Galerkin problem: " << relaxed_steps << " of "
                    << gmres_steps << " steps with relaxed products"
                    << std::endl;
            }
          else
            {
              gmres.solve(block, x, rhs, calderon_preconditioner);
              gmres_steps = control.last_step();
            }
          pcout << "Mixed Galerkin problem: "
                << calderon_preconditioner.n_applications()
                << " applications of the preconditioner in "
                << calderon_preconditioner.seconds() << " s, of which "
                << calderon_preconditioner.mass_seconds()
                << " s in the mass solves (process 0)" << std::endl;
        }
      else
        {
          set_tolerance(preconditioner);
          gmres.solve(block, x, rhs, preconditioner);
          gmres_steps = control.last_step();
        }
      {
        // The true residual, block by block.
        Vector<double> residual;
        block.vmult(residual, x);
        residual -= rhs;
        double r[2] = {0, 0}, f[2] = {0, 0};
        for (types::global_dof_index i = 0; i < n; ++i)
          if (role[i] == flux || role[i] == free_potential)
            {
              const unsigned int b = role[i] == flux ? 0 : 1;
              r[b] += residual[i] * residual[i];
              f[b] += rhs[i] * rhs[i];
            }
        pcout << "Mixed Galerkin problem: relative residual of the flux rows "
              << std::sqrt(r[0] / f[0]) << ", of the potential rows "
              << std::sqrt(r[1] / f[1]) << std::endl;
      }
      pcout << "Mixed Galerkin problem: GMRES iterations: "
            << gmres_steps << ", in "
            << std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - start)
                 .count()
            << " s since the start of the mixed solve" << std::endl;
      if (calderon)
        calderon->print_statistics(pcout.get_stream());

      // The traces on the whole surface.
      Vector<double> u_full(n), t_full(n);
      for (types::global_dof_index i = 0; i < n; ++i)
        {
          switch (role[i])
            {
              case flux:
                t_full[i] = x[i];
                u_full[i] = u_data[i];
                break;
              case free_potential:
              case tied_potential:
                u_full[i] = x[master[i]];
                t_full[i] =
                  g_data[i] - (robin[i] ? robin_coefficient[i] * u_full[i] : 0.0);
                break;
              case known_potential:
                u_full[i] = u_data[i];
                t_full[i] = g_data[i];
                break;
              case hanging:
                break;
            }
        }
      constraints.distribute(u_full);
      constraints.distribute(t_full);
      // The traces on the cells, for the next solve on a refined mesh.
      if (reuse_near_field)
        {
          transfer_values.clear();
          transfer_degree = fe->degree;
          for (const auto &cell : dh.active_cell_iterators())
            {
              const auto         &cell_dofs = indices_of(cell);
              std::vector<double> local(fe->dofs_per_cell);
              const bool          d = is_dirichlet_cell(cell);
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                local[i] = (d ? t_full : u_full)[cell_dofs[i]];
              transfer_values[{{cell->level(), cell->index()}}] =
                std::move(local);
            }
        }
      // At degrees of freedom of both parts, the Dirichlet side.
      for (const auto i : this_cpu_set)
        {
          phi[i]     = u_full[i];
          dphi_dn[i] = t_full[i];
        }
      phi.compress(VectorOperation::insert);
      dphi_dn.compress(VectorOperation::insert);

      // Error indicators for adaptive refinement: on each cell the jumps of
      // the conormal derivative of the discrete trace, the flux on the
      // Dirichlet part and the potential on the Neumann part, across its
      // edges to cells of the same part and the same smooth piece (material),
      // half of each edge to each side. With a_F = |T|/|F| the extent of the
      // cell across the edge F, sum_F a_F/2 int_F [d_nu f]^2 measures
      // |f - f_h|^2_{H^1(T)} and sum_F a_F^3/(2 pi^2) int_F [d_nu f]^2
      // measures ||f - f_h||^2_{L^2(T)}, with the Poincare constant a/pi,
      // also on elongated cells; the norms of the traces weight them by
      // a = min_F a_F: a ||e||^2_{L^2(T)} for the flux and a |e|^2_{H^1(T)}
      // for the potential. With an error dump they are
      // computed and written in any case.
      const char *dump_requested = std::getenv("PIBEM_MIXED_ERROR_DUMP");
      // The smallest extent of a cell across its edges, min_F |T|/|F|.
      const auto cell_extent = [](const auto &cell) {
        double a = 1e300;
        for (const unsigned int face : cell->face_indices())
          a = std::min(a, cell->measure() / cell->face(face)->measure());
        return a;
      };
      Vector<double> jump_indicator;
      if (mixed_galerkin_error_indicator == "jumps" ||
          dump_requested != nullptr)
        {
          const auto &tria = dh.get_triangulation();
          jump_indicator.reinit(tria.n_active_cells());
          const QGauss<dim - 2>  face_quadrature(fe->degree + 1);
          const unsigned int     nq    = face_quadrature.size();
          const UpdateFlags      flags = update_gradients |
                                    update_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values;
          FEFaceValues<dim - 1, dim>    here(*mapping, *fe, face_quadrature, flags);
          FEFaceValues<dim - 1, dim>    there(*mapping, *fe, face_quadrature, flags);
          // The conormal derivative of the trace at the quadrature points.
          const auto derivative = [&](const FEValuesBase<dim - 1, dim> &v,
                                      const auto                       &cell,
                                      const Vector<double>             &f) {
            const auto         &dofs = indices_of(cell);
            std::vector<double> d(nq, 0.);
            for (unsigned int q = 0; q < nq; ++q)
              {
                Tensor<1, dim> g;
                for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                  g += f[dofs[i]] * v.shape_grad(i, q);
                d[q] = g * v.normal_vector(q);
              }
            return d;
          };
          // int_F [d_nu f]^2, the quadrature points of the two sides matched
          // in the same or the reversed order.
          const auto jump = [&](const FEValuesBase<dim - 1, dim> &a,
                                const auto                       &cell_a,
                                const FEValuesBase<dim - 1, dim> &b,
                                const auto                       &cell_b,
                                const Vector<double>             &f,
                                const char                       *kind) {
            const std::vector<double> da = derivative(a, cell_a, f);
            const std::vector<double> db = derivative(b, cell_b, f);
            double same = 0, reversed = 0;
            for (unsigned int q = 0; q < nq; ++q)
              {
                same += a.quadrature_point(q).distance(b.quadrature_point(q));
                reversed +=
                  a.quadrature_point(q).distance(b.quadrature_point(nq - 1 - q));
              }
            const double length =
              a.quadrature_point(0).distance(a.quadrature_point(nq - 1));
            if (std::min(same, reversed) > 1e-6 * nq * length)
              {
                std::ostringstream message;
                message << "Error indicator: the quadrature points of the two "
                           "sides of an edge do not match ("
                        << kind << ", cells at " << cell_a->center() << " and "
                        << cell_b->center() << "):";
                for (unsigned int q = 0; q < nq; ++q)
                  message << " [" << a.quadrature_point(q) << " | "
                          << b.quadrature_point(q) << "]";
                AssertThrow(false, ExcMessage(message.str()));
              }
            double sum = 0;
            for (unsigned int q = 0; q < nq; ++q)
              {
                const double j =
                  da[q] + db[same <= reversed ? q : nq - 1 - q];
                sum += j * j * a.JxW(q);
              }
            return sum;
          };
          // An edge with a hanging node: on the exact geometry the hanging
          // vertex need not lie at the midpoint of the reference edge of the
          // coarse cell, so the coarse side is evaluated at the quadrature
          // points of the fine edge, pulled back onto the mapped edge of the
          // coarse cell (samples, then Newton in the edge parameter), with
          // the conormal of the fine side reversed.
          const auto jump_hanging = [&](const FEValuesBase<dim - 1, dim> &fine,
                                        const auto                       &fine_cell,
                                        const auto                       &coarse,
                                        const unsigned int coarse_face,
                                        const Vector<double>             &f) {
            const std::vector<double> d_fine = derivative(fine, fine_cell, f);
            const auto on_face = [&](const double t) {
              return coarse_face < 2 ? Point<dim - 1>(double(coarse_face), t) :
                                       Point<dim - 1>(t, double(coarse_face - 2));
            };
            const auto x_of = [&](const double t) {
              return mapping->transform_unit_to_real_cell(coarse, on_face(t));
            };
            std::vector<Point<dim - 1>> unit(nq);
            for (unsigned int q = 0; q < nq; ++q)
              {
                const Point<dim> target = fine.quadrature_point(q);
                double t = 0, gap = target.distance(x_of(0.));
                for (unsigned int k = 1; k <= 16; ++k)
                  if (target.distance(x_of(k / 16.)) < gap)
                    {
                      t   = k / 16.;
                      gap = target.distance(x_of(t));
                    }
                for (unsigned int step = 0; step < 30; ++step)
                  {
                    const double         lo = std::max(0., t - 1e-7),
                                 hi = std::min(1., t + 1e-7);
                    const Tensor<1, dim> tangent = (x_of(hi) - x_of(lo)) / (hi - lo);
                    const double next = std::clamp(
                      t + (target - x_of(t)) * tangent / tangent.norm_square(),
                      0.,
                      1.);
                    const bool done = std::abs(next - t) < 1e-13;
                    t               = next;
                    if (done)
                      break;
                  }
                unit[q] = on_face(t);
              }
            FEValues<dim - 1, dim> values(*mapping,
                                          *fe,
                                          Quadrature<dim - 1>(unit),
                                          update_gradients);
            values.reinit(coarse);
            const auto &dofs = indices_of(coarse);
            double      sum  = 0;
            for (unsigned int q = 0; q < nq; ++q)
              {
                Tensor<1, dim> g;
                for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                  g += f[dofs[i]] * values.shape_grad(i, q);
                const double j = d_fine[q] - g * fine.normal_vector(q);
                sum += j * j * fine.JxW(q);
              }
            return sum;
          };
          const auto same_piece = [&](const auto &a, const auto &b) {
            return is_dirichlet_cell(a) == is_dirichlet_cell(b) &&
                   a->material_id() == b->material_id();
          };
          for (const auto &cell : dh.active_cell_iterators())
            {
              if (cell->active_cell_index() % n_mpi_processes !=
                  this_mpi_process)
                continue;
              const bool            d = is_dirichlet_cell(cell);
              const Vector<double> &f = d ? t_full : u_full;
              double                sum  = 0;
              const double          area = cell->measure();
              // The weight of an edge: a_F/2 for the potential, a_F^3/(2 pi^2)
              // for the flux.
              const bool l2 = mixed_galerkin_indicator_norm == "L2";
              const auto weight = [&](const unsigned int face) {
                const double a = area / cell->face(face)->measure();
                return 0.5 * (d || l2 ? a * a * a / (M_PI * M_PI) : a);
              };
              for (const unsigned int face : cell->face_indices())
                {
                  if (cell->at_boundary(face))
                    continue;
                  if (cell->face(face)->has_children())
                    for (unsigned int s = 0;
                         s < cell->face(face)->n_children();
                         ++s)
                      {
                        const auto child =
                          cell->neighbor_child_on_subface(face, s);
                        if (!same_piece(cell, child))
                          continue;
                        unsigned int child_face = numbers::invalid_unsigned_int;
                        for (const unsigned int g : child->face_indices())
                          if (!child->at_boundary(g) &&
                              child->neighbor(g) == cell)
                            child_face = g;
                        AssertThrow(child_face != numbers::invalid_unsigned_int,
                                    ExcInternalError());
                        there.reinit(child, child_face);
                        sum += weight(face) *
                               jump_hanging(there, child, cell, face, f);
                      }
                  else
                    {
                      const auto neighbor = cell->neighbor(face);
                      if (!same_piece(cell, neighbor))
                        continue;
                      here.reinit(cell, face);
                      if (cell->neighbor_is_coarser(face))
                        sum += weight(face) *
                               jump_hanging(here,
                                            cell,
                                            neighbor,
                                            cell->neighbor_of_coarser_neighbor(face).first,
                                            f);
                      else
                        {
                          there.reinit(neighbor,
                                       cell->neighbor_of_neighbor(face));
                          sum += weight(face) *
                                 jump(here, cell, there, neighbor, f, "same level");
                        }
                    }
                }
              jump_indicator[cell->active_cell_index()] =
                (l2 ? 1. : cell_extent(cell)) * sum;
            }
          Utilities::MPI::sum(jump_indicator, mpi_communicator, jump_indicator);
        }
      std::vector<double> cell_squared_error(
        dh.get_triangulation().n_active_cells(), 0.);
      // The squared L2 norms of the exact flux on the Dirichlet part and of
      // the exact potential on the Neumann part.
      double part_norm_squared[2] = {1., 1.};

      // L2 errors: of the flux on the Dirichlet part, of the potential on the
      // Neumann part.
      {
        const QGauss<dim - 1>  error_quadrature(fe->degree + 4);
        FEValues<dim - 1, dim> error_values(*mapping,
                                      *fe,
                                      error_quadrature,
                                      update_values | update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
        FEValues<dim - 1, dim> error_data_values(*data_mapping,
                                                 *fe,
                                                 error_quadrature,
                                                 update_normal_vectors |
                                                   update_quadrature_points);
        double flux_error = 0, flux_norm = 0, potential_error = 0,
               potential_norm = 0, flux_max = 0, potential_max = 0,
               flux_error_edge = 0, flux_norm_edge = 0, robin_flux_error = 0,
               robin_flux_norm = 0;
        Point<dim>     flux_where, potential_where;
        Vector<double> g(dim);
        // PIBEM_MIXED_ERROR_DUMP=file: one line per cell, its center, whether
        // it is a Dirichlet cell and touches a double node, its squared
        // errors and squared norm, and the squared error of the best
        // approximation of the exact values by the shape functions of the
        // cell alone; then the largest errors and exact value at the
        // quadrature points and whether the cell has a hanging node.
        const char   *dump_name = std::getenv("PIBEM_MIXED_ERROR_DUMP");
        std::ofstream dump;
        if (dump_name != nullptr && this_mpi_process == 0)
          dump.open(dump_name);
        for (const auto &cell : dh.active_cell_iterators())
          {
            const double cell_error_before =
              flux_error + potential_error;
            const double cell_norm_before = flux_norm + potential_norm;
            FullMatrix<double> local_mass(fe->dofs_per_cell);
            Vector<double>     local_rhs(fe->dofs_per_cell);
            double             local_square = 0;
            error_values.reinit(cell);
            const FEValues<dim - 1, dim> &error_data =
              separate_data ? (error_data_values.reinit(cell),
                               error_data_values) :
                              error_values;
            const auto &cell_dofs = indices_of(cell);
            const bool  dirichlet = is_dirichlet_cell(cell);
            bool        at_edge   = false;
            for (const auto e : cell_dofs)
              if (double_nodes_set[original[e]].size() > 1 ||
                  copy_of[original[e]] != numbers::invalid_dof_index)
                at_edge = true;
            bool hanging_cell = false;
            for (const auto e : cell_dofs)
              if (role[e] == hanging)
                hanging_cell = true;
            double              cell_max_error = 0, cell_max_exact = 0;
            std::vector<double> cell_exact(error_quadrature.size());
            for (unsigned int q = 0; q < error_quadrature.size(); ++q)
              {
                double value = 0;
                for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                  value += error_values.shape_value(i, q) *
                           (dirichlet ? t_full : u_full)[cell_dofs[i]];
                double exact_value;
                if (dirichlet)
                  {
                    gradient.vector_value(error_data.quadrature_point(q), g);
                    exact_value = 0;
                    for (unsigned int d = 0; d < dim; ++d)
                      exact_value += g[d] * error_data.normal_vector(q)[d];
                    flux_error += (value - exact_value) *
                                  (value - exact_value) * error_values.JxW(q);
                    if (dump.is_open())
                      {
                        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                          {
                            for (unsigned int j = 0; j < fe->dofs_per_cell;
                                 ++j)
                              local_mass(i, j) +=
                                error_values.shape_value(i, q) *
                                error_values.shape_value(j, q) *
                                error_values.JxW(q);
                            local_rhs(i) += error_values.shape_value(i, q) *
                                            exact_value * error_values.JxW(q);
                          }
                        local_square +=
                          exact_value * exact_value * error_values.JxW(q);
                      }
                    flux_norm += exact_value * exact_value * error_values.JxW(q);
                    if (at_edge)
                      {
                        flux_error_edge += (value - exact_value) *
                                           (value - exact_value) *
                                           error_values.JxW(q);
                        flux_norm_edge +=
                          exact_value * exact_value * error_values.JxW(q);
                      }
                    if (std::abs(value - exact_value) > flux_max)
                      {
                        flux_max   = std::abs(value - exact_value);
                        flux_where = error_values.quadrature_point(q);
                      }
                  }
                else
                  {
                    exact_value =
                      potential.value(error_data.quadrature_point(q));
                    potential_error += (value - exact_value) *
                                       (value - exact_value) * error_values.JxW(q);
                    potential_norm +=
                      exact_value * exact_value * error_values.JxW(q);
                    if (std::abs(value - exact_value) > potential_max)
                      {
                        potential_max   = std::abs(value - exact_value);
                        potential_where = error_values.quadrature_point(q);
                      }
                    // On the Robin part the flux g/b - (a/b) u as well.
                    if (has_robin && is_robin_cell(cell))
                      {
                        double flux_value = 0, exact_flux = 0;
                        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                          flux_value += error_values.shape_value(i, q) *
                                        t_full[cell_dofs[i]];
                        gradient.vector_value(error_data.quadrature_point(q),
                                              g);
                        for (unsigned int d = 0; d < dim; ++d)
                          exact_flux += g[d] * error_data.normal_vector(q)[d];
                        robin_flux_error += (flux_value - exact_flux) *
                                            (flux_value - exact_flux) *
                                            error_values.JxW(q);
                        robin_flux_norm +=
                          exact_flux * exact_flux * error_values.JxW(q);
                      }
                  }
                cell_exact[q]  = exact_value;
                cell_max_error =
                  std::max(cell_max_error, std::abs(value - exact_value));
                cell_max_exact = std::max(cell_max_exact, std::abs(exact_value));
              }
            cell_squared_error[cell->active_cell_index()] =
              flux_error + potential_error - cell_error_before;
            if (dump.is_open())
              {
                double best = 0, best_max = 0;
                if (dirichlet)
                  {
                    Vector<double> coefficients(local_rhs);
                    local_mass.gauss_jordan();
                    local_mass.vmult(coefficients, local_rhs);
                    best = std::max(local_square - (coefficients * local_rhs),
                                    0.);
                    for (unsigned int q = 0; q < error_quadrature.size(); ++q)
                      {
                        double b = 0;
                        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                          b += coefficients[i] * error_values.shape_value(i, q);
                        best_max = std::max(best_max, std::abs(b - cell_exact[q]));
                      }
                  }
                // Then, at the quadrature points of the cell, the largest
                // error, the largest exact value and the largest error of
                // the local best approximation, and whether the cell has a
                // hanging node.
                dump << cell->center() << ' ' << dirichlet << ' ' << at_edge
                     << ' ' << flux_error + potential_error - cell_error_before
                     << ' ' << flux_norm + potential_norm - cell_norm_before
                     << ' ' << best << ' ' << cell_max_error << ' '
                     << cell_max_exact << ' ' << best_max << ' '
                     << hanging_cell << ' '
                     << jump_indicator[cell->active_cell_index()] << '\n';
              }
          }
        part_norm_squared[0] = flux_norm;
        part_norm_squared[1] = potential_norm;
        // Zero without a Dirichlet part (Robin conditions everywhere).
        pcout << "Mixed Galerkin problem: flux on the Dirichlet part, "
                 "relative L2 error "
              << (flux_norm > 0 ? std::sqrt(flux_error / flux_norm) : 0.)
              << "; potential on the Neumann part, relative L2 error "
              << std::sqrt(potential_error / potential_norm) << std::endl
              << "  flux on the cells at double nodes: absolute L2 error "
              << std::sqrt(flux_error_edge) << " (norm "
              << std::sqrt(flux_norm_edge) << "), elsewhere "
              << std::sqrt(flux_error - flux_error_edge) << " (norm "
              << std::sqrt(flux_norm - flux_norm_edge) << ")" << std::endl
              << "  absolute L2 errors: flux " << std::sqrt(flux_error)
              << " (norm " << std::sqrt(flux_norm) << "), potential "
              << std::sqrt(potential_error) << " (norm "
              << std::sqrt(potential_norm) << ")" << std::endl
              << "  largest error of the flux " << flux_max << " at "
              << flux_where << ", of the potential " << potential_max
              << " at " << potential_where << std::endl;
        if (has_robin)
          pcout << "Mixed Galerkin problem: flux on the Robin part, relative "
                   "L2 error "
                << std::sqrt(robin_flux_error / robin_flux_norm) << std::endl;
      }

      // The indicators that adaptive_refinement() uses, and their sums on the
      // two parts.
      if (mixed_galerkin_error_indicator != "none")
        {
          mixed_indicators.reinit(dh.get_triangulation().n_active_cells());
          double sums[2] = {0, 0};
          for (const auto &cell : dh.active_cell_iterators())
            {
              const unsigned int c = cell->active_cell_index();
              const double       h = cell_extent(cell);
              const bool         d = is_dirichlet_cell(cell);
              const bool jumps = mixed_galerkin_error_indicator == "jumps";
              if (mixed_galerkin_indicator_norm == "L2")
                mixed_indicators[c] =
                  (jumps ? jump_indicator[c] : cell_squared_error[c]) /
                  part_norm_squared[d ? 0 : 1];
              else
                mixed_indicators[c] =
                  jumps ? jump_indicator[c] :
                          (d ? h * cell_squared_error[c] :
                               cell_squared_error[c] / h);
              sums[d ? 0 : 1] += mixed_indicators[c];
            }
          pcout << "Mixed Galerkin problem: error indicator ("
                << mixed_galerkin_error_indicator << ", "
                << mixed_galerkin_indicator_norm << "): flux part "
                << std::sqrt(sums[0]) << ", potential part "
                << std::sqrt(sums[1]) << ", total "
                << std::sqrt(sums[0] + sums[1]) << ", on "
                << dh.get_triangulation().n_active_cells() << " cells"
                << std::endl;
        }
    }
  else
    {
      (void)phi;
      (void)dphi_dn;
      (void)potential;
      (void)gradient;
      AssertThrow(false,
                  ExcMessage("The mixed Galerkin problem is implemented in "
                             "3D only."));
    }
}


template void
BEMProblem<2>::solve_galerkin_mixed(TrilinosWrappers::MPI::Vector &,
                                    TrilinosWrappers::MPI::Vector &,
                                    const Function<2> &,
                                    const Function<2> &);
template void
BEMProblem<3>::solve_galerkin_mixed(TrilinosWrappers::MPI::Vector &,
                                    TrilinosWrappers::MPI::Vector &,
                                    const Function<3> &,
                                    const Function<3> &);
