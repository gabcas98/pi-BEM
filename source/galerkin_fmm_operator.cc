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

#include "../include/galerkin_fmm_operator.h"

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/numerics/rtree.h>

#include <map>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>

using namespace dealii;

namespace
{
  double
  seconds_since(const std::chrono::steady_clock::time_point &start)
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
      .count();
  }

  void
  sum_over_processes(Vector<double> &v, const MPI_Comm comm)
  {
    Utilities::MPI::sum(ArrayView<const double>(v.begin(), v.size()),
                        comm,
                        ArrayView<double>(v.begin(), v.size()));
  }

  constexpr double inverse_4pi = 1. / (4. * numbers::PI);
} // namespace



GalerkinFMMOperator::GalerkinFMMOperator(const MPI_Comm comm)
  : mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
  , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
  , fmm(comm)
  , rhs_fmm(comm)
{}



void
GalerkinFMMOperator::reinit(const DoFHandler<2, 3>          &dh,
                            const Mapping<2, 3>             &mapping,
                            const AffineConstraints<double> &constraints_in,
                            const GalerkinPanelPairIntegrator<3> &integrator,
                            const IndexSet          &locally_owned_dofs,
                            const AdditionalData    &data,
                            const RightHandSideData &rhs)
{
  dof_handler = &dh;
  constraints.copy_from(constraints_in);
  regularize                     = rhs.regularize;
  const FiniteElement<2, 3> &fe  = dh.get_fe();
  const unsigned int         dpc = fe.dofs_per_cell;

  // Geometry and data at the quadrature points of every cell.
  auto start = std::chrono::steady_clock::now();
  cells.clear();
  own_cells.clear();
  for (const auto &cell : dh.active_cell_iterators())
    {
      if (cell->subdomain_id() == this_mpi_process)
        own_cells.push_back(cells.size());
      cells.push_back(cell);
    }
  const QGauss<2> quadrature(data.quadrature_order);
  n_q_points   = quadrature.size();
  mapping_info = std::make_unique<NonMatching::MappingInfo<2, 3>>(
    mapping,
    update_values | update_JxW_values | update_quadrature_points |
      update_normal_vectors);
  std::vector<Triangulation<2, 3>::cell_iterator> tria_cells(cells.begin(),
                                                             cells.end());
  mapping_info->reinit_cells(tria_cells,
                             std::vector<Quadrature<2>>(cells.size(),
                                                        quadrature));
  evaluator = std::make_unique<FEPointEvaluation<1, 2, 3>>(*mapping_info, fe);
  const unsigned int    n_points = cells.size() * n_q_points;
  std::vector<Point<3>> points(n_points);
  JxW.resize(n_points);
  normals.resize(n_points);
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      evaluator->reinit(c);
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          points[c * n_q_points + q]  = evaluator->quadrature_point(q);
          JxW[c * n_q_points + q]     = evaluator->JxW(q);
          normals[c * n_q_points + q] = evaluator->normal_vector(q);
        }
    }
  dirichlet_values.clear();
  if (rhs.dirichlet != nullptr)
    {
      dirichlet_values.resize(n_points);
      for (unsigned int p = 0; p < n_points; ++p)
        dirichlet_values[p] = rhs.dirichlet->value(points[p]);
    }
  flux_values.clear();
  if (rhs.gradient != nullptr)
    {
      flux_values.resize(n_points);
      Vector<double> gradient(3);
      for (unsigned int p = 0; p < n_points; ++p)
        {
          rhs.gradient->vector_value(points[p], gradient);
          flux_values[p] = gradient[0] * normals[p][0] +
                           gradient[1] * normals[p][1] +
                           gradient[2] * normals[p][2];
        }
    }
  setup_geometry = seconds_since(start);

  start = std::chrono::steady_clock::now();
  fmm.reinit(points, data.fmm);
  if (rhs.dirichlet != nullptr)
    {
      LaplaceFMM::AdditionalData rhs_data = data.fmm;
      rhs_data.order += data.rhs_order_increase;
      rhs_fmm.reinit(points, rhs_data);
    }
  setup_tree = seconds_since(start);

  // The test cells of the near field of this process: every n-th cell, so
  // that the pairs near the contact, where the mesh is graded, are shared.
  start = std::chrono::steady_clock::now();
  std::vector<unsigned int> near_test_cells;
  for (unsigned int c = this_mpi_process; c < cells.size();
       c += n_mpi_processes)
    near_test_cells.push_back(c);

  // Candidate pairs: cells whose bounding boxes, each enlarged by the ratio
  // times its own diagonal, intersect. Any pair within the ratio times the
  // larger diameter is among them; the integrator then decides.
  std::vector<std::pair<BoundingBox<3>, unsigned int>> boxes;
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      BoundingBox<3> box      = mapping.get_bounding_box(cells[c]);
      const double   diagonal = box.get_boundary_points().first.distance(
        box.get_boundary_points().second);
      box.extend(data.near_distance_ratio * diagonal + 1e-12 * diagonal);
      boxes.emplace_back(box, c);
    }
  const auto tree = pack_rtree(boxes);
  // Each unordered pair once: the kernel is symmetric, and the double layer
  // of the exchanged pair comes from the same points. The third entry is a
  // rough cost, by the kind of rule the pair needs.
  std::vector<std::array<unsigned int, 3>> found;
  for (const unsigned int t : near_test_cells)
    {
      std::vector<std::pair<BoundingBox<3>, unsigned int>> hits;
      tree.query(boost::geometry::index::intersects(boxes[t].first),
                 std::back_inserter(hits));
      for (const auto &hit : hits)
        {
          const unsigned int r = hit.second;
          if (r < t || integrator.is_far_regular_pair(cells[t], cells[r]))
            continue;
          const GalerkinPanelPairType type =
            integrator.classify(cells[t], cells[r]);
          const unsigned int cost =
            type == GalerkinPanelPairType::same_panel         ? 6 :
            type == GalerkinPanelPairType::common_edge ||
                type == GalerkinPanelPairType::nonconforming_edge ? 5 :
            type == GalerkinPanelPairType::common_vertex        ? 2 :
                                                                  1;
          found.push_back({{t, r, cost}});
        }
    }

  // Deal all pairs out by cost, the most expensive first to the least
  // loaded process: the singular pairs cluster where the mesh is graded.
  std::vector<std::array<unsigned int, 3>> all_pairs;
  for (const auto &part :
       Utilities::MPI::all_gather(mpi_communicator, found))
    all_pairs.insert(all_pairs.end(), part.begin(), part.end());
  std::sort(all_pairs.begin(),
            all_pairs.end(),
            [](const auto &x, const auto &y) {
              return x[2] != y[2] ? x[2] > y[2] :
                     x[0] != y[0] ? x[0] < y[0] :
                                    x[1] < y[1];
            });
  std::vector<unsigned long> load(n_mpi_processes, 0);
  std::map<unsigned int, std::vector<unsigned int>> my_pairs;
  n_near_pairs = 0;
  for (const auto &pair : all_pairs)
    {
      const unsigned int process =
        std::min_element(load.begin(), load.end()) - load.begin();
      load[process] += pair[2];
      if (process == this_mpi_process)
        {
          my_pairs[pair[0]].push_back(pair[1]);
          n_near_pairs += pair[0] == pair[1] ? 1 : 2;
        }
    }
  near_test_cells.clear();
  std::vector<std::vector<unsigned int>> near_trial_cells;
  for (auto &[t, trials] : my_pairs)
    {
      near_test_cells.push_back(t);
      near_trial_cells.push_back(std::move(trials));
    }

  // Sparsity of the near field, condensed like the operator: both blocks
  // of every pair. The cells belong to any process, and constraints write
  // into the rows of the masters.
  IndexSet                             writable_rows(dh.n_dofs());
  std::vector<types::global_dof_index> test_dofs(dpc), trial_dofs(dpc);
  writable_rows.add_indices(locally_owned_dofs);
  auto add_rows = [&](const std::vector<types::global_dof_index> &dofs) {
    for (const auto i : dofs)
      {
        writable_rows.add_index(i);
        if (constraints.is_constrained(i))
          for (const auto &entry : *constraints.get_constraint_entries(i))
            writable_rows.add_index(entry.first);
      }
  };
  for (unsigned int k = 0; k < near_test_cells.size(); ++k)
    {
      cells[near_test_cells[k]]->get_dof_indices(test_dofs);
      add_rows(test_dofs);
      for (const unsigned int r : near_trial_cells[k])
        {
          cells[r]->get_dof_indices(trial_dofs);
          add_rows(trial_dofs);
        }
    }
  writable_rows.compress();
  TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                             locally_owned_dofs,
                                             writable_rows,
                                             mpi_communicator);
  for (unsigned int k = 0; k < near_test_cells.size(); ++k)
    {
      cells[near_test_cells[k]]->get_dof_indices(test_dofs);
      for (const unsigned int r : near_trial_cells[k])
        {
          cells[r]->get_dof_indices(trial_dofs);
          constraints.add_entries_local_to_global(test_dofs,
                                                  trial_dofs,
                                                  sparsity,
                                                  true);
          constraints.add_entries_local_to_global(trial_dofs,
                                                  test_dofs,
                                                  sparsity,
                                                  true);
        }
    }
  sparsity.compress();
  near_exact.reinit(sparsity);
  near_correction.reinit(sparsity);
  near_double_layer.reinit(rhs.dirichlet != nullptr ? dh.n_dofs() : 0);
  near_manufactured.reinit(rhs.gradient != nullptr ? dh.n_dofs() : 0);

  // Shape values at the reference quadrature points, the same on every
  // cell for a Lagrange element.
  FullMatrix<double> shape(n_q_points, dpc);
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int i = 0; i < dpc; ++i)
      shape(q, i) = fe.shape_value(i, quadrature.point(q));

  FullMatrix<double> exact(dpc, dpc), correction(dpc, dpc),
    transposed(dpc, dpc), kernel(n_q_points, n_q_points),
    partial(dpc, n_q_points);
  Vector<double> exact_test_rhs(dpc), exact_trial_rhs(dpc), local(dpc);
  std::vector<double> test_side(n_q_points), trial_side(n_q_points);
  for (unsigned int k = 0; k < near_test_cells.size(); ++k)
    {
      const unsigned int t = near_test_cells[k];
      cells[t]->get_dof_indices(test_dofs);
      for (const unsigned int r : near_trial_cells[k])
        {
          cells[r]->get_dof_indices(trial_dofs);
          const bool self = r == t;

          // Exact contributions: the single-layer block, and with it in one
          // traversal the double-layer right-hand side of the test cell and,
          // for two cells, of the trial cell.
          if (rhs.dirichlet != nullptr)
            {
              const FullMatrix<double> both =
                self ? integrator.integrate_single_and_double_layer(
                         cells[t],
                         cells[r],
                         data.quadrature_order,
                         *rhs.dirichlet,
                         rhs.regularize) :
                       integrator.integrate_single_and_double_layer_both(
                         cells[t],
                         cells[r],
                         data.quadrature_order,
                         *rhs.dirichlet,
                         rhs.regularize);
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  for (unsigned int j = 0; j < dpc; ++j)
                    exact(i, j) = both(i, j);
                  exact_test_rhs[i] = both(i, dpc);
                  if (!self)
                    exact_trial_rhs[i] = both(i, dpc + 1);
                }
            }
          else
            exact = integrator.integrate_single_layer(cells[t],
                                                      cells[r],
                                                      data.quadrature_order);

          // The contributions that the point sums contain, without the
          // coincident points of a cell with itself: the kernel block, and
          // the dipole sums with either cell as target.
          std::fill(test_side.begin(), test_side.end(), 0.);
          std::fill(trial_side.begin(), trial_side.end(), 0.);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const unsigned int a = t * n_q_points + q;
              for (unsigned int s = 0; s < n_q_points; ++s)
                {
                  const unsigned int b = r * n_q_points + s;
                  if (a == b)
                    {
                      kernel(q, s) = 0;
                      continue;
                    }
                  const Tensor<1, 3> x_minus_y = points[a] - points[b];
                  const double       distance  = x_minus_y.norm();
                  const double       weight    = inverse_4pi * JxW[a] * JxW[b];
                  kernel(q, s)                 = weight / distance;
                  if (rhs.dirichlet != nullptr)
                    {
                      const double cube = distance * distance * distance;
                      test_side[q] +=
                        weight * (x_minus_y * normals[b]) / cube *
                        (dirichlet_values[b] -
                         (rhs.regularize ? dirichlet_values[a] : 0.));
                      if (!self)
                        trial_side[s] -=
                          weight * (x_minus_y * normals[a]) / cube *
                          (dirichlet_values[a] -
                           (rhs.regularize ? dirichlet_values[b] : 0.));
                    }
                }
            }
          for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int s = 0; s < n_q_points; ++s)
              {
                double sum = 0;
                for (unsigned int q = 0; q < n_q_points; ++q)
                  sum += shape(q, i) * kernel(q, s);
                partial(i, s) = sum;
              }
          for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int j = 0; j < dpc; ++j)
              {
                double sum = 0;
                for (unsigned int s = 0; s < n_q_points; ++s)
                  sum += partial(i, s) * shape(s, j);
                correction(i, j) = exact(i, j) - sum;
              }
          constraints.distribute_local_to_global(exact,
                                                 test_dofs,
                                                 trial_dofs,
                                                 near_exact);
          constraints.distribute_local_to_global(correction,
                                                 test_dofs,
                                                 trial_dofs,
                                                 near_correction);
          if (!self)
            {
              transposed.copy_transposed(exact);
              constraints.distribute_local_to_global(transposed,
                                                     trial_dofs,
                                                     test_dofs,
                                                     near_exact);
              transposed.copy_transposed(correction);
              constraints.distribute_local_to_global(transposed,
                                                     trial_dofs,
                                                     test_dofs,
                                                     near_correction);
            }

          if (rhs.dirichlet != nullptr)
            {
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  double sum = 0;
                  for (unsigned int q = 0; q < n_q_points; ++q)
                    sum += shape(q, i) * test_side[q];
                  local[i] = exact_test_rhs[i] - sum;
                }
              constraints.distribute_local_to_global(local,
                                                     test_dofs,
                                                     near_double_layer);
              if (!self)
                {
                  for (unsigned int j = 0; j < dpc; ++j)
                    {
                      double sum = 0;
                      for (unsigned int s = 0; s < n_q_points; ++s)
                        sum += shape(s, j) * trial_side[s];
                      local[j] = exact_trial_rhs[j] - sum;
                    }
                  constraints.distribute_local_to_global(local,
                                                         trial_dofs,
                                                         near_double_layer);
                }
            }
          if (rhs.gradient != nullptr)
            {
              // Diagnostic only: each direction with its own traversal.
              local = integrator.integrate_single_layer_flux(
                cells[t], cells[r], data.quadrature_order, *rhs.gradient);
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  double sum = 0;
                  for (unsigned int s = 0; s < n_q_points; ++s)
                    sum += partial(i, s) * flux_values[r * n_q_points + s];
                  local[i] -= sum;
                }
              constraints.distribute_local_to_global(local,
                                                     test_dofs,
                                                     near_manufactured);
              if (!self)
                {
                  local = integrator.integrate_single_layer_flux(
                    cells[r], cells[t], data.quadrature_order, *rhs.gradient);
                  // partial^T: the kernel block seen from the trial cell
                  for (unsigned int j = 0; j < dpc; ++j)
                    {
                      double sum = 0;
                      for (unsigned int s = 0; s < n_q_points; ++s)
                        for (unsigned int q = 0; q < n_q_points; ++q)
                          sum += shape(s, j) * kernel(q, s) *
                                 flux_values[t * n_q_points + q];
                      local[j] -= sum;
                    }
                  constraints.distribute_local_to_global(local,
                                                         trial_dofs,
                                                         near_manufactured);
                }
            }
        }
    }
  const double own_near_time = seconds_since(start);
  near_exact.compress(VectorOperation::add);
  near_correction.compress(VectorOperation::add);
  if (near_double_layer.size() > 0)
    sum_over_processes(near_double_layer, mpi_communicator);
  if (near_manufactured.size() > 0)
    sum_over_processes(near_manufactured, mpi_communicator);
  setup_near = seconds_since(start);
  setup_near_imbalance =
    Utilities::MPI::max(own_near_time, mpi_communicator) /
    std::max(1e-300,
             Utilities::MPI::sum(own_near_time, mpi_communicator) /
               n_mpi_processes);

  time_evaluate = time_fmm = time_test = time_near = time_rhs = 0;
  n_vmult                                                     = 0;
}



void
GalerkinFMMOperator::test_potentials(const std::vector<double> &values,
                                     const Vector<double>      &near_field,
                                     const double               sign,
                                     TrilinosWrappers::MPI::Vector &dst) const
{
  const unsigned int dpc = dof_handler->get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> dofs(dpc);
  std::vector<double>                  local(dpc);
  Vector<double>                       y(dof_handler->n_dofs());
  for (const unsigned int c : own_cells)
    {
      evaluator->reinit(c);
      for (unsigned int q = 0; q < n_q_points; ++q)
        evaluator->submit_value(values[c * n_q_points + q], q);
      evaluator->test_and_sum(local, EvaluationFlags::values);
      cells[c]->get_dof_indices(dofs);
      constraints.distribute_local_to_global(local, dofs, y);
    }
  sum_over_processes(y, mpi_communicator);
  if (near_field.size() > 0)
    y += near_field;
  for (const auto i : dst.locally_owned_elements())
    dst[i] = sign * y[i];
  dst.compress(VectorOperation::insert);
}



void
GalerkinFMMOperator::vmult(TrilinosWrappers::MPI::Vector       &dst,
                           const TrilinosWrappers::MPI::Vector &src) const
{
  const unsigned int dpc = dof_handler->get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> dofs(dpc);
  std::vector<double>                  local(dpc);

  // Density at every quadrature point: the charges of the point sum.
  auto           start = std::chrono::steady_clock::now();
  Vector<double> x(src);
  if (constraints.n_constraints() > 0)
    constraints.distribute(x);
  std::vector<double> charges(cells.size() * n_q_points);
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      cells[c]->get_dof_indices(dofs);
      for (unsigned int i = 0; i < dpc; ++i)
        local[i] = x[dofs[i]];
      evaluator->reinit(c);
      evaluator->evaluate(local, EvaluationFlags::values);
      for (unsigned int q = 0; q < n_q_points; ++q)
        charges[c * n_q_points + q] =
          inverse_4pi * JxW[c * n_q_points + q] * evaluator->get_value(q);
    }
  time_evaluate += seconds_since(start);

  start = std::chrono::steady_clock::now();
  std::vector<double> potentials;
  fmm.evaluate(charges, {}, potentials);
  time_fmm += seconds_since(start);

  start = std::chrono::steady_clock::now();
  for (unsigned int p = 0; p < potentials.size(); ++p)
    potentials[p] *= JxW[p];
  test_potentials(potentials, Vector<double>(), 1., dst);
  time_test += seconds_since(start);

  start = std::chrono::steady_clock::now();
  TrilinosWrappers::MPI::Vector correction(dst);
  near_correction.vmult(correction, src);
  dst += correction;
  time_near += seconds_since(start);
  ++n_vmult;
}



void
GalerkinFMMOperator::double_layer_rhs(TrilinosWrappers::MPI::Vector &rhs) const
{
  AssertThrow(!dirichlet_values.empty(),
              ExcMessage("The Dirichlet datum was not given to reinit()."));
  const auto                start = std::chrono::steady_clock::now();
  const unsigned int        n     = JxW.size();
  std::vector<Tensor<1, 3>> dipoles(n);
  std::vector<double>       with_datum, without_datum;
  for (unsigned int p = 0; p < n; ++p)
    dipoles[p] = inverse_4pi * JxW[p] * dirichlet_values[p] * normals[p];
  rhs_fmm.evaluate({}, dipoles, with_datum);
  if (regularize)
    {
      for (unsigned int p = 0; p < n; ++p)
        dipoles[p] = inverse_4pi * JxW[p] * normals[p];
      rhs_fmm.evaluate({}, dipoles, without_datum);
    }
  for (unsigned int p = 0; p < n; ++p)
    with_datum[p] =
      JxW[p] * (with_datum[p] -
                (regularize ? dirichlet_values[p] * without_datum[p] : 0.));
  test_potentials(with_datum, near_double_layer, -1., rhs);
  time_rhs += seconds_since(start);
}



void
GalerkinFMMOperator::manufactured_rhs(TrilinosWrappers::MPI::Vector &rhs) const
{
  AssertThrow(!flux_values.empty(),
              ExcMessage("The exact gradient was not given to reinit()."));
  const auto          start = std::chrono::steady_clock::now();
  const unsigned int  n     = JxW.size();
  std::vector<double> charges(n), potentials;
  for (unsigned int p = 0; p < n; ++p)
    charges[p] = inverse_4pi * JxW[p] * flux_values[p];
  fmm.evaluate(charges, {}, potentials);
  for (unsigned int p = 0; p < n; ++p)
    potentials[p] *= JxW[p];
  test_potentials(potentials, near_manufactured, 1., rhs);
  time_rhs += seconds_since(start);
}



void
GalerkinFMMOperator::print_statistics(std::ostream &out) const
{
  const unsigned long pairs =
    Utilities::MPI::sum(n_near_pairs, mpi_communicator);
  const unsigned int calls = std::max(1u, n_vmult);
  const double       te =
    Utilities::MPI::max(time_evaluate / calls, mpi_communicator);
  const double tf = Utilities::MPI::max(time_fmm / calls, mpi_communicator);
  const double tt = Utilities::MPI::max(time_test / calls, mpi_communicator);
  const double tn = Utilities::MPI::max(time_near / calls, mpi_communicator);
  const double tr = Utilities::MPI::max(time_rhs, mpi_communicator);
  const double sg = Utilities::MPI::max(setup_geometry, mpi_communicator);
  const double st = Utilities::MPI::max(setup_tree, mpi_communicator);
  const double sn = Utilities::MPI::max(setup_near, mpi_communicator);
  fmm.print_statistics(out);
  const auto precision = out.precision();
  if (this_mpi_process == 0)
    out << std::setprecision(3) << "Galerkin FMM: " << cells.size()
        << " cells, " << n_q_points << " points each; " << pairs
        << " near pairs corrected, " << near_exact.n_nonzero_elements()
        << " near-field entries\n"
        << "  setup: geometry " << sg << " s, trees " << st
        << " s, near field " << sn << " s (slowest process "
        << setup_near_imbalance << " times the mean)\n"
        << "  right-hand sides " << tr << " s\n"
        << "  seconds per product, slowest process (" << n_vmult
        << " products): density " << te << ", FMM " << tf << ", testing "
        << tt << ", near correction " << tn << '\n'
        << std::setprecision(precision);
}
