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
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/rtree.h>

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
} // namespace



GalerkinFMMOperator::GalerkinFMMOperator(const MPI_Comm comm)
  : mpi_communicator(comm)
  , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
  , fmm(comm)
{}



void
GalerkinFMMOperator::reinit(const DoFHandler<2, 3>               &dh,
                            const Mapping<2, 3>                  &mapping,
                            const AffineConstraints<double>      &constraints_in,
                            const GalerkinPanelPairIntegrator<3> &integrator,
                            const IndexSet &locally_owned_dofs,
                            const AdditionalData &data)
{
  dof_handler = &dh;
  constraints.copy_from(constraints_in);
  const FiniteElement<2, 3> &fe  = dh.get_fe();
  const unsigned int         dpc = fe.dofs_per_cell;

  // Geometry of the quadrature points of every cell.
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
  n_q_points = quadrature.size();
  mapping_info =
    std::make_unique<NonMatching::MappingInfo<2, 3>>(mapping,
                                                     update_values |
                                                       update_JxW_values |
                                                       update_quadrature_points);
  std::vector<Triangulation<2, 3>::cell_iterator> tria_cells(cells.begin(),
                                                             cells.end());
  mapping_info->reinit_cells(tria_cells,
                             std::vector<Quadrature<2>>(cells.size(),
                                                        quadrature));
  evaluator = std::make_unique<FEPointEvaluation<1, 2, 3>>(*mapping_info, fe);
  std::vector<Point<3>> points(cells.size() * n_q_points);
  JxW.resize(points.size());
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      evaluator->reinit(c);
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          points[c * n_q_points + q] = evaluator->quadrature_point(q);
          JxW[c * n_q_points + q]    = evaluator->JxW(q);
        }
    }
  setup_geometry = seconds_since(start);

  start = std::chrono::steady_clock::now();
  fmm.reinit(points, data.fmm);
  setup_tree = seconds_since(start);

  // Candidate pairs: cells whose bounding boxes, each enlarged by the ratio
  // times its own diagonal, intersect. Any pair within the ratio times the
  // larger diameter is among them; the integrator then decides.
  start = std::chrono::steady_clock::now();
  std::vector<std::pair<BoundingBox<3>, unsigned int>> boxes;
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      BoundingBox<3> box = mapping.get_bounding_box(cells[c]);
      const double   diagonal =
        box.get_boundary_points().first.distance(
          box.get_boundary_points().second);
      box.extend(data.near_distance_ratio * diagonal + 1e-12 * diagonal);
      boxes.emplace_back(box, c);
    }
  const auto tree = pack_rtree(boxes);

  std::vector<std::vector<unsigned int>> near_trial_cells(own_cells.size());
  for (unsigned int k = 0; k < own_cells.size(); ++k)
    {
      const unsigned int                                   t = own_cells[k];
      std::vector<std::pair<BoundingBox<3>, unsigned int>> hits;
      tree.query(boost::geometry::index::intersects(boxes[t].first),
                 std::back_inserter(hits));
      for (const auto &hit : hits)
        if (!integrator.is_far_regular_pair(cells[t], cells[hit.second]))
          near_trial_cells[k].push_back(hit.second);
      n_near_pairs += near_trial_cells[k].size();
    }

  // Sparsity of the near field, condensed like the operator. A continuous
  // element writes into rows of other processes, and constraints into the
  // rows of the masters.
  IndexSet                             writable_rows(dh.n_dofs());
  std::vector<types::global_dof_index> test_dofs(dpc), trial_dofs(dpc);
  writable_rows.add_indices(locally_owned_dofs);
  for (const unsigned int t : own_cells)
    {
      cells[t]->get_dof_indices(test_dofs);
      for (const auto i : test_dofs)
        {
          writable_rows.add_index(i);
          if (constraints.is_constrained(i))
            for (const auto &entry : *constraints.get_constraint_entries(i))
              writable_rows.add_index(entry.first);
        }
    }
  writable_rows.compress();
  TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                             locally_owned_dofs,
                                             writable_rows,
                                             mpi_communicator);
  for (unsigned int k = 0; k < own_cells.size(); ++k)
    {
      cells[own_cells[k]]->get_dof_indices(test_dofs);
      for (const unsigned int r : near_trial_cells[k])
        {
          cells[r]->get_dof_indices(trial_dofs);
          constraints.add_entries_local_to_global(test_dofs, trial_dofs, sparsity, true);
        }
    }
  sparsity.compress();
  near_exact.reinit(sparsity);
  near_correction.reinit(sparsity);

  // Shape values at the reference quadrature points, the same on every
  // cell for a Lagrange element.
  FullMatrix<double> shape(n_q_points, dpc);
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int i = 0; i < dpc; ++i)
      shape(q, i) = fe.shape_value(i, quadrature.point(q));

  const double       inverse_4pi = 1. / (4. * numbers::PI);
  FullMatrix<double> exact, point_rule(dpc, dpc), kernel(n_q_points,
                                                          n_q_points),
    partial(dpc, n_q_points);
  for (unsigned int k = 0; k < own_cells.size(); ++k)
    {
      const unsigned int t = own_cells[k];
      cells[t]->get_dof_indices(test_dofs);
      for (const unsigned int r : near_trial_cells[k])
        {
          cells[r]->get_dof_indices(trial_dofs);
          exact = integrator.integrate_single_layer(cells[t],
                                                    cells[r],
                                                    data.quadrature_order);

          // The block that the point sum contains, without the coincident
          // points of a cell with itself.
          for (unsigned int q = 0; q < n_q_points; ++q)
            for (unsigned int s = 0; s < n_q_points; ++s)
              {
                const unsigned int a = t * n_q_points + q;
                const unsigned int b = r * n_q_points + s;
                kernel(q, s) =
                  a == b ? 0. :
                           inverse_4pi * JxW[a] * JxW[b] /
                             points[a].distance(points[b]);
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
                point_rule(i, j) = sum;
              }

          constraints.distribute_local_to_global(exact,
                                       test_dofs,
                                       trial_dofs,
                                       near_exact);
          FullMatrix<double> correction(exact);
          correction.add(-1., point_rule);
          constraints.distribute_local_to_global(correction,
                                       test_dofs,
                                       trial_dofs,
                                       near_correction);
        }
    }
  near_exact.compress(VectorOperation::add);
  near_correction.compress(VectorOperation::add);
  setup_near = seconds_since(start);

  time_evaluate = time_fmm = time_test = time_near = 0;
  n_vmult                                        = 0;
}



void
GalerkinFMMOperator::vmult(TrilinosWrappers::MPI::Vector       &dst,
                           const TrilinosWrappers::MPI::Vector &src) const
{
  const FiniteElement<2, 3> &fe  = dof_handler->get_fe();
  const unsigned int         dpc = fe.dofs_per_cell;
  std::vector<types::global_dof_index> dofs(dpc);
  std::vector<double>                  local(dpc);

  // Density at every quadrature point: the charges of the point sum.
  auto           start = std::chrono::steady_clock::now();
  Vector<double> x(src);
  if (constraints.n_constraints() > 0)
    constraints.distribute(x);
  const double        inverse_4pi = 1. / (4. * numbers::PI);
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

  // Potential tested against the shape functions of the own cells,
  // condensed onto the masters, then summed over the processes.
  start = std::chrono::steady_clock::now();
  Vector<double> y(dof_handler->n_dofs());
  for (const unsigned int c : own_cells)
    {
      evaluator->reinit(c);
      for (unsigned int q = 0; q < n_q_points; ++q)
        evaluator->submit_value(potentials[c * n_q_points + q] *
                                  JxW[c * n_q_points + q],
                                q);
      evaluator->test_and_sum(local, EvaluationFlags::values);
      cells[c]->get_dof_indices(dofs);
      constraints.distribute_local_to_global(local, dofs, y);
    }
  Utilities::MPI::sum(ArrayView<const double>(y.begin(), y.size()),
                      mpi_communicator,
                      ArrayView<double>(y.begin(), y.size()));
  for (const auto i : dst.locally_owned_elements())
    dst[i] = y[i];
  dst.compress(VectorOperation::insert);
  time_test += seconds_since(start);

  start = std::chrono::steady_clock::now();
  TrilinosWrappers::MPI::Vector correction(dst);
  near_correction.vmult(correction, src);
  dst += correction;
  time_near += seconds_since(start);
  ++n_vmult;
}



void
GalerkinFMMOperator::print_statistics(std::ostream &out) const
{
  const unsigned long pairs =
    Utilities::MPI::sum(n_near_pairs, mpi_communicator);
  const unsigned int  calls = std::max(1u, n_vmult);
  const double        te =
    Utilities::MPI::max(time_evaluate / calls, mpi_communicator);
  const double tf = Utilities::MPI::max(time_fmm / calls, mpi_communicator);
  const double tt = Utilities::MPI::max(time_test / calls, mpi_communicator);
  const double tn = Utilities::MPI::max(time_near / calls, mpi_communicator);
  const double sg = Utilities::MPI::max(setup_geometry, mpi_communicator);
  const double st = Utilities::MPI::max(setup_tree, mpi_communicator);
  const double sn = Utilities::MPI::max(setup_near, mpi_communicator);
  fmm.print_statistics(out);
  const auto precision = out.precision();
  if (this_mpi_process == 0)
    out << std::setprecision(3) << "Galerkin FMM: " << cells.size() << " cells, " << n_q_points
        << " points each; " << pairs << " near pairs corrected, "
        << near_exact.n_nonzero_elements() << " near-field entries\n"
        << "  setup: geometry " << sg << " s, tree " << st
        << " s, near field " << sn << " s\n"
        << "  seconds per product, slowest process (" << n_vmult
        << " products): density " << te << ", FMM " << tf << ", testing "
        << tt << ", near correction " << tn << '\n'
        << std::setprecision(precision);
}
