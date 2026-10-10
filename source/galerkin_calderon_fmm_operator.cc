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

#include "../include/galerkin_calderon_fmm_operator.h"

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/rtree.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <map>
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



GalerkinCalderonFMMOperator::GalerkinCalderonFMMOperator(const MPI_Comm comm)
  : mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
  , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
  , charge_fmm(comm)
  , dipole_fmm(comm)
  , preconditioner_fmm(comm)
{}



void
GalerkinCalderonFMMOperator::reinit(
  const DoFHandler<2, 3>                                  &dh,
  const Mapping<2, 3>                                     &mapping,
  const std::vector<std::vector<types::global_dof_index>> &cell_indices,
  const types::global_dof_index                            n,
  const AffineConstraints<double>                         &constraints_in,
  const GalerkinPanelPairIntegrator<3>                    &integrator,
  const AdditionalData                                    &data)
{
  AssertThrow(integrator.get_screening() == 0.,
              ExcMessage("The mixed problem through the FMM supports the "
                         "Laplace kernel only."));
  n_indices = n;
  indices   = cell_indices;
  constraints.copy_from(constraints_in);
  const FiniteElement<2, 3> &fe = dh.get_fe();
  dofs_per_cell                 = fe.dofs_per_cell;
  const unsigned int dpc        = dofs_per_cell;

  // Geometry, shape functions and their surface curls at the quadrature
  // points of every cell.
  auto start = std::chrono::steady_clock::now();
  std::vector<DoFHandler<2, 3>::active_cell_iterator> cells;
  own_cells.clear();
  for (const auto &cell : dh.active_cell_iterators())
    {
      AssertDimension(cell->active_cell_index(), cells.size());
      if (cell->subdomain_id() == this_mpi_process)
        own_cells.push_back(cells.size());
      cells.push_back(cell);
    }
  n_cells = cells.size();
  AssertDimension(indices.size(), n_cells);
  const QGauss<2> quadrature(data.quadrature_order);
  n_q_points = quadrature.size();
  shape.reinit(n_q_points, dpc);
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int i = 0; i < dpc; ++i)
      shape(q, i) = fe.shape_value(i, quadrature.point(q));
  FEValues<2, 3> values(mapping,
                        fe,
                        quadrature,
                        update_gradients | update_normal_vectors |
                          update_quadrature_points | update_JxW_values);
  const unsigned int n_points = n_cells * n_q_points;
  points.resize(n_points);
  JxW.resize(n_points);
  normals.resize(n_points);
  shape_curls.resize(n_points * dpc);
  for (unsigned int c = 0; c < n_cells; ++c)
    {
      values.reinit(cells[c]);
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const unsigned int p = c * n_q_points + q;
          points[p]            = values.quadrature_point(q);
          JxW[p]               = values.JxW(q);
          normals[p]           = values.normal_vector(q);
          for (unsigned int i = 0; i < dpc; ++i)
            shape_curls[p * dpc + i] =
              cross_product_3d(normals[p], values.shape_grad(i, q));
        }
    }
  setup_geometry = seconds_since(start);

  start                               = std::chrono::steady_clock::now();
  LaplaceFMM::AdditionalData fmm_data = data.fmm;
  fmm_data.kappa                      = 0.;
  charge_fmm.reinit(points, fmm_data);
  separate_preconditioner =
    data.preconditioner_order > 0 || data.preconditioner_theta > 0;
  if (separate_preconditioner)
    {
      LaplaceFMM::AdditionalData coarse = fmm_data;
      if (data.preconditioner_order > 0)
        coarse.order = data.preconditioner_order;
      if (data.preconditioner_theta > 0)
        coarse.theta = data.preconditioner_theta;
      preconditioner_fmm.reinit(points, coarse);
    }
  fmm_data.order += data.dipole_order_increase;
  dipole_fmm.reinit(points, fmm_data);
  setup_tree = seconds_since(start);

  // The near pairs, found and dealt out by cost as in GalerkinFMMOperator:
  // candidates from bounding boxes enlarged by the ratio times their own
  // diagonal, each unordered pair once.
  start = std::chrono::steady_clock::now();
  std::vector<std::pair<BoundingBox<3>, unsigned int>> boxes;
  for (unsigned int c = 0; c < n_cells; ++c)
    {
      BoundingBox<3> box      = mapping.get_bounding_box(cells[c]);
      const double   diagonal = box.get_boundary_points().first.distance(
        box.get_boundary_points().second);
      box.extend(data.near_distance_ratio * diagonal + 1e-12 * diagonal);
      boxes.emplace_back(box, c);
    }
  const auto tree = pack_rtree(boxes);
  std::vector<std::array<unsigned int, 3>> found;
  for (unsigned int t = this_mpi_process; t < n_cells; t += n_mpi_processes)
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
            type == GalerkinPanelPairType::same_panel        ? 6 :
            type == GalerkinPanelPairType::common_edge ||
                type == GalerkinPanelPairType::nonconforming_edge ? 5 :
            type == GalerkinPanelPairType::common_vertex       ? 2 :
                                                                 1;
          found.push_back({{t, r, cost}});
        }
    }
  std::vector<std::array<unsigned int, 3>> all_pairs;
  for (const auto &part : Utilities::MPI::all_gather(mpi_communicator, found))
    all_pairs.insert(all_pairs.end(), part.begin(), part.end());
  std::sort(all_pairs.begin(), all_pairs.end(), [](const auto &x, const auto &y) {
    return x[2] != y[2] ? x[2] > y[2] : x[0] != y[0] ? x[0] < y[0] : x[1] < y[1];
  });
  std::vector<unsigned long>                           load(n_mpi_processes, 0);
  std::vector<std::pair<unsigned int, unsigned int>> my_pairs;
  n_near_pairs = 0;
  for (const auto &pair : all_pairs)
    {
      const unsigned int process =
        std::min_element(load.begin(), load.end()) - load.begin();
      load[process] += pair[2];
      if (process == this_mpi_process)
        {
          my_pairs.emplace_back(pair[0], pair[1]);
          n_near_pairs += pair[0] == pair[1] ? 1 : 2;
        }
    }

  // Both blocks of every pair, condensed.
  DynamicSparsityPattern dsp(n_indices, n_indices);
  for (const auto &[t, r] : my_pairs)
    {
      constraints.add_entries_local_to_global(indices[t], indices[r], dsp, true);
      constraints.add_entries_local_to_global(indices[r], indices[t], dsp, true);
    }
  near_sparsity.copy_from(dsp);
  near_single.reinit(near_sparsity);
  near_double.reinit(near_sparsity);
  near_hypersingular.reinit(near_sparsity);
  single_diagonal.reinit(n_indices);
  hypersingular_diagonal.reinit(n_indices);

  FullMatrix<double> kernel(n_q_points, n_q_points),
    dipole(n_q_points, n_q_points), dipole_back(n_q_points, n_q_points),
    partial(dpc, n_q_points), block(dpc, dpc), back(dpc, dpc),
    transposed(dpc, dpc);
  const auto contract = [&](const FullMatrix<double> &k,
                            FullMatrix<double>       &result) {
    // result = shape^T k shape
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int s = 0; s < n_q_points; ++s)
        {
          double sum = 0;
          for (unsigned int q = 0; q < n_q_points; ++q)
            sum += shape(q, i) * k(q, s);
          partial(i, s) = sum;
        }
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int j = 0; j < dpc; ++j)
        {
          double sum = 0;
          for (unsigned int s = 0; s < n_q_points; ++s)
            sum += partial(i, s) * shape(s, j);
          result(i, j) = sum;
        }
  };
  for (const auto &[t, r] : my_pairs)
    {
      const bool self = t == r;
      // Exact blocks: V, K and W of (t, r), and K of (r, t) from the same
      // points, transposed in a fourth block.
      const FullMatrix<double> exact =
        self ? integrator.integrate_single_double_hypersingular(
                 cells[t], cells[r], data.quadrature_order) :
               integrator.integrate_single_double_hypersingular_both(
                 cells[t], cells[r], data.quadrature_order);

      // The part of the point sums: the kernels of the pairs of points,
      // without the coincident points of a cell with itself.
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const unsigned int a = t * n_q_points + q;
          for (unsigned int s = 0; s < n_q_points; ++s)
            {
              const unsigned int b = r * n_q_points + s;
              if (a == b)
                {
                  kernel(q, s) = dipole(q, s) = dipole_back(s, q) = 0;
                  continue;
                }
              const Tensor<1, 3> x_minus_y = points[a] - points[b];
              const double       distance  = x_minus_y.norm();
              const double weight = inverse_4pi * JxW[a] * JxW[b] / distance;
              const double cube   = distance * distance;
              kernel(q, s)        = weight;
              dipole(q, s)        = weight * (x_minus_y * normals[b]) / cube;
              dipole_back(s, q)   = -weight * (x_minus_y * normals[a]) / cube;
            }
        }

      // Single layer.
      contract(kernel, block);
      for (unsigned int i = 0; i < dpc; ++i)
        for (unsigned int j = 0; j < dpc; ++j)
          block(i, j) = exact(i, j) - block(i, j);
      constraints.distribute_local_to_global(block,
                                             indices[t],
                                             indices[r],
                                             near_single);
      if (!self)
        {
          transposed.copy_transposed(block);
          constraints.distribute_local_to_global(transposed,
                                                 indices[r],
                                                 indices[t],
                                                 near_single);
        }
      // Double layer, both ways.
      contract(dipole, block);
      for (unsigned int i = 0; i < dpc; ++i)
        for (unsigned int j = 0; j < dpc; ++j)
          block(i, j) = exact(i, dpc + j) - block(i, j);
      constraints.distribute_local_to_global(block,
                                             indices[t],
                                             indices[r],
                                             near_double);
      if (!self)
        {
          contract(dipole_back, back);
          for (unsigned int j = 0; j < dpc; ++j)
            for (unsigned int i = 0; i < dpc; ++i)
              back(j, i) = exact(i, 3 * dpc + j) - back(j, i);
          constraints.distribute_local_to_global(back,
                                                 indices[r],
                                                 indices[t],
                                                 near_double);
        }
      // Hypersingular: the kernel block against the curls of both cells.
      block = 0;
      for (unsigned int d = 0; d < 3; ++d)
        {
          for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int s = 0; s < n_q_points; ++s)
              {
                double sum = 0;
                for (unsigned int q = 0; q < n_q_points; ++q)
                  sum += shape_curls[(t * n_q_points + q) * dpc + i][d] *
                         kernel(q, s);
                partial(i, s) = sum;
              }
          for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int j = 0; j < dpc; ++j)
              {
                double sum = 0;
                for (unsigned int s = 0; s < n_q_points; ++s)
                  sum += partial(i, s) *
                         shape_curls[(r * n_q_points + s) * dpc + j][d];
                block(i, j) += sum;
              }
        }
      for (unsigned int i = 0; i < dpc; ++i)
        for (unsigned int j = 0; j < dpc; ++j)
          block(i, j) = exact(i, 2 * dpc + j) - block(i, j);
      constraints.distribute_local_to_global(block,
                                             indices[t],
                                             indices[r],
                                             near_hypersingular);
      if (!self)
        {
          transposed.copy_transposed(block);
          constraints.distribute_local_to_global(transposed,
                                                 indices[r],
                                                 indices[t],
                                                 near_hypersingular);
        }

      // The diagonals of V and W, from the exact blocks, for the
      // preconditioner; an exchanged pair adds the same entry again.
      for (unsigned int i = 0; i < dpc; ++i)
        for (unsigned int j = 0; j < dpc; ++j)
          if (indices[t][i] == indices[r][j])
            {
              const double times = self ? 1. : 2.;
              single_diagonal[indices[t][i]] += times * exact(i, j);
              hypersingular_diagonal[indices[t][i]] +=
                times * exact(i, 2 * dpc + j);
            }
    }
  const double own_near_time = seconds_since(start);
  sum_over_processes(single_diagonal, mpi_communicator);
  sum_over_processes(hypersingular_diagonal, mpi_communicator);
  setup_near = seconds_since(start);
  setup_near_imbalance =
    Utilities::MPI::max(own_near_time, mpi_communicator) /
    std::max(1e-300,
             Utilities::MPI::sum(own_near_time, mpi_communicator) /
               n_mpi_processes);
  time_product = {{0, 0, 0, 0, 0, 0}};
  n_products   = {{0, 0, 0, 0, 0, 0}};
}



void
GalerkinCalderonFMMOperator::evaluate_field(
  const Vector<double>      &x,
  std::vector<double>       &values,
  std::vector<Tensor<1, 3>> *curls) const
{
  const unsigned int dpc = dofs_per_cell;
  values.assign(points.size(), 0.);
  if (curls != nullptr)
    curls->assign(points.size(), Tensor<1, 3>());
  std::vector<double> local(dpc);
  for (unsigned int c = 0; c < n_cells; ++c)
    {
      for (unsigned int i = 0; i < dpc; ++i)
        local[i] = x[indices[c][i]];
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const unsigned int p     = c * n_q_points + q;
          double             value = 0;
          for (unsigned int i = 0; i < dpc; ++i)
            value += shape(q, i) * local[i];
          values[p] = value;
          if (curls != nullptr)
            for (unsigned int i = 0; i < dpc; ++i)
              (*curls)[p] += local[i] * shape_curls[p * dpc + i];
        }
    }
}



void
GalerkinCalderonFMMOperator::test_values(const std::vector<double> &values,
                                         Vector<double>            &y) const
{
  const unsigned int  dpc = dofs_per_cell;
  Vector<double>      local(dpc);
  for (const unsigned int c : own_cells)
    {
      for (unsigned int i = 0; i < dpc; ++i)
        {
          double sum = 0;
          for (unsigned int q = 0; q < n_q_points; ++q)
            sum += shape(q, i) * values[c * n_q_points + q];
          local[i] = sum;
        }
      constraints.distribute_local_to_global(local, indices[c], y);
    }
}



void
GalerkinCalderonFMMOperator::vmult(const Operator        op,
                                   Vector<double>       &dst,
                                   const Vector<double> &src) const
{
  const auto         start = std::chrono::steady_clock::now();
  const unsigned int dpc   = dofs_per_cell;
  const unsigned int np    = points.size();
  Vector<double>     x(src);
  if (constraints.n_constraints() > 0)
    constraints.distribute(x);
  Vector<double>      y(n_indices);
  std::vector<double> values, potentials;

  switch (op)
    {
      case Operator::single_layer:
        {
          evaluate_field(x, values, nullptr);
          std::vector<double> charges(np);
          for (unsigned int p = 0; p < np; ++p)
            charges[p] = inverse_4pi * JxW[p] * values[p];
          charge_fmm.evaluate(charges, {}, potentials);
          for (unsigned int p = 0; p < np; ++p)
            potentials[p] *= JxW[p];
          test_values(potentials, y);
          near_single.vmult_add(y, src);
          (void)dpc;
          break;
        }
      case Operator::double_layer:
        {
          evaluate_field(x, values, nullptr);
          std::vector<Tensor<1, 3>> dipoles(np);
          for (unsigned int p = 0; p < np; ++p)
            dipoles[p] = inverse_4pi * JxW[p] * values[p] * normals[p];
          dipole_fmm.evaluate({}, dipoles, potentials);
          for (unsigned int p = 0; p < np; ++p)
            potentials[p] *= JxW[p];
          test_values(potentials, y);
          near_double.vmult_add(y, src);
          break;
        }
      case Operator::double_layer_transpose:
        {
          evaluate_field(x, values, nullptr);
          add_transpose_double_layer(values, y);
          near_double.Tvmult_add(y, src);
          break;
        }
      case Operator::hypersingular:
        {
          std::vector<Tensor<1, 3>> curls;
          evaluate_field(x, values, &curls);
          add_hypersingular(curls, y, charge_fmm);
          near_hypersingular.vmult_add(y, src);
          break;
        }
    }
  sum_over_processes(y, mpi_communicator);
  dst = y;
  time_product[static_cast<unsigned int>(op)] += seconds_since(start);
  ++n_products[static_cast<unsigned int>(op)];
}



void
GalerkinCalderonFMMOperator::add_transpose_double_layer(
  const std::vector<double> &values,
  Vector<double>            &y) const
{
  const unsigned int  np = points.size();
  std::vector<double> normal_component(np, 0.), potentials;
  if (std::getenv("PIBEM_CALDERON_DIPOLE_KT") != nullptr)
    {
      // The components of the field from three dipole sums, for
      // comparison.
      std::vector<Tensor<1, 3>> dipoles(np);
      for (unsigned int d = 0; d < 3; ++d)
        {
          for (unsigned int p = 0; p < np; ++p)
            {
              dipoles[p]    = Tensor<1, 3>();
              dipoles[p][d] = -inverse_4pi * JxW[p] * values[p];
            }
          dipole_fmm.evaluate({}, dipoles, potentials);
          for (unsigned int p = 0; p < np; ++p)
            normal_component[p] += normals[p][d] * potentials[p];
        }
    }
  else
    {
      // The gradient of the potential of the charges.
      std::vector<double>       charges(np);
      std::vector<Tensor<1, 3>> gradients;
      for (unsigned int p = 0; p < np; ++p)
        charges[p] = inverse_4pi * JxW[p] * values[p];
      charge_fmm.evaluate_with_gradient(charges, potentials, gradients);
      for (unsigned int p = 0; p < np; ++p)
        normal_component[p] = normals[p] * gradients[p];
    }
  for (unsigned int p = 0; p < np; ++p)
    normal_component[p] *= JxW[p];
  test_values(normal_component, y);
}



void
GalerkinCalderonFMMOperator::add_hypersingular(
  const std::vector<Tensor<1, 3>> &curls,
  Vector<double>                  &y,
  const LaplaceFMM                &fmm) const
{
  const unsigned int dpc = dofs_per_cell;
  const unsigned int np  = points.size();
  // The three components of the curl in one evaluation.
  std::vector<std::vector<double>> charges(3, std::vector<double>(np)),
    curl_potentials;
  for (unsigned int d = 0; d < 3; ++d)
    for (unsigned int p = 0; p < np; ++p)
      charges[d][p] = inverse_4pi * JxW[p] * curls[p][d];
  fmm.evaluate_multiple(charges, curl_potentials);
  Vector<double> local(dpc);
  for (const unsigned int c : own_cells)
    {
      local = 0;
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const unsigned int p = c * n_q_points + q;
          const Tensor<1, 3> potential({curl_potentials[0][p],
                                        curl_potentials[1][p],
                                        curl_potentials[2][p]});
          for (unsigned int i = 0; i < dpc; ++i)
            local[i] += JxW[p] * (shape_curls[p * dpc + i] * potential);
        }
      constraints.distribute_local_to_global(local, indices[c], y);
    }
}



void
GalerkinCalderonFMMOperator::vmult_rows(const Vector<double> &t,
                                        const Vector<double> &u,
                                        const double          k,
                                        Vector<double>       &dirichlet,
                                        Vector<double>       &neumann) const
{
  const auto         start = std::chrono::steady_clock::now();
  const unsigned int np    = points.size();
  Vector<double>     x_t(t), x_u(u);
  if (constraints.n_constraints() > 0)
    {
      constraints.distribute(x_t);
      constraints.distribute(x_u);
    }
  std::vector<double>       values_t, values_u, potentials;
  std::vector<Tensor<1, 3>> curls_u;
  evaluate_field(x_t, values_t, nullptr);
  evaluate_field(x_u, values_u, &curls_u);

  // Both rows in one vector, for one sum over the processes.
  Vector<double> rows(2 * n_indices);
  Vector<double> d_rows(n_indices), n_rows(n_indices);

  // V t and K^T t from the potential and the gradient of the charges of t,
  // K u from the dipoles of u.
  std::vector<double>       charges(np), values(np);
  std::vector<Tensor<1, 3>> dipoles(np), gradients;
  for (unsigned int p = 0; p < np; ++p)
    {
      charges[p] = inverse_4pi * JxW[p] * values_t[p];
      dipoles[p] = inverse_4pi * JxW[p] * values_u[p] * normals[p];
    }
  charge_fmm.evaluate_with_gradient(charges, potentials, gradients);
  for (unsigned int p = 0; p < np; ++p)
    values[p] = JxW[p] * potentials[p];
  test_values(values, d_rows);
  near_single.vmult_add(d_rows, t);
  Vector<double> transpose(n_indices);
  for (unsigned int p = 0; p < np; ++p)
    values[p] = JxW[p] * (normals[p] * gradients[p]);
  test_values(values, transpose);
  near_double.Tvmult_add(transpose, t);
  dipole_fmm.evaluate({}, dipoles, potentials);
  Vector<double> k_u(n_indices);
  for (unsigned int p = 0; p < np; ++p)
    values[p] = JxW[p] * potentials[p];
  test_values(values, k_u);
  near_double.vmult_add(k_u, u);
  d_rows.add(-k, k_u);

  // W u + k K^T t.
  add_hypersingular(curls_u, n_rows, charge_fmm);
  near_hypersingular.vmult_add(n_rows, u);
  n_rows.add(k, transpose);

  for (types::global_dof_index i = 0; i < n_indices; ++i)
    {
      rows[i]             = d_rows[i];
      rows[n_indices + i] = n_rows[i];
    }
  sum_over_processes(rows, mpi_communicator);
  dirichlet.reinit(n_indices);
  neumann.reinit(n_indices);
  for (types::global_dof_index i = 0; i < n_indices; ++i)
    {
      dirichlet[i] = rows[i];
      neumann[i]   = rows[n_indices + i];
    }
  time_product[4] += seconds_since(start);
  ++n_products[4];
}



void
GalerkinCalderonFMMOperator::apply(const Operator        op,
                                   Vector<double>       &dst,
                                   const Vector<double> &src,
                                   const LaplaceFMM     &fmm) const
{
  AssertThrow(op == Operator::single_layer || op == Operator::hypersingular,
              ExcMessage("V or W only."));
  const unsigned int np = points.size();
  Vector<double>     x(src);
  if (constraints.n_constraints() > 0)
    constraints.distribute(x);
  Vector<double>            y(n_indices);
  std::vector<double>       values, potentials;
  std::vector<Tensor<1, 3>> curls;
  if (op == Operator::single_layer)
    {
      evaluate_field(x, values, nullptr);
      std::vector<double> charges(np);
      for (unsigned int p = 0; p < np; ++p)
        charges[p] = inverse_4pi * JxW[p] * values[p];
      fmm.evaluate(charges, {}, potentials);
      for (unsigned int p = 0; p < np; ++p)
        potentials[p] *= JxW[p];
      test_values(potentials, y);
      near_single.vmult_add(y, src);
    }
  else
    {
      evaluate_field(x, values, &curls);
      add_hypersingular(curls, y, fmm);
      near_hypersingular.vmult_add(y, src);
    }
  sum_over_processes(y, mpi_communicator);
  dst = y;
}



void
GalerkinCalderonFMMOperator::vmult_preconditioner(
  const Operator        op,
  Vector<double>       &dst,
  const Vector<double> &src) const
{
  const auto start = std::chrono::steady_clock::now();
  apply(op, dst, src, separate_preconditioner ? preconditioner_fmm : charge_fmm);
  time_product[5] += seconds_since(start);
  ++n_products[5];
}



const Vector<double> &
GalerkinCalderonFMMOperator::diagonal(const Operator op) const
{
  AssertThrow(op == Operator::single_layer || op == Operator::hypersingular,
              ExcMessage("Only V and W have their diagonal stored."));
  return op == Operator::single_layer ? single_diagonal :
                                        hypersingular_diagonal;
}



void
GalerkinCalderonFMMOperator::print_statistics(std::ostream &out) const
{
  const unsigned long pairs =
    Utilities::MPI::sum(n_near_pairs, mpi_communicator);
  const unsigned long entries = Utilities::MPI::sum(
    static_cast<unsigned long>(near_sparsity.n_nonzero_elements()),
    mpi_communicator);
  std::array<double, 6> mean;
  for (unsigned int k = 0; k < 6; ++k)
    mean[k] = Utilities::MPI::max(time_product[k] /
                                    std::max(1u, n_products[k]),
                                  mpi_communicator);
  const double sg = Utilities::MPI::max(setup_geometry, mpi_communicator);
  const double st = Utilities::MPI::max(setup_tree, mpi_communicator);
  const double sn = Utilities::MPI::max(setup_near, mpi_communicator);
  charge_fmm.print_statistics(out);
  dipole_fmm.print_statistics(out);
  if (separate_preconditioner)
    preconditioner_fmm.print_statistics(out);
  const auto   precision = out.precision();
  if (this_mpi_process == 0)
    out << std::setprecision(3) << "Galerkin Calderon FMM: " << n_cells
        << " cells, " << n_q_points << " points each; " << pairs
        << " near pairs corrected, " << entries
        << " near-field entries per operator\n"
        << "  setup: geometry " << sg << " s, trees " << st
        << " s, near field " << sn << " s (slowest process "
        << setup_near_imbalance << " times the mean)\n"
        << "  seconds per product, slowest process: V " << mean[0] << " ("
        << n_products[0] << "), K " << mean[1] << " (" << n_products[1]
        << "), K^T " << mean[2] << " (" << n_products[2] << "), W "
        << mean[3] << " (" << n_products[3] << "), both rows of the mixed "
        << "problem " << mean[4] << " (" << n_products[4]
        << "), preconditioner V or W " << mean[5] << " (" << n_products[5]
        << ")\n"
        << std::setprecision(precision);
}
