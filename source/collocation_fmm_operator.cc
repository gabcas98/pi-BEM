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

#include "../include/collocation_fmm_operator.h"

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parallel.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/rtree.h>

#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <ostream>

namespace
{
  double
  seconds_since(const std::chrono::steady_clock::time_point &start)
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
      .count();
  }

  constexpr double inverse_4pi = 1. / (4. * numbers::PI);
} // namespace



CollocationFMMOperator::CollocationFMMOperator(const MPI_Comm comm)
  : mpi_communicator(comm)
  , fmm(comm)
{}



void
CollocationFMMOperator::reinit(const DoFHandler<2, 3>           &dh,
                               const Mapping<2, 3>              &mapping,
                               const Quadrature<2>              &quadrature,
                               const std::vector<Point<3>>      &support_points,
                               const IndexSet                   &owned,
                               const double                      near_ratio,
                               const RowIntegrator              &rows,
                               const LaplaceFMM::AdditionalData &fmm_data)
{
  const FiniteElement<2, 3> &fe  = dh.get_fe();
  const unsigned int         dpc = fe.dofs_per_cell;
  n_dofs                         = dh.n_dofs();
  AssertDimension(support_points.size(), n_dofs);

  // Geometry at the points of the regular rule of every cell.
  auto                      start = std::chrono::steady_clock::now();
  std::vector<CellIterator> cells;
  cell_dofs.clear();
  for (const auto &cell : dh.active_cell_iterators())
    {
      cells.push_back(cell);
      cell_dofs.emplace_back(dpc);
      cell->get_dof_indices(cell_dofs.back());
    }
  n_q_points = quadrature.size();
  shape.reinit(n_q_points, dpc);
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int j = 0; j < dpc; ++j)
      shape(q, j) = fe.shape_value(j, quadrature.point(q));
  FEValues<2, 3> values(mapping,
                        fe,
                        quadrature,
                        update_quadrature_points | update_normal_vectors |
                          update_JxW_values);
  n_sources = cells.size() * n_q_points;
  std::vector<Point<3>> points(n_sources + n_dofs);
  JxW.resize(n_sources);
  normals.resize(n_sources);
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      values.reinit(cells[c]);
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const unsigned int p = c * n_q_points + q;
          points[p]            = values.quadrature_point(q);
          JxW[p]               = values.JxW(q);
          normals[p]           = values.normal_vector(q);
        }
    }
  for (types::global_dof_index i = 0; i < n_dofs; ++i)
    points[n_sources + i] = support_points[i];
  setup_geometry = seconds_since(start);

  // The support points are the only targets.
  start = std::chrono::steady_clock::now();
  std::vector<bool> targets(points.size(), false);
  std::fill(targets.begin() + n_sources, targets.end(), true);
  fmm.reinit(points, fmm_data, targets);
  setup_tree = seconds_since(start);

  // The cells within the reach of the quasi-singular rule of each own
  // support point: its distance from the centre of the cell is at most
  // near_ratio + 1 diameters, so the support point lies in the bounding box
  // of the cell enlarged by as much.
  start = std::chrono::steady_clock::now();
  std::vector<std::pair<BoundingBox<3>, unsigned int>> boxes;
  for (unsigned int c = 0; c < cells.size(); ++c)
    {
      BoundingBox<3> box = mapping.get_bounding_box(cells[c]);
      box.extend((near_ratio + 1.) * cells[c]->diameter());
      boxes.emplace_back(box, c);
    }
  const auto tree = pack_rtree(boxes);

  // Each row: the band entries of N and D, and their corrections.
  const std::vector<types::global_dof_index> own_rows(owned.begin(),
                                                      owned.end());
  std::vector<std::map<types::global_dof_index, std::array<double, 4>>> entries(
    own_rows.size());
  std::vector<unsigned long> row_pairs(own_rows.size(), 0),
    row_corrections(own_rows.size(), 0);
  const double kappa     = fmm_data.kappa;
  const auto   integrate = [&](const std::size_t begin, const std::size_t end) {
    Vector<double> exact_n(dpc), exact_d(dpc), point_n(dpc), point_d(dpc);
    std::vector<std::pair<BoundingBox<3>, unsigned int>> hits;
    for (std::size_t r = begin; r < end; ++r)
      {
        const types::global_dof_index i = own_rows[r];
        const Point<3>               &x = support_points[i];
        hits.clear();
        tree.query(boost::geometry::index::intersects(
                     BoundingBox<3>(std::make_pair(x, x))),
                   std::back_inserter(hits));
        for (const auto &hit : hits)
          {
            const unsigned int c = hit.second;
            const bool near = rows(cells[c], cell_dofs[c], i, exact_n, exact_d);
            // The regular rule, as the multipole method sums it: the kernel
            // of the dense assembly, without a point at the support point.
            point_n = 0;
            point_d = 0;
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const unsigned int p = c * n_q_points + q;
                const Tensor<1, 3> R = points[p] - x;
                const double       r = R.norm();
                if (r == 0.)
                  continue;
                const double exponential =
                  kappa == 0. ? 1. : std::exp(-kappa * r);
                const double single = inverse_4pi * exponential / r * JxW[p];
                const double dipole = -inverse_4pi * (R * normals[p]) *
                                      (1. + kappa * r) * exponential /
                                      (r * r * r) * JxW[p];
                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    point_n[j] += dipole * shape(q, j);
                    point_d[j] += single * shape(q, j);
                  }
              }
            auto &row = entries[r];
            for (unsigned int j = 0; j < dpc; ++j)
              {
                std::array<double, 4> &entry = row[cell_dofs[c][j]];
                if (near)
                  {
                    entry[0] += exact_n[j];
                    entry[1] += exact_d[j];
                    entry[2] += exact_n[j] - point_n[j];
                    entry[3] += exact_d[j] - point_d[j];
                  }
                else
                  {
                    entry[0] += point_n[j];
                    entry[1] += point_d[j];
                  }
              }
            ++row_pairs[r];
            row_corrections[r] += near;
          }
      }
  };
  parallel::apply_to_subranges(std::size_t(0), own_rows.size(), integrate, 16);

  TrilinosWrappers::SparsityPattern sparsity(owned, owned, mpi_communicator);
  n_band_pairs      = 0;
  n_corrected_pairs = 0;
  for (std::size_t r = 0; r < own_rows.size(); ++r)
    {
      for (const auto &[j, entry] : entries[r])
        sparsity.add(own_rows[r], j);
      n_band_pairs += row_pairs[r];
      n_corrected_pairs += row_corrections[r];
    }
  sparsity.compress();
  for (auto *matrix : {&correction_n, &correction_d, &band_n, &band_d})
    matrix->reinit(sparsity);
  for (std::size_t r = 0; r < own_rows.size(); ++r)
    for (const auto &[j, entry] : entries[r])
      {
        band_n.set(own_rows[r], j, entry[0]);
        band_d.set(own_rows[r], j, entry[1]);
        correction_n.set(own_rows[r], j, entry[2]);
        correction_d.set(own_rows[r], j, entry[3]);
      }
  for (auto *matrix : {&correction_n, &correction_d, &band_n, &band_d})
    matrix->compress(VectorOperation::insert);
  setup_near    = seconds_since(start);
  time_products = 0;
  n_products    = 0;
}



void
CollocationFMMOperator::vmult(
  TrilinosWrappers::MPI::Vector       &dst,
  const TrilinosWrappers::MPI::Vector &phi,
  const TrilinosWrappers::MPI::Vector &dphi_dn) const
{
  vmult({&dst}, {&phi}, {&dphi_dn});
}



void
CollocationFMMOperator::vmult(
  const std::vector<TrilinosWrappers::MPI::Vector *>       &dst,
  const std::vector<const TrilinosWrappers::MPI::Vector *> &phi,
  const std::vector<const TrilinosWrappers::MPI::Vector *> &dphi_dn) const
{
  const unsigned int k = dst.size();
  AssertThrow(phi.size() == k && dphi_dn.size() == k,
              ExcDimensionMismatch(phi.size(), k));
  const auto start = std::chrono::steady_clock::now();
  // The sources on all cells, from the values on all degrees of freedom:
  // the dipoles of phi and the charges of -dphi_dn.
  const unsigned int                     dpc = shape.n();
  std::vector<std::vector<double>>       charges(k), potentials;
  std::vector<std::vector<Tensor<1, 3>>> dipoles(k);
  for (unsigned int s = 0; s < k; ++s)
    {
      const Vector<double> phi_all(*phi[s]), dphi_dn_all(*dphi_dn[s]);
      charges[s].assign(n_sources + n_dofs, 0.);
      dipoles[s].assign(n_sources + n_dofs, Tensor<1, 3>());
      for (unsigned int c = 0; c < cell_dofs.size(); ++c)
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const unsigned int p     = c * n_q_points + q;
            double             value = 0, normal_derivative = 0;
            for (unsigned int j = 0; j < dpc; ++j)
              {
                value += shape(q, j) * phi_all[cell_dofs[c][j]];
                normal_derivative += shape(q, j) * dphi_dn_all[cell_dofs[c][j]];
              }
            charges[s][p] = -inverse_4pi * JxW[p] * normal_derivative;
            dipoles[s][p] = inverse_4pi * JxW[p] * value * normals[p];
          }
    }
  fmm.evaluate_combined_sets(charges, dipoles, potentials);

  for (unsigned int s = 0; s < k; ++s)
    {
      TrilinosWrappers::MPI::Vector &result = *dst[s];
      TrilinosWrappers::MPI::Vector  single(result), far(result);
      correction_n.vmult(result, *phi[s]);
      correction_d.vmult(single, *dphi_dn[s]);
      result -= single;
      for (const auto i : result.locally_owned_elements())
        far(i) = potentials[s][n_sources + i];
      far.compress(VectorOperation::insert);
      result += far;
    }
  time_products += seconds_since(start);
  n_products += k;
}



void
CollocationFMMOperator::print_statistics(std::ostream &out) const
{
  const double pairs =
    Utilities::MPI::sum(static_cast<double>(n_band_pairs), mpi_communicator);
  const double corrected =
    Utilities::MPI::sum(static_cast<double>(n_corrected_pairs),
                        mpi_communicator);
  const double near_time = Utilities::MPI::max(setup_near, mpi_communicator);
  const double product_time =
    Utilities::MPI::max(time_products, mpi_communicator);
  fmm.print_statistics(out);
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    out << "  collocation FMM: " << n_sources << " quadrature points, "
        << n_dofs << " support points; " << pairs
        << " pairs of a support point and a cell in the near band, of which "
        << corrected << " corrected; setup: geometry " << setup_geometry
        << " s, tree " << setup_tree << " s, near field " << near_time
        << " s (slowest process); " << n_products << " products, "
        << (n_products > 0 ? product_time / n_products : 0.)
        << " s each (slowest process)" << std::endl;
}
