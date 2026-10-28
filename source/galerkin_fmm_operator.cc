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

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/rtree.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <ostream>
#include <set>

#include "../include/cellwise_function.h"

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
  , rhs_laplace_fmm(comm)
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
  regularize = rhs.regularize;
  kappa      = integrator.get_screening();
  AssertThrow(!integrator.is_helmholtz(),
              ExcMessage("The FMM has the expansions of the Laplace and the "
                         "screened kernel only."));
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
  n_q_points = quadrature.size();
  // The cells are stored for the mappings that FEPointEvaluation evaluates
  // on the cell, as MappingFEField (Mapping Type = FE).
  mapping_info = std::make_unique<NonMatching::MappingInfo<2, 3>>(
    mapping,
    update_values | update_JxW_values | update_quadrature_points |
      update_normal_vectors,
    NonMatching::MappingInfo<2, 3>::AdditionalData(false, true));
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
  quadrature_points = points;
  dirichlet_values.clear();
  if (rhs.dirichlet != nullptr)
    {
      // The datum of the slot of the cell of every point.
      dirichlet_values.resize(n_points);
      for (unsigned int p = 0; p < n_points; ++p)
        dirichlet_values[p] =
          on_cell(*rhs.dirichlet, cells[p / n_q_points]).value(points[p]);
    }
  flux_values.clear();
  if (rhs.gradient != nullptr)
    {
      flux_values.resize(n_points);
      Vector<double> gradient(3);
      for (unsigned int p = 0; p < n_points; ++p)
        {
          on_cell(*rhs.gradient, cells[p / n_q_points])
            .vector_value(points[p], gradient);
          flux_values[p] = gradient[0] * normals[p][0] +
                           gradient[1] * normals[p][1] +
                           gradient[2] * normals[p][2];
        }
    }
  setup_geometry = seconds_since(start);

  start                               = std::chrono::steady_clock::now();
  LaplaceFMM::AdditionalData fmm_data = data.fmm;
  fmm_data.kappa                      = kappa;
  fmm.reinit(points, fmm_data);
  if (rhs.dirichlet != nullptr)
    {
      LaplaceFMM::AdditionalData rhs_data = fmm_data;
      rhs_data.order += data.rhs_order_increase;
      rhs_fmm.reinit(points, rhs_data);
      if (kappa > 0 && rhs.regularize)
        {
          rhs_data.kappa = 0.;
          rhs_laplace_fmm.reinit(points, rhs_data);
        }
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
            type == GalerkinPanelPairType::same_panel ? 6 :
            type == GalerkinPanelPairType::common_edge ||
                type == GalerkinPanelPairType::nonconforming_edge ?
                                                        5 :
            type == GalerkinPanelPairType::common_vertex ? 2 :
                                                           1;
          found.push_back({{t, r, cost}});
        }
    }

  // Deal all pairs out by cost, the most expensive first to the least
  // loaded process: the singular pairs cluster where the mesh is graded.
  std::vector<std::array<unsigned int, 3>> all_pairs;
  for (const auto &part : Utilities::MPI::all_gather(mpi_communicator, found))
    all_pairs.insert(all_pairs.end(), part.begin(), part.end());
  std::sort(all_pairs.begin(),
            all_pairs.end(),
            [](const auto &x, const auto &y) {
              return x[2] != y[2] ? x[2] > y[2] :
                     x[0] != y[0] ? x[0] < y[0] :
                                    x[1] < y[1];
            });
  std::vector<unsigned long>                        load(n_mpi_processes, 0);
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
  reference_shapes.reinit(n_q_points, dpc);
  for (unsigned int q = 0; q < n_q_points; ++q)
    for (unsigned int i = 0; i < dpc; ++i)
      reference_shapes(q, i) = fe.shape_value(i, quadrature.point(q));
  const FullMatrix<double> &shape = reference_shapes;
  computed_pairs.clear();
  computed_blocks.clear();
  patch_dofs.clear();
  patch_inverses.clear();

  FullMatrix<double> exact(dpc, dpc), correction(dpc, dpc),
    transposed(dpc, dpc), kernel(n_q_points, n_q_points),
    partial(dpc, n_q_points);
  Vector<double> exact_test_rhs(dpc), exact_trial_rhs(dpc), local(dpc);
  // The consistent right-hand side from the same traversal as the single
  // layer and the double layer, when the integrator carries the gradient.
  const bool fused_flux = rhs.gradient != nullptr && rhs.dirichlet != nullptr &&
                          integrator.get_flux_function() == rhs.gradient;
  Vector<double>      exact_test_flux(dpc), exact_trial_flux(dpc);
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
                  if (fused_flux)
                    {
                      exact_test_flux[i] = both(i, self ? dpc + 1 : dpc + 2);
                      if (!self)
                        exact_trial_flux[i] = both(i, dpc + 3);
                    }
                }
            }
          else
            exact = integrator.integrate_single_layer(cells[t],
                                                      cells[r],
                                                      data.quadrature_order);
          computed_pairs.emplace_back(t, r);
          for (unsigned int i = 0; i < dpc; ++i)
            for (unsigned int j = 0; j < dpc; ++j)
              computed_blocks.push_back(exact(i, j));

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
                  const double       screening = std::exp(-kappa * distance);
                  kernel(q, s)                 = weight * screening / distance;
                  if (rhs.dirichlet != nullptr)
                    {
                      // The Laplace dipole against E(r) phi(y) - phi(x),
                      // E = (1 + kappa r) e^{-kappa r}, as in the dense
                      // assembly and in double_layer_rhs().
                      const double cube = distance * distance * distance;
                      const double E    = (1. + kappa * distance) * screening;
                      test_side[q] +=
                        weight * (x_minus_y * normals[b]) / cube *
                        (E * dirichlet_values[b] -
                         (rhs.regularize ? dirichlet_values[a] : 0.));
                      if (!self)
                        trial_side[s] -=
                          weight * (x_minus_y * normals[a]) / cube *
                          (E * dirichlet_values[a] -
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
              // From the fused traversal, or else each direction with its
              // own.
              local =
                fused_flux ?
                  exact_test_flux :
                  integrator.integrate_single_layer_flux(cells[t],
                                                         cells[r],
                                                         data.quadrature_order,
                                                         *rhs.gradient);
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
                  local = fused_flux ? exact_trial_flux :
                                       integrator.integrate_single_layer_flux(
                                         cells[r],
                                         cells[t],
                                         data.quadrature_order,
                                         *rhs.gradient);
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
GalerkinFMMOperator::setup_patch_preconditioner()
{
  const auto         start = std::chrono::steady_clock::now();
  const unsigned int dpc   = dof_handler->get_fe().dofs_per_cell;

  // The exact blocks of all near pairs, from all processes, gathered as
  // plain arrays (a serializing gather of these megabytes takes seconds).
  std::vector<unsigned int> my_pairs;
  for (const auto &[t, r] : computed_pairs)
    {
      my_pairs.push_back(t);
      my_pairs.push_back(r);
    }
  auto gather = [&](const auto &mine, const MPI_Datatype type) {
    using T                = typename std::decay_t<decltype(mine)>::value_type;
    const int        count = mine.size();
    std::vector<int> counts(n_mpi_processes), offsets(n_mpi_processes, 0);
    int              ierr = MPI_Allgather(
      &count, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi_communicator);
    AssertThrowMPI(ierr);
    for (unsigned int p = 1; p < n_mpi_processes; ++p)
      offsets[p] = offsets[p - 1] + counts[p - 1];
    std::vector<T> all(offsets.back() + counts.back());
    ierr = MPI_Allgatherv(mine.data(),
                          count,
                          type,
                          all.data(),
                          counts.data(),
                          offsets.data(),
                          type,
                          mpi_communicator);
    AssertThrowMPI(ierr);
    return all;
  };
  const std::vector<unsigned int> all_pairs = gather(my_pairs, MPI_UNSIGNED);
  const std::vector<double> all_blocks = gather(computed_blocks, MPI_DOUBLE);
  std::map<std::pair<unsigned int, unsigned int>, const double *> exact;
  for (unsigned int k = 0; 2 * k < all_pairs.size(); ++k)
    exact[{all_pairs[2 * k], all_pairs[2 * k + 1]}] =
      &all_blocks[k * dpc * dpc];

  // The single-layer block of a pair of cells: exact from the near field,
  // or from the point rule, the rule of the dense path for a far pair.
  FullMatrix<double> kernel(n_q_points, n_q_points), partial(dpc, n_q_points);
  const double       inverse_4pi = 1. / (4. * numbers::PI);
  auto               block       = [&](const unsigned int  a,
                   const unsigned int  b,
                   FullMatrix<double> &result) {
    result.reinit(dpc, dpc);
    const auto hit = exact.find({std::min(a, b), std::max(a, b)});
    if (hit != exact.end())
      {
        for (unsigned int i = 0; i < dpc; ++i)
          for (unsigned int j = 0; j < dpc; ++j)
            result(i, j) =
              a <= b ? hit->second[i * dpc + j] : hit->second[j * dpc + i];
        return;
      }
    for (unsigned int q = 0; q < n_q_points; ++q)
      for (unsigned int s = 0; s < n_q_points; ++s)
        {
          const unsigned int x = a * n_q_points + q, y = b * n_q_points + s;
          const double       distance =
            quadrature_points[x].distance(quadrature_points[y]);
          kernel(q, s) = inverse_4pi * JxW[x] * JxW[y] *
                         std::exp(-kappa * distance) / distance;
        }
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int s = 0; s < n_q_points; ++s)
        {
          double sum = 0;
          for (unsigned int q = 0; q < n_q_points; ++q)
            sum += reference_shapes(q, i) * kernel(q, s);
          partial(i, s) = sum;
        }
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int j = 0; j < dpc; ++j)
        {
          double sum = 0;
          for (unsigned int s = 0; s < n_q_points; ++s)
            sum += partial(i, s) * reference_shapes(s, j);
          result(i, j) = sum;
        }
  };

  // A local unknown expands into free unknowns: itself, or the masters of
  // its constraint with their weights; computed once per cell.
  std::vector<
    std::vector<std::vector<std::pair<types::global_dof_index, double>>>>
                                       expansion(cells.size());
  std::vector<types::global_dof_index> dofs(dpc);
  auto expand = [&](const unsigned int c) -> const auto & {
    auto &e = expansion[c];
    if (e.empty())
      {
        cells[c]->get_dof_indices(dofs);
        e.resize(dpc);
        for (unsigned int i = 0; i < dpc; ++i)
          if (!constraints.is_constrained(dofs[i]))
            e[i].emplace_back(dofs[i], 1.);
          else
            for (const auto &entry :
                 *constraints.get_constraint_entries(dofs[i]))
              e[i].push_back(entry);
      }
    return e;
  };

  // The blocks of the pairs, each computed once although the patches
  // overlap.
  std::map<std::pair<unsigned int, unsigned int>, FullMatrix<double>> blocks;
  auto pair_block = [&](const unsigned int a,
                        const unsigned int b) -> const FullMatrix<double> & {
    auto hit = blocks.find({a, b});
    if (hit == blocks.end())
      {
        hit = blocks.emplace(std::make_pair(a, b), FullMatrix<double>()).first;
        block(a, b, hit->second);
      }
    return hit->second;
  };

  // The support of every free unknown: the cells with a local unknown that
  // expands into it. A patch keeps only the unknowns whose support it
  // contains, for which its block is complete: with continuous elements an
  // unknown on the rim of a patch also lives in cells outside it, and the
  // block would miss their share, lose the property of a principal
  // submatrix and, as it turned out, its definiteness.
  std::vector<unsigned int>            support_size(dof_handler->n_dofs(), 0);
  std::vector<types::global_dof_index> cell_support;
  auto                                 support_of = [&](const unsigned int c) {
    cell_support.clear();
    for (const auto &terms : expand(c))
      for (const auto &term : terms)
        cell_support.push_back(term.first);
    std::sort(cell_support.begin(), cell_support.end());
    cell_support.erase(std::unique(cell_support.begin(), cell_support.end()),
                       cell_support.end());
    return cell_support;
  };
  std::vector<std::vector<unsigned int>> support(dof_handler->n_dofs());
  for (unsigned int c = 0; c < cells.size(); ++c)
    for (const auto f : support_of(c))
      {
        ++support_size[f];
        support[f].push_back(c);
      }

  const auto vertex_cells =
    GridTools::vertex_to_cell_map(dof_handler->get_triangulation());
  patch_dofs.resize(own_cells.size());
  patch_inverses.resize(own_cells.size());
  std::vector<std::vector<std::vector<std::pair<int, double>>>> local;
  for (unsigned int k = 0; k < own_cells.size(); ++k)
    {
      // The cells around the vertices of the cell, and those that share an
      // unknown with it, directly or through a constraint: then the
      // unknowns of the cell are always in its patch.
      const unsigned int     c = own_cells[k];
      std::set<unsigned int> patch_set;
      for (const unsigned int v : cells[c]->vertex_indices())
        for (const auto &other : vertex_cells[cells[c]->vertex_index(v)])
          patch_set.insert(other->active_cell_index());
      for (const auto f : support_of(c))
        patch_set.insert(support[f].begin(), support[f].end());
      const std::vector<unsigned int> patch_cells(patch_set.begin(),
                                                  patch_set.end());

      // The free unknowns of the patch, and where each unknown of each
      // cell goes among them.
      std::vector<types::global_dof_index>           &free = patch_dofs[k];
      std::map<types::global_dof_index, unsigned int> in_patch;
      for (const unsigned int a : patch_cells)
        for (const auto f : support_of(a))
          ++in_patch[f];
      free.clear();
      for (const auto &[f, count] : in_patch)
        if (count == support_size[f])
          free.push_back(f);
      local.assign(patch_cells.size(), {});
      for (unsigned int a = 0; a < patch_cells.size(); ++a)
        {
          local[a].resize(dpc);
          const auto &e = expand(patch_cells[a]);
          for (unsigned int i = 0; i < dpc; ++i)
            for (const auto &[f, w] : e[i])
              {
                const auto it = std::lower_bound(free.begin(), free.end(), f);
                if (it != free.end() && *it == f)
                  local[a][i].emplace_back(it - free.begin(), w);
              }
        }

      // The condensed operator on them, pair of cells by pair of cells.
      FullMatrix<double> patch(free.size(), free.size());
      for (unsigned int a = 0; a < patch_cells.size(); ++a)
        for (unsigned int b = 0; b < patch_cells.size(); ++b)
          {
            const FullMatrix<double> &v =
              pair_block(patch_cells[a], patch_cells[b]);
            for (unsigned int i = 0; i < dpc; ++i)
              for (const auto &[row, wi] : local[a][i])
                for (unsigned int j = 0; j < dpc; ++j)
                  for (const auto &[column, wj] : local[b][j])
                    patch(row, column) += wi * wj * v(i, j);
          }
      // Gauss-Jordan in place: FullMatrix::invert() may call a threaded
      // LAPACK, which on eight processes oversubscribes the cores and makes
      // an 81 x 81 inverse take 20 ms instead of 0.1.
      patch.gauss_jordan();
      patch_inverses[k] = std::move(patch);
    }
  computed_pairs.clear();
  computed_blocks.clear();
  setup_patches = seconds_since(start);
}



void
GalerkinFMMOperator::apply_patch_preconditioner(
  TrilinosWrappers::MPI::Vector       &dst,
  const TrilinosWrappers::MPI::Vector &src) const
{
  const Vector<double> x(src);
  Vector<double>       y(x.size());
  Vector<double>       local_x, local_y;
  for (unsigned int k = 0; k < patch_dofs.size(); ++k)
    {
      const auto &free = patch_dofs[k];
      local_x.reinit(free.size());
      local_y.reinit(free.size());
      for (unsigned int i = 0; i < free.size(); ++i)
        local_x[i] = x[free[i]];
      patch_inverses[k].vmult(local_y, local_x);
      for (unsigned int i = 0; i < free.size(); ++i)
        y[free[i]] += local_y[i];
    }
  sum_over_processes(y, mpi_communicator);
  for (const auto i : dst.locally_owned_elements())
    dst[i] = constraints.is_constrained(i) ? x[i] : y[i];
  dst.compress(VectorOperation::insert);
}



void
GalerkinFMMOperator::test_potentials(const std::vector<double>     &values,
                                     const Vector<double>          &near_field,
                                     const double                   sign,
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
      // The regularization uses the Laplace kernel, screened or not.
      for (unsigned int p = 0; p < n; ++p)
        dipoles[p] = inverse_4pi * JxW[p] * normals[p];
      (kappa > 0 ? rhs_laplace_fmm : rhs_fmm)
        .evaluate({}, dipoles, without_datum);
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
  const double sp = Utilities::MPI::max(setup_patches, mpi_communicator);
  fmm.print_statistics(out);
  const auto precision = out.precision();
  if (this_mpi_process == 0)
    out << std::setprecision(3) << "Galerkin FMM: " << cells.size()
        << " cells, " << n_q_points << " points each; " << pairs
        << " near pairs corrected, " << near_exact.n_nonzero_elements()
        << " near-field entries\n"
        << "  setup: geometry " << sg << " s, trees " << st << " s, near field "
        << sn << " s (slowest process " << setup_near_imbalance
        << " times the mean), patches " << sp << " s\n"
        << "  right-hand sides " << tr << " s\n"
        << "  seconds per product, slowest process (" << n_vmult
        << " products): density " << te << ", FMM " << tf << ", testing " << tt
        << ", near correction " << tn << '\n'
        << std::setprecision(precision);
}
