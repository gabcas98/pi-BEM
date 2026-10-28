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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <set>
#include <unordered_map>

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
  const AdditionalData                                    &data,
  NearFieldCache                                          *cache)
{
  kappa = integrator.get_screening();
  AssertThrow(!integrator.is_helmholtz(),
              ExcMessage("The FMM has the expansions of the Laplace and the "
                         "screened kernel only."));
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
  FEValues<2, 3>     values(mapping,
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
  fmm_data.kappa                      = kappa;
  charge_fmm.reinit(points, fmm_data);
  separate_preconditioner = data.preconditioner_order > 0 ||
                            data.preconditioner_theta > 0 ||
                            data.preconditioner_points_per_leaf > 0;
  LaplaceFMM::AdditionalData coarse = fmm_data;
  if (separate_preconditioner)
    {
      if (data.preconditioner_order > 0)
        coarse.order = data.preconditioner_order;
      if (data.preconditioner_theta > 0)
        coarse.theta = data.preconditioner_theta;
      if (data.preconditioner_points_per_leaf > 0)
        coarse.max_points_per_leaf = data.preconditioner_points_per_leaf;
      preconditioner_fmm.reinit(points, coarse);
    }
  separate_relaxed = data.relaxed_order > 0 || data.relaxed_theta > 0;
  if (separate_relaxed)
    {
      LaplaceFMM::AdditionalData relaxed = fmm_data;
      relaxed.order =
        data.relaxed_order > 0 ? data.relaxed_order : coarse.order;
      relaxed.theta =
        data.relaxed_theta > 0 ? data.relaxed_theta : coarse.theta;
      relaxed_fmm.reinit(points, relaxed);
    }
  fmm_data.order += data.dipole_order_increase;
  dipole_fmm.reinit(points, fmm_data);
  // PIBEM_UNFUSED_ROWS: the three separate evaluations, for comparison.
  // The fused pass holds the multipoles of five sets of all boxes at once:
  // only while they take less than 256 MB per process. The screened
  // expansions have no fused pass.
  fused_rows = kappa == 0. && data.dipole_order_increase == 0 &&
               charge_fmm.expansion_bytes(5) < (std::size_t(256) << 20) &&
               std::getenv("PIBEM_UNFUSED_ROWS") == nullptr;
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
  const auto tree     = pack_rtree(boxes);
  const auto cell_key = [&](const unsigned int c) {
    return std::array<int, 2>{{cells[c]->level(), cells[c]->index()}};
  };
  const auto pair_key = [&](const unsigned int t, const unsigned int r) {
    return std::array<int, 4>{{cells[t]->level(),
                               cells[t]->index(),
                               cells[r]->level(),
                               cells[r]->index()}};
  };
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
            type == GalerkinPanelPairType::same_panel ? 6 :
            type == GalerkinPanelPairType::common_edge ||
                type == GalerkinPanelPairType::nonconforming_edge ?
                                                        5 :
            type == GalerkinPanelPairType::common_vertex ? 2 :
                                                           1;
          // With a cache, the cells of a pair in the order of their levels
          // and indices, which refinement does not change.
          if (cache != nullptr && cell_key(r) < cell_key(t))
            found.push_back({{r, t, cost}});
          else
            found.push_back({{t, r, cost}});
        }
    }
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
  std::vector<std::pair<unsigned int, unsigned int>> my_pairs;
  // With a cache, each pair goes to the process of a hash of its key, so
  // that it stays there when the mesh is refined around it; there are many
  // pairs of each kind, and the costs even out.
  const auto hashed_process = [&](const unsigned int t, const unsigned int r) {
    const auto    key = pair_key(t, r);
    std::uint64_t h   = 0x9e3779b97f4a7c15ull;
    for (const int v : key)
      {
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) +
             0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= h >> 31;
        h *= 0xbf58476d1ce4e5b9ull;
      }
    return static_cast<unsigned int>(h % n_mpi_processes);
  };

  // PIBEM_NEAR_MEMORY: the memory of the process and the rule cache of the
  // integrator every 2000 pairs of the near field.
  static const bool report_memory = std::getenv("PIBEM_NEAR_MEMORY") != nullptr;
  static const bool report_profile =
    std::getenv("PIBEM_NEAR_PROFILE") != nullptr;
  std::set<std::array<int, 4>> used_keys;
  n_cached_pairs                               = 0;
  n_integrated_pairs                           = 0;
  std::array<double, 7>        profile_seconds = {{0, 0, 0, 0, 0, 0, 0}};
  std::array<unsigned long, 6> profile_count   = {{0, 0, 0, 0, 0, 0}};
  unsigned int                 pairs_done      = 0;
  // The exact blocks of the pairs of this process: V, K and W of (t, r),
  // and K of (r, t) from the same points, transposed in a fourth block;
  // in the cache if there is one, which keeps them, here otherwise.
  std::deque<FullMatrix<double>>          own_blocks;
  std::vector<const FullMatrix<double> *> exact_blocks;
  // Congruent pairs: on a surface swept along a straight line, as a wing of
  // constant section meshed evenly along the span, most near pairs are
  // translates of others, and their blocks are the same, the kernels being
  // invariant under translations. A pair is keyed by the quadrature points
  // of its two cells relative to the first one of the test cell, rounded to
  // 1e-9 of the size of that cell (which fixes the configuration, the rules
  // and the order of the shape functions), by its normals and by the
  // materials of its cells, through two 64-bit hashes; a process integrates
  // the first pair of each class it meets and uses its block for the
  // others.
  struct CongruenceHash
  {
    std::size_t
    operator()(const std::pair<std::uint64_t, std::uint64_t> &key) const
    {
      return key.first ^ (key.second * 0x9e3779b97f4a7c15ull);
    }
  };
  std::unordered_map<std::pair<std::uint64_t, std::uint64_t>,
                     const FullMatrix<double> *,
                     CongruenceHash>
    congruent;
  n_congruent_pairs         = 0;
  const auto congruence_key = [&](const unsigned int t, const unsigned int r) {
    const Point<3> &origin = points[t * n_q_points];
    const double    size =
      origin.distance(points[t * n_q_points + n_q_points - 1]);
    const double  quantum = 1e-9 * size;
    std::uint64_t h1 = 0x9e3779b97f4a7c15ull, h2 = 0xc2b2ae3d27d4eb4full;
    const auto    mix = [&](const std::int64_t value) {
      const std::uint64_t v = static_cast<std::uint64_t>(value);
      h1 ^= v + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2);
      h1 *= 0xbf58476d1ce4e5b9ull;
      h2 ^= v + 0x94d049bb133111ebull + (h2 << 7) + (h2 >> 3);
      h2 *= 0x94d049bb133111ebull;
    };
    for (const unsigned int c : {t, r})
      {
        mix(cells[c]->material_id());
        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int d = 0; d < 3; ++d)
            mix(std::llround((points[c * n_q_points + q][d] - origin[d]) /
                             quantum));
        for (unsigned int d = 0; d < 3; ++d)
          mix(std::llround(normals[c * n_q_points][d] * 1e9));
      }
    return std::make_pair(h1, h2);
  };
  const auto integrate_pair = [&](const unsigned int t, const unsigned int r) {
    if (report_memory && pairs_done % 2000 == 0)
      {
        std::ifstream statm("/proc/self/statm");
        unsigned long size = 0, resident = 0;
        statm >> size >> resident;
        std::cerr << "near field, process " << this_mpi_process << ", pair "
                  << pairs_done << ": RSS " << resident * 4096 / 1048576
                  << " MB, " << integrator.cache_statistics() << std::endl;
      }
    ++pairs_done;
    const bool self = t == r;
    // PIBEM_NEAR_PROFILE: the time of the exact blocks by type of pair,
    // close regular pairs apart.
    const auto   pair_start   = std::chrono::steady_clock::now();
    unsigned int profile_kind = 0;
    if (report_profile)
      {
        const auto type = integrator.classify(cells[t], cells[r]);
        profile_kind    = type == GalerkinPanelPairType::same_panel  ? 0 :
                          type == GalerkinPanelPairType::common_edge ? 1 :
                          type == GalerkinPanelPairType::nonconforming_edge ? 2 :
                          type == GalerkinPanelPairType::common_vertex ? 3 :
                          integrator.is_near_singular_pair(cells[t], cells[r]) ?
                                                                         4 :
                                                                         5;
      }
    const FullMatrix<double> *exact = nullptr;
    if (cache != nullptr)
      {
        const auto key = pair_key(t, r);
        const auto hit = cache->blocks.find(key);
        if (hit != cache->blocks.end())
          {
            exact = &hit->second;
            ++n_cached_pairs;
          }
        used_keys.insert(key);
      }
    if (exact == nullptr)
      {
        // PIBEM_NO_CONGRUENCE: every pair integrated, for comparison.
        static const bool use_congruence =
          std::getenv("PIBEM_NO_CONGRUENCE") == nullptr;
        const auto key = congruence_key(t, r);
        const auto found =
          use_congruence ? congruent.find(key) : congruent.end();
        if (found != congruent.end())
          {
            ++n_congruent_pairs;
            if (cache != nullptr)
              exact = &(cache->blocks[pair_key(t, r)] = *found->second);
            else
              exact = found->second;
          }
        else
          {
            FullMatrix<double> computed =
              self ? integrator.integrate_single_double_hypersingular(
                       cells[t], cells[r], data.quadrature_order) :
                     integrator.integrate_single_double_hypersingular_both(
                       cells[t], cells[r], data.quadrature_order);
            ++n_integrated_pairs;
            if (cache != nullptr)
              exact = &(cache->blocks[pair_key(t, r)] = std::move(computed));
            else
              {
                own_blocks.push_back(std::move(computed));
                exact = &own_blocks.back();
              }
            congruent.emplace(key, exact);
          }
      }
    exact_blocks.push_back(exact);
    my_pairs.emplace_back(t, r);
    if (report_profile)
      {
        profile_seconds[profile_kind] += seconds_since(pair_start);
        ++profile_count[profile_kind];
      }
  };

  if (cache == nullptr && n_mpi_processes > 1)
    {
      // Without a cache, the pairs are taken as the processes get free:
      // each fetches the next few, in the order of decreasing cost, from a
      // counter on process 0, so that the faster ones take more and the
      // costs need not be known.
      constexpr long chunk   = 4;
      long          *counter = nullptr;
      MPI_Win        window;
      MPI_Win_allocate(this_mpi_process == 0 ? sizeof(long) : 0,
                       sizeof(long),
                       MPI_INFO_NULL,
                       mpi_communicator,
                       &counter,
                       &window);
      if (this_mpi_process == 0)
        {
          MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window);
          *counter = 0;
          MPI_Win_unlock(0, window);
        }
      MPI_Barrier(mpi_communicator);
      MPI_Win_lock_all(0, window);
      const long n_pairs = all_pairs.size();
      while (true)
        {
          long first = 0;
          MPI_Fetch_and_op(&chunk, &first, MPI_LONG, 0, 0, MPI_SUM, window);
          MPI_Win_flush(0, window);
          if (first >= n_pairs)
            break;
          for (long k = first; k < std::min(first + chunk, n_pairs); ++k)
            integrate_pair(all_pairs[k][0], all_pairs[k][1]);
        }
      MPI_Win_unlock_all(window);
      MPI_Win_free(&window);
    }
  else
    {
      std::vector<unsigned long> load(n_mpi_processes, 0);
      for (const auto &pair : all_pairs)
        {
          const unsigned int process =
            cache != nullptr ?
              hashed_process(pair[0], pair[1]) :
              static_cast<unsigned int>(
                std::min_element(load.begin(), load.end()) - load.begin());
          load[process] += pair[2];
          if (process == this_mpi_process)
            integrate_pair(pair[0], pair[1]);
        }
    }
  n_near_pairs = 0;
  for (const auto &[t, r] : my_pairs)
    n_near_pairs += t == r ? 1 : 2;

  // Both blocks of every pair, condensed.
  DynamicSparsityPattern dsp(n_indices, n_indices);
  for (const auto &[t, r] : my_pairs)
    {
      constraints.add_entries_local_to_global(indices[t],
                                              indices[r],
                                              dsp,
                                              true);
      constraints.add_entries_local_to_global(indices[r],
                                              indices[t],
                                              dsp,
                                              true);
    }
  near_sparsity.copy_from(dsp);
  near_single.reinit(near_sparsity);
  near_double.reinit(near_sparsity);
  near_hypersingular.reinit(near_sparsity);
  single_diagonal.reinit(n_indices);
  hypersingular_diagonal.reinit(n_indices);

  FullMatrix<double> kernel(n_q_points, n_q_points),
    dipole(n_q_points, n_q_points), dipole_back(n_q_points, n_q_points),
    normal_kernel(kappa > 0. ? n_q_points : 0, kappa > 0. ? n_q_points : 0),
    partial(dpc, n_q_points), block(dpc, dpc), back(dpc, dpc),
    transposed(dpc, dpc), normal_block(dpc, dpc);
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
  for (unsigned int k = 0; k < my_pairs.size(); ++k)
    {
      const auto [t, r]               = my_pairs[k];
      const bool                self  = t == r;
      const FullMatrix<double> &exact = *exact_blocks[k];
      const auto correction_start     = std::chrono::steady_clock::now();

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
                  if (kappa > 0.)
                    normal_kernel(q, s) = 0;
                  continue;
                }
              const Tensor<1, 3> x_minus_y = points[a] - points[b];
              const double       distance  = x_minus_y.norm();
              const double weight = inverse_4pi * JxW[a] * JxW[b] / distance;
              const double cube   = distance * distance;
              // The screened kernel e^{-kappa r} G_0, its normal derivatives
              // those of G_0 times (1 + kappa r) e^{-kappa r}, and the term
              // kappa^2 (n_x . n_y) G of W.
              const double exponential =
                kappa == 0. ? 1. : std::exp(-kappa * distance);
              const double factor = (1. + kappa * distance) * exponential;
              kernel(q, s)        = weight * exponential;
              dipole(q, s) = weight * (x_minus_y * normals[b]) / cube * factor;
              dipole_back(s, q) =
                -weight * (x_minus_y * normals[a]) / cube * factor;
              if (kappa > 0.)
                normal_kernel(q, s) =
                  kappa * kappa * kernel(q, s) * (normals[a] * normals[b]);
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
      if (kappa > 0.)
        {
          contract(normal_kernel, normal_block);
          block.add(1., normal_block);
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
      if (report_profile)
        profile_seconds[6] += seconds_since(correction_start);
    }
  const double own_near_time = seconds_since(start);
  integrator.release_work_memory();
  // The blocks of the pairs that are gone, refined, leave the cache.
  if (cache != nullptr)
    {
      for (auto it = cache->blocks.begin(); it != cache->blocks.end();)
        if (used_keys.count(it->first) == 0)
          it = cache->blocks.erase(it);
        else
          ++it;
      cache->hits += n_cached_pairs;
      cache->misses += n_integrated_pairs;
    }
  if (report_profile)
    {
      // The parts of the direct path of the integrator over the near field,
      // summed over the processes (it holds the totals since its creation;
      // the integrator is created for this operator).
      const auto  &ip       = integrator.rhs_profile();
      const double parts[9] = {ip.geometry_seconds,
                               ip.shape_seconds,
                               ip.kernel_seconds,
                               ip.setup_seconds,
                               ip.singular_seconds,
                               ip.regular_seconds,
                               ip.kernel_weight_seconds,
                               ip.kernel_sum_seconds,
                               ip.kernel_curl_seconds};
      double       sums[9];
      for (unsigned int k = 0; k < 9; ++k)
        sums[k] = Utilities::MPI::sum(parts[k], mpi_communicator);
      const double points =
        Utilities::MPI::sum(static_cast<double>(ip.singular_points),
                            mpi_communicator);
      const double geometry_points =
        Utilities::MPI::sum(static_cast<double>(ip.geometry_points),
                            mpi_communicator);
      const double leaves[4] = {
        Utilities::MPI::sum(static_cast<double>(ip.singular_leaves),
                            mpi_communicator),
        Utilities::MPI::sum(static_cast<double>(ip.regular_leaves),
                            mpi_communicator),
        Utilities::MPI::sum(static_cast<double>(ip.transformed_leaves),
                            mpi_communicator),
        Utilities::MPI::sum(static_cast<double>(ip.half_leaves),
                            mpi_communicator)};
      if (this_mpi_process == 0)
        std::cout << "Near field, direct path, seconds summed over the "
                     "processes: geometry "
                  << sums[0] << " (" << geometry_points << " points), shapes "
                  << sums[1] << ", kernel loops " << sums[2] << " (weights "
                  << sums[6] << ", sums " << sums[7] << ", curls " << sums[8]
                  << ")"
                  << ", rule setup " << sums[3] << "; singular leaves "
                  << leaves[0] << " in " << sums[4] << " s (" << points
                  << " pair points; " << leaves[2]
                  << " on patches with the shapes of their rule, " << leaves[3]
                  << " on half rules), tensor leaves " << leaves[1] << " in "
                  << sums[5] << " s" << std::endl;
    }
  near_profile_seconds.fill(0.);
  near_profile_count.fill(0);
  if (report_profile)
    {
      for (unsigned int k = 0; k < 7; ++k)
        near_profile_seconds[k] =
          Utilities::MPI::sum(profile_seconds[k], mpi_communicator);
      for (unsigned int k = 0; k < 6; ++k)
        near_profile_count[k] =
          Utilities::MPI::sum(profile_count[k], mpi_communicator);
    }
  sum_over_processes(single_diagonal, mpi_communicator);
  sum_over_processes(hypersingular_diagonal, mpi_communicator);
  setup_near = seconds_since(start);
  setup_near_imbalance =
    Utilities::MPI::max(own_near_time, mpi_communicator) /
    std::max(1e-300,
             Utilities::MPI::sum(own_near_time, mpi_communicator) /
               n_mpi_processes);
  time_product = {{0, 0, 0, 0, 0, 0, 0}};
  n_products   = {{0, 0, 0, 0, 0, 0, 0}};
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
  const unsigned int dpc = dofs_per_cell;
  Vector<double>     local(dpc);
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
          add_hypersingular(values, curls, y, charge_fmm);
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
  // The gradient of the potential of the charges, for the screened kernel,
  // or with PIBEM_CALDERON_DIPOLE_KT for comparison, from three dipole
  // sums.
  std::vector<double>       charges(np);
  std::vector<Tensor<1, 3>> gradients;
  for (unsigned int p = 0; p < np; ++p)
    charges[p] = inverse_4pi * JxW[p] * values[p];
  if (kappa > 0. || std::getenv("PIBEM_CALDERON_DIPOLE_KT") != nullptr)
    gradient_from_dipoles(charges, dipole_fmm, gradients);
  else
    charge_fmm.evaluate_with_gradient(charges, potentials, gradients);
  for (unsigned int p = 0; p < np; ++p)
    normal_component[p] = JxW[p] * (normals[p] * gradients[p]);
  test_values(normal_component, y);
}



void
GalerkinCalderonFMMOperator::gradient_from_dipoles(
  const std::vector<double> &charges,
  const LaplaceFMM          &fmm,
  std::vector<Tensor<1, 3>> &gradients) const
{
  // The derivative along e_c of the kernel at x is the potential of the
  // dipole -e_c at y.
  const unsigned int                     np = points.size();
  std::vector<std::vector<Tensor<1, 3>>> dipoles(3,
                                                 std::vector<Tensor<1, 3>>(np));
  std::vector<std::vector<double>>       potentials;
  for (unsigned int d = 0; d < 3; ++d)
    for (unsigned int p = 0; p < np; ++p)
      dipoles[d][p][d] = -charges[p];
  fmm.evaluate_sets({}, dipoles, potentials);
  gradients.resize(np);
  for (unsigned int p = 0; p < np; ++p)
    for (unsigned int d = 0; d < 3; ++d)
      gradients[p][d] = potentials[d][p];
}



void
GalerkinCalderonFMMOperator::add_screened_hypersingular_charges(
  const std::vector<double>        &values,
  const std::vector<Tensor<1, 3>>  &curls,
  std::vector<std::vector<double>> &sets) const
{
  const unsigned int np = points.size();
  for (unsigned int d = 0; d < 3; ++d)
    {
      sets.emplace_back(np);
      for (unsigned int p = 0; p < np; ++p)
        sets.back()[p] = inverse_4pi * JxW[p] * curls[p][d];
    }
  for (unsigned int d = 0; d < 3; ++d)
    {
      sets.emplace_back(np);
      for (unsigned int p = 0; p < np; ++p)
        sets.back()[p] = inverse_4pi * JxW[p] * values[p] * normals[p][d];
    }
}



void
GalerkinCalderonFMMOperator::test_screened_hypersingular(
  std::vector<std::vector<double>> &potentials,
  const unsigned int                first,
  Vector<double>                   &y) const
{
  const unsigned int np = points.size();
  test_curl_potentials({{std::move(potentials[first]),
                         std::move(potentials[first + 1]),
                         std::move(potentials[first + 2])}},
                       y);
  // kappa^2 (n_x . n_y) G, from the potentials of the components of n x.
  std::vector<double> normal_part(np, 0.);
  for (unsigned int d = 0; d < 3; ++d)
    for (unsigned int p = 0; p < np; ++p)
      normal_part[p] += normals[p][d] * potentials[first + 3 + d][p];
  for (unsigned int p = 0; p < np; ++p)
    normal_part[p] *= kappa * kappa * JxW[p];
  test_values(normal_part, y);
}



void
GalerkinCalderonFMMOperator::add_hypersingular(
  const std::vector<double>       &values,
  const std::vector<Tensor<1, 3>> &curls,
  Vector<double>                  &y,
  const LaplaceFMM                &fmm) const
{
  const unsigned int               np = points.size();
  std::vector<std::vector<double>> charges, potentials;
  if (kappa > 0.)
    {
      // The screened W: six sets.
      add_screened_hypersingular_charges(values, curls, charges);
      fmm.evaluate_sets(charges, {}, potentials);
      test_screened_hypersingular(potentials, 0, y);
      return;
    }
  // The three components of the curl in one evaluation.
  charges.assign(3, std::vector<double>(np));
  for (unsigned int d = 0; d < 3; ++d)
    for (unsigned int p = 0; p < np; ++p)
      charges[d][p] = inverse_4pi * JxW[p] * curls[p][d];
  fmm.evaluate_multiple(charges, potentials);
  test_curl_potentials({{std::move(potentials[0]),
                         std::move(potentials[1]),
                         std::move(potentials[2])}},
                       y);
}



void
GalerkinCalderonFMMOperator::test_curl_potentials(
  const std::array<std::vector<double>, 3> &curl_potentials,
  Vector<double>                           &y) const
{
  const unsigned int dpc = dofs_per_cell;
  Vector<double>     local(dpc);
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
                                        Vector<double>       &neumann,
                                        const bool            relaxed) const
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
  // With the same expansions for charges and dipoles, all the sums from one
  // pass: V t and K^T t, K u, and the three components of the curl of u;
  // relaxed, always so, through the expansions of the preconditioner.
  const bool use_relaxed =
    relaxed && (separate_relaxed || separate_preconditioner);
  const LaplaceFMM &fmm   = !use_relaxed     ? charge_fmm :
                            separate_relaxed ? relaxed_fmm :
                                               preconditioner_fmm;
  const bool        fused = (use_relaxed || fused_rows) && kappa == 0.;
  // The potentials of the screened sums, whose W part is tested below.
  std::vector<std::vector<double>> screened_potentials;
  // The dipole sums of the screened rows: those of the relaxed expansions,
  // or of their own order.
  const LaplaceFMM   &dipole_sums = use_relaxed ? fmm : dipole_fmm;
  std::vector<double> dipole_potentials;
  std::array<std::vector<double>, 3> curl_potentials;
  if (fused)
    {
      std::array<std::vector<double>, 3> curl_charges;
      for (unsigned int c = 0; c < 3; ++c)
        {
          curl_charges[c].resize(np);
          for (unsigned int p = 0; p < np; ++p)
            curl_charges[c][p] = inverse_4pi * JxW[p] * curls_u[p][c];
        }
      fmm.evaluate_mixed(charges,
                         dipoles,
                         curl_charges,
                         potentials,
                         gradients,
                         dipole_potentials,
                         curl_potentials);
    }
  else if (kappa == 0.)
    charge_fmm.evaluate_with_gradient(charges, potentials, gradients);
  else
    {
      // The screened sums in two calls: the charges of t, of the curl of u
      // and of n u; the dipoles of the gradient of the potential of t, and
      // those of K u.
      std::vector<std::vector<double>> charge_sets{charges};
      add_screened_hypersingular_charges(values_u, curls_u, charge_sets);
      fmm.evaluate_sets(charge_sets, {}, screened_potentials);
      potentials = screened_potentials[0];
      std::vector<std::vector<Tensor<1, 3>>> dipole_sets(
        4, std::vector<Tensor<1, 3>>(np));
      for (unsigned int p = 0; p < np; ++p)
        {
          for (unsigned int d = 0; d < 3; ++d)
            dipole_sets[d][p][d] = -charges[p];
          dipole_sets[3][p] = dipoles[p];
        }
      std::vector<std::vector<double>> dipole_set_potentials;
      dipole_sums.evaluate_sets({}, dipole_sets, dipole_set_potentials);
      gradients.resize(np);
      for (unsigned int p = 0; p < np; ++p)
        for (unsigned int d = 0; d < 3; ++d)
          gradients[p][d] = dipole_set_potentials[d][p];
      dipole_potentials = std::move(dipole_set_potentials[3]);
    }
  for (unsigned int p = 0; p < np; ++p)
    values[p] = JxW[p] * potentials[p];
  test_values(values, d_rows);
  near_single.vmult_add(d_rows, t);
  Vector<double> transpose(n_indices);
  for (unsigned int p = 0; p < np; ++p)
    values[p] = JxW[p] * (normals[p] * gradients[p]);
  test_values(values, transpose);
  near_double.Tvmult_add(transpose, t);
  if (fused || kappa > 0.)
    potentials = std::move(dipole_potentials);
  else
    dipole_sums.evaluate({}, dipoles, potentials);
  Vector<double> k_u(n_indices);
  for (unsigned int p = 0; p < np; ++p)
    values[p] = JxW[p] * potentials[p];
  test_values(values, k_u);
  near_double.vmult_add(k_u, u);
  d_rows.add(-k, k_u);

  // W u + k K^T t.
  if (fused)
    test_curl_potentials(curl_potentials, n_rows);
  else if (kappa > 0.)
    test_screened_hypersingular(screened_potentials, 1, n_rows);
  else
    add_hypersingular(values_u, curls_u, n_rows, fmm);
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
  const unsigned int kind = use_relaxed ? 6 : 4;
  time_product[kind] += seconds_since(start);
  ++n_products[kind];
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
      add_hypersingular(values, curls, y, fmm);
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
  apply(op,
        dst,
        src,
        separate_preconditioner ? preconditioner_fmm : charge_fmm);
  time_product[5] += seconds_since(start);
  ++n_products[5];
}



void
GalerkinCalderonFMMOperator::vmult_preconditioner_pair(
  const Vector<double> &v_src,
  const Vector<double> &w_src,
  Vector<double>       &v,
  Vector<double>       &w) const
{
  const auto        start = std::chrono::steady_clock::now();
  const LaplaceFMM &fmm =
    separate_preconditioner ? preconditioner_fmm : charge_fmm;
  const unsigned int np = points.size();
  Vector<double>     x_v(v_src), x_w(w_src);
  if (constraints.n_constraints() > 0)
    {
      constraints.distribute(x_v);
      constraints.distribute(x_w);
    }
  std::vector<double>       values_v, values_w;
  std::vector<Tensor<1, 3>> curls_w;
  evaluate_field(x_v, values_v, nullptr);
  evaluate_field(x_w, values_w, &curls_w);
  std::vector<std::vector<double>> charges(4, std::vector<double>(np)),
    potentials;
  for (unsigned int p = 0; p < np; ++p)
    {
      charges[0][p] = inverse_4pi * JxW[p] * values_v[p];
      for (unsigned int d = 0; d < 3; ++d)
        charges[1 + d][p] = inverse_4pi * JxW[p] * curls_w[p][d];
    }
  Vector<double> rows(2 * n_indices), v_rows(n_indices), w_rows(n_indices);
  if (kappa == 0.)
    {
      fmm.evaluate_multiple(charges, potentials);
      test_curl_potentials({{std::move(potentials[1]),
                             std::move(potentials[2]),
                             std::move(potentials[3])}},
                           w_rows);
    }
  else
    {
      // The screened W: six sets of its own after that of V.
      charges.resize(1);
      add_screened_hypersingular_charges(values_w, curls_w, charges);
      fmm.evaluate_sets(charges, {}, potentials);
      test_screened_hypersingular(potentials, 1, w_rows);
    }
  for (unsigned int p = 0; p < np; ++p)
    potentials[0][p] *= JxW[p];
  test_values(potentials[0], v_rows);
  near_single.vmult_add(v_rows, v_src);
  near_hypersingular.vmult_add(w_rows, w_src);
  for (types::global_dof_index i = 0; i < n_indices; ++i)
    {
      rows[i]             = v_rows[i];
      rows[n_indices + i] = w_rows[i];
    }
  sum_over_processes(rows, mpi_communicator);
  v.reinit(n_indices);
  w.reinit(n_indices);
  for (types::global_dof_index i = 0; i < n_indices; ++i)
    {
      v[i] = rows[i];
      w[i] = rows[n_indices + i];
    }
  time_product[5] += seconds_since(start);
  n_products[5] += 2;
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
  const unsigned long cached =
    Utilities::MPI::sum(n_cached_pairs, mpi_communicator);
  const unsigned long integrated =
    Utilities::MPI::sum(n_integrated_pairs, mpi_communicator);
  const unsigned long congruent_pairs =
    Utilities::MPI::sum(n_congruent_pairs, mpi_communicator);
  const unsigned long entries =
    Utilities::MPI::sum(static_cast<unsigned long>(
                          near_sparsity.n_nonzero_elements()),
                        mpi_communicator);
  std::array<double, 7> mean;
  for (unsigned int k = 0; k < 7; ++k)
    mean[k] = Utilities::MPI::max(time_product[k] / std::max(1u, n_products[k]),
                                  mpi_communicator);
  const double sg = Utilities::MPI::max(setup_geometry, mpi_communicator);
  const double st = Utilities::MPI::max(setup_tree, mpi_communicator);
  const double sn = Utilities::MPI::max(setup_near, mpi_communicator);
  charge_fmm.print_statistics(out);
  dipole_fmm.print_statistics(out);
  if (separate_preconditioner)
    preconditioner_fmm.print_statistics(out);
  if (separate_relaxed)
    relaxed_fmm.print_statistics(out);
  const auto precision = out.precision();
  if (this_mpi_process == 0)
    out << std::setprecision(3) << "Galerkin Calderon FMM: " << n_cells
        << " cells, " << n_q_points << " points each; " << pairs
        << " near pairs corrected, " << entries
        << " near-field entries per operator"
        << ", " << congruent_pairs << " pair blocks from congruent pairs"
        << (cached > 0 ? ", " + std::to_string(cached) +
                           " pair blocks from the previous mesh, " +
                           std::to_string(integrated) + " integrated" :
                         std::string())
        << "\n"
        << "  setup: geometry " << sg << " s, trees " << st << " s, near field "
        << sn << " s (slowest process " << setup_near_imbalance
        << " times the mean)\n";
  if (this_mpi_process == 0 &&
      near_profile_count[0] + near_profile_count[1] > 0)
    {
      static const char *names[6] = {"coincident",
                                     "common edge",
                                     "non-conforming edge",
                                     "common vertex",
                                     "close regular",
                                     "other regular"};
      out << "  near field by pair, seconds summed over the processes:";
      for (unsigned int k = 0; k < 6; ++k)
        out << " " << names[k] << " " << near_profile_seconds[k] << " ("
            << near_profile_count[k] << ")";
      out << ", point-sum corrections " << near_profile_seconds[6] << "\n";
    }
  if (this_mpi_process == 0)
    out << "  seconds per product, slowest process: V " << mean[0] << " ("
        << n_products[0] << "), K " << mean[1] << " (" << n_products[1]
        << "), K^T " << mean[2] << " (" << n_products[2] << "), W " << mean[3]
        << " (" << n_products[3] << "), both rows of the mixed "
        << "problem " << mean[4] << " (" << n_products[4]
        << "), preconditioner V or W " << mean[5] << " (" << n_products[5]
        << "), relaxed rows " << mean[6] << " (" << n_products[6] << ")\n"
        << std::setprecision(precision);
}
