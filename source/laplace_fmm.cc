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

#include "../include/laplace_fmm.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#if defined(__AVX2__) && defined(__FMA__)
#  include <immintrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>

using namespace dealii;
namespace LE = LaplaceExpansions;
namespace YE = YukawaExpansions;

namespace
{
  // Deepest level of the tree, a guard against coincident points.
  constexpr unsigned int max_level = 40;

  // With screening, pairs of boxes whose gap g has kappa g above this are
  // dropped: e^{-40} = 4e-18 relative to the unscreened interaction.
  constexpr double screening_cutoff = 40.;

  // Octant of a child in its parent, as build_tree() numbers them.
  template <typename Box>
  unsigned int
  octant_of(const Box &child, const Box &parent)
  {
    return (child.center[0] > parent.center[0]) |
           ((child.center[1] > parent.center[1]) << 1) |
           ((child.center[2] > parent.center[2]) << 2);
  }

  double
  seconds_since(const std::chrono::steady_clock::time_point &start)
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
      .count();
  }

  void
  sum_over_processes(std::vector<LE::complex> &v, const MPI_Comm comm)
  {
    // std::complex<double> is laid out as two doubles.
    const int ierr = MPI_Allreduce(MPI_IN_PLACE,
                                   reinterpret_cast<double *>(v.data()),
                                   2 * v.size(),
                                   MPI_DOUBLE,
                                   MPI_SUM,
                                   comm);
    AssertThrowMPI(ierr);
  }

  void
  sum_over_processes(std::vector<double> &v, const MPI_Comm comm)
  {
    const int ierr = MPI_Allreduce(
      MPI_IN_PLACE, v.data(), v.size(), MPI_DOUBLE, MPI_SUM, comm);
    AssertThrowMPI(ierr);
  }
} // namespace



LaplaceFMM::LaplaceFMM(const MPI_Comm comm)
  : mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(comm))
  , this_mpi_process(Utilities::MPI::this_mpi_process(comm))
{}



void
LaplaceFMM::reinit(const std::vector<Point<3>> &points_in,
                   const AdditionalData        &data_in)
{
  AssertThrow(data_in.theta > 0 && data_in.theta < 1,
              ExcMessage("The opening criterion must lie in (0, 1)."));
  AssertThrow(data_in.max_points_per_leaf > 0,
              ExcMessage("A leaf must hold at least one point."));
  AssertThrow(data_in.kappa >= 0,
              ExcMessage("The screening parameter must not be negative."));
  const auto start = std::chrono::steady_clock::now();
  data             = data_in;
  points           = points_in;
  build_tree();
  m2l_pairs.clear();
  p2p_pairs.clear();
  n_screened_out = 0;
  if (!boxes.empty())
    traverse(0, 0);
  m2l_count.assign(boxes.size(), 0);
  p2p_count.assign(boxes.size(), 0);
  for (const auto &[a, b] : m2l_pairs)
    ++m2l_count[a];
  for (const auto &[a, b] : p2p_pairs)
    p2p_count[a] += static_cast<unsigned long>(boxes[a].end - boxes[a].begin) *
                    (boxes[b].end - boxes[b].begin);
  speed.clear();
  // Rough relative costs until the first evaluation measures them; the
  // screened translations are cheaper, its direct interactions dearer.
  cost_m2l   = 1.;
  cost_p2p   = data.kappa > 0 ? 1. / 150. : 1. / 1500.;
  cost_point = data.kappa > 0 ? 0.5 : 0.05;
  rotations  = std::make_unique<YE::Operators>(data.order);
  if (data.kappa > 0)
    {
      // Translations between a parent and its child in each octant, level
      // by level; v is the old centre minus the new one. The normalization
      // of each level.
      const unsigned int p     = data.order;
      const unsigned int nc    = YE::n_coefficients(p);
      unsigned int       depth = 0;
      for (const auto &box : boxes)
        depth = std::max(depth, box.level);
      yukawa_m2m.assign((depth + 1) * 8 * nc * nc, 0.);
      yukawa_l2l.assign((depth + 1) * 8 * nc * nc, 0.);
      yukawa_outgoing_scale.resize((depth + 1) * (p + 1));
      yukawa_local_scale.resize((depth + 1) * (p + 1));
      const double root_half_side = boxes.empty() ? 1. : boxes[0].half_side;
      for (unsigned int level = 0; level <= depth; ++level)
        {
          const double h   = root_half_side / double(1u << level);
          const double rho = std::sqrt(3.) * h;
          YE::bessel_k<double>(p,
                               data.kappa * rho,
                               &yukawa_outgoing_scale[level * (p + 1)]);
          YE::bessel_i<double>(p,
                               data.kappa * rho,
                               &yukawa_local_scale[level * (p + 1)]);
          for (unsigned int n = 0; n <= p; ++n)
            yukawa_outgoing_scale[level * (p + 1) + n] =
              1. / yukawa_outgoing_scale[level * (p + 1) + n];
          if (level == 0)
            continue;
          for (unsigned int o = 0; o < 8; ++o)
            {
              Tensor<1, 3> child_minus_parent;
              for (unsigned int d = 0; d < 3; ++d)
                child_minus_parent[d] = ((o >> d) & 1) ? h : -h;
              rotations->dense_translation(
                YE::Basis::outgoing,
                YE::Basis::outgoing,
                data.kappa,
                child_minus_parent,
                rho,
                2. * rho,
                &yukawa_m2m[(level * 8 + o) * nc * nc]);
              rotations->dense_translation(
                YE::Basis::local,
                YE::Basis::local,
                data.kappa,
                -child_minus_parent,
                2. * rho,
                rho,
                &yukawa_l2l[(level * 8 + o) * nc * nc]);
            }
        }
      yukawa_multipoles.assign(boxes.size() * nc, 0.);
      yukawa_locals.assign(boxes.size() * nc, 0.);
      multipoles.clear();
      locals.clear();
    }
  else
    {
      const LE::RealBasis basis(data.order);
      real_outgoing = basis.outgoing;
      real_local    = basis.local;
      yukawa_m2m.clear();
      yukawa_l2l.clear();
      yukawa_outgoing_scale.clear();
      yukawa_local_scale.clear();
      yukawa_multipoles.clear();
      yukawa_locals.clear();
      const unsigned int n_expansions =
        boxes.size() * LE::n_coefficients(data.order);
      multipoles.assign(n_expansions, 0.);
      locals.assign(n_expansions, 0.);
    }
  distribute_work();
  phase_time.fill(0.);
  n_evaluations = 0;
  setup_time    = seconds_since(start);
}



void
LaplaceFMM::build_tree()
{
  const unsigned int n = points.size();
  permutation.resize(n);
  std::iota(permutation.begin(), permutation.end(), 0u);
  boxes.clear();
  if (n == 0)
    {
      sorted_points.clear();
      return;
    }

  // Root: the smallest cube containing the points, slightly enlarged.
  Point<3> lower = points[0], upper = points[0];
  for (const auto &x : points)
    for (unsigned int d = 0; d < 3; ++d)
      {
        lower[d] = std::min(lower[d], x[d]);
        upper[d] = std::max(upper[d], x[d]);
      }
  Box root;
  root.center = 0.5 * (lower + upper);
  root.half_side =
    0.5 *
      std::max(
        {upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]}) *
      (1. + 1e-10) +
    1e-300;
  root.begin = 0;
  root.end   = n;
  boxes.push_back(root);

  // Split boxes in breadth-first order; the children of a box are appended
  // together, so they are consecutive and follow their parent.
  std::array<std::vector<unsigned int>, 8> octant;
  for (unsigned int b = 0; b < boxes.size(); ++b)
    {
      const Box box = boxes[b];
      if (box.end - box.begin <= data.max_points_per_leaf ||
          box.level >= max_level)
        continue;
      for (auto &o : octant)
        o.clear();
      for (unsigned int i = box.begin; i < box.end; ++i)
        {
          const Point<3> &x = points[permutation[i]];
          octant[(x[0] > box.center[0]) | ((x[1] > box.center[1]) << 1) |
                 ((x[2] > box.center[2]) << 2)]
            .push_back(permutation[i]);
        }
      unsigned int next    = box.begin;
      boxes[b].first_child = boxes.size();
      for (unsigned int o = 0; o < 8; ++o)
        {
          if (octant[o].empty())
            continue;
          Box child;
          child.half_side = 0.5 * box.half_side;
          for (unsigned int d = 0; d < 3; ++d)
            child.center[d] = box.center[d] + ((o >> d) & 1 ? child.half_side :
                                                              -child.half_side);
          child.begin = next;
          for (const unsigned int i : octant[o])
            permutation[next++] = i;
          child.end    = next;
          child.parent = b;
          child.level  = box.level + 1;
          boxes.push_back(child);
          ++boxes[b].n_children;
        }
    }

  sorted_points.resize(n);
  for (unsigned int i = 0; i < n; ++i)
    sorted_points[i] = points[permutation[i]];
  for (unsigned int d = 0; d < 3; ++d)
    {
      sorted_coordinates[d].resize(n);
      for (unsigned int i = 0; i < n; ++i)
        sorted_coordinates[d][i] = sorted_points[i][d];
    }
  for (auto &box : boxes)
    {
      double r2 = 0;
      for (unsigned int i = box.begin; i < box.end; ++i)
        r2 = std::max(r2, (sorted_points[i] - box.center).norm_square());
      box.radius = std::sqrt(r2);
    }
}



void
LaplaceFMM::traverse(const unsigned int a, const unsigned int b)
{
  const Box &A = boxes[a];
  const Box &B = boxes[b];
  if (a == b)
    {
      if (A.is_leaf())
        p2p_pairs.emplace_back(a, a);
      else
        for (unsigned int i = 0; i < A.n_children; ++i)
          for (unsigned int j = 0; j < A.n_children; ++j)
            traverse(A.first_child + i, A.first_child + j);
      return;
    }
  const double distance = A.center.distance(B.center);
  if (data.kappa > 0 &&
      data.kappa * (distance - A.radius - B.radius) > screening_cutoff)
    {
      ++n_screened_out;
      return;
    }
  if (A.radius + B.radius < data.theta * distance)
    {
      m2l_pairs.emplace_back(a, b);
      return;
    }
  if (A.is_leaf() && B.is_leaf())
    {
      p2p_pairs.emplace_back(a, b);
      return;
    }
  if (B.is_leaf() || (!A.is_leaf() && A.radius >= B.radius))
    for (unsigned int i = 0; i < A.n_children; ++i)
      traverse(A.first_child + i, b);
  else
    for (unsigned int j = 0; j < B.n_children; ++j)
      traverse(a, B.first_child + j);
}



void
LaplaceFMM::distribute_work() const
{
  const unsigned int n_boxes = boxes.size();
  owner.assign(n_boxes, invalid);
  if (speed.size() != n_mpi_processes)
    speed.assign(n_mpi_processes, 1.);

  if (n_boxes > 0)
    {
      // Modelled cost of every box, and of the subtree below it,
      // accumulated from the leaves up (children follow their parents, so a
      // reverse sweep).
      std::vector<double> cost(n_boxes, 0.);
      for (unsigned int b = 0; b < n_boxes; ++b)
        {
          cost[b] = cost_m2l * m2l_count[b] + cost_p2p * p2p_count[b];
          if (boxes[b].is_leaf())
            cost[b] += cost_point * (boxes[b].end - boxes[b].begin);
        }
      for (unsigned int b = n_boxes - 1; b > 0; --b)
        cost[boxes[b].parent] += cost[b];

      // Roots of the subtrees: starting from the root of the tree, the
      // heaviest subtree is replaced by those of its children until none
      // exceeds a small share of the total, or it is a leaf. On a graded
      // mesh the work concentrates in a few deep subtrees, which a cut at
      // a fixed level would leave whole to one process. The boxes above the
      // roots are handled by every process.
      const double              share = cost[0] / (16. * n_mpi_processes);
      std::vector<unsigned int> roots;
      {
        const auto lighter = [&](const unsigned int x, const unsigned int y) {
          return cost[x] < cost[y];
        };
        std::vector<unsigned int> heap = {0};
        while (!heap.empty())
          {
            std::pop_heap(heap.begin(), heap.end(), lighter);
            const unsigned int b = heap.back();
            heap.pop_back();
            if (cost[b] <= share)
              {
                // The others are lighter still.
                roots.push_back(b);
                roots.insert(roots.end(), heap.begin(), heap.end());
                break;
              }
            if (boxes[b].is_leaf())
              {
                roots.push_back(b);
                continue;
              }
            for (unsigned int c = 0; c < boxes[b].n_children; ++c)
              {
                heap.push_back(boxes[b].first_child + c);
                std::push_heap(heap.begin(), heap.end(), lighter);
              }
          }
      }
      std::vector<bool> is_root(n_boxes, false);
      for (const unsigned int r : roots)
        is_root[r] = true;

      // Largest processing time first, with capacities: the heaviest
      // subtree goes to the process that would finish it first.
      std::sort(roots.begin(),
                roots.end(),
                [&](const unsigned int x, const unsigned int y) {
                  return cost[x] > cost[y];
                });
      std::vector<double> load(n_mpi_processes, 0.);
      for (const unsigned int r : roots)
        {
          unsigned int best = 0;
          for (unsigned int p = 1; p < n_mpi_processes; ++p)
            if ((load[p] + cost[r]) / speed[p] <
                (load[best] + cost[r]) / speed[best])
              best = p;
          owner[r] = best;
          load[best] += cost[r];
        }
      // Parents precede children, so one sweep propagates the owners.
      for (unsigned int b = 1; b < n_boxes; ++b)
        if (owner[b] == invalid && owner[boxes[b].parent] != invalid)
          owner[b] = owner[boxes[b].parent];
    }

  my_leaves.clear();
  my_boxes.clear();
  for (unsigned int b = 0; b < n_boxes; ++b)
    {
      if (owner[b] == this_mpi_process || owner[b] == invalid)
        my_boxes.push_back(b);
      if (boxes[b].is_leaf() && owner[b] == this_mpi_process)
        my_leaves.push_back(b);
    }
  my_m2l_pairs.clear();
  for (const auto &pair : m2l_pairs)
    if (owner[pair.first] == this_mpi_process || owner[pair.first] == invalid)
      my_m2l_pairs.push_back(pair);
  my_p2p_pairs.clear();
  for (const auto &pair : p2p_pairs)
    if (owner[pair.first] == this_mpi_process)
      my_p2p_pairs.push_back(pair);

  // Groups of M2L pairs with the same scaled offset; within a group the
  // pairs are sorted by target, so that a local expansion stays in cache.
  std::map<std::array<int, 3>, std::vector<unsigned int>> by_offset;
  m2l_unit.resize(my_m2l_pairs.size());
  for (unsigned int i = 0; i < my_m2l_pairs.size(); ++i)
    {
      const Box         &A    = boxes[my_m2l_pairs[i].first];
      const Box         &B    = boxes[my_m2l_pairs[i].second];
      const double       unit = std::min(A.half_side, B.half_side);
      std::array<int, 3> key;
      for (unsigned int d = 0; d < 3; ++d)
        key[d] =
          static_cast<int>(std::lround((A.center[d] - B.center[d]) / unit));
      m2l_unit[i] = unit;
      by_offset[key].push_back(i);
    }
  // The translation along z of every pair: for the Laplace kernel one for
  // each |tau|, in units u; with screening one for each distance |tau| u,
  // which the square of tau and the level of the smaller box determine. The
  // pairs of a group are sorted by it first, so that a batch mostly shares
  // one.
  yukawa_tz.clear();
  m2l_tz.assign(my_m2l_pairs.size(), 0);
  if (data.kappa == 0.)
    {
      std::map<int, unsigned int> start_of;
      const unsigned int          size = rotations->z_translation_size();
      for (const auto &[key, members] : by_offset)
        {
          const int tau2 = key[0] * key[0] + key[1] * key[1] + key[2] * key[2];
          const auto [entry, inserted] =
            start_of.try_emplace(tau2, yukawa_tz.size());
          if (inserted)
            {
              yukawa_tz.resize(yukawa_tz.size() + size);
              rotations->z_translation_laplace(std::sqrt(double(tau2)),
                                               &yukawa_tz[entry->second]);
            }
          for (const unsigned int i : members)
            m2l_tz[i] = entry->second;
        }
    }
  else
    {
      std::map<std::pair<int, unsigned int>, unsigned int> start_of;
      const unsigned int size = rotations->z_translation_size();
      for (const auto &[key, members] : by_offset)
        {
          const int tau2 = key[0] * key[0] + key[1] * key[1] + key[2] * key[2];
          for (const unsigned int i : members)
            {
              const unsigned int level =
                std::max(boxes[my_m2l_pairs[i].first].level,
                         boxes[my_m2l_pairs[i].second].level);
              const auto [entry, inserted] =
                start_of.try_emplace({tau2, level}, yukawa_tz.size());
              if (inserted)
                {
                  yukawa_tz.resize(yukawa_tz.size() + size);
                  rotations->z_translation_unscaled(YE::Basis::outgoing,
                                                    YE::Basis::local,
                                                    data.kappa,
                                                    std::sqrt(double(tau2)) *
                                                      m2l_unit[i],
                                                    &yukawa_tz[entry->second]);
                }
              m2l_tz[i] = entry->second;
            }
        }
    }

  m2l_groups.clear();
  m2l_order.clear();
  for (auto &[key, members] : by_offset)
    {
      std::sort(members.begin(),
                members.end(),
                [&](const unsigned int x, const unsigned int y) {
                  return std::make_pair(m2l_tz[x], my_m2l_pairs[x]) <
                         std::make_pair(m2l_tz[y], my_m2l_pairs[y]);
                });
      M2LGroup group;
      group.tau =
        Tensor<1, 3>({double(key[0]), double(key[1]), double(key[2])});
      group.begin = m2l_order.size();
      m2l_order.insert(m2l_order.end(), members.begin(), members.end());
      group.end = m2l_order.size();
      m2l_groups.push_back(group);
    }

  // The rotations about y of the polar angles of the groups, which the
  // offset determines by its component along z and its distance from the
  // axis.
  m2l_rotations.clear();
  if (data.kappa == 0.)
    {
      const std::size_t size = rotations->rotation_size();
      std::map<std::array<long, 2>, std::size_t> start_of;
      YE::Operators::Angles                      angles;
      for (auto &group : m2l_groups)
        {
          const long z = std::lround(group.tau[2]);
          const long r = std::lround(group.tau[0] * group.tau[0] +
                                     group.tau[1] * group.tau[1]);
          const auto [entry, inserted] =
            start_of.try_emplace({{z, r}}, m2l_rotations.size());
          if (inserted)
            {
              m2l_rotations.resize(m2l_rotations.size() + size);
              rotations->angles(-group.tau, angles);
              rotations->rotation_blocks(angles, &m2l_rotations[entry->second]);
            }
          group.rotation = entry->second;
        }
    }
}



void
LaplaceFMM::rebalance(const std::array<double, 6> &times) const
{
  // Global cost per operation, from the times and counts of all processes.
  unsigned long my_points = 0, my_interactions = 0;
  for (const unsigned int leaf : my_leaves)
    my_points += boxes[leaf].end - boxes[leaf].begin;
  for (const auto &[a, b] : my_p2p_pairs)
    my_interactions +=
      static_cast<unsigned long>(boxes[a].end - boxes[a].begin) *
      (boxes[b].end - boxes[b].begin);
  const double m2l_time   = times[2];
  const double p2p_time   = times[4];
  const double point_time = times[0] + times[3];
  const double total_m2l =
    Utilities::MPI::sum(double(my_m2l_pairs.size()), mpi_communicator);
  const double total_p2p =
    Utilities::MPI::sum(double(my_interactions), mpi_communicator);
  const double total_points =
    Utilities::MPI::sum(double(my_points), mpi_communicator);
  cost_m2l =
    Utilities::MPI::sum(m2l_time, mpi_communicator) / std::max(1., total_m2l);
  cost_p2p =
    Utilities::MPI::sum(p2p_time, mpi_communicator) / std::max(1., total_p2p);
  cost_point = Utilities::MPI::sum(point_time, mpi_communicator) /
               std::max(1., total_points);

  // Speed of every process: modelled cost of its work over its time.
  const double modelled = cost_m2l * my_m2l_pairs.size() +
                          cost_p2p * my_interactions + cost_point * my_points;
  const double measured = m2l_time + p2p_time + point_time;
  const double my_speed =
    measured > 0 && modelled > 0 ? modelled / measured : 1.;
  speed = Utilities::MPI::all_gather(mpi_communicator, my_speed);
  distribute_work();
}



#if defined(__AVX2__) && defined(__FMA__)
namespace
{
  /**
   * 1/sqrt(r2) for four doubles, zero where r2 is zero: the estimate of the
   * single-precision instruction, relative error 4e-4, refined by two
   * Newton steps in double precision to about 1e-13, well below the
   * truncation of the expansions. The zeros are replaced by ones before the
   * estimate, so that no infinity or NaN arises.
   */
  inline __m256d
  inverse_sqrt(const __m256d r2)
  {
    const __m256d positive = _mm256_cmp_pd(r2, _mm256_setzero_pd(), _CMP_GT_OQ);
    const __m256d one      = _mm256_set1_pd(1.);
    const __m256d safe     = _mm256_blendv_pd(one, r2, positive);
    __m256d       y = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(safe)));
    const __m256d half_r2      = _mm256_mul_pd(_mm256_set1_pd(0.5), safe);
    const __m256d three_halves = _mm256_set1_pd(1.5);
    y                          = _mm256_mul_pd(
      y, _mm256_fnmadd_pd(half_r2, _mm256_mul_pd(y, y), three_halves));
    y = _mm256_mul_pd(
      y, _mm256_fnmadd_pd(half_r2, _mm256_mul_pd(y, y), three_halves));
    return _mm256_and_pd(y, positive);
  }

  /**
   * e^x for four doubles x <= 0: x = n ln 2 + r with |r| <= ln 2 / 2, e^r
   * by its Taylor polynomial of degree 10 (relative error below 3e-13,
   * far below the truncation of the expansions), and 2^n put into the
   * exponent bits. Arguments below -700 are raised to it; their results,
   * below 1e-304, are zero for every purpose here.
   */
  inline __m256d
  exp_nonpositive(__m256d x)
  {
    x = _mm256_max_pd(x, _mm256_set1_pd(-700.));
    const __m256d n =
      _mm256_round_pd(_mm256_mul_pd(x, _mm256_set1_pd(1.4426950408889634)),
                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // Cody and Waite: ln 2 in two parts, the first exact in n * ln2_high.
    __m256d r = _mm256_fnmadd_pd(n, _mm256_set1_pd(6.93145751953125e-1), x);
    r = _mm256_fnmadd_pd(n, _mm256_set1_pd(1.42860682030941723212e-6), r);
    constexpr double inverse_factorial[11] = {1.,
                                              1.,
                                              1. / 2,
                                              1. / 6,
                                              1. / 24,
                                              1. / 120,
                                              1. / 720,
                                              1. / 5040,
                                              1. / 40320,
                                              1. / 362880,
                                              1. / 3628800};
    __m256d          p = _mm256_set1_pd(inverse_factorial[10]);
    for (int k = 9; k >= 0; --k)
      p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(inverse_factorial[k]));
    // n + 2^52 + 2^51 holds n in its low bits.
    const __m256d magic = _mm256_set1_pd(6755399441055744.);
    __m256i       bits =
      _mm256_sub_epi64(_mm256_castpd_si256(_mm256_add_pd(n, magic)),
                       _mm256_castpd_si256(magic));
    bits =
      _mm256_slli_epi64(_mm256_add_epi64(bits, _mm256_set1_epi64x(1023)), 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(bits));
  }

  inline double
  horizontal_sum(const __m256d v)
  {
    const __m128d low  = _mm256_castpd256_pd128(v);
    const __m128d high = _mm256_extractf128_pd(v, 1);
    const __m128d pair = _mm_add_pd(low, high);
    return _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
  }
} // namespace
#endif



template <bool with_charges, bool with_dipoles, bool screened>
void
LaplaceFMM::particle_to_particle(const unsigned int a,
                                 const unsigned int b,
                                 const double      *q,
                                 const double      *d,
                                 double            *u) const
{
  const Box         &A  = boxes[a];
  const Box         &B  = boxes[b];
  const double      *x  = sorted_coordinates[0].data();
  const double      *y  = sorted_coordinates[1].data();
  const double      *z  = sorted_coordinates[2].data();
  const unsigned int n  = points.size();
  const double      *dx = d, *dy = d + n, *dz = d + 2 * n;
  const double       kappa = data.kappa;
  for (unsigned int i = A.begin; i < A.end; ++i)
    {
      const double xi = x[i], yi = y[i], zi = z[i];
      double       sum = 0;
      unsigned int j   = B.begin;
#if defined(__AVX2__) && defined(__FMA__)
      const __m256d vx = _mm256_set1_pd(xi), vy = _mm256_set1_pd(yi),
                    vz = _mm256_set1_pd(zi);
      __m256d acc      = _mm256_setzero_pd();
      // Four sources at a time; the last one to three through masked
      // loads, whose empty lanes carry zero charges and dipoles, rather
      // than through the scalar loop, which with the exponential costs
      // many times more per interaction.
      const auto interact = [&](const __m256d px,
                                const __m256d py,
                                const __m256d pz,
                                const __m256d qj,
                                const __m256d dxj,
                                const __m256d dyj,
                                const __m256d dzj) {
        const __m256d rx = _mm256_sub_pd(vx, px);
        const __m256d ry = _mm256_sub_pd(vy, py);
        const __m256d rz = _mm256_sub_pd(vz, pz);
        const __m256d r2 =
          _mm256_fmadd_pd(rx,
                          rx,
                          _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz)));
        const __m256d inverse = inverse_sqrt(r2);
        // With screening, e^{-kappa r} / r and, for the dipoles, the
        // factor (1 + kappa r) of the derivative.
        __m256d kernel = inverse, kappa_r = _mm256_setzero_pd();
        if constexpr (screened)
          {
            kappa_r =
              _mm256_mul_pd(_mm256_set1_pd(kappa), _mm256_mul_pd(r2, inverse));
            kernel =
              _mm256_mul_pd(inverse,
                            exp_nonpositive(
                              _mm256_sub_pd(_mm256_setzero_pd(), kappa_r)));
          }
        if constexpr (with_charges)
          acc = _mm256_fmadd_pd(qj, kernel, acc);
        if constexpr (with_dipoles)
          {
            __m256d dot =
              _mm256_fmadd_pd(dxj,
                              rx,
                              _mm256_fmadd_pd(dyj, ry, _mm256_mul_pd(dzj, rz)));
            if constexpr (screened)
              dot = _mm256_fmadd_pd(dot, kappa_r, dot);
            const __m256d inverse3 =
              _mm256_mul_pd(kernel, _mm256_mul_pd(inverse, inverse));
            acc = _mm256_fmadd_pd(dot, inverse3, acc);
          }
      };
      const __m256d zero = _mm256_setzero_pd();
      for (; j + 4 <= B.end; j += 4)
        interact(_mm256_loadu_pd(x + j),
                 _mm256_loadu_pd(y + j),
                 _mm256_loadu_pd(z + j),
                 with_charges ? _mm256_loadu_pd(q + j) : zero,
                 with_dipoles ? _mm256_loadu_pd(dx + j) : zero,
                 with_dipoles ? _mm256_loadu_pd(dy + j) : zero,
                 with_dipoles ? _mm256_loadu_pd(dz + j) : zero);
      if (j < B.end)
        {
          const __m256i mask =
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(B.end - j),
                               _mm256_set_epi64x(3, 2, 1, 0));
          interact(_mm256_maskload_pd(x + j, mask),
                   _mm256_maskload_pd(y + j, mask),
                   _mm256_maskload_pd(z + j, mask),
                   with_charges ? _mm256_maskload_pd(q + j, mask) : zero,
                   with_dipoles ? _mm256_maskload_pd(dx + j, mask) : zero,
                   with_dipoles ? _mm256_maskload_pd(dy + j, mask) : zero,
                   with_dipoles ? _mm256_maskload_pd(dz + j, mask) : zero);
          j = B.end;
        }
      sum = horizontal_sum(acc);
#endif
      // Without AVX2, all of it. The point itself, at distance zero, is
      // masked out without a division by zero.
      for (; j < B.end; ++j)
        {
          const double rx = xi - x[j], ry = yi - y[j], rz = zi - z[j];
          const double r2       = rx * rx + ry * ry + rz * rz;
          const bool   distinct = r2 > 0.;
          const double inverse =
            (distinct ? 1. : 0.) / std::sqrt(distinct ? r2 : 1.);
          double screening = 1., derivative = 1.;
          if constexpr (screened)
            {
              const double kappa_r = kappa * r2 * inverse;
              screening            = std::exp(-kappa_r);
              derivative           = 1. + kappa_r;
            }
          if constexpr (with_charges)
            sum += q[j] * screening * inverse;
          if constexpr (with_dipoles)
            sum += (dx[j] * rx + dy[j] * ry + dz[j] * rz) * derivative *
                   screening * inverse * inverse * inverse;
        }
      u[i] += sum;
    }
}



void
LaplaceFMM::particle_to_particle_gradient(const unsigned int a,
                                          const unsigned int b,
                                          const double      *q,
                                          double            *u,
                                          double            *g) const
{
  const Box         &A  = boxes[a];
  const Box         &B  = boxes[b];
  const double      *x  = sorted_coordinates[0].data();
  const double      *y  = sorted_coordinates[1].data();
  const double      *z  = sorted_coordinates[2].data();
  const unsigned int n  = points.size();
  double            *gx = g, *gy = g + n, *gz = g + 2 * n;
  for (unsigned int i = A.begin; i < A.end; ++i)
    {
      const double xi = x[i], yi = y[i], zi = z[i];
      double       sum = 0, sx = 0, sy = 0, sz = 0;
      unsigned int j = B.begin;
#if defined(__AVX2__) && defined(__FMA__)
      const __m256d vx = _mm256_set1_pd(xi), vy = _mm256_set1_pd(yi),
                    vz = _mm256_set1_pd(zi);
      __m256d acc = _mm256_setzero_pd(), acc_x = _mm256_setzero_pd(),
              acc_y = _mm256_setzero_pd(), acc_z = _mm256_setzero_pd();
      // grad_x q / |x - y| = -q (x - y) / |x - y|^3; the point itself, at
      // distance zero, has a zero inverse.
      const auto interact = [&](const __m256d px,
                                const __m256d py,
                                const __m256d pz,
                                const __m256d qj) {
        const __m256d rx = _mm256_sub_pd(vx, px);
        const __m256d ry = _mm256_sub_pd(vy, py);
        const __m256d rz = _mm256_sub_pd(vz, pz);
        const __m256d r2 =
          _mm256_fmadd_pd(rx,
                          rx,
                          _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz)));
        const __m256d inverse = inverse_sqrt(r2);
        const __m256d qk      = _mm256_mul_pd(qj, inverse);
        acc                   = _mm256_add_pd(acc, qk);
        const __m256d f = _mm256_mul_pd(qk, _mm256_mul_pd(inverse, inverse));
        acc_x           = _mm256_fnmadd_pd(f, rx, acc_x);
        acc_y           = _mm256_fnmadd_pd(f, ry, acc_y);
        acc_z           = _mm256_fnmadd_pd(f, rz, acc_z);
      };
      for (; j + 4 <= B.end; j += 4)
        interact(_mm256_loadu_pd(x + j),
                 _mm256_loadu_pd(y + j),
                 _mm256_loadu_pd(z + j),
                 _mm256_loadu_pd(q + j));
      if (j < B.end)
        {
          const __m256i mask =
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(B.end - j),
                               _mm256_set_epi64x(3, 2, 1, 0));
          interact(_mm256_maskload_pd(x + j, mask),
                   _mm256_maskload_pd(y + j, mask),
                   _mm256_maskload_pd(z + j, mask),
                   _mm256_maskload_pd(q + j, mask));
          j = B.end;
        }
      sum = horizontal_sum(acc);
      sx  = horizontal_sum(acc_x);
      sy  = horizontal_sum(acc_y);
      sz  = horizontal_sum(acc_z);
#endif
      for (; j < B.end; ++j)
        {
          const double rx = xi - x[j], ry = yi - y[j], rz = zi - z[j];
          const double r2       = rx * rx + ry * ry + rz * rz;
          const bool   distinct = r2 > 0.;
          const double inverse =
            (distinct ? 1. : 0.) / std::sqrt(distinct ? r2 : 1.);
          const double qk = q[j] * inverse;
          const double f  = qk * inverse * inverse;
          sum += qk;
          sx -= f * rx;
          sy -= f * ry;
          sz -= f * rz;
        }
      u[i] += sum;
      gx[i] += sx;
      gy[i] += sy;
      gz[i] += sz;
    }
}



template <unsigned int k>
void
LaplaceFMM::particle_to_particle_multiple(const unsigned int a,
                                          const unsigned int b,
                                          const double      *q,
                                          double            *u) const
{
  const Box         &A = boxes[a];
  const Box         &B = boxes[b];
  const double      *x = sorted_coordinates[0].data();
  const double      *y = sorted_coordinates[1].data();
  const double      *z = sorted_coordinates[2].data();
  const unsigned int n = points.size();
  for (unsigned int i = A.begin; i < A.end; ++i)
    {
      const double xi = x[i], yi = y[i], zi = z[i];
      double       sum[k];
      for (unsigned int s = 0; s < k; ++s)
        sum[s] = 0;
      unsigned int j = B.begin;
#if defined(__AVX2__) && defined(__FMA__)
      const __m256d vx = _mm256_set1_pd(xi), vy = _mm256_set1_pd(yi),
                    vz = _mm256_set1_pd(zi);
      __m256d acc[k];
      for (unsigned int s = 0; s < k; ++s)
        acc[s] = _mm256_setzero_pd();
      const auto inverse_at =
        [&](const __m256d px, const __m256d py, const __m256d pz) {
          const __m256d rx = _mm256_sub_pd(vx, px);
          const __m256d ry = _mm256_sub_pd(vy, py);
          const __m256d rz = _mm256_sub_pd(vz, pz);
          return inverse_sqrt(_mm256_fmadd_pd(
            rx, rx, _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz))));
        };
      for (; j + 4 <= B.end; j += 4)
        {
          const __m256d inverse = inverse_at(_mm256_loadu_pd(x + j),
                                             _mm256_loadu_pd(y + j),
                                             _mm256_loadu_pd(z + j));
          for (unsigned int s = 0; s < k; ++s)
            acc[s] =
              _mm256_fmadd_pd(_mm256_loadu_pd(q + s * n + j), inverse, acc[s]);
        }
      if (j < B.end)
        {
          const __m256i mask =
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(B.end - j),
                               _mm256_set_epi64x(3, 2, 1, 0));
          const __m256d inverse = inverse_at(_mm256_maskload_pd(x + j, mask),
                                             _mm256_maskload_pd(y + j, mask),
                                             _mm256_maskload_pd(z + j, mask));
          for (unsigned int s = 0; s < k; ++s)
            acc[s] = _mm256_fmadd_pd(_mm256_maskload_pd(q + s * n + j, mask),
                                     inverse,
                                     acc[s]);
          j = B.end;
        }
      for (unsigned int s = 0; s < k; ++s)
        sum[s] = horizontal_sum(acc[s]);
#endif
      for (; j < B.end; ++j)
        {
          const double rx = xi - x[j], ry = yi - y[j], rz = zi - z[j];
          const double r2       = rx * rx + ry * ry + rz * rz;
          const bool   distinct = r2 > 0.;
          const double inverse =
            (distinct ? 1. : 0.) / std::sqrt(distinct ? r2 : 1.);
          for (unsigned int s = 0; s < k; ++s)
            sum[s] += q[s * n + j] * inverse;
        }
      for (unsigned int s = 0; s < k; ++s)
        u[s * n + i] += sum[s];
    }
}



namespace
{
  /**
   * X[a * B + j] = source[j][a], a < n, for the B lanes of a batch: the
   * coefficients of each expansion, contiguous, become the rows of the
   * batch, by transposes of 4 x 4 blocks where AVX2 is available.
   */
  template <unsigned int B>
  void
  gather_lanes(const double *const *source, const unsigned int n, double *X)
  {
    unsigned int a = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; a + 4 <= n; a += 4)
      for (unsigned int j = 0; j < B; j += 4)
        {
          const __m256d r0 = _mm256_loadu_pd(source[j] + a);
          const __m256d r1 = _mm256_loadu_pd(source[j + 1] + a);
          const __m256d r2 = _mm256_loadu_pd(source[j + 2] + a);
          const __m256d r3 = _mm256_loadu_pd(source[j + 3] + a);
          const __m256d t0 = _mm256_unpacklo_pd(r0, r1);
          const __m256d t1 = _mm256_unpackhi_pd(r0, r1);
          const __m256d t2 = _mm256_unpacklo_pd(r2, r3);
          const __m256d t3 = _mm256_unpackhi_pd(r2, r3);
          _mm256_storeu_pd(X + a * B + j, _mm256_permute2f128_pd(t0, t2, 0x20));
          _mm256_storeu_pd(X + (a + 1) * B + j,
                           _mm256_permute2f128_pd(t1, t3, 0x20));
          _mm256_storeu_pd(X + (a + 2) * B + j,
                           _mm256_permute2f128_pd(t0, t2, 0x31));
          _mm256_storeu_pd(X + (a + 3) * B + j,
                           _mm256_permute2f128_pd(t1, t3, 0x31));
        }
#endif
    for (; a < n; ++a)
      for (unsigned int j = 0; j < B; ++j)
        X[a * B + j] = source[j][a];
  }

  /**
   * target[j][a] += Y[a * B + j], a < n, for the first count lanes, one
   * lane after the other, so that lanes may share a target.
   */
  template <unsigned int B>
  void
  scatter_lanes(const double      *Y,
                const unsigned int n,
                const unsigned int count,
                double *const     *target)
  {
    unsigned int a = 0;
#if defined(__AVX2__) && defined(__FMA__)
    for (; a + 4 <= n; a += 4)
      for (unsigned int j = 0; j < count; j += 4)
        {
          const __m256d r0      = _mm256_loadu_pd(Y + a * B + j);
          const __m256d r1      = _mm256_loadu_pd(Y + (a + 1) * B + j);
          const __m256d r2      = _mm256_loadu_pd(Y + (a + 2) * B + j);
          const __m256d r3      = _mm256_loadu_pd(Y + (a + 3) * B + j);
          const __m256d t0      = _mm256_unpacklo_pd(r0, r1);
          const __m256d t1      = _mm256_unpackhi_pd(r0, r1);
          const __m256d t2      = _mm256_unpacklo_pd(r2, r3);
          const __m256d t3      = _mm256_unpackhi_pd(r2, r3);
          const __m256d lane[4] = {_mm256_permute2f128_pd(t0, t2, 0x20),
                                   _mm256_permute2f128_pd(t1, t3, 0x20),
                                   _mm256_permute2f128_pd(t0, t2, 0x31),
                                   _mm256_permute2f128_pd(t1, t3, 0x31)};
          for (unsigned int l = 0; l < 4 && j + l < count; ++l)
            {
              double *t = target[j + l] + a;
              _mm256_storeu_pd(t, _mm256_add_pd(_mm256_loadu_pd(t), lane[l]));
            }
        }
#endif
    for (; a < n; ++a)
      for (unsigned int j = 0; j < count; ++j)
        target[j][a] += Y[a * B + j];
  }
} // namespace



void
LaplaceFMM::real_multipoles(LE::complex *M, const unsigned int k) const
{
  const unsigned int  p       = data.order;
  const unsigned int  nc      = LE::n_coefficients(p);
  const unsigned int  nr      = YE::n_coefficients(p);
  const std::size_t   n_boxes = boxes.size();
  std::vector<double> real(nr);
  for (unsigned int s = 0; s < k; ++s)
    for (std::size_t b = 0; b < n_boxes; ++b)
      {
        LE::complex *const Mb      = M + (s * n_boxes + b) * nc;
        const double       inverse = 1. / boxes[b].half_side;
        double             scale   = 1.;
        for (unsigned int n = 0; n <= p; ++n, scale *= inverse)
          {
            const unsigned int c  = LE::index(n, 0);
            real[YE::index(n, 0)] = scale * real_outgoing[c] * Mb[c].real();
            for (unsigned int m = 1; m <= n; ++m)
              {
                const double f              = scale * real_outgoing[c + m];
                real[YE::index(n, m)]       = f * Mb[c + m].real();
                real[YE::index(n, -int(m))] = -f * Mb[c + m].imag();
              }
          }
        std::copy(real.begin(), real.end(), reinterpret_cast<double *>(Mb));
      }
}



void
LaplaceFMM::complex_locals(LE::complex *L, const unsigned int k) const
{
  const unsigned int  p       = data.order;
  const unsigned int  nc      = LE::n_coefficients(p);
  const unsigned int  nr      = YE::n_coefficients(p);
  const std::size_t   n_boxes = boxes.size();
  std::vector<double> real(nr);
  for (unsigned int s = 0; s < k; ++s)
    for (const unsigned int b : my_boxes)
      {
        LE::complex *const Lb = L + (s * n_boxes + b) * nc;
        const double      *Lr = reinterpret_cast<const double *>(Lb);
        std::copy(Lr, Lr + nr, real.begin());
        const double inverse = 1. / boxes[b].half_side;
        double       scale   = inverse;
        for (unsigned int n = 0; n <= p; ++n, scale *= inverse)
          {
            const unsigned int c = LE::index(n, 0);
            Lb[c] = scale * real_local[c] * real[YE::index(n, 0)];
            for (unsigned int m = 1; m <= n; ++m)
              Lb[c + m] =
                scale * real_local[c + m] *
                LE::complex(real[YE::index(n, m)], real[YE::index(n, -int(m))]);
          }
      }
}



void
LaplaceFMM::multipole_to_local(const double      *M,
                               double            *L,
                               const unsigned int k) const
{
  // The pairs of a group convert from the unit u of the group, the half
  // side of the smaller box, to those of their boxes: the coefficients of
  // degree n of the source are scaled by (h_B / u)^n on the way in, those
  // of degree n of the target by (h_A / u)^(n + 1) on the way out; a power
  // of two, one for boxes of the same size.
  constexpr unsigned int B       = 16;
  const unsigned int     p       = data.order;
  const std::size_t      stride  = 2 * LE::n_coefficients(p);
  const unsigned int     nr      = YE::n_coefficients(p);
  const std::size_t      n_boxes = boxes.size();
  std::vector<double>    X(nr * B), Y(nr * B), zeros(nr, 0.);
  alignas(32) double     ratio[B], power[B];
  const double          *source[B];
  double                *target[B];
  YE::Operators::Angles  angles;
  for (const auto &group : m2l_groups)
    {
      rotations->angles(-group.tau, angles);
      const double      *Tz    = &yukawa_tz[m2l_tz[m2l_order[group.begin]]];
      const double      *R     = &m2l_rotations[group.rotation];
      const unsigned int total = (group.end - group.begin) * k;
      for (unsigned int first = 0; first < total; first += B)
        {
          const unsigned int count  = std::min(B, total - first);
          bool               scaled = false;
          for (unsigned int j = 0; j < count; ++j)
            {
              const unsigned int pair =
                m2l_order[group.begin + (first + j) / k];
              const unsigned int s = (first + j) % k;
              const unsigned int b = my_m2l_pairs[pair].second;
              source[j]            = M + (s * n_boxes + b) * stride;
              target[j] = L + (s * n_boxes + my_m2l_pairs[pair].first) * stride;
              ratio[j]  = boxes[b].half_side / m2l_unit[pair];
              scaled    = scaled || ratio[j] != 1.;
            }
          for (unsigned int j = count; j < B; ++j)
            {
              source[j] = zeros.data();
              ratio[j]  = 1.;
            }
          gather_lanes<B>(source, nr, X.data());
          if (scaled)
            {
#pragma omp simd
              for (unsigned int j = 0; j < B; ++j)
                power[j] = 1.;
              for (unsigned int n = 1; n <= p; ++n)
                {
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    power[j] *= ratio[j];
                  for (unsigned int a = n * n; a < (n + 1) * (n + 1); ++a)
#pragma omp simd
                    for (unsigned int j = 0; j < B; ++j)
                      X[a * B + j] *= power[j];
                }
            }
          rotations->translate_blocks<B>(angles, R, Tz, X.data(), Y.data());
          scaled = false;
          for (unsigned int j = 0; j < count; ++j)
            {
              const unsigned int pair =
                m2l_order[group.begin + (first + j) / k];
              ratio[j] =
                boxes[my_m2l_pairs[pair].first].half_side / m2l_unit[pair];
              scaled = scaled || ratio[j] != 1.;
            }
          for (unsigned int j = count; j < B; ++j)
            ratio[j] = 1.;
          if (scaled)
            {
#pragma omp simd
              for (unsigned int j = 0; j < B; ++j)
                power[j] = 1.;
              for (unsigned int n = 0; n <= p; ++n)
                {
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    power[j] *= ratio[j];
                  for (unsigned int a = n * n; a < (n + 1) * (n + 1); ++a)
#pragma omp simd
                    for (unsigned int j = 0; j < B; ++j)
                      Y[a * B + j] *= power[j];
                }
            }
          scatter_lanes<B>(Y.data(), nr, count, target);
        }
    }
}



void
LaplaceFMM::far_field_laplace_multiple(
  const unsigned int                       k,
  const std::vector<double>               &q,
  std::vector<double>                     &u,
  const std::vector<dealii::Tensor<1, 3>> *dipoles,
  std::vector<double>                     *gradient) const
{
  const unsigned int       p       = data.order;
  const unsigned int       nc      = LE::n_coefficients(p);
  const unsigned int       n       = points.size();
  const unsigned int       n_boxes = boxes.size();
  std::vector<LE::complex> M(k * n_boxes * nc, 0.);
  // The local expansions only on the boxes of this process: zero pages
  // from calloc, which the boxes of the other processes never touch, so
  // that k sets cost the memory of about one.
  const std::unique_ptr<LE::complex[], void (*)(void *)> L_buffer(
    static_cast<LE::complex *>(
      std::calloc(std::size_t(k) * n_boxes * nc, sizeof(LE::complex))),
    std::free);
  AssertThrow(L_buffer != nullptr, ExcMessage("Out of memory."));
  LE::complex *const L = L_buffer.get();
  LE::Workspace      w(p);

  auto start = std::chrono::steady_clock::now();
  for (unsigned int s = 0; s < k; ++s)
    {
      LE::complex *const Ms         = &M[s * n_boxes * nc];
      const bool         dipole_set = dipoles != nullptr && s == 1;
      for (const unsigned int leaf : my_leaves)
        {
          const Box &box = boxes[leaf];
          for (unsigned int i = box.begin; i < box.end; ++i)
            if (dipole_set)
              LE::add_dipole(w,
                             (*dipoles)[i],
                             sorted_points[i] - box.center,
                             &Ms[leaf * nc]);
            else
              LE::add_charge(w,
                             q[s * n + i],
                             sorted_points[i] - box.center,
                             &Ms[leaf * nc]);
        }
      for (auto b = my_boxes.rbegin(); b != my_boxes.rend(); ++b)
        if (boxes[*b].parent != invalid)
          {
            const Box &box = boxes[*b];
            LE::multipole_to_multipole(w,
                                       &Ms[*b * nc],
                                       box.center - boxes[box.parent].center,
                                       &Ms[box.parent * nc]);
          }
    }
  phase_time[0] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  if (n_mpi_processes > 1)
    sum_over_processes(M, mpi_communicator);
  phase_time[1] += seconds_since(start);

  // M2L in the real coefficients, all sets together.
  start = std::chrono::steady_clock::now();
  real_multipoles(M.data(), k);
  multipole_to_local(reinterpret_cast<const double *>(M.data()),
                     reinterpret_cast<double *>(L),
                     k);
  complex_locals(L, k);
  phase_time[2] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  for (unsigned int s = 0; s < k; ++s)
    {
      LE::complex *Ls = &L[s * n_boxes * nc];
      for (const unsigned int b : my_boxes)
        if (boxes[b].parent != invalid)
          {
            const Box &box = boxes[b];
            LE::local_to_local(w,
                               &Ls[box.parent * nc],
                               box.center - boxes[box.parent].center,
                               &Ls[b * nc]);
          }
      for (const unsigned int leaf : my_leaves)
        {
          const Box &box = boxes[leaf];
          for (unsigned int i = box.begin; i < box.end; ++i)
            {
              u[s * n + i] += LE::evaluate_local(w,
                                                 &Ls[leaf * nc],
                                                 sorted_points[i] - box.center);
              if (gradient != nullptr && s == 0)
                {
                  const Tensor<1, 3> g =
                    LE::evaluate_local_gradient(w,
                                                &Ls[leaf * nc],
                                                sorted_points[i] - box.center);
                  for (unsigned int c = 0; c < 3; ++c)
                    (*gradient)[c * n + i] += g[c];
                }
            }
        }
    }
  phase_time[3] += seconds_since(start);
}



void
LaplaceFMM::evaluate_multiple(
  const std::vector<std::vector<double>> &charges,
  std::vector<std::vector<double>>       &potentials) const
{
  AssertThrow(data.kappa == 0.,
              ExcMessage("Several sets of charges are implemented for the "
                         "Laplace kernel."));
  const unsigned int k = charges.size();
  const unsigned int n = points.size();
  AssertThrow(k >= 1 && k <= 4,
              ExcMessage("One to four sets of charges at once."));
  std::vector<double> q(k * n);
  for (unsigned int s = 0; s < k; ++s)
    {
      AssertThrow(charges[s].size() == n,
                  ExcDimensionMismatch(charges[s].size(), n));
      for (unsigned int i = 0; i < n; ++i)
        q[s * n + i] = charges[s][permutation[i]];
    }
  const std::array<double, 6> times_before = phase_time;
  std::vector<double>         u(k * n, 0.);
  far_field_laplace_multiple(k, q, u);

  auto start = std::chrono::steady_clock::now();
  for (const auto &[a, b] : my_p2p_pairs)
    switch (k)
      {
        case 1:
          particle_to_particle_multiple<1>(a, b, q.data(), u.data());
          break;
        case 2:
          particle_to_particle_multiple<2>(a, b, q.data(), u.data());
          break;
        case 3:
          particle_to_particle_multiple<3>(a, b, q.data(), u.data());
          break;
        default:
          particle_to_particle_multiple<4>(a, b, q.data(), u.data());
      }
  phase_time[4] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  std::vector<double> all(k * n, 0.);
  for (unsigned int s = 0; s < k; ++s)
    for (unsigned int i = 0; i < n; ++i)
      all[s * n + permutation[i]] = u[s * n + i];
  if (n_mpi_processes > 1)
    sum_over_processes(all, mpi_communicator);
  potentials.resize(k);
  for (unsigned int s = 0; s < k; ++s)
    potentials[s].assign(all.begin() + s * n, all.begin() + (s + 1) * n);
  phase_time[5] += seconds_since(start);
  ++n_evaluations;
  if (n_evaluations == 1 && n_mpi_processes > 1)
    {
      std::array<double, 6> times;
      for (unsigned int i = 0; i < times.size(); ++i)
        times[i] = phase_time[i] - times_before[i];
      rebalance(times);
    }
}



void
LaplaceFMM::particle_to_particle_mixed(const unsigned int a,
                                       const unsigned int b,
                                       const double      *q,
                                       const double      *d,
                                       double            *u,
                                       double            *g) const
{
  const Box         &A  = boxes[a];
  const Box         &B  = boxes[b];
  const double      *x  = sorted_coordinates[0].data();
  const double      *y  = sorted_coordinates[1].data();
  const double      *z  = sorted_coordinates[2].data();
  const unsigned int n  = points.size();
  const double      *q0 = q, *c1 = q + 2 * n, *c2 = q + 3 * n, *c3 = q + 4 * n;
  const double      *dx = d, *dy = d + n, *dz = d + 2 * n;
  for (unsigned int i = A.begin; i < A.end; ++i)
    {
      const double xi = x[i], yi = y[i], zi = z[i];
      double s0 = 0, sx = 0, sy = 0, sz = 0, sd = 0, s1 = 0, s2 = 0, s3 = 0;
      unsigned int j = B.begin;
#if defined(__AVX2__) && defined(__FMA__)
      const __m256d vx = _mm256_set1_pd(xi), vy = _mm256_set1_pd(yi),
                    vz = _mm256_set1_pd(zi);
      __m256d acc0 = _mm256_setzero_pd(), acc_x = _mm256_setzero_pd(),
              acc_y = _mm256_setzero_pd(), acc_z = _mm256_setzero_pd(),
              acc_d = _mm256_setzero_pd(), acc1 = _mm256_setzero_pd(),
              acc2 = _mm256_setzero_pd(), acc3 = _mm256_setzero_pd();
      // 1 / r and (x - y) / r^3 once for all: the potential and the
      // gradient of q, the potential of the dipoles, those of c.
      const auto interact = [&](const __m256d px,
                                const __m256d py,
                                const __m256d pz,
                                const __m256d qj,
                                const __m256d dxj,
                                const __m256d dyj,
                                const __m256d dzj,
                                const __m256d cj1,
                                const __m256d cj2,
                                const __m256d cj3) {
        const __m256d rx = _mm256_sub_pd(vx, px);
        const __m256d ry = _mm256_sub_pd(vy, py);
        const __m256d rz = _mm256_sub_pd(vz, pz);
        const __m256d r2 =
          _mm256_fmadd_pd(rx,
                          rx,
                          _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz)));
        const __m256d inverse = inverse_sqrt(r2);
        const __m256d inverse3 =
          _mm256_mul_pd(inverse, _mm256_mul_pd(inverse, inverse));
        acc0            = _mm256_fmadd_pd(qj, inverse, acc0);
        const __m256d f = _mm256_mul_pd(qj, inverse3);
        acc_x           = _mm256_fnmadd_pd(f, rx, acc_x);
        acc_y           = _mm256_fnmadd_pd(f, ry, acc_y);
        acc_z           = _mm256_fnmadd_pd(f, rz, acc_z);
        const __m256d dot =
          _mm256_fmadd_pd(dxj,
                          rx,
                          _mm256_fmadd_pd(dyj, ry, _mm256_mul_pd(dzj, rz)));
        acc_d = _mm256_fmadd_pd(dot, inverse3, acc_d);
        acc1  = _mm256_fmadd_pd(cj1, inverse, acc1);
        acc2  = _mm256_fmadd_pd(cj2, inverse, acc2);
        acc3  = _mm256_fmadd_pd(cj3, inverse, acc3);
      };
      for (; j + 4 <= B.end; j += 4)
        interact(_mm256_loadu_pd(x + j),
                 _mm256_loadu_pd(y + j),
                 _mm256_loadu_pd(z + j),
                 _mm256_loadu_pd(q0 + j),
                 _mm256_loadu_pd(dx + j),
                 _mm256_loadu_pd(dy + j),
                 _mm256_loadu_pd(dz + j),
                 _mm256_loadu_pd(c1 + j),
                 _mm256_loadu_pd(c2 + j),
                 _mm256_loadu_pd(c3 + j));
      if (j < B.end)
        {
          const __m256i mask =
            _mm256_cmpgt_epi64(_mm256_set1_epi64x(B.end - j),
                               _mm256_set_epi64x(3, 2, 1, 0));
          interact(_mm256_maskload_pd(x + j, mask),
                   _mm256_maskload_pd(y + j, mask),
                   _mm256_maskload_pd(z + j, mask),
                   _mm256_maskload_pd(q0 + j, mask),
                   _mm256_maskload_pd(dx + j, mask),
                   _mm256_maskload_pd(dy + j, mask),
                   _mm256_maskload_pd(dz + j, mask),
                   _mm256_maskload_pd(c1 + j, mask),
                   _mm256_maskload_pd(c2 + j, mask),
                   _mm256_maskload_pd(c3 + j, mask));
          j = B.end;
        }
      s0 = horizontal_sum(acc0);
      sx = horizontal_sum(acc_x);
      sy = horizontal_sum(acc_y);
      sz = horizontal_sum(acc_z);
      sd = horizontal_sum(acc_d);
      s1 = horizontal_sum(acc1);
      s2 = horizontal_sum(acc2);
      s3 = horizontal_sum(acc3);
#endif
      for (; j < B.end; ++j)
        {
          const double rx = xi - x[j], ry = yi - y[j], rz = zi - z[j];
          const double r2       = rx * rx + ry * ry + rz * rz;
          const bool   distinct = r2 > 0.;
          const double inverse =
            (distinct ? 1. : 0.) / std::sqrt(distinct ? r2 : 1.);
          const double inverse3 = inverse * inverse * inverse;
          s0 += q0[j] * inverse;
          sx -= q0[j] * inverse3 * rx;
          sy -= q0[j] * inverse3 * ry;
          sz -= q0[j] * inverse3 * rz;
          sd += (dx[j] * rx + dy[j] * ry + dz[j] * rz) * inverse3;
          s1 += c1[j] * inverse;
          s2 += c2[j] * inverse;
          s3 += c3[j] * inverse;
        }
      u[i] += s0;
      u[n + i] += sd;
      u[2 * n + i] += s1;
      u[3 * n + i] += s2;
      u[4 * n + i] += s3;
      g[i] += sx;
      g[n + i] += sy;
      g[2 * n + i] += sz;
    }
}



std::size_t
LaplaceFMM::expansion_bytes(const unsigned int sets) const
{
  return std::size_t(sets) * boxes.size() * LE::n_coefficients(data.order) *
         sizeof(LE::complex);
}



void
LaplaceFMM::evaluate_mixed(
  const std::vector<double>                &q,
  const std::vector<Tensor<1, 3>>          &d,
  const std::array<std::vector<double>, 3> &c,
  std::vector<double>                      &potentials_q,
  std::vector<Tensor<1, 3>>                &gradients_q,
  std::vector<double>                      &potentials_d,
  std::array<std::vector<double>, 3>       &potentials_c) const
{
  AssertThrow(data.kappa == 0.,
              ExcMessage("The mixed sums are implemented for the Laplace "
                         "kernel."));
  const unsigned int n = points.size();
  AssertThrow(q.size() == n && d.size() == n && c[0].size() == n &&
                c[1].size() == n && c[2].size() == n,
              ExcMessage("Sources of the wrong size."));
  // Five sets in the sorted numbering: q, the dipole set (charges unused),
  // and c; the dipoles as tensors and by component.
  std::vector<double>       sources(5 * n, 0.), d_components(3 * n);
  std::vector<Tensor<1, 3>> d_sorted(n);
  for (unsigned int i = 0; i < n; ++i)
    {
      const unsigned int k = permutation[i];
      sources[i]           = q[k];
      for (unsigned int s = 0; s < 3; ++s)
        sources[(2 + s) * n + i] = c[s][k];
      d_sorted[i] = d[k];
      for (unsigned int s = 0; s < 3; ++s)
        d_components[s * n + i] = d[k][s];
    }
  const std::array<double, 6> times_before = phase_time;
  std::vector<double>         u(5 * n, 0.), g(3 * n, 0.);
  far_field_laplace_multiple(5, sources, u, &d_sorted, &g);

  auto start = std::chrono::steady_clock::now();
  for (const auto &[a, b] : my_p2p_pairs)
    particle_to_particle_mixed(
      a, b, sources.data(), d_components.data(), u.data(), g.data());
  phase_time[4] += seconds_since(start);

  // One sum over the processes: the five potentials and the gradient.
  start = std::chrono::steady_clock::now();
  std::vector<double> all(8 * n, 0.);
  for (unsigned int i = 0; i < n; ++i)
    {
      const unsigned int k = permutation[i];
      for (unsigned int s = 0; s < 5; ++s)
        all[s * n + k] = u[s * n + i];
      for (unsigned int s = 0; s < 3; ++s)
        all[(5 + s) * n + k] = g[s * n + i];
    }
  if (n_mpi_processes > 1)
    sum_over_processes(all, mpi_communicator);
  potentials_q.assign(all.begin(), all.begin() + n);
  potentials_d.assign(all.begin() + n, all.begin() + 2 * n);
  for (unsigned int s = 0; s < 3; ++s)
    potentials_c[s].assign(all.begin() + (2 + s) * n,
                           all.begin() + (3 + s) * n);
  gradients_q.resize(n);
  for (unsigned int k = 0; k < n; ++k)
    for (unsigned int s = 0; s < 3; ++s)
      gradients_q[k][s] = all[(5 + s) * n + k];
  phase_time[5] += seconds_since(start);
  ++n_evaluations;
  if (n_evaluations == 1 && n_mpi_processes > 1)
    {
      std::array<double, 6> times;
      for (unsigned int i = 0; i < times.size(); ++i)
        times[i] = phase_time[i] - times_before[i];
      rebalance(times);
    }
}



void
LaplaceFMM::evaluate_with_gradient(const std::vector<double> &charges,
                                   std::vector<double>       &potentials,
                                   std::vector<Tensor<1, 3>> &gradients) const
{
  AssertThrow(data.kappa == 0.,
              ExcMessage("Gradients are implemented for the Laplace kernel."));
  const unsigned int n = points.size();
  AssertThrow(charges.size() == n, ExcDimensionMismatch(charges.size(), n));
  std::vector<double> q(n);
  for (unsigned int i = 0; i < n; ++i)
    q[i] = charges[permutation[i]];

  const std::array<double, 6> times_before = phase_time;
  // Potentials and the three components of the gradient, sorted numbering.
  std::vector<double> u(n, 0.), g(3 * n, 0.);
  far_field_laplace(q, {}, u, &g);

  auto start = std::chrono::steady_clock::now();
  for (const auto &[a, b] : my_p2p_pairs)
    particle_to_particle_gradient(a, b, q.data(), u.data(), g.data());
  phase_time[4] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  std::vector<double> all(4 * n, 0.);
  for (unsigned int i = 0; i < n; ++i)
    {
      const unsigned int k = permutation[i];
      all[4 * k]           = u[i];
      for (unsigned int c = 0; c < 3; ++c)
        all[4 * k + 1 + c] = g[c * n + i];
    }
  if (n_mpi_processes > 1)
    sum_over_processes(all, mpi_communicator);
  potentials.resize(n);
  gradients.resize(n);
  for (unsigned int k = 0; k < n; ++k)
    {
      potentials[k] = all[4 * k];
      for (unsigned int c = 0; c < 3; ++c)
        gradients[k][c] = all[4 * k + 1 + c];
    }
  phase_time[5] += seconds_since(start);
  ++n_evaluations;
  if (n_evaluations == 1 && n_mpi_processes > 1)
    {
      std::array<double, 6> times;
      for (unsigned int i = 0; i < times.size(); ++i)
        times[i] = phase_time[i] - times_before[i];
      rebalance(times);
    }
}



void
LaplaceFMM::evaluate(const std::vector<double>       &charges,
                     const std::vector<Tensor<1, 3>> &dipoles,
                     std::vector<double>             &potentials) const
{
  const unsigned int n = points.size();
  AssertThrow(charges.empty() || charges.size() == n,
              ExcDimensionMismatch(charges.size(), n));
  AssertThrow(dipoles.empty() || dipoles.size() == n,
              ExcDimensionMismatch(dipoles.size(), n));
  // Sources in the sorted numbering; the dipoles also by component, for
  // the direct interactions.
  std::vector<double>       q(charges.empty() ? 0 : n);
  std::vector<Tensor<1, 3>> d(dipoles.empty() ? 0 : n);
  std::vector<double>       d_components(3 * d.size());
  for (unsigned int i = 0; i < q.size(); ++i)
    q[i] = charges[permutation[i]];
  for (unsigned int i = 0; i < d.size(); ++i)
    {
      d[i] = dipoles[permutation[i]];
      for (unsigned int c = 0; c < 3; ++c)
        d_components[c * n + i] = d[i][c];
    }

  const std::array<double, 6> times_before = phase_time;
  std::vector<double>         u(n, 0.);
  if (data.kappa > 0)
    far_field_yukawa(q, d, u);
  else
    far_field_laplace(q, d, u);

  auto       start      = std::chrono::steady_clock::now();
  const auto near_field = [&](auto screened) {
    constexpr bool s = decltype(screened)::value;
    for (const auto &[a, b] : my_p2p_pairs)
      if (d.empty())
        particle_to_particle<true, false, s>(a, b, q.data(), nullptr, u.data());
      else if (q.empty())
        particle_to_particle<false, true, s>(
          a, b, nullptr, d_components.data(), u.data());
      else
        particle_to_particle<true, true, s>(
          a, b, q.data(), d_components.data(), u.data());
  };
  if (data.kappa > 0)
    near_field(std::true_type());
  else
    near_field(std::false_type());
  phase_time[4] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  potentials.assign(n, 0.);
  for (unsigned int i = 0; i < n; ++i)
    potentials[permutation[i]] = u[i];
  if (n_mpi_processes > 1)
    sum_over_processes(potentials, mpi_communicator);
  phase_time[5] += seconds_since(start);
  ++n_evaluations;

  // The first evaluation calibrates the split of the work.
  if (n_evaluations == 1 && n_mpi_processes > 1)
    {
      std::array<double, 6> times;
      for (unsigned int i = 0; i < times.size(); ++i)
        times[i] = phase_time[i] - times_before[i];
      rebalance(times);
    }
}



void
LaplaceFMM::far_field_laplace(const std::vector<double>       &q,
                              const std::vector<Tensor<1, 3>> &d,
                              std::vector<double>             &u,
                              std::vector<double>             *g) const
{
  const unsigned int p  = data.order;
  const unsigned int nc = LE::n_coefficients(p);
  std::fill(multipoles.begin(), multipoles.end(), 0.);
  std::fill(locals.begin(), locals.end(), 0.);
  LE::Workspace w(p);

  // Upward pass: multipoles of the own leaves, translated up through the
  // own and the shared boxes (children after parents, so reversed).
  auto start = std::chrono::steady_clock::now();
  for (const unsigned int leaf : my_leaves)
    {
      const Box         &box = boxes[leaf];
      LE::complex *const M   = &multipoles[leaf * nc];
      for (unsigned int i = box.begin; i < box.end; ++i)
        {
          const Tensor<1, 3> offset = sorted_points[i] - box.center;
          if (!q.empty())
            LE::add_charge(w, q[i], offset, M);
          if (!d.empty())
            LE::add_dipole(w, d[i], offset, M);
        }
    }
  for (auto b = my_boxes.rbegin(); b != my_boxes.rend(); ++b)
    if (boxes[*b].parent != invalid)
      {
        const Box &box = boxes[*b];
        LE::multipole_to_multipole(w,
                                   &multipoles[*b * nc],
                                   box.center - boxes[box.parent].center,
                                   &multipoles[box.parent * nc]);
      }
  phase_time[0] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  if (n_mpi_processes > 1)
    sum_over_processes(multipoles, mpi_communicator);
  phase_time[1] += seconds_since(start);

  // M2L by groups of pairs with the same scaled offset, by rotation onto the
  // z-axis in the real harmonics (see YukawaExpansions::Operators), on the
  // real coefficients scaled by the sizes of the boxes.
  start = std::chrono::steady_clock::now();
  real_multipoles(multipoles.data(), 1);
  multipole_to_local(reinterpret_cast<const double *>(multipoles.data()),
                     reinterpret_cast<double *>(locals.data()),
                     1);
  complex_locals(locals.data(), 1);
  phase_time[2] += seconds_since(start);

  // Downward pass and evaluation at the own points.
  start = std::chrono::steady_clock::now();
  for (const unsigned int b : my_boxes)
    if (boxes[b].parent != invalid)
      {
        const Box &box = boxes[b];
        LE::local_to_local(w,
                           &locals[box.parent * nc],
                           box.center - boxes[box.parent].center,
                           &locals[b * nc]);
      }
  const unsigned int n = u.size();
  for (const unsigned int leaf : my_leaves)
    {
      const Box &box = boxes[leaf];
      for (unsigned int i = box.begin; i < box.end; ++i)
        {
          u[i] += LE::evaluate_local(w,
                                     &locals[leaf * nc],
                                     sorted_points[i] - box.center);
          if (g != nullptr)
            {
              const Tensor<1, 3> gradient =
                LE::evaluate_local_gradient(w,
                                            &locals[leaf * nc],
                                            sorted_points[i] - box.center);
              for (unsigned int c = 0; c < 3; ++c)
                (*g)[c * n + i] += gradient[c];
            }
        }
    }
  phase_time[3] += seconds_since(start);
}



void
LaplaceFMM::far_field_yukawa(const std::vector<double>       &q,
                             const std::vector<Tensor<1, 3>> &d,
                             std::vector<double>             &u) const
{
  const unsigned int p     = data.order;
  const unsigned int nc    = YE::n_coefficients(p);
  const double       kappa = data.kappa;
  std::fill(yukawa_multipoles.begin(), yukawa_multipoles.end(), 0.);
  std::fill(yukawa_locals.begin(), yukawa_locals.end(), 0.);
  YE::Workspace w(p);

  // Upward pass: multipoles of the own leaves, translated up through the
  // own and the shared boxes with the matrix of their level and octant.
  auto start = std::chrono::steady_clock::now();
  for (const unsigned int leaf : my_leaves)
    {
      const Box    &box = boxes[leaf];
      double *const M   = &yukawa_multipoles[leaf * nc];
      const double  rho = yukawa_radius(box);
      for (unsigned int i = box.begin; i < box.end; ++i)
        {
          const Tensor<1, 3> offset = sorted_points[i] - box.center;
          if (!q.empty())
            YE::add_charge(w, kappa, rho, q[i], offset, M);
          if (!d.empty())
            YE::add_dipole(w, kappa, rho, d[i], offset, M);
        }
    }
  for (auto b = my_boxes.rbegin(); b != my_boxes.rend(); ++b)
    if (boxes[*b].parent != invalid)
      {
        const Box    &box = boxes[*b];
        const double *T =
          &yukawa_m2m[(box.level * 8 + octant_of(box, boxes[box.parent])) * nc *
                      nc];
        const double *M      = &yukawa_multipoles[*b * nc];
        double       *parent = &yukawa_multipoles[box.parent * nc];
        for (unsigned int r = 0; r < nc; ++r)
          {
            double sum = 0.;
#pragma omp simd reduction(+ : sum)
            for (unsigned int c = 0; c < nc; ++c)
              sum += T[r * nc + c] * M[c];
            parent[r] += sum;
          }
      }
  phase_time[0] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  if (n_mpi_processes > 1)
    sum_over_processes(yukawa_multipoles, mpi_communicator);
  phase_time[1] += seconds_since(start);

  // M2L by groups of pairs with the same scaled offset: the rotation of the
  // group and the translation along z of each pair, between unnormalized
  // bases, applied to batches of B pairs.
  start = std::chrono::steady_clock::now();
  {
    constexpr unsigned int B = 8;
    std::vector<double>    X(nc * B), Y(nc * B), work(nc * B),
      Tz_lanes(rotations->z_translation_size() * B);
    std::array<const double *, B> Tz;
    YE::Operators::Angles         angles;
    for (const auto &group : m2l_groups)
      {
        // v = c_B - c_A = -tau u
        rotations->angles(-group.tau, angles);
        for (unsigned int first = group.begin; first < group.end; first += B)
          {
            const unsigned int count = std::min(B, group.end - first);
            for (unsigned int j = 0; j < count; ++j)
              {
                const unsigned int pair = m2l_order[first + j];
                const unsigned int b    = my_m2l_pairs[pair].second;
                const double      *M    = &yukawa_multipoles[b * nc];
                const double      *scale =
                  &yukawa_outgoing_scale[boxes[b].level * (p + 1)];
                for (unsigned int n = 0; n <= p; ++n)
                  for (unsigned int a = n * n; a < (n + 1) * (n + 1); ++a)
                    X[a * B + j] = scale[n] * M[a];
                Tz[j] = &yukawa_tz[m2l_tz[pair]];
              }
            for (unsigned int j = count; j < B; ++j)
              {
                for (unsigned int a = 0; a < nc; ++a)
                  X[a * B + j] = 0.;
                Tz[j] = Tz[0];
              }
            std::fill(Y.begin(), Y.end(), 0.);
            rotations->translate<B>(angles,
                                    Tz.data(),
                                    X.data(),
                                    Y.data(),
                                    work.data(),
                                    Tz_lanes.data());
            for (unsigned int j = 0; j < count; ++j)
              {
                const unsigned int a_box =
                  my_m2l_pairs[m2l_order[first + j]].first;
                double       *L = &yukawa_locals[a_box * nc];
                const double *scale =
                  &yukawa_local_scale[boxes[a_box].level * (p + 1)];
                for (unsigned int n = 0; n <= p; ++n)
                  for (unsigned int a = n * n; a < (n + 1) * (n + 1); ++a)
                    L[a] += scale[n] * Y[a * B + j];
              }
          }
      }
  }
  phase_time[2] += seconds_since(start);

  // Downward pass and evaluation at the own points.
  start = std::chrono::steady_clock::now();
  for (const unsigned int b : my_boxes)
    if (boxes[b].parent != invalid)
      {
        const Box    &box = boxes[b];
        const double *T =
          &yukawa_l2l[(box.level * 8 + octant_of(box, boxes[box.parent])) * nc *
                      nc];
        const double *parent = &yukawa_locals[box.parent * nc];
        double       *L      = &yukawa_locals[b * nc];
        for (unsigned int r = 0; r < nc; ++r)
          {
            double sum = 0.;
#pragma omp simd reduction(+ : sum)
            for (unsigned int c = 0; c < nc; ++c)
              sum += T[r * nc + c] * parent[c];
            L[r] += sum;
          }
      }
  for (const unsigned int leaf : my_leaves)
    {
      const Box   &box = boxes[leaf];
      const double rho = yukawa_radius(box);
      for (unsigned int i = box.begin; i < box.end; ++i)
        u[i] += YE::evaluate_local(w,
                                   kappa,
                                   rho,
                                   &yukawa_locals[leaf * nc],
                                   sorted_points[i] - box.center);
    }
  phase_time[3] += seconds_since(start);
}



std::vector<double>
LaplaceFMM::evaluate_directly(const std::vector<double>       &charges,
                              const std::vector<Tensor<1, 3>> &dipoles,
                              const std::vector<unsigned int> &targets) const
{
  std::vector<double> u(targets.size(), 0.);
  for (unsigned int t = 0; t < targets.size(); ++t)
    {
      const Point<3> &x = points[targets[t]];
      for (unsigned int j = 0; j < points.size(); ++j)
        {
          if (j == targets[t])
            continue;
          const Tensor<1, 3> r         = x - points[j];
          const double       distance  = r.norm();
          const double       inverse   = 1. / distance;
          const double       screening = std::exp(-data.kappa * distance);
          if (!charges.empty())
            u[t] += charges[j] * screening * inverse;
          if (!dipoles.empty())
            u[t] += (dipoles[j] * r) * (1. + data.kappa * distance) *
                    screening * inverse * inverse * inverse;
        }
    }
  return u;
}



void
LaplaceFMM::print_statistics(std::ostream &out) const
{
  unsigned int n_leaves = 0, depth = 0;
  for (const auto &box : boxes)
    {
      n_leaves += box.is_leaf();
      depth = std::max(depth, box.level);
    }
  // Structure of the M2L pairs: in units of half the side of the smaller
  // box the offset between the centres is an integer vector, and pairs
  // with the same one share their translation operator.
  unsigned long                same_level = 0;
  std::set<std::array<int, 3>> offsets;
  for (const auto &[a, b] : m2l_pairs)
    {
      same_level += boxes[a].level == boxes[b].level;
      const double unit = std::min(boxes[a].half_side, boxes[b].half_side);
      std::array<int, 3> o;
      for (unsigned int d = 0; d < 3; ++d)
        o[d] = static_cast<int>(
          std::lround((boxes[a].center[d] - boxes[b].center[d]) / unit));
      offsets.insert(o);
    }
  unsigned long n_direct = 0;
  for (const auto &[a, b] : p2p_pairs)
    n_direct += static_cast<unsigned long>(boxes[a].end - boxes[a].begin) *
                (boxes[b].end - boxes[b].begin);

  // Time per evaluation of each phase, largest over the processes.
  std::array<double, 6> t;
  for (unsigned int i = 0; i < t.size(); ++i)
    t[i] = Utilities::MPI::max(phase_time[i] / std::max(1u, n_evaluations),
                               mpi_communicator);
  const double largest_share =
    Utilities::MPI::max(static_cast<double>(my_m2l_pairs.size()),
                        mpi_communicator) /
    std::max<double>(1, m2l_pairs.size());
  const unsigned int largest_groups =
    Utilities::MPI::max(static_cast<unsigned int>(m2l_groups.size()),
                        mpi_communicator);

  if (this_mpi_process == 0)
    {
      const auto precision = out.precision();
      out << std::setprecision(3) << "LaplaceFMM: " << points.size()
          << " points, "
          << (data.kappa > 0 ?
                "screened kernel, kappa " + std::to_string(data.kappa) + ", " :
                std::string())
          << "order " << data.order << ", theta " << data.theta << ", at most "
          << data.max_points_per_leaf << " points per leaf\n"
          << "  tree: " << boxes.size() << " boxes, " << n_leaves
          << " leaves, depth " << depth << "; setup " << setup_time << " s\n"
          << "  lists: " << m2l_pairs.size() << " M2L pairs (largest share "
          << 100 * largest_share << "%; "
          << 100. * same_level / std::max<std::size_t>(1, m2l_pairs.size())
          << "% within a level, " << offsets.size()
          << " distinct offsets, at most " << largest_groups
          << " groups on a process, "
          << m2l_rotations.size() /
               std::max<std::size_t>(1, rotations->rotation_size())
          << " rotations), " << p2p_pairs.size() << " P2P pairs with "
          << n_direct << " direct interactions ("
          << double(n_direct) / std::max<std::size_t>(1, points.size())
          << " per point)"
          << (data.kappa > 0 ?
                ", " + std::to_string(n_screened_out) + " pairs screened out" :
                std::string())
          << "\n"
          << "  seconds per evaluation, slowest process: P2M+M2M " << t[0]
          << ", sum of multipoles " << t[1] << ", M2L " << t[2] << ", L2L+L2P "
          << t[3] << ", P2P " << t[4] << ", sum of potentials " << t[5] << '\n'
          << std::setprecision(precision);
    }
}
