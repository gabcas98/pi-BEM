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
#include <iomanip>
#include <map>
#include <numeric>
#include <ostream>
#include <set>

using namespace dealii;
namespace LE = LaplaceExpansions;

namespace
{
  // Deepest level of the tree, a guard against coincident points.
  constexpr unsigned int max_level = 40;

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
  const auto start = std::chrono::steady_clock::now();
  data             = data_in;
  points           = points_in;
  build_tree();
  m2l_pairs.clear();
  p2p_pairs.clear();
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
  cost_m2l   = 1.;
  cost_p2p   = 1. / 1500.;
  cost_point = 0.05;
  distribute_work();
  const unsigned int n_expansions =
    boxes.size() * LE::n_coefficients(data.order);
  multipoles.assign(n_expansions, 0.);
  locals.assign(n_expansions, 0.);
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
    0.5 * std::max({upper[0] - lower[0], upper[1] - lower[1],
                    upper[2] - lower[2]}) *
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
      unsigned int next = box.begin;
      boxes[b].first_child = boxes.size();
      for (unsigned int o = 0; o < 8; ++o)
        {
          if (octant[o].empty())
            continue;
          Box child;
          child.half_side = 0.5 * box.half_side;
          for (unsigned int d = 0; d < 3; ++d)
            child.center[d] = box.center[d] + ((o >> d) & 1 ?
                                                 child.half_side :
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
  if (A.radius + B.radius < data.theta * A.center.distance(B.center))
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
      // Roots of the subtrees: the boxes of the first level at which there
      // are enough of them, together with the leaves above that level.
      const unsigned int wanted = 16 * n_mpi_processes;
      unsigned int       depth  = 0;
      for (const auto &box : boxes)
        depth = std::max(depth, box.level);
      std::vector<unsigned int> roots;
      for (unsigned int level = 0; level <= depth; ++level)
        {
          roots.clear();
          for (unsigned int b = 0; b < n_boxes; ++b)
            if (boxes[b].level == level ||
                (boxes[b].level < level && boxes[b].is_leaf()))
              roots.push_back(b);
          if (roots.size() >= wanted)
            break;
        }
      std::vector<bool> is_root(n_boxes, false);
      for (const unsigned int r : roots)
        is_root[r] = true;

      // Modelled cost of every subtree, accumulated from the leaves up to
      // the roots (children follow their parents, so a reverse sweep).
      std::vector<double> cost(n_boxes, 0.);
      for (unsigned int b = 0; b < n_boxes; ++b)
        {
          cost[b] = cost_m2l * m2l_count[b] + cost_p2p * p2p_count[b];
          if (boxes[b].is_leaf())
            cost[b] += cost_point * (boxes[b].end - boxes[b].begin);
        }
      for (unsigned int b = n_boxes - 1; b > 0; --b)
        if (!is_root[b])
          cost[boxes[b].parent] += cost[b];

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
      const Box   &A    = boxes[my_m2l_pairs[i].first];
      const Box   &B    = boxes[my_m2l_pairs[i].second];
      const double unit = std::min(A.half_side, B.half_side);
      std::array<int, 3> key;
      for (unsigned int d = 0; d < 3; ++d)
        key[d] = static_cast<int>(
          std::lround((A.center[d] - B.center[d]) / unit));
      m2l_unit[i] = unit;
      by_offset[key].push_back(i);
    }
  m2l_groups.clear();
  m2l_order.clear();
  for (auto &[key, members] : by_offset)
    {
      std::sort(members.begin(),
                members.end(),
                [&](const unsigned int x, const unsigned int y) {
                  return my_m2l_pairs[x] < my_m2l_pairs[y];
                });
      M2LGroup group;
      group.tau   = Tensor<1, 3>({double(key[0]), double(key[1]),
                                  double(key[2])});
      group.begin = m2l_order.size();
      m2l_order.insert(m2l_order.end(), members.begin(), members.end());
      group.end = m2l_order.size();
      m2l_groups.push_back(group);
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
    my_interactions += static_cast<unsigned long>(boxes[a].end -
                                                  boxes[a].begin) *
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
  cost_m2l = Utilities::MPI::sum(m2l_time, mpi_communicator) /
             std::max(1., total_m2l);
  cost_p2p = Utilities::MPI::sum(p2p_time, mpi_communicator) /
             std::max(1., total_p2p);
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
    __m256d y = _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(safe)));
    const __m256d half_r2      = _mm256_mul_pd(_mm256_set1_pd(0.5), safe);
    const __m256d three_halves = _mm256_set1_pd(1.5);
    y = _mm256_mul_pd(
      y, _mm256_fnmadd_pd(half_r2, _mm256_mul_pd(y, y), three_halves));
    y = _mm256_mul_pd(
      y, _mm256_fnmadd_pd(half_r2, _mm256_mul_pd(y, y), three_halves));
    return _mm256_and_pd(y, positive);
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



template <bool with_charges, bool with_dipoles>
void
LaplaceFMM::particle_to_particle(const unsigned int a,
                                 const unsigned int b,
                                 const double      *q,
                                 const double      *d,
                                 double            *u) const
{
  const Box    &A = boxes[a];
  const Box    &B = boxes[b];
  const double *x = sorted_coordinates[0].data();
  const double *y = sorted_coordinates[1].data();
  const double *z = sorted_coordinates[2].data();
  const unsigned int n  = points.size();
  const double      *dx = d, *dy = d + n, *dz = d + 2 * n;
  for (unsigned int i = A.begin; i < A.end; ++i)
    {
      const double xi = x[i], yi = y[i], zi = z[i];
      double       sum = 0;
      unsigned int j   = B.begin;
#if defined(__AVX2__) && defined(__FMA__)
      const __m256d vx = _mm256_set1_pd(xi), vy = _mm256_set1_pd(yi),
                    vz  = _mm256_set1_pd(zi);
      __m256d       acc = _mm256_setzero_pd();
      for (; j + 4 <= B.end; j += 4)
        {
          const __m256d rx = _mm256_sub_pd(vx, _mm256_loadu_pd(x + j));
          const __m256d ry = _mm256_sub_pd(vy, _mm256_loadu_pd(y + j));
          const __m256d rz = _mm256_sub_pd(vz, _mm256_loadu_pd(z + j));
          const __m256d r2 = _mm256_fmadd_pd(
            rx, rx, _mm256_fmadd_pd(ry, ry, _mm256_mul_pd(rz, rz)));
          const __m256d inverse = inverse_sqrt(r2);
          if constexpr (with_charges)
            acc = _mm256_fmadd_pd(_mm256_loadu_pd(q + j), inverse, acc);
          if constexpr (with_dipoles)
            {
              const __m256d dot = _mm256_fmadd_pd(
                _mm256_loadu_pd(dx + j),
                rx,
                _mm256_fmadd_pd(_mm256_loadu_pd(dy + j),
                                ry,
                                _mm256_mul_pd(_mm256_loadu_pd(dz + j), rz)));
              const __m256d inverse3 =
                _mm256_mul_pd(inverse, _mm256_mul_pd(inverse, inverse));
              acc = _mm256_fmadd_pd(dot, inverse3, acc);
            }
        }
      sum = horizontal_sum(acc);
#endif
      // The remainder, or all of it without AVX2. The point itself, at
      // distance zero, is masked out without a division by zero.
      for (; j < B.end; ++j)
        {
          const double rx = xi - x[j], ry = yi - y[j], rz = zi - z[j];
          const double r2       = rx * rx + ry * ry + rz * rz;
          const bool   distinct = r2 > 0.;
          const double inverse =
            (distinct ? 1. : 0.) / std::sqrt(distinct ? r2 : 1.);
          if constexpr (with_charges)
            sum += q[j] * inverse;
          if constexpr (with_dipoles)
            sum += (dx[j] * rx + dy[j] * ry + dz[j] * rz) * inverse *
                   inverse * inverse;
        }
      u[i] += sum;
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
  const unsigned int p  = data.order;
  const unsigned int nc = LE::n_coefficients(p);

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
  std::fill(multipoles.begin(), multipoles.end(), 0.);
  std::fill(locals.begin(), locals.end(), 0.);
  LE::Workspace w(p);

  // Upward pass: multipoles of the own leaves, translated up through the
  // own and the shared boxes (children after parents, so reversed).
  auto start = std::chrono::steady_clock::now();
  for (const unsigned int leaf : my_leaves)
    {
      const Box        &box = boxes[leaf];
      LE::complex *const M  = &multipoles[leaf * nc];
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

  // M2L by groups of pairs with the same scaled offset: one operator per
  // group, applied to batches of B pairs. A pair scales its multipole by
  // u^{-n} on the way in and the result by u^{-(k+1)} on the way out.
  start = std::chrono::steady_clock::now();
  {
    constexpr unsigned int B       = 8;
    const unsigned int     columns = (p + 1) * (p + 1);
    std::vector<double>    T_re(nc * columns), T_im(nc * columns);
    std::vector<double>    X_re(columns * B), X_im(columns * B);
    std::vector<double>    Y_re(nc * B), Y_im(nc * B);
    std::vector<double>    power(p + 2);
    for (const auto &group : m2l_groups)
      {
        LE::multipole_to_local_operator(w,
                                        group.tau,
                                        T_re.data(),
                                        T_im.data());
        for (unsigned int first = group.begin; first < group.end; first += B)
          {
            const unsigned int count = std::min(B, group.end - first);
            std::fill(X_re.begin(), X_re.end(), 0.);
            std::fill(X_im.begin(), X_im.end(), 0.);
            for (unsigned int j = 0; j < count; ++j)
              {
                const unsigned int pair = m2l_order[first + j];
                const LE::complex *M =
                  &multipoles[my_m2l_pairs[pair].second * nc];
                const double inverse_unit = 1. / m2l_unit[pair];
                double       scale        = 1.;
                for (unsigned int n = 0; n <= p; ++n, scale *= inverse_unit)
                  for (unsigned int m = 0; m <= n; ++m)
                    {
                      const LE::complex v = scale * M[LE::index(n, m)];
                      X_re[(n * n + n + m) * B + j] = v.real();
                      X_im[(n * n + n + m) * B + j] = v.imag();
                      if (m > 0)
                        {
                          // M_n^{-m} = (-1)^m conj(M_n^m)
                          const double sign = (m & 1) ? -1. : 1.;
                          X_re[(n * n + n - m) * B + j] = sign * v.real();
                          X_im[(n * n + n - m) * B + j] = -sign * v.imag();
                        }
                    }
              }
            LE::multipole_to_local_batch<B>(p,
                                            T_re.data(),
                                            T_im.data(),
                                            X_re.data(),
                                            X_im.data(),
                                            Y_re.data(),
                                            Y_im.data());
            for (unsigned int j = 0; j < count; ++j)
              {
                const unsigned int pair = m2l_order[first + j];
                LE::complex       *L = &locals[my_m2l_pairs[pair].first * nc];
                const double       inverse_unit = 1. / m2l_unit[pair];
                double             scale        = inverse_unit;
                for (unsigned int k = 0; k <= p; ++k, scale *= inverse_unit)
                  for (unsigned int l = 0; l <= k; ++l)
                    {
                      const unsigned int r = LE::index(k, l);
                      L[r] += LE::complex(scale * Y_re[r * B + j],
                                          scale * Y_im[r * B + j]);
                    }
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
        const Box &box = boxes[b];
        LE::local_to_local(w,
                           &locals[box.parent * nc],
                           box.center - boxes[box.parent].center,
                           &locals[b * nc]);
      }
  std::vector<double> u(n, 0.);
  for (const unsigned int leaf : my_leaves)
    {
      const Box &box = boxes[leaf];
      for (unsigned int i = box.begin; i < box.end; ++i)
        u[i] += LE::evaluate_local(w,
                                   &locals[leaf * nc],
                                   sorted_points[i] - box.center);
    }
  phase_time[3] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  for (const auto &[a, b] : my_p2p_pairs)
    if (d.empty())
      particle_to_particle<true, false>(a, b, q.data(), nullptr, u.data());
    else if (q.empty())
      particle_to_particle<false, true>(
        a, b, nullptr, d_components.data(), u.data());
    else
      particle_to_particle<true, true>(
        a, b, q.data(), d_components.data(), u.data());
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
          const Tensor<1, 3> r       = x - points[j];
          const double       inverse = 1. / r.norm();
          if (!charges.empty())
            u[t] += charges[j] * inverse;
          if (!dipoles.empty())
            u[t] += (dipoles[j] * r) * inverse * inverse * inverse;
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

  if (this_mpi_process == 0)
    {
      const auto precision = out.precision();
      out << std::setprecision(3) << "LaplaceFMM: " << points.size()
          << " points, order "
          << data.order << ", theta " << data.theta << ", at most "
          << data.max_points_per_leaf << " points per leaf\n"
          << "  tree: " << boxes.size() << " boxes, " << n_leaves
          << " leaves, depth " << depth << "; setup " << setup_time << " s\n"
          << "  lists: " << m2l_pairs.size() << " M2L pairs (largest share "
          << 100 * largest_share << "%; "
          << 100. * same_level / std::max<std::size_t>(1, m2l_pairs.size())
          << "% within a level, " << offsets.size() << " distinct offsets), "
          << p2p_pairs.size() << " P2P pairs with " << n_direct
          << " direct interactions ("
          << double(n_direct) / std::max<std::size_t>(1, points.size())
          << " per point)\n"
          << "  seconds per evaluation, slowest process: P2M+M2M " << t[0]
          << ", sum of multipoles " << t[1] << ", M2L " << t[2]
          << ", L2L+L2P " << t[3] << ", P2P " << t[4]
          << ", sum of potentials " << t[5] << '\n'
          << std::setprecision(precision);
    }
}
