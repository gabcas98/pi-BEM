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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>

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
LaplaceFMM::distribute_work()
{
  const unsigned int n_boxes = boxes.size();
  owner.assign(n_boxes, invalid);

  if (n_boxes > 0)
    {
      // Roots of the subtrees: the boxes of the first level at which there
      // are enough of them, together with the leaves above that level.
      const unsigned int wanted = 4 * n_mpi_processes;
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

      // Largest processing time first: the heaviest subtree goes to the
      // least loaded process. The weight is the number of points.
      std::sort(roots.begin(),
                roots.end(),
                [&](const unsigned int x, const unsigned int y) {
                  return boxes[x].end - boxes[x].begin >
                         boxes[y].end - boxes[y].begin;
                });
      std::vector<unsigned long> load(n_mpi_processes, 0);
      for (const unsigned int r : roots)
        {
          const unsigned int p =
            std::min_element(load.begin(), load.end()) - load.begin();
          owner[r] = p;
          load[p] += boxes[r].end - boxes[r].begin;
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
}



void
LaplaceFMM::particle_to_particle(const unsigned int  a,
                                 const unsigned int  b,
                                 const double       *q,
                                 const Tensor<1, 3> *d,
                                 double             *u) const
{
  const Box &A = boxes[a];
  const Box &B = boxes[b];
  for (unsigned int i = A.begin; i < A.end; ++i)
    {
      const Point<3> &x   = sorted_points[i];
      double          sum = 0;
      for (unsigned int j = B.begin; j < B.end; ++j)
        {
          if (j == i)
            continue;
          const Tensor<1, 3> r       = x - sorted_points[j];
          const double       inverse = 1. / r.norm();
          if (q != nullptr)
            sum += q[j] * inverse;
          if (d != nullptr)
            sum += (d[j] * r) * inverse * inverse * inverse;
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

  // Sources in the sorted numbering.
  std::vector<double>       q(charges.empty() ? 0 : n);
  std::vector<Tensor<1, 3>> d(dipoles.empty() ? 0 : n);
  for (unsigned int i = 0; i < q.size(); ++i)
    q[i] = charges[permutation[i]];
  for (unsigned int i = 0; i < d.size(); ++i)
    d[i] = dipoles[permutation[i]];

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

  start = std::chrono::steady_clock::now();
  for (const auto &[a, b] : my_m2l_pairs)
    LE::multipole_to_local(w,
                           &multipoles[b * nc],
                           boxes[a].center - boxes[b].center,
                           &locals[a * nc]);
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
    particle_to_particle(a,
                         b,
                         q.empty() ? nullptr : q.data(),
                         d.empty() ? nullptr : d.data(),
                         u.data());
  phase_time[4] += seconds_since(start);

  start = std::chrono::steady_clock::now();
  potentials.assign(n, 0.);
  for (unsigned int i = 0; i < n; ++i)
    potentials[permutation[i]] = u[i];
  if (n_mpi_processes > 1)
    sum_over_processes(potentials, mpi_communicator);
  phase_time[5] += seconds_since(start);
  ++n_evaluations;
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
          << 100 * largest_share << "%), "
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
