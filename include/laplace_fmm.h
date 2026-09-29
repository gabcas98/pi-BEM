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

#ifndef laplace_fmm_h
#define laplace_fmm_h

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <mpi.h>

#include <array>
#include <iosfwd>
#include <utility>
#include <vector>

#include "laplace_expansions.h"

/**
 * Fast multipole method for the Laplace kernel on a fixed set of points,
 * each of which is both a source and a target:
 * @f[
 *   u_i = \sum_{j \ne i} \frac{q_j}{|x_i - x_j|}
 *         + \frac{d_j \cdot (x_i - x_j)}{|x_i - x_j|^3},
 * @f]
 * with charges $q_j$ and dipole moments $d_j$; the second term is
 * $d_j \cdot \nabla_y |x_i - y|^{-1}$ at $y = x_j$. Nothing in the class
 * knows about boundary elements: in a Galerkin method the points are the
 * quadrature points, the charges and dipoles carry the density and the
 * quadrature weights, and the self-interaction $j = i$ is the one that the
 * singular quadrature replaces.
 *
 * The points are sorted into an adaptive octree whose leaves hold at most
 * a given number of points. A dual traversal of the tree pairs target and
 * source boxes: a pair is approximated by a multipole-to-local translation
 * when $R_A + R_B < \theta\, |c_A - c_B|$, where $c$ is the centre of a box
 * and $R$ the largest distance of its points from it, and computed
 * directly when both boxes are leaves; otherwise the larger box is split.
 * Measuring $R$ on the points rather than on the box makes the criterion
 * sharp for points on a surface, which fill only a thin shell of a box.
 * The lists of both kinds of pairs are built once, in reinit().
 *
 * Every process holds the whole tree and all points. The work is split by
 * subtrees of the targets: the subtrees rooted at the first level with
 * enough boxes are distributed over the processes, balancing their points,
 * and the few boxes above that level are handled by every process. All
 * steps are linear in the sources, so each process forms the multipoles of
 * its own leaves, the partial expansions are summed over the processes, and
 * each process computes the potentials of its own points; a final sum over
 * the processes gives every process all potentials.
 */
class LaplaceFMM
{
public:
  struct AdditionalData
  {
    /// Truncation order of the expansions.
    unsigned int order = 8;
    /// Largest number of points in a leaf.
    unsigned int max_points_per_leaf = 64;
    /// Opening criterion of the dual traversal, in (0, 1).
    double theta = 0.5;
  };

  explicit LaplaceFMM(const MPI_Comm comm = MPI_COMM_WORLD);

  /// Build the tree and the interaction lists for the given points.
  void
  reinit(const std::vector<dealii::Point<3>> &points,
         const AdditionalData                &data);

  /**
   * Potentials at all points, on every process, of the given charges and
   * dipole moments, both indexed like the points. Either may be empty.
   */
  void
  evaluate(const std::vector<double>               &charges,
           const std::vector<dealii::Tensor<1, 3>> &dipoles,
           std::vector<double>                     &potentials) const;

  /// Direct sum at the listed targets, for tests.
  std::vector<double>
  evaluate_directly(const std::vector<double>               &charges,
                    const std::vector<dealii::Tensor<1, 3>> &dipoles,
                    const std::vector<unsigned int>         &targets) const;

  /// Sizes of the tree and of the lists, and the time of each phase.
  void
  print_statistics(std::ostream &out) const;

  unsigned int
  n_points() const
  {
    return points.size();
  }

private:
  static constexpr unsigned int invalid = static_cast<unsigned int>(-1);

  struct Box
  {
    dealii::Point<3> center;
    double           half_side;
    /// Largest distance of the points of the box from its centre.
    double       radius;
    unsigned int begin, end; // range in the sorted points
    unsigned int first_child = invalid, n_children = 0;
    unsigned int parent      = invalid;
    unsigned int level       = 0;

    bool
    is_leaf() const
    {
      return n_children == 0;
    }
  };

  void
  build_tree();

  void
  traverse(const unsigned int a, const unsigned int b);

  /**
   * Deal the subtrees of the targets out to the processes. The cost of a
   * subtree is modelled as a M2L time per translation, a P2P time per
   * interaction and a time per point for its multipoles and local
   * evaluations, and each process receives work in proportion to its
   * speed. Before the first evaluation the model uses rough constants and
   * equal speeds; after it, both come from the measured times, which also
   * captures processors of different speed, such as the performance and
   * efficiency cores of a hybrid chip.
   */
  void
  distribute_work() const;

  /// Calibrate the cost model and the speeds on the last evaluation, and
  /// deal the work out again; collective.
  void
  rebalance(const std::array<double, 6> &times) const;

  /// Direct interaction of the points of the source leaf b with those of
  /// the target leaf a, in sorted numbering; q holds the charges and d the
  /// three components of the dipoles, each of the used kinds.
  template <bool with_charges, bool with_dipoles>
  void
  particle_to_particle(const unsigned int a,
                       const unsigned int b,
                       const double      *q,
                       const double      *d,
                       double            *u) const;

  MPI_Comm     mpi_communicator;
  unsigned int n_mpi_processes, this_mpi_process;

  AdditionalData data;

  /// The points in the order of the user, and sorted by boxes.
  std::vector<dealii::Point<3>> points, sorted_points;
  /// sorted_points[i] = points[permutation[i]]
  std::vector<unsigned int> permutation;
  /// The sorted coordinates, one array per component, for the direct
  /// interactions, which run over them in vectorized loops.
  std::array<std::vector<double>, 3> sorted_coordinates;

  /// Boxes in breadth-first order: parents precede their children, which
  /// are consecutive.
  std::vector<Box> boxes;

  /// Pairs (target box, source box) of the two lists.
  std::vector<std::pair<unsigned int, unsigned int>> m2l_pairs, p2p_pairs;

  /// Process that computes the targets of each box, or invalid for the
  /// boxes above the subtrees, which every process handles. The split
  /// adapts to the measured speeds, hence mutable.
  mutable std::vector<unsigned int> owner;
  /// Work of this process: leaves whose multipoles and potentials it forms,
  /// M2L and P2P pairs whose target it owns (or shares), boxes whose
  /// expansions it translates.
  mutable std::vector<unsigned int>                          my_leaves, my_boxes;
  mutable std::vector<std::pair<unsigned int, unsigned int>> my_m2l_pairs,
    my_p2p_pairs;

  /**
   * The M2L pairs of this process grouped by their offset: in units u of
   * half the side of the smaller box, the offset of the centres is an
   * integer vector tau, and the pairs with the same tau share one
   * translation operator (see
   * LaplaceExpansions::multipole_to_local_operator()), which is built once
   * per group and applied to the pairs in batches.
   */
  struct M2LGroup
  {
    dealii::Tensor<1, 3> tau;
    unsigned int         begin, end; // range in m2l_order
  };
  mutable std::vector<M2LGroup>     m2l_groups;
  mutable std::vector<unsigned int> m2l_order;
  /// The unit u of every pair of my_m2l_pairs.
  mutable std::vector<double> m2l_unit;

  /// Expansions of all boxes, n_coefficients(order) each.
  mutable std::vector<LaplaceExpansions::complex> multipoles, locals;

  /// Cost model: seconds per M2L, per P2P interaction and per point, and
  /// the relative speed of every process.
  mutable double              cost_m2l = 1., cost_p2p = 1. / 1500.,
                 cost_point = 0.05;
  mutable std::vector<double> speed;
  /// Counts per box, for the cost model: M2L translations and P2P
  /// interactions for which the box is the target.
  std::vector<unsigned int>  m2l_count;
  std::vector<unsigned long> p2p_count;

  /// Wall-clock time of the phases, summed over the calls of evaluate().
  mutable std::array<double, 6> phase_time = {{0, 0, 0, 0, 0, 0}};
  mutable unsigned int          n_evaluations = 0;
  double                        setup_time    = 0;
};

#endif
