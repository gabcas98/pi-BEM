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
#include <cmath>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>

#include "laplace_expansions.h"
#include "yukawa_expansions.h"

/**
 * Fast multipole method for the Laplace kernel on a fixed set of points,
 * each of which is both a source and a target:
 * @f[
 *   u_i = \sum_{j \ne i} \frac{q_j}{|x_i - x_j|}
 *         + \frac{d_j \cdot (x_i - x_j)}{|x_i - x_j|^3},
 * @f]
 * with charges $q_j$ and dipole moments $d_j$; the second term is
 * $d_j \cdot \nabla_y |x_i - y|^{-1}$ at $y = x_j$. With a screening
 * parameter $\kappa > 0$ the kernel is instead the screened one,
 * $e^{-\kappa r}/r$, of the modified Helmholtz equation
 * $-\Delta u + \kappa^2 u = 0$, and the dipole term
 * $d_j \cdot (x_i - x_j)(1 + \kappa r) e^{-\kappa r} / r^3$; the tree, the
 * lists and the split of the work are the same, the expansions those of
 * yukawa_expansions.h. Pairs of boxes so far apart that
 * $e^{-\kappa (|c_A - c_B| - R_A - R_B)}$ is below $10^{-17}$ are dropped:
 * their interaction is negligible against that of the near sources.
 *
 * Nothing in the class knows about boundary elements: in a Galerkin method
 * the points are the quadrature points, the charges and dipoles carry the
 * density and the quadrature weights, and the self-interaction $j = i$ is
 * the one that the singular quadrature replaces.
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
 * subtrees of the targets, each small enough against the total work, which
 * are distributed over the processes by their modelled cost; the few boxes
 * above the subtrees are handled by every process. All
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
    /// Screening parameter kappa of the kernel e^{-kappa r}/r; zero for the
    /// Laplace kernel.
    double kappa = 0.;
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

  /**
   * Potentials and their gradients at all points, on every process, of the
   * given charges: sum_{j != i} q_j / |x_i - x_j| and its gradient with
   * respect to x_i. The Laplace kernel only. The gradient of the local
   * expansions converges one order more slowly than the potential.
   */
  void
  evaluate_with_gradient(const std::vector<double>         &charges,
                         std::vector<double>               &potentials,
                         std::vector<dealii::Tensor<1, 3>> &gradients) const;

  /**
   * The potentials at all points, on every process, of several sets of
   * charges at once, Laplace kernel: the tree passes, the direct
   * interactions, whose kernel is computed once for all sets, and the sums
   * over the processes are shared, the translations repeated for each set.
   */
  void
  evaluate_multiple(const std::vector<std::vector<double>> &charges,
                    std::vector<std::vector<double>>       &potentials) const;

  /**
   * In one pass, the sums of the mixed Galerkin problem, Laplace kernel:
   * the potentials and their gradients of the charges q, the potentials of
   * the dipoles d, and the potentials of three more sets of charges c, all
   * indexed like the points; the same as evaluate_with_gradient(q),
   * evaluate({}, d) and evaluate_multiple(c) with the expansions of this
   * object. The tree passes, the rotations of the translations, the kernel
   * of the direct interactions and the sums over the processes are shared.
   */
  void
  evaluate_mixed(const std::vector<double>                &q,
                 const std::vector<dealii::Tensor<1, 3>>  &d,
                 const std::array<std::vector<double>, 3> &c,
                 std::vector<double>                      &potentials_q,
                 std::vector<dealii::Tensor<1, 3>>        &gradients_q,
                 std::vector<double>                      &potentials_d,
                 std::array<std::vector<double>, 3>       &potentials_c) const;

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

  /// The memory of the multipole expansions of all boxes for the given
  /// number of sets of sources, in bytes.
  std::size_t
  expansion_bytes(const unsigned int sets) const;

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
    unsigned int parent = invalid;
    unsigned int level  = 0;

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
   * evaluations. Starting from the whole tree, the heaviest subtree is
   * split into those of its children until each carries at most
   * 1/(16 n_processes) of the total or is a leaf, which adapts the cut to
   * graded meshes, where the work gathers in a few deep subtrees. The
   * subtrees are then dealt out, heaviest first, and each process receives
   * work in proportion to its speed. Before the first evaluation the model
   * uses rough constants and equal speeds; after it, both come from the
   * measured times, which also captures processors of different speed,
   * such as the performance and efficiency cores of a hybrid chip.
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
  template <bool with_charges, bool with_dipoles, bool screened>
  void
  particle_to_particle(const unsigned int a,
                       const unsigned int b,
                       const double      *q,
                       const double      *d,
                       double            *u) const;

  /// Direct interaction of the k sets of charges q (k arrays of the points,
  /// one after the other) of the source leaf b with the points of the
  /// target leaf a, potentials u (k arrays), sorted numbering, Laplace
  /// kernel.
  template <unsigned int k>
  void
  particle_to_particle_multiple(const unsigned int a,
                                const unsigned int b,
                                const double      *q,
                                double            *u) const;

  /// The far field of k sets of charges (k arrays of the points), Laplace
  /// kernel, potentials added to u (k arrays); with dipoles, the set 1 is
  /// that of these dipoles (its charges are not used), and with gradient
  /// the gradient of the potential of the set 0 is added to it (three
  /// arrays).
  void
  far_field_laplace_multiple(
    const unsigned int                       k,
    const std::vector<double>               &q,
    std::vector<double>                     &u,
    const std::vector<dealii::Tensor<1, 3>> *dipoles  = nullptr,
    std::vector<double>                     *gradient = nullptr) const;

  /// The direct interactions of evaluate_mixed(): q holds five arrays of
  /// the points, the charges with the gradient, the dipole set (unused),
  /// and three more sets of charges; d the three components of the dipoles;
  /// u five arrays of potentials and g three of the gradient of the first.
  void
  particle_to_particle_mixed(const unsigned int a,
                             const unsigned int b,
                             const double      *q,
                             const double      *d,
                             double            *u,
                             double            *g) const;

  /// Direct interaction of the charges of the source leaf b with the points
  /// of the target leaf a, potentials u and gradients g (three arrays, one
  /// per component), sorted numbering, Laplace kernel.
  void
  particle_to_particle_gradient(const unsigned int a,
                                const unsigned int b,
                                const double      *q,
                                double            *u,
                                double            *g) const;

  /// The far field through the expansions, from the multipoles of the own
  /// leaves to the potentials u at the own points, for each kernel, and for
  /// The multipoles of all boxes, k sets of n_boxes, converted in place
  /// to the real coefficients of the M2L translations, YukawaExpansions'
  /// (n + 1)^2 doubles at the start of each expansion, coefficient (n, m)
  /// scaled by h^{-n} with h the half side of the box.
  void
  real_multipoles(LaplaceExpansions::complex *M, const unsigned int k) const;

  /// The local expansions of the own boxes, k sets of n_boxes, accumulated
  /// by the M2L translations in the real coefficients, converted in place
  /// back to the complex ones, coefficient (n, m) scaled by h^{-(n+1)}.
  void
  complex_locals(LaplaceExpansions::complex *L, const unsigned int k) const;

  /// The M2L translations of the own pairs, by groups of the same scaled
  /// offset: the rotation and the translation along z of a group are shared
  /// by its pairs and by the k sets of expansions, which fill the batches
  /// together. On the expansions of real_multipoles(), adding to the real
  /// coefficients of complex_locals() (Laplace kernel).
  void
  multipole_to_local(const double *M, double *L, const unsigned int k) const;

  /// the Laplace kernel optionally their gradients g (three arrays of the
  /// size of u, one per component); the times of the phases are added to
  /// phase_time[0..3].
  void
  far_field_laplace(const std::vector<double>               &q,
                    const std::vector<dealii::Tensor<1, 3>> &d,
                    std::vector<double>                     &u,
                    std::vector<double>                     *g = nullptr) const;
  void
  far_field_yukawa(const std::vector<double>               &q,
                   const std::vector<dealii::Tensor<1, 3>> &d,
                   std::vector<double>                     &u) const;

  /// Normalization radius of the screened expansions of a box.
  static double
  yukawa_radius(const Box &box)
  {
    return std::sqrt(3.) * box.half_side;
  }

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
  mutable std::vector<unsigned int> my_leaves, my_boxes;
  mutable std::vector<std::pair<unsigned int, unsigned int>> my_m2l_pairs,
    my_p2p_pairs;

  /**
   * The M2L pairs of this process grouped by their offset: in units u of
   * half the side of the smaller box, the offset of the centres is an
   * integer vector tau, and the pairs with the same tau share the rotation
   * that takes tau onto the z-axis (YukawaExpansions::Operators), applied
   * to the pairs in batches with their translation along z. For the
   * Laplace kernel the translation along z depends on |tau| alone, so the
   * whole M2L operator is shared by the group whatever the size of its
   * boxes; the expansions are converted to real coefficients scaled by
   * powers of u on the way in and back on the way out
   * (LaplaceExpansions::RealBasis).
   */
  struct M2LGroup
  {
    dealii::Tensor<1, 3> tau;
    unsigned int         begin, end; // range in m2l_order
    /// Start in m2l_rotations of the blocks of the rotation about y of the
    /// polar angle of -tau (Laplace kernel).
    std::size_t rotation = 0;
  };
  mutable std::vector<M2LGroup>     m2l_groups;
  mutable std::vector<unsigned int> m2l_order;
  /// The unit u of every pair of my_m2l_pairs.
  mutable std::vector<double> m2l_unit;

  /// Expansions of all boxes, n_coefficients(order) each.
  mutable std::vector<LaplaceExpansions::complex> multipoles, locals;

  /// The rotations and translations along z of the real harmonics, with
  /// which both kernels apply their M2L translations.
  std::unique_ptr<YukawaExpansions::Operators> rotations;
  /// The factors between the complex Laplace coefficients and the real
  /// ones (LaplaceExpansions::RealBasis).
  std::vector<double> real_outgoing, real_local;

  /// The screened expansions of all boxes, and their translations: the
  /// dense M2M and L2L matrices of a child at each level in each octant of
  /// its parent, at (level * 8 + octant) times the square of the number of
  /// coefficients.
  mutable std::vector<double> yukawa_multipoles, yukawa_locals;
  std::vector<double>         yukawa_m2m, yukawa_l2l;
  /// The normalization of the screened expansions of each level, 1/k_n and
  /// i_n of kappa times its radius, at level * (order + 1) + n: the M2L
  /// translations work between the unnormalized bases.
  std::vector<double> yukawa_outgoing_scale, yukawa_local_scale;
  /**
   * The M2L translations along z of this process, stored one after another,
   * and the start of the one of each pair of my_m2l_pairs. A group shares
   * the rotation of its offset tau. For the Laplace kernel the translation
   * depends on |tau| alone, in units u; with screening on the distance
   * |tau| u, so that the pairs of a group, of different sizes, may need
   * different ones.
   */
  mutable std::vector<double> yukawa_tz;
  /// The rotations about y of the polar angles of the groups
  /// (YukawaExpansions::Operators::rotation_blocks()), shared by the
  /// groups of the same polar angle (Laplace kernel).
  mutable std::vector<double>       m2l_rotations;
  mutable std::vector<unsigned int> m2l_tz;
  /// Pairs of boxes dropped because the screening makes them negligible.
  unsigned long n_screened_out = 0;

  /// Cost model: seconds per M2L, per P2P interaction and per point, and
  /// the relative speed of every process.
  mutable double cost_m2l = 1., cost_p2p = 1. / 1500., cost_point = 0.05;
  mutable std::vector<double> speed;
  /// Counts per box, for the cost model: M2L translations and P2P
  /// interactions for which the box is the target.
  std::vector<unsigned int>  m2l_count;
  std::vector<unsigned long> p2p_count;

  /// Wall-clock time of the phases, summed over the calls of evaluate().
  mutable std::array<double, 6> phase_time    = {{0, 0, 0, 0, 0, 0}};
  mutable unsigned int          n_evaluations = 0;
  double                        setup_time    = 0;
};

#endif
