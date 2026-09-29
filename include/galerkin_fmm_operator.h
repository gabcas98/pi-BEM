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

#ifndef galerkin_fmm_operator_h
#define galerkin_fmm_operator_h

#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/non_matching/mapping_info.h>

#include <iosfwd>
#include <memory>
#include <vector>

#include "galerkin_panel_pair_integrator.h"
#include "laplace_fmm.h"

/**
 * The Galerkin single-layer operator of the Laplace kernel and the
 * right-hand sides of the Dirichlet problem, without forming a dense
 * matrix: the quantities that the dense path assembles in
 * GalerkinPanelPairIntegrator::assemble_single_layer_and_double_layer_rhs()
 * and assemble_single_layer_flux_rhs(), up to the truncation of the fast
 * multipole method.
 *
 * The dense assembly integrates a panel pair with the tensor Gauss rule of
 * the given order on both cells unless the integrator marks it as singular
 * or near-singular. With that rule on every cell, the sum over all regular
 * pairs is a sum over pairs of quadrature points,
 * @f[
 *   (V\mu)_i \approx \sum_q \psi_i(x_q) w_q \sum_{r \ne q}
 *     \frac{w_r\, \mu(x_r)}{4\pi |x_q - x_r|},
 * @f]
 * which LaplaceFMM evaluates with the charges $w_r \mu(x_r) / 4\pi$. The
 * regularised double-layer right-hand side is, in the same way,
 * @f[
 *   b_i = -\sum_q \psi_i(x_q) w_q \bigl(A(x_q) - \varphi(x_q) B(x_q)\bigr),
 *   \quad A(x) = \sum_{r} \frac{w_r \varphi(x_r)\, n_r \cdot (x - x_r)}
 *                              {4\pi |x - x_r|^3},
 * @f]
 * and $B$ the same with $\varphi = 1$: two evaluations with dipoles. The
 * manufactured right-hand side $\langle V\lambda, \psi_i \rangle$ is one
 * with the charges $w_r \lambda(x_r) / 4\pi$. The pairs that the
 * integrator treats specially enter through sparse corrections: their
 * exact contribution minus the one of the point rule, which is what the
 * multipole method has included for them. The coincident points of a cell
 * with itself are left out of both. The exact single-layer block and the
 * double-layer contribution of a pair come from one traversal of the
 * integrator, as in the fused dense assembly.
 *
 * The density is evaluated at the quadrature points, and the potentials
 * tested against the shape functions, with FEPointEvaluation on the
 * geometry that NonMatching::MappingInfo computes once for all cells; the
 * matrix-free FEEvaluation of deal.II needs a DoFHandler whose dimension
 * equals that of the space, which a surface mesh is not.
 *
 * With hanging-node constraints the operator is the condensed C^T V C of
 * the dense path: the constrained values are distributed from their masters
 * before the density is evaluated, and the tested potential is condensed
 * onto the masters. The exact near-field blocks are kept in a second
 * sparse matrix, from which the cell-block and diagonal preconditioners
 * take their entries.
 *
 * Every process evaluates the density on all cells, since the multipole
 * method needs all charges, and tests the potentials on its own cells. The
 * pairs of the near field, which cluster where the mesh is graded, are
 * gathered from all processes and dealt out in turn, so that each process
 * integrates the same number of them whatever its cells.
 */
class GalerkinFMMOperator
{
public:
  struct AdditionalData
  {
    /// Order of the tensor Gauss rule on every cell, the one of the regular
    /// pairs of the dense assembly.
    unsigned int quadrature_order = 6;
    /// Near-singular distance ratio of the integrator: pairs closer than
    /// this ratio times the larger diameter may need the special rules.
    double near_distance_ratio = 0.0;
    /// The expansions of the operator. The right-hand sides use the same
    /// tree with an order higher by rhs_order_increase, because dipoles
    /// converge one order more slowly than charges.
    LaplaceFMM::AdditionalData fmm;
    unsigned int               rhs_order_increase = 4;
  };

  /// The data of the right-hand sides whose near field is integrated
  /// together with that of the operator; either may be absent.
  struct RightHandSideData
  {
    /// Dirichlet datum of the double-layer right-hand side, and whether it
    /// is regularised by subtracting its value at the test point.
    const Function<3> *dirichlet  = nullptr;
    bool               regularize = true;
    /// Exact gradient, whose normal component is the flux of the
    /// manufactured right-hand side.
    const Function<3> *gradient = nullptr;
  };

  explicit GalerkinFMMOperator(const MPI_Comm comm);

  /**
   * Compute the geometry of the quadrature points, build the multipole
   * trees, find the pairs of cells that the integrator treats specially and
   * integrate their corrections, for the operator and the requested
   * right-hand sides.
   */
  void
  reinit(const DoFHandler<2, 3>               &dof_handler,
         const Mapping<2, 3>                  &mapping,
         const AffineConstraints<double>      &constraints,
         const GalerkinPanelPairIntegrator<3> &integrator,
         const IndexSet                       &locally_owned_dofs,
         const AdditionalData                 &data,
         const RightHandSideData              &rhs);

  /// dst = V src, with the constrained entries of src ignored.
  void
  vmult(TrilinosWrappers::MPI::Vector       &dst,
        const TrilinosWrappers::MPI::Vector &src) const;

  /// The double-layer right-hand side, with the sign of the dense path.
  void
  double_layer_rhs(TrilinosWrappers::MPI::Vector &rhs) const;

  /// The manufactured right-hand side <V lambda, psi_i>.
  void
  manufactured_rhs(TrilinosWrappers::MPI::Vector &rhs) const;

  /// The exact single-layer blocks of the pairs integrated specially,
  /// condensed like the operator, for preconditioners.
  const TrilinosWrappers::SparseMatrix &
  near_field_matrix() const
  {
    return near_exact;
  }

  /**
   * Set up the additive Schwarz preconditioner on overlapping patches: the
   * patch of a cell is the cell with the cells that share a vertex with
   * it, and its block is the principal submatrix of the condensed operator
   * on the free unknowns of the patch, exact for the pairs of the near
   * field and from the point rule, which is the rule of the dense path for
   * them, for the others. Principal submatrices of a positive definite
   * matrix are positive definite, so the preconditioner is, whatever the
   * truncation of the near field. Each process sets up the patches of its
   * own cells; collective.
   */
  void
  setup_patch_preconditioner();

  /// dst = sum over the patches of R^T B^{-1} R src, with the constrained
  /// entries of dst set to those of src; collective.
  void
  apply_patch_preconditioner(TrilinosWrappers::MPI::Vector       &dst,
                             const TrilinosWrappers::MPI::Vector &src) const;

  /// Sizes, times of the setup and mean times of vmult(); collective.
  void
  print_statistics(std::ostream &out) const;

private:
  /// Test the potentials at all points against the shape functions of the
  /// own cells, condense, add the near-field vector and sum over the
  /// processes into dst, multiplied by sign.
  void
  test_potentials(const std::vector<double>     &values_times_JxW,
                  const Vector<double>          &near_field,
                  const double                   sign,
                  TrilinosWrappers::MPI::Vector &dst) const;

  MPI_Comm     mpi_communicator;
  unsigned int n_mpi_processes, this_mpi_process;

  ObserverPointer<const DoFHandler<2, 3>> dof_handler;
  /// A copy of the hanging-node constraints of the caller.
  AffineConstraints<double> constraints;
  unsigned int              n_q_points;

  std::vector<DoFHandler<2, 3>::active_cell_iterator> cells;
  /// Cells whose test integrals this process computes.
  std::vector<unsigned int> own_cells;

  std::unique_ptr<NonMatching::MappingInfo<2, 3>>     mapping_info;
  mutable std::unique_ptr<FEPointEvaluation<1, 2, 3>> evaluator;
  /// Data at the quadrature points, cell by cell.
  std::vector<Point<3>>     quadrature_points;
  std::vector<double>       JxW;
  std::vector<Tensor<1, 3>> normals;
  std::vector<double>       dirichlet_values, flux_values;
  bool                      regularize = true;

  LaplaceFMM fmm, rhs_fmm;

  TrilinosWrappers::SparseMatrix near_exact, near_correction;
  /// Near-field corrections of the right-hand sides, without the sign of
  /// the double layer, summed over the processes.
  Vector<double> near_double_layer, near_manufactured;

  /// Shape values at the reference quadrature points, shape(q, i).
  FullMatrix<double> reference_shapes;

  /// The exact single-layer blocks integrated by this process, pair (t, r)
  /// with t <= r, row by row, kept until the patches are set up.
  std::vector<std::pair<unsigned int, unsigned int>> computed_pairs;
  std::vector<double>                                computed_blocks;

  /// The patches of the own cells: free unknowns, and the inverse of the
  /// block.
  std::vector<std::vector<types::global_dof_index>> patch_dofs;
  std::vector<FullMatrix<double>>                   patch_inverses;
  double                                            setup_patches = 0;

  unsigned long  n_near_pairs = 0;
  double         setup_geometry = 0, setup_tree = 0, setup_near = 0,
         setup_near_imbalance = 0;
  mutable double time_evaluate = 0, time_fmm = 0, time_test = 0,
                 time_near = 0, time_rhs = 0;
  mutable unsigned int n_vmult = 0;
};

#endif
