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

#ifndef galerkin_calderon_fmm_operator_h
#define galerkin_calderon_fmm_operator_h

#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <iosfwd>
#include <vector>

#include "galerkin_panel_pair_integrator.h"
#include "laplace_fmm.h"

/**
 * The four boundary integral operators of the symmetric Galerkin
 * formulation of the mixed problem of the Laplace equation, the single
 * layer V, the double layer K, its transpose and the weak hypersingular
 * operator W, applied without dense matrices: the products that
 * BEMProblem::solve_galerkin_mixed() takes from the matrices of
 * GalerkinPanelPairIntegrator::assemble_single_double_hypersingular(), up
 * to the truncation of the fast multipole method.
 *
 * The vectors are those of the mixed problem: the caller gives the indices
 * of the shape functions of every cell (a degree of freedom on cells of
 * both parts has a second index on the cells of one of them) and the
 * hanging-node constraints among them, and the operators are condensed,
 * C^T A C, as the dense matrices.
 *
 * As in GalerkinFMMOperator, the regular pairs use the tensor Gauss rule of
 * the given order on every cell, so that their sum is a sum over pairs of
 * quadrature points, which LaplaceFMM evaluates; with x the function of the
 * coefficients, w the weights and n the normals at the points:
 *  - V: the charges w_r x(y_r) / 4 pi, tested against the shape functions;
 *  - K: the dipoles w_r x(y_r) n_r / 4 pi, whose kernel is the double-layer
 *    kernel (x - y) . n_y / (4 pi |x - y|^3) of the integrator;
 *  - K^T: at each point x_q, n_q . sum_r w_r x(y_r) (y_r - x_q) / (4 pi
 *    |x_q - y_r|^3), whose component along e_c is the potential of the
 *    dipoles -e_c w_r x(y_r) / 4 pi: three evaluations;
 *  - W: the weak form, the single layer of the surface curl
 *    n x grad x against the curls of the shape functions: three
 *    evaluations with the components of the curl as charges.
 * The dipole sums converge one order more slowly than the charges and use
 * a second expansion order. The pairs that the integrator treats specially
 * enter as corrections: their V, K and W, and K the other way, from one
 * traversal, integrate_single_double_hypersingular_both(), minus the part
 * of the point sums, without the
 * coincident points of a cell with itself, which the multipole method
 * leaves out as well.
 *
 * Every process holds whole vectors, as the mixed solver does, evaluates
 * the fields on all cells, since the multipole method needs all sources,
 * tests on its own cells and keeps the corrections of its share of the
 * near pairs, which are dealt out by cost; the results are summed over the
 * processes. The Laplace kernel only.
 */
class GalerkinCalderonFMMOperator
{
public:
  enum class Operator
  {
    single_layer,
    double_layer,
    double_layer_transpose,
    hypersingular
  };

  struct AdditionalData
  {
    /// Order of the tensor Gauss rule on every cell, that of the regular
    /// pairs.
    unsigned int quadrature_order = 6;
    /// Near-singular distance ratio of the integrator: pairs closer than
    /// this ratio times the larger diameter may need the special rules.
    double near_distance_ratio = 0.;
    /// The expansions of the charge sums (V and W).
    LaplaceFMM::AdditionalData fmm;
    /// Increase of the order for the dipole sums (K and K^T).
    unsigned int dipole_order_increase = 2;
    /// The expansions of the products of the preconditioner
    /// (vmult_preconditioner()), which need not be accurate, only fixed:
    /// order and opening criterion, zero for those of the operator.
    unsigned int preconditioner_order = 0;
    double       preconditioner_theta = 0.;
  };

  explicit GalerkinCalderonFMMOperator(const MPI_Comm comm);

  /**
   * Compute the geometry and the shape functions with their surface curls
   * at the quadrature points, build the multipole trees, find the pairs of
   * cells that the integrator treats specially and integrate their
   * corrections. The cell indices are those of every active cell, by
   * active_cell_index(); collective.
   */
  void
  reinit(const DoFHandler<2, 3>                                  &dof_handler,
         const Mapping<2, 3>                                     &mapping,
         const std::vector<std::vector<types::global_dof_index>> &cell_indices,
         const types::global_dof_index                            n_indices,
         const AffineConstraints<double>                         &constraints,
         const GalerkinPanelPairIntegrator<3>                    &integrator,
         const AdditionalData                                    &data);

  /// dst = A src, whole vectors on every process; collective.
  void
  vmult(const Operator op, Vector<double> &dst, const Vector<double> &src) const;

  /// dst = V src or W src through the expansions of the preconditioner
  /// (AdditionalData::preconditioner_order), with the exact near field: a
  /// fixed approximation of the operator, for preconditioners; collective.
  void
  vmult_preconditioner(const Operator        op,
                       Vector<double>       &dst,
                       const Vector<double> &src) const;

  /**
   * The two rows of the block operator of the mixed problem,
   * dirichlet = V t - k K u and neumann = W u + k K^T t, whole vectors on
   * every process: V t and K u from one evaluation, with charges and
   * dipoles, and one sum over the processes for both; collective.
   */
  void
  vmult_rows(const Vector<double> &t,
             const Vector<double> &u,
             const double          k,
             Vector<double>       &dirichlet,
             Vector<double>       &neumann) const;

  /// The diagonal of V or of W, from the exact blocks of the near field,
  /// whole on every process.
  const Vector<double> &
  diagonal(const Operator op) const;

  /// Sizes, times of the setup and mean times of the products; collective.
  void
  print_statistics(std::ostream &out) const;

private:
  /// The function of the coefficients at every point: its values, and with
  /// curls its surface curl.
  void
  evaluate_field(const Vector<double>      &x,
                 std::vector<double>       &values,
                 std::vector<Tensor<1, 3>> *curls) const;

  /// Add the integrals of the values at the points against the shape
  /// functions of the own cells, condensed, to y.
  void
  test_values(const std::vector<double> &values, Vector<double> &y) const;

  /// Add K^T x, without the near field, for the values of x at the points,
  /// to y (tested on the own cells).
  void
  add_transpose_double_layer(const std::vector<double> &values,
                             Vector<double>            &y) const;

  /// Add W x, without the near field, for the curls of x at the points, to
  /// y (tested on the own cells), with the given expansions.
  void
  add_hypersingular(const std::vector<Tensor<1, 3>> &curls,
                    Vector<double>                  &y,
                    const LaplaceFMM                &fmm) const;

  /// V src or W src with the given expansions.
  void
  apply(const Operator        op,
        Vector<double>       &dst,
        const Vector<double> &src,
        const LaplaceFMM     &fmm) const;

  MPI_Comm     mpi_communicator;
  unsigned int n_mpi_processes, this_mpi_process;

  types::global_dof_index                           n_indices = 0;
  std::vector<std::vector<types::global_dof_index>> indices;
  AffineConstraints<double>                         constraints;
  unsigned int                                      dofs_per_cell = 0;
  unsigned int                                      n_q_points    = 0;
  unsigned int                                      n_cells       = 0;
  /// Cells whose test integrals this process computes.
  std::vector<unsigned int> own_cells;

  /// Data at the quadrature points, cell by cell, and the surface curls of
  /// the shape functions, at (point * dofs_per_cell + i).
  std::vector<dealii::Point<3>> points;
  std::vector<double>           JxW;
  std::vector<Tensor<1, 3>>     normals, shape_curls;
  /// Shape values at the reference quadrature points, shape(q, i).
  FullMatrix<double> shape;

  LaplaceFMM charge_fmm, dipole_fmm, preconditioner_fmm;
  bool       separate_preconditioner = false;

  /// Corrections of the near pairs of this process, condensed.
  SparsityPattern      near_sparsity;
  SparseMatrix<double> near_single, near_double, near_hypersingular;
  Vector<double>       single_diagonal, hypersingular_diagonal;

  unsigned long  n_near_pairs = 0;
  double         setup_geometry = 0, setup_tree = 0, setup_near = 0,
         setup_near_imbalance = 0;
  /// Times and counts of the products by operator, of the rows, and of
  /// the products of the preconditioner.
  mutable std::array<double, 6>       time_product = {{0, 0, 0, 0, 0, 0}};
  mutable std::array<unsigned int, 6> n_products   = {{0, 0, 0, 0, 0, 0}};
};

#endif
