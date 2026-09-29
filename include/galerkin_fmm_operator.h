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

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/non_matching/mapping_info.h>

#include <iosfwd>
#include <memory>
#include <vector>

#include "galerkin_panel_pair_integrator.h"
#include "laplace_fmm.h"

/**
 * The Galerkin single-layer operator of the Laplace kernel, applied without
 * forming its matrix: the operator that the dense path assembles in
 * GalerkinPanelPairIntegrator::assemble_single_layer(), up to the truncation
 * of the fast multipole method.
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
 * pairs that the integrator treats specially enter through a sparse
 * correction: their exact block minus the block of the point rule, which is
 * what the multipole method has included for them. The coincident points
 * of a cell with itself are left out of both.
 *
 * The density is evaluated at the quadrature points, and the potential
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
 * method needs all charges, and tests the potential on its own cells.
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
    LaplaceFMM::AdditionalData fmm;
  };

  explicit GalerkinFMMOperator(const MPI_Comm comm);

  /**
   * Compute the geometry of the quadrature points, build the multipole
   * tree, find the pairs of cells that the integrator treats specially and
   * integrate their corrections. The test cells of this process are those
   * of its subdomain, as in the dense assembly.
   */
  void
  reinit(const DoFHandler<2, 3>               &dof_handler,
         const Mapping<2, 3>                  &mapping,
         const AffineConstraints<double>      &constraints,
         const GalerkinPanelPairIntegrator<3> &integrator,
         const IndexSet                       &locally_owned_dofs,
         const AdditionalData                 &data);

  /// dst = V src, with the constrained entries of src ignored.
  void
  vmult(TrilinosWrappers::MPI::Vector       &dst,
        const TrilinosWrappers::MPI::Vector &src) const;

  /// The exact single-layer blocks of the pairs integrated specially,
  /// condensed like the operator, for preconditioners.
  const TrilinosWrappers::SparseMatrix &
  near_field_matrix() const
  {
    return near_exact;
  }

  /// Sizes, times of the setup and mean times of vmult(); collective.
  void
  print_statistics(std::ostream &out) const;

private:
  MPI_Comm     mpi_communicator;
  unsigned int this_mpi_process;

  ObserverPointer<const DoFHandler<2, 3>> dof_handler;
  /// A copy of the hanging-node constraints of the caller.
  AffineConstraints<double> constraints;
  unsigned int              n_q_points;

  std::vector<DoFHandler<2, 3>::active_cell_iterator> cells;
  /// Cells whose test integrals this process computes.
  std::vector<unsigned int> own_cells;

  std::unique_ptr<NonMatching::MappingInfo<2, 3>>   mapping_info;
  mutable std::unique_ptr<FEPointEvaluation<1, 2, 3>> evaluator;
  /// Quadrature weights times Jacobians, cell by cell.
  std::vector<double> JxW;

  LaplaceFMM fmm;

  TrilinosWrappers::SparseMatrix near_exact, near_correction;

  unsigned long n_near_pairs = 0;
  double        setup_geometry = 0, setup_tree = 0, setup_near = 0;
  mutable double       time_evaluate = 0, time_fmm = 0, time_test = 0,
                 time_near = 0;
  mutable unsigned int n_vmult = 0;
};

#endif
