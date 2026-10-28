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

#ifndef collocation_fmm_operator_h
#define collocation_fmm_operator_h

#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <functional>
#include <iosfwd>
#include <vector>

#include "laplace_fmm.h"

using namespace dealii;

/**
 * The matrices of the collocation method in 3D, of the Laplace or of the
 * screened kernel, applied without forming them: the double layer N phi
 * and the single layer D dphi_dn at the support points of the degrees of
 * freedom, the products of the matrices of BEMProblem::assemble_system()
 * up to the truncation of the fast multipole method.
 *
 * On a cell far from a support point the dense assembly takes the regular
 * rule of the cell, so that its sum over these cells is a sum over the
 * quadrature points of all cells, which LaplaceFMM evaluates with the
 * support points as its only targets: the dipoles w_q phi(y_q) n_q / 4 pi
 * give N phi, whose kernel (x - y) . n_y / (4 pi |x - y|^3) is that of the
 * dense assembly, and the charges w_q dphi_dn(y_q) / 4 pi give D dphi_dn.
 * The collocation takes only N phi - D dphi_dn, so that the dipoles and the
 * charges, negated, share one evaluation.
 * On the cells where the dense assembly takes the singular or the
 * quasi-singular rule, its rows enter as sparse corrections: the rows of
 * that rule minus those of the regular rule, which the multipole method
 * has included for them.
 *
 * The near band holds, for every support point, the rows on all cells
 * within the reach of the quasi-singular rule, those of the rule the dense
 * assembly takes; the preconditioner of the collocation system is built on
 * it.
 *
 * Every process evaluates the sources on all cells, as the multipole
 * method needs, and holds the corrections and the band of its own rows.
 */
class CollocationFMMOperator
{
public:
  using CellIterator = DoFHandler<2, 3>::active_cell_iterator;

  /// The rows of the dense assembly at the support point of i on a cell,
  /// for the cells within the reach of the quasi-singular rule: true, with
  /// the rows, if they are those of the singular or of the quasi-singular
  /// rule; false if they are those of the regular rule, which it need not
  /// fill.
  using RowIntegrator =
    std::function<bool(const CellIterator                         &cell,
                       const std::vector<types::global_dof_index> &dofs,
                       const types::global_dof_index               i,
                       Vector<double>                             &neumann_row,
                       Vector<double> &dirichlet_row)>;

  explicit CollocationFMMOperator(const MPI_Comm comm);

  /**
   * Compute the geometry at the points of the regular rule of every cell,
   * build the multipole tree of these points and of the support points,
   * and integrate the corrections and the band of the rows of this
   * process: the cells within near_ratio + 1 diameters of a support point
   * from the centre of the cell are offered to rows. Collective.
   */
  void
  reinit(const DoFHandler<2, 3>           &dof_handler,
         const Mapping<2, 3>              &mapping,
         const Quadrature<2>              &quadrature,
         const std::vector<Point<3>>      &support_points,
         const IndexSet                   &locally_owned_dofs,
         const double                      near_ratio,
         const RowIntegrator              &rows,
         const LaplaceFMM::AdditionalData &fmm_data);

  /// dst = N phi - D dphi_dn on the own rows, the combination that the
  /// collocation system and its right-hand side take, in one evaluation of
  /// the charges and the dipoles together; collective.
  void
  vmult(TrilinosWrappers::MPI::Vector       &dst,
        const TrilinosWrappers::MPI::Vector &phi,
        const TrilinosWrappers::MPI::Vector &dphi_dn) const;

  /// The same for up to four pairs at once, dst[s] = N phi[s] - D
  /// dphi_dn[s], such as the real and the imaginary part of a complex
  /// problem: the direct interactions and the sums over the processes are
  /// shared.
  void
  vmult(
    const std::vector<TrilinosWrappers::MPI::Vector *>       &dst,
    const std::vector<const TrilinosWrappers::MPI::Vector *> &phi,
    const std::vector<const TrilinosWrappers::MPI::Vector *> &dphi_dn) const;

  /**
   * The Helmholtz kernel (LaplaceFMM::AdditionalData::wavenumber): dst +
   * i dst_imag = N (phi + i phi_imag) - D (dphi_dn + i dphi_dn_imag), with
   * the complex matrices of the kernel exp(i k r) / (4 pi r). The near
   * corrections, and the band, are those of its real part, singular: the
   * dense assembly integrates the imaginary part, analytic, with the
   * regular rule on every cell, as the point sums do. Collective.
   */
  void
  vmult_complex(TrilinosWrappers::MPI::Vector       &dst,
                TrilinosWrappers::MPI::Vector       &dst_imag,
                const TrilinosWrappers::MPI::Vector &phi,
                const TrilinosWrappers::MPI::Vector &phi_imag,
                const TrilinosWrappers::MPI::Vector &dphi_dn,
                const TrilinosWrappers::MPI::Vector &dphi_dn_imag) const;

  /// The rows of N and of D on the near band.
  const TrilinosWrappers::SparseMatrix &
  band_neumann() const
  {
    return band_n;
  }

  const TrilinosWrappers::SparseMatrix &
  band_dirichlet() const
  {
    return band_d;
  }

  /// Sizes, times of the setup and mean time of a product; collective.
  void
  print_statistics(std::ostream &out) const;

private:
  MPI_Comm mpi_communicator;

  types::global_dof_index n_dofs     = 0;
  unsigned int            n_q_points = 0;
  unsigned int            n_sources  = 0;
  /// The wavenumber of the Helmholtz kernel, zero for the others.
  double wavenumber = 0.;

  /// The degrees of freedom of every cell.
  std::vector<std::vector<types::global_dof_index>> cell_dofs;
  /// Weights and normals at the quadrature points, cell by cell.
  std::vector<double>       JxW;
  std::vector<Tensor<1, 3>> normals;
  /// Shape values at the reference quadrature points, shape(q, j).
  FullMatrix<double> shape;

  /// The quadrature points, then the support points as the targets.
  LaplaceFMM fmm;

  /// The corrections of the near pairs of the own rows, and the near band.
  TrilinosWrappers::SparseMatrix correction_n, correction_d, band_n, band_d;

  unsigned long        n_band_pairs = 0, n_corrected_pairs = 0;
  double               setup_geometry = 0, setup_tree = 0, setup_near = 0;
  mutable double       time_products = 0;
  mutable unsigned int n_products    = 0;
};

#endif
