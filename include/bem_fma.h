// ---------------------------------------------------------------------
//
// Copyright (C) 2014 - 2026 by the pi-BEM authors.
//
// This file is part of the pi-BEM library.
//
// The pi-BEM is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License version 2.1 as published by the Free Software Foundation.
// The full text of the license can be found in the file LICENSE at
// the top level of the pi-BEM distribution.
//
// Authors: Nicola Giuliani, Andrea Mola, Luca Heltai
//
// ---------------------------------------------------------------------

#ifndef bem_fma_h
#define bem_fma_h

#include <deal.II/base/parameter_acceptor.h>

using namespace dealii;

/**
 * The parameters of the fast multipole method, in the section BEMFMA<dim>
 * of the parameter file: the truncation order, which by default grows with
 * the mesh (mesh_order()), the opening criterion and
 * the size of the leaves of LaplaceFMM, for the collocation
 * (CollocationFMMOperator) and the Galerkin formulation
 * (GalerkinFMMOperator, GalerkinCalderonFMMOperator), and the settings of
 * the mixed Galerkin problem.
 *
 * The section once held the octree of the original FMA, which LaplaceFMM
 * has replaced; its entries (Octree Params, Granularity for TBB simple for
 * cycles) are still accepted, so that the parameter files of upstream run,
 * and are ignored.
 */
template <int dim>
class BEMFMA : public ParameterAcceptor
{
public:
  BEMFMA() = default;

  virtual void
  declare_parameters(ParameterHandler &prm) override;

  virtual void
  parse_parameters(ParameterHandler &prm) override;

  /// Truncation order of the expansions in the parameter file.
  unsigned int
  truncation_order() const
  {
    return trunc_order;
  }

  /// Whether the order of the expansions grows with the mesh
  /// (truncation_order(diameter, cell_size, degree)).
  bool
  order_from_mesh() const
  {
    return trunc_order_from_mesh;
  }

  /**
   * The truncation order that keeps the error of the products of the
   * multipole method below the discretisation error of a smooth solution,
   * with elements of the given degree p on a surface of the given diameter
   * D whose cells have the given mean size h. The error of the products
   * decreases by about a digit every two orders, the discretisation error
   * as (h/D)^(p+1), so that the order grows as 2 (p+1) log10(D/h):
   *
   *   ceil(2 (p+1) log10(D/h) - 2),
   *
   * at most 24. The constant comes from the sphere of the campaign of the
   * report, on which degree 2 needed order 8 with 1536 cells, 10 with 6144
   * and 12 with 24576, for both formulations and the three kernels, and
   * degree 1 order 8 with 98304. On a singular solution, whose
   * discretisation error is larger, the order is higher than it needs to be.
   */
  static unsigned int
  mesh_order(const double       diameter,
             const double       cell_size,
             const unsigned int degree);

  /// The truncation order on a mesh: that of the parameter file, or, with
  /// FMA truncation order from the mesh, the larger of it and mesh_order().
  unsigned int
  truncation_order(const double       diameter,
                   const double       cell_size,
                   const unsigned int degree) const;

  /// Opening criterion of the octree.
  double
  opening_criterion() const
  {
    return galerkin_opening_criterion;
  }

  /// Largest number of points in a leaf of the octree.
  unsigned int
  points_per_leaf() const
  {
    return galerkin_points_per_leaf;
  }

  /// Increase of the truncation order for the dipole sums of the mixed
  /// Galerkin problem.
  unsigned int
  dipole_order_increase() const
  {
    return galerkin_dipole_order_increase;
  }

  /// Order and opening criterion of the preconditioner products of the
  /// mixed Galerkin problem, zero for those of the operator.
  unsigned int
  preconditioner_order() const
  {
    return galerkin_preconditioner_order;
  }

  double
  preconditioner_theta() const
  {
    return galerkin_preconditioner_theta;
  }

  /// Points per leaf of the preconditioner, and the expansions of the
  /// relaxed products, zero for the defaults.
  unsigned int
  preconditioner_points_per_leaf() const
  {
    return galerkin_preconditioner_points_per_leaf;
  }

  unsigned int
  relaxed_order() const
  {
    return galerkin_relaxed_order;
  }

  double
  relaxed_theta() const
  {
    return galerkin_relaxed_theta;
  }

private:
  unsigned int trunc_order                             = 6;
  bool         trunc_order_from_mesh                   = true;
  double       galerkin_opening_criterion              = 0.5;
  unsigned int galerkin_points_per_leaf                = 64;
  unsigned int galerkin_dipole_order_increase          = 2;
  unsigned int galerkin_preconditioner_order           = 0;
  double       galerkin_preconditioner_theta           = 0.;
  unsigned int galerkin_preconditioner_points_per_leaf = 0;
  unsigned int galerkin_relaxed_order                  = 0;
  double       galerkin_relaxed_theta                  = 0.;
};

#endif
