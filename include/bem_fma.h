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
 * of the parameter file: the truncation order, the opening criterion and
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

  /// Truncation order of the expansions.
  unsigned int
  truncation_order() const
  {
    return trunc_order;
  }

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
