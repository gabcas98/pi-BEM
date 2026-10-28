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

#include "../include/bem_fma.h"

template <int dim>
void
BEMFMA<dim>::declare_parameters(ParameterHandler &prm)
{
  // The octree of the original FMA, which LaplaceFMM has replaced: accepted
  // for the parameter files of upstream, and ignored.
  prm.enter_subsection("Octree Params");
  {
    prm.declare_entry("Number of Octree Levels",
                      "10",
                      Patterns::Integer(),
                      "Ignored: the octree of LaplaceFMM ends at Galerkin "
                      "maximum points per leaf.");

    prm.declare_entry(
      "Maximum Number of Collocation Points per Childless Block",
      "20",
      Patterns::Integer(),
      "Ignored: the octree of LaplaceFMM ends at Galerkin maximum points "
      "per leaf.");
  }
  prm.leave_subsection();

  prm.enter_subsection("FMA Params");
  {
    prm.declare_entry("FMA Truncation Order", "6", Patterns::Integer());
    prm.declare_entry(
      "Galerkin opening criterion",
      "0.5",
      Patterns::Double(0.0, 1.0),
      "Collocation and Galerkin FMM: two boxes of the octree interact "
      "through their expansions when the sum of their radii is below this "
      "fraction of the distance of their centres. Smaller is more accurate "
      "and slower.");
    prm.declare_entry(
      "Galerkin maximum points per leaf",
      "64",
      Patterns::Integer(1),
      "Collocation and Galerkin FMM: largest number of points in a leaf of "
      "the octree (quadrature points, and the support points of the "
      "collocation).");
    prm.declare_entry(
      "Galerkin dipole order increase",
      "2",
      Patterns::Integer(0),
      "Mixed Galerkin problem through the FMM only: the double layer and its "
      "transpose, sums of dipoles, use the truncation order plus this "
      "increase.");
    prm.declare_entry(
      "Galerkin preconditioner truncation order",
      "0",
      Patterns::Integer(0),
      "Mixed Galerkin problem through the FMM only: the order of the "
      "expansions in the products of the Calderon preconditioner, which need "
      "not be accurate; zero for the truncation order.");
    prm.declare_entry(
      "Galerkin preconditioner opening criterion",
      "0",
      Patterns::Double(0.0, 1.0),
      "Mixed Galerkin problem through the FMM only: the opening criterion "
      "of the products of the Calderon preconditioner; zero for that of the "
      "operator.");
    prm.declare_entry(
      "Galerkin preconditioner maximum points per leaf",
      "0",
      Patterns::Integer(0),
      "Mixed Galerkin problem through the FMM only: the largest number of "
      "points in a leaf of the octree of the preconditioner, whose low order "
      "favours small leaves; zero for that of the operator.");
    prm.declare_entry(
      "Galerkin relaxed truncation order",
      "0",
      Patterns::Integer(0),
      "Mixed Galerkin problem through the FMM only: the order of the "
      "expansions of the relaxed products of GMRES (Mixed Galerkin relaxed "
      "products below); zero for those of the preconditioner.");
    prm.declare_entry(
      "Galerkin relaxed opening criterion",
      "0",
      Patterns::Double(0.0, 1.0),
      "Mixed Galerkin problem through the FMM only: the opening criterion of "
      "the relaxed products; zero for that of the preconditioner.");
    prm.declare_entry("Granularity for TBB simple for cycles",
                      "10",
                      Patterns::Integer(),
                      "Ignored: a setting of the original FMA.");
  }
  prm.leave_subsection();
}

template <int dim>
void
BEMFMA<dim>::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("FMA Params");
  {
    trunc_order                = prm.get_integer("FMA Truncation Order");
    galerkin_opening_criterion = prm.get_double("Galerkin opening criterion");
    galerkin_points_per_leaf =
      prm.get_integer("Galerkin maximum points per leaf");
    galerkin_dipole_order_increase =
      prm.get_integer("Galerkin dipole order increase");
    galerkin_preconditioner_order =
      prm.get_integer("Galerkin preconditioner truncation order");
    galerkin_preconditioner_theta =
      prm.get_double("Galerkin preconditioner opening criterion");
    galerkin_preconditioner_points_per_leaf =
      prm.get_integer("Galerkin preconditioner maximum points per leaf");
    galerkin_relaxed_order =
      prm.get_integer("Galerkin relaxed truncation order");
    galerkin_relaxed_theta =
      prm.get_double("Galerkin relaxed opening criterion");
  }
  prm.leave_subsection();
}

template class BEMFMA<2>;
template class BEMFMA<3>;
