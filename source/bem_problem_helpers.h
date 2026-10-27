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

// What the source files of BEMProblem (bem_problem*.cc) share: the timers,
// defined in bem_problem.cc, and the creation of the finite elements.

#ifndef bem_problem_helpers_h
#define bem_problem_helpers_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <memory>
#include <string>

#include "Teuchos_TimeMonitor.hpp"

extern Teuchos::RCP<Teuchos::Time> AssembleTime, SurfaceGradientTime,
  GradientTime;

namespace bem_problem_internal
{
  using namespace dealii;

  template <int dim, int spacedim, bool system = false>
  std::unique_ptr<FiniteElement<dim, spacedim>>
  createFiniteElementPointer(const std::string &type, const unsigned int order)
  {
    if (type == "FEQ")
      {
        if (system)
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FESystem<dim, spacedim>(FE_Q<dim, spacedim>(order),
                                          spacedim));
          }
        else
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FE_Q<dim, spacedim>(order));
          }
      }
    else if (type == "FEDGQ")
      {
        if (system)
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FESystem<dim, spacedim>(FE_DGQ<dim, spacedim>(order),
                                          spacedim));
          }
        else
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FE_DGQ<dim, spacedim>(order));
          }
      }
    else if (type == "FEDGQGauss")
      {
        const FE_DGQArbitraryNodes<dim, spacedim> scalar_element(
          QGauss<1>(order + 1));
        if (system)
          return std::unique_ptr<FiniteElement<dim, spacedim>>(
            new FESystem<dim, spacedim>(scalar_element, spacedim));
        return std::unique_ptr<FiniteElement<dim, spacedim>>(
          new FE_DGQArbitraryNodes<dim, spacedim>(QGauss<1>(order + 1)));
      }

    AssertThrow(false, ExcMessage("Unknown finite element type: " + type));
    return nullptr;
  }
} // namespace bem_problem_internal

#endif
