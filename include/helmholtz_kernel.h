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

#ifndef helmholtz_kernel_h
#define helmholtz_kernel_h

#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>

#include <cmath>

/**
 * The kernel of the Helmholtz equation Delta u + k^2 u = 0 in 3D, the
 * outgoing fundamental solution G = exp(i k r) / (4 pi r), r = |R|, and its
 * gradient with respect to the source point y, R = y - x:
 *
 *   grad_y G = -(1 - i k r) exp(i k r) R / (4 pi r^3),
 *
 * the screened kernel exp(-kappa r) / (4 pi r) with kappa = -i k. Its real
 * and imaginary parts, as the real arguments of the collocation: the real
 * part, cos(k r) / (4 pi r), is singular as the Laplace kernel; the
 * imaginary part, sin(k r) / (4 pi r), is analytic, its gradient
 * -(sin(k r) - k r cos(k r)) R / (4 pi r^3) bounded, so that the regular
 * rule integrates it on every cell.
 */
namespace HelmholtzKernel
{
  using namespace dealii;

  /// The real parts: d = cos(k r) / (4 pi r) and D its gradient in y.
  inline void
  real_kernels(const Tensor<1, 3> &R,
               Tensor<1, 3>       &D,
               double             &d,
               const double        k)
  {
    const double r  = R.norm();
    const double kr = k * r;
    const double c = std::cos(kr), s = std::sin(kr);
    d = c / (4 * numbers::PI * r);
    D = (c + kr * s) * R / (-4 * numbers::PI * r * r * r);
  }

  /// The imaginary parts: d = sin(k r) / (4 pi r), k / (4 pi) at r = 0,
  /// and D its gradient in y, zero at r = 0.
  inline void
  imaginary_kernels(const Tensor<1, 3> &R,
                    Tensor<1, 3>       &D,
                    double             &d,
                    const double        k)
  {
    const double r  = R.norm();
    const double kr = k * r;
    if (kr < 1e-3)
      {
        // sin(x) / x and (sin(x) - x cos(x)) / x^3 by their series, which
        // the direct formulas lose to cancellation.
        const double x2 = kr * kr;
        d               = k * (1. - x2 / 6.) / (4 * numbers::PI);
        D = k * k * k * (1. / 3. - x2 / 30.) * R / (-4 * numbers::PI);
        return;
      }
    const double c = std::cos(kr), s = std::sin(kr);
    d = s / (4 * numbers::PI * r);
    D = (s - kr * c) * R / (-4 * numbers::PI * r * r * r);
  }
} // namespace HelmholtzKernel

#endif
