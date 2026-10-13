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

#ifndef inexact_gmres_h
#define inexact_gmres_h

#include <deal.II/lac/vector.h>

#include <cmath>
#include <limits>
#include <vector>

/**
 * GMRES without restarts whose products may be inexact (Simoncini and
 * Szyld 2003, Bouras and Fraysse 2005): the product of each step may err by
 * about the tolerance over the residual reached, so that the caller can
 * choose cheaper products once the residual is small. The system is left
 * preconditioned, P A x = P b, and the operator receives the norm of the
 * preconditioned residual reached.
 */
namespace InexactGMRES
{
  using dealii::Vector;

  /**
   * Solve P A x = P b from the given x until the preconditioned residual
   * falls below the tolerance; apply(y, v, residual) computes y = P A v,
   * residual being the norm of the preconditioned residual reached
   * (infinite for the product of the start). Returns the number of steps.
   */
  template <typename Apply>
  unsigned int
  solve(const Apply          &apply,
        const Vector<double> &preconditioned_rhs,
        Vector<double>       &x,
        const double          tolerance,
        const unsigned int    max_iterations)
  {
    const unsigned int n = x.size();
    Vector<double>     w(n);
    // r = P b - P A x
    apply(w, x, std::numeric_limits<double>::infinity());
    Vector<double> r(preconditioned_rhs);
    r -= w;
    const double beta = r.l2_norm();
    if (beta <= tolerance)
      return 0;
    std::vector<Vector<double>> V;
    V.emplace_back(r);
    V.back() /= beta;
    // The columns of the Hessenberg matrix and of its rotated form, grown
    // step by step: column m has m + 2 entries.
    std::vector<std::vector<double>> H, R;
    std::vector<double>              cs, sn, g(1, beta);
    unsigned int                     m = 0;
    for (; m < max_iterations;)
      {
        // w = P A v_m, orthogonalized against the basis twice.
        H.emplace_back(m + 2, 0.);
        R.emplace_back(m + 2, 0.);
        cs.push_back(0.);
        sn.push_back(0.);
        g.push_back(0.);
        apply(w, V[m], std::abs(g[m]));
        for (unsigned int pass = 0; pass < 2; ++pass)
          for (unsigned int i = 0; i <= m; ++i)
            {
              const double h = V[i] * w;
              H[m][i] += h;
              w.add(-h, V[i]);
            }
        const double h = w.l2_norm();
        H[m][m + 1]    = h;
        // The least-squares problem by Givens rotations.
        R[m] = H[m];
        for (unsigned int i = 0; i < m; ++i)
          {
            const double a = R[m][i], b = R[m][i + 1];
            R[m][i]        = cs[i] * a + sn[i] * b;
            R[m][i + 1]    = -sn[i] * a + cs[i] * b;
          }
        const double a = R[m][m], b = R[m][m + 1];
        const double norm = std::hypot(a, b);
        cs[m]             = a / norm;
        sn[m]             = b / norm;
        R[m][m]           = norm;
        R[m][m + 1]       = 0.;
        g[m + 1]          = -sn[m] * g[m];
        g[m]              = cs[m] * g[m];
        ++m;
        if (std::abs(g[m]) <= tolerance || h == 0.)
          break;
        V.emplace_back(w);
        V.back() /= h;
      }
    // y from the triangular system, x += V y.
    std::vector<double> y(m);
    for (int i = m - 1; i >= 0; --i)
      {
        double sum = g[i];
        for (unsigned int j = i + 1; j < m; ++j)
          sum -= R[j][i] * y[j];
        y[i] = sum / R[i][i];
      }
    for (unsigned int i = 0; i < m; ++i)
      x.add(y[i], V[i]);
    return m;
  }
} // namespace InexactGMRES

#endif
