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

#ifndef tensor_map_kernel_h
#define tensor_map_kernel_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <algorithm>

/**
 * The evaluation of the tensor Lagrange map of a cell, its position and its
 * two tangents, at many reference points, shared by the direct path of the
 * Galerkin integrator and by the evaluation of potentials on the surface.
 */
namespace TensorMapKernel
{
  /**
   * The map of a cell with n nodes per direction, given by the nodes in
   * structure-of-arrays form (three arrays of n * n values, node i + n j
   * at reference point (x_i, x_j)) and the one-dimensional basis in product
   * form (the nodes x_k followed by the weights
   * w_i = 1 / prod_{k != i} (x_i - x_k)), at m reference points.
   *
   * The points go in blocks of four, one vector register of doubles, and
   * with n known at compile time all loops over the nodes unroll and the
   * accumulators of a block stay in registers: the basis and its derivative
   * in each direction from prefix and suffix products,
   * L_i = w_i P_i S_i, which is accurate to rounding at any degree, then the
   * sums over the nodes, factorized, first over the nodes i of each line j,
   * then over the lines. A partial last block repeats its last point. For
   * each block, output(begin, count, s) receives s[0..2] the position,
   * s[3..5] and s[6..8] the derivatives in the two reference directions,
   * each for the four points of the block.
   */
  template <unsigned int n, typename Output>
  void
  evaluate(const double             *nodes,
           const double             *basis,
           const dealii::Point<2>   *reference_points,
           const unsigned int        m,
           Output                  &&output)
  {
    constexpr unsigned int block = 4;
    for (unsigned int begin = 0; begin < m; begin += block)
      {
        const unsigned int count = std::min(block, m - begin);
        alignas(32) double t[block], u[block];
        for (unsigned int q = 0; q < block; ++q)
          {
            const dealii::Point<2> &p =
              reference_points[begin + std::min(q, count - 1)];
            t[q] = p[0];
            u[q] = p[1];
          }
        alignas(32) double v0[n][block], d0[n][block], v1[n][block],
          d1[n][block];
        const double *x = basis, *w = basis + n;
        for (unsigned int direction = 0; direction < 2; ++direction)
          {
            const double *r = direction == 0 ? t : u;
            double(&v)[n][block] = direction == 0 ? v0 : v1;
            double(&d)[n][block] = direction == 0 ? d0 : d1;
            alignas(32) double P[n][block], dP[n][block], S[n][block],
              dS[n][block];
#pragma omp simd
            for (unsigned int q = 0; q < block; ++q)
              {
                P[0][q]      = 1.0;
                dP[0][q]     = 0.0;
                S[n - 1][q]  = 1.0;
                dS[n - 1][q] = 0.0;
              }
#pragma GCC unroll 16
            for (unsigned int k = 0; k + 1 < n; ++k)
#pragma omp simd
              for (unsigned int q = 0; q < block; ++q)
                {
                  const double f = r[q] - x[k];
                  dP[k + 1][q]   = dP[k][q] * f + P[k][q];
                  P[k + 1][q]    = P[k][q] * f;
                }
#pragma GCC unroll 16
            for (unsigned int k = n - 1; k > 0; --k)
#pragma omp simd
              for (unsigned int q = 0; q < block; ++q)
                {
                  const double f = r[q] - x[k];
                  dS[k - 1][q]   = dS[k][q] * f + S[k][q];
                  S[k - 1][q]    = S[k][q] * f;
                }
#pragma GCC unroll 16
            for (unsigned int i = 0; i < n; ++i)
#pragma omp simd
              for (unsigned int q = 0; q < block; ++q)
                {
                  v[i][q] = w[i] * P[i][q] * S[i][q];
                  d[i][q] = w[i] * (dP[i][q] * S[i][q] + P[i][q] * dS[i][q]);
                }
          }
        alignas(32) double s[9][block] = {};
        // The loops over the nodes are unrolled in full whatever n, so that
        // the accumulators stay in registers.
#pragma GCC unroll 16
        for (unsigned int j = 0; j < n; ++j)
          {
            alignas(32) double line[6][block] = {};
#pragma GCC unroll 16
            for (unsigned int i = 0; i < n; ++i)
#pragma GCC unroll 3
              for (unsigned int c = 0; c < 3; ++c)
                {
                  const double value = nodes[c * n * n + i + n * j];
#pragma omp simd
                  for (unsigned int q = 0; q < block; ++q)
                    {
                      line[c][q] += value * v0[i][q];
                      line[3 + c][q] += value * d0[i][q];
                    }
                }
            for (unsigned int c = 0; c < 3; ++c)
#pragma omp simd
              for (unsigned int q = 0; q < block; ++q)
                {
                  s[c][q] += v1[j][q] * line[c][q];
                  s[3 + c][q] += v1[j][q] * line[3 + c][q];
                  s[6 + c][q] += d1[j][q] * line[c][q];
                }
          }
        output(begin, count, s);
      }
  }


  /// The same for n known at run time, from 2 to 12.
  template <typename Output>
  void
  evaluate(const unsigned int        n,
           const double             *nodes,
           const double             *basis,
           const dealii::Point<2>   *reference_points,
           const unsigned int        m,
           Output                  &&output)
  {
    switch (n)
      {
#define PIBEM_TENSOR_MAP_CASE(N)                                \
  case N:                                                      \
    evaluate<N>(nodes, basis, reference_points, m, output);    \
    break;
        PIBEM_TENSOR_MAP_CASE(2)
        PIBEM_TENSOR_MAP_CASE(3)
        PIBEM_TENSOR_MAP_CASE(4)
        PIBEM_TENSOR_MAP_CASE(5)
        PIBEM_TENSOR_MAP_CASE(6)
        PIBEM_TENSOR_MAP_CASE(7)
        PIBEM_TENSOR_MAP_CASE(8)
        PIBEM_TENSOR_MAP_CASE(9)
        PIBEM_TENSOR_MAP_CASE(10)
        PIBEM_TENSOR_MAP_CASE(11)
        PIBEM_TENSOR_MAP_CASE(12)
#undef PIBEM_TENSOR_MAP_CASE
        default:
          AssertThrow(false,
                      dealii::ExcMessage("Unsupported number of nodes."));
      }
  }
} // namespace TensorMapKernel

#endif
