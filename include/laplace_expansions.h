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

#ifndef laplace_expansions_h
#define laplace_expansions_h

#include <deal.II/base/tensor.h>

#include <complex>
#include <vector>

/**
 * Multipole and local expansions of the Laplace kernel 1/|x - y| in solid
 * harmonics, the building blocks of a fast multipole method.
 *
 * With the regular and irregular solid harmonics
 * @f[
 *   R_n^m(r) = (-1)^m \frac{|r|^n P_n^m(\cos\theta) e^{im\varphi}}{(n+m)!},
 *   \qquad
 *   I_n^m(r) = (-1)^m (n-m)! \frac{P_n^m(\cos\theta) e^{im\varphi}}
 *              {|r|^{n+1}},
 * @f]
 * where $P_n^m$ carries the Condon-Shortley phase, both families follow
 * from recurrences in the Cartesian components of $r$, without
 * trigonometric functions or powers, and satisfy
 * $R_n^{-m} = (-1)^m \overline{R_n^m}$, $I_n^{-m} = (-1)^m \overline{I_n^m}$.
 * The kernel and the translations follow from the addition theorems
 * @f[
 *   \frac{1}{|x-y|} = \sum_{n,m} \overline{R_n^m(y)}\, I_n^m(x), \quad
 *   R_n^m(x+y) = \sum_{k \le n} \sum_l R_k^l(y)\, R_{n-k}^{m-l}(x), \quad
 *   I_n^m(x-y) = \sum_{k,l} \overline{R_k^l(y)}\, I_{n+k}^{m+l}(x),
 * @f]
 * the first and last for $|y| < |x|$. With them, for sources $y_i$ of
 * strength $q_i$,
 *  - multipole expansion about $z$: $M_n^m = \sum_i q_i
 *    \overline{R_n^m(y_i - z)}$, and $\psi(x) = \sum M_n^m I_n^m(x - z)$;
 *  - local expansion about $z$: $\psi(x) = \sum L_n^m
 *    \overline{R_n^m(x - z)}$;
 *  - M2M: $M'^m_n = \sum_{k,l} \overline{R_k^l(z - z')}\, M_{n-k}^{m-l}$;
 *  - M2L: $L_k^l = (-1)^k \sum_{n,m} M_n^m I_{n+k}^{m+l}(z_L - z_M)$;
 *  - L2L: $L'^i_j = \sum_{k \ge j, l} L_k^l
 *    \overline{R_{k-j}^{l-i}(z' - z)}$.
 * A dipole of moment $d$ at $y$, the kernel $d \cdot \nabla_y 1/|x-y|$,
 * enters through the first-order term of the addition theorem of $R$:
 * $d \cdot \nabla R_n^m(y) = \sum_{l=-1}^{1} R_1^l(d)\, R_{n-1}^{m-l}(y)$.
 *
 * Coefficients are stored for $0 \le m \le n \le p$ at index
 * $n(n+1)/2 + m$; those with $m < 0$ follow from the symmetry, which the
 * expansions share with the harmonics because the potential is real.
 * Every translation costs $O(p^4)$ multiply-adds after $O(p^2)$ work on
 * the harmonics of the translation vector.
 */
namespace LaplaceExpansions
{
  using complex = std::complex<double>;

  /// Number of stored coefficients of an expansion of order p.
  inline unsigned int
  n_coefficients(const unsigned int p)
  {
    return (p + 1) * (p + 2) / 2;
  }

  /// Index of the coefficient (n, m), 0 <= m <= n.
  inline unsigned int
  index(const unsigned int n, const unsigned int m)
  {
    return n * (n + 1) / 2 + m;
  }

  /// Coefficient (n, m) for any |m| <= n, from the stored ones.
  inline complex
  get(const complex *c, const unsigned int n, const int m)
  {
    if (m >= 0)
      return c[index(n, m)];
    const complex v = std::conj(c[index(n, -m)]);
    return (m & 1) ? -v : v;
  }

  /**
   * Regular solid harmonics R_n^m(r), 0 <= m <= n <= p, into
   * out[index(n, m)].
   */
  inline void
  regular_harmonics(const unsigned int          p,
                    const dealii::Tensor<1, 3> &r,
                    complex                    *out)
  {
    const double  r2 = r * r;
    const complex w(r[0], r[1]);
    complex       diagonal(1., 0.);
    for (unsigned int m = 0; m <= p; ++m)
      {
        if (m > 0)
          diagonal *= w / (2. * m);
        out[index(m, m)] = diagonal;
        if (m + 1 <= p)
          out[index(m + 1, m)] = r[2] * diagonal;
        for (unsigned int n = m + 2; n <= p; ++n)
          out[index(n, m)] =
            ((2. * n - 1.) * r[2] * out[index(n - 1, m)] -
             r2 * out[index(n - 2, m)]) /
            double((n - m) * (n + m));
      }
  }

  /**
   * Irregular solid harmonics I_n^m(r), 0 <= m <= n <= p, into
   * out[index(n, m)].
   */
  inline void
  irregular_harmonics(const unsigned int          p,
                      const dealii::Tensor<1, 3> &r,
                      complex                    *out)
  {
    const double  inv_r2 = 1. / (r * r);
    const complex w(r[0] * inv_r2, r[1] * inv_r2);
    const double  z = r[2] * inv_r2;
    complex       diagonal(std::sqrt(inv_r2), 0.);
    for (unsigned int m = 0; m <= p; ++m)
      {
        if (m > 0)
          diagonal *= (2. * m - 1.) * w;
        out[index(m, m)] = diagonal;
        if (m + 1 <= p)
          out[index(m + 1, m)] = (2. * m + 1.) * z * diagonal;
        for (unsigned int n = m + 2; n <= p; ++n)
          out[index(n, m)] =
            (2. * n - 1.) * z * out[index(n - 1, m)] -
            double((n - 1) * (n - 1) - m * m) * inv_r2 * out[index(n - 2, m)];
      }
  }

  /**
   * Work space for the expansions of one order, so that a thread can
   * translate many expansions without allocating.
   */
  class Workspace
  {
  public:
    explicit Workspace(const unsigned int p)
      : p(p)
      , harmonics(n_coefficients(2 * p))
      , full_harmonics((2 * p + 1) * (2 * p + 1))
      , full_expansion((p + 1) * (p + 1))
    {}

    const unsigned int   p;
    std::vector<complex> harmonics;
    /// Harmonics for all -n <= m <= n, at n * n + n + m.
    std::vector<complex> full_harmonics;
    /// Expansion coefficients for all -n <= m <= n, at n * n + n + m.
    std::vector<complex> full_expansion;
  };

  /// Unfold the stored coefficients of order p to all -n <= m <= n.
  inline void
  unfold(const unsigned int p, const complex *c, complex *full)
  {
    for (unsigned int n = 0; n <= p; ++n)
      for (int m = -int(n); m <= int(n); ++m)
        full[n * n + n + m] = get(c, n, m);
  }

  /// P2M: add a charge q at the offset y - z to the multipole M.
  inline void
  add_charge(Workspace                  &w,
             const double                q,
             const dealii::Tensor<1, 3> &y_minus_z,
             complex                    *M)
  {
    regular_harmonics(w.p, y_minus_z, w.harmonics.data());
    for (unsigned int i = 0; i < n_coefficients(w.p); ++i)
      M[i] += q * std::conj(w.harmonics[i]);
  }

  /**
   * P2M: add a dipole of moment d, the kernel d . grad_y 1/|x - y|, at the
   * offset y - z to the multipole M.
   */
  inline void
  add_dipole(Workspace                  &w,
             const dealii::Tensor<1, 3> &d,
             const dealii::Tensor<1, 3> &y_minus_z,
             complex                    *M)
  {
    if (w.p == 0)
      return;
    regular_harmonics(w.p - 1, y_minus_z, w.harmonics.data());
    // R_1^0(d) = d_z, R_1^1(d) = (d_x + i d_y) / 2, R_1^{-1} = -conj(R_1^1)
    const complex r10(d[2], 0.);
    const complex r11(0.5 * d[0], 0.5 * d[1]);
    const complex r1m1 = -std::conj(r11);
    const complex *R   = w.harmonics.data();
    for (unsigned int n = 1; n <= w.p; ++n)
      for (int m = 0; m <= int(n); ++m)
        {
          complex g = r10 * (std::abs(m) <= int(n) - 1 ? get(R, n - 1, m) : 0.);
          if (std::abs(m - 1) <= int(n) - 1)
            g += r11 * get(R, n - 1, m - 1);
          if (std::abs(m + 1) <= int(n) - 1)
            g += r1m1 * get(R, n - 1, m + 1);
          M[index(n, m)] += std::conj(g);
        }
  }

  /// M2P: the potential of the multipole M at the offset x - z.
  inline double
  evaluate_multipole(Workspace                  &w,
                     const complex              *M,
                     const dealii::Tensor<1, 3> &x_minus_z)
  {
    irregular_harmonics(w.p, x_minus_z, w.harmonics.data());
    double psi = 0;
    for (unsigned int n = 0; n <= w.p; ++n)
      {
        psi += (M[index(n, 0)] * w.harmonics[index(n, 0)]).real();
        for (unsigned int m = 1; m <= n; ++m)
          psi += 2. * (M[index(n, m)] * w.harmonics[index(n, m)]).real();
      }
    return psi;
  }

  /// L2P: the potential of the local expansion L at the offset x - z.
  inline double
  evaluate_local(Workspace                  &w,
                 const complex              *L,
                 const dealii::Tensor<1, 3> &x_minus_z)
  {
    regular_harmonics(w.p, x_minus_z, w.harmonics.data());
    double psi = 0;
    for (unsigned int n = 0; n <= w.p; ++n)
      {
        psi += (L[index(n, 0)] * std::conj(w.harmonics[index(n, 0)])).real();
        for (unsigned int m = 1; m <= n; ++m)
          psi += 2. *
                 (L[index(n, m)] * std::conj(w.harmonics[index(n, m)])).real();
      }
    return psi;
  }

  /**
   * M2M: add the multipole M about z, translated to z', to M_out; the
   * argument is z - z'.
   */
  inline void
  multipole_to_multipole(Workspace                  &w,
                         const complex              *M,
                         const dealii::Tensor<1, 3> &z_minus_z_new,
                         complex                    *M_out)
  {
    const unsigned int p = w.p;
    regular_harmonics(p, z_minus_z_new, w.harmonics.data());
    unfold(p, w.harmonics.data(), w.full_harmonics.data());
    unfold(p, M, w.full_expansion.data());
    const complex *R  = w.full_harmonics.data();
    const complex *Mf = w.full_expansion.data();
    for (unsigned int n = 0; n <= p; ++n)
      for (int m = 0; m <= int(n); ++m)
        {
          complex sum = 0.;
          for (unsigned int k = 0; k <= n; ++k)
            {
              const int nk = n - k;
              // |l| <= k and |m - l| <= n - k
              const int l_min = std::max(-int(k), m - nk);
              const int l_max = std::min(int(k), m + nk);
              for (int l = l_min; l <= l_max; ++l)
                sum += std::conj(R[k * k + k + l]) * Mf[nk * nk + nk + m - l];
            }
          M_out[index(n, m)] += sum;
        }
  }

  /**
   * M2L: add the local expansion about z_L of the multipole M about z_M to
   * L; the argument is z_L - z_M.
   */
  inline void
  multipole_to_local(Workspace                  &w,
                     const complex              *M,
                     const dealii::Tensor<1, 3> &zL_minus_zM,
                     complex                    *L)
  {
    const unsigned int p = w.p;
    irregular_harmonics(2 * p, zL_minus_zM, w.harmonics.data());
    unfold(2 * p, w.harmonics.data(), w.full_harmonics.data());
    unfold(p, M, w.full_expansion.data());
    const complex *I  = w.full_harmonics.data();
    const complex *Mf = w.full_expansion.data();
    for (unsigned int k = 0; k <= p; ++k)
      for (int l = 0; l <= int(k); ++l)
        {
          complex sum = 0.;
          for (unsigned int n = 0; n <= p; ++n)
            {
              const unsigned int nk  = n + k;
              const complex     *Ink = I + nk * nk + nk + l;
              const complex     *Mn  = Mf + n * n + n;
              for (int m = -int(n); m <= int(n); ++m)
                sum += Mn[m] * Ink[m];
            }
          L[index(k, l)] += (k & 1) ? -sum : sum;
        }
  }

  /**
   * L2L: add the local expansion L about z, translated to z', to L_out; the
   * argument is z' - z.
   */
  inline void
  local_to_local(Workspace                  &w,
                 const complex              *L,
                 const dealii::Tensor<1, 3> &z_new_minus_z,
                 complex                    *L_out)
  {
    const unsigned int p = w.p;
    regular_harmonics(p, z_new_minus_z, w.harmonics.data());
    unfold(p, w.harmonics.data(), w.full_harmonics.data());
    unfold(p, L, w.full_expansion.data());
    const complex *R  = w.full_harmonics.data();
    const complex *Lf = w.full_expansion.data();
    for (unsigned int j = 0; j <= p; ++j)
      for (int i = 0; i <= int(j); ++i)
        {
          complex sum = 0.;
          for (unsigned int k = j; k <= p; ++k)
            {
              const int kj = k - j;
              // |l| <= k and |l - i| <= k - j
              const int l_min = std::max(-int(k), i - kj);
              const int l_max = std::min(int(k), i + kj);
              for (int l = l_min; l <= l_max; ++l)
                sum +=
                  Lf[k * k + k + l] * std::conj(R[kj * kj + kj + l - i]);
            }
          L_out[index(j, i)] += sum;
        }
  }
} // namespace LaplaceExpansions

#endif
