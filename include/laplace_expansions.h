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

#include <cmath>
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
 * the harmonics of the translation vector. LaplaceFMM applies the M2L
 * translation in $O(p^3)$ instead, by rotation in the real harmonics of
 * yukawa_expansions.h, through the factors of RealBasis.
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
          out[index(n, m)] = ((2. * n - 1.) * r[2] * out[index(n - 1, m)] -
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
      , harmonics_re((2 * p + 1) * (2 * p + 1))
      , harmonics_im((2 * p + 1) * (2 * p + 1))
      , expansion_re((p + 1) * (p + 1))
      , expansion_im((p + 1) * (p + 1))
    {}

    const unsigned int   p;
    std::vector<complex> harmonics;
    /// Real and imaginary parts of harmonics and expansion coefficients for
    /// all -n <= m <= n, at n * n + n + m: the translations run over them
    /// in real arithmetic, which vectorizes, where std::complex does not.
    std::vector<double> harmonics_re, harmonics_im;
    std::vector<double> expansion_re, expansion_im;
  };

  /// Unfold the stored coefficients of order p to all -n <= m <= n, split
  /// into real and imaginary parts, conjugated if asked.
  inline void
  unfold(const unsigned int p,
         const complex     *c,
         double            *re,
         double            *im,
         const bool         conjugate = false)
  {
    const double s = conjugate ? -1. : 1.;
    for (unsigned int n = 0; n <= p; ++n)
      {
        double *r = re + n * n + n, *i = im + n * n + n;
        for (unsigned int m = 0; m <= n; ++m)
          {
            const complex v = c[index(n, m)];
            r[m]            = v.real();
            i[m]            = s * v.imag();
            if (m == 0)
              continue;
            // c_n^{-m} = (-1)^m conj(c_n^m)
            const double sign = (m & 1) ? -1. : 1.;
            r[-int(m)]        = sign * v.real();
            i[-int(m)]        = -sign * s * v.imag();
          }
      }
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
    const complex  r10(d[2], 0.);
    const complex  r11(0.5 * d[0], 0.5 * d[1]);
    const complex  r1m1 = -std::conj(r11);
    const complex *R    = w.harmonics.data();
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
          psi +=
            2. * (L[index(n, m)] * std::conj(w.harmonics[index(n, m)])).real();
      }
    return psi;
  }

  /**
   * L2P: the gradient of the local expansion L at the offset x - z. With
   * the harmonics of this namespace, d/dz R_n^m = R_{n-1}^m and
   * (d/dx - i d/dy) R_n^m = R_{n-1}^{m-1}, so that for the real potential
   * phi = sum L_n^m conj(R_n^m),
   * d phi/dx + i d phi/dy = sum L_n^m conj(R_{n-1}^{m-1}) and
   * d phi/dz = sum L_n^m conj(R_{n-1}^m).
   */
  inline dealii::Tensor<1, 3>
  evaluate_local_gradient(Workspace                  &w,
                          const complex              *L,
                          const dealii::Tensor<1, 3> &x_minus_z)
  {
    dealii::Tensor<1, 3> gradient;
    if (w.p == 0)
      return gradient;
    regular_harmonics(w.p - 1, x_minus_z, w.harmonics.data());
    const complex *R = w.harmonics.data();
    complex        xy(0., 0.), z(0., 0.);
    for (unsigned int n = 1; n <= w.p; ++n)
      {
        const int k = n - 1;
        for (int m = -k; m <= k; ++m)
          z += get(L, n, m) * std::conj(get(R, k, m));
        for (int m = -k + 1; m <= int(n); ++m)
          xy += get(L, n, m) * std::conj(get(R, k, m - 1));
      }
    gradient[0] = xy.real();
    gradient[1] = xy.imag();
    gradient[2] = z.real();
    return gradient;
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
    // conj(R)
    unfold(p,
           w.harmonics.data(),
           w.harmonics_re.data(),
           w.harmonics_im.data(),
           true);
    unfold(p, M, w.expansion_re.data(), w.expansion_im.data());
    const double *Rr = w.harmonics_re.data(), *Ri = w.harmonics_im.data();
    const double *Mr = w.expansion_re.data(), *Mi = w.expansion_im.data();
    for (unsigned int n = 0; n <= p; ++n)
      for (int m = 0; m <= int(n); ++m)
        {
          double sr = 0, si = 0;
          for (unsigned int k = 0; k <= n; ++k)
            {
              const int nk = n - k;
              // |l| <= k and |m - l| <= n - k
              const int     l_min = std::max(-int(k), m - nk);
              const int     l_max = std::min(int(k), m + nk);
              const double *ar    = Rr + k * k + k;
              const double *ai    = Ri + k * k + k;
              const double *br    = Mr + nk * nk + nk + m;
              const double *bi    = Mi + nk * nk + nk + m;
#pragma omp simd reduction(+ : sr, si)
              for (int l = l_min; l <= l_max; ++l)
                {
                  sr += ar[l] * br[-l] - ai[l] * bi[-l];
                  si += ar[l] * bi[-l] + ai[l] * br[-l];
                }
            }
          M_out[index(n, m)] += complex(sr, si);
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
    unfold(2 * p,
           w.harmonics.data(),
           w.harmonics_re.data(),
           w.harmonics_im.data());
    unfold(p, M, w.expansion_re.data(), w.expansion_im.data());
    const double *Ir = w.harmonics_re.data(), *Ii = w.harmonics_im.data();
    const double *Mr = w.expansion_re.data(), *Mi = w.expansion_im.data();
    for (unsigned int k = 0; k <= p; ++k)
      for (int l = 0; l <= int(k); ++l)
        {
          double sr = 0, si = 0;
          for (unsigned int n = 0; n <= p; ++n)
            {
              const unsigned int nk = n + k;
              const double      *ar = Mr + n * n + n;
              const double      *ai = Mi + n * n + n;
              const double      *br = Ir + nk * nk + nk + l;
              const double      *bi = Ii + nk * nk + nk + l;
#pragma omp simd reduction(+ : sr, si)
              for (int m = -int(n); m <= int(n); ++m)
                {
                  sr += ar[m] * br[m] - ai[m] * bi[m];
                  si += ar[m] * bi[m] + ai[m] * br[m];
                }
            }
          const double sign = (k & 1) ? -1. : 1.;
          L[index(k, l)] += complex(sign * sr, sign * si);
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
    // conj(R)
    unfold(p,
           w.harmonics.data(),
           w.harmonics_re.data(),
           w.harmonics_im.data(),
           true);
    unfold(p, L, w.expansion_re.data(), w.expansion_im.data());
    const double *Rr = w.harmonics_re.data(), *Ri = w.harmonics_im.data();
    const double *Lr = w.expansion_re.data(), *Li = w.expansion_im.data();
    for (unsigned int j = 0; j <= p; ++j)
      for (int i = 0; i <= int(j); ++i)
        {
          double sr = 0, si = 0;
          for (unsigned int k = j; k <= p; ++k)
            {
              const int kj = k - j;
              // |l| <= k and |l - i| <= k - j
              const int     l_min = std::max(-int(k), i - kj);
              const int     l_max = std::min(int(k), i + kj);
              const double *ar    = Lr + k * k + k;
              const double *ai    = Li + k * k + k;
              const double *br    = Rr + kj * kj + kj - i;
              const double *bi    = Ri + kj * kj + kj - i;
#pragma omp simd reduction(+ : sr, si)
              for (int l = l_min; l <= l_max; ++l)
                {
                  sr += ar[l] * br[l] - ai[l] * bi[l];
                  si += ar[l] * bi[l] + ai[l] * br[l];
                }
            }
          L_out[index(j, i)] += complex(sr, si);
        }
  }
  /**
   * The M2L operator of the offset tau in the scaled form used for batches:
   * for boxes at distance u tau, with u a length, the local expansion is
   * L_k^l = u^{-(k+1)} sum_{n,m} T_{(k,l),(n,m)} u^{-n} M_n^m with
   * T_{(k,l),(n,m)} = (-1)^k I_{n+k}^{m+l}(tau), by the homogeneity
   * I_j(u tau) = u^{-(j+1)} I_j(tau). T depends on tau only, so all pairs
   * of boxes with the same tau, whatever their size, share it. Rows are
   * index(k, l), columns n * n + n + m, real and imaginary parts apart.
   */
  inline void
  multipole_to_local_operator(Workspace                  &w,
                              const dealii::Tensor<1, 3> &tau,
                              double                     *T_re,
                              double                     *T_im)
  {
    const unsigned int p       = w.p;
    const unsigned int columns = (p + 1) * (p + 1);
    irregular_harmonics(2 * p, tau, w.harmonics.data());
    unfold(2 * p,
           w.harmonics.data(),
           w.harmonics_re.data(),
           w.harmonics_im.data());
    for (unsigned int k = 0; k <= p; ++k)
      for (unsigned int l = 0; l <= k; ++l)
        {
          const double sign = (k & 1) ? -1. : 1.;
          double      *tr   = T_re + index(k, l) * columns;
          double      *ti   = T_im + index(k, l) * columns;
          for (unsigned int n = 0; n <= p; ++n)
            {
              const unsigned int nk = n + k;
              const double      *hr = w.harmonics_re.data() + nk * nk + nk + l;
              const double      *hi = w.harmonics_im.data() + nk * nk + nk + l;
              for (int m = -int(n); m <= int(n); ++m)
                {
                  tr[n * n + n + m] = sign * hr[m];
                  ti[n * n + n + m] = sign * hi[m];
                }
            }
        }
  }

  /**
   * Y = T X for a batch of B expansions: X holds the (p + 1)^2 unfolded
   * coefficients of each of them, coefficient by coefficient (X[c * B + j]
   * for the expansion j), and Y the n_coefficients(p) results the same
   * way. The loop over the batch runs innermost, where it vectorizes, and
   * the B partial sums of a row stay in registers.
   */
  template <unsigned int B>
  inline void
  multipole_to_local_batch(const unsigned int p,
                           const double      *T_re,
                           const double      *T_im,
                           const double      *X_re,
                           const double      *X_im,
                           double            *Y_re,
                           double            *Y_im)
  {
    const unsigned int rows    = n_coefficients(p);
    const unsigned int columns = (p + 1) * (p + 1);
    for (unsigned int r = 0; r < rows; ++r)
      {
        double        yr[B] = {}, yi[B] = {};
        const double *tr = T_re + r * columns;
        const double *ti = T_im + r * columns;
        for (unsigned int c = 0; c < columns; ++c)
          {
            const double  a = tr[c], b = ti[c];
            const double *xr = X_re + c * B;
            const double *xi = X_im + c * B;
#pragma omp simd
            for (unsigned int j = 0; j < B; ++j)
              {
                yr[j] += a * xr[j] - b * xi[j];
                yi[j] += a * xi[j] + b * xr[j];
              }
          }
        for (unsigned int j = 0; j < B; ++j)
          {
            Y_re[r * B + j] = yr[j];
            Y_im[r * B + j] = yi[j];
          }
      }
  }

  /**
   * The factors between the complex coefficients of this namespace and real
   * ones in the orthonormal real spherical harmonics Y_a, a = (n, m), of
   * YukawaExpansions::real_harmonics(), whose P_n^m carry the same
   * Condon-Shortley phase. With the norms c_m N_n^m of those harmonics
   * (c_0 = 1, c_m = sqrt(2)), P_n^m e^{im phi} = (Y_(n,m) + i Y_(n,-m)) /
   * (c_m N_n^m) for m > 0, so that a multipole expansion
   * sum M_n^m I_n^m(r), summed over both signs of m, is
   * sum_a c_a r^{-(n+1)} Y_a(r^) with
   *   c_(n,0) = outgoing(n,0) Re M_n^0,
   *   c_(n,m) = outgoing(n,m) Re M_n^m, c_(n,-m) = -outgoing(n,m) Im M_n^m,
   * outgoing(n,m) = (m > 0 ? 2 : 1) (-1)^m (n-m)! / (c_m N_n^m); and a local
   * expansion sum L_n^m conj(R_n^m(r)) with real coefficients d_a in the
   * basis r^n Y_a(r^) has
   *   L_n^0 = local(n,0) d_(n,0), L_n^m = local(n,m) (d_(n,m) + i d_(n,-m)),
   * local(n,m) = (-1)^m c_m N_n^m (n+m)! / (m > 0 ? 2 : 1). Both are indexed
   * like the stored coefficients, index(n, m) with m >= 0.
   */
  struct RealBasis
  {
    explicit RealBasis(const unsigned int p)
      : p(p)
      , outgoing(n_coefficients(p))
      , local(n_coefficients(p))
    {
      std::vector<double> factorial(2 * p + 1, 1.);
      for (unsigned int k = 1; k <= 2 * p; ++k)
        factorial[k] = factorial[k - 1] * k;
      for (unsigned int n = 0; n <= p; ++n)
        for (unsigned int m = 0; m <= n; ++m)
          {
            const double norm = (m > 0 ? std::sqrt(2.) : 1.) *
                                std::sqrt((2. * n + 1.) / (4. * M_PI) *
                                          factorial[n - m] / factorial[n + m]);
            const double sign = (m & 1) ? -1. : 1.;
            outgoing[index(n, m)] =
              (m > 0 ? 2. : 1.) * sign * factorial[n - m] / norm;
            local[index(n, m)] =
              sign * norm * factorial[n + m] / (m > 0 ? 2. : 1.);
          }
    }

    unsigned int        p;
    std::vector<double> outgoing, local;
  };
} // namespace LaplaceExpansions

#endif
