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

#ifndef helmholtz_expansions_h
#define helmholtz_expansions_h

#include <deal.II/base/tensor.h>

#include <cmath>
#include <complex>
#include <vector>

#include "yukawa_expansions.h"

/**
 * Multipole and local expansions of the Helmholtz kernel e^{i k |x - y|} /
 * |x - y|, for a fast multipole method in the low-frequency regime, where
 * k times the size of a box stays moderate.
 *
 * The kernel is the screened one e^{-kappa r} / r of YukawaExpansions at the
 * complex kappa = -i k, and its addition theorem, translations and their
 * derivation by Hobson's theorem hold unchanged by analytic continuation:
 * the coefficients are complex in the same real harmonics, the rotations
 * real and the translations along z complex
 * (YukawaExpansions::Operators::z_translation_unscaled_complex()). The
 * radial functions at the imaginary argument are those of the Helmholtz
 * equation, i_n(-i x) = i^{-n} j_n(x) and k_n(-i x) = -i^n h_n(x), with the
 * spherical Bessel functions j_n and the Hankel functions h_n = j_n + i y_n
 * of real x.
 *
 * The normalizations cannot be the radial functions at the radius of the
 * box, as for the screened kernel: j_n has zeros. They are real and
 * positive, of the size of the functions: |h_n(k rho)| for the outgoing
 * basis and 1 / (k rho |h_n(k rho)|), which behaves like j_n(k rho) for
 * small arguments and like its envelope for large ones, for the local
 * basis. A diagonal scaling does not change the precision; it keeps the
 * coefficients from under- and overflowing.
 */
namespace HelmholtzExpansions
{
  namespace YE  = YukawaExpansions;
  using complex = std::complex<double>;

  /**
   * The spherical Bessel functions j_n(x), n = 0..N, of real or complex x:
   * the power series for small |x|, Miller's downward recurrence otherwise,
   * normalized by j_0 or j_1, whichever is larger: one of them is not zero.
   */
  template <typename Number>
  void
  bessel_j(const unsigned int N, const Number x, Number *out)
  {
    using std::abs;
    if (abs(x) < 0.5)
      {
        // j_n(x) = x^n / (2n+1)!! sum_k (-x^2/2)^k / (k! (2n+3)...(2n+2k+1))
        const Number half_x2 = -0.5 * x * x;
        Number       leading = 1.;
        for (unsigned int n = 0; n <= N; ++n)
          {
            if (n > 0)
              leading *= x / double(2 * n + 1);
            Number sum = 1., term = 1.;
            for (unsigned int k = 1; k < 40; ++k)
              {
                term *= half_x2 / (double(k) * double(2 * n + 2 * k + 1));
                sum += term;
                if (abs(term) < 1e-18 * abs(sum))
                  break;
              }
            out[n] = leading * sum;
          }
        return;
      }
    const unsigned int start =
      std::max(N, 1u) + 20 + static_cast<unsigned int>(abs(x));
    const Number        inverse_x = 1. / x;
    Number              above = 0., current = 1e-280;
    std::vector<Number> values(std::max(N, 1u) + 1);
    for (unsigned int n = start; n > 0; --n)
      {
        // j_{n-1} = (2n+1)/x j_n - j_{n+1}
        const Number below = (double(2 * n + 1) * inverse_x) * current - above;
        above              = current;
        current            = below;
        if (n - 1 < values.size())
          values[n - 1] = current;
        if (abs(current) > 1e250)
          {
            above *= 1e-250;
            current *= 1e-250;
            for (unsigned int k = n - 1; k < values.size(); ++k)
              values[k] *= 1e-250;
          }
      }
    using std::cos;
    using std::sin;
    const Number j0    = sin(x) * inverse_x;
    const Number j1    = (sin(x) * inverse_x - cos(x)) * inverse_x;
    const Number scale = abs(j0) >= abs(j1) ? j0 / values[0] : j1 / values[1];
    for (unsigned int n = 0; n <= N; ++n)
      out[n] = scale * values[n];
  }

  /**
   * The spherical Hankel functions of the first kind h_n(x) = j_n(x) +
   * i y_n(x), n = 0..N, x > 0: y_n by its upward recurrence, which is
   * stable.
   */
  inline void
  bessel_h(const unsigned int N, const double x, complex *out)
  {
    std::vector<double> j(N + 1), y(std::max(N, 1u) + 1);
    bessel_j<double>(N, x, j.data());
    y[0] = -std::cos(x) / x;
    y[1] = -std::cos(x) / (x * x) - std::sin(x) / x;
    for (unsigned int n = 1; n + 1 <= N; ++n)
      y[n + 1] = double(2 * n + 1) / x * y[n] - y[n - 1];
    for (unsigned int n = 0; n <= N; ++n)
      out[n] = complex(j[n], y[n]);
  }

  /// i_n(kappa r) = i^{-n} j_n(k r) at kappa = -i k, n = 0..N.
  inline void
  modified_i(const unsigned int N, const double kr, complex *out)
  {
    std::vector<double> j(N + 1);
    bessel_j<double>(N, kr, j.data());
    complex power = 1.;
    for (unsigned int n = 0; n <= N; ++n, power *= complex(0., -1.))
      out[n] = power * j[n];
  }

  /// k_n(kappa r) = -i^n h_n(k r) at kappa = -i k, n = 0..N.
  inline void
  modified_k(const unsigned int N, const double kr, complex *out)
  {
    bessel_h(N, kr, out);
    complex power = -1.;
    for (unsigned int n = 0; n <= N; ++n, power *= complex(0., 1.))
      out[n] *= power;
  }

  /**
   * The normalizations of the expansions of a box of radius rho, n = 0..p,
   * in the form that LaplaceFMM applies around the translations, which work
   * between the unnormalized bases k_n Y_a and i_n Y_a: outgoing[n] =
   * 1 / |h_n(k rho)|, by which an outgoing coefficient is multiplied before,
   * and local[n] = 1 / (k rho |h_n(k rho)|), by which a local one is
   * multiplied after.
   */
  inline void
  normalizations(const unsigned int p,
                 const double       k,
                 const double       rho,
                 double            *outgoing,
                 double            *local)
  {
    std::vector<complex> h(p + 1);
    bessel_h(p, k * rho, h.data());
    for (unsigned int n = 0; n <= p; ++n)
      {
        outgoing[n] = 1. / std::abs(h[n]);
        local[n]    = 1. / (k * rho * std::abs(h[n]));
      }
  }

  /**
   * The translation along z by the distance |v| from an expansion of kind
   * `from` to one of kind `to`, between the unnormalized bases, complex.
   */
  inline void
  z_translation(const YE::Operators &operators,
                const YE::Basis      from,
                const YE::Basis      to,
                const double         k,
                const double         distance,
                complex             *Tz)
  {
    const unsigned int   p = operators.order();
    std::vector<complex> f(2 * p + 1);
    if (from == YE::Basis::outgoing && to == YE::Basis::local)
      modified_k(2 * p, k * distance, f.data());
    else
      modified_i(2 * p, k * distance, f.data());
    operators.z_translation_unscaled_complex(from, to, f.data(), Tz);
  }

  /**
   * Buffers of the expansions of order p, and the normalizations of the
   * last radius.
   */
  struct Workspace
  {
    explicit Workspace(const unsigned int p)
      : p(p)
      , norms(YE::harmonic_norms(p))
      , harmonics(YE::n_coefficients(p))
      , complex_harmonics(YE::n_coefficients(p))
      , radial(p + 1)
      , outgoing(p + 1)
      , local(p + 1)
    {}

    unsigned int         p;
    std::vector<double>  norms, harmonics;
    std::vector<complex> complex_harmonics, radial;
    std::vector<double>  outgoing, local;

    void
    set_radius(const double k, const double rho)
    {
      if (k == cached_k && rho == cached_rho)
        return;
      normalizations(p, k, rho, outgoing.data(), local.data());
      cached_k   = k;
      cached_rho = rho;
    }

  private:
    double cached_k = -1., cached_rho = -1.;
  };

  /**
   * P2M for a charge q at y - c: the coefficients of q e^{i k|x-y|}/|x-y|
   * in the outgoing basis k_n(kappa r) Y_a / |h_n(k rho)|, by the addition
   * theorem 4 pi kappa |h_n(k rho)| q i_n(kappa |y-c|) Y_a(y-c), added to M.
   */
  inline void
  add_charge(Workspace                  &w,
             const double                k,
             const double                rho,
             const complex               q,
             const dealii::Tensor<1, 3> &y_minus_c,
             complex                    *M)
  {
    const unsigned int p = w.p;
    const double       r = y_minus_c.norm();
    const complex      kappa(0., -k);
    w.set_radius(k, rho);
    modified_i(p, k * r, w.radial.data());
    if (r == 0.)
      {
        M[0] += 4. * M_PI * kappa / w.outgoing[0] * q * w.norms[0];
        return;
      }
    YE::real_harmonics<double>(p,
                               y_minus_c[0],
                               y_minus_c[1],
                               y_minus_c[2],
                               w.harmonics.data(),
                               w.norms.data());
    for (unsigned int n = 0; n <= p; ++n)
      {
        const complex factor =
          4. * M_PI * kappa / w.outgoing[n] * q * w.radial[n];
        for (int m = -int(n); m <= int(n); ++m)
          M[YE::index(n, m)] += factor * w.harmonics[YE::index(n, m)];
      }
  }

  /**
   * P2M for a dipole d, complex: the coefficients of d . grad_y
   * e^{i k|x-y|}/|x-y|. The coefficient of a unit charge is 4 pi kappa
   * |h_n| i^{-n} F_a(y - c), with the real function F_a = j_n(k r) Y_a; its
   * derivative along the real and the imaginary part of d by a complex step,
   * as YukawaExpansions::add_dipole() does for the screened kernel.
   */
  inline void
  add_dipole(Workspace                           &w,
             const double                         k,
             const double                         rho,
             const dealii::Tensor<1, 3, complex> &d,
             const dealii::Tensor<1, 3>          &y_minus_c,
             complex                             *M)
  {
    const unsigned int p = w.p;
    const complex      kappa(0., -k);
    w.set_radius(k, rho);
    for (const bool imaginary_part : {false, true})
      {
        dealii::Tensor<1, 3> e;
        for (unsigned int c = 0; c < 3; ++c)
          e[c] = imaginary_part ? d[c].imag() : d[c].real();
        const double e_norm = e.norm();
        if (e_norm == 0.)
          continue;
        const double  h = 1e-30 * std::max(y_minus_c.norm(), rho);
        const complex x(y_minus_c[0], h * e[0] / e_norm),
          y(y_minus_c[1], h * e[1] / e_norm),
          z(y_minus_c[2], h * e[2] / e_norm);
        const complex r = std::sqrt(x * x + y * y + z * z);
        bessel_j<complex>(p, k * r, w.radial.data());
        YE::real_harmonics<complex>(
          p, x, y, z, w.complex_harmonics.data(), w.norms.data());
        // d = Re d + i Im d
        const complex part = imaginary_part ? complex(0., 1.) : complex(1., 0.);
        complex       power = 1.;
        for (unsigned int n = 0; n <= p; ++n, power *= complex(0., -1.))
          {
            const complex factor =
              part * e_norm / h * 4. * M_PI * kappa / w.outgoing[n] * power;
            for (int m = -int(n); m <= int(n); ++m)
              M[YE::index(n, m)] +=
                factor *
                (w.radial[n] * w.complex_harmonics[YE::index(n, m)]).imag();
          }
      }
  }

  /// L2P: the value of the local expansion L, in the basis i_n(kappa r) Y_a
  /// k rho |h_n(k rho)|, at x - c.
  inline complex
  evaluate_local(Workspace                  &w,
                 const double                k,
                 const double                rho,
                 const complex              *L,
                 const dealii::Tensor<1, 3> &x_minus_c)
  {
    const unsigned int p = w.p;
    const double       r = x_minus_c.norm();
    w.set_radius(k, rho);
    modified_i(p, k * r, w.radial.data());
    if (r == 0.)
      return L[0] * w.radial[0] / w.local[0] * w.norms[0];
    YE::real_harmonics<double>(p,
                               x_minus_c[0],
                               x_minus_c[1],
                               x_minus_c[2],
                               w.harmonics.data(),
                               w.norms.data());
    complex u = 0.;
    for (unsigned int n = 0; n <= p; ++n)
      {
        complex sum = 0.;
        for (int m = -int(n); m <= int(n); ++m)
          sum += L[YE::index(n, m)] * w.harmonics[YE::index(n, m)];
        u += sum * w.radial[n] / w.local[n];
      }
    return u;
  }

  /// M2P: the value of the outgoing expansion M at x - c, for tests.
  inline complex
  evaluate_outgoing(Workspace                  &w,
                    const double                k,
                    const double                rho,
                    const complex              *M,
                    const dealii::Tensor<1, 3> &x_minus_c)
  {
    const unsigned int   p = w.p;
    std::vector<complex> radial(p + 1);
    w.set_radius(k, rho);
    modified_k(p, k * x_minus_c.norm(), radial.data());
    YE::real_harmonics<double>(p,
                               x_minus_c[0],
                               x_minus_c[1],
                               x_minus_c[2],
                               w.harmonics.data(),
                               w.norms.data());
    complex u = 0.;
    for (unsigned int n = 0; n <= p; ++n)
      {
        complex sum = 0.;
        for (int m = -int(n); m <= int(n); ++m)
          sum += M[YE::index(n, m)] * w.harmonics[YE::index(n, m)];
        u += sum * radial[n] * w.outgoing[n];
      }
    return u;
  }
} // namespace HelmholtzExpansions

#endif
