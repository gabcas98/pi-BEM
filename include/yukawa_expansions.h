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

#ifndef yukawa_expansions_h
#define yukawa_expansions_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

/**
 * Multipole and local expansions of the screened Coulomb (Yukawa) kernel
 * e^{-kappa |x - y|} / |x - y|, the building blocks of a fast multipole
 * method for the screened Poisson equation.
 *
 * With the modified spherical Bessel functions i_n (first kind) and k_n
 * (third kind, scaled so that k_0(z) = e^{-z}/z) and the real orthonormal
 * spherical harmonics Y_a, a = (n, m), -n <= m <= n, the addition theorem
 * @f[
 *   \frac{e^{-\kappa|x-y|}}{|x-y|}
 *   = 4\pi\kappa \sum_{n,m} i_n(\kappa|y|)\, Y_n^m(\hat y)\,
 *     k_n(\kappa|x|)\, Y_n^m(\hat x), \qquad |y| < |x|,
 * @f]
 * gives outgoing expansions about the centre c of a box, in the basis
 * O_a(x) = k_n(kappa r) / k_n(kappa rho) Y_a(r^), and local ones in the basis
 * J_a(x) = i_n(kappa r) / i_n(kappa rho) Y_a(r^), r = x - c, both
 * normalized on a sphere of radius rho about the centre so that their
 * coefficients stay of the size of the field: without this, i_n and k_n
 * under- and overflow when kappa times the size of a box is large or small.
 *
 * All coefficients are real. The translations between expansions (M2M,
 * M2L, L2L) follow from the addition theorem and the Gaunt coefficients of
 * the harmonics, applied by rotating the translation vector onto the
 * z-axis, where they are diagonal in the azimuthal order (class Operators).
 * A second construction, by projecting the field of each basis function
 * on a sphere about the new centre, needs nothing but the basis functions
 * and serves as reference in the tests.
 *
 * Expansions of order p have (p + 1)^2 coefficients, a = n * n + n + m.
 */
namespace YukawaExpansions
{
  inline unsigned int
  n_coefficients(const unsigned int p)
  {
    return (p + 1) * (p + 1);
  }

  inline unsigned int
  index(const unsigned int n, const int m)
  {
    return n * n + n + m;
  }

  /**
   * Modified spherical Bessel functions of the first kind i_n(z), n = 0..N,
   * for z >= 0, in any real or complex floating-point type: the power series
   * for small |z|, where the downward recurrence would overflow, and Miller's
   * downward recurrence normalized by i_0 = sinh(z)/z otherwise.
   */
  template <typename Number>
  void
  bessel_i(const unsigned int N, const Number z, Number *out)
  {
    using std::abs;
    if (abs(z) < 0.5)
      {
        // i_n(z) = z^n / (2n+1)!! sum_k (z^2/2)^k / (k! (2n+3)...(2n+2k+1))
        const Number half_z2 = 0.5 * z * z;
        Number       leading = 1.;
        for (unsigned int n = 0; n <= N; ++n)
          {
            if (n > 0)
              leading *= z / double(2 * n + 1);
            Number sum = 1., term = 1.;
            for (unsigned int k = 1; k < 40; ++k)
              {
                term *= half_z2 / (double(k) * double(2 * n + 2 * k + 1));
                sum += term;
                if (abs(term) < 1e-18 * abs(sum))
                  break;
              }
            out[n] = leading * sum;
          }
        return;
      }
    // Miller: i_{n-1} = i_{n+1} + (2n+1)/z i_n from well above N down to
    // zero, rescaled when it grows large, and normalized by i_0.
    const unsigned int start =
      N + 20 + static_cast<unsigned int>(std::abs(z));
    const Number inverse_z = 1. / z;
    Number       above = 0., current = 1e-280;
    for (unsigned int n = start; n > 0; --n)
      {
        const Number below = above + (double(2 * n + 1) * inverse_z) * current;
        above              = current;
        current            = below;
        if (n - 1 <= N)
          out[n - 1] = current;
        if (abs(current) > 1e250)
          {
            above *= 1e-250;
            current *= 1e-250;
            for (unsigned int k = n - 1; k <= N; ++k)
              out[k] *= 1e-250;
          }
      }
    using std::sinh;
    const Number scale = sinh(z) / z / out[0];
    for (unsigned int n = 0; n <= N; ++n)
      out[n] *= scale;
  }

  /**
   * Modified spherical Bessel functions of the third kind, scaled so that
   * k_0(z) = e^{-z}/z, n = 0..N, by the upward recurrence
   * k_{n+1} = k_{n-1} + (2n+1)/z k_n, which is stable.
   */
  template <typename Number>
  void
  bessel_k(const unsigned int N, const Number z, Number *out)
  {
    using std::exp;
    const Number inverse_z = 1. / z;
    const Number e         = exp(-z) * inverse_z;
    out[0]                 = e;
    if (N >= 1)
      out[1] = e * (1. + z) * inverse_z;
    for (unsigned int n = 1; n + 1 <= N; ++n)
      out[n + 1] = out[n - 1] + (double(2 * n + 1) * inverse_z) * out[n];
  }

  /**
   * Real orthonormal spherical harmonics Y_a of the direction of r, degree
   * n <= N, at out[n*n + n + m]: sqrt(2) N_n^m P_n^m(cos theta) cos(m phi)
   * for m > 0, the same with sin(|m| phi) for m < 0, N_n^0 P_n(cos theta)
   * for m = 0. The factors sqrt(2) N_n^m may be passed in norms, indexed
   * like the harmonics for m >= 0 (see harmonic_norms()). The factors
   * sin^m(theta) cos(m phi) and sin^m(theta) sin(m phi) are the real and
   * imaginary parts of ((x + i y)/|r|)^m, and
   * P_n^m / sin^m(theta) follows from a recurrence in cos(theta): no
   * trigonometric function is evaluated, and the formula holds for complex
   * components too.
   */
  template <typename Number>
  void
  real_harmonics(const unsigned int N,
                 const Number       x,
                 const Number       y,
                 const Number       z,
                 Number            *out,
                 const double      *norms = nullptr)
  {
    using std::sqrt;
    const Number r  = sqrt(x * x + y * y + z * z);
    const Number ct = z / r;
    const Number wx = x / r, wy = y / r;
    // (wx + i wy)^m as (c_m, s_m) with real arithmetic in Number
    Number cm = 1., sm = 0.;
    // p_m^m = (-1)^m (2m-1)!! without the sin^m factor
    Number pmm = 1.;
    for (unsigned int m = 0; m <= N; ++m)
      {
        if (m > 0)
          {
            const Number c = cm * wx - sm * wy;
            sm             = cm * wy + sm * wx;
            cm             = c;
            pmm *= -double(2 * m - 1);
          }
        // normalization sqrt((2n+1)/(4 pi) (n-m)!/(n+m)!) built along n
        double ratio = 1.; // (n-m)!/(n+m)! at n = m: 1/(2m)!
        if (norms == nullptr)
          for (unsigned int k = 1; k <= 2 * m; ++k)
            ratio /= double(k);
        Number p_prev = 0., p = pmm;
        for (unsigned int n = m; n <= N; ++n)
          {
            if (n > m)
              {
                const Number p_next =
                  n == m + 1 ?
                    ct * double(2 * m + 1) * pmm :
                    (double(2 * n - 1) * ct * p -
                     double(n + m - 1) * p_prev) /
                      double(n - m);
                p_prev = p;
                p      = p_next;
                // (n-m)!/(n+m)! from (n-1-m)!/(n-1+m)!
                if (norms == nullptr)
                  ratio *= double(n - m) / double(n + m);
              }
            const double norm =
              norms != nullptr ?
                norms[index(n, m)] :
                (m > 0 ? std::sqrt(2.) : 1.) *
                  std::sqrt((2. * n + 1.) / (4. * M_PI) * ratio);
            if (m == 0)
              out[index(n, 0)] = norm * p;
            else
              {
                out[index(n, m)]       = norm * p * cm;
                out[index(n, -int(m))] = norm * p * sm;
              }
          }
      }
  }

  /// The factors of real_harmonics(): sqrt(2) N_n^m for m > 0, N_n^0.
  inline std::vector<double>
  harmonic_norms(const unsigned int N)
  {
    std::vector<double> norms(n_coefficients(N), 0.);
    for (unsigned int n = 0; n <= N; ++n)
      for (unsigned int m = 0; m <= n; ++m)
        {
          double ratio = 1.; // (n-m)!/(n+m)!
          for (unsigned int k = n - m + 1; k <= n + m; ++k)
            ratio /= double(k);
          norms[index(n, m)] = (m > 0 ? std::sqrt(2.) : 1.) *
                               std::sqrt((2. * n + 1.) / (4. * M_PI) * ratio);
        }
    return norms;
  }

  /**
   * A rule on the unit sphere: Gauss-Legendre points in cos(theta) times
   * equispaced points in phi, exact for the spherical harmonics up to
   * degree 2 n_theta - 1 in theta and n_phi - 1 in the azimuthal order.
   */
  struct SphereRule
  {
    std::vector<dealii::Tensor<1, 3>> directions;
    std::vector<double>               weights;

    SphereRule() = default;

    SphereRule(const unsigned int n_theta, const unsigned int n_phi)
    {
      const dealii::QGauss<1> gauss(n_theta);
      for (unsigned int i = 0; i < n_theta; ++i)
        {
          const double ct = 2. * gauss.point(i)[0] - 1.;
          const double st = std::sqrt(std::max(0., 1. - ct * ct));
          for (unsigned int j = 0; j < n_phi; ++j)
            {
              const double phi = 2. * M_PI * j / n_phi;
              directions.push_back(dealii::Tensor<1, 3>(
                {st * std::cos(phi), st * std::sin(phi), ct}));
              weights.push_back(2. * gauss.weight(i) * 2. * M_PI / n_phi);
            }
        }
    }

    unsigned int
    size() const
    {
      return weights.size();
    }
  };

  /// The rule for expansions of order p: exact up to degree p + extra.
  inline SphereRule
  sphere_rule(const unsigned int p, const unsigned int extra = 8)
  {
    return SphereRule(p + 1 + extra, 2 * (p + extra) + 2);
  }

  /**
   * Buffers and tables for the expansions of order p: the harmonics, the
   * radial functions, and their complex counterparts for the dipoles.
   */
  struct Workspace
  {
    explicit Workspace(const unsigned int p)
      : p(p)
      , norms(harmonic_norms(p))
      , harmonics(n_coefficients(p))
      , radial(p + 1)
      , radial_rho(p + 1)
      , complex_harmonics(n_coefficients(p))
      , complex_radial(p + 1)
    {}

    unsigned int                      p;
    std::vector<double>               norms, harmonics, radial, radial_rho;
    std::vector<std::complex<double>> complex_harmonics, complex_radial;

    /// radial_rho = k_n(kappa rho) or i_n(kappa rho), computed only when
    /// the arguments change: all points of a box share them.
    void
    set_radius(const bool outgoing, const double kappa, const double rho)
    {
      if (outgoing == cached_outgoing && kappa == cached_kappa &&
          rho == cached_rho)
        return;
      if (outgoing)
        bessel_k<double>(p, kappa * rho, radial_rho.data());
      else
        bessel_i<double>(p, kappa * rho, radial_rho.data());
      cached_outgoing = outgoing;
      cached_kappa    = kappa;
      cached_rho      = rho;
    }

  private:
    bool   cached_outgoing = false;
    double cached_kappa = -1., cached_rho = -1.;
  };

  /**
   * The values of the outgoing basis functions O_a, a < (p+1)^2, of a box
   * with normalization radius rho at the offset x - c from its centre.
   */
  inline void
  outgoing_basis(Workspace                  &w,
                 const double                kappa,
                 const double                rho,
                 const dealii::Tensor<1, 3> &x_minus_c,
                 double                     *out)
  {
    const unsigned int p = w.p;
    bessel_k<double>(p, kappa * x_minus_c.norm(), w.radial.data());
    w.set_radius(true, kappa, rho);
    real_harmonics<double>(
      p, x_minus_c[0], x_minus_c[1], x_minus_c[2], out, w.norms.data());
    for (unsigned int n = 0; n <= p; ++n)
      {
        const double ratio = w.radial[n] / w.radial_rho[n];
        for (int m = -int(n); m <= int(n); ++m)
          out[index(n, m)] *= ratio;
      }
  }

  /// The same for the local basis functions J_a.
  inline void
  local_basis(Workspace                  &w,
              const double                kappa,
              const double                rho,
              const dealii::Tensor<1, 3> &x_minus_c,
              double                     *out)
  {
    const unsigned int p = w.p;
    const double       r = x_minus_c.norm();
    bessel_i<double>(p, kappa * r, w.radial.data());
    w.set_radius(false, kappa, rho);
    if (r == 0.)
      {
        for (unsigned int a = 0; a < n_coefficients(p); ++a)
          out[a] = 0.;
        out[0] = w.norms[0] / w.radial_rho[0];
        return;
      }
    real_harmonics<double>(
      p, x_minus_c[0], x_minus_c[1], x_minus_c[2], out, w.norms.data());
    for (unsigned int n = 0; n <= p; ++n)
      {
        const double ratio = w.radial[n] / w.radial_rho[n];
        for (int m = -int(n); m <= int(n); ++m)
          out[index(n, m)] *= ratio;
      }
  }

  /**
   * P2M for a charge q at y - c: the coefficients of q e^{-kappa|x-y|}/|x-y|
   * in the outgoing basis of radius rho, from the addition theorem,
   * 4 pi kappa k_n(kappa rho) q i_n(kappa |y-c|) Y_a(y-c), added to M.
   */
  inline void
  add_charge(Workspace                  &w,
             const double                kappa,
             const double                rho,
             const double                q,
             const dealii::Tensor<1, 3> &y_minus_c,
             double                     *M)
  {
    const unsigned int p = w.p;
    const double       r = y_minus_c.norm();
    bessel_i<double>(p, kappa * r, w.radial.data());
    w.set_radius(true, kappa, rho);
    if (r == 0.)
      {
        M[0] += 4. * M_PI * kappa * w.radial_rho[0] * q * w.norms[0];
        return;
      }
    real_harmonics<double>(p,
                           y_minus_c[0],
                           y_minus_c[1],
                           y_minus_c[2],
                           w.harmonics.data(),
                           w.norms.data());
    for (unsigned int n = 0; n <= p; ++n)
      {
        const double factor =
          4. * M_PI * kappa * w.radial_rho[n] * q * w.radial[n];
        for (int m = -int(n); m <= int(n); ++m)
          M[index(n, m)] += factor * w.harmonics[index(n, m)];
      }
  }

  /**
   * P2M for a dipole: the coefficients of d . grad_y e^{-kappa|x-y|}/|x-y|,
   * the derivative of those of a unit charge along d, computed by a complex
   * step: f(y + i h e) = f(y) + i h e.grad f(y) + O(h^2) with no
   * cancellation, so a step of 1e-30 relative gives the derivative to
   * rounding. It holds at y = c too, where the harmonics of the complex
   * offset reduce to those of the direction e.
   */
  inline void
  add_dipole(Workspace                  &w,
             const double                kappa,
             const double                rho,
             const dealii::Tensor<1, 3> &d,
             const dealii::Tensor<1, 3> &y_minus_c,
             double                     *M)
  {
    using complex       = std::complex<double>;
    const unsigned int p      = w.p;
    const double       d_norm = d.norm();
    if (d_norm == 0.)
      return;
    const double  h = 1e-30 * std::max(y_minus_c.norm(), rho);
    const complex x(y_minus_c[0], h * d[0] / d_norm),
      y(y_minus_c[1], h * d[1] / d_norm), z(y_minus_c[2], h * d[2] / d_norm);
    const complex r = std::sqrt(x * x + y * y + z * z);
    bessel_i<complex>(p, kappa * r, w.complex_radial.data());
    w.set_radius(true, kappa, rho);
    real_harmonics<complex>(
      p, x, y, z, w.complex_harmonics.data(), w.norms.data());
    for (unsigned int n = 0; n <= p; ++n)
      {
        const complex factor =
          4. * M_PI * kappa * w.radial_rho[n] * w.complex_radial[n];
        const double scale = d_norm / h;
        for (int m = -int(n); m <= int(n); ++m)
          M[index(n, m)] +=
            scale * (factor * w.complex_harmonics[index(n, m)]).imag();
      }
  }

  /// L2P: the value of the local expansion L at x - c.
  inline double
  evaluate_local(Workspace                  &w,
                 const double                kappa,
                 const double                rho,
                 const double               *L,
                 const dealii::Tensor<1, 3> &x_minus_c)
  {
    local_basis(w, kappa, rho, x_minus_c, w.harmonics.data());
    double u = 0.;
    for (unsigned int a = 0; a < n_coefficients(w.p); ++a)
      u += L[a] * w.harmonics[a];
    return u;
  }

  /// M2P: the value of the outgoing expansion M at x - c.
  inline double
  evaluate_outgoing(Workspace                  &w,
                    const double                kappa,
                    const double                rho,
                    const double               *M,
                    const dealii::Tensor<1, 3> &x_minus_c)
  {
    outgoing_basis(w, kappa, rho, x_minus_c, w.harmonics.data());
    double u = 0.;
    for (unsigned int a = 0; a < n_coefficients(w.p); ++a)
      u += M[a] * w.harmonics[a];
    return u;
  }

  enum class Basis
  {
    outgoing,
    local
  };

  /**
   * The matrix T, (p+1)^2 x (p+1)^2 row by row, that takes the coefficients
   * of an expansion of kind `from` about c_from with radius rho_from to
   * those of kind `to` about c_to with radius rho_to, by projection: T_{ab}
   * is the coefficient of the basis function a of the new expansion in the
   * basis function b of the old one, the projection of the latter on the
   * spherical harmonics on the sphere of radius s about c_to. The sphere
   * must lie where both expansions converge. A reference for the analytic
   * operators of the class Operators; too slow for the method itself.
   */
  inline void
  projected_translation(const unsigned int          p,
                        const double                kappa,
                        const SphereRule           &rule,
                        const Basis                 from,
                        const double                rho_from,
                        const Basis                 to,
                        const double                rho_to,
                        const dealii::Tensor<1, 3> &c_to_minus_c_from,
                        const double                s,
                        double                     *T)
  {
    const unsigned int  nc = n_coefficients(p);
    Workspace           w(p);
    std::vector<double> basis(nc), Y(nc), radial_s(p + 1), radial_rho(p + 1);
    std::vector<double> sampled(rule.size() * nc), harmonics(rule.size() * nc);
    for (unsigned int q = 0; q < rule.size(); ++q)
      {
        const dealii::Tensor<1, 3> x =
          c_to_minus_c_from + s * rule.directions[q];
        if (from == Basis::outgoing)
          outgoing_basis(w, kappa, rho_from, x, basis.data());
        else
          local_basis(w, kappa, rho_from, x, basis.data());
        const dealii::Tensor<1, 3> &e = rule.directions[q];
        real_harmonics<double>(p, e[0], e[1], e[2], Y.data());
        for (unsigned int b = 0; b < nc; ++b)
          {
            sampled[q * nc + b]   = basis[b] * rule.weights[q];
            harmonics[q * nc + b] = Y[b];
          }
      }
    // Projected coefficient of Y_a on the sphere of radius s, converted to
    // the normalization at rho_to.
    if (to == Basis::outgoing)
      {
        bessel_k<double>(p, kappa * s, radial_s.data());
        bessel_k<double>(p, kappa * rho_to, radial_rho.data());
      }
    else
      {
        bessel_i<double>(p, kappa * s, radial_s.data());
        bessel_i<double>(p, kappa * rho_to, radial_rho.data());
      }
    for (unsigned int n = 0; n <= p; ++n)
      for (int m = -int(n); m <= int(n); ++m)
        {
          const unsigned int a     = index(n, m);
          const double       scale = radial_rho[n] / radial_s[n];
          for (unsigned int b = 0; b < nc; ++b)
            {
              double sum = 0.;
              for (unsigned int q = 0; q < rule.size(); ++q)
                sum += harmonics[q * nc + a] * sampled[q * nc + b];
              T[a * nc + b] = scale * sum;
            }
        }
  }

  /**
   * The translations of expansions of order p, M2M, M2L and L2L, by
   * rotation and translation along the z-axis ("point and shoot"), in
   * O(p^3) operations instead of the O(p^4) of a dense matrix. The
   * rotations act on any expansion in the real harmonics, and the class
   * also gives the M2L translation along z of the Laplace kernel
   * (z_translation_laplace()), which LaplaceFMM applies in the same way.
   *
   * Let v = c_old - c_new be the vector from the new centre to the old one.
   * The derivation rests on two facts. By Hobson's theorem a basis function
   * is a harmonic polynomial of the gradient applied to the zeroth one,
   * k_n(kappa r) Y_a(r^) = (-kappa)^{-n} P_a(grad) k_0(kappa r) and
   * i_n(kappa r) Y_a(r^) = kappa^{-n} P_a(grad) i_0(kappa r), with
   * P_a(x) = |x|^n Y_a(x^); and the product of two such polynomials is a sum
   * of |x|^{2j} times harmonic polynomials with the Gaunt coefficients
   * G(a, b, c) = int Y_a Y_b Y_c as weights, where |grad|^2 = kappa^2 on
   * both kinds of functions. Expanding the zeroth function of the old
   * centre about the new one by the addition theorem gives
   * @f[
   *   T_{a'a} = 4\pi\, s_{n n'} \sum_{l, M} G(a, (l, M), a')\,
   *             f_l(\kappa |v|)\, Y_l^M(\hat v),
   * @f]
   * the coefficient of the new basis function a' in the old function a
   * (without the normalization radii), with f = i and s = 1 for M2M,
   * f = i and s = (-1)^{n+n'} for L2L, f = k and s = (-1)^n for M2L. When v
   * points along z only M = 0 remains, and T is diagonal in m: blocks of
   * size (p + 1 - |m|)^2, the same for m and -m. A general v is rotated to
   * the z-axis by Q = R_y(-theta) R_z(-phi), with theta and phi the angles
   * of v. On the coefficients a rotation about z acts on the pairs
   * (m, -m) by 2 x 2 rotations; one about y is P R_z P, with P the
   * reflection that exchanges y and z, whose matrix on each degree is
   * computed once. The translation is then
   * @f[
   *   T = D(Q)^T\, T_z\, D(Q), \quad D(Q) = D(P)\, Z(\theta)\, D(P)\,
   *       Z(-\phi).
   * @f]
   * The functions that apply the steps work on B expansions at once, stored
   * coefficient by coefficient (X[a * B + j] for expansion j), so that the
   * innermost loops run over the batch and vectorize.
   */
  class Operators
  {
  public:
    explicit Operators(const unsigned int p)
      : p(p)
      , nc(n_coefficients(p))
      , norms(harmonic_norms(2 * p))
    {
      // Gaunt coefficients int Y_(n,m) Y_(n',m) Y_(l,0), the same for m and
      // -m, by Gauss-Legendre in cos(theta): the integrand is a polynomial
      // of degree n + n' + l <= 4p.
      const dealii::QGauss<1>                      gauss(2 * p + 2);
      std::vector<std::vector<double>>             legendre(gauss.size());
      for (unsigned int k = 0; k < gauss.size(); ++k)
        {
          const double x = 2. * gauss.point(k)[0] - 1.;
          legendre[k].resize(n_coefficients(2 * p));
          real_harmonics<double>(2 * p,
                                 std::sqrt(std::max(0., 1. - x * x)),
                                 0.,
                                 x,
                                 legendre[k].data(),
                                 norms.data());
        }
      gaunt.assign((p + 1) * (p + 1) * (p + 1) * (2 * p + 1), 0.);
      for (unsigned int m = 0; m <= p; ++m)
        for (unsigned int n = m; n <= p; ++n)
          for (unsigned int n2 = m; n2 <= p; ++n2)
            for (unsigned int l = (n > n2 ? n - n2 : n2 - n); l <= n + n2;
                 l += 2)
              {
                // Y_(n,m)(phi = 0) = sqrt(2) N P for m > 0, and the
                // azimuthal integral of cos^2 is pi instead of 2 pi.
                double sum = 0.;
                for (unsigned int k = 0; k < gauss.size(); ++k)
                  sum += 2. * gauss.weight(k) * legendre[k][index(n, m)] *
                         legendre[k][index(n2, m)] * legendre[k][index(l, 0)];
                gaunt[gaunt_index(m, n, n2, l)] =
                  (m > 0 ? M_PI : 2. * M_PI) * sum;
              }

      // The matrix of the exchange of y and z on each degree, by an exact
      // rule: D(P)_{ba} = int Y_b(w) Y_a(P w) dw.
      const SphereRule    rule(p + 1, 2 * p + 2);
      std::vector<double> Y(nc), Yp(nc), D(nc * nc, 0.);
      for (unsigned int q = 0; q < rule.size(); ++q)
        {
          const dealii::Tensor<1, 3> &e = rule.directions[q];
          real_harmonics<double>(p, e[0], e[1], e[2], Y.data(), norms.data());
          real_harmonics<double>(p, e[0], e[2], e[1], Yp.data(), norms.data());
          for (unsigned int n = 0; n <= p; ++n)
            for (unsigned int b = n * n; b < (n + 1) * (n + 1); ++b)
              for (unsigned int a = n * n; a < (n + 1) * (n + 1); ++a)
                D[b * nc + a] += rule.weights[q] * Y[b] * Yp[a];
        }
      swap_start.assign(1, 0);
      for (unsigned int b = 0; b < nc; ++b)
        {
          for (unsigned int a = 0; a < nc; ++a)
            if (std::abs(D[b * nc + a]) > 1e-13)
              {
                swap_column.push_back(a);
                swap_value.push_back(D[b * nc + a]);
              }
          swap_start.push_back(swap_column.size());
        }

      axis_factor.resize(2 * p + 1);
      for (unsigned int l = 0; l <= 2 * p; ++l)
        axis_factor[l] = std::sqrt(4. * M_PI * (2. * l + 1.));
      scratch_f.resize(2 * p + 1);
      scratch_from.resize(p + 1);
      scratch_to.resize(p + 1);

      block_start.resize(p + 2);
      block_start[0] = 0;
      for (unsigned int m = 0; m <= p; ++m)
        block_start[m + 1] = block_start[m] + (p + 1 - m) * (p + 1 - m);

      rotation_start.resize(p + 2);
      rotation_start[0] = 0;
      for (unsigned int n = 0; n <= p; ++n)
        rotation_start[n + 1] =
          rotation_start[n] + (2 * n + 1) * (2 * n + 1);
      swap_blocks.resize(rotation_start[p + 1]);
      for (unsigned int n = 0; n <= p; ++n)
        for (unsigned int b = 0; b < 2 * n + 1; ++b)
          for (unsigned int a = 0; a < 2 * n + 1; ++a)
            swap_blocks[rotation_start[n] + b * (2 * n + 1) + a] =
              D[(n * n + b) * nc + n * n + a];
    }

    unsigned int
    order() const
    {
      return p;
    }

    /// Size of the translation along z, as z_translation() writes it.
    unsigned int
    z_translation_size() const
    {
      return block_start[p + 1];
    }

    /**
     * The translation along z by the distance |v| from an expansion of kind
     * `from` with radius rho_from to one of kind `to` with radius rho_to,
     * written block by block: for m = 0..p the rows n' = m..p and the
     * columns n = m..p. The normalization radii are included.
     */
    void
    z_translation(const Basis  from,
                  const Basis  to,
                  const double kappa,
                  const double distance,
                  const double rho_from,
                  const double rho_to,
                  double      *Tz) const
    {
      z_translation_unscaled(from, to, kappa, distance, Tz);
      std::vector<double> &from_rho = scratch_from, &to_rho = scratch_to;
      if (from == Basis::outgoing)
        bessel_k<double>(p, kappa * rho_from, from_rho.data());
      else
        bessel_i<double>(p, kappa * rho_from, from_rho.data());
      if (to == Basis::outgoing)
        bessel_k<double>(p, kappa * rho_to, to_rho.data());
      else
        bessel_i<double>(p, kappa * rho_to, to_rho.data());
      for (unsigned int m = 0; m <= p; ++m)
        {
          double *block = Tz + block_start[m];
          for (unsigned int n2 = m; n2 <= p; ++n2)
            for (unsigned int n = m; n <= p; ++n)
              block[(n2 - m) * (p + 1 - m) + (n - m)] *=
                to_rho[n2] / from_rho[n];
        }
    }

    /**
     * The same between the unnormalized bases k_n Y_a and i_n Y_a. The
     * normalization is a scaling of each degree, which commutes with the
     * rotations; applied separately, it lets the pairs of boxes of
     * different sizes share one translation along z.
     */
    void
    z_translation_unscaled(const Basis  from,
                           const Basis  to,
                           const double kappa,
                           const double distance,
                           double      *Tz) const
    {
      AssertThrow(!(from == Basis::local && to == Basis::outgoing),
                  dealii::ExcMessage("A local expansion has no outgoing "
                                     "one."));
      std::vector<double> &f = scratch_f;
      if (from == Basis::outgoing && to == Basis::local)
        bessel_k<double>(2 * p, kappa * distance, f.data());
      else
        bessel_i<double>(2 * p, kappa * distance, f.data());
      // 4 pi Y_(l,0)(z^) = sqrt(4 pi (2l + 1))
      for (unsigned int l = 0; l <= 2 * p; ++l)
        f[l] *= axis_factor[l];
      for (unsigned int m = 0; m <= p; ++m)
        {
          double *block = Tz + block_start[m];
          for (unsigned int n2 = m; n2 <= p; ++n2)
            for (unsigned int n = m; n <= p; ++n)
              {
                double sum = 0.;
                for (unsigned int l = (n > n2 ? n - n2 : n2 - n); l <= n + n2;
                     l += 2)
                  sum += gaunt[gaunt_index(m, n, n2, l)] * f[l];
                double sign = 1.;
                if (from == Basis::outgoing && to == Basis::local)
                  sign = (n % 2) ? -1. : 1.;
                else if (from == Basis::local)
                  sign = ((n + n2) % 2) ? -1. : 1.;
                block[(n2 - m) * (p + 1 - m) + (n - m)] = sign * sum;
              }
        }
    }

    /**
     * The translation along z by the distance |v| of the Laplace kernel 1/r,
     * from the outgoing basis r^{-(n+1)} Y_a to the local basis r^n Y_a,
     * in the layout of z_translation(): the limit kappa -> 0 of the screened
     * multipole-to-local translation, in which, with
     * k_n(z) ~ (2n-1)!!/z^{n+1} and i_n(z) ~ z^n/(2n+1)!!, only the term
     * l = n + n' of the sum remains:
     * T_{(n',m),(n,m)} = (-1)^n sqrt(4 pi (2l+1)) G(n, n', l, m)
     *                    (2l-1)!! / ((2n-1)!! (2n'+1)!!) / |v|^{l+1}.
     */
    void
    z_translation_laplace(const double distance, double *Tz) const
    {
      // (2k-1)!!, k = 0..2p+1, and the powers of 1/|v|; computed once per
      // distance, so plain vectors.
      std::vector<double> double_factorial(2 * p + 2),
        inverse_power(2 * p + 2);
      double_factorial[0] = 1.;
      for (unsigned int k = 1; k < double_factorial.size(); ++k)
        double_factorial[k] = double_factorial[k - 1] * (2. * k - 1.);
      inverse_power[0] = 1.;
      for (unsigned int k = 1; k < inverse_power.size(); ++k)
        inverse_power[k] = inverse_power[k - 1] / distance;
      for (unsigned int m = 0; m <= p; ++m)
        {
          double *block = Tz + block_start[m];
          for (unsigned int n2 = m; n2 <= p; ++n2)
            for (unsigned int n = m; n <= p; ++n)
              {
                const unsigned int l = n + n2;
                block[(n2 - m) * (p + 1 - m) + (n - m)] =
                  ((n % 2) ? -1. : 1.) * axis_factor[l] *
                  gaunt[gaunt_index(m, n, n2, l)] * double_factorial[l] /
                  (double_factorial[n] * double_factorial[n2 + 1]) *
                  inverse_power[l + 1];
              }
        }
    }

    /**
     * cos(m alpha) and sin(m alpha), m = 0..p, of the angles theta and phi
     * of v, the input of translate().
     */
    struct Angles
    {
      std::vector<double> cos_theta, sin_theta, cos_phi, sin_phi;
    };

    void
    angles(const dealii::Tensor<1, 3> &v, Angles &result) const
    {
      // e^{i theta} = (v_z + i rho) / |v| and e^{i phi} = (v_x + i v_y) /
      // rho, rho the distance from the axis, and their powers: no
      // trigonometric function.
      const double r   = v.norm();
      const double rho = std::sqrt(v[0] * v[0] + v[1] * v[1]);
      const std::complex<double> e_theta(v[2] / r, rho / r);
      const std::complex<double> e_phi =
        rho > 0 ? std::complex<double>(v[0] / rho, v[1] / rho) :
                  std::complex<double>(1., 0.);
      result.cos_theta.resize(p + 1);
      result.sin_theta.resize(p + 1);
      result.cos_phi.resize(p + 1);
      result.sin_phi.resize(p + 1);
      std::complex<double> power_theta(1., 0.), power_phi(1., 0.);
      for (unsigned int m = 0; m <= p; ++m)
        {
          result.cos_theta[m] = power_theta.real();
          result.sin_theta[m] = power_theta.imag();
          result.cos_phi[m]   = power_phi.real();
          result.sin_phi[m]   = power_phi.imag();
          power_theta *= e_theta;
          power_phi *= e_phi;
        }
    }

    /**
     * Y += T X for B expansions, with T = D(Q)^T T_z D(Q) given by the
     * angles of v and the translation along z; X is overwritten.
     */
    template <unsigned int B>
    void
    translate(const Angles &a,
              const double *Tz,
              double       *X,
              double       *Y,
              double       *work) const
    {
      rotate_z<B>(a.cos_phi.data(), a.sin_phi.data(), -1., X);
      apply_swap<B>(X, work);
      rotate_z<B>(a.cos_theta.data(), a.sin_theta.data(), 1., work);
      apply_swap<B>(work, X);
      apply_z_translation<B>(Tz, X, work);
      apply_swap<B>(work, X);
      rotate_z<B>(a.cos_theta.data(), a.sin_theta.data(), -1., X);
      apply_swap<B>(X, work);
      rotate_z<B>(a.cos_phi.data(), a.sin_phi.data(), 1., work);
      for (unsigned int i = 0; i < nc * B; ++i)
        Y[i] += work[i];
    }

    /**
     * The same with a translation along z for each of the B expansions,
     * Tz[j] for expansion j; the rotation is common. When all are the same
     * this is translate(), otherwise they are gathered into Tz_lanes, of
     * z_translation_size() * B entries.
     */
    template <unsigned int B>
    void
    translate(const Angles        &a,
              const double *const *Tz,
              double              *X,
              double              *Y,
              double              *work,
              double              *Tz_lanes) const
    {
      bool common = true;
      for (unsigned int j = 1; j < B; ++j)
        common = common && Tz[j] == Tz[0];
      if (common)
        {
          translate<B>(a, Tz[0], X, Y, work);
          return;
        }
      const unsigned int size = z_translation_size();
      for (unsigned int e = 0; e < size; ++e)
        for (unsigned int j = 0; j < B; ++j)
          Tz_lanes[e * B + j] = Tz[j][e];
      rotate_z<B>(a.cos_phi.data(), a.sin_phi.data(), -1., X);
      apply_swap<B>(X, work);
      rotate_z<B>(a.cos_theta.data(), a.sin_theta.data(), 1., work);
      apply_swap<B>(work, X);
      apply_z_translation_lanes<B>(Tz_lanes, X, work);
      apply_swap<B>(work, X);
      rotate_z<B>(a.cos_theta.data(), a.sin_theta.data(), -1., X);
      apply_swap<B>(X, work);
      rotate_z<B>(a.cos_phi.data(), a.sin_phi.data(), 1., work);
      for (unsigned int i = 0; i < nc * B; ++i)
        Y[i] += work[i];
    }

    /// The number of entries of the blocks of a rotation, the sum of
    /// (2n + 1)^2 over the degrees.
    unsigned int
    rotation_size() const
    {
      return rotation_start[p + 1];
    }

    /**
     * The blocks on the degrees of D(P) Z(theta) D(P), the rotation by theta
     * about the y-axis, for the angle theta of v (the angle phi is not
     * used), row by row (block n at the sum of (2k + 1)^2 over k < n), of
     * rotation_size() entries. With Z(-phi), cheap, they give the rotation D(Q) of
     * translate(), and translate_blocks() applies them as dense products;
     * the blocks depend on the polar angle of the offset only, so that the
     * groups of offsets of a tree share them.
     */
    void
    rotation_blocks(const Angles &a, double *R) const
    {
      double C[(2 * maximum_rotation_order + 1) *
               (2 * maximum_rotation_order + 1)];
      AssertThrow(p <= maximum_rotation_order,
                  dealii::ExcMessage("Order too high for the rotation blocks."));
      for (unsigned int n = 0; n <= p; ++n)
        {
          const unsigned int s     = 2 * n + 1;
          const unsigned int first = n * n;
          const double      *P     = swap_blocks.data() + rotation_start[n];
          // C = Z(theta) D(P): the rows of the orders m and -m of D(P),
          // n + m and n - m, rotated.
          for (unsigned int k = 0; k < s * s; ++k)
            C[k] = P[k];
          for (unsigned int m = 1; m <= n; ++m)
            {
              double      *u  = C + (n + m) * s;
              double      *w  = C + (n - m) * s;
              const double cm = a.cos_theta[m], sm = a.sin_theta[m];
              for (unsigned int k = 0; k < s; ++k)
                {
                  const double x = u[k], y = w[k];
                  u[k]           = cm * x - sm * y;
                  w[k]           = sm * x + cm * y;
                }
            }
          // R = D(P) C, by the rows of D(P).
          double *block = R + rotation_start[n];
          for (unsigned int b = 0; b < s; ++b)
            {
              double *row = block + b * s;
              for (unsigned int k = 0; k < s; ++k)
                row[k] = 0.;
              for (unsigned int e = swap_start[first + b];
                   e < swap_start[first + b + 1];
                   ++e)
                {
                  const double  v = swap_value[e];
                  const double *c = C + (swap_column[e] - first) * s;
                  for (unsigned int k = 0; k < s; ++k)
                    row[k] += v * c[k];
                }
            }
        }
    }

    /**
     * Y = D(Q)^T T_z D(Q) X for B expansions, the translation of
     * translate() without the sum, with D(Q) = R Z(-phi): the blocks R of
     * rotation_blocks() for theta and the angle phi of a; dense products,
     * whose sums stay in registers. X is overwritten.
     */
    template <unsigned int B>
    void
    translate_blocks(const Angles &a,
                     const double *R,
                     const double *Tz,
                     double       *X,
                     double       *Y) const
    {
      rotate_z<B>(a.cos_phi.data(), a.sin_phi.data(), -1., X);
      for (unsigned int n = 0; n <= p; ++n)
        block_product<B, false>(R + rotation_start[n],
                                2 * n + 1,
                                X + n * n * B,
                                Y + n * n * B);
      apply_z_translation<B>(Tz, Y, X);
      for (unsigned int n = 0; n <= p; ++n)
        block_product<B, true>(R + rotation_start[n],
                               2 * n + 1,
                               X + n * n * B,
                               Y + n * n * B);
      rotate_z<B>(a.cos_phi.data(), a.sin_phi.data(), 1., Y);
    }

    /// The dense matrix of a translation, row by row, for the translations
    /// between the levels of a tree and for tests.
    void
    dense_translation(const Basis                 from,
                      const Basis                 to,
                      const double                kappa,
                      const dealii::Tensor<1, 3> &v,
                      const double                rho_from,
                      const double                rho_to,
                      double                     *T) const
    {
      std::vector<double> Tz(z_translation_size());
      z_translation(from, to, kappa, v.norm(), rho_from, rho_to, Tz.data());
      Angles a;
      angles(v, a);
      std::vector<double> X(nc), Y(nc), work(nc);
      for (unsigned int b = 0; b < nc; ++b)
        {
          std::fill(X.begin(), X.end(), 0.);
          std::fill(Y.begin(), Y.end(), 0.);
          X[b] = 1.;
          translate<1>(a, Tz.data(), X.data(), Y.data(), work.data());
          for (unsigned int r = 0; r < nc; ++r)
            T[r * nc + b] = Y[r];
        }
    }

  private:
    unsigned int
    gaunt_index(const unsigned int m,
                const unsigned int n,
                const unsigned int n2,
                const unsigned int l) const
    {
      return ((m * (p + 1) + n) * (p + 1) + n2) * (2 * p + 1) + l;
    }

    /// Y = A X, or with transpose Y = A^T X, for the s x s matrix A, row by
    /// row, and B expansions: rows of Y in groups whose sums fill the
    /// vector registers.
    template <unsigned int B, bool transpose>
    static void
    block_product(const double      *A,
                  const unsigned int s,
                  const double      *X,
                  double            *Y)
    {
      constexpr unsigned int G = B >= 16 ? 2 : 4;
      const unsigned int     row_stride    = transpose ? 1 : s;
      const unsigned int     column_stride = transpose ? s : 1;
      unsigned int           r             = 0;
      for (; r + G <= s; r += G)
        {
          double sums[G][B];
          for (unsigned int g = 0; g < G; ++g)
#pragma omp simd
            for (unsigned int j = 0; j < B; ++j)
              sums[g][j] = 0.;
          for (unsigned int c = 0; c < s; ++c)
            {
              const double *x = X + c * B;
              for (unsigned int g = 0; g < G; ++g)
                {
                  const double a =
                    A[(r + g) * row_stride + c * column_stride];
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    sums[g][j] += a * x[j];
                }
            }
          for (unsigned int g = 0; g < G; ++g)
            {
              double *y = Y + (r + g) * B;
#pragma omp simd
              for (unsigned int j = 0; j < B; ++j)
                y[j] = sums[g][j];
            }
        }
      for (; r < s; ++r)
        {
          double sums[B];
#pragma omp simd
          for (unsigned int j = 0; j < B; ++j)
            sums[j] = 0.;
          for (unsigned int c = 0; c < s; ++c)
            {
              const double  a = A[r * row_stride + c * column_stride];
              const double *x = X + c * B;
#pragma omp simd
              for (unsigned int j = 0; j < B; ++j)
                sums[j] += a * x[j];
            }
          double *y = Y + r * B;
#pragma omp simd
          for (unsigned int j = 0; j < B; ++j)
            y[j] = sums[j];
        }
    }

    /// X <- Z(sign alpha) X, a rotation by sign alpha about the z-axis.
    template <unsigned int B>
    void
    rotate_z(const double *c, const double *s, const double sign, double *X)
      const
    {
      for (unsigned int n = 1; n <= p; ++n)
        for (unsigned int m = 1; m <= n; ++m)
          {
            double      *x  = X + index(n, m) * B;
            double      *y  = X + index(n, -int(m)) * B;
            const double cm = c[m], sm = sign * s[m];
#pragma omp simd
            for (unsigned int j = 0; j < B; ++j)
              {
                const double u = x[j], w = y[j];
                x[j]           = cm * u - sm * w;
                y[j]           = sm * u + cm * w;
              }
          }
    }

    /// Y = D(P) X, row by row with the sums in registers, two per lane:
    /// accumulated in Y, each product would wait for the previous one.
    template <unsigned int B>
    void
    apply_swap(const double *X, double *Y) const
    {
      for (unsigned int b = 0; b < nc; ++b)
        {
          double s0[B], s1[B];
#pragma omp simd
          for (unsigned int j = 0; j < B; ++j)
            s0[j] = s1[j] = 0.;
          unsigned int       e   = swap_start[b];
          const unsigned int end = swap_start[b + 1];
          for (; e + 1 < end; e += 2)
            {
              const double  v0 = swap_value[e], v1 = swap_value[e + 1];
              const double *x0 = X + swap_column[e] * B;
              const double *x1 = X + swap_column[e + 1] * B;
#pragma omp simd
              for (unsigned int j = 0; j < B; ++j)
                {
                  s0[j] += v0 * x0[j];
                  s1[j] += v1 * x1[j];
                }
            }
          if (e < end)
            {
              const double  v0 = swap_value[e];
              const double *x0 = X + swap_column[e] * B;
#pragma omp simd
              for (unsigned int j = 0; j < B; ++j)
                s0[j] += v0 * x0[j];
            }
          double *y = Y + b * B;
#pragma omp simd
          for (unsigned int j = 0; j < B; ++j)
            y[j] = s0[j] + s1[j];
        }
    }

    /// Y = T_z X, two rows at a time with their sums in registers.
    template <unsigned int B>
    void
    apply_z_translation(const double *Tz, const double *X, double *Y) const
    {
      apply_z_blocks<B, false>(Tz, X, Y);
    }

    /// Y = T_z X with a translation for each expansion, entry e of
    /// expansion j at Tz[e * B + j].
    template <unsigned int B>
    void
    apply_z_translation_lanes(const double *Tz, const double *X, double *Y)
      const
    {
      apply_z_blocks<B, true>(Tz, X, Y);
    }

    /// The blocks of T_z, of order m, on the coefficients of orders m and
    /// -m; with lanes, an entry for each expansion.
    template <unsigned int B, bool lanes>
    void
    apply_z_blocks(const double *Tz, const double *X, double *Y) const
    {
      constexpr unsigned int L = lanes ? B : 1;
      for (unsigned int m = 0; m <= p; ++m)
        {
          const double      *block = Tz + block_start[m] * L;
          const unsigned int size  = p + 1 - m;
          for (int sign = (m == 0 ? 1 : -1); sign <= 1; sign += 2)
            {
              const int    order = sign * int(m);
              unsigned int n2    = m;
              for (; n2 + 1 <= p; n2 += 2)
                {
                  double s0[B], s1[B];
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    s0[j] = s1[j] = 0.;
                  const double *t0 = block + (n2 - m) * size * L;
                  const double *t1 = t0 + size * L;
                  for (unsigned int n = m; n <= p; ++n)
                    {
                      const double *x  = X + index(n, order) * B;
                      const double *a0 = t0 + (n - m) * L;
                      const double *a1 = t1 + (n - m) * L;
#pragma omp simd
                      for (unsigned int j = 0; j < B; ++j)
                        {
                          s0[j] += a0[lanes ? j : 0] * x[j];
                          s1[j] += a1[lanes ? j : 0] * x[j];
                        }
                    }
                  double *y0 = Y + index(n2, order) * B;
                  double *y1 = Y + index(n2 + 1, order) * B;
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    {
                      y0[j] = s0[j];
                      y1[j] = s1[j];
                    }
                }
              if (n2 <= p)
                {
                  double s0[B];
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    s0[j] = 0.;
                  const double *t0 = block + (n2 - m) * size * L;
                  for (unsigned int n = m; n <= p; ++n)
                    {
                      const double *x  = X + index(n, order) * B;
                      const double *a0 = t0 + (n - m) * L;
#pragma omp simd
                      for (unsigned int j = 0; j < B; ++j)
                        s0[j] += a0[lanes ? j : 0] * x[j];
                    }
                  double *y0 = Y + index(n2, order) * B;
#pragma omp simd
                  for (unsigned int j = 0; j < B; ++j)
                    y0[j] = s0[j];
                }
            }
        }
    }

    unsigned int              p, nc;
    std::vector<double>       norms;
    std::vector<double>       gaunt;
    std::vector<double>       axis_factor;
    /// D(P) by rows: the columns and values of row b from swap_start[b].
    std::vector<unsigned int> swap_start, swap_column;
    std::vector<double>       swap_value;
    std::vector<unsigned int> block_start;
    /// Start of the block of degree n of a rotation.
    std::vector<unsigned int> rotation_start;
    /// D(P) by blocks on the degrees, like a rotation.
    std::vector<double> swap_blocks;
    /// The highest order of rotation_blocks().
    static constexpr unsigned int maximum_rotation_order = 30;
    /// Radial functions of z_translation(), kept to avoid allocations; an
    /// object is used by one thread.
    mutable std::vector<double> scratch_f, scratch_from, scratch_to;
  };
} // namespace YukawaExpansions

#endif
