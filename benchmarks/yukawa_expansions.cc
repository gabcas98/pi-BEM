// ---------------------------------------------------------------------
//
// Accuracy and cost of the expansions of the screened kernel
// e^{-kappa r}/r in yukawa_expansions.h.
//
// 1. The modified spherical Bessel functions against the Wronskian
//    i_n k_{n+1} + i_{n+1} k_n = 1/z^2.
// 2. The addition theorem of the kernel.
// 3. The analytic translations of the class Operators against those by
//    projection, for M2M, M2L and L2L: largest entry of the difference over
//    the largest entry.
// 4. The chain P2M (leaf) -> M2M (parent) -> M2L -> L2L (leaf) -> L2P for
//    charges and dipoles in a leaf of side 1 against the direct sum, for
//    several kappa and orders, with the targets in a leaf of the parent box
//    at offset (2, 2, 1) or (3, 2, 1.5) parent sides. The error is relative
//    to the largest value of the field of the same sources without
//    screening (sum |q|/r, or sum |d|/r^2): the screened far field decays
//    like e^{-kappa d}, and what matters is its error against the size of
//    the whole field, of which the near sources, barely screened, make up
//    most. In parentheses, the error of the outgoing expansion of the leaf
//    evaluated directly (M2P).
// 5. The time of the translations: building T_z, applying the rotated
//    translation to a batch of 8 expansions, and a dense matrix.
//
// ---------------------------------------------------------------------

#include "yukawa_expansions.h"

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace dealii;
namespace YE = YukawaExpansions;

namespace
{
  double
  seconds_since(const std::chrono::steady_clock::time_point &start)
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
      .count();
  }

  void
  apply(const std::vector<double> &T,
        const std::vector<double> &from,
        std::vector<double>       &to)
  {
    const unsigned int nc = from.size();
    for (unsigned int a = 0; a < nc; ++a)
      {
        double s = 0;
        for (unsigned int b = 0; b < nc; ++b)
          s += T[a * nc + b] * from[b];
        to[a] = s;
      }
  }
} // namespace

int
main()
{
  std::cout << std::scientific << std::setprecision(2);

  // 1. Wronskian
  {
    double worst = 0;
    for (const double z : {1e-3, 0.1, 0.49, 0.51, 1., 5., 20., 80.})
      {
        const unsigned int  N = 30;
        std::vector<double> i(N + 1), k(N + 1);
        YE::bessel_i<double>(N, z, i.data());
        YE::bessel_k<double>(N, z, k.data());
        for (unsigned int n = 0; n < N; ++n)
          worst = std::max(
            worst, std::abs((i[n] * k[n + 1] + i[n + 1] * k[n]) * z * z - 1.));
      }
    std::cout << "Wronskian, largest relative error: " << worst << std::endl;
  }

  // 2. Addition theorem
  for (const double kappa : {0.01, 1., 10.})
    {
      const unsigned int  p = 40;
      const Tensor<1, 3>  x({1.3, -0.4, 0.9}), y({0.2, 0.3, -0.25});
      const unsigned int  nc = YE::n_coefficients(p);
      std::vector<double> Yx(nc), Yy(nc), i(p + 1), k(p + 1);
      YE::real_harmonics<double>(p, x[0], x[1], x[2], Yx.data());
      YE::real_harmonics<double>(p, y[0], y[1], y[2], Yy.data());
      YE::bessel_i<double>(p, kappa * y.norm(), i.data());
      YE::bessel_k<double>(p, kappa * x.norm(), k.data());
      double sum = 0;
      for (unsigned int n = 0; n <= p; ++n)
        for (int m = -int(n); m <= int(n); ++m)
          sum += 4 * M_PI * kappa * i[n] * k[n] * Yx[YE::index(n, m)] *
                 Yy[YE::index(n, m)];
      const double r = (x - y).norm();
      std::cout << "addition theorem, kappa " << kappa << ", relative error: "
                << std::abs(sum / (std::exp(-kappa * r) / r) - 1.) << std::endl;
    }

  const double rho_leaf = std::sqrt(3.) / 2, rho_parent = std::sqrt(3.);

  // 3. Analytic against projected translations
  for (const unsigned int p : {6u, 10u})
    {
      const unsigned int   nc = YE::n_coefficients(p);
      const YE::Operators  operators(p);
      const YE::SphereRule rule = YE::sphere_rule(p, 12);
      std::vector<double>  T(nc * nc), R(nc * nc);
      const auto           compare = [&]() {
        double difference = 0, largest = 0;
        for (unsigned int i = 0; i < nc * nc; ++i)
          {
            difference = std::max(difference, std::abs(T[i] - R[i]));
            largest    = std::max(largest, std::abs(R[i]));
          }
        return difference / largest;
      };
      for (const double kappa : {0.01, 1., 5.})
        {
          const Tensor<1, 3> half({0.5, -0.5, 0.5});
          const Tensor<1, 3> far({4., -2., 3.});
          // M2M from a child at c_parent - half to the parent
          operators.dense_translation(YE::Basis::outgoing,
                                      YE::Basis::outgoing,
                                      kappa,
                                      -half,
                                      rho_leaf,
                                      rho_parent,
                                      T.data());
          YE::projected_translation(p,
                                    kappa,
                                    rule,
                                    YE::Basis::outgoing,
                                    rho_leaf,
                                    YE::Basis::outgoing,
                                    rho_parent,
                                    half,
                                    3 * rho_parent,
                                    R.data());
          const double m2m = compare();
          // M2L from a box at c_target + far
          operators.dense_translation(YE::Basis::outgoing,
                                      YE::Basis::local,
                                      kappa,
                                      far,
                                      rho_parent,
                                      rho_parent,
                                      T.data());
          YE::projected_translation(p,
                                    kappa,
                                    rule,
                                    YE::Basis::outgoing,
                                    rho_parent,
                                    YE::Basis::local,
                                    rho_parent,
                                    -far,
                                    rho_parent,
                                    R.data());
          const double m2l = compare();
          // L2L from the parent at c_child - half to the child
          operators.dense_translation(YE::Basis::local,
                                      YE::Basis::local,
                                      kappa,
                                      -half,
                                      rho_parent,
                                      rho_leaf,
                                      T.data());
          YE::projected_translation(p,
                                    kappa,
                                    rule,
                                    YE::Basis::local,
                                    rho_parent,
                                    YE::Basis::local,
                                    rho_leaf,
                                    half,
                                    rho_leaf,
                                    R.data());
          const double l2l = compare();
          std::cout << "analytic vs projected, p=" << p << ", kappa " << kappa
                    << ": M2M " << m2m << ", M2L " << m2l << ", L2L " << l2l
                    << std::endl;
        }
    }

  // 4. Chain
  std::mt19937                           rng(7);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  const Point<3>                         source_parent(0, 0, 0);
  const Point<3>                         source_leaf(-0.5, -0.5, -0.5);
  std::vector<Point<3>>                  y(50);
  std::vector<double>                    q(y.size());
  std::vector<Tensor<1, 3>>              d(y.size());
  for (unsigned int j = 0; j < y.size(); ++j)
    {
      y[j] = source_leaf + Point<3>(u(rng), u(rng), u(rng));
      q[j] = 2 * u(rng);
      d[j] = Tensor<1, 3>({u(rng), u(rng), u(rng)});
    }

  for (const Tensor<1, 3> offset :
       {Tensor<1, 3>({4., 4., 2.}), Tensor<1, 3>({6., 4., 3.})})
    {
      const Point<3> target_parent = source_parent + offset;
      const Point<3> target_leaf   = target_parent + Point<3>(0.5, 0.5, 0.5);
      std::vector<Point<3>> x(20);
      for (auto &t : x)
        t = target_leaf + Point<3>(u(rng), u(rng), u(rng));
      std::cout << "offset " << offset << std::endl;
      for (const double kappa : {1e-3, 0.3, 1., 3., 10.})
        for (const bool dipoles : {false, true})
          {
            std::cout << "  kappa " << kappa
                      << (dipoles ? " dipoles " : " charges ");
            for (const unsigned int p : {4u, 8u, 12u, 16u})
              {
                const unsigned int  nc = YE::n_coefficients(p);
                const YE::Operators operators(p);
                YE::Workspace       w(p);
                std::vector<double> M(nc), Mp(nc), Lp(nc), L(nc);
                std::vector<double> T(nc * nc);
                for (unsigned int j = 0; j < y.size(); ++j)
                  if (dipoles)
                    YE::add_dipole(
                      w, kappa, rho_leaf, d[j], y[j] - source_leaf, M.data());
                  else
                    YE::add_charge(
                      w, kappa, rho_leaf, q[j], y[j] - source_leaf, M.data());
                operators.dense_translation(YE::Basis::outgoing,
                                            YE::Basis::outgoing,
                                            kappa,
                                            source_leaf - source_parent,
                                            rho_leaf,
                                            rho_parent,
                                            T.data());
                apply(T, M, Mp);
                operators.dense_translation(YE::Basis::outgoing,
                                            YE::Basis::local,
                                            kappa,
                                            source_parent - target_parent,
                                            rho_parent,
                                            rho_parent,
                                            T.data());
                apply(T, Mp, Lp);
                operators.dense_translation(YE::Basis::local,
                                            YE::Basis::local,
                                            kappa,
                                            target_parent - target_leaf,
                                            rho_parent,
                                            rho_leaf,
                                            T.data());
                apply(T, Lp, L);

                double error = 0, error_m2p = 0, scale = 0;
                for (const auto &t : x)
                  {
                    double direct = 0, laplace = 0;
                    for (unsigned int j = 0; j < y.size(); ++j)
                      {
                        const Tensor<1, 3> r  = t - y[j];
                        const double       rn = r.norm();
                        const double       g  = std::exp(-kappa * rn) / rn;
                        laplace += dipoles ? d[j].norm() / (rn * rn) :
                                             std::abs(q[j]) / rn;
                        direct += dipoles ? g * (1 + kappa * rn) / (rn * rn) *
                                              (d[j] * r) :
                                            q[j] * g;
                      }
                    const double fmm = YE::evaluate_local(
                      w, kappa, rho_leaf, L.data(), t - target_leaf);
                    const double m2p = YE::evaluate_outgoing(
                      w, kappa, rho_leaf, M.data(), t - source_leaf);
                    scale     = std::max(scale, laplace);
                    error     = std::max(error, std::abs(fmm - direct));
                    error_m2p = std::max(error_m2p, std::abs(m2p - direct));
                  }
                std::cout << " p=" << p << ": " << error / scale << " ("
                          << error_m2p / scale << ")";
              }
            std::cout << std::endl;
          }
    }

  // 5. Cost of the translations
  for (const unsigned int p : {8u, 10u, 12u})
    {
      const unsigned int     nc = YE::n_coefficients(p);
      const YE::Operators    operators(p);
      std::vector<double>    Tz(operators.z_translation_size()), T(nc * nc);
      constexpr unsigned int B = 8;
      std::vector<double>    X(nc * B, 1.), Y(nc * B, 0.), work(nc * B);
      YE::Operators::Angles  angles;
      const unsigned int     n = 2000;

      auto start = std::chrono::steady_clock::now();
      for (unsigned int k = 0; k < n; ++k)
        operators.z_translation(YE::Basis::outgoing,
                                YE::Basis::local,
                                1.,
                                5. + 1e-3 * k,
                                rho_parent,
                                rho_parent,
                                Tz.data());
      const double t_z = seconds_since(start) / n;

      operators.angles(Tensor<1, 3>({4., 2., 3.}), angles);
      start = std::chrono::steady_clock::now();
      for (unsigned int k = 0; k < n; ++k)
        {
          X[k % (nc * B)] = 1. + 1e-3 * k;
          operators.translate<B>(
            angles, Tz.data(), X.data(), Y.data(), work.data());
        }
      const double t_batch = seconds_since(start) / n;

      start = std::chrono::steady_clock::now();
      for (unsigned int k = 0; k < 20; ++k)
        operators.dense_translation(YE::Basis::outgoing,
                                    YE::Basis::local,
                                    1.,
                                    Tensor<1, 3>({4., 2. + k, 3.}),
                                    rho_parent,
                                    rho_parent,
                                    T.data());
      const double t_dense = seconds_since(start) / 20;
      std::cout << "p=" << p << ": T_z " << 1e6 * t_z
                << " us, rotated translation per expansion "
                << 1e6 * t_batch / B << " us (" << Y[0] << "), dense matrix "
                << 1e6 * t_dense << " us" << std::endl;
    }
}
