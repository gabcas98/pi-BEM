// ---------------------------------------------------------------------
//
// Accuracy and cost of the multipole and local expansions of the FMA.
//
// Charges and dipoles are placed at random in a leaf box of side h, and
// the potential they induce is evaluated at random points of a target
// leaf box, directly and through the chain of the FMA:
//
//   P2M (leaf) -> M2M (parent) -> M2L (parent to parent) -> L2L (leaf) -> L2P
//
// for the offsets between the parent boxes that the interaction lists of
// an octree produce: the nearest well-separated box (2, 0, 0), the
// farthest corner of the list (3, 3, 3), and one in between. The error
// is relative to the largest direct value. Two implementations are
// compared: the classes MultipoleExpansion and LocalExpansion of the
// original FMA ("old") and the solid harmonics of laplace_expansions.h
// ("new"). The time of each operation is measured on its own, in
// microseconds, for every truncation order, and last the M2L translation as
// LaplaceFMM applies it, grouped by offset in batches of eight.
//
// Usage: fmm_expansions [n_sources n_targets max_order]
//
// ---------------------------------------------------------------------

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "ass_leg_function.h"
#include "laplace_expansions.h"
#include "local_expansion.h"
#include "multipole_expansion.h"
#include "yukawa_expansions.h"

using namespace dealii;
namespace LE = LaplaceExpansions;

namespace
{
  // The original classes are limited by their precomputed tables.
  constexpr unsigned int max_old_order = 10;

  struct Sources
  {
    std::vector<Point<3>>     points;
    std::vector<double>       strengths;
    std::vector<Tensor<1, 3>> normals;
  };

  // Random points in the cube of side h centred at c.
  std::vector<Point<3>>
  random_points(const Point<3>    &c,
                const double       h,
                const unsigned int n,
                std::mt19937      &rng)
  {
    std::uniform_real_distribution<double> u(-0.5 * h, 0.5 * h);
    std::vector<Point<3>>                  points(n);
    for (auto &p : points)
      p = c + Point<3>(u(rng), u(rng), u(rng));
    return points;
  }

  Sources
  random_sources(const Point<3>    &c,
                 const double       h,
                 const unsigned int n,
                 std::mt19937      &rng)
  {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Sources                                s;
    s.points = random_points(c, h, n, rng);
    s.strengths.resize(n);
    s.normals.resize(n);
    for (unsigned int i = 0; i < n; ++i)
      {
        s.strengths[i] = u(rng);
        Tensor<1, 3> v({u(rng), u(rng), u(rng)});
        s.normals[i] = v / v.norm();
      }
    return s;
  }

  // Kernels of the expansions: 1/|x - y| for charges, and its normal
  // derivative in y, (x - y).n / |x - y|^3, for dipoles, as in the tests of
  // the expansions.
  double
  direct(const Sources &s, const Point<3> &x, const bool dipoles)
  {
    double u = 0;
    for (unsigned int i = 0; i < s.points.size(); ++i)
      {
        const Tensor<1, 3> r = x - s.points[i];
        const double       d = r.norm();
        u += dipoles ? s.strengths[i] * (r * s.normals[i]) / (d * d * d) :
                       s.strengths[i] / d;
      }
    return u;
  }

  template <typename F>
  double
  microseconds(F &&f, const unsigned int repetitions)
  {
    const auto start = std::chrono::steady_clock::now();
    for (unsigned int r = 0; r < repetitions; ++r)
      f();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(stop - start).count() /
           repetitions;
  }

  struct Boxes
  {
    Point<3> source_leaf, source_parent, target_parent, target_leaf;
  };

  // The whole chain with the original classes; potentials at the targets.
  std::vector<double>
  chain_old(const unsigned int           p,
            const Boxes                 &b,
            const Sources               &s,
            const bool                   dipoles,
            const std::vector<Point<3>> &targets,
            const AssLegFunction        &legendre)
  {
    MultipoleExpansion leaf(p, b.source_leaf, &legendre);
    for (unsigned int i = 0; i < s.points.size(); ++i)
      if (dipoles)
        leaf.AddNormDer(s.strengths[i], s.points[i], s.normals[i]);
      else
        leaf.Add(s.strengths[i], s.points[i]);
    MultipoleExpansion parent(p, b.source_parent, &legendre);
    parent.Add(leaf);
    LocalExpansion parent_local(p, b.target_parent, &legendre);
    parent_local.Add(parent);
    LocalExpansion local(p, b.target_leaf, &legendre);
    local.Add(parent_local);
    std::vector<double> u;
    for (const auto &x : targets)
      u.push_back(local.Evaluate(x));
    return u;
  }

  // The whole chain with the solid harmonics.
  std::vector<double>
  chain_new(const unsigned int           p,
            const Boxes                 &b,
            const Sources               &s,
            const bool                   dipoles,
            const std::vector<Point<3>> &targets)
  {
    LE::Workspace            w(p);
    const unsigned int       n = LE::n_coefficients(p);
    std::vector<LE::complex> leaf(n), parent(n), parent_local(n), local(n);
    for (unsigned int i = 0; i < s.points.size(); ++i)
      if (dipoles)
        LE::add_dipole(w,
                       s.strengths[i] * s.normals[i],
                       s.points[i] - b.source_leaf,
                       leaf.data());
      else
        LE::add_charge(w,
                       s.strengths[i],
                       s.points[i] - b.source_leaf,
                       leaf.data());
    LE::multipole_to_multipole(w,
                               leaf.data(),
                               b.source_leaf - b.source_parent,
                               parent.data());
    LE::multipole_to_local(w,
                           parent.data(),
                           b.target_parent - b.source_parent,
                           parent_local.data());
    LE::local_to_local(w,
                       parent_local.data(),
                       b.target_leaf - b.target_parent,
                       local.data());
    std::vector<double> u;
    for (const auto &x : targets)
      u.push_back(LE::evaluate_local(w, local.data(), x - b.target_leaf));
    return u;
  }
} // namespace

int
main(int argc, char **argv)
{
  const unsigned int n_sources = argc > 1 ? std::atoi(argv[1]) : 200;
  const unsigned int n_targets = argc > 2 ? std::atoi(argv[2]) : 200;
  const unsigned int max_order = argc > 3 ? std::atoi(argv[3]) : 16;
  const double       h         = 1.0; // side of a leaf box

  AssLegFunction legendre;
  std::mt19937   rng(42);

  // Source leaf: first child of the parent box centred at the origin.
  Boxes b;
  b.source_parent       = Point<3>(0, 0, 0);
  b.source_leaf         = b.source_parent + Point<3>(-h / 2, -h / 2, -h / 2);
  const Sources sources = random_sources(b.source_leaf, h, n_sources, rng);

  // Offsets between parent boxes (side 2h), in units of the parent side.
  const std::vector<Point<3>> offsets = {Point<3>(2, 0, 0),
                                         Point<3>(2, 2, 1),
                                         Point<3>(3, 3, 3)};

  std::cout << "Expansions of the FMA against the direct sum, " << n_sources
            << " sources in a leaf of side " << h << ", " << n_targets
            << " targets in the leaf of a well-separated parent box\n\n"
            << "Error: max |FMA - direct| / max |direct| over the targets, "
               "chain P2M-M2M-M2L-L2L-L2P; old / new\n"
            << std::setw(3) << "p";
  for (const auto &o : offsets)
    for (const std::string kind : {"charges", "dipoles"})
      std::cout << std::setw(13) << kind << " (" << o[0] << "," << o[1] << ","
                << o[2] << ")";
  std::cout << '\n';

  for (unsigned int p = 2; p <= max_order; ++p)
    {
      std::cout << std::setw(3) << p;
      for (const auto &o : offsets)
        {
          b.target_parent = b.source_parent + 2 * h * o;
          b.target_leaf   = b.target_parent + Point<3>(h / 2, h / 2, h / 2);
          const std::vector<Point<3>> targets =
            random_points(b.target_leaf, h, n_targets, rng);
          for (const bool dipoles : {false, true})
            {
              std::vector<double> exact;
              double              scale = 0;
              for (const auto &x : targets)
                {
                  exact.push_back(direct(sources, x, dipoles));
                  scale = std::max(scale, std::abs(exact.back()));
                }
              auto error = [&](const std::vector<double> &u) {
                double e = 0;
                for (unsigned int i = 0; i < u.size(); ++i)
                  e = std::max(e, std::abs(u[i] - exact[i]));
                return e / scale;
              };
              std::cout << std::scientific << std::setprecision(1);
              if (p <= max_old_order)
                std::cout << std::setw(10)
                          << error(chain_old(
                               p, b, sources, dipoles, targets, legendre));
              else
                std::cout << std::setw(10) << "-";
              std::cout << std::setw(10)
                        << error(chain_new(p, b, sources, dipoles, targets));
            }
        }
      std::cout << '\n';
    }

  std::cout << "\nTime per operation [microseconds], old / new\n"
            << std::setw(3) << "p";
  for (const std::string op :
       {"P2M/point", "P2Mdip/pt", "M2M", "M2L", "L2L", "L2P/point"})
    std::cout << std::setw(20) << op;
  std::cout << '\n';
  b.target_parent = b.source_parent + 2 * h * offsets[0];
  b.target_leaf   = b.target_parent + Point<3>(h / 2, h / 2, h / 2);
  const std::vector<Point<3>> targets =
    random_points(b.target_leaf, h, n_targets, rng);
  double sink = 0; // keeps the compiler from dropping the evaluations
  for (unsigned int p = 2; p <= max_order; ++p)
    {
      const unsigned int repetitions = 4000 / (p * p) + 5;
      const unsigned int np          = sources.points.size();
      std::vector<double> t_old(6, 0.), t_new(6, 0.);

      if (p <= max_old_order)
        {
          MultipoleExpansion leaf(p, b.source_leaf, &legendre);
          t_old[0] = microseconds(
                       [&]() {
                         for (unsigned int i = 0; i < np; ++i)
                           leaf.Add(sources.strengths[i], sources.points[i]);
                       },
                       repetitions) /
                     np;
          MultipoleExpansion dip(p, b.source_leaf, &legendre);
          t_old[1] = microseconds(
                       [&]() {
                         for (unsigned int i = 0; i < np; ++i)
                           dip.AddNormDer(sources.strengths[i],
                                          sources.points[i],
                                          sources.normals[i]);
                       },
                       repetitions) /
                     np;
          MultipoleExpansion parent(p, b.source_parent, &legendre);
          t_old[2] =
            microseconds([&]() { parent.Add(leaf); }, 20 * repetitions);
          LocalExpansion parent_local(p, b.target_parent, &legendre);
          t_old[3] =
            microseconds([&]() { parent_local.Add(parent); }, repetitions);
          LocalExpansion local(p, b.target_leaf, &legendre);
          t_old[4] =
            microseconds([&]() { local.Add(parent_local); }, repetitions);
          t_old[5] = microseconds(
                       [&]() {
                         for (const auto &x : targets)
                           sink += local.Evaluate(x);
                       },
                       repetitions) /
                     targets.size();
        }

      {
        LE::Workspace            w(p);
        const unsigned int       n = LE::n_coefficients(p);
        std::vector<LE::complex> leaf(n), dip(n), parent(n), parent_local(n),
          local(n);
        t_new[0] = microseconds(
                     [&]() {
                       for (unsigned int i = 0; i < np; ++i)
                         LE::add_charge(w,
                                        sources.strengths[i],
                                        sources.points[i] - b.source_leaf,
                                        leaf.data());
                     },
                     repetitions) /
                   np;
        t_new[1] = microseconds(
                     [&]() {
                       for (unsigned int i = 0; i < np; ++i)
                         LE::add_dipole(w,
                                        sources.strengths[i] *
                                          sources.normals[i],
                                        sources.points[i] - b.source_leaf,
                                        dip.data());
                     },
                     repetitions) /
                   np;
        t_new[2] = microseconds(
          [&]() {
            LE::multipole_to_multipole(w,
                                       leaf.data(),
                                       b.source_leaf - b.source_parent,
                                       parent.data());
          },
          20 * repetitions);
        t_new[3] = microseconds(
          [&]() {
            LE::multipole_to_local(w,
                                   parent.data(),
                                   b.target_parent - b.source_parent,
                                   parent_local.data());
          },
          repetitions);
        t_new[4] = microseconds(
          [&]() {
            LE::local_to_local(w,
                               parent_local.data(),
                               b.target_leaf - b.target_parent,
                               local.data());
          },
          repetitions);
        t_new[5] = microseconds(
                     [&]() {
                       for (const auto &x : targets)
                         sink += LE::evaluate_local(w,
                                                    local.data(),
                                                    x - b.target_leaf);
                     },
                     repetitions) /
                   targets.size();
      }

      std::cout << std::setw(3) << p << std::fixed;
      for (unsigned int i = 0; i < 6; ++i)
        {
          std::cout << std::setprecision(3);
          if (p <= max_old_order)
            std::cout << std::setw(10) << t_old[i];
          else
            std::cout << std::setw(10) << "-";
          std::cout << std::setw(10) << t_new[i];
        }
      std::cout << '\n';
    }
  // M2L by rotation onto the z-axis in the real harmonics of
  // yukawa_expansions.h, against the direct formula: the multipole of the
  // sources converted to real coefficients, rotated, translated along z
  // and rotated back, and the local expansion converted back.
  std::cout << "\nM2L by rotation against the direct formula, largest "
               "relative difference of the local coefficients\n";
  for (unsigned int p = 4; p <= std::min(max_order, 16u); p += 4)
    {
      namespace YE = YukawaExpansions;
      LE::Workspace           w(p);
      const LE::RealBasis     basis(p);
      const YE::Operators     operators(p);
      const unsigned int      nc = LE::n_coefficients(p);
      std::vector<LE::complex> M(nc, 0.), L_direct(nc, 0.), L_rotated(nc, 0.);
      std::uniform_real_distribution<double> u(-0.5, 0.5);
      for (unsigned int i = 0; i < 30; ++i)
        LE::add_charge(w,
                       u(rng),
                       Tensor<1, 3>({u(rng), u(rng), u(rng)}),
                       M.data());
      double worst = 0;
      for (const Tensor<1, 3> zL_minus_zM :
           {Tensor<1, 3>({4., 2., 2.}), Tensor<1, 3>({-3., 0., 0.}),
            Tensor<1, 3>({0., 0., -5.}), Tensor<1, 3>({1., -6., 3.})})
        {
          std::fill(L_direct.begin(), L_direct.end(), 0.);
          LE::multipole_to_local(w, M.data(), zL_minus_zM, L_direct.data());
          const unsigned int  nr = YE::n_coefficients(p);
          std::vector<double> X(nr), Y(nr, 0.), work(nr),
            Tz(operators.z_translation_size());
          for (unsigned int n = 0; n <= p; ++n)
            for (unsigned int m = 0; m <= n; ++m)
              {
                const LE::complex c = M[LE::index(n, m)];
                const double      f = basis.outgoing[LE::index(n, m)];
                X[YE::index(n, m)]  = f * c.real();
                if (m > 0)
                  X[YE::index(n, -int(m))] = -f * c.imag();
              }
          operators.z_translation_laplace(zL_minus_zM.norm(), Tz.data());
          YE::Operators::Angles angles;
          operators.angles(-zL_minus_zM, angles);
          operators.translate<1>(
            angles, Tz.data(), X.data(), Y.data(), work.data());
          double largest = 0, difference = 0;
          for (unsigned int n = 0; n <= p; ++n)
            for (unsigned int m = 0; m <= n; ++m)
              {
                const double f = basis.local[LE::index(n, m)];
                L_rotated[LE::index(n, m)] =
                  m == 0 ? LE::complex(f * Y[YE::index(n, 0)], 0.) :
                           f * LE::complex(Y[YE::index(n, m)],
                                           Y[YE::index(n, -int(m))]);
                largest = std::max(largest, std::abs(L_direct[LE::index(n, m)]));
                difference =
                  std::max(difference,
                           std::abs(L_rotated[LE::index(n, m)] -
                                    L_direct[LE::index(n, m)]));
              }
          worst = std::max(worst, difference / largest);
        }
      std::cout << "  p=" << p << ": " << std::scientific
                << std::setprecision(2) << worst << std::defaultfloat << '\n';
    }

  // M2L as LaplaceFMM applies it: one operator per offset, built once and
  // applied to batches of eight expansions stored coefficient by
  // coefficient; the time per translation, without the operator.
  std::cout << "\nM2L grouped by offset, batches of 8 [microseconds per "
               "translation]\n";
  for (unsigned int p = 4; p <= std::min(max_order, 12u); p += 2)
    {
      constexpr unsigned int B       = 8;
      LE::Workspace          w(p);
      const unsigned int     rows    = LE::n_coefficients(p);
      const unsigned int     columns = (p + 1) * (p + 1);
      std::vector<double>    T_re(rows * columns), T_im(rows * columns);
      std::vector<double>    X_re(columns * B), X_im(columns * B),
        Y_re(rows * B), Y_im(rows * B);
      std::uniform_real_distribution<double> u(-1., 1.);
      for (unsigned int i = 0; i < columns * B; ++i)
        {
          X_re[i] = u(rng);
          X_im[i] = u(rng);
        }
      LE::multipole_to_local_operator(
        w, Tensor<1, 3>({4., 2., 2.}), T_re.data(), T_im.data());
      const unsigned int n     = 20000;
      const auto         start = std::chrono::steady_clock::now();
      for (unsigned int k = 0; k < n; ++k)
        {
          X_re[k % (columns * B)] += 1e-9;
          LE::multipole_to_local_batch<B>(p,
                                          T_re.data(),
                                          T_im.data(),
                                          X_re.data(),
                                          X_im.data(),
                                          Y_re.data(),
                                          Y_im.data());
          sink += Y_re[k % (rows * B)];
        }
      const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      start)
          .count();
      std::cout << "  p=" << p << ": " << std::fixed << std::setprecision(3)
                << 1e6 * seconds / (double(n) * B) << '\n';
    }
  std::cout << "\n(checksum " << std::scientific << sink << ")\n";
}
