// ---------------------------------------------------------------------
//
// Accuracy and cost of LaplaceFMM on points of a sphere-box geometry.
//
// Points lie on the unit sphere and on the surface of the box
// [-2, 2]^2 x [-3.003, -1.003], whose top face is at 0.3% of the radius
// from the south pole of the sphere, as in the benchmark of the paper. A
// third of the points of each surface cluster near the contact, like the
// quadrature points of a graded mesh. Every point carries a random charge
// and a random dipole along the surface normal. The error, relative to the
// largest value, is measured against the direct sum at 500 random targets.
//
// Usage: mpirun -np P fmm_points [n_points max_points_per_leaf]
//        followed by pairs "order theta" (default: a small sweep)
//
// The screening parameter kappa of the kernel e^{-kappa r}/r is read from
// the environment variable FMM_KAPPA; without it the kernel is 1/r.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "laplace_fmm.h"

using namespace dealii;

namespace
{
  // Points and outer normals of the sphere-box surfaces.
  void
  sphere_box_points(const unsigned int         n,
                    std::vector<Point<3>>     &points,
                    std::vector<Tensor<1, 3>> &normals)
  {
    std::mt19937                           rng(1);
    std::uniform_real_distribution<double> u(0., 1.);
    std::normal_distribution<double>       g(0., 1.);
    const double                           top = -1.003;
    for (unsigned int i = 0; i < n / 2; ++i)
      {
        Tensor<1, 3> v;
        if (i % 3 == 0)
          {
            // near the south pole: polar angle from the pole ~ 0.5 u^2
            const double t = 0.5 * u(rng) * u(rng), f = 2 * M_PI * u(rng);
            v = Tensor<1, 3>(
              {std::sin(t) * std::cos(f), std::sin(t) * std::sin(f),
               -std::cos(t)});
          }
        else
          {
            v = Tensor<1, 3>({g(rng), g(rng), g(rng)});
            v /= v.norm();
          }
        points.emplace_back(v);
        normals.push_back(v);
      }
    // Box faces, by area: top and bottom 16 each, sides 8 each.
    const double area[6] = {16, 16, 8, 8, 8, 8};
    for (unsigned int i = n / 2; i < n; ++i)
      {
        if (i % 3 == 0)
          {
            // on the top face near the contact point
            const double r = 0.5 * u(rng) * u(rng), f = 2 * M_PI * u(rng);
            points.emplace_back(r * std::cos(f), r * std::sin(f), top);
            normals.push_back(Tensor<1, 3>({0, 0, 1}));
            continue;
          }
        double       s    = 64 * u(rng);
        unsigned int face = 0;
        while (s > area[face])
          s -= area[face++];
        const double a = 4 * u(rng) - 2, b = 4 * u(rng) - 2,
                     c = top - 2 * u(rng);
        switch (face)
          {
            case 0:
              points.emplace_back(a, b, top);
              normals.push_back(Tensor<1, 3>({0, 0, 1}));
              break;
            case 1:
              points.emplace_back(a, b, top - 2);
              normals.push_back(Tensor<1, 3>({0, 0, -1}));
              break;
            case 2:
              points.emplace_back(-2, a, c);
              normals.push_back(Tensor<1, 3>({-1, 0, 0}));
              break;
            case 3:
              points.emplace_back(2, a, c);
              normals.push_back(Tensor<1, 3>({1, 0, 0}));
              break;
            case 4:
              points.emplace_back(a, -2, c);
              normals.push_back(Tensor<1, 3>({0, -1, 0}));
              break;
            default:
              points.emplace_back(a, 2, c);
              normals.push_back(Tensor<1, 3>({0, 1, 0}));
          }
      }
  }
} // namespace

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const MPI_Comm                   comm = MPI_COMM_WORLD;
  const bool root = Utilities::MPI::this_mpi_process(comm) == 0;

  const char        *kappa_variable = std::getenv("FMM_KAPPA");
  const double       kappa = kappa_variable ? std::atof(kappa_variable) : 0.;
  const unsigned int n    = argc > 1 ? std::atoi(argv[1]) : 100000;
  const unsigned int leaf = argc > 2 ? std::atoi(argv[2]) : 64;
  std::vector<std::pair<unsigned int, double>> settings;
  for (int i = 3; i + 1 < argc; i += 2)
    settings.emplace_back(std::atoi(argv[i]), std::atof(argv[i + 1]));
  if (settings.empty())
    settings = {{6, 0.5}, {8, 0.5}, {10, 0.5}, {8, 0.4}, {8, 0.6}};

  std::vector<Point<3>>     points;
  std::vector<Tensor<1, 3>> normals;
  sphere_box_points(n, points, normals);
  std::mt19937                           rng(2);
  std::uniform_real_distribution<double> u(-1., 1.);
  std::vector<double>                    charges(n);
  std::vector<Tensor<1, 3>>              dipoles(n);
  for (unsigned int i = 0; i < n; ++i)
    {
      charges[i] = u(rng);
      dipoles[i] = u(rng) * normals[i];
    }
  std::vector<unsigned int> targets(500);
  std::uniform_int_distribution<unsigned int> pick(0, n - 1);
  for (auto &t : targets)
    t = pick(rng);

  for (const auto &[order, theta] : settings)
    {
      LaplaceFMM                 fmm(comm);
      LaplaceFMM::AdditionalData data;
      data.order               = order;
      data.theta               = theta;
      data.max_points_per_leaf = leaf;
      data.kappa               = kappa;
      fmm.reinit(points, data);

      std::vector<double> potentials;
      const auto          start = std::chrono::steady_clock::now();
      fmm.evaluate(charges, dipoles, potentials);
      const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      start)
          .count();
      fmm.print_statistics(std::cout);

      for (const bool with_dipoles : {false, true})
        {
          // charges only, then charges and dipoles
          std::vector<double> u_fmm;
          const std::vector<Tensor<1, 3>> none;
          if (!with_dipoles)
            fmm.evaluate(charges, none, u_fmm);
          else
            u_fmm = potentials;
          if (root)
            {
              const std::vector<double> exact = fmm.evaluate_directly(
                charges, with_dipoles ? dipoles : none, targets);
              double error = 0, scale = 0;
              for (unsigned int t = 0; t < targets.size(); ++t)
                {
                  error = std::max(error, std::abs(u_fmm[targets[t]] - exact[t]));
                  scale = std::max(scale, std::abs(exact[t]));
                }
              std::cout << "  relative error, "
                        << (with_dipoles ? "charges and dipoles" :
                                           "charges only")
                        << ": " << std::scientific << std::setprecision(2)
                        << error / scale << std::defaultfloat << '\n';
            }
        }
      const auto again = std::chrono::steady_clock::now();
      fmm.evaluate(charges, dipoles, potentials);
      const double seconds_again =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      again)
          .count();
      if (root)
        std::cout << "  first evaluation " << std::setprecision(3) << seconds
                  << " s, after rebalancing " << seconds_again << " s\n\n";
    }
}
