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

// LaplaceFMM on points of two nearby spheres against the direct sum, for
// charges, dipoles and both, at two truncation orders, for the Laplace
// kernel and for the screened one with kappa = 3 and 30: the boxes then
// span up to about 2 and 20 screening lengths, and with kappa = 30 the
// pairs of boxes more than 4/3 apart are dropped.

#include "laplace_fmm.h"

#include <random>

#include "tests.h"

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();

  // Points on two unit spheres 0.05 apart, with outer normals.
  std::mt19937                     rng(3);
  std::normal_distribution<double> g(0., 1.);
  std::vector<Point<3>>            points;
  std::vector<Tensor<1, 3>>        normals;
  for (unsigned int i = 0; i < 6000; ++i)
    {
      Tensor<1, 3> v({g(rng), g(rng), g(rng)});
      v /= v.norm();
      points.push_back(Point<3>(v) + Point<3>(0, 0, i % 2 ? 1.025 : -1.025));
      normals.push_back(v);
    }
  std::uniform_real_distribution<double> u(-1., 1.);
  std::vector<double>                    q(points.size());
  std::vector<Tensor<1, 3>>              d(points.size());
  for (unsigned int i = 0; i < points.size(); ++i)
    {
      q[i] = u(rng);
      d[i] = u(rng) * normals[i];
    }
  std::vector<unsigned int> targets;
  for (unsigned int i = 0; i < points.size(); i += 37)
    targets.push_back(i);

  for (const double kappa : {0., 3., 30.})
    for (const unsigned int order : {6u, 12u})
      {
        LaplaceFMM                 fmm;
        LaplaceFMM::AdditionalData data;
        data.order               = order;
        data.theta               = 0.5;
        data.max_points_per_leaf = 32;
        data.kappa               = kappa;
        fmm.reinit(points, data);
        for (const unsigned int kind : {0u, 1u, 2u})
          {
            const std::vector<Tensor<1, 3>> no_d;
            const std::vector<double>       no_q;
            const auto                     &charges = kind == 1 ? no_q : q;
            const auto                     &dipoles = kind == 0 ? no_d : d;
            std::vector<double>             potentials;
            fmm.evaluate(charges, dipoles, potentials);
            const std::vector<double> exact =
              fmm.evaluate_directly(charges, dipoles, targets);
            double error = 0, scale = 0;
            for (unsigned int t = 0; t < targets.size(); ++t)
              {
                error =
                  std::max(error, std::abs(potentials[targets[t]] - exact[t]));
                scale = std::max(scale, std::abs(exact[t]));
              }
            // Loose bounds: charges converge like theta^(p+1), dipoles one
            // order slower; a wrong translation gives an error of order one.
            const double bound = std::pow(0.5, order - (kind == 0 ? 0. : 1.));
            deallog << "kappa " << kappa << ", order " << order << ", "
                    << (kind == 0 ? "charges" :
                        kind == 1 ? "dipoles" :
                                    "charges and dipoles")
                    << ": relative error below " << bound << ": "
                    << (error / scale < bound ? "yes" : "no") << std::endl;
          }
        if (kappa > 0)
          continue;
        // The gradient of the potential of the charges, against the direct
        // sum of -q_j (x_i - x_j) / |x_i - x_j|^3; one order slower.
        std::vector<double>       potentials, plain;
        std::vector<Tensor<1, 3>> gradients;
        fmm.evaluate_with_gradient(q, potentials, gradients);
        fmm.evaluate(q, {}, plain);
        double error = 0, scale = 0, difference = 0;
        for (const unsigned int i : targets)
          {
            Tensor<1, 3> exact;
            for (unsigned int j = 0; j < points.size(); ++j)
              if (j != i)
                {
                  const Tensor<1, 3> r = points[i] - points[j];
                  exact -= q[j] * r / std::pow(r.norm(), 3);
                }
            error = std::max(error, (gradients[i] - exact).norm());
            scale = std::max(scale, exact.norm());
            difference =
              std::max(difference, std::abs(potentials[i] - plain[i]));
          }
        const double bound = std::pow(0.5, order - 1.);
        deallog << "kappa 0, order " << order
                << ", gradient of the charges: relative error below " << bound
                << ": " << (error / scale < bound ? "yes" : "no")
                << "; potential as without the gradient: "
                << (difference < 1e-12 ? "yes" : "no") << std::endl;
        // Several sets of charges at once, against one at a time.
        std::vector<double> q2(points.size());
        for (unsigned int i = 0; i < points.size(); ++i)
          q2[i] = q[i] * normals[i][0];
        std::vector<std::vector<double>> sets;
        fmm.evaluate_multiple({q, q2}, sets);
        std::vector<double> single2;
        fmm.evaluate(q2, {}, single2);
        double deviation = 0, size = 0;
        for (unsigned int i = 0; i < points.size(); ++i)
          {
            deviation = std::max({deviation,
                                  std::abs(sets[0][i] - plain[i]),
                                  std::abs(sets[1][i] - single2[i])});
            size      = std::max(size, std::abs(plain[i]));
          }
        deallog << "kappa 0, order " << order
                << ", two sets of charges at once as one at a time: "
                << (deviation < 1e-12 * size ? "yes" : "no") << std::endl;
      }
}
