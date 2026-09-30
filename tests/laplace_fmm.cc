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

#include <random>

#include "laplace_fmm.h"
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
          const auto &charges = kind == 1 ? no_q : q;
          const auto &dipoles = kind == 0 ? no_d : d;
          std::vector<double> potentials;
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
    }
}
