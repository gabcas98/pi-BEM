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

// LaplaceFMM with the Helmholtz kernel e^{i k r} / r on points of two
// nearby unit spheres against the direct sum, for complex charges, dipoles
// and both, k = 1 and 3 (boxes of up to some k r = 5 interact through
// expansions), at two truncation orders; several sets at once and
// combined against one at a time, and the potentials at some of the
// points of one sphere only.

#include <random>

#include "laplace_fmm.h"
#include "tests.h"

namespace
{
  using complex = std::complex<double>;

  std::vector<complex>
  direct(const std::vector<Point<3>>              &points,
         const std::vector<complex>               &q,
         const std::vector<Tensor<1, 3, complex>> &d,
         const double                              k,
         const std::vector<unsigned int>          &targets)
  {
    std::vector<complex> u(targets.size(), 0.);
    for (unsigned int t = 0; t < targets.size(); ++t)
      {
        const Point<3> &x = points[targets[t]];
        for (unsigned int j = 0; j < points.size(); ++j)
          {
            if (j == targets[t])
              continue;
            const Tensor<1, 3> r        = x - points[j];
            const double       distance = r.norm();
            const complex      e =
              complex(std::cos(k * distance), std::sin(k * distance)) /
              distance;
            if (!q.empty())
              u[t] += q[j] * e;
            if (!d.empty())
              u[t] += (d[j][0] * r[0] + d[j][1] * r[1] + d[j][2] * r[2]) *
                      complex(1., -k * distance) * e / (distance * distance);
          }
      }
    return u;
  }
} // namespace

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
  for (unsigned int i = 0; i < 4000; ++i)
    {
      Tensor<1, 3> v({g(rng), g(rng), g(rng)});
      v /= v.norm();
      points.push_back(Point<3>(v) + Point<3>(0, 0, i % 2 ? 1.025 : -1.025));
      normals.push_back(v);
    }
  std::uniform_real_distribution<double> u(-1., 1.);
  std::vector<complex>                   q(points.size());
  std::vector<Tensor<1, 3, complex>>     d(points.size());
  for (unsigned int i = 0; i < points.size(); ++i)
    {
      q[i] = complex(u(rng), u(rng));
      for (unsigned int c = 0; c < 3; ++c)
        d[i][c] = complex(u(rng), u(rng)) * normals[i][c];
    }
  std::vector<unsigned int> targets;
  for (unsigned int i = 0; i < points.size(); i += 37)
    targets.push_back(i);

  for (const double k : {1., 3.})
    for (const unsigned int order : {8u, 14u})
      {
        LaplaceFMM::AdditionalData data;
        data.order      = order;
        data.theta      = 0.5;
        data.wavenumber = k;
        LaplaceFMM fmm;
        fmm.reinit(points, data);

        const std::vector<complex>                 none;
        const std::vector<Tensor<1, 3, complex>>   no_dipoles;
        const std::array<std::pair<bool, bool>, 3> kinds = {
          {{true, false}, {false, true}, {true, true}}};
        const char *names[3] = {"charges", "dipoles", "charges and dipoles"};
        for (unsigned int c = 0; c < 3; ++c)
          {
            const auto &qc = kinds[c].first ? q : none;
            const auto &dc = kinds[c].second ? d : no_dipoles;
            std::vector<std::vector<complex>> potentials;
            fmm.evaluate_complex_sets({qc}, {dc}, potentials, true);
            const std::vector<complex> exact =
              direct(points, qc, dc, k, targets);
            double error = 0, scale = 0;
            for (unsigned int t = 0; t < targets.size(); ++t)
              {
                error =
                  std::max(error,
                           std::abs(potentials[0][targets[t]] - exact[t]));
                scale = std::max(scale, std::abs(exact[t]));
              }
            // Measured: 2e-7 to 9e-7 at order 8, 1e-10 to 4e-10 at order
            // 14; an error in a translation is of order one.
            const double bound = order == 8 ? 1e-5 : 1e-8;
            deallog << "k " << k << ", order " << order << ", " << names[c]
                    << ": relative error below " << bound << ": "
                    << (error / scale < bound ? "yes" : "no") << std::endl;
          }

        // Two sets of charges and one of dipoles at once, and two combined
        // sets, against one at a time.
        std::vector<complex> q2(points.size());
        for (unsigned int i = 0; i < points.size(); ++i)
          q2[i] = q[i] * normals[i][0];
        std::vector<std::vector<complex>> sets, first, second, third;
        fmm.evaluate_complex_sets({q, q2}, {d}, sets);
        fmm.evaluate_complex_sets({q}, {}, first);
        fmm.evaluate_complex_sets({q2}, {}, second);
        fmm.evaluate_complex_sets({}, {d}, third);
        double deviation = 0, size = 0;
        for (unsigned int i = 0; i < points.size(); ++i)
          {
            deviation = std::max({deviation,
                                  std::abs(sets[0][i] - first[0][i]),
                                  std::abs(sets[1][i] - second[0][i]),
                                  std::abs(sets[2][i] - third[0][i])});
            size =
              std::max({size, std::abs(first[0][i]), std::abs(third[0][i])});
          }
        std::vector<std::vector<complex>> combined, single;
        fmm.evaluate_complex_sets({q, q2}, {d, d}, combined, true);
        fmm.evaluate_complex_sets({q2}, {d}, single, true);
        for (unsigned int i = 0; i < points.size(); ++i)
          deviation =
            std::max({deviation,
                      std::abs(combined[0][i] - first[0][i] - third[0][i]),
                      std::abs(combined[1][i] - single[0][i])});
        deallog << "k " << k << ", order " << order
                << ", several and combined sets as one at a time: "
                << (deviation < 1e-12 * size ? "yes" : "no") << std::endl;

        // With every third point of the lower sphere (the even ones) a
        // target, the same potentials there and zero at the others.
        std::vector<bool> flags(points.size());
        for (unsigned int i = 0; i < points.size(); i += 6)
          flags[i] = true;
        LaplaceFMM target_fmm;
        target_fmm.reinit(points, data, flags);
        std::vector<std::vector<complex>> some;
        target_fmm.evaluate_complex_sets({q}, {d}, some, true);
        std::vector<std::vector<complex>> all;
        fmm.evaluate_complex_sets({q}, {d}, all, true);
        deviation = 0;
        size      = 0;
        for (unsigned int i = 0; i < points.size(); ++i)
          {
            deviation =
              std::max(deviation,
                       std::abs(some[0][i] - (flags[i] ? all[0][i] : 0.)));
            size = std::max(size, std::abs(all[0][i]));
          }
        deallog << "k " << k << ", order " << order
                << ", potentials at a third of the points of one sphere "
                   "only: "
                << (deviation < 1e-12 * size ? "yes" : "no") << std::endl;
      }
}
