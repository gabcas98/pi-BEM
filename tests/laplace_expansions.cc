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

// The solid-harmonic expansions of laplace_expansions.h: the addition
// theorem of the kernel, and the chain P2M-M2M-M2L-L2L-L2P for charges and
// dipoles against the direct sum.

#include "laplace_expansions.h"

#include <random>

#include "tests.h"

namespace LE = LaplaceExpansions;

int
main()
{
  initlog();

  // Addition theorem: 1/|x - y| = sum conj(R_n^m(y)) I_n^m(x), |y| < |x|.
  {
    const unsigned int       p = 40;
    const Tensor<1, 3>       x({1.3, -0.4, 0.9}), y({0.2, 0.3, -0.25});
    std::vector<LE::complex> R(LE::n_coefficients(p)), I(LE::n_coefficients(p));
    LE::regular_harmonics(p, y, R.data());
    LE::irregular_harmonics(p, x, I.data());
    double sum = 0;
    for (unsigned int n = 0; n <= p; ++n)
      for (int m = -int(n); m <= int(n); ++m)
        sum +=
          (std::conj(LE::get(R.data(), n, m)) * LE::get(I.data(), n, m)).real();
    deallog << "addition theorem, relative error below 1e-13: "
            << (std::abs(sum * (x - y).norm() - 1.) < 1e-13 ? "yes" : "no")
            << std::endl;
  }

  // Sources in the leaf (-1/2, -1/2, -1/2) + [-1/2, 1/2]^3 of the parent
  // box of side 2 centred at the origin; targets in a leaf of the parent box
  // at offset (2, 2, 1) parent sides.
  std::mt19937                           rng(7);
  std::uniform_real_distribution<double> u(-0.5, 0.5);
  const Point<3>                         source_parent(0, 0, 0);
  const Point<3>                         source_leaf(-0.5, -0.5, -0.5);
  const Point<3>                         target_parent(4, 4, 2);
  const Point<3>                         target_leaf(4.5, 4.5, 2.5);
  std::vector<Point<3>>                  y(50), x(20);
  std::vector<double>                    q(y.size());
  std::vector<Tensor<1, 3>>              d(y.size());
  for (unsigned int i = 0; i < y.size(); ++i)
    {
      y[i] = source_leaf + Point<3>(u(rng), u(rng), u(rng));
      q[i] = 2 * u(rng);
      d[i] = Tensor<1, 3>({u(rng), u(rng), u(rng)});
    }
  for (auto &t : x)
    t = target_leaf + Point<3>(u(rng), u(rng), u(rng));

  for (const bool dipoles : {false, true})
    for (const unsigned int p : {4u, 8u, 10u, 16u})
      {
        LE::Workspace            w(p);
        const unsigned int       n = LE::n_coefficients(p);
        std::vector<LE::complex> M(n), M_parent(n), L_parent(n), L(n);
        for (unsigned int i = 0; i < y.size(); ++i)
          if (dipoles)
            LE::add_dipole(w, d[i], y[i] - source_leaf, M.data());
          else
            LE::add_charge(w, q[i], y[i] - source_leaf, M.data());
        LE::multipole_to_multipole(w,
                                   M.data(),
                                   source_leaf - source_parent,
                                   M_parent.data());
        LE::multipole_to_local(w,
                               M_parent.data(),
                               target_parent - source_parent,
                               L_parent.data());
        LE::local_to_local(w,
                           L_parent.data(),
                           target_leaf - target_parent,
                           L.data());

        double error = 0, scale = 0;
        for (const auto &t : x)
          {
            double exact = 0;
            for (unsigned int i = 0; i < y.size(); ++i)
              {
                const Tensor<1, 3> r    = t - y[i];
                const double       dist = r.norm();
                exact +=
                  dipoles ? (r * d[i]) / (dist * dist * dist) : q[i] / dist;
              }
            error =
              std::max(error,
                       std::abs(
                         LE::evaluate_local(w, L.data(), t - target_leaf) -
                         exact));
            scale = std::max(scale, std::abs(exact));
          }
        // Loose bound on the accuracy at this offset, from the convergence
        // in p of the benchmark fmm_expansions; an error in a translation
        // gives an error of order one.
        const double bound = std::pow(10., -0.6 * p + 1.);
        deallog << (dipoles ? "dipoles" : "charges") << ", p = " << p
                << ": relative error below " << bound << ": "
                << (error / scale < bound ? "yes" : "no") << std::endl;
      }
}
