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

// The Jacobians of the projections onto an exact geometry against central
// differences:
//  - closest_point_jacobian() on a torus, a helix and an ellipse, with the
//    closest point computed by Newton's method, and the identity
//    [I + rho J_n] J = I - n n of the surface case (Theorem 2 of Gambarini
//    et al.);
//  - the derivative J V of the projected point with respect to the
//    reference coordinates of a curved cell, for the closest point and the
//    projections along the normal of the mesh, along a vector field and
//    along the normal interpolated from the vertices onto a sphere and a
//    circle, and the closest point on a cylinder; the
//    projection along the normal of the mesh needs its derivative, from the
//    second derivatives of the mapping.

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <functional>

#include "projected_quadrature.h"
#include "tests.h"

using namespace ExactGeometry;

template <int dim, int spacedim>
using Parametrization =
  std::function<void(const Point<dim> &,
                     Point<spacedim> &,
                     DerivativeForm<1, dim, spacedim> &,
                     DerivativeForm<2, dim, spacedim> &)>;


// The field d(x) = (I + s K)(x - c), K the rotation by a right angle about
// the last axis in 2d and about the z-axis in 3d: radial plus a swirl.
template <int spacedim>
class SwirlField : public Function<spacedim>
{
public:
  SwirlField(const Point<spacedim> &center, const double swirl)
    : Function<spacedim>(spacedim)
    , center(center)
  {
    for (unsigned int i = 0; i < spacedim; ++i)
      M[i][i] = 1;
    M[0][1] = -swirl;
    M[1][0] = swirl;
  }

  virtual double
  value(const Point<spacedim> &p, const unsigned int i) const override
  {
    return (M * (p - center))[i];
  }

  virtual Tensor<1, spacedim>
  gradient(const Point<spacedim> &, const unsigned int i) const override
  {
    return M[i];
  }

private:
  const Point<spacedim> center;
  Tensor<2, spacedim>   M;
};


// Newton's method on |S(theta) - x|^2 / 2 from theta.
template <int dim, int spacedim>
Point<dim>
closest_point(const Parametrization<dim, spacedim> &S,
              const Point<spacedim>                &x,
              Point<dim>                            theta)
{
  for (unsigned int it = 0; it < 50; ++it)
    {
      Point<spacedim>                  y;
      DerivativeForm<1, dim, spacedim> dS;
      DerivativeForm<2, dim, spacedim> d2S;
      S(theta, y, dS, d2S);
      Tensor<1, dim> g;
      Tensor<2, dim> M;
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int i = 0; i < spacedim; ++i)
          {
            g[a] += (x - y)[i] * dS[i][a];
            for (unsigned int b = 0; b < dim; ++b)
              M[a][b] += dS[i][a] * dS[i][b] - (x - y)[i] * d2S[i][a][b];
          }
      const Tensor<1, dim> delta = invert(M) * g;
      theta += delta;
      if (delta.norm() < 1e-15)
        break;
    }
  return theta;
}


template <int dim, int spacedim>
void
check_closest_point(const std::string                    &name,
                    const Parametrization<dim, spacedim> &S,
                    const Point<dim>                     &theta0,
                    const Tensor<1, spacedim>            &offset)
{
  Point<spacedim>                  y;
  DerivativeForm<1, dim, spacedim> dS;
  DerivativeForm<2, dim, spacedim> d2S;
  S(theta0, y, dS, d2S);
  const Point<spacedim>     x = y + offset;
  const Tensor<2, spacedim> J = closest_point_jacobian(dS, d2S, offset);

  const double        h = 1e-5;
  Tensor<2, spacedim> J_fd;
  for (unsigned int j = 0; j < spacedim; ++j)
    {
      Point<spacedim> xp = x, xm = x, yp, ym;
      xp[j] += h;
      xm[j] -= h;
      S(closest_point(S, xp, theta0), yp, dS, d2S);
      S(closest_point(S, xm, theta0), ym, dS, d2S);
      for (unsigned int i = 0; i < spacedim; ++i)
        J_fd[i][j] = (yp - ym)[i] / (2 * h);
    }
  const double error = (J - J_fd).norm() / J.norm();
  deallog << name << ": |J| = " << std::setprecision(6) << J.norm()
          << ", difference to central differences "
          << (error < 1e-8 ? std::string("below 1e-8") :
                             std::to_string(error))
          << std::endl;

  if constexpr (dim + 1 == spacedim)
    {
      // [I + rho J_n] J = I - n n, with x - y = rho n.
      S(theta0, y, dS, d2S);
      const Tensor<1, spacedim> n   = unit_normal(dS);
      const double              rho = offset * n;
      Tensor<2, spacedim>       I;
      for (unsigned int i = 0; i < spacedim; ++i)
        I[i][i] = 1;
      const double residual =
        ((I + rho * unit_normal_jacobian(dS, d2S)) * J - I +
         outer_product(n, n))
          .norm();
      deallog << name << ": [I + rho J_n] J = I - n n "
              << (residual < 1e-12 ? std::string("holds") :
                                     std::to_string(residual))
              << std::endl;
    }
}


// The derivative of the projected point with respect to the reference
// coordinates at a point of a cell, J V, against central differences.
template <int dim, int spacedim>
void
check_cell(const std::string                                    &name,
           const Mapping<dim, spacedim>                         &mapping,
           const typename Triangulation<dim, spacedim>::cell_iterator &cell,
           const Projector<dim, spacedim>                       &projector,
           const Point<dim>                                     &xi)
{
  const FE_Nothing<dim, spacedim> fe;
  auto cache = projector.create_cache();
  projector.reinit(mapping, cell, *cache);

  const auto project = [&](const Point<dim> &p) {
    FEValues<dim, spacedim> fe_values(mapping,
                                      fe,
                                      Quadrature<dim>(p),
                                      update_quadrature_points |
                                        update_jacobians |
                                        update_jacobian_grads);
    fe_values.reinit(cell);
    MeshPoint<dim, spacedim> x;
    x.cell       = cell;
    x.unit_point = p;
    x.point      = fe_values.quadrature_point(0);
    x.jacobian = fe_values.jacobian(0);
    x.hessian  = fe_values.jacobian_grad(0);
    return std::make_pair(projector.project(x, *cache), x);
  };

  const auto [result, x] = project(xi);
  const double h         = 1e-5;
  double       error = 0, size = 0;
  for (unsigned int a = 0; a < dim; ++a)
    {
      Point<dim> xp = xi, xm = xi;
      xp[a] += h;
      xm[a] -= h;
      const Tensor<1, spacedim> fd =
        (project(xp).first.point - project(xm).first.point) / (2 * h);
      Tensor<1, spacedim> JV;
      for (unsigned int i = 0; i < spacedim; ++i)
        for (unsigned int j = 0; j < spacedim; ++j)
          JV[i] += result.jacobian[i][j] * x.jacobian[j][a];
      error = std::max(error, (JV - fd).norm());
      size  = std::max(size, JV.norm());
    }
  deallog << name << ": projected point " << std::setprecision(5)
          << result.point
          << ", difference of J V to central differences "
          << (error / size < 1e-8 ? std::string("below 1e-8") :
                                    std::to_string(error / size))
          << std::endl;
}


int
main()
{
  initlog();

  // A torus about the z-axis, R = 2, r = 0.5, from inside and outside.
  const Parametrization<2, 3> torus = [](const Point<2>          &t,
                                         Point<3>                &S,
                                         DerivativeForm<1, 2, 3> &dS,
                                         DerivativeForm<2, 2, 3> &d2S) {
    const double R = 2, r = 0.5;
    const double cu = std::cos(t[0]), su = std::sin(t[0]);
    const double cv = std::cos(t[1]), sv = std::sin(t[1]);
    S        = Point<3>((R + r * cv) * cu, (R + r * cv) * su, r * sv);
    dS[0][0] = -(R + r * cv) * su;
    dS[1][0] = (R + r * cv) * cu;
    dS[2][0] = 0;
    dS[0][1] = -r * sv * cu;
    dS[1][1] = -r * sv * su;
    dS[2][1] = r * cv;
    const double second[3][2][2] = {
      {{-(R + r * cv) * cu, r * sv * su}, {r * sv * su, -r * cv * cu}},
      {{-(R + r * cv) * su, -r * sv * cu}, {-r * sv * cu, -r * cv * su}},
      {{0, 0}, {0, -r * sv}}};
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int a = 0; a < 2; ++a)
        for (unsigned int b = 0; b < 2; ++b)
          d2S[i][a][b] = second[i][a][b];
  };
  {
    Point<3>                y;
    DerivativeForm<1, 2, 3> dS;
    DerivativeForm<2, 2, 3> d2S;
    const Point<2>          theta0(0.3, 1.1);
    torus(theta0, y, dS, d2S);
    const Tensor<1, 3> n = unit_normal(dS);
    check_closest_point("torus, rho = 0.2",
                        torus,
                        theta0,
                        Tensor<1, 3>(0.2 * n));
    check_closest_point("torus, rho = -0.3",
                        torus,
                        theta0,
                        Tensor<1, 3>(-0.3 * n));
  }

  // A helix, from a point off its normal plane directions.
  const Parametrization<1, 3> helix = [](const Point<1>          &t,
                                         Point<3>                &S,
                                         DerivativeForm<1, 1, 3> &dS,
                                         DerivativeForm<2, 1, 3> &d2S) {
    const double a = 1, b = 0.3;
    S            = Point<3>(a * std::cos(t[0]), a * std::sin(t[0]), b * t[0]);
    dS[0][0]     = -a * std::sin(t[0]);
    dS[1][0]     = a * std::cos(t[0]);
    dS[2][0]     = b;
    d2S[0][0][0] = -a * std::cos(t[0]);
    d2S[1][0][0] = -a * std::sin(t[0]);
    d2S[2][0][0] = 0;
  };
  {
    Point<3>                y;
    DerivativeForm<1, 1, 3> dS;
    DerivativeForm<2, 1, 3> d2S;
    const Point<1>          theta0(0.7);
    helix(theta0, y, dS, d2S);
    const Tensor<1, 3> t        = dS.transpose()[0] / dS.norm();
    const Tensor<1, 3> normal   = Tensor<1, 3>(-(y - Point<3>(0, 0, y[2])));
    const Tensor<1, 3> binormal = cross_product_3d(t, normal);
    check_closest_point("helix", helix, theta0, 0.2 * normal + 0.1 * binormal);
  }

  // An ellipse in the plane.
  const Parametrization<1, 2> ellipse = [](const Point<1>          &t,
                                           Point<2>                &S,
                                           DerivativeForm<1, 1, 2> &dS,
                                           DerivativeForm<2, 1, 2> &d2S) {
    S            = Point<2>(2 * std::cos(t[0]), std::sin(t[0]));
    dS[0][0]     = -2 * std::sin(t[0]);
    dS[1][0]     = std::cos(t[0]);
    d2S[0][0][0] = -2 * std::cos(t[0]);
    d2S[1][0][0] = -std::sin(t[0]);
  };
  {
    Point<2>                y;
    DerivativeForm<1, 1, 2> dS;
    DerivativeForm<2, 1, 2> d2S;
    const Point<1>          theta0(0.4);
    ellipse(theta0, y, dS, d2S);
    check_closest_point("ellipse",
                        ellipse,
                        theta0,
                        Tensor<1, 2>(0.3 * unit_normal(dS)));
  }

  // Curved cells of a sphere and a circle, whose normal varies within the
  // cell.
  {
    Triangulation<2, 3> tria;
    const Point<3>      center(0.1, -0.2, 0.3);
    GridGenerator::hyper_sphere(tria, center, 1.3);
    tria.refine_global(1);
    const MappingQ<2, 3> mapping(3);
    const auto           cell = std::next(tria.begin_active(), 5);
    const Point<2>       xi(0.31, 0.77);

    // A sphere off the centre of the mesh, so that the normal of the mesh
    // is not radial and the two projections differ.
    const Point<3> shifted = center + Point<3>(0.05, -0.03, 0.04);
    const SphericalProjector<2, 3> closest(shifted, 1.25);
    const SphericalProjector<2, 3> along_normal(shifted,
                                                1.25,
                                                Direction<3>::mesh_normal());
    const SwirlField<3>            field(center, 0.4);
    const SphericalProjector<2, 3> along_field(shifted,
                                               1.25,
                                               Direction<3>(field));
    const CylindricalProjector     cylinder(center,
                                        Tensor<1, 3>({0.2, 0.1, 1}),
                                        1);
    check_cell<2, 3>("sphere, closest point", mapping, cell, closest, xi);
    check_cell<2, 3>("sphere, mesh normal", mapping, cell, along_normal, xi);
    check_cell<2, 3>("sphere, vector field", mapping, cell, along_field, xi);
    const SphericalProjector<2, 3> along_vertex_normal(
      shifted, 1.25, Direction<3>::interpolated_mesh_normal(tria, mapping));
    check_cell<2, 3>("sphere, interpolated normal",
                     mapping,
                     cell,
                     along_vertex_normal,
                     xi);
    check_cell<2, 3>("cylinder, closest point", mapping, cell, cylinder, xi);
  }
  {
    Triangulation<1, 2> tria;
    const Point<2>      center(0.1, -0.2);
    GridGenerator::hyper_sphere(tria, center, 1.3);
    const MappingQ<1, 2> mapping(3);
    const auto           cell = std::next(tria.begin_active(), 2);
    const Point<1>       xi(0.31);

    const Point<2>                 shifted = center + Point<2>(0.05, -0.03);
    const SphericalProjector<1, 2> closest(shifted, 1.4);
    const SphericalProjector<1, 2> along_normal(shifted,
                                                1.4,
                                                Direction<2>::mesh_normal());
    check_cell<1, 2>("circle, closest point", mapping, cell, closest, xi);
    const SwirlField<2>            field(center, 0.4);
    const SphericalProjector<1, 2> along_field(shifted,
                                               1.4,
                                               Direction<2>(field));
    check_cell<1, 2>("circle, mesh normal", mapping, cell, along_normal, xi);
    check_cell<1, 2>("circle, vector field", mapping, cell, along_field, xi);
    const SphericalProjector<1, 2> along_vertex_normal(
      shifted, 1.4, Direction<2>::interpolated_mesh_normal(tria, mapping));
    check_cell<1, 2>("circle, interpolated normal",
                     mapping,
                     cell,
                     along_vertex_normal,
                     xi);
  }
}
