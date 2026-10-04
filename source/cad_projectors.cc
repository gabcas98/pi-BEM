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

#include "cad_projectors.h"

#ifdef DEAL_II_WITH_OPENCASCADE

#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/polynomial.h>
#  include <deal.II/base/quadrature_lib.h>

#  include <deal.II/fe/mapping_q.h>

#  include <BRepBndLib.hxx>
#  include <BRepTopAdaptor_FClass2d.hxx>
#  include <BRep_Tool.hxx>
#  include <Bnd_Box.hxx>
#  include <Geom_Curve.hxx>
#  include <Geom_Surface.hxx>
#  include <ShapeAnalysis_Curve.hxx>
#  include <ShapeAnalysis_Surface.hxx>
#  include <TopExp.hxx>
#  include <TopExp_Explorer.hxx>
#  include <TopTools_IndexedMapOfShape.hxx>
#  include <TopoDS.hxx>
#  include <TopoDS_Edge.hxx>
#  include <TopoDS_Face.hxx>
#  include <gp_Pnt.hxx>
#  include <gp_Pnt2d.hxx>
#  include <gp_Vec.hxx>

#  include <algorithm>
#  include <array>
#  include <cmath>
#  include <limits>
#  include <map>

DEAL_II_NAMESPACE_OPEN

namespace OpenCASCADE
{
  using namespace ExactGeometry;

  namespace
  {
    constexpr unsigned int none = static_cast<unsigned int>(-1);

    Point<3>
    to_point(const gp_Pnt &p)
    {
      return Point<3>(p.X(), p.Y(), p.Z());
    }

    gp_Pnt
    to_gp(const Point<3> &p)
    {
      return gp_Pnt(p[0], p[1], p[2]);
    }

    void
    set_column(const gp_Vec            &v,
               const unsigned int       a,
               DerivativeForm<1, 2, 3> &A)
    {
      for (unsigned int i = 0; i < 3; ++i)
        A[i][a] = v.Coord(i + 1);
    }

    template <int dim>
    Tensor<2, dim>
    gram(const DerivativeForm<1, dim, 3> &A)
    {
      Tensor<2, dim> G;
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          for (unsigned int i = 0; i < 3; ++i)
            G[a][b] += A[i][a] * A[i][b];
      return G;
    }

    // Positive definiteness of a symmetric 1x1 or 2x2 matrix.
    template <int dim>
    bool
    positive_definite(const Tensor<2, dim> &M)
    {
      return M[0][0] > 0. && determinant(M) > 0.;
    }

    // The reference quadrilateral of deal.II: its vertices, and the vertices
    // of its edges x = 0, x = 1, y = 0, y = 1, each from the smaller
    // parameter t to the larger.
    const std::array<Point<2>, 4> unit_vertices = {
      {Point<2>(0, 0), Point<2>(1, 0), Point<2>(0, 1), Point<2>(1, 1)}};
    constexpr unsigned int edge_vertices[4][2] = {{0, 2}, {1, 3}, {0, 1}, {2, 3}};

    Point<2>
    edge_point(const unsigned int edge, const double t)
    {
      switch (edge)
        {
          case 0:
            return Point<2>(0, t);
          case 1:
            return Point<2>(1, t);
          case 2:
            return Point<2>(t, 0);
          default:
            return Point<2>(t, 1);
        }
    }
  } // namespace



  namespace internal
  {
    template <int dim>
    struct CADPatch
    {
      TopoDS_Shape         shape;
      Handle(Geom_Surface) surface;
      Handle(Geom_Curve)   curve;
      bool                 reversed = false;

      /// Range of the parameters of the trimmed patch (edges), and of the
      /// parametrization, to which Newton's iterates are clamped in the
      /// directions that are not periodic.
      std::array<double, dim> first, last, lower, upper;
      std::array<bool, dim>   periodic;
      std::array<double, dim> period;

      /// Bounding box and its diameter, the length scale of the patch.
      std::array<double, 3> box_min, box_max;
      double                diameter;

      /// Faces: the indices of their trimming curves.
      std::vector<unsigned int> trims;

      void
      evaluate(const Point<dim>          &theta,
               Point<3>                  &S,
               DerivativeForm<1, dim, 3> &dS,
               DerivativeForm<2, dim, 3> &d2S) const
      {
        if constexpr (dim == 2)
          {
            gp_Pnt P;
            gp_Vec D1U, D1V, D2U, D2V, D2UV;
            surface->D2(theta[0], theta[1], P, D1U, D1V, D2U, D2V, D2UV);
            S = to_point(P);
            set_column(D1U, 0, dS);
            set_column(D1V, 1, dS);
            for (unsigned int i = 0; i < 3; ++i)
              {
                d2S[i][0][0] = D2U.Coord(i + 1);
                d2S[i][0][1] = d2S[i][1][0] = D2UV.Coord(i + 1);
                d2S[i][1][1] = D2V.Coord(i + 1);
              }
          }
        else
          {
            gp_Pnt P;
            gp_Vec D1, D2;
            curve->D2(theta[0], P, D1, D2);
            S = to_point(P);
            for (unsigned int i = 0; i < 3; ++i)
              {
                dS[i][0]     = D1.Coord(i + 1);
                d2S[i][0][0] = D2.Coord(i + 1);
              }
          }
      }

      void
      clamp(Point<dim> &theta) const
      {
        for (unsigned int a = 0; a < dim; ++a)
          if (!periodic[a])
            theta[a] = std::clamp(theta[a], lower[a], upper[a]);
      }

      double
      box_distance(const Point<3> &x) const
      {
        double d2 = 0;
        for (unsigned int i = 0; i < 3; ++i)
          {
            const double d =
              std::max({box_min[i] - x[i], 0., x[i] - box_max[i]});
            d2 += d * d;
          }
        return std::sqrt(d2);
      }

      void
      set_box()
      {
        Bnd_Box box;
        BRepBndLib::Add(shape, box);
        box.Get(
          box_min[0], box_min[1], box_min[2], box_max[0], box_max[1], box_max[2]);
        double d2 = 0;
        for (unsigned int i = 0; i < 3; ++i)
          d2 += std::pow(box_max[i] - box_min[i], 2);
        diameter = std::sqrt(d2);
      }
    };
  } // namespace internal



  namespace
  {
    using internal::CADPatch;

    std::unique_ptr<CADPatch<2>>
    make_face_patch(const TopoDS_Shape &shape)
    {
      const TopoDS_Face face  = TopoDS::Face(shape);
      auto              patch = std::make_unique<CADPatch<2>>();
      patch->shape            = shape;
      patch->reversed         = (shape.Orientation() == TopAbs_REVERSED);
      patch->surface          = BRep_Tool::Surface(face);
      if (patch->surface.IsNull())
        return nullptr;
      patch->surface->Bounds(patch->lower[0],
                             patch->upper[0],
                             patch->lower[1],
                             patch->upper[1]);
      patch->periodic = {{bool(patch->surface->IsUPeriodic()),
                          bool(patch->surface->IsVPeriodic())}};
      patch->first    = patch->lower;
      patch->last     = patch->upper;
      patch->set_box();
      return patch;
    }

    std::unique_ptr<CADPatch<1>>
    make_curve_patch(const TopoDS_Shape &shape)
    {
      const TopoDS_Edge edge = TopoDS::Edge(shape);
      if (BRep_Tool::Degenerated(edge))
        return nullptr;
      auto patch      = std::make_unique<CADPatch<1>>();
      patch->shape    = shape;
      patch->reversed = (shape.Orientation() == TopAbs_REVERSED);
      double first, last;
      patch->curve = BRep_Tool::Curve(edge, first, last);
      if (patch->curve.IsNull())
        return nullptr;
      patch->first    = {{first}};
      patch->last     = {{last}};
      patch->lower    = {{patch->curve->FirstParameter()}};
      patch->upper    = {{patch->curve->LastParameter()}};
      patch->periodic = {{bool(patch->curve->IsPeriodic())}};
      patch->period   = {{patch->periodic[0] ? patch->curve->Period() : 0.}};
      patch->set_box();
      return patch;
    }



    template <int dim>
    struct ShapeCache : public Projector<dim, 3>::Cache
    {
      /// Per face, created when first needed: the global projection and
      /// the classification of parameters against the trimmed face.
      std::vector<Handle(ShapeAnalysis_Surface)>            analysis;
      std::vector<std::unique_ptr<BRepTopAdaptor_FClass2d>> classifiers;

      /// The patch of the current cell, and the parameters of the foot of
      /// its centre, from which Newton's method starts.
      unsigned int patch = none;
      Point<dim>   theta;

      /// The parameters of the feet of the mapping at the nodes 0, 1/2, 1
      /// of the reference cell in each direction, numbered
      /// lexicographically: their quadratic interpolation at a point of
      /// the cell starts Newton's method close to the solution.
      std::array<Point<dim>, dim == 2 ? 9 : 3> node_theta;

      Point<dim>
      start(const Point<dim> &unit_point) const
      {
        const auto basis = [](const double t) {
          return std::array<double, 3>{{2 * (t - 0.5) * (t - 1),
                                        -4 * t * (t - 1),
                                        2 * t * (t - 0.5)}};
        };
        Point<dim>                  result;
        const std::array<double, 3> b0 = basis(unit_point[0]);
        if constexpr (dim == 1)
          for (unsigned int i = 0; i < 3; ++i)
            result += b0[i] * node_theta[i];
        else
          {
            const std::array<double, 3> b1 = basis(unit_point[1]);
            for (unsigned int j = 0; j < 3; ++j)
              for (unsigned int i = 0; i < 3; ++i)
                result += b0[i] * b1[j] * node_theta[i + 3 * j];
          }
        return result;
      }

      /// The transfinite map of the current cell, if it has edges on
      /// trimming curves (faces only): per edge the trimming curve or none,
      /// the mapping at the Gauss-Lobatto points of the edge, the curve
      /// parameters of its ends and the images of its ends by its own
      /// projection; the images of the vertices; and the Lagrange basis of
      /// the Gauss-Lobatto points.
      struct Edge
      {
        unsigned int          curve = none;
        std::vector<Point<3>> nodes;
        std::array<double, 2> parameter;
        std::array<Point<3>, 2> end;
      };
      unsigned int                                       n_trimmed = 0;
      std::array<Edge, 4>                                edges;
      std::array<Point<3>, 4>                            corners;
      const std::vector<Polynomials::Polynomial<double>> *basis = nullptr;
      std::map<unsigned int, std::vector<Polynomials::Polynomial<double>>>
        bases;
    };



    /**
     * The closest point of x on the patch by Newton's method on
     * f = |S(theta) - x|^2 / 2, from theta. Where the Hessian G - H is not
     * positive definite, the step is that of Levenberg and Marquardt, with
     * G plus a small multiple of the identity, which also copes with a
     * degenerate parametrization, as at the poles of a sphere, where G is
     * singular. Each step is halved until f decreases. Returns whether the
     * iteration converged; S, dS and d2S are those of the last theta.
     */
    template <int dim>
    bool
    closest_point(const CADPatch<dim>       &patch,
                  const Point<3>            &x,
                  Point<dim>                &theta,
                  Point<3>                  &S,
                  DerivativeForm<1, dim, 3> &dS,
                  DerivativeForm<2, dim, 3> &d2S)
    {
      // Newton's method converges quadratically: once the step is below
      // 1e-9 of the size of the patch, the iterate it leads to is exact to
      // rounding, and the iteration stops there.
      const double tolerance = 1e-9 * std::max(1., patch.diameter);
      patch.evaluate(theta, S, dS, d2S);
      double f = (x - S).norm_square();
      for (unsigned int iteration = 0; iteration < 50; ++iteration)
        {
          const Tensor<1, 3> r = x - S;
          Tensor<1, dim>     g;
          for (unsigned int a = 0; a < dim; ++a)
            for (unsigned int i = 0; i < 3; ++i)
              g[a] += r[i] * dS[i][a];
          Tensor<2, dim> H;
          for (unsigned int i = 0; i < 3; ++i)
            H += r[i] * d2S[i];
          Tensor<2, dim> M = gram(dS) - H;
          if (!positive_definite(M))
            {
              M = gram(dS);
              for (unsigned int a = 0; a < dim; ++a)
                M[a][a] += 1e-10 * trace(M) + 1e-300;
            }
          const Tensor<1, dim> delta = invert(M) * g;

          Tensor<1, 3> step;
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int a = 0; a < dim; ++a)
              step[i] += dS[i][a] * delta[a];
          const bool last = (step.norm() <= tolerance);

          // Close to the minimum the full step is taken: there f changes
          // by less than its rounding error, which makes the descent test
          // meaningless. Elsewhere the step is halved until f decreases, up
          // to that rounding error.
          const bool local = (step.norm() <= 1e-6 * patch.diameter);
          const double noise =
            1e-14 * std::max(1., x.norm()) * std::sqrt(f) + 1e-300;
          Point<dim>                trial;
          Point<3>                  S_trial;
          DerivativeForm<1, dim, 3> dS_trial;
          DerivativeForm<2, dim, 3> d2S_trial;
          double                    f_trial = f;
          bool                      descent = false;
          for (double length = 1.; length > 1e-9; length /= 2)
            {
              trial = theta + length * delta;
              patch.clamp(trial);
              patch.evaluate(trial, S_trial, dS_trial, d2S_trial);
              f_trial = (x - S_trial).norm_square();
              if (local || f_trial <= f + noise)
                {
                  descent = true;
                  break;
                }
            }
          if (!descent || !std::isfinite(f_trial))
            // No decrease along the step: a minimum up to round-off, or a
            // failure.
            return std::isfinite(f) && step.norm() <= 1e-8 * patch.diameter;
          theta = trial;
          S     = S_trial;
          dS    = dS_trial;
          d2S   = d2S_trial;
          f     = f_trial;
          if (last)
            return true;
        }
      return false;
    }



    // Whether the closest point S of x is a regular minimum in the interior
    // of the parametrization: orthogonality, and a positive definite
    // Hessian G - H.
    template <int dim>
    bool
    regular_minimum(const CADPatch<dim>             &patch,
                    const Point<3>                  &x,
                    const Point<3>                  &S,
                    const DerivativeForm<1, dim, 3> &dS,
                    const DerivativeForm<2, dim, 3> &d2S)
    {
      const Tensor<1, 3> r = x - S;
      Tensor<2, dim>     H;
      for (unsigned int i = 0; i < 3; ++i)
        H += r[i] * d2S[i];
      const Tensor<2, dim> G = gram(dS);
      for (unsigned int a = 0; a < dim; ++a)
        {
          double g = 0;
          for (unsigned int i = 0; i < 3; ++i)
            g += r[i] * dS[i][a];
          if (std::abs(g) >
              1e-10 * std::max(1., patch.diameter) * std::sqrt(G[a][a]))
            return false;
        }
      return positive_definite(Tensor<2, dim>(G - H));
    }



    /// The closest point on a trimming curve, from the parameter t.
    ProjectionResult<3>
    project_onto_curve(const CADPatch<1> &curve,
                       const Point<3>    &x,
                       double            &t,
                       const double       tolerance)
    {
      Point<1>                theta(t);
      Point<3>                S;
      DerivativeForm<1, 1, 3> dS;
      DerivativeForm<2, 1, 3> d2S;
      bool converged = closest_point(curve, x, theta, S, dS, d2S);
      if (!converged)
        {
          gp_Pnt projection;
          double parameter;
          ShapeAnalysis_Curve().Project(curve.curve,
                                        to_gp(x),
                                        tolerance,
                                        projection,
                                        parameter,
                                        curve.first[0],
                                        curve.last[0],
                                        false);
          theta     = Point<1>(parameter);
          converged = closest_point(curve, x, theta, S, dS, d2S);
        }
      ProjectionResult<3> result;
      result.point   = S;
      result.regular = converged && regular_minimum(curve, x, S, dS, d2S);
      result.jacobian =
        result.regular ? closest_point_jacobian(dS, d2S, x - S) :
                         closest_point_jacobian(dS, d2S, Tensor<1, 3>());
      t = theta[0];
      return result;
    }



    /**
     * The projection of x onto a patch, from the parameters theta, which
     * return those of the image: the closest point, or along the direction
     * unless closest_only, followed by the trimming test for faces.
     *
     * Where the parametrization of a face degenerates at the image, as at
     * the poles of a surface of revolution, where a column of A vanishes,
     * the formulas in A, G and H lose accuracy as the rounding error over
     * the square root of the condition number of G, and the image point is
     * still accurate but the Jacobian and the normal are not. There, the two
     * are taken from the projections of the points x +- delta t_i and
     * x +- 2 delta t_i, with t_1, t_2 tangent to the face, whose images lie
     * away from the degeneracy: the Richardson extrapolation of the means
     * of the two stencils, (4 M(delta) - M(2 delta)) / 3, has an error of
     * order delta^4.
     */
    template <int dim>
    ProjectionResult<3>
    project_onto_patch(const CADPatch<dim>     &patch,
                       ShapeCache<dim>         &data,
                       const Direction<3>      &direction,
                       const double             tolerance,
                       const MeshPoint<dim, 3> &x,
                       Point<dim>              &theta,
                       const bool               closest_only = false,
                       const bool               allow_stencil = true)
    {
      ProjectionResult<3>       result;
      Point<3>                  S;
      DerivativeForm<1, dim, 3> dS;
      DerivativeForm<2, dim, 3> d2S;
      result.regular = closest_point(patch, x.point, theta, S, dS, d2S) &&
                       regular_minimum(patch, x.point, S, dS, d2S);
      if (!result.regular)
        {
          // Start again from the global projection of the point: Newton's
          // method may stop on a degenerate boundary of the parameters, as a
          // pole of a surface of revolution, where it cannot turn towards a
          // foot on the other side of the pole.
          if constexpr (dim == 2)
            {
              const gp_Pnt2d uv =
                data.analysis[data.patch]->ValueOfUV(to_gp(x.point), tolerance);
              theta = Point<2>(uv.X(), uv.Y());
            }
          else
            {
              gp_Pnt projection;
              double t;
              ShapeAnalysis_Curve().Project(patch.curve,
                                            to_gp(x.point),
                                            tolerance,
                                            projection,
                                            t,
                                            patch.first[0],
                                            patch.last[0],
                                            false);
              theta = Point<1>(t);
            }
          result.regular = closest_point(patch, x.point, theta, S, dS, d2S);
        }
      AssertThrow(std::isfinite(S.norm_square()),
                  ExcMessage("The projection onto the CAD patch failed."));
      if (!regular_minimum(patch, x.point, S, dS, d2S))
        result.regular = false;

      result.point = S;
      if constexpr (dim == 2)
        {
          result.normal = unit_normal(dS);
          if (patch.reversed)
            result.normal = -result.normal;
        }
      result.jacobian =
        result.regular ? closest_point_jacobian(dS, d2S, x.point - S) :
                         closest_point_jacobian(dS, d2S, Tensor<1, 3>());

      if constexpr (dim == 2)
        if (!direction.is_closest_point() && !closest_only)
          {
            // Newton's method on S(u, v) - x - zeta d = 0 from the closest
            // point.
            const auto [d, J_d] = direction.evaluate(x);
            double     zeta      = (S - x.point) * d / (d * d);
            Point<2>   uv        = theta;
            bool       converged = false;
            for (unsigned int iteration = 0; iteration < 50; ++iteration)
              {
                patch.evaluate(uv, S, dS, d2S);
                const Tensor<1, 3> F = S - x.point - zeta * d;
                Tensor<2, 3>       M;
                for (unsigned int i = 0; i < 3; ++i)
                  {
                    M[i][0] = dS[i][0];
                    M[i][1] = dS[i][1];
                    M[i][2] = -d[i];
                  }
                if (std::abs(determinant(M)) <
                    1e-12 * std::sqrt(determinant(gram(dS))))
                  break;
                const Tensor<1, 3> delta = invert(M) * F;
                uv[0] -= delta[0];
                uv[1] -= delta[1];
                zeta -= delta[2];
                patch.clamp(uv);
                Tensor<1, 3> step = -delta[2] * d;
                for (unsigned int i = 0; i < 3; ++i)
                  step[i] += dS[i][0] * delta[0] + dS[i][1] * delta[1];
                if (step.norm() <= 1e-13 * std::max(1., patch.diameter))
                  {
                    patch.evaluate(uv, S, dS, d2S);
                    converged = ((S - x.point - zeta * d).norm() <=
                                 1e-10 * std::max(1., patch.diameter));
                    break;
                  }
              }
            if (!converged)
              // The line misses the patch: keep the closest point.
              result.regular = false;
            else
              {
                theta         = uv;
                result.point  = S;
                result.normal = unit_normal(dS);
                if (patch.reversed)
                  result.normal = -result.normal;
                result.jacobian =
                  directional_projection_jacobian(d, result.normal, zeta, J_d);
              }
          }

      if constexpr (dim == 2)
        {
          const Tensor<2, 2> G = gram(dS);
          if (allow_stencil &&
              determinant(G) < 1e-9 * trace(G) * trace(G))
            {
              // A normal at the image: along x - S if x lies off the face,
              // else across the non-degenerate column of A and the mixed
              // derivative, the limit direction of the vanishing column.
              Tensor<1, 3> n = x.point - result.point;
              if (n.norm() <= 1e-8 * std::max(1., patch.diameter))
                {
                  Tensor<1, 3> a, b;
                  const unsigned int k = (G[0][0] > G[1][1]) ? 0 : 1;
                  for (unsigned int i = 0; i < 3; ++i)
                    {
                      a[i] = dS[i][k];
                      b[i] = d2S[i][0][1];
                    }
                  n = cross_product_3d(a, b);
                }
              n /= n.norm();
              Tensor<1, 3> t1 = cross_product_3d(n, Tensor<1, 3>({1, 0, 0}));
              if (t1.norm() < 0.5)
                t1 = cross_product_3d(n, Tensor<1, 3>({0, 1, 0}));
              t1 /= t1.norm();
              const Tensor<1, 3> t2 = cross_product_3d(n, t1);

              const double delta = 3e-4 * std::max(1., patch.diameter);
              std::array<Tensor<2, 3>, 2> mean_J;
              std::array<Tensor<1, 3>, 2> mean_n;
              for (unsigned int level = 0; level < 2; ++level)
                for (const Tensor<1, 3> &t : {t1, -t1, t2, -t2})
                  {
                    MeshPoint<dim, 3> y = x;
                    y.point += (level + 1) * delta * t;
                    Point<dim> theta_y = theta;
                    const ProjectionResult<3> r = project_onto_patch(
                      patch, data, direction, tolerance, y, theta_y,
                      closest_only, false);
                    mean_J[level] += 0.25 * r.jacobian;
                    mean_n[level] += 0.25 * r.normal;
                  }
              result.jacobian = (4 * mean_J[0] - mean_J[1]) / 3;
              result.normal   = (4 * mean_n[0] - mean_n[1]) / 3;
              result.normal /= result.normal.norm();
            }
        }

      // Inside the trimmed patch; a face without trimming curves, such as
      // a closed surface, contains every point of its surface.
      if constexpr (dim == 2)
        {
          if (!patch.trims.empty())
            {
              if (!data.classifiers[data.patch])
                data.classifiers[data.patch] =
                  std::make_unique<BRepTopAdaptor_FClass2d>(
                    TopoDS::Face(patch.shape), tolerance);
              if (!std::isfinite(theta.norm_square()) ||
                  data.classifiers[data.patch]->Perform(
                    gp_Pnt2d(theta[0], theta[1])) == TopAbs_OUT)
                result.regular = false;
            }
        }
      else
        {
          double t = theta[0];
          if (patch.periodic[0])
            t = patch.first[0] +
                std::fmod(std::fmod(t - patch.first[0], patch.period[0]) +
                            patch.period[0],
                          patch.period[0]);
          const double slack = tolerance / std::max(dS.norm(), 1e-300);
          if (t < patch.first[0] - slack || t > patch.last[0] + slack)
            result.regular = false;
        }
      return result;
    }



    // The mapping on an edge of the cell, interpolated from its values at
    // the Gauss-Lobatto points, and its derivative in t.
    std::pair<Point<3>, Tensor<1, 3>>
    interpolate(const std::vector<Point<3>>                         &nodes,
                const std::vector<Polynomials::Polynomial<double>> &basis,
                const double                                         t)
    {
      Point<3>     p;
      Tensor<1, 3> dp;
      double       values[2];
      for (unsigned int j = 0; j < nodes.size(); ++j)
        {
          basis[j].value(t, 1, values);
          p += values[0] * nodes[j];
          dp += values[1] * nodes[j];
        }
      return {p, dp};
    }
  } // namespace



  template <int dim>
  ShapeProjector<dim>::ShapeProjector(const TopoDS_Shape &shape,
                                      const Direction<3> &direction,
                                      const double        tolerance)
    : direction(direction)
    , tolerance(tolerance)
  {
    AssertThrow(dim == 2 || direction.is_closest_point(),
                ExcMessage("Curves are projected onto their closest point "
                           "only."));

    TopTools_IndexedMapOfShape map, edges;
    TopExp::MapShapes(shape, dim == 2 ? TopAbs_FACE : TopAbs_EDGE, map);
    for (int i = 1; i <= map.Extent(); ++i)
      {
        if constexpr (dim == 2)
          {
            auto patch = make_face_patch(map(i));
            if (!patch)
              continue;
            // The trimming curves of the face: its edges but the degenerate
            // ones and the seams of periodic surfaces.
            const TopoDS_Face face = TopoDS::Face(map(i));
            for (TopExp_Explorer e(face, TopAbs_EDGE); e.More(); e.Next())
              {
                const TopoDS_Edge edge = TopoDS::Edge(e.Current());
                if (BRep_Tool::Degenerated(edge) ||
                    BRep_Tool::IsClosed(edge, face))
                  continue;
                int index = edges.FindIndex(edge);
                if (index == 0)
                  {
                    auto curve = make_curve_patch(edge);
                    if (!curve)
                      continue;
                    index = edges.Add(edge);
                    AssertDimension(static_cast<unsigned int>(index),
                                    trimming_curves.size() + 1);
                    trimming_curves.push_back(std::move(curve));
                  }
                if (std::find(patch->trims.begin(),
                              patch->trims.end(),
                              index - 1) == patch->trims.end())
                  patch->trims.push_back(index - 1);
              }
            patches.push_back(std::move(patch));
          }
        else
          {
            auto patch = make_curve_patch(map(i));
            if (patch)
              patches.push_back(std::move(patch));
          }
      }
    AssertThrow(!patches.empty(),
                ExcMessage("The shape has no face or edge to project onto."));
  }



  template <int dim>
  ShapeProjector<dim>::~ShapeProjector() = default;



  template <int dim>
  void
  ShapeProjector<dim>::set_follow_trimming_curves(const bool follow)
  {
    follow_trimming_curves = follow;
  }



  template <int dim>
  unsigned int
  ShapeProjector<dim>::n_patches() const
  {
    return patches.size();
  }



  template <int dim>
  unsigned int
  ShapeProjector<dim>::current_patch(const Cache &cache) const
  {
    return static_cast<const ShapeCache<dim> &>(cache).patch;
  }



  template <int dim>
  unsigned int
  ShapeProjector<dim>::n_trimmed_edges(const Cache &cache) const
  {
    return static_cast<const ShapeCache<dim> &>(cache).n_trimmed;
  }



  template <int dim>
  std::unique_ptr<typename ShapeProjector<dim>::Cache>
  ShapeProjector<dim>::create_cache() const
  {
    auto cache = std::make_unique<ShapeCache<dim>>();
    if constexpr (dim == 2)
      {
        cache->analysis.resize(patches.size());
        cache->classifiers.resize(patches.size());
      }
    return cache;
  }



  template <int dim>
  void
  ShapeProjector<dim>::reinit(
    const Mapping<dim, 3>                               &mapping,
    const typename Triangulation<dim, 3>::cell_iterator &cell,
    Cache                                               &cache) const
  {
    auto          &data   = static_cast<ShapeCache<dim> &>(cache);
    const Point<3> center = cell->center();

    std::vector<std::pair<double, unsigned int>> order;
    for (unsigned int p = 0; p < patches.size(); ++p)
      order.emplace_back(patches[p]->box_distance(center), p);
    std::sort(order.begin(), order.end());

    // The closest patch among those that contain the foot of the centre,
    // or else the closest of all.
    double       best_inside = std::numeric_limits<double>::max();
    double       best_any    = best_inside;
    unsigned int inside      = none, any = none;
    Point<dim>   theta_inside, theta_any;
    for (const auto &[box_distance, p] : order)
      {
        if (box_distance > best_inside + tolerance)
          break;
        const auto &patch = *patches[p];
        Point<dim>  theta;
        double      distance;
        bool        is_inside;
        if constexpr (dim == 2)
          {
            if (data.analysis[p].IsNull())
              data.analysis[p] = new ShapeAnalysis_Surface(patch.surface);
            const gp_Pnt2d uv =
              data.analysis[p]->ValueOfUV(to_gp(center), tolerance);
            distance = data.analysis[p]->Gap();
            theta    = Point<2>(uv.X(), uv.Y());
            if (!data.classifiers[p])
              data.classifiers[p] = std::make_unique<BRepTopAdaptor_FClass2d>(
                TopoDS::Face(patch.shape), tolerance);
            is_inside = (data.classifiers[p]->Perform(uv) != TopAbs_OUT);
          }
        else
          {
            gp_Pnt projection;
            double t;
            distance  = ShapeAnalysis_Curve().Project(patch.curve,
                                                     to_gp(center),
                                                     tolerance,
                                                     projection,
                                                     t,
                                                     patch.first[0],
                                                     patch.last[0],
                                                     false);
            theta     = Point<1>(t);
            is_inside = true;
          }
        if (is_inside && distance < best_inside)
          {
            best_inside  = distance;
            inside       = p;
            theta_inside = theta;
          }
        if (distance < best_any)
          {
            best_any  = distance;
            any       = p;
            theta_any = theta;
          }
      }
    data.patch     = (inside != none) ? inside : any;
    data.theta     = (inside != none) ? theta_inside : theta_any;
    data.n_trimmed = 0;

    // The feet of the mapping at the nodes of the cell, unwrapped on
    // periodic parameters to the period of the centre.
    {
      const auto &patch = *patches[data.patch];
      for (unsigned int k = 0; k < data.node_theta.size(); ++k)
        {
          Point<dim> unit;
          unit[0] = 0.5 * (k % 3);
          if constexpr (dim == 2)
            unit[1] = 0.5 * (k / 3);
          Point<dim>                theta = data.theta;
          Point<3>                  S;
          DerivativeForm<1, dim, 3> dS;
          DerivativeForm<2, dim, 3> d2S;
          if (!closest_point(patch,
                             mapping.transform_unit_to_real_cell(cell, unit),
                             theta,
                             S,
                             dS,
                             d2S) ||
              !std::isfinite(theta.norm_square()))
            theta = data.theta;
          for (unsigned int a = 0; a < dim; ++a)
            if (patch.periodic[a])
              {
                const double period = patch.upper[a] - patch.lower[a];
                theta[a] -= period * std::round((theta[a] - data.theta[a]) /
                                                period);
              }
          data.node_theta[k] = theta;
        }
    }

    if constexpr (dim == 2)
      {
        const auto &face = *patches[data.patch];
        if (!follow_trimming_curves || direction.is_mesh_normal() ||
            face.trims.empty() ||
            cell->reference_cell() != ReferenceCells::Quadrilateral)
          return;

        // The edges of the cell on trimming curves: vertices on the curve,
        // and the midpoint close to it.
        std::array<Point<3>, 4> vertices;
        for (unsigned int v = 0; v < 4; ++v)
          vertices[v] =
            mapping.transform_unit_to_real_cell(cell, unit_vertices[v]);
        for (unsigned int e = 0; e < 4; ++e)
          {
            auto &edge = data.edges[e];
            edge.curve = none;
            const Point<3> &a      = vertices[edge_vertices[e][0]];
            const Point<3> &b      = vertices[edge_vertices[e][1]];
            const double    length = (b - a).norm();
            const double    near   = 1e-3 * length + 10 * tolerance;
            for (const unsigned int k : face.trims)
              {
                const auto &curve = *trimming_curves[k];
                if (curve.box_distance(a) > near ||
                    curve.box_distance(b) > near)
                  continue;
                gp_Pnt       projection;
                double       s[3];
                const double distance_a =
                  ShapeAnalysis_Curve().Project(curve.curve,
                                                to_gp(a),
                                                tolerance,
                                                projection,
                                                s[0],
                                                curve.first[0],
                                                curve.last[0],
                                                false);
                const double distance_b =
                  ShapeAnalysis_Curve().Project(curve.curve,
                                                to_gp(b),
                                                tolerance,
                                                projection,
                                                s[1],
                                                curve.first[0],
                                                curve.last[0],
                                                false);
                if (distance_a > near || distance_b > near)
                  continue;
                const double distance_m = ShapeAnalysis_Curve().Project(
                  curve.curve,
                  to_gp(mapping.transform_unit_to_real_cell(cell,
                                                            edge_point(e, 0.5))),
                  tolerance,
                  projection,
                  s[2],
                  curve.first[0],
                  curve.last[0],
                  false);
                if (distance_m > 0.25 * length)
                  continue;
                // The parameters of the ends, unwrapped on periodic curves
                // so that the parameter runs along the edge.
                if (curve.periodic[0] &&
                    std::abs(s[1] - s[0]) > 0.5 * curve.period[0])
                  s[1] += (s[1] < s[0] ? 1. : -1.) * curve.period[0];
                edge.curve     = k;
                edge.parameter = {{s[0], s[1]}};
                ++data.n_trimmed;
                break;
              }
          }
        if (data.n_trimmed == 0)
          return;

        // The Lagrange basis of the Gauss-Lobatto points, as many as the
        // degree of a polynomial mapping plus one.
        unsigned int n_nodes = 9;
        if (const auto q = dynamic_cast<const MappingQ<2, 3> *>(&mapping))
          n_nodes = std::min(q->get_degree() + 1, 9u);
        auto basis = data.bases.find(n_nodes);
        if (basis == data.bases.end())
          basis = data.bases
                    .emplace(n_nodes,
                             Polynomials::generate_complete_Lagrange_basis(
                               QGaussLobatto<1>(n_nodes).get_points()))
                    .first;
        data.basis = &basis->second;
        const std::vector<Point<1>> nodes =
          QGaussLobatto<1>(n_nodes).get_points();

        // The images of the vertices by the projection, and the mapping on
        // the edges; the ends of the edges on trimming curves projected
        // onto the curves.
        // A Jacobian of the cell at its vertices, for directions that need
        // one: the two edges from the vertex.
        const auto vertex_jacobian = [&](const unsigned int v) {
          DerivativeForm<1, 2, 3> V;
          const unsigned int      along_s = v ^ 1, along_t = v ^ 2;
          for (unsigned int i = 0; i < 3; ++i)
            {
              V[i][0] = (vertices[along_s][i] - vertices[v][i]) *
                        (unit_vertices[along_s][0] - unit_vertices[v][0]);
              V[i][1] = (vertices[along_t][i] - vertices[v][i]) *
                        (unit_vertices[along_t][1] - unit_vertices[v][1]);
            }
          return V;
        };
        for (unsigned int v = 0; v < 4; ++v)
          {
            MeshPoint<2, 3> x;
            x.cell         = cell;
            x.unit_point   = unit_vertices[v];
            x.point        = vertices[v];
            x.jacobian     = vertex_jacobian(v);
            Point<2> theta = data.start(unit_vertices[v]);
            data.corners[v] =
              project_onto_patch(face, data, direction, tolerance, x, theta)
                .point;
          }
        for (unsigned int e = 0; e < 4; ++e)
          {
            auto &edge = data.edges[e];
            edge.nodes.resize(n_nodes);
            for (unsigned int j = 0; j < n_nodes; ++j)
              edge.nodes[j] =
                mapping.transform_unit_to_real_cell(cell,
                                                    edge_point(e,
                                                               nodes[j][0]));
            for (unsigned int i = 0; i < 2; ++i)
              if (edge.curve == none)
                edge.end[i] = data.corners[edge_vertices[e][i]];
              else
                edge.end[i] =
                  project_onto_curve(*trimming_curves[edge.curve],
                                     vertices[edge_vertices[e][i]],
                                     edge.parameter[i],
                                     tolerance)
                    .point;
          }
      }
  }



  template <int dim>
  ProjectionResult<3>
  ShapeProjector<dim>::project(const MeshPoint<dim, 3> &x, Cache &cache) const
  {
    auto &data = static_cast<ShapeCache<dim> &>(cache);
    Assert(data.patch != none, ExcMessage("Call reinit() first."));
    const auto &patch = *patches[data.patch];

    if constexpr (dim == 2)
      if (data.n_trimmed > 0)
        {
          // The transfinite interpolation of the images of the four edges,
          // X(s, t) = sum of the edges blended linearly - bilinear
          // interpolation of the vertices, projected onto the face.
          const double s = x.unit_point[0], t = x.unit_point[1];
          std::array<Point<3>, 4>     c;
          std::array<Tensor<1, 3>, 4> dc;
          bool                        regular = true;
          for (unsigned int e = 0; e < 4; ++e)
            {
              const auto  &edge = data.edges[e];
              const double r    = (e < 2) ? t : s;
              const auto [p, dp] = interpolate(edge.nodes, *data.basis, r);
              ProjectionResult<3> image;
              if (edge.curve == none)
                {
                  // A Jacobian of the cell at the point of the edge for the
                  // direction: the tangent of the edge and, across it, the
                  // other tangent of the cell; only the derivative along
                  // the edge enters.
                  MeshPoint<2, 3> y;
                  y.cell                   = x.cell;
                  y.unit_point             = edge_point(e, r);
                  y.point                  = p;
                  y.jacobian               = x.jacobian;
                  const unsigned int along = (e < 2) ? 1 : 0;
                  for (unsigned int i = 0; i < 3; ++i)
                    y.jacobian[i][along] = dp[i];
                  Point<2> theta = data.start(edge_point(e, r));
                  image =
                    project_onto_patch(patch, data, direction, tolerance, y, theta);
                }
              else
                {
                  double parameter =
                    (1 - r) * edge.parameter[0] + r * edge.parameter[1];
                  image = project_onto_curve(*trimming_curves[edge.curve],
                                             p,
                                             parameter,
                                             tolerance);
                }
              regular = regular && image.regular;
              // Shift the edge linearly to end at the images of the
              // vertices; zero unless the vertices lie off the curve.
              const Tensor<1, 3> shift_a =
                data.corners[edge_vertices[e][0]] - edge.end[0];
              const Tensor<1, 3> shift_b =
                data.corners[edge_vertices[e][1]] - edge.end[1];
              c[e]  = image.point + (1 - r) * shift_a + r * shift_b;
              dc[e] = image.jacobian * dp + shift_b - shift_a;
            }
          const auto &V = data.corners;
          Point<3>    X;
          X += (1 - s) * c[0] + s * c[1] + (1 - t) * c[2] + t * c[3];
          X -= (1 - s) * (1 - t) * V[0] + s * (1 - t) * V[1] +
               (1 - s) * t * V[2] + s * t * V[3];
          const Tensor<1, 3> X_s =
            c[1] - c[0] + (1 - t) * dc[2] + t * dc[3] -
            ((1 - t) * (V[1] - V[0]) + t * (V[3] - V[2]));
          const Tensor<1, 3> X_t =
            (1 - s) * dc[0] + s * dc[1] + c[3] - c[2] -
            ((1 - s) * (V[2] - V[0]) + s * (V[3] - V[1]));

          MeshPoint<2, 3> y;
          y.point        = X;
          Point<2> theta = data.start(x.unit_point);
          ProjectionResult<3> result =
            project_onto_patch(patch, data, direction, tolerance, y, theta, true);
          result.regular = result.regular && regular;

          // The derivative of the map with respect to the reference
          // coordinates, D = J_P [X_s, X_t], returned as D (V^T V)^{-1} V^T,
          // whose product with V is D.
          DerivativeForm<1, 2, 3> D;
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
              {
                D[i][0] += result.jacobian[i][j] * X_s[j];
                D[i][1] += result.jacobian[i][j] * X_t[j];
              }
          const Tensor<2, 2> G_inverse = invert(gram(x.jacobian));
          Tensor<2, 3>       J;
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
              for (unsigned int a = 0; a < 2; ++a)
                for (unsigned int b = 0; b < 2; ++b)
                  J[i][j] += D[i][a] * G_inverse[a][b] * x.jacobian[j][b];
          result.jacobian = J;
          return result;
        }

    Point<dim> theta = data.start(x.unit_point);
    return project_onto_patch(patch, data, direction, tolerance, x, theta);
  }



  template <int dim>
  UpdateFlags
  ShapeProjector<dim>::get_needed_update_flags() const
  {
    return direction.get_needed_update_flags();
  }



  template class ShapeProjector<1>;
  template class ShapeProjector<2>;
} // namespace OpenCASCADE

DEAL_II_NAMESPACE_CLOSE

#endif
