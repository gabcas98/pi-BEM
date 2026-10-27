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
#  include <Geom2d_Curve.hxx>
#  include <Geom_BSplineSurface.hxx>
#  include <Geom_Curve.hxx>
#  include <Geom_RectangularTrimmedSurface.hxx>
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
#  include <gp_Vec2d.hxx>

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
    constexpr unsigned int edge_vertices[4][2] = {{0, 2},
                                                  {1, 3},
                                                  {0, 1},
                                                  {2, 3}};

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
      TopoDS_Shape shape;
      Handle(Geom_Surface) surface;
      Handle(Geom_Curve) curve;
      bool reversed = false;

      /// Range of the parameters of the trimmed patch (edges), and of the
      /// parametrization, to which Newton's iterates are clamped in the
      /// directions that are not periodic.
      std::array<double, dim> first, last, lower, upper;
      std::array<bool, dim>   periodic;
      std::array<double, dim> period;

      /// Bounding box and its diameter, the length scale of the patch.
      std::array<double, 3> box_min, box_max;
      double                diameter;

      /// Faces: the indices of their trimming curves, and their poles, the
      /// points of their degenerate edges.
      std::vector<unsigned int> trims;
      std::vector<Point<3>>     poles;

      /// Faces: the parameters of the interior knot lines of a B-spline
      /// surface across which its parametrization is only continuous (a
      /// knot of multiplicity at least the degree) while the surface itself
      /// is smooth, its normal continuous, per direction. A map linear in
      /// the parameters has a kink there, which the closest point does not
      /// have. Knot lines along which the surface has an edge are left out:
      /// every map has a kink there.
      std::array<std::vector<double>, dim> kinks;

      /// Whether the open interval between a and b in the parameter a
      /// direction contains a kink.
      bool
      crosses_kink(const unsigned int direction,
                   const double       a,
                   const double       b) const
      {
        const double slack = 1e-9 * (upper[direction] - lower[direction]);
        for (const double k : kinks[direction])
          if (k > std::min(a, b) + slack && k < std::max(a, b) - slack)
            return true;
        return false;
      }

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
                d2S[i][1][1]                = D2V.Coord(i + 1);
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
        box.Get(box_min[0],
                box_min[1],
                box_min[2],
                box_max[0],
                box_max[1],
                box_max[2]);
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
      Handle(Geom_Surface) basis = patch->surface;
      if (const auto trimmed =
            Handle(Geom_RectangularTrimmedSurface)::DownCast(basis);
          !trimmed.IsNull())
        basis = trimmed->BasisSurface();
      if (const auto spline = Handle(Geom_BSplineSurface)::DownCast(basis);
          !spline.IsNull())
        {
          // The normal on the two sides of a knot line, at points along it.
          const auto smooth = [&](const unsigned int a, const double knot) {
            const double       eps = 1e-8 * (patch->upper[a] - patch->lower[a]);
            const unsigned int b   = 1 - a;
            for (unsigned int j = 0; j <= 20; ++j)
              {
                const double t = patch->lower[b] +
                                 (patch->upper[b] - patch->lower[b]) * j / 20.;
                std::array<Tensor<1, 3>, 2> normal;
                for (const int side : {0, 1})
                  {
                    Point<2> theta;
                    theta[a] = knot + (side == 0 ? -eps : eps);
                    theta[b] = t;
                    Point<3>                S;
                    DerivativeForm<1, 2, 3> dS;
                    DerivativeForm<2, 2, 3> d2S;
                    patch->evaluate(theta, S, dS, d2S);
                    const Tensor<1, 3> n = cross_product_3d(
                      Tensor<1, 3>({dS[0][0], dS[1][0], dS[2][0]}),
                      Tensor<1, 3>({dS[0][1], dS[1][1], dS[2][1]}));
                    if (n.norm() < 1e-300)
                      return false;
                    normal[side] = n / n.norm();
                  }
                if ((normal[0] - normal[1]).norm() > 1e-4)
                  return false;
              }
            return true;
          };
          for (int i = 2; i < spline->NbUKnots(); ++i)
            if (spline->UMultiplicity(i) >= spline->UDegree() &&
                smooth(0, spline->UKnot(i)))
              patch->kinks[0].push_back(spline->UKnot(i));
          for (int i = 2; i < spline->NbVKnots(); ++i)
            if (spline->VMultiplicity(i) >= spline->VDegree() &&
                smooth(1, spline->VKnot(i)))
              patch->kinks[1].push_back(spline->VKnot(i));
        }
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
      using NodeTheta = std::array<Point<dim>, dim == 2 ? 9 : 3>;
      NodeTheta node_theta;

      /// A cell whose points have feet outside its face, as across the
      /// junction of two faces: the faces close to it, onto the closest of
      /// which each point is projected, and the parameters of the feet of
      /// the nodes on each; under the parametric map, its own face for a
      /// cell across a kink of the parametrization; empty otherwise.
      std::vector<unsigned int> faces;
      std::vector<NodeTheta>    face_node_theta;

      Point<dim>
      start(const Point<dim> &unit_point) const
      {
        return start(unit_point, node_theta);
      }

      static Point<dim>
      start(const Point<dim> &unit_point, const NodeTheta &nodes)
      {
        const auto basis = [](const double t) {
          return std::array<double, 3>{
            {2 * (t - 0.5) * (t - 1), -4 * t * (t - 1), 2 * t * (t - 0.5)}};
        };
        Point<dim>                  result;
        const std::array<double, 3> b0 = basis(unit_point[0]);
        if constexpr (dim == 1)
          for (unsigned int i = 0; i < 3; ++i)
            result += b0[i] * nodes[i];
        else
          {
            const std::array<double, 3> b1 = basis(unit_point[1]);
            for (unsigned int j = 0; j < 3; ++j)
              for (unsigned int i = 0; i < 3; ++i)
                result += b0[i] * b1[j] * nodes[i + 3 * j];
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
        unsigned int            curve = none;
        std::vector<Point<3>>   nodes;
        std::array<double, 2>   parameter;
        std::array<Point<3>, 2> end;

        /// The parametric map (set_parametric()): the parameters of the
        /// ends on the face; on a trimming curve, the curve of the edge on
        /// the face (null where its parameter differs from that of the
        /// curve in space, which is then pulled back point by point), the
        /// period added to it, and the shifts of its ends to the
        /// parameters of the vertices.
        std::array<Point<2>, 2> theta;
        bool                    projected = false;
        Handle(Geom2d_Curve) pcurve;
        Tensor<1, 2>                offset;
        std::array<Tensor<1, 2>, 2> shift;

        /// A cell on several faces under the parametric map: the face on
        /// which the edge is the segment between the parameters theta of
        /// its ends, as for the cell of that face along it, or none.
        unsigned int face = none;
      };
      unsigned int n_trimmed  = 0;
      unsigned int n_segments = 0;

      /// The images of the points of the edges by their projection, for
      /// the parameters r met so far on the current cell: the points of a
      /// tensor rule share them. Kept for the closest point only, whose
      /// image does not depend on the Jacobian of the cell.
      std::array<std::vector<std::pair<double, ProjectionResult<3>>>, 4>
                              edge_images;
      std::array<Edge, 4>     edges;
      std::array<Point<3>, 4> corners;

      /// Whether the current cell is mapped through the parametrization of
      /// its face, the parameters of its vertices, and whether the curves of
      /// its edges do not close in the parameters (a vertex at a degenerate
      /// point of the parametrization, or a pole inside the cell), so that
      /// the cell is blended in space.
      bool                    parametric_cell = false;
      bool                    degenerate      = false;
      std::array<Point<2>, 4> vertex_theta;
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
          const bool   local = (step.norm() <= 1e-6 * patch.diameter);
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



    // Whether S, where Newton's method stopped, is the foot of x on the
    // surface of the patch: x - S normal to it, unlike a point of the
    // boundary of the parameters to which the iterate was clamped, as for a
    // point beyond an edge of a face that is not trimmed there.
    bool
    is_foot(const CADPatch<2>             &patch,
            const Point<3>                &x,
            const Point<3>                &S,
            const DerivativeForm<1, 2, 3> &dS)
    {
      const Tensor<1, 3> r = x - S;
      for (unsigned int a = 0; a < 2; ++a)
        {
          double g = 0, t = 0;
          for (unsigned int i = 0; i < 3; ++i)
            {
              g += r[i] * dS[i][a];
              t += dS[i][a] * dS[i][a];
            }
          t = std::sqrt(t);
          if (std::abs(g) >
              (1e-6 * r.norm() + 1e-8 * std::max(1., patch.diameter)) * t)
            return false;
        }
      return true;
    }



    // Whether the closest point of x is a strict local minimum at S, from
    // the Hessian G - H of |S - x|^2 / 2: x lies closer to the surface than
    // its centres of curvature, where the projection of a cell does not
    // fold.
    bool
    within_reach(const Point<3>                &x,
                 const Point<3>                &S,
                 const DerivativeForm<1, 2, 3> &dS,
                 const DerivativeForm<2, 2, 3> &d2S)
    {
      const Tensor<1, 3> r = x - S;
      Tensor<2, 2>       H;
      for (unsigned int i = 0; i < 3; ++i)
        H += r[i] * d2S[i];
      return positive_definite(Tensor<2, 2>(gram(dS) - H));
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
      result.point    = S;
      result.regular  = converged && regular_minimum(curve, x, S, dS, d2S);
      result.jacobian = result.regular ?
                          closest_point_jacobian(dS, d2S, x - S) :
                          closest_point_jacobian(dS, d2S, Tensor<1, 3>());
      t               = theta[0];
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
                       const unsigned int       index,
                       ShapeCache<dim>         &data,
                       const Direction<3>      &direction,
                       const double             tolerance,
                       const MeshPoint<dim, 3> &x,
                       Point<dim>              &theta,
                       const bool               closest_only  = false,
                       const bool               allow_stencil = true,
                       const bool               allow_restart = true)
    {
      ProjectionResult<3>       result;
      Point<3>                  S;
      DerivativeForm<1, dim, 3> dS;
      DerivativeForm<2, dim, 3> d2S;
      result.regular = closest_point(patch, x.point, theta, S, dS, d2S) &&
                       regular_minimum(patch, x.point, S, dS, d2S);
      if (!result.regular && allow_restart)
        {
          // Start again from the global projection of the point: Newton's
          // method may stop on a degenerate boundary of the parameters, as a
          // pole of a surface of revolution, where it cannot turn towards a
          // foot on the other side of the pole.
          if constexpr (dim == 2)
            {
              if (data.analysis[index].IsNull())
                data.analysis[index] = new ShapeAnalysis_Surface(patch.surface);
              const gp_Pnt2d uv =
                data.analysis[index]->ValueOfUV(to_gp(x.point), tolerance);
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
      result.jacobian = result.regular ?
                          closest_point_jacobian(dS, d2S, x.point - S) :
                          closest_point_jacobian(dS, d2S, Tensor<1, 3>());

      if constexpr (dim == 2)
        if (!direction.is_closest_point() && !closest_only)
          {
            // Newton's method on S(u, v) - x - zeta d = 0 from the closest
            // point.
            const auto [d, J_d] = direction.evaluate(x);
            double   zeta       = (S - x.point) * d / (d * d);
            Point<2> uv         = theta;
            bool     converged  = false;
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
          if (allow_stencil && determinant(G) < 1e-9 * trace(G) * trace(G))
            {
              // A normal at the image: along x - S if x lies off the face,
              // else across the non-degenerate column of A and the mixed
              // derivative, the limit direction of the vanishing column.
              Tensor<1, 3> n = x.point - result.point;
              if (n.norm() <= 1e-8 * std::max(1., patch.diameter))
                {
                  Tensor<1, 3>       a, b;
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
                    Point<dim>                theta_y = theta;
                    const ProjectionResult<3> r =
                      project_onto_patch(patch,
                                         index,
                                         data,
                                         direction,
                                         tolerance,
                                         y,
                                         theta_y,
                                         closest_only,
                                         false);
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
              if (!data.classifiers[index])
                data.classifiers[index] =
                  std::make_unique<BRepTopAdaptor_FClass2d>(
                    TopoDS::Face(patch.shape), tolerance);
              if (!std::isfinite(theta.norm_square()) ||
                  data.classifiers[index]->Perform(
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



    /**
     * The projection of x, at the given point of the reference cell, onto
     * the faces of the current cell: onto its face; or, for a cell whose
     * points have feet on several faces, onto the closest of those faces
     * that contain the foot, else onto the closest of all, as onto the
     * union of the faces. The parameters of the feet of the nodes on each
     * face start Newton's method.
     */
    template <int dim>
    ProjectionResult<3>
    project_onto_faces(
      const std::vector<std::unique_ptr<CADPatch<dim>>> &patches,
      ShapeCache<dim>                                   &data,
      const Direction<3>                                &direction,
      const double                                       tolerance,
      const MeshPoint<dim, 3>                           &x,
      const Point<dim>                                  &unit_point,
      const bool                                         closest_only = false)
    {
      if (data.faces.empty())
        {
          Point<dim> theta = data.start(unit_point);
          return project_onto_patch(*patches[data.patch],
                                    data.patch,
                                    data,
                                    direction,
                                    tolerance,
                                    x,
                                    theta,
                                    closest_only);
        }
      // A face that does not contain the foot stops Newton's method on its
      // boundary: the global restart of the projection, meant for the poles
      // of a parametrization, is tried only if no face contains it.
      ProjectionResult<3> best;
      double              best_distance = std::numeric_limits<double>::max();
      bool                best_inside   = false;
      for (const bool restart : {false, true})
        {
          for (unsigned int k = 0; k < data.faces.size(); ++k)
            {
              const unsigned int p = data.faces[k];
              Point<dim>         theta =
                ShapeCache<dim>::start(unit_point, data.face_node_theta[k]);
              const ProjectionResult<3> r = project_onto_patch(*patches[p],
                                                               p,
                                                               data,
                                                               direction,
                                                               tolerance,
                                                               x,
                                                               theta,
                                                               closest_only,
                                                               true,
                                                               restart);
              const double              distance = (r.point - x.point).norm();
              if ((r.regular && !best_inside) ||
                  (r.regular == best_inside && distance < best_distance))
                {
                  best          = r;
                  best_distance = distance;
                  best_inside   = r.regular;
                }
            }
          if (best_inside)
            break;
        }
      return best;
    }



    /**
     * The parameters on a face of the point of a trimming curve at its
     * parameter t, and their derivative in t: from the curve of the edge on
     * the face where there is one, else from the closest point of the curve
     * point on the face, started from theta, with the derivative of the
     * curve pulled back, d theta / dt = G^{-1} A^T dC/dt, as the curve lies
     * on the face.
     */
    void
    curve_parameters(const Handle(Geom2d_Curve) & pcurve,
                     const CADPatch<1> &curve,
                     const CADPatch<2> &face,
                     const double       t,
                     Point<2>          &theta,
                     Tensor<1, 2>      &dtheta)
    {
      if (!pcurve.IsNull())
        {
          gp_Pnt2d P;
          gp_Vec2d V;
          pcurve->D1(t, P, V);
          theta  = Point<2>(P.X(), P.Y());
          dtheta = Tensor<1, 2>({V.X(), V.Y()});
          return;
        }
      Point<3>                C;
      DerivativeForm<1, 1, 3> dC;
      DerivativeForm<2, 1, 3> d2C;
      curve.evaluate(Point<1>(t), C, dC, d2C);
      Point<3>                S;
      DerivativeForm<1, 2, 3> dS;
      DerivativeForm<2, 2, 3> d2S;
      closest_point(face, C, theta, S, dS, d2S);
      Tensor<1, 2> g;
      for (unsigned int a = 0; a < 2; ++a)
        for (unsigned int i = 0; i < 3; ++i)
          g[a] += dS[i][a] * dC[i][0];
      dtheta = invert(gram(dS)) * g;
    }



    // D (V^T V)^{-1} V^T, the Jacobian whose product with V is D.
    Tensor<2, 3>
    jacobian_of(const DerivativeForm<1, 2, 3> &D,
                const DerivativeForm<1, 2, 3> &V)
    {
      const Tensor<2, 2> G_inverse = invert(gram(V));
      Tensor<2, 3>       J;
      for (unsigned int i = 0; i < 3; ++i)
        for (unsigned int j = 0; j < 3; ++j)
          for (unsigned int a = 0; a < 2; ++a)
            for (unsigned int b = 0; b < 2; ++b)
              J[i][j] += D[i][a] * G_inverse[a][b] * V[j][b];
      return J;
    }



    // The number of Gauss-Lobatto points on which the mapping of an edge is
    // interpolated: the degree of a polynomial mapping plus one, at most
    // nine.
    template <int dim>
    unsigned int
    parametric_nodes(const Mapping<dim, 3> &mapping)
    {
      if (const auto q = dynamic_cast<const MappingQ<dim, 3> *>(&mapping))
        return std::min(q->get_degree() + 1, 9u);
      return 9;
    }



    // The mapping on an edge of the cell, interpolated from its values at
    // the Gauss-Lobatto points, and its derivative in t.
    std::pair<Point<3>, Tensor<1, 3>>
    interpolate(const std::vector<Point<3>>                        &nodes,
                const std::vector<Polynomials::Polynomial<double>> &basis,
                const double                                        t)
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
                if (BRep_Tool::Degenerated(edge))
                  {
                    patch->poles.push_back(
                      to_point(BRep_Tool::Pnt(TopExp::FirstVertex(edge))));
                    continue;
                  }
                if (BRep_Tool::IsClosed(edge, face))
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
  void
  ShapeProjector<dim>::set_parametric(const bool parametric)
  {
    this->parametric = parametric;
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
    unsigned int inside = none, any = none;
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
    data.patch      = (inside != none) ? inside : any;
    data.theta      = (inside != none) ? theta_inside : theta_any;
    data.n_trimmed  = 0;
    data.n_segments = 0;
    if constexpr (dim == 2)
      for (auto &images : data.edge_images)
        images.clear();

    // The feet of the mapping at the nodes of the cell on a patch, from
    // the parameters of the foot of the centre, unwrapped on periodic
    // parameters to the period of the centre.
    const auto node_feet = [&](const unsigned int p,
                               const Point<dim>  &theta_center,
                               typename ShapeCache<dim>::NodeTheta &feet) {
      const auto &patch = *patches[p];
      for (unsigned int k = 0; k < feet.size(); ++k)
        {
          Point<dim> unit;
          unit[0] = 0.5 * (k % 3);
          if constexpr (dim == 2)
            unit[1] = 0.5 * (k / 3);
          Point<dim>                theta = theta_center;
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
            theta = theta_center;
          for (unsigned int a = 0; a < dim; ++a)
            if (patch.periodic[a])
              {
                const double period = patch.upper[a] - patch.lower[a];
                theta[a] -=
                  period * std::round((theta[a] - theta_center[a]) / period);
              }
          feet[k] = theta;
        }
    };
    node_feet(data.patch, data.theta, data.node_theta);

    // Whether the nodes of the cell lie within reach of the closest point
    // of their face, closer to it than its centres of curvature, so that
    // its projection does not fold the cell, as it does where a coarse cell
    // spans a strongly curved part of the surface.
    const auto within_reach_of_face = [&]() {
      if constexpr (dim == 2)
        {
          const auto &patch = *patches[data.patch];
          for (unsigned int k = 0; k < data.node_theta.size(); ++k)
            {
              Point<3>                S;
              DerivativeForm<1, 2, 3> dS;
              DerivativeForm<2, 2, 3> d2S;
              patch.evaluate(data.node_theta[k], S, dS, d2S);
              if (!within_reach(mapping.transform_unit_to_real_cell(
                                  cell, Point<2>(0.5 * (k % 3), 0.5 * (k / 3))),
                                S,
                                dS,
                                d2S))
                return false;
            }
        }
      return true;
    };

    // A cell of a shape of several faces with a node (a vertex, the middle
    // of an edge or the centre) whose foot lies off its face, beyond a trimming
    // curve or at an edge of the parameters where Newton's method stopped, as
    // across the junction of two faces, lies on several faces: those whose
    // boxes come within a diameter of the cell, the closest of which each point
    // then takes, by projection; the foot may lie off the other faces as well,
    // in a gap between them. The parametric map holds on one face only, and on
    // the bow of a hull it lost 3e-4 of the area on cells that leave their face
    // along a free edge, with no other face close, which are then mapped by
    // projection too. Only a cell within a diameter of a trimming curve of
    // its face can leave it.
    data.faces.clear();
    data.face_node_theta.clear();
    if constexpr (dim == 2)
      if (const double reach = cell->diameter() + tolerance;
          patches.size() > 1 &&
          std::any_of(patches[data.patch]->trims.begin(),
                      patches[data.patch]->trims.end(),
                      [&](const unsigned int k) {
                        return trimming_curves[k]->box_distance(center) <=
                               reach;
                      }))
        {
          const auto &patch = *patches[data.patch];
          if (!data.classifiers[data.patch])
            data.classifiers[data.patch] =
              std::make_unique<BRepTopAdaptor_FClass2d>(
                TopoDS::Face(patch.shape), tolerance);
          bool several = false;
          for (unsigned int k = 0; k < data.node_theta.size() && !several; ++k)
            {
              const Point<2> unit(0.5 * (k % 3), 0.5 * (k / 3));
              const Point<3> y =
                mapping.transform_unit_to_real_cell(cell, unit);
              const Point<2>         &theta = data.node_theta[k];
              Point<3>                S;
              DerivativeForm<1, 2, 3> dS;
              DerivativeForm<2, 2, 3> d2S;
              patch.evaluate(theta, S, dS, d2S);
              several = !std::isfinite(theta.norm_square()) ||
                        !is_foot(patch, y, S, dS) ||
                        data.classifiers[data.patch]->Perform(
                          gp_Pnt2d(theta[0], theta[1])) == TopAbs_OUT;
            }
          // At a free edge, with no other face close, the parametric map
          // remains if the projection would fold the cell.
          if (several && parametric &&
              std::count_if(order.begin(),
                            order.end(),
                            [&](const auto &o) { return o.first <= reach; }) <=
                1 &&
              !within_reach_of_face())
            several = false;
          if (several)
            for (const auto &[box_distance, p] : order)
              {
                if (box_distance > reach)
                  break;
                if (data.analysis[p].IsNull())
                  data.analysis[p] =
                    new ShapeAnalysis_Surface(patches[p]->surface);
                const gp_Pnt2d uv =
                  data.analysis[p]->ValueOfUV(to_gp(center), tolerance);
                data.faces.push_back(p);
                data.face_node_theta.emplace_back();
                node_feet(p,
                          Point<dim>(uv.X(), uv.Y()),
                          data.face_node_theta.back());
              }
        }

    // Under the parametric map, a cell on one face whose parameters cross a
    // kink of the parametrization of the face takes the same way as a cell
    // on several faces, with its face only: the projection of the
    // transfinite map of its edges, whose closest point does not depend on
    // the parametrization; unless that projection would fold the cell, which
    // is worse than the kink.
    if constexpr (dim == 2)
      if (parametric && data.faces.empty())
        {
          const auto &patch = *patches[data.patch];
          bool        kink  = false;
          for (unsigned int a = 0; a < 2 && !kink; ++a)
            if (!patch.kinks[a].empty())
              {
                double low = data.node_theta[0][a], high = low;
                for (const auto &theta : data.node_theta)
                  {
                    low  = std::min(low, theta[a]);
                    high = std::max(high, theta[a]);
                  }
                kink = patch.crosses_kink(a, low, high);
              }
          if (kink && within_reach_of_face())
            {
              data.faces.push_back(data.patch);
              data.face_node_theta.push_back(data.node_theta);
            }
        }

    if constexpr (dim == 2)
      {
        const auto &face     = *patches[data.patch];
        data.parametric_cell = false;
        if (cell->reference_cell() != ReferenceCells::Quadrilateral)
          return;
        // The trimming curves of the faces of the cell.
        std::vector<unsigned int> trims = face.trims;
        for (const unsigned int p : data.faces)
          for (const unsigned int k : patches[p]->trims)
            if (std::find(trims.begin(), trims.end(), k) == trims.end())
              trims.push_back(k);
        if (parametric && data.faces.empty())
          data.parametric_cell = true;
        else if (!follow_trimming_curves || direction.is_mesh_normal() ||
                 trims.empty())
          return;

        // The edges of the cell on trimming curves: vertices on the curve,
        // and the midpoint close to it.
        std::array<Point<3>, 4> vertices;
        for (unsigned int v = 0; v < 4; ++v)
          vertices[v] =
            mapping.transform_unit_to_real_cell(cell, unit_vertices[v]);
        for (unsigned int e = 0; e < 4; ++e)
          {
            auto &edge             = data.edges[e];
            edge.curve             = none;
            edge.face              = none;
            const Point<3> &a      = vertices[edge_vertices[e][0]];
            const Point<3> &b      = vertices[edge_vertices[e][1]];
            const double    length = (b - a).norm();
            const double    near   = 1e-3 * length + 10 * tolerance;
            for (const unsigned int k : trims)
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
                  to_gp(mapping.transform_unit_to_real_cell(
                    cell, edge_point(e, 0.5))),
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
        if (data.parametric_cell)
          {
            reinit_parametric(mapping, cell, vertices, data);
            return;
          }

        // A cell on several faces under the parametric map: an edge whose
        // points have their feet on one face is the segment between the
        // parameters of its ends on that face, as for the cell of that face
        // on its other side, so that the images of the two cells meet; the
        // other edges are projected, as by the cells on their other side.
        // The cell is then the projection of the transfinite interpolation
        // of its edges onto its faces.
        if (parametric && !data.faces.empty())
          {
            constexpr unsigned int vertex_node[4] = {0, 2, 6, 8};
            for (unsigned int e = 0; e < 4; ++e)
              {
                auto &edge = data.edges[e];
                if (edge.curve != none)
                  continue;
                for (unsigned int k = 0;
                     k < data.faces.size() && edge.face == none;
                     ++k)
                  {
                    const unsigned int p      = data.faces[k];
                    const auto        &patch  = *patches[p];
                    bool               inside = true;
                    if (!data.classifiers[p])
                      data.classifiers[p] =
                        std::make_unique<BRepTopAdaptor_FClass2d>(
                          TopoDS::Face(patch.shape), tolerance);
                    for (unsigned int i = 0; i <= 4 && inside; ++i)
                      {
                        const Point<2> unit = edge_point(e, 0.25 * i);
                        Point<2>       theta =
                          ShapeCache<2>::start(unit, data.face_node_theta[k]);
                        Point<3>                S;
                        DerivativeForm<1, 2, 3> dS;
                        DerivativeForm<2, 2, 3> d2S;
                        const Point<3>          y =
                          mapping.transform_unit_to_real_cell(cell, unit);
                        closest_point(patch, y, theta, S, dS, d2S);
                        inside = std::isfinite(theta.norm_square()) &&
                                 is_foot(patch, y, S, dS) &&
                                 data.classifiers[p]->Perform(
                                   gp_Pnt2d(theta[0], theta[1])) != TopAbs_OUT;
                      }
                    // Nor across a kink of the parametrization, where the
                    // segment would have a kink.
                    const auto &ends = data.face_node_theta[k];
                    const auto &a    = ends[vertex_node[edge_vertices[e][0]]];
                    const auto &b    = ends[vertex_node[edge_vertices[e][1]]];
                    if (!inside || patch.crosses_kink(0, a[0], b[0]) ||
                        patch.crosses_kink(1, a[1], b[1]))
                      continue;
                    edge.face  = p;
                    edge.theta = {
                      {data
                         .face_node_theta[k][vertex_node[edge_vertices[e][0]]],
                       data.face_node_theta[k]
                                           [vertex_node[edge_vertices[e][1]]]}};
                    ++data.n_segments;
                  }
              }
          }
        if (data.n_trimmed + data.n_segments == 0)
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
            x.cell       = cell;
            x.unit_point = unit_vertices[v];
            x.point      = vertices[v];
            x.jacobian   = vertex_jacobian(v);
            data.corners[v] =
              project_onto_faces(
                patches, data, direction, tolerance, x, unit_vertices[v])
                .point;
          }
        for (unsigned int e = 0; e < 4; ++e)
          {
            auto &edge = data.edges[e];
            edge.nodes.resize(n_nodes);
            for (unsigned int j = 0; j < n_nodes; ++j)
              edge.nodes[j] =
                mapping.transform_unit_to_real_cell(cell,
                                                    edge_point(e, nodes[j][0]));
            for (unsigned int i = 0; i < 2; ++i)
              if (edge.face != none)
                {
                  Point<3>                S;
                  DerivativeForm<1, 2, 3> dS;
                  DerivativeForm<2, 2, 3> d2S;
                  patches[edge.face]->evaluate(edge.theta[i], S, dS, d2S);
                  edge.end[i] = S;
                }
              else if (edge.curve == none)
                edge.end[i] = data.corners[edge_vertices[e][i]];
              else
                edge.end[i] = project_onto_curve(*trimming_curves[edge.curve],
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

    if constexpr (dim == 2)
      if (data.parametric_cell)
        return project_parametric(x, cache);

    if constexpr (dim == 2)
      if (data.n_trimmed + data.n_segments > 0)
        {
          // The transfinite interpolation of the images of the four edges,
          // X(s, t) = sum of the edges blended linearly - bilinear
          // interpolation of the vertices, projected onto the face.
          const double                s = x.unit_point[0], t = x.unit_point[1];
          std::array<Point<3>, 4>     c;
          std::array<Tensor<1, 3>, 4> dc;
          bool                        regular = true;
          for (unsigned int e = 0; e < 4; ++e)
            {
              const auto  &edge  = data.edges[e];
              const double r     = (e < 2) ? t : s;
              const auto [p, dp] = interpolate(edge.nodes, *data.basis, r);
              ProjectionResult<3> image;
              if (edge.face != none)
                {
                  // The segment between the parameters of the ends, whose
                  // derivative in r is that of the edge.
                  const Tensor<1, 2> dtheta = edge.theta[1] - edge.theta[0];
                  DerivativeForm<1, 2, 3> dS;
                  DerivativeForm<2, 2, 3> d2S;
                  patches[edge.face]->evaluate(
                    Point<2>(edge.theta[0] + r * dtheta), image.point, dS, d2S);
                  Tensor<1, 3> tangent;
                  for (unsigned int i = 0; i < 3; ++i)
                    tangent[i] = dS[i][0] * dtheta[0] + dS[i][1] * dtheta[1];
                  const Tensor<1, 3> shift_a =
                    data.corners[edge_vertices[e][0]] - edge.end[0];
                  const Tensor<1, 3> shift_b =
                    data.corners[edge_vertices[e][1]] - edge.end[1];
                  c[e]  = image.point + (1 - r) * shift_a + r * shift_b;
                  dc[e] = tangent + shift_b - shift_a;
                  continue;
                }
              auto      &images = data.edge_images[e];
              const auto known  = std::find_if(images.begin(),
                                              images.end(),
                                              [r](const auto &image) {
                                                return image.first == r;
                                              });
              if (known != images.end())
                image = known->second;
              else if (edge.curve == none)
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
                  image = project_onto_faces(
                    patches, data, direction, tolerance, y, edge_point(e, r));
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
              if (known == images.end() && direction.is_closest_point())
                images.emplace_back(r, image);
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
          y.point                    = X;
          ProjectionResult<3> result = project_onto_faces(
            patches, data, direction, tolerance, y, x.unit_point, true);
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

    return project_onto_faces(
      patches, data, direction, tolerance, x, x.unit_point);
  }



  template <int dim>
  void
  ShapeProjector<dim>::reinit_parametric(
    const Mapping<dim, 3>                               &mapping,
    const typename Triangulation<dim, 3>::cell_iterator &cell,
    const std::array<Point<3>, 4>                       &vertices,
    Cache                                               &cache) const
  {
    if constexpr (dim == 2)
      {
        auto       &data = static_cast<ShapeCache<2> &>(cache);
        const auto &face = *patches[data.patch];

        // The parameters of the vertices: those of their closest points,
        // from the feet at the corner nodes of reinit(), unwrapped on
        // periodic parameters to the period of the centre. A parameter is
        // free where its column of dS/dtheta vanishes, as the longitude at a
        // pole.
        constexpr unsigned int             vertex_node[4] = {0, 2, 6, 8};
        std::array<std::array<bool, 2>, 4> free{};
        data.degenerate = false;
        for (unsigned int v = 0; v < 4; ++v)
          {
            MeshPoint<2, 3> x;
            x.point        = vertices[v];
            Point<2> theta = data.node_theta[vertex_node[v]];
            project_onto_patch(face,
                               data.patch,
                               data,
                               direction,
                               tolerance,
                               x,
                               theta,
                               true,
                               false);
            for (unsigned int a = 0; a < 2; ++a)
              if (face.periodic[a])
                {
                  const double period = face.upper[a] - face.lower[a];
                  theta[a] -=
                    period * std::round((theta[a] - data.theta[a]) / period);
                }
            data.vertex_theta[v] = theta;
            Point<3>                S;
            DerivativeForm<1, 2, 3> dS;
            DerivativeForm<2, 2, 3> d2S;
            face.evaluate(theta, S, dS, d2S);
            data.corners[v]      = S;
            const Tensor<2, 2> G = gram(dS);
            for (unsigned int a = 0; a < 2; ++a)
              if (G[a][a] <= 1e-16 * trace(G))
                {
                  free[v][a]      = true;
                  data.degenerate = true;
                }
          }

        for (unsigned int e = 0; e < 4; ++e)
          {
            auto              &edge = data.edges[e];
            const unsigned int a    = edge_vertices[e][0];
            const unsigned int b    = edge_vertices[e][1];
            // A free parameter of an end is that of the other end. On a
            // periodic parameter the edge takes the shorter way, which
            // depends on its vertices alone, as for the cell on its other
            // side; a cell whose edges then do not close in the parameters,
            // as around a pole, is blended in space.
            edge.theta = {{data.vertex_theta[a], data.vertex_theta[b]}};
            for (unsigned int c = 0; c < 2; ++c)
              {
                if (free[a][c] && !free[b][c])
                  edge.theta[0][c] = edge.theta[1][c];
                if (free[b][c] && !free[a][c])
                  edge.theta[1][c] = edge.theta[0][c];
                if (face.periodic[c])
                  {
                    const double period = face.upper[c] - face.lower[c];
                    const double turns  = std::round(
                      (edge.theta[1][c] - edge.theta[0][c]) / period);
                    if (turns != 0)
                      {
                        edge.theta[1][c] -= turns * period;
                        data.degenerate = true;
                      }
                  }
              }
            edge.pcurve.Nullify();
            edge.offset = Tensor<1, 2>();
            edge.shift  = {{Tensor<1, 2>(), Tensor<1, 2>()}};

            // Close to a pole the segments in the parameters wind about it:
            // an edge closer to a pole than twice its length is the
            // closest point of the edge of the mesh, as both cells along it
            // find, and the cell is blended in space.
            edge.projected = false;
            if (edge.curve == none)
              {
                const Point<3>    &x_a    = vertices[a];
                const Point<3>    &x_b    = vertices[b];
                const Tensor<1, 3> chord  = x_b - x_a;
                const double       length = chord.norm();
                for (const Point<3> &pole : face.poles)
                  {
                    const double r =
                      std::clamp((pole - x_a) * chord /
                                   std::max(length * length, 1e-300),
                                 0.,
                                 1.);
                    if (pole.distance(x_a + r * chord) < 2 * length)
                      edge.projected = true;
                  }
                if (edge.projected)
                  {
                    data.degenerate = true;
                    if (data.basis == nullptr ||
                        data.basis->size() != parametric_nodes(mapping))
                      {
                        const unsigned int n     = parametric_nodes(mapping);
                        auto               basis = data.bases.find(n);
                        if (basis == data.bases.end())
                          basis =
                            data.bases
                              .emplace(
                                n,
                                Polynomials::generate_complete_Lagrange_basis(
                                  QGaussLobatto<1>(n).get_points()))
                              .first;
                        data.basis = &basis->second;
                      }
                    const std::vector<Point<1>> nodes =
                      QGaussLobatto<1>(data.basis->size()).get_points();
                    edge.nodes.resize(nodes.size());
                    for (unsigned int j = 0; j < nodes.size(); ++j)
                      edge.nodes[j] = mapping.transform_unit_to_real_cell(
                        cell, edge_point(e, nodes[j][0]));
                  }
              }
            if (edge.curve == none)
              continue;

            // On a trimming curve: the parameters of the closest points of
            // the vertices on the curve, and the curve of the edge on the
            // face if its parameter is that of the curve in space.
            const auto &curve = *trimming_curves[edge.curve];
            for (unsigned int i = 0; i < 2; ++i)
              project_onto_curve(curve,
                                 vertices[edge_vertices[e][i]],
                                 edge.parameter[i],
                                 tolerance);
            double first, last;
            edge.pcurve = BRep_Tool::CurveOnSurface(TopoDS::Edge(curve.shape),
                                                    TopoDS::Face(face.shape),
                                                    first,
                                                    last);
            if (!edge.pcurve.IsNull())
              {
                const double middle =
                  0.5 * (edge.parameter[0] + edge.parameter[1]);
                const gp_Pnt2d          uv = edge.pcurve->Value(middle);
                Point<3>                S, C;
                DerivativeForm<1, 2, 3> dS;
                DerivativeForm<2, 2, 3> d2S;
                DerivativeForm<1, 1, 3> dC;
                DerivativeForm<2, 1, 3> d2C;
                face.evaluate(Point<2>(uv.X(), uv.Y()), S, dS, d2S);
                curve.evaluate(Point<1>(middle), C, dC, d2C);
                if (S.distance(C) > 1e-9 * std::max(1., face.diameter))
                  edge.pcurve.Nullify();
              }

            // The ends of the curve on the face, the period that brings them
            // to the parameters of the vertices, and the remaining shifts,
            // zero where the vertices lie on the curve; a free parameter of
            // a vertex is that of the curve.
            std::array<Point<2>, 2> ends = edge.theta;
            for (unsigned int i = 0; i < 2; ++i)
              {
                Tensor<1, 2> derivative;
                curve_parameters(edge.pcurve,
                                 curve,
                                 face,
                                 edge.parameter[i],
                                 ends[i],
                                 derivative);
              }
            for (unsigned int c = 0; c < 2; ++c)
              if (face.periodic[c])
                {
                  const double period = face.upper[c] - face.lower[c];
                  edge.offset[c] =
                    period *
                    std::round((edge.theta[0][c] - ends[0][c]) / period);
                }
            for (unsigned int i = 0; i < 2; ++i)
              {
                edge.shift[i] = edge.theta[i] - (ends[i] + edge.offset);
                for (unsigned int c = 0; c < 2; ++c)
                  if (free[edge_vertices[e][i]][c])
                    edge.shift[i][c] = 0;
              }
          }
      }
    else
      {
        (void)mapping;
        (void)cell;
        (void)vertices;
        (void)cache;
      }
  }



  template <int dim>
  ProjectionResult<3>
  ShapeProjector<dim>::project_parametric(const MeshPoint<dim, 3> &x,
                                          Cache                   &cache) const
  {
    ProjectionResult<3> result;
    if constexpr (dim == 2)
      {
        auto        &data = static_cast<ShapeCache<2> &>(cache);
        const auto  &face = *patches[data.patch];
        const double s = x.unit_point[0], t = x.unit_point[1];

        // The curves of the edges in the parameters of the face.
        std::array<Tensor<1, 2>, 4> c, dc;
        for (unsigned int e = 0; e < 4; ++e)
          {
            const auto        &edge  = data.edges[e];
            const double       r     = (e < 2) ? t : s;
            const Tensor<1, 2> chord = edge.theta[1] - edge.theta[0];
            if (edge.projected)
              continue;
            if (edge.curve == none)
              {
                c[e]  = edge.theta[0] + r * chord;
                dc[e] = chord;
              }
            else
              {
                const double parameter =
                  (1 - r) * edge.parameter[0] + r * edge.parameter[1];
                Point<2>     theta = edge.theta[0] + r * chord - edge.offset;
                Tensor<1, 2> dtheta;
                curve_parameters(edge.pcurve,
                                 *trimming_curves[edge.curve],
                                 face,
                                 parameter,
                                 theta,
                                 dtheta);
                c[e] = theta + edge.offset + (1 - r) * edge.shift[0] +
                       r * edge.shift[1];
                dc[e] = (edge.parameter[1] - edge.parameter[0]) * dtheta +
                        edge.shift[1] - edge.shift[0];
              }
          }

        Point<3>                S;
        DerivativeForm<1, 2, 3> dS;
        DerivativeForm<2, 2, 3> d2S;
        DerivativeForm<1, 2, 3> D;
        if (!data.degenerate)
          {
            // The transfinite interpolation in the parameters, mapped by S.
            std::array<Tensor<1, 2>, 4> V;
            for (unsigned int v = 0; v < 4; ++v)
              V[v] = data.vertex_theta[v];
            const Tensor<1, 2> theta =
              (1 - s) * c[0] + s * c[1] + (1 - t) * c[2] + t * c[3] -
              ((1 - s) * (1 - t) * V[0] + s * (1 - t) * V[1] +
               (1 - s) * t * V[2] + s * t * V[3]);
            const Tensor<1, 2> theta_s =
              c[1] - c[0] + (1 - t) * dc[2] + t * dc[3] -
              ((1 - t) * (V[1] - V[0]) + t * (V[3] - V[2]));
            const Tensor<1, 2> theta_t =
              (1 - s) * dc[0] + s * dc[1] + c[3] - c[2] -
              ((1 - s) * (V[2] - V[0]) + s * (V[3] - V[1]));
            face.evaluate(Point<2>(theta), S, dS, d2S);
            for (unsigned int i = 0; i < 3; ++i)
              {
                D[i][0] = dS[i][0] * theta_s[0] + dS[i][1] * theta_s[1];
                D[i][1] = dS[i][0] * theta_t[0] + dS[i][1] * theta_t[1];
              }
            result.point  = S;
            result.normal = unit_normal(dS);
            if (face.reversed)
              result.normal = -result.normal;
            for (unsigned int a = 0; a < 2; ++a)
              if (!face.periodic[a] &&
                  (theta[a] <
                     face.lower[a] - 1e-9 * (face.upper[a] - face.lower[a]) ||
                   theta[a] >
                     face.upper[a] + 1e-9 * (face.upper[a] - face.lower[a])))
                result.regular = false;
          }
        else
          {
            // The closest point of the transfinite interpolation of the
            // images of the edges.
            std::array<Point<3>, 4>     image;
            std::array<Tensor<1, 3>, 4> d_image;
            for (unsigned int e = 0; e < 4; ++e)
              {
                const auto &edge = data.edges[e];
                if (edge.projected)
                  {
                    const double r = (e < 2) ? t : s;
                    const auto [p, dp] =
                      interpolate(edge.nodes, *data.basis, r);
                    MeshPoint<2, 3> y;
                    y.point        = p;
                    Point<2> theta = data.start(edge_point(e, r));
                    const ProjectionResult<3> projection =
                      project_onto_patch(face,
                                         data.patch,
                                         data,
                                         direction,
                                         tolerance,
                                         y,
                                         theta,
                                         true);
                    image[e]   = projection.point;
                    d_image[e] = projection.jacobian * dp;
                    continue;
                  }
                face.evaluate(Point<2>(c[e]), S, dS, d2S);
                image[e] = S;
                for (unsigned int i = 0; i < 3; ++i)
                  d_image[e][i] = dS[i][0] * dc[e][0] + dS[i][1] * dc[e][1];
              }
            const auto &V = data.corners;
            Point<3>    X;
            X += (1 - s) * image[0] + s * image[1] + (1 - t) * image[2] +
                 t * image[3];
            X -= (1 - s) * (1 - t) * V[0] + s * (1 - t) * V[1] +
                 (1 - s) * t * V[2] + s * t * V[3];
            const Tensor<1, 3> X_s =
              image[1] - image[0] + (1 - t) * d_image[2] + t * d_image[3] -
              ((1 - t) * (V[1] - V[0]) + t * (V[3] - V[2]));
            const Tensor<1, 3> X_t =
              (1 - s) * d_image[0] + s * d_image[1] + image[3] - image[2] -
              ((1 - s) * (V[2] - V[0]) + s * (V[3] - V[1]));
            MeshPoint<2, 3> y;
            y.point        = X;
            Point<2> theta = data.start(x.unit_point);
            result         = project_onto_patch(
              face, data.patch, data, direction, tolerance, y, theta, true);
            for (unsigned int i = 0; i < 3; ++i)
              for (unsigned int j = 0; j < 3; ++j)
                {
                  D[i][0] += result.jacobian[i][j] * X_s[j];
                  D[i][1] += result.jacobian[i][j] * X_t[j];
                }
          }
        result.jacobian = jacobian_of(D, x.jacobian);
      }
    else
      {
        (void)x;
        (void)cache;
      }
    return result;
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
