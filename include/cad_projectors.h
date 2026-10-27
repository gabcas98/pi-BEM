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

#ifndef cad_projectors_h
#define cad_projectors_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_OPENCASCADE

#  include <TopoDS_Shape.hxx>

#  include <array>
#  include <memory>
#  include <vector>

#  include "projected_quadrature.h"

DEAL_II_NAMESPACE_OPEN

namespace OpenCASCADE
{
  namespace internal
  {
    /// A face (dim = 2) or an edge (dim = 1) of a shape with its
    /// parametrization.
    template <int dim>
    struct CADPatch;
  } // namespace internal

  /**
   * The projection of the points of a mesh onto the faces (dim = 2) or the
   * edges (dim = 1) of a CAD shape, with its Jacobian, for
   * ExactGeometry::ProjectedQuadrature.
   *
   * Each cell is associated, in reinit(), with one patch of the shape: a
   * face or an edge, the closest to the centre of the cell among those
   * that contain the foot of the centre in their trimmed domain. The faces
   * are assumed to be smooth, and the mesh to resolve the shape: a cell
   * lies on one patch and does not straddle its edges. The points of the
   * cell are then projected onto that patch alone, so that the
   * projection is smooth across the cell, by Newton's method on the
   * parametrization S(theta) of the patch, started from the quadratic
   * interpolation of the parameters of the feet of nine nodes of the cell,
   * computed in reinit(), and stopped one step after the step falls below
   * 1e-9 of the size of the patch, where the quadratic convergence makes
   * the iterate exact to rounding:
   *  - the closest point: the minimum of |S(theta) - x|^2, whose Hessian
   *    G - H is the matrix of ExactGeometry::closest_point_jacobian();
   *  - along a direction d (faces only, see ExactGeometry::Direction): the
   *    root of S(u, v) - x - zeta d = 0, from the closest point, with the
   *    Jacobian of ExactGeometry::directional_projection_jacobian().
   * Newton converges to machine precision, which the Jacobians need: they
   * hold where the orthogonality or the intersection conditions hold.
   *
   * <h3>Trimmed faces</h3>
   *
   * A projection maps the edges of the cells along the boundary of a face
   * onto curves of its surface that in general differ from the trimming
   * curves: the projection onto a plane leaves a chord of a circle where it
   * is. The images of the cells then miss the parts of the face between the
   * cells and its trimming curves. A quadrilateral with an edge on a
   * trimming curve, an edge whose vertices lie on the curve and whose
   * midpoint lies close to it, is therefore mapped by the transfinite
   * interpolation of Gordon and Hall of its four edges, projected onto the
   * surface of the face:
   *  - an edge on a trimming curve is mapped onto the curve by the closest
   *    point on the curve;
   *  - any other edge by the projection of the projector, as the
   *    neighbouring cell maps it, so that the images of the cells still
   *    partition the face;
   *  - the vertices are the projections of the vertices onto the face, and
   *    the edges on trimming curves are corrected linearly to end there,
   *    which changes them only where the vertices of the mesh lie off the
   *    curves;
   *  - the interior is the closest point on the surface of the blend of the
   *    four edges, with the Jacobian of the composition.
   * The mapping of the cell is evaluated on its edges at Gauss-Lobatto points
   * in reinit(), as many as the degree of a MappingQ plus one, or nine for
   * other mappings, and interpolated in between: the edges are exact for
   * polynomial mappings up to degree eight. The transfinite map applies to
   * the closest point and to projections along a vector field, not along
   * the normal of the mesh, which does not partition the face anyway.
   *
   * <h3>Parametric map</h3>
   *
   * The closest point of a flat cell converges slowly where the surface
   * bends over a length comparable with the cell, as across the nose of a
   * wing: the centres of curvature, where the projection is singular, lie
   * close to the cell, and the polynomial interpolation of the projected map
   * (ExactGeometry::interpolate_exact_geometry()) gains only a factor of two
   * per degree. With set_parametric(), the faces are instead mapped through
   * their parametrization S(u, v): the parameters of the vertices of the
   * cell are those of their closest points, every edge is given a curve in
   * the parameters, and the interior is S of the transfinite interpolation
   * of Gordon and Hall of the four curves:
   *  - an edge inside a face is the segment between the parameters of its
   *    vertices, which both cells along it share;
   *  - an edge on a trimming curve runs along the curve, its parameter
   *    linear in that of the edge, through the curve of the edge on the face
   *    (or, where its parameter differs from that of the edge in space, the
   *    closest points of the curve on the face); two faces along a curve
   *    then map their edges onto the same points when their curves share
   *    the parametrization up to an affine change, as the curves of an
   *    edge do in OpenCASCADE, and as the boundaries of patches built from
   *    the same poles do;
   *  - as for the transfinite map, the curves are shifted linearly to end at
   *    the parameters of the vertices, which changes them only where the
   *    vertices of the mesh lie off the trimming curves.
   * No point is projected but the vertices, and the map is analytic on a
   * cell wherever the surface is: on a polynomial patch it is a polynomial,
   * which the interpolation of the exact geometry reproduces at that
   * degree. A vertex at a degenerate point of the parametrization, such as
   * a pole, takes the free parameter of the other end of each of its edges,
   * so that the edges run along the meridians, and an edge takes the
   * shorter way on a periodic parameter, so that two cells agree on it
   * across a seam. Near a pole, however, the segments in the parameters
   * wind about it: an edge closer to a pole of its face (a degenerate edge
   * of the face) than twice its length is mapped by the closest point of
   * the edge of the mesh, a rule that depends on the edge alone. Where an
   * edge is so projected, or the curves of the edges do not close in the
   * parameters, the interior is the closest point of the transfinite
   * interpolation of the images of the edges in space. The map ignores the
   * direction of the projector.
   *
   * The projection of a point is not regular when Newton does not
   * converge, when the projected point falls outside the trimmed patch, or
   * when it lies beyond a focal point of the patch, and when the line along
   * the direction misses the patch, in which case the closest point is
   * returned.
   *
   * <h3>Degenerate parametrizations</h3>
   *
   * At a pole of a surface of revolution a column of dS/dtheta vanishes.
   * Newton's method, clamped to the range of the parameters, can stop on
   * the pole while the foot lies across it; it then restarts from the
   * global projection of the point. Close to the pole, the formulas in the
   * parameters lose accuracy as the rounding error over the distance from
   * the pole: where the condition number of G exceeds 1e9, the Jacobian
   * and the normal are taken from the projections of four points around x
   * at two distances, combined by Richardson extrapolation (error of order
   * 1e-11).
   */
  template <int dim>
  class ShapeProjector : public ExactGeometry::Projector<dim, 3>
  {
  public:
    using Cache = typename ExactGeometry::Projector<dim, 3>::Cache;

    /**
     * Project onto the faces (dim = 2) or the edges (dim = 1) of the
     * shape. The tolerance is that of the trimming tests and of the global
     * projections of the centres of the cells.
     */
    ShapeProjector(const TopoDS_Shape                &shape,
                   const ExactGeometry::Direction<3> &direction = {},
                   const double                       tolerance = 1e-7);

    ~ShapeProjector();

    virtual std::unique_ptr<Cache>
    create_cache() const override;

    virtual void
    reinit(const Mapping<dim, 3>                               &mapping,
           const typename Triangulation<dim, 3>::cell_iterator &cell,
           Cache &cache) const override;

    virtual ExactGeometry::ProjectionResult<3>
    project(const ExactGeometry::MeshPoint<dim, 3> &x,
            Cache                                  &cache) const override;

    virtual UpdateFlags
    get_needed_update_flags() const override;

    /// Map the cells along the trimming curves of the faces by transfinite
    /// interpolation (the default), or project them like the others.
    void
    set_follow_trimming_curves(const bool follow);

    /// Map the cells of faces through the parametrization of the faces
    /// (see the class documentation) instead of projecting them; off by
    /// default.
    void
    set_parametric(const bool parametric);

    /// The number of faces or edges onto which the projector projects.
    unsigned int
    n_patches() const;

    /// The patch of the cell of the last reinit() with the cache, in the
    /// order of TopExp::MapShapes over the shape, without the degenerate
    /// edges.
    unsigned int
    current_patch(const Cache &cache) const;

    /// The number of edges of the cell of the last reinit() with the cache
    /// that lie on trimming curves, and are mapped onto them.
    unsigned int
    n_trimmed_edges(const Cache &cache) const;

  private:
    /// The parametric map of the current cell, from the positions of its
    /// vertices on the mesh (faces only).
    void
    reinit_parametric(const Mapping<dim, 3> &mapping,
                      const typename Triangulation<dim, 3>::cell_iterator &cell,
                      const std::array<Point<3>, 4> &vertices,
                      Cache                         &cache) const;

    /// A point of the current cell under the parametric map (faces only).
    ExactGeometry::ProjectionResult<3>
    project_parametric(const ExactGeometry::MeshPoint<dim, 3> &x,
                       Cache                                  &cache) const;

    std::vector<std::unique_ptr<internal::CADPatch<dim>>> patches;
    /// The trimming curves of the faces: their edges that are neither
    /// degenerate nor seams.
    std::vector<std::unique_ptr<internal::CADPatch<1>>> trimming_curves;
    const ExactGeometry::Direction<3>                   direction;
    const double                                        tolerance;
    bool follow_trimming_curves = true;
    bool parametric             = false;
  };
} // namespace OpenCASCADE

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
