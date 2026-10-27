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

#ifndef projected_quadrature_h
#define projected_quadrature_h

#include <deal.II/base/config.h>

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * Integration over the exact geometry of a surface or curve that a mesh only
 * approximates (Gambarini, Mola, Ciaramella, Miglio, "Proximal operator
 * Jacobians for multi-patch surface mesh smoothing and surface integrals").
 *
 * The quadrature points of a mesh cell are mapped onto the exact geometry
 * $\Gamma$ by a projection $\mathcal P$, the closest-point map or a
 * projection along a direction, and the change of variables from the cell
 * $\Gamma_h$ to its image is given by the Jacobian $J$ of $\mathcal P$:
 * with $V = \partial x/\partial\xi$ the derivative of the mapping of the
 * cell at the reference point $\xi_q$,
 * @f[
 *   \int_{\mathcal P(K)} f\,d\Gamma
 *   \approx \sum_q f(\mathcal P(x_q))\, w_q
 *     \sqrt{\det\bigl((JV)^T (JV)\bigr)}
 *   = \sum_q f(\mathcal P(x_q))\, JxW_q\,
 *     \sqrt{\frac{\det\bigl((JV)^T (JV)\bigr)}{\det(V^T V)}} .
 * @f]
 * The integrand sees the exact geometry, while the finite element functions
 * keep living on the mesh cells: the quadrature can be passed, cell by cell,
 * to FEValues, FEPointEvaluation or NonMatching::MappingInfo, which then
 * produce the JxW values of the exact surface.
 *
 * The images of the cells partition the exact geometry when a point shared
 * by two cells has the same image from both: then the quadrature converges
 * exponentially in its order, on meshes of flat cells too, as the integrand
 * and the Jacobian of a smooth projection are analytic on each cell. This
 * holds for the closest point and for the projection along a continuous
 * vector field, not along the normal of the mesh (see Direction).
 *
 * The namespace contains the Jacobians of the projections as free
 * functions, the interface Projector of a projection onto a geometry with
 * its Direction, analytic projectors onto spheres and cylinders, a
 * ProjectorCollection by manifold id, and the class ProjectedQuadrature.
 * The projector onto the faces and edges of CAD shapes,
 * OpenCASCADE::ShapeProjector, is in cad_projectors.h.
 */
namespace ExactGeometry
{
  /**
   * The image of a point under a projection onto the exact geometry.
   */
  template <int spacedim>
  struct ProjectionResult
  {
    /// The projected point.
    Point<spacedim> point;

    /// The unit normal of the geometry at the projected point, for a
    /// geometry of codimension one; zero otherwise. Its orientation is that
    /// of the geometry; ProjectedQuadrature orients it like the mesh.
    Tensor<1, spacedim> normal;

    /// The derivative of the projection at the original point.
    Tensor<2, spacedim> jacobian;

    /// False where the projection is not smooth at the point, for instance
    /// where a closest point lies on the boundary of a patch, or where a
    /// projector had to fall back to another projection.
    bool regular = true;
  };

  /**
   * A point of a mesh cell: its coordinates on the reference cell, its
   * position, and the first and, when a projector asks for them, second
   * derivatives of the mapping there.
   */
  template <int dim, int spacedim>
  struct MeshPoint
  {
    Point<dim>                       unit_point;
    Point<spacedim>                  point;
    DerivativeForm<1, dim, spacedim> jacobian;
    DerivativeForm<2, dim, spacedim> hessian;
    /// The cell, for directions defined cell by cell (see Direction).
    typename Triangulation<dim, spacedim>::cell_iterator cell;
  };

  /**
   * The Jacobian of the closest-point map onto a manifold of any dimension
   * and codimension, parametrized by S(theta), at a point x whose closest
   * point is y = S(theta): with A = dS/dtheta, G = A^T A and
   * H_{ab} = (x - y) . d^2 S / dtheta_a dtheta_b,
   * @f[
   *   J = A\,(G - H)^{-1} A^T .
   * @f]
   * It follows from differentiating the orthogonality conditions
   * (x - S(theta)) . dS/dtheta_a = 0. For a surface, x - y = rho n and
   * H = rho B with B the second fundamental form, which is Theorem 2 of
   * the paper, [I + rho J_n] J = I - n n; for a curve it is
   * y' y'^T / (|y'|^2 - (x - y) . y''), Theorem 3. The formula needs no
   * sign convention for the distance and no normal.
   */
  template <int dim, int spacedim>
  Tensor<2, spacedim>
  closest_point_jacobian(const DerivativeForm<1, dim, spacedim> &dS,
                         const DerivativeForm<2, dim, spacedim> &d2S,
                         const Tensor<1, spacedim>              &x_minus_y);

  /**
   * The Jacobian of the projection onto a surface along a direction d(x),
   * z = x + zeta d(x), with n the normal of the surface at z and J_d the
   * derivative of the direction (zero for a fixed direction), Proposition 1
   * of the paper:
   * @f[
   *   J = \Bigl(I - \frac{d \otimes n}{d \cdot n}\Bigr)
   *       \bigl(I + \zeta J_d\bigr).
   * @f]
   */
  template <int spacedim>
  Tensor<2, spacedim>
  directional_projection_jacobian(const Tensor<1, spacedim> &d,
                                  const Tensor<1, spacedim> &n,
                                  const double               zeta,
                                  const Tensor<2, spacedim> &J_d);

  /**
   * The unit normal of a manifold of codimension one with derivatives dS,
   * with the orientation of the normals of deal.II mappings: (dS_0 x dS_1)/|.|
   * in 3d and the tangent rotated counterclockwise in 2d.
   */
  template <int dim, int spacedim>
  Tensor<1, spacedim>
  unit_normal(const DerivativeForm<1, dim, spacedim> &dS);

  /**
   * The derivative of the unit normal of a manifold of codimension one,
   * as a function on the manifold: J_n = -A G^{-1} B G^{-1} A^T, with the
   * second fundamental form B_{ab} = n . d^2 S / dtheta_a dtheta_b. It is
   * symmetric and maps tangent vectors to tangent vectors.
   */
  template <int dim, int spacedim>
  Tensor<2, spacedim>
  unit_normal_jacobian(const DerivativeForm<1, dim, spacedim> &dS,
                       const DerivativeForm<2, dim, spacedim> &d2S);

  /**
   * The ratio of the measure of the image under J of the parallelotope
   * spanned by the columns of V to that of the parallelotope itself,
   * sqrt(det((JV)^T JV) / det(V^T V)).
   */
  template <int dim, int spacedim>
  double
  measure_ratio(const Tensor<2, spacedim>              &J,
                const DerivativeForm<1, dim, spacedim> &V);


  /**
   * A projection of the points of a mesh of dimension dim onto an exact
   * geometry of the same dimension.
   *
   * The functions are const and the state of a computation, if any, lives
   * in a Cache that every user creates for itself: one projector can serve
   * several threads, each with its own cache. A projector that associates
   * the cells of the mesh with parts of the geometry, the patches of a CAD
   * shape for instance, does so in reinit(), once per cell, where it may
   * also evaluate the mapping of the cell elsewhere than at the points to
   * be projected.
   *
   * The images of the cells under a projector need not be the images of a
   * single projection: a projector may map a cell by a construction of its
   * own, such as the transfinite interpolation of
   * OpenCASCADE::ShapeProjector on the cells along the trimming curves of a
   * face, as long as it returns the Jacobian J of the map with respect to
   * the position on the cell, of which only the product J V with the
   * derivative V of the mapping matters.
   */
  template <int dim, int spacedim>
  class Projector
  {
  public:
    /// The state of a computation, per user; empty by default.
    class Cache
    {
    public:
      virtual ~Cache() = default;
    };

    virtual ~Projector() = default;

    virtual std::unique_ptr<Cache>
    create_cache() const;

    /// Prepare the projection of the points of a cell, mapped by the given
    /// mapping.
    virtual void
    reinit(const Mapping<dim, spacedim>                               &mapping,
           const typename Triangulation<dim, spacedim>::cell_iterator &cell,
           Cache &cache) const;

    /// Project a point of the current cell.
    virtual ProjectionResult<spacedim>
    project(const MeshPoint<dim, spacedim> &x, Cache &cache) const = 0;

    /// The update flags of the mapping that project() needs beyond the
    /// point and its Jacobian, e.g. update_jacobian_grads for the
    /// derivative of the mesh normal.
    virtual UpdateFlags
    get_needed_update_flags() const;
  };


  /**
   * The direction of a projection: none for the closest point, the normal of
   * the mesh, or a vector field d(x) with spacedim components and its
   * gradient, given as a Function, which must outlive the object.
   *
   * The images of the cells partition the exact geometry only if a point
   * shared by two cells has the same image from both, which holds for the
   * closest point and for a vector field, functions of the point alone.
   * The normal of the mesh jumps across the edges of the cells, which
   * therefore overlap or leave gaps on the geometry unless the edges lie on
   * it, as for curves meshed by points on them: on a sphere meshed with a
   * mapping of degree p, the area so computed has an error decreasing as
   * h^2 for flat cells and about h^(2p+2) for p >= 2, while the closest
   * point integrates it to round-off. The normal interpolated from the
   * vertices of the mesh is continuous and conforming.
   */
  template <int spacedim>
  class Direction
  {
  public:
    /// The closest point.
    Direction() = default;

    /// Along the vector field d(x) of the function, with spacedim
    /// components; it need not be a unit field.
    explicit Direction(const Function<spacedim> &field);

    /// Along the normal of the mesh.
    static Direction<spacedim>
    mesh_normal();

    /**
     * Along the normal of the mesh interpolated from its vertices: at every
     * vertex the average of the unit normals of the cells that share it, and
     * on every cell the d-linear interpolation of those at its vertices. At a
     * hanging vertex the normal is interpolated from the ends of the coarse
     * edge it splits. The field is continuous across the edges of the
     * cells, so that, unlike the normal of the mesh, the projection along it
     * is conforming. Evaluating it needs MeshPoint::cell and
     * MeshPoint::unit_point.
     */
    template <int dim>
    static Direction<spacedim>
    interpolated_mesh_normal(const Triangulation<dim, spacedim> &triangulation,
                             const Mapping<dim, spacedim>       &mapping);

    bool
    is_closest_point() const
    {
      return field == nullptr && !along_mesh_normal && !vertex_normals;
    }

    bool
    is_mesh_normal() const
    {
      return along_mesh_normal;
    }

    /**
     * The direction d at a point of a cell and its derivative J_d, the
     * latter as a matrix of which only the product with the tangent vectors
     * of the cell matters; for the normal of the mesh, from the second
     * derivatives of the mapping.
     */
    template <int dim>
    std::pair<Tensor<1, spacedim>, Tensor<2, spacedim>>
    evaluate(const MeshPoint<dim, spacedim> &x) const;

    /// update_jacobian_grads for the normal of the mesh.
    UpdateFlags
    get_needed_update_flags() const;

  private:
    const Function<spacedim> *field             = nullptr;
    bool                      along_mesh_normal = false;
    /// The normals at the vertices, by vertex index, for the interpolated
    /// normal of the mesh.
    std::shared_ptr<const std::vector<Tensor<1, spacedim>>> vertex_normals;
  };


  /**
   * The projection onto a sphere (dim = 2, spacedim = 3) or a circle
   * (dim = 1, spacedim = 2) of given centre and radius. The closest point is
   * y = c + R e, e = (x - c)/|x - c|, with J = (R/|x - c|)(I - e e); along a
   * direction d, z = x + zeta d with |z - c| = R and the root zeta of
   * smaller magnitude.
   */
  template <int dim, int spacedim>
  class SphericalProjector : public Projector<dim, spacedim>
  {
  public:
    SphericalProjector(const Point<spacedim>     &center,
                       const double               radius,
                       const Direction<spacedim> &direction = {});

    virtual ProjectionResult<spacedim>
    project(const MeshPoint<dim, spacedim>           &x,
            typename Projector<dim, spacedim>::Cache &cache) const override;

    virtual UpdateFlags
    get_needed_update_flags() const override;

  private:
    const Point<spacedim>     center;
    const double              radius;
    const Direction<spacedim> direction;
  };


  /**
   * The closest-point projection onto the cylinder of given radius about the
   * line through a point with a direction: with a the unit direction and
   * r = (x - c) - ((x - c) . a) a the radial part, the closest point is
   * c + ((x - c) . a) a + R r/|r|, with
   * J = a a + (R/|r|)(I - a a - e e), e = r/|r|.
   */
  class CylindricalProjector : public Projector<2, 3>
  {
  public:
    CylindricalProjector(const Point<3>     &point_on_axis,
                         const Tensor<1, 3> &axis_direction,
                         const double        radius);

    virtual ProjectionResult<3>
    project(const MeshPoint<2, 3>  &x,
            Projector<2, 3>::Cache &cache) const override;

  private:
    const Point<3>     point_on_axis;
    const Tensor<1, 3> axis;
    const double       radius;
  };


  /**
   * Different projectors on the cells of different manifold ids, for a mesh
   * whose parts approximate different geometries. The cells of manifold ids
   * without a projector are integrated on the mesh, as with the identity
   * projection. The projectors must outlive the collection.
   */
  template <int dim, int spacedim>
  class ProjectorCollection : public Projector<dim, spacedim>
  {
  public:
    void
    add(const types::manifold_id        manifold_id,
        const Projector<dim, spacedim> &projector);

    virtual std::unique_ptr<typename Projector<dim, spacedim>::Cache>
    create_cache() const override;

    virtual void
    reinit(const Mapping<dim, spacedim>                               &mapping,
           const typename Triangulation<dim, spacedim>::cell_iterator &cell,
           typename Projector<dim, spacedim>::Cache &cache) const override;

    virtual ProjectionResult<spacedim>
    project(const MeshPoint<dim, spacedim>           &x,
            typename Projector<dim, spacedim>::Cache &cache) const override;

    virtual UpdateFlags
    get_needed_update_flags() const override;

  private:
    std::map<types::manifold_id, unsigned int>    index;
    std::vector<const Projector<dim, spacedim> *> projectors;
  };


  /**
   * The quadrature of the cells of a mesh over the exact geometry.
   *
   * After reinit() on a cell, the class provides
   *  - a Quadrature<dim> with the reference points of the base quadrature
   *    and the weights w_q times the measure ratio of the projection, which
   *    FEValues, FEPointEvaluation or NonMatching::MappingInfo turn into the
   *    JxW values of the exact geometry together with the shape functions of
   *    the mesh;
   *  - the projected points, at which the integrand is evaluated, the JxW
   *    values of the exact geometry, and for a geometry of codimension one
   *    its normals, oriented like those of the mesh.
   *
   * Like FEValues, an object is meant for one thread.
   */
  template <int dim, int spacedim>
  class ProjectedQuadrature
  {
  public:
    ProjectedQuadrature(const Mapping<dim, spacedim>   &mapping,
                        const Quadrature<dim>          &quadrature,
                        const Projector<dim, spacedim> &projector);

    void
    reinit(const typename Triangulation<dim, spacedim>::cell_iterator &cell);

    /// Reference points with the weights of the exact geometry.
    const Quadrature<dim> &
    get_quadrature() const
    {
      return projected_quadrature;
    }

    /// The quadrature points on the exact geometry.
    const std::vector<Point<spacedim>> &
    get_points() const
    {
      return points;
    }

    /// The JxW values of the exact geometry.
    const std::vector<double> &
    get_JxW_values() const
    {
      return JxW;
    }

    /// The normals of the exact geometry, oriented like the mesh; only for
    /// codimension one.
    const std::vector<Tensor<1, spacedim>> &
    get_normal_vectors() const
    {
      return normals;
    }

    /// The Jacobians of the projection at the quadrature points.
    const std::vector<Tensor<2, spacedim>> &
    get_projection_jacobians() const
    {
      return jacobians;
    }

    /// The quadrature points of the mesh cell, before the projection.
    const std::vector<Point<spacedim>> &
    get_mesh_points() const
    {
      return fe_values.get_quadrature_points();
    }

    /// The JxW values of the mesh cell.
    const std::vector<double> &
    get_mesh_JxW_values() const
    {
      return fe_values.get_JxW_values();
    }

    /// The normals of the mesh cell; only for codimension one.
    const std::vector<Tensor<1, spacedim>> &
    get_mesh_normal_vectors() const
    {
      return fe_values.get_normal_vectors();
    }

    /// Points of the last reinit() whose projection is not regular.
    unsigned int
    n_irregular_points() const
    {
      return n_irregular;
    }

  private:
    const Mapping<dim, spacedim>                             &mapping;
    const Quadrature<dim>                                     quadrature;
    const Projector<dim, spacedim>                           &projector;
    std::unique_ptr<typename Projector<dim, spacedim>::Cache> cache;
    const FE_Nothing<dim, spacedim>                           fe;
    FEValues<dim, spacedim>                                   fe_values;

    Quadrature<dim>                  projected_quadrature;
    std::vector<Point<spacedim>>     points;
    std::vector<double>              JxW;
    std::vector<Tensor<1, spacedim>> normals;
    std::vector<Tensor<2, spacedim>> jacobians;
    unsigned int                     n_irregular = 0;
  };
} // namespace ExactGeometry

DEAL_II_NAMESPACE_CLOSE

#endif
