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

#include "projected_quadrature.h"

#include <deal.II/base/exceptions.h>

#include <cmath>

DEAL_II_NAMESPACE_OPEN

namespace ExactGeometry
{
  namespace
  {
    // The matrix A^T B of the inner products of the columns of A and B.
    template <int dim, int spacedim>
    Tensor<2, dim>
    gram(const DerivativeForm<1, dim, spacedim> &A,
         const DerivativeForm<1, dim, spacedim> &B)
    {
      Tensor<2, dim> G;
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          for (unsigned int i = 0; i < spacedim; ++i)
            G[a][b] += A[i][a] * B[i][b];
      return G;
    }

    // The matrix A M A^T.
    template <int dim, int spacedim>
    Tensor<2, spacedim>
    push_forward(const DerivativeForm<1, dim, spacedim> &A,
                 const Tensor<2, dim>                   &M)
    {
      Tensor<2, spacedim> J;
      for (unsigned int i = 0; i < spacedim; ++i)
        for (unsigned int j = 0; j < spacedim; ++j)
          for (unsigned int a = 0; a < dim; ++a)
            for (unsigned int b = 0; b < dim; ++b)
              J[i][j] += A[i][a] * M[a][b] * A[j][b];
      return J;
    }

    // The matrix v . d^2 S / dtheta_a dtheta_b.
    template <int dim, int spacedim>
    Tensor<2, dim>
    contract(const Tensor<1, spacedim>              &v,
             const DerivativeForm<2, dim, spacedim> &d2S)
    {
      Tensor<2, dim> H;
      for (unsigned int i = 0; i < spacedim; ++i)
        H += v[i] * d2S[i];
      return H;
    }

    template <int spacedim>
    Tensor<2, spacedim>
    identity()
    {
      Tensor<2, spacedim> I;
      for (unsigned int i = 0; i < spacedim; ++i)
        I[i][i] = 1.;
      return I;
    }
  } // namespace



  template <int dim, int spacedim>
  Tensor<2, spacedim>
  closest_point_jacobian(const DerivativeForm<1, dim, spacedim> &dS,
                         const DerivativeForm<2, dim, spacedim> &d2S,
                         const Tensor<1, spacedim>              &x_minus_y)
  {
    const Tensor<2, dim> M = gram(dS, dS) - contract(x_minus_y, d2S);
    Assert(determinant(M) > 0,
           ExcMessage("The point lies beyond a focal point of the manifold: "
                      "its closest point is not a local minimum of the "
                      "distance."));
    return push_forward(dS, invert(M));
  }



  template <int spacedim>
  Tensor<2, spacedim>
  directional_projection_jacobian(const Tensor<1, spacedim> &d,
                                  const Tensor<1, spacedim> &n,
                                  const double               zeta,
                                  const Tensor<2, spacedim> &J_d)
  {
    const double d_n = d * n;
    Assert(d_n != 0.,
           ExcMessage("The direction of the projection is tangent to the "
                      "surface."));
    return (identity<spacedim>() - outer_product(d, n) / d_n) *
           (identity<spacedim>() + zeta * J_d);
  }



  template <int dim, int spacedim>
  Tensor<1, spacedim>
  unit_normal(const DerivativeForm<1, dim, spacedim> &dS)
  {
    static_assert(dim + 1 == spacedim,
                  "Normals are defined for codimension one only.");
    const DerivativeForm<1, spacedim, dim> tangents = dS.transpose();
    Tensor<1, spacedim>                    n;
    if constexpr (spacedim == 2)
      n = cross_product_2d(-tangents[0]);
    else
      n = cross_product_3d(tangents[0], tangents[1]);
    return n / n.norm();
  }



  template <int dim, int spacedim>
  Tensor<2, spacedim>
  unit_normal_jacobian(const DerivativeForm<1, dim, spacedim> &dS,
                       const DerivativeForm<2, dim, spacedim> &d2S)
  {
    const Tensor<2, dim> G_inverse = invert(gram(dS, dS));
    return push_forward(dS,
                        Tensor<2, dim>(-G_inverse *
                                       contract(unit_normal(dS), d2S) *
                                       G_inverse));
  }



  template <int dim, int spacedim>
  double
  measure_ratio(const Tensor<2, spacedim>              &J,
                const DerivativeForm<1, dim, spacedim> &V)
  {
    DerivativeForm<1, dim, spacedim> W;
    for (unsigned int i = 0; i < spacedim; ++i)
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int j = 0; j < spacedim; ++j)
          W[i][a] += J[i][j] * V[j][a];
    return std::sqrt(determinant(gram(W, W)) / determinant(gram(V, V)));
  }



  template <int dim, int spacedim>
  std::unique_ptr<typename Projector<dim, spacedim>::Cache>
  Projector<dim, spacedim>::create_cache() const
  {
    return std::make_unique<Cache>();
  }



  template <int dim, int spacedim>
  void
  Projector<dim, spacedim>::reinit(
    const Mapping<dim, spacedim> &,
    const typename Triangulation<dim, spacedim>::cell_iterator &,
    Cache &) const
  {}



  template <int dim, int spacedim>
  UpdateFlags
  Projector<dim, spacedim>::get_needed_update_flags() const
  {
    return update_default;
  }



  template <int spacedim>
  Direction<spacedim>::Direction(const Function<spacedim> &field)
    : field(&field)
  {
    AssertDimension(field.n_components, spacedim);
  }



  template <int spacedim>
  Direction<spacedim>
  Direction<spacedim>::mesh_normal()
  {
    Direction<spacedim> direction;
    direction.along_mesh_normal = true;
    return direction;
  }



  template <int spacedim>
  template <int dim>
  Direction<spacedim>
  Direction<spacedim>::interpolated_mesh_normal(
    const Triangulation<dim, spacedim> &triangulation,
    const Mapping<dim, spacedim>       &mapping)
  {
    static_assert(dim + 1 == spacedim,
                  "The mesh has a normal in codimension one only.");
    std::vector<Tensor<1, spacedim>> normals(triangulation.n_vertices());
    std::vector<Point<dim>>          vertices;
    for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
      vertices.push_back(GeometryInfo<dim>::unit_cell_vertex(v));
    const FE_Nothing<dim, spacedim> fe;
    FEValues<dim, spacedim>         values(mapping,
                                   fe,
                                   Quadrature<dim>(vertices),
                                   update_normal_vectors);
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        values.reinit(cell);
        for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
          normals[cell->vertex_index(v)] += values.normal_vector(v);
      }
    for (auto &n : normals)
      if (n.norm() > 0)
        n /= n.norm();
    // Hanging vertices, coarse cells first: the normal of the midpoint of a
    // refined face is the mean of those at its ends.
    if constexpr (dim == 2)
      for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
        for (const auto &cell : triangulation.active_cell_iterators_on_level(
               level))
          for (const unsigned int f : cell->face_indices())
            if (!cell->at_boundary(f) && cell->face(f)->has_children())
              {
                const auto face = cell->face(f);
                Tensor<1, spacedim> mean =
                  normals[face->vertex_index(0)] +
                  normals[face->vertex_index(1)];
                normals[face->child(0)->vertex_index(1)] = mean / mean.norm();
              }
    for (const auto &n : normals)
      AssertThrow(n.norm() == 0 || std::abs(n.norm() - 1) < 1e-12,
                  ExcMessage("The normals of the cells around a vertex "
                             "cancel: the mesh is not consistently "
                             "oriented."));
    Direction<spacedim> direction;
    direction.vertex_normals =
      std::make_shared<const std::vector<Tensor<1, spacedim>>>(
        std::move(normals));
    return direction;
  }



  template <int spacedim>
  template <int dim>
  std::pair<Tensor<1, spacedim>, Tensor<2, spacedim>>
  Direction<spacedim>::evaluate(const MeshPoint<dim, spacedim> &x) const
  {
    Assert(!is_closest_point(), ExcMessage("There is no direction."));
    Tensor<1, spacedim> d;
    Tensor<2, spacedim> J_d;
    if (vertex_normals)
      {
        // d = sum_v phi_v n_v, and its derivative along the cell,
        // M = sum_v n_v (grad phi_v)^T = J_d V, returned as M G^{-1} V^T.
        AssertThrow(x.cell.state() == IteratorState::valid,
                    ExcMessage("The interpolated normal of the mesh needs "
                               "MeshPoint::cell."));
        DerivativeForm<1, dim, spacedim> M;
        for (const unsigned int v : GeometryInfo<dim>::vertex_indices())
          {
            const Tensor<1, spacedim> &n =
              (*vertex_normals)[x.cell->vertex_index(v)];
            const double phi =
              GeometryInfo<dim>::d_linear_shape_function(x.unit_point, v);
            const Tensor<1, dim> gradient =
              GeometryInfo<dim>::d_linear_shape_function_gradient(x.unit_point,
                                                                  v);
            d += phi * n;
            for (unsigned int i = 0; i < spacedim; ++i)
              for (unsigned int a = 0; a < dim; ++a)
                M[i][a] += n[i] * gradient[a];
          }
        Tensor<2, dim> G;
        for (unsigned int a = 0; a < dim; ++a)
          for (unsigned int b = 0; b < dim; ++b)
            for (unsigned int i = 0; i < spacedim; ++i)
              G[a][b] += x.jacobian[i][a] * x.jacobian[i][b];
        if (determinant(G) > 0)
          {
            const Tensor<2, dim> G_inverse = invert(G);
            for (unsigned int i = 0; i < spacedim; ++i)
              for (unsigned int j = 0; j < spacedim; ++j)
                for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                    J_d[i][j] += M[i][a] * G_inverse[a][b] * x.jacobian[j][b];
          }
        return {d, J_d};
      }
    if (along_mesh_normal)
      {
        if constexpr (dim + 1 == spacedim)
          {
            d   = unit_normal(x.jacobian);
            J_d = unit_normal_jacobian(x.jacobian, x.hessian);
          }
        else
          AssertThrow(false,
                      ExcMessage("The mesh has no normal: its codimension "
                                 "is not one."));
      }
    else
      for (unsigned int i = 0; i < spacedim; ++i)
        {
          d[i]   = field->value(x.point, i);
          J_d[i] = field->gradient(x.point, i);
        }
    return {d, J_d};
  }



  template <int spacedim>
  UpdateFlags
  Direction<spacedim>::get_needed_update_flags() const
  {
    return along_mesh_normal ? update_jacobian_grads : update_default;
  }



  template <int dim, int spacedim>
  SphericalProjector<dim, spacedim>::SphericalProjector(
    const Point<spacedim>     &center,
    const double               radius,
    const Direction<spacedim> &direction)
    : center(center)
    , radius(radius)
    , direction(direction)
  {
    static_assert(dim + 1 == spacedim,
                  "The projector is implemented for spheres and circles.");
  }



  template <int dim, int spacedim>
  ProjectionResult<spacedim>
  SphericalProjector<dim, spacedim>::project(
    const MeshPoint<dim, spacedim> &x,
    typename Projector<dim, spacedim>::Cache &) const
  {
    ProjectionResult<spacedim> result;
    const Tensor<1, spacedim>  r        = x.point - center;
    const double               distance = r.norm();
    AssertThrow(distance > 0.,
                ExcMessage("The centre has no closest point on the sphere."));

    if (!direction.is_closest_point())
      {
        // z = x + zeta d with |z - c| = R: a zeta^2 + 2 b zeta + c0 = 0. The
        // root of smaller magnitude is c0/q, with q/a the other one,
        // computed without cancellation.
        const auto [d, J_d] = direction.evaluate(x);
        const double a      = d * d;
        const double b      = d * r;
        const double c0     = (distance - radius) * (distance + radius);
        const double discriminant = b * b - a * c0;
        if (discriminant >= 0.)
          {
            const double q = -(b + std::copysign(std::sqrt(discriminant), b));
            const double zeta = (q == 0.) ? 0. : c0 / q;
            result.point      = x.point + zeta * d;
            result.normal     = (result.point - center) / radius;
            result.jacobian =
              directional_projection_jacobian(d, result.normal, zeta, J_d);
            return result;
          }
        // The line misses the sphere: closest point instead.
        result.regular = false;
      }

    const Tensor<1, spacedim> e = r / distance;
    result.point                = center + radius * e;
    result.normal               = e;
    result.jacobian =
      (radius / distance) * (identity<spacedim>() - outer_product(e, e));
    return result;
  }



  template <int dim, int spacedim>
  UpdateFlags
  SphericalProjector<dim, spacedim>::get_needed_update_flags() const
  {
    return direction.get_needed_update_flags();
  }



  CylindricalProjector::CylindricalProjector(const Point<3>     &point_on_axis,
                                             const Tensor<1, 3> &axis_direction,
                                             const double        radius)
    : point_on_axis(point_on_axis)
    , axis(axis_direction / axis_direction.norm())
    , radius(radius)
  {}



  ProjectionResult<3>
  CylindricalProjector::project(const MeshPoint<2, 3> &x,
                                Projector<2, 3>::Cache &) const
  {
    const Tensor<1, 3> relative = x.point - point_on_axis;
    const double       height   = relative * axis;
    const Tensor<1, 3> r        = relative - height * axis;
    const double       distance = r.norm();
    AssertThrow(distance > 0.,
                ExcMessage("The axis has no closest point on the cylinder."));
    const Tensor<1, 3> e = r / distance;

    ProjectionResult<3> result;
    result.point    = point_on_axis + height * axis + radius * e;
    result.normal   = e;
    result.jacobian = outer_product(axis, axis) +
                      (radius / distance) * (identity<3>() -
                                             outer_product(axis, axis) -
                                             outer_product(e, e));
    return result;
  }



  template <int dim, int spacedim>
  void
  ProjectorCollection<dim, spacedim>::add(
    const types::manifold_id        manifold_id,
    const Projector<dim, spacedim> &projector)
  {
    AssertThrow(index.find(manifold_id) == index.end(),
                ExcMessage("The manifold id has a projector already."));
    index[manifold_id] = projectors.size();
    projectors.push_back(&projector);
  }



  namespace
  {
    template <int dim, int spacedim>
    struct CollectionCache : public Projector<dim, spacedim>::Cache
    {
      static constexpr unsigned int none = static_cast<unsigned int>(-1);

      std::vector<std::unique_ptr<typename Projector<dim, spacedim>::Cache>>
                   caches;
      unsigned int current = none;
    };
  } // namespace



  template <int dim, int spacedim>
  std::unique_ptr<typename Projector<dim, spacedim>::Cache>
  ProjectorCollection<dim, spacedim>::create_cache() const
  {
    auto cache = std::make_unique<CollectionCache<dim, spacedim>>();
    for (const auto projector : projectors)
      cache->caches.push_back(projector->create_cache());
    return cache;
  }



  template <int dim, int spacedim>
  void
  ProjectorCollection<dim, spacedim>::reinit(
    const Mapping<dim, spacedim>                               &mapping,
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    typename Projector<dim, spacedim>::Cache                   &cache) const
  {
    auto &data = static_cast<CollectionCache<dim, spacedim> &>(cache);
    const auto found = index.find(cell->manifold_id());
    data.current     = (found == index.end()) ? data.none : found->second;
    if (data.current != data.none)
      projectors[data.current]->reinit(mapping,
                                       cell,
                                       *data.caches[data.current]);
  }



  template <int dim, int spacedim>
  ProjectionResult<spacedim>
  ProjectorCollection<dim, spacedim>::project(
    const MeshPoint<dim, spacedim>           &x,
    typename Projector<dim, spacedim>::Cache &cache) const
  {
    auto &data = static_cast<CollectionCache<dim, spacedim> &>(cache);
    if (data.current != data.none)
      return projectors[data.current]->project(x, *data.caches[data.current]);

    ProjectionResult<spacedim> result;
    result.point    = x.point;
    result.jacobian = identity<spacedim>();
    if constexpr (dim + 1 == spacedim)
      result.normal = unit_normal(x.jacobian);
    return result;
  }



  template <int dim, int spacedim>
  UpdateFlags
  ProjectorCollection<dim, spacedim>::get_needed_update_flags() const
  {
    UpdateFlags flags = update_default;
    for (const auto projector : projectors)
      flags |= projector->get_needed_update_flags();
    return flags;
  }



  template <int dim, int spacedim>
  ProjectedQuadrature<dim, spacedim>::ProjectedQuadrature(
    const Mapping<dim, spacedim>   &mapping,
    const Quadrature<dim>          &quadrature,
    const Projector<dim, spacedim> &projector)
    : mapping(mapping)
    , quadrature(quadrature)
    , projector(projector)
    , cache(projector.create_cache())
    , fe()
    , fe_values(mapping,
                fe,
                quadrature,
                update_quadrature_points | update_jacobians |
                  update_JxW_values |
                  (dim + 1 == spacedim ? update_normal_vectors :
                                         update_default) |
                  projector.get_needed_update_flags())
    , projected_quadrature(quadrature)
    , points(quadrature.size())
    , JxW(quadrature.size())
    , normals(dim + 1 == spacedim ? quadrature.size() : 0)
    , jacobians(quadrature.size())
  {}



  template <int dim, int spacedim>
  void
  ProjectedQuadrature<dim, spacedim>::reinit(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell)
  {
    fe_values.reinit(cell);
    projector.reinit(mapping, cell, *cache);

    const bool with_hessians =
      (fe_values.get_update_flags() & update_jacobian_grads) != 0;
    std::vector<double> weights(quadrature.size());
    n_irregular = 0;
    for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        MeshPoint<dim, spacedim> x;
        x.cell       = cell;
        x.unit_point = quadrature.point(q);
        x.point      = fe_values.quadrature_point(q);
        x.jacobian = fe_values.jacobian(q);
        if (with_hessians)
          x.hessian = fe_values.jacobian_grad(q);

        const ProjectionResult<spacedim> result = projector.project(x, *cache);
        const double ratio = measure_ratio(result.jacobian, x.jacobian);

        weights[q]   = quadrature.weight(q) * ratio;
        JxW[q]       = fe_values.JxW(q) * ratio;
        points[q]    = result.point;
        jacobians[q] = result.jacobian;
        if constexpr (dim + 1 == spacedim)
          normals[q] = (result.normal * fe_values.normal_vector(q) < 0.) ?
                         -result.normal :
                         result.normal;
        if (!result.regular)
          ++n_irregular;
      }
    projected_quadrature = Quadrature<dim>(quadrature.get_points(), weights);
  }



  template Tensor<2, 2>
  closest_point_jacobian(const DerivativeForm<1, 1, 2> &,
                         const DerivativeForm<2, 1, 2> &,
                         const Tensor<1, 2> &);
  template Tensor<2, 3>
  closest_point_jacobian(const DerivativeForm<1, 1, 3> &,
                         const DerivativeForm<2, 1, 3> &,
                         const Tensor<1, 3> &);
  template Tensor<2, 3>
  closest_point_jacobian(const DerivativeForm<1, 2, 3> &,
                         const DerivativeForm<2, 2, 3> &,
                         const Tensor<1, 3> &);

  template Tensor<2, 2>
  directional_projection_jacobian(const Tensor<1, 2> &,
                                  const Tensor<1, 2> &,
                                  const double,
                                  const Tensor<2, 2> &);
  template Tensor<2, 3>
  directional_projection_jacobian(const Tensor<1, 3> &,
                                  const Tensor<1, 3> &,
                                  const double,
                                  const Tensor<2, 3> &);

  template Tensor<1, 2>
  unit_normal(const DerivativeForm<1, 1, 2> &);
  template Tensor<1, 3>
  unit_normal(const DerivativeForm<1, 2, 3> &);

  template Tensor<2, 2>
  unit_normal_jacobian(const DerivativeForm<1, 1, 2> &,
                       const DerivativeForm<2, 1, 2> &);
  template Tensor<2, 3>
  unit_normal_jacobian(const DerivativeForm<1, 2, 3> &,
                       const DerivativeForm<2, 2, 3> &);

  template double
  measure_ratio(const Tensor<2, 2> &, const DerivativeForm<1, 1, 2> &);
  template double
  measure_ratio(const Tensor<2, 3> &, const DerivativeForm<1, 1, 3> &);
  template double
  measure_ratio(const Tensor<2, 3> &, const DerivativeForm<1, 2, 3> &);

  template class Direction<2>;
  template class Direction<3>;
  template Direction<2>
  Direction<2>::interpolated_mesh_normal(const Triangulation<1, 2> &,
                                         const Mapping<1, 2> &);
  template Direction<3>
  Direction<3>::interpolated_mesh_normal(const Triangulation<2, 3> &,
                                         const Mapping<2, 3> &);
  template std::pair<Tensor<1, 2>, Tensor<2, 2>>
  Direction<2>::evaluate(const MeshPoint<1, 2> &) const;
  template std::pair<Tensor<1, 3>, Tensor<2, 3>>
  Direction<3>::evaluate(const MeshPoint<1, 3> &) const;
  template std::pair<Tensor<1, 3>, Tensor<2, 3>>
  Direction<3>::evaluate(const MeshPoint<2, 3> &) const;

  template class Projector<1, 2>;
  template class Projector<1, 3>;
  template class Projector<2, 3>;
  template class SphericalProjector<1, 2>;
  template class SphericalProjector<2, 3>;
  template class ProjectorCollection<1, 2>;
  template class ProjectorCollection<1, 3>;
  template class ProjectorCollection<2, 3>;
  template class ProjectedQuadrature<1, 2>;
  template class ProjectedQuadrature<1, 3>;
  template class ProjectedQuadrature<2, 3>;
} // namespace ExactGeometry

DEAL_II_NAMESPACE_CLOSE
