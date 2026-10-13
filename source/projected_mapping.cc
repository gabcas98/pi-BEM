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

#include "projected_mapping.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_related_data.h>

#include <cmath>
#include <limits>
#include <mutex>

DEAL_II_NAMESPACE_OPEN

namespace ExactGeometry
{
  namespace
  {
    // D = J V.
    template <int dim, int spacedim>
    DerivativeForm<1, dim, spacedim>
    compose(const Tensor<2, spacedim>              &J,
            const DerivativeForm<1, dim, spacedim> &V)
    {
      DerivativeForm<1, dim, spacedim> D;
      for (unsigned int i = 0; i < spacedim; ++i)
        for (unsigned int a = 0; a < dim; ++a)
          for (unsigned int j = 0; j < spacedim; ++j)
            D[i][a] += J[i][j] * V[j][a];
      return D;
    }

    template <int dim, int spacedim>
    double
    surface_element(const DerivativeForm<1, dim, spacedim> &D)
    {
      Tensor<2, dim> G;
      for (unsigned int a = 0; a < dim; ++a)
        for (unsigned int b = 0; b < dim; ++b)
          for (unsigned int i = 0; i < spacedim; ++i)
            G[a][b] += D[i][a] * D[i][b];
      return std::sqrt(determinant(G));
    }

    template <int dim, int spacedim>
    Tensor<1, spacedim>
    column(const DerivativeForm<1, dim, spacedim> &D, const unsigned int a)
    {
      Tensor<1, spacedim> t;
      for (unsigned int i = 0; i < spacedim; ++i)
        t[i] = D[i][a];
      return t;
    }
  } // namespace



  template <int dim, int spacedim>
  ProjectedMapping<dim, spacedim>::InternalData::InternalData(
    const ProjectedMapping<dim, spacedim> &mapping)
    : MappingQ<dim, spacedim>::InternalData(mapping.fe_dgq_1d)
    , cache(mapping.projector->create_cache())
  {}



  template <int dim, int spacedim>
  ProjectedMapping<dim, spacedim>::ProjectedMapping(
    const unsigned int              degree,
    const Projector<dim, spacedim> &projector)
    : MappingQ<dim, spacedim>(degree)
    , projector(&projector)
    , mesh_mapping(std::make_unique<MappingQ<dim, spacedim>>(degree))
    , shared_cache(projector.create_cache())
  {}



  template <int dim, int spacedim>
  ProjectedMapping<dim, spacedim>::ProjectedMapping(
    const ProjectedMapping<dim, spacedim> &other)
    : MappingQ<dim, spacedim>(other)
    , projector(other.projector)
    , mesh_mapping(
        std::make_unique<MappingQ<dim, spacedim>>(other.polynomial_degree))
    , shared_cache(other.projector->create_cache())
  {}



  template <int dim, int spacedim>
  std::unique_ptr<Mapping<dim, spacedim>>
  ProjectedMapping<dim, spacedim>::clone() const
  {
    return std::make_unique<ProjectedMapping<dim, spacedim>>(*this);
  }



  template <int dim, int spacedim>
  bool
  ProjectedMapping<dim, spacedim>::preserves_vertex_locations() const
  {
    return false;
  }



  template <int dim, int spacedim>
  UpdateFlags
  ProjectedMapping<dim, spacedim>::requires_update_flags(
    const UpdateFlags in) const
  {
    const UpdateFlags geometric =
      update_quadrature_points | update_jacobians | update_JxW_values |
      update_normal_vectors | update_inverse_jacobians |
      update_covariant_transformation | update_contravariant_transformation |
      update_boundary_forms | update_jacobian_grads | update_volume_elements;
    UpdateFlags out = in;
    if (in & geometric)
      out |= update_quadrature_points | update_jacobians |
             projector->get_needed_update_flags();
    Assert(!(in & (update_jacobian_pushed_forward_grads |
                   update_jacobian_2nd_derivatives |
                   update_jacobian_pushed_forward_2nd_derivatives |
                   update_jacobian_3rd_derivatives |
                   update_jacobian_pushed_forward_3rd_derivatives)),
           ExcMessage("The projected mapping provides first and second "
                      "derivatives only."));
    return MappingQ<dim, spacedim>::requires_update_flags(out);
  }



  template <int dim, int spacedim>
  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  ProjectedMapping<dim, spacedim>::get_data(
    const UpdateFlags      update_flags,
    const Quadrature<dim> &quadrature) const
  {
    auto data = std::make_unique<InternalData>(*this);
    data->reinit(update_flags, quadrature);
    return data;
  }



  template <int dim, int spacedim>
  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  ProjectedMapping<dim, spacedim>::get_face_data(
    const UpdateFlags               update_flags,
    const hp::QCollection<dim - 1> &quadrature) const
  {
    AssertDimension(quadrature.size(), 1);
    auto data = std::make_unique<InternalData>(*this);
    data->initialize_face(this->requires_update_flags(update_flags),
                          QProjector<dim>::project_to_all_faces(
                            ReferenceCells::get_hypercube<dim>(), quadrature),
                          quadrature[0].size());
    return data;
  }



  template <int dim, int spacedim>
  std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
  ProjectedMapping<dim, spacedim>::get_subface_data(
    const UpdateFlags          update_flags,
    const Quadrature<dim - 1> &quadrature) const
  {
    auto data = std::make_unique<InternalData>(*this);
    data->initialize_face(this->requires_update_flags(update_flags),
                          QProjector<dim>::project_to_all_subfaces(
                            ReferenceCells::get_hypercube<dim>(), quadrature),
                          quadrature.size());
    return data;
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::evaluate(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const std::vector<Point<dim>>                              &unit_points,
    typename Projector<dim, spacedim>::Cache                   &cache,
    std::vector<Point<spacedim>>                               &points,
    std::vector<DerivativeForm<1, dim, spacedim>>              &jacobians) const
  {
    const Quadrature<dim> quadrature(unit_points);
    const UpdateFlags     flags = MappingQ<dim, spacedim>::requires_update_flags(
      update_quadrature_points | update_jacobians |
      projector->get_needed_update_flags());
    const auto base_data = MappingQ<dim, spacedim>::get_data(flags, quadrature);
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim> output;
    output.initialize(unit_points.size(), flags);
    MappingQ<dim, spacedim>::fill_fe_values(
      cell, CellSimilarity::none, quadrature, *base_data, output);

    points.resize(unit_points.size());
    jacobians.resize(unit_points.size());
    for (unsigned int q = 0; q < unit_points.size(); ++q)
      {
        MeshPoint<dim, spacedim> x;
        x.cell       = cell;
        x.unit_point = unit_points[q];
        x.point      = output.quadrature_points[q];
        x.jacobian   = output.jacobians[q];
        if (flags & update_jacobian_grads)
          x.hessian = output.jacobian_grads[q];
        const ProjectionResult<spacedim> r = projector->project(x, cache);
        points[q]                          = r.point;
        jacobians[q]                       = compose(r.jacobian, x.jacobian);
      }
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::project_output(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const std::vector<Point<dim>>                              &unit_points,
    const std::vector<double>                                  &weights,
    const unsigned int                                          face_no,
    const InternalData                                         &data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output) const
  {
    const UpdateFlags flags = data.update_each;
    if (!(flags & update_quadrature_points) || !(flags & update_jacobians))
      return;
    const bool on_face = (face_no != numbers::invalid_unsigned_int);

    projector->reinit(*mesh_mapping, cell, *data.cache);
    const unsigned int n = unit_points.size();
    for (unsigned int q = 0; q < n; ++q)
      {
        MeshPoint<dim, spacedim> x;
        x.cell       = cell;
        x.unit_point = unit_points[q];
        x.point      = output.quadrature_points[q];
        x.jacobian   = output.jacobians[q];
        if (flags & update_jacobian_grads)
          x.hessian = output.jacobian_grads[q];
        const ProjectionResult<spacedim> r = projector->project(x, *data.cache);
        const DerivativeForm<1, dim, spacedim> D = compose(r.jacobian, x.jacobian);

        // The unit normal of the exact geometry, oriented like the mesh.
        Tensor<1, spacedim> normal;
        if constexpr (dim + 1 == spacedim)
          {
            normal = unit_normal(D);
            if (normal * unit_normal(x.jacobian) < 0)
              normal = -normal;
            if (!cell->direction_flag())
              normal = -normal;
          }

        if (!on_face)
          {
            if (flags & update_JxW_values)
              output.JxW_values[q] = weights[q] * surface_element(D);
            if constexpr (dim + 1 == spacedim)
              if (flags & update_normal_vectors)
                output.normal_vectors[q] = normal;
          }
        else
          {
            // Faces: the measure of the face scales with the image of its
            // reference tangent, and the normal of the face is the
            // conormal in the tangent plane of the exact geometry, with the
            // orientation of the mesh.
            Tensor<1, spacedim> conormal;
            double              scale = 1.;
            if constexpr (dim == 2)
              {
                Tensor<1, dim> t;
                t[face_no < 2 ? 1 : 0] = 1.;
                Tensor<1, spacedim> mesh_tangent, tangent;
                for (unsigned int i = 0; i < spacedim; ++i)
                  for (unsigned int a = 0; a < dim; ++a)
                    {
                      mesh_tangent[i] += x.jacobian[i][a] * t[a];
                      tangent[i] += D[i][a] * t[a];
                    }
                scale = tangent.norm() / mesh_tangent.norm();
                if constexpr (spacedim == 3)
                  conormal = cross_product_3d(tangent, normal);
              }
            else
              conormal = column(D, 0);
            if (conormal.norm() > 0)
              conormal /= conormal.norm();
            if (flags & update_normal_vectors)
              {
                if (conormal * output.normal_vectors[q] < 0)
                  conormal = -conormal;
                output.normal_vectors[q] = conormal;
              }
            if (flags & update_boundary_forms)
              {
                const double size = output.boundary_forms[q].norm() * scale;
                if (conormal * output.boundary_forms[q] < 0)
                  conormal = -conormal;
                output.boundary_forms[q] = size * conormal;
              }
            if (flags & update_JxW_values)
              output.JxW_values[q] *= scale;
          }

        output.quadrature_points[q] = r.point;
        output.jacobians[q]         = D;
        // MappingQ keeps the covariant transformation of the shape
        // gradients in the inverse Jacobians of the output.
        if ((flags & (update_inverse_jacobians |
                      update_covariant_transformation)) &&
            q < output.inverse_jacobians.size())
          output.inverse_jacobians[q] = D.covariant_form().transpose();
        if ((flags & update_volume_elements) && q < data.volume_elements.size())
          data.volume_elements[q] = surface_element(D);
      }

    // Second derivatives: central differences of the exact first ones.
    if (flags & update_jacobian_grads)
      {
        const double            h = 1e-4;
        std::vector<Point<dim>> shifted;
        for (unsigned int q = 0; q < n; ++q)
          for (unsigned int b = 0; b < dim; ++b)
            for (const double sign : {1., -1.})
              {
                Point<dim> p = unit_points[q];
                p[b] += sign * h;
                shifted.push_back(p);
              }
        std::vector<Point<spacedim>>                  points;
        std::vector<DerivativeForm<1, dim, spacedim>> jacobians;
        evaluate(cell, shifted, *data.cache, points, jacobians);
        for (unsigned int q = 0; q < n; ++q)
          {
            DerivativeForm<2, dim, spacedim> grad;
            for (unsigned int b = 0; b < dim; ++b)
              {
                const auto &plus  = jacobians[(q * dim + b) * 2];
                const auto &minus = jacobians[(q * dim + b) * 2 + 1];
                for (unsigned int i = 0; i < spacedim; ++i)
                  for (unsigned int a = 0; a < dim; ++a)
                    grad[i][a][b] = (plus[i][a] - minus[i][a]) / (2 * h);
              }
            for (unsigned int i = 0; i < spacedim; ++i)
              for (unsigned int a = 0; a < dim; ++a)
                for (unsigned int b = 0; b < a; ++b)
                  grad[i][a][b] = grad[i][b][a] =
                    0.5 * (grad[i][a][b] + grad[i][b][a]);
            output.jacobian_grads[q] = grad;
          }
      }
  }



  template <int dim, int spacedim>
  CellSimilarity::Similarity
  ProjectedMapping<dim, spacedim>::fill_fe_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const CellSimilarity::Similarity,
    const Quadrature<dim>                                   &quadrature,
    const typename Mapping<dim, spacedim>::InternalDataBase &internal_data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output_data) const
  {
    MappingQ<dim, spacedim>::fill_fe_values(
      cell, CellSimilarity::none, quadrature, internal_data, output_data);
    project_output(cell,
                   quadrature.get_points(),
                   quadrature.get_weights(),
                   numbers::invalid_unsigned_int,
                   static_cast<const InternalData &>(internal_data),
                   output_data);
    return CellSimilarity::none;
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::fill_fe_face_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const hp::QCollection<dim - 1>                             &quadrature,
    const typename Mapping<dim, spacedim>::InternalDataBase    &internal_data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output_data) const
  {
    MappingQ<dim, spacedim>::fill_fe_face_values(
      cell, face_no, quadrature, internal_data, output_data);
    const Quadrature<dim> projected =
      QProjector<dim>::project_to_face(ReferenceCells::get_hypercube<dim>(),
                                       quadrature[0],
                                       face_no,
                                       cell->combined_face_orientation(
                                         face_no));
    project_output(cell,
                   projected.get_points(),
                   projected.get_weights(),
                   face_no,
                   static_cast<const InternalData &>(internal_data),
                   output_data);
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::fill_fe_subface_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const unsigned int                                          subface_no,
    const Quadrature<dim - 1>                                  &quadrature,
    const typename Mapping<dim, spacedim>::InternalDataBase    &internal_data,
    internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
      &output_data) const
  {
    MappingQ<dim, spacedim>::fill_fe_subface_values(
      cell, face_no, subface_no, quadrature, internal_data, output_data);
    if constexpr (dim > 1)
      {
        const Quadrature<dim> projected = QProjector<dim>::project_to_subface(
          ReferenceCells::get_hypercube<dim>(),
          quadrature,
          face_no,
          subface_no,
          cell->combined_face_orientation(face_no),
          RefinementCase<dim - 1>::isotropic_refinement);
        project_output(cell,
                       projected.get_points(),
                       projected.get_weights(),
                       face_no,
                       static_cast<const InternalData &>(internal_data),
                       output_data);
      }
  }



  template <int dim, int spacedim>
  Point<spacedim>
  ProjectedMapping<dim, spacedim>::transform_unit_to_real_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Point<dim>                                           &p) const
  {
    std::lock_guard<std::mutex>                   lock(shared_cache_mutex);
    std::vector<Point<spacedim>>                  points;
    std::vector<DerivativeForm<1, dim, spacedim>> jacobians;
    projector->reinit(*mesh_mapping, cell, *shared_cache);
    evaluate(cell, {p}, *shared_cache, points, jacobians);
    return points[0];
  }



  template <int dim, int spacedim>
  Point<dim>
  ProjectedMapping<dim, spacedim>::transform_real_to_unit_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Point<spacedim>                                      &p) const
  {
    Point<dim> xi;
    for (unsigned int d = 0; d < dim; ++d)
      xi[d] = 0.5;
    try
      {
        xi = MappingQ<dim, spacedim>::transform_real_to_unit_cell(cell, p);
      }
    catch (...)
      {}

    std::lock_guard<std::mutex>                   lock(shared_cache_mutex);
    std::vector<Point<spacedim>>                  points;
    std::vector<DerivativeForm<1, dim, spacedim>> jacobians;
    projector->reinit(*mesh_mapping, cell, *shared_cache);
    const double tolerance = 1e-12 * cell->diameter();
    for (unsigned int iteration = 0; iteration < 30; ++iteration)
      {
        evaluate(cell, {xi}, *shared_cache, points, jacobians);
        const DerivativeForm<1, dim, spacedim> &D = jacobians[0];
        const Tensor<1, spacedim>               r = p - points[0];
        Tensor<2, dim>                          G;
        Tensor<1, dim>                          g;
        for (unsigned int a = 0; a < dim; ++a)
          for (unsigned int i = 0; i < spacedim; ++i)
            {
              g[a] += D[i][a] * r[i];
              for (unsigned int b = 0; b < dim; ++b)
                G[a][b] += D[i][a] * D[i][b];
            }
        const Tensor<1, dim> delta = invert(G) * g;
        xi += delta;
        double step = 0;
        for (unsigned int i = 0; i < spacedim; ++i)
          {
            double s = 0;
            for (unsigned int a = 0; a < dim; ++a)
              s += D[i][a] * delta[a];
            step += s * s;
          }
        if (std::sqrt(step) <= tolerance)
          return xi;
      }
    AssertThrow(false, (typename Mapping<dim, spacedim>::ExcTransformationFailed()));
    return xi;
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::transform_points_real_to_unit_cell(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const ArrayView<const Point<spacedim>>                     &real_points,
    const ArrayView<Point<dim>>                                &unit_points) const
  {
    AssertDimension(real_points.size(), unit_points.size());
    for (unsigned int i = 0; i < real_points.size(); ++i)
      try
        {
          unit_points[i] = transform_real_to_unit_cell(cell, real_points[i]);
        }
      catch (typename Mapping<dim, spacedim>::ExcTransformationFailed &)
        {
          unit_points[i][0] = std::numeric_limits<double>::infinity();
        }
  }



  template <int dim, int spacedim>
  Point<spacedim>
  ProjectedMapping<dim, spacedim>::get_center(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const bool map_barycenter_of_reference_cell) const
  {
    if (!map_barycenter_of_reference_cell)
      return MappingQ<dim, spacedim>::get_center(cell, false);
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.5;
    return transform_unit_to_real_cell(cell, center);
  }



  template <int dim, int spacedim>
  BoundingBox<spacedim>
  ProjectedMapping<dim, spacedim>::get_bounding_box(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell) const
  {
    const unsigned int      n = this->polynomial_degree + 3;
    std::vector<Point<dim>> grid;
    for (unsigned int j = 0; j < (dim > 1 ? n : 1); ++j)
      for (unsigned int i = 0; i < n; ++i)
        {
          Point<dim> p;
          p[0] = i / (n - 1.);
          if constexpr (dim > 1)
            p[1] = j / (n - 1.);
          grid.push_back(p);
        }
    std::vector<Point<spacedim>>                  points;
    std::vector<DerivativeForm<1, dim, spacedim>> jacobians;
    {
      std::lock_guard<std::mutex> lock(shared_cache_mutex);
      projector->reinit(*mesh_mapping, cell, *shared_cache);
      evaluate(cell, grid, *shared_cache, points, jacobians);
    }
    for (const unsigned int v : cell->vertex_indices())
      points.push_back(cell->vertex(v));
    BoundingBox<spacedim> box(points);
    box.extend(0.01 * box.side_length(0) + 1e-12);
    return box;
  }



  template <int dim, int spacedim>
  ProjectedMapping<dim, spacedim>::Evaluator::Evaluator(
    const ProjectedMapping<dim, spacedim> &mapping)
    : mapping(mapping)
  {
    for (auto &cache : caches)
      cache = mapping.projector->create_cache();
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::Evaluator::reinit(
    const typename Triangulation<dim, spacedim>::cell_iterator &new_cell)
  {
    for (unsigned int s = 0; s < n_slots; ++s)
      if (cells[s].state() == IteratorState::valid && cells[s] == new_cell)
        {
          current = s;
          return;
        }
    current = next;
    next    = (next + 1) % n_slots;
    mapping.projector->reinit(*mapping.mesh_mapping, new_cell, *caches[current]);
    cells[current] = new_cell;
  }



  template <int dim, int spacedim>
  void
  ProjectedMapping<dim, spacedim>::Evaluator::project(
    const Point<dim>                       &unit_point,
    const Point<spacedim>                  &x,
    const DerivativeForm<1, dim, spacedim> &V,
    Point<spacedim>                        &y,
    DerivativeForm<1, dim, spacedim>       &D) const
  {
    MeshPoint<dim, spacedim> m;
    m.cell                             = cells[current];
    m.unit_point                       = unit_point;
    m.point                            = x;
    m.jacobian                         = V;
    const ProjectionResult<spacedim> r =
      mapping.projector->project(m, *caches[current]);
    y                                  = r.point;
    D                                  = compose(r.jacobian, V);
  }



  template <int dim, int spacedim>
  std::pair<double, double>
  interpolate_exact_geometry(MappingQCache<dim, spacedim>          &cache,
                             const ProjectedMapping<dim, spacedim> &exact,
                             const Triangulation<dim, spacedim> &triangulation,
                             const MPI_Comm                      communicator)
  {
    const unsigned int degree = cache.get_degree();
    const unsigned int n_processes =
      Utilities::MPI::n_mpi_processes(communicator);
    const unsigned int rank = Utilities::MPI::this_mpi_process(communicator);
    // FE_Q orders its support points, the Gauss-Lobatto points, as
    // MappingQ orders its own: vertices, lines, interior.
    const FE_Q<dim, spacedim>       support_element(degree);
    const Quadrature<dim>           support(
      support_element.get_unit_support_points());
    const unsigned int              n_points = support.size();
    const FE_Nothing<dim, spacedim> nothing;

    // MappingQCache keeps the support points of the cells of all levels.
    // Each process projects those of every n_processes-th cell, in the
    // order of the iterators, into a table indexed by level and index that
    // is zero elsewhere; the sum over the processes completes it.
    std::vector<std::size_t> level_start(triangulation.n_levels() + 1, 0);
    for (unsigned int l = 0; l < triangulation.n_levels(); ++l)
      level_start[l + 1] = level_start[l] + triangulation.n_raw_cells(l);
    const auto position =
      [&](const typename Triangulation<dim, spacedim>::cell_iterator &cell) {
        return (level_start[cell->level()] + cell->index()) * n_points *
               spacedim;
      };
    std::vector<double> coordinates(level_start.back() * n_points * spacedim,
                                    0.);
    {
      FEValues<dim, spacedim> values(exact,
                                     nothing,
                                     support,
                                     update_quadrature_points);
      unsigned int            index = 0;
      for (const auto &cell : triangulation.cell_iterators())
        if (index++ % n_processes == rank)
          {
            values.reinit(cell);
            double *c = &coordinates[position(cell)];
            for (unsigned int q = 0; q < n_points; ++q)
              for (unsigned int d = 0; d < spacedim; ++d)
                c[q * spacedim + d] = values.quadrature_point(q)[d];
          }
    }
    if (n_processes > 1)
      Utilities::MPI::sum(coordinates, communicator, coordinates);
    cache.initialize(
      triangulation,
      [&](const typename Triangulation<dim, spacedim>::cell_iterator &cell) {
        const double                *c = &coordinates[position(cell)];
        std::vector<Point<spacedim>> points(n_points);
        for (unsigned int q = 0; q < n_points; ++q)
          for (unsigned int d = 0; d < spacedim; ++d)
            points[q][d] = c[q * spacedim + d];
        return points;
      });

    const QGauss<dim>       check(degree + 1);
    FEValues<dim, spacedim> interpolated(cache,
                                         nothing,
                                         check,
                                         update_quadrature_points |
                                           update_JxW_values);
    FEValues<dim, spacedim> projected(exact,
                                      nothing,
                                      check,
                                      update_quadrature_points |
                                        update_JxW_values);
    double       point = 0, element = 0;
    unsigned int index = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        if (index++ % n_processes != rank)
          continue;
        interpolated.reinit(cell);
        projected.reinit(cell);
        const double diameter = cell->diameter();
        for (unsigned int q = 0; q < check.size(); ++q)
          {
            point   = std::max(point,
                             interpolated.quadrature_point(q).distance(
                               projected.quadrature_point(q)) /
                               diameter);
            element = std::max(element,
                               std::abs(interpolated.JxW(q) /
                                          projected.JxW(q) -
                                        1));
          }
      }
    return {Utilities::MPI::max(point, communicator),
            Utilities::MPI::max(element, communicator)};
  }



  template <int dim, int spacedim>
  std::pair<double, double>
  estimate_interpolation_deviation(
    const ProjectedMapping<dim, spacedim> &exact,
    const Triangulation<dim, spacedim>    &triangulation,
    const unsigned int                     degree,
    const unsigned int                     stride,
    const unsigned int                     part,
    const unsigned int                     n_parts)
  {
    const FE_Q<dim, spacedim>       element(degree);
    const Quadrature<dim>           support(element.get_unit_support_points());
    const QGauss<dim>               check(degree + 1);
    const FE_Nothing<dim, spacedim> nothing;
    FEValues<dim, spacedim>         at_support(exact,
                                       nothing,
                                       support,
                                       update_quadrature_points);
    FEValues<dim, spacedim>         at_check(exact,
                                     nothing,
                                     check,
                                     update_quadrature_points |
                                       update_JxW_values);
    // The shape values and reference gradients of FE_Q at the check points.
    const unsigned int n = element.n_dofs_per_cell();
    Table<2, double>   values(check.size(), n);
    Table<2, Tensor<1, dim>> gradients(check.size(), n);
    for (unsigned int q = 0; q < check.size(); ++q)
      for (unsigned int k = 0; k < n; ++k)
        {
          values(q, k)    = element.shape_value(k, check.point(q));
          gradients(q, k) = element.shape_grad(k, check.point(q));
        }
    double             point = 0, surface = 0;
    unsigned int       index = 0;
    const unsigned long n_cells = triangulation.n_active_cells();
    const unsigned int  begin   = part * n_cells / n_parts;
    const unsigned int  end     = (part + 1) * n_cells / n_parts;
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const unsigned int i = index++;
        if (i < begin || i >= end || (i - begin) % stride != 0)
          continue;
        at_support.reinit(cell);
        at_check.reinit(cell);
        const double diameter = cell->diameter();
        for (unsigned int q = 0; q < check.size(); ++q)
          {
            Point<spacedim>                  x;
            DerivativeForm<1, dim, spacedim> D;
            for (unsigned int k = 0; k < n; ++k)
              {
                const Point<spacedim> &X = at_support.quadrature_point(k);
                x += values(q, k) * X;
                for (unsigned int i = 0; i < spacedim; ++i)
                  for (unsigned int a = 0; a < dim; ++a)
                    D[i][a] += X[i] * gradients(q, k)[a];
              }
            point   = std::max(point,
                             x.distance(at_check.quadrature_point(q)) /
                               diameter);
            surface = std::max(surface,
                               std::abs(surface_element(D) * check.weight(q) /
                                          at_check.JxW(q) -
                                        1));
          }
      }
    return {point, surface};
  }



  template class ProjectedMapping<1, 2>;
  template class ProjectedMapping<2, 3>;

  template std::pair<double, double>
  estimate_interpolation_deviation(const ProjectedMapping<1, 2> &,
                                   const Triangulation<1, 2> &,
                                   const unsigned int,
                                   const unsigned int,
                                   const unsigned int,
                                   const unsigned int);
  template std::pair<double, double>
  estimate_interpolation_deviation(const ProjectedMapping<2, 3> &,
                                   const Triangulation<2, 3> &,
                                   const unsigned int,
                                   const unsigned int,
                                   const unsigned int,
                                   const unsigned int);

  template std::pair<double, double>
  interpolate_exact_geometry(MappingQCache<1, 2> &,
                             const ProjectedMapping<1, 2> &,
                             const Triangulation<1, 2> &,
                             const MPI_Comm);
  template std::pair<double, double>
  interpolate_exact_geometry(MappingQCache<2, 3> &,
                             const ProjectedMapping<2, 3> &,
                             const Triangulation<2, 3> &,
                             const MPI_Comm);
} // namespace ExactGeometry

DEAL_II_NAMESPACE_CLOSE
