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

#ifndef projected_mapping_h
#define projected_mapping_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/mutex.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <array>
#include <memory>
#include <vector>

#include "projected_quadrature.h"

DEAL_II_NAMESPACE_OPEN

namespace ExactGeometry
{
  /**
   * A mapping onto the exact geometry: each cell is the image, under a
   * Projector, of the cell of a polynomial mapping of the mesh,
   * Phi_K = P o chi_K, with the derivative D Phi = J V of
   * ProjectedQuadrature. Everything that evaluates the geometry through the
   * Mapping interface then works on the exact geometry: FEValues,
   * FEFaceValues and FESubfaceValues return the projected quadrature
   * points, the Jacobians J V and their pseudo-inverses, the surface
   * elements of the exact geometry and its normals, oriented like those of
   * the mesh, and transform the gradients of the shape functions with them;
   * NonMatching::MappingInfo and FEPointEvaluation go through the same
   * fill_fe_values(); support points, transform_unit_to_real_cell() and
   * its inverse are those of the exact geometry. The finite element spaces
   * stay those of the mesh, on its reference cells.
   *
   * The class derives from MappingQ, whose evaluation it runs first for the
   * mapping of the mesh. It does not preserve vertex locations, so that the
   * library does not take shortcuts that assume the polynomial geometry.
   * Second derivatives of the mapping (update_jacobian_grads) are central
   * differences of the exact first derivatives in the reference
   * coordinates, accurate to about 1e-8 relative; higher derivatives are
   * not available.
   *
   * The projector must outlive the mapping. Each FEValues object owns the
   * cache of the projector in its internal data; the functions that do not
   * go through internal data, such as transform_unit_to_real_cell(), share
   * one cache under a mutex.
   */
  template <int dim, int spacedim>
  class ProjectedMapping : public MappingQ<dim, spacedim>
  {
  public:
    ProjectedMapping(const unsigned int              degree,
                     const Projector<dim, spacedim> &projector);

    ProjectedMapping(const ProjectedMapping<dim, spacedim> &other);

    virtual std::unique_ptr<Mapping<dim, spacedim>>
    clone() const override;

    virtual bool
    preserves_vertex_locations() const override;

    virtual Point<spacedim>
    transform_unit_to_real_cell(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<dim> &p) const override;

    /// Newton's method on the exact map, from the inverse of the mapping of
    /// the mesh.
    virtual Point<dim>
    transform_real_to_unit_cell(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const Point<spacedim> &p) const override;

    virtual void
    transform_points_real_to_unit_cell(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const ArrayView<const Point<spacedim>>                     &real_points,
      const ArrayView<Point<dim>> &unit_points) const override;

    virtual Point<spacedim>
    get_center(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const bool map_barycenter_of_reference_cell = true) const override;

    /// A box that contains the exact cell: that of the mapping of the mesh
    /// and of the projections of a grid of points, enlarged by a percent.
    virtual BoundingBox<spacedim>
    get_bounding_box(const typename Triangulation<dim, spacedim>::cell_iterator
                       &cell) const override;

    const Projector<dim, spacedim> &
    get_projector() const
    {
      return *projector;
    }

    /**
     * The mapping of the mesh, a MappingQ of the same degree, for code that
     * evaluates the polynomial geometry itself and projects it with an
     * Evaluator.
     */
    const MappingQ<dim, spacedim> &
    get_mesh_mapping() const
    {
      return *mesh_mapping;
    }

    /**
     * Projection of points of a cell given with the mapping of the mesh,
     * for code that evaluates that mapping itself. Like FEValues, one object
     * serves one thread. It keeps the projector prepared for the last few
     * cells, since loops over pairs of cells alternate between two.
     */
    class Evaluator
    {
    public:
      Evaluator(const ProjectedMapping<dim, spacedim> &mapping);

      /// Prepare the cell; nothing to do if it is the current one.
      void
      reinit(const typename Triangulation<dim, spacedim>::cell_iterator &cell);

      /// The image y of the point x = chi(unit_point) of the current cell,
      /// with mesh Jacobian V, and the derivative D = J V of the exact map.
      void
      project(const Point<dim>                       &unit_point,
              const Point<spacedim>                  &x,
              const DerivativeForm<1, dim, spacedim> &V,
              Point<spacedim>                        &y,
              DerivativeForm<1, dim, spacedim>       &D) const;

    private:
      static constexpr unsigned int          n_slots = 4;
      const ProjectedMapping<dim, spacedim> &mapping;
      std::array<std::unique_ptr<typename Projector<dim, spacedim>::Cache>,
                 n_slots>
        caches;
      std::array<typename Triangulation<dim, spacedim>::cell_iterator, n_slots>
                   cells;
      unsigned int current = 0, next = 0;
    };

  protected:
    virtual UpdateFlags
    requires_update_flags(const UpdateFlags update_flags) const override;

    virtual std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    get_data(const UpdateFlags      update_flags,
             const Quadrature<dim> &quadrature) const override;

    using Mapping<dim, spacedim>::get_face_data;

    virtual std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    get_face_data(const UpdateFlags               update_flags,
                  const hp::QCollection<dim - 1> &quadrature) const override;

    virtual std::unique_ptr<typename Mapping<dim, spacedim>::InternalDataBase>
    get_subface_data(const UpdateFlags          update_flags,
                     const Quadrature<dim - 1> &quadrature) const override;

    virtual CellSimilarity::Similarity
    fill_fe_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const CellSimilarity::Similarity                         cell_similarity,
      const Quadrature<dim>                                   &quadrature,
      const typename Mapping<dim, spacedim>::InternalDataBase &internal_data,
      internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
        &output_data) const override;

    using Mapping<dim, spacedim>::fill_fe_face_values;

    virtual void
    fill_fe_face_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const unsigned int                                          face_no,
      const hp::QCollection<dim - 1>                             &quadrature,
      const typename Mapping<dim, spacedim>::InternalDataBase    &internal_data,
      internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
        &output_data) const override;

    virtual void
    fill_fe_subface_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const unsigned int                                          face_no,
      const unsigned int                                          subface_no,
      const Quadrature<dim - 1>                                  &quadrature,
      const typename Mapping<dim, spacedim>::InternalDataBase    &internal_data,
      internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
        &output_data) const override;

  private:
    /// The internal data of MappingQ with the cache of the projector.
    class InternalData : public MappingQ<dim, spacedim>::InternalData
    {
    public:
      InternalData(const ProjectedMapping<dim, spacedim> &mapping);

      mutable std::unique_ptr<typename Projector<dim, spacedim>::Cache> cache;
    };

    /**
     * Replace the geometry of the mesh in output_data, evaluated by MappingQ
     * at the given points of the reference cell, by the exact one. For
     * faces, the face JxW values, normals and boundary forms are rescaled
     * with the reference tangent of the face.
     */
    void
    project_output(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const std::vector<Point<dim>>                              &unit_points,
      const std::vector<double>                                  &weights,
      const unsigned int                                          face_no,
      const InternalData                                         &data,
      internal::FEValuesImplementation::MappingRelatedData<dim, spacedim>
        &output_data) const;

    /// The mapping of the mesh and the exact map at points of a cell, with
    /// a cache that has been reinit for it.
    void
    evaluate(const typename Triangulation<dim, spacedim>::cell_iterator &cell,
             const std::vector<Point<dim>>                 &unit_points,
             typename Projector<dim, spacedim>::Cache      &cache,
             std::vector<Point<spacedim>>                  &points,
             std::vector<DerivativeForm<1, dim, spacedim>> &jacobians) const;

    const Projector<dim, spacedim>                *projector;
    std::unique_ptr<const MappingQ<dim, spacedim>> mesh_mapping;

    /// The cache of the functions without internal data.
    mutable std::unique_ptr<typename Projector<dim, spacedim>::Cache>
                           shared_cache;
    mutable Threads::Mutex shared_cache_mutex;
  };

  /**
   * Fill a MappingQCache with the exact geometry: its support points, the
   * Gauss-Lobatto points of its degree n, are the images under the
   * projected mapping, so that every cell is mapped by the polynomial
   * interpolant of degree n of its exact map. The interpolants of two cells
   * agree on their common edge, which only the support points on it
   * determine, since the exact maps do. They converge exponentially in n,
   * the exact maps being analytic on each cell, and cost what a MappingQ of
   * degree n costs, with no projection after the initialization, which
   * projects (n+1)^dim points per cell once.
   *
   * Returns the largest deviation of the interpolant from the exact map at
   * the points of QGauss(n + 1) on the active cells: of the points,
   * relative to the diameter of the cell, and of the surface elements,
   * relative to themselves. The cache is cleared when the triangulation
   * changes, and must then be filled again.
   *
   * The processes of the communicator, each with the whole triangulation,
   * share the work: each projects the support points of every P-th cell
   * and checks every P-th active cell, and all of them receive all support
   * points and the largest deviations, the same as from a single process.
   */
  template <int dim, int spacedim>
  std::pair<double, double>
  interpolate_exact_geometry(MappingQCache<dim, spacedim>          &cache,
                             const ProjectedMapping<dim, spacedim> &exact,
                             const Triangulation<dim, spacedim> &triangulation,
                             const MPI_Comm communicator = MPI_COMM_SELF);

  /**
   * The largest deviation of the interpolant of degree n of the exact map
   * from the exact map, as in interpolate_exact_geometry(), estimated on
   * every stride-th active cell of the given one of n_parts contiguous
   * parts of the active cells, without building a MappingQCache: the
   * interpolant is evaluated from the shape functions of FE_Q(n) at the
   * images of its support points. Cheap enough to choose the degree. The
   * cells of a part follow one another, so that the projections of a cell
   * start from those of a close one (scattered cells cost the projector
   * thirty times as much).
   */
  template <int dim, int spacedim>
  std::pair<double, double>
  estimate_interpolation_deviation(
    const ProjectedMapping<dim, spacedim> &exact,
    const Triangulation<dim, spacedim>    &triangulation,
    const unsigned int                     degree,
    const unsigned int                     stride,
    const unsigned int                     part    = 0,
    const unsigned int                     n_parts = 1);
} // namespace ExactGeometry

DEAL_II_NAMESPACE_CLOSE

#endif
