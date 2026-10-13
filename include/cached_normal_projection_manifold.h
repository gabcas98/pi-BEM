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

#ifndef cached_normal_projection_manifold_h
#define cached_normal_projection_manifold_h

#include <deal.II/base/point.h>
#include <deal.II/opencascade/manifold_lib.h>

#include <IntCurvesFace_ShapeIntersector.hxx>

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>

DEAL_II_NAMESPACE_OPEN

/**
 * OpenCASCADE::NormalToMeshProjectionManifold with the same new points,
 * computed faster: the intersector of the shape, which the library loads
 * anew for every point (the face classifiers of the shape), is loaded once,
 * and the normals of the shape at the vertices, which the midpoint of every
 * line needs at its two ends, are kept for the vertices met before. The
 * global refinement of a mesh on CAD surfaces spends most of its time
 * there.
 */
class CachedNormalToMeshProjectionManifold
  : public OpenCASCADE::NormalToMeshProjectionManifold<2, 3>
{
public:
  CachedNormalToMeshProjectionManifold(const TopoDS_Shape &shape,
                                       const double        tolerance = 1e-7);

  virtual std::unique_ptr<Manifold<2, 3>>
  clone() const override;

  virtual Point<3>
  project_to_manifold(const ArrayView<const Point<3>> &surrounding_points,
                      const Point<3> &candidate) const override;

private:
  /// The normal of the shape at its closest point to the given one.
  Tensor<1, 3>
  normal_at(const Point<3> &point) const;

  struct Hash
  {
    std::size_t
    operator()(const std::array<std::uint64_t, 3> &key) const
    {
      std::uint64_t h = 0x9e3779b97f4a7c15ull;
      for (const auto k : key)
        {
          h ^= k + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
          h *= 0xbf58476d1ce4e5b9ull;
        }
      return h;
    }
  };

  const TopoDS_Shape shape;
  const double       tolerance;
  mutable std::unique_ptr<IntCurvesFace_ShapeIntersector> intersector;
  mutable std::unordered_map<std::array<std::uint64_t, 3>, Tensor<1, 3>, Hash>
    normals;
};

DEAL_II_NAMESPACE_CLOSE

#endif
