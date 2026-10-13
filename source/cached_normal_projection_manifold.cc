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

#include "../include/cached_normal_projection_manifold.h"

#include <deal.II/opencascade/utilities.h>

#include <gp_Ax1.hxx>
#include <gp_Dir.hxx>
#include <gp_Lin.hxx>
#include <gp_Pnt.hxx>

#include <cstring>

DEAL_II_NAMESPACE_OPEN

CachedNormalToMeshProjectionManifold::CachedNormalToMeshProjectionManifold(
  const TopoDS_Shape &shape,
  const double        tolerance)
  : OpenCASCADE::NormalToMeshProjectionManifold<2, 3>(shape, tolerance)
  , shape(shape)
  , tolerance(tolerance)
{}



std::unique_ptr<Manifold<2, 3>>
CachedNormalToMeshProjectionManifold::clone() const
{
  return std::make_unique<CachedNormalToMeshProjectionManifold>(shape,
                                                                tolerance);
}



Tensor<1, 3>
CachedNormalToMeshProjectionManifold::normal_at(const Point<3> &point) const
{
  std::array<std::uint64_t, 3> key;
  for (unsigned int d = 0; d < 3; ++d)
    std::memcpy(&key[d], &point[d], sizeof(double));
  const auto found = normals.find(key);
  if (found != normals.end())
    return found->second;
  const Tensor<1, 3> normal = std::get<1>(
    OpenCASCADE::closest_point_and_differential_forms(shape, point, tolerance));
  normals.emplace(key, normal);
  return normal;
}



Point<3>
CachedNormalToMeshProjectionManifold::project_to_manifold(
  const ArrayView<const Point<3>> &surrounding_points,
  const Point<3>                  &candidate) const
{
  // The direction of OpenCASCADE::NormalToMeshProjectionManifold (deal.II
  // 9.7): the average normal of the shape at the ends of a line, normal to
  // the line; the average of the normals of the triangles of the vertices
  // of a quadrilateral, or of its vertices and midpoints; of all triangles
  // otherwise.
  const auto cross = [](const Tensor<1, 3> &u, const Tensor<1, 3> &v) {
    const double c[3] = {u[1] * v[2] - u[2] * v[1],
                         u[2] * v[0] - u[0] * v[2],
                         u[0] * v[1] - u[1] * v[0]};
    return Tensor<1, 3>(c);
  };
  const auto unit = [](const Tensor<1, 3> &n) { return n / n.norm(); };
  Tensor<1, 3> average_normal;
  const auto  &p = surrounding_points;
  switch (p.size())
    {
      case 2:
        {
          for (const auto &point : p)
            average_normal += normal_at(point);
          average_normal /= 2.0;
          Assert(average_normal.norm() > 1e-4,
                 ExcMessage("Null average normal at a line."));
          Tensor<1, 3> T = p[0] - p[1];
          T /= T.norm();
          average_normal = average_normal - (average_normal * T) * T;
          average_normal /= average_normal.norm();
          break;
        }
      case 4:
        {
          const Tensor<1, 3> n1 = unit(cross(p[1] - p[0], p[2] - p[0]));
          const Tensor<1, 3> n2 = unit(cross(p[2] - p[3], p[1] - p[3]));
          average_normal        = (n1 + n2) / 2.0;
          average_normal /= average_normal.norm();
          break;
        }
      case 8:
        {
          const Tensor<1, 3> n1 = unit(cross(p[1] - p[0], p[2] - p[0]));
          const Tensor<1, 3> n2 = unit(cross(p[2] - p[3], p[1] - p[3]));
          const Tensor<1, 3> n3 = unit(cross(p[4] - p[7], p[6] - p[7]));
          const Tensor<1, 3> n4 = unit(cross(p[6] - p[7], p[5] - p[7]));
          average_normal        = (n1 + n2 + n3 + n4) / 4.0;
          average_normal /= average_normal.norm();
          break;
        }
      default:
        {
          for (unsigned int i = 0; i < p.size(); ++i)
            for (unsigned int j = 0; j < p.size(); ++j)
              if (j != i)
                for (unsigned int k = 0; k < p.size(); ++k)
                  if (k != j && k != i)
                    {
                      Tensor<1, 3> n1 = cross(p[i] - p[j], p[i] - p[k]);
                      if (n1.norm() > tolerance)
                        {
                          n1 = n1 / n1.norm();
                          if (average_normal.norm() < tolerance)
                            average_normal = n1;
                          else if (n1 * average_normal > 0)
                            average_normal += n1;
                          else
                            average_normal -= n1;
                        }
                    }
          average_normal = average_normal / average_normal.norm();
          break;
        }
    }

  // OpenCASCADE::line_intersection with the intersector loaded once: the
  // intersection of the line through the candidate closest to it.
  if (!intersector)
    {
      intersector = std::make_unique<IntCurvesFace_ShapeIntersector>();
      intersector->Load(shape, tolerance);
    }
  const gp_Pnt P0(candidate[0], candidate[1], candidate[2]);
  const gp_Lin line(
    gp_Ax1(P0, gp_Dir(average_normal[0], average_normal[1], average_normal[2])));
  intersector->Perform(line, -RealLast(), +RealLast());
  AssertThrow(intersector->IsDone(), ExcMessage("Could not project point."));
  double   minimum = 1e7;
  Point<3> result;
  for (int i = 0; i < intersector->NbPnt(); ++i)
    {
      const double distance = P0.Distance(intersector->Pnt(i + 1));
      if (distance < minimum)
        {
          minimum = distance;
          result  = Point<3>(intersector->Pnt(i + 1).X(),
                            intersector->Pnt(i + 1).Y(),
                            intersector->Pnt(i + 1).Z());
        }
    }
  return result;
}

DEAL_II_NAMESPACE_CLOSE
