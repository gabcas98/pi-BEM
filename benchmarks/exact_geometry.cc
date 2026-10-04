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

// Accuracy and cost of the quadrature over the exact geometry
// (projected_quadrature.h, cad_projectors.h), written as CSV files to the
// directory given as the first argument:
//
//   exact_geometry_order.csv    sphere of 96 flat cells and circle of 32
//                               segments: relative errors of measure, volume
//                               and second moment against the number of
//                               Gauss points per direction, on the mesh and
//                               with the closest point, the normal of the
//                               mesh and a vector field;
//   exact_geometry_normal_check.csv  the projection along the normal of the
//                               mesh checked without its Jacobian: areas of
//                               the images by spherical subdivision, and the
//                               part of the sphere they leave uncovered;
//   exact_geometry_mapping.csv  sphere meshed with mappings of degree 1 to 4
//                               on the spherical manifold, levels 0 to 3:
//                               area errors with 12 x 12 points;
//   exact_geometry_cad.csv      OpenCASCADE shapes: agreement of the sphere
//                               and of its NURBS conversion with the analytic
//                               projector, torus, cylinder and curves;
//   exact_geometry_cost.csv     microseconds per point of the projectors and
//                               of the per-point projections of deal.II;
//   exact_geometry_trimmed.csv  trimmed faces (cylinder with caps, plate
//                               with a hole, spherical cap closed by a
//                               disk): area and volume errors against the
//                               number of Gauss points, on the mesh, with
//                               the projection alone and with the
//                               transfinite map along the trimming curves,
//                               for the closest point and along a field.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>

#include "projected_quadrature.h"

#ifdef DEAL_II_WITH_OPENCASCADE
#  include <deal.II/opencascade/utilities.h>

#  include <BRepAlgoAPI_Cut.hxx>
#  include <BRepBuilderAPI_MakeEdge.hxx>
#  include <BRepBuilderAPI_MakeFace.hxx>
#  include <BRepBuilderAPI_NurbsConvert.hxx>
#  include <BRepPrimAPI_MakeCylinder.hxx>
#  include <BRepPrimAPI_MakeSphere.hxx>
#  include <BRepPrimAPI_MakeTorus.hxx>
#  include <ElCLib.hxx>
#  include <gp.hxx>
#  include <gp_Ax2.hxx>
#  include <gp_Pln.hxx>
#  include <gp_Circ.hxx>
#  include <gp_Elips.hxx>

#  include "cad_projectors.h"
#endif

using namespace dealii;
using namespace dealii::ExactGeometry;

namespace
{
  double
  now()
  {
    return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }


  // d(x) = (I + s K)(x - c), K the rotation by a right angle about the last
  // axis in 2d and about the z-axis in 3d.
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


  // Measure, enclosed volume (1/spacedim) int (x - c) . n (codimension one)
  // and second moment of the last coordinate about c; cells for which
  // select(normal at the first point) is false are skipped.
  template <int dim, int spacedim>
  std::array<double, 3>
  integrate(const Triangulation<dim, spacedim> &tria,
            const Mapping<dim, spacedim>       &mapping,
            const Projector<dim, spacedim>     &projector,
            const unsigned int                  n_points,
            const Point<spacedim>              &c,
            const std::function<bool(const Tensor<1, spacedim> &)> &select =
              [](const Tensor<1, spacedim> &) { return true; })
  {
    ProjectedQuadrature<dim, spacedim> quadrature(mapping,
                                                  QGauss<dim>(n_points),
                                                  projector);
    std::array<double, 3>              result = {{0, 0, 0}};
    for (const auto &cell : tria.active_cell_iterators())
      {
        quadrature.reinit(cell);
        if constexpr (dim + 1 == spacedim)
          if (!select(quadrature.get_normal_vectors()[0]))
            continue;
        for (unsigned int q = 0; q < quadrature.get_points().size(); ++q)
          {
            const Tensor<1, spacedim> r  = quadrature.get_points()[q] - c;
            const double              dx = quadrature.get_JxW_values()[q];
            result[0] += dx;
            if constexpr (dim + 1 == spacedim)
              result[1] +=
                r * quadrature.get_normal_vectors()[q] / spacedim * dx;
            result[2] += r[spacedim - 1] * r[spacedim - 1] * dx;
          }
      }
    return result;
  }


  template <int dim, int spacedim>
  void
  order(std::ostream                       &out,
        const std::string                  &name,
        const Triangulation<dim, spacedim> &tria,
        const Point<spacedim>              &center,
        const double                        R,
        const std::array<double, 3>        &exact)
  {
    const MappingQ<dim, spacedim>            mapping(1);
    const ProjectorCollection<dim, spacedim> mesh;
    const SphericalProjector<dim, spacedim>  closest(center, R);
    const SphericalProjector<dim, spacedim>  normal(
      center, R, Direction<spacedim>::mesh_normal());
    const SwirlField<spacedim>              field(center, 0.5);
    const SphericalProjector<dim, spacedim> along_field(
      center, R, Direction<spacedim>(field));
    const SphericalProjector<dim, spacedim> along_vertex_normal(
      center,
      R,
      Direction<spacedim>::interpolated_mesh_normal(tria, mapping));
    const std::vector<std::pair<std::string, const Projector<dim, spacedim> *>>
      methods = {{"mesh", &mesh},
                 {"closest", &closest},
                 {"mesh_normal", &normal},
                 {"field", &along_field},
                 {"vertex_normal", &along_vertex_normal}};
    for (unsigned int n = 1; n <= 10; ++n)
      for (const auto &[method, projector] : methods)
        {
          const auto integrals =
            integrate(tria, mapping, *projector, n, center);
          out << name << ',' << tria.n_active_cells() << ',' << n << ','
              << method;
          for (unsigned int i = 0; i < 3; ++i)
            out << ',' << std::abs(integrals[i] / exact[i] - 1);
          out << '\n';
        }
  }


  void
  write_order(const std::string &directory)
  {
    std::ofstream out(directory + "/exact_geometry_order.csv");
    out << std::setprecision(4)
        << "geometry,cells,points,method,measure,volume,moment\n";
    {
      const Point<3>      center(0.1, -0.2, 0.3);
      const double        R = 1.3;
      Triangulation<2, 3> curved, tria;
      GridGenerator::hyper_sphere(curved, center, R);
      curved.refine_global(2);
      GridGenerator::flatten_triangulation(curved, tria);
      order<2, 3>(out,
                  "sphere",
                  tria,
                  center,
                  R,
                  {{4 * numbers::PI * R * R,
                    4. / 3 * numbers::PI * R * R * R,
                    4. / 3 * numbers::PI * std::pow(R, 4)}});
    }
    {
      const Point<2>      center(0.1, -0.2);
      const double        R = 1.3;
      Triangulation<1, 2> curved, tria;
      GridGenerator::hyper_sphere(curved, center, R);
      curved.refine_global(3);
      GridGenerator::flatten_triangulation(curved, tria);
      order<1, 2>(out,
                  "circle",
                  tria,
                  center,
                  R,
                  {{2 * numbers::PI * R,
                    numbers::PI * R * R,
                    numbers::PI * R * R * R}});
    }
  }


  // Area of the spherical triangle with vertices a, b, c relative to the
  // centre, |a| = |b| = |c| = R (Van Oosterom and Strackee).
  double
  spherical_triangle(const Tensor<1, 3> &a,
                     const Tensor<1, 3> &b,
                     const Tensor<1, 3> &c,
                     const double        R)
  {
    const double num = std::abs(a * cross_product_3d(b, c));
    const double den = R * R * R + (a * b) * R + (b * c) * R + (c * a) * R;
    return 2 * std::atan2(num, den) * R * R;
  }


  // The projection along the normal of the mesh onto a sphere of flat
  // cells, checked without its Jacobian: the area of the image of every cell
  // from a subdivision into spherical triangles, and the part of the sphere
  // that no image covers, by Monte Carlo (a point y is in the image of K
  // when its projection onto the plane of K along n_K falls inside K; the
  // cells are taken as planar, which they are to second order). Written to
  // exact_geometry_normal_check.csv with the estimate beta^2/6 of the gaps,
  // beta the mean angular size of the cells.
  void
  write_normal_check(const std::string &directory)
  {
    std::ofstream out(directory + "/exact_geometry_normal_check.csv");
    out << std::setprecision(4)
        << "level,cells,quadrature,subdivision,uncovered,overlapped,"
           "estimate\n";
    const Point<3> center(0.1, -0.2, 0.3);
    const double   R = 1.3, exact = 4 * numbers::PI * R * R;
    for (unsigned int level = 1; level <= 4; ++level)
      {
        Triangulation<2, 3> curved, tria;
        GridGenerator::hyper_sphere(curved, center, R);
        curved.refine_global(level);
        GridGenerator::flatten_triangulation(curved, tria);
        const MappingQ<2, 3>           mapping(1);
        const SphericalProjector<2, 3> along_normal(
          center, R, Direction<3>::mesh_normal());
        auto                   cache = along_normal.create_cache();
        const FE_Nothing<2, 3> fe;
        const unsigned int     N = 64;
        std::vector<Point<2>>  grid;
        for (unsigned int j = 0; j <= N; ++j)
          for (unsigned int i = 0; i <= N; ++i)
            grid.emplace_back(double(i) / N, double(j) / N);
        FEValues<2, 3> values(mapping,
                              fe,
                              Quadrature<2>(grid),
                              update_quadrature_points | update_jacobians);
        double subdivision = 0;
        std::vector<typename Triangulation<2, 3>::active_cell_iterator> cells;
        std::vector<Tensor<1, 3>> normals;
        for (const auto &cell : tria.active_cell_iterators())
          {
            cells.push_back(cell);
            values.reinit(cell);
            along_normal.reinit(mapping, cell, *cache);
            std::vector<Tensor<1, 3>> image(grid.size());
            for (unsigned int k = 0; k < grid.size(); ++k)
              {
                MeshPoint<2, 3> x;
                x.cell       = cell;
                x.unit_point = grid[k];
                x.point      = values.quadrature_point(k);
                x.jacobian   = values.jacobian(k);
                image[k]     = along_normal.project(x, *cache).point - center;
              }
            for (unsigned int j = 0; j < N; ++j)
              for (unsigned int i = 0; i < N; ++i)
                {
                  const auto &a = image[i + (N + 1) * j];
                  const auto &b = image[i + 1 + (N + 1) * j];
                  const auto &c = image[i + 1 + (N + 1) * (j + 1)];
                  const auto &d = image[i + (N + 1) * (j + 1)];
                  subdivision += spherical_triangle(a, b, c, R) +
                                 spherical_triangle(a, c, d, R);
                }
            normals.push_back(unit_normal(values.jacobian(0)));
          }
        const double quadrature =
          integrate(tria, mapping, along_normal, 10, center)[0];

        std::mt19937                     generator(level);
        std::normal_distribution<double> gauss(0, 1);
        const unsigned int               samples = 100000;
        unsigned int                     uncovered = 0, overlapped = 0;
        for (unsigned int s = 0; s < samples; ++s)
          {
            Tensor<1, 3> e({gauss(generator), gauss(generator),
                            gauss(generator)});
            const Point<3> y       = center + R * e / e.norm();
            unsigned int   covered = 0;
            for (unsigned int k = 0; k < cells.size(); ++k)
              {
                if (cells[k]->center().distance(y) > 1.5 * cells[k]->diameter())
                  continue;
                const Point<3> x =
                  y - ((y - cells[k]->vertex(0)) * normals[k]) * normals[k];
                Point<2> xi;
                try
                  {
                    xi = mapping.transform_real_to_unit_cell(cells[k], x);
                  }
                catch (...)
                  {
                    continue;
                  }
                if (xi[0] >= 0 && xi[0] <= 1 && xi[1] >= 0 && xi[1] <= 1)
                  ++covered;
              }
            if (covered == 0)
              ++uncovered;
            if (covered > 1)
              ++overlapped;
          }
        // beta: the mean angular size of the cells, sqrt(4 pi / cells).
        const double beta2 = 4 * numbers::PI / tria.n_active_cells();
        out << level << ',' << tria.n_active_cells() << ','
            << std::abs(quadrature / exact - 1) << ','
            << std::abs(subdivision / exact - 1) << ','
            << double(uncovered) / samples << ','
            << double(overlapped) / samples << ',' << beta2 / 6 << '\n';
      }
  }


  void
  write_mapping(const std::string &directory)
  {
    std::ofstream out(directory + "/exact_geometry_mapping.csv");
    out << std::setprecision(4)
        << "degree,level,cells,mesh,closest,mesh_normal,field,vertex_normal\n";
    const Point<3> c(0.1, -0.2, 0.3);
    const double   R = 1.3, exact = 4 * numbers::PI * R * R;
    for (unsigned int p = 1; p <= 4; ++p)
      for (unsigned int level = 0; level <= 3; ++level)
        {
          Triangulation<2, 3> tria;
          GridGenerator::hyper_sphere(tria, c, R);
          tria.refine_global(level);
          const MappingQ<2, 3>            mapping(p);
          const ProjectorCollection<2, 3> mesh;
          const SphericalProjector<2, 3>  closest(c, R);
          const SphericalProjector<2, 3>  normal(c,
                                                R,
                                                Direction<3>::mesh_normal());
          const SwirlField<3>             field(c, 0.5);
          const SphericalProjector<2, 3>  along_field(c,
                                                     R,
                                                     Direction<3>(field));
          const SphericalProjector<2, 3>  along_vertex_normal(
            c, R, Direction<3>::interpolated_mesh_normal(tria, mapping));
          out << p << ',' << level << ',' << tria.n_active_cells();
          for (const Projector<2, 3> *projector :
               {static_cast<const Projector<2, 3> *>(&mesh),
                static_cast<const Projector<2, 3> *>(&closest),
                static_cast<const Projector<2, 3> *>(&normal),
                static_cast<const Projector<2, 3> *>(&along_field),
                static_cast<const Projector<2, 3> *>(&along_vertex_normal)})
            out << ','
                << std::abs(
                     integrate(tria, mapping, *projector, 12, c)[0] / exact -
                     1);
          out << '\n';
        }
  }


#ifdef DEAL_II_WITH_OPENCASCADE
  // Largest differences of the points, the Jacobians and the JxW values of
  // two projectors.
  std::array<double, 3>
  compare(const Triangulation<2, 3> &tria,
          const Projector<2, 3>     &projector,
          const Projector<2, 3>     &reference)
  {
    const MappingQ<2, 3>      mapping(1);
    ProjectedQuadrature<2, 3> a(mapping, QGauss<2>(4), projector);
    ProjectedQuadrature<2, 3> b(mapping, QGauss<2>(4), reference);
    std::array<double, 3>     d = {{0, 0, 0}};
    for (const auto &cell : tria.active_cell_iterators())
      {
        a.reinit(cell);
        b.reinit(cell);
        for (unsigned int q = 0; q < a.get_points().size(); ++q)
          {
            d[0] =
              std::max(d[0], (a.get_points()[q] - b.get_points()[q]).norm());
            d[1] = std::max(d[1],
                            (a.get_projection_jacobians()[q] -
                             b.get_projection_jacobians()[q])
                              .norm());
            d[2] = std::max(d[2],
                            std::abs(a.get_JxW_values()[q] /
                                       b.get_JxW_values()[q] -
                                     1));
          }
      }
    return d;
  }


  Triangulation<1, 3>
  polygon(const std::vector<Point<3>> &vertices)
  {
    std::vector<CellData<1>> cells(vertices.size());
    for (unsigned int i = 0; i < vertices.size(); ++i)
      cells[i].vertices = {i,
                           static_cast<unsigned int>((i + 1) %
                                                     vertices.size())};
    Triangulation<1, 3> tria;
    tria.create_triangulation(vertices, cells, SubCellData());
    return tria;
  }


  void
  write_cad(const std::string &directory)
  {
    std::ofstream out(directory + "/exact_geometry_cad.csv");
    out << std::setprecision(4) << "case,cells,points,method,a,b,c\n";
    const MappingQ<2, 3> mapping(1);

    // Sphere and its NURBS conversion against the analytic projector:
    // points, Jacobians, JxW; area and volume with 8 x 8 points.
    {
      const Point<3>      center(0.1, -0.2, 0.3);
      const double        R = 1.3;
      Triangulation<2, 3> curved, tria;
      GridGenerator::hyper_sphere(curved, center, R);
      curved.refine_global(2);
      GridGenerator::flatten_triangulation(curved, tria);
      const TopoDS_Shape sphere =
        BRepPrimAPI_MakeSphere(gp_Pnt(center[0], center[1], center[2]), R)
          .Shape();
      const SwirlField<3> field(center, 0.5);
      for (const auto &[name, shape] :
           {std::make_pair(std::string("sphere"), sphere),
            std::make_pair(std::string("nurbs_sphere"),
                           BRepBuilderAPI_NurbsConvert(sphere).Shape())})
        {
          const OpenCASCADE::ShapeProjector<2> closest(shape);
          const OpenCASCADE::ShapeProjector<2> along_field(shape,
                                                           Direction<3>(
                                                             field));
          const auto d1 =
            compare(tria, closest, SphericalProjector<2, 3>(center, R));
          const auto d2 = compare(tria,
                                  along_field,
                                  SphericalProjector<2, 3>(center,
                                                           R,
                                                           Direction<3>(
                                                             field)));
          out << name << ',' << tria.n_active_cells() << ",4,closest,"
              << d1[0] << ',' << d1[1] << ',' << d1[2] << '\n'
              << name << ',' << tria.n_active_cells() << ",4,field," << d2[0]
              << ',' << d2[1] << ',' << d2[2] << '\n';
          const auto integrals =
            integrate(tria, mapping, closest, 8, Point<3>());
          out << name << "_integrals," << tria.n_active_cells()
              << ",8,closest,"
              << std::abs(integrals[0] / (4 * numbers::PI * R * R) - 1) << ','
              << std::abs(integrals[1] / (4. / 3 * numbers::PI * R * R * R) -
                          1)
              << ",0\n";
        }
    }

    // Torus: area and volume against the number of points.
    {
      const double        R = 2, r = 0.5;
      Triangulation<2, 3> curved, tria;
      GridGenerator::torus(curved, R, r);
      curved.refine_global(2);
      GridGenerator::flatten_triangulation(curved, tria);
      const TopoDS_Shape shape =
        BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), R, r)
          .Shape();
      const OpenCASCADE::ShapeProjector<2> projector(shape);
      const ProjectorCollection<2, 3>      mesh;
      for (unsigned int n = 1; n <= 10; ++n)
        for (const auto &[method, p] :
             {std::make_pair(std::string("mesh"),
                             static_cast<const Projector<2, 3> *>(&mesh)),
              std::make_pair(std::string("closest"),
                             static_cast<const Projector<2, 3> *>(
                               &projector))})
          {
            const auto integrals = integrate(tria, mapping, *p, n, Point<3>());
            out << "torus," << tria.n_active_cells() << ',' << n << ','
                << method << ','
                << std::abs(integrals[0] /
                              (4 * numbers::PI * numbers::PI * R * r) -
                            1)
                << ','
                << std::abs(integrals[1] /
                              (2 * numbers::PI * numbers::PI * R * r * r) -
                            1)
                << ",0\n";
          }
    }

    // Cylinder with its caps: lateral area, area of the caps against the
    // exact one and against the polygons of the mesh.
    {
      const double     R = 1, half_length = 1.5;
      Triangulation<3> volume, flat_volume;
      GridGenerator::cylinder(volume, R, half_length);
      volume.refine_global(2);
      GridGenerator::flatten_triangulation(volume, flat_volume);
      Triangulation<2, 3> tria;
      GridGenerator::extract_boundary_mesh(flat_volume, tria);
      const TopoDS_Shape shape =
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-half_length, 0, 0),
                                        gp_Dir(1, 0, 0)),
                                 R,
                                 2 * half_length)
          .Shape();
      const OpenCASCADE::ShapeProjector<2> projector(shape);
      const ProjectorCollection<2, 3>      mesh;
      const auto lateral = [](const Tensor<1, 3> &n) {
        return std::abs(n[0]) < 0.5;
      };
      const auto caps = [](const Tensor<1, 3> &n) {
        return std::abs(n[0]) > 0.5;
      };
      const double side =
        integrate<2, 3>(tria, mapping, projector, 8, Point<3>(), lateral)[0];
      const double side_mesh =
        integrate<2, 3>(tria, mapping, mesh, 8, Point<3>(), lateral)[0];
      const double cap =
        integrate<2, 3>(tria, mapping, projector, 8, Point<3>(), caps)[0];
      const double cap_mesh =
        integrate<2, 3>(tria, mapping, mesh, 8, Point<3>(), caps)[0];
      const double exact_side = 4 * numbers::PI * R * half_length,
                   exact_cap  = 2 * numbers::PI * R * R;
      out << "cylinder_lateral," << tria.n_active_cells() << ",8,closest,"
          << std::abs(side / exact_side - 1) << ','
          << std::abs(side_mesh / exact_side - 1) << ",0\n"
          << "cylinder_caps," << tria.n_active_cells() << ",8,closest,"
          << std::abs(cap / exact_cap - 1) << ','
          << std::abs(cap_mesh / exact_cap - 1) << ','
          << std::abs(cap / cap_mesh - 1) << '\n';
    }

    // Circle and ellipse of 24 segments in a tilted plane; the perimeter of
    // the ellipse by the series of Gauss and Kummer.
    {
      const unsigned int n = 24;
      const gp_Ax2       frame(gp_Pnt(0.1, 0.2, -0.3), gp_Dir(1, 1, 1));
      const gp_Circ      circle(frame, 1.3);
      const gp_Elips     ellipse(frame, 2., 1.);
      const double       h = std::pow(1. / 3, 2);
      double             perimeter = 0, coefficient = 1;
      for (unsigned int k = 0; k < 40; ++k)
        {
          perimeter += coefficient * coefficient * std::pow(h, k);
          coefficient *= (0.5 - k) / (k + 1);
        }
      perimeter *= numbers::PI * 3.;
      const MappingQ<1, 3> line_mapping(1);
      for (const bool is_circle : {true, false})
        {
          std::vector<Point<3>> vertices;
          for (unsigned int i = 0; i < n; ++i)
            {
              const double t = 2 * numbers::PI * i / n;
              const gp_Pnt p = is_circle ? ElCLib::Value(t, circle) :
                                           ElCLib::Value(t, ellipse);
              vertices.emplace_back(p.X(), p.Y(), p.Z());
            }
          const Triangulation<1, 3>            tria = polygon(vertices);
          const OpenCASCADE::ShapeProjector<1> projector(
            is_circle ? BRepBuilderAPI_MakeEdge(circle).Edge() :
                        BRepBuilderAPI_MakeEdge(ellipse).Edge());
          const ProjectorCollection<1, 3> mesh;
          const double exact = is_circle ? 2 * numbers::PI * 1.3 : perimeter;
          for (unsigned int q = 1; q <= 10; ++q)
            out << (is_circle ? "circle," : "ellipse,") << n << ',' << q
                << ",closest,"
                << std::abs(integrate(tria,
                                      line_mapping,
                                      projector,
                                      q,
                                      Point<3>())[0] /
                              exact -
                            1)
                << ','
                << std::abs(
                     integrate(tria, line_mapping, mesh, q, Point<3>())[0] /
                       exact -
                     1)
                << ",0\n";
        }
    }
  }


  void
  write_cost(const std::string &directory)
  {
    std::ofstream out(directory + "/exact_geometry_cost.csv");
    out << std::setprecision(4)
        << "shape,cells,points,closest,field,pull_back,line_intersection\n";
    const MappingQ<2, 3> mapping(1);

    // With the field d = x - c where it is transversal to the surface.
    const auto measure = [&](const std::string         &name,
                             const Triangulation<2, 3> &tria,
                             const TopoDS_Shape        &shape,
                             const Point<3>            &c,
                             const bool                 with_field) {
      const SwirlField<3> field(c, 0.);
      double              seconds[2] = {0, 0};
      unsigned int        n_points   = 0;
      for (unsigned int mode = 0; mode < (with_field ? 2 : 1); ++mode)
        {
          const OpenCASCADE::ShapeProjector<2> projector(
            shape, mode == 0 ? Direction<3>() : Direction<3>(field));
          ProjectedQuadrature<2, 3> quadrature(mapping,
                                               QGauss<2>(6),
                                               projector);
          n_points          = 0;
          const double t0   = now();
          for (const auto &cell : tria.active_cell_iterators())
            {
              quadrature.reinit(cell);
              n_points += quadrature.get_points().size();
            }
          seconds[mode] = now() - t0;
        }

      // The per-point projections of deal.II, on every 16th cell.
      const FE_Nothing<2, 3>  fe;
      FEValues<2, 3>          fe_values(mapping,
                               fe,
                               QGauss<2>(6),
                               update_quadrature_points);
      double                  pull_back = 0, intersection = 0;
      unsigned int            n_sampled = 0;
      for (const auto &cell : tria.active_cell_iterators())
        {
          if (cell->active_cell_index() % 16 != 0)
            continue;
          fe_values.reinit(cell);
          for (const auto &p : fe_values.get_quadrature_points())
            {
              double t0 = now();
              OpenCASCADE::project_point_and_pull_back(shape, p, 1e-7);
              pull_back += now() - t0;
              t0 = now();
              OpenCASCADE::line_intersection(shape, p, p - c, 1e-7);
              intersection += now() - t0;
              ++n_sampled;
            }
        }
      out << name << ',' << tria.n_active_cells() << ',' << n_points << ','
          << seconds[0] / n_points * 1e6 << ',';
      if (with_field)
        out << seconds[1] / n_points * 1e6;
      out << ',' << pull_back / n_sampled * 1e6 << ','
          << intersection / n_sampled * 1e6 << '\n';
    };

    {
      const Point<3>      c(0.1, -0.2, 0.3);
      const double        R = 1.3;
      Triangulation<2, 3> curved, tria;
      GridGenerator::hyper_sphere(curved, c, R);
      curved.refine_global(4);
      GridGenerator::flatten_triangulation(curved, tria);
      const TopoDS_Shape sphere =
        BRepPrimAPI_MakeSphere(gp_Pnt(c[0], c[1], c[2]), R).Shape();
      measure("sphere", tria, sphere, c, true);
      measure("nurbs_sphere",
              tria,
              BRepBuilderAPI_NurbsConvert(sphere).Shape(),
              c,
              true);
    }
    {
      const double        R = 2, r = 0.5;
      Triangulation<2, 3> curved, tria;
      GridGenerator::torus(curved, R, r);
      curved.refine_global(3);
      GridGenerator::flatten_triangulation(curved, tria);
      const TopoDS_Shape torus =
        BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), R, r)
          .Shape();
      measure("nurbs_torus",
              tria,
              BRepBuilderAPI_NurbsConvert(torus).Shape(),
              Point<3>(),
              false);
    }
  }


  // A mesh of a surface from a planar mesh, mapping its vertices; flip
  // reverses the orientation of the cells.
  template <typename Map>
  Triangulation<2, 3>
  lift(const Triangulation<2> &flat, const Map &map, const bool flip)
  {
    std::vector<Point<3>> vertices;
    for (const auto &v : flat.get_vertices())
      vertices.push_back(map(v));
    std::vector<CellData<2>> cells;
    for (const auto &cell : flat.active_cell_iterators())
      {
        CellData<2> c;
        c.vertices = {cell->vertex_index(0),
                      cell->vertex_index(flip ? 2 : 1),
                      cell->vertex_index(flip ? 1 : 2),
                      cell->vertex_index(3)};
        cells.push_back(c);
      }
    SubCellData subcell_data;
    GridTools::delete_unused_vertices(vertices, cells, subcell_data);
    Triangulation<2, 3> tria;
    tria.create_triangulation(vertices, cells, subcell_data);
    return tria;
  }


  void
  write_trimmed(const std::string &directory)
  {
    std::ofstream out(directory + "/exact_geometry_trimmed.csv");
    out << std::setprecision(4)
        << "case,cells,points,method,area,volume\n";
    const MappingQ<2, 3> mapping(1);

    const auto run = [&](const std::string         &name,
                         const Triangulation<2, 3> &tria,
                         const TopoDS_Shape        &shape,
                         const Point<3>            &c,
                         const double               area,
                         const double               volume) {
      OpenCASCADE::ShapeProjector<2> transfinite(shape);
      OpenCASCADE::ShapeProjector<2> projection(shape);
      projection.set_follow_trimming_curves(false);
      const SwirlField<3>             field(c, 0.3);
      OpenCASCADE::ShapeProjector<2>  along_field(shape, Direction<3>(field));
      const ProjectorCollection<2, 3> mesh;
      const std::vector<
        std::pair<std::string, const Projector<2, 3> *>>
        methods = {{"mesh", &mesh},
                   {"projection", &projection},
                   {"transfinite", &transfinite},
                   {"transfinite_field", &along_field}};
      for (unsigned int n = 1; n <= 10; ++n)
        for (const auto &[method, projector] : methods)
          {
            const auto integrals =
              integrate(tria, mapping, *projector, n, Point<3>());
            out << name << ',' << tria.n_active_cells() << ',' << n << ','
                << method << ',' << std::abs(integrals[0] / area - 1) << ','
                << (volume > 0 ? std::abs(integrals[1] / volume - 1) : 0.)
                << '\n';
          }
    };

    {
      const double     R = 1, half_length = 1.5;
      Triangulation<3> volume, flat_volume;
      GridGenerator::cylinder(volume, R, half_length);
      volume.refine_global(2);
      GridGenerator::flatten_triangulation(volume, flat_volume);
      Triangulation<2, 3> tria;
      GridGenerator::extract_boundary_mesh(flat_volume, tria);
      run("cylinder",
          tria,
          BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-half_length, 0, 0),
                                          gp_Dir(1, 0, 0)),
                                   R,
                                   2 * half_length)
            .Shape(),
          Point<3>(),
          4 * numbers::PI * R * half_length + 2 * numbers::PI * R * R,
          2 * numbers::PI * R * R * half_length);
    }
    {
      const double     r = 0.5, a = 1;
      Triangulation<2> curved, flat;
      GridGenerator::hyper_cube_with_cylindrical_hole(curved, r, a);
      curved.refine_global(2);
      GridGenerator::flatten_triangulation(curved, flat);
      const TopoDS_Shape square =
        BRepBuilderAPI_MakeFace(gp_Pln(gp::XOY()), -a, a, -a, a).Face();
      const TopoDS_Shape hole =
        BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)),
                                 r,
                                 2)
          .Shape();
      // The field d = (I + s K)(x - c) from a point above the plate.
      run("plate",
          lift(
            flat,
            [](const Point<2> &p) { return Point<3>(p[0], p[1], 0); },
            false),
          BRepAlgoAPI_Cut(square, hole).Shape(),
          Point<3>(0, 0, 2),
          4 * a * a - numbers::PI * r * r,
          0);
    }
    {
      const double R = 1.3, latitude = numbers::PI / 4;
      const double polar = numbers::PI / 2 - latitude;
      const double rho = R * std::sin(polar), z0 = R * std::cos(polar);
      Triangulation<2> curved, disk;
      GridGenerator::hyper_ball(curved, Point<2>(), rho);
      curved.refine_global(2);
      GridGenerator::flatten_triangulation(curved, disk);
      const Triangulation<2, 3> cap = lift(
        disk,
        [&](const Point<2> &p) {
          const double theta = polar * p.norm() / rho;
          const double phi   = std::atan2(p[1], p[0]);
          return Point<3>(R * std::sin(theta) * std::cos(phi),
                          R * std::sin(theta) * std::sin(phi),
                          R * std::cos(theta));
        },
        false);
      const Triangulation<2, 3> bottom = lift(
        disk,
        [&](const Point<2> &p) { return Point<3>(p[0], p[1], z0); },
        true);
      Triangulation<2, 3> tria;
      GridGenerator::merge_triangulations(cap, bottom, tria, 1e-10);
      const double h = R - z0;
      run("spherical_cap",
          tria,
          BRepPrimAPI_MakeSphere(R, latitude, numbers::PI / 2).Shape(),
          Point<3>(0, 0, 0.9),
          2 * numbers::PI * R * h + numbers::PI * rho * rho,
          numbers::PI * h * h * (3 * R - h) / 3);
    }
  }
#endif
} // namespace


int
main(int argc, char **argv)
{
  const std::string directory = (argc > 1) ? argv[1] : ".";
  write_order(directory);
  write_mapping(directory);
  write_normal_check(directory);
#ifdef DEAL_II_WITH_OPENCASCADE
  write_cad(directory);
  write_trimmed(directory);
  write_cost(directory);
#endif
  std::cout << "Written to " << directory << std::endl;
}
