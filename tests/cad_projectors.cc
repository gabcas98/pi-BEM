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

// Projections onto OpenCASCADE shapes and the integrals over them, from
// meshes of flat cells:
//  - a sphere, as built by OpenCASCADE and converted to NURBS: points and
//    Jacobians against the analytic projector, and the area and volume;
//  - a torus: area and volume against the number of Gauss points;
//  - trimmed faces: a cylinder with its two planar caps, a square plate
//    with a circular hole, and a spherical cap closed by a disk. The
//    projection alone leaves out the parts of the faces between the cells
//    and the trimming curves; the transfinite map of the cells along the
//    curves covers them, with Jacobians checked against central
//    differences;
//  - a surface of two faces: the lateral surface of a cylinder split into
//    two halves, each converted to NURBS so that its surface ends at its
//    edges, under a mesh whose cells straddle the two junctions, whose
//    points take the closest of the two faces, and the knots of the NURBS,
//    across which the parametric map takes the projection;
//  - curves: a circle and an ellipse in space, and the two rims of the
//    cylinder, picked out of the edges of the solid.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <BRepAlgoAPI_Cut.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <gp_Ax3.hxx>
#include <gp_Cylinder.hxx>
#include <gp.hxx>
#include <gp_Ax2.hxx>
#include <gp_Pln.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <ElCLib.hxx>

#include "cad_projectors.h"
#include "tests.h"

using namespace ExactGeometry;


std::string
format(const double error)
{
  if (std::abs(error) < 1e-13)
    return "<1e-13 ";
  std::ostringstream out;
  out << std::scientific << std::setprecision(1) << std::abs(error);
  return out.str();
}


// The field d(x) = (I + s K)(x - c), K the rotation by a right angle about
// the z-axis: radial plus a swirl.
class SwirlField : public Function<3>
{
public:
  SwirlField(const Point<3> &center, const double swirl)
    : Function<3>(3)
    , center(center)
  {
    for (unsigned int i = 0; i < 3; ++i)
      M[i][i] = 1;
    M[0][1] = -swirl;
    M[1][0] = swirl;
  }

  virtual double
  value(const Point<3> &p, const unsigned int i) const override
  {
    return (M * (p - center))[i];
  }

  virtual Tensor<1, 3>
  gradient(const Point<3> &, const unsigned int i) const override
  {
    return M[i];
  }

private:
  const Point<3> center;
  Tensor<2, 3>   M;
};


// Measure, and for surfaces the volume (1/3) int x . n, with the given
// projector; the integrals over the cells for which select(cell, normal at
// the first point) holds.
template <int dim>
std::array<double, 2>
integrate(
  const Triangulation<dim, 3> &tria,
  const Projector<dim, 3>     &projector,
  const unsigned int           n_points,
  unsigned int                &n_irregular,
  const std::function<bool(const Tensor<1, 3> &)> &select =
    [](const Tensor<1, 3> &) { return true; })
{
  const MappingQ<dim, 3>     mapping(1);
  ProjectedQuadrature<dim, 3> quadrature(mapping,
                                         QGauss<dim>(n_points),
                                         projector);
  std::array<double, 2> result = {{0, 0}};
  for (const auto &cell : tria.active_cell_iterators())
    {
      quadrature.reinit(cell);
      if constexpr (dim == 2)
        if (!select(quadrature.get_normal_vectors()[0]))
          continue;
      n_irregular += quadrature.n_irregular_points();
      for (unsigned int q = 0; q < quadrature.get_points().size(); ++q)
        {
          const double dx = quadrature.get_JxW_values()[q];
          result[0] += dx;
          if constexpr (dim == 2)
            result[1] += quadrature.get_points()[q] *
                         quadrature.get_normal_vectors()[q] / 3. * dx;
        }
    }
  return result;
}


// The largest differences of the projected points and of the Jacobians of
// two projectors.
void
compare(const std::string         &name,
        const Triangulation<2, 3> &tria,
        const Projector<2, 3>     &projector,
        const Projector<2, 3>     &reference)
{
  const MappingQ<2, 3>      mapping(1);
  ProjectedQuadrature<2, 3> a(mapping, QGauss<2>(4), projector);
  ProjectedQuadrature<2, 3> b(mapping, QGauss<2>(4), reference);
  double                    point = 0, jacobian = 0, weight = 0;
  unsigned int              n_irregular = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      a.reinit(cell);
      b.reinit(cell);
      n_irregular += a.n_irregular_points();
      for (unsigned int q = 0; q < a.get_points().size(); ++q)
        {
          point =
            std::max(point, (a.get_points()[q] - b.get_points()[q]).norm());
          jacobian = std::max(jacobian,
                              (a.get_projection_jacobians()[q] -
                               b.get_projection_jacobians()[q])
                                .norm());
          weight = std::max(weight,
                            std::abs(a.get_JxW_values()[q] /
                                       b.get_JxW_values()[q] -
                                     1));
        }
    }
  deallog << name << ": largest differences to the analytic projection: "
          << "points " << format(point) << ", Jacobians " << format(jacobian)
          << ", JxW " << format(weight) << "; irregular projections "
          << n_irregular << std::endl;
}


void
sphere()
{
  const Point<3>      center(0.1, -0.2, 0.3);
  const double        R = 1.3;
  Triangulation<2, 3> curved, tria;
  GridGenerator::hyper_sphere(curved, center, R);
  curved.refine_global(2);
  GridGenerator::flatten_triangulation(curved, tria);

  const TopoDS_Shape occ_sphere =
    BRepPrimAPI_MakeSphere(gp_Pnt(center[0], center[1], center[2]), R).Shape();
  const TopoDS_Shape nurbs_sphere =
    BRepBuilderAPI_NurbsConvert(occ_sphere).Shape();
  const SwirlField field(center, 0.5);

  for (const auto &[name, shape] :
       {std::make_pair(std::string("sphere"), occ_sphere),
        std::make_pair(std::string("NURBS sphere"), nurbs_sphere)})
    {
      const OpenCASCADE::ShapeProjector<2> closest(shape);
      const OpenCASCADE::ShapeProjector<2> along_field(shape,
                                                       Direction<3>(field));
      const MappingQ<2, 3> mapping(1);
      const Direction<3>   vertex_normal =
        Direction<3>::interpolated_mesh_normal(tria, mapping);
      const OpenCASCADE::ShapeProjector<2> along_vertex_normal(shape,
                                                               vertex_normal);
      compare(name + ", interpolated normal",
              tria,
              along_vertex_normal,
              SphericalProjector<2, 3>(center, R, vertex_normal));
      compare(name + ", closest point",
              tria,
              closest,
              SphericalProjector<2, 3>(center, R));
      compare(name + ", vector field",
              tria,
              along_field,
              SphericalProjector<2, 3>(center, R, Direction<3>(field)));

      unsigned int n_irregular = 0;
      const auto   integrals   = integrate(tria, closest, 8, n_irregular);
      deallog << name << ", 8 x 8 points: relative errors of area "
              << format(integrals[0] / (4 * numbers::PI * R * R) - 1)
              << ", volume "
              << format(integrals[1] / (4. / 3 * numbers::PI * R * R * R) - 1)
              << std::endl;
    }
}


void
torus()
{
  const double        R = 2, r = 0.5;
  Triangulation<2, 3> curved, tria;
  GridGenerator::torus(curved, R, r);
  curved.refine_global(2);
  GridGenerator::flatten_triangulation(curved, tria);

  // The torus of deal.II has its centre line in the xz-plane.
  const TopoDS_Shape shape =
    BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)), R, r)
      .Shape();
  const OpenCASCADE::ShapeProjector<2> projector(shape);
  const ProjectorCollection<2, 3>      mesh;

  deallog << "torus, " << tria.n_active_cells()
          << " flat cells; relative errors of area and volume" << std::endl
          << "points   mesh               projected" << std::endl;
  unsigned int n_irregular = 0;
  for (unsigned int n = 2; n <= 10; n += 2)
    {
      const auto on_mesh = integrate(tria, mesh, n, n_irregular);
      const auto exact   = integrate(tria, projector, n, n_irregular);
      deallog << std::setw(2) << n << "     ";
      for (const auto &integrals : {on_mesh, exact})
        deallog << format(integrals[0] /
                            (4 * numbers::PI * numbers::PI * R * r) -
                          1)
                << " "
                << format(integrals[1] /
                            (2 * numbers::PI * numbers::PI * R * r * r) -
                          1)
                << "    ";
      deallog << std::endl;
    }
  deallog << "irregular projections: " << n_irregular << std::endl;
}


// The derivative J V of the map of the cells with edges on trimming curves,
// or of all cells, against central differences in the reference
// coordinates, at one point of every such cell.
void
check_transfinite(const std::string                    &name,
                  const Triangulation<2, 3>            &tria,
                  const OpenCASCADE::ShapeProjector<2> &projector,
                  const bool                            all_cells = false)
{
  const MappingQ<2, 3>  mapping(1);
  const FE_Nothing<2, 3> fe;
  auto                   cache   = projector.create_cache();
  unsigned int           n_cells = 0, n_edges = 0;
  double                 error   = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      projector.reinit(mapping, cell, *cache);
      if (projector.n_trimmed_edges(*cache) == 0 && !all_cells)
        continue;
      ++n_cells;
      n_edges += projector.n_trimmed_edges(*cache);
      const auto project = [&](const Point<2> &p) {
        FEValues<2, 3> fe_values(mapping,
                                 fe,
                                 Quadrature<2>(p),
                                 update_quadrature_points | update_jacobians);
        fe_values.reinit(cell);
        MeshPoint<2, 3> x;
        x.cell       = cell;
        x.unit_point = p;
        x.point      = fe_values.quadrature_point(0);
        x.jacobian   = fe_values.jacobian(0);
        return std::make_pair(projector.project(x, *cache), x);
      };
      const Point<2> xi(0.31, 0.77);
      const auto [result, x] = project(xi);
      const double h         = 1e-4;
      for (unsigned int a = 0; a < 2; ++a)
        {
          Point<2> xp = xi, xm = xi;
          xp[a] += h;
          xm[a] -= h;
          const Tensor<1, 3> fd =
            (project(xp).first.point - project(xm).first.point) / (2 * h);
          Tensor<1, 3> JV;
          for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
              JV[i] += result.jacobian[i][j] * x.jacobian[j][a];
          error = std::max(error, (JV - fd).norm() / JV.norm());
        }
    }
  deallog << name << ": " << n_cells << " cells with " << n_edges
          << " edges on trimming curves; difference of J V to central "
             "differences "
          << (error < 1e-6 ? std::string("below 1e-6") : format(error))
          << std::endl;
}


// Area and volume with the projection alone and with the transfinite map
// along the trimming curves, against the exact values.
void
trimmed(const std::string         &name,
        const Triangulation<2, 3> &tria,
        const TopoDS_Shape        &shape,
        const double               area,
        const double               volume)
{
  OpenCASCADE::ShapeProjector<2> projector(shape);
  OpenCASCADE::ShapeProjector<2> projection_only(shape);
  projection_only.set_follow_trimming_curves(false);
  const MappingQ<2, 3>           mapping(1);
  OpenCASCADE::ShapeProjector<2> along_vertex_normal(
    shape, Direction<3>::interpolated_mesh_normal(tria, mapping));
  OpenCASCADE::ShapeProjector<2> parametric(shape);
  parametric.set_parametric(true);
  const ProjectorCollection<2, 3> mesh;
  deallog << name << ": " << projector.n_patches() << " faces, "
          << tria.n_active_cells()
          << " flat cells; relative errors of area and volume, 8 x 8 points"
          << std::endl;
  unsigned int n_irregular = 0;
  for (const auto &[method, p] :
       {std::make_pair(std::string("mesh"),
                       static_cast<const Projector<2, 3> *>(&mesh)),
        std::make_pair(std::string("projection"),
                       static_cast<const Projector<2, 3> *>(&projection_only)),
        std::make_pair(std::string("trimming curves"),
                       static_cast<const Projector<2, 3> *>(&projector)),
        std::make_pair(std::string("trimming curves, interpolated normal"),
                       static_cast<const Projector<2, 3> *>(
                         &along_vertex_normal)),
        std::make_pair(std::string("parametric"),
                       static_cast<const Projector<2, 3> *>(&parametric))})
    {
      const auto integrals = integrate(tria, *p, 8, n_irregular);
      deallog << "  " << method << ": " << format(integrals[0] / area - 1);
      if (volume > 0)
        deallog << ", " << format(integrals[1] / volume - 1);
      deallog << std::endl;
    }
  deallog << "  irregular projections: " << n_irregular << std::endl;
  check_transfinite(name, tria, projector);
  check_transfinite(name + ", interpolated normal", tria, along_vertex_normal);
  check_transfinite(name + ", parametric", tria, parametric, true);
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
cylinder()
{
  const double     R = 1, half_length = 1.5;
  Triangulation<3> volume, flat_volume;
  GridGenerator::cylinder(volume, R, half_length);
  volume.refine_global(2);
  GridGenerator::flatten_triangulation(volume, flat_volume);
  Triangulation<2, 3> tria;
  GridGenerator::extract_boundary_mesh(flat_volume, tria);

  // Along the x-axis, as the cylinder of deal.II: a lateral face and two
  // planar caps, trimmed by circles.
  const TopoDS_Shape shape =
    BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-half_length, 0, 0),
                                    gp_Dir(1, 0, 0)),
                             R,
                             2 * half_length)
      .Shape();
  trimmed("cylinder",
          tria,
          shape,
          4 * numbers::PI * R * half_length + 2 * numbers::PI * R * R,
          2 * numbers::PI * R * R * half_length);
}


void
plate()
{
  // A square of side 2 with a hole of radius 0.5: a planar face trimmed by
  // a circle inside.
  const double     r = 0.5, a = 1;
  Triangulation<2> curved, flat;
  GridGenerator::hyper_cube_with_cylindrical_hole(curved, r, a);
  curved.refine_global(2);
  GridGenerator::flatten_triangulation(curved, flat);
  const Triangulation<2, 3> tria = lift(
    flat, [](const Point<2> &p) { return Point<3>(p[0], p[1], 0); }, false);

  const TopoDS_Shape square =
    BRepBuilderAPI_MakeFace(gp_Pln(gp::XOY()), -a, a, -a, a).Face();
  const TopoDS_Shape hole =
    BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), r, 2)
      .Shape();
  const TopoDS_Shape shape = BRepAlgoAPI_Cut(square, hole).Shape();
  trimmed("plate with a hole", tria, shape, 4 * a * a - numbers::PI * r * r, 0);
}


void
split_cylinder()
{
  // The lateral surface along the x-axis, its halves z > 0 and z < 0 as two
  // NURBS faces; the cells of the mesh turned by 0.3 about the axis, so
  // that columns of cells straddle the junctions z = 0, and the knots of
  // the two rational arcs of each half, a quarter turn from the junctions,
  // where a map linear in the parameters has a kink.
  const double        R = 1, length = 3, turn = 0.3;
  const unsigned int  n_around = 12;
  Triangulation<2>    flat;
  GridGenerator::subdivided_hyper_rectangle(flat,
                                            {n_around, 3},
                                            Point<2>(turn, -length / 2),
                                            Point<2>(turn + 2 * numbers::PI,
                                                     length / 2));
  const Triangulation<2, 3> tria = lift(
    flat,
    [&](const Point<2> &p) {
      return Point<3>(p[1], R * std::cos(p[0]), R * std::sin(p[0]));
    },
    false);

  const gp_Cylinder cylinder(gp_Ax3(gp_Pnt(-length / 2, 0, 0),
                                    gp_Dir(1, 0, 0),
                                    gp_Dir(0, 1, 0)),
                             R);
  BRep_Builder    builder;
  TopoDS_Compound shape;
  builder.MakeCompound(shape);
  for (const double start : {0., numbers::PI})
    builder.Add(shape,
                BRepBuilderAPI_NurbsConvert(
                  BRepBuilderAPI_MakeFace(
                    cylinder, start, start + numbers::PI, 0, length)
                    .Face())
                  .Shape());
  trimmed("split cylinder", tria, shape, 2 * numbers::PI * R * length, 0);
}


void
spherical_cap()
{
  // The part of a sphere above the latitude of 45 degrees, closed by a
  // disk: a spherical face and a planar one, trimmed by the same circle.
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
    disk, [&](const Point<2> &p) { return Point<3>(p[0], p[1], z0); }, true);
  Triangulation<2, 3> tria;
  GridGenerator::merge_triangulations(cap, bottom, tria, 1e-10);

  const TopoDS_Shape shape =
    BRepPrimAPI_MakeSphere(R, latitude, numbers::PI / 2).Shape();
  const double h = R - z0;
  trimmed("spherical cap",
          tria,
          shape,
          2 * numbers::PI * R * h + numbers::PI * rho * rho,
          numbers::PI * h * h * (3 * R - h) / 3);
}


// A closed curve meshed by n segments between the given points.
Triangulation<1, 3>
polygon(const std::vector<Point<3>> &vertices)
{
  std::vector<CellData<1>> cells(vertices.size());
  for (unsigned int i = 0; i < vertices.size(); ++i)
    cells[i].vertices = {i,
                         static_cast<unsigned int>((i + 1) % vertices.size())};
  Triangulation<1, 3> tria;
  tria.create_triangulation(vertices, cells, SubCellData());
  return tria;
}


void
curves()
{
  const unsigned int n = 24;

  // A circle and an ellipse in a tilted plane, and the perimeter of the
  // latter by the series of Gauss and Kummer in h = ((a - b)/(a + b))^2.
  const gp_Ax2   frame(gp_Pnt(0.1, 0.2, -0.3), gp_Dir(1, 1, 1));
  const gp_Circ  circle(frame, 1.3);
  const gp_Elips ellipse(frame, 2., 1.);
  const double   h = std::pow(1. / 3, 2);
  double         perimeter = 0, coefficient = 1;
  for (unsigned int k = 0; k < 40; ++k)
    {
      perimeter += coefficient * coefficient * std::pow(h, k);
      coefficient *= (0.5 - k) / (k + 1);
    }
  perimeter *= numbers::PI * 3.;

  for (const auto &[name, edge, exact] :
       {std::make_tuple(std::string("circle"),
                        BRepBuilderAPI_MakeEdge(circle).Edge(),
                        2 * numbers::PI * 1.3),
        std::make_tuple(std::string("ellipse"),
                        BRepBuilderAPI_MakeEdge(ellipse).Edge(),
                        perimeter)})
    {
      std::vector<Point<3>> vertices;
      for (unsigned int i = 0; i < n; ++i)
        {
          const double t = 2 * numbers::PI * i / n;
          const gp_Pnt p = (name == "circle") ? ElCLib::Value(t, circle) :
                                                ElCLib::Value(t, ellipse);
          vertices.emplace_back(p.X(), p.Y(), p.Z());
        }
      const Triangulation<1, 3>            tria = polygon(vertices);
      const OpenCASCADE::ShapeProjector<1> projector(edge);
      const ProjectorCollection<1, 3>      mesh;
      unsigned int                         n_irregular = 0;
      deallog << name << ", " << n << " segments: relative error of the length"
              << std::endl
              << "points   mesh       projected" << std::endl;
      for (unsigned int q = 2; q <= 10; q += 2)
        deallog << std::setw(2) << q << "     "
                << format(integrate(tria, mesh, q, n_irregular)[0] / exact - 1)
                << "    "
                << format(
                     integrate(tria, projector, q, n_irregular)[0] / exact - 1)
                << std::endl;
      deallog << "irregular projections: " << n_irregular << std::endl;
    }

  // The two rims of a cylinder along x, among the edges of the solid: two
  // circles and the seam.
  const double       R = 1, half_length = 1.5;
  const TopoDS_Shape shape =
    BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-half_length, 0, 0),
                                    gp_Dir(1, 0, 0)),
                             R,
                             2 * half_length)
      .Shape();
  std::vector<Point<3>> left, right;
  for (unsigned int i = 0; i < n; ++i)
    {
      const double t = 2 * numbers::PI * (i + 0.5) / n;
      left.emplace_back(-half_length, R * std::cos(t), R * std::sin(t));
      right.emplace_back(half_length, R * std::cos(t), R * std::sin(t));
    }
  Triangulation<1, 3> rims;
  GridGenerator::merge_triangulations(polygon(left), polygon(right), rims, 0.);
  const OpenCASCADE::ShapeProjector<1> projector(shape);
  unsigned int                         n_irregular = 0;
  deallog << "rims of the cylinder, " << projector.n_patches()
          << " edges, 8 points: relative error of the length "
          << format(integrate(rims, projector, 8, n_irregular)[0] /
                      (4 * numbers::PI * R) -
                    1)
          << ", irregular projections " << n_irregular << std::endl;
}


int
main()
{
  initlog();
  sphere();
  deallog << std::endl;
  torus();
  deallog << std::endl;
  cylinder();
  plate();
  spherical_cap();
  split_cylinder();
  deallog << std::endl;
  curves();
}
