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

// ProjectedMapping on a sphere of flat cells, through the library:
//  - the JxW values of FEValues agree with ProjectedQuadrature, and their
//    sum is the area of the sphere;
//  - the surface divergence theorem on every cell,
//      int_K grad_G phi = int_dK phi nu + int_K phi (2/R) n,
//    n the outward normal,
//    with the gradients of the shape functions, the face JxW values and
//    conormals of FEFaceValues and the cell normals, holds to rounding: the
//    cells are patches of the exact sphere;
//  - the support points of a finite element lie on the sphere, and
//    transform_real_to_unit_cell inverts transform_unit_to_real_cell;
//  - the second derivatives of the mapping are consistent with the first;
//  - the interpolation of the exact geometry in a MappingQCache converges
//    exponentially in its degree, and the estimate on a sample of cells
//    agrees with the check on all of them;
//  - on a disk trimmed by a circle (a cap of the cylinder of OpenCASCADE),
//    the boundary faces of the mesh have the length of the circle.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "projected_mapping.h"
#include "projected_quadrature.h"
#include "tests.h"

#include <numeric>

#ifdef DEAL_II_WITH_OPENCASCADE
#  include <BRepPrimAPI_MakeCylinder.hxx>
#  include <gp_Ax2.hxx>

#  include "cad_projectors.h"
#endif

using namespace ExactGeometry;


class Smooth : public Function<3>
{
public:
  virtual double
  value(const Point<3> &p, const unsigned int) const override
  {
    return std::sin(p[0]) * std::cos(2 * p[1]) + p[2] * p[2] * p[0];
  }
};


std::string
format(const double error)
{
  if (std::abs(error) < 1e-12)
    return "below 1e-12";
  std::ostringstream out;
  out << std::scientific << std::setprecision(1) << std::abs(error);
  return out.str();
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

  const SphericalProjector<2, 3> projector(center, R);
  const ProjectedMapping<2, 3>   mapping(1, projector);
  const MappingQ<2, 3>           mesh_mapping(1);

  // Areas.
  {
    const QGauss<2>           quadrature(8);
    const FE_Q<2, 3>          fe(1);
    FEValues<2, 3>            fe_values(mapping,
                             fe,
                             quadrature,
                             update_JxW_values | update_quadrature_points);
    ProjectedQuadrature<2, 3> projected(mesh_mapping, quadrature, projector);
    double                    area = 0, difference = 0;
    for (const auto &cell : tria.active_cell_iterators())
      {
        fe_values.reinit(cell);
        projected.reinit(cell);
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            area += fe_values.JxW(q);
            difference =
              std::max({difference,
                        std::abs(fe_values.JxW(q) -
                                 projected.get_JxW_values()[q]),
                        fe_values.quadrature_point(q).distance(
                          projected.get_points()[q])});
          }
      }
    deallog << "area through FEValues: relative error "
            << format(area / (4 * numbers::PI * R * R) - 1)
            << ", difference to ProjectedQuadrature " << format(difference)
            << std::endl;
  }

  // The surface divergence theorem on every cell, for a quadratic finite
  // element function.
  {
    const FE_Q<2, 3>  fe(2);
    DoFHandler<2, 3>  dof_handler(tria);
    dof_handler.distribute_dofs(fe);
    Vector<double> phi(dof_handler.n_dofs());
    VectorTools::interpolate(mapping,
                             dof_handler,
                             Smooth(),
                             phi);

    const QGauss<2>   quadrature(8);
    const QGauss<1>   face_quadrature(8);
    FEValues<2, 3>    fe_values(mapping,
                             fe,
                             quadrature,
                             update_values | update_gradients |
                               update_JxW_values | update_quadrature_points);
    FEFaceValues<2, 3> fe_face_values(mapping,
                                      fe,
                                      face_quadrature,
                                      update_values | update_JxW_values |
                                        update_normal_vectors);
    std::vector<double>         values(quadrature.size());
    std::vector<Tensor<1, 3>>   gradients(quadrature.size());
    std::vector<double>         face_values(face_quadrature.size());
    double                      residual = 0, size = 0;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(phi, values);
        fe_values.get_function_gradients(phi, gradients);
        Tensor<1, 3> left, right;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            left += gradients[q] * fe_values.JxW(q);
            right += values[q] * (2 / R) *
                     ((fe_values.quadrature_point(q) - center) / R) *
                     fe_values.JxW(q);
          }
        for (const unsigned int f : cell->face_indices())
          {
            fe_face_values.reinit(cell, f);
            fe_face_values.get_function_values(phi, face_values);
            for (unsigned int q = 0; q < face_quadrature.size(); ++q)
              right += face_values[q] * fe_face_values.normal_vector(q) *
                       fe_face_values.JxW(q);
          }
        residual = std::max(residual, (left - right).norm());
        size     = std::max(size, left.norm());
      }
    deallog << "surface divergence theorem on every cell: relative residual "
            << format(residual / size) << std::endl;

    // Support points on the sphere.
    std::vector<Point<3>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping,
                                         dof_handler,
                                         support_points);
    double off_sphere = 0;
    for (const auto &p : support_points)
      off_sphere = std::max(off_sphere, std::abs(p.distance(center) - R));
    deallog << "support points: largest distance from the sphere "
            << format(off_sphere) << std::endl;
  }

  // Inverse map and second derivatives.
  {
    double     inverse = 0, second = 0;
    const auto cell    = std::next(tria.begin_active(), 7);
    for (const Point<2> &p : {Point<2>(0.2, 0.3), Point<2>(0.9, 0.05)})
      {
        const Point<3> x = mapping.transform_unit_to_real_cell(cell, p);
        inverse = std::max(inverse,
                           p.distance(mapping.transform_real_to_unit_cell(cell,
                                                                         x)));
        // d/dxi of the Jacobian against central differences of the
        // Jacobian at the quadrature points of a shifted rule.
        const double  h = 1e-3;
        const FE_Q<2, 3> fe(1);
        FEValues<2, 3>   center_values(mapping,
                                     fe,
                                     Quadrature<2>(p),
                                     update_jacobians | update_jacobian_grads);
        center_values.reinit(cell);
        for (unsigned int b = 0; b < 2; ++b)
          {
            Point<2> plus = p, minus = p;
            plus[b] += h;
            minus[b] -= h;
            FEValues<2, 3> plus_values(mapping,
                                       fe,
                                       Quadrature<2>(plus),
                                       update_jacobians);
            FEValues<2, 3> minus_values(mapping,
                                        fe,
                                        Quadrature<2>(minus),
                                        update_jacobians);
            plus_values.reinit(cell);
            minus_values.reinit(cell);
            for (unsigned int i = 0; i < 3; ++i)
              for (unsigned int a = 0; a < 2; ++a)
                second = std::max(
                  second,
                  std::abs(center_values.jacobian_grad(0)[i][a][b] -
                           (plus_values.jacobian(0)[i][a] -
                            minus_values.jacobian(0)[i][a]) /
                             (2 * h)));
          }
      }
    deallog << "transform_real_to_unit_cell: largest error "
            << format(inverse) << std::endl
            << "second derivatives against central differences of the "
               "first: "
            << (second < 1e-5 ? std::string("below 1e-5") : format(second))
            << std::endl;
  }
}


// The interpolation of the exact geometry in a MappingQCache: deviation
// from the exact map against the degree, exponential, and the estimate on a
// sample of cells, which chooses the degree in pi-BEM, against the check on
// all cells; the area of the sphere with the interpolated geometry.
void
interpolation()
{
  const Point<3>      center(0.1, -0.2, 0.3);
  const double        R = 1.3;
  Triangulation<2, 3> curved, tria;
  GridGenerator::hyper_sphere(curved, center, R);
  curved.refine_global(2);
  GridGenerator::flatten_triangulation(curved, tria);
  const SphericalProjector<2, 3> projector(center, R);
  const ProjectedMapping<2, 3>   exact(1, projector);
  deallog << "interpolation of the exact sphere, " << tria.n_active_cells()
          << " flat cells: degree, deviation of points and surface elements, "
             "estimate on every fourth cell, relative error of the area"
          << std::endl;
  for (unsigned int degree = 2; degree <= 8; degree += 2)
    {
      MappingQCache<2, 3> cache(degree);
      const auto deviation = interpolate_exact_geometry(cache, exact, tria);
      const auto estimate =
        estimate_interpolation_deviation(exact, tria, degree, 4);
      const FE_Q<2, 3> fe(1);
      FEValues<2, 3>   values(cache, fe, QGauss<2>(degree + 2),
                            update_JxW_values);
      double           area = 0;
      for (const auto &cell : tria.active_cell_iterators())
        {
          values.reinit(cell);
          for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
            area += values.JxW(q);
        }
      deallog << degree << ": " << format(deviation.first) << ", "
              << format(deviation.second) << "; estimate "
              << format(estimate.first) << ", " << format(estimate.second)
              << "; area " << format(area / (4 * numbers::PI * R * R) - 1)
              << std::endl;
    }
}


#ifdef DEAL_II_WITH_OPENCASCADE
void
trimmed_disk()
{
  // The boundary of the cylinder of deal.II and the solid of OpenCASCADE;
  // the length of the edges of the cap x = -L/2 on its trimming circle.
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
  const ProjectedMapping<2, 3>         mapping(1, projector);
  const FE_Q<2, 3>                     fe(1);
  FEFaceValues<2, 3>                   fe_face_values(mapping,
                                    fe,
                                    QGauss<1>(8),
                                    update_JxW_values |
                                      update_quadrature_points);
  FEValues<2, 3>                       fe_values(mapping,
                               fe,
                               QGauss<2>(8),
                               update_JxW_values | update_normal_vectors);
  double                               length = 0, area = 0;
  for (const auto &cell : tria.active_cell_iterators())
    {
      fe_values.reinit(cell);
      area += std::accumulate(fe_values.get_JxW_values().begin(),
                              fe_values.get_JxW_values().end(),
                              0.);
      if (std::abs(cell->center()[0] + half_length) > 1e-10)
        continue;
      // Edges of the cap on the rim: both vertices on the circle.
      for (const unsigned int f : cell->face_indices())
        if (std::abs(cell->face(f)->vertex(0).distance(
                       Point<3>(-half_length, 0, 0)) -
                     R) < 1e-10 &&
            std::abs(cell->face(f)->vertex(1).distance(
                       Point<3>(-half_length, 0, 0)) -
                     R) < 1e-10)
          {
            fe_face_values.reinit(cell, f);
            for (unsigned int q = 0; q < fe_face_values.n_quadrature_points;
                 ++q)
              length += fe_face_values.JxW(q);
          }
    }
  deallog << "cap of the cylinder: length of the edges on the trimming "
             "circle, relative error "
          << format(length / (2 * numbers::PI * R) - 1) << std::endl
          << "cylinder: area through FEValues, relative error "
          << format(area / (4 * numbers::PI * R * half_length +
                            2 * numbers::PI * R * R) -
                    1)
          << std::endl;
}


void
parametric_interpolation()
{
  // The cylinder of deal.II without refinement: its lateral cells span a
  // quarter of the circle. The closest point of a flat cell is singular on
  // the axis, close to the cell, and its interpolation converges slowly; the
  // map through the parametrization of the faces is the cylinder with the
  // angle linear along the cell, and that of the caps a plane trimmed by
  // the circle.
  const double     R = 1, half_length = 1.5;
  Triangulation<3> volume, flat_volume;
  GridGenerator::cylinder(volume, R, half_length);
  GridGenerator::flatten_triangulation(volume, flat_volume);
  Triangulation<2, 3> tria;
  GridGenerator::extract_boundary_mesh(flat_volume, tria);
  const TopoDS_Shape shape =
    BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-half_length, 0, 0),
                                    gp_Dir(1, 0, 0)),
                             R,
                             2 * half_length)
      .Shape();
  OpenCASCADE::ShapeProjector<2> projection(shape), parametric(shape);
  parametric.set_parametric(true);
  const ProjectedMapping<2, 3> by_projection(1, projection);
  const ProjectedMapping<2, 3> by_parameters(1, parametric);
  deallog << "interpolation of the exact cylinder, " << tria.n_active_cells()
          << " flat cells: degree, deviation of points and surface elements "
             "of the projection and of the parametric map"
          << std::endl;
  for (unsigned int degree = 2; degree <= 10; degree += 2)
    {
      MappingQCache<2, 3> cache(degree);
      const auto a = interpolate_exact_geometry(cache, by_projection, tria);
      const auto b = interpolate_exact_geometry(cache, by_parameters, tria);
      deallog << degree << ": " << format(a.first) << ", " << format(a.second)
              << "; " << format(b.first) << ", " << format(b.second)
              << std::endl;
    }
}
#endif


int
main()
{
  initlog();
  sphere();
  interpolation();
#ifdef DEAL_II_WITH_OPENCASCADE
  trimmed_disk();
  parametric_interpolation();
#endif
}
