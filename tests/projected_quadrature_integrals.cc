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

// Integrals over the exact geometry from meshes of flat cells: the measure,
// the enclosed volume (1/spacedim) int (x - c) . n and a second moment of a
// sphere and a circle, against the number of Gauss points per direction. On
// the mesh the error stays at O(h^2); with the closest point and along a
// vector field it decays exponentially down to round-off. Along the normal
// of the mesh it does so on the circle, whose cells meet at points of the
// circle, but not on the sphere: the normal jumps across the edges of the
// cells, whose images then overlap or leave gaps. Then the weights of the
// projected quadrature in FEValues, and a mesh of a sphere, a cylinder and a
// square with a ProjectorCollection.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "projected_quadrature.h"
#include "tests.h"

using namespace ExactGeometry;


// The field d(x) = (I + s K)(x - c), K the rotation by a right angle about
// the last axis in 2d and about the z-axis in 3d: radial plus a swirl.
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


std::string
format(const double error)
{
  if (std::abs(error) < 1e-13)
    return "<1e-13 ";
  std::ostringstream out;
  out << std::scientific << std::setprecision(1) << std::abs(error);
  return out.str();
}


// Measure, volume and the second moment of the last coordinate about c.
template <int dim, int spacedim>
std::array<double, 3>
integrate(const Triangulation<dim, spacedim> &tria,
          const Projector<dim, spacedim>     &projector,
          const unsigned int                  n_points,
          const Point<spacedim>              &c,
          unsigned int                       &n_irregular)
{
  const MappingQ<dim, spacedim>      mapping(1);
  ProjectedQuadrature<dim, spacedim> quadrature(mapping,
                                                QGauss<dim>(n_points),
                                                projector);
  std::array<double, 3>              result = {{0, 0, 0}};
  for (const auto &cell : tria.active_cell_iterators())
    {
      quadrature.reinit(cell);
      n_irregular += quadrature.n_irregular_points();
      for (unsigned int q = 0; q < quadrature.get_points().size(); ++q)
        {
          const Tensor<1, spacedim> r  = quadrature.get_points()[q] - c;
          const double              dx = quadrature.get_JxW_values()[q];
          result[0] += dx;
          result[1] += r * quadrature.get_normal_vectors()[q] / spacedim * dx;
          result[2] += r[spacedim - 1] * r[spacedim - 1] * dx;
        }
    }
  return result;
}


template <int dim, int spacedim>
void
convergence(const std::string                  &name,
            const Triangulation<dim, spacedim> &tria,
            const Point<spacedim>              &center,
            const double                        radius,
            const std::array<double, 3>        &exact)
{
  const ProjectorCollection<dim, spacedim> mesh; // the identity
  const SphericalProjector<dim, spacedim>  closest(center, radius);
  const SphericalProjector<dim, spacedim>  along_normal(
    center, radius, Direction<spacedim>::mesh_normal());
  const SwirlField<spacedim>              field(center, 0.5);
  const SphericalProjector<dim, spacedim> along_field(
    center, radius, Direction<spacedim>(field));
  const MappingQ<dim, spacedim>           mapping(1);
  const SphericalProjector<dim, spacedim> along_vertex_normal(
    center,
    radius,
    Direction<spacedim>::interpolated_mesh_normal(tria, mapping));

  deallog << name << ", " << tria.n_active_cells()
          << " flat cells; relative errors of measure / volume / second "
             "moment"
          << std::endl;
  deallog << "points   mesh                       closest point"
             "              mesh normal                vector field"
             "               interpolated normal"
          << std::endl;
  unsigned int n_irregular = 0;
  for (unsigned int n = 1; n <= 10; ++n)
    {
      deallog << std::setw(2) << n << "     ";
      for (const Projector<dim, spacedim> *projector :
           {static_cast<const Projector<dim, spacedim> *>(&mesh),
            static_cast<const Projector<dim, spacedim> *>(&closest),
            static_cast<const Projector<dim, spacedim> *>(&along_normal),
            static_cast<const Projector<dim, spacedim> *>(&along_field),
            static_cast<const Projector<dim, spacedim> *>(
              &along_vertex_normal)})
        {
          const auto integrals =
            integrate(tria, *projector, n, center, n_irregular);
          for (unsigned int i = 0; i < 3; ++i)
            deallog << format(integrals[i] / exact[i] - 1.) << " ";
          deallog << "   ";
        }
      deallog << std::endl;
    }
  deallog << "irregular projections: " << n_irregular << std::endl;
}


int
main()
{
  initlog();
  deallog.get_file_stream() << std::setprecision(10);

  // A sphere of flat cells.
  {
    const Point<3>      center(0.1, -0.2, 0.3);
    const double        R = 1.3;
    Triangulation<2, 3> curved, tria;
    GridGenerator::hyper_sphere(curved, center, R);
    curved.refine_global(2);
    GridGenerator::flatten_triangulation(curved, tria);
    convergence<2, 3>("sphere",
                      tria,
                      center,
                      R,
                      {{4 * numbers::PI * R * R,
                        4. / 3 * numbers::PI * R * R * R,
                        4. / 3 * numbers::PI * std::pow(R, 4)}});

    // The projected weights in FEValues give the JxW values of the exact
    // surface with the shape functions of the mesh.
    const MappingQ<2, 3>           mapping(1);
    const SphericalProjector<2, 3> projector(center, R);
    ProjectedQuadrature<2, 3>      quadrature(mapping, QGauss<2>(4), projector);
    const FE_Q<2, 3>               fe(1);
    double                         difference = 0;
    for (const auto &cell : tria.active_cell_iterators())
      {
        quadrature.reinit(cell);
        FEValues<2, 3> fe_values(mapping,
                                 fe,
                                 quadrature.get_quadrature(),
                                 update_JxW_values);
        fe_values.reinit(cell);
        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          difference = std::max(difference,
                                std::abs(fe_values.JxW(q) -
                                         quadrature.get_JxW_values()[q]));
      }
    deallog << "FEValues with the projected quadrature: JxW values "
            << (difference < 1e-15 ? std::string("agree") :
                                     std::to_string(difference))
            << std::endl
            << std::endl;
  }

  // A circle of straight segments.
  {
    const Point<2>      center(0.1, -0.2);
    const double        R = 1.3;
    Triangulation<1, 2> curved, tria;
    GridGenerator::hyper_sphere(curved, center, R);
    curved.refine_global(3);
    GridGenerator::flatten_triangulation(curved, tria);
    convergence<1, 2>(
      "circle",
      tria,
      center,
      R,
      {{2 * numbers::PI * R, numbers::PI * R * R, numbers::PI * R * R * R}});
    deallog << std::endl;
  }

  // A sphere (manifold id 1), the lateral surface of a cylinder (2) and a
  // square (3, no projector: integrated on the mesh).
  {
    const Point<3>      center(0, 0, 0), axis_point(5, 0, 0);
    const double        R = 1, R_cylinder = 0.8, length = 2;
    Triangulation<2, 3> curved, sphere, cylinder, square, tria, merged;
    GridGenerator::hyper_sphere(curved, center, R);
    curved.refine_global(2);
    GridGenerator::flatten_triangulation(curved, sphere);
    GridGenerator::subdivided_hyper_rectangle(
      cylinder, {16, 4}, Point<2>(0, 0), Point<2>(2 * numbers::PI, length));
    GridTools::transform(
      [&](const Point<3> &p) {
        return axis_point + Point<3>(R_cylinder * std::cos(p[0]),
                                     R_cylinder * std::sin(p[0]),
                                     p[1]);
      },
      cylinder);
    GridGenerator::subdivided_hyper_rectangle(square,
                                              {2, 2},
                                              Point<2>(-10, -10),
                                              Point<2>(-8, -8));
    const std::pair<Triangulation<2, 3> *, types::material_id> meshes[] = {
      {&sphere, 1}, {&cylinder, 2}, {&square, 3}};
    for (const auto &[mesh, id] : meshes)
      for (const auto &cell : mesh->active_cell_iterators())
        cell->set_material_id(id);
    GridGenerator::merge_triangulations(sphere, cylinder, merged, 0.);
    GridGenerator::merge_triangulations(merged, square, tria, 0.);
    for (const auto &cell : tria.active_cell_iterators())
      cell->set_manifold_id(cell->material_id());

    const SphericalProjector<2, 3> on_sphere(center, R);
    const CylindricalProjector     on_cylinder(axis_point,
                                           Tensor<1, 3>({0, 0, 1}),
                                           R_cylinder);
    ProjectorCollection<2, 3>      projector;
    projector.add(1, on_sphere);
    projector.add(2, on_cylinder);

    const double exact =
      4 * numbers::PI * R * R + 2 * numbers::PI * R_cylinder * length + 4;
    deallog << "sphere, cylinder and square, " << tria.n_active_cells()
            << " flat cells; relative error of the area" << std::endl;
    unsigned int n_irregular = 0;
    for (unsigned int n = 2; n <= 10; n += 2)
      deallog << std::setw(2) << n << "     "
              << format(integrate(tria, projector, n, center, n_irregular)[0] /
                          exact -
                        1.)
              << std::endl;
    deallog << "irregular projections: " << n_irregular << std::endl;
  }
}
