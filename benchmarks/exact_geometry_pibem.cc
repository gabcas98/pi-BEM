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

// The quadrature over the exact geometry on the meshes of pi-BEM: the
// sphere-box domain of the verification, built by ComputationalDomain from
// the parameter file given as first argument, refined globally, with maps of
// degree 1 and 2. The sphere is projected onto the CAD surface of
// Color_1.iges, or onto the analytic sphere; the faces of the box are planar
// and exact on the mesh. For each mesh, mapping and geometry it writes, to
// the CSV file given as second argument, the relative errors of
//  - the area of the sphere;
//  - Gauss' lemma, int_Gamma dG/dn_y (x0, y) dGamma_y = -1 for x0 in the
//    domain, a purely geometric identity;
//  - the representation of the harmonic field u = 1/|x| of the verification
//    at points x0 of the domain, u(x0) = int_Gamma (G du/dn - u dG/dn_y),
//    with the exact data: the identity that the boundary element method
//    discretizes, exact on the exact geometry.
// The normals point out of the domain. Gauss' lemma and the representation
// hold on any closed surface that encloses x0 and a region where u is
// harmonic, the mesh included, when u and its normal derivative are taken
// on that surface, as in the verification, where u is known in space. The
// geometry matters when the data are known on the exact surface only: the
// geometry "mesh_surface_data" integrates over the mesh, with its points,
// normals and surface elements in the kernel, the data u(P(x)) and
// du/dn(P(x)) of the corresponding points P(x) of the exact surface, as an
// isoparametric boundary element method with data given on a CAD surface.

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping_q.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "cad_projectors.h"
#include "computational_domain.h"
#include "projected_quadrature.h"

using namespace dealii;
using namespace dealii::ExactGeometry;

namespace
{
  const std::array<Point<3>, 3> targets = {
    {Point<3>(0, 0, 2.5), Point<3>(2, -1, 1.5), Point<3>(-2, 2, 0)}};

  double
  u(const Point<3> &y)
  {
    return 1 / y.norm();
  }

  Tensor<1, 3>
  grad_u(const Point<3> &y)
  {
    return -Tensor<1, 3>(y) / std::pow(y.norm(), 3);
  }

  struct Results
  {
    double                area = 0, volume = 0;
    std::array<double, 3> gauss{}, representation{};
  };

  // The integrals over all cells, the area over those of material id 1;
  // sigma orients the normals out of the domain. With surface_data, the
  // kernel and the surface elements are those of the mesh and the data
  // those of the projected points.
  Results
  integrate(const Triangulation<2, 3> &tria,
            const Mapping<2, 3>       &mapping,
            const Projector<2, 3>     &projector,
            const unsigned int         n_points,
            const double               sigma,
            const bool                 surface_data = false)
  {
    ProjectedQuadrature<2, 3> quadrature(mapping,
                                         QGauss<2>(n_points),
                                         projector);
    Results                   r;
    for (const auto &cell : tria.active_cell_iterators())
      {
        quadrature.reinit(cell);
        for (unsigned int q = 0; q < quadrature.get_points().size(); ++q)
          {
            const Point<3>    &p = quadrature.get_points()[q];
            const Tensor<1, 3> m = sigma * quadrature.get_normal_vectors()[q];
            const double       data_u = u(p), data_dudn = grad_u(p) * m;
            const Point<3>    &y =
              surface_data ? quadrature.get_mesh_points()[q] : p;
            const Tensor<1, 3> n =
              surface_data ? sigma * quadrature.get_mesh_normal_vectors()[q] :
                             m;
            const double dx = surface_data ?
                                quadrature.get_mesh_JxW_values()[q] :
                                quadrature.get_JxW_values()[q];
            if (cell->material_id() == 1)
              r.area += dx;
            r.volume += y * n / 3 * dx;
            for (unsigned int k = 0; k < targets.size(); ++k)
              {
                const Tensor<1, 3> d  = targets[k] - y;
                const double       rr = d.norm();
                const double       G  = 1 / (4 * numbers::PI * rr);
                const double dG = d * n / (4 * numbers::PI * rr * rr * rr);
                r.gauss[k] += dG * dx;
                r.representation[k] += (G * data_dudn - data_u * dG) * dx;
              }
          }
      }
    return r;
  }
} // namespace


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  AssertThrow(argc > 2,
              ExcMessage("Usage: exact_geometry_pibem parameters.prm out.csv"));

  ComputationalDomain<3> domain(MPI_COMM_WORLD);
  ParameterAcceptor::initialize(argv[1], "used_domain.prm");
  domain.read_domain();
  domain.refine_and_resize(0);
  AssertThrow(!domain.cad_surfaces.empty(),
              ExcMessage("The parameter file must load Color_1.iges."));

  const OpenCASCADE::ShapeProjector<2> cad_sphere(domain.cad_surfaces[0]);
  const SphericalProjector<2, 3>       analytic_sphere(Point<3>(), 1.);
  ProjectorCollection<2, 3>            mesh, cad, analytic;
  cad.add(1, cad_sphere);
  analytic.add(1, analytic_sphere);

  const double box    = 13. * 13. * (6.5 + 1.003);
  const double volume = box - 4. / 3 * numbers::PI;

  std::ofstream out(argv[2]);
  out << "level,cells,degree,geometry,points,area,volume,gauss,"
         "representation\n"
      << std::setprecision(4);
  double sigma = 0;
  for (unsigned int level = 0; level <= 3; ++level)
    {
      if (level > 0)
        {
          domain.tria.refine_global(1);
          domain.make_edges_conformal();
        }
      for (const unsigned int degree : {1u, 2u})
        {
          const MappingQ<2, 3> mapping(degree);
          if (sigma == 0)
            sigma = integrate(domain.tria, mapping, mesh, 4, 1.).volume > 0 ?
                      1. :
                      -1.;
          for (const auto &[name, projector] :
               {std::make_pair(std::string("mesh"), &mesh),
                std::make_pair(std::string("mesh_surface_data"), &analytic),
                std::make_pair(std::string("cad"), &cad),
                std::make_pair(std::string("analytic"), &analytic)})
            for (const unsigned int n : {16u, 24u})
              {
                const Results r     = integrate(domain.tria,
                                            mapping,
                                            *projector,
                                            n,
                                            sigma,
                                            name == "mesh_surface_data");
                double        gauss = 0, representation = 0;
                for (unsigned int k = 0; k < targets.size(); ++k)
                  {
                    gauss = std::max(gauss, std::abs(r.gauss[k] + 1));
                    representation =
                      std::max(representation,
                               std::abs(r.representation[k] / u(targets[k]) -
                                        1));
                  }
                out << level << ',' << domain.tria.n_active_cells() << ','
                    << degree << ',' << name << ',' << n << ','
                    << std::abs(r.area / (4 * numbers::PI) - 1) << ','
                    << std::abs(r.volume / volume - 1) << ',' << gauss << ','
                    << representation << std::endl;
              }
        }
    }
  std::cout << "Written to " << argv[2] << std::endl;
}
