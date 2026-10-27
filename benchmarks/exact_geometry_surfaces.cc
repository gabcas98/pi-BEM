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

// Surface integrals over a CAD model meshed by pi-BEM, such as the hull of
// a ship: ComputationalDomain builds the mesh from the parameter file given
// as first argument (Color_i.iges on the cells of material i), refined
// globally up to the given level. For each level, geometry and number of
// Gauss points per direction it writes, to the CSV file given as second
// argument,
//  - the area of the cells of each material;
//  - the volume, int z n_z, and the first moment int x z n_z;
//  - the waterplane area, -int n_z;
// with the normals out of the body. The last three are those of the body
// closed by a lid in the plane z = 0, to which the lid adds nothing: a hull
// open at its waterline z = 0 needs no lid. The geometries are the mesh
// with maps of degree 1 and 2 (the second through the manifolds of the CAD
// surfaces), the closest point of the exact surface, its parametrization,
// and the interpolation of the parametrization that the solver uses
// (Exact geometry = true), at the lowest degree that meets 1e-10. The
// largest distance of the vertices from the CAD surfaces and the area of
// the CAD faces by OpenCASCADE are printed for reference.
//
// Usage: exact_geometry_surfaces domain.prm out.csv [levels] [points]

#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/opencascade/utilities.h>

#include <BRepGProp.hxx>
#include <GProp_GProps.hxx>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "cad_projectors.h"
#include "computational_domain.h"
#include "projected_mapping.h"
#include "projected_quadrature.h"

using namespace dealii;
using namespace dealii::ExactGeometry;

namespace
{
  struct Results
  {
    std::map<types::material_id, double> area;
    double                               volume = 0, moment = 0, waterplane = 0;
  };

  Results
  integrate(const Triangulation<2, 3> &tria,
            const Mapping<2, 3>       &mapping,
            const Projector<2, 3>     &projector,
            const unsigned int         n_points,
            const double               sigma)
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
            const Point<3>    &y  = quadrature.get_points()[q];
            const Tensor<1, 3> n  = sigma * quadrature.get_normal_vectors()[q];
            const double       dx = quadrature.get_JxW_values()[q];
            r.area[cell->material_id()] += dx;
            r.volume += y[2] * n[2] * dx;
            r.moment += y[0] * y[2] * n[2] * dx;
            r.waterplane -= n[2] * dx;
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
              ExcMessage("Usage: exact_geometry_surfaces domain.prm out.csv "
                         "[levels] [points]"));
  const unsigned int levels   = argc > 3 ? std::atoi(argv[3]) : 4;
  const unsigned int n_points = argc > 4 ? std::atoi(argv[4]) : 16;

  ComputationalDomain<3> domain(MPI_COMM_WORLD);
  ParameterAcceptor::initialize(argv[1], "used_domain.prm");
  domain.read_domain();
  domain.refine_and_resize(0);
  const unsigned int n_surfaces = domain.cad_surfaces.size();
  AssertThrow(n_surfaces > 0,
              ExcMessage("The parameter file must load Color_1.iges."));

  std::cout << std::setprecision(15);
  for (unsigned int i = 0; i < n_surfaces; ++i)
    {
      GProp_GProps props;
      BRepGProp::SurfaceProperties(domain.cad_surfaces[i], props, 1e-12);
      std::cout << "Color_" << i + 1 << ".iges: area by OpenCASCADE "
                << props.Mass() << std::endl;
    }

  // The cells of material i on Color_i, projected or through the
  // parametrization; the mesh collection leaves every cell on the mesh.
  std::vector<std::unique_ptr<OpenCASCADE::ShapeProjector<2>>> projectors;
  ProjectorCollection<2, 3> mesh, closest, parametric;
  for (unsigned int i = 0; i < n_surfaces; ++i)
    for (const bool through_parameters : {false, true})
      {
        projectors.push_back(std::make_unique<OpenCASCADE::ShapeProjector<2>>(
          domain.cad_surfaces[i]));
        projectors.back()->set_parametric(through_parameters);
        (through_parameters ? parametric : closest)
          .add(1 + i, *projectors.back());
      }
  const ProjectedMapping<2, 3> exact(1, parametric);

  std::ofstream out(argv[2]);
  out << "level,cells,geometry,degree,points,deviation";
  for (unsigned int i = 0; i < n_surfaces; ++i)
    out << ",area_" << i + 1;
  out << ",volume,moment,waterplane\n" << std::setprecision(16);

  double sigma = 0;
  for (unsigned int level = 0; level <= levels; ++level)
    {
      if (level > 0)
        {
          domain.tria.refine_global(1);
          domain.make_edges_conformal();
        }

      // The vertices on the CAD surface of their cells.
      double distance = 0;
      for (const auto &cell : domain.tria.active_cell_iterators())
        if (cell->material_id() >= 1 && cell->material_id() <= n_surfaces)
          for (const unsigned int v : cell->vertex_indices())
            distance =
              std::max(distance,
                       cell->vertex(v).distance(OpenCASCADE::closest_point(
                         domain.cad_surfaces[cell->material_id() - 1],
                         cell->vertex(v),
                         1e-7)));
      std::cout << "level " << level << ": " << domain.tria.n_active_cells()
                << " cells, largest distance of a vertex from the CAD "
                << distance << std::endl;

      // The interpolated exact geometry of the solver.
      std::unique_ptr<MappingQCache<2, 3>> interpolated;
      unsigned int                         degree         = 0;
      double                               best_deviation = 0;
      for (unsigned int d = 2; d <= 10; ++d)
        {
          auto       cache = std::make_unique<MappingQCache<2, 3>>(d);
          const auto deviation_pair =
            interpolate_exact_geometry(*cache, exact, domain.tria);
          const double deviation =
            std::max(deviation_pair.first, deviation_pair.second);
          if (interpolated == nullptr || deviation < best_deviation)
            {
              interpolated   = std::move(cache);
              degree         = d;
              best_deviation = deviation;
            }
          if (deviation <= 1e-10)
            break;
        }
      std::cout << "  interpolated exact geometry: degree " << degree
                << ", deviation " << best_deviation << std::endl;

      const MappingQ<2, 3> q1(1), q2(2);
      if (sigma == 0)
        sigma = integrate(domain.tria, q1, mesh, 4, 1.).volume > 0 ? 1. : -1.;
      struct Variant
      {
        std::string            name;
        const Mapping<2, 3>   *mapping;
        const Projector<2, 3> *projector;
        unsigned int           degree;
        double                 deviation;
      };
      for (const Variant &variant :
           {Variant{"mesh_q1", &q1, &mesh, 1, 0},
            Variant{"mesh_q2", &q2, &mesh, 2, 0},
            Variant{"projection", &q1, &closest, 0, 0},
            Variant{"parametric", &q1, &parametric, 0, 0},
            Variant{"interpolated",
                    interpolated.get(),
                    &mesh,
                    degree,
                    best_deviation}})
        for (const unsigned int n : {n_points, n_points + 8})
          {
            const Results r = integrate(
              domain.tria, *variant.mapping, *variant.projector, n, sigma);
            out << level << ',' << domain.tria.n_active_cells() << ','
                << variant.name << ',' << variant.degree << ',' << n << ','
                << variant.deviation;
            for (unsigned int i = 0; i < n_surfaces; ++i)
              {
                const auto it = r.area.find(1 + i);
                out << ',' << (it == r.area.end() ? 0. : it->second);
              }
            out << ',' << r.volume << ',' << r.moment << ',' << r.waterplane
                << std::endl;
          }
    }
  std::cout << "Written to " << argv[2] << std::endl;
}
