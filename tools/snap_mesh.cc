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

// Move the vertices of a coarse UCD mesh onto its CAD model, as
// ComputationalDomain reads both: a vertex of line elements of manifold id
// 10 + j onto Curve_j.iges, a vertex on two such curves onto their meeting
// point (alternating projections), and every other vertex of a quadrilateral
// of material i onto Color_i.iges. Coordinates written with a few digits
// leave the vertices off the curves; the exact geometry recognises an edge
// on a trimming curve only if its vertices lie on it to 1e-3 of its length,
// which the edges of the refined meshes then miss.
//
// Usage: snap_mesh input.inp cad_folder/ output.inp

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <deal.II/opencascade/utilities.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

using namespace dealii;

int
main(int argc, char **argv)
{
  AssertThrow(argc == 4,
              ExcMessage("Usage: snap_mesh input.inp cad_folder/ output.inp"));
  const std::string folder = std::string(argv[2]) + "/";

  // The mesh: header, vertices, cells, the blank lines skipped.
  std::ifstream                         in(argv[1]);
  std::vector<std::vector<std::string>> rows;
  for (std::string line; std::getline(in, line);)
    {
      std::istringstream       words(line);
      std::vector<std::string> row;
      for (std::string w; words >> w;)
        row.push_back(w);
      if (!row.empty())
        rows.push_back(row);
    }
  const unsigned int              n_vertices = std::stoi(rows[0][0]);
  std::map<std::string, Point<3>> vertices;
  for (unsigned int k = 1; k <= n_vertices; ++k)
    vertices[rows[k][0]] = Point<3>(std::stod(rows[k][1]),
                                    std::stod(rows[k][2]),
                                    std::stod(rows[k][3]));
  std::map<std::string, std::set<int>> curves_of, materials_of;
  for (unsigned int k = 1 + n_vertices; k < rows.size(); ++k)
    {
      const int material = std::stoi(rows[k][1]);
      for (unsigned int v = 3; v < rows[k].size(); ++v)
        (rows[k][2] == "line" ? curves_of : materials_of)[rows[k][v]].insert(
          material);
    }

  // The shapes, in the units of the mesh.
  std::map<int, TopoDS_Shape> curves, surfaces;
  for (unsigned int j = 1;
       std::filesystem::exists(folder + "Curve_" + std::to_string(j) + ".iges");
       ++j)
    curves[10 + j] =
      OpenCASCADE::read_IGES(folder + "Curve_" + std::to_string(j) + ".iges",
                             1e-3);
  for (unsigned int i = 1;
       std::filesystem::exists(folder + "Color_" + std::to_string(i) + ".iges");
       ++i)
    surfaces[i] =
      OpenCASCADE::read_IGES(folder + "Color_" + std::to_string(i) + ".iges",
                             1e-3);

  double       moved = 0, corner_gap = 0;
  unsigned int on_curve = 0, on_corner = 0, on_surface = 0, kept = 0;
  for (auto &[name, p] : vertices)
    {
      std::vector<int> on;
      for (const int m : curves_of[name])
        if (curves.count(m))
          on.push_back(m);
      Point<3> q = p;
      if (on.size() == 1)
        {
          q = OpenCASCADE::closest_point(curves[on[0]], p, 1e-12);
          ++on_curve;
        }
      else if (on.size() >= 2)
        {
          // Alternating projections onto the first two curves converge to
          // their meeting point; the gap left between them is reported.
          for (unsigned int it = 0; it < 200; ++it)
            for (unsigned int c = 0; c < 2; ++c)
              q = OpenCASCADE::closest_point(curves[on[c]], q, 1e-12);
          corner_gap =
            std::max(corner_gap,
                     q.distance(
                       OpenCASCADE::closest_point(curves[on[0]], q, 1e-12)));
          ++on_corner;
        }
      else if (!materials_of[name].empty() &&
               surfaces.count(*materials_of[name].begin()))
        {
          q = OpenCASCADE::closest_point(surfaces[*materials_of[name].begin()],
                                         p,
                                         1e-12);
          ++on_surface;
        }
      else
        ++kept;
      moved = std::max(moved, p.distance(q));
      p     = q;
    }

  std::ofstream out(argv[3]);
  out << std::setprecision(17);
  out << rows[0][0] << ' ' << rows.size() - 1 - n_vertices << " 0 0 0\n";
  for (unsigned int k = 1; k <= n_vertices; ++k)
    {
      const Point<3> &p = vertices[rows[k][0]];
      out << rows[k][0] << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << '\n';
    }
  for (unsigned int k = 1 + n_vertices; k < rows.size(); ++k)
    {
      for (unsigned int w = 0; w < rows[k].size(); ++w)
        out << (w ? " " : "") << rows[k][w];
      out << '\n';
    }
  std::cout << "vertices on one curve " << on_curve << ", on two curves "
            << on_corner << ", on a surface " << on_surface << ", kept " << kept
            << "; largest move " << moved
            << ", largest gap between the curves at a corner " << corner_gap
            << std::endl;
}
