#include "planar_mesh_quality.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

namespace PlanarMeshQuality
{
  using namespace dealii;

  Statistics
  smooth(Triangulation<2, 3> &tria,
         const std::set<types::material_id> &material_ids,
         const unsigned int maximum_sweeps)
  {
    Statistics result;
    if (material_ids.empty() || maximum_sweeps == 0)
      return result;

    using Cell = Triangulation<2, 3>::active_cell_iterator;
    struct Patch
    {
      Point<3> origin;
      Tensor<1, 3> normal;
      Tensor<1, 3> axis;
      double diameter = 0.;
    };
    std::map<types::material_id, Patch> patches;
    std::vector<Cell> cells;
    std::vector<Tensor<1, 3>> normals;
    std::vector<std::vector<unsigned int>> incident(tria.n_vertices());
    std::vector<std::set<unsigned int>> neighbors(tria.n_vertices());
    std::vector<bool> fixed(tria.n_vertices(), false);
    std::vector<Point<3> *> positions(tria.n_vertices(), nullptr);
    const std::array<unsigned int, 4> perimeter{{0, 1, 3, 2}};

    // Freeze both sides of a hanging face: moving its coarse endpoints
    // while leaving the fine vertices fixed would change the contact.
    using Face = Triangulation<2, 3>::face_iterator;
    std::function<void(const Face &)> freeze_face = [&](const Face &face) {
      for (const auto v : face->vertex_indices())
        fixed[face->vertex_index(v)] = true;
      if (face->has_children())
        for (unsigned int c = 0; c < face->n_children(); ++c)
          freeze_face(face->child(c));
    };
    for (const auto &cell : tria.active_cell_iterators())
      {
        const bool selected = material_ids.count(cell->material_id()) != 0;
        for (const auto f : cell->face_indices())
          if (cell->at_boundary(f) || cell->face(f)->has_children())
            freeze_face(cell->face(f));
        if (!selected)
          {
            for (const auto v : cell->vertex_indices())
              fixed[cell->vertex_index(v)] = true;
            continue;
          }
        Tensor<1, 3> normal = cross_product_3d(
          cell->vertex(1) - cell->vertex(0),
          cell->vertex(2) - cell->vertex(0));
        AssertThrow(normal.norm() > 0., ExcMessage("Degenerate planar patch."));
        normal /= normal.norm();
        auto [it, inserted] = patches.try_emplace(cell->material_id());
        if (inserted)
          {
            it->second.origin = cell->vertex(0);
            it->second.normal = normal;
            it->second.axis = cell->vertex(1) - cell->vertex(0);
            it->second.axis /= it->second.axis.norm();
          }
        it->second.diameter = std::max(it->second.diameter,
                                      it->second.origin.distance(cell->center()) +
                                        cell->diameter());
        const unsigned int index = cells.size();
        cells.push_back(cell);
        normals.push_back(normal);
        for (unsigned int i = 0; i < 4; ++i)
          {
            const auto v = cell->vertex_index(perimeter[i]);
            incident[v].push_back(index);
            positions[v] = &cell->vertex(perimeter[i]);
            neighbors[v].insert(cell->vertex_index(perimeter[(i + 1) % 4]));
            neighbors[v].insert(cell->vertex_index(perimeter[(i + 3) % 4]));
          }
      }
    for (unsigned int v = 0; v < incident.size(); ++v)
      for (const auto c : incident[v])
        if (cells[c]->material_id() != cells[incident[v][0]]->material_id())
          fixed[v] = true;
    // Validate all patches before the first mutation. Smoothing a curved
    // patch in its tangent plane would silently change the geometry.
    for (const auto &cell : cells)
      {
        const auto &patch = patches.at(cell->material_id());
        for (const auto v : cell->vertex_indices())
          AssertThrow(std::abs((cell->vertex(v) - patch.origin) * patch.normal) <=
                        1e-10 * std::max(patch.diameter, 1e-12),
                      ExcMessage("Mesh quality smoothing requires a planar "
                                 "patch for every selected material id."));
      }

    const auto quality = [&](const unsigned int c) {
      double minimum = 1.;
      for (unsigned int i = 0; i < 4; ++i)
        {
          const auto &cell = cells[c];
          const Point<3> p = cell->vertex(perimeter[i]);
          const Tensor<1, 3> a = cell->vertex(perimeter[(i + 1) % 4]) - p;
          const Tensor<1, 3> b = cell->vertex(perimeter[(i + 3) % 4]) - p;
          const double denominator = a.norm_square() + b.norm_square();
          minimum = std::min(minimum, denominator > 0. ?
            2. * (cross_product_3d(a, b) * normals[c]) / denominator : -1.);
        }
      return minimum;
    };
    const auto summary = [&]() {
      std::pair<double, double> value(1., 0.);
      for (unsigned int c = 0; c < cells.size(); ++c)
        {
          const double q = quality(c);
          value.first = std::min(value.first, q);
          value.second += q;
        }
      if (!cells.empty())
        value.second /= cells.size();
      return value;
    };
    std::tie(result.minimum_before, result.mean_before) = summary();
    AssertThrow(result.minimum_before > 0.,
                ExcMessage("Planar smoothing requires convex input quads."));

    // Deterministic vertex sweeps with a local line search. Every incident
    // cell keeps at least its previous minimum corner quality; maximize
    // their sum without trading a better cell for a worse neighbor.
    const auto &vertices = tria.get_vertices();
    for (; result.sweeps < maximum_sweeps; ++result.sweeps)
      {
        unsigned int moved = 0;
        for (unsigned int v = 0; v < incident.size(); ++v)
          {
            if (fixed[v] || incident[v].empty())
              continue;
            const auto &patch = patches.at(cells[incident[v][0]]->material_id());
            const Tensor<1, 3> axis2 = cross_product_3d(patch.normal, patch.axis);
            const Point<3> old = vertices[v];
            Point<3> average;
            double step = std::numeric_limits<double>::max();
            for (const auto n : neighbors[v])
              {
                average += vertices[n];
                step = std::min(step, old.distance(vertices[n]));
              }
            average /= neighbors[v].size();
            const auto score = [&]() {
              std::pair<double, double> value(1., 0.);
              for (const auto c : incident[v])
                {
                  const double q = quality(c);
                  value.first = std::min(value.first, q);
                  value.second += q;
                }
              return value;
            };
            auto best_score = score();
            std::vector<double> floors;
            for (const auto c : incident[v])
              floors.push_back(quality(c));
            Point<3> best = old;
            std::vector<Tensor<1, 3>> directions{average - old};
            for (int i = -1; i <= 1; ++i)
              for (int j = -1; j <= 1; ++j)
                if (i != 0 || j != 0)
                  directions.push_back(0.25 * step * (i * patch.axis + j * axis2));
            for (const auto &direction : directions)
              for (const double fraction : {1., 0.5, 0.25})
                {
                  *positions[v] = old + fraction * direction;
                  // Remove accumulated roundoff normal to the plane.
                  *positions[v] -= ((vertices[v] - patch.origin) * patch.normal) *
                                   patch.normal;
                  const auto candidate = score();
                  bool admissible = candidate.first > 0.;
                  for (unsigned int i = 0; i < incident[v].size(); ++i)
                    admissible = admissible &&
                      quality(incident[v][i]) >= floors[i] - 1e-14;
                  if (admissible && candidate.second > best_score.second + 1e-12)
                    {
                      best_score = candidate;
                      best = vertices[v];
                    }
                }
            *positions[v] = best;
            if (best.distance(old) > 1e-12 * patch.diameter)
              ++moved;
          }
        result.accepted_moves += moved;
        if (moved == 0)
          break;
      }
    std::tie(result.minimum_after, result.mean_after) = summary();
    return result;
  }
}
