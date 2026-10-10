#include "mapping_hanging_constraints.h"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/exceptions.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <vector>

DEAL_II_NAMESPACE_OPEN
namespace ExactGeometry
{
  using namespace dealii;

  template <int dim, int spacedim>
  HangingConstraintStatistics
  make_mapping_hanging_node_constraints(
    const Mapping<dim, spacedim> &mapping,
    const DoFHandler<dim, spacedim> &dh,
    AffineConstraints<double> &constraints)
  {
    AssertThrow(constraints.n_constraints() == 0,
                ExcMessage("Mapping hanging constraints require an empty object."));
    AffineConstraints<double> standard;
    DoFTools::make_hanging_node_constraints(dh, standard);
    HangingConstraintStatistics result;
    using Entries = std::vector<std::pair<types::global_dof_index, double>>;
    std::map<types::global_dof_index, Entries> replacement;
    const auto &fe = dh.get_fe();

    if constexpr (dim == 2)
      if (standard.n_constraints() != 0 && fe.is_primitive() &&
          fe.has_support_points() && fe.dofs_per_face != 0)
        {
          std::vector<Point<spacedim>> points(dh.n_dofs());
          DoFTools::map_dofs_to_support_points(mapping, dh, points);
          std::vector<types::global_dof_index> coarse(fe.dofs_per_cell),
            fine(fe.dofs_per_face);
          using Face = typename DoFHandler<dim, spacedim>::face_iterator;
          for (const auto &cell : dh.active_cell_iterators())
            for (const auto f : cell->face_indices())
              if (cell->face(f)->has_children())
                {
                  cell->get_dof_indices(coarse);
                  const auto ref = [&](const double t) {
                    return f < 2 ? Point<2>(double(f), t) :
                                   Point<2>(t, double(f - 2));
                  };
                  const auto physical = [&](const double t) {
                    return mapping.transform_unit_to_real_cell(cell, ref(t));
                  };
                  const double length = physical(0.).distance(physical(1.));
                  AssertThrow(length > 0., ExcMessage("Degenerate hanging edge."));
                  std::function<void(const Face &)> visit = [&](const Face &face) {
                    if (face->has_children())
                      {
                        for (unsigned int c = 0; c < face->n_children(); ++c)
                          visit(face->child(c));
                        return;
                      }
                    face->get_dof_indices(fine);
                    for (unsigned int i = 0; i < fine.size(); ++i)
                      {
                        const auto index = fine[i];
                        if (!standard.is_constrained(index) ||
                            std::find(coarse.begin(), coarse.end(), index) != coarse.end())
                          continue;
                        const unsigned int component =
                          fe.face_system_to_component_index(i).first;
                        // A bounded one-dimensional pullback onto the coarse
                        // edge, initialized by samples and refined by Newton.
                        double t = 0., distance = points[index].distance(physical(0.));
                        for (unsigned int q = 1; q <= 16; ++q)
                          {
                            const double candidate = double(q) / 16.;
                            const double gap = points[index].distance(physical(candidate));
                            if (gap < distance)
                              t = candidate, distance = gap;
                          }
                        for (unsigned int step = 0; step < 24; ++step)
                          {
                            const auto x = physical(t);
                            const double lo = std::max(0., t - 1e-6);
                            const double hi = std::min(1., t + 1e-6);
                            const auto tangent = (physical(hi) - physical(lo)) / (hi - lo);
                            if (tangent.norm_square() == 0.)
                              break;
                            const double next = std::clamp(
                              t + (points[index] - x) * tangent / tangent.norm_square(), 0., 1.);
                            if (std::abs(next - t) < 1e-13)
                              break;
                            double fraction = 1.;
                            while (fraction > 1. / 64. &&
                                   points[index].distance(physical(t + fraction * (next - t))) >
                                     points[index].distance(x))
                              fraction *= 0.5;
                            t += fraction * (next - t);
                          }
                        const double gap = points[index].distance(physical(t));
                        result.maximum_geometric_gap = std::max(result.maximum_geometric_gap, gap);
                        if (gap > std::max(1e-12, 1e-8 * length))
                          {
                            ++result.off_edge_points;
                            continue;
                          }
                        Entries entries;
                        std::map<types::global_dof_index, double> difference;
                        for (const auto &[master, weight] :
                             *standard.get_constraint_entries(index))
                          difference[master] -= weight;
                        for (unsigned int j = 0; j < coarse.size(); ++j)
                          if (fe.system_to_component_index(j).first == component)
                            {
                              const double weight = fe.shape_value_component(j, ref(t), component);
                              difference[coarse[j]] += weight;
                              if (std::abs(weight) > 1e-14)
                                entries.emplace_back(coarse[j], weight);
                            }
                        double change = 0.;
                        for (const auto &[master, weight] : difference)
                          change = std::max(change, std::abs(weight));
                        if (change > 1e-10)
                          {
                            replacement[index] = std::move(entries);
                            result.maximum_weight_change = std::max(result.maximum_weight_change, change);
                          }
                      }
                  };
                  for (unsigned int c = 0; c < cell->face(f)->n_children(); ++c)
                    visit(cell->face(f)->child(c));
                }
        }

    for (const auto &line : standard.get_lines())
      if (constraints.can_store_line(line.index))
        {
          constraints.add_line(line.index);
          const auto found = replacement.find(line.index);
          if (found != replacement.end())
            {
              constraints.add_entries(line.index, found->second);
              ++result.corrected_lines;
            }
          else
            constraints.add_entries(line.index, line.entries);
          constraints.set_inhomogeneity(line.index, line.inhomogeneity);
        }
    return result;
  }

  template HangingConstraintStatistics make_mapping_hanging_node_constraints(
    const Mapping<2, 3> &, const DoFHandler<2, 3> &, AffineConstraints<double> &);
  template HangingConstraintStatistics make_mapping_hanging_node_constraints(
    const Mapping<1, 2> &, const DoFHandler<1, 2> &, AffineConstraints<double> &);
}
DEAL_II_NAMESPACE_CLOSE
