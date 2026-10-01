#include "../include/galerkin_panel_pair_integrator.h"

#include <chrono>

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{
  Point<2>
  square_symmetry(const Point<2> &point, const unsigned int symmetry)
  {
    const double u = point[0];
    const double v = point[1];
    switch (symmetry)
      {
        case 0:
          return Point<2>(u, v);
        case 1:
          return Point<2>(1.0 - u, v);
        case 2:
          return Point<2>(u, 1.0 - v);
        case 3:
          return Point<2>(1.0 - u, 1.0 - v);
        case 4:
          return Point<2>(v, u);
        case 5:
          return Point<2>(1.0 - v, u);
        case 6:
          return Point<2>(v, 1.0 - u);
        case 7:
          return Point<2>(1.0 - v, 1.0 - u);
        default:
          AssertThrow(false, ExcInternalError());
          return Point<2>();
      }
  }

  bool
  point_lexicographic_less(const Point<3> &left, const Point<3> &right)
  {
    for (unsigned int d = 0; d < 3; ++d)
      {
        if (left[d] < right[d])
          return true;
        if (left[d] > right[d])
          return false;
      }
    return false;
  }

  Point<2>
  transform_patch_point(const Point<2> &point,
                        const Point<2> &lower,
                        const Point<2> &upper)
  {
    Point<2> transformed;
    for (unsigned int d = 0; d < 2; ++d)
      transformed[d] = lower[d] + (upper[d] - lower[d]) * point[d];
    return transformed;
  }

  constexpr unsigned int maximum_lagrange_nodes = 12;
  using LagrangeValues = std::array<double, maximum_lagrange_nodes>;

  // The Lagrange basis through the given nodes in monomial form: row i holds
  // the coefficients of basis function i, from the constant term up, and the
  // next n rows those of its derivative. Evaluated by Horner's rule, the
  // basis costs a few multiply-adds per function and no branch, which
  // matters at the tens of thousands of points of a singular rule.
  // Normal trace of a vector-valued function at a point.
  template <int spacedim>
  double
  normal_trace(const Function<spacedim>  &gradient,
               const Point<spacedim>     &point,
               const Tensor<1, spacedim> &normal)
  {
    double trace = 0.0;
    for (unsigned int d = 0; d < spacedim; ++d)
      trace += gradient.value(point, d) * normal[d];
    return trace;
  }

  std::vector<double>
  monomial_coefficients(const std::vector<double> &nodes)
  {
    const unsigned int  n = nodes.size();
    std::vector<double> result(2 * n * n, 0.0);
    for (unsigned int i = 0; i < n; ++i)
      {
        std::vector<double> c(n, 0.0);
        c[0] = 1.0;
        unsigned int degree = 0;
        for (unsigned int k = 0; k < n; ++k)
          if (k != i)
            {
              const double scale = 1.0 / (nodes[i] - nodes[k]);
              for (unsigned int m = degree + 1; m > 0; --m)
                c[m] = scale * (c[m - 1] - nodes[k] * c[m]);
              c[0] = -scale * nodes[k] * c[0];
              ++degree;
            }
        for (unsigned int m = 0; m < n; ++m)
          result[i * n + m] = c[m];
        for (unsigned int m = 1; m < n; ++m)
          result[(n + i) * n + m - 1] = m * c[m];
      }
    return result;
  }

  // Values and, if asked, first derivatives of the basis at t.
  void
  lagrange_basis(const std::vector<double> &coefficients,
                 const unsigned int         n,
                 const double               t,
                 LagrangeValues            &value,
                 LagrangeValues            *derivative)
  {
    for (unsigned int i = 0; i < n; ++i)
      {
        const double *c = &coefficients[i * n];
        double        v = c[n - 1];
        for (unsigned int m = n - 1; m > 0; --m)
          v = v * t + c[m - 1];
        value[i] = v;
      }
    if (derivative != nullptr)
      for (unsigned int i = 0; i < n; ++i)
        {
          const double *c = &coefficients[(n + i) * n];
          double        d = n > 1 ? c[n - 2] : 0.0;
          for (unsigned int m = n > 1 ? n - 2 : 0; m > 0; --m)
            d = d * t + c[m - 1];
          (*derivative)[i] = d;
        }
  }
} // namespace


template <int spacedim>
GalerkinPanelPairIntegrator<spacedim>::GalerkinPanelPairIntegrator(
  const Mapping<spacedim - 1, spacedim>       &mapping,
  const FiniteElement<spacedim - 1, spacedim> &finite_element,
  const GalerkinSingularQuadrature             singular_quadrature,
  const unsigned int                           singular_order,
  const unsigned int                           duffy_outer_subdivisions,
  const double                                 near_singular_distance_ratio,
  const unsigned int                           near_singular_maximum_depth,
  const unsigned int                           near_singular_quadrature_order)
  : mapping(mapping)
  , finite_element(finite_element)
  , singular_quadrature(singular_quadrature)
  , singular_order(singular_order)
  , duffy_outer_subdivisions(duffy_outer_subdivisions)
  , near_singular_distance_ratio(near_singular_distance_ratio)
  , near_singular_maximum_depth(near_singular_maximum_depth)
  , near_singular_quadrature_order(near_singular_quadrature_order)
  , near_singular_pairs(0)
  , near_singular_leaves(0)
{
  AssertThrow(duffy_outer_subdivisions > 0,
              ExcMessage("Duffy outer subdivisions must be positive."));
}


template <int spacedim>
bool
GalerkinPanelPairIntegrator<spacedim>::is_far_regular_pair(
  const CellIterator &test_cell,
  const CellIterator &trial_cell) const
{
  const ReferencePatch full_patch{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)};
  return classify(test_cell, trial_cell) == GalerkinPanelPairType::regular &&
         !is_near_singular(test_cell, full_patch, trial_cell, full_patch);
}


template <int spacedim>
unsigned int
GalerkinPanelPairIntegrator<spacedim>::n_near_singular_pairs() const
{
  return near_singular_pairs;
}


template <int spacedim>
unsigned int
GalerkinPanelPairIntegrator<spacedim>::n_near_singular_leaves() const
{
  return near_singular_leaves;
}


template <int spacedim>
std::vector<typename GalerkinPanelPairIntegrator<spacedim>::CommonVertex>
GalerkinPanelPairIntegrator<spacedim>::common_vertices(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch) const
{
  const double tolerance =
    1e-12 * std::max({1.0, test_cell->diameter(), trial_cell->diameter()});

  // Map the four corners of each patch once. The previous form mapped every
  // one of the sixteen combinations, so each corner was transformed four
  // times over; on a whole-cell patch the corner is the cell vertex and no
  // mapping is needed at all.
  const auto corners = [this](const CellIterator   &cell,
                              const ReferencePatch &patch) {
    std::array<Point<spacedim>, GeometryInfo<2>::vertices_per_cell> points;
    const bool full = is_full_patch(patch);
    for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
      points[i] =
        full ? cell->vertex(i) :
               map_to_real_cell(
                 cell,
                 transform_patch_point(GeometryInfo<2>::unit_cell_vertex(i),
                                       patch.lower,
                                       patch.upper));
    return points;
  };
  const auto test_corners  = corners(test_cell, test_patch);
  const auto trial_corners = corners(trial_cell, trial_patch);

  std::vector<CommonVertex> result;
  for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
    for (unsigned int j = 0; j < GeometryInfo<2>::vertices_per_cell; ++j)
      if (test_corners[i].distance(trial_corners[j]) <= tolerance)
        result.push_back({test_corners[i], i, j});

  std::sort(result.begin(),
            result.end(),
            [](const CommonVertex &left, const CommonVertex &right) {
              return point_lexicographic_less(left.point, right.point);
            });
  return result;
}


template <int spacedim>
bool
GalerkinPanelPairIntegrator<spacedim>::share_nonconforming_edge(
  const CellIterator &test_cell,
  const CellIterator &trial_cell) const
{
  const auto is_nonconforming_neighbor = [](const CellIterator &cell,
                                             const CellIterator &other) {
    for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; ++face)
      if (!cell->at_boundary(face))
        {
          if (cell->neighbor_is_coarser(face) &&
              cell->neighbor(face) == other)
            return true;

          if (cell->face(face)->has_children())
            for (unsigned int subface = 0;
                 subface < cell->face(face)->n_children();
                 ++subface)
              if (cell->neighbor_child_on_subface(face, subface) == other)
                return true;
        }
    return false;
  };

  return test_cell->level() != trial_cell->level() &&
         (is_nonconforming_neighbor(test_cell, trial_cell) ||
          is_nonconforming_neighbor(trial_cell, test_cell));
}


template <int spacedim>
GalerkinPanelPairType
GalerkinPanelPairIntegrator<spacedim>::classify(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch) const
{
  const bool identical_patches =
    test_cell == trial_cell && test_patch.lower == trial_patch.lower &&
    test_patch.upper == trial_patch.upper;
  if (identical_patches)
    return GalerkinPanelPairType::same_panel;

  const unsigned int n_common =
    common_vertices(test_cell, test_patch, trial_cell, trial_patch).size();
  if (n_common == 2)
    return GalerkinPanelPairType::common_edge;
  if (n_common == 1)
    return GalerkinPanelPairType::common_vertex;
  if (n_common == 0)
    return GalerkinPanelPairType::regular;
  return GalerkinPanelPairType::unsupported;
}


template <int spacedim>
GalerkinPanelPairType
GalerkinPanelPairIntegrator<spacedim>::classify(
  const CellIterator &test_cell,
  const CellIterator &trial_cell) const
{
  if (test_cell == trial_cell)
    return GalerkinPanelPairType::same_panel;
  if (share_nonconforming_edge(test_cell, trial_cell))
    return GalerkinPanelPairType::nonconforming_edge;

  const double tolerance =
    1e-12 * std::max({1.0, test_cell->diameter(), trial_cell->diameter()});
  unsigned int n_common_vertices = 0;
  for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
    for (unsigned int j = 0; j < GeometryInfo<2>::vertices_per_cell; ++j)
      if (test_cell->vertex(i).distance(trial_cell->vertex(j)) <= tolerance)
        ++n_common_vertices;

  if (n_common_vertices == 2)
    return GalerkinPanelPairType::common_edge;
  if (n_common_vertices == 1)
    return GalerkinPanelPairType::common_vertex;
  if (n_common_vertices == 0)
    return GalerkinPanelPairType::regular;
  return GalerkinPanelPairType::unsupported;
}


template <int spacedim>
std::vector<typename GalerkinPanelPairIntegrator<spacedim>::ReferencePatch>
GalerkinPanelPairIntegrator<spacedim>::uniform_subpatches(
  const unsigned int refinement_levels) const
{
  AssertThrow(refinement_levels < 16,
              ExcMessage("Excessive nonconforming refinement difference."));
  const unsigned int subdivisions = 1U << refinement_levels;
  std::vector<ReferencePatch> patches;
  patches.reserve(subdivisions * subdivisions);
  for (unsigned int i = 0; i < subdivisions; ++i)
    for (unsigned int j = 0; j < subdivisions; ++j)
      patches.push_back(
        {Point<2>(static_cast<double>(i) / subdivisions,
                  static_cast<double>(j) / subdivisions),
         Point<2>(static_cast<double>(i + 1) / subdivisions,
                  static_cast<double>(j + 1) / subdivisions)});
  return patches;
}


template <int spacedim>
std::vector<typename GalerkinPanelPairIntegrator<spacedim>::ReferencePatch>
GalerkinPanelPairIntegrator<spacedim>::split_patch(
  const ReferencePatch &patch) const
{
  const Point<2> midpoint(0.5 * (patch.lower[0] + patch.upper[0]),
                          0.5 * (patch.lower[1] + patch.upper[1]));
  return {{patch.lower, midpoint},
          {Point<2>(midpoint[0], patch.lower[1]),
           Point<2>(patch.upper[0], midpoint[1])},
          {Point<2>(patch.lower[0], midpoint[1]),
           Point<2>(midpoint[0], patch.upper[1])},
          {midpoint, patch.upper}};
}


template <int spacedim>
std::vector<Point<spacedim>>
GalerkinPanelPairIntegrator<spacedim>::patch_sample_points(
  const CellIterator   &cell,
  const ReferencePatch &patch) const
{
  std::vector<Point<spacedim>> points;
  points.reserve(9);
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      {
        const Point<2> reference(
          patch.lower[0] +
            0.5 * i * (patch.upper[0] - patch.lower[0]),
          patch.lower[1] +
            0.5 * j * (patch.upper[1] - patch.lower[1]));
        points.push_back(map_to_real_cell(cell, reference));
      }
  return points;
}


template <int spacedim>
bool
GalerkinPanelPairIntegrator<spacedim>::is_full_patch(
  const ReferencePatch &patch) const
{
  return patch.lower[0] == 0.0 && patch.lower[1] == 0.0 &&
         patch.upper[0] == 1.0 && patch.upper[1] == 1.0;
}


template <int spacedim>
const typename GalerkinPanelPairIntegrator<spacedim>::CellGeometry &
GalerkinPanelPairIntegrator<spacedim>::full_patch_geometry(
  const CellIterator &cell) const
{
  const unsigned int index = cell->active_cell_index();
  if (cell_geometry_cache.size() <= index)
    {
      const unsigned int n = cell->get_triangulation().n_active_cells();
      cell_geometry_cache.resize(std::max<unsigned int>(n, index + 1));
      cell_geometry_valid.resize(cell_geometry_cache.size(), false);
    }
  if (!cell_geometry_valid[index])
    {
      CellGeometry geometry;
      geometry.samples = patch_sample_points(
        cell, ReferencePatch{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)});
      Point<spacedim> centre;
      for (const auto &p : geometry.samples)
        centre += p;
      centre /= static_cast<double>(geometry.samples.size());
      geometry.centre   = centre;
      geometry.radius   = 0.0;
      geometry.diameter = 0.0;
      for (const auto &p : geometry.samples)
        geometry.radius = std::max(geometry.radius, centre.distance(p));
      for (const auto &a : geometry.samples)
        for (const auto &b : geometry.samples)
          geometry.diameter = std::max(geometry.diameter, a.distance(b));
      cell_geometry_cache[index] = std::move(geometry);
      cell_geometry_valid[index] = true;
    }
  return cell_geometry_cache[index];
}


template <int spacedim>
bool
GalerkinPanelPairIntegrator<spacedim>::is_near_singular(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch) const
{
  if (near_singular_distance_ratio <= 0.0 ||
      near_singular_maximum_depth == 0)
    return false;

  // Whole-cell patches are the overwhelmingly common case and are asked about
  // once per ordered pair, so they go through the cache and a bounding-sphere
  // rejection before any per-point work.
  if (is_full_patch(test_patch) && is_full_patch(trial_patch))
    {
      const CellGeometry &test  = full_patch_geometry(test_cell);
      const CellGeometry &trial = full_patch_geometry(trial_cell);
      const double        threshold =
        near_singular_distance_ratio *
        std::max(test.diameter, trial.diameter);
      const double separation = test.centre.distance(trial.centre) -
                                test.radius - trial.radius;
      if (separation >= threshold)
        return false;

      double distance = std::numeric_limits<double>::max();
      for (const auto &left : test.samples)
        for (const auto &right : trial.samples)
          distance = std::min(distance, left.distance(right));
      return distance < threshold;
    }

  const auto test_points = patch_sample_points(test_cell, test_patch);
  const auto trial_points = patch_sample_points(trial_cell, trial_patch);
  double test_diameter  = 0.0;
  double trial_diameter = 0.0;
  double distance       = std::numeric_limits<double>::max();
  for (const auto &left : test_points)
    for (const auto &right : test_points)
      test_diameter = std::max(test_diameter, left.distance(right));
  for (const auto &left : trial_points)
    for (const auto &right : trial_points)
      trial_diameter = std::max(trial_diameter, left.distance(right));
  for (const auto &left : test_points)
    for (const auto &right : trial_points)
      distance = std::min(distance, left.distance(right));

  return distance < near_singular_distance_ratio *
                      std::max(test_diameter, trial_diameter);
}


template <int spacedim>
std::array<double, 2>
GalerkinPanelPairIntegrator<spacedim>::patch_extents(
  const CellIterator &cell, const ReferencePatch &patch) const
{
  // patch_sample_points lays out a 3x3 grid as index i * 3 + j, with i along
  // the first reference direction and j along the second.
  const auto points = patch_sample_points(cell, patch);
  return {{points[0].distance(points[6]), points[0].distance(points[2])}};
}


template <int spacedim>
std::array<typename GalerkinPanelPairIntegrator<spacedim>::ReferencePatch, 2>
GalerkinPanelPairIntegrator<spacedim>::split_patch_direction(
  const ReferencePatch &patch, const unsigned int direction) const
{
  const double middle =
    0.5 * (patch.lower[direction] + patch.upper[direction]);
  ReferencePatch low  = patch;
  ReferencePatch high = patch;
  low.upper[direction]  = middle;
  high.lower[direction] = middle;
  return {{low, high}};
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_balanced_singular(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order,
  const OperatorType    operator_type,
  const Function<spacedim> *trial_function,
  const bool                regularize_trial_function,
  const unsigned int        depth) const
{
  const GalerkinPanelPairType type =
    classify(test_cell, test_patch, trial_cell, trial_patch);

  // A subpatch that no longer touches is an ordinary regular pair, and it is
  // usually close enough to need the near-singular subdivision.
  if (type == GalerkinPanelPairType::regular)
    return integrate_adaptive_regular(test_cell, test_patch, trial_cell,
                                      trial_patch, order, operator_type,
                                      trial_function,
                                      regularize_trial_function, 0);

  const unsigned int effective_order =
    singular_order == 0 ? order : singular_order;
  const auto test_size  = patch_extents(test_cell, test_patch);
  const auto trial_size = patch_extents(trial_cell, trial_patch);
  const double target =
    std::min({test_size[0], test_size[1], trial_size[0], trial_size[1]});

  // Halve whichever of the four side lengths is furthest out of proportion,
  // so every step brings the pair closer to two comparable, roughly square
  // panels - the configuration the rule is built for. Coincident panels are
  // balanced by construction.
  unsigned int worst_direction = 0;
  double       worst_extent    = 0.0;
  bool         worst_is_test   = true;
  for (unsigned int d = 0; d < 2; ++d)
    {
      if (test_size[d] > worst_extent)
        {
          worst_extent  = test_size[d];
          worst_direction = d;
          worst_is_test = true;
        }
      if (trial_size[d] > worst_extent)
        {
          worst_extent  = trial_size[d];
          worst_direction = d;
          worst_is_test = false;
        }
    }

  if (type == GalerkinPanelPairType::same_panel ||
      depth >= singular_balance_maximum_depth ||
      target <= 0.0 ||
      worst_extent <= singular_balance_ratio * target)
    return integrate(test_cell, test_patch, trial_cell, trial_patch,
                     effective_order, operator_type, trial_function,
                     regularize_trial_function);

  FullMatrix<double> result(finite_element.dofs_per_cell,
                            n_result_columns(operator_type, trial_function));
  if (worst_is_test)
    for (const auto &child :
         split_patch_direction(test_patch, worst_direction))
      result.add(1.0,
                 integrate_balanced_singular(test_cell, child, trial_cell,
                                             trial_patch, order,
                                             operator_type, trial_function,
                                             regularize_trial_function,
                                             depth + 1));
  else
    for (const auto &child :
         split_patch_direction(trial_patch, worst_direction))
      result.add(1.0,
                 integrate_balanced_singular(test_cell, test_patch,
                                             trial_cell, child, order,
                                             operator_type, trial_function,
                                             regularize_trial_function,
                                             depth + 1));
  return result;
}


template <int spacedim>
double
GalerkinPanelPairIntegrator<spacedim>::error_energy_balanced(
  const CellIterator        &test_cell,
  const ReferencePatch      &test_patch,
  const CellIterator        &trial_cell,
  const ReferencePatch      &trial_patch,
  const unsigned int         order,
  const Function<spacedim>  &exact_gradient,
  const std::vector<double> &test_coefficients,
  const std::vector<double> &trial_coefficients,
  const unsigned int         depth) const
{
  const GalerkinPanelPairType type =
    classify(test_cell, test_patch, trial_cell, trial_patch);

  if (type == GalerkinPanelPairType::regular)
    return error_energy_adaptive(test_cell, test_patch, trial_cell,
                                 trial_patch, order, exact_gradient,
                                 test_coefficients, trial_coefficients, 0);

  const unsigned int effective_order =
    singular_order == 0 ? order : singular_order;
  const auto   test_size  = patch_extents(test_cell, test_patch);
  const auto   trial_size = patch_extents(trial_cell, trial_patch);
  const double target =
    std::min({test_size[0], test_size[1], trial_size[0], trial_size[1]});

  unsigned int worst_direction = 0;
  double       worst_extent    = 0.0;
  bool         worst_is_test   = true;
  for (unsigned int d = 0; d < 2; ++d)
    {
      if (test_size[d] > worst_extent)
        {
          worst_extent    = test_size[d];
          worst_direction = d;
          worst_is_test   = true;
        }
      if (trial_size[d] > worst_extent)
        {
          worst_extent    = trial_size[d];
          worst_direction = d;
          worst_is_test   = false;
        }
    }

  if (type == GalerkinPanelPairType::same_panel ||
      depth >= singular_balance_maximum_depth ||
      target <= 0.0 ||
      worst_extent <= singular_balance_ratio * target)
    return error_energy_leaf(test_cell, test_patch, trial_cell, trial_patch,
                             effective_order, exact_gradient,
                             test_coefficients, trial_coefficients);

  double total = 0.0;
  if (worst_is_test)
    for (const auto &child :
         split_patch_direction(test_patch, worst_direction))
      total += error_energy_balanced(test_cell, child, trial_cell,
                                     trial_patch, order, exact_gradient,
                                     test_coefficients, trial_coefficients,
                                     depth + 1);
  else
    for (const auto &child :
         split_patch_direction(trial_patch, worst_direction))
      total += error_energy_balanced(test_cell, test_patch, trial_cell,
                                     child, order, exact_gradient,
                                     test_coefficients, trial_coefficients,
                                     depth + 1);
  return total;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_adaptive_regular(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order,
  const OperatorType    operator_type,
  const Function<spacedim> *trial_function,
  const bool regularize_trial_function,
  const unsigned int depth) const
{
  if (!is_near_singular(test_cell,
                        test_patch,
                        trial_cell,
                        trial_patch))
    {
      ++near_singular_leaves;
      return integrate(test_cell,
                       test_patch,
                       trial_cell,
                       trial_patch,
                       order,
                       operator_type,
                       trial_function,
                       regularize_trial_function);
    }

  if (depth >= near_singular_maximum_depth)
    {
      ++near_singular_leaves;
      return integrate(test_cell,
                       test_patch,
                       trial_cell,
                       trial_patch,
                       std::max(order, near_singular_quadrature_order),
                       operator_type,
                       trial_function,
                       regularize_trial_function);
    }

  const auto test_points = patch_sample_points(test_cell, test_patch);
  const auto trial_points = patch_sample_points(trial_cell, trial_patch);
  const double test_diameter =
    test_points.front().distance(test_points.back());
  const double trial_diameter =
    trial_points.front().distance(trial_points.back());
  const bool split_test =
    test_diameter > trial_diameter * (1.0 + 1e-12) ||
    (std::abs(test_diameter - trial_diameter) <=
       1e-12 * std::max({1.0, test_diameter, trial_diameter}) &&
     test_cell->active_cell_index() < trial_cell->active_cell_index());

  FullMatrix<double> result(finite_element.dofs_per_cell,
                            n_result_columns(operator_type, trial_function));
  if (split_test)
    for (const auto &child : split_patch(test_patch))
      result.add(1.0,
                 integrate_adaptive_regular(test_cell,
                                            child,
                                            trial_cell,
                                            trial_patch,
                                            order,
                                            operator_type,
                                            trial_function,
                                            regularize_trial_function,
                                            depth + 1));
  else
    for (const auto &child : split_patch(trial_patch))
      result.add(1.0,
                 integrate_adaptive_regular(test_cell,
                                            test_patch,
                                            trial_cell,
                                            child,
                                            order,
                                            operator_type,
                                            trial_function,
                                            regularize_trial_function,
                                            depth + 1));
  return result;
}


template <int spacedim>
Point<2>
GalerkinPanelPairIntegrator<spacedim>::orient_from_canonical_vertex(
  const Point<2>   &point,
  const unsigned int local_vertex) const
{
  const Point<2> canonical_origin(0.0, 0.0);
  const Point<2> target = GeometryInfo<2>::unit_cell_vertex(local_vertex);
  for (unsigned int symmetry = 0; symmetry < 8; ++symmetry)
    if (square_symmetry(canonical_origin, symmetry).distance(target) < 1e-14)
      return square_symmetry(point, symmetry);
  AssertThrow(false, ExcInternalError());
  return Point<2>();
}


template <int spacedim>
Point<2>
GalerkinPanelPairIntegrator<spacedim>::orient_from_canonical_edge(
  const Point<2>   &point,
  const unsigned int first_local_vertex,
  const unsigned int second_local_vertex) const
{
  const Point<2> canonical_first(0.0, 0.0);
  const Point<2> canonical_second(1.0, 0.0);
  const Point<2> target_first =
    GeometryInfo<2>::unit_cell_vertex(first_local_vertex);
  const Point<2> target_second =
    GeometryInfo<2>::unit_cell_vertex(second_local_vertex);
  for (unsigned int symmetry = 0; symmetry < 8; ++symmetry)
    if (square_symmetry(canonical_first, symmetry).distance(target_first) <
          1e-14 &&
        square_symmetry(canonical_second, symmetry).distance(target_second) <
          1e-14)
      return square_symmetry(point, symmetry);
  AssertThrow(false,
              ExcMessage("The common vertices do not form a cell edge."));
  return Point<2>();
}


template <int spacedim>
Point<2>
GalerkinPanelPairIntegrator<spacedim>::orient_to_canonical_edge(
  const Point<2>   &point,
  const unsigned int first_local_vertex,
  const unsigned int second_local_vertex) const
{
  const Point<2> canonical_first(0.0, 0.0);
  const Point<2> canonical_second(1.0, 0.0);
  const Point<2> source_first =
    GeometryInfo<2>::unit_cell_vertex(first_local_vertex);
  const Point<2> source_second =
    GeometryInfo<2>::unit_cell_vertex(second_local_vertex);
  for (unsigned int symmetry = 0; symmetry < 8; ++symmetry)
    if (square_symmetry(source_first, symmetry).distance(canonical_first) <
          1e-14 &&
        square_symmetry(source_second, symmetry).distance(canonical_second) <
          1e-14)
      return square_symmetry(point, symmetry);
  AssertThrow(false,
              ExcMessage("The common vertices do not form a cell edge."));
  return Point<2>();
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_single_layer(
  const CellIterator &test_cell,
  const CellIterator &trial_cell,
  const unsigned int  order) const
{
  FullMatrix<double> result = integrate_operator(
    test_cell, trial_cell, order, OperatorType::single_layer);
  if (singular_quadrature == GalerkinSingularQuadrature::iterated_duffy &&
      test_cell == trial_cell)
    for (unsigned int i = 0; i < result.m(); ++i)
      for (unsigned int j = i + 1; j < result.n(); ++j)
        {
          const double symmetric_value = 0.5 * (result(i, j) + result(j, i));
          result(i, j) = symmetric_value;
          result(j, i) = symmetric_value;
        }
  return result;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_double_layer(
  const CellIterator &test_cell,
  const CellIterator &trial_cell,
  const unsigned int  order) const
{
  return integrate_operator(
    test_cell, trial_cell, order, OperatorType::double_layer);
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_single_and_double_layer(
  const CellIterator       &test_cell,
  const CellIterator       &trial_cell,
  const unsigned int        order,
  const Function<spacedim> &trial_function,
  const bool                regularize_trial_function) const
{
  FullMatrix<double> result =
    integrate_operator(test_cell,
                       trial_cell,
                       order,
                       OperatorType::single_and_double_layer,
                       &trial_function,
                       regularize_trial_function);
  // As in integrate_single_layer(): the single-layer block of a panel with
  // itself is symmetrised under the iterated Duffy rule.
  if (singular_quadrature == GalerkinSingularQuadrature::iterated_duffy &&
      test_cell == trial_cell)
    for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < finite_element.dofs_per_cell; ++j)
        {
          const double symmetric_value = 0.5 * (result(i, j) + result(j, i));
          result(i, j) = symmetric_value;
          result(j, i) = symmetric_value;
        }
  return result;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_single_and_double_layer_both(
  const CellIterator       &test_cell,
  const CellIterator       &trial_cell,
  const unsigned int        order,
  const Function<spacedim> &trial_function,
  const bool                regularize_trial_function) const
{
  AssertThrow(takes_light_path(OperatorType::single_and_double_layer_both,
                               &trial_function),
              ExcMessage("The pair integrated both ways needs the direct "
                         "evaluation of the geometry (Galerkin light "
                         "geometry = true)."));
  return integrate_operator(test_cell,
                            trial_cell,
                            order,
                            OperatorType::single_and_double_layer_both,
                            &trial_function,
                            regularize_trial_function);
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_adjoint_double_layer(
  const CellIterator &test_cell,
  const CellIterator &trial_cell,
  const unsigned int  order) const
{
  return integrate_operator(
    test_cell, trial_cell, order, OperatorType::adjoint_double_layer);
}


template <int spacedim>
Vector<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_adjoint_double_layer_flux(
  const CellIterator       &test_cell,
  const CellIterator       &trial_cell,
  const unsigned int        order,
  const Function<spacedim> &gradient) const
{
  AssertDimension(gradient.n_components, spacedim);
  const auto column = integrate_operator(test_cell,
                                         trial_cell,
                                         order,
                                         OperatorType::adjoint_double_layer,
                                         &gradient,
                                         false);
  Vector<double> result(column.m());
  for (unsigned int i = 0; i < column.m(); ++i)
    result[i] = column(i, 0);
  return result;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_weak_hypersingular(
  const CellIterator &test_cell,
  const CellIterator &trial_cell,
  const unsigned int  order) const
{
  static_assert(spacedim == 3,
                "The surface-curl weak form is implemented in 3D only.");
  return integrate_operator(
    test_cell, trial_cell, order, OperatorType::weak_hypersingular);
}


template <int spacedim>
typename GalerkinPanelPairIntegrator<spacedim>::CrossLayerMatrices
GalerkinPanelPairIntegrator<spacedim>::integrate_cross_layers(
  const CellIterator &test_cell,
  const CellIterator &trial_cell,
  const unsigned int  order) const
{
  AssertThrow(order > 0, ExcMessage("Quadrature order must be positive."));
  if (classify(test_cell, trial_cell) != GalerkinPanelPairType::regular)
    {
      CrossLayerMatrices result{
        integrate_single_layer(test_cell, trial_cell, order),
        FullMatrix<double>(finite_element.dofs_per_cell,
                           finite_element.dofs_per_cell)};
      const FullMatrix<double> transposed_pair =
        integrate_double_layer(trial_cell, test_cell, order);
      for (unsigned int i = 0; i < result.adjoint_double_layer.m(); ++i)
        for (unsigned int j = 0; j < result.adjoint_double_layer.n(); ++j)
          result.adjoint_double_layer(i, j) = transposed_pair(j, i);
      return result;
    }
  const ReferencePatch full_patch{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)};
  if (is_near_singular(test_cell,
                       full_patch,
                       trial_cell,
                       full_patch))
    {
      ++near_singular_pairs;
      return integrate_adaptive_cross_layers(test_cell,
                                             full_patch,
                                             trial_cell,
                                             full_patch,
                                             order,
                                             0);
    }
  return integrate_cross_layers(test_cell,
                                full_patch,
                                trial_cell,
                                full_patch,
                                order);
}


template <int spacedim>
typename GalerkinPanelPairIntegrator<spacedim>::CrossLayerMatrices
GalerkinPanelPairIntegrator<spacedim>::integrate_cross_layers(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order) const
{
  const QGauss<2> base_quadrature(order);
  const double test_patch_jacobian =
    (test_patch.upper[0] - test_patch.lower[0]) *
    (test_patch.upper[1] - test_patch.lower[1]);
  const double trial_patch_jacobian =
    (trial_patch.upper[0] - trial_patch.lower[0]) *
    (trial_patch.upper[1] - trial_patch.lower[1]);
  std::vector<Point<2>> test_points;
  std::vector<Point<2>> trial_points;
  std::vector<double> test_weights;
  std::vector<double> trial_weights;
  test_points.reserve(base_quadrature.size());
  trial_points.reserve(base_quadrature.size());
  test_weights.reserve(base_quadrature.size());
  trial_weights.reserve(base_quadrature.size());
  for (unsigned int q = 0; q < base_quadrature.size(); ++q)
    {
      test_points.push_back(transform_patch_point(base_quadrature.point(q),
                                                  test_patch.lower,
                                                  test_patch.upper));
      trial_points.push_back(transform_patch_point(base_quadrature.point(q),
                                                   trial_patch.lower,
                                                   trial_patch.upper));
      test_weights.push_back(base_quadrature.weight(q) *
                             test_patch_jacobian);
      trial_weights.push_back(base_quadrature.weight(q) *
                              trial_patch_jacobian);
    }

  const Quadrature<2> test_quadrature(test_points, test_weights);
  const Quadrature<2> trial_quadrature(trial_points, trial_weights);
  FEValues<2, spacedim> test_values(mapping,
                                    finite_element,
                                    test_quadrature,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
  FEValues<2, spacedim> trial_values(mapping,
                                     finite_element,
                                     trial_quadrature,
                                     update_values |
                                       update_quadrature_points |
                                       update_JxW_values);
  test_values.reinit(test_cell);
  trial_values.reinit(trial_cell);

  CrossLayerMatrices result{
    FullMatrix<double>(finite_element.dofs_per_cell,
                       finite_element.dofs_per_cell),
    FullMatrix<double>(finite_element.dofs_per_cell,
                       finite_element.dofs_per_cell)};
  for (unsigned int qx = 0; qx < test_quadrature.size(); ++qx)
    for (unsigned int qy = 0; qy < trial_quadrature.size(); ++qy)
      {
        const Tensor<1, spacedim> displacement =
          trial_values.quadrature_point(qy) -
          test_values.quadrature_point(qx);
        const double distance = displacement.norm();
        const double common_factor = test_values.JxW(qx) *
                                     trial_values.JxW(qy) *
                                     single_layer_kernel(distance);
        const double adjoint_factor =
          (displacement * test_values.normal_vector(qx)) *
          test_values.JxW(qx) * trial_values.JxW(qy) *
          screening_factor(distance) /
          (4.0 * numbers::PI * distance * distance * distance);
        AssertThrow(std::isfinite(common_factor) &&
                      std::isfinite(adjoint_factor),
                    ExcMessage("Non-finite combined cross-layer integrand."));
        for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < finite_element.dofs_per_cell; ++j)
            {
              const double shape_product =
                test_values.shape_value(i, qx) *
                trial_values.shape_value(j, qy);
              result.single_layer(i, j) +=
                shape_product * common_factor;
              result.adjoint_double_layer(i, j) +=
                shape_product * adjoint_factor;
            }
      }
  return result;
}


template <int spacedim>
typename GalerkinPanelPairIntegrator<spacedim>::CrossLayerMatrices
GalerkinPanelPairIntegrator<spacedim>::integrate_adaptive_cross_layers(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order,
  const unsigned int    depth) const
{
  if (!is_near_singular(test_cell,
                        test_patch,
                        trial_cell,
                        trial_patch))
    {
      ++near_singular_leaves;
      return integrate_cross_layers(test_cell,
                                    test_patch,
                                    trial_cell,
                                    trial_patch,
                                    order);
    }
  if (depth >= near_singular_maximum_depth)
    {
      ++near_singular_leaves;
      return integrate_cross_layers(
        test_cell,
        test_patch,
        trial_cell,
        trial_patch,
        std::max(order, near_singular_quadrature_order));
    }

  const auto test_points = patch_sample_points(test_cell, test_patch);
  const auto trial_points = patch_sample_points(trial_cell, trial_patch);
  const double test_diameter =
    test_points.front().distance(test_points.back());
  const double trial_diameter =
    trial_points.front().distance(trial_points.back());
  const bool split_test =
    test_diameter > trial_diameter * (1.0 + 1e-12) ||
    (std::abs(test_diameter - trial_diameter) <=
       1e-12 * std::max({1.0, test_diameter, trial_diameter}) &&
     test_cell->active_cell_index() < trial_cell->active_cell_index());

  CrossLayerMatrices result{
    FullMatrix<double>(finite_element.dofs_per_cell,
                       finite_element.dofs_per_cell),
    FullMatrix<double>(finite_element.dofs_per_cell,
                       finite_element.dofs_per_cell)};
  const auto patches = split_test ? split_patch(test_patch) :
                                    split_patch(trial_patch);
  for (const auto &child : patches)
    {
      const CrossLayerMatrices contribution = split_test ?
        integrate_adaptive_cross_layers(test_cell,
                                        child,
                                        trial_cell,
                                        trial_patch,
                                        order,
                                        depth + 1) :
        integrate_adaptive_cross_layers(test_cell,
                                        test_patch,
                                        trial_cell,
                                        child,
                                        order,
                                        depth + 1);
      result.single_layer.add(1.0, contribution.single_layer);
      result.adjoint_double_layer.add(
        1.0, contribution.adjoint_double_layer);
    }
  return result;
}


template <int spacedim>
Vector<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_double_layer(
  const CellIterator       &test_cell,
  const CellIterator       &trial_cell,
  const unsigned int        order,
  const Function<spacedim> &trial_function,
  const bool                regularize_trial_function) const
{
  const FullMatrix<double> local_column = integrate_operator(
    test_cell,
    trial_cell,
    order,
    OperatorType::double_layer,
    &trial_function,
    regularize_trial_function);
  Vector<double> result(local_column.m());
  for (unsigned int i = 0; i < local_column.m(); ++i)
    result[i] = local_column(i, 0);
  return result;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_operator(
  const CellIterator &test_cell,
  const CellIterator &trial_cell,
  const unsigned int  order,
  const OperatorType  operator_type,
  const Function<spacedim> *trial_function,
  const bool regularize_trial_function) const
{
  AssertThrow(order > 0, ExcMessage("Quadrature order must be positive."));
  const unsigned int effective_order =
    classify(test_cell, trial_cell) == GalerkinPanelPairType::regular ||
        singular_order == 0 ?
      order :
      singular_order;
  const ReferencePatch full_patch{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)};
  if (classify(test_cell, trial_cell) == GalerkinPanelPairType::regular &&
      is_near_singular(test_cell,
                       full_patch,
                       trial_cell,
                       full_patch))
    {
      ++near_singular_pairs;
      return integrate_adaptive_regular(test_cell,
                                        full_patch,
                                        trial_cell,
                                        full_patch,
                                        order,
                                        operator_type,
                                        trial_function,
                                        regularize_trial_function,
                                        0);
    }
  if (!share_nonconforming_edge(test_cell, trial_cell))
    return integrate_balanced_singular(test_cell,
                                       full_patch,
                                       trial_cell,
                                       full_patch,
                                       order,
                                       operator_type,
                                       trial_function,
                                       regularize_trial_function,
                                       0);

  const unsigned int level_difference =
    test_cell->level() > trial_cell->level() ?
      test_cell->level() - trial_cell->level() :
      trial_cell->level() - test_cell->level();
  const std::vector<ReferencePatch> coarse_patches =
    uniform_subpatches(level_difference);

  FullMatrix<double> result(finite_element.dofs_per_cell,
                            n_result_columns(operator_type, trial_function));
  for (const ReferencePatch &coarse_patch : coarse_patches)
    {
      const FullMatrix<double> contribution =
        test_cell->level() < trial_cell->level() ?
          integrate(test_cell,
                    coarse_patch,
                    trial_cell,
                    full_patch,
                    effective_order,
                    operator_type,
                    trial_function,
                    regularize_trial_function) :
          integrate(test_cell,
                    full_patch,
                    trial_cell,
                    coarse_patch,
                    effective_order,
                    operator_type,
                    trial_function,
                    regularize_trial_function);
      result.add(1.0, contribution);
    }
  return result;
}


template <int spacedim>
void
GalerkinPanelPairIntegrator<spacedim>::unit_pair_rule(
  const GalerkinPanelPairType      type,
  const std::vector<CommonVertex> &shared,
  const unsigned int               order,
  std::vector<Point<2>>           &test_reference_points,
  std::vector<Point<2>>           &trial_reference_points,
  std::vector<double>             &pair_weights) const
{
  // The singular rules in the coordinates of the two patches, before these
  // are placed in their cells: they depend only on the configuration, the
  // order and which corners of the patches are shared.
  test_reference_points.clear();
  trial_reference_points.clear();
  pair_weights.clear();
  if (singular_quadrature ==
           GalerkinSingularQuadrature::iterated_duffy)
    {
      Quadrature<2> outer_quadrature;
      if (type == GalerkinPanelPairType::same_panel)
        outer_quadrature =
          QIterated<2>(QGauss<1>(4), duffy_outer_subdivisions);
      else
        outer_quadrature =
          QIterated<2>(QGauss<1>(2), duffy_outer_subdivisions);

      for (unsigned int qx = 0; qx < outer_quadrature.size(); ++qx)
        {
          Point<2> split_point;
          if (type == GalerkinPanelPairType::same_panel)
            split_point = outer_quadrature.point(qx);
          else if (type == GalerkinPanelPairType::common_edge)
            {
              const Point<2> canonical_test = orient_to_canonical_edge(
                outer_quadrature.point(qx),
                shared[0].test_index,
                shared[1].test_index);
              split_point = orient_from_canonical_edge(
                Point<2>(canonical_test[0], 0.0),
                shared[0].trial_index,
                shared[1].trial_index);
            }
          else
            split_point =
              GeometryInfo<2>::unit_cell_vertex(shared[0].trial_index);

          const QSplit<2> inner_quadrature(
            QDuffy(order, 1.0), split_point);
          for (unsigned int qy = 0; qy < inner_quadrature.size(); ++qy)
            {
              test_reference_points.push_back(outer_quadrature.point(qx));
              trial_reference_points.push_back(inner_quadrature.point(qy));
              pair_weights.push_back(outer_quadrature.weight(qx) *
                                     inner_quadrature.weight(qy));
            }
        }
    }
  else
    {
      const QGauss<1> one_dimensional_quadrature(order);
      std::vector<Point<4>> parametric_points;
      std::vector<double>   parametric_weights;
      const unsigned int n_1d = one_dimensional_quadrature.size();
      parametric_points.reserve(n_1d * n_1d * n_1d * n_1d);
      parametric_weights.reserve(n_1d * n_1d * n_1d * n_1d);
      for (unsigned int q0 = 0; q0 < n_1d; ++q0)
        for (unsigned int q1 = 0; q1 < n_1d; ++q1)
          for (unsigned int q2 = 0; q2 < n_1d; ++q2)
            for (unsigned int q3 = 0; q3 < n_1d; ++q3)
              {
                Point<4> point;
                point[0] = one_dimensional_quadrature.point(q0)[0];
                point[1] = one_dimensional_quadrature.point(q1)[0];
                point[2] = one_dimensional_quadrature.point(q2)[0];
                point[3] = one_dimensional_quadrature.point(q3)[0];
                parametric_points.push_back(point);
                parametric_weights.push_back(
                  one_dimensional_quadrature.weight(q0) *
                  one_dimensional_quadrature.weight(q1) *
                  one_dimensional_quadrature.weight(q2) *
                  one_dimensional_quadrature.weight(q3));
              }
      const unsigned int n_regions =
        type == GalerkinPanelPairType::same_panel ? 8 :
        type == GalerkinPanelPairType::common_edge ? 6 : 4;
      test_reference_points.reserve(n_regions * parametric_points.size());
      trial_reference_points.reserve(n_regions * parametric_points.size());
      pair_weights.reserve(n_regions * parametric_points.size());

      for (unsigned int q = 0; q < parametric_points.size(); ++q)
        {
          const Point<4> &p = parametric_points[q];
          for (unsigned int region = 0; region < n_regions; ++region)
            {
              Point<2> x;
              Point<2> y;
              double   transformation_jacobian = 0.0;

              if (type == GalerkinPanelPairType::same_panel)
                {
                  const std::array<double, 4> z = {{
                    (1.0 - p[0]) * p[3],
                    (1.0 - p[0] * p[1]) * p[2],
                    p[0] + (1.0 - p[0]) * p[3],
                    p[0] * p[1] + (1.0 - p[0] * p[1]) * p[2]}};
                  const std::array<std::array<unsigned int, 4>, 8> indices = {{
                    {{0, 1, 2, 3}},
                    {{1, 0, 3, 2}},
                    {{0, 3, 2, 1}},
                    {{1, 2, 3, 0}},
                    {{2, 1, 0, 3}},
                    {{3, 0, 1, 2}},
                    {{2, 3, 0, 1}},
                    {{3, 2, 1, 0}}
                  }};
                  x = Point<2>(z[indices[region][0]],
                               z[indices[region][1]]);
                  y = Point<2>(z[indices[region][2]],
                               z[indices[region][3]]);
                  transformation_jacobian =
                    p[0] * (1.0 - p[0]) * (1.0 - p[0] * p[1]);
                }
              else if (type == GalerkinPanelPairType::common_edge)
                {
                  const std::array<double, 4> z1 = {{
                    (1.0 - p[0]) * p[3] + p[0],
                    p[0] * p[2],
                    (1.0 - p[0]) * p[3],
                    p[0] * p[1]}};
                  const std::array<double, 4> z2 = {{
                    (1.0 - p[0] * p[1]) * p[3] + p[0] * p[1],
                    p[0] * p[2],
                    (1.0 - p[0] * p[1]) * p[3],
                    p[0]}};
                  const std::array<std::array<unsigned int, 4>, 2> first = {{
                    {{0, 1, 2, 3}}, {{2, 1, 0, 3}}
                  }};
                  const std::array<std::array<unsigned int, 4>, 4> second = {{
                    {{0, 1, 2, 3}},
                    {{0, 3, 2, 1}},
                    {{2, 1, 0, 3}},
                    {{2, 3, 0, 1}}
                  }};
                  const auto &z = region < 2 ? z1 : z2;
                  const auto &idx =
                    region < 2 ? first[region] : second[region - 2];
                  x = Point<2>(z[idx[0]], z[idx[1]]);
                  y = Point<2>(z[idx[2]], z[idx[3]]);
                  transformation_jacobian =
                    region < 2 ? p[0] * p[0] * (1.0 - p[0]) :
                                 p[0] * p[0] * (1.0 - p[0] * p[1]);
                  x = orient_from_canonical_edge(
                    x, shared[0].test_index, shared[1].test_index);
                  y = orient_from_canonical_edge(
                    y, shared[0].trial_index, shared[1].trial_index);
                }
              else
                {
                  const std::array<double, 4> z = {{
                    p[0], p[0] * p[1], p[0] * p[2], p[0] * p[3]}};
                  const std::array<std::array<unsigned int, 4>, 4> indices = {{
                    {{0, 1, 2, 3}},
                    {{1, 0, 2, 3}},
                    {{1, 2, 0, 3}},
                    {{1, 2, 3, 0}}
                  }};
                  x = Point<2>(z[indices[region][0]],
                               z[indices[region][1]]);
                  y = Point<2>(z[indices[region][2]],
                               z[indices[region][3]]);
                  transformation_jacobian = p[0] * p[0] * p[0];
                  x = orient_from_canonical_vertex(x, shared[0].test_index);
                  y = orient_from_canonical_vertex(y, shared[0].trial_index);
                }

              test_reference_points.push_back(x);
              trial_reference_points.push_back(y);
              pair_weights.push_back(parametric_weights[q] *
                                     transformation_jacobian);
            }
        }
    }

}


template <int spacedim>
void
GalerkinPanelPairIntegrator<spacedim>::pair_quadrature(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order,
  std::vector<Point<2>> &test_reference_points,
  std::vector<Point<2>> &trial_reference_points,
  std::vector<std::array<unsigned int, 2>> &pair_indices,
  std::vector<double>   &pair_weights) const
{
  test_reference_points.clear();
  trial_reference_points.clear();
  pair_indices.clear();
  pair_weights.clear();
  const GalerkinPanelPairType pair_type =
    classify(test_cell, test_patch, trial_cell, trial_patch);
  AssertThrow(pair_type != GalerkinPanelPairType::unsupported,
              ExcMessage("Unsupported overlapping panel pair."));


  if (pair_type == GalerkinPanelPairType::regular)
    {
      // A regular pair uses a tensor rule, so the two sides have only
      // `order^2` distinct points each even though the pair has `order^4`
      // entries. Storing the points once and pairing them by index keeps the
      // mapping and shape-function evaluations down by that same factor.
      const QGauss<2> quadrature(order);
      const unsigned int n = quadrature.size();
      test_reference_points.reserve(n);
      trial_reference_points.reserve(n);
      for (unsigned int q = 0; q < n; ++q)
        {
          test_reference_points.push_back(transform_patch_point(
            quadrature.point(q), test_patch.lower, test_patch.upper));
          trial_reference_points.push_back(transform_patch_point(
            quadrature.point(q), trial_patch.lower, trial_patch.upper));
        }
      pair_indices.reserve(n * n);
      pair_weights.reserve(n * n);
      for (unsigned int qx = 0; qx < n; ++qx)
        for (unsigned int qy = 0; qy < n; ++qy)
          {
            pair_indices.push_back({{qx, qy}});
            pair_weights.push_back(quadrature.weight(qx) *
                                   quadrature.weight(qy));
          }
    }
  else
    {
      const std::vector<CommonVertex> shared =
        common_vertices(test_cell, test_patch, trial_cell, trial_patch);
      unit_pair_rule(pair_type,
                     shared,
                     order,
                     test_reference_points,
                     trial_reference_points,
                     pair_weights);
      for (Point<2> &point : test_reference_points)
        point =
          transform_patch_point(point, test_patch.lower, test_patch.upper);
      for (Point<2> &point : trial_reference_points)
        point =
          transform_patch_point(point, trial_patch.lower, trial_patch.upper);
    }

  if (pair_indices.empty())
    {
      // The singular rules emit one point per side for each pair entry, so
      // the pairing is the identity.
      pair_indices.resize(pair_weights.size());
      for (unsigned int q = 0; q < pair_weights.size(); ++q)
        pair_indices[q] = {{q, q}};
    }

  const double test_patch_jacobian =
    (test_patch.upper[0] - test_patch.lower[0]) *
    (test_patch.upper[1] - test_patch.lower[1]);
  const double trial_patch_jacobian =
    (trial_patch.upper[0] - trial_patch.lower[0]) *
    (trial_patch.upper[1] - trial_patch.lower[1]);
  for (double &weight : pair_weights)
    weight *= test_patch_jacobian * trial_patch_jacobian;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order,
  const OperatorType    operator_type,
  const Function<spacedim> *trial_function,
  const bool regularize_trial_function) const
{
  if (takes_light_path(operator_type, trial_function))
    return integrate_light(test_cell,
                           test_patch,
                           trial_cell,
                           trial_patch,
                           order,
                           operator_type,
                           trial_function,
                           regularize_trial_function);

  AssertThrow(operator_type != OperatorType::single_and_double_layer_both,
              ExcMessage("The pair integrated both ways needs the direct "
                         "evaluation of the geometry (Galerkin light "
                         "geometry = true)."));
  const bool wants_gradients =
    operator_type == OperatorType::weak_hypersingular;

  // Whole-cell patches are the common case, and for them the rule depends
  // only on the configuration, so it is built once and reused. Sub-patches,
  // which the near-singular recursion produces, differ pair by pair and are
  // not cached.
  // The switch exists so the cache can be measured against its own absence
  // on the same machine and the same load.
  static const bool cache_disabled =
    std::getenv("PIBEM_NO_PAIR_CACHE") != nullptr;
  const bool cacheable = !cache_disabled && is_full_patch(test_patch) &&
                         is_full_patch(trial_patch);
  std::array<int, 7> key{{-1, -1, -1, -1, -1, -1, -1}};
  if (cacheable)
    {
      const GalerkinPanelPairType type =
        classify(test_cell, test_patch, trial_cell, trial_patch);
      key[0] = static_cast<int>(type);
      key[1] = static_cast<int>(order);
      key[2] = wants_gradients ? 1 : 0;
      if (type == GalerkinPanelPairType::common_edge ||
          type == GalerkinPanelPairType::common_vertex)
        {
          const std::vector<CommonVertex> shared =
            common_vertices(test_cell, test_patch, trial_cell, trial_patch);
          for (unsigned int k = 0; k < shared.size() && k < 2; ++k)
            {
              key[3 + 2 * k] = static_cast<int>(shared[k].test_index);
              key[4 + 2 * k] = static_cast<int>(shared[k].trial_index);
            }
        }
      const auto hit = pair_rule_cache.find(key);
      if (hit != pair_rule_cache.end())
        {
          CachedPairRule &rule = hit->second;
          rule.test_values->reinit(test_cell);
          rule.trial_values->reinit(trial_cell);
          return integrate_pairs(*rule.test_values,
                                 *rule.trial_values,
                                 rule.indices,
                                 rule.weights,
                                 operator_type,
                                 trial_function,
                                 regularize_trial_function);
        }
    }

  const auto setup_start = std::chrono::steady_clock::now();
  ++profile.uncached_pairs;
  std::vector<Point<2>>                    test_reference_points;
  std::vector<Point<2>>                    trial_reference_points;
  std::vector<std::array<unsigned int, 2>> pair_indices;
  std::vector<double>                      pair_weights;
  pair_quadrature(test_cell, test_patch, trial_cell, trial_patch, order,
                  test_reference_points, trial_reference_points, pair_indices,
                  pair_weights);


  const std::vector<double> unit_weights(test_reference_points.size(), 1.0);
  const Quadrature<2> test_quadrature(test_reference_points, unit_weights);
  const Quadrature<2> trial_quadrature(trial_reference_points, unit_weights);
  // Only the hypersingular form reads shape gradients. Asking for them
  // regardless made every singular panel pair pay for the covariant
  // transform of a gradient at each of its thousands of quadrature points,
  // twice, for a quantity the single- and double-layer kernels never touch.
  const UpdateFlags flags =
    update_values | update_normal_vectors | update_quadrature_points |
    update_JxW_values |
    (operator_type == OperatorType::weak_hypersingular ? update_gradients :
                                                         UpdateFlags());
  auto test_values_owned =
    std::make_unique<FEValues<2, spacedim>>(mapping,
                                            finite_element,
                                            test_quadrature,
                                            flags);
  auto trial_values_owned =
    std::make_unique<FEValues<2, spacedim>>(mapping,
                                            finite_element,
                                            trial_quadrature,
                                            flags);
  test_values_owned->reinit(test_cell);
  trial_values_owned->reinit(trial_cell);
  profile.setup_seconds += std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - setup_start)
                             .count();

  if (cacheable && pair_rule_cache.size() < maximum_cached_pair_rules)
    {
      CachedPairRule rule;
      rule.indices      = pair_indices;
      rule.weights      = pair_weights;
      rule.test_values  = std::move(test_values_owned);
      rule.trial_values = std::move(trial_values_owned);
      CachedPairRule &stored = pair_rule_cache.emplace(key, std::move(rule))
                                 .first->second;
      return integrate_pairs(*stored.test_values,
                             *stored.trial_values,
                             stored.indices,
                             stored.weights,
                             operator_type,
                             trial_function,
                             regularize_trial_function);
    }

  return integrate_pairs(*test_values_owned,
                         *trial_values_owned,
                         pair_indices,
                         pair_weights,
                         operator_type,
                         trial_function,
                         regularize_trial_function);
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_pairs(
  const FEValues<2, spacedim>                    &test_values,
  const FEValues<2, spacedim>                    &trial_values,
  const std::vector<std::array<unsigned int, 2>> &pair_indices,
  const std::vector<double>                      &pair_weights,
  const OperatorType                              operator_type,
  const Function<spacedim>                       *trial_function,
  const bool regularize_trial_function) const
{
  // The trial function is a property of a quadrature point, not of a pair.
  // In the factorised regular branch the pair list is order^4 long while each
  // side holds only order^2 distinct points, so evaluating it inside the pair
  // loop would repeat every call about order^2 times over - and these calls
  // are typically into a parsed symbolic expression, the most expensive thing
  // in the loop.
  AssertThrow(operator_type != OperatorType::weak_hypersingular ||
                screening == 0.0,
              ExcMessage("The weak hypersingular operator is implemented for "
                         "the Laplace kernel only."));
  const auto kernel_start = std::chrono::steady_clock::now();
  const unsigned int n_test_points  = test_values.get_quadrature().size();
  const unsigned int n_trial_points = trial_values.get_quadrature().size();
  std::vector<double>              trial_scalar;
  std::vector<double>              test_scalar;
  std::vector<double>              trial_flux;
  std::vector<Tensor<1, spacedim>> trial_gradient;
  const auto function_start = std::chrono::steady_clock::now();
  if (trial_function != nullptr)
    {
      profile.function_calls +=
        n_trial_points + (regularize_trial_function ? n_test_points : 0);
      if (operator_type == OperatorType::double_layer ||
          operator_type == OperatorType::single_and_double_layer)
        {
          trial_scalar.resize(n_trial_points);
          for (unsigned int k = 0; k < n_trial_points; ++k)
            trial_scalar[k] =
              trial_function->value(trial_values.quadrature_point(k));
          if (regularize_trial_function)
            {
              test_scalar.resize(n_test_points);
              for (unsigned int k = 0; k < n_test_points; ++k)
                test_scalar[k] =
                  trial_function->value(test_values.quadrature_point(k));
            }
        }
      else if (operator_type == OperatorType::single_layer ||
               operator_type == OperatorType::adjoint_double_layer)
        {
          trial_flux.resize(n_trial_points);
          for (unsigned int k = 0; k < n_trial_points; ++k)
            {
              Tensor<1, spacedim> gradient;
              for (unsigned int d = 0; d < spacedim; ++d)
                gradient[d] =
                  trial_function->value(trial_values.quadrature_point(k), d);
              trial_flux[k] = gradient * trial_values.normal_vector(k);
            }
        }
      else
        {
          trial_gradient.resize(n_trial_points);
          for (unsigned int k = 0; k < n_trial_points; ++k)
            for (unsigned int d = 0; d < spacedim; ++d)
              trial_gradient[k][d] =
                trial_function->value(trial_values.quadrature_point(k), d);
        }
    }

  profile.function_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                function_start)
                                .count();

  const bool fused = operator_type == OperatorType::single_and_double_layer;
  FullMatrix<double> local_matrix(finite_element.dofs_per_cell,
                                  n_result_columns(operator_type,
                                                   trial_function));
  for (unsigned int p = 0; p < pair_weights.size(); ++p)
    {
      const unsigned int        q  = pair_indices[p][0];
      const unsigned int        qt = pair_indices[p][1];
      const Tensor<1, spacedim> displacement =
        trial_values.quadrature_point(qt) - test_values.quadrature_point(q);
      const double distance = displacement.norm();
      const double single = single_layer_kernel(distance);
      // For the Laplace kernel dG/dn_y = -(y - x).n_y / (4 pi r^3) for the
      // double layer and dG/dn_x = (y - x).n_x / (4 pi r^3) for its adjoint;
      // the screened kernel multiplies both by E(r).
      const double laplace_double_layer =
        operator_type == OperatorType::double_layer || fused ?
          -(displacement * trial_values.normal_vector(qt)) /
            (4.0 * numbers::PI * distance * distance * distance) :
          0.0;
      const double factor = screening_factor(distance);
      const double kernel =
        operator_type == OperatorType::double_layer ?
          laplace_double_layer * factor :
        operator_type == OperatorType::adjoint_double_layer ?
          (displacement * test_values.normal_vector(q)) * factor /
            (4.0 * numbers::PI * distance * distance * distance) :
          single;
      const double weight = pair_weights[p] * test_values.JxW(q) *
                            trial_values.JxW(qt);
      AssertThrow(std::isfinite(kernel) && std::isfinite(weight),
                  ExcMessage("Non-finite Galerkin panel-pair integrand."));
      for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
        if (fused)
          {
            const double test_value = test_values.shape_value(i, q) * weight;
            for (unsigned int j = 0; j < finite_element.dofs_per_cell; ++j)
              local_matrix(i, j) +=
                test_value * single * trial_values.shape_value(j, qt);
            local_matrix(i, finite_element.dofs_per_cell) +=
              test_value * laplace_double_layer *
              (factor * trial_scalar[qt] -
               (regularize_trial_function ? test_scalar[q] : 0.0));
          }
        else if (trial_function == nullptr)
          for (unsigned int j = 0; j < finite_element.dofs_per_cell; ++j)
            if (operator_type == OperatorType::weak_hypersingular)
              {
                const Tensor<1, spacedim> test_curl = cross_product_3d(
                  test_values.normal_vector(q), test_values.shape_grad(i, q));
                const Tensor<1, spacedim> trial_curl = cross_product_3d(
                  trial_values.normal_vector(qt),
                  trial_values.shape_grad(j, qt));
                local_matrix(i, j) +=
                  (test_curl * trial_curl) * single * weight;
              }
            else
              local_matrix(i, j) += test_values.shape_value(i, q) * kernel *
                                    trial_values.shape_value(j, qt) * weight;
        else if (operator_type == OperatorType::weak_hypersingular)
          {
            const auto test_curl = cross_product_3d(
              test_values.normal_vector(q), test_values.shape_grad(i, q));
            // The trial curl is taken at the trial point: with the factorised
            // regular rule the two indices differ, and using the test index
            // here would evaluate the trial normal at the wrong point.
            const auto trial_curl = cross_product_3d(
              trial_values.normal_vector(qt), trial_gradient[qt]);
            local_matrix(i, 0) +=
              (test_curl * trial_curl) * single * weight;
          }
        else if (operator_type == OperatorType::adjoint_double_layer)
          local_matrix(i, 0) += test_values.shape_value(i, q) * kernel *
                                trial_flux[qt] * weight;
        else if (operator_type == OperatorType::single_layer)
          {
            // V applied to a flux given as a function rather than as a
            // finite element field: the trial function supplies the exact
            // gradient and its normal trace is the flux. This builds the
            // right-hand side that Galerkin orthogonality actually asks for,
            // <V lambda, phi_i>, through the single-layer kernel - which
            // behaves like 1/r and is what the Sauter-Schwab rule is built
            // for - instead of through the double layer's 1/r^2.
            local_matrix(i, 0) += test_values.shape_value(i, q) * single *
                                  trial_flux[qt] * weight;
          }
        else
          {
            AssertThrow(operator_type == OperatorType::double_layer,
                        ExcInternalError());
            // The regularised form integrates the Laplace kernel against
            // E(r) phi(y) - phi(x); with E = 1 this is the subtraction of
            // the solid-angle identity, and for kappa > 0 the remainder
            // (E - 1) phi(x) has a bounded kernel.
            const double function_value =
              factor * trial_scalar[qt] -
              (regularize_trial_function ? test_scalar[q] : 0.0);
            local_matrix(i, 0) += test_values.shape_value(i, q) *
                                  laplace_double_layer * function_value *
                                  weight;
          }
    }
  profile.kernel_seconds += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - kernel_start)
                              .count();
  return local_matrix;
}

template <int spacedim>
Vector<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_single_layer_flux(
  const CellIterator       &test_cell,
  const CellIterator       &trial_cell,
  const unsigned int        order,
  const Function<spacedim> &gradient) const
{
  AssertDimension(gradient.n_components, spacedim);
  const auto column = integrate_operator(test_cell, trial_cell, order,
                                         OperatorType::single_layer,
                                         &gradient, false);
  Vector<double> result(column.m());
  for (unsigned int i = 0; i < column.m(); ++i)
    result[i] = column(i, 0);
  return result;
}


template <int spacedim>
Vector<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_hypersingular_rhs(
  const CellIterator &test_cell, const CellIterator &trial_cell,
  const unsigned int order, const Function<spacedim> &gradient) const
{
  AssertDimension(gradient.n_components, spacedim);
  const auto column = integrate_operator(test_cell, trial_cell, order,
    OperatorType::weak_hypersingular, &gradient);
  Vector<double> result(column.m());
  for (unsigned int i = 0; i < column.m(); ++i)
    result[i] = column(i, 0);
  return result;
}

template <int spacedim>
double
GalerkinPanelPairIntegrator<spacedim>::error_energy_leaf(
  const CellIterator        &test_cell,
  const ReferencePatch      &test_patch,
  const CellIterator        &trial_cell,
  const ReferencePatch      &trial_patch,
  const unsigned int         order,
  const Function<spacedim>  &exact_gradient,
  const std::vector<double> &test_coefficients,
  const std::vector<double> &trial_coefficients) const
{
  if (light_geometry)
    return error_energy_light(test_cell,
                              test_patch,
                              trial_cell,
                              trial_patch,
                              order,
                              exact_gradient,
                              test_coefficients,
                              trial_coefficients);

  std::vector<Point<2>>                    test_reference_points;
  std::vector<Point<2>>                    trial_reference_points;
  std::vector<std::array<unsigned int, 2>> pair_indices;
  std::vector<double>                      pair_weights;
  pair_quadrature(test_cell,
                  test_patch,
                  trial_cell,
                  trial_patch,
                  order,
                  test_reference_points,
                  trial_reference_points,
                  pair_indices,
                  pair_weights);

  const std::vector<double> unit_weights(test_reference_points.size(), 1.0);
  const Quadrature<2>       test_quadrature(test_reference_points,
                                            unit_weights);
  const Quadrature<2>       trial_quadrature(trial_reference_points,
                                             unit_weights);
  const UpdateFlags         flags = update_values | update_normal_vectors |
                            update_quadrature_points | update_JxW_values;
  FEValues<2, spacedim> test_values(mapping, finite_element, test_quadrature,
                                    flags);
  FEValues<2, spacedim> trial_values(mapping, finite_element, trial_quadrature,
                                     flags);
  test_values.reinit(test_cell);
  trial_values.reinit(trial_cell);

  // The error is a property of a quadrature point, not of a pair, so it is
  // evaluated once per point per side. In the factorised regular branch the
  // pair list is order^4 long while each side holds only order^2 distinct
  // points, so evaluating it inside the pair loop would repeat every exact
  // gradient and every shape function about order^2 times over.
  const auto errors_at = [&](const FEValues<2, spacedim> &values,
                             const std::vector<double>   &coefficients) {
    const unsigned int  n_points = values.get_quadrature().size();
    std::vector<double> result(n_points);
    Vector<double>      gradient(spacedim);
    for (unsigned int q = 0; q < n_points; ++q)
      {
        exact_gradient.vector_value(values.quadrature_point(q), gradient);
        Tensor<1, spacedim> normal = values.normal_vector(q);
        const double        norm   = normal.norm();
        AssertThrow(norm > 1e-14,
                    ExcMessage("Cannot evaluate the exact flux with a zero "
                               "normal."));
        normal /= norm;
        double exact = 0.0;
        for (unsigned int d = 0; d < spacedim; ++d)
          exact += gradient[d] * normal[d];
        double discrete = 0.0;
        for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
          discrete += coefficients[i] * values.shape_value(i, q);
        result[q] = exact - discrete;
      }
    return result;
  };

  const std::vector<double> test_error =
    errors_at(test_values, test_coefficients);
  const std::vector<double> trial_error =
    errors_at(trial_values, trial_coefficients);

  double contribution = 0.0;
  for (unsigned int p = 0; p < pair_weights.size(); ++p)
    {
      const unsigned int        q  = pair_indices[p][0];
      const unsigned int        qt = pair_indices[p][1];
      const Tensor<1, spacedim> displacement =
        trial_values.quadrature_point(qt) - test_values.quadrature_point(q);
      const double distance = displacement.norm();
      const double kernel   = single_layer_kernel(distance);
      const double weight =
        pair_weights[p] * test_values.JxW(q) * trial_values.JxW(qt);
      AssertThrow(std::isfinite(kernel) && std::isfinite(weight),
                  ExcMessage("Non-finite energy-error integrand."));
      contribution += test_error[q] * kernel * trial_error[qt] * weight;
    }
  return contribution;
}


template <int spacedim>
double
GalerkinPanelPairIntegrator<spacedim>::error_energy_adaptive(
  const CellIterator        &test_cell,
  const ReferencePatch      &test_patch,
  const CellIterator        &trial_cell,
  const ReferencePatch      &trial_patch,
  const unsigned int         order,
  const Function<spacedim>  &exact_gradient,
  const std::vector<double> &test_coefficients,
  const std::vector<double> &trial_coefficients,
  const unsigned int         depth) const
{
  if (!is_near_singular(test_cell, test_patch, trial_cell, trial_patch))
    return error_energy_leaf(test_cell, test_patch, trial_cell, trial_patch,
                             order, exact_gradient, test_coefficients,
                             trial_coefficients);

  if (depth >= near_singular_maximum_depth)
    return error_energy_leaf(test_cell, test_patch, trial_cell, trial_patch,
                             std::max(order, near_singular_quadrature_order),
                             exact_gradient, test_coefficients,
                             trial_coefficients);

  const auto   test_points  = patch_sample_points(test_cell, test_patch);
  const auto   trial_points = patch_sample_points(trial_cell, trial_patch);
  const double test_diameter =
    test_points.front().distance(test_points.back());
  const double trial_diameter =
    trial_points.front().distance(trial_points.back());
  const bool split_test =
    test_diameter > trial_diameter * (1.0 + 1e-12) ||
    (std::abs(test_diameter - trial_diameter) <=
       1e-12 * std::max({1.0, test_diameter, trial_diameter}) &&
     test_cell->active_cell_index() < trial_cell->active_cell_index());

  double total = 0.0;
  if (split_test)
    for (const auto &child : split_patch(test_patch))
      total += error_energy_adaptive(test_cell, child, trial_cell, trial_patch,
                                     order, exact_gradient, test_coefficients,
                                     trial_coefficients, depth + 1);
  else
    for (const auto &child : split_patch(trial_patch))
      total += error_energy_adaptive(test_cell, test_patch, trial_cell, child,
                                     order, exact_gradient, test_coefficients,
                                     trial_coefficients, depth + 1);
  return total;
}


template <int spacedim>
double
GalerkinPanelPairIntegrator<spacedim>::integrate_error_energy(
  const CellIterator        &test_cell,
  const CellIterator        &trial_cell,
  const unsigned int         order,
  const Function<spacedim>  &exact_gradient,
  const std::vector<double> &test_coefficients,
  const std::vector<double> &trial_coefficients) const
{
  AssertDimension(exact_gradient.n_components, spacedim);
  AssertDimension(test_coefficients.size(), finite_element.dofs_per_cell);
  AssertDimension(trial_coefficients.size(), finite_element.dofs_per_cell);

  const ReferencePatch full_patch{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)};
  const GalerkinPanelPairType pair_type = classify(test_cell, trial_cell);
  const unsigned int          effective_order =
    pair_type == GalerkinPanelPairType::regular || singular_order == 0 ?
      order :
      singular_order;

  if (pair_type == GalerkinPanelPairType::regular &&
      is_near_singular(test_cell, full_patch, trial_cell, full_patch))
    return error_energy_adaptive(test_cell, full_patch, trial_cell, full_patch,
                                 order, exact_gradient, test_coefficients,
                                 trial_coefficients, 0);

  if (!share_nonconforming_edge(test_cell, trial_cell))
    return error_energy_balanced(test_cell, full_patch, trial_cell,
                                 full_patch, order, exact_gradient,
                                 test_coefficients, trial_coefficients, 0);

  // Non-conforming edge: pair the coarse cell's matching subpatches with the
  // fine neighbour, exactly as the operator integrals do.
  const unsigned int level_difference =
    test_cell->level() > trial_cell->level() ?
      test_cell->level() - trial_cell->level() :
      trial_cell->level() - test_cell->level();
  double total = 0.0;
  for (const ReferencePatch &coarse_patch : uniform_subpatches(level_difference))
    total += test_cell->level() < trial_cell->level() ?
               error_energy_leaf(test_cell, coarse_patch, trial_cell,
                                 full_patch, effective_order, exact_gradient,
                                 test_coefficients, trial_coefficients) :
               error_energy_leaf(test_cell, full_patch, trial_cell,
                                 coarse_patch, effective_order, exact_gradient,
                                 test_coefficients, trial_coefficients);
  return total;
}

template <int spacedim>
bool
GalerkinPanelPairIntegrator<spacedim>::set_light_geometry(const bool enable)
{
  light_geometry = false;
  light_cells.clear();
  light_cell_valid.clear();
  light_rule_cache.clear();
  light_check_values.reset();
  if (!enable)
    return false;

  const auto *polynomial_mapping =
    dynamic_cast<const MappingQ<spacedim - 1, spacedim> *>(&mapping);
  if (polynomial_mapping == nullptr)
    return false;
  const unsigned int degree = std::max(1U, polynomial_mapping->get_degree());
  AssertThrow(degree + 1 <= maximum_lagrange_nodes,
              ExcMessage("Mapping degree too high for the direct path."));
  const QGaussLobatto<1> lobatto(degree + 1);
  geometry_nodes.clear();
  for (unsigned int i = 0; i < lobatto.size(); ++i)
    geometry_nodes.push_back(lobatto.point(i)[0]);

  // FE_DGQ and FE_Q are tensor Lagrange bases on their support points:
  // basis function k is L_a(x) L_b(y), with (a, b) the position of its
  // support point in the tensor grid. FE_DGQ numbers the points
  // lexicographically, FE_Q vertices first; the position is read from the
  // support point itself, and the basis is checked at a generic point. Any
  // other element is evaluated through the library.
  shape_nodes.clear();
  shape_tensor_index.clear();
  if (finite_element.has_support_points())
    {
      const std::vector<Point<2>> &support =
        finite_element.get_unit_support_points();
      std::vector<double> nodes;
      for (const Point<2> &point : support)
        if (std::none_of(nodes.begin(), nodes.end(), [&](const double x) {
              return std::abs(x - point[0]) < 1e-12;
            }))
          nodes.push_back(point[0]);
      std::sort(nodes.begin(), nodes.end());
      const unsigned int n1 = nodes.size();
      bool tensor = n1 * n1 == support.size() && n1 <= maximum_lagrange_nodes;
      const auto position = [&](const double x) {
        for (unsigned int a = 0; a < n1; ++a)
          if (std::abs(nodes[a] - x) < 1e-12)
            return a;
        return numbers::invalid_unsigned_int;
      };
      std::vector<std::array<unsigned int, 2>> index(support.size());
      std::vector<bool>                        seen(n1 * n1, false);
      for (unsigned int k = 0; tensor && k < support.size(); ++k)
        {
          index[k] = {{position(support[k][0]), position(support[k][1])}};
          if (index[k][0] == numbers::invalid_unsigned_int ||
              index[k][1] == numbers::invalid_unsigned_int ||
              seen[index[k][0] + n1 * index[k][1]])
            tensor = false;
          else
            seen[index[k][0] + n1 * index[k][1]] = true;
        }
      if (tensor)
        {
          const std::vector<double> coefficients =
            monomial_coefficients(nodes);
          const Point<2> probe(0.3141, 0.7182);
          LagrangeValues v0, v1;
          lagrange_basis(coefficients, n1, probe[0], v0, nullptr);
          lagrange_basis(coefficients, n1, probe[1], v1, nullptr);
          for (unsigned int k = 0; tensor && k < support.size(); ++k)
            if (std::abs(v0[index[k][0]] * v1[index[k][1]] -
                         finite_element.shape_value(k, probe)) > 1e-12)
              tensor = false;
        }
      if (tensor)
        {
          shape_nodes        = nodes;
          shape_tensor_index = index;
        }
    }
  geometry_coefficients = monomial_coefficients(geometry_nodes);
  shape_coefficients    = monomial_coefficients(shape_nodes);
  light_geometry = true;
  return true;
}


template <int spacedim>
const typename GalerkinPanelPairIntegrator<spacedim>::LightCell &
GalerkinPanelPairIntegrator<spacedim>::light_cell(
  const CellIterator &cell) const
{
  const unsigned int index = cell->active_cell_index();
  if (light_cells.size() <= index)
    {
      const unsigned int n = cell->get_triangulation().n_active_cells();
      light_cells.resize(std::max<unsigned int>(n, index + 1));
      light_cell_valid.resize(light_cells.size(), false);
    }
  if (light_cell_valid[index])
    return light_cells[index];

  LightCell         &data = light_cells[index];
  const unsigned int n    = geometry_nodes.size();
  data.nodes.resize(n * n);
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < n; ++i)
      data.nodes[i + n * j] = mapping.transform_unit_to_real_cell(
        cell, Point<2>(geometry_nodes[i], geometry_nodes[j]));
  data.orientation       = 1.0;
  light_cell_valid[index] = true;

  // The orientation of the normal is the library's, and the reconstructed
  // map must reproduce the library's on every cell it is used on.
  if (!light_check_values)
    light_check_values = std::make_unique<FEValues<2, spacedim>>(
      mapping,
      finite_element,
      QGauss<2>(3),
      update_quadrature_points | update_normal_vectors | update_JxW_values);
  FEValues<2, spacedim> &check = *light_check_values;
  check.reinit(cell);
  const Quadrature<2> &rule = check.get_quadrature();
  PointSet             set;
  evaluate_geometry(cell, rule.get_points(), set);
  data.orientation = set.normals[0] * check.normal_vector(0) < 0.0 ? -1.0 : 1.0;
  const double diameter = cell->diameter();
  for (unsigned int q = 0; q < rule.size(); ++q)
    {
      const double deviation = std::max(
        {set.points[q].distance(check.quadrature_point(q)) / diameter,
         (data.orientation * set.normals[q] - check.normal_vector(q)).norm(),
         std::abs(set.jacobians[q] * rule.weight(q) - check.JxW(q)) /
           check.JxW(q)});
      maximum_light_deviation = std::max(maximum_light_deviation, deviation);
    }
  AssertThrow(maximum_light_deviation < 1e-9,
              ExcMessage("The reconstructed cell geometry does not reproduce "
                         "the mapping."));
  return data;
}


template <int spacedim>
Point<spacedim>
GalerkinPanelPairIntegrator<spacedim>::map_to_real_cell(
  const CellIterator &cell,
  const Point<2>     &reference) const
{
  if (!light_geometry)
    return mapping.transform_unit_to_real_cell(cell, reference);
  const LightCell   &data = light_cell(cell);
  const unsigned int n    = geometry_nodes.size();
  LagrangeValues     v0, v1;
  lagrange_basis(geometry_coefficients, geometry_nodes.size(), reference[0], v0, nullptr);
  lagrange_basis(geometry_coefficients, geometry_nodes.size(), reference[1], v1, nullptr);
  Tensor<1, spacedim> x;
  for (unsigned int j = 0; j < n; ++j)
    {
      Tensor<1, spacedim> along;
      for (unsigned int i = 0; i < n; ++i)
        along += v0[i] * data.nodes[i + n * j];
      x += v1[j] * along;
    }
  return Point<spacedim>(x);
}


template <int spacedim>
void
GalerkinPanelPairIntegrator<spacedim>::evaluate_geometry(
  const CellIterator          &cell,
  const std::vector<Point<2>> &reference_points,
  PointSet                    &set) const
{
  const LightCell   &data = light_cell(cell);
  const unsigned int n    = geometry_nodes.size();
  const unsigned int m    = reference_points.size();
  set.n                   = m;
  set.points.resize(m);
  set.normals.resize(m);
  set.jacobians.resize(m);

  // All loops run over the points innermost, where they vectorize: the
  // Lagrange basis and its derivative in each direction by Horner's scheme,
  // then the sums over the nodes for the position and the two tangents.
  geometry_work.resize((4 * n + 9) * m);
  double *const v0 = geometry_work.data();
  double *const d0 = v0 + n * m;
  double *const v1 = d0 + n * m;
  double *const d1 = v1 + n * m;
  double *const x  = d1 + n * m; // 3 components, then x1, then x2
  double *const x1 = x + 3 * m;
  double *const x2 = x1 + 3 * m;
  const double *coefficients = geometry_coefficients.data();
  for (unsigned int direction = 0; direction < 2; ++direction)
    {
      double *const v = direction == 0 ? v0 : v1;
      double *const d = direction == 0 ? d0 : d1;
      for (unsigned int i = 0; i < n; ++i)
        {
          const double *c  = coefficients + i * n;
          const double *cd = coefficients + (n + i) * n;
          double       *vi = v + i * m;
          double       *di = d + i * m;
#pragma omp simd
          for (unsigned int q = 0; q < m; ++q)
            {
              const double t     = reference_points[q][direction];
              double       value = c[n - 1];
              for (unsigned int k = n - 1; k > 0; --k)
                value = value * t + c[k - 1];
              double derivative = n > 1 ? cd[n - 2] : 0.0;
              for (unsigned int k = n > 1 ? n - 2 : 0; k > 0; --k)
                derivative = derivative * t + cd[k - 1];
              vi[q] = value;
              di[q] = derivative;
            }
        }
    }
  std::fill(x, x + 9 * m, 0.0);
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < n; ++i)
      {
        const Tensor<1, spacedim> &node = data.nodes[i + n * j];
        const double *vi = v0 + i * m, *di = d0 + i * m;
        const double *vj = v1 + j * m, *dj = d1 + j * m;
        for (unsigned int c = 0; c < 3; ++c)
          {
            const double value = node[c];
            double      *xc = x + c * m, *x1c = x1 + c * m, *x2c = x2 + c * m;
#pragma omp simd
            for (unsigned int q = 0; q < m; ++q)
              {
                xc[q] += value * vi[q] * vj[q];
                x1c[q] += value * di[q] * vj[q];
                x2c[q] += value * vi[q] * dj[q];
              }
          }
      }
  for (unsigned int q = 0; q < m; ++q)
    {
      Tensor<1, spacedim> tangent1, tangent2;
      for (unsigned int c = 0; c < 3; ++c)
        {
          set.points[q][c] = x[c * m + q];
          tangent1[c]      = x1[c * m + q];
          tangent2[c]      = x2[c * m + q];
        }
      const Tensor<1, spacedim> normal   = cross_product_3d(tangent1, tangent2);
      const double              jacobian = normal.norm();
      set.normals[q]   = (data.orientation / jacobian) * normal;
      set.jacobians[q] = jacobian;
    }
}


template <int spacedim>
void
GalerkinPanelPairIntegrator<spacedim>::evaluate_shapes(
  const std::vector<Point<2>> &reference_points,
  std::vector<double>         &shapes) const
{
  const unsigned int dpc = finite_element.dofs_per_cell;
  const unsigned int m   = reference_points.size();
  shapes.resize(dpc * m);
  if (shape_nodes.empty())
    {
      for (unsigned int k = 0; k < dpc; ++k)
        for (unsigned int q = 0; q < m; ++q)
          shapes[k * m + q] =
            finite_element.shape_value(k, reference_points[q]);
      return;
    }
  LagrangeValues v0, v1;
  for (unsigned int q = 0; q < m; ++q)
    {
      lagrange_basis(shape_coefficients, shape_nodes.size(), reference_points[q][0], v0, nullptr);
      lagrange_basis(shape_coefficients, shape_nodes.size(), reference_points[q][1], v1, nullptr);
      for (unsigned int k = 0; k < dpc; ++k)
        shapes[k * m + q] =
          v0[shape_tensor_index[k][0]] * v1[shape_tensor_index[k][1]];
    }
}


template <int spacedim>
bool
GalerkinPanelPairIntegrator<spacedim>::takes_light_path(
  const OperatorType        operator_type,
  const Function<spacedim> *trial_function) const
{
  return light_geometry &&
         ((operator_type == OperatorType::single_layer &&
           trial_function == nullptr) ||
          operator_type == OperatorType::double_layer ||
          operator_type == OperatorType::adjoint_double_layer ||
          operator_type == OperatorType::single_and_double_layer ||
          operator_type == OperatorType::single_and_double_layer_both);
}


template <int spacedim>
void
GalerkinPanelPairIntegrator<spacedim>::prepare_light_leaf(
  const CellIterator   &test_cell,
  const ReferencePatch &test_patch,
  const CellIterator   &trial_cell,
  const ReferencePatch &trial_patch,
  const unsigned int    order,
  LightLeaf            &leaf) const
{
  const GalerkinPanelPairType type =
    classify(test_cell, test_patch, trial_cell, trial_patch);
  AssertThrow(type != GalerkinPanelPairType::unsupported,
              ExcMessage("Unsupported overlapping panel pair."));

  if (type == GalerkinPanelPairType::regular)
    {
      // A tensor Gauss rule on each patch: the pair rule is their product,
      // so each side is evaluated at its own points only.
      auto rule_hit = gauss_rules.find(order);
      if (rule_hit == gauss_rules.end())
        rule_hit = gauss_rules.emplace(order, QGauss<2>(order)).first;
      const QGauss<2>   &rule = rule_hit->second;
      const unsigned int m    = rule.size();
      std::vector<Point<2>> test_points(m), trial_points(m);
      for (unsigned int q = 0; q < m; ++q)
        {
          test_points[q] = transform_patch_point(rule.point(q),
                                                 test_patch.lower,
                                                 test_patch.upper);
          trial_points[q] = transform_patch_point(rule.point(q),
                                                  trial_patch.lower,
                                                  trial_patch.upper);
        }
      evaluate_geometry(test_cell, test_points, test_set);
      evaluate_shapes(test_points, test_set.shapes);
      evaluate_geometry(trial_cell, trial_points, trial_set);
      evaluate_shapes(trial_points, trial_set.shapes);
      const double test_patch_jacobian =
        (test_patch.upper[0] - test_patch.lower[0]) *
        (test_patch.upper[1] - test_patch.lower[1]);
      const double trial_patch_jacobian =
        (trial_patch.upper[0] - trial_patch.lower[0]) *
        (trial_patch.upper[1] - trial_patch.lower[1]);
      leaf.tensor = true;
      leaf.test_weights.resize(m);
      leaf.trial_weights.resize(m);
      for (unsigned int q = 0; q < m; ++q)
        {
          leaf.test_weights[q] =
            rule.weight(q) * test_patch_jacobian * test_set.jacobians[q];
          leaf.trial_weights[q] =
            rule.weight(q) * trial_patch_jacobian * trial_set.jacobians[q];
        }
      leaf.test_shapes  = &test_set.shapes;
      leaf.trial_shapes = &trial_set.shapes;
      ++profile.uncached_pairs;
      return;
    }

  // A singular rule. In the coordinates of its two patches it depends only
  // on the configuration, so it is built once, with its shape values for
  // whole cells; the patches of the subdivision place it in their cells by
  // an affine map.
  const std::vector<CommonVertex> shared =
    type == GalerkinPanelPairType::same_panel ?
      std::vector<CommonVertex>() :
      common_vertices(test_cell, test_patch, trial_cell, trial_patch);
  std::array<int, 7> key{{static_cast<int>(type),
                          static_cast<int>(order),
                          0, -1, -1, -1, -1}};
  if (type == GalerkinPanelPairType::common_edge ||
      type == GalerkinPanelPairType::common_vertex)
    for (unsigned int k = 0; k < shared.size() && k < 2; ++k)
      {
        key[3 + 2 * k] = static_cast<int>(shared[k].test_index);
        key[4 + 2 * k] = static_cast<int>(shared[k].trial_index);
      }
  auto hit = light_rule_cache.find(key);
  if (hit == light_rule_cache.end())
    {
      ++profile.uncached_pairs;
      LightRule built;
      unit_pair_rule(type,
                     shared,
                     order,
                     built.test_points,
                     built.trial_points,
                     built.weights);
      evaluate_shapes(built.test_points, built.test_shapes);
      evaluate_shapes(built.trial_points, built.trial_shapes);
      hit = light_rule_cache.emplace(key, std::move(built)).first;
    }
  const LightRule &rule = hit->second;

  leaf.tensor = false;
  if (is_full_patch(test_patch) && is_full_patch(trial_patch))
    {
      evaluate_geometry(test_cell, rule.test_points, test_set);
      evaluate_geometry(trial_cell, rule.trial_points, trial_set);
      leaf.test_shapes  = &rule.test_shapes;
      leaf.trial_shapes = &rule.trial_shapes;
      leaf.weights      = &rule.weights;
      return;
    }
  std::vector<Point<2>> test_points(rule.test_points.size());
  std::vector<Point<2>> trial_points(rule.trial_points.size());
  for (unsigned int p = 0; p < test_points.size(); ++p)
    {
      test_points[p]  = transform_patch_point(rule.test_points[p],
                                             test_patch.lower,
                                             test_patch.upper);
      trial_points[p] = transform_patch_point(rule.trial_points[p],
                                              trial_patch.lower,
                                              trial_patch.upper);
    }
  evaluate_geometry(test_cell, test_points, test_set);
  evaluate_geometry(trial_cell, trial_points, trial_set);
  evaluate_shapes(test_points, leaf.own_test_shapes);
  evaluate_shapes(trial_points, leaf.own_trial_shapes);
  const double patch_jacobians =
    (test_patch.upper[0] - test_patch.lower[0]) *
    (test_patch.upper[1] - test_patch.lower[1]) *
    (trial_patch.upper[0] - trial_patch.lower[0]) *
    (trial_patch.upper[1] - trial_patch.lower[1]);
  leaf.own_weights = rule.weights;
  for (double &w : leaf.own_weights)
    w *= patch_jacobians;
  leaf.test_shapes  = &leaf.own_test_shapes;
  leaf.trial_shapes = &leaf.own_trial_shapes;
  leaf.weights      = &leaf.own_weights;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_light(
  const CellIterator       &test_cell,
  const ReferencePatch     &test_patch,
  const CellIterator       &trial_cell,
  const ReferencePatch     &trial_patch,
  const unsigned int        order,
  const OperatorType        operator_type,
  const Function<spacedim> *trial_function,
  const bool                regularize_trial_function) const
{
  const auto setup_start = std::chrono::steady_clock::now();
  LightLeaf  leaf;
  prepare_light_leaf(test_cell, test_patch, trial_cell, trial_patch, order,
                     leaf);
  profile.setup_seconds += std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - setup_start)
                             .count();
  FullMatrix<double> result =
    leaf.tensor ? integrate_tensor_sets(test_set,
                                        leaf.test_weights,
                                        trial_set,
                                        leaf.trial_weights,
                                        operator_type,
                                        trial_function,
                                        regularize_trial_function) :
                  integrate_paired_sets(test_set,
                                        *leaf.test_shapes,
                                        trial_set,
                                        *leaf.trial_shapes,
                                        *leaf.weights,
                                        operator_type,
                                        trial_function,
                                        regularize_trial_function);
  const double seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - setup_start)
                           .count();
  if (leaf.tensor)
    {
      ++profile.regular_leaves;
      profile.regular_seconds += seconds;
    }
  else
    {
      ++profile.singular_leaves;
      profile.singular_points += leaf.weights->size();
      profile.singular_seconds += seconds;
    }
  return result;
}


template <int spacedim>
double
GalerkinPanelPairIntegrator<spacedim>::error_energy_light(
  const CellIterator        &test_cell,
  const ReferencePatch      &test_patch,
  const CellIterator        &trial_cell,
  const ReferencePatch      &trial_patch,
  const unsigned int         order,
  const Function<spacedim>  &exact_gradient,
  const std::vector<double> &test_coefficients,
  const std::vector<double> &trial_coefficients) const
{
  LightLeaf leaf;
  prepare_light_leaf(test_cell, test_patch, trial_cell, trial_patch, order,
                     leaf);

  // The error, exact normal trace minus discrete flux, at the points of
  // each side, as in error_energy_leaf().
  const unsigned int dpc      = finite_element.dofs_per_cell;
  const auto         errors_at = [&](const PointSet            &set,
                                     const std::vector<double> &shapes,
                                     const std::vector<double> &coefficients,
                                     std::vector<double>       &error) {
    const unsigned int n = set.n;
    error.resize(n);
    Vector<double> gradient(spacedim);
    for (unsigned int q = 0; q < n; ++q)
      {
        exact_gradient.vector_value(set.points[q], gradient);
        double exact = 0.0;
        for (unsigned int d = 0; d < spacedim; ++d)
          exact += gradient[d] * set.normals[q][d];
        double discrete = 0.0;
        for (unsigned int i = 0; i < dpc; ++i)
          discrete += coefficients[i] * shapes[i * n + q];
        error[q] = exact - discrete;
      }
  };
  std::vector<double> test_error, trial_error;
  errors_at(test_set, *leaf.test_shapes, test_coefficients, test_error);
  errors_at(trial_set, *leaf.trial_shapes, trial_coefficients, trial_error);

  double contribution = 0.0;
  if (leaf.tensor)
    for (unsigned int qx = 0; qx < test_set.n; ++qx)
      {
        double sum = 0.0;
        for (unsigned int qy = 0; qy < trial_set.n; ++qy)
          sum += single_layer_kernel(
                   test_set.points[qx].distance(trial_set.points[qy])) *
                 trial_error[qy] * leaf.trial_weights[qy];
        contribution += test_error[qx] * leaf.test_weights[qx] * sum;
      }
  else
    for (unsigned int p = 0; p < leaf.weights->size(); ++p)
      contribution += test_error[p] *
                      single_layer_kernel(
                        test_set.points[p].distance(trial_set.points[p])) *
                      trial_error[p] * (*leaf.weights)[p] *
                      test_set.jacobians[p] * trial_set.jacobians[p];
  return contribution;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_tensor_sets(
  const PointSet            &test,
  const std::vector<double> &test_weights,
  const PointSet            &trial,
  const std::vector<double> &trial_weights,
  const OperatorType         operator_type,
  const Function<spacedim>  *trial_function,
  const bool                 regularize_trial_function) const
{
  const auto         kernel_start = std::chrono::steady_clock::now();
  const unsigned int nx           = test.n;
  const unsigned int ny           = trial.n;
  const unsigned int dpc          = finite_element.dofs_per_cell;
  const bool both = operator_type == OperatorType::single_and_double_layer_both;
  const bool fused =
    operator_type == OperatorType::single_and_double_layer || both;
  // A matrix block, of the single, the double or the adjoint double layer,
  // and a column of the double layer applied to the trial function, or of
  // the adjoint double layer applied to the normal trace of a gradient.
  const bool adjoint = operator_type == OperatorType::adjoint_double_layer;
  const bool matrix_part = fused || trial_function == nullptr;
  const bool single_kernel = operator_type == OperatorType::single_layer || fused;
  const bool function_part = trial_function != nullptr;
  const bool regularize    = function_part && regularize_trial_function &&
                          !adjoint;
  const unsigned int function_column = fused ? dpc : 0;
  // The consistent right-hand sides, <V lambda, phi>, in the last columns.
  const bool         flux        = fused && flux_function != nullptr;
  const unsigned int flux_column = dpc + (both ? 2 : 1);

  std::vector<double> trial_flux, test_flux;
  if (flux)
    {
      const auto function_start = std::chrono::steady_clock::now();
      trial_flux.resize(ny);
      for (unsigned int qy = 0; qy < ny; ++qy)
        trial_flux[qy] =
          normal_trace(*flux_function, trial.points[qy], trial.normals[qy]);
      if (both)
        {
          test_flux.resize(nx);
          for (unsigned int qx = 0; qx < nx; ++qx)
            test_flux[qx] =
              normal_trace(*flux_function, test.points[qx], test.normals[qx]);
        }
      profile.function_calls += ny + (both ? nx : 0);
      profile.function_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      function_start)
          .count();
    }

  std::vector<double> trial_values, test_values;
  if (function_part)
    {
      const auto function_start = std::chrono::steady_clock::now();
      trial_values.resize(ny);
      for (unsigned int qy = 0; qy < ny; ++qy)
        trial_values[qy] = adjoint ?
                             normal_trace(*trial_function, trial.points[qy],
                                          trial.normals[qy]) :
                             trial_function->value(trial.points[qy]);
      if (regularize || both)
        {
          test_values.resize(nx);
          for (unsigned int qx = 0; qx < nx; ++qx)
            test_values[qx] = trial_function->value(test.points[qx]);
        }
      profile.function_calls += ny + (regularize || both ? nx : 0);
      profile.function_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      function_start)
          .count();
    }

  std::vector<double> kernel(matrix_part ? nx * ny : 0);
  std::vector<double> weighted(function_part ? nx : 0);
  // The exchanged pair: the test side at the trial points, the kernel
  // dG/dn at the test points, the datum at the test points.
  std::vector<double> exchanged_function(both ? ny : 0),
    exchanged_kernel(both ? ny : 0);
  for (unsigned int qx = 0; qx < nx; ++qx)
    {
      double sum_function = 0.0;
      double sum_kernel   = 0.0;
      for (unsigned int qy = 0; qy < ny; ++qy)
        {
          const Tensor<1, spacedim> displacement =
            trial.points[qy] - test.points[qx];
          const double distance = displacement.norm();
          const double exponential =
            screening == 0.0 ? 1.0 : std::exp(-screening * distance);
          const double factor = (1.0 + screening * distance) * exponential;
          // For the adjoint double layer this is dG/dn_x instead,
          // (y - x).n_x / (4 pi r^3), times the screening factor.
          const double laplace_double_layer =
            adjoint ?
              (displacement * test.normals[qx]) /
                (4.0 * numbers::PI * distance * distance * distance) :
            (function_part || !single_kernel) ?
              -(displacement * trial.normals[qy]) /
                (4.0 * numbers::PI * distance * distance * distance) :
              0.0;
          if (matrix_part)
            kernel[qx * ny + qy] =
              trial_weights[qy] *
              (single_kernel ? exponential / (4.0 * numbers::PI * distance) :
                               laplace_double_layer * factor);
          if (function_part)
            {
              sum_function +=
                trial_weights[qy] * laplace_double_layer * factor *
                trial_values[qy];
              sum_kernel += trial_weights[qy] * laplace_double_layer;
            }
          if (both)
            {
              // -(x - y).n_x / (4 pi r^3) with the displacement y - x
              const double exchanged_double_layer =
                (displacement * test.normals[qx]) /
                (4.0 * numbers::PI * distance * distance * distance);
              exchanged_function[qy] += test_weights[qx] *
                                        exchanged_double_layer * factor *
                                        test_values[qx];
              exchanged_kernel[qy] += test_weights[qx] * exchanged_double_layer;
            }
        }
      if (function_part)
        weighted[qx] =
          test_weights[qx] *
          (sum_function - (regularize ? sum_kernel * test_values[qx] : 0.0));
    }

  FullMatrix<double> result(dpc, n_result_columns(operator_type,
                                                  trial_function));
  if (matrix_part)
    {
      // Contract the test index first, then the trial one: dpc (nx + dpc) ny
      // operations instead of dpc^2 nx ny.
      std::vector<double> partial(dpc * ny, 0.0);
      for (unsigned int i = 0; i < dpc; ++i)
        {
          double *row = &partial[i * ny];
          for (unsigned int qx = 0; qx < nx; ++qx)
            {
              const double  a = test.shapes[i * nx + qx] * test_weights[qx];
              const double *k = &kernel[qx * ny];
              for (unsigned int qy = 0; qy < ny; ++qy)
                row[qy] += a * k[qy];
            }
        }
      for (unsigned int i = 0; i < dpc; ++i)
        for (unsigned int j = 0; j < dpc; ++j)
          {
            const double *row = &partial[i * ny];
            const double *v   = &trial.shapes[j * ny];
            double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
            for (unsigned int qy = 0; qy < ny; ++qy)
              acc += row[qy] * v[qy];
            result(i, j) = acc;
          }
    }
  if (function_part)
    for (unsigned int i = 0; i < dpc; ++i)
      {
        double acc = 0.0;
        for (unsigned int qx = 0; qx < nx; ++qx)
          acc += test.shapes[i * nx + qx] * weighted[qx];
        result(i, function_column) = acc;
      }
  if (both)
    for (unsigned int j = 0; j < dpc; ++j)
      {
        double acc = 0.0;
        for (unsigned int qy = 0; qy < ny; ++qy)
          acc += trial.shapes[j * ny + qy] * trial_weights[qy] *
                 (exchanged_function[qy] -
                  (regularize ? exchanged_kernel[qy] * trial_values[qy] : 0.0));
        result(j, dpc + 1) = acc;
      }
  if (flux)
    {
      // The kernel holds the trial weights: sum over y of G lambda(y) w_y
      // for the test cell, and over x of w_x lambda(x) G w_y for the trial
      // cell.
      std::vector<double> at_test(nx, 0.0), at_trial(both ? ny : 0, 0.0);
      for (unsigned int qx = 0; qx < nx; ++qx)
        {
          const double *k   = &kernel[qx * ny];
          double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
          for (unsigned int qy = 0; qy < ny; ++qy)
            acc += k[qy] * trial_flux[qy];
          at_test[qx] = test_weights[qx] * acc;
          if (both)
            {
              const double a = test_weights[qx] * test_flux[qx];
              for (unsigned int qy = 0; qy < ny; ++qy)
                at_trial[qy] += a * k[qy];
            }
        }
      for (unsigned int i = 0; i < dpc; ++i)
        {
          double acc = 0.0;
          for (unsigned int qx = 0; qx < nx; ++qx)
            acc += test.shapes[i * nx + qx] * at_test[qx];
          result(i, flux_column) = acc;
        }
      if (both)
        for (unsigned int j = 0; j < dpc; ++j)
          {
            double acc = 0.0;
            for (unsigned int qy = 0; qy < ny; ++qy)
              acc += trial.shapes[j * ny + qy] * at_trial[qy];
            result(j, flux_column + 1) = acc;
          }
    }
  profile.kernel_seconds += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - kernel_start)
                              .count();
  return result;
}


template <int spacedim>
FullMatrix<double>
GalerkinPanelPairIntegrator<spacedim>::integrate_paired_sets(
  const PointSet            &test,
  const std::vector<double> &test_shapes,
  const PointSet            &trial,
  const std::vector<double> &trial_shapes,
  const std::vector<double> &weights,
  const OperatorType         operator_type,
  const Function<spacedim>  *trial_function,
  const bool                 regularize_trial_function) const
{
  const auto         kernel_start = std::chrono::steady_clock::now();
  const unsigned int n            = weights.size();
  const unsigned int dpc          = finite_element.dofs_per_cell;
  AssertDimension(test.n, n);
  AssertDimension(trial.n, n);
  const bool both = operator_type == OperatorType::single_and_double_layer_both;
  const bool fused =
    operator_type == OperatorType::single_and_double_layer || both;
  const bool adjoint       = operator_type == OperatorType::adjoint_double_layer;
  const bool matrix_part   = fused || trial_function == nullptr;
  const bool single_kernel = operator_type == OperatorType::single_layer || fused;
  const bool function_part = trial_function != nullptr;
  const bool regularize    = function_part && regularize_trial_function &&
                          !adjoint;
  const unsigned int function_column = fused ? dpc : 0;
  const bool         flux            = fused && flux_function != nullptr;
  const unsigned int flux_column     = dpc + (both ? 2 : 1);

  std::vector<double> trial_flux, test_flux;
  if (flux)
    {
      const auto function_start = std::chrono::steady_clock::now();
      trial_flux.resize(n);
      for (unsigned int p = 0; p < n; ++p)
        trial_flux[p] =
          normal_trace(*flux_function, trial.points[p], trial.normals[p]);
      if (both)
        {
          test_flux.resize(n);
          for (unsigned int p = 0; p < n; ++p)
            test_flux[p] =
              normal_trace(*flux_function, test.points[p], test.normals[p]);
        }
      profile.function_calls += (both ? 2 : 1) * n;
      profile.function_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      function_start)
          .count();
    }

  std::vector<double> trial_values, test_values;
  if (function_part)
    {
      const auto function_start = std::chrono::steady_clock::now();
      trial_values.resize(n);
      for (unsigned int p = 0; p < n; ++p)
        trial_values[p] = adjoint ?
                            normal_trace(*trial_function, trial.points[p],
                                         trial.normals[p]) :
                            trial_function->value(trial.points[p]);
      if (regularize || both)
        {
          test_values.resize(n);
          for (unsigned int p = 0; p < n; ++p)
            test_values[p] = trial_function->value(test.points[p]);
        }
      profile.function_calls += n + (regularize || both ? n : 0);
      profile.function_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      function_start)
          .count();
    }

  std::vector<double> matrix_weight(matrix_part ? n : 0);
  std::vector<double> function_weight(function_part ? n : 0);
  std::vector<double> exchanged_weight(both ? n : 0);
  bool                non_finite = false;
  for (unsigned int p = 0; p < n; ++p)
    {
      const Tensor<1, spacedim> displacement = trial.points[p] - test.points[p];
      const double              distance     = displacement.norm();
      const double              exponential =
        screening == 0.0 ? 1.0 : std::exp(-screening * distance);
      const double factor = (1.0 + screening * distance) * exponential;
      const double weight =
        weights[p] * test.jacobians[p] * trial.jacobians[p];
      const double laplace_double_layer =
        adjoint ?
          (displacement * test.normals[p]) /
            (4.0 * numbers::PI * distance * distance * distance) :
        (function_part || !single_kernel) ?
          -(displacement * trial.normals[p]) /
            (4.0 * numbers::PI * distance * distance * distance) :
          0.0;
      if (!(std::isfinite(weight) && std::isfinite(distance) &&
            distance > 0.0))
        non_finite = true;
      if (matrix_part)
        matrix_weight[p] =
          weight *
          (single_kernel ? exponential / (4.0 * numbers::PI * distance) :
                           laplace_double_layer * factor);
      if (function_part)
        function_weight[p] =
          weight * laplace_double_layer *
          (factor * trial_values[p] - (regularize ? test_values[p] : 0.0));
      if (both)
        // the pair exchanged: -(x - y).n_x / (4 pi r^3), datum at x
        exchanged_weight[p] =
          weight * (displacement * test.normals[p]) /
          (4.0 * numbers::PI * distance * distance * distance) *
          (factor * test_values[p] - (regularize ? trial_values[p] : 0.0));
    }

  AssertThrow(!non_finite,
              ExcMessage("Non-finite Galerkin panel-pair integrand."));
  FullMatrix<double> result(dpc, n_result_columns(operator_type,
                                                  trial_function));
  if (matrix_part)
    {
      std::vector<double> weighted_test(n);
      for (unsigned int i = 0; i < dpc; ++i)
        {
          const double *u = &test_shapes[i * n];
          for (unsigned int p = 0; p < n; ++p)
            weighted_test[p] = u[p] * matrix_weight[p];
          for (unsigned int j = 0; j < dpc; ++j)
            {
              const double *v   = &trial_shapes[j * n];
              double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
              for (unsigned int p = 0; p < n; ++p)
                acc += weighted_test[p] * v[p];
              result(i, j) = acc;
            }
        }
    }
  if (function_part)
    for (unsigned int i = 0; i < dpc; ++i)
      {
        const double *u   = &test_shapes[i * n];
        double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
        for (unsigned int p = 0; p < n; ++p)
          acc += u[p] * function_weight[p];
        result(i, function_column) = acc;
      }
  if (both)
    for (unsigned int j = 0; j < dpc; ++j)
      {
        const double *v   = &trial_shapes[j * n];
        double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
        for (unsigned int p = 0; p < n; ++p)
          acc += v[p] * exchanged_weight[p];
        result(j, dpc + 1) = acc;
      }
  if (flux)
    {
      for (unsigned int i = 0; i < dpc; ++i)
        {
          const double *u   = &test_shapes[i * n];
          double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
          for (unsigned int p = 0; p < n; ++p)
            acc += u[p] * matrix_weight[p] * trial_flux[p];
          result(i, flux_column) = acc;
        }
      if (both)
        for (unsigned int j = 0; j < dpc; ++j)
          {
            const double *v   = &trial_shapes[j * n];
            double        acc = 0.0;
#pragma omp simd reduction(+ : acc)
            for (unsigned int p = 0; p < n; ++p)
              acc += v[p] * matrix_weight[p] * test_flux[p];
            result(j, flux_column + 1) = acc;
          }
    }
  profile.kernel_seconds += std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - kernel_start)
                              .count();
  return result;
}


template class GalerkinPanelPairIntegrator<3>;
