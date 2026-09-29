#include "../include/surface_potential_evaluator.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace
{
  constexpr unsigned int maximum_nodes = 8;
  using Basis = std::array<double, maximum_nodes>;

  // Values, first and second derivatives of the Lagrange basis through the
  // given nodes, at t.
  void
  lagrange_basis(const std::vector<double> &nodes,
                 const double               t,
                 Basis                     &value,
                 Basis                     &first,
                 Basis                     &second)
  {
    const unsigned int n = nodes.size();
    for (unsigned int i = 0; i < n; ++i)
      {
        double v = 1.0;
        for (unsigned int k = 0; k < n; ++k)
          if (k != i)
            v *= (t - nodes[k]) / (nodes[i] - nodes[k]);
        value[i] = v;

        double d = 0.0;
        for (unsigned int m = 0; m < n; ++m)
          {
            if (m == i)
              continue;
            double term = 1.0 / (nodes[i] - nodes[m]);
            for (unsigned int k = 0; k < n; ++k)
              if (k != i && k != m)
                term *= (t - nodes[k]) / (nodes[i] - nodes[k]);
            d += term;
          }
        first[i] = d;

        double dd = 0.0;
        for (unsigned int m = 0; m < n; ++m)
          {
            if (m == i)
              continue;
            for (unsigned int l = 0; l < n; ++l)
              {
                if (l == i || l == m)
                  continue;
                double term =
                  1.0 / ((nodes[i] - nodes[m]) * (nodes[i] - nodes[l]));
                for (unsigned int k = 0; k < n; ++k)
                  if (k != i && k != m && k != l)
                    term *= (t - nodes[k]) / (nodes[i] - nodes[k]);
                dd += term;
              }
          }
        second[i] = dd;
      }
  }

  // The same basis in monomial form: rows 0..n-1 hold the coefficients of
  // the basis functions from the constant term up, rows n..2n-1 those of
  // their first derivatives and rows 2n..3n-1 those of their second. Horner's
  // rule then evaluates only the derivatives a caller asks for, with no
  // division; the product form above costs a cubic loop per point.
  std::vector<double>
  monomial_basis(const std::vector<double> &nodes)
  {
    const unsigned int  n = nodes.size();
    std::vector<double> result(3 * n * n, 0.0);
    for (unsigned int i = 0; i < n; ++i)
      {
        std::vector<double> c(n, 0.0);
        c[0]                = 1.0;
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
        for (unsigned int m = 2; m < n; ++m)
          result[(2 * n + i) * n + m - 2] = m * (m - 1) * c[m];
      }
    return result;
  }

  void
  horner_basis(const std::vector<double> &coefficients,
               const unsigned int         n,
               const double               t,
               Basis                     *value,
               Basis                     *first,
               Basis                     *second)
  {
    Basis *const out[3] = {value, first, second};
    for (unsigned int block = 0; block < 3; ++block)
      if (out[block] != nullptr)
        for (unsigned int i = 0; i < n; ++i)
          {
            const double *c = &coefficients[(block * n + i) * n];
            double        v = c[n - 1];
            for (unsigned int m = n - 1; m > 0; --m)
              v = v * t + c[m - 1];
            (*out[block])[i] = v;
          }
  }

  const QGauss<1> &
  gauss_rule(const unsigned int n)
  {
    static std::vector<std::unique_ptr<QGauss<1>>> rules;
    if (rules.size() <= n)
      rules.resize(n + 1);
    if (!rules[n])
      rules[n] = std::make_unique<QGauss<1>>(n);
    return *rules[n];
  }
} // namespace


SurfacePotentialEvaluator::SurfacePotentialEvaluator(
  const Mapping<2, 3>    &mapping,
  const DoFHandler<2, 3> &dof_handler,
  const unsigned int      geometry_degree,
  const Densities        &densities,
  const unsigned int      duffy_order,
  const double            near_ratio,
  const unsigned int      maximum_depth,
  const double            screening)
  : mapping(mapping)
  , dof_handler(dof_handler)
  , geometry_degree(std::max(1U, geometry_degree))
  , densities(densities)
  , duffy_order(duffy_order)
  , near_ratio(near_ratio)
  , maximum_depth(maximum_depth)
  , screening(screening)
  , wants_curl(densities.trace_gradient != nullptr)
  , gradient_buffer(3)
{
  AssertThrow((densities.flux_coefficients != nullptr) !=
                (densities.flux_gradient != nullptr),
              ExcMessage("Give the flux either as coefficients or as the "
                         "normal trace of a gradient."));
  AssertThrow((densities.trace_coefficients != nullptr) !=
                (densities.trace_function != nullptr),
              ExcMessage("Give the trace either as coefficients or as a "
                         "function."));

  AssertThrow(this->geometry_degree + 1 <= maximum_nodes,
              ExcMessage("Geometry degree too high for the evaluator."));
  const QGaussLobatto<1> lobatto(this->geometry_degree + 1);
  for (unsigned int i = 0; i < lobatto.size(); ++i)
    lagrange_nodes.push_back(lobatto.point(i)[0]);
  lagrange_coefficients = monomial_basis(lagrange_nodes);

  const FiniteElement<2, 3> &fe = dof_handler.get_fe();

  // Evaluating the element at arbitrary points through the library is the
  // most expensive part of a near-field source point. FE_DGQ and FE_Q are
  // tensor Lagrange bases on their support points: basis function k is
  // L_a(x) L_b(y), with (a, b) the position of its support point in the
  // tensor grid - lexicographic for FE_DGQ, vertices first for FE_Q. That is
  // checked here, and used if it holds.
  if (fe.has_support_points())
    {
      const std::vector<Point<2>> &support = fe.get_unit_support_points();
      std::vector<double>          nodes;
      for (const Point<2> &point : support)
        if (std::none_of(nodes.begin(), nodes.end(), [&](const double x) {
              return std::abs(x - point[0]) < 1e-12;
            }))
          nodes.push_back(point[0]);
      std::sort(nodes.begin(), nodes.end());
      const unsigned int n1 = nodes.size();
      bool tensor = n1 * n1 == support.size() && n1 <= maximum_nodes;
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
          const Point<2> probe(0.3141, 0.7182);
          Basis v0, d0, s0, v1, d1, s1;
          lagrange_basis(nodes, probe[0], v0, d0, s0);
          lagrange_basis(nodes, probe[1], v1, d1, s1);
          for (unsigned int k = 0; tensor && k < support.size(); ++k)
            if (std::abs(v0[index[k][0]] * v1[index[k][1]] -
                         fe.shape_value(k, probe)) > 1e-12)
              tensor = false;
        }
      if (tensor)
        {
          shape_nodes        = nodes;
          shape_tensor_index = index;
          shape_coefficients = monomial_basis(nodes);
        }
    }

  // Far-field rules, from the coarsest to the finest; evaluate() picks one
  // by the separation of the panel.
  const std::array<unsigned int, 5> far_orders{{3, 4, 6, 8, 12}};
  std::vector<std::unique_ptr<FEValues<2, 3>>> far_values;
  for (const unsigned int order : far_orders)
    far_values.push_back(std::make_unique<FEValues<2, 3>>(
      mapping,
      fe,
      QGauss<2>(order),
      update_values | update_quadrature_points | update_normal_vectors |
        update_JxW_values));

  std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
  cells.reserve(dof_handler.get_triangulation().n_active_cells());
  cell_position.assign(dof_handler.get_triangulation().n_active_cells(),
                       numbers::invalid_unsigned_int);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      CellData data;
      data.cell = cell;
      const unsigned int n = lagrange_nodes.size();
      data.nodes.resize(n * n);
      for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < n; ++i)
          data.nodes[i + n * j] = mapping.transform_unit_to_real_cell(
            cell, Point<2>(lagrange_nodes[i], lagrange_nodes[j]));

      cell->get_dof_indices(dof_indices);
      if (densities.flux_coefficients != nullptr)
        {
          data.flux_local.resize(fe.dofs_per_cell);
          for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
            data.flux_local[k] = (*densities.flux_coefficients)(dof_indices[k]);
        }
      if (densities.trace_coefficients != nullptr)
        {
          data.trace_local.resize(fe.dofs_per_cell);
          for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
            data.trace_local[k] =
              (*densities.trace_coefficients)(dof_indices[k]);
        }

      // A generous bounding sphere: curved cells bulge between the nodes.
      data.centre = geometry(data, Point<2>(0.5, 0.5)).point;
      double radius = 0.0;
      for (unsigned int j = 0; j <= 4; ++j)
        for (unsigned int i = 0; i <= 4; ++i)
          radius = std::max(radius,
                            (geometry(data, Point<2>(0.25 * i, 0.25 * j))
                               .point -
                             data.centre)
                              .norm());
      data.radius   = 1.05 * radius;
      data.diameter = 2.0 * data.radius;

      for (unsigned int r = 0; r < far_orders.size(); ++r)
        {
          FEValues<2, 3> &values = *far_values[r];
          values.reinit(cell);
          const Quadrature<2> &rule = values.get_quadrature();
          // The orientation of the reconstructed normal is taken from the
          // library's normal at the first point of the first rule.
          if (r == 0)
            {
              const Geometry g = geometry(data, rule.point(0));
              data.orientation =
                (g.normal * values.normal_vector(0) < 0.0) ? -1.0 : 1.0;
            }
          data.far_rules[r].resize(rule.size());
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              Source &s = data.far_rules[r][q];
              s.point   = values.quadrature_point(q);
              s.normal  = values.normal_vector(q);
              s.weight  = values.JxW(q);
              if (densities.flux_coefficients != nullptr)
                {
                  s.flux = 0.0;
                  for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
                    s.flux += data.flux_local[k] * values.shape_value(k, q);
                }
              else
                {
                  densities.flux_gradient->vector_value(s.point,
                                                        gradient_buffer);
                  s.flux = 0.0;
                  for (unsigned int d = 0; d < 3; ++d)
                    s.flux += gradient_buffer[d] * s.normal[d];
                }
              if (densities.trace_coefficients != nullptr)
                {
                  s.trace = 0.0;
                  for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
                    s.trace += data.trace_local[k] * values.shape_value(k, q);
                }
              else
                s.trace = densities.trace_function->value(s.point);
              if (wants_curl)
                {
                  densities.trace_gradient->vector_value(s.point,
                                                         gradient_buffer);
                  Tensor<1, 3> g;
                  for (unsigned int d = 0; d < 3; ++d)
                    g[d] = gradient_buffer[d];
                  s.curl_density = cross_product_3d(s.normal, g);
                }

              // The reconstructed map must reproduce the library's.
              if (r + 1 == far_orders.size())
                {
                  const Geometry g =
                    geometry(data, rule.point(q));
                  const double deviation =
                    std::max((g.point - s.point).norm() / data.diameter,
                             std::max((g.normal - s.normal).norm(),
                                      std::abs(g.jacobian * rule.weight(q) -
                                               s.weight) /
                                        std::max(s.weight, 1e-300)));
                  maximum_geometry_deviation =
                    std::max(maximum_geometry_deviation, deviation);
                }
            }
        }

      cell_position[cell->active_cell_index()] = cells.size();
      cells.push_back(std::move(data));
    }
}


SurfacePotentialEvaluator::Geometry
SurfacePotentialEvaluator::geometry(const CellIterator &cell,
                                    const Point<2>     &reference) const
{
  return geometry(cells[cell_position[cell->active_cell_index()]],
                  reference,
                  true);
}


SurfacePotentialEvaluator::Geometry
SurfacePotentialEvaluator::geometry(const CellData &data,
                                    const Point<2> &reference,
                                    const bool      with_curvature) const
{
  const unsigned int n = lagrange_nodes.size();
  Basis              v0, d0, s0, v1, d1, s1;
  horner_basis(lagrange_coefficients, n, reference[0], &v0, &d0,
               with_curvature ? &s0 : nullptr);
  horner_basis(lagrange_coefficients, n, reference[1], &v1, &d1,
               with_curvature ? &s1 : nullptr);

  Tensor<1, 3> x, x1, x2, x11, x12, x22;
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < n; ++i)
      {
        const Tensor<1, 3> &node = data.nodes[i + n * j];
        x += v0[i] * v1[j] * node;
        x1 += d0[i] * v1[j] * node;
        x2 += v0[i] * d1[j] * node;
        if (with_curvature)
          {
            x11 += s0[i] * v1[j] * node;
            x12 += d0[i] * d1[j] * node;
            x22 += v0[i] * s1[j] * node;
          }
      }

  Geometry g;
  g.point            = Point<3>(x);
  g.tangents         = {{x1, x2}};
  const Tensor<1, 3> N = cross_product_3d(x1, x2);
  g.jacobian         = N.norm();
  g.normal           = data.orientation * N / g.jacobian;
  if (!with_curvature)
    return g;

  // div_S n = -tr(G^{-1} II), with II_ab = n . x_ab (Weingarten).
  const double g11 = x1 * x1, g12 = x1 * x2, g22 = x2 * x2;
  const double det = g11 * g22 - g12 * g12;
  const double h11 = g.normal * x11, h12 = g.normal * x12,
               h22 = g.normal * x22;
  g.curvature = -(g22 * h11 - 2.0 * g12 * h12 + g11 * h22) / det;
  return g;
}


SurfacePotentialEvaluator::Source
SurfacePotentialEvaluator::source(const CellData &data,
                                  const Point<2> &reference,
                                  const double    weight) const
{
  ++n_sources;
  const Geometry g = geometry(data, reference, false);
  Source         s;
  s.point  = g.point;
  s.normal = g.normal;
  s.weight = weight * g.jacobian;

  const FiniteElement<2, 3> &fe = dof_handler.get_fe();
  if (!data.flux_local.empty() || !data.trace_local.empty())
    {
      Basis              v0, v1;
      const unsigned int n1 = shape_nodes.size();
      if (n1 > 0)
        {
          horner_basis(shape_coefficients, n1, reference[0], &v0, nullptr,
                       nullptr);
          horner_basis(shape_coefficients, n1, reference[1], &v1, nullptr,
                       nullptr);
        }
      double flux = 0.0, trace = 0.0;
      for (unsigned int k = 0; k < fe.dofs_per_cell; ++k)
        {
          const double shape =
            n1 > 0 ? v0[shape_tensor_index[k][0]] * v1[shape_tensor_index[k][1]] :
                     fe.shape_value(k, reference);
          if (!data.flux_local.empty())
            flux += data.flux_local[k] * shape;
          if (!data.trace_local.empty())
            trace += data.trace_local[k] * shape;
        }
      s.flux  = flux;
      s.trace = trace;
    }
  if (densities.flux_gradient != nullptr)
    {
      densities.flux_gradient->vector_value(s.point, gradient_buffer);
      s.flux = 0.0;
      for (unsigned int d = 0; d < 3; ++d)
        s.flux += gradient_buffer[d] * s.normal[d];
    }
  if (densities.trace_function != nullptr)
    s.trace = densities.trace_function->value(s.point);
  if (wants_curl)
    {
      densities.trace_gradient->vector_value(s.point, gradient_buffer);
      Tensor<1, 3> grad;
      for (unsigned int d = 0; d < 3; ++d)
        grad[d] = gradient_buffer[d];
      s.curl_density = cross_product_3d(s.normal, grad);
    }
  return s;
}


void
SurfacePotentialEvaluator::add(const Point<3>     &x,
                               const Tensor<1, 3> &normal,
                               const double        trace_at_x,
                               const Source       &y,
                               Values             &values) const
{
  const Tensor<1, 3> r        = y.point - x;
  const double       distance = r.norm();
  if (distance <= 0.0)
    return;
  const double decay =
    screening == 0.0 ? 1.0 : std::exp(-screening * distance);
  const double E  = screening == 0.0 ? 1.0 : (1.0 + screening * distance) * decay;
  const double g0 = y.weight / (4.0 * numbers::PI * distance);
  const double g  = g0 * decay;
  const double g3 = g0 / (distance * distance);
  values.single_layer += g * y.flux;
  // dG0/dn_y = -(y - x).n_y / (4 pi r^3), the Laplace double layer, against
  // E phi(y) - phi(x): the regularised (1/2 I + K) phi of either kernel.
  values.double_layer += -(r * y.normal) * g3 * (E * y.trace - trace_at_x);
  // dG/dn_x = E (y - x).n_x / (4 pi r^3), the kernel of the adjoint.
  values.adjoint_double_layer += (r * normal) * g3 * E * y.flux;
  if (wants_curl)
    {
      values.curl_single_layer += g * y.curl_density;
      if (screening != 0.0)
        values.normal_single_layer += (g * y.trace) * y.normal;
    }
}


std::pair<Point<2>, double>
SurfacePotentialEvaluator::closest_point(const CellData &data,
                                         const Point<3> &x) const
{
  Point<2> best(0.5, 0.5);
  double   best_distance = std::numeric_limits<double>::max();
  for (unsigned int j = 0; j <= 4; ++j)
    for (unsigned int i = 0; i <= 4; ++i)
      {
        const Point<2> xi(0.25 * i, 0.25 * j);
        const double   d = (geometry(data, xi, false).point - x).norm();
        if (d < best_distance)
          {
            best_distance = d;
            best          = xi;
          }
      }

  Point<2> xi = best;
  for (unsigned int iteration = 0; iteration < 60; ++iteration)
    {
      const Geometry     g  = geometry(data, xi, false);
      const Tensor<1, 3> r  = g.point - x;
      const double       a  = g.tangents[0] * g.tangents[0];
      const double       b  = g.tangents[0] * g.tangents[1];
      const double       c  = g.tangents[1] * g.tangents[1];
      const double       r0 = g.tangents[0] * r;
      const double       r1 = g.tangents[1] * r;
      const double       det = a * c - b * b;
      Point<2>           next(xi[0] - (c * r0 - b * r1) / det,
                    xi[1] - (a * r1 - b * r0) / det);
      next[0] = std::min(1.0, std::max(0.0, next[0]));
      next[1] = std::min(1.0, std::max(0.0, next[1]));
      const double step = (next - xi).norm();
      xi                = next;
      if (step < 1e-15)
        break;
    }
  return {xi, (geometry(data, xi, false).point - x).norm()};
}


void
SurfacePotentialEvaluator::integrate_near(const CellData     &data,
                                          const Point<2>     &apex,
                                          const double        distance,
                                          const Point<3>     &x,
                                          const Tensor<1, 3> &normal,
                                          const double        trace_at_x,
                                          Values             &values) const
{
  // The square is cut into the triangles joining the closest point P to each
  // side P does not lie on. On a triangle (P, A, B) the collapsed map
  //     xi = P + s ((A - P) + t (B - A))
  // has Jacobian s h |B - A|, with h the distance of P from the side, and the
  // factor s cancels the 1/r singularity at the apex.
  //
  // Two near singularities remain, and each is resolved by a sinh map
  // (Johnston and Elliott), which spreads the Gauss points over the scale of
  // the singularity instead of subdividing:
  //  - in t, when P is close to the side: the distance from P to the point
  //    of the side, L(t), has a sharp minimum of width H / |B - A| at the
  //    foot of the perpendicular, H being the physical height;
  //  - in s, when x is off the panel at distance d: along each ray the
  //    kernel behaves like 1/sqrt(s^2 L(t)^2 + d^2), on the scale
  //    d / L(t).
  // Lengths are measured with the Jacobian at P, so that anisotropic cells
  // are treated in their physical metric.
  const double   tolerance = 1e-12;
  const Geometry at_apex   = geometry(data, apex);
  const auto     physical  = [&](const Tensor<1, 2> &v) {
    return v[0] * at_apex.tangents[0] + v[1] * at_apex.tangents[1];
  };

  static const std::array<Point<2>, 4> corners{
    {Point<2>(0, 0), Point<2>(1, 0), Point<2>(1, 1), Point<2>(0, 1)}};
  const std::array<double, 4> heights{
    {apex[1], 1.0 - apex[0], 1.0 - apex[1], apex[0]}};
  const QGauss<1> &rule = gauss_rule(duffy_order);

  for (unsigned int side = 0; side < 4; ++side)
    {
      if (heights[side] < tolerance)
        continue;
      const Point<2>     &A      = corners[side];
      const Point<2>     &B      = corners[(side + 1) % 4];
      const Tensor<1, 2>  base   = B - A;
      const Tensor<1, 2>  leg    = A - apex;
      const double        factor = heights[side] * base.norm();

      // Foot of the perpendicular in the physical metric, and the width of
      // the minimum of L(t) around it.
      const Tensor<1, 3> base_x = physical(base);
      const Tensor<1, 3> leg_x  = physical(leg);
      const double       base_length = base_x.norm();
      const double       foot = std::min(
        1.0, std::max(0.0, -(leg_x * base_x) / (base_length * base_length)));
      const double width = (leg_x + foot * base_x).norm() / base_length;
      const bool   stretch_t = width < 0.5;
      const double u0 = stretch_t ? std::asinh((0.0 - foot) / width) : 0.0;
      const double u1 = stretch_t ? std::asinh((1.0 - foot) / width) : 1.0;

      for (unsigned int j = 0; j < rule.size(); ++j)
        {
          double t = rule.point(j)[0], dt = 1.0;
          if (stretch_t)
            {
              const double u = u0 + rule.point(j)[0] * (u1 - u0);
              t              = foot + width * std::sinh(u);
              dt             = width * std::cosh(u) * (u1 - u0);
            }
          const Tensor<1, 2> ray        = leg + t * base;
          const double       ray_length = physical(ray).norm();
          const double       delta      = distance / ray_length;
          const bool         stretch_s  = delta > tolerance && delta < 1.0;
          const double       span = stretch_s ? std::asinh(1.0 / delta) : 1.0;
          for (unsigned int i = 0; i < rule.size(); ++i)
            {
              const double tau = rule.point(i)[0];
              double       s = tau, ds = 1.0;
              if (stretch_s)
                {
                  s  = delta * std::sinh(tau * span);
                  ds = delta * span * std::cosh(tau * span);
                }
              const Point<2> xi = apex + s * ray;
              add(x,
                  normal,
                  trace_at_x,
                  source(data,
                         xi,
                         rule.weight(i) * rule.weight(j) * s * ds * dt *
                           factor),
                  values);
            }
        }
    }
}


void
SurfacePotentialEvaluator::integrate_patch(const CellData     &data,
                                           const Point<2>     &lower,
                                           const Point<2>     &upper,
                                           const unsigned int  depth,
                                           const Point<3>     &x,
                                           const Tensor<1, 3> &normal,
                                           const double        trace_at_x,
                                           Values &values) const
{
  const Point<2> middle((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2);
  const Point<3> centre = geometry(data, middle).point;
  double         radius = 0.0;
  for (unsigned int j = 0; j <= 2; ++j)
    for (unsigned int i = 0; i <= 2; ++i)
      {
        const Point<2> xi(lower[0] + 0.5 * i * (upper[0] - lower[0]),
                          lower[1] + 0.5 * j * (upper[1] - lower[1]));
        radius = std::max(radius, (geometry(data, xi).point - centre).norm());
      }
  radius *= 1.05;
  const double diameter = 2.0 * radius;
  const double gap      = (x - centre).norm() - radius;

  // A leaf at one diameter of separation, with eight points per direction,
  // is accurate to about 1e-10 (Bernstein ellipse of parameter 2 + sqrt 5).
  if (gap >= diameter || depth >= maximum_depth)
    {
      ++n_leaves;
      const QGauss<1> &rule = gauss_rule(depth >= maximum_depth ? 12 : 8);
      const double     area =
        (upper[0] - lower[0]) * (upper[1] - lower[1]);
      for (unsigned int j = 0; j < rule.size(); ++j)
        for (unsigned int i = 0; i < rule.size(); ++i)
          {
            const Point<2> xi(lower[0] +
                                rule.point(i)[0] * (upper[0] - lower[0]),
                              lower[1] +
                                rule.point(j)[0] * (upper[1] - lower[1]));
            add(x,
                normal,
                trace_at_x,
                source(data, xi, rule.weight(i) * rule.weight(j) * area),
                values);
          }
      return;
    }

  // Split across the physically longer direction only when the patch is
  // elongated, so that the pieces tend to be square in space and the
  // separation criterion, which sees the longest side, is not met by
  // needlessly thin leaves.
  const double extent0 =
    (geometry(data, Point<2>(upper[0], middle[1])).point -
     geometry(data, Point<2>(lower[0], middle[1])).point)
      .norm();
  const double extent1 =
    (geometry(data, Point<2>(middle[0], upper[1])).point -
     geometry(data, Point<2>(middle[0], lower[1])).point)
      .norm();
  if (extent0 > 2.0 * extent1)
    {
      integrate_patch(data, lower, Point<2>(middle[0], upper[1]), depth + 1,
                      x, normal, trace_at_x, values);
      integrate_patch(data, Point<2>(middle[0], lower[1]), upper, depth + 1,
                      x, normal, trace_at_x, values);
      return;
    }
  if (extent1 > 2.0 * extent0)
    {
      integrate_patch(data, lower, Point<2>(upper[0], middle[1]), depth + 1,
                      x, normal, trace_at_x, values);
      integrate_patch(data, Point<2>(lower[0], middle[1]), upper, depth + 1,
                      x, normal, trace_at_x, values);
      return;
    }
  const std::array<Point<2>, 4> lowers{
    {lower,
     Point<2>(middle[0], lower[1]),
     Point<2>(lower[0], middle[1]),
     middle}};
  const std::array<Point<2>, 4> uppers{
    {middle,
     Point<2>(upper[0], middle[1]),
     Point<2>(middle[0], upper[1]),
     upper}};
  for (unsigned int k = 0; k < 4; ++k)
    integrate_patch(
      data, lowers[k], uppers[k], depth + 1, x, normal, trace_at_x, values);
}


SurfacePotentialEvaluator::Values
SurfacePotentialEvaluator::evaluate(const Point<3>     &x,
                                    const Tensor<1, 3> &normal,
                                    const double        trace_at_x) const
{
  Values values;
  for (const CellData &data : cells)
    {
      const double gap = (x - data.centre).norm() - data.radius;
      if (gap >= near_ratio * data.diameter)
        {
          // Gauss convergence against 1/r at a separation of g diameters
          // goes like rho^(-2n), rho = 2g + sqrt(4g^2 + 1): about 1e-10 with
          // three points from ten diameters, four from four, six from two,
          // eight from one and twelve from half a diameter.
          const unsigned int rule = gap >= 10.0 * data.diameter ? 0 :
                                    gap >= 4.0 * data.diameter  ? 1 :
                                    gap >= 2.0 * data.diameter  ? 2 :
                                    gap >= 1.0 * data.diameter  ? 3 :
                                                                  4;
          ++n_far;
          for (const Source &s : data.far_rules[rule])
            add(x, normal, trace_at_x, s, values);
          continue;
        }

      auto [apex, distance] = closest_point(data, x);
      if (distance <= 1e-9 * data.diameter)
        distance = 0.0;
      ++n_singular;
      integrate_near(data, apex, distance, x, normal, trace_at_x, values);
    }
  return values;
}
