#include "../include/surface_potential_evaluator.h"

#include "../include/cellwise_function.h"
#include "../include/tensor_map_kernel.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace
{
  constexpr unsigned int maximum_nodes = 12;
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

  // The same basis for fast evaluation: the nodes x_k and the weights
  // w_i = 1 / prod_{k != i} (x_i - x_k), with L_i(t) = w_i P_i(t) S_i(t),
  // P_i and S_i the products of (t - x_k) over k < i and k > i. Built up by
  // prefix and suffix products with their derivatives (fast_basis), the
  // basis and its first two derivatives cost O(n) per function, and are
  // accurate to rounding at any degree, where the monomial form on [0, 1]
  // loses a digit per node; the product form above costs a cubic loop.
  std::vector<double>
  lagrange_data(const std::vector<double> &nodes)
  {
    const unsigned int  n = nodes.size();
    std::vector<double> result(2 * n);
    for (unsigned int i = 0; i < n; ++i)
      {
        double product = 1.0;
        for (unsigned int k = 0; k < n; ++k)
          if (k != i)
            product *= nodes[i] - nodes[k];
        result[i]     = nodes[i];
        result[n + i] = 1.0 / product;
      }
    return result;
  }

  // Values and first and second derivatives of the basis at t, from the
  // prefix and suffix products and their derivatives.
  void
  fast_basis(const std::vector<double> &data,
               const unsigned int         n,
               const double               t,
               Basis                     *value,
               Basis                     *first,
               Basis                     *second)
  {
    const double *x = data.data(), *w = data.data() + n;
    Basis         P, dP, d2P, S, dS, d2S;
    P[0] = 1.0, dP[0] = 0.0, d2P[0] = 0.0;
    S[n - 1] = 1.0, dS[n - 1] = 0.0, d2S[n - 1] = 0.0;
    for (unsigned int k = 0; k + 1 < n; ++k)
      {
        const double f = t - x[k];
        d2P[k + 1]     = d2P[k] * f + 2.0 * dP[k];
        dP[k + 1]      = dP[k] * f + P[k];
        P[k + 1]       = P[k] * f;
      }
    for (unsigned int k = n - 1; k > 0; --k)
      {
        const double f = t - x[k];
        d2S[k - 1]     = d2S[k] * f + 2.0 * dS[k];
        dS[k - 1]      = dS[k] * f + S[k];
        S[k - 1]       = S[k] * f;
      }
    for (unsigned int i = 0; i < n; ++i)
      {
        if (value != nullptr)
          (*value)[i] = w[i] * P[i] * S[i];
        if (first != nullptr)
          (*first)[i] = w[i] * (dP[i] * S[i] + P[i] * dS[i]);
        if (second != nullptr)
          (*second)[i] =
            w[i] * (d2P[i] * S[i] + 2.0 * dP[i] * dS[i] + P[i] * d2S[i]);
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
  lagrange_coefficients = lagrange_data(lagrange_nodes);

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
          shape_coefficients = lagrange_data(nodes);
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

  projected_mapping =
    dynamic_cast<const ExactGeometry::ProjectedMapping<2, 3> *>(&mapping);
  if (projected_mapping != nullptr)
    exact_evaluator =
      std::make_unique<ExactGeometry::ProjectedMapping<2, 3>::Evaluator>(
        *projected_mapping);

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
      const Mapping<2, 3> &nodal_mapping =
        projected_mapping != nullptr ?
          static_cast<const Mapping<2, 3> &>(
            projected_mapping->get_mesh_mapping()) :
          mapping;
      for (unsigned int j = 0; j < n; ++j)
        for (unsigned int i = 0; i < n; ++i)
          data.nodes[i + n * j] = nodal_mapping.transform_unit_to_real_cell(
            cell, Point<2>(lagrange_nodes[i], lagrange_nodes[j]));
      data.soa_nodes.resize(3 * n * n);
      for (unsigned int k = 0; k < n * n; ++k)
        for (unsigned int c = 0; c < 3; ++c)
          data.soa_nodes[c * n * n + k] = data.nodes[k][c];

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
                  on_cell(*densities.flux_gradient, cell)
                    .vector_value(s.point, gradient_buffer);
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
                s.trace =
                  on_cell(*densities.trace_function, cell).value(s.point);
              if (wants_curl)
                {
                  on_cell(*densities.trace_gradient, cell)
                    .vector_value(s.point, gradient_buffer);
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
SurfacePotentialEvaluator::geometry_of_nodes(const CellData &data,
                                             const Point<2> &reference) const
{
  const unsigned int n = lagrange_nodes.size();
  Basis              v0, d0, v1, d1;
  fast_basis(lagrange_coefficients, n, reference[0], &v0, &d0, nullptr);
  fast_basis(lagrange_coefficients, n, reference[1], &v1, &d1, nullptr);
  Tensor<1, 3> x, x1, x2;
  for (unsigned int j = 0; j < n; ++j)
    for (unsigned int i = 0; i < n; ++i)
      {
        const Tensor<1, 3> &node = data.nodes[i + n * j];
        x += v0[i] * v1[j] * node;
        x1 += d0[i] * v1[j] * node;
        x2 += v0[i] * d1[j] * node;
      }
  Geometry g;
  g.point    = Point<3>(x);
  g.tangents = {{x1, x2}};
  return g;
}


SurfacePotentialEvaluator::Geometry
SurfacePotentialEvaluator::geometry(const CellData &data,
                                    const Point<2> &reference,
                                    const bool      with_curvature) const
{
  const unsigned int n = lagrange_nodes.size();
  Basis              v0, d0, s0, v1, d1, s1;
  fast_basis(lagrange_coefficients, n, reference[0], &v0, &d0,
               with_curvature ? &s0 : nullptr);
  fast_basis(lagrange_coefficients, n, reference[1], &v1, &d1,
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
  if (exact_evaluator)
    {
      // The point of the mesh mapping and its tangents, projected onto the
      // exact geometry; the second derivatives of the exact map are central
      // differences of its first ones.
      const auto project = [&](const Point<2>     &r,
                               const Tensor<1, 3> &y0,
                               const Tensor<1, 3> &y1,
                               const Tensor<1, 3> &y2,
                               Point<3>           &image,
                               Tensor<1, 3>       &t1,
                               Tensor<1, 3>       &t2) {
        DerivativeForm<1, 2, 3> V, D;
        for (unsigned int c = 0; c < 3; ++c)
          {
            V[c][0] = y1[c];
            V[c][1] = y2[c];
          }
        exact_evaluator->project(r, Point<3>(y0), V, image, D);
        for (unsigned int c = 0; c < 3; ++c)
          {
            t1[c] = D[c][0];
            t2[c] = D[c][1];
          }
      };
      exact_evaluator->reinit(data.cell);
      project(reference, x, x1, x2, g.point, x1, x2);
      if (with_curvature)
        {
          const double h = 1e-4;
          std::array<std::array<Tensor<1, 3>, 2>, 2> tangent_plus,
            tangent_minus;
          for (unsigned int b = 0; b < 2; ++b)
            for (const int sign : {1, -1})
              {
                Point<2> r = reference;
                r[b] += sign * h;
                const Geometry mesh = geometry_of_nodes(data, r);
                Point<3>       image;
                auto &t = (sign > 0) ? tangent_plus[b] : tangent_minus[b];
                project(r,
                        mesh.point,
                        mesh.tangents[0],
                        mesh.tangents[1],
                        image,
                        t[0],
                        t[1]);
              }
          x11 = (tangent_plus[0][0] - tangent_minus[0][0]) / (2 * h);
          x22 = (tangent_plus[1][1] - tangent_minus[1][1]) / (2 * h);
          x12 = 0.5 * ((tangent_plus[1][0] - tangent_minus[1][0]) +
                       (tangent_plus[0][1] - tangent_minus[0][1])) /
                (2 * h);
        }
    }
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
  complete_source(data, reference, s);
  return s;
}


void
SurfacePotentialEvaluator::sources(const CellData              &data,
                                   const std::vector<Point<2>> &references,
                                   const std::vector<double>   &weights,
                                   std::vector<Source>         &out) const
{
  const unsigned int m = references.size();
  out.resize(m);
  if (exact_evaluator)
    {
      for (unsigned int q = 0; q < m; ++q)
        out[q] = source(data, references[q], weights[q]);
      return;
    }
  n_sources += m;
  const double orientation = data.orientation;
  TensorMapKernel::evaluate(
    lagrange_nodes.size(),
    data.soa_nodes.data(),
    lagrange_coefficients.data(),
    references.data(),
    m,
    [&](const unsigned int begin,
        const unsigned int count,
        const double (&s)[9][4]) {
      for (unsigned int q = 0; q < count; ++q)
        {
          Source &y = out[begin + q];
          for (unsigned int c = 0; c < 3; ++c)
            y.point[c] = s[c][q];
          const double nx = s[4][q] * s[8][q] - s[5][q] * s[7][q];
          const double ny = s[5][q] * s[6][q] - s[3][q] * s[8][q];
          const double nz = s[3][q] * s[7][q] - s[4][q] * s[6][q];
          const double j  = std::sqrt(nx * nx + ny * ny + nz * nz);
          y.normal[0]     = orientation * nx / j;
          y.normal[1]     = orientation * ny / j;
          y.normal[2]     = orientation * nz / j;
          y.weight        = weights[begin + q] * j;
        }
    });
  for (unsigned int q = 0; q < m; ++q)
    complete_source(data, references[q], out[q]);
}


void
SurfacePotentialEvaluator::complete_source(const CellData &data,
                                           const Point<2> &reference,
                                           Source         &s) const
{
  const FiniteElement<2, 3> &fe = dof_handler.get_fe();
  if (!data.flux_local.empty() || !data.trace_local.empty())
    {
      Basis              v0, v1;
      const unsigned int n1 = shape_nodes.size();
      if (n1 > 0)
        {
          fast_basis(shape_coefficients, n1, reference[0], &v0, nullptr,
                       nullptr);
          fast_basis(shape_coefficients, n1, reference[1], &v1, nullptr,
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
      on_cell(*densities.flux_gradient, data.cell)
        .vector_value(s.point, gradient_buffer);
      s.flux = 0.0;
      for (unsigned int d = 0; d < 3; ++d)
        s.flux += gradient_buffer[d] * s.normal[d];
    }
  if (densities.trace_function != nullptr)
    s.trace = on_cell(*densities.trace_function, data.cell).value(s.point);
  if (wants_curl)
    {
      on_cell(*densities.trace_gradient, data.cell)
        .vector_value(s.point, gradient_buffer);
      Tensor<1, 3> grad;
      for (unsigned int d = 0; d < 3; ++d)
        grad[d] = gradient_buffer[d];
      s.curl_density = cross_product_3d(s.normal, grad);
    }
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

      batch_points.clear();
      batch_weights.clear();
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
              batch_points.push_back(apex + s * ray);
              batch_weights.push_back(rule.weight(i) * rule.weight(j) * s *
                                      ds * dt * factor);
            }
        }
      sources(data, batch_points, batch_weights, batch_sources);
      for (const Source &y : batch_sources)
        add(x, normal, trace_at_x, y, values);
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
      batch_points.clear();
      batch_weights.clear();
      for (unsigned int j = 0; j < rule.size(); ++j)
        for (unsigned int i = 0; i < rule.size(); ++i)
          {
            batch_points.emplace_back(
              lower[0] + rule.point(i)[0] * (upper[0] - lower[0]),
              lower[1] + rule.point(j)[0] * (upper[1] - lower[1]));
            batch_weights.push_back(rule.weight(i) * rule.weight(j) * area);
          }
      sources(data, batch_points, batch_weights, batch_sources);
      for (const Source &y : batch_sources)
        add(x, normal, trace_at_x, y, values);
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
