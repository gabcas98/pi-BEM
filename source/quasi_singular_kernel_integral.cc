#include "../include/quasi_singular_kernel_integral.h"

#include <algorithm>
#include <cmath>
#include <limits>


template <>
QuasiSingularKernelIntegral<3>::QuasiSingularKernelIntegral(
  const DoFHandler<2, 3>::active_cell_iterator &in_cell,
  const FiniteElement<2, 3>                    &in_fe,
  const Mapping<2, 3>                          &in_mapping,
  const Point<3>                               &in_external_point)
  : cell(in_cell)
  , fe(in_fe)
  , mapping(in_mapping)
{
  eta = find_closest_reference_cell_point(in_external_point);
}


template <>
QuasiSingularKernelIntegral<2>::QuasiSingularKernelIntegral(
  const DoFHandler<1, 2>::active_cell_iterator &in_cell,
  const FiniteElement<1, 2>                    &in_fe,
  const Mapping<1, 2>                          &in_mapping,
  const Point<2> & /*in_external_point*/)
  : cell(in_cell)
  , fe(in_fe)
  , mapping(in_mapping)
{}


template <int dim>
Point<dim - 1>
QuasiSingularKernelIntegral<dim>::find_closest_reference_cell_point(
  const Point<dim> &external_point)
{
  AssertThrow(dim == 3, ExcNotImplemented());

  // A coarse search supplies a reliable basin for curved or distorted cells.
  Point<2> eta_sol;
  double   best_distance_square = std::numeric_limits<double>::infinity();
  constexpr unsigned int samples_per_direction = 3;
  for (unsigned int i = 0; i < samples_per_direction; ++i)
    for (unsigned int j = 0; j < samples_per_direction; ++j)
      {
        const Point<2> candidate(static_cast<double>(i) /
                                   (samples_per_direction - 1),
                                 static_cast<double>(j) /
                                   (samples_per_direction - 1));
        const double   distance_square = external_point.distance_square(
          mapping.transform_unit_to_real_cell(cell, candidate));
        if (distance_square < best_distance_square)
          {
            best_distance_square = distance_square;
            eta_sol              = candidate;
          }
      }

  // Projection makes edges and vertices part of the same minimization.
  for (unsigned int iteration = 0; iteration < 30; ++iteration)
    {
      const Quadrature<2> point_quadrature(std::vector<Point<2>>{eta_sol});
      FEValues<2, 3>      point_values(mapping,
                                  fe,
                                  point_quadrature,
                                  update_quadrature_points | update_jacobians);
      point_values.reinit(cell);

      const Point<3>     real_point = point_values.quadrature_point(0);
      const auto         jacobian   = point_values.jacobian(0);
      const Tensor<1, 3> residual   = external_point - real_point;

      double normal_matrix[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
      double right_hand_side[2]  = {0.0, 0.0};
      for (unsigned int a = 0; a < 2; ++a)
        {
          for (unsigned int b = 0; b < 2; ++b)
            for (unsigned int d = 0; d < 3; ++d)
              normal_matrix[a][b] += jacobian[d][a] * jacobian[d][b];
          for (unsigned int d = 0; d < 3; ++d)
            right_hand_side[a] += jacobian[d][a] * residual[d];
        }

      const double determinant = normal_matrix[0][0] * normal_matrix[1][1] -
                                 normal_matrix[0][1] * normal_matrix[1][0];
      const double scale = normal_matrix[0][0] + normal_matrix[1][1];
      if (!std::isfinite(determinant) ||
          std::abs(determinant) <= 1e-14 * scale * scale)
        break;

      const double step[2] = {(normal_matrix[1][1] * right_hand_side[0] -
                               normal_matrix[0][1] * right_hand_side[1]) /
                                determinant,
                              (-normal_matrix[1][0] * right_hand_side[0] +
                               normal_matrix[0][0] * right_hand_side[1]) /
                                determinant};

      bool     accepted             = false;
      double   damping              = 1.0;
      Point<2> next_eta             = eta_sol;
      double   next_distance_square = best_distance_square;
      for (unsigned int line_search = 0; line_search < 16; ++line_search)
        {
          for (unsigned int d = 0; d < 2; ++d)
            next_eta[d] =
              std::max(0.0, std::min(1.0, eta_sol[d] + damping * step[d]));
          next_distance_square = external_point.distance_square(
            mapping.transform_unit_to_real_cell(cell, next_eta));
          if (next_distance_square < best_distance_square)
            {
              accepted = true;
              break;
            }
          damping *= 0.5;
        }

      if (!accepted)
        break;

      const double parameter_step = eta_sol.distance(next_eta);
      eta_sol                     = next_eta;
      best_distance_square        = next_distance_square;
      if (parameter_step < 1e-12)
        break;
    }

  this->min_distance = std::sqrt(best_distance_square);
  return eta_sol;
}
