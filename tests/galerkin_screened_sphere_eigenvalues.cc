// The single layer, the double layer and the weak hypersingular operator of
// the Laplace and of the screened kernel exp(-kappa r) / (4 pi r) on the
// unit sphere, where the spherical harmonics of degree n are their
// eigenfunctions:
//   V Y_n = kappa i_n(kappa) k_n(kappa) Y_n,
//   K Y_n = kappa^2 / 2 (i_n(kappa) k_n'(kappa) + i_n'(kappa) k_n(kappa)) Y_n,
//   W Y_n = -kappa^3 i_n'(kappa) k_n'(kappa) Y_n,
// with the modified spherical Bessel functions i_n and k_n, k_0(z) =
// e^{-z} / z; for kappa = 0, 1 / (2n + 1), -1 / (2 (2n + 1)) and
// n (n + 1) / (2n + 1). The Rayleigh quotients of the interpolants of z
// (n = 1) and xy (n = 2) err by the square of the interpolation error. The
// screened W has, besides the product of the surface curls, the term
// kappa^2 (n_x . n_y) psi(x) phi(y) G. Also: the light and the direct path
// of the integrator give the same matrices, and assemble_weak_hypersingular()
// the W of the joint assembly.

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <memory>

#include "galerkin_panel_pair_integrator.h"
#include "tests.h"

namespace
{
  // i_n and k_n, and their derivatives from i_n' = i_{n-1} - (n+1)/z i_n
  // and k_n' = -k_{n-1} - (n+1)/z k_n.
  double
  bessel_i(const unsigned int n, const double z)
  {
    return std::sqrt(numbers::PI / (2. * z)) * std::cyl_bessel_i(n + 0.5, z);
  }

  double
  bessel_k(const unsigned int n, const double z)
  {
    return std::sqrt(2. / (numbers::PI * z)) * std::cyl_bessel_k(n + 0.5, z);
  }

  double
  bessel_i_derivative(const unsigned int n, const double z)
  {
    return bessel_i(n - 1, z) - (n + 1.) / z * bessel_i(n, z);
  }

  double
  bessel_k_derivative(const unsigned int n, const double z)
  {
    return -bessel_k(n - 1, z) - (n + 1.) / z * bessel_k(n, z);
  }

  std::array<double, 3>
  eigenvalues(const unsigned int n, const double kappa)
  {
    if (kappa == 0.)
      return {{1. / (2. * n + 1.),
               -1. / (2. * (2. * n + 1.)),
               n * (n + 1.) / (2. * n + 1.)}};
    const double i = bessel_i(n, kappa), k = bessel_k(n, kappa),
                 di = bessel_i_derivative(n, kappa),
                 dk = bessel_k_derivative(n, kappa);
    return {{kappa * i * k,
             kappa * kappa / 2. * (i * dk + di * k),
             -kappa * kappa * kappa * di * dk}};
  }

  struct Operators
  {
    SparseMatrix<double> single_layer, double_layer, hypersingular;
  };

  double
  relative_difference(const SparseMatrix<double> &a,
                      const SparseMatrix<double> &b)
  {
    double difference = 0., norm = 0.;
    for (unsigned int i = 0; i < a.m(); ++i)
      for (unsigned int j = 0; j < a.n(); ++j)
        {
          difference += std::pow(a.el(i, j) - b.el(i, j), 2);
          norm += std::pow(a.el(i, j), 2);
        }
    return std::sqrt(difference / norm);
  }
} // namespace


int
main()
{
  initlog();
  deallog << std::setprecision(6);

  Triangulation<2, 3> triangulation;
  GridGenerator::hyper_sphere(triangulation, Point<3>(), 1.);
  triangulation.refine_global(1);
  const FE_Q<2, 3> fe(2);
  DoFHandler<2, 3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const MappingQ<2, 3>      mapping(3);
  AffineConstraints<double> constraints;
  constraints.close();
  const unsigned int n = dof_handler.n_dofs();

  DynamicSparsityPattern dynamic_pattern(n, n);
  for (unsigned int i = 0; i < n; ++i)
    for (unsigned int j = 0; j < n; ++j)
      dynamic_pattern.add(i, j);
  SparsityPattern pattern;
  pattern.copy_from(dynamic_pattern);
  SparseMatrix<double> mass(pattern);

  const auto make_integrator = [&](const double kappa, const bool light) {
    auto integrator = std::make_unique<GalerkinPanelPairIntegrator<3>>(
      mapping, fe, GalerkinSingularQuadrature::sauter_schwab, 7, 4, 0.5, 3, 6);
    integrator->set_screening(kappa);
    AssertThrow(integrator->set_light_geometry(light) == light,
                ExcInternalError());
    return integrator;
  };
  const auto assemble = [&](const double kappa, const bool light) {
    const auto integrator = make_integrator(kappa, light);
    Operators  operators;
    operators.single_layer.reinit(pattern);
    operators.double_layer.reinit(pattern);
    operators.hypersingular.reinit(pattern);
    integrator->assemble_single_double_hypersingular(dof_handler,
                                                     constraints,
                                                     operators.single_layer,
                                                     operators.double_layer,
                                                     operators.hypersingular,
                                                     6);
    if (mass.frobenius_norm() == 0.)
      integrator->assemble_mass_matrix(dof_handler, constraints, mass, 6);
    return operators;
  };

  // z and xy, harmonics of degree 1 and 2.
  const std::array<std::function<double(const Point<3> &)>, 2> harmonic = {
    {[](const Point<3> &p) { return p[2]; },
     [](const Point<3> &p) { return p[0] * p[1]; }}};
  std::array<Vector<double>, 2> interpolant;
  for (unsigned int h = 0; h < 2; ++h)
    {
      interpolant[h].reinit(n);
      VectorTools::interpolate(mapping,
                               dof_handler,
                               ScalarFunctionFromFunctionObject<3>(harmonic[h]),
                               interpolant[h]);
    }

  for (const double kappa : {0., 2.})
    {
      const Operators operators = assemble(kappa, true);
      for (unsigned int degree = 1; degree <= 2; ++degree)
        {
          const Vector<double>       &y        = interpolant[degree - 1];
          const double                norm     = mass.matrix_norm_square(y);
          const std::array<double, 3> computed = {
            {operators.single_layer.matrix_norm_square(y) / norm,
             operators.double_layer.matrix_norm_square(y) / norm,
             operators.hypersingular.matrix_norm_square(y) / norm}};
          const std::array<double, 3> exact   = eigenvalues(degree, kappa);
          const char                 *name[3] = {"V", "K", "W"};
          for (unsigned int o = 0; o < 3; ++o)
            {
              deallog << "kappa " << kappa << ", n = " << degree << ": "
                      << name[o] << " " << computed[o] << ", exact " << exact[o]
                      << std::endl;
              AssertThrow(std::abs(computed[o] - exact[o]) <
                            2e-3 * std::abs(exact[o]),
                          ExcMessage("The Rayleigh quotient misses the "
                                     "eigenvalue."));
            }
        }

      // The direct path, and W alone.
      const Operators direct       = assemble(kappa, false);
      const double    light_direct = std::max(
        {relative_difference(operators.single_layer, direct.single_layer),
            relative_difference(operators.double_layer, direct.double_layer),
            relative_difference(operators.hypersingular, direct.hypersingular)});
      SparseMatrix<double> hypersingular(pattern);
      make_integrator(kappa, true)
        ->assemble_weak_hypersingular(dof_handler,
                                      constraints,
                                      hypersingular,
                                      6);
      const double alone =
        relative_difference(operators.hypersingular, hypersingular);
      deallog << "kappa " << kappa << ": light and direct path agree "
              << (light_direct < 1e-10 ? "to 1e-10" : "NOT") << std::endl;
      deallog << "kappa " << kappa << ": W alone agrees "
              << (alone < 1e-10 ? "to 1e-10" : "NOT") << std::endl;
    }
}
