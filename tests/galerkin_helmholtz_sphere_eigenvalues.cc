// The single layer, the double layer and the weak hypersingular operator of
// the Helmholtz kernel exp(i k r) / (4 pi r) on the unit sphere, k = 2, from
// their real and imaginary parts (GalerkinPanelPairIntegrator::
// set_helmholtz()): the kernel is the screened one with kappa = -i k, so
// that the eigenvalues on the spherical harmonics of degree n are those of
// galerkin_screened_sphere_eigenvalues at complex kappa,
//   V Y_n = kappa i_n(kappa) k_n(kappa) Y_n,
//   K Y_n = kappa^2 / 2 (i_n(kappa) k_n'(kappa) + i_n'(kappa) k_n(kappa)) Y_n,
//   W Y_n = -kappa^3 i_n'(kappa) k_n'(kappa) Y_n,
// with i_n and k_n of complex argument in closed form, k_0(z) = e^{-z} / z.
// The Rayleigh quotients of the interpolants of z (n = 1) and xy (n = 2)
// with each part give the real and the imaginary part of the eigenvalue.
// W has the term -k^2 (n_x . n_y) psi(x) phi(y) G. Also: the light and the
// direct path give the same matrices, and assemble_weak_hypersingular() the
// W of the joint assembly, for both parts.

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <complex>
#include <memory>

#include "galerkin_panel_pair_integrator.h"
#include "tests.h"

namespace
{
  using complex = std::complex<double>;

  // i_n by its recurrence from i_0 = sinh z / z and i_1 = cosh z / z -
  // sinh z / z^2, stable for these n and |z|.
  complex
  bessel_i(const unsigned int n, const complex z)
  {
    complex previous = std::sinh(z) / z,
            current  = std::cosh(z) / z - std::sinh(z) / (z * z);
    if (n == 0)
      return previous;
    for (unsigned int m = 1; m < n; ++m)
      {
        const complex next = previous - (2. * m + 1.) / z * current;
        previous           = current;
        current            = next;
      }
    return current;
  }

  // k_n(z) = e^{-z} / z sum_{m <= n} (n + m)! / (m! (n - m)!) (2 z)^{-m}.
  complex
  bessel_k(const unsigned int n, const complex z)
  {
    complex sum = 0., power = 1.;
    for (unsigned int m = 0; m <= n; ++m, power /= 2. * z)
      sum += std::tgamma(n + m + 1.) /
             (std::tgamma(m + 1.) * std::tgamma(n - m + 1.)) * power;
    return std::exp(-z) / z * sum;
  }

  std::array<complex, 3>
  eigenvalues(const unsigned int n, const double k)
  {
    const complex kappa(0., -k);
    const complex i = bessel_i(n, kappa), b = bessel_k(n, kappa),
                  di = bessel_i(n - 1, kappa) - (n + 1.) / kappa * i,
                  db = -bessel_k(n - 1, kappa) - (n + 1.) / kappa * b;
    return {{kappa * i * b,
             kappa * kappa / 2. * (i * db + di * b),
             -kappa * kappa * kappa * di * db}};
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

  const double k               = 2.;
  const auto   make_integrator = [&](const bool imaginary, const bool light) {
    auto integrator = std::make_unique<GalerkinPanelPairIntegrator<3>>(
      mapping, fe, GalerkinSingularQuadrature::sauter_schwab, 7, 4, 0.5, 3, 6);
    integrator->set_helmholtz(k, imaginary);
    AssertThrow(integrator->set_light_geometry(light) == light,
                ExcInternalError());
    return integrator;
  };
  const auto assemble = [&](const bool imaginary, const bool light) {
    const auto integrator = make_integrator(imaginary, light);
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

  const std::array<Operators, 2> parts = {
    {assemble(false, true), assemble(true, true)}};
  for (unsigned int degree = 1; degree <= 2; ++degree)
    {
      const Vector<double> &y        = interpolant[degree - 1];
      const double          norm     = mass.matrix_norm_square(y);
      const auto            quotient = [&](const Operators &o) {
        return std::array<double, 3>{
          {o.single_layer.matrix_norm_square(y) / norm,
           o.double_layer.matrix_norm_square(y) / norm,
           o.hypersingular.matrix_norm_square(y) / norm}};
      };
      const std::array<double, 3> real     = quotient(parts[0]),
                                  imag     = quotient(parts[1]);
      const std::array<complex, 3> exact   = eigenvalues(degree, k);
      const char                  *name[3] = {"V", "K", "W"};
      for (unsigned int o = 0; o < 3; ++o)
        {
          const complex computed(real[o], imag[o]);
          deallog << "k " << k << ", n = " << degree << ": " << name[o] << " "
                  << computed.real() << " + " << computed.imag() << " i, exact "
                  << exact[o].real() << " + " << exact[o].imag() << " i"
                  << std::endl;
          AssertThrow(std::abs(computed - exact[o]) < 2e-3 * std::abs(exact[o]),
                      ExcMessage("The Rayleigh quotient misses the "
                                 "eigenvalue."));
        }
    }

  // The direct path, and W alone, for both parts.
  for (const bool imaginary : {false, true})
    {
      const Operators &light        = parts[imaginary];
      const Operators  direct       = assemble(imaginary, false);
      const double     light_direct = std::max(
        {relative_difference(light.single_layer, direct.single_layer),
             relative_difference(light.double_layer, direct.double_layer),
             relative_difference(light.hypersingular, direct.hypersingular)});
      SparseMatrix<double> hypersingular(pattern);
      make_integrator(imaginary, true)
        ->assemble_weak_hypersingular(dof_handler,
                                      constraints,
                                      hypersingular,
                                      6);
      const double alone =
        relative_difference(light.hypersingular, hypersingular);
      const char *part = imaginary ? "imaginary" : "real";
      deallog << part << " part: light and direct path agree "
              << (light_direct < 1e-10 ? "to 1e-10" : "NOT") << std::endl;
      deallog << part << " part: W alone agrees "
              << (alone < 1e-10 ? "to 1e-10" : "NOT") << std::endl;
    }
}
