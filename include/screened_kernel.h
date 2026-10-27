
#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/utilities.h>
// #include <boost/math/special_functions/bessel.hpp> 		// for modified Bessel
// functions of second kind (not needed?)

// And here are a few C++ standard header
// files that we will need:
#include <cmath> // for std::exp and modified Bessel functions of second kind
#include <fstream>
#include <iostream>
#include <set>
#include <string>


namespace ScreenedKernel
{
  template <int dim>
  double
  single_layer(const Point<dim> &R, const double kappa)
  {
    switch (dim)
      {
        case 2:
          // G(x, y) = (1 / 2π) * K0(κ * |R|)
          return (std::cyl_bessel_k(0, kappa * R.norm()) / (2 * numbers::PI));

        case 3:
          // G(x, y) = (1 / 4π) * exp(-κ * |R|) / |R|
          return (std::exp(-kappa * R.norm()) / (R.norm() * 4 * numbers::PI));

        default:
          Assert(false, ExcInternalError());
          return 0.;
      }
  }



  template <int dim>
  Point<dim>
  double_layer(const Point<dim> &R, const double kappa)
  {
    double r = R.norm();
    // The Bessel functions are needed in 2D only, and cost far more than the
    // exponential: they are evaluated in the branch that uses them.
    switch (dim)
      {
        case 2:
          // ∇G = -κ * K1(κ * |R|) * R / (2π * |R|)
          return (-kappa * std::cyl_bessel_k(1, kappa * r) * R /
                  (-2 * numbers::PI * r));
        case 3:
          // ∇G = (κ * |R| + 1) * exp(-κ * |R|) * R / (-4π |R|^3)
          return ((kappa * r + 1.) * std::exp(-kappa * r) * R /
                  (-4 * numbers::PI * R.square() * r));

        default:
          Assert(false, ExcInternalError());
          return Point<dim>();
      }
  }

  template <int dim> // mio//
  Tensor<2, dim>
  hypersingular(const Point<dim> &R, const double kappa)
  {
    double         r = R.norm();
    Tensor<2, dim> Hyper;
    switch (dim)
      {
        case 2:
          {
            const double K0_kr = std::cyl_bessel_k(0, kappa * r);
            const double K1_kr = std::cyl_bessel_k(1, kappa * r);
            Hyper[0][0]        = kappa * kappa * R.square() * K0_kr +
                          kappa * r * K1_kr + kappa * R[0] * R[0] * K1_kr;
            Hyper[1][1] = kappa * kappa * R.square() * K0_kr +
                          kappa * r * K1_kr + kappa * R[1] * R[1] * K1_kr;
            Hyper[0][1] = kappa * K1_kr * R[0] * R[1];
            Hyper[1][0] = kappa * K1_kr * R[0] * R[1];
            Hyper       = Hyper / (2 * numbers::PI * R.square() * R.norm());
            return Hyper;
          }
        case 3:
          {
            for (unsigned int i = 0; i < dim; ++i)
              for (unsigned int j = 0; j < dim; ++j)
                if (i == j)
                  Hyper[j][j] =
                    (kappa * kappa * R.square() + 3 * kappa * r + 3) * R[j] *
                      R[j] -
                    (kappa * r + 1.) * R.square();
                else
                  Hyper[i][j] =
                    (kappa * kappa * R.square() + 3 * kappa * r + 3) * R[i] *
                    R[j];
            return (Hyper * std::exp(-kappa * r) /
                    (4 * numbers::PI * R.square() * R.square() * r));
          }

        default:
          Assert(false, ExcInternalError());
          return Point<dim>();
      }
  }

  template <int dim>
  void
  kernels(const Tensor<1, dim> &R,
          Tensor<1, dim>       &D,
          double               &d,
          const double          kappa)
  {
    double r  = R.norm();
    double r2 = r * r;
    switch (dim)
      {
        case 2:
          d = std::cyl_bessel_k(0, kappa * r) / (2 * numbers::PI);
          D = kappa * std::cyl_bessel_k(1, kappa * r) * R /
              (-2 * numbers::PI * r);
          break;
        case 3:
          {
            const double e = std::exp(-kappa * r);
            d              = e / (4 * numbers::PI * r);
            D = e * (kappa * r + 1.) * R / (-4 * numbers::PI * r2 * r);
            break;
          }

        default:
          Assert(false, ExcInternalError());
      }
  }

  template <int dim> // mio//
  void
  kernels(const Tensor<1, dim> &R,
          Tensor<2, dim>       &H,
          Tensor<1, dim>       &D,
          double               &d,
          const double          kappa)
  {
    double r  = R.norm();
    double r2 = r * r;
    switch (dim)
      {
        case 2:
          {
            const double K0_kr = std::cyl_bessel_k(0, kappa * r);
            const double K1_kr = std::cyl_bessel_k(1, kappa * r);
            d                  = K0_kr / (2 * numbers::PI);
            D                  = kappa * K1_kr * R / (-2 * numbers::PI * r);
            H                  = 0;
            H[0][0] = kappa * kappa * r2 * K0_kr + kappa * r * K1_kr +
                      kappa * R[0] * R[0] * K1_kr;
            H[1][1] = kappa * kappa * r2 * K0_kr + kappa * r * K1_kr +
                      kappa * R[1] * R[1] * K1_kr;
            H[0][1] = kappa * K1_kr * R[0] * R[1];
            H[1][0] = kappa * K1_kr * R[0] * R[1];
            H       = H / (2 * numbers::PI * r2 * r);
            break;
          }
        case 3:
          {
            const double e = std::exp(-kappa * r);
            d              = e / (4 * numbers::PI * r);
            D = e * (kappa * r + 1.) * R / (-4 * numbers::PI * r2 * r);
            H = 0;
            /*	for(unsigned int i=0;i<dim;++i)
                  for(unsigned int j=0;j<dim;++j)
                    if(i==j)
                {
                H[j][j]=-2*(R[j]*R[j]);
                for(unsigned int k=0;k<dim;++k)
                              if(k!=i)
                  H[j][j]+=(R[k]*R[k]);
                }
                    else H[i][j]=-3*R[i]*R[j];
              H=H/( 4*numbers::PI * r2 *r2*r );
            */
            for (unsigned int i = 0; i < dim; ++i)
              for (unsigned int j = 0; j < dim; ++j)
                H[i][j] =
                  (kappa * kappa * r2 + 3 * kappa * r + 3) * R[i] * R[j];
            for (unsigned int i = 0; i < dim; ++i)
              H[i][i] -= (kappa * r + 1.) * r2;
            H = H * e / (4 * numbers::PI * r2 * r2 * r);
            break;
          }

        default:
          Assert(false, ExcInternalError());
      }
  }

} // namespace ScreenedKernel
