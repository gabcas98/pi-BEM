#include "telles_quadrature.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/mapping_q1.h>

#include <cmath>
#include <vector>
#include <utility>

using namespace dealii;


// Table from Telles & Oliveira 1994, for alpha = 0,1,2,3
static double r_bar_from_table(double D, int alpha)
{
  // rows: {D, r_bar_alpha0, r_bar_alpha1, r_bar_alpha2, r_bar_alpha3}
  static const std::vector<std::array<double,5>> table = {
    {{0.01,     0.0,        0.06915407, 0.0,        0.0       }},
    {{0.016667, 0.0,        0.09831843, 0.05747304, 0.0       }},
    {{0.05,     0.44836450, 0.20502630, 0.13773590, 0.11184720}},
    {{0.10,     0.52877950, 0.32425940, 0.22398780, 0.18485610}},
    {{0.20,     0.64464500, 0.48307190, 0.37314880, 0.30286860}},
    {{0.30,     0.72158940, 0.58950260, 0.49163050, 0.41799020}},
    {{0.50,     0.81810350, 0.72706620, 0.65196120, 0.58999270}},
    {{0.70,     0.87475540, 0.80949360, 0.75230580, 0.70242170}},
    {{0.90,     0.91011310, 0.86210700, 0.81812040, 0.77833510}},
    {{1.50,     0.95881710, 0.93744350, 0.91627500, 0.89566070}},
    {{2.50,     0.98214420, 0.97358650, 0.96520330, 0.95694540}},
    {{3.50,     0.98998790, 0.98520440, 0.98055080, 0.97601060}},
    {{4.50,     0.99363380, 0.99056080, 0.98755660, 0.98461700}},
    {{14.0,     0.99928560, 0.99893010, 0.99857580, 0.99822270}}
  };

  if (D > 14.0) return 1.0;
  if (D <= table.front()[0]) return table.front()[1 + alpha];

  // linear interpolation of the table
  for (unsigned int i = 1; i < table.size(); ++i)
  {
    if (D <= table[i][0])
    {
      double t = (D - table[i-1][0]) / (table[i][0] - table[i-1][0]);
      double v0 = table[i-1][1 + alpha];
      double v1 = table[i  ][1 + alpha];
      return v0 + t * (v1 - v0);
    }
  }
    
    
  return 1.0;
}


// compute transformation coefficients a,b,c,d
// and gamma_bar from eta_bar and r_bar 
static void compute_telles_coeffs(
    double eta_bar, double r_bar,
    double &a, double &b, double &c, double &d)
{
  double eta_bar2 = eta_bar * eta_bar;
  double one_p_2r = 1.0 + 2.0 * r_bar;

  double p = (1.0/ (3.0 * one_p_2r * one_p_2r))
           * ( 4.0*r_bar*(1.0 - r_bar) + 3.0*(1.0 - eta_bar2) );
           

  double q = (1.0 / (2.0 * one_p_2r))
           * ( eta_bar*(3.0 - 2.0*r_bar) - 2.0*eta_bar2*eta_bar / one_p_2r - eta_bar );

  double qq_ppp = q*q + p*p*p;
  double gamma_bar = std::cbrt(-q + std::sqrt(qq_ppp)) + std::cbrt(-q - std::sqrt(qq_ppp)) + eta_bar / one_p_2r;

  double Q = 1.0 + 3.0 * gamma_bar * gamma_bar;
  a =  (1.0 - r_bar) / Q;
  b = -3.0*(1.0 - r_bar)*gamma_bar / Q;
  c =  (r_bar + 3.0*gamma_bar*gamma_bar) / Q;
  d = -b;
}


// main
Quadrature<2> telles_quadrature(
    const typename DoFHandler<2,3>::active_cell_iterator &cell,
    const Mapping<2,3> &mapping,
    const Point<3> &singularity,
    const Point<2> &ref_qsing,
    const unsigned int quadrature_order,
    const int alpha)
{
  Point<3> closest_real = mapping.transform_unit_to_real_cell(cell, ref_qsing);
  double R_min = (singularity - closest_real).norm();

  // convert ref_closest to [-1,1]
  double eta1_bar = 2.0 * ref_qsing[0] - 1.0;
  double eta2_bar = 2.0 * ref_qsing[1] - 1.0;

  // compute physical length l in each direction
  Point<2> p1_m(0.0, ref_qsing[1]);
  Point<2> p1_p(1.0, ref_qsing[1]);
  double l1 = (mapping.transform_unit_to_real_cell(cell, p1_p)
             - mapping.transform_unit_to_real_cell(cell, p1_m)).norm();

  Point<2> p2_m(ref_qsing[0], 0.0);
  Point<2> p2_p(ref_qsing[0], 1.0);
  double l2 = (mapping.transform_unit_to_real_cell(cell, p2_p)
             - mapping.transform_unit_to_real_cell(cell, p2_m)).norm();

  // normalized distances D
  double D1 = 2.0 * R_min / l1;
  double D2 = 2.0 * R_min / l2;

  // r_bar from table
  double r_bar1 = r_bar_from_table(D1, alpha);
  double r_bar2 = r_bar_from_table(D2, alpha);

  // transformation coefficients
  double a1,b1,c1,d1, a2,b2,c2,d2;
  compute_telles_coeffs(eta1_bar, r_bar1, a1, b1, c1, d1);
  compute_telles_coeffs(eta2_bar, r_bar2, a2, b2, c2, d2);

  // apply transformation to Gauss points
  QGauss<1> gauss_1d(quadrature_order);
  const unsigned int n = gauss_1d.size();

  std::vector<Point<2>>  points;
  std::vector<double>    weights;
  points.reserve(n*n);
  weights.reserve(n*n);

  for (unsigned int i = 0; i < n; ++i)
  {
    for (unsigned int j = 0; j < n; ++j)
    {
      // standard Gauss points in [-1,1]
      double g1 = 2.0*gauss_1d.point(i)[0] - 1.0;
      double g2 = 2.0*gauss_1d.point(j)[0] - 1.0;

      // transformed coords in [-1,1]
      double eta1 = a1*g1*g1*g1 + b1*g1*g1 + c1*g1 + d1;
      double eta2 = a2*g2*g2*g2 + b2*g2*g2 + c2*g2 + d2;

      // Jacobians
      double J1 = 3.0*a1*g1*g1 + 2.0*b1*g1 + c1;
      double J2 = 3.0*a2*g2*g2 + 2.0*b2*g2 + c2;

      // back to [0,1]
      Point<2> pt((eta1+1.0)/2.0, (eta2+1.0)/2.0);

      // weight: original Gauss weight (already in [0,1])
      // times Telles Jacobian (J is for [-1,1], factor 0.5 each dim)
      double w = gauss_1d.weight(i) * gauss_1d.weight(j)
               * J1 * J2;

      points.push_back(pt);
      weights.push_back(w);
    }
  }
  
  return Quadrature<2>(points, weights);
}


Quadrature<2> nested_quasi_singular_quadrature(
    const Point<2> &ref_qsing,
    const unsigned int quadrature_order,
    const unsigned int subdivision_levels)
{
  struct ReferenceBox
  {
    Point<2> lower;
    Point<2> upper;
  };

  std::vector<ReferenceBox> boxes;
  boxes.reserve(3 * subdivision_levels + 1);
  ReferenceBox current{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)};

  // Refine only the child containing the closest point and retain its three
  // siblings as leaves.  This gives O(levels), rather than O(4^levels), cells.
  for (unsigned int level = 0; level < subdivision_levels; ++level)
    {
      const ReferenceBox parent = current;
      const Point<2> midpoint = 0.5 * (parent.lower + parent.upper);
      const unsigned int selected_x = ref_qsing[0] >= midpoint[0] ? 1 : 0;
      const unsigned int selected_y = ref_qsing[1] >= midpoint[1] ? 1 : 0;
      ReferenceBox selected_child;
      for (unsigned int child_x = 0; child_x < 2; ++child_x)
        for (unsigned int child_y = 0; child_y < 2; ++child_y)
          {
            ReferenceBox child;
            child.lower[0] = child_x == 0 ? parent.lower[0] : midpoint[0];
            child.upper[0] = child_x == 0 ? midpoint[0] : parent.upper[0];
            child.lower[1] = child_y == 0 ? parent.lower[1] : midpoint[1];
            child.upper[1] = child_y == 0 ? midpoint[1] : parent.upper[1];
            if (child_x == selected_x && child_y == selected_y)
              selected_child = child;
            else
              boxes.push_back(child);
          }
      current = selected_child;
    }
  boxes.push_back(current);

  const QGauss<2> gauss(quadrature_order);
  std::vector<Point<2>> points;
  std::vector<double>   weights;
  points.reserve(boxes.size() * gauss.size());
  weights.reserve(boxes.size() * gauss.size());
  for (const ReferenceBox &box : boxes)
    {
      const double width  = box.upper[0] - box.lower[0];
      const double height = box.upper[1] - box.lower[1];
      for (unsigned int q = 0; q < gauss.size(); ++q)
        {
          points.emplace_back(box.lower[0] + width * gauss.point(q)[0],
                              box.lower[1] + height * gauss.point(q)[1]);
          weights.push_back(width * height * gauss.weight(q));
        }
    }

  double weight_sum = 0.0;
  for (const double weight : weights)
    weight_sum += weight;
  AssertThrow(std::abs(weight_sum - 1.0) < 1e-12,
              ExcMessage("Nested quadrature does not cover the unit cell."));

  return Quadrature<2>(points, weights);
}


// overload 1 dim
Quadrature<1> telles_quadrature(
    const typename DoFHandler<1,2>::active_cell_iterator &cell,
    const Mapping<1,2> &mapping,
    const Point<2> &singularity,
    const Point<1> &ref_qsing,
    const unsigned int quadrature_order,
    const int alpha)
{
  // closest point in real space
  Point<2> closest_real = mapping.transform_unit_to_real_cell(cell, ref_qsing);

  double R_min = (singularity - closest_real).norm();

  // reference point in [-1,1]
  double eta_bar = 2.0 * ref_qsing[0] - 1.0;

  // physical element length
  Point<1> p_m(0.0);
  Point<1> p_p(1.0);

  double l =
      (mapping.transform_unit_to_real_cell(cell, p_p) -
       mapping.transform_unit_to_real_cell(cell, p_m)).norm();

  // normalized distance
  double D = 2.0 * R_min / l;

  // r_bar from table
  double r_bar = r_bar_from_table(D, alpha);

  // Telles coefficients
  double a, b, c, d;
  compute_telles_coeffs(eta_bar, r_bar, a, b, c, d);

  // Gauss rule
  QGauss<1> gauss_1d(quadrature_order);
  const unsigned int n = gauss_1d.size();

  std::vector<Point<1>> points;
  std::vector<double> weights;

  points.reserve(n);
  weights.reserve(n);

  for (unsigned int i = 0; i < n; ++i)
  {
    // Gauss point in [-1,1]
    double g = 2.0 * gauss_1d.point(i)[0] - 1.0;

    // Telles mapping
    double eta = a*g*g*g + b*g*g + c*g + d;

    // Jacobian of transformation
    double J = 3.0*a*g*g + 2.0*b*g + c;

    // back to [0,1]
    Point<1> x((eta + 1.0) / 2.0);

    points.push_back(x);

    // weight includes mapping Jacobian
    double w = gauss_1d.weight(i) * J;
    weights.push_back(w);
  }

  return Quadrature<1>(points, weights);
}
