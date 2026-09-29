#include "tests.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "laplace_kernel.h"
#include "telles_quadrature.h"

int
main()
{
  initlog();

  Triangulation<2, 3> triangulation;
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  FE_Q<2, 3>     fe(1);
  DoFHandler<2, 3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  MappingQ1<2, 3> mapping;

  const double   distance = 0.003;
  const Point<3> source(0.5, 0.5, distance);
  const Point<2> projection(0.5, 0.5);
  const double half_width = 0.5;
  const double solid_angle =
    4.0 * std::atan(half_width * half_width /
                    (distance *
                     std::sqrt(distance * distance +
                               2.0 * half_width * half_width)));
  const double exact = solid_angle / (4.0 * numbers::PI);

  const auto cell = dof_handler.begin_active();
  double integral = 0.0;
  double area     = 0.0;
  for (const unsigned int order : {8U, 12U, 18U, 27U, 41U, 62U, 93U, 128U})
    {
      const Quadrature<2> quadrature =
        telles_quadrature(cell, mapping, source, projection, order, 2);
      FEValues<2, 3> fe_values(mapping,
                               fe,
                               quadrature,
                               update_quadrature_points |
                                 update_normal_vectors | update_JxW_values);
      fe_values.reinit(cell);

      integral = 0.0;
      area     = 0.0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          Tensor<1, 3> double_layer;
          double       single_layer;
          LaplaceKernel::kernels(fe_values.quadrature_point(q) - source,
                                 double_layer,
                                 single_layer);
          integral += double_layer * fe_values.normal_vector(q) *
                      fe_values.JxW(q);
          area += fe_values.JxW(q);
        }
    }

  AssertThrow(std::abs(area - 1.0) < 1.e-12,
              ExcMessage("Telles transformation does not preserve area."));
  AssertThrow(std::abs(integral - exact) / exact < 1.e-8,
              ExcMessage("Telles near-singular integral did not converge."));
  deallog << "Telles area and near-singular Neumann integral: OK" << std::endl;
}
