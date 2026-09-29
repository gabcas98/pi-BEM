#ifndef TELLES_QUADRATURE_H
#define TELLES_QUADRATURE_H

#include <deal.II/base/quadrature.h>
#include <deal.II/base/point.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

Quadrature<2> telles_quadrature(
    const typename DoFHandler<2,3>::active_cell_iterator &cell,
    const Mapping<2,3> &mapping,
    const Point<3> &singularity,
    const Point<2> &ref_qsing,
    const unsigned int quadrature_order,
    const int alpha = 1  // default to 1/r kernel
);

Quadrature<2> nested_quasi_singular_quadrature(
    const Point<2> &ref_qsing,
    const unsigned int quadrature_order,
    const unsigned int subdivision_levels
);

Quadrature<1> telles_quadrature(
    const typename DoFHandler<1,2>::active_cell_iterator &cell,
    const Mapping<1,2> &mapping,
    const Point<2> &singularity,
    const Point<1> &ref_qsing,
    const unsigned int quadrature_order,
    const int alpha = 1  // default to 1/r kernel
);

#endif
