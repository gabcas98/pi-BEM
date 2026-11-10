// The Sauter-Schwab rules adapted to the geometry of the pair
// (GalerkinPanelPairIntegrator::set_singular_adaptation()): a rhombus of 20
// degrees, as the cells of a planar tip at the sharp trailing edge of a
// wing, with the square that meets it at a right angle along an edge, as
// the cell of the surface of the wing, and with itself. With seven points
// per variable the adapted rule has to be far more accurate than the plain
// one, against the plain rule of order 16; with 16 points it has to agree
// with it; and a pair of well-shaped cells has to keep its rule.

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>

#include "galerkin_panel_pair_integrator.h"
#include "tests.h"

int
main()
{
  initlog();
  const double                c        = std::cos(20. * numbers::PI / 180.);
  const double                s        = std::sin(20. * numbers::PI / 180.);
  const std::vector<Point<3>> vertices = {Point<3>(0, 0, 0),
                                          Point<3>(1, 0, 0),
                                          Point<3>(0, -1, 0),
                                          Point<3>(1, -1, 0),
                                          Point<3>(c, 0, s),
                                          Point<3>(1 + c, 0, s),
                                          Point<3>(0, -2, 0),
                                          Point<3>(1, -2, 0)};
  std::vector<CellData<2>>    data(3);
  // The rhombus in the plane y = 0, the square in the plane z = 0 along
  // their common edge on the x axis, and a second square beyond it.
  data[0].vertices = {{0, 1, 4, 5}};
  data[1].vertices = {{2, 3, 0, 1}};
  data[2].vertices = {{6, 7, 2, 3}};
  GridTools::consistently_order_cells(data);
  Triangulation<2, 3> tria;
  tria.create_triangulation(vertices, data, SubCellData());
  const FE_Q<2, 3>      fe(1);
  const MappingQ1<2, 3> mapping;
  DoFHandler<2, 3>      dh(tria);
  dh.distribute_dofs(fe);
  const auto rhombus = dh.begin_active();
  const auto square  = std::next(rhombus);
  const auto beyond  = std::next(square);

  const auto rule =
    [&](const unsigned int order, const bool adapt, const double ratio = 0.3) {
      GalerkinPanelPairIntegrator<3> integrator(
        mapping,
        fe,
        GalerkinSingularQuadrature::sauter_schwab,
        order,
        4,
        0.5,
        6,
        4);
      integrator.set_singular_adaptation(adapt, ratio);
      return integrator;
    };
  // Largest difference of the blocks V, K and W, relative to the largest
  // entry of each.
  const auto difference = [&](const FullMatrix<double> &x,
                              const FullMatrix<double> &reference) {
    std::array<double, 3> diff = {{0, 0, 0}}, scale = {{0, 0, 0}};
    for (unsigned int i = 0; i < x.m(); ++i)
      for (unsigned int j = 0; j < x.n(); ++j)
        {
          const unsigned int k = j / fe.dofs_per_cell;
          diff[k]  = std::max(diff[k], std::abs(x(i, j) - reference(i, j)));
          scale[k] = std::max(scale[k], std::abs(reference(i, j)));
        }
    for (unsigned int k = 0; k < 3; ++k)
      diff[k] = scale[k] > 0 ? diff[k] / scale[k] : 0.;
    return diff;
  };

  // The rhombus and the square across the sharp edge.
  {
    const auto reference =
      rule(16, false).integrate_single_double_hypersingular(rhombus, square, 8);
    const auto plain = difference(
      rule(7, false).integrate_single_double_hypersingular(rhombus, square, 8),
      reference);
    auto       adapted_rule = rule(7, true);
    const auto adapted      = difference(
      adapted_rule.integrate_single_double_hypersingular(rhombus, square, 8),
      reference);
    AssertThrow(adapted_rule.n_adapted_pairs() > 0,
                ExcMessage("The pair across the edge was not adapted."));
    for (unsigned int k = 0; k < 3; ++k)
      AssertThrow(adapted[k] < 0.05 * plain[k],
                  ExcMessage("The adapted rule did not improve the pair "
                             "across the edge."));
    const auto high = difference(
      rule(16, true).integrate_single_double_hypersingular(rhombus, square, 8),
      reference);
    for (unsigned int k = 0; k < 3; ++k)
      AssertThrow(high[k] < 1e-6,
                  ExcMessage("The adapted rule of order 16 disagrees."));
    deallog << "Rhombus and square across a sharp edge: OK" << std::endl;
  }

  // The rhombus with itself, adapted from the ratio 0.5 on.
  {
    const auto reference =
      rule(16, false).integrate_single_double_hypersingular(rhombus,
                                                            rhombus,
                                                            8);
    const auto plain = difference(
      rule(7, false).integrate_single_double_hypersingular(rhombus, rhombus, 8),
      reference);
    const auto adapted =
      difference(rule(7, true, 0.5)
                   .integrate_single_double_hypersingular(rhombus, rhombus, 8),
                 reference);
    AssertThrow(adapted[0] < 0.05 * plain[0] && adapted[2] < 0.05 * plain[2],
                ExcMessage("The adapted rule did not improve the rhombus "
                           "with itself."));
    deallog << "Rhombus with itself: OK" << std::endl;
  }

  // Two squares in a plane keep the plain rule, entry by entry.
  {
    auto       adapted_rule = rule(7, true);
    const auto adapted =
      adapted_rule.integrate_single_double_hypersingular(square, beyond, 8);
    const auto plain =
      rule(7, false).integrate_single_double_hypersingular(square, beyond, 8);
    AssertThrow(adapted_rule.n_adapted_pairs() == 0,
                ExcMessage("A pair of squares was adapted."));
    for (unsigned int i = 0; i < plain.m(); ++i)
      for (unsigned int j = 0; j < plain.n(); ++j)
        AssertThrow(adapted(i, j) == plain(i, j),
                    ExcMessage("The rule of a pair of squares changed."));
    deallog << "Squares in a plane: OK" << std::endl;
  }
}
