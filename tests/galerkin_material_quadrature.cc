#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>

#include "galerkin_panel_pair_integrator.h"
#include "tests.h"

int
main()
{
  initlog();
  // Two skew quads meet orthogonally, as a wing surface and a planar tip.
  const std::vector<Point<3>> vertices = {Point<3>(0, 0, 0),
                                          Point<3>(1, 0, 0),
                                          Point<3>(0, 1, 0),
                                          Point<3>(1, 1, 0),
                                          Point<3>(0.4, 0, 0.2),
                                          Point<3>(1.4, 0, 0.3)};
  std::vector<CellData<2>>    data(2);
  data[0].vertices    = {{0, 1, 2, 3}};
  data[0].material_id = 4;
  data[1].vertices    = {{0, 4, 1, 5}};
  data[1].material_id = 2;
  GridTools::consistently_order_cells(data);
  Triangulation<2, 3> tria;
  tria.create_triangulation(vertices, data, SubCellData());
  const FE_Q<2, 3>      fe(1);
  const MappingQ1<2, 3> mapping;
  DoFHandler<2, 3>      dh(tria);
  dh.distribute_dofs(fe);
  const auto                     a = dh.begin_active();
  const auto                     b = std::next(a);
  GalerkinPanelPairIntegrator<3> low(
    mapping, fe, GalerkinSingularQuadrature::sauter_schwab, 3, 4, 0.5, 6, 4);
  GalerkinPanelPairIntegrator<3> high(
    mapping, fe, GalerkinSingularQuadrature::sauter_schwab, 8, 4, 0.5, 6, 4);
  GalerkinPanelPairIntegrator<3> selected(
    mapping, fe, GalerkinSingularQuadrature::sauter_schwab, 3, 4, 0.5, 6, 4);
  selected.set_folded_singular_order(8);
  const auto baseline = low.integrate_single_double_hypersingular(a, b, 4);
  const auto folded   = selected.integrate_single_double_hypersingular(a, b, 4);
  selected.set_material_singular_order({2}, 8);
  const auto promoted = selected.integrate_single_double_hypersingular(a, b, 4);
  const auto reference = high.integrate_single_double_hypersingular(a, b, 4);
  double     change    = 0.;
  for (unsigned int i = 0; i < reference.m(); ++i)
    for (unsigned int j = 0; j < reference.n(); ++j)
      {
        AssertThrow(std::abs(folded(i, j) - baseline(i, j)) < 1e-14,
                    ExcMessage("Orthogonal contact was classified as folded."));
        AssertThrow(std::abs(promoted(i, j) - reference(i, j)) < 1e-12,
                    ExcMessage("Selected material order was not applied."));
        change = std::max(change, std::abs(promoted(i, j) - baseline(i, j)));
      }
  AssertThrow(change > 1e-8,
              ExcMessage("The quadrature comparison is trivial."));
  // Pairs without a selected material must retain the original rule.
  const auto unselected = selected.integrate_single_layer(a, a, 4);
  const auto original   = low.integrate_single_layer(a, a, 4);
  for (unsigned int i = 0; i < original.m(); ++i)
    for (unsigned int j = 0; j < original.n(); ++j)
      AssertThrow(std::abs(unselected(i, j) - original(i, j)) < 1e-14,
                  ExcMessage("Unselected material order changed."));
  deallog << "Selected material quadrature at orthogonal contacts: OK"
          << std::endl;
}
