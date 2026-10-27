#ifndef cellwise_function_h
#define cellwise_function_h

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

#include <map>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * The boundary data of BoundaryConditions, one function for each slot of the
 * manifolds (ComputationalDomain: Manifold to boundary slot map). The
 * Galerkin code takes on every cell the function of its slot, through
 * on_cell(); a point is evaluated with the function of the cell it lies on,
 * also in the regularised integrals, where the datum at the test point
 * follows the test cell. Elsewhere this is the function of the first slot.
 */
template <int dim>
class CellwiseFunction : public Function<dim>
{
public:
  CellwiseFunction(const std::vector<const Function<dim> *>        &pieces,
                   const std::map<types::manifold_id, unsigned int> &slots)
    : Function<dim>(pieces[0]->n_components)
    , pieces(pieces)
    , slots(slots)
  {}

  /// The function of the slot of the manifold of the cell.
  template <class CellIterator>
  const Function<dim> &
  on(const CellIterator &cell) const
  {
    const auto slot = slots.find(cell->manifold_id());
    return *pieces[slot == slots.end() ? 0 : slot->second];
  }

  /// Whether some manifold takes another function than the first.
  bool
  several() const
  {
    for (const auto &[manifold, slot] : slots)
      if (slot != 0 && pieces[slot] != pieces[0])
        return true;
    return false;
  }

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    return pieces[0]->value(p, component);
  }

  void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    pieces[0]->vector_value(p, values);
  }

private:
  const std::vector<const Function<dim> *>   pieces;
  const std::map<types::manifold_id, unsigned int> slots;
};

/// The function of the cell: the piece of a CellwiseFunction, or the
/// function itself.
template <int dim, class CellIterator>
const Function<dim> &
on_cell(const Function<dim> &function, const CellIterator &cell)
{
  if (const auto *cellwise =
        dynamic_cast<const CellwiseFunction<dim> *>(&function))
    return cellwise->on(cell);
  return function;
}

DEAL_II_NAMESPACE_CLOSE

#endif
