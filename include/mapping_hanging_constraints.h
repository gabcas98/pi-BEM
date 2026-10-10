#ifndef mapping_hanging_constraints_h
#define mapping_hanging_constraints_h

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>

DEAL_II_NAMESPACE_OPEN
namespace ExactGeometry
{
  struct HangingConstraintStatistics
  {
    unsigned int corrected_lines = 0;
    unsigned int off_edge_points = 0;
    double maximum_weight_change = 0.;
    double maximum_geometric_gap = 0.;
  };

  /** Build hanging-node constraints at the physical support points of the
   * mapping. On CAD-refined curved edges, a child midpoint need not be the
   * midpoint of the coarse cell's parameter interval. For primitive nodal
   * elements on quadrilateral surfaces, evaluate the coarse trace at the
   * actual fine support point instead of using the topological fraction.
   * Noncoincident edges retain their standard constraint and are counted.
   * The input constraint object must have no existing lines; the caller
   * closes it after any further constraints have been added.
   */
  template <int dim, int spacedim>
  HangingConstraintStatistics
  make_mapping_hanging_node_constraints(
    const dealii::Mapping<dim, spacedim> &mapping,
    const dealii::DoFHandler<dim, spacedim> &dh,
    dealii::AffineConstraints<double> &constraints);
}
DEAL_II_NAMESPACE_CLOSE

#endif
