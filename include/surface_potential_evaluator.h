#ifndef SURFACE_POTENTIAL_EVALUATOR_H
#define SURFACE_POTENTIAL_EVALUATOR_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <vector>

using namespace dealii;

/**
 * Layer potentials evaluated at points of the boundary itself.
 *
 * The Galerkin form of the gradient representation needs, at quadrature
 * points in the interior and on the edges of every cell, the values
 *
 *   V lambda (x)   = int G(x,y) lambda(y)
 *   R phi (x)      = int dG0/dn_y (x,y) (E(r) phi(y) - phi(x))
 *   K' lambda (x)  = int dG/dn_x (x,y) lambda(y)
 *   V a (x)        = int G(x,y) a(y),        a = n x grad phi,
 *   V (n phi) (x)  = int G(x,y) n_y phi(y),
 *
 * with G = exp(-kappa r) / (4 pi r) the kernel of -Delta + kappa^2, G0 the
 * Laplace kernel and E(r) = (1 + kappa r) exp(-kappa r). For kappa = 0 all
 * of them are the Laplace potentials; R phi is (1/2 I + K) phi at smooth
 * points in either case.
 *
 * all of them weakly singular on a piecewise smooth surface. Panels far from
 * the point use a tensor Gauss rule whose order follows the separation.
 * Panels that contain the point, or come close to it, use a Duffy rule on a
 * box around the closest point of the panel: the box is cut into triangles
 * with their apex there, the collapsing map cancels the 1/r singularity, and
 * when the point is off the panel a sinh map in the radial variable resolves
 * the near singularity. The rest of the panel is subdivided until every
 * piece is well separated.
 *
 * Near and singular panels need the geometry at arbitrary reference points.
 * Rather than building an FEValues object for each of them, the map of every
 * cell is recovered once as the tensor Lagrange polynomial through the
 * mapping evaluated at Gauss-Lobatto nodes. For a polynomial mapping of the
 * given degree this reproduces the map exactly; the reproduction is checked
 * against FEValues at the far-field points.
 */
class SurfacePotentialEvaluator
{
public:
  using CellIterator = typename DoFHandler<2, 3>::active_cell_iterator;

  struct Densities
  {
    /// Flux as finite element coefficients on the scalar dof handler...
    const Vector<double> *flux_coefficients = nullptr;
    /// ... or as the normal component of an exact gradient (self-check).
    const Function<3> *flux_gradient = nullptr;
    /// Trace as an analytical function...
    const Function<3> *trace_function = nullptr;
    /// ... or as finite element coefficients.
    const Vector<double> *trace_coefficients = nullptr;
    /// Gradient of the trace, for a = n x grad phi. Optional.
    const Function<3> *trace_gradient = nullptr;
  };

  struct Values
  {
    double         single_layer         = 0.0;
    double         double_layer         = 0.0;
    double         adjoint_double_layer = 0.0;
    Tensor<1, 3>   curl_single_layer;
    /// V (n phi): only with a trace gradient and kappa > 0, for the term
    /// kappa^2 (n_x . n_y) of the screened hypersingular operator.
    Tensor<1, 3>   normal_single_layer;
  };

  struct Geometry
  {
    Point<3>     point;
    Tensor<1, 3> normal;
    double       jacobian = 0.0;
    /// Columns of the Jacobian of the map, d x / d xi_a.
    std::array<Tensor<1, 3>, 2> tangents;
    /// Total curvature div_S n, from the second fundamental form.
    double curvature = 0.0;
  };

  SurfacePotentialEvaluator(const Mapping<2, 3>       &mapping,
                            const DoFHandler<2, 3>    &dof_handler,
                            const unsigned int         geometry_degree,
                            const Densities           &densities,
                            const unsigned int         duffy_order = 10,
                            const double               near_ratio  = 0.5,
                            const unsigned int         maximum_depth = 10,
                            const double               screening   = 0.0);

  /// Potentials at x on the surface, with n_x its normal and phi_x the trace
  /// used by the regularisation of the double layer.
  Values
  evaluate(const Point<3>     &x,
           const Tensor<1, 3> &normal,
           const double        trace_at_x) const;

  /// Geometry of a cell at a reference point.
  Geometry
  geometry(const CellIterator &cell, const Point<2> &reference) const;

  /// Largest distance between the reconstructed map and FEValues at the
  /// far-field points, relative to the cell diameter.
  double
  geometry_deviation() const
  {
    return maximum_geometry_deviation;
  }

  /// Counts of panel treatments since construction: far panels, leaves of
  /// the subdivision, near or singular panels (Duffy).
  std::array<unsigned long long, 3>
  counts() const
  {
    return {{n_far, n_leaves, n_singular}};
  }

  /// Number of source points evaluated off the precomputed far-field rules.
  unsigned long long
  source_evaluations() const
  {
    return n_sources;
  }

private:
  struct Source
  {
    Point<3>     point;
    Tensor<1, 3> normal;
    double       weight = 0.0;
    double       flux   = 0.0;
    double       trace  = 0.0;
    Tensor<1, 3> curl_density;
  };

  struct CellData
  {
    CellIterator              cell;
    std::vector<Point<3>>     nodes;
    double                    orientation = 1.0;
    Point<3>                  centre;
    double                    radius   = 0.0;
    double                    diameter = 0.0;
    std::vector<double>       flux_local;
    std::vector<double>       trace_local;
    std::array<std::vector<Source>, 5> far_rules;
  };

  Source
  source(const CellData &data, const Point<2> &reference, const double weight)
    const;

  Geometry
  geometry(const CellData &data,
           const Point<2> &reference,
           const bool      with_curvature = false) const;

  void
  add(const Point<3>     &x,
      const Tensor<1, 3> &normal,
      const double        trace_at_x,
      const Source       &y,
      Values             &values) const;

  /// Closest point of the cell to x, by projected Gauss-Newton.
  std::pair<Point<2>, double>
  closest_point(const CellData &data, const Point<3> &x) const;

  /// Panel containing x (distance zero) or close to it: Duffy rule on a
  /// box around the closest point, with a sinh map in the radial variable
  /// when the distance is positive; subdivision on the rest of the panel.
  void
  integrate_near(const CellData     &data,
                     const Point<2>     &apex,
                     const double        distance,
                     const Point<3>     &x,
                     const Tensor<1, 3> &normal,
                     const double        trace_at_x,
                     Values             &values) const;

  void
  integrate_patch(const CellData     &data,
                  const Point<2>     &lower,
                  const Point<2>     &upper,
                  const unsigned int  depth,
                  const Point<3>     &x,
                  const Tensor<1, 3> &normal,
                  const double        trace_at_x,
                  Values             &values) const;

  const Mapping<2, 3>    &mapping;
  const DoFHandler<2, 3> &dof_handler;
  const unsigned int      geometry_degree;
  const Densities         densities;
  const unsigned int      duffy_order;
  const double            near_ratio;
  const unsigned int      maximum_depth;
  const double            screening;
  const bool              wants_curl;

  std::vector<double>   lagrange_nodes;
  /// The Lagrange bases through lagrange_nodes and shape_nodes in monomial
  /// form, for Horner evaluation: values, first and second derivatives.
  std::vector<double>   lagrange_coefficients;
  std::vector<double>   shape_coefficients;
  /// One-dimensional nodes of the scalar element, when its basis is the
  /// tensor Lagrange basis on them in lexicographic order (checked); empty
  /// otherwise, and the element is asked directly.
  std::vector<double>   shape_nodes;
  /// Position (a, b) of the support point of each basis function in the
  /// tensor grid of shape_nodes: the function is L_a(x) L_b(y).
  std::vector<std::array<unsigned int, 2>> shape_tensor_index;
  std::vector<CellData> cells;
  std::vector<unsigned int> cell_position;

  double maximum_geometry_deviation = 0.0;

  mutable Vector<double>             gradient_buffer;
  mutable unsigned long long         n_far      = 0;
  mutable unsigned long long         n_leaves   = 0;
  mutable unsigned long long         n_singular = 0;
  mutable unsigned long long         n_sources  = 0;
};

#endif
