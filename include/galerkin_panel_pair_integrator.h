#ifndef GALERKIN_PANEL_PAIR_INTEGRATOR_H
#define GALERKIN_PANEL_PAIR_INTEGRATOR_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>

#include <functional>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/function.h>

#include "projected_mapping.h"
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

enum class GalerkinPanelPairType
{
  same_panel,
  common_edge,
  nonconforming_edge,
  common_vertex,
  regular,
  unsupported
};

enum class GalerkinSingularQuadrature
{
  sauter_schwab,
  iterated_duffy
};

/**
 * Integrates local Galerkin single-layer blocks on quadrilateral surface
 * cells. Singular pairs can use either the four-dimensional changes of
 * variables from Sauter and Schwab or an iterated Duffy rule. Regular pairs
 * use tensor-product Gauss quadrature.
 *
 * This class is intentionally independent from BEMProblem so that pair
 * quadrature can be validated before it is used by a global formulation.
 */
template <int spacedim>
class GalerkinPanelPairIntegrator
{
public:
  static_assert(spacedim == 3,
                "The panel-pair integrator currently supports surfaces in 3D.");

  using CellIterator =
    typename DoFHandler<spacedim - 1, spacedim>::active_cell_iterator;

  struct CrossLayerMatrices
  {
    FullMatrix<double> single_layer;
    FullMatrix<double> adjoint_double_layer;
  };

  GalerkinPanelPairIntegrator(
    const Mapping<spacedim - 1, spacedim>       &mapping,
    const FiniteElement<spacedim - 1, spacedim> &finite_element,
    const GalerkinSingularQuadrature singular_quadrature =
      GalerkinSingularQuadrature::sauter_schwab,
    const unsigned int singular_order = 0,
    const unsigned int duffy_outer_subdivisions = 4,
    const double near_singular_distance_ratio = 0.0,
    const unsigned int near_singular_maximum_depth = 0,
    const unsigned int near_singular_quadrature_order = 0);

  /** Allow the far field to use a lower quadrature order where the panels are
   * well separated. The cost of a regular pair grows as the fourth power of
   * the order, so dropping from six to three on a distant pair is sixteen
   * times cheaper. The order is chosen per pair from the separation ratio
   * rho = (r_test + r_trial) / |c_test - c_trial|, which bounds the
   * convergence factor of the tensor Gauss rule against the 1/r kernel: the
   * smallest order n with rho^(2n) below the tolerance is used. A tolerance
   * of zero, the default, disables the reduction and keeps the full order
   * everywhere.
   */
  /**
   * The order of the singular rule for folded pairs, cells with a common
   * edge or vertex whose normals make a cosine below the given one, as the
   * upper and lower cells at a thin trailing edge: the integrand is then
   * nearly singular across both cells, and the rule of the other singular
   * pairs loses accuracy. Zero, the default, uses the singular order for
   * them too.
   */
  void
  set_folded_singular_order(const unsigned int order,
                            const double       cosine = -0.5)
  {
    folded_singular_order  = order;
    folded_singular_cosine = cosine;
  }

  /** Promote touching pairs involving any selected material, including
   * coincident pairs. This also covers orthogonal surface--tip contacts,
   * which the folded-pair normal criterion does not select. Empty ids or
   * order zero leave the ordinary singular rule unchanged.
   */
  void
  set_material_singular_order(const std::set<types::material_id> &ids,
                              const unsigned int order)
  {
    singular_material_ids = ids;
    material_singular_order = order;
  }

  /** Promote touching pairs involving a skewed cell: one with an interior
   * angle, measured at its vertices, below the given one, as the thin
   * rhombi of a planar face near a sharp corner of its boundary. Their
   * edges have similar lengths, so the anisotropic balancing does not split
   * them, and the singular rules lose accuracy on them. Order zero leaves
   * the ordinary singular rule unchanged.
   */
  void
  set_skewed_singular_order(const unsigned int order,
                            const double       minimum_angle_degrees = 30.)
  {
    skewed_singular_order = order;
    skewed_minimum_angle  = minimum_angle_degrees * M_PI / 180.;
  }

  /** Promote touching pairs whose normals make an angle above the given
   * one, the cells on the two sides of a sharp edge of the surface, as the
   * edge between the tip and the surface of a wing, at about 90 degrees:
   * the singular rules of the pairs within a smooth surface lose accuracy
   * on them, most where the surface also bends, as at the leading edge.
   * The normals are those of the vertices of the cells. Folded pairs
   * (set_folded_singular_order()) are kinked pairs too; the higher of the
   * two orders applies. Order zero leaves the ordinary singular rule
   * unchanged.
   */
  void
  set_kinked_singular_order(const unsigned int order,
                            const double       minimum_angle_degrees = 60.)
  {
    kinked_singular_order  = order;
    kinked_singular_cosine = std::cos(minimum_angle_degrees * M_PI / 180.);
  }

  /** Widen the near-singular test for kinked pairs that do not touch
   * (set_kinked_singular_order() gives their angle), as the upper and lower
   * surfaces of a thin body, or the tip of a wing and the surfaces next to
   * it: the double-layer kernel (x - y) . n / |x - y|^3 of one cell across
   * the other carries a weight of order one within a distance of the order
   * of their separation, which the tensor rule of the far field resolves
   * only at separations of several cell sizes, while on one smooth surface
   * it nearly vanishes. Kinked pairs closer than the given ratio times the
   * larger diameter are near-singular; a ratio not above the near-singular
   * distance ratio leaves the test unchanged.
   */
  void
  set_kinked_distance_ratio(const double ratio)
  {
    kinked_distance_ratio = ratio;
  }

  void
  set_far_field_tolerance(const double tolerance)
  {
    far_field_tolerance = tolerance;
    far_field_order_histogram.fill(0);
  }

  /** Screening parameter of the modified Helmholtz equation
   * -Delta u + kappa^2 u = 0, whose fundamental solution is
   * exp(-kappa r) / (4 pi r). Zero, the default, is the Laplace kernel. The
   * normal derivatives of the screened kernel are those of the Laplace kernel
   * times E(r) = (1 + kappa r) exp(-kappa r), so the singularities, and with
   * them the quadrature, are unchanged.
   */
  void
  set_screening(const double kappa)
  {
    screening = kappa;
  }

  /// Free the work arrays that the direct path keeps from pair to pair, at
  /// the end of an assembly, so that they do not add to the memory of what
  /// follows; they grow again when needed.
  void
  release_work_memory() const;

  /// The cached singular rules of the direct path: their number and size,
  /// split into the base order and the raised orders, and the size of the
  /// point sets; for diagnosing memory.
  std::string
  cache_statistics() const;

  double
  get_screening() const
  {
    return screening;
  }

  /** A gradient whose normal trace lambda the fused traversals of the
   * single layer and the double-layer right-hand side also integrate
   * against the single layer: integrate_single_and_double_layer() then
   * returns <V lambda, phi_i> of the test cell in one more column, and
   * integrate_single_and_double_layer_both() that of the test cell and that
   * of the trial cell in two more. The consistent right-hand side of a
   * manufactured solution so costs the evaluation of the gradient at the
   * points of the rules, which are already there, instead of a traversal
   * of its own. nullptr, the default, disables it.
   */
  void
  set_flux_function(const Function<spacedim> *gradient)
  {
    AssertThrow(gradient == nullptr || gradient->n_components == spacedim,
                ExcMessage("The flux function is a gradient, with one "
                           "component per space dimension."));
    flux_function = gradient;
  }

  const Function<spacedim> *
  get_flux_function() const
  {
    return flux_function;
  }

  /** Evaluate the geometry and the shape functions of the near-field rules
   * directly, instead of through FEValues. The mapping is a tensor
   * polynomial on each cell, so its values at the Gauss-Lobatto nodes of its
   * degree determine it exactly; the element is the tensor Lagrange basis on
   * its support points. Building an FEValues object costs as much as the
   * kernel loop of the pair it serves, and the near-singular subdivision
   * builds one pair of them for every leaf. Only the single-layer and
   * double-layer operators take this path; the others keep FEValues.
   *
   * Returns whether the path is active: it needs a MappingQ, and it is
   * checked against FEValues on every cell, the largest deviation being
   * available from light_geometry_deviation(). On an
   * ExactGeometry::ProjectedMapping the reconstruction is that of the mesh
   * mapping, and every point is projected onto the exact geometry.
   */
  bool
  set_light_geometry(const bool enable);

  /// Largest deviation of the reconstructed geometry from FEValues, over the
  /// cells seen so far: point distance relative to the cell diameter, normal
  /// difference and relative surface element.
  double
  light_geometry_deviation() const
  {
    return maximum_light_deviation;
  }

  /// Global indices of the degrees of freedom of a cell for
  /// assemble_single_layer(), assemble_double_layer() and
  /// assemble_weak_hypersingular(), in place of those of the DoFHandler: a
  /// block formulation may give a shape function different indices on
  /// different cells. An empty function restores the DoFHandler.
  void
  set_dof_index_map(
    std::function<void(const CellIterator &,
                       std::vector<types::global_dof_index> &)> map)
  {
    dof_index_map = std::move(map);
  }

  /// Whether the direct path evaluates the element as a tensor Lagrange
  /// basis, rather than through the library.
  bool
  tensor_shape_functions() const
  {
    return !shape_nodes.empty();
  }

  /// Profile of the double-layer right-hand side assembly: seconds in far
  /// pairs, seconds in near and singular pairs, and, within the latter, the
  /// number of evaluations of the trial function and the seconds they took.
  struct RhsProfile
  {
    double             far_seconds       = 0.0;
    double             near_seconds      = 0.0;
    double             function_seconds  = 0.0;
    unsigned long long function_calls    = 0;
    /// Rule construction (pair quadrature and FEValues) for the pairs that
    /// miss the cache, and how many they were.
    double             setup_seconds     = 0.0;
    unsigned long long uncached_pairs    = 0;
    /// Everything spent in integrate_pairs, the kernel loops.
    double             kernel_seconds    = 0.0;
    /// On the direct path: leaves integrated with a singular rule and with a
    /// tensor rule, the seconds each took, and the pair points of the former.
    unsigned long long singular_leaves   = 0;
    double             singular_seconds  = 0.0;
    unsigned long long singular_points   = 0;
    unsigned long long regular_leaves    = 0;
    double             regular_seconds   = 0.0;
    /// On the direct path: points at which the cell geometry was evaluated,
    /// the seconds it took, and the seconds spent on the shape values.
    unsigned long long geometry_points   = 0;
    double             geometry_seconds  = 0.0;
    double             shape_seconds     = 0.0;
    /// Parts of the kernel loops of the joint operators on paired rules:
    /// the kernel weights, the surface curls and the sums.
    double             kernel_weight_seconds = 0.0;
    double             kernel_curl_seconds   = 0.0;
    double             kernel_sum_seconds    = 0.0;
    /// Singular leaves on patches with the shapes of their kept rule
    /// (LightLeaf::transformed), and those of a panel with itself on half
    /// the rule (LightLeaf::half).
    unsigned long long transformed_leaves = 0;
    unsigned long long half_leaves        = 0;
  };

  const RhsProfile &
  rhs_profile() const
  {
    return profile;
  }

  const std::array<unsigned long long, 32> &
  far_field_order_counts() const
  {
    return far_field_order_histogram;
  }

  /** True when the pair needs no singular or near-singular treatment, so a
   * caller holding precomputed per-cell quadrature data can integrate it with
   * a plain tensor sum instead of going through the full machinery.
   */
  bool
  is_far_regular_pair(const CellIterator &test_cell,
                      const CellIterator &trial_cell) const;

  unsigned int
  n_near_singular_pairs() const;

  unsigned int
  n_near_singular_leaves() const;

  /// Whether two whole cells that do not touch are close enough for the
  /// near-singular subdivision.
  bool
  is_near_singular_pair(const CellIterator &test_cell,
                        const CellIterator &trial_cell) const
  {
    const ReferencePatch whole{Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)};
    return is_near_singular(test_cell, whole, trial_cell, whole);
  }

  GalerkinPanelPairType
  classify(const CellIterator &test_cell,
           const CellIterator &trial_cell) const;

  /** Contribution of one panel pair to the squared energy norm of the flux
   * error, that is to the double integral of the single-layer kernel against
   * e(x) e(y) with e = lambda - lambda_h. The energy norm is the one in which
   * the Galerkin method for the first-kind equation is quasi-optimal; the L2
   * norm is not, and on a strongly graded mesh the two disagree.
   *
   * The error is evaluated as the exact normal flux, obtained by contracting
   * the given gradient with the surface normal, minus the discrete flux built
   * from the local coefficients. Forming e first avoids the cancellation that
   * would come from subtracting two nearly equal energy products.
   */
  double
  integrate_error_energy(const CellIterator        &test_cell,
                         const CellIterator        &trial_cell,
                         const unsigned int         order,
                         const Function<spacedim>  &exact_gradient,
                         const std::vector<double> &test_coefficients,
                         const std::vector<double> &trial_coefficients) const;

  FullMatrix<double>
  integrate_single_layer(const CellIterator &test_cell,
                         const CellIterator &trial_cell,
                         const unsigned int  order) const;

  FullMatrix<double>
  integrate_double_layer(const CellIterator &test_cell,
                         const CellIterator &trial_cell,
                         const unsigned int  order) const;

  FullMatrix<double>
  integrate_weak_hypersingular(const CellIterator &test_cell,
                               const CellIterator &trial_cell,
                               const unsigned int  order) const;

  /// V, K and W of the Laplace kernel for one panel pair from the same
  /// quadrature points: the columns of V, then of K, then of W.
  FullMatrix<double>
  integrate_single_double_hypersingular(const CellIterator &test_cell,
                                        const CellIterator &trial_cell,
                                        const unsigned int  order) const;

  /// The same, and in a fourth block of columns K of the pair the other way
  /// round, trial cell as test cell, transposed: entry (i, 3 dpc + j) is
  /// entry (j, i) of the K of (trial cell, test cell), from the same points.
  FullMatrix<double>
  integrate_single_double_hypersingular_both(const CellIterator &test_cell,
                                             const CellIterator &trial_cell,
                                             const unsigned int  order) const;

  /** Integrate V(test,trial) and K'(test,trial) in one traversal. This
   * routine is intended for geometrically disjoint panel pairs, for which
   * both kernels are regular or nearly singular.
   */
  CrossLayerMatrices
  integrate_cross_layers(const CellIterator &test_cell,
                         const CellIterator &trial_cell,
                         const unsigned int  order) const;

  Vector<double>
  integrate_double_layer(const CellIterator      &test_cell,
                         const CellIterator      &trial_cell,
                         const unsigned int       order,
                         const Function<spacedim> &trial_function,
                         const bool regularize_trial_function = false) const;

  /** Contribution of one panel pair to <V lambda, phi_i>, with lambda given
   * as the normal trace of the supplied gradient. This is the right-hand
   * side the Galerkin equations need for a manufactured solution, expressed
   * through the single-layer kernel rather than the double layer.
   */
  Vector<double>
  integrate_single_layer_flux(const CellIterator       &test_cell,
                              const CellIterator       &trial_cell,
                              const unsigned int        order,
                              const Function<spacedim> &gradient) const;

  /** Assemble <V lambda, phi_i> over the whole surface. The caller owns
   * vector initialization and compression.
   */
  template <typename VectorType>
  void
  assemble_single_layer_flux_rhs(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    VectorType                                &rhs,
    const unsigned int                         order,
    const Function<spacedim>                  &gradient,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id) const;

  /** Contribution of one panel pair to <K' lambda, phi_i>, with the kernel
   * dG/dn_x taken at the test point. Without a function the trial side is
   * the finite element basis; with one, lambda is the normal trace of the
   * supplied gradient.
   */
  FullMatrix<double>
  integrate_adjoint_double_layer(const CellIterator &test_cell,
                                 const CellIterator &trial_cell,
                                 const unsigned int  order) const;

  Vector<double>
  integrate_adjoint_double_layer_flux(const CellIterator       &test_cell,
                                      const CellIterator       &trial_cell,
                                      const unsigned int        order,
                                      const Function<spacedim> &gradient) const;

  Vector<double>
  integrate_hypersingular_rhs(const CellIterator &test_cell,
                             const CellIterator &trial_cell,
                             unsigned int order,
                             const Function<spacedim> &gradient) const;

  /**
   * Assemble the weak single-layer operator. Constraints are applied to both
   * test and trial functions, as required for hanging-node elimination.
   * The caller owns matrix initialization and compression.
   */
  template <typename MatrixType>
  void
  assemble_single_layer(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &matrix,
    const unsigned int                         order,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const types::material_id selected_test_material =
      numbers::invalid_material_id,
    const types::material_id selected_trial_material =
      numbers::invalid_material_id) const;

  template <typename MatrixType>
  void
  assemble_double_layer(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &matrix,
    const unsigned int                         order,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const types::material_id selected_test_material =
      numbers::invalid_material_id,
    const types::material_id selected_trial_material =
      numbers::invalid_material_id) const;

  /** Assemble the adjoint double-layer operator K' as the transpose weak
   * form of K. Test-cell ownership is retained, so this also works with
   * distributed row matrices.
   */
  template <typename MatrixType>
  void
  assemble_adjoint_double_layer(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &matrix,
    const unsigned int                         order,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const types::material_id selected_test_material =
      numbers::invalid_material_id,
    const types::material_id selected_trial_material =
      numbers::invalid_material_id) const;

  template <typename VectorType>
  void
  assemble_double_layer_rhs(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    VectorType                                &rhs,
    const unsigned int                         order,
    const Function<spacedim>                  &trial_function,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const bool regularize_trial_function = false,
    const std::vector<unsigned int> *trial_cell_components = nullptr,
    const unsigned int selected_trial_component =
      numbers::invalid_unsigned_int,
    const types::material_id selected_test_material =
      numbers::invalid_material_id) const;

  /** The single-layer block of a panel pair and its contribution to the
   * double-layer right-hand side, in one traversal: the first dofs_per_cell
   * columns hold V(test, trial), the last one the right-hand side, without
   * the overall sign that assemble_double_layer_rhs() applies. The two
   * operators share the singular and near-singular subdivision and every
   * quadrature rule, which the separate assemblies build twice.
   */
  FullMatrix<double>
  integrate_single_and_double_layer(
    const CellIterator       &test_cell,
    const CellIterator       &trial_cell,
    const unsigned int        order,
    const Function<spacedim> &trial_function,
    const bool                regularize_trial_function) const;

  /** The pair integrated once for both orders of its cells: the result of
   * integrate_single_and_double_layer(), and in the column dofs_per_cell + 1
   * the right-hand side of the trial cell, row j for its shape function j,
   * that is the double layer with test and trial cell exchanged. The
   * single-layer block of the exchanged pair is the transpose of the first
   * one. A quadrature rule of the pair is a rule on the product of the two
   * cells, whatever their roles, and integrates the exchanged kernel, which
   * has the same singularity, as well: the pair costs one traversal instead
   * of two. Needs the direct evaluation of the geometry.
   */
  FullMatrix<double>
  integrate_single_and_double_layer_both(
    const CellIterator       &test_cell,
    const CellIterator       &trial_cell,
    const unsigned int        order,
    const Function<spacedim> &trial_function,
    const bool                regularize_trial_function) const;

  /** Assemble the single-layer matrix and the double-layer right-hand side
   * together. The result equals assemble_single_layer() followed by
   * assemble_double_layer_rhs() up to rounding. The caller owns
   * initialization and compression of both.
   */
  template <typename MatrixType, typename VectorType>
  void
  assemble_single_layer_and_double_layer_rhs(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &matrix,
    VectorType                                &rhs,
    const unsigned int                         order,
    const Function<spacedim>                  &trial_function,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const bool regularize_trial_function = false) const;

  template <typename MatrixType>
  void
  assemble_mass_matrix(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &matrix,
    const unsigned int                         order,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const types::material_id selected_material =
      numbers::invalid_material_id) const;

  template <typename MatrixType>
  void
  assemble_weak_hypersingular(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &matrix,
    const unsigned int                         order,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id,
    const types::material_id selected_test_material =
      numbers::invalid_material_id,
    const types::material_id selected_trial_material =
      numbers::invalid_material_id) const;

  /** Assemble the single layer V, the double layer K and the weak
   * hypersingular operator W of the Laplace kernel in one traversal of the
   * panel pairs. The far pairs share their points, the order chosen by the
   * far-field tolerance and the kernel block of 1/(4 pi r), which V and W
   * contract with the shape functions and with their surface curls, while K
   * contracts the block of the double-layer kernel; every contraction goes
   * one index at a time. The near and singular pairs integrate the three
   * from the same points, integrate_single_double_hypersingular(). Constraints
   * apply to test and trial functions; the caller owns initialization and
   * compression.
   */
  template <typename MatrixType>
  void
  assemble_single_double_hypersingular(
    const DoFHandler<spacedim - 1, spacedim> &dof_handler,
    const AffineConstraints<double>           &constraints,
    MatrixType                                &single_layer,
    MatrixType                                &double_layer,
    MatrixType                                &hypersingular,
    const unsigned int                         order,
    const types::subdomain_id locally_owned_subdomain =
      numbers::invalid_subdomain_id) const;

private:
  enum class OperatorType
  {
    single_layer,
    double_layer,
    adjoint_double_layer,
    weak_hypersingular,
    // The single layer and the double-layer right-hand side of the same pair,
    // integrated together: the local result holds the single-layer block in
    // the first dofs_per_cell columns and the right-hand side in the last.
    single_and_double_layer,
    // The same, and in one more column the right-hand side of the pair seen
    // the other way round, trial cell as test cell, from the same points.
    single_and_double_layer_both,
    // V, K and W of the Laplace kernel from the same points: the local
    // result holds V in the first dofs_per_cell columns, K in the next and
    // W in the last.
    single_double_hypersingular,
    // The same, and in a fourth block K of the pair the other way round,
    // transposed.
    single_double_hypersingular_both
  };

  /// Whether the operator is V, K and W together.
  static bool
  is_joint(const OperatorType operator_type)
  {
    return operator_type == OperatorType::single_double_hypersingular ||
           operator_type == OperatorType::single_double_hypersingular_both;
  }

  struct ReferencePatch
  {
    Point<2> lower;
    Point<2> upper;
  };

  struct CommonVertex
  {
    Point<spacedim> point;
    unsigned int    test_index;
    unsigned int    trial_index;
  };

  std::vector<CommonVertex>
  common_vertices(const CellIterator   &test_cell,
                  const ReferencePatch &test_patch,
                  const CellIterator   &trial_cell,
                  const ReferencePatch &trial_patch) const;

  /**
   * The contact of two cells along part of an edge: an edge of one of them,
   * the containing cell, holds an edge of the other, and the two edges do
   * not coincide, as across a hanging node. The ends of the shorter edge are
   * pulled back onto the mapped edge of the containing cell, which on the
   * exact geometry is curved and passes through the hanging vertex while
   * its chord does not; their reference coordinates along it bound the
   * contact. The test does not use the refinement levels: after anisotropic
   * refinement, cells of different levels may share a whole edge (a
   * conforming pair), and cells of one level only part of one.
   */
  struct EdgeContact
  {
    bool         found = false;
    /// Whether the test cell (true) or the trial cell holds the edge.
    bool         test_contains = true;
    /// The face of the containing cell, and the reference coordinates along
    /// it, increasing, of the ends of the edge of the other cell.
    unsigned int face = 0;
    double       begin = 0, end = 1;
  };

  EdgeContact
  nonconforming_contact(const CellIterator &test_cell,
                        const CellIterator &trial_cell) const;

  bool
  share_nonconforming_edge(const CellIterator &test_cell,
                           const CellIterator &trial_cell) const;

  /// The containing cell of a contact cut into strips across its edge at
  /// the ends of the contact: the middle strip shares a whole edge with the
  /// other cell, the others touch it at a vertex or not at all.
  std::vector<ReferencePatch>
  contact_strips(const EdgeContact &contact) const;

  /// For two patches with a common edge, the reference direction of each
  /// along that edge: the balancing of a singular pair halves both patches
  /// across the middle of the edge rather than one, which would leave half
  /// an edge shared, a pair that no rule treats.
  std::array<unsigned int, 2>
  common_edge_directions(const CellIterator   &test_cell,
                         const ReferencePatch &test_patch,
                         const CellIterator   &trial_cell,
                         const ReferencePatch &trial_patch) const;

  GalerkinPanelPairType
  classify(const CellIterator   &test_cell,
           const ReferencePatch &test_patch,
           const CellIterator   &trial_cell,
           const ReferencePatch &trial_patch) const;

  FullMatrix<double>
  integrate_operator(const CellIterator &test_cell,
                     const CellIterator &trial_cell,
                     const unsigned int  order,
                     const OperatorType  operator_type,
                     const Function<spacedim> *trial_function = nullptr,
                     const bool regularize_trial_function = false) const;

  FullMatrix<double>
  integrate(const CellIterator   &test_cell,
            const ReferencePatch &test_patch,
            const CellIterator   &trial_cell,
            const ReferencePatch &trial_patch,
            const unsigned int    order,
            const OperatorType    operator_type,
            const Function<spacedim> *trial_function = nullptr,
            const bool regularize_trial_function = false) const;

  /** Build the panel-pair quadrature: reference points on each cell and the
   * matching weights, already scaled by the patch jacobians. Shared by every
   * pair integral so they all inherit the same singular treatment.
   */
  /** Leaf of the energy-error integration: the contribution of one pair of
   * reference patches, with no further subdivision.
   */
  double
  error_energy_leaf(const CellIterator        &test_cell,
                    const ReferencePatch      &test_patch,
                    const CellIterator        &trial_cell,
                    const ReferencePatch      &trial_patch,
                    const unsigned int         order,
                    const Function<spacedim>  &exact_gradient,
                    const std::vector<double> &test_coefficients,
                    const std::vector<double> &trial_coefficients) const;

  /** Near-singular recursion for the energy-error integration, mirroring the
   * subdivision used by the operator integrals. Without it the integrand,
   * which is not pointwise positive because the error changes sign, is
   * mis-integrated badly enough to return a negative energy.
   */
  double
  error_energy_adaptive(const CellIterator        &test_cell,
                        const ReferencePatch      &test_patch,
                        const CellIterator        &trial_cell,
                        const ReferencePatch      &trial_patch,
                        const unsigned int         order,
                        const Function<spacedim>  &exact_gradient,
                        const std::vector<double> &test_coefficients,
                        const std::vector<double> &trial_coefficients,
                        const unsigned int         depth) const;

  void
  pair_quadrature(const CellIterator    &test_cell,
                  const ReferencePatch  &test_patch,
                  const CellIterator    &trial_cell,
                  const ReferencePatch  &trial_patch,
                  const unsigned int     order,
                  std::vector<Point<2>> &test_reference_points,
                  std::vector<Point<2>> &trial_reference_points,
                  std::vector<std::array<unsigned int, 2>> &pair_indices,
                  std::vector<double>   &pair_weights) const;

  void
  unit_pair_rule(const GalerkinPanelPairType      type,
                 const std::vector<CommonVertex> &shared,
                 const unsigned int               order,
                 std::vector<Point<2>>           &test_reference_points,
                 std::vector<Point<2>>           &trial_reference_points,
                 std::vector<double>             &pair_weights,
                 const bool                       half = false) const;

  /// The transform of a leaf on a patch (LightLeaf::transformed): row i,
  /// the values of shape i of the cell at the support points of the patch.
  void
  patch_transform(const ReferencePatch &patch,
                  std::vector<double>  &transform) const;

  std::vector<ReferencePatch>
  uniform_subpatches(const unsigned int refinement_levels) const;

  std::vector<ReferencePatch>
  split_patch(const ReferencePatch &patch) const;

  std::vector<Point<spacedim>>
  patch_sample_points(const CellIterator   &cell,
                      const ReferencePatch &patch) const;

  /** Per-cell geometry for the whole-cell patch, computed once and reused.
   * The near-singular test is asked about every ordered pair of cells, so
   * recomputing the mapped sample points there costs O(cells^2) mapping
   * evaluations; caching them, and rejecting distant pairs by their bounding
   * spheres first, turns the common case into a handful of arithmetic ops.
   */
  struct CellGeometry
  {
    std::vector<Point<spacedim>> samples;
    Point<spacedim>              centre;
    double                       radius;
    double                       diameter;
  };

  const CellGeometry &
  full_patch_geometry(const CellIterator &cell) const;

  bool
  is_full_patch(const ReferencePatch &patch) const;

  bool
  is_near_singular(const CellIterator   &test_cell,
                   const ReferencePatch &test_patch,
                   const CellIterator   &trial_cell,
                   const ReferencePatch &trial_patch) const;

  // Sauter-Schwab assumes the two touching panels are of comparable size.
  // On a graded mesh they are not, and the rule then loses several orders of
  // accuracy that no increase of its order recovers. These split the larger
  // panel until the touching pair is balanced; the subpatches that stop
  // touching are classified as regular by the patch-aware dispatch and
  // integrated as such. An elongated panel with itself is halved across its
  // long direction in both copies.
  // The two physical side lengths of a patch, in the order of the reference
  // directions. Sauter-Schwab degrades with the aspect ratio of the touching
  // panels, not with their diameter, so these are what the balancing test
  // must look at.
  std::array<double, 2>
  patch_extents(const CellIterator   &cell,
                const ReferencePatch &patch) const;

  // Halve a patch along one reference direction. Splitting only the long
  // direction keeps the recursion linear: at every level a single half still
  // touches the singularity and the other becomes an ordinary regular pair.
  std::array<ReferencePatch, 2>
  split_patch_direction(const ReferencePatch &patch,
                        const unsigned int    direction) const;

  FullMatrix<double>
  integrate_balanced_singular(const CellIterator   &test_cell,
                              const ReferencePatch &test_patch,
                              const CellIterator   &trial_cell,
                              const ReferencePatch &trial_patch,
                              const unsigned int    order,
                              const OperatorType    operator_type,
                              const Function<spacedim> *trial_function,
                              const bool regularize_trial_function,
                              const unsigned int depth) const;

  double
  error_energy_balanced(const CellIterator        &test_cell,
                        const ReferencePatch      &test_patch,
                        const CellIterator        &trial_cell,
                        const ReferencePatch      &trial_patch,
                        const unsigned int         order,
                        const Function<spacedim>  &exact_gradient,
                        const std::vector<double> &test_coefficients,
                        const std::vector<double> &trial_coefficients,
                        const unsigned int         depth) const;

  // A touching pair is integrated directly once neither panel is more than
  // this many times the other; the depth cap bounds the recursion for
  // pathological size ratios.
  static constexpr double       singular_balance_ratio = 2.0;
  /// The largest ratio of the depths of the two patches of a folded pair
  /// from their common edge.
  static constexpr double       folded_balance_ratio = 1.25;
  static constexpr unsigned int singular_balance_maximum_depth = 10;

  FullMatrix<double>
  integrate_adaptive_regular(const CellIterator   &test_cell,
                             const ReferencePatch &test_patch,
                             const CellIterator   &trial_cell,
                             const ReferencePatch &trial_patch,
                             const unsigned int    order,
                             const OperatorType    operator_type,
                             const Function<spacedim> *trial_function,
                             const bool regularize_trial_function,
                             const unsigned int depth) const;

  CrossLayerMatrices
  integrate_cross_layers(const CellIterator   &test_cell,
                         const ReferencePatch &test_patch,
                         const CellIterator   &trial_cell,
                         const ReferencePatch &trial_patch,
                         const unsigned int    order) const;

  CrossLayerMatrices
  integrate_adaptive_cross_layers(const CellIterator   &test_cell,
                                  const ReferencePatch &test_patch,
                                  const CellIterator   &trial_cell,
                                  const ReferencePatch &trial_patch,
                                  const unsigned int    order,
                                  const unsigned int    depth) const;

  Point<2>
  orient_from_canonical_vertex(const Point<2>   &point,
                               const unsigned int local_vertex) const;

  Point<2>
  orient_from_canonical_edge(const Point<2>   &point,
                             const unsigned int first_local_vertex,
                             const unsigned int second_local_vertex) const;

  Point<2>
  orient_to_canonical_edge(const Point<2>   &point,
                           const unsigned int first_local_vertex,
                           const unsigned int second_local_vertex) const;

  // Columns of a local result: the element basis for an operator, one for an
  // operator applied to a function, and the basis plus one for the fused
  // single layer and right-hand side.
  unsigned int
  n_result_columns(const OperatorType        operator_type,
                   const Function<spacedim> *trial_function) const
  {
    const unsigned int dpc  = finite_element.dofs_per_cell;
    const unsigned int flux = flux_function != nullptr ? 1 : 0;
    return operator_type == OperatorType::single_double_hypersingular ?
             3 * dpc :
           operator_type == OperatorType::single_double_hypersingular_both ?
             4 * dpc :
           operator_type == OperatorType::single_and_double_layer_both ?
             dpc + 2 + 2 * flux :
           operator_type == OperatorType::single_and_double_layer ?
             dpc + 1 + flux :
           trial_function != nullptr ? 1 :
                                       dpc;
  }

  /// Geometry of a cell for the direct evaluation: the mapping at the
  /// Gauss-Lobatto nodes of its degree, numbered lexicographically, and the
  /// sign that turns the cross product of the tangents into the normal of
  /// the library.
  struct LightCell
  {
    std::vector<Tensor<1, spacedim>> nodes;
    double                           orientation = 0.0;
  };

  /// Points, unit normals, surface elements and shape values at a set of
  /// reference points of one cell; the shape value of basis function i at
  /// point q is shapes[i * n + q].
  struct PointSet
  {
    unsigned int                     n = 0;
    std::vector<Point<spacedim>>     points;
    std::vector<Tensor<1, spacedim>> normals;
    std::vector<double>              jacobians;
    std::vector<double>              shapes;
    // With light_derivatives, the tangents of the map in the two reference
    // directions times the orientation of the cell: the surface curl of a
    // shape function is (d0 phi tangents1 - d1 phi tangents0) / jacobian.
    std::vector<Tensor<1, spacedim>> tangents0;
    std::vector<Tensor<1, spacedim>> tangents1;
  };

  /// A singular rule of the direct path, in the coordinates of its two
  /// patches: it depends only on the configuration of the pair, so its
  /// points, weights and, for whole cells, shape values are built once.
  struct LightRule
  {
    std::vector<Point<2>> test_points;
    std::vector<Point<2>> trial_points;
    /// The distinct points of each side and, for every pair of points, the
    /// index of its own among them, where they are fewer: on one side of a
    /// pair with a common vertex there are n^2 of the n^4 pairs of a
    /// region, on each side of one with a common edge n^3. Empty where
    /// the points are (nearly) all distinct.
    std::vector<Point<2>>     test_unique, trial_unique;
    std::vector<unsigned int> test_index, trial_index;
    /// For the rules of raised orders, which do not keep the shapes at all
    /// their points: the shapes and their derivatives at the distinct
    /// points, laid out as the values, gathered for each leaf.
    std::vector<double> test_unique_shapes, trial_unique_shapes;
    std::vector<double> test_unique_derivatives0, test_unique_derivatives1,
      trial_unique_derivatives0, trial_unique_derivatives1;
    std::vector<double>   weights;
    std::vector<double>   test_shapes;
    std::vector<double>   trial_shapes;
    // The derivatives of the shape functions in the two reference
    // directions, laid out as the values.
    std::vector<double>   test_derivatives0;
    std::vector<double>   test_derivatives1;
    std::vector<double>   trial_derivatives0;
    std::vector<double>   trial_derivatives1;
    /// The last lookup, for the eviction of the rules of raised orders.
    unsigned long long    last_use = 0;

    std::size_t
    memory_consumption() const;
  };

  const LightCell &
  light_cell(const CellIterator &cell) const;

  /// The mapping at a reference point: from the reconstructed geometry on
  /// the direct path, through the library otherwise. The subdivision asks
  /// for the corners and sample points of every patch it visits.
  Point<spacedim>
  map_to_real_cell(const CellIterator &cell, const Point<2> &reference) const;

  void
  evaluate_geometry(const CellIterator          &cell,
                    const std::vector<Point<2>> &reference_points,
                    PointSet                    &set) const;

  void
  evaluate_shape_derivatives(const std::vector<Point<2>> &reference_points,
                             std::vector<double>         &derivatives0,
                             std::vector<double>         &derivatives1) const;

  /// Set by integrate_light() for the operators that need the surface
  /// curls: evaluate_geometry() then keeps the tangents and
  /// prepare_light_leaf() the derivatives of the shape functions.
  mutable bool light_derivatives = false;
  /// Set by integrate_light() for the operators that it takes to the shapes
  /// of the cells itself: prepare_light_leaf() may then use the shapes of a
  /// kept rule on a patch (LightLeaf::transformed) and, with light_half, the
  /// half rule of a panel with itself (LightLeaf::half).
  mutable bool light_affine = false, light_half = false;

  void
  evaluate_shapes(const std::vector<Point<2>> &reference_points,
                  std::vector<double>         &shapes) const;

  /// The geometry of one side of a kept rule, placed on the patch, at all
  /// its points: evaluated at its distinct points (unique, with the index
  /// of every point among them) and gathered, or at all of them if unique
  /// is empty.
  void
  evaluate_rule_geometry(const CellIterator              &cell,
                         const ReferencePatch            &patch,
                         const std::vector<Point<2>>     &points,
                         const std::vector<Point<2>>     &unique,
                         const std::vector<unsigned int> &index,
                         PointSet                        &set) const;

  bool
  takes_light_path(const OperatorType        operator_type,
                   const Function<spacedim> *trial_function) const;

  /// A leaf of the direct path, ready for a kernel loop: the geometry of
  /// both sides in test_set and trial_set, and either a tensor rule, with
  /// per-side weights and the shapes of the point sets, or a paired rule,
  /// with its weights and shapes, cached or built for the patches here.
  struct LightLeaf
  {
    bool                       tensor = false;
    std::vector<double>        test_weights;
    std::vector<double>        trial_weights;
    const std::vector<double> *test_shapes  = nullptr;
    const std::vector<double> *trial_shapes = nullptr;
    const std::vector<double> *weights      = nullptr;
    std::vector<double>        own_test_shapes;
    std::vector<double>        own_trial_shapes;
    std::vector<double>        own_weights;
    // Derivatives of the shape functions, with light_derivatives.
    const std::vector<double> *test_derivatives0  = nullptr;
    const std::vector<double> *test_derivatives1  = nullptr;
    const std::vector<double> *trial_derivatives0 = nullptr;
    const std::vector<double> *trial_derivatives1 = nullptr;
    std::vector<double>        own_derivatives[4];
    /// On patches smaller than the cells, with a rule kept with its shapes:
    /// the shapes and derivatives are those of the rule, in the coordinates
    /// of the patches, and the result goes over to the shapes of the cells
    /// by the transforms, row i the values of shape i of the cell at the
    /// support points of the patch (Q_p is invariant under the affine map
    /// of a patch); the derivative in direction d is scaled by the inverse
    /// extent of the patch in d and the weights by weight_scale, the
    /// product of the areas of the patches.
    bool                  transformed = false;
    std::vector<double>   test_transform, trial_transform;
    std::array<double, 2> test_derivative_scale  = {{1., 1.}};
    std::array<double, 2> trial_derivative_scale = {{1., 1.}};
    double                weight_scale           = 1.;
    /// A panel with itself on half the regions of its rule, the others being
    /// their images under the exchange of the two points: the result is
    /// completed by symmetry (joint operators in both directions).
    bool half = false;

    /// Forget the rule and the pointers, keep the capacity of the arrays.
    void
    reset()
    {
      tensor             = false;
      test_shapes        = nullptr;
      trial_shapes       = nullptr;
      weights            = nullptr;
      test_derivatives0  = nullptr;
      test_derivatives1  = nullptr;
      trial_derivatives0 = nullptr;
      trial_derivatives1 = nullptr;
      transformed        = false;
      half               = false;
      test_derivative_scale  = {{1., 1.}};
      trial_derivative_scale = {{1., 1.}};
      weight_scale           = 1.;
    }
  };
  /// The leaf of the direct path, kept from pair to pair so that its arrays
  /// keep their memory; one leaf is in use at a time.
  mutable LightLeaf light_leaf;

  /// W, or V, K and W, of a leaf of the direct path, whose point sets carry
  /// tangents and whose leaf carries the derivatives of the shape functions.
  FullMatrix<double>
  integrate_light_hypersingular(const LightLeaf   &leaf,
                                const OperatorType operator_type) const;

  /// The result of integrate_light_hypersingular() for the shapes of the
  /// cells (LightLeaf::transformed) and the whole rule (LightLeaf::half,
  /// from the four blocks of the joint operators in both directions to the
  /// first n_columns of them).
  FullMatrix<double>
  complete_light_leaf(const LightLeaf    &leaf,
                      FullMatrix<double> &&result,
                      const unsigned int   n_columns) const;

  void
  prepare_light_leaf(const CellIterator   &test_cell,
                     const ReferencePatch &test_patch,
                     const CellIterator   &trial_cell,
                     const ReferencePatch &trial_patch,
                     const unsigned int    order,
                     LightLeaf            &leaf) const;

  double
  error_energy_light(const CellIterator        &test_cell,
                     const ReferencePatch      &test_patch,
                     const CellIterator        &trial_cell,
                     const ReferencePatch      &trial_patch,
                     const unsigned int         order,
                     const Function<spacedim>  &exact_gradient,
                     const std::vector<double> &test_coefficients,
                     const std::vector<double> &trial_coefficients) const;

  FullMatrix<double>
  integrate_light(const CellIterator       &test_cell,
                  const ReferencePatch     &test_patch,
                  const CellIterator       &trial_cell,
                  const ReferencePatch     &trial_patch,
                  const unsigned int        order,
                  const OperatorType        operator_type,
                  const Function<spacedim> *trial_function,
                  const bool                regularize_trial_function) const;

  // The kernel loops of the direct path. A tensor rule pairs every test
  // point with every trial point, with product weights, and is contracted
  // one index at a time; a singular rule pairs point q with point q.
  FullMatrix<double>
  integrate_tensor_sets(const PointSet            &test,
                        const std::vector<double> &test_weights,
                        const PointSet            &trial,
                        const std::vector<double> &trial_weights,
                        const OperatorType         operator_type,
                        const Function<spacedim>  *trial_function,
                        const bool regularize_trial_function) const;

  FullMatrix<double>
  integrate_paired_sets(const PointSet            &test,
                        const std::vector<double> &test_shapes,
                        const PointSet            &trial,
                        const std::vector<double> &trial_shapes,
                        const std::vector<double> &weights,
                        const OperatorType         operator_type,
                        const Function<spacedim>  *trial_function,
                        const bool regularize_trial_function) const;

  bool                             light_geometry = false;
  std::vector<double>              geometry_nodes;
  std::vector<double>              shape_nodes;
  /// Position (a, b) of the support point of each basis function in the
  /// tensor grid of shape_nodes: the function is L_a(x) L_b(y).
  std::vector<std::array<unsigned int, 2>> shape_tensor_index;
  /// The Lagrange bases through geometry_nodes and shape_nodes: nodes and
  /// weights of the product form.
  std::vector<double>              geometry_coefficients;
  std::vector<double>              shape_coefficients;
  mutable std::vector<LightCell>   light_cells;
  mutable std::vector<bool>        light_cell_valid;
  mutable double                   maximum_light_deviation = 0.0;
  mutable std::map<std::array<int, 7>, LightRule> light_rule_cache;
  /// The rules of orders above singular_order (folded, material and skewed
  /// pairs) serve few pairs but are large, 6 n^4 points for a common edge,
  /// in up to 32 configurations: they stay cached while their total stays
  /// below this size, the least recently used evicted first.
  static constexpr std::size_t maximum_raised_rule_bytes = std::size_t(128)
                                                           << 20;
  mutable std::size_t        raised_rule_bytes = 0;
  mutable unsigned long long light_rule_clock  = 0;
  // Point sets of the direct path, reused from pair to pair, the FEValues
  // that checks the reconstructed geometry, and the tensor Gauss rules.
  mutable PointSet                                test_set;
  mutable PointSet                                trial_set;
  /// The geometry at the distinct points of a side of a rule, before it is
  /// gathered to all its points (evaluate_rule_geometry()).
  mutable PointSet                                unique_set;
  mutable std::vector<Point<2>>                   unique_points;
  /// Work arrays of evaluate_geometry(): basis values and derivatives in
  /// the two directions, node by node, and the coordinates and tangents,
  /// component by component, each over all points.
  mutable std::vector<double> geometry_work;
  /// Work arrays of the kernel loops of the joint operators, kept from pair
  /// to pair: allocated anew, arrays of megabytes cost a page fault per
  /// page on every pair. The surface curls of both sides, the kernel weights
  /// and the weighted test shapes of a block of points.
  mutable std::vector<double> kernel_curls_x, kernel_curls_y,
    kernel_weights, kernel_block;
  /// The points of the subpatches of a leaf, kept for the same reason.
  mutable std::vector<Point<2>> leaf_test_points, leaf_trial_points;
  mutable std::unique_ptr<FEValues<2, spacedim>>  light_check_values;
  /// On a mapping onto the exact geometry (ExactGeometry::ProjectedMapping),
  /// the direct path reconstructs the cells of the mesh mapping and projects
  /// every point with this evaluator.
  const ExactGeometry::ProjectedMapping<2, spacedim> *projected_mapping =
    nullptr;
  mutable std::unique_ptr<
    typename ExactGeometry::ProjectedMapping<2, spacedim>::Evaluator>
    exact_evaluator;
  mutable std::map<unsigned int, QGauss<2>>       gauss_rules;

  const Mapping<spacedim - 1, spacedim>       &mapping;
  const FiniteElement<spacedim - 1, spacedim> &finite_element;
  const GalerkinSingularQuadrature             singular_quadrature;
  const unsigned int                           singular_order;
  unsigned int folded_singular_order  = 0;
  double       folded_singular_cosine = -0.5;
  std::set<types::material_id> singular_material_ids;
  unsigned int material_singular_order = 0;
  /// The order for pairs with a skewed cell, and the angle below which a
  /// cell is skewed (set_skewed_singular_order()).
  unsigned int skewed_singular_order = 0;
  double       skewed_minimum_angle  = 0.;
  /// The order for kinked pairs, and the cosine of the angle of their
  /// normals above which a pair is kinked (set_kinked_singular_order()).
  unsigned int kinked_singular_order  = 0;
  double       kinked_singular_cosine = 0.5;
  /// The near-singular distance ratio of kinked pairs
  /// (set_kinked_distance_ratio()).
  double kinked_distance_ratio = 0.;
  /// The highest order of the tensor rule of a kinked pair near only by its
  /// own ratio (integrate_adaptive_regular()).
  static constexpr unsigned int maximum_kinked_regular_order = 12;

  /// The order of the singular rule for a pair (set_folded_singular_order()).
  unsigned int
  singular_order_for(const CellIterator &test_cell,
                     const CellIterator &trial_cell,
                     const unsigned int  order) const;

  /// Whether two cells meet with nearly opposite normals, the cosine of
  /// their normals below folded_singular_cosine.
  bool
  is_folded_pair(const CellIterator &test_cell,
                 const CellIterator &trial_cell) const;

  /// The cosine of the angle of the normals of two cells, from their
  /// vertices and oriented as the cells.
  double
  normal_cosine(const CellIterator &test_cell,
                const CellIterator &trial_cell) const;

  /// The smallest distance of the sample points of two patches over the
  /// larger of their diameters.
  double
  separation_ratio(const CellIterator   &test_cell,
                   const ReferencePatch &test_patch,
                   const CellIterator   &trial_cell,
                   const ReferencePatch &trial_patch) const;
  const unsigned int                           duffy_outer_subdivisions;
  const double                                 near_singular_distance_ratio;
  const unsigned int                           near_singular_maximum_depth;
  const unsigned int                           near_singular_quadrature_order;
  mutable unsigned int                         near_singular_pairs;
  mutable unsigned int                         near_singular_leaves;
  /** A panel-pair quadrature rule together with the FEValues that evaluate
   * it. The reference points of a singular pair depend only on the pair
   * type, the order and which local vertices are shared - never on the cells
   * - so the rule and the shape-function data built from it are the same for
   * every pair in that configuration, and only reinit() has to be repeated.
   */
  FullMatrix<double>
  integrate_pairs(const FEValues<2, spacedim>                    &test_values,
                  const FEValues<2, spacedim>                    &trial_values,
                  const std::vector<std::array<unsigned int, 2>> &pair_indices,
                  const std::vector<double>                      &pair_weights,
                  const OperatorType                              operator_type,
                  const Function<spacedim>                       *trial_function,
                  const bool regularize_trial_function) const;

  struct CachedPairRule
  {
    std::vector<std::array<unsigned int, 2>>   indices;
    std::vector<double>                        weights;
    std::unique_ptr<FEValues<2, spacedim>>     test_values;
    std::unique_ptr<FEValues<2, spacedim>>     trial_values;
  };

  // Keyed by {pair type, order, gradients wanted, shared local vertices}.
  // There are only a few dozen distinct configurations on a mesh, but the
  // cap keeps a pathological one from growing without bound.
  static constexpr std::size_t maximum_cached_pair_rules = 48;
  mutable std::map<std::array<int, 7>, CachedPairRule> pair_rule_cache;

  double                                      far_field_tolerance = 0.0;
  double                                      screening           = 0.0;
  const Function<spacedim>                   *flux_function       = nullptr;
  mutable RhsProfile                          profile;
  std::function<void(const CellIterator &,
                     std::vector<types::global_dof_index> &)>
    dof_index_map;

  void
  cell_dof_indices(const CellIterator                   &cell,
                   std::vector<types::global_dof_index> &indices) const
  {
    if (dof_index_map)
      dof_index_map(cell, indices);
    else
      cell->get_dof_indices(indices);
  }

  /// exp(-kappa r) / (4 pi r).
  double
  single_layer_kernel(const double distance) const
  {
    return (screening == 0.0 ? 1.0 : std::exp(-screening * distance)) /
           (4.0 * numbers::PI * distance);
  }

  /// E(r) = (1 + kappa r) exp(-kappa r): the normal derivatives of the
  /// screened kernel are those of the Laplace kernel times E.
  double
  screening_factor(const double distance) const
  {
    return screening == 0.0 ? 1.0 :
                              (1.0 + screening * distance) *
                                std::exp(-screening * distance);
  }
  // How many far-field pairs each order ended up serving; a diagnostic for
  // whether the reduction is worth its complexity on a given mesh.
  mutable std::array<unsigned long long, 32>  far_field_order_histogram{};
  mutable std::vector<CellGeometry>            cell_geometry_cache;
  mutable std::vector<bool>                    cell_geometry_valid;
};


template <int spacedim>
template <typename MatrixType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_single_layer(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &matrix,
  const unsigned int                         order,
  const types::subdomain_id locally_owned_subdomain,
  const types::material_id selected_test_material,
  const types::material_id selected_trial_material) const
{
  AssertDimension(dof_handler.get_fe().dofs_per_cell,
                  finite_element.dofs_per_cell);

  struct CellQuadratureData
  {
    CellIterator                              cell;
    std::vector<Point<spacedim>>              points;
    std::vector<double>                       jxw;
    std::vector<std::vector<double>>          shape_values;
  };

  // Per-cell quadrature data for every order the far field may select. With
  // the reduction switched off only the requested order is built.
  const unsigned int lowest_order =
    far_field_tolerance > 0.0 ? std::min(2u, order) : order;
  std::vector<std::vector<CellQuadratureData>> data_by_order(order + 1);
  for (unsigned int o = lowest_order; o <= order; ++o)
    {
      const QGauss<spacedim - 1>       rule(o);
      FEValues<spacedim - 1, spacedim> values(mapping,
                                              finite_element,
                                              rule,
                                              update_values |
                                                update_quadrature_points |
                                                update_JxW_values);
      auto &set = data_by_order[o];
      set.reserve(dof_handler.get_triangulation().n_active_cells());
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          values.reinit(cell);
          CellQuadratureData data;
          data.cell   = cell;
          data.points = values.get_quadrature_points();
          data.jxw.resize(rule.size());
          data.shape_values.assign(finite_element.dofs_per_cell,
                                   std::vector<double>(rule.size()));
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              data.jxw[q] = values.JxW(q);
              for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
                data.shape_values[i][q] = values.shape_value(i, q);
            }
          set.push_back(std::move(data));
        }
    }
  const QGauss<spacedim - 1>            quadrature(order);
  const std::vector<CellQuadratureData> &cell_data = data_by_order[order];

  // Smallest order whose convergence factor meets the tolerance.
  const auto far_field_order = [&](const CellIterator &test_cell,
                                   const CellIterator &trial_cell) {
    if (far_field_tolerance <= 0.0)
      return order;
    const CellGeometry &g1 = full_patch_geometry(test_cell);
    const CellGeometry &g2 = full_patch_geometry(trial_cell);
    const double        separation = g1.centre.distance(g2.centre);
    if (separation <= 0.0)
      return order;
    const double rho = (g1.radius + g2.radius) / separation;
    if (rho >= 1.0 || rho <= 0.0)
      return order;
    const unsigned int needed = static_cast<unsigned int>(
      std::ceil(std::log(far_field_tolerance) / (2.0 * std::log(rho))));
    return std::min(order, std::max(lowest_order, needed));
  };

  std::vector<types::global_dof_index> test_indices(
    finite_element.dofs_per_cell);
  std::vector<types::global_dof_index> trial_indices(
    finite_element.dofs_per_cell);

  // The far-field contribution of a panel pair is a triple product,
  //     M_ij = sum_qx sum_qy  K(qx,qy) u_i(qx) v_j(qy),
  // and four nested loops evaluate it in n_q^2 * dofs^2 operations.
  // Contracting one index at a time costs n_q^2 * dofs + n_q * dofs^2
  // instead - about sevenfold fewer at degree two with a sixth-order rule.
  const unsigned int  n_q = quadrature.size();
  const unsigned int  dpc = finite_element.dofs_per_cell;
  std::vector<double> kernel_block(n_q * n_q);
  std::vector<double> partial(dpc * n_q);

  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;
      if (selected_test_material != numbers::invalid_material_id &&
          test_cell->material_id() != selected_test_material)
        continue;

      cell_dof_indices(test_cell, test_indices);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          cell_dof_indices(trial_cell, trial_indices);
          FullMatrix<double> local_matrix(finite_element.dofs_per_cell,
                                          finite_element.dofs_per_cell);
          const auto pair_start = std::chrono::steady_clock::now();
          bool       far_pair   = false;
          if (classify(test_cell, trial_cell) ==
                GalerkinPanelPairType::regular &&
              !is_near_singular(test_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                                trial_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)}))
            {
              far_pair = true;
              const unsigned int pair_order =
                far_field_order(test_cell, trial_cell);
              ++far_field_order_histogram[pair_order];
              const std::size_t  index = &test_data - cell_data.data();
              const CellQuadratureData &t_data =
                data_by_order[pair_order][index];
              const CellQuadratureData &r_data =
                data_by_order[pair_order][&trial_data - cell_data.data()];
              const unsigned int m = t_data.jxw.size();
              for (unsigned int qx = 0; qx < m; ++qx)
                for (unsigned int qy = 0; qy < m; ++qy)
                  kernel_block[qx * m + qy] =
                    t_data.jxw[qx] * r_data.jxw[qy] *
                    single_layer_kernel(
                      t_data.points[qx].distance(r_data.points[qy]));
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  const std::vector<double> &u = t_data.shape_values[i];
                  for (unsigned int qy = 0; qy < m; ++qy)
                    {
                      double acc = 0.0;
                      for (unsigned int qx = 0; qx < m; ++qx)
                        acc += u[qx] * kernel_block[qx * m + qy];
                      partial[i * m + qy] = acc;
                    }
                }
              for (unsigned int i = 0; i < dpc; ++i)
                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    const std::vector<double> &v = r_data.shape_values[j];
                    double acc = 0.0;
                    for (unsigned int qy = 0; qy < m; ++qy)
                      acc += partial[i * m + qy] * v[qy];
                    local_matrix(i, j) += acc;
                  }
            }
          else
            local_matrix = integrate_single_layer(test_cell,
                                                  trial_cell,
                                                  order);
          (far_pair ? profile.far_seconds : profile.near_seconds) +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          pair_start)
              .count();
          if (constraints.n_constraints() == 0)
            matrix.add(test_indices, trial_indices, local_matrix);
          else
            constraints.distribute_local_to_global(local_matrix,
                                                   test_indices,
                                                   trial_indices,
                                                   matrix);
        }
    }
}


template <int spacedim>
template <typename VectorType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_single_layer_flux_rhs(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  VectorType                                &rhs,
  const unsigned int                         order,
  const Function<spacedim>                  &gradient,
  const types::subdomain_id locally_owned_subdomain) const
{
  AssertDimension(gradient.n_components, spacedim);
  AssertDimension(dof_handler.get_fe().dofs_per_cell,
                  finite_element.dofs_per_cell);

  struct CellQuadratureData
  {
    CellIterator                     cell;
    std::vector<Point<spacedim>>     points;
    std::vector<double>              jxw;
    std::vector<double>              flux;
    std::vector<std::vector<double>> shape_values;
  };

  const QGauss<spacedim - 1>       quadrature(order);
  FEValues<spacedim - 1, spacedim> values(mapping,
                                          finite_element,
                                          quadrature,
                                          update_values |
                                            update_normal_vectors |
                                            update_quadrature_points |
                                            update_JxW_values);
  std::vector<CellQuadratureData> cell_data;
  cell_data.reserve(dof_handler.get_triangulation().n_active_cells());
  Vector<double> gradient_value(spacedim);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      CellQuadratureData data;
      data.cell   = cell;
      data.points = values.get_quadrature_points();
      data.jxw.resize(quadrature.size());
      data.flux.resize(quadrature.size());
      data.shape_values.assign(finite_element.dofs_per_cell,
                               std::vector<double>(quadrature.size()));
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          data.jxw[q] = values.JxW(q);
          gradient.vector_value(data.points[q], gradient_value);
          double trace = 0.0;
          for (unsigned int d = 0; d < spacedim; ++d)
            trace += gradient_value[d] * values.normal_vector(q)[d];
          data.flux[q] = trace;
          for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
            data.shape_values[i][q] = values.shape_value(i, q);
        }
      cell_data.push_back(std::move(data));
    }

  std::vector<types::global_dof_index> test_indices(
    finite_element.dofs_per_cell);

  // Same contraction as in the single-layer assembly: the shape function
  // index is summed last, so the quadrature pair loop runs once instead of
  // once per local degree of freedom.
  const unsigned int  n_q = quadrature.size();
  const unsigned int  dpc = finite_element.dofs_per_cell;
  std::vector<double> weighted(n_q);
  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;

      Vector<double> local_rhs(finite_element.dofs_per_cell);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          if (classify(test_cell, trial_cell) ==
                GalerkinPanelPairType::regular &&
              !is_near_singular(test_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                                trial_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)}))
            {
              for (unsigned int qx = 0; qx < n_q; ++qx)
                {
                  double acc = 0.0;
                  for (unsigned int qy = 0; qy < n_q; ++qy)
                    acc += trial_data.jxw[qy] * trial_data.flux[qy] *
                           single_layer_kernel(test_data.points[qx].distance(
                             trial_data.points[qy]));
                  weighted[qx] = acc * test_data.jxw[qx];
                }
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  const std::vector<double> &u = test_data.shape_values[i];
                  double acc = 0.0;
                  for (unsigned int qx = 0; qx < n_q; ++qx)
                    acc += u[qx] * weighted[qx];
                  local_rhs[i] += acc;
                }
            }
          else
            local_rhs += integrate_single_layer_flux(test_cell, trial_cell,
                                                     order, gradient);
        }
      test_cell->get_dof_indices(test_indices);
      constraints.distribute_local_to_global(local_rhs, test_indices, rhs);
    }
}


template <int spacedim>
template <typename VectorType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_double_layer_rhs(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  VectorType                                &rhs,
  const unsigned int                         order,
  const Function<spacedim>                  &trial_function,
  const types::subdomain_id locally_owned_subdomain,
  const bool                regularize_trial_function,
  const std::vector<unsigned int> *trial_cell_components,
  const unsigned int selected_trial_component,
  const types::material_id selected_test_material) const
{
  struct CellQuadratureData
  {
    CellIterator                              cell;
    std::vector<Point<spacedim>>              points;
    std::vector<Tensor<1, spacedim>>          normals;
    std::vector<double>                       jxw;
    std::vector<std::vector<double>>          shape_values;
    std::vector<double>                       function_values;
    unsigned int component = numbers::invalid_unsigned_int;
  };

  const QGauss<spacedim - 1> quadrature(order);
  FEValues<spacedim - 1, spacedim> values(mapping,
                                           finite_element,
                                           quadrature,
                                           update_values |
                                             update_normal_vectors |
                                             update_quadrature_points |
                                             update_JxW_values);
  std::vector<CellQuadratureData> cell_data;
  cell_data.reserve(dof_handler.get_triangulation().n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      CellQuadratureData data;
      data.cell = cell;
      if (trial_cell_components != nullptr)
        {
          AssertIndexRange(cell->active_cell_index(),
                           trial_cell_components->size());
          data.component =
            (*trial_cell_components)[cell->active_cell_index()];
        }
      data.points = values.get_quadrature_points();
      data.normals = values.get_normal_vectors();
      data.jxw.resize(quadrature.size());
      data.function_values.resize(quadrature.size());
      data.shape_values.assign(finite_element.dofs_per_cell,
                               std::vector<double>(quadrature.size()));
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          data.jxw[q] = values.JxW(q);
          data.function_values[q] = trial_function.value(data.points[q]);
          for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
            data.shape_values[i][q] = values.shape_value(i, q);
        }
      cell_data.push_back(std::move(data));
    }

  std::vector<types::global_dof_index> test_indices(
    finite_element.dofs_per_cell);

  // Same contraction as in the single-layer assembly: the shape function
  // index is summed last, so the quadrature pair loop runs once instead of
  // once per local degree of freedom.
  const unsigned int  n_q = quadrature.size();
  const unsigned int  dpc = finite_element.dofs_per_cell;
  std::vector<double> weighted(n_q);
  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;
      if (selected_test_material != numbers::invalid_material_id &&
          test_cell->material_id() != selected_test_material)
        continue;

      Vector<double> local_rhs(finite_element.dofs_per_cell);
      for (const auto &trial_data : cell_data)
        {
          if (selected_trial_component != numbers::invalid_unsigned_int &&
              trial_data.component != selected_trial_component)
            continue;

          const auto &trial_cell = trial_data.cell;
          const auto pair_start = std::chrono::steady_clock::now();
          bool       far_pair   = false;
          if (classify(test_cell, trial_cell) ==
                GalerkinPanelPairType::regular &&
              !is_near_singular(test_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                                trial_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)}))
            {
              far_pair = true;
              for (unsigned int qx = 0; qx < n_q; ++qx)
                {
                  double sum_kernel   = 0.0;
                  double sum_weighted = 0.0;
                  for (unsigned int qy = 0; qy < n_q; ++qy)
                    {
                      const Tensor<1, spacedim> displacement =
                        trial_data.points[qy] - test_data.points[qx];
                      const double distance = displacement.norm();
                      const double factor =
                        -(displacement * trial_data.normals[qy]) *
                        test_data.jxw[qx] * trial_data.jxw[qy] /
                        (4.0 * numbers::PI * distance * distance * distance);
                      // (1/2 I + K_kappa) phi (x) is the integral of the
                      // Laplace kernel against E(r) phi(y) - phi(x): the
                      // Laplace part carries the solid-angle identity, and
                      // the screened remainder is bounded.
                      sum_kernel += factor;
                      sum_weighted += factor * screening_factor(distance) *
                                      trial_data.function_values[qy];
                    }
                  weighted[qx] = sum_weighted -
                                 (regularize_trial_function ?
                                    sum_kernel * test_data.function_values[qx] :
                                    0.0);
                }
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  const std::vector<double> &u = test_data.shape_values[i];
                  double acc = 0.0;
                  for (unsigned int qx = 0; qx < n_q; ++qx)
                    acc += u[qx] * weighted[qx];
                  local_rhs[i] += acc;
                }
            }
          else
            local_rhs += integrate_double_layer(test_cell,
                                                trial_cell,
                                                order,
                                                trial_function,
                                                regularize_trial_function);
          const double pair_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          pair_start)
              .count();
          (far_pair ? profile.far_seconds : profile.near_seconds) +=
            pair_seconds;
        }

      local_rhs *= -1.0;
      test_cell->get_dof_indices(test_indices);
      constraints.distribute_local_to_global(local_rhs, test_indices, rhs);
    }
}


template <int spacedim>
template <typename MatrixType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_double_layer(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &matrix,
  const unsigned int                         order,
  const types::subdomain_id locally_owned_subdomain,
  const types::material_id selected_test_material,
  const types::material_id selected_trial_material) const
{
  AssertDimension(dof_handler.get_fe().dofs_per_cell,
                  finite_element.dofs_per_cell);

  std::vector<types::global_dof_index> test_indices(
    finite_element.dofs_per_cell);
  std::vector<types::global_dof_index> trial_indices(
    finite_element.dofs_per_cell);

  for (const auto &test_cell : dof_handler.active_cell_iterators())
    {
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;
      if (selected_test_material != numbers::invalid_material_id &&
          test_cell->material_id() != selected_test_material)
        continue;

      cell_dof_indices(test_cell, test_indices);
      for (const auto &trial_cell : dof_handler.active_cell_iterators())
        {
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          cell_dof_indices(trial_cell, trial_indices);
          const FullMatrix<double> local_matrix =
            integrate_double_layer(test_cell, trial_cell, order);
          constraints.distribute_local_to_global(local_matrix,
                                                 test_indices,
                                                 trial_indices,
                                                 matrix);
        }
    }
}


template <int spacedim>
template <typename MatrixType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_adjoint_double_layer(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &matrix,
  const unsigned int                         order,
  const types::subdomain_id locally_owned_subdomain,
  const types::material_id selected_test_material,
  const types::material_id selected_trial_material) const
{
  AssertDimension(dof_handler.get_fe().dofs_per_cell,
                  finite_element.dofs_per_cell);

  struct CellQuadratureData
  {
    CellIterator                     cell;
    std::vector<Point<spacedim>>     points;
    std::vector<Tensor<1, spacedim>> normals;
    std::vector<double>              jxw;
    std::vector<std::vector<double>> shape_values;
  };

  const QGauss<spacedim - 1> quadrature(order);
  FEValues<spacedim - 1, spacedim> values(mapping,
                                           finite_element,
                                           quadrature,
                                           update_values |
                                             update_normal_vectors |
                                             update_quadrature_points |
                                             update_JxW_values);
  std::vector<CellQuadratureData> cell_data;
  cell_data.reserve(dof_handler.get_triangulation().n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      CellQuadratureData data;
      data.cell    = cell;
      data.points  = values.get_quadrature_points();
      data.normals = values.get_normal_vectors();
      data.jxw.resize(quadrature.size());
      data.shape_values.assign(finite_element.dofs_per_cell,
                               std::vector<double>(quadrature.size()));
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          data.jxw[q] = values.JxW(q);
          for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
            data.shape_values[i][q] = values.shape_value(i, q);
        }
      cell_data.push_back(std::move(data));
    }

  std::vector<types::global_dof_index> test_indices(
    finite_element.dofs_per_cell);
  std::vector<types::global_dof_index> trial_indices(
    finite_element.dofs_per_cell);
  FullMatrix<double> local_adjoint(finite_element.dofs_per_cell,
                                   finite_element.dofs_per_cell);

  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;
      if (selected_test_material != numbers::invalid_material_id &&
          test_cell->material_id() != selected_test_material)
        continue;

      test_cell->get_dof_indices(test_indices);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          trial_cell->get_dof_indices(trial_indices);
          local_adjoint = 0.0;
          if (classify(test_cell, trial_cell) ==
                GalerkinPanelPairType::regular &&
              !is_near_singular(test_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                                trial_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)}))
            for (unsigned int qx = 0; qx < quadrature.size(); ++qx)
              for (unsigned int qy = 0; qy < quadrature.size(); ++qy)
                {
                  const Tensor<1, spacedim> displacement =
                    trial_data.points[qy] - test_data.points[qx];
                  const double distance = displacement.norm();
                  const double factor =
                    (displacement * test_data.normals[qx]) *
                    test_data.jxw[qx] * trial_data.jxw[qy] *
                    screening_factor(distance) /
                    (4.0 * numbers::PI * distance * distance * distance);
                  for (unsigned int i = 0;
                       i < finite_element.dofs_per_cell;
                       ++i)
                    for (unsigned int j = 0;
                         j < finite_element.dofs_per_cell;
                         ++j)
                      local_adjoint(i, j) +=
                        test_data.shape_values[i][qx] *
                        trial_data.shape_values[j][qy] * factor;
                }
          else
            {
              const FullMatrix<double> transposed_pair =
                integrate_double_layer(trial_cell, test_cell, order);
              for (unsigned int i = 0; i < local_adjoint.m(); ++i)
                for (unsigned int j = 0; j < local_adjoint.n(); ++j)
                  local_adjoint(i, j) = transposed_pair(j, i);
            }
          constraints.distribute_local_to_global(local_adjoint,
                                                 test_indices,
                                                 trial_indices,
                                                 matrix);
        }
    }
}


template <int spacedim>
template <typename MatrixType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_mass_matrix(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &matrix,
  const unsigned int                         order,
  const types::subdomain_id locally_owned_subdomain,
  const types::material_id selected_material) const
{
  const QGauss<spacedim - 1> quadrature(order);
  FEValues<spacedim - 1, spacedim> values(mapping,
                                           finite_element,
                                           quadrature,
                                           update_values |
                                             update_JxW_values);
  FullMatrix<double> local_matrix(finite_element.dofs_per_cell,
                                  finite_element.dofs_per_cell);
  std::vector<types::global_dof_index> local_indices(
    finite_element.dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          cell->subdomain_id() != locally_owned_subdomain)
        continue;
      if (selected_material != numbers::invalid_material_id &&
          cell->material_id() != selected_material)
        continue;

      values.reinit(cell);
      local_matrix = 0.0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < finite_element.dofs_per_cell; ++j)
            local_matrix(i, j) += values.shape_value(i, q) *
                                  values.shape_value(j, q) * values.JxW(q);
      cell->get_dof_indices(local_indices);
      constraints.distribute_local_to_global(local_matrix,
                                             local_indices,
                                             local_indices,
                                             matrix);
    }
}


template <int spacedim>
template <typename MatrixType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_weak_hypersingular(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &matrix,
  const unsigned int                         order,
  const types::subdomain_id locally_owned_subdomain,
  const types::material_id selected_test_material,
  const types::material_id selected_trial_material) const
{
  // The screened hypersingular operator has a further term,
  // kappa^2 (n_x . n_y) psi(x) phi(y) G, which this assembly does not have.
  AssertThrow(screening == 0.0,
              ExcMessage("The weak hypersingular operator is implemented for "
                         "the Laplace kernel only."));
  struct CellQuadratureData
  {
    CellIterator cell;
    std::vector<Point<spacedim>> points;
    std::vector<double>          jxw;
    std::vector<std::vector<Tensor<1, spacedim>>> surface_curls;
  };

  const QGauss<spacedim - 1> quadrature(order);
  FEValues<spacedim - 1, spacedim> values(mapping,
                                           finite_element,
                                           quadrature,
                                           update_gradients |
                                             update_normal_vectors |
                                             update_quadrature_points |
                                             update_JxW_values);
  std::vector<CellQuadratureData> cell_data;
  cell_data.reserve(dof_handler.get_triangulation().n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      CellQuadratureData data;
      data.cell   = cell;
      data.points = values.get_quadrature_points();
      data.jxw.resize(quadrature.size());
      data.surface_curls.assign(
        finite_element.dofs_per_cell,
        std::vector<Tensor<1, spacedim>>(quadrature.size()));
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          data.jxw[q] = values.JxW(q);
          for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
            data.surface_curls[i][q] = cross_product_3d(
              values.normal_vector(q), values.shape_grad(i, q));
        }
      cell_data.push_back(std::move(data));
    }

  std::vector<types::global_dof_index> test_indices(
    finite_element.dofs_per_cell);
  std::vector<types::global_dof_index> trial_indices(
    finite_element.dofs_per_cell);

  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;
      if (selected_test_material != numbers::invalid_material_id &&
          test_cell->material_id() != selected_test_material)
        continue;

      cell_dof_indices(test_cell, test_indices);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          cell_dof_indices(trial_cell, trial_indices);
          FullMatrix<double> local_matrix(finite_element.dofs_per_cell,
                                          finite_element.dofs_per_cell);
          if (classify(test_cell, trial_cell) ==
                GalerkinPanelPairType::regular &&
              !is_near_singular(test_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                                trial_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)}))
            for (unsigned int qx = 0; qx < quadrature.size(); ++qx)
              for (unsigned int qy = 0; qy < quadrature.size(); ++qy)
                {
                  const double distance =
                    test_data.points[qx].distance(trial_data.points[qy]);
                  const double factor =
                    test_data.jxw[qx] * trial_data.jxw[qy] /
                    (4.0 * numbers::PI * distance);
                  for (unsigned int i = 0;
                       i < finite_element.dofs_per_cell;
                       ++i)
                    for (unsigned int j = 0;
                         j < finite_element.dofs_per_cell;
                         ++j)
                      local_matrix(i, j) +=
                        (test_data.surface_curls[i][qx] *
                         trial_data.surface_curls[j][qy]) * factor;
                }
          else
            local_matrix =
              integrate_weak_hypersingular(test_cell, trial_cell, order);
          if (constraints.n_constraints() == 0)
            matrix.add(test_indices, trial_indices, local_matrix);
          else
            constraints.distribute_local_to_global(local_matrix,
                                                   test_indices,
                                                   trial_indices,
                                                   matrix);
        }
    }
}

template <int spacedim>
template <typename MatrixType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_single_double_hypersingular(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &single_layer,
  MatrixType                                &double_layer,
  MatrixType                                &hypersingular,
  const unsigned int                         order,
  const types::subdomain_id                  locally_owned_subdomain) const
{
  AssertThrow(screening == 0.0,
              ExcMessage("The joint assembly of V, K and W is implemented "
                         "for the Laplace kernel only."));
  AssertDimension(dof_handler.get_fe().dofs_per_cell,
                  finite_element.dofs_per_cell);
  const unsigned int dpc = finite_element.dofs_per_cell;

  struct CellQuadratureData
  {
    CellIterator                                  cell;
    std::vector<Point<spacedim>>                  points;
    std::vector<Tensor<1, spacedim>>              normals;
    std::vector<double>                           jxw;
    std::vector<std::vector<double>>              shape_values;
    std::vector<std::vector<Tensor<1, spacedim>>> surface_curls;
  };

  // Per-cell data for every order the far field may select.
  const unsigned int lowest_order =
    far_field_tolerance > 0.0 ? std::min(2u, order) : order;
  std::vector<std::vector<CellQuadratureData>> data_by_order(order + 1);
  for (unsigned int o = lowest_order; o <= order; ++o)
    {
      const QGauss<spacedim - 1>       rule(o);
      FEValues<spacedim - 1, spacedim> values(mapping,
                                              finite_element,
                                              rule,
                                              update_values | update_gradients |
                                                update_normal_vectors |
                                                update_quadrature_points |
                                                update_JxW_values);
      auto &set = data_by_order[o];
      set.reserve(dof_handler.get_triangulation().n_active_cells());
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          values.reinit(cell);
          CellQuadratureData data;
          data.cell    = cell;
          data.points  = values.get_quadrature_points();
          data.normals = values.get_normal_vectors();
          data.jxw.resize(rule.size());
          data.shape_values.assign(dpc, std::vector<double>(rule.size()));
          data.surface_curls.assign(
            dpc, std::vector<Tensor<1, spacedim>>(rule.size()));
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              data.jxw[q] = values.JxW(q);
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  data.shape_values[i][q] = values.shape_value(i, q);
                  data.surface_curls[i][q] = cross_product_3d(
                    values.normal_vector(q), values.shape_grad(i, q));
                }
            }
          set.push_back(std::move(data));
        }
    }
  const std::vector<CellQuadratureData> &cell_data = data_by_order[order];

  // Smallest order whose convergence factor meets the tolerance, as for the
  // single layer: the double-layer kernel is singular at the same points.
  const auto far_field_order = [&](const CellIterator &test_cell,
                                   const CellIterator &trial_cell) {
    if (far_field_tolerance <= 0.0)
      return order;
    const CellGeometry &g1         = full_patch_geometry(test_cell);
    const CellGeometry &g2         = full_patch_geometry(trial_cell);
    const double        separation = g1.centre.distance(g2.centre);
    if (separation <= 0.0)
      return order;
    const double rho = (g1.radius + g2.radius) / separation;
    if (rho >= 1.0 || rho <= 0.0)
      return order;
    const unsigned int needed = static_cast<unsigned int>(
      std::ceil(std::log(far_field_tolerance) / (2.0 * std::log(rho))));
    return std::min(order, std::max(lowest_order, needed));
  };

  const unsigned int  n_q = QGauss<spacedim - 1>(order).size();
  std::vector<double> single_block(n_q * n_q), double_block(n_q * n_q),
    partial(dpc * n_q);
  std::vector<types::global_dof_index> test_indices(dpc), trial_indices(dpc);
  FullMatrix<double> v_local(dpc, dpc), k_local(dpc, dpc), w_local(dpc, dpc);

  // local(i, j) += sum_qx sum_qy u_i(qx) block(qx, qy) v_j(qy), one index at
  // a time.
  const auto contract = [&](const unsigned int                       m,
                            const std::vector<double>               &block,
                            const auto                              &u,
                            const auto                              &v,
                            FullMatrix<double>                      &local) {
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int qy = 0; qy < m; ++qy)
        {
          double acc = 0.0;
          for (unsigned int qx = 0; qx < m; ++qx)
            acc += u(i, qx) * block[qx * m + qy];
          partial[i * m + qy] = acc;
        }
    for (unsigned int i = 0; i < dpc; ++i)
      for (unsigned int j = 0; j < dpc; ++j)
        {
          double acc = 0.0;
          for (unsigned int qy = 0; qy < m; ++qy)
            acc += partial[i * m + qy] * v(j, qy);
          local(i, j) += acc;
        }
  };

  const auto add = [&](MatrixType &matrix, const FullMatrix<double> &local) {
    if (constraints.n_constraints() == 0)
      matrix.add(test_indices, trial_indices, local);
    else
      constraints.distribute_local_to_global(local,
                                             test_indices,
                                             trial_indices,
                                             matrix);
  };

  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;
      cell_dof_indices(test_cell, test_indices);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          cell_dof_indices(trial_cell, trial_indices);
          const auto pair_start = std::chrono::steady_clock::now();
          const bool far_pair =
            classify(test_cell, trial_cell) ==
              GalerkinPanelPairType::regular &&
            !is_near_singular(test_cell,
                              {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                              trial_cell,
                              {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)});
          if (far_pair)
            {
              const unsigned int pair_order =
                far_field_order(test_cell, trial_cell);
              ++far_field_order_histogram[pair_order];
              const CellQuadratureData &t =
                data_by_order[pair_order][&test_data - cell_data.data()];
              const CellQuadratureData &r =
                data_by_order[pair_order][&trial_data - cell_data.data()];
              const unsigned int m = t.jxw.size();
              for (unsigned int qx = 0; qx < m; ++qx)
                for (unsigned int qy = 0; qy < m; ++qy)
                  {
                    const Tensor<1, spacedim> displacement =
                      r.points[qy] - t.points[qx];
                    const double distance = displacement.norm();
                    const double weight   = t.jxw[qx] * r.jxw[qy] /
                                          (4.0 * numbers::PI * distance);
                    single_block[qx * m + qy] = weight;
                    double_block[qx * m + qy] =
                      -weight * (displacement * r.normals[qy]) /
                      (distance * distance);
                  }
              v_local = 0;
              k_local = 0;
              w_local = 0;
              const auto values_of = [](const CellQuadratureData &d) {
                return [&d](const unsigned int i, const unsigned int q) {
                  return d.shape_values[i][q];
                };
              };
              contract(m, single_block, values_of(t), values_of(r), v_local);
              contract(m, double_block, values_of(t), values_of(r), k_local);
              for (unsigned int c = 0; c < spacedim; ++c)
                {
                  const auto curl_of = [c](const CellQuadratureData &d) {
                    return [&d, c](const unsigned int i, const unsigned int q) {
                      return d.surface_curls[i][q][c];
                    };
                  };
                  contract(m, single_block, curl_of(t), curl_of(r), w_local);
                }
            }
          else
            {
              const FullMatrix<double> joint =
                integrate_single_double_hypersingular(test_cell,
                                                      trial_cell,
                                                      order);
              for (unsigned int i = 0; i < dpc; ++i)
                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    v_local(i, j) = joint(i, j);
                    k_local(i, j) = joint(i, dpc + j);
                    w_local(i, j) = joint(i, 2 * dpc + j);
                  }
            }
          (far_pair ? profile.far_seconds : profile.near_seconds) +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          pair_start)
              .count();
          add(single_layer, v_local);
          add(double_layer, k_local);
          add(hypersingular, w_local);
        }
    }
}


template <int spacedim>
template <typename MatrixType, typename VectorType>
void
GalerkinPanelPairIntegrator<spacedim>::assemble_single_layer_and_double_layer_rhs(
  const DoFHandler<spacedim - 1, spacedim> &dof_handler,
  const AffineConstraints<double>           &constraints,
  MatrixType                                &matrix,
  VectorType                                &rhs,
  const unsigned int                         order,
  const Function<spacedim>                  &trial_function,
  const types::subdomain_id                  locally_owned_subdomain,
  const bool                                 regularize_trial_function) const
{
  AssertDimension(dof_handler.get_fe().dofs_per_cell,
                  finite_element.dofs_per_cell);

  struct CellQuadratureData
  {
    CellIterator                     cell;
    std::vector<Point<spacedim>>     points;
    std::vector<Tensor<1, spacedim>> normals;
    std::vector<double>              jxw;
    std::vector<std::vector<double>> shape_values;
    std::vector<double>              function_values;
  };

  // As in assemble_single_layer(): per-cell data for every order the far
  // field may select. A far pair integrates the single layer and the
  // right-hand side with the same, possibly reduced, order, so every order
  // carries the normals and the trial function.
  const unsigned int lowest_order =
    far_field_tolerance > 0.0 ? std::min(2u, order) : order;
  std::vector<std::vector<CellQuadratureData>> data_by_order(order + 1);
  for (unsigned int o = lowest_order; o <= order; ++o)
    {
      const QGauss<spacedim - 1>       rule(o);
      FEValues<spacedim - 1, spacedim> values(
        mapping,
        finite_element,
        rule,
        update_values | update_quadrature_points | update_JxW_values |
          update_normal_vectors);
      auto &set = data_by_order[o];
      set.reserve(dof_handler.get_triangulation().n_active_cells());
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          values.reinit(cell);
          CellQuadratureData data;
          data.cell    = cell;
          data.points  = values.get_quadrature_points();
          data.normals = values.get_normal_vectors();
          data.function_values.resize(rule.size());
          data.jxw.resize(rule.size());
          data.shape_values.assign(finite_element.dofs_per_cell,
                                   std::vector<double>(rule.size()));
          for (unsigned int q = 0; q < rule.size(); ++q)
            {
              data.jxw[q] = values.JxW(q);
              data.function_values[q] = trial_function.value(data.points[q]);
              for (unsigned int i = 0; i < finite_element.dofs_per_cell; ++i)
                data.shape_values[i][q] = values.shape_value(i, q);
            }
          set.push_back(std::move(data));
        }
    }
  const std::vector<CellQuadratureData> &cell_data = data_by_order[order];

  const auto far_field_order = [&](const CellIterator &test_cell,
                                   const CellIterator &trial_cell) {
    if (far_field_tolerance <= 0.0)
      return order;
    const CellGeometry &g1 = full_patch_geometry(test_cell);
    const CellGeometry &g2 = full_patch_geometry(trial_cell);
    const double        separation = g1.centre.distance(g2.centre);
    if (separation <= 0.0)
      return order;
    const double rho = (g1.radius + g2.radius) / separation;
    if (rho >= 1.0 || rho <= 0.0)
      return order;
    const unsigned int needed = static_cast<unsigned int>(
      std::ceil(std::log(far_field_tolerance) / (2.0 * std::log(rho))));
    return std::min(order, std::max(lowest_order, needed));
  };

  const unsigned int dpc = finite_element.dofs_per_cell;
  std::vector<types::global_dof_index> test_indices(dpc);
  std::vector<types::global_dof_index> trial_indices(dpc);
  const unsigned int  n_q = cell_data.front().points.size();
  std::vector<double> kernel_block(n_q * n_q);
  std::vector<double> partial(dpc * n_q);
  std::vector<double> weighted(n_q);
  FullMatrix<double>  local_matrix(dpc, dpc);

  for (const auto &test_data : cell_data)
    {
      const auto &test_cell = test_data.cell;
      if (locally_owned_subdomain != numbers::invalid_subdomain_id &&
          test_cell->subdomain_id() != locally_owned_subdomain)
        continue;

      test_cell->get_dof_indices(test_indices);
      Vector<double> local_rhs(dpc);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          trial_cell->get_dof_indices(trial_indices);
          const auto pair_start = std::chrono::steady_clock::now();
          bool       far_pair   = false;
          local_matrix          = 0.0;
          if (classify(test_cell, trial_cell) ==
                GalerkinPanelPairType::regular &&
              !is_near_singular(test_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)},
                                trial_cell,
                                {Point<2>(0.0, 0.0), Point<2>(1.0, 1.0)}))
            {
              far_pair = true;
              const unsigned int pair_order =
                far_field_order(test_cell, trial_cell);
              ++far_field_order_histogram[pair_order];
              const std::size_t test_index  = &test_data - cell_data.data();
              const std::size_t trial_index = &trial_data - cell_data.data();

              // The single layer and the right-hand side with the same rule,
              // sharing the distance and the exponential of the screened
              // kernel.
              const CellQuadratureData &t_data =
                data_by_order[pair_order][test_index];
              const CellQuadratureData &r_data =
                data_by_order[pair_order][trial_index];
              const unsigned int m = t_data.jxw.size();
              for (unsigned int qx = 0; qx < m; ++qx)
                {
                  double sum_kernel   = 0.0;
                  double sum_weighted = 0.0;
                  for (unsigned int qy = 0; qy < m; ++qy)
                    {
                      const Tensor<1, spacedim> displacement =
                        r_data.points[qy] - t_data.points[qx];
                      const double distance = displacement.norm();
                      const double exponential =
                        screening == 0.0 ? 1.0 :
                                           std::exp(-screening * distance);
                      const double weight = t_data.jxw[qx] * r_data.jxw[qy];
                      const double factor =
                        -(displacement * r_data.normals[qy]) * weight /
                        (4.0 * numbers::PI * distance * distance * distance);
                      sum_kernel += factor;
                      sum_weighted += factor * (1.0 + screening * distance) *
                                      exponential * r_data.function_values[qy];
                      kernel_block[qx * m + qy] =
                        weight * exponential / (4.0 * numbers::PI * distance);
                    }
                  weighted[qx] = sum_weighted -
                                 (regularize_trial_function ?
                                    sum_kernel * t_data.function_values[qx] :
                                    0.0);
                }
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  const std::vector<double> &u   = t_data.shape_values[i];
                  double                     acc = 0.0;
                  for (unsigned int qx = 0; qx < m; ++qx)
                    acc += u[qx] * weighted[qx];
                  local_rhs[i] += acc;
                }
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  const std::vector<double> &u = t_data.shape_values[i];
                  for (unsigned int qy = 0; qy < m; ++qy)
                    {
                      double acc = 0.0;
                      for (unsigned int qx = 0; qx < m; ++qx)
                        acc += u[qx] * kernel_block[qx * m + qy];
                      partial[i * m + qy] = acc;
                    }
                }
              for (unsigned int i = 0; i < dpc; ++i)
                for (unsigned int j = 0; j < dpc; ++j)
                  {
                    const std::vector<double> &v = r_data.shape_values[j];
                    double                     acc = 0.0;
                    for (unsigned int qy = 0; qy < m; ++qy)
                      acc += partial[i * m + qy] * v[qy];
                    local_matrix(i, j) += acc;
                  }
            }
          else
            {
              const FullMatrix<double> both =
                integrate_single_and_double_layer(test_cell,
                                                  trial_cell,
                                                  order,
                                                  trial_function,
                                                  regularize_trial_function);
              for (unsigned int i = 0; i < dpc; ++i)
                {
                  for (unsigned int j = 0; j < dpc; ++j)
                    local_matrix(i, j) = both(i, j);
                  local_rhs[i] += both(i, dpc);
                }
            }
          (far_pair ? profile.far_seconds : profile.near_seconds) +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          pair_start)
              .count();
          if (constraints.n_constraints() == 0)
            matrix.add(test_indices, trial_indices, local_matrix);
          else
            constraints.distribute_local_to_global(local_matrix,
                                                   test_indices,
                                                   trial_indices,
                                                   matrix);
        }

      local_rhs *= -1.0;
      constraints.distribute_local_to_global(local_rhs, test_indices, rhs);
    }
}

#endif
