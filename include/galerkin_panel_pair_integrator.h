#ifndef GALERKIN_PANEL_PAIR_INTEGRATOR_H
#define GALERKIN_PANEL_PAIR_INTEGRATOR_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>

#include <chrono>
#include <map>
#include <memory>
#include <deal.II/fe/mapping.h>
#include <deal.II/base/function.h>
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

  double
  get_screening() const
  {
    return screening;
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
   * available from light_geometry_deviation().
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
    single_and_double_layer
  };

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

  bool
  share_nonconforming_edge(const CellIterator &test_cell,
                           const CellIterator &trial_cell) const;

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
                 std::vector<double>             &pair_weights) const;

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
  // integrated as such.
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
    const unsigned int dpc = finite_element.dofs_per_cell;
    return operator_type == OperatorType::single_and_double_layer ? dpc + 1 :
           trial_function != nullptr                              ? 1 :
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
  };

  /// A singular rule of the direct path, in the coordinates of its two
  /// patches: it depends only on the configuration of the pair, so its
  /// points, weights and, for whole cells, shape values are built once.
  struct LightRule
  {
    std::vector<Point<2>> test_points;
    std::vector<Point<2>> trial_points;
    std::vector<double>   weights;
    std::vector<double>   test_shapes;
    std::vector<double>   trial_shapes;
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
  evaluate_shapes(const std::vector<Point<2>> &reference_points,
                  std::vector<double>         &shapes) const;

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
  };

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
  std::vector<double>              geometry_coefficients;
  std::vector<double>              shape_coefficients;
  mutable std::vector<LightCell>   light_cells;
  mutable std::vector<bool>        light_cell_valid;
  mutable double                   maximum_light_deviation = 0.0;
  mutable std::map<std::array<int, 7>, LightRule> light_rule_cache;
  // Point sets of the direct path, reused from pair to pair, the FEValues
  // that checks the reconstructed geometry, and the tensor Gauss rules.
  mutable PointSet                                test_set;
  mutable PointSet                                trial_set;
  mutable std::unique_ptr<FEValues<2, spacedim>>  light_check_values;
  mutable std::map<unsigned int, QGauss<2>>       gauss_rules;

  const Mapping<spacedim - 1, spacedim>       &mapping;
  const FiniteElement<spacedim - 1, spacedim> &finite_element;
  const GalerkinSingularQuadrature             singular_quadrature;
  const unsigned int                           singular_order;
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
  mutable RhsProfile                          profile;

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

      test_cell->get_dof_indices(test_indices);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          trial_cell->get_dof_indices(trial_indices);
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

      test_cell->get_dof_indices(test_indices);
      for (const auto &trial_cell : dof_handler.active_cell_iterators())
        {
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          trial_cell->get_dof_indices(trial_indices);
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

      test_cell->get_dof_indices(test_indices);
      for (const auto &trial_data : cell_data)
        {
          const auto &trial_cell = trial_data.cell;
          if (selected_trial_material != numbers::invalid_material_id &&
              trial_cell->material_id() != selected_trial_material)
            continue;
          trial_cell->get_dof_indices(trial_indices);
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
