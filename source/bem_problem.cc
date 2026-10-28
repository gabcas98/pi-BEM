

#include "../include/bem_problem.h"

#include <deal.II/base/parallel.h>

#include <deal.II/lac/arpack_solver.h>

#include <deal.II/numerics/error_estimator.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "../include/cellwise_function.h"
#include "../include/constrained_matrix_complex.h"
#include "../include/galerkin_calderon_fmm_operator.h"
#include "../include/galerkin_panel_pair_integrator.h"
#include "../include/laplace_kernel.h"
#include "../include/quasi_singular_kernel_integral.h"
#include "../include/screened_kernel.h"
#include "../include/singular_kernel_integral.h"
#include "../include/surface_potential_evaluator.h"
#include "../include/telles_quadrature.h"
#include "Teuchos_TimeMonitor.hpp"
#include "bem_problem_helpers.h"
#include "mapping_hanging_constraints.h"

using Teuchos::RCP;
using Teuchos::Time;
using Teuchos::TimeMonitor;
using namespace std;
using bem_problem_internal::createFiniteElementPointer;

#define ENTRY EntryRaiiObject obj##LINE(__FUNCTION__);

struct EntryRaiiObject
{
  EntryRaiiObject(const char *f)
    : f_(f)
  {
    printf("Entered into %s\n", f_);
  }

  ~EntryRaiiObject()
  {
    printf("Exited from %s\n", f_);
  }

  const char *f_;
};
namespace
{
  double
  clamp_to_unit_interval(const double value)
  {
    return std::max(-1.0, std::min(1.0, value));
  }

  template <typename VEC>
  void
  vector_shift(VEC &in_vec, double a_scalar)
  {
    for (auto i : in_vec.locally_owned_elements())
      in_vec[i] += a_scalar;
  }

  template <int dim>
  bool
  is_finite_tensor(const Tensor<1, dim> &value)
  {
    for (unsigned int d = 0; d < dim; ++d)
      if (!std::isfinite(value[d]))
        return false;
    return true;
  }

  struct QuadratureConvergenceState
  {
    double previous_difference  = std::numeric_limits<double>::infinity();
    double previous_contraction = std::numeric_limits<double>::infinity();
  };

  bool
  quadrature_vectors_converged(const Vector<double>       &previous,
                               const Vector<double>       &current,
                               const double                relative_tolerance,
                               const double                absolute_tolerance,
                               QuadratureConvergenceState &state)
  {
    AssertDimension(previous.size(), current.size());
    double difference = 0.0;
    double scale      = 0.0;
    for (types::global_dof_index i = 0; i < current.size(); ++i)
      {
        difference = std::max(difference, std::abs(current[i] - previous[i]));
        scale      = std::max(scale,
                         std::max(std::abs(current[i]), std::abs(previous[i])));
      }
    const double tolerance = absolute_tolerance + relative_tolerance * scale;
    const bool   direct_estimate_converged = difference <= tolerance;

    double contraction = std::numeric_limits<double>::infinity();
    bool   asymptotic_estimate_converged = false;
    if (std::isfinite(state.previous_difference) &&
        state.previous_difference > 0.0)
      {
        contraction = difference / state.previous_difference;
        if (contraction < 1.0 && contraction <= 0.5 &&
            state.previous_contraction <= 0.5)
          {
            const double estimated_remaining_error =
              difference * contraction / (1.0 - contraction);
            asymptotic_estimate_converged =
              estimated_remaining_error <= tolerance;
          }
      }

    state.previous_difference  = difference;
    state.previous_contraction = contraction;
    return direct_estimate_converged || asymptotic_estimate_converged;
  }

  bool
  quadrature_scalars_converged(const double                previous,
                               const double                current,
                               const double                relative_tolerance,
                               const double                absolute_tolerance,
                               QuadratureConvergenceState &state)
  {
    const double difference = std::abs(current - previous);
    const double scale      = std::max(std::abs(current), std::abs(previous));
    const double tolerance  = absolute_tolerance + relative_tolerance * scale;
    const bool   direct_estimate_converged = difference <= tolerance;

    double contraction = std::numeric_limits<double>::infinity();
    bool   asymptotic_estimate_converged = false;
    if (std::isfinite(state.previous_difference) &&
        state.previous_difference > 0.0)
      {
        contraction = difference / state.previous_difference;
        if (contraction < 1.0 && contraction <= 0.5 &&
            state.previous_contraction <= 0.5)
          asymptotic_estimate_converged =
            difference * contraction / (1.0 - contraction) <= tolerance;
      }

    state.previous_difference  = difference;
    state.previous_contraction = contraction;
    return direct_estimate_converged || asymptotic_estimate_converged;
  }

  unsigned int
  next_adaptive_order(const unsigned int current_order,
                      const unsigned int maximum_order,
                      const double       growth)
  {
    const unsigned int grown_order = static_cast<unsigned int>(
      std::ceil(growth * static_cast<double>(current_order)));
    return std::min(maximum_order, std::max(current_order + 1, grown_order));
  }

  /** Symmetric diagonal scaling taken from the Galerkin single-layer
   * operator. The single layer is symmetric and positive definite, so its
   * diagonal is strictly positive and this preconditioner stays symmetric,
   * as the conjugate gradient method requires. The diagonal is read directly
   * from the assembled operator, so no explicit system has to be built. On
   * the unknowns fixed by constraints the operator is the identity, and so is
   * the preconditioner.
   */
  class DiagonalPreconditioner
  {
  public:
    DiagonalPreconditioner(const TrilinosWrappers::SparseMatrix &matrix,
                           const IndexSet                  &locally_owned_dofs,
                           const MPI_Comm                   communicator,
                           const AffineConstraints<double> &constraints)
      : inverse_diagonal(locally_owned_dofs, communicator)
    {
      for (const auto i : locally_owned_dofs)
        {
          if (constraints.is_constrained(i))
            {
              inverse_diagonal[i] = 1.0;
              continue;
            }
          const double entry = matrix.diag_element(i);
          AssertThrow(entry > 0.0,
                      ExcMessage("The single-layer diagonal must be positive "
                                 "for a symmetric diagonal preconditioner."));
          inverse_diagonal[i] = 1.0 / entry;
        }
      inverse_diagonal.compress(VectorOperation::insert);
    }

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const
    {
      dst = src;
      dst.scale(inverse_diagonal);
    }

  private:
    TrilinosWrappers::MPI::Vector inverse_diagonal;
  };

  /** Additive Schwarz preconditioner on the cells: the sum over the cells of
   * the inverse of the block of the Galerkin single layer that couples the
   * free unknowns of the cell,
   *     P^{-1} = sum_K R_K^T A_K^{-1} R_K.
   * Every term is symmetric positive semidefinite and together they cover
   * every free unknown, so P^{-1} is symmetric positive definite, as the
   * conjugate gradient method requires. For discontinuous elements the
   * blocks do not overlap and this is block Jacobi; for continuous ones an
   * unknown shared by several cells receives the contribution of each. On
   * the unknowns fixed by constraints the operator is the identity, and so
   * is the preconditioner.
   *
   * The blocks are small and the vectors are exchanged whole: every process
   * owns the rows of its unknowns only, so each fills the rows it owns and
   * the blocks are summed over the processes once; each process then applies
   * the blocks of its own cells, and the contributions are summed.
   */
  class CellBlockPreconditioner
  {
  public:
    CellBlockPreconditioner(
      const TrilinosWrappers::SparseMatrix                    &matrix,
      const std::vector<std::vector<types::global_dof_index>> &cell_dofs,
      const std::vector<bool>                                 &applies_cell,
      const IndexSet                  &locally_owned_dofs,
      const AffineConstraints<double> &constraints,
      const MPI_Comm                   communicator)
      : cell_dofs(cell_dofs)
      , applies_cell(applies_cell)
      , locally_owned_dofs(locally_owned_dofs)
      , constraints(constraints)
      , communicator(communicator)
    {
      std::vector<std::size_t> offset(cell_dofs.size() + 1, 0);
      for (unsigned int c = 0; c < cell_dofs.size(); ++c)
        offset[c + 1] = offset[c] + cell_dofs[c].size() * cell_dofs[c].size();
      std::vector<double> entries(offset.back(), 0.0);
      for (unsigned int c = 0; c < cell_dofs.size(); ++c)
        {
          const auto        &dofs = cell_dofs[c];
          const unsigned int n    = dofs.size();
          for (unsigned int i = 0; i < n; ++i)
            if (locally_owned_dofs.is_element(dofs[i]))
              for (unsigned int j = 0; j < n; ++j)
                entries[offset[c] + i * n + j] = matrix.el(dofs[i], dofs[j]);
        }
      Utilities::MPI::sum(ArrayView<const double>(entries.data(),
                                                  entries.size()),
                          communicator,
                          ArrayView<double>(entries.data(), entries.size()));

      inverses.resize(cell_dofs.size());
      for (unsigned int c = 0; c < cell_dofs.size(); ++c)
        if (applies_cell[c] && !cell_dofs[c].empty())
          {
            const unsigned int n = cell_dofs[c].size();
            inverses[c].reinit(n, n);
            for (unsigned int i = 0; i < n; ++i)
              for (unsigned int j = 0; j < n; ++j)
                inverses[c](i, j) = entries[offset[c] + i * n + j];
            inverses[c].gauss_jordan();
          }
    }

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const
    {
      const Vector<double> full_src(src);
      Vector<double>       sum(full_src.size());
      Vector<double>       local_src, local_dst;
      for (unsigned int c = 0; c < cell_dofs.size(); ++c)
        if (applies_cell[c] && !cell_dofs[c].empty())
          {
            const auto &dofs = cell_dofs[c];
            local_src.reinit(dofs.size());
            local_dst.reinit(dofs.size());
            for (unsigned int i = 0; i < dofs.size(); ++i)
              local_src[i] = full_src[dofs[i]];
            inverses[c].vmult(local_dst, local_src);
            for (unsigned int i = 0; i < dofs.size(); ++i)
              sum[dofs[i]] += local_dst[i];
          }
      Utilities::MPI::sum(ArrayView<const double>(sum.begin(), sum.size()),
                          communicator,
                          ArrayView<double>(sum.begin(), sum.size()));
      for (const auto i : locally_owned_dofs)
        dst[i] = constraints.is_constrained(i) ? full_src[i] : sum[i];
      dst.compress(VectorOperation::insert);
    }

  private:
    const std::vector<std::vector<types::global_dof_index>> cell_dofs;
    const std::vector<bool>                                 applies_cell;
    const IndexSet                                          locally_owned_dofs;
    const AffineConstraints<double>                        &constraints;
    const MPI_Comm                                          communicator;
    std::vector<FullMatrix<double>>                         inverses;
  };

#ifdef DEAL_II_WITH_MUMPS
  class CalderonOperatorPreconditioner
  {
  public:
    CalderonOperatorPreconditioner(
      const TrilinosWrappers::SparseMatrix &hypersingular_matrix,
      const TrilinosWrappers::SparseMatrix &mass_matrix,
      const IndexSet                       &locally_owned_dofs,
      const MPI_Comm                        communicator,
      const double                          stabilization)
      : hypersingular_matrix(hypersingular_matrix)
      , mass_matrix(mass_matrix)
      , mass_inverse(
          SparseDirectMUMPS::AdditionalData(false, false, false, false),
          communicator)
      , mass_inverse_source(locally_owned_dofs, communicator)
      , operator_result(locally_owned_dofs, communicator)
      , mass_result(locally_owned_dofs, communicator)
      , stabilization(stabilization)
    {
      mass_inverse.initialize(mass_matrix);
    }

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const
    {
      mass_inverse.vmult(mass_inverse_source, src);
      hypersingular_matrix.vmult(operator_result, mass_inverse_source);
      mass_matrix.vmult(mass_result, mass_inverse_source);
      operator_result.add(stabilization, mass_result);
      mass_inverse.vmult(dst, operator_result);
    }

  private:
    const TrilinosWrappers::SparseMatrix &hypersingular_matrix;
    const TrilinosWrappers::SparseMatrix &mass_matrix;
    mutable SparseDirectMUMPS             mass_inverse;
    mutable TrilinosWrappers::MPI::Vector mass_inverse_source;
    mutable TrilinosWrappers::MPI::Vector operator_result;
    mutable TrilinosWrappers::MPI::Vector mass_result;
    const double                          stabilization;
  };
#endif

  class CalderonDualPreconditioner
  {
  public:
    CalderonDualPreconditioner(
      const TrilinosWrappers::SparseMatrix &hypersingular_matrix,
      const TrilinosWrappers::SparseMatrix &mass_matrix,
      const IndexSet                       &locally_owned_cells,
      const IndexSet                       &locally_owned_vertices,
      const MPI_Comm                        communicator,
      const std::vector<std::vector<types::global_dof_index>> &cell_vertices,
      const std::vector<unsigned int>                         &vertex_valence,
      const std::vector<double>                               &cell_areas,
      const double hypersingular_stabilization,
      const double bubble_scaling)
      : hypersingular_matrix(hypersingular_matrix)
      , mass_matrix(mass_matrix)
      , locally_owned_cells(locally_owned_cells)
      , communicator(communicator)
      , cell_vertices(cell_vertices)
      , vertex_valence(vertex_valence)
      , cell_areas(cell_areas)
      , vertex_source(locally_owned_vertices, communicator)
      , vertex_operator_result(locally_owned_vertices, communicator)
      , vertex_mass_result(locally_owned_vertices, communicator)
      , hypersingular_stabilization(hypersingular_stabilization)
      , bubble_scaling(bubble_scaling)
    {}

    void
    vmult(TrilinosWrappers::MPI::Vector       &dst,
          const TrilinosWrappers::MPI::Vector &src) const
    {
      const Vector<double> localized_src(src);
      std::vector<double>  area_scaled(cell_vertices.size(), 0.0);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size();
           ++cell)
        area_scaled[cell] = localized_src[cell] / cell_areas[cell];

      const std::vector<double> projected = apply_p(area_scaled);
      vertex_source                       = 0.0;
      for (const auto vertex : vertex_source.locally_owned_elements())
        vertex_source[vertex] = projected[vertex];
      vertex_source.compress(VectorOperation::insert);

      hypersingular_matrix.vmult(vertex_operator_result, vertex_source);
      mass_matrix.vmult(vertex_mass_result, vertex_source);
      vertex_operator_result.add(hypersingular_stabilization,
                                 vertex_mass_result);
      const Vector<double> localized_vertex_result(vertex_operator_result);

      std::vector<double> result = apply_pt(localized_vertex_result);
      std::vector<double> bubble = apply_q(area_scaled);
      for (types::global_dof_index cell = 0; cell < bubble.size(); ++cell)
        bubble[cell] *= bubble_scaling * std::sqrt(cell_areas[cell]);
      bubble = apply_q(bubble);
      for (types::global_dof_index cell = 0; cell < result.size(); ++cell)
        result[cell] = (result[cell] + bubble[cell]) / cell_areas[cell];

      dst = 0.0;
      for (const auto cell : locally_owned_cells)
        dst[cell] = result[cell];
      dst.compress(VectorOperation::insert);
    }

  private:
    std::vector<double>
    apply_p(const std::vector<double> &cell_values) const
    {
      std::vector<double> vertex_values(vertex_valence.size(), 0.0);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size();
           ++cell)
        for (const auto vertex : cell_vertices[cell])
          vertex_values[vertex] +=
            cell_values[cell] / static_cast<double>(vertex_valence[vertex]);
      return vertex_values;
    }

    std::vector<double>
    apply_pt(const Vector<double> &vertex_values) const
    {
      std::vector<double> cell_values(cell_vertices.size(), 0.0);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size();
           ++cell)
        for (const auto vertex : cell_vertices[cell])
          cell_values[cell] +=
            vertex_values[vertex] / static_cast<double>(vertex_valence[vertex]);
      return cell_values;
    }

    std::vector<double>
    apply_q(const std::vector<double> &cell_values) const
    {
      const std::vector<double> projected = apply_p(cell_values);
      std::vector<double>       result(cell_values);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size();
           ++cell)
        {
          double vertex_average = 0.0;
          for (const auto vertex : cell_vertices[cell])
            vertex_average += projected[vertex];
          result[cell] -=
            vertex_average / static_cast<double>(cell_vertices[cell].size());
        }
      return result;
    }

    const TrilinosWrappers::SparseMatrix &hypersingular_matrix;
    const TrilinosWrappers::SparseMatrix &mass_matrix;
    const IndexSet                       &locally_owned_cells;
    const MPI_Comm                        communicator;
    const std::vector<std::vector<types::global_dof_index>> &cell_vertices;
    const std::vector<unsigned int>                         &vertex_valence;
    const std::vector<double>                               &cell_areas;
    mutable TrilinosWrappers::MPI::Vector                    vertex_source;
    mutable TrilinosWrappers::MPI::Vector vertex_operator_result;
    mutable TrilinosWrappers::MPI::Vector vertex_mass_result;
    const double                          hypersingular_stabilization;
    const double                          bubble_scaling;
  };
} // namespace

RCP<Time> ConstraintsTime =
  Teuchos::TimeMonitor::getNewTimer("Compute Constraints Time");
RCP<Time> AssembleTime = Teuchos::TimeMonitor::getNewTimer("Assemble Time");
RCP<Time> NormalsTime  = Teuchos::TimeMonitor::getNewTimer("Normals Time");
RCP<Time> SurfaceGradientTime =
  Teuchos::TimeMonitor::getNewTimer("SurfaceGradientTime Time");
RCP<Time> GradientTime = Teuchos::TimeMonitor::getNewTimer("Gradient Time");
RCP<Time> LacSolveTime = Teuchos::TimeMonitor::getNewTimer("LAC Solve Time");
RCP<Time> ReinitTime =
  Teuchos::TimeMonitor::getNewTimer("BEM Reinitialisation Time");

// @sect4{BEMProblem::BEMProblem and
// BEMProblem::read_parameters}
// The constructor initializes the
// variuous object in much the same
// way as done in the finite element
// programs such as step-4 or
// step-6. The only new ingredient
// here is the ParsedFunction object,
// which needs, at construction time,
// the specification of the number of
// components.
//
// For the exact solution the number
// of vector components is one, and
// no action is required since one is
// the default value for a
// ParsedFunction object. The wind,
// however, requires dim components
// to be specified. Notice that when
// declaring entries in a parameter
// file for the expression of the
// Functions::ParsedFunction, we need
// to specify the number of
// components explicitly, since the
// function
// Functions::ParsedFunction::declare_parameters
// is static, and has no knowledge of
// the number of components.
template <>
BEMProblem<3>::BEMProblem(ComputationalDomain<3> &comp_dom,
                          MPI_Comm                comm,
                          unsigned int            n_components)
  : n_components(n_components)
  , current_component(0)
  , pcout(std::cout)
  , comp_dom(comp_dom)
  , dh(comp_dom.tria)
  , gradient_dh(comp_dom.tria)
  , calderon_dh(comp_dom.tria)
  , mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , vector_gradients_solutions(n_components)
  , vector_surface_gradients_solutions(n_components)
  , collocation_fmm(comm)
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}

template <>
BEMProblem<2>::BEMProblem(ComputationalDomain<2> &comp_dom,
                          MPI_Comm                comm,
                          unsigned int            n_components)
  : n_components(n_components)
  , current_component(0)
  , pcout(std::cout)
  , comp_dom(comp_dom)
  , dh(comp_dom.tria)
  , gradient_dh(comp_dom.tria)
  , calderon_dh(comp_dom.tria)
  , mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , vector_gradients_solutions(n_components)
  , vector_surface_gradients_solutions(n_components)
  , collocation_fmm(comm)
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}


template <int dim>
void
BEMProblem<dim>::reinit()
{
  // ENTRY
  Teuchos::TimeMonitor LocalTimer(*ReinitTime);
  // The Galerkin FMM and the operators of the mixed problem belong to the
  // previous mesh; the next solve sets them up.
  galerkin_fmm.reset();
  mixed_single_layer.clear();
  mixed_double_layer.clear();
  mixed_hypersingular.clear();
  mixed_calderon.reset();
  mixed_operators_size = 0;

  fe          = createFiniteElementPointer<dim - 1, dim, false>(scalar_fe_type,
                                                       scalar_fe_order);
  gradient_fe = createFiniteElementPointer<dim - 1, dim, true>(vector_fe_type,
                                                               vector_fe_order);
  // fe = new FE_DGQArbitraryNodes<dim-1, dim>(QGauss<1> (2));
  // gradient_fe = new
  // FESystem<dim-1,dim>(FE_DGQArbitraryNodes<dim-1,dim>(QGauss<1> (2)),dim);
  // // auto hhh = new FE_DGQArbitraryNodes<dim-1, dim>(QGauss<1> (2));
  std::string foo = fe->get_name();
  pcout << "FE name " << foo << std::endl;
  // FiniteElement<dim-1,dim> * pippo = FETools::get_fe_by_name<dim-1,
  // dim>(foo); std::cout<<pippo->get_name()<<std::endl;

  dh.distribute_dofs(*fe);
  gradient_dh.distribute_dofs(*gradient_fe);

  // we should choose the appropriate renumbering strategy and then stick with
  // it. in step 32 they use component_wise which is very straight-forward but
  // maybe the quickest is subdomain_wise (step 17, 18)
  DoFRenumbering::component_wise(dh);
  DoFRenumbering::component_wise(gradient_dh);

  pcout << "re-ordering vector" << std::endl;

  compute_reordering_vectors();

  DoFRenumbering::subdomain_wise(dh);
  DoFRenumbering::subdomain_wise(gradient_dh);

  vector_constraints.reinit();
  DoFTools::make_hanging_node_constraints(gradient_dh, vector_constraints);
  vector_constraints.close();
  if (mapping_type == "FE")
    {
      map_vector.reinit(gradient_dh.n_dofs());
      // Fills the euler vector with information from the Triangulation
      VectorTools::get_position_vector(gradient_dh, map_vector);
      vector_constraints.distribute(map_vector);
    }

  if (!mapping)
    {
      if (comp_dom.spheroid_bool && comp_dom.used_spherical_manifold)
        {
          for (types::global_dof_index ii = 0; ii < gradient_dh.n_dofs() / dim;
               ++ii)
            {
              map_vector[vec_original_to_sub_wise[ii]] *=
                comp_dom.spheroid_x_axis;
              map_vector[vec_original_to_sub_wise[ii + gradient_dh.n_dofs() /
                                                         dim]] *=
                comp_dom.spheroid_y_axis;
              if (dim == 3)
                {
                  map_vector[vec_original_to_sub_wise
                               [ii + gradient_dh.n_dofs() / dim]] *=
                    comp_dom.spheroid_z_axis;
                }
            }
        }

      if (mapping_type == "FE")
        {
          mapping = std::make_shared<MappingFEField<dim - 1, dim>>(gradient_dh,
                                                                   map_vector);
        }
      else if (exact_geometry)
        {
          exact_mapping = std::dynamic_pointer_cast<
            ExactGeometry::ProjectedMapping<dim - 1, dim>>(
            make_exact_geometry_mapping());
          if (exact_geometry_degree > 0 || exact_geometry_tolerance > 0)
            interpolate_exact_geometry = true;
          else
            mapping = exact_mapping;
        }
      else
        {
          mapping = std::make_shared<MappingQ<dim - 1, dim>>(mapping_degree);
        }
    }
  // The interpolated exact geometry is cleared whenever the triangulation
  // changes; it is filled again here, at the given degree or at the lowest
  // degree that meets the tolerance on the current mesh.
  if (interpolate_exact_geometry)
    {
      const auto   start  = std::chrono::steady_clock::now();
      unsigned int degree = exact_geometry_degree > 0 ?
                              exact_geometry_degree :
                              std::max(2u, mapping_degree);
      // The degree from an estimate on every eighth cell, the processes
      // taking contiguous parts: the lowest that meets the tolerance, up to
      // 10. The deviation may stay nearly flat over the first degrees, then
      // falls exponentially. Once it has fallen by ten in one degree, a
      // deviation that does not halve over two degrees no longer follows
      // the interpolation but a geometry that is not smooth or not precise
      // enough for the tolerance, as a CAD surface written to a few digits;
      // the search then stops at the degree of the smallest deviation. The
      // interpolation is checked on all cells, and the degree raised as
      // long as that halves the deviation.
      const auto deviation_of = [](const std::pair<double, double> &d) {
        return std::max(d.first, d.second);
      };
      const unsigned int n_processes =
        Utilities::MPI::n_mpi_processes(mpi_communicator);
      bool stalled    = false;
      bool converging = false;
      if (exact_geometry_degree == 0)
        {
          std::ostringstream estimates;
          double             smallest    = std::numeric_limits<double>::max();
          double             previous    = 0;
          unsigned int       best_degree = degree, without_gain = 0;
          for (;; ++degree)
            {
              auto estimate = ExactGeometry::estimate_interpolation_deviation(
                *exact_mapping,
                comp_dom.tria,
                degree,
                8,
                Utilities::MPI::this_mpi_process(mpi_communicator),
                n_processes);
              estimate.first =
                Utilities::MPI::max(estimate.first, mpi_communicator);
              estimate.second =
                Utilities::MPI::max(estimate.second, mpi_communicator);
              const double deviation = deviation_of(estimate);
              estimates << ' ' << degree << ": " << deviation;
              if (deviation <= exact_geometry_tolerance)
                break;
              // From the second degree on.
              converging = converging || deviation < previous / 10;
              previous   = deviation;
              if (deviation < smallest / 2)
                {
                  smallest     = deviation;
                  best_degree  = degree;
                  without_gain = 0;
                }
              else if (++without_gain >= 2 && converging)
                {
                  stalled = true;
                  degree  = best_degree;
                  break;
                }
              if (degree == 10)
                {
                  degree = best_degree;
                  break;
                }
            }
          pcout << "Exact geometry: estimated deviation by degree"
                << estimates.str() << std::endl;
        }
      auto interpolation =
        std::make_shared<MappingQCache<dim - 1, dim>>(degree);
      auto deviation = ExactGeometry::interpolate_exact_geometry(
        *interpolation, *exact_mapping, comp_dom.tria, mpi_communicator);
      // The deviation falls in pairs of degrees, little from an odd degree
      // to the next even one: the degree is raised by one or two.
      while (exact_geometry_degree == 0 && !stalled &&
             deviation_of(deviation) > exact_geometry_tolerance)
        {
          bool         raised = false;
          unsigned int step   = 1;
          for (; step <= 2 && degree + step <= 10; ++step)
            {
              auto higher =
                std::make_shared<MappingQCache<dim - 1, dim>>(degree + step);
              const auto higher_deviation =
                ExactGeometry::interpolate_exact_geometry(*higher,
                                                          *exact_mapping,
                                                          comp_dom.tria,
                                                          mpi_communicator);
              if (deviation_of(higher_deviation) < deviation_of(deviation) / 2)
                {
                  interpolation = higher;
                  deviation     = higher_deviation;
                  degree += step;
                  raised = true;
                  break;
                }
            }
          if (!raised)
            {
              // Two degrees tried without gain once converging: the
              // geometry; otherwise the convergence is too slow, or the
              // degree the largest.
              stalled = step > 2 && converging;
              break;
            }
        }
      retired_mapping_cache  = exact_mapping_cache;
      exact_mapping_cache    = interpolation;
      mapping                = exact_mapping_cache;
      geometry_degree_in_use = degree;
      pcout << "Exact geometry: interpolation of degree " << degree << " in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             start)
                 .count()
            << " s; largest deviation from the exact map, points "
            << deviation.first << " of the cell diameter, surface elements "
            << deviation.second << std::endl;
      if (exact_geometry_degree == 0 &&
          deviation_of(deviation) > exact_geometry_tolerance)
        {
          if (stalled)
            pcout << "Warning: the deviation of the interpolation stops "
                     "decreasing at degree "
                  << degree << ", above the tolerance "
                  << exact_geometry_tolerance
                  << ": the geometry is not smooth or not precise enough "
                     "for it"
                  << std::endl;
          else
            pcout << "Warning: the interpolation misses the tolerance "
                  << exact_geometry_tolerance << " up to degree 10"
                  << std::endl;
        }
    }


  if (exact_geometry)
    {
      vector_constraints.clear();
      const auto corrected =
        ExactGeometry::make_mapping_hanging_node_constraints(
          *mapping, gradient_dh, vector_constraints);
      vector_constraints.close();
      pcout << "Mapping hanging constraints: " << corrected.corrected_lines
            << " vector lines corrected; maximum weight change "
            << corrected.maximum_weight_change << ", off-edge support points "
            << corrected.off_edge_points << std::endl;
    }

  const types::global_dof_index n_dofs = dh.n_dofs();

  pcout << "phi dofs: " << dh.n_dofs()
        << " gradient phi dofs: " << gradient_dh.n_dofs() << std::endl;
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);

  DoFTools::get_subdomain_association(dh, dofs_domain_association);
  std::vector<types::subdomain_id> vector_dofs_domain_association(
    gradient_dh.n_dofs());

  DoFTools::get_subdomain_association(gradient_dh,
                                      vector_dofs_domain_association);

  this_cpu_set.clear();
  vector_this_cpu_set.clear();
  this_cpu_set.set_size(dh.n_dofs());
  vector_this_cpu_set.set_size(gradient_dh.n_dofs());

  // We compute this two vector in order to use an eventual
  // DoFRenumbering::subdomain_wise At the time being we don't. We need to
  // decide the better strategy.

  // We need to enforce consistency between the non-ghosted IndexSets.
  // To be changed accordingly with the DoFRenumbering strategy.
  pcout << "you are using " << sizeof(dh.n_dofs()) << " bytes indices"
        << std::endl;
  pcout << "setting cpu_sets" << std::endl;

  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    if (dofs_domain_association[i] == this_mpi_process)
      {
        this_cpu_set.add_index(i);
        types::global_dof_index dummy = sub_wise_to_original[i];
        for (unsigned int idim = 0; idim < dim; ++idim)
          {
            vector_this_cpu_set.add_index(
              vec_original_to_sub_wise[gradient_dh.n_dofs() / dim * idim +
                                       dummy]);
          }
      }

  this_cpu_set.compress();
  vector_this_cpu_set.compress();

  // At this point we just need to create a ghosted IndexSet for the scalar
  // DoFHandler. This can be through the builtin dealii function.
  MPI_Barrier(mpi_communicator);
  ghosted_set.clear();
  ghosted_set.set_size(dh.n_dofs());
  ghosted_set =
    DoFTools::dof_indices_with_subdomain_association(dh, this_mpi_process);
  ghosted_set.compress();

  // standard TrilinosWrappers::MPI::Vector reinitialization.
  system_rhs.reinit(this_cpu_set, mpi_communicator);
  exact_regularized_dirichlet_rhs.reinit(this_cpu_set, mpi_communicator);
  sol.reinit(this_cpu_set, mpi_communicator);
  alpha.reinit(this_cpu_set, mpi_communicator);
  serv_phi.reinit(this_cpu_set, mpi_communicator);
  serv_phi_imag.reinit(this_cpu_set, mpi_communicator);
  serv_dphi_dn.reinit(this_cpu_set, mpi_communicator);
  serv_dphi_dn_imag.reinit(this_cpu_set, mpi_communicator);

  if (preconditioner_type == "CalderonDual")
    initialize_calderon_dual_space();

  std::vector<IndexSet> parallel_partition{this_cpu_set, this_cpu_set};
  sol_blocked.reinit(parallel_partition);
  system_rhs_blocked.reinit(parallel_partition);

  // TrilinosWrappers::SparsityPattern for the BEM matrices reinitialization
  pcout << "re-initializing sparsity patterns and matrices" << std::endl;
  if (solution_method == "Direct")
    {
      full_sparsity_pattern.reinit(this_cpu_set, mpi_communicator);

      // TODO: candidate for parallelization - one row per thread
      for (auto i : this_cpu_set)
        {
          for (types::global_dof_index j = 0; j < dh.n_dofs(); ++j)
            {
              full_sparsity_pattern.add(i, j);
            }
        }

      full_sparsity_pattern.compress();
      dirichlet_matrix.reinit(full_sparsity_pattern);
      if (boundary_integral_formulation == "collocation")
        neumann_matrix.reinit(full_sparsity_pattern);
      else if (preconditioner_type == "Calderon")
        {
          calderon_hypersingular_matrix.reinit(full_sparsity_pattern);
          calderon_mass_matrix.reinit(full_sparsity_pattern);
        }
    }
  laplace_patch_correction_initialized = false;
  pcout << "re-initialized sparsity patterns and matrices" << std::endl;
  preconditioner_band = 100;
  preconditioner_sparsity_pattern.reinit(this_cpu_set,
                                         mpi_communicator,
                                         (types::global_dof_index)
                                           preconditioner_band);
  is_preconditioner_initialized = false;

  IndexSet this_cpu_set_complex(2 * this_cpu_set.size());
  this_cpu_set_complex.add_indices(this_cpu_set);
  this_cpu_set_complex.add_indices(this_cpu_set, this_cpu_set.size());
  this_cpu_set_complex.compress();

  preconditioner_complex_sparsity_pattern.reinit(this_cpu_set_complex,
                                                 mpi_communicator,
                                                 (types::global_dof_index)
                                                   preconditioner_band);
  is_preconditioner_complex_initialized = false;

  dirichlet_nodes.reinit(this_cpu_set, mpi_communicator);
  neumann_nodes.reinit(this_cpu_set, mpi_communicator);
  robin_nodes.reinit(this_cpu_set, mpi_communicator);
  dirichlet_flags.reinit(this_cpu_set, mpi_communicator);
  neumann_flags.reinit(this_cpu_set, mpi_communicator);
  robin_flags.reinit(this_cpu_set, mpi_communicator);
  compute_dirichlet_and_neumann_dofs_vectors();
  compute_double_nodes_set();


  /* TODO: unused
  // We need a TrilinosWrappers::MPI::Vector to reinit the SparsityPattern for
  // the parallel mass matrices.
  TrilinosWrappers::MPI::Vector helper(vector_this_cpu_set, mpi_communicator);
  IndexSet                      trial_index_set;
  trial_index_set.clear();
  trial_index_set =
    DoFTools::dof_indices_with_subdomain_association(gradient_dh,
                                                     this_mpi_process);
  */
  // This is the only way we could create the SparsityPattern, through the
  // Epetramap of an existing vector.
  vector_sparsity_pattern.reinit(vector_this_cpu_set,
                                 vector_this_cpu_set,
                                 mpi_communicator);
  DoFTools::make_sparsity_pattern(gradient_dh,
                                  vector_sparsity_pattern,
                                  vector_constraints,
                                  true,
                                  this_mpi_process);
  vector_sparsity_pattern.compress();


  hyp_alpha.reinit(this_cpu_set, mpi_communicator);
  C_ij.resize(dim * dim);
  for (unsigned int i = 0; i < dim * dim; ++i)
    C_ij[i].reinit(this_cpu_set, mpi_communicator);
  b_i.resize(dim);
  for (unsigned int i = 0; i < dim; ++i)
    b_i[i].reinit(this_cpu_set, mpi_communicator);
}


template <>
const Quadrature<2>
BEMProblem<3>::get_quasi_singular_quadrature(
  const typename DoFHandler<2, 3>::active_cell_iterator &cell,
  const Mapping<2, 3>                                   &mapping,
  const Point<3>                                        &singularity,
  const Point<2>                                        &ref_projection,
  const unsigned int                                     exponent,
  const unsigned int quadrature_order_override,
  const double       normalized_distance) const
{
  Assert(exponent == 1 || exponent == 2, ExcNotImplemented());
  const unsigned int quadrature_order =
    quadrature_order_override > 0 ?
      quadrature_order_override :
      (exponent == 2 ? quasi_singular_neumann_quadrature_order :
                       quasi_singular_dirichlet_quadrature_order);
  if (near_singular_subdivision &&
      normalized_distance < near_singular_subdivision_ratio)
    {
      const double       safe_distance = std::max(normalized_distance, 1e-12);
      const unsigned int levels =
        std::min(near_singular_subdivision_max_levels,
                 std::max(1u,
                          static_cast<unsigned int>(
                            std::ceil(std::log2(0.5 / safe_distance)))));
      return nested_quasi_singular_quadrature(ref_projection,
                                              quadrature_order,
                                              levels);
    }
  return telles_quadrature(
    cell, mapping, singularity, ref_projection, quadrature_order, exponent);
}

template <>
const Quadrature<1>
BEMProblem<2>::get_quasi_singular_quadrature(
  const typename DoFHandler<1, 2>::active_cell_iterator &cell,
  const Mapping<1, 2>                                   &mapping,
  const Point<2>                                        &singularity,
  const Point<1>                                        &ref_projection,
  const unsigned int                                     exponent,
  const unsigned int quadrature_order_override,
  const double /*normalized_distance*/) const
{
  Assert(exponent == 1 || exponent == 2, ExcNotImplemented());
  const unsigned int quadrature_order =
    quadrature_order_override > 0 ?
      quadrature_order_override :
      (exponent == 2 ? quasi_singular_neumann_quadrature_order :
                       quasi_singular_dirichlet_quadrature_order);
  return telles_quadrature(
    cell, mapping, singularity, ref_projection, quadrature_order, exponent);
}

template <int dim>
std::shared_ptr<Mapping<dim - 1, dim>>
BEMProblem<dim>::make_exact_geometry_mapping()
{
  exact_projectors.clear();
  exact_projector_collection =
    std::make_unique<ExactGeometry::ProjectorCollection<dim - 1, dim>>();
  const Triangulation<dim - 1, dim> &tria = comp_dom.tria;

  // The manifold ids in use, with a vertex of each, and the largest distance
  // of their vertices from the centre of a spherical manifold.
  std::map<types::manifold_id, std::vector<Point<dim>>> vertices;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->manifold_id() != numbers::flat_manifold_id)
      for (const unsigned int v : cell->vertex_indices())
        vertices[cell->manifold_id()].push_back(cell->vertex(v));

  if (exact_geometry_source == "cad")
    {
      if constexpr (dim == 3)
        {
          AssertThrow(!comp_dom.cad_surfaces.empty(),
                      ExcMessage("Exact geometry source = cad needs the "
                                 "Color_i.iges files and Use iges surfaces "
                                 "and curves = true."));
          for (unsigned int i = 0; i < comp_dom.cad_surfaces.size(); ++i)
            {
              auto projector = std::make_unique<OpenCASCADE::ShapeProjector<2>>(
                comp_dom.cad_surfaces[i]);
              projector->set_parametric(exact_geometry_map == "parametric");
              exact_projectors.push_back(std::move(projector));
              exact_projector_collection->add(1 + i, *exact_projectors.back());
              pcout << "Exact geometry: cells of manifold id " << 1 + i
                    << (exact_geometry_map == "parametric" ?
                          " mapped through the parametrization of Color_" :
                          " projected onto Color_")
                    << 1 + i << ".iges" << std::endl;
            }
        }
      else
        AssertThrow(false,
                    ExcMessage("CAD surfaces are available in 3d only."));
    }
  else
    for (const auto &[id, points] : vertices)
      {
        const auto *sphere =
          dynamic_cast<const SphericalManifold<dim - 1, dim> *>(
            &tria.get_manifold(id));
        if (sphere == nullptr)
          {
            pcout << "Exact geometry: cells of manifold id " << id
                  << " stay on the mesh (not a spherical manifold)"
                  << std::endl;
            continue;
          }
        double radius = 0;
        for (const auto &p : points)
          radius += p.distance(sphere->get_center());
        radius /= points.size();
        exact_projectors.push_back(
          std::make_unique<ExactGeometry::SphericalProjector<dim - 1, dim>>(
            sphere->get_center(), radius));
        exact_projector_collection->add(id, *exact_projectors.back());
        pcout << "Exact geometry: cells of manifold id " << id
              << " projected onto the sphere of centre " << sphere->get_center()
              << " and radius " << radius << std::endl;
      }
  pcout << "Exact geometry: projected mapping of degree " << mapping_degree
        << std::endl;
  return std::make_shared<ExactGeometry::ProjectedMapping<dim - 1, dim>>(
    mapping_degree, *exact_projector_collection);
}


template <int dim>
void
BEMProblem<dim>::declare_parameters(ParameterHandler &prm)
{
  // In the solver section, we set
  // all SolverControl
  // parameters. The object will then
  // be fed to the GMRES solver in
  // the solve_system() function.

  prm.enter_subsection("Solver");
  SolverControl::declare_parameters(prm);
  prm.leave_subsection();

  prm.declare_entry("Preconditioner",
                    "ILU",
                    Patterns::Selection(
                      "None|ILU|AMG|Jacobi|BlockJacobi|PatchSchwarz|"
                      "Calderon|CalderonDual"));

  prm.declare_entry("Calderon stabilization", "1.0", Patterns::Double(0.0));
  prm.declare_entry("Calderon dual bubble scaling",
                    "1.0",
                    Patterns::Double(0.0));
  // The sphere/cube split of the right-hand side, removed in the cleanup of
  // 2026-10-27: read, so that older files run, and ignored.
  prm.declare_entry("Galerkin component RHS diagnostics",
                    "false",
                    Patterns::Bool());

  // The energy norm costs a full pass over the panel pairs, comparable to one
  // matrix assembly, so it is off unless a convergence study asks for it.
  prm.declare_entry("Compute energy norm error", "false", Patterns::Bool());

  // The error breakdowns of the sphere-in-box geometry, removed in the
  // cleanup of 2026-10-27: read, so that older files run, and ignored.
  prm.declare_entry("Detailed error diagnostics", "false", Patterns::Bool());

  // How the surface gradient is recovered once the trace and the flux are
  // known. "projection" takes the L2 projection of the gradient reconstructed
  // from the discrete fields; "hypersingular" evaluates the hypersingular
  // boundary integral equation at the support points, which for discontinuous
  // elements are interior to their cell and so avoid the corner ambiguity of
  // the free term that a continuous space would face at vertices.
  // weak_joint, weak_discrete and hypersingular_weak_discrete are removed
  // variants: still accepted by the pattern, so that older files are read,
  // and refused by parse_parameters().
  prm.declare_entry("Gradient recovery",
                    "projection",
                    Patterns::Selection(
                      "projection|weak|weak_joint|weak_discrete|hypersingular|"
                      "hypersingular_weak|hypersingular_weak_discrete"));

  // Accuracy asked of the far-field quadrature. The cost of a regular panel
  // pair grows as the fourth power of the order, and well-separated panels do
  // not need the order the close ones do. Zero keeps the full order
  // everywhere, which is the safe default; 1e-10 is a reasonable setting.
  prm.declare_entry("Galerkin far field tolerance",
                    "0.0",
                    Patterns::Double(0.0));

  // The single layer and the double-layer right-hand side go through the
  // same panel pairs, the same singular and near-singular subdivision and the
  // same quadrature rules; assembled together, these are built once. The
  // result is the same up to rounding.
  prm.declare_entry("Galerkin fused right-hand side", "true", Patterns::Bool());

  // Evaluate the near-field quadrature rules from the cell geometry,
  // reconstructed exactly from the mapping at its Gauss-Lobatto nodes, rather
  // than building FEValues objects for every leaf of the subdivision. The
  // reconstruction is checked against FEValues on every cell.
  prm.declare_entry("Galerkin light geometry", "true", Patterns::Bool());

  prm.declare_entry("Solution method",
                    "Direct",
                    Patterns::Selection("Direct|FMA"));

  prm.declare_entry("Linear solver",
                    "GMRES",
                    Patterns::Selection("GMRES|MUMPS|CG"));

  prm.declare_entry("MUMPS ordering",
                    "QAMD",
                    Patterns::Selection(
                      "automatic|AMD|AMF|SCOTCH|PORD|METIS|QAMD"));

  prm.declare_entry("Boundary integral formulation",
                    "collocation",
                    Patterns::Selection("collocation|galerkin"));

  // The adjoint double layer and hybrid formulation of the Neumann problem
  // was removed in the cleanup of 2026-10-27 (in the git history). Its
  // parameters, down to the hybrid first-kind polishing, are read, so that
  // older files run, and ignored; adjoint_double_layer is refused.
  prm.declare_entry("Galerkin Neumann formulation",
                    "single_layer",
                    Patterns::Selection("single_layer|adjoint_double_layer"));
  prm.declare_entry("Galerkin adjoint continuity",
                    "global",
                    Patterns::Selection("global|material_wise|normal_wise"));
  prm.declare_entry("Galerkin adjoint component weighting",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin adjoint primary material id",
                    "1",
                    Patterns::Integer(0));
  prm.declare_entry("Galerkin adjoint primary material ids",
                    "",
                    Patterns::Anything());
  prm.declare_entry("Galerkin adjoint primary eta scale",
                    "1.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin adjoint secondary eta scale",
                    "1.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin adjoint local eta scaling",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin adjoint dual residual stabilization",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin adjoint dual residual lift",
                    "mass",
                    Patterns::Selection("mass|single_layer|hypersingular"));
  prm.declare_entry("Galerkin adjoint block completion",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin adjoint hybrid system",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin hybrid exact trace diagnostics",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin analytical hypersingular RHS",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin hybrid conforming complement",
                    "false",
                    Patterns::Bool());
  prm.declare_entry("Galerkin hybrid complement mass regularization",
                    "0.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin hybrid local mass regularization",
                    "0.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin hybrid local fluctuation regularization",
                    "0.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin hybrid local jump regularization",
                    "0.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin hybrid local jump h scaling",
                    "true",
                    Patterns::Bool());
  prm.declare_entry("Galerkin hybrid local mass material id",
                    "3",
                    Patterns::Integer(0));
  prm.declare_entry("Galerkin hybrid local mass radius min",
                    "0.35",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin hybrid local mass radius max",
                    "1.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin hybrid first-kind polishing steps",
                    "0",
                    Patterns::Integer(0));
  prm.declare_entry("Galerkin hybrid first-kind polishing relaxation",
                    "0.25",
                    Patterns::Double(0.0, 1.0));

  prm.declare_entry("Kernel type",
                    "laplace",
                    Patterns::Selection("laplace|screened"));

  // manufactured_flux builds the Galerkin right-hand side as <V lambda, phi_i>
  // from the exact flux, through the single-layer kernel. It applies only
  // when the exact solution is known, so it is a convergence-study tool, but
  // it is the right-hand side the Galerkin equations actually require, and it
  // avoids the double layer's 1/r^2 kernel entirely.
  prm.declare_entry("Dirichlet RHS treatment",
                    "nodal",
                    Patterns::Selection(
                      "nodal|exact_regularized|manufactured_flux"));

  // The hypersingular trace correction is removed: its parameters are read,
  // so that older files run, and steps other than 0 are refused.
  prm.declare_entry("Hypersingular correction steps",
                    "0",
                    Patterns::Integer(0));
  prm.declare_entry("Hypersingular correction relaxation",
                    "1.0",
                    Patterns::Double(0.0, 1.0));
  prm.declare_entry("Hypersingular correction maximum relative update",
                    "0.25",
                    Patterns::Double(0.0));
  prm.declare_entry("Enforce Laplace linear patch", "false", Patterns::Bool());

  prm.declare_entry("Screened Kappa", "1.0", Patterns::Double());

  prm.enter_subsection("Quadrature rules");
  {
    prm.declare_entry("Quadrature type",
                      "gauss",
                      Patterns::Selection(
                        QuadratureSelector<(dim - 1)>::get_quadrature_names()));
    prm.declare_entry("Quadrature order", "4", Patterns::Integer());
    prm.declare_entry("Singular quadrature order", "5", Patterns::Integer());
    prm.declare_entry("Galerkin quadrature order", "6", Patterns::Integer(2));
    prm.declare_entry("Galerkin singular quadrature",
                      "sauter_schwab",
                      Patterns::Selection("sauter_schwab|iterated_duffy"));
    prm.declare_entry("Galerkin singular quadrature order",
                      "6",
                      Patterns::Integer(2));
    // The singular rule for pairs of cells that meet with nearly opposite
    // normals (cosine below -0.5), as at a thin trailing edge; zero uses the
    // singular order (GalerkinPanelPairIntegrator::set_folded_singular_order).
    prm.declare_entry("Galerkin singular quadrature order for folded pairs",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry(
      "Galerkin singular quadrature order for selected materials",
      "0",
      Patterns::Integer(0));
    prm.declare_entry("Galerkin singular quadrature material ids",
                      "",
                      Patterns::List(Patterns::Integer(0)));
    // The singular rule for pairs with a skewed cell, one with an interior
    // angle below the given one (degrees); zero uses the singular order.
    prm.declare_entry("Galerkin singular quadrature order for skewed cells",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry("Galerkin skewed cell angle",
                      "30",
                      Patterns::Double(0., 90.));
    // The singular rule for kinked pairs, touching cells whose normals make
    // an angle above the given one (degrees), the two sides of a sharp edge
    // of the surface such as the edge between the tip and the surface of a
    // wing; zero uses the singular order.
    prm.declare_entry("Galerkin singular quadrature order for kinked pairs",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry("Galerkin kinked pair angle",
                      "60",
                      Patterns::Double(0., 180.));
    prm.declare_entry("Galerkin Duffy outer subdivisions",
                      "4",
                      Patterns::Integer(1));
    prm.declare_entry("Galerkin near singular distance ratio",
                      "0.0",
                      Patterns::Double(0.0));
    // The near-singular distance ratio of kinked pairs that do not touch
    // (normals at more than the kinked pair angle), such as the two
    // surfaces of a thin body, where the double layer of one across the
    // other needs it larger; zero, or a value not above the near-singular
    // distance ratio, keeps that ratio
    // (GalerkinPanelPairIntegrator::set_kinked_distance_ratio).
    prm.declare_entry("Galerkin near singular distance ratio for kinked pairs",
                      "0",
                      Patterns::Double(0.));
    prm.declare_entry("Galerkin near singular maximum depth",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry("Galerkin near singular quadrature order",
                      "4",
                      Patterns::Integer(2));
    prm.declare_entry("Quasi singular quadrature order",
                      "5",
                      Patterns::Integer());
    prm.declare_entry("Quasi singular Neumann quadrature order",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry("Quasi singular Dirichlet quadrature order",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry("Adaptive quadrature", "true", Patterns::Bool());
    prm.declare_entry("Adaptive quadrature relative tolerance",
                      "1.e-8",
                      Patterns::Double(0.0));
    prm.declare_entry("Adaptive quadrature absolute tolerance",
                      "1.e-12",
                      Patterns::Double(0.0));
    prm.declare_entry("Adaptive quadrature maximum order",
                      "128",
                      Patterns::Integer(2));
    prm.declare_entry("Adaptive quadrature order growth",
                      "1.5",
                      Patterns::Double(1.01));
    prm.declare_entry("Quasi singular distance ratio",
                      "2.0",
                      Patterns::Double(0.0));
    prm.declare_entry("Near singular subdivision", "false", Patterns::Bool());
    prm.declare_entry("Near singular subdivision ratio",
                      "0.25",
                      Patterns::Double(0.0));
    prm.declare_entry("Near singular subdivision maximum levels",
                      "10",
                      Patterns::Integer(1));
  }
  prm.leave_subsection();

  prm.declare_entry("Mapping Type", "FE", Patterns::Selection("FE|Q"));

  prm.declare_entry("Mapping Q Degree", "1", Patterns::Integer());

  // Integrate over the exact geometry rather than over the cells of the
  // mapping: every cell is mapped onto the surface it approximates by the
  // projection of the points of its MappingQ of the degree above, with the
  // Jacobian of the projection (ExactGeometry::ProjectedMapping). All
  // integrals, collocation and Galerkin, the right-hand sides, the support
  // points and the errors then see the exact geometry; the finite element
  // spaces are those of the mesh. The source of the geometry: "cad" projects
  // the cells of manifold id i onto the surface of Color_i.iges (Use iges
  // surfaces and curves = true), "manifold" onto the analytic manifold
  // attached to the cells (spheres); cells of other manifold ids stay on the
  // mesh. Needs Mapping Type = Q.
  prm.declare_entry("Exact geometry", "false", Patterns::Bool());
  prm.declare_entry("Exact geometry source",
                    "cad",
                    Patterns::Selection("cad|manifold"));
  // How a cell is mapped onto a CAD face: by the closest point of its
  // MappingQ, or through the parametrization S(u, v) of the face, with the
  // parameters of the cell interpolated from those of its vertices and
  // edges (OpenCASCADE::ShapeProjector::set_parametric()). The closest point
  // of a flat cell is singular at the centres of curvature, and its
  // interpolation converges slowly where the surface bends over a length
  // comparable with the cell, as across the nose of a wing; the parametric
  // map is analytic wherever the surface is.
  // The preconditioner of the mixed Galerkin problem: the diagonals of V
  // and W, or operator preconditioning by the Calderon identities, which
  // costs a product with V and one with W per step and keeps the number of
  // steps nearly independent of the mesh.
  prm.declare_entry("Mixed Galerkin preconditioner",
                    "diagonal",
                    Patterns::Selection("diagonal|calderon"));
  // The Dirichlet datum of the mixed Galerkin problem on the Dirichlet part:
  // its L2 projection on the cells of the part, or its interpolant at the
  // nodes, whose error, smooth where the cells change size, the map from the
  // datum to the flux turns into an error of the flux one order lower.
  // Inexact GMRES for the mixed problem through the FMM: once the
  // preconditioned residual is below this fraction of the preconditioned
  // right-hand side, the products use the expansions of the preconditioner;
  // about the solver reduction over their error, which the solver prints.
  // Zero, the default, keeps the full expansions.
  prm.declare_entry("Mixed Galerkin relaxed products below",
                    "0",
                    Patterns::Double(0.));
  prm.declare_entry("Mixed Galerkin Dirichlet datum",
                    "projection",
                    Patterns::Selection("interpolation|projection"));
  // Error indicators of the mixed Galerkin problem, which drive the
  // adaptive refinement when Set Global Refinement is false: none (the
  // Kelly estimator of the potential), the jumps of the conormal derivative
  // of the flux on the Dirichlet part and of the potential on the Neumann
  // part across the edges inside each smooth piece, weighted for the norms
  // of H^{-1/2} and H^{1/2} by the extents of the cell, or the exact errors
  // of a manufactured solution, a ||e||^2 and ||e||^2 / a with a the
  // smallest extent of the cell, for comparison. The marked cells carry the
  // given fraction of the sum of the squares.
  prm.declare_entry("Mixed Galerkin error indicator",
                    "none",
                    Patterns::Selection("none|jumps|exact"));
  prm.declare_entry("Mixed Galerkin refinement fraction",
                    "0.5",
                    Patterns::Double(0., 1.));
  // The norm the indicators measure: the energy norm of the mixed problem,
  // H^{-1/2} for the flux and H^{1/2} for the potential, or the sum of the
  // squared relative L2 errors of the flux and of the potential.
  // The reduction of the preconditioned residual at which GMRES stops in
  // the mixed Galerkin problem; far below the error of the discretisation
  // by default.
  prm.declare_entry("Mixed Galerkin solver reduction",
                    "1e-12",
                    Patterns::Double(0., 1.));
  prm.declare_entry("Mixed Galerkin indicator norm",
                    "energy",
                    Patterns::Selection("energy|L2"));
  prm.declare_entry("Exact geometry map",
                    "projection",
                    Patterns::Selection("projection|parametric"));
  // The exact map of every cell is interpolated once at the Gauss-Lobatto
  // points of a degree n, and the geometry is then evaluated as a MappingQ
  // of degree n, without projections: the cost of a polynomial mapping, at
  // an accuracy that converges exponentially in n. With a positive degree,
  // n is that degree. With zero, n is the lowest degree from that of the
  // mapping up to 10 at which the interpolant deviates from the exact map by
  // less than the tolerance, in points relative to the cell diameter and in
  // surface elements; with zero degree and zero tolerance every point is
  // projected, which is exact but slow on CAD surfaces.
  prm.declare_entry("Exact geometry interpolation degree",
                    "0",
                    Patterns::Integer(0, 11));
  prm.declare_entry("Exact geometry interpolation tolerance",
                    "1e-10",
                    Patterns::Double(0.0));
  // The nodal normals of collocation: the L2 projection of the normals of
  // the mapping, or their values at the nodes.
  prm.declare_entry("Nodal normals",
                    "projection",
                    Patterns::Selection("projection|interpolation"));
  // The data of the mixed Galerkin problem at the points of the exact
  // surface (the CAD surfaces or the spherical manifolds of Exact geometry
  // source) while the computation uses the mesh.
  prm.declare_entry("Boundary data on the exact surface",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Continuos gradient across edges",
                    "true",
                    Patterns::Bool());
  prm.enter_subsection("Scalar Finite Element");
  {
    prm.declare_entry("Finite Element type",
                      "FEQ",
                      Patterns::Selection("FEQ|FEDGQ|FEDGQGauss"));
    prm.declare_entry("Finite element order", "1", Patterns::Integer());
  }
  prm.leave_subsection();

  prm.enter_subsection("Vector Finite Element");
  {
    prm.declare_entry("Finite Element type",
                      "FEQ",
                      Patterns::Selection("FEQ|FEDGQ|FEDGQGauss"));
    prm.declare_entry("Finite element order", "1", Patterns::Integer());
  }
  prm.leave_subsection();


  prm.enter_subsection("Grid Refinement");
  {
    prm.declare_entry("Coarsening threshold", "0.03", Patterns::Double());
    prm.declare_entry("Refinement threshold", "0.3", Patterns::Double());
  }
  prm.leave_subsection();
}

template <int dim>
void
BEMProblem<dim>::parse_parameters(ParameterHandler &prm)
{
  // Variants removed in the cleanup of 2026-10-27, still in the git history:
  // a file that selects one stops here.
  const auto removed = [](const bool selected, const std::string &variant) {
    AssertThrow(!selected,
                ExcMessage(variant +
                           " was removed from this pi-BEM in the cleanup of "
                           "2026-10-27; it is in the git history."));
  };

  prm.enter_subsection("Solver");
  solver_control.parse_parameters(prm);
  prm.leave_subsection();

  preconditioner_type          = prm.get("Preconditioner");
  calderon_stabilization       = prm.get_double("Calderon stabilization");
  calderon_dual_bubble_scaling = prm.get_double("Calderon dual bubble scaling");
  compute_energy_norm          = prm.get_bool("Compute energy norm error");
  gradient_recovery            = prm.get("Gradient recovery");
  removed(gradient_recovery == "weak_joint" ||
            gradient_recovery == "weak_discrete" ||
            gradient_recovery == "hypersingular_weak_discrete",
          "Gradient recovery = " + gradient_recovery);
  galerkin_far_field_tolerance = prm.get_double("Galerkin far field tolerance");
  galerkin_fused_rhs           = prm.get_bool("Galerkin fused right-hand side");
  galerkin_light_geometry      = prm.get_bool("Galerkin light geometry");

  solution_method = prm.get("Solution method");

  linear_solver = prm.get("Linear solver");

  mumps_ordering = prm.get("MUMPS ordering");

  boundary_integral_formulation = prm.get("Boundary integral formulation");
  removed(prm.get("Galerkin Neumann formulation") == "adjoint_double_layer",
          "Galerkin Neumann formulation = adjoint_double_layer (with the "
          "hybrid system)");

  kernel_type = prm.get("Kernel type");

  dirichlet_rhs_treatment = prm.get("Dirichlet RHS treatment");

  removed(prm.get_integer("Hypersingular correction steps") != 0,
          "The hypersingular trace correction (Hypersingular correction "
          "steps)");
  enforce_laplace_linear_patch = prm.get_bool("Enforce Laplace linear patch");

  screened_kappa = prm.get_double("Screened Kappa");

  prm.enter_subsection("Quadrature rules");
  {
    quadrature = std::shared_ptr<Quadrature<dim - 1>>(
      new QuadratureSelector<dim - 1>(prm.get("Quadrature type"),
                                      prm.get_integer("Quadrature order")));
    quadrature_order             = prm.get_integer("Quadrature order");
    singular_quadrature_order    = prm.get_integer("Singular quadrature order");
    galerkin_quadrature_order    = prm.get_integer("Galerkin quadrature order");
    galerkin_singular_quadrature = prm.get("Galerkin singular quadrature");
    galerkin_singular_quadrature_order =
      prm.get_integer("Galerkin singular quadrature order");
    galerkin_folded_singular_quadrature_order =
      prm.get_integer("Galerkin singular quadrature order for folded pairs");
    galerkin_material_singular_quadrature_order = prm.get_integer(
      "Galerkin singular quadrature order for selected materials");
    galerkin_skewed_singular_quadrature_order =
      prm.get_integer("Galerkin singular quadrature order for skewed cells");
    galerkin_skewed_minimum_angle =
      prm.get_double("Galerkin skewed cell angle");
    galerkin_kinked_singular_quadrature_order =
      prm.get_integer("Galerkin singular quadrature order for kinked pairs");
    galerkin_kinked_minimum_angle =
      prm.get_double("Galerkin kinked pair angle");
    galerkin_singular_quadrature_material_ids.clear();
    for (const auto &id : Utilities::split_string_list(
           prm.get("Galerkin singular quadrature material ids")))
      if (!id.empty())
        galerkin_singular_quadrature_material_ids.insert(
          static_cast<types::material_id>(Utilities::string_to_int(id)));
    galerkin_duffy_outer_subdivisions =
      prm.get_integer("Galerkin Duffy outer subdivisions");
    galerkin_near_singular_distance_ratio =
      prm.get_double("Galerkin near singular distance ratio");
    galerkin_kinked_distance_ratio =
      prm.get_double("Galerkin near singular distance ratio for kinked pairs");
    galerkin_near_singular_maximum_depth =
      prm.get_integer("Galerkin near singular maximum depth");
    galerkin_near_singular_quadrature_order =
      prm.get_integer("Galerkin near singular quadrature order");
    const unsigned int legacy_quasi_singular_order =
      prm.get_integer("Quasi singular quadrature order");
    quasi_singular_neumann_quadrature_order =
      prm.get_integer("Quasi singular Neumann quadrature order");
    quasi_singular_dirichlet_quadrature_order =
      prm.get_integer("Quasi singular Dirichlet quadrature order");
    if (quasi_singular_neumann_quadrature_order == 0)
      quasi_singular_neumann_quadrature_order = legacy_quasi_singular_order;
    if (quasi_singular_dirichlet_quadrature_order == 0)
      quasi_singular_dirichlet_quadrature_order = legacy_quasi_singular_order;
    adaptive_quadrature = prm.get_bool("Adaptive quadrature");
    adaptive_quadrature_relative_tolerance =
      prm.get_double("Adaptive quadrature relative tolerance");
    adaptive_quadrature_absolute_tolerance =
      prm.get_double("Adaptive quadrature absolute tolerance");
    adaptive_quadrature_max_order =
      prm.get_integer("Adaptive quadrature maximum order");
    adaptive_quadrature_order_growth =
      prm.get_double("Adaptive quadrature order growth");
    quasi_singular_distance_ratio =
      prm.get_double("Quasi singular distance ratio");
    near_singular_subdivision = prm.get_bool("Near singular subdivision");
    near_singular_subdivision_ratio =
      prm.get_double("Near singular subdivision ratio");
    near_singular_subdivision_max_levels =
      prm.get_integer("Near singular subdivision maximum levels");
    AssertThrow(adaptive_quadrature_max_order >= singular_quadrature_order &&
                  adaptive_quadrature_max_order >=
                    quasi_singular_neumann_quadrature_order &&
                  adaptive_quadrature_max_order >=
                    quasi_singular_dirichlet_quadrature_order,
                ExcMessage("Adaptive quadrature maximum order must not be "
                           "smaller than an initial quadrature order."));
  }
  prm.leave_subsection();

  if (boundary_integral_formulation == "galerkin")
    {
      AssertThrow(dim == 3,
                  ExcMessage("Galerkin assembly currently supports 3D only."));
      // Solution method = FMA applies the single layer through
      // GalerkinFMMOperator; the rest of the Galerkin machinery, and the
      // dense path, are unchanged.
      AssertThrow(solution_method == "Direct" ||
                    ((kernel_type == "laplace" || kernel_type == "screened") &&
                     linear_solver == "CG" &&
                     (preconditioner_type == "None" ||
                      preconditioner_type == "Jacobi" ||
                      preconditioner_type == "BlockJacobi" ||
                      preconditioner_type == "PatchSchwarz")),
                  ExcMessage("The Galerkin FMM (Solution method = FMA) "
                             "supports the Laplace and screened kernels, the "
                             "single-layer "
                             "formulation, Linear solver = CG and "
                             "Preconditioner = None, Jacobi, BlockJacobi or "
                             "PatchSchwarz."));
      AssertThrow(preconditioner_type != "PatchSchwarz" ||
                    solution_method == "FMA",
                  ExcMessage("Preconditioner = PatchSchwarz builds its blocks "
                             "from the near field of the Galerkin FMM "
                             "(Solution method = FMA)."));
      // The screened kernel exp(-kappa r)/(4 pi r) enters the single-layer
      // formulation, its right-hand sides, the energy functional and the
      // gradient recovery. The second-kind and Calderon paths use the weak
      // hypersingular operator, whose screened form is not implemented.
      AssertThrow(kernel_type == "laplace" ||
                    (preconditioner_type != "Calderon" &&
                     preconditioner_type != "CalderonDual"),
                  ExcMessage("The screened kernel is supported by the "
                             "single-layer Galerkin formulation without the "
                             "Calderon preconditioner."));
      AssertThrow(dirichlet_rhs_treatment == "nodal" ||
                    dirichlet_rhs_treatment == "exact_regularized" ||
                    dirichlet_rhs_treatment == "manufactured_flux",
                  ExcMessage("Unsupported Galerkin Dirichlet RHS treatment."));
      AssertThrow(!enforce_laplace_linear_patch,
                  ExcMessage("The collocation linear-patch correction is not "
                             "used by the Galerkin formulation."));
      if (preconditioner_type == "Calderon" ||
          preconditioner_type == "CalderonDual")
        {
          AssertThrow(linear_solver == "GMRES",
                      ExcMessage("The Calderon operator preconditioner "
                                 "requires Linear solver = GMRES."));
        }
    }
  else
    {
      AssertThrow(linear_solver != "CG",
                  ExcMessage("The collocation operator is not symmetric; the "
                             "conjugate gradient path requires Boundary "
                             "integral formulation = galerkin."));
    }

  mapping_type                   = prm.get("Mapping Type");
  mapping_degree                 = prm.get_integer("Mapping Q Degree");
  exact_geometry                 = prm.get_bool("Exact geometry");
  exact_geometry_source          = prm.get("Exact geometry source");
  exact_geometry_map             = prm.get("Exact geometry map");
  mixed_galerkin_preconditioner  = prm.get("Mixed Galerkin preconditioner");
  mixed_galerkin_dirichlet_datum = prm.get("Mixed Galerkin Dirichlet datum");
  mixed_galerkin_relaxation =
    prm.get_double("Mixed Galerkin relaxed products below");
  mixed_galerkin_error_indicator = prm.get("Mixed Galerkin error indicator");
  mixed_galerkin_refinement_fraction =
    prm.get_double("Mixed Galerkin refinement fraction");
  mixed_galerkin_indicator_norm = prm.get("Mixed Galerkin indicator norm");
  mixed_galerkin_solver_reduction =
    prm.get_double("Mixed Galerkin solver reduction");
  exact_geometry_degree =
    prm.get_integer("Exact geometry interpolation degree");
  exact_geometry_tolerance =
    prm.get_double("Exact geometry interpolation tolerance");
  nodal_normals = prm.get("Nodal normals");
  boundary_data_on_exact_surface =
    prm.get_bool("Boundary data on the exact surface");
  AssertThrow(!exact_geometry || mapping_type == "Q",
              ExcMessage("Exact geometry needs Mapping Type = Q."));
  continuos_gradient = prm.get_bool("Continuos gradient across edges");


  prm.enter_subsection("Scalar Finite Element");
  {
    scalar_fe_type  = prm.get("Finite Element type");
    scalar_fe_order = prm.get_integer("Finite element order");
  }
  prm.leave_subsection();

  prm.enter_subsection("Vector Finite Element");
  {
    vector_fe_type  = prm.get("Finite Element type");
    vector_fe_order = prm.get_integer("Finite element order");
  }
  prm.leave_subsection();

  if (boundary_integral_formulation == "galerkin")
    {
      AssertThrow(scalar_fe_type == vector_fe_type &&
                    scalar_fe_order == vector_fe_order,
                  ExcMessage("The current Galerkin implementation requires "
                             "matching scalar and vector finite element "
                             "spaces for normal and gradient recovery."));
      if (preconditioner_type == "Calderon")
        AssertThrow(scalar_fe_type == "FEQ",
                    ExcMessage("The weak hypersingular Calderon operator "
                               "requires Scalar FE = FEQ."));
      if (preconditioner_type == "CalderonDual")
        AssertThrow(scalar_fe_type == "FEDGQ" && scalar_fe_order == 0,
                    ExcMessage("CalderonDual requires Scalar FE = FEDGQ of "
                               "order 0."));
    }

  prm.enter_subsection("Grid Refinement");
  {
    coarsening_threshold = prm.get_double("Coarsening threshold");
    refinement_threshold = prm.get_double("Refinement threshold");
  }
  prm.leave_subsection();
}

template <int dim>
void
BEMProblem<dim>::compute_dirichlet_and_neumann_dofs_vectors()
{
  can_determine_phi = false;

  Vector<double> non_partitioned_dirichlet_nodes(dh.n_dofs());
  Vector<double> non_partitioned_neumann_nodes(dh.n_dofs());
  Vector<double> non_partitioned_robin_nodes(dh.n_dofs());

  Vector<double> non_partitioned_dirichlet_flags(dh.n_dofs());
  Vector<double> non_partitioned_neumann_flags(dh.n_dofs());
  Vector<double> non_partitioned_robin_flags(dh.n_dofs());

  // defaulting to neumann
  // vector_shift(non_partitioned_neumann_nodes, 1.);
  std::vector<types::global_dof_index> dofs(fe->dofs_per_cell);
  unsigned int                         local_can_determine_phi = 0;

  // Every process looks at all the cells around its dofs, so that the type of
  // a dof does not depend on the partition: a dof shared by cells of
  // different types is a Dirichlet dof if one of them is Dirichlet, else a
  // Neumann dof if one of them is Neumann, else a Robin dof.
  for (const auto &cell : dh.active_cell_iterators())
    {
      const auto has_material = [&](const std::vector<unsigned int> &ids) {
        return std::find(ids.begin(), ids.end(), cell->material_id()) !=
               ids.end();
      };
      cell->get_dof_indices(dofs);
      if (has_material(comp_dom.dirichlet_boundary_ids))
        {
          for (auto i : dofs)
            {
              non_partitioned_dirichlet_flags(i) = 1;

              // mark dofs on masking vectors
              non_partitioned_dirichlet_nodes(i) = 1;
              non_partitioned_neumann_nodes(i)   = 0;
              non_partitioned_robin_nodes(i)     = 0;
            }

          local_can_determine_phi = 1;
        }
      else if (has_material(comp_dom.neumann_boundary_ids))
        {
          for (auto i : dofs)
            {
              non_partitioned_neumann_flags(i) = 1;

              // mark dofs on masking vectors
              if (!non_partitioned_dirichlet_nodes(i))
                {
                  non_partitioned_neumann_nodes(i) = 1;
                  non_partitioned_robin_nodes(i)   = 0;
                }
            }
        }
      else
        {
          Assert(has_material(comp_dom.robin_boundary_ids), ExcInternalError());
          for (auto i : dofs)
            {
              non_partitioned_robin_flags(i) = 1;

              // mark dofs on masking vectors
              if (!non_partitioned_dirichlet_nodes(i) &&
                  !non_partitioned_neumann_nodes(i))
                non_partitioned_robin_nodes(i) = 1;
            }

          local_can_determine_phi = 1;
        }
    }

  for (auto i : this_cpu_set)
    {
      dirichlet_nodes(i) = non_partitioned_dirichlet_nodes(i);
      neumann_nodes(i)   = non_partitioned_neumann_nodes(i);
      robin_nodes(i)     = non_partitioned_robin_nodes(i);

      dirichlet_flags(i) = non_partitioned_dirichlet_flags(i);
      neumann_flags(i)   = non_partitioned_neumann_flags(i);
      robin_flags(i)     = non_partitioned_robin_flags(i);
    }

  {
    Vector<double> localized_dirichlet(dirichlet_nodes);
    pcout << "Number of Dirichlet dofs: "
          << (int)(localized_dirichlet.size() *
                   localized_dirichlet.mean_value())
          << std::endl;
    Vector<double> localized_neumann(neumann_nodes);
    pcout << "Number of Neumann dofs: "
          << (int)(localized_neumann.size() * localized_neumann.mean_value())
          << std::endl;
    Vector<double> localized_robin(robin_nodes);
    pcout << "Number of Robin dofs: "
          << (int)(localized_robin.size() * localized_robin.mean_value())
          << std::endl;

    Vector<double> localized_dirichlet2(dirichlet_flags);
    pcout << "Number of Dirichlet flags: "
          << (int)(localized_dirichlet2.size() *
                   localized_dirichlet2.mean_value())
          << std::endl;
    Vector<double> localized_neumann2(neumann_flags);
    pcout << "Number of Neumann flags: "
          << (int)(localized_neumann2.size() * localized_neumann2.mean_value())
          << std::endl;
    Vector<double> localized_robin2(robin_flags);
    pcout << "Number of Robin flags: "
          << (int)(localized_robin2.size() * localized_robin2.mean_value())
          << std::endl;
  }

  unsigned int global_can_determine_phi;
  MPI_Allreduce(&local_can_determine_phi,
                &global_can_determine_phi,
                1,
                MPI_UNSIGNED,
                MPI_MAX,
                mpi_communicator);

  if (global_can_determine_phi > 0)
    {
      can_determine_phi = true;
    }
}

template <int dim>
void
BEMProblem<dim>::compute_double_nodes_set()
{
  double tol = 1e-10;
  double_nodes_set.clear();
  double_nodes_set.resize(dh.n_dofs());
  std::vector<Point<dim>> support_points(dh.n_dofs());

  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  edge_set.clear();
  edge_set.set_size(dh.n_dofs());
  for (const auto &cell : dh.active_cell_iterators())
    {
      for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell; ++f)
        {
          if (cell->face(f)->at_boundary())
            {
              cell->face(f)->get_dof_indices(face_dofs);
              edge_set.add_indices(face_dofs.cbegin(), face_dofs.cend());
            }
        }
    }
  edge_set.compress();

  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    {
      double_nodes_set[i].insert(i);
    }
  for (auto i : edge_set)
    {
      auto jiter = edge_set.at(i);
      ++jiter;
      for (; jiter != edge_set.end(); ++jiter)
        {
          const auto j = *jiter;
          if (support_points[i].distance_square(support_points[j]) <
              (tol * tol))
            {
              double_nodes_set[i].insert(j);
              double_nodes_set[j].insert(i);
            }
        }
    }
}

template <int dim>
void
BEMProblem<dim>::compute_reordering_vectors()
{
  original_to_sub_wise.resize(dh.n_dofs());
  sub_wise_to_original.resize(dh.n_dofs());
  vec_original_to_sub_wise.resize(gradient_dh.n_dofs());
  vec_sub_wise_to_original.resize(gradient_dh.n_dofs());

  DoFRenumbering::compute_subdomain_wise(original_to_sub_wise, dh);
  DoFRenumbering::compute_subdomain_wise(vec_original_to_sub_wise, gradient_dh);

  for (types::global_dof_index i = 0; i < gradient_dh.n_dofs(); ++i)
    {
      if (i < dh.n_dofs())
        {
          sub_wise_to_original[original_to_sub_wise[i]] = i;
        }
      vec_sub_wise_to_original[vec_original_to_sub_wise[i]] = i;
    }
}


template <int dim>
void
BEMProblem<dim>::assemble_system()
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  if (boundary_integral_formulation == "galerkin")
    {
      assemble_galerkin_system();
      return;
    }

  pcout << "(Directly) Assembling system matrices" << std::endl;

  neumann_matrix   = 0;
  dirichlet_matrix = 0;

  // Counted by the threads that integrate the rows.
  CollocationRuleStatistics statistics;


  // Next, we initialize an FEValues
  // object with the quadrature
  // formula for the integration of
  // the kernel in non singular
  // cells. This quadrature is
  // selected with the parameter
  // file, and needs to be quite
  // precise, since the functions we
  // are integrating are not
  // polynomial functions.
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  pcout << fe->dofs_per_cell << " " << std::endl;
  // Unlike in finite element
  // methods, if we use a collocation
  // boundary element method, then in
  // each assembly loop we only
  // assemble the information that
  // refers to the coupling between
  // one degree of freedom (the
  // degree associated with support
  // point $i$) and the current
  // cell. This is done using a
  // vector of fe->dofs_per_cell
  // elements, which will then be
  // distributed to the matrix in the
  // global row $i$. The following
  // object will hold this
  // information:
  // The rows of this process: for every cell they are integrated in
  // parallel (MultithreadInfo::n_threads()), each into its own vectors, and
  // then added to the matrices one after the other.
  const std::vector<types::global_dof_index> rows(this_cpu_set.begin(),
                                                  this_cpu_set.end());
  std::vector<Vector<double>>                neumann_rows(rows.size(),
                                           Vector<double>(fe->dofs_per_cell)),
    dirichlet_rows(rows.size(), Vector<double>(fe->dofs_per_cell));

  // Now that we have checked that
  // the number of vertices is equal
  // to the number of degrees of
  // freedom, we construct a vector
  // of support points which will be
  // used in the local integrations:
  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);


  // After doing so, we can start the
  // integration loop over all cells,
  // where we first initialize the
  // FEValues object and get the
  // values of $\mathbf{\tilde v}$ at
  // the quadrature points (this
  // vector field should be constant,
  // but it doesn't hurt to be more
  // general):


  for (const auto &cell : dh.active_cell_iterators())
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      // We then form the integral over
      // the current cell for all
      // degrees of freedom (note that
      // this includes degrees of
      // freedom not located on the
      // current cell, a deviation from
      // the usual finite element
      // integrals). The integral that
      // we need to perform is singular
      // if one of the local degrees of
      // freedom is the same as the
      // support point $i$. A the
      // beginning of the loop we
      // therefore check wether this is
      // the case, and we store which
      // one is the singular index.
      // We do the same for quasi singular cells.
      const auto integrate_rows = [&](const std::size_t begin,
                                      const std::size_t end) {
        for (std::size_t r = begin; r < end; ++r)
          {
            const types::global_dof_index i = rows[r];
            integrate_collocation_rows(cell,
                                       local_dof_indices,
                                       i,
                                       support_points,
                                       &fe_v,
                                       neumann_rows[r],
                                       dirichlet_rows[r],
                                       statistics);
          } // end loop on rows
      };
      parallel::apply_to_subranges(std::size_t(0),
                                   rows.size(),
                                   integrate_rows,
                                   16);

      // Finally, we need to add
      // the contributions of the
      // current cell to the
      // global matrix.
      for (std::size_t r = 0; r < rows.size(); ++r)
        for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
          {
            neumann_matrix.add(rows[r],
                               local_dof_indices[j],
                               neumann_rows[r](j));
            dirichlet_matrix.add(rows[r],
                                 local_dof_indices[j],
                                 dirichlet_rows[r](j));
          }
    } // end loop on cells

  // The second part of the integral
  // operator is the term
  // $\alpha(\mathbf{x}_i)
  // \phi_j(\mathbf{x}_i)$. Since we
  // use a collocation scheme,
  // $\phi_j(\mathbf{x}_i)=\delta_{ij}$
  // and the corresponding matrix is
  // a diagonal one with entries
  // equal to $\alpha(\mathbf{x}_i)$.

  // One quick way to compute this
  // diagonal matrix of the solid
  // angles, is to use the Neumann
  // matrix itself. It is enough to
  // multiply the matrix with a
  // vector of elements all equal to
  // -1, to get the diagonal matrix
  // of the alpha angles, or solid
  // angles (see the formula in the
  // introduction for this). The
  // result is then added back onto
  // the system matrix object to
  // yield the final form of the
  // matrix:

  /*
    pcout<<"Neumann"<<std::endl;
    for (unsigned int i = 0; i < dh.n_dofs(); i++)
        {
        if (this_cpu_set.is_element(i))
           {
           pcout<<this_mpi_process<<" *** ";
           for (unsigned int j = 0; j < dh.n_dofs(); j++)
               {
               pcout<<neumann_matrix(i,j)<<" ";
               }
           pcout<<std::endl;
           }
        }


    pcout<<"Dirichlet"<<std::endl;
    for (unsigned int i = 0; i < dh.n_dofs(); i++)
        {
        if (this_cpu_set.is_element(i))
           {
           pcout<<this_mpi_process<<" *** ";
           for (unsigned int j = 0; j < dh.n_dofs(); j++)
               {
               pcout<<dirichlet_matrix(i,j)<<" ";
               }
           pcout<<std::endl;
           }
        }
        //*/
  if (adaptive_quadrature)
    {
      const unsigned int total_singular =
        Utilities::MPI::sum(statistics.adaptive_singular_interactions.load(),
                            mpi_communicator);
      const unsigned int total_quasi_neumann = Utilities::MPI::sum(
        statistics.adaptive_quasi_neumann_interactions.load(),
        mpi_communicator);
      const unsigned int total_quasi_dirichlet = Utilities::MPI::sum(
        statistics.adaptive_quasi_dirichlet_interactions.load(),
        mpi_communicator);
      const unsigned int total_unconverged_singular_neumann =
        Utilities::MPI::sum(
          statistics.adaptive_unconverged_singular_neumann.load(),
          mpi_communicator);
      const unsigned int total_unconverged_singular_dirichlet =
        Utilities::MPI::sum(
          statistics.adaptive_unconverged_singular_dirichlet.load(),
          mpi_communicator);
      const unsigned int total_unconverged_quasi_neumann = Utilities::MPI::sum(
        statistics.adaptive_unconverged_quasi_neumann.load(), mpi_communicator);
      const unsigned int total_unconverged_quasi_dirichlet =
        Utilities::MPI::sum(
          statistics.adaptive_unconverged_quasi_dirichlet.load(),
          mpi_communicator);
      const unsigned int largest_order =
        Utilities::MPI::max(statistics.adaptive_largest_order_used.load(),
                            mpi_communicator);
      pcout << "Adaptive quadrature interactions: singular " << total_singular
            << ", quasi Neumann " << total_quasi_neumann << ", quasi Dirichlet "
            << total_quasi_dirichlet << std::endl
            << "Adaptive quadrature largest order: " << largest_order
            << ", unconverged singular-N/singular-D/quasi-N/quasi-D: "
            << total_unconverged_singular_neumann << '/'
            << total_unconverged_singular_dirichlet << '/'
            << total_unconverged_quasi_neumann << '/'
            << total_unconverged_quasi_dirichlet << std::endl;
    }
  pcout << "done assembling system matrices" << std::endl;
  // std::cout<<"printing Neumann Matrix"<<std::endl;
  // for(unsigned int i=0; i<dh.n_dofs(); ++i)
  // {
  //   for(unsigned int j=0; j<dh.n_dofs(); ++j)
  //     std::cout<<neumann_matrix(i,j)<<" ";
  //   std::cout<<std::endl;
  // }
  // std::cout<<"printing Dirichlet Matrix"<<std::endl;
  // for(unsigned int i=0; i<dh.n_dofs(); ++i)
  // {
  //   for(unsigned int j=0; j<dh.n_dofs(); ++j)
  //     std::cout<<dirichlet_matrix(i,j)<<" ";
  //   std::cout<<std::endl;
  // }
}

template <int dim>
typename BEMProblem<dim>::CollocationRule
BEMProblem<dim>::integrate_collocation_rows(
  const typename DoFHandler<dim - 1, dim>::active_cell_iterator &cell,
  const std::vector<types::global_dof_index> &local_dof_indices,
  const types::global_dof_index               i,
  const std::vector<Point<dim>>              &support_points,
  const FEValues<dim - 1, dim>               *regular_values,
  Vector<double>                             &local_neumann_matrix_row_i,
  Vector<double>                             &local_dirichlet_matrix_row_i,
  CollocationRuleStatistics                  &statistics) const
{
  Point<dim> D;
  double     s;
  local_neumann_matrix_row_i   = 0;
  local_dirichlet_matrix_row_i = 0;
  Point<dim>     singularity   = support_points[i];
  Point<dim - 1> ref_projection(0.0, 0.0);
  bool           is_singular    = false;
  unsigned int   singular_index = numbers::invalid_unsigned_int;

  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
    // if(local_dof_indices[j] == i)
    if (double_nodes_set[i].count(local_dof_indices[j]) > 0)
      {
        singular_index = j;
        is_singular    = true;
        break;
      }

  bool   is_quasi_singular = false;
  double distance_ratio    = std::numeric_limits<double>::infinity();
  if (is_singular == false)
    {
      Point<dim>   singularity    = support_points[i];
      const double dist_to_center = (singularity - cell->center()).norm();
      const double h              = cell->diameter();
      if (dist_to_center <= (quasi_singular_distance_ratio + 1.0) * h)
        {
          QuasiSingularKernelIntegral<dim> qski(cell,
                                                *fe,
                                                *mapping,
                                                singularity);
          ref_projection            = qski.get_closest_reference_point();
          const double dist_to_cell = qski.min_distance;

          distance_ratio = dist_to_cell / h;
          if (distance_ratio < quasi_singular_distance_ratio)
            is_quasi_singular = true;
        }
    }

  // We then perform the
  // integral. If the index $i$
  // is not one of the local
  // degrees of freedom, we
  // simply have to add the
  // single layer terms to the
  // right hand side, and the
  // double layer terms to the
  // matrix:
  if ((is_singular == false) && (is_quasi_singular == false))
    {
      if (regular_values == nullptr)
        return CollocationRule::regular;
      const FEValues<dim - 1, dim>  &fe_v        = *regular_values;
      const std::vector<Point<dim>> &q_points    = fe_v.get_quadrature_points();
      const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();
      const unsigned int                 n_q_points = fe_v.n_quadrature_points;
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const Tensor<1, dim> R = q_points[q] - support_points[i];
          if (kernel_type == "laplace")
            LaplaceKernel::kernels(R, D, s);
          else if (kernel_type == "screened")
            ScreenedKernel::kernels(R, D, s, screened_kappa);
          else
            AssertThrow(false,
                        ExcMessage("Unknown kernel type: " + kernel_type));
          AssertThrow(std::isfinite(s) && is_finite_tensor(D),
                      ExcMessage("Non-finite screened kernel in regular "
                                 "assembly."));
          // if(support_points[i][0]==0.25&&support_points[i][1]==0.25)
          //   pcout<<"D "<<D<<" s "<<s<<" , ";
          for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
            {
              local_neumann_matrix_row_i(j) +=
                ((D * normals[q]) * fe_v.shape_value(j, q) * fe_v.JxW(q));
              local_dirichlet_matrix_row_i(j) +=
                (s * fe_v.shape_value(j, q) * fe_v.JxW(q));
            }
        }
    }
  else if (is_quasi_singular == true)
    {
      const auto integrate_quasi_singular =
        [&](const unsigned int exponent,
            const unsigned int order) -> Vector<double> {
        const Quadrature<dim - 1> transformed_quadrature =
          get_quasi_singular_quadrature(cell,
                                        *mapping,
                                        singularity,
                                        ref_projection,
                                        exponent,
                                        order,
                                        distance_ratio);
        AssertThrow(transformed_quadrature.size() > 0,
                    ExcMessage("Empty quasi-singular quadrature."));
        FEValues<dim - 1, dim> transformed_fe_values(
          *mapping,
          *fe,
          transformed_quadrature,
          update_values | update_normal_vectors | update_quadrature_points |
            update_JxW_values);
        transformed_fe_values.reinit(cell);

        Vector<double> result(fe->dofs_per_cell);
        const auto    &transformed_points =
          transformed_fe_values.get_quadrature_points();
        const auto &transformed_normals =
          transformed_fe_values.get_normal_vectors();
        for (unsigned int q = 0; q < transformed_quadrature.size(); ++q)
          {
            const Tensor<1, dim> R = transformed_points[q] - support_points[i];
            if (kernel_type == "laplace")
              LaplaceKernel::kernels(R, D, s);
            else if (kernel_type == "screened")
              ScreenedKernel::kernels(R, D, s, screened_kappa);
            else
              AssertThrow(false,
                          ExcMessage("Unknown kernel type: " + kernel_type));
            AssertThrow(std::isfinite(s) && is_finite_tensor(D),
                        ExcMessage("Non-finite quasi-singular kernel."));
            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
              result[j] += (exponent == 2 ? D * transformed_normals[q] : s) *
                           transformed_fe_values.shape_value(j, q) *
                           transformed_fe_values.JxW(q);
          }
        return result;
      };

      const auto adapt_quasi_singular =
        [&](const unsigned int exponent,
            const unsigned int initial_order) -> Vector<double> {
        Vector<double> result =
          integrate_quasi_singular(exponent, initial_order);
        QuadratureConvergenceState convergence_state;
        unsigned int               used_order = initial_order;
        bool                       converged  = !adaptive_quadrature;
        while (adaptive_quadrature &&
               used_order < adaptive_quadrature_max_order)
          {
            const unsigned int next_order =
              next_adaptive_order(used_order,
                                  adaptive_quadrature_max_order,
                                  adaptive_quadrature_order_growth);
            Vector<double> refined =
              integrate_quasi_singular(exponent, next_order);
            converged = quadrature_vectors_converged(
              result,
              refined,
              adaptive_quadrature_relative_tolerance,
              adaptive_quadrature_absolute_tolerance,
              convergence_state);
            result.swap(refined);
            used_order = next_order;
            if (converged)
              break;
          }
        statistics.raise_largest_order(used_order);
        if (adaptive_quadrature && !converged)
          {
            if (exponent == 2)
              ++statistics.adaptive_unconverged_quasi_neumann;
            else
              ++statistics.adaptive_unconverged_quasi_dirichlet;
          }
        return result;
      };

      local_neumann_matrix_row_i +=
        adapt_quasi_singular(2, quasi_singular_neumann_quadrature_order);
      ++statistics.adaptive_quasi_neumann_interactions;
      local_dirichlet_matrix_row_i +=
        adapt_quasi_singular(1, quasi_singular_dirichlet_quadrature_order);
      ++statistics.adaptive_quasi_dirichlet_interactions;
    }
  else
    {
      // Now we treat the more
      // delicate case. If we
      // are here, this means
      // that the cell that
      // runs on the $j$ index
      // contains
      // support_point[i]. In
      // this case both the
      // single and the double
      // layer potential are
      // singular, and they
      // require special
      // treatment.
      //
      // Whenever the
      // integration is
      // performed with the
      // singularity inside the
      // given cell, then a
      // special quadrature
      // formula is used that
      // allows one to
      // integrate arbitrary
      // functions against a
      // singular weight on the
      // reference cell.
      // Notice that singular
      // integration requires a
      // careful selection of
      // the quadrature
      // rules. In particular
      // the deal.II library
      // provides quadrature
      // rules which are
      // taylored for
      // logarithmic
      // singularities
      // (QGaussLog,
      // QGaussLogR), as well
      // as for 1/R
      // singularities
      // (QGaussOneOverR).
      //
      // Singular integration
      // is typically obtained
      // by constructing
      // weighted quadrature
      // formulas with singular
      // weights, so that it is
      // possible to write
      //
      // \f[
      //   \int_K f(x) s(x) dx = \sum_{i=1}^N w_i f(q_i)
      // \f]
      //
      // where $s(x)$ is a given
      // singularity, and the weights
      // and quadrature points
      // $w_i,q_i$ are carefully
      // selected to make the formula
      // above an equality for a
      // certain class of functions
      // $f(x)$.
      //
      // In all the finite
      // element examples we
      // have seen so far, the
      // weight of the
      // quadrature itself
      // (namely, the function
      // $s(x)$), was always
      // constantly equal to 1.
      // For singular
      // integration, we have
      // two choices: we can
      // use the definition
      // above, factoring out
      // the singularity from
      // the integrand (i.e.,
      // integrating $f(x)$
      // with the special
      // quadrature rule), or
      // we can ask the
      // quadrature rule to
      // "normalize" the
      // weights $w_i$ with
      // $s(q_i)$:
      //
      // \f[
      //   \int_K f(x) s(x) dx =
      //   \int_K g(x) dx = \sum_{i=1}^N \frac{w_i}{s(q_i)} g(q_i)
      // \f]
      //
      // We use this second
      // option, through the @p
      // factor_out_singularity
      // parameter of both
      // QGaussLogR and
      // QGaussOneOverR.
      //
      // These integrals are
      // somewhat delicate,
      // especially in two
      // dimensions, due to the
      // transformation from
      // the real to the
      // reference cell, where
      // the variable of
      // integration is scaled
      // with the determinant
      // of the transformation.
      //
      // In two dimensions this
      // process does not
      // result only in a
      // factor appearing as a
      // constant factor on the
      // entire integral, but
      // also on an additional
      // integral alltogether
      // that needs to be
      // evaluated:
      //
      // \f[
      //  \int_0^1 f(x)\ln(x/\alpha) dx =
      //  \int_0^1 f(x)\ln(x) dx - \int_0^1 f(x) \ln(\alpha) dx.
      // \f]
      //
      // This process is taken care of by
      // the constructor of the QGaussLogR
      // class, which adds additional
      // quadrature points and weights to
      // take into consideration also the
      // second part of the integral.
      //
      // A similar reasoning
      // should be done in the
      // three dimensional
      // case, since the
      // singular quadrature is
      // taylored on the
      // inverse of the radius
      // $r$ in the reference
      // cell, while our
      // singular function
      // lives in real space,
      // however in the three
      // dimensional case
      // everything is simpler
      // because the
      // singularity scales
      // linearly with the
      // determinant of the
      // transformation. This
      // allows us to build the
      // singular two
      // dimensional quadrature
      // rules once and for all
      // outside the loop over
      // all cells, using only
      // a pointer where needed.
      //
      // Notice that in one
      // dimensional
      // integration this is
      // not possible, since we
      // need to know the
      // scaling parameter for
      // the quadrature, which
      // is not known a
      // priori. Here, the
      // quadrature rule itself
      // depends also on the
      // size of the current
      // cell. For this reason,
      // it is necessary to
      // create a new
      // quadrature for each
      // singular
      // integration. Since we
      // create it using the
      // new operator of C++,
      // we also need to
      // destroy it using the
      // dual of new:
      // delete. This is done
      // at the end, and only
      // if dim == 2.
      //
      // Putting all this into a
      // dimension independent
      // framework requires a little
      // trick. The problem is that,
      // depending on dimension, we'd
      // like to either assign a
      // QGaussLogR<1> or a
      // QGaussOneOverR<2> to a
      // Quadrature<dim-1>. C++
      // doesn't allow this right
      // away, and neither is a
      // static_cast
      // possible. However, we can
      // attempt a dynamic_cast: the
      // implementation will then
      // look up at run time whether
      // the conversion is possible
      // (which we <em>know</em> it
      // is) and if that isn't the
      // case simply return a null
      // pointer. To be sure we can
      // then add a safety check at
      // the end:
      Assert(singular_index != numbers::invalid_unsigned_int,
             ExcInternalError());

      struct SingularRows
      {
        Vector<double> neumann;
        Vector<double> dirichlet;

        explicit SingularRows(const unsigned int size)
          : neumann(size)
          , dirichlet(size)
        {}
      };

      const auto integrate_singular =
        [&](const unsigned int order) -> SingularRows {
        Quadrature<dim - 1> singular_quadrature;
        if constexpr (dim == 3)
          singular_quadrature =
            QSplit<2>(QDuffy(order, 1.0),
                      fe->get_unit_support_points()[singular_index]);
        else
          singular_quadrature =
            QTelles<1>(order, fe->get_unit_support_points()[singular_index]);

        FEValues<dim - 1, dim> singular_fe_values(*mapping,
                                                  *fe,
                                                  singular_quadrature,
                                                  update_values |
                                                    update_normal_vectors |
                                                    update_quadrature_points |
                                                    update_JxW_values);
        singular_fe_values.reinit(cell);

        SingularRows result(fe->dofs_per_cell);
        const auto  &singular_normals = singular_fe_values.get_normal_vectors();
        const auto  &singular_points =
          singular_fe_values.get_quadrature_points();
        for (unsigned int q = 0; q < singular_quadrature.size(); ++q)
          {
            const Tensor<1, dim> R = singular_points[q] - support_points[i];
            if (kernel_type == "laplace")
              LaplaceKernel::kernels(R, D, s);
            else if (kernel_type == "screened")
              ScreenedKernel::kernels(R, D, s, screened_kappa);
            else
              AssertThrow(false,
                          ExcMessage("Unknown kernel type: " + kernel_type));
            AssertThrow(std::isfinite(s) && is_finite_tensor(D),
                        ExcMessage("Non-finite singular kernel."));

            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
              {
                const double shape_j = singular_fe_values.shape_value(j, q);
                const double weight  = singular_fe_values.JxW(q);
                result.neumann[j] +=
                  (D * singular_normals[q]) * shape_j * weight;
                result.dirichlet[j] += s * shape_j * weight;
              }
          }
        return result;
      };

      SingularRows singular_rows =
        integrate_singular(singular_quadrature_order);
      unsigned int               used_order        = singular_quadrature_order;
      bool                       neumann_converged = !adaptive_quadrature;
      bool                       dirichlet_converged = !adaptive_quadrature;
      QuadratureConvergenceState neumann_convergence_state;
      QuadratureConvergenceState dirichlet_convergence_state;
      while (adaptive_quadrature && used_order < adaptive_quadrature_max_order)
        {
          const unsigned int next_order =
            next_adaptive_order(used_order,
                                adaptive_quadrature_max_order,
                                adaptive_quadrature_order_growth);
          SingularRows refined = integrate_singular(next_order);
          neumann_converged =
            quadrature_vectors_converged(singular_rows.neumann,
                                         refined.neumann,
                                         adaptive_quadrature_relative_tolerance,
                                         adaptive_quadrature_absolute_tolerance,
                                         neumann_convergence_state);
          dirichlet_converged =
            quadrature_vectors_converged(singular_rows.dirichlet,
                                         refined.dirichlet,
                                         adaptive_quadrature_relative_tolerance,
                                         adaptive_quadrature_absolute_tolerance,
                                         dirichlet_convergence_state);
          singular_rows = std::move(refined);
          used_order    = next_order;
          if (neumann_converged && dirichlet_converged)
            break;
        }
      statistics.raise_largest_order(used_order);
      if (adaptive_quadrature && !neumann_converged)
        ++statistics.adaptive_unconverged_singular_neumann;
      if (adaptive_quadrature && !dirichlet_converged)
        ++statistics.adaptive_unconverged_singular_dirichlet;
      ++statistics.adaptive_singular_interactions;
      local_neumann_matrix_row_i += singular_rows.neumann;
      local_dirichlet_matrix_row_i += singular_rows.dirichlet;
    } // end else
  return is_singular       ? CollocationRule::singular :
         is_quasi_singular ? CollocationRule::quasi_singular :
                             CollocationRule::regular;
}


template <int dim>
void
BEMProblem<dim>::assemble_exact_regularized_dirichlet_rhs(
  const Function<dim> &dirichlet_function)
{
  AssertThrow(solution_method == "Direct" && kernel_type == "laplace",
              ExcMessage("Exact regularized Dirichlet RHS is implemented only "
                         "for the direct Laplace formulation."));

  pcout << "Assembling exact regularized Dirichlet RHS" << std::endl;
  exact_regularized_dirichlet_rhs = 0;

  FEValues<dim - 1, dim>               regular_values(*mapping,
                                        *fe,
                                        *quadrature,
                                        update_normal_vectors |
                                          update_quadrature_points |
                                          update_JxW_values);
  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  std::vector<Point<dim>>              support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);

  unsigned int singular_interactions = 0;
  unsigned int quasi_interactions    = 0;
  unsigned int unconverged_singular  = 0;
  unsigned int unconverged_quasi     = 0;
  unsigned int largest_order_used    = 0;

  for (auto cell = dh.begin_active(); cell != dh.end(); ++cell)
    {
      regular_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      const auto &regular_points  = regular_values.get_quadrature_points();
      const auto &regular_normals = regular_values.get_normal_vectors();

      for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
        if (this_cpu_set.is_element(i))
          {
            const Point<dim> &singularity = support_points[i];
            const double source_value   = dirichlet_function.value(singularity);
            bool         is_singular    = false;
            unsigned int singular_index = numbers::invalid_unsigned_int;
            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
              if (double_nodes_set[i].count(local_dof_indices[j]) > 0)
                {
                  is_singular    = true;
                  singular_index = j;
                  break;
                }

            bool   is_quasi_singular = false;
            double distance_ratio    = std::numeric_limits<double>::infinity();
            Point<dim - 1> ref_projection;
            if (!is_singular)
              {
                const double h = cell->diameter();
                if (singularity.distance(cell->center()) <=
                    (quasi_singular_distance_ratio + 1.0) * h)
                  {
                    QuasiSingularKernelIntegral<dim> closest_point(cell,
                                                                   *fe,
                                                                   *mapping,
                                                                   singularity);
                    ref_projection =
                      closest_point.get_closest_reference_point();
                    distance_ratio = closest_point.min_distance / h;
                    is_quasi_singular =
                      distance_ratio < quasi_singular_distance_ratio;
                  }
              }

            const auto integrate_special =
              [&](const Quadrature<dim - 1> &integration_quadrature) -> double {
              FEValues<dim - 1, dim> values(*mapping,
                                            *fe,
                                            integration_quadrature,
                                            update_normal_vectors |
                                              update_quadrature_points |
                                              update_JxW_values);
              values.reinit(cell);
              const auto &points   = values.get_quadrature_points();
              const auto &normals  = values.get_normal_vectors();
              double      integral = 0.0;
              for (unsigned int q = 0; q < integration_quadrature.size(); ++q)
                {
                  Tensor<1, dim> double_layer_kernel;
                  double         single_layer_kernel;
                  LaplaceKernel::kernels(points[q] - singularity,
                                         double_layer_kernel,
                                         single_layer_kernel);
                  const double value_difference =
                    dirichlet_function.value(points[q]) - source_value;
                  integral += (double_layer_kernel * normals[q]) *
                              value_difference * values.JxW(q);
                }
              AssertThrow(std::isfinite(integral),
                          ExcMessage("Non-finite exact regularized Dirichlet "
                                     "integral."));
              return integral;
            };

            double cell_integral = 0.0;
            if (!is_singular && !is_quasi_singular)
              {
                for (unsigned int q = 0; q < regular_values.n_quadrature_points;
                     ++q)
                  {
                    Tensor<1, dim> double_layer_kernel;
                    double         single_layer_kernel;
                    LaplaceKernel::kernels(regular_points[q] - singularity,
                                           double_layer_kernel,
                                           single_layer_kernel);
                    cell_integral +=
                      (double_layer_kernel * regular_normals[q]) *
                      (dirichlet_function.value(regular_points[q]) -
                       source_value) *
                      regular_values.JxW(q);
                  }
              }
            else
              {
                const unsigned int initial_order =
                  is_singular ? singular_quadrature_order :
                                quasi_singular_neumann_quadrature_order;
                const auto make_quadrature =
                  [&](const unsigned int order) -> Quadrature<dim - 1> {
                  if (is_singular)
                    {
                      Assert(singular_index != numbers::invalid_unsigned_int,
                             ExcInternalError());
                      if constexpr (dim == 3)
                        return QSplit<2>(
                          QDuffy(order, 1.0),
                          fe->get_unit_support_points()[singular_index]);
                      else
                        return QTelles<1>(
                          order, fe->get_unit_support_points()[singular_index]);
                    }
                  return get_quasi_singular_quadrature(cell,
                                                       *mapping,
                                                       singularity,
                                                       ref_projection,
                                                       2,
                                                       order,
                                                       distance_ratio);
                };

                unsigned int used_order = initial_order;
                cell_integral = integrate_special(make_quadrature(used_order));
                bool                       converged = !adaptive_quadrature;
                QuadratureConvergenceState convergence_state;
                while (adaptive_quadrature &&
                       used_order < adaptive_quadrature_max_order)
                  {
                    const unsigned int next_order =
                      next_adaptive_order(used_order,
                                          adaptive_quadrature_max_order,
                                          adaptive_quadrature_order_growth);
                    const double refined =
                      integrate_special(make_quadrature(next_order));
                    converged = quadrature_scalars_converged(
                      cell_integral,
                      refined,
                      adaptive_quadrature_relative_tolerance,
                      adaptive_quadrature_absolute_tolerance,
                      convergence_state);
                    cell_integral = refined;
                    used_order    = next_order;
                    if (converged)
                      break;
                  }
                largest_order_used = std::max(largest_order_used, used_order);
                if (is_singular)
                  {
                    ++singular_interactions;
                    if (adaptive_quadrature && !converged)
                      ++unconverged_singular;
                  }
                else
                  {
                    ++quasi_interactions;
                    if (adaptive_quadrature && !converged)
                      ++unconverged_quasi;
                  }
              }

            exact_regularized_dirichlet_rhs[i] -= cell_integral;
          }
    }

  exact_regularized_dirichlet_rhs.compress(VectorOperation::add);
  pcout << "Exact regularized RHS interactions: singular "
        << Utilities::MPI::sum(singular_interactions, mpi_communicator)
        << ", quasi "
        << Utilities::MPI::sum(quasi_interactions, mpi_communicator)
        << ", largest order "
        << Utilities::MPI::max(largest_order_used, mpi_communicator)
        << ", unconverged singular/quasi "
        << Utilities::MPI::sum(unconverged_singular, mpi_communicator) << '/'
        << Utilities::MPI::sum(unconverged_quasi, mpi_communicator)
        << std::endl;
}

template <int dim>
void
BEMProblem<dim>::compute_hypersingular_free_coeffs()
{
  pcout << "Computing free coefficients for hypersingular BIE" << std::endl;

  pcout << "Computing C_ij tensor" << endl;

  Assert(fe->has_support_points(),
         ExcMessage(
           "The FE selected has no support points. This is not supported."));
  const std::vector<Point<dim - 1>> &ref_dofs_location =
    fe->get_unit_support_points();
  // we use these points as quadrature points for a quadrature rule
  std::vector<double> weights(ref_dofs_location.size(), 1.0);


  // here's the quadrature rule obtained with the points and weights generated
  Quadrature<dim - 1> dofs_quadrature(ref_dofs_location, weights);

  // and here's the FEValues class resulting by it
  FEValues<dim - 1, dim> dofs_fe_values(*mapping,
                                        *fe,
                                        dofs_quadrature,
                                        update_values | update_gradients |
                                          update_quadrature_points |
                                          update_JxW_values |
                                          update_normal_vectors |
                                          update_jacobians |
                                          update_jacobian_grads);

  cell_it cell = dh.begin_active(), endc = dh.end();
  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);

  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);

  // check which are the global dofs with on the sphere (material id 1)
  std::vector<bool> dof_in_material_1(dh.n_dofs(), false);
  if (comp_dom.input_grid_name == "../grids/sphere_box_flipped" ||
      comp_dom.input_grid_name == "../grids/sphere_box_flipped_double_nodes" ||
      comp_dom.input_grid_name == "../grids/sphere_box_flipped_new")
    {
      for (auto cell = dh.begin_active(); cell != dh.end(); ++cell)
        {
          if (int(cell->material_id()) == 1)
            {
              cell->get_dof_indices(local_dof_indices);

              for (const auto dof : local_dof_indices)
                {
                  dof_in_material_1[dof] = true;
                }
            }
        }
    }



  for (types::global_dof_index i = 0; i < dh.n_dofs();
       ++i) // these must now be the locally owned dofs. the rest should stay
            // the same
    {
      std::vector<Tensor<1, dim>> normals;
      if (this_cpu_set.is_element(i))
        {
          for (cell = dh.begin_active(); cell != endc; ++cell)
            {
              cell->get_dof_indices(local_dof_indices);
              for (unsigned int i_loc = 0; i_loc < fe->dofs_per_cell; ++i_loc)
                {
                  std::set<types::global_dof_index> doubles =
                    double_nodes_set[local_dof_indices[i_loc]];
                  for (std::set<types::global_dof_index>::iterator it =
                         doubles.begin();
                       it != doubles.end();
                       it++)
                    if (*it == i)
                      {
                        dofs_fe_values.reinit(cell);
                        Tensor<1, dim> normal =
                          dofs_fe_values.normal_vector(i_loc);
                        normals.push_back(normal);
                      }
                }
            }
          // cout<<i<<"  --> "<<normals.size()<<endl;
          // for (unsigned int k=0; k<normals.size(); ++k)
          //     cout<<normals[k]<<endl;
          std::vector<Tensor<1, dim>> unique_normals;
          std::vector<Tensor<1, dim>> unique_projected_normals;

          std::vector<Tensor<1, dim>> unique_tangents;
          for (unsigned int k = 0; k < normals.size(); ++k)
            {
              bool found = false;
              for (unsigned int j = 0; j < unique_normals.size(); ++j)
                if ((normals[k] - unique_normals[j]).norm() < 1e-5)
                  found = true;
              if (!found)
                unique_normals.push_back(normals[k]);
            }
          // cout<<i<<"  -->Unique normals: "<<unique_normals.size()<<endl;
          std::vector<Tensor<1, dim>> unique_normals_copy(unique_normals);
          // std::vector<Tensor<1, dim> >
          // unique_ordered_normals(unique_normals);
          std::vector<Tensor<1, dim>> unique_ordered_normals(unique_normals);
          // for (unsigned int k=0; k<unique_ordered_normals.size(); ++k)
          //     cout<<unique_ordered_normals[k]<<"   *** vs *** "<<
          //     unique_normals_copy[k]<<endl;

          // initialize geometric alpha = 2*pi then subtracts the angle between
          // every couple of normals
          double geom_alpha = 0.5;
          if (unique_normals.size() > 1)
            {
              Tensor<1, dim> average_normal;
              for (unsigned int k = 0; k < unique_normals.size(); ++k)
                average_normal +=
                  unique_normals[k] * (1.0 / unique_normals.size());
              const double average_normal_norm = average_normal.norm();
              AssertThrow(average_normal_norm > 1e-14,
                          ExcMessage("Cannot order free-coefficient normals: "
                                     "their average is zero."));
              average_normal /= average_normal_norm;
              // for (unsigned int k=0; k<unique_normals.size(); ++k)
              //     cout<<unique_normals[k]<<endl;
              Tensor<2, dim> projection_matrix =
                -1.0 * outer_product(average_normal, average_normal);
              for (unsigned int d = 0; d < dim; ++d)
                projection_matrix[d][d] += 1.0;
              for (unsigned int k = 0; k < unique_normals.size(); ++k)
                {
                  Tensor<1, dim> projected_normal =
                    projection_matrix * unique_normals[k];
                  const double projected_normal_norm = projected_normal.norm();
                  AssertThrow(projected_normal_norm > 1e-14,
                              ExcMessage(
                                "Cannot order free-coefficient normals: "
                                "a projected normal is zero."));
                  unique_projected_normals.push_back(projected_normal /
                                                     projected_normal_norm);
                }

              // for (unsigned int k=0; k<unique_normals.size(); ++k)
              //     cout<<"n: "<<unique_normals[k]<<"  proj:
              //     "<<unique_projected_normals[k]<<endl;
              unique_ordered_normals.clear();
              if (dim == 3)
                {
                  // Sort all projected normals around the average normal.  A
                  // greedy search from the previous normal can get stuck at
                  // the 2*pi-to-zero branch cut even for a valid vertex fan.
                  const Tensor<1, dim> angular_origin =
                    unique_projected_normals[0];
                  std::vector<std::pair<double, unsigned int>> angular_order;
                  angular_order.reserve(unique_normals.size());

                  for (unsigned int k = 0; k < unique_normals.size(); ++k)
                    {
                      double angle = std::atan2(
                        average_normal *
                          cross_product_3d(angular_origin,
                                           unique_projected_normals[k]),
                        clamp_to_unit_interval(angular_origin *
                                               unique_projected_normals[k]));
                      if (angle < 0.0)
                        angle += 2.0 * numbers::PI;
                      angular_order.emplace_back(angle, k);
                    }

                  std::sort(angular_order.begin(), angular_order.end());
                  for (const auto &entry : angular_order)
                    unique_ordered_normals.push_back(
                      unique_normals[entry.second]);
                }
              else if (dim == 2)
                unique_ordered_normals = unique_normals;
              unique_tangents.resize(unique_ordered_normals.size() + 1);
              if (dim == 3)
                unique_tangents[0] = cross_product_3d(
                  unique_ordered_normals[0],
                  unique_ordered_normals[unique_ordered_normals.size() - 1]);
              if (dim == 2)
                unique_tangents[0] =
                  cross_product_2d(unique_ordered_normals[0]);
              for (unsigned int k = 1; k < unique_ordered_normals.size(); ++k)
                if (dim == 3)
                  unique_tangents[k] =
                    cross_product_3d(unique_ordered_normals[k],
                                     unique_ordered_normals[k - 1]);
                else if (dim == 2)
                  unique_tangents[k] =
                    cross_product_2d(unique_ordered_normals[k]);

              unique_tangents[unique_ordered_normals.size()] =
                unique_tangents[0];
              // cout<<i<<"  -->Unique tangents:
              // "<<unique_tangents.size()<<endl;
              for (unsigned int k = 0; k < unique_tangents.size(); ++k)
                {
                  const double tangent_norm = unique_tangents[k].norm();
                  AssertThrow(tangent_norm > 1e-14,
                              ExcMessage(
                                "Cannot compute free-coefficient tangent: "
                                "two consecutive normals are parallel."));
                  unique_tangents[k] /= tangent_norm;
                  // cout<<unique_tangents[k]<<endl;
                }

              geom_alpha = 2 * numbers::PI;
              for (unsigned int k = 1; k < unique_ordered_normals.size(); ++k)
                geom_alpha -= acos(clamp_to_unit_interval(
                  unique_ordered_normals[k] * unique_ordered_normals[k - 1]));
              geom_alpha -= acos(clamp_to_unit_interval(
                unique_ordered_normals[unique_ordered_normals.size() - 1] *
                unique_ordered_normals[0]));
              // then geom_alpha is normalized to get a value between 0 and 1
              geom_alpha /= 4 * numbers::PI;
            }

          // for concave doimains we need the external angle
          if (comp_dom.input_grid_name == "../grids/sphere_box_flipped" ||
              comp_dom.input_grid_name ==
                "../grids/sphere_box_flipped_double_nodes" ||
              comp_dom.input_grid_name == "../grids/sphere_box_flipped_new")
            {
              if (dof_in_material_1[i])
                geom_alpha = 1 - geom_alpha;
            }

          AssertThrow(std::isfinite(geom_alpha),
                      ExcMessage("Non-finite geometric free coefficient."));

          hyp_alpha(i) = geom_alpha;

          // just in case we need to check the code
          // const Point<dim> refinement_center(0, 0, -1);
          // const double distance_from_center =
          // refinement_center.distance(support_points[i]);
          // if (std::abs(geom_alpha - alpha(i)) > 0.0003)
          //   pcout << i << " \t" << support_points[i]
          //         << "->\t geom_alpha: " << geom_alpha
          //         << " \t alpha(i): " << alpha(i) << endl;
          // if (fabs(geom_alpha-alpha(i)) > 1e-3)
          //   pcout<<"HELP! 	fabs(geom_alpha-alpha(i)) > 1e-3"<<endl;


          Tensor<2, dim> C_matrix;
          for (unsigned int d = 0; d < dim; ++d)
            C_matrix[d][d] += geom_alpha;
          if (unique_ordered_normals.size() > 1)
            {
              for (unsigned int k = 0; k < unique_ordered_normals.size(); ++k)
                if (dim == 3)
                  {
                    C_matrix -=
                      1. / 4. / dealii::numbers::PI *
                      outer_product(cross_product_3d(unique_tangents[k + 1] -
                                                       unique_tangents[k],
                                                     unique_ordered_normals[k]),
                                    unique_ordered_normals[k]);
                    // cout<<"unique_tangents[k+1]:
                    // "<<unique_tangents[k+1]<<endl; cout<<"unique_tangents[k]:
                    // "<<unique_tangents[k]<<endl;
                    // cout<<"unique_ordered_normals[k]"<<unique_ordered_normals[k]<<endl;
                  }
                else if (dim == 2)
                  C_matrix -= 1 / 2 / dealii::numbers::PI *
                              outer_product(unique_tangents[k],
                                            unique_ordered_normals[k]);
            }
          // pcout<<"C_matrix: "<<C_matrix<<endl;
          for (unsigned int di = 0; di < dim; ++di)
            for (unsigned int dj = 0; dj < dim; ++dj)
              {
                C_ij[di * dim + dj][i] = C_matrix[di][dj];
                AssertThrow(std::isfinite(C_matrix[di][dj]),
                            ExcMessage("Non-finite C_ij free coefficient."));
                // pcout<<C_ij[di*dim+dj][i]<<std::endl;
              }
        }
    }

  pcout << "Done computing C_ij tensor" << endl;

  if (kernel_type == "laplace" &&
      boundary_integral_formulation == "collocation")
    {
      for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
        if (this_cpu_set.is_element(i))
          {
            AssertThrow(std::isfinite(alpha(i)),
                        ExcMessage("Non-finite integral alpha."));
            AssertThrow(std::isfinite(hyp_alpha(i)),
                        ExcMessage("Non-finite geometric alpha."));
          }

      TrilinosWrappers::MPI::Vector error(hyp_alpha);
      error.sadd(-1.0, alpha);

      pcout << "Alpha abs error: " << error.l2_norm() << endl;
      pcout << "Alpha rel error: " << error.l2_norm() / alpha.l2_norm() << endl;
      pcout << "Alpha maximum error: " << error.linfty_norm() << endl;
      if (fe->degree > 1)
        pcout << "  (diagnostic geometric discrepancy; the consistent "
                 "row-sum alpha is retained for Laplace)"
              << endl;
    }
  else if (kernel_type == "laplace")
    pcout << "Galerkin jump term uses the consistent 0.5 mass matrix" << endl;
  else
    pcout << "Using geometric alpha for screened kernel" << endl;

  // For Laplace, compute_alpha() obtains a discretely consistent jump
  // coefficient from the assembled double-layer row sum.  In particular this
  // enforces the constant patch at higher-order interior collocation points.
  // Screened kernels have no constant-solution identity and use geometry.
  if (kernel_type != "laplace")
    alpha = hyp_alpha;

  // Calculating second free term as in Mantic et al. paper
  // Existence and evaluation of the two free terms in the hypersingular
  // boundary integral equation of potential theory
  // V Mantič, F Paris - Engineering Analysis with Boundary Elements, 1995
  // At each DOF, the free coefficient is a Tensor<1,dim>. It is null on dofs
  // located on smooth surfaces and becomes not null for non-smooth geometries.
  // We compute it here by means of integrals carried out in the cell parametric
  // plane. Because these are similar to those computed to obtain the
  // hypersingular quadrature, we implemented a method for such a computation in
  // the SingularKernelIntegral class. Cell by cell, and DOF by DOF, we compute
  // the portion of integral that gives the contribution of each cell to the
  // free coefficient of its DOFS
  pcout << "Computing vector b_i" << std::endl;
  std::vector<Tensor<1, dim>> free_term_b_all(dh.n_dofs());

  // we initialize an FEValues
  // object with the quadrature
  // formula for the integration of
  // the kernel in non singular
  // cells. This quadrature is
  // selected with the parameter
  // file, and needs to be quite
  // precise, since the functions we
  // are integrating are not
  // polynomial functions.
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int local_id = 0; local_id < fe->dofs_per_cell; ++local_id)
        {
          int global_id = local_dof_indices[local_id];
          if (this_cpu_set.is_element(global_id))
            {
              Assert(
                (*fe).has_support_points(),
                ExcMessage(
                  "The FE selected has no support points. This is not supported."));
              Point<dim - 1> P = (*fe).unit_support_point(local_id);

              SingularKernelIntegral<dim> singular_kernel_integrator(cell,
                                                                     *fe,
                                                                     *mapping,
                                                                     P);
              Tensor<1, dim>              b =
                singular_kernel_integrator.evaluate_free_term_b();
              std::set<types::global_dof_index> doubles =
                double_nodes_set[global_id];
              for (std::set<types::global_dof_index>::iterator it =
                     doubles.begin();
                   it != doubles.end();
                   it++)
                for (unsigned int d = 0; d < dim; ++d)
                  b_i[d][*it] += b[d];
            }
        }
    }
  for (unsigned int d = 0; d < dim; ++d)
    b_i[d].compress(VectorOperation::add);

  pcout << "Done computing vector b_i" << std::endl;

  pcout << "done computing free coefficients for hypersingular BIE"
        << std::endl;
}


// Compute the free coefficients for the Laplace problem
template <int dim>
void
BEMProblem<dim>::compute_alpha()
{
  if (boundary_integral_formulation == "galerkin")
    {
      alpha = 0.0;
      for (const auto i : this_cpu_set)
        alpha[i] = 0.5;
      alpha.compress(VectorOperation::insert);
      return;
    }

  static TrilinosWrappers::MPI::Vector ones, zeros, dummy;
  if (ones.size() != dh.n_dofs())
    {
      ones.reinit(this_cpu_set, mpi_communicator);
      vector_shift(ones, -1.);
      zeros.reinit(this_cpu_set, mpi_communicator);
      dummy.reinit(this_cpu_set, mpi_communicator);
    }

  if (solution_method == "Direct")
    {
      neumann_matrix.vmult(alpha, ones);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      collocation_fmm_products(ones, zeros, alpha, dummy);
    }
}


// Compute the free coefficients for the Screened Poisson problem
template <int dim>
void
BEMProblem<dim>::compute_alpha(const double kappa)
{
  // normalizers to the size of the mesh
  // Bounding box del dominio
  BoundingBox<dim> bbox = GridTools::compute_bounding_box(comp_dom.tria);
  auto [p_min, p_max]   = bbox.get_boundary_points(); // Point<dim>, Point<dim>

  // Diametro della mesh (distanza massima tra due vertici)
  // double diam = GridTools::diameter(comp_dom.tria);

  Point<dim> centroid;
  double     extent[dim];
  for (unsigned int d = 0; d < dim; ++d)
    {
      centroid(d) = 0.5 * (p_min(d) + p_max(d));
      extent[d]   = p_max(d) - p_min(d);
    }

  // pesi a_d bilanciati sull'estensione del dominio
  double sum_inv_sq = 0.0;
  for (unsigned int d = 0; d < dim; ++d)
    sum_inv_sq += 1.0 / (extent[d] * extent[d]);

  double a[dim];
  for (unsigned int d = 0; d < dim; ++d)
    a[d] = kappa / (extent[d] * std::sqrt(sum_inv_sq));

  // original section, used in the last part
  static TrilinosWrappers::MPI::Vector ones, zeros, dum;
  if (ones.size() != dh.n_dofs())
    {
      ones.reinit(this_cpu_set, mpi_communicator);
      vector_shift(ones, -1.);
      zeros.reinit(this_cpu_set, mpi_communicator);
      dum.reinit(this_cpu_set, mpi_communicator);
    }

  // hanging node constraints matrix
  AffineConstraints<double> c_hn;
  if (exact_geometry)
    ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, c_hn);
  else
    DoFTools::make_hanging_node_constraints(dh, c_hn);
  c_hn.close();

  // Define static variables for function values and normal derivative
  static TrilinosWrappers::MPI::Vector function_coeff, normal_derivative_coeff;
  static TrilinosWrappers::MPI::Vector tmp1, tmp2;

  if (function_coeff.size() != dh.n_dofs() ||
      normal_derivative_coeff.size() != dh.n_dofs())
    {
      // Allocate vectors
      function_coeff.reinit(this_cpu_set, mpi_communicator);
      normal_derivative_coeff.reinit(this_cpu_set, mpi_communicator);
      tmp1.reinit(this_cpu_set, mpi_communicator);
      tmp2.reinit(this_cpu_set, mpi_communicator);
    }

  function_coeff          = 0;
  normal_derivative_coeff = 0;
  tmp1                    = 0;
  tmp2                    = 0;

  // Support points of scalar and gradient DoFs
  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);

  std::vector<Point<dim>> vec_support_points(gradient_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     gradient_dh,
                                                     vec_support_points);

  // Initialize FEValues for integration on each cell
  cell_it            cell = dh.begin_active(), endc = dh.end();
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  FEValues<dim - 1, dim>               fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  // Loop over cells
  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      fe_v.reinit(cell);

      // Retrieve DoF indices on this cell
      cell->get_dof_indices(local_dof_indices);

      // Loop over DoF indices on this cell
      for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
        {
          // skip hanging nodes
          if (c_hn.is_constrained(local_dof_indices[j]))
            continue;

          const Point<dim> &punto = support_points[local_dof_indices[j]];

          double xp[dim], ch[dim], sh[dim];
          for (unsigned int d = 0; d < dim; ++d)
            {
              xp[d] = punto[d] - centroid(d);
              ch[d] = std::cosh(a[d] * xp[d]);
              sh[d] = std::sinh(a[d] * xp[d]);
            }

          // Normal vector (local_normal) initialized to zero
          Tensor<1, dim> local_normal;
          for (unsigned int ii = 0; ii < dim; ++ii)
            local_normal[ii] = 0;

          for (unsigned int d = 0; d < dim; ++d)
            {
              // Map indices between subdivided and original systems (to obtain
              // vec_index)
              types::global_dof_index dummy =
                sub_wise_to_original[local_dof_indices[j]];

              types::global_dof_index vec_index =
                vec_original_to_sub_wise[gradient_dh.n_dofs() / dim * d +
                                         dummy];

              // Consistency check
              Assert(vector_this_cpu_set.is_element(vec_index),
                     ExcMessage("vector cpu set and cpu set are inconsistent"));

              // Build local_normal component by component
              local_normal[d] = vector_normals_solution[vec_index];
            }

          // Function value: cosh(a1 x) * cosh(a2 y) * cosh(a3 z)
          double u_val = 1.0;
          for (unsigned int d = 0; d < dim; ++d)
            u_val *= ch[d];
          function_coeff(local_dof_indices[j]) = u_val;

          // gradient of the ch * ch * ch function
          double grad_u[dim];
          for (unsigned int d = 0; d < dim; ++d)
            {
              double g = a[d] * sh[d];
              for (unsigned int k = 0; k < dim; ++k)
                if (k != d)
                  g *= ch[k];
              grad_u[d] = g;
            }

          // normal derivative value
          double dudn = 0;
          for (unsigned int d = 0; d < dim; ++d)
            dudn += grad_u[d] * local_normal[d];
          normal_derivative_coeff(local_dof_indices[j]) = dudn;
        }
    }

  c_hn.distribute(function_coeff);
  c_hn.distribute(normal_derivative_coeff);

  for (types::global_dof_index ii = 0; ii < dh.n_dofs(); ++ii)
    if (this_cpu_set.is_element(ii))
      {
        AssertThrow(std::isfinite(function_coeff(ii)),
                    ExcMessage("Non-finite screened function coefficient."));
        AssertThrow(std::isfinite(normal_derivative_coeff(ii)),
                    ExcMessage("Non-finite screened normal derivative "
                               "coefficient."));
        AssertThrow(std::abs(function_coeff(ii)) > 1e-14,
                    ExcMessage("Zero screened function coefficient."));
      }

  // Compute alpha depending on solver type
  if (solution_method == "Direct")
    {
      // Multiply coefficients and Newmann and Dirichlet matrices
      neumann_matrix.vmult(tmp1, function_coeff);
      dirichlet_matrix.vmult(tmp2, normal_derivative_coeff);
      for (types::global_dof_index ii = 0; ii < dh.n_dofs(); ++ii)
        if (this_cpu_set.is_element(ii))
          {
            AssertThrow(std::isfinite(tmp1(ii)),
                        ExcMessage(
                          "Non-finite screened alpha Neumann product."));
            AssertThrow(std::isfinite(tmp2(ii)),
                        ExcMessage(
                          "Non-finite screened alpha Dirichlet product."));
          }

      // alpha = (-tmp1 - tmp2) / function_coeff
      alpha.reinit(this_cpu_set, mpi_communicator);
      alpha = tmp2;
      alpha -= tmp1;

      for (unsigned int ii = 0; ii < alpha.size(); ++ii)
        {
          if (this_cpu_set.is_element(ii))
            alpha[ii] /= function_coeff[ii];
        }
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      collocation_fmm_products(ones, zeros, alpha, dum);
    }

  // fill hanging node values by interpolating from master dofs
  c_hn.distribute(alpha);

  // alpha.print(pcout);
  // for (unsigned int i=0; i<alpha.size(); ++i)
  //   {
  //     pcout << i << "-> 	alpha(i): " << std::setprecision(10) << alpha(i)
  //     <<endl;
  //   }
}


template <int dim>
void
BEMProblem<dim>::vmult(TrilinosWrappers::MPI::Vector       &dst,
                       const TrilinosWrappers::MPI::Vector &src) const
{
  // the Robin nodes participate with their unknowns carrying phi
  //(alpha + N) * serv_phi - D * serv_dphi_dn
  // becomes
  //(alpha + N) * (serv_phi + serv_phi_robin) - D * (serv_dphi_dn -
  // robin_matrix_diagonal.scale(serv_phi_robin))

  serv_phi = src;
  if (!can_determine_phi)
    {
      vector_shift(serv_phi, -serv_phi.l2_norm());
    }
  serv_dphi_dn   = src;
  serv_phi_robin = serv_phi;

  dst = 0;

  serv_phi.scale(neumann_nodes);
  serv_dphi_dn.scale(dirichlet_nodes);
  serv_phi_robin.scale(robin_nodes);

  if (solution_method == "Direct" ||
      boundary_integral_formulation == "galerkin")
    {
      serv_phi += serv_phi_robin;
      serv_phi_robin.scale(robin_matrix_diagonal);
      serv_dphi_dn -= serv_phi_robin;

      apply_single_layer(dst, serv_dphi_dn);
      dst *= -1;
      if (boundary_integral_formulation == "collocation")
        neumann_matrix.vmult_add(dst, serv_phi);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      serv_phi += serv_phi_robin;
      serv_phi_robin.scale(robin_matrix_diagonal);
      serv_dphi_dn -= serv_phi_robin;

      static TrilinosWrappers::MPI::Vector matrVectProdN(this_cpu_set,
                                                         mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdD(this_cpu_set,
                                                         mpi_communicator);
      if (matrVectProdN.size() != src.size())
        {
          matrVectProdN.reinit(this_cpu_set, mpi_communicator);
          matrVectProdD.reinit(this_cpu_set, mpi_communicator);
        }

      collocation_fmm_products(serv_phi,
                               serv_dphi_dn,
                               matrVectProdN,
                               matrVectProdD);

      dst.add(-1, matrVectProdD, 1, matrVectProdN);
    }

  if (boundary_integral_formulation == "collocation")
    {
      TrilinosWrappers::MPI::Vector jump_term(this_cpu_set, mpi_communicator);
      apply_jump_term(jump_term, serv_phi);
      dst += jump_term;
    }

  if (!can_determine_phi)
    {
      vector_shift(dst, -dst.l2_norm());
    }
  dst.compress(VectorOperation::add);
}

template <int dim>
void
BEMProblem<dim>::vmult(TrilinosWrappers::MPI::Vector       &dst,
                       TrilinosWrappers::MPI::Vector       &dst_imag,
                       const TrilinosWrappers::MPI::Vector &src,
                       const TrilinosWrappers::MPI::Vector &src_imag) const
{
  // the Robin nodes participate with their unknowns carrying phi
  //(alpha + N) * serv_phi - D * serv_dphi_dn
  // becomes
  //(alpha + N) * (serv_phi + serv_phi_robin) - D * (serv_dphi_dn -
  // robin_matrix_diagonal.scale(serv_phi_robin))

  serv_phi      = src;
  serv_phi_imag = src_imag;
  if (!can_determine_phi)
    {
      auto shift = std::sqrt(serv_phi.norm_sqr() + serv_phi_imag.l2_norm());
      vector_shift(serv_phi, -shift);
      vector_shift(serv_phi_imag, -shift);
    }
  serv_dphi_dn        = src;
  serv_dphi_dn_imag   = src_imag;
  serv_phi_robin      = serv_phi;
  serv_phi_robin_imag = serv_phi_imag;

  static TrilinosWrappers::MPI::Vector tmp(this_cpu_set, mpi_communicator);
  if (tmp.size() != src.size())
    {
      tmp.reinit(this_cpu_set, mpi_communicator);
    }

  dst      = 0;
  dst_imag = 0;

  serv_phi.scale(neumann_nodes);
  serv_phi_imag.scale(neumann_nodes);
  serv_dphi_dn.scale(dirichlet_nodes);
  serv_dphi_dn_imag.scale(dirichlet_nodes);
  serv_phi_robin.scale(robin_nodes);
  serv_phi_robin_imag.scale(robin_nodes);

  if (solution_method == "Direct")
    {
      serv_phi += serv_phi_robin;
      serv_phi_imag += serv_phi_robin_imag;

      // robin_matrix_diagonal is complex
      // serv_dphi_dn += -robin_matrix_diagonal*serv_phi_robin +
      // robin_matrix_diagonal_imag * serv_phi_robin_imag
      // and
      // serv_dphi_dn_imag += -robin_matrix_diagonal_imag*serv_phi_robin -
      // robin_matrix_diagonal * serv_phi_robin_imag

      tmp = serv_phi_robin;
      tmp.scale(robin_matrix_diagonal);
      serv_dphi_dn -= tmp;
      tmp = serv_phi_robin_imag;
      tmp.scale(robin_matrix_diagonal_imag);
      serv_dphi_dn += tmp;

      // these can be destructive
      serv_phi_robin.scale(robin_matrix_diagonal_imag);
      serv_phi_robin_imag.scale(robin_matrix_diagonal);
      serv_dphi_dn_imag.add(-1, serv_phi_robin, -1, serv_phi_robin_imag);

      // now, products between the plain matrices and the spiked vectors
      dirichlet_matrix.vmult(dst, serv_dphi_dn);
      dirichlet_matrix.vmult(dst_imag, serv_dphi_dn_imag);
      dst *= -1;
      dst_imag *= -1;
      neumann_matrix.vmult_add(dst, serv_phi);
      neumann_matrix.vmult_add(dst_imag, serv_phi_imag);
      serv_phi.scale(alpha);
      serv_phi_imag.scale(alpha);
      dst += serv_phi;
      dst_imag += serv_phi_imag;
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      serv_phi += serv_phi_robin;
      serv_phi_imag += serv_phi_robin_imag;

      // the dphi_dn vectors will be spiked with the contribution from robin
      // nodes
      tmp = serv_phi_robin;
      tmp.scale(robin_matrix_diagonal);
      serv_dphi_dn -= tmp;
      tmp = serv_phi_robin_imag;
      tmp.scale(robin_matrix_diagonal_imag);
      serv_dphi_dn += tmp;

      // these can be destructive
      serv_phi_robin.scale(robin_matrix_diagonal_imag);
      serv_phi_robin_imag.scale(robin_matrix_diagonal);
      serv_dphi_dn_imag.add(-1, serv_phi_robin, -1, serv_phi_robin_imag);

      static TrilinosWrappers::MPI::Vector matrVectProdN(this_cpu_set,
                                                         mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdN_imag(this_cpu_set,
                                                              mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdD(this_cpu_set,
                                                         mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdD_imag(this_cpu_set,
                                                              mpi_communicator);
      if (matrVectProdN.size() != src.size())
        {
          matrVectProdN.reinit(this_cpu_set, mpi_communicator);
          matrVectProdN_imag.reinit(this_cpu_set, mpi_communicator);
          matrVectProdD.reinit(this_cpu_set, mpi_communicator);
          matrVectProdD_imag.reinit(this_cpu_set, mpi_communicator);
        }

      // now, products between the plain matrices and the spiked vectors
      collocation_fmm_products(serv_phi,
                               serv_dphi_dn,
                               matrVectProdN,
                               matrVectProdD);

      collocation_fmm_products(serv_phi_imag,
                               serv_dphi_dn_imag,
                               matrVectProdN_imag,
                               matrVectProdD_imag);

      dst.add(-1, matrVectProdD, 1, matrVectProdN);
      dst_imag.add(-1, matrVectProdD_imag, 1, matrVectProdN_imag);

      serv_phi.scale(alpha);
      serv_phi_imag.scale(alpha);
      dst += serv_phi;
      dst_imag += serv_phi_imag;
    }

  if (!can_determine_phi)
    {
      auto shift = std::sqrt(dst.norm_sqr() + dst_imag.l2_norm());
      vector_shift(dst, -shift);
      vector_shift(dst_imag, -shift);
    }
  dst.compress(VectorOperation::add);
  dst_imag.compress(VectorOperation::add);
}

template <int dim>
void
BEMProblem<dim>::apply_single_layer(
  TrilinosWrappers::MPI::Vector       &dst,
  const TrilinosWrappers::MPI::Vector &src) const
{
  if (galerkin_fmm)
    galerkin_fmm->vmult(dst, src);
  else
    dirichlet_matrix.vmult(dst, src);
  if (laplace_patch_correction_initialized)
    for (unsigned int d = 0; d < dim; ++d)
      dst.add(laplace_patch_normal_components[d] * src,
              laplace_patch_correction_coefficients[d]);
}


template <int dim>
void
BEMProblem<dim>::apply_jump_term(TrilinosWrappers::MPI::Vector       &dst,
                                 const TrilinosWrappers::MPI::Vector &src) const
{
  if (boundary_integral_formulation == "galerkin")
    {
      jump_matrix.vmult(dst, src);
      dst *= 0.5;
    }
  else
    {
      dst = src;
      dst.scale(alpha);
    }
}


template <int dim>
void
BEMProblem<dim>::compute_rhs(TrilinosWrappers::MPI::Vector       &dst,
                             const TrilinosWrappers::MPI::Vector &src) const
{
  if (boundary_integral_formulation == "galerkin")
    {
      dst = exact_regularized_dirichlet_rhs;
      return;
    }

  if (dirichlet_rhs_treatment == "exact_regularized")
    {
      dst = exact_regularized_dirichlet_rhs;
      return;
    }

  // the Robin nodes participate with their unknowns carrying phi; the
  // inhomogeneity is accounted for as if included in dphi_dn
  //-(alpha + N) * serv_phi + D * serv_dphi_dn
  // becomes
  //-(alpha + N) * serv_phi + D * (serv_dphi_dn + robin_rhs)
  serv_phi     = src;
  serv_dphi_dn = src;

  // Robin nodes are accounted for by robin_rhs
  serv_phi.scale(dirichlet_nodes);
  serv_dphi_dn.scale(neumann_nodes);
  // cut the robin_rhs to only the true robin nodes
  serv_phi_robin = robin_rhs;
  serv_phi_robin.scale(robin_nodes);

  if (solution_method == "Direct")
    {
      neumann_matrix.vmult(dst, serv_phi);
      TrilinosWrappers::MPI::Vector jump_term(this_cpu_set, mpi_communicator);
      apply_jump_term(jump_term, serv_phi);
      dst += jump_term;
      dst *= -1;
      serv_dphi_dn += serv_phi_robin;
      TrilinosWrappers::MPI::Vector single_layer_term(this_cpu_set,
                                                      mpi_communicator);
      apply_single_layer(single_layer_term, serv_dphi_dn);
      dst += single_layer_term;
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      // The Robin datum before the expansions: the far field needs it as
      // well as the near field.
      serv_dphi_dn += serv_phi_robin;

      static TrilinosWrappers::MPI::Vector matrVectProdN(this_cpu_set,
                                                         mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdD(this_cpu_set,
                                                         mpi_communicator);
      if (matrVectProdN.size() != src.size())
        {
          matrVectProdN.reinit(this_cpu_set, mpi_communicator);
          matrVectProdD.reinit(this_cpu_set, mpi_communicator);
        }

      collocation_fmm_products(serv_phi,
                               serv_dphi_dn,
                               matrVectProdN,
                               matrVectProdD);
      dst += matrVectProdN;
      TrilinosWrappers::MPI::Vector jump_term(this_cpu_set, mpi_communicator);
      apply_jump_term(jump_term, serv_phi);
      dst += jump_term;
      dst *= -1;
      dst += matrVectProdD;
    }
}

template <int dim>
void
BEMProblem<dim>::compute_rhs(
  TrilinosWrappers::MPI::Vector       &dst,
  TrilinosWrappers::MPI::Vector       &dst_imag,
  const TrilinosWrappers::MPI::Vector &src,
  const TrilinosWrappers::MPI::Vector &src_imag) const
{
  // the Robin nodes participate with their unknowns carrying phi; the
  // inhomogeneity is accounted for as if included in dphi_dn
  //-(alpha + N) * serv_phi + D * serv_dphi_dn
  // becomes
  //-(alpha + N) * serv_phi + D * (serv_dphi_dn + robin_rhs)
  serv_phi            = src;
  serv_phi_imag       = src_imag;
  serv_dphi_dn        = src;
  serv_dphi_dn_imag   = src_imag;
  serv_phi_robin      = robin_rhs;
  serv_phi_robin_imag = robin_rhs_imag;

  // Robin nodes are accounted for by robin_rhs
  serv_phi.scale(dirichlet_nodes);
  serv_phi_imag.scale(dirichlet_nodes);
  serv_dphi_dn.scale(neumann_nodes);
  serv_dphi_dn_imag.scale(neumann_nodes);
  // cut the robin_rhs to only the true robin nodes
  serv_phi_robin.scale(robin_nodes);
  serv_phi_robin_imag.scale(robin_nodes);

  if (solution_method == "Direct")
    {
      neumann_matrix.vmult(dst, serv_phi);
      neumann_matrix.vmult(dst_imag, serv_phi_imag);

      serv_phi.scale(alpha);
      serv_phi_imag.scale(alpha);

      dst += serv_phi;
      dst_imag += serv_phi_imag;

      dst *= -1;
      dst_imag *= -1;

      serv_dphi_dn += serv_phi_robin;
      serv_dphi_dn_imag += serv_phi_robin_imag;

      dirichlet_matrix.vmult_add(dst, serv_dphi_dn);
      dirichlet_matrix.vmult_add(dst_imag, serv_dphi_dn_imag);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      serv_dphi_dn += serv_phi_robin;
      serv_dphi_dn_imag += serv_phi_robin_imag;

      static TrilinosWrappers::MPI::Vector matrVectProdN(this_cpu_set,
                                                         mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdN_imag(this_cpu_set,
                                                              mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdD(this_cpu_set,
                                                         mpi_communicator);
      static TrilinosWrappers::MPI::Vector matrVectProdD_imag(this_cpu_set,
                                                              mpi_communicator);
      if (matrVectProdN.size() != src.size())
        {
          matrVectProdN.reinit(this_cpu_set, mpi_communicator);
          matrVectProdN_imag.reinit(this_cpu_set, mpi_communicator);
          matrVectProdD.reinit(this_cpu_set, mpi_communicator);
          matrVectProdD_imag.reinit(this_cpu_set, mpi_communicator);
        }

      collocation_fmm_products(serv_phi,
                               serv_dphi_dn,
                               matrVectProdN,
                               matrVectProdD);

      collocation_fmm_products(serv_phi_imag,
                               serv_dphi_dn_imag,
                               matrVectProdN_imag,
                               matrVectProdD_imag);

      serv_phi.scale(alpha);
      serv_phi_imag.scale(alpha);

      dst.add(-1, matrVectProdN, -1, serv_phi);
      dst_imag.add(-1, matrVectProdN_imag, -1, serv_phi_imag);
      dst += matrVectProdD;
      dst_imag += matrVectProdD_imag;
    }
}


template <int dim>
void
BEMProblem<dim>::assemble_explicit_system(
  TrilinosWrappers::SparseMatrix &system_matrix) const
{
  AssertThrow(solution_method == "Direct",
              ExcMessage("MUMPS requires explicitly assembled operators."));
  AssertThrow(can_determine_phi,
              ExcMessage("The explicit MUMPS path does not yet support the "
                         "pure-Neumann null-space correction."));

  system_matrix.reinit(full_sparsity_pattern);
  const Vector<double> localized_dirichlet_nodes(dirichlet_nodes);
  const Vector<double> localized_neumann_nodes(neumann_nodes);

  std::vector<Vector<double>> localized_patch_normals;
  std::vector<Vector<double>> localized_patch_corrections;
  if (laplace_patch_correction_initialized)
    {
      localized_patch_normals.reserve(dim);
      localized_patch_corrections.reserve(dim);
      for (unsigned int d = 0; d < dim; ++d)
        {
          localized_patch_normals.emplace_back(
            laplace_patch_normal_components[d]);
          localized_patch_corrections.emplace_back(
            laplace_patch_correction_coefficients[d]);
        }
    }

  for (const auto i : this_cpu_set)
    {
      if (constraints.is_constrained(i))
        {
          system_matrix.set(i, i, 1.0);
          const auto *entries = constraints.get_constraint_entries(i);
          if (entries != nullptr)
            for (const auto &entry : *entries)
              system_matrix.set(i, entry.first, -entry.second);
          continue;
        }

      for (types::global_dof_index j = 0; j < dh.n_dofs(); ++j)
        {
          double value = 0.0;
          if (localized_dirichlet_nodes[j] != 0.0)
            {
              value = -dirichlet_matrix(i, j);
              for (unsigned int d = 0; d < localized_patch_normals.size(); ++d)
                value -= localized_patch_corrections[d][i] *
                         localized_patch_normals[d][j];
            }
          else if (localized_neumann_nodes[j] != 0.0)
            {
              value = neumann_matrix(i, j);
              if (i == j)
                value += alpha[i];
            }

          if (value != 0.0)
            system_matrix.set(i, j, value);
        }
    }
  system_matrix.compress(VectorOperation::insert);
}


// @sect4{BEMProblem::solve_system}

// The next function simply solves
// the linear system.
template <int dim>
void
BEMProblem<dim>::solve_system(TrilinosWrappers::MPI::Vector       &phi,
                              TrilinosWrappers::MPI::Vector       &dphi_dn,
                              const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  Teuchos::TimeMonitor LocalTimer(*LacSolveTime);
  system_rhs = 0;
  sol        = 0;
  alpha      = 0;


  if (kernel_type == "laplace")
    {
      compute_alpha();

      if (enforce_laplace_linear_patch && solution_method == "Direct")
        {
          laplace_patch_normal_components.resize(dim);
          laplace_patch_correction_coefficients.resize(dim);
          std::vector<TrilinosWrappers::MPI::Vector> coordinates(dim);
          std::vector<TrilinosWrappers::MPI::Vector> residuals(dim);
          std::vector<Point<dim>> support_points(dh.n_dofs());
          DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                             dh,
                                                             support_points);

          for (unsigned int d = 0; d < dim; ++d)
            {
              coordinates[d].reinit(this_cpu_set, mpi_communicator);
              residuals[d].reinit(this_cpu_set, mpi_communicator);
              laplace_patch_normal_components[d].reinit(this_cpu_set,
                                                        mpi_communicator);
              laplace_patch_correction_coefficients[d].reinit(this_cpu_set,
                                                              mpi_communicator);
              for (const auto i : this_cpu_set)
                {
                  const types::global_dof_index original_i =
                    sub_wise_to_original[i];
                  coordinates[d][i] = support_points[i][d];
                  laplace_patch_normal_components[d][i] =
                    vector_normals_solution
                      [vec_original_to_sub_wise[d * dh.n_dofs() + original_i]];
                }
              coordinates[d].compress(VectorOperation::insert);
              laplace_patch_normal_components[d].compress(
                VectorOperation::insert);

              dirichlet_matrix.vmult(residuals[d],
                                     laplace_patch_normal_components[d]);
              residuals[d] *= -1.0;
              neumann_matrix.vmult_add(residuals[d], coordinates[d]);
              TrilinosWrappers::MPI::Vector alpha_coordinate(coordinates[d]);
              alpha_coordinate.scale(alpha);
              residuals[d] += alpha_coordinate;
            }

          FullMatrix<double> normal_gram(dim, dim);
          FullMatrix<double> inverse_normal_gram(dim, dim);
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int e = 0; e < dim; ++e)
              normal_gram(d, e) = laplace_patch_normal_components[d] *
                                  laplace_patch_normal_components[e];
          inverse_normal_gram.invert(normal_gram);

          for (const auto i : this_cpu_set)
            for (unsigned int d = 0; d < dim; ++d)
              for (unsigned int e = 0; e < dim; ++e)
                laplace_patch_correction_coefficients[d][i] +=
                  inverse_normal_gram(d, e) * residuals[e][i];
          for (unsigned int d = 0; d < dim; ++d)
            laplace_patch_correction_coefficients[d].compress(
              VectorOperation::insert);
          laplace_patch_correction_initialized = true;
          pcout << "Applied rank-" << dim << " Laplace linear patch correction"
                << std::endl;
        }
    }
  else if (kernel_type == "screened")
    {
      // The jump coefficient is geometric also for modified Helmholtz.
      // compute_hypersingular_free_coeffs() evaluates it below.
    }
  else
    AssertThrow(false, ExcMessage("Unknown kernel type: " + kernel_type));

  // The free coefficients are the matrix C of the hypersingular equation, so
  // recovering the gradient that way needs them too; without them C is zero
  // and its inverse is not a number.
  // The Galerkin formulation has no free term: the screened case needs the
  // coefficients only for collocation.
  if ((kernel_type == "screened" &&
       boundary_integral_formulation != "galerkin") ||
      gradient_recovery == "hypersingular")
    compute_hypersingular_free_coeffs();
  else
    pcout << "Skipping hypersingular free coefficients: not required by the "
             "current Laplace solve"
          << std::endl;
  //   for (unsigned int i = 0; i < alpha.size(); i++)
  //      if (this_cpu_set.is_element(i))
  //         pcout<<std::setprecision(20)<<alpha(i)<<std::endl;


  compute_rhs(system_rhs, tmp_rhs);

  compute_constraints(constr_cpu_set, constraints, tmp_rhs);
  ConstrainedOperator<TrilinosWrappers::MPI::Vector, BEMProblem<dim>> cc(
    *this, constraints, constr_cpu_set, mpi_communicator);

  cc.distribute_rhs(system_rhs);
  system_rhs.compress(VectorOperation::insert);

  if (linear_solver == "CG")
    {
      // The Galerkin single-layer operator is symmetric and positive
      // definite. BEMProblem::vmult returns -V, so operator and right hand
      // side are both negated here to hand CG a positive definite system.
      // Unlike the direct path this never forms an explicit matrix, so it
      // avoids both the quadratic assembly loop and the dense factorization.
      //
      // A discontinuous trace space carries no constraints, and the operator
      // is V itself. A continuous one on a locally refined mesh carries
      // hanging-node constraints, which the assembly has already condensed:
      // the free rows and columns of V hold C^T V C. ConstrainedOperator
      // appends the constraint equations as rows, which is not symmetric, so
      // CG is given the condensed block on the free unknowns and the identity
      // on the constrained ones, whose right-hand side is zero; their values
      // follow from the free ones after the solve.
      AssertThrow(boundary_integral_formulation == "galerkin" ||
                    constraints.n_constraints() == 0,
                  ExcMessage("The conjugate gradient path with constraints "
                             "is implemented for the Galerkin formulation."));

      struct NegatedOperator
      {
        const ConstrainedOperator<TrilinosWrappers::MPI::Vector,
                                  BEMProblem<dim>> &constrained_operator;
        const AffineConstraints<double>            &constraints;
        const IndexSet                             &locally_owned;

        void
        vmult(TrilinosWrappers::MPI::Vector       &dst,
              const TrilinosWrappers::MPI::Vector &src) const
        {
          if (constraints.n_constraints() == 0)
            {
              constrained_operator.vmult(dst, src);
              dst *= -1.0;
              return;
            }
          TrilinosWrappers::MPI::Vector free_part(src);
          for (const auto i : locally_owned)
            if (constraints.is_constrained(i))
              free_part[i] = 0.0;
          free_part.compress(VectorOperation::insert);
          constrained_operator.vmult(dst, free_part);
          dst *= -1.0;
          for (const auto i : locally_owned)
            if (constraints.is_constrained(i))
              dst[i] = src[i];
          dst.compress(VectorOperation::insert);
        }
      };
      const NegatedOperator negated_operator{cc, constraints, this_cpu_set};

      TrilinosWrappers::MPI::Vector negated_rhs(system_rhs);
      negated_rhs *= -1.0;

      // The tolerance is taken relative to the norm of the right-hand side.
      // The Dirichlet data may be of any size - a screened solution with a
      // large kappa is 1e-5 on one surface and 1e35 on another - and an
      // absolute tolerance is then either reached after a few iterations or
      // not reachable at all in double precision.
      const double  rhs_norm = negated_rhs.l2_norm();
      SolverControl cg_control(solver_control.max_steps(),
                               solver_control.tolerance() *
                                 std::max(rhs_norm, 1e-300),
                               solver_control.log_history(),
                               solver_control.log_result());
      SolverCG<TrilinosWrappers::MPI::Vector> solver(cg_control);
      if (preconditioner_type == "Jacobi")
        {
          pcout << "Iterative solver: CG with diagonal preconditioning"
                << std::endl;
          const DiagonalPreconditioner diagonal_preconditioner(
            galerkin_preconditioner_matrix(),
            this_cpu_set,
            mpi_communicator,
            constraints);
          solver.solve(negated_operator,
                       sol,
                       negated_rhs,
                       diagonal_preconditioner);
        }
      else if (preconditioner_type == "BlockJacobi")
        {
          pcout << "Iterative solver: CG with cell-block preconditioning"
                << std::endl;
          // The free unknowns of every cell; each process applies the blocks
          // of its own cells.
          std::vector<std::vector<types::global_dof_index>> cell_dofs;
          std::vector<bool>                                 applies_cell;
          std::vector<types::global_dof_index> dofs(fe->dofs_per_cell);
          for (const auto &cell : dh.active_cell_iterators())
            {
              cell->get_dof_indices(dofs);
              cell_dofs.emplace_back();
              for (const auto i : dofs)
                if (!constraints.is_constrained(i))
                  cell_dofs.back().push_back(i);
              applies_cell.push_back(cell->subdomain_id() == this_mpi_process);
            }
          const CellBlockPreconditioner block_preconditioner(
            galerkin_preconditioner_matrix(),
            cell_dofs,
            applies_cell,
            this_cpu_set,
            constraints,
            mpi_communicator);
          solver.solve(negated_operator,
                       sol,
                       negated_rhs,
                       block_preconditioner);
        }
      else if (preconditioner_type == "PatchSchwarz")
        {
          // Additive Schwarz on overlapping patches of cells, from the near
          // field of the FMM.
          AssertThrow(galerkin_fmm != nullptr, ExcInternalError());
          pcout << "Iterative solver: CG with additive Schwarz on "
                   "overlapping patches (a cell and its vertex neighbours)"
                << std::endl;
          galerkin_fmm->setup_patch_preconditioner();
          struct PatchPreconditioner
          {
            const GalerkinFMMOperator &fmm_operator;
            void
            vmult(TrilinosWrappers::MPI::Vector       &dst,
                  const TrilinosWrappers::MPI::Vector &src) const
            {
              fmm_operator.apply_patch_preconditioner(dst, src);
            }
          };
          const PatchPreconditioner patch_preconditioner{*galerkin_fmm};
          solver.solve(negated_operator,
                       sol,
                       negated_rhs,
                       patch_preconditioner);
        }
      else
        {
          AssertThrow(preconditioner_type == "None",
                      ExcMessage("The conjugate gradient path supports "
                                 "Preconditioner = None, Jacobi, BlockJacobi "
                                 "or PatchSchwarz; the other preconditioners "
                                 "are not symmetric."));
          pcout << "Iterative solver: CG without preconditioning" << std::endl;
          PreconditionIdentity identity_preconditioner;
          solver.solve(negated_operator,
                       sol,
                       negated_rhs,
                       identity_preconditioner);
        }
      if (constraints.n_constraints() > 0)
        constraints.distribute(sol);
      if (galerkin_fmm)
        galerkin_fmm->print_statistics(pcout.get_stream());
      pcout << "CG iterations: " << cg_control.last_step()
            << ", final residual: " << cg_control.last_value()
            << " (relative to the right-hand side "
            << cg_control.last_value() / std::max(rhs_norm, 1e-300) << ")"
            << std::endl;
    }
  else if (linear_solver == "MUMPS")
    {
#ifdef DEAL_II_WITH_MUMPS
      AssertThrow(solution_method == "Direct",
                  ExcMessage("MUMPS is available only with Solution method "
                             "= Direct."));
      TrilinosWrappers::SparseMatrix explicit_system;
      assemble_explicit_system(explicit_system);

      pcout << "Direct solver: MUMPS (no iterative preconditioner)"
            << std::endl;
      pcout << "Factoring explicit " << boundary_integral_formulation
            << " matrix with MUMPS" << std::endl;
      SparseDirectMUMPS direct_solver(
        SparseDirectMUMPS::AdditionalData(false, false, false, false),
        mpi_communicator);
      const int ordering_code      = mumps_ordering == "AMD"    ? 0 :
                                     mumps_ordering == "AMF"    ? 2 :
                                     mumps_ordering == "SCOTCH" ? 3 :
                                     mumps_ordering == "PORD"   ? 4 :
                                     mumps_ordering == "METIS"  ? 5 :
                                     mumps_ordering == "QAMD"   ? 6 :
                                                                  7;
      direct_solver.get_icntl()[6] = ordering_code;
      pcout << "MUMPS ordering: " << mumps_ordering
            << " (ICNTL(7)=" << ordering_code << ")" << std::endl;
      direct_solver.initialize(explicit_system);
      direct_solver.vmult(sol, system_rhs);

      TrilinosWrappers::MPI::Vector residual(system_rhs);
      explicit_system.vmult(residual, sol);
      residual -= system_rhs;
      const double rhs_norm = system_rhs.l2_norm();
      pcout << "MUMPS explicit relative residual: "
            << residual.l2_norm() / std::max(rhs_norm, 1e-30) << std::endl;

#else
      AssertThrow(false,
                  ExcMessage("This deal.II installation was built without "
                             "MUMPS support."));
#endif
    }
  else if (solution_method == "Direct")
    {
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
        solver_control,
        SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(100));
      if (preconditioner_type == "Calderon")
        {
#ifdef DEAL_II_WITH_MUMPS
          const double boundary_diameter = GridTools::diameter(comp_dom.tria);
          const double stabilization =
            calderon_stabilization /
            std::max(boundary_diameter, std::numeric_limits<double>::min());
          pcout << "Calderon preconditioner: M^-1 (W + beta M) M^-1, beta = "
                << stabilization << std::endl;
          CalderonOperatorPreconditioner calderon_preconditioner(
            calderon_hypersingular_matrix,
            calderon_mass_matrix,
            this_cpu_set,
            mpi_communicator,
            stabilization);
          solver.solve(cc, sol, system_rhs, calderon_preconditioner);
#else
          AssertThrow(false,
                      ExcMessage("The exact mass inverses in the Calderon "
                                 "preconditioner require MUMPS."));
#endif
        }
      else if (preconditioner_type == "CalderonDual")
        {
          const double boundary_diameter = GridTools::diameter(comp_dom.tria);
          const double stabilization =
            calderon_stabilization /
            std::max(boundary_diameter, std::numeric_limits<double>::min());
          pcout << "CalderonDual preconditioner: D^-1 [P^T (W + gamma M) P "
                   "+ Q^T B_b Q] D^-1, gamma = "
                << stabilization
                << ", bubble scaling = " << calderon_dual_bubble_scaling
                << std::endl;
          CalderonDualPreconditioner calderon_dual_preconditioner(
            calderon_hypersingular_matrix,
            calderon_mass_matrix,
            this_cpu_set,
            calderon_cpu_set,
            mpi_communicator,
            calderon_dual_cell_vertices,
            calderon_dual_vertex_valence,
            calderon_dual_cell_areas,
            stabilization,
            calderon_dual_bubble_scaling);
          solver.solve(cc, sol, system_rhs, calderon_dual_preconditioner);
        }
      else
        {
          if (preconditioner_type == "None")
            {
              PreconditionIdentity identity_preconditioner;
              solver.solve(cc, sol, system_rhs, identity_preconditioner);
            }
          else
            {
              assemble_preconditioner();
              if (preconditioner_type == "AMG")
                solver.solve(cc, sol, system_rhs, get_preconditioner(false));
              else
                solver.solve(cc, sol, system_rhs, preconditioner);
            }
        }
      pcout << "GMRES iterations: " << solver_control.last_step()
            << ", final residual: " << solver_control.last_value() << std::endl;
    }
  else
    {
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
        solver_control,
        SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(100));
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      TrilinosWrappers::PreconditionBase &fma_preconditioner =
        assemble_fmm_preconditioner();
      solver.solve(cc, sol, system_rhs, fma_preconditioner);
      pcout << "GMRES iterations: " << solver_control.last_step()
            << ", final residual: " << solver_control.last_value() << std::endl;
      collocation_fmm.print_statistics(pcout.get_stream());
    }

  TrilinosWrappers::MPI::Vector unpreconditioned_residual(this_cpu_set,
                                                          mpi_communicator);
  cc.vmult(unpreconditioned_residual, sol);
  unpreconditioned_residual -= system_rhs;
  pcout << "Unpreconditioned system relative residual: "
        << unpreconditioned_residual.l2_norm() /
             std::max(system_rhs.l2_norm(), 1e-30)
        << std::endl;

  // cc.apply_constraint(sol);
  // pcout<<"sol = [";
  // for (unsigned int i = 0; i < dh.n_dofs(); i++)
  //    pcout<<sol(i)<<"; ";
  // pcout<<"];"<<std::endl;

  for (auto i : this_cpu_set)
    {
      if (neumann_nodes(i) == 1)
        {
          phi(i) = sol(i);
        }
      else if (dirichlet_nodes(i) == 1)
        {
          dphi_dn(i) = sol(i);
        }
      else
        {
          Assert(robin_nodes(i) == 1, ExcInternalError());
          phi(i)     = sol(i);
          dphi_dn(i) = robin_rhs(i) - robin_matrix_diagonal(i) * phi(i);
        }
    }

  phi(this_cpu_set.nth_index_in_set(0)) = phi(this_cpu_set.nth_index_in_set(0));
  dphi_dn(this_cpu_set.nth_index_in_set(0)) =
    dphi_dn(this_cpu_set.nth_index_in_set(0));
  phi.compress(VectorOperation::insert);
  dphi_dn.compress(VectorOperation::insert);
}

template <int dim>
void
BEMProblem<dim>::solve_system(TrilinosWrappers::MPI::Vector       &phi,
                              TrilinosWrappers::MPI::Vector       &phi_imag,
                              TrilinosWrappers::MPI::Vector       &dphi_dn,
                              TrilinosWrappers::MPI::Vector       &dphi_dn_imag,
                              const TrilinosWrappers::MPI::Vector &tmp_rhs,
                              const TrilinosWrappers::MPI::Vector &tmp_rhs_imag)
{
  Teuchos::TimeMonitor LocalTimer(*LacSolveTime);

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(1000));

  system_rhs_blocked = 0;
  sol_blocked        = 0;
  alpha              = 0;

  compute_alpha();
  compute_rhs(system_rhs_blocked.block(0),
              system_rhs_blocked.block(1),
              tmp_rhs,
              tmp_rhs_imag);

  compute_constraints(constr_cpu_set, constraints, tmp_rhs);
  set_current_phi_component(current_component + 1);
  compute_constraints(constr_cpu_set, constraints_imag, tmp_rhs_imag);
  set_current_phi_component(current_component - 1);

  BlockConstrainedComplexOperator<TrilinosWrappers::MPI::BlockVector,
                                  BEMProblem<dim>>
    cc(*this, constraints, constraints_imag, constr_cpu_set, mpi_communicator);

  cc.distribute_rhs(system_rhs_blocked);
  system_rhs_blocked.compress(VectorOperation::insert);

  if (solution_method == "Direct")
    {
      assemble_preconditioner();
      auto precond = BlockPreconditioner<TrilinosWrappers::PreconditionBase>(
        get_preconditioner(false));

      sol_blocked.sadd(1., 0., system_rhs_blocked);
      solver.solve(cc, sol_blocked, system_rhs_blocked, precond);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      auto &fma_preconditioner = assemble_fmm_preconditioner();
      auto  precond = BlockPreconditioner<TrilinosWrappers::PreconditionBase>(
        fma_preconditioner);

      sol_blocked.sadd(1., 0., system_rhs_blocked);
      solver.solve(cc, sol_blocked, system_rhs_blocked, precond);
    }

  pcout << "Solve terminated at step " << solver_control.last_step()
        << std::endl;
  if (solution_method == "FMA")
    collocation_fmm.print_statistics(pcout.get_stream());

  for (auto i : this_cpu_set)
    {
      if (neumann_nodes(i) == 1)
        {
          phi(i)      = sol_blocked.block(0)(i);
          phi_imag(i) = sol_blocked.block(1)(i);
        }
      else if (dirichlet_nodes(i) == 1)
        {
          dphi_dn(i)      = sol_blocked.block(0)(i);
          dphi_dn_imag(i) = sol_blocked.block(1)(i);
        }
      else
        {
          AssertThrow(robin_nodes(i) == 1,
                      ExcMessage("Inconsistent boundary condition map"));

          phi(i)      = sol_blocked.block(0)(i);
          phi_imag(i) = sol_blocked.block(1)(i);

          // retrieval of dphi_dn using complex coefficients
          std::complex<double> c0_c1(robin_matrix_diagonal(i),
                                     robin_matrix_diagonal_imag(i));
          std::complex<double> c2_c1(robin_rhs(i), robin_rhs_imag(i));
          std::complex<double> ph(phi(i), phi_imag(i));
          std::complex<double> dph_dn(c2_c1 - c0_c1 * ph);

          dphi_dn(i)      = std::real(dph_dn);
          dphi_dn_imag(i) = std::imag(dph_dn);
        }
    }

  phi(this_cpu_set.nth_index_in_set(0)) = phi(this_cpu_set.nth_index_in_set(0));
  phi_imag(this_cpu_set.nth_index_in_set(0)) =
    phi_imag(this_cpu_set.nth_index_in_set(0));
  dphi_dn(this_cpu_set.nth_index_in_set(0)) =
    dphi_dn(this_cpu_set.nth_index_in_set(0));
  dphi_dn_imag(this_cpu_set.nth_index_in_set(0)) =
    dphi_dn_imag(this_cpu_set.nth_index_in_set(0));
  phi.compress(VectorOperation::insert);
  phi_imag.compress(VectorOperation::insert);
  dphi_dn.compress(VectorOperation::insert);
  dphi_dn_imag.compress(VectorOperation::insert);
}

// This method performs a Bem resolution,
// either in a direct or multipole method
template <int dim>
void
BEMProblem<dim>::solve(TrilinosWrappers::MPI::Vector       &phi,
                       TrilinosWrappers::MPI::Vector       &dphi_dn,
                       const TrilinosWrappers::MPI::Vector &tmp_rhs,
                       const Function<dim>                 *dirichlet_function,
                       const Function<dim>                 *dirichlet_gradient,
                       bool                                 reset_matrix)
{
  if (boundary_integral_formulation == "galerkin")
    {
      AssertThrow(dirichlet_function != nullptr,
                  ExcMessage("Galerkin Dirichlet assembly requires the "
                             "analytical boundary function."));
      unsigned int locally_dirichlet = 0;
      for (const auto i : this_cpu_set)
        if (dirichlet_nodes[i] > 0.5)
          ++locally_dirichlet;
      const unsigned int globally_dirichlet =
        Utilities::MPI::sum(locally_dirichlet, mpi_communicator);
      // PIBEM_FORCE_MIXED: the mixed solver also without a Neumann part, to
      // compare a pure Dirichlet problem through the same code.
      if (globally_dirichlet != dh.n_dofs() ||
          std::getenv("PIBEM_FORCE_MIXED") != nullptr)
        {
          AssertThrow(dirichlet_gradient != nullptr,
                      ExcMessage("The mixed Galerkin problem needs the "
                                 "gradient of the solution for the Neumann "
                                 "datum."));
          galerkin_reuse_operators = !reset_matrix;
          solve_galerkin_mixed({&phi},
                               {&dphi_dn},
                               {dirichlet_function},
                               {dirichlet_gradient});
          return;
        }
    }

  if (dirichlet_rhs_treatment == "exact_regularized")
    {
      AssertThrow(dirichlet_function != nullptr,
                  ExcMessage("Exact regularized Dirichlet RHS requires a "
                             "boundary function."));
      // The Galerkin assembly regularises the screened double layer too,
      // through the Laplace part of its kernel; the collocation one does not.
      AssertThrow((solution_method == "Direct" ||
                   boundary_integral_formulation == "galerkin") &&
                    (kernel_type == "laplace" ||
                     boundary_integral_formulation == "galerkin"),
                  ExcMessage("Exact regularized Dirichlet RHS currently "
                             "requires Direct Laplace, or the Galerkin "
                             "formulation."));
      unsigned int locally_dirichlet = 0;
      for (const auto i : this_cpu_set)
        if (dirichlet_nodes[i] > 0.5)
          ++locally_dirichlet;
      const unsigned int globally_dirichlet =
        Utilities::MPI::sum(locally_dirichlet, mpi_communicator);
      AssertThrow(globally_dirichlet == dh.n_dofs(),
                  ExcMessage("Exact regularized Dirichlet RHS currently "
                             "requires pure Dirichlet boundary conditions."));
    }

  // Another component reuses the matrices (reset_matrix false). The
  // Galerkin load vector is then assembled alone, without the single layer
  // (with it, when the matrices are assembled, if fused); the exact
  // regularized right-hand side of the collocation is assembled with the
  // matrices, and the Galerkin FMM of the Dirichlet problem is set up with
  // its datum, so both are set up again.
  const bool galerkin = boundary_integral_formulation == "galerkin";
  if (!galerkin && dirichlet_rhs_treatment == "exact_regularized")
    reset_matrix = true;

  if (solution_method == "Direct" && (reset_matrix || galerkin))
    {
      if (reset_matrix)
        {
          if (galerkin)
            galerkin_fused_dirichlet_function = dirichlet_function;
          assemble_system();
          galerkin_fused_dirichlet_function = nullptr;
        }
      else
        pcout << "Galerkin problem: the single layer of the previous "
                 "component"
              << std::endl;
      if (boundary_integral_formulation == "galerkin")
        assemble_exact_galerkin_dirichlet_rhs(*dirichlet_function,
                                              dirichlet_gradient);
      else if (dirichlet_rhs_treatment == "exact_regularized")
        assemble_exact_regularized_dirichlet_rhs(*dirichlet_function);
      // neumann_matrix.print(std::cout);
      // dirichlet_matrix.print(std::cout);
    }
  else if (galerkin)
    {
      // The single layer through the FMM; the right-hand side is assembled
      // as in the dense path when it is not fused with the matrix. A mixed
      // problem sets up its own operators in solve_galerkin_mixed().
      unsigned int locally_dirichlet = 0;
      for (const auto i : this_cpu_set)
        if (dirichlet_nodes[i] > 0.5)
          ++locally_dirichlet;
      if (Utilities::MPI::sum(locally_dirichlet, mpi_communicator) ==
          dh.n_dofs())
        {
          setup_galerkin_fmm(dirichlet_function, dirichlet_gradient);
          assemble_exact_galerkin_dirichlet_rhs(*dirichlet_function,
                                                dirichlet_gradient);
        }
    }
  else if (reset_matrix)
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      setup_collocation_fmm();
    }

  solve_system(phi, dphi_dn, tmp_rhs);
}

template <int dim>
void
BEMProblem<dim>::solve(
  TrilinosWrappers::MPI::Vector              &phi,
  TrilinosWrappers::MPI::Vector              &phi_imag,
  TrilinosWrappers::MPI::Vector              &dphi_dn,
  TrilinosWrappers::MPI::Vector              &dphi_dn_imag,
  const TrilinosWrappers::MPI::Vector        &tmp_rhs,
  const TrilinosWrappers::MPI::Vector        &tmp_rhs_imag,
  const std::array<const Function<dim> *, 2> &dirichlet_functions,
  const std::array<const Function<dim> *, 2> &dirichlet_gradients,
  bool                                        reset_matrix)
{
  if (boundary_integral_formulation == "galerkin")
    {
      AssertThrow(dirichlet_functions[0] != nullptr &&
                    dirichlet_functions[1] != nullptr,
                  ExcMessage("The complex Galerkin problem requires the "
                             "analytical boundary functions of the real and "
                             "of the imaginary part."));
      unsigned int locally_dirichlet = 0;
      for (const auto i : this_cpu_set)
        if (dirichlet_nodes[i] > 0.5)
          ++locally_dirichlet;
      const unsigned int globally_dirichlet =
        Utilities::MPI::sum(locally_dirichlet, mpi_communicator);
      // Without a Robin part the two parts are not coupled: on the whole
      // surface Dirichlet conditions are two real problems, the second with
      // the single layer of the first; otherwise one mixed system for both.
      if (globally_dirichlet == dh.n_dofs() &&
          std::getenv("PIBEM_FORCE_MIXED") == nullptr)
        {
          pcout << "Complex Galerkin problem: the real part" << std::endl;
          solve(phi,
                dphi_dn,
                tmp_rhs,
                dirichlet_functions[0],
                dirichlet_gradients[0],
                reset_matrix);
          pcout << "Complex Galerkin problem: the imaginary part" << std::endl;
          set_current_phi_component(current_component + 1);
          solve(phi_imag,
                dphi_dn_imag,
                tmp_rhs_imag,
                dirichlet_functions[1],
                dirichlet_gradients[1],
                false);
          set_current_phi_component(current_component - 1);
          return;
        }
      AssertThrow(dirichlet_gradients[0] != nullptr &&
                    dirichlet_gradients[1] != nullptr,
                  ExcMessage("The mixed Galerkin problem needs the "
                             "gradient of the solution for the Neumann "
                             "datum."));
      galerkin_reuse_operators = !reset_matrix;
      solve_galerkin_mixed({&phi, &phi_imag},
                           {&dphi_dn, &dphi_dn_imag},
                           {dirichlet_functions[0], dirichlet_functions[1]},
                           {dirichlet_gradients[0], dirichlet_gradients[1]});
      return;
    }

  if (reset_matrix)
    {
      if (solution_method == "Direct")
        assemble_system();
      else
        {
          AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

          setup_collocation_fmm();
        }
    }

  solve_system(phi, phi_imag, dphi_dn, dphi_dn_imag, tmp_rhs, tmp_rhs_imag);
}


template <int dim>
void
BEMProblem<dim>::compute_constraints(
  IndexSet                            &c_cpu_set,
  AffineConstraints<double>           &c,
  const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  Teuchos::TimeMonitor LocalTimer(*ConstraintsTime);

  if (boundary_integral_formulation == "galerkin")
    {
      c.clear();
      if (exact_geometry)
        ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, c);
      else
        DoFTools::make_hanging_node_constraints(dh, c);
      c.close();
      c_cpu_set = this_cpu_set;
      pcout << "Galerkin constraints: " << c.n_constraints()
            << " hanging-node constraints; trace space = " << scalar_fe_type
            << std::endl;
      return;
    }

  // We need both the normal vector and surface gradients to apply correctly
  // dirichlet-dirichlet double node constraints. compute_normals();
  compute_surface_gradients(tmp_rhs);

  // communication is needed here: there is one matrix per process: thus the
  // vector needed to set inhomogeneities has to be copied locally
  Vector<double> localized_surface_gradients(
    get_vector_surface_gradients_solution());
  Vector<double> localized_normals(vector_normals_solution);
  Vector<double> localized_dirichlet_nodes(dirichlet_nodes);
  Vector<double> localized_neumann_nodes(neumann_nodes);
  Vector<double> localized_robin_nodes(robin_nodes);
  // Vector<double> localized_robin_flags(robin_flags);
  // Vector<double> localized_robin_rhs(robin_rhs);
  // Vector<double> localized_robin_matrix_diagonal(robin_matrix_diagonal);
  Vector<double> loc_tmp_rhs(tmp_rhs.size());
  loc_tmp_rhs = tmp_rhs;

  // we start clearing the constraint matrix
  c.clear();

  // here we prepare the constraint matrix so as to account for the presence
  // hanging nodes
  AffineConstraints<double> c_hn;
  if (exact_geometry)
    ExactGeometry::make_mapping_hanging_node_constraints(*mapping, dh, c_hn);
  else
    DoFTools::make_hanging_node_constraints(dh, c_hn);
  c_hn.close();

  std::vector<types::subdomain_id> dofs_domain_association(dh.n_dofs());
  DoFTools::get_subdomain_association(dh, dofs_domain_association);
  // here we prepare the constraint matrix so as to account for the presence
  // of double and triple dofs

  // we start looping on the dofs
  for (types::global_dof_index i = 0; i < tmp_rhs.size(); i++)
    {
      // in the next line we compute the "first" among the set of double
      // nodes: this node is the first dirichlet node in the set, and if no
      // dirichlet node is there, we get the first neumann node
      auto doubles        = double_nodes_set[i];
      auto firstOfDoubles = *doubles.begin();
      for (auto j : doubles)
        {
          // if(this_cpu_set.is_element(j))
          if (localized_dirichlet_nodes(j) == 1)
            {
              firstOfDoubles = j;
              break;
            }
        }
      // do not bind from a robin node if possible
      if (localized_robin_nodes(firstOfDoubles) == 1)
        {
          // only neumann and robins remain
          for (auto j : doubles)
            {
              // if(this_cpu_set.is_element(j))
              if (localized_neumann_nodes(j) == 1)
                {
                  firstOfDoubles = j;
                  break;
                }
            }
        }

      // for each set of double nodes, we will perform the correction only
      // once, and precisely when the current node is the first of the set
      if (i == firstOfDoubles)
        {
          // the vector entry corresponding to the first node of the set does
          // i is the source of the constraints, thus we erase it from the set
          doubles.erase(i);

          // TODO: when coinciding with Robin nodes, the rhs should be updated
          // to reflect the known value
          // will these global constraints solve the need for then
          // redistributing the updated rhs?

          // if the current (first) node is a dirichlet node, for all its
          // neumann doubles we will impose that the potential is equal to
          // that of the first node: this means that in the matrix vector
          // product we will put the potential value of the double node
          if (localized_dirichlet_nodes(i) == 1)
            {
              for (auto j : doubles)
                {
                  // if(this_cpu_set.is_element(j))
                  {
                    if (localized_dirichlet_nodes(j) == 1)
                      {
                        // this is the dirichlet-dirichlet case on flat edges:
                        // here we impose that dphi_dn on the two (or more)
                        // sides is equal.
                        double normal_alignment = 0;

                        // types::global_dof_index owner_el_1 =
                        // DoFTools::count_dofs_with_subdomain_association
                        // (dh, dofs_domain_association[i]);
                        // types::global_dof_index owner_el_2 =
                        // DoFTools::count_dofs_with_subdomain_association
                        // (dh, dofs_domain_association[*it]);

                        for (unsigned int idim = 0; idim < dim; ++idim)
                          {
                            types::global_dof_index dummy_1 =
                              sub_wise_to_original[i];
                            types::global_dof_index dummy_2 =
                              sub_wise_to_original[j];

                            types::global_dof_index index1 =
                              vec_original_to_sub_wise[gradient_dh.n_dofs() /
                                                         dim * idim +
                                                       dummy_1];
                            types::global_dof_index index2 =
                              vec_original_to_sub_wise[gradient_dh.n_dofs() /
                                                         dim * idim +
                                                       dummy_2];

                            normal_alignment += localized_normals[index1] *
                                                localized_normals[index2];
                          }

                        if (std::abs(1.0 - normal_alignment) < 1e-4)
                          {
                            c.add_line(j);
                            c.add_entry(j, i, 1);
                          }
                        else if (continuos_gradient)
                          {
                            // this is the dirichlet-dirichlet case on sharp
                            // edges: both normal gradients can be computed
                            // from surface gradients of phi and assigned as
                            // BC
                            double norm_i_norm_j = 0;
                            double surf_j_norm_i = 0;
                            double surf_i_norm_j = 0;

                            // types::global_dof_index owner_el_1 =
                            // DoFTools::count_dofs_with_subdomain_association
                            // (dh, dofs_domain_association[i]);
                            // types::global_dof_index owner_el_2 =
                            // DoFTools::count_dofs_with_subdomain_association
                            // (dh, dofs_domain_association[*it]);

                            // We no longer have a std::vector of Point<dim>
                            // so we need to perform the scalar product
                            for (unsigned int idim = 0; idim < dim; ++idim)
                              {
                                types::global_dof_index dummy_1 =
                                  sub_wise_to_original[i];
                                types::global_dof_index dummy_2 =
                                  sub_wise_to_original[j];

                                types::global_dof_index index1 =
                                  vec_original_to_sub_wise
                                    [gradient_dh.n_dofs() / dim * idim +
                                     dummy_1];
                                types::global_dof_index index2 =
                                  vec_original_to_sub_wise
                                    [gradient_dh.n_dofs() / dim * idim +
                                     dummy_2];

                                norm_i_norm_j += localized_normals[index1] *
                                                 localized_normals[index2];
                                surf_j_norm_i +=
                                  localized_surface_gradients[index2] *
                                  localized_normals[index1];
                                surf_i_norm_j +=
                                  localized_surface_gradients[index1] *
                                  localized_normals[index2];
                              }

                            double this_normal_gradient =
                              (1.0 / (1.0 - norm_i_norm_j * norm_i_norm_j)) *
                              (surf_j_norm_i +
                               (surf_i_norm_j) * (norm_i_norm_j));
                            double other_normal_gradient =
                              (1.0 / (1.0 - norm_i_norm_j * norm_i_norm_j)) *
                              (surf_i_norm_j +
                               (surf_j_norm_i) * (norm_i_norm_j));

                            c.add_line(i);
                            c.set_inhomogeneity(i, this_normal_gradient);
                            c.add_line(j);
                            c.set_inhomogeneity(j, other_normal_gradient);
                          }
                      }
                    else
                      {
                        c.add_line(j);
                        c.set_inhomogeneity(j, loc_tmp_rhs(i));
                      }
                  }
                }
            }

          // if the current (first) node is a neumann node, for all its
          // doubles we will impose that the potential is equal to that of the
          // first node: this means that in the matrix vector product we will
          // put the difference between the potential at the fist node in the
          // doubles set, and the current double node
          if (localized_dirichlet_nodes(i) == 0)
            {
              for (auto j : doubles)
                {
                  c.add_line(j);
                  c.add_entry(j, i, 1);
                }
            }
        }

      // pcout << "processed double node constraints for dof " << i <<
      // std::endl;
    }

  c.merge(c_hn);
  c.close();

  c_cpu_set.clear();
  c_cpu_set.set_size(this_cpu_set.size());
  for (auto i : this_cpu_set)
    {
      c_cpu_set.add_index(i);
      if (c.is_constrained(i))
        {
          const std::vector<std::pair<types::global_dof_index, double>>
            *entries = c.get_constraint_entries(i);
          for (const auto &pair : *entries)
            {
              c_cpu_set.add_index(pair.first);
            }
        }
    }
  c_cpu_set.compress();
}


template <int dim>
void
BEMProblem<dim>::assemble_preconditioner()
{
  // Each owned matrix row also contains columns owned by other ranks.
  // Import column data collectively before the rank-local assembly loops.
  const Vector<double> column_dirichlet_nodes(dirichlet_nodes);
  const Vector<double> column_robin_nodes(robin_nodes);
  const Vector<double> column_robin_diagonal(robin_matrix_diagonal);
  if (!is_preconditioner_initialized)
    {
      for (auto i : this_cpu_set)
        {
          types::global_dof_index start_helper =
            ((i) > preconditioner_band / 2) ? (i - preconditioner_band / 2) :
                                              ((types::global_dof_index)0);
          for (types::global_dof_index j = start_helper;
               j <
               std::min((types::global_dof_index)(i + preconditioner_band / 2),
                        (types::global_dof_index)dh.n_dofs());
               ++j)
            {
              preconditioner_sparsity_pattern.add(i, j);
            }
        }
      preconditioner_sparsity_pattern.compress();
      band_system.reinit(preconditioner_sparsity_pattern);
      is_preconditioner_initialized = true;
    }
  else
    {
      band_system = 0;
    }

  for (auto i : this_cpu_set)
    {
      if (constraints.is_constrained(i))
        {
          band_system.add(i, i, 1);
        }
      else
        {
          types::global_dof_index start_helper =
            ((i) > preconditioner_band / 2) ? (i - preconditioner_band / 2) :
                                              ((types::global_dof_index)0);

          for (types::global_dof_index j = start_helper;
               j <
               std::min((types::global_dof_index)i + preconditioner_band / 2,
                        (types::global_dof_index)dh.n_dofs());
               ++j)
            {
              // if (dirichlet_nodes(i) == 0)
              if (column_dirichlet_nodes(j) == 0)
                {
                  // Nodo di Neumann - or Robin
                  band_system.add(i, j, neumann_matrix(i, j));

                  // TODO: account for Robin node
                  if (column_robin_nodes(j) == 1)
                    {
                      band_system.add(i,
                                      j,
                                      dirichlet_matrix(i, j) *
                                        column_robin_diagonal(j));
                    }

                  if (boundary_integral_formulation == "galerkin")
                    band_system.add(i, j, 0.5 * jump_matrix(i, j));
                  else if (i == j)
                    band_system.add(i, j, alpha(i));
                }
              else
                {
                  // Nodo di Dirichlet
                  band_system.add(i, j, -dirichlet_matrix(i, j));
                }
            }
        }
    }

  // The other preconditioner types reach the GMRES of the collocation, and
  // of the Galerkin formulation without CG, with the ILU of the band.
  if (preconditioner_type == "AMG")
    {
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      data.higher_order_elements = fe->degree > 1;

      preconditioner_amg.initialize(band_system, data);
    }
  else
    {
      preconditioner.initialize(band_system);
    }
}

template <int dim>
void
BEMProblem<dim>::setup_collocation_fmm()
{
  if constexpr (dim == 3)
    {
      Teuchos::TimeMonitor       LocalTimer(*AssembleTime);
      LaplaceFMM::AdditionalData data;
      data.order               = fma.truncation_order();
      data.theta               = fma.opening_criterion();
      data.max_points_per_leaf = fma.points_per_leaf();
      data.kappa = kernel_type == "screened" ? screened_kappa : 0.;
      pcout << "Setting up the collocation FMM: order " << data.order
            << ", opening criterion " << data.theta << ", at most "
            << data.max_points_per_leaf << " points per leaf" << std::endl;
      std::vector<Point<dim>> support_points(dh.n_dofs());
      DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                         dh,
                                                         support_points);
      // The rows of the dense assembly on the near cells; the regular ones
      // are those of the point sums.
      CollocationRuleStatistics statistics;
      collocation_fmm.reinit(
        dh,
        *mapping,
        *quadrature,
        support_points,
        this_cpu_set,
        quasi_singular_distance_ratio,
        [&](const typename DoFHandler<dim - 1, dim>::active_cell_iterator &cell,
            const std::vector<types::global_dof_index>                    &dofs,
            const types::global_dof_index                                  i,
            Vector<double> &neumann_row,
            Vector<double> &dirichlet_row) {
          return integrate_collocation_rows(cell,
                                            dofs,
                                            i,
                                            support_points,
                                            nullptr,
                                            neumann_row,
                                            dirichlet_row,
                                            statistics) !=
                 CollocationRule::regular;
        },
        data);
    }
  else
    AssertThrow(false, ExcMessage("FMA only works in 3D"));
}



template <int dim>
void
BEMProblem<dim>::collocation_fmm_products(
  const TrilinosWrappers::MPI::Vector &phi,
  const TrilinosWrappers::MPI::Vector &dphi_dn,
  TrilinosWrappers::MPI::Vector       &neumann,
  TrilinosWrappers::MPI::Vector       &dirichlet) const
{
  AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));
  collocation_fmm.vmult(phi, dphi_dn, neumann, dirichlet);
}



template <int dim>
TrilinosWrappers::PreconditionBase &
BEMProblem<dim>::assemble_fmm_preconditioner()
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  // The entries of assemble_preconditioner() on the near band: the double
  // layer and the free term on the columns of the unknown potential (with
  // the Robin flux -(a/b) u), minus the single layer on those of the
  // unknown flux.
  const Vector<double> column_dirichlet_nodes(dirichlet_nodes);
  const Vector<double> column_robin_nodes(robin_nodes);
  const Vector<double> column_robin_diagonal(robin_matrix_diagonal);
  const TrilinosWrappers::SparseMatrix &band_n = collocation_fmm.band_neumann();
  const TrilinosWrappers::SparseMatrix &band_d =
    collocation_fmm.band_dirichlet();
  fmm_band_system.reinit(band_n);
  fmm_band_system = 0;
  for (const auto i : this_cpu_set)
    {
      if (constraints.is_constrained(i))
        {
          fmm_band_system.set(i, i, 1.);
          continue;
        }
      auto d = band_d.begin(i);
      for (auto n = band_n.begin(i); n != band_n.end(i); ++n, ++d)
        {
          const types::global_dof_index j = n->column();
          Assert(d->column() == j, ExcInternalError());
          double value;
          if (column_dirichlet_nodes(j) == 0)
            {
              value = n->value();
              if (column_robin_nodes(j) == 1)
                value += d->value() * column_robin_diagonal(j);
              if (i == j)
                value += alpha(i);
            }
          else
            value = -d->value();
          fmm_band_system.set(i, j, value);
        }
    }
  fmm_band_system.compress(VectorOperation::insert);
  if (preconditioner_type == "AMG")
    {
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      data.higher_order_elements = fe->degree > 1;
      preconditioner_amg.initialize(fmm_band_system, data);
      return preconditioner_amg;
    }
  preconditioner.initialize(fmm_band_system);
  return preconditioner;
}



template <int dim>
void
BEMProblem<dim>::assemble_preconditioner_complex()
{
  // Each owned matrix row also contains columns owned by other ranks.
  // Import column data collectively before the rank-local assembly loops.
  const Vector<double> column_dirichlet_nodes(dirichlet_nodes);
  const Vector<double> column_robin_nodes(robin_nodes);
  const Vector<double> column_robin_diagonal(robin_matrix_diagonal);
  const Vector<double> column_robin_diagonal_imag(robin_matrix_diagonal_imag);
  if (!is_preconditioner_complex_initialized)
    {
      for (auto i : this_cpu_set)
        {
          auto from = (i >= preconditioner_band / 2) ?
                        (i - preconditioner_band / 2) :
                        (types::global_dof_index)0;
          auto to =
            std::min((types::global_dof_index)(i + preconditioner_band / 2),
                     this_cpu_set.size());

          for (auto j = from; j < to; ++j)
            {
              preconditioner_complex_sparsity_pattern.add(i, j);
              preconditioner_complex_sparsity_pattern.add(
                i + this_cpu_set.size(), j + this_cpu_set.size());
            }
        }
      preconditioner_complex_sparsity_pattern.compress();
      band_system_complex.reinit(preconditioner_complex_sparsity_pattern);
      is_preconditioner_complex_initialized = true;
    }
  else
    {
      band_system_complex = 0;
    }

  for (auto i : this_cpu_set)
    {
      if (constraints.is_constrained(i))
        {
          band_system_complex.add(i, i, 1);
          band_system_complex.add(i + this_cpu_set.size(),
                                  i + this_cpu_set.size(),
                                  1);
        }
      else
        {
          auto from = (i >= preconditioner_band / 2) ?
                        (i - preconditioner_band / 2) :
                        (types::global_dof_index)0;
          auto to =
            std::min((types::global_dof_index)i + preconditioner_band / 2,
                     this_cpu_set.size());

          for (auto j = from; j < to; ++j)
            {
              if (column_dirichlet_nodes(j) == 0)
                {
                  // neumann and robin nodes are implicitly merged, here
                  band_system_complex.add(i, j, neumann_matrix(i, j));
                  band_system_complex.add(i + this_cpu_set.size(),
                                          j + this_cpu_set.size(),
                                          neumann_matrix(i, j));

                  if (column_robin_nodes(j) == 1)
                    {
                      // scale the row from D, using the robin matrix
                      // diagonal
                      // for now, ignore the pairing parts
                      band_system_complex.add(i,
                                              j,
                                              dirichlet_matrix(i, j) *
                                                column_robin_diagonal(j));
                      band_system_complex.add(i + this_cpu_set.size(),
                                              j + this_cpu_set.size(),
                                              dirichlet_matrix(i, j) *
                                                column_robin_diagonal(j));

                      // check for pairing elements
                      if ((robin_nodes(i) == 1) &&
                          j + this_cpu_set.size() < i + preconditioner_band / 2)
                        {
                          // the pairing element of the current real
                          // variable and its imag version is available in
                          // the band
                          band_system_complex.add(i,
                                                  j + this_cpu_set.size(),
                                                  -dirichlet_matrix(i, j) *
                                                    column_robin_diagonal_imag(
                                                      j));
                          band_system_complex.add(i + this_cpu_set.size(),
                                                  j,
                                                  dirichlet_matrix(i, j) *
                                                    column_robin_diagonal_imag(
                                                      j));
                        }
                    }

                  if (i == j)
                    {
                      band_system_complex.add(i, j, alpha(i));
                      band_system_complex.add(i + this_cpu_set.size(),
                                              j + this_cpu_set.size(),
                                              alpha(i));
                    }
                }
              else
                {
                  band_system_complex.add(i, j, -dirichlet_matrix(i, j));
                  band_system_complex.add(i + this_cpu_set.size(),
                                          j + this_cpu_set.size(),
                                          -dirichlet_matrix(i, j));
                }
            }
        }
    }

  if (preconditioner_type == "ILU")
    {
      preconditioner_complex.initialize(band_system_complex);
    }
  else if (preconditioner_type == "AMG")
    {
      TrilinosWrappers::PreconditionAMG::AdditionalData data;
      data.higher_order_elements = fe->degree > 1;

      preconditioner_complex_amg.initialize(band_system_complex, data);
    }
  else
    {
      AssertThrow(false, ExcMessage("Invalid preconditioner type"));
    }
}


template <int dim>
void
BEMProblem<dim>::compute_normals()
{
  Teuchos::TimeMonitor LocalTimer(*NormalsTime);
  vector_normals_solution.reinit(vector_this_cpu_set, mpi_communicator);

  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;

  TrilinosWrappers::SparseMatrix vector_normals_matrix;
  TrilinosWrappers::MPI::Vector  vector_normals_rhs(vector_this_cpu_set,
                                                   mpi_communicator);

  vector_normals_matrix.reinit(vector_sparsity_pattern);

  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  const unsigned int vector_n_q_points = vector_fe_v.n_quadrature_points;

  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;

  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);

  std::vector<Vector<double>> q_vector_normals_solution(vector_n_q_points,
                                                        Vector<double>(dim));

  FullMatrix<double> local_normals_matrix(vector_dofs_per_cell,
                                          vector_dofs_per_cell);
  Vector<double>     local_normals_rhs(vector_dofs_per_cell);

  for (const auto &vector_cell : gradient_dh.active_cell_iterators())
    {
      if (vector_cell->subdomain_id() == this_mpi_process)
        {
          vector_fe_v.reinit(vector_cell);
          local_normals_matrix = 0;
          local_normals_rhs    = 0;
          const std::vector<Tensor<1, dim>> &vector_node_normals =
            vector_fe_v.get_normal_vectors();
          unsigned int comp_i, comp_j;

          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            {
              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  comp_i = gradient_fe->system_to_component_index(i).first;
                  for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
                    {
                      comp_j = gradient_fe->system_to_component_index(j).first;
                      if (comp_i == comp_j)
                        {
                          local_normals_matrix(i, j) +=
                            vector_fe_v.shape_value(i, q) *
                            vector_fe_v.shape_value(j, q) * vector_fe_v.JxW(q);
                        }
                    }

                  local_normals_rhs(i) += (vector_fe_v.shape_value(i, q)) *
                                          vector_node_normals[q][comp_i] *
                                          vector_fe_v.JxW(q);
                }
            }
          vector_cell->get_dof_indices(vector_local_dof_indices);

          vector_constraints.distribute_local_to_global(
            local_normals_matrix,
            local_normals_rhs,
            vector_local_dof_indices,
            vector_normals_matrix,
            vector_normals_rhs);
        }
    }

  vector_normals_matrix.compress(VectorOperation::add);
  vector_normals_rhs.compress(VectorOperation::add);

  if (nodal_normals == "interpolation")
    {
      // The normals of the mapping at the support points of every cell,
      // summed over the cells that share a node and normalised below: on a
      // smooth surface the normal at the node, where the projection averages
      // the normals of the neighbouring cells.
      const Quadrature<dim - 1> nodes(
        gradient_fe->base_element(0).get_unit_support_points());
      FEValues<dim - 1, dim> at_nodes(*mapping,
                                      *gradient_fe,
                                      nodes,
                                      update_normal_vectors);
      vector_normals_solution = 0;
      for (const auto &cell : gradient_dh.active_cell_iterators())
        if (cell->subdomain_id() == this_mpi_process)
          {
            at_nodes.reinit(cell);
            cell->get_dof_indices(vector_local_dof_indices);
            std::vector<double> values(vector_dofs_per_cell);
            for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
              {
                const auto component =
                  gradient_fe->system_to_component_index(i);
                values[i] =
                  at_nodes.normal_vector(component.second)[component.first];
              }
            vector_normals_solution.add(vector_local_dof_indices, values);
          }
      vector_normals_solution.compress(VectorOperation::add);
    }
  else
    {
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
        solver_control,
        SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));
      TrilinosWrappers::PreconditionAMG mass_prec;
      mass_prec.initialize(vector_normals_matrix);

      solver.solve(vector_normals_matrix,
                   vector_normals_solution,
                   vector_normals_rhs,
                   mass_prec);
    }

  vector_constraints.distribute(vector_normals_solution);

  // The summed normals of the interpolation are normalised. The projection
  // is not: it is the normal of the discrete surface in the space of the
  // gradients, and only with it is the Neumann datum of the collocation
  // consistent with that surface. Normalised, the potential of the DN
  // sample on the sphere converges at order 2 instead of 3 (Direct, three
  // refinements: 5.2e-4 against 3.0e-5).
  if (nodal_normals == "interpolation")
    {
      for (const auto i : this_cpu_set)
        {
          const types::global_dof_index original_index =
            sub_wise_to_original[i];
          std::vector<types::global_dof_index> component_indices(dim);
          double                               normal_norm_square = 0.0;
          for (unsigned int d = 0; d < dim; ++d)
            {
              component_indices[d] =
                vec_original_to_sub_wise[d * dh.n_dofs() + original_index];
              Assert(vector_this_cpu_set.is_element(component_indices[d]),
                     ExcInternalError());
              const double component =
                vector_normals_solution[component_indices[d]];
              normal_norm_square += component * component;
            }

          const double normal_norm = std::sqrt(normal_norm_square);
          AssertThrow(normal_norm > 1e-14,
                      ExcMessage("Cannot normalize a zero nodal normal."));
          for (unsigned int d = 0; d < dim; ++d)
            vector_normals_solution[component_indices[d]] /= normal_norm;
        }
      vector_normals_solution.compress(VectorOperation::insert);
    }
}

template <int dim>
void
BEMProblem<dim>::adaptive_refinement(
  const TrilinosWrappers::MPI::Vector &error_vector)
{
  // The indicators of the mixed Galerkin problem, if the solve computed
  // them: refine the cells that carry the given fraction of their sum.
  if (mixed_indicators.size() == comp_dom.tria.n_active_cells())
    {
      GridRefinement::refine_and_coarsen_fixed_fraction(
        comp_dom.tria,
        mixed_indicators,
        mixed_galerkin_refinement_fraction,
        0.);
      unsigned int marked = 0;
      for (const auto &cell : comp_dom.tria.active_cell_iterators())
        if (cell->refine_flag_set())
          ++marked;
      pcout << "Adaptive refinement: " << marked << " of "
            << comp_dom.tria.n_active_cells() << " cells marked" << std::endl;
      mixed_indicators.reinit(0);
      comp_dom.tria.prepare_coarsening_and_refinement();
      comp_dom.tria.execute_coarsening_and_refinement();
      return;
    }
  Vector<float>  estimated_error_per_cell(comp_dom.tria.n_active_cells());
  Vector<double> helper(error_vector);

  KellyErrorEstimator<dim - 1, dim>::estimate(
    *mapping, dh, QGauss<dim - 2>(3), {}, helper, estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number(comp_dom.tria,
                                                  estimated_error_per_cell,
                                                  refinement_threshold,
                                                  coarsening_threshold);

  comp_dom.tria.prepare_coarsening_and_refinement();
  comp_dom.tria.execute_coarsening_and_refinement();
}

template class BEMProblem<2>;
template class BEMProblem<3>;
