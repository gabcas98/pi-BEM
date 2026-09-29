

#include <cstdlib>
#include "../include/bem_problem.h"

#include <deal.II/numerics/error_estimator.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "../include/laplace_kernel.h"
#include "../include/galerkin_panel_pair_integrator.h"
#include "../include/surface_potential_evaluator.h"
#include "../include/screened_kernel.h"
#include "../include/singular_kernel_integral.h"
#include "../include/quasi_singular_kernel_integral.h"
#include "../include/telles_quadrature.h"
#include "Teuchos_TimeMonitor.hpp"

using Teuchos::RCP;
using Teuchos::Time;
using Teuchos::TimeMonitor;
using namespace std;

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
  quadrature_vectors_converged(const Vector<double>      &previous,
                               const Vector<double>      &current,
                               const double               relative_tolerance,
                               const double               absolute_tolerance,
                               QuadratureConvergenceState &state)
  {
    AssertDimension(previous.size(), current.size());
    double difference = 0.0;
    double scale      = 0.0;
    for (types::global_dof_index i = 0; i < current.size(); ++i)
      {
        difference =
          std::max(difference, std::abs(current[i] - previous[i]));
        scale = std::max(scale, std::max(std::abs(current[i]),
                                         std::abs(previous[i])));
      }
    const double tolerance =
      absolute_tolerance + relative_tolerance * scale;
    const bool direct_estimate_converged = difference <= tolerance;

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
  quadrature_scalars_converged(const double               previous,
                               const double               current,
                               const double               relative_tolerance,
                               const double               absolute_tolerance,
                               QuadratureConvergenceState &state)
  {
    const double difference = std::abs(current - previous);
    const double scale = std::max(std::abs(current), std::abs(previous));
    const double tolerance = absolute_tolerance + relative_tolerance * scale;
    const bool direct_estimate_converged = difference <= tolerance;

    double contraction = std::numeric_limits<double>::infinity();
    bool asymptotic_estimate_converged = false;
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
    return std::min(maximum_order,
                    std::max(current_order + 1, grown_order));
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
      const TrilinosWrappers::SparseMatrix                     &matrix,
      const std::vector<std::vector<types::global_dof_index>> &cell_dofs,
      const std::vector<bool>                                  &applies_cell,
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
    mutable SparseDirectMUMPS              mass_inverse;
    mutable TrilinosWrappers::MPI::Vector  mass_inverse_source;
    mutable TrilinosWrappers::MPI::Vector  operator_result;
    mutable TrilinosWrappers::MPI::Vector  mass_result;
    const double                           stabilization;
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
      const std::vector<unsigned int> &vertex_valence,
      const std::vector<double>       &cell_areas,
      const double                     hypersingular_stabilization,
      const double                     bubble_scaling)
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
      std::vector<double> area_scaled(cell_vertices.size(), 0.0);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size(); ++cell)
        area_scaled[cell] = localized_src[cell] / cell_areas[cell];

      const std::vector<double> projected = apply_p(area_scaled);
      vertex_source = 0.0;
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
      for (types::global_dof_index cell = 0; cell < cell_vertices.size(); ++cell)
        for (const auto vertex : cell_vertices[cell])
          vertex_values[vertex] +=
            cell_values[cell] / static_cast<double>(vertex_valence[vertex]);
      return vertex_values;
    }

    std::vector<double>
    apply_pt(const Vector<double> &vertex_values) const
    {
      std::vector<double> cell_values(cell_vertices.size(), 0.0);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size(); ++cell)
        for (const auto vertex : cell_vertices[cell])
          cell_values[cell] +=
            vertex_values[vertex] /
            static_cast<double>(vertex_valence[vertex]);
      return cell_values;
    }

    std::vector<double>
    apply_q(const std::vector<double> &cell_values) const
    {
      const std::vector<double> projected = apply_p(cell_values);
      std::vector<double>       result(cell_values);
      for (types::global_dof_index cell = 0; cell < cell_vertices.size(); ++cell)
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
    const std::vector<unsigned int> &vertex_valence;
    const std::vector<double>       &cell_areas;
    mutable TrilinosWrappers::MPI::Vector vertex_source;
    mutable TrilinosWrappers::MPI::Vector vertex_operator_result;
    mutable TrilinosWrappers::MPI::Vector vertex_mass_result;
    const double hypersingular_stabilization;
    const double bubble_scaling;
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
                          // const unsigned int fe_degree,
                          MPI_Comm comm)
  : pcout(std::cout)
  , comp_dom(comp_dom)
  , dh(comp_dom.tria)
  , gradient_dh(comp_dom.tria)
  , calderon_dh(comp_dom.tria)
  , mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}
template <>
BEMProblem<2>::BEMProblem(ComputationalDomain<2> &comp_dom,
                          // const unsigned int fe_degree,
                          MPI_Comm comm)
  : pcout(std::cout)
  , comp_dom(comp_dom)
  , dh(comp_dom.tria)
  , gradient_dh(comp_dom.tria)
  , calderon_dh(comp_dom.tria)
  , mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}


namespace
{
  template <int dim, int spacedim, bool system = false>
  std::unique_ptr<FiniteElement<dim, spacedim>>
  createFiniteElementPointer(const std::string &type, const unsigned int order)
  {
    if (type == "FEQ")
      {
        if (system)
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FESystem<dim, spacedim>(FE_Q<dim, spacedim>(order),
                                          spacedim));
          }
        else
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FE_Q<dim, spacedim>(order));
          }
      }
    else if (type == "FEDGQ")
      {
        if (system)
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FESystem<dim, spacedim>(FE_DGQ<dim, spacedim>(order),
                                          spacedim));
          }
        else
          {
            return std::unique_ptr<FiniteElement<dim, spacedim>>(
              new FE_DGQ<dim, spacedim>(order));
          }
      }
    else if (type == "FEDGQGauss")
      {
        const FE_DGQArbitraryNodes<dim, spacedim> scalar_element(
          QGauss<1>(order + 1));
        if (system)
          return std::unique_ptr<FiniteElement<dim, spacedim>>(
            new FESystem<dim, spacedim>(scalar_element, spacedim));
        return std::unique_ptr<FiniteElement<dim, spacedim>>(
          new FE_DGQArbitraryNodes<dim, spacedim>(QGauss<1>(order + 1)));
      }

    AssertThrow(false, ExcMessage("Unknown finite element type: " + type));
    return nullptr;
  }
} // namespace
template <int dim>
void
BEMProblem<dim>::reinit()
{
  // ENTRY
  Teuchos::TimeMonitor LocalTimer(*ReinitTime);

  fe          = createFiniteElementPointer<dim - 1, dim, false>(scalar_fe_type,
                                                       scalar_fe_order);
  gradient_fe = createFiniteElementPointer<dim - 1, dim, true>(vector_fe_type,
                                                               vector_fe_order);
  // fe = new FE_DGQArbitraryNodes<dim-1, dim>(QGauss<1> (2));
  // gradient_fe = new
  // FESystem<dim-1,dim>(FE_DGQArbitraryNodes<dim-1,dim>(QGauss<1> (2)),dim);
  // // auto hhh = new FE_DGQArbitraryNodes<dim-1, dim>(QGauss<1> (2));
  std::string foo = fe->get_name();
  std::cout << foo << std::endl;
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
  // mapping_degree = fe->get_degree();
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
                map_vector[vec_original_to_sub_wise[ii + gradient_dh.n_dofs() /
                                                           dim]] *=
                  comp_dom.spheroid_z_axis;
            }
        }
      if (mapping_type == "FE")
        mapping = std::make_shared<MappingFEField<dim - 1, dim>>(gradient_dh,
                                                                 map_vector);
      else
        mapping = std::make_shared<MappingQ<dim - 1, dim>>(mapping_degree);
    }



  const types::global_dof_index n_dofs = dh.n_dofs();

  pcout << dh.n_dofs() << " " << gradient_dh.n_dofs() << std::endl;
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);

  DoFTools::get_subdomain_association(dh, dofs_domain_association);
  std::vector<types::subdomain_id> vector_dofs_domain_association(
    gradient_dh.n_dofs());

  DoFTools::get_subdomain_association(gradient_dh,
                                      vector_dofs_domain_association);


  this_cpu_set.clear();
  vector_this_cpu_set.clear();
  this_cpu_set.set_size(n_dofs);
  vector_this_cpu_set.set_size(gradient_dh.n_dofs());


  // We compute this two vector in order to use an eventual
  // DoFRenumbering::subdomain_wise At the time being we don't. We need to
  // decide the better strategy.


  // We need to enforce consistency between the non-ghosted IndexSets.
  // To be changed accordingly with the DoFRenumbering strategy.
  pcout << "you are using " << sizeof(dh.n_dofs()) << " bytes indices"
        << std::endl;
  pcout << "setting cpu_sets" << std::endl;

  for (types::global_dof_index i = 0; i < n_dofs; ++i)
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


  // for (unsigned int i=0; i<gradient_dh.n_dofs(); ++i)
  //   if (vector_dofs_domain_association[i] == this_mpi_process)
  //     {
  //       vector_this_cpu_set.add_index(i);
  //       // for(unsigned int idim=0; idim<dim; ++idim)
  //       // {
  //       //   vector_this_cpu_set.add_index(i*dim+idim);
  //       // }
  //     }

  this_cpu_set.compress();
  vector_this_cpu_set.compress();
  // std::cout<<"set the cpu sets"<<std::endl;
  // std::vector<types::global_dof_index> localized_ndfos(n_mpi_processes);
  // std::vector<types::global_dof_index>
  // localized_vector_ndfos(n_mpi_processes); start_per_process.resize
  // (n_mpi_processes); vector_start_per_process.resize (n_mpi_processes);
  //
  // localized_ndfos[this_mpi_process] = this_cpu_set.n_elements();
  // localized_vector_ndfos[this_mpi_process] =
  // vector_this_cpu_set.n_elements();
  //
  // Utilities::MPI::sum (localized_ndfos, mpi_communicator, start_per_process);
  // Utilities::MPI::sum (localized_vector_ndfos, mpi_communicator,
  // vector_start_per_process);
  //
  // for(unsigned int i=start_per_process.size()-1; i>0; --i)
  // {
  //   start_per_process[i] = start_per_process[i-1];
  //   vector_start_per_process[i] = vector_start_per_process[i-1];
  // }
  // start_per_process[0] = 0;
  // vector_start_per_process[0] = 0;
  // for(unsigned int i=2; i<start_per_process.size(); ++i)
  // {
  //   start_per_process[i] += start_per_process[i-1];
  //   vector_start_per_process[i] += vector_start_per_process[i-1];
  // }
  // start_per_process[0] = 0;
  // vector_start_per_process[0] = 0;

  // std::cout<<this_mpi_process<<" "<<start_per_process[this_mpi_process]<<"
  // "<<vector_start_per_process[this_mpi_process]<<std::endl;


  // At this point we just need to create a ghosted IndexSet for the scalar
  // DoFHandler. This can be through the builtin dealii function.
  // this_cpu_set.print(std::cout);
  MPI_Barrier(mpi_communicator);
  ghosted_set.clear();
  ghosted_set.set_size(dh.n_dofs());
  ghosted_set =
    DoFTools::dof_indices_with_subdomain_association(dh, this_mpi_process);
  ghosted_set.compress();
  // std::cout<<"set ghosted set"<<std::endl;

  // standard TrilinosWrappers::MPI::Vector reinitialization.
  system_rhs.reinit(this_cpu_set, mpi_communicator);
  exact_regularized_dirichlet_rhs.reinit(this_cpu_set, mpi_communicator);
  sol.reinit(this_cpu_set, mpi_communicator);
  alpha.reinit(this_cpu_set, mpi_communicator);
  serv_phi.reinit(this_cpu_set, mpi_communicator);
  serv_dphi_dn.reinit(this_cpu_set, mpi_communicator);
  serv_tmp_rhs.reinit(this_cpu_set, mpi_communicator);

  if (preconditioner_type == "CalderonDual" ||
      galerkin_neumann_formulation == "adjoint_double_layer")
    initialize_calderon_dual_space();


  // TrilinosWrappers::SparsityPattern for the BEM matricesreinitialization
  pcout << "re-initializing sparsity patterns and matrices" << std::endl;
  if (solution_method == "Direct")
    {
      full_sparsity_pattern.reinit(this_cpu_set, mpi_communicator);

      for (auto i : this_cpu_set)
        {
          for (types::global_dof_index j = 0; j < dh.n_dofs(); ++j)
            full_sparsity_pattern.add(i, j);
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

  dirichlet_nodes.reinit(this_cpu_set, mpi_communicator);
  neumann_nodes.reinit(this_cpu_set, mpi_communicator);
  compute_dirichlet_and_neumann_dofs_vectors();
  compute_double_nodes_set();

  fma.init_fma(dh,
               double_nodes_set,
               dirichlet_nodes,
               *mapping,
               quadrature_order,
               singular_quadrature_order);



  // We need a TrilinosWrappers::MPI::Vector to reinit the SparsityPattern for
  // the parallel mass matrices.
  TrilinosWrappers::MPI::Vector helper(vector_this_cpu_set, mpi_communicator);
  // These are just for test
  // IndexSet vector_active_dofs;
  // IndexSet vector_relevant_dofs;
  IndexSet trial_index_set;
  // vector_active_dofs.clear();
  // vector_relevant_dofs.clear();
  trial_index_set.clear();
  // DoFTools::extract_locally_active_dofs(gradient_dh, vector_active_dofs);//,
  // vector_active_dofs);
  trial_index_set =
    DoFTools::dof_indices_with_subdomain_association(gradient_dh,
                                                     this_mpi_process);
  // Assert(trial_index_set == vector_this_cpu_set, ExcNotImplemented());
  // // The following functions returns the entire dof set.
  // DoFTools::extract_locally_relevant_dofs(gradient_dh, vector_relevant_dofs);
  // pcout<<vector_active_dofs.n_elements()<<"
  // "<<vector_relevant_dofs.n_elements()<<"
  // "<<vector_this_cpu_set.n_elements()<<std::endl;


  // This is the only way we could create the SparsityPattern, through the
  // Epetramap of an existing vector.
  // vector_sparsity_pattern.reinit(helper.vector_partitioner(),
  // helper.vector_partitioner());
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


template <int dim>
void
BEMProblem<dim>::initialize_calderon_dual_space()
{
  AssertThrow(dim == 3, ExcNotImplemented());
  const unsigned int auxiliary_order =
    preconditioner_type == "CalderonDual" ? 1 : scalar_fe_order;
  calderon_fe =
    createFiniteElementPointer<dim - 1, dim, false>("FEQ",
                                                     auxiliary_order);
  calderon_dh.distribute_dofs(*calderon_fe);
  DoFRenumbering::component_wise(calderon_dh);
  DoFRenumbering::subdomain_wise(calderon_dh);
  pcout << "Initializing adjoint auxiliary space: dofs distributed"
        << std::endl;

  calderon_cpu_set.clear();
  calderon_cpu_set.set_size(calderon_dh.n_dofs());
  std::vector<types::subdomain_id> dof_subdomains(calderon_dh.n_dofs());
  DoFTools::get_subdomain_association(calderon_dh, dof_subdomains);
  for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
    if (dof_subdomains[i] == this_mpi_process)
      calderon_cpu_set.add_index(i);
  calderon_cpu_set.compress();

  calderon_constraints.clear();
  DoFTools::make_hanging_node_constraints(calderon_dh,
                                          calderon_constraints);
  if (preconditioner_type == "CalderonDual")
    AssertThrow(calderon_constraints.n_constraints() == 0,
                ExcMessage("CalderonDual currently requires a conforming "
                           "active mesh without hanging nodes."));

  std::vector<Point<dim>> support_points(calderon_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     calderon_dh,
                                                     support_points);
  pcout << "Initializing adjoint auxiliary space: support points mapped"
        << std::endl;
  calderon_dof_material_ids.assign(calderon_dh.n_dofs(), {});
  std::vector<Tensor<1, dim>> calderon_dof_normal_sums(
    calderon_dh.n_dofs());
  std::vector<unsigned int> calderon_dof_normal_counts(
    calderon_dh.n_dofs(), 0);
  std::vector<types::global_dof_index> material_cell_indices(
    calderon_fe->dofs_per_cell);
  const QMidpoint<dim - 1> auxiliary_normal_quadrature;
  FEValues<dim - 1, dim> auxiliary_normal_values(
    *mapping,
    *calderon_fe,
    auxiliary_normal_quadrature,
    update_normal_vectors);
  for (const auto &cell : calderon_dh.active_cell_iterators())
    {
      cell->get_dof_indices(material_cell_indices);
      auxiliary_normal_values.reinit(cell);
      const Tensor<1, dim> cell_normal =
        auxiliary_normal_values.normal_vector(0);
      for (const auto dof : material_cell_indices)
        {
          calderon_dof_material_ids[dof].insert(cell->material_id());
          calderon_dof_normal_sums[dof] += cell_normal;
          ++calderon_dof_normal_counts[dof];
        }
    }
  unsigned int topologically_shared_material_interface_dofs = 0;
  for (const auto &material_ids : calderon_dof_material_ids)
    if (material_ids.size() > 1)
      ++topologically_shared_material_interface_dofs;

  std::vector<types::global_dof_index> master(calderon_dh.n_dofs());
  std::iota(master.begin(), master.end(), 0);
  const auto root = [&master](types::global_dof_index index) {
    while (master[index] != index)
      index = master[index];
    return index;
  };
  const auto share_material = [this](
                                const types::global_dof_index i,
                                const types::global_dof_index j) {
    for (const auto material_id : calderon_dof_material_ids[i])
      if (calderon_dof_material_ids[j].find(material_id) !=
          calderon_dof_material_ids[j].end())
        return true;
    return false;
  };
  std::vector<Tensor<1, dim>> calderon_dof_normals(
    calderon_dh.n_dofs());
  for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
    if (calderon_dof_normal_counts[i] > 0)
      {
        const double normal_norm = calderon_dof_normal_sums[i].norm();
        AssertThrow(normal_norm > 1e-14,
                    ExcMessage("Cannot determine an auxiliary trace normal."));
        calderon_dof_normals[i] = calderon_dof_normal_sums[i] / normal_norm;
      }
  constexpr double duplicate_tolerance = 1e-10;
  constexpr double smooth_normal_cosine = 1.0 - 1e-8;
  unsigned int skipped_material_interface_pairs = 0;
  unsigned int skipped_sharp_edge_pairs = 0;
  for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
    for (types::global_dof_index j = i + 1; j < calderon_dh.n_dofs(); ++j)
      if (support_points[i].distance(support_points[j]) <
          duplicate_tolerance)
        {
          if (preconditioner_type != "CalderonDual" &&
              galerkin_adjoint_continuity == "material_wise" &&
              !share_material(i, j))
            {
              ++skipped_material_interface_pairs;
              continue;
            }
          if (preconditioner_type != "CalderonDual" &&
              galerkin_adjoint_continuity == "normal_wise" &&
              scalar_product(calderon_dof_normals[i],
                             calderon_dof_normals[j]) < smooth_normal_cosine)
            {
              ++skipped_sharp_edge_pairs;
              continue;
            }
          const types::global_dof_index root_i = root(i);
          const types::global_dof_index root_j = root(j);
          if (root_i != root_j)
            {
              const types::global_dof_index canonical =
                std::min(root_i, root_j);
              master[std::max(root_i, root_j)] = canonical;
            }
        }
  std::vector<types::global_dof_index> class_root(master.size());
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    class_root[i] = root(i);

  std::vector<types::global_dof_index> class_representative(
    master.size(), numbers::invalid_dof_index);
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    if (!calderon_constraints.is_constrained(i))
      {
        auto &representative = class_representative[class_root[i]];
        if (representative == numbers::invalid_dof_index ||
            i < representative)
          representative = i;
      }
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    {
      const auto representative = class_representative[class_root[i]];
      master[i] = representative != numbers::invalid_dof_index ?
                    representative :
                    class_root[i];
    }
  pcout << "Initializing adjoint auxiliary space: duplicate points matched"
        << std::endl;
  unsigned int merged_vertices = 0;
  for (types::global_dof_index i = 0; i < master.size(); ++i)
    if (master[i] != i && !calderon_constraints.is_constrained(i))
      {
        calderon_constraints.add_line(i);
        calderon_constraints.add_entry(i, master[i], 1.0);
        ++merged_vertices;
      }
  calderon_constraints.close();

  if (preconditioner_type != "CalderonDual")
    {
      pcout << "Initializing adjoint auxiliary space: allocating dense "
               "sparsity pattern"
            << std::endl;
      calderon_sparsity_pattern.reinit(calderon_cpu_set,
                                       mpi_communicator);
      for (const auto i : calderon_cpu_set)
        for (types::global_dof_index j = 0; j < calderon_dh.n_dofs(); ++j)
          calderon_sparsity_pattern.add(i, j);
      pcout << "Initializing adjoint auxiliary space: compressing dense "
               "sparsity pattern"
            << std::endl;
      calderon_sparsity_pattern.compress();
      pcout << "Initializing adjoint auxiliary space: allocating four "
               "operator matrices"
            << std::endl;
      calderon_hypersingular_matrix.reinit(calderon_sparsity_pattern);
      calderon_mass_matrix.reinit(calderon_sparsity_pattern);
      adjoint_double_layer_matrix.reinit(calderon_sparsity_pattern);
      adjoint_single_layer_matrix.reinit(calderon_sparsity_pattern);
      adjoint_double_layer_rhs.reinit(calderon_cpu_set,
                                      mpi_communicator);
      adjoint_first_kind_rhs.reinit(calderon_cpu_set,
                                    mpi_communicator);
      pcout << "Initializing adjoint auxiliary space: complete" << std::endl;
      pcout << "Adjoint second-kind auxiliary FEQ(" << auxiliary_order
            << ") space: " << calderon_dh.n_dofs()
            << " algebraic dofs, " << merged_vertices
            << " duplicate CAD dofs constrained, "
            << calderon_constraints.n_constraints()
            << " total constraints" << std::endl
            << "Adjoint continuity: " << galerkin_adjoint_continuity
            << "; skipped duplicate pairs across material interfaces: "
            << skipped_material_interface_pairs
            << "; skipped duplicate pairs across sharp edges: "
            << skipped_sharp_edge_pairs << std::endl
            << "Topologically shared dofs touching multiple materials: "
            << topologically_shared_material_interface_dofs << std::endl;
      return;
    }

  calderon_dual_cell_vertices.assign(dh.n_dofs(), {});
  calderon_dual_vertex_valence.assign(calderon_dh.n_dofs(), 0);
  calderon_dual_cell_areas.assign(dh.n_dofs(), 0.0);
  calderon_dual_cell_components.assign(
    dh.get_triangulation().n_active_cells(),
    numbers::invalid_unsigned_int);
  std::vector<types::global_dof_index> primary_indices(fe->dofs_per_cell);
  std::vector<types::global_dof_index> vertex_indices(
    calderon_fe->dofs_per_cell);
  const QGauss<dim - 1> area_quadrature(
    std::max(2U, mapping_degree + 1));
  FEValues<dim - 1, dim> area_values(*mapping,
                                     *calderon_fe,
                                     area_quadrature,
                                     update_JxW_values);
  auto primary_cell = dh.begin_active();
  auto vertex_cell  = calderon_dh.begin_active();
  for (; primary_cell != dh.end(); ++primary_cell, ++vertex_cell)
    {
      AssertThrow(vertex_cell != calderon_dh.end(), ExcInternalError());
      primary_cell->get_dof_indices(primary_indices);
      vertex_cell->get_dof_indices(vertex_indices);
      AssertDimension(primary_indices.size(), 1);
      const types::global_dof_index cell_dof = primary_indices[0];
      auto &cell_vertices = calderon_dual_cell_vertices[cell_dof];
      cell_vertices.reserve(vertex_indices.size());
      for (const auto vertex : vertex_indices)
        {
          const types::global_dof_index canonical_vertex = master[vertex];
          if (std::find(cell_vertices.begin(),
                        cell_vertices.end(),
                        canonical_vertex) == cell_vertices.end())
            cell_vertices.push_back(canonical_vertex);
        }
      AssertDimension(cell_vertices.size(), vertex_indices.size());
      for (const auto vertex : cell_vertices)
        ++calderon_dual_vertex_valence[vertex];

      area_values.reinit(vertex_cell);
      for (unsigned int q = 0; q < area_quadrature.size(); ++q)
        calderon_dual_cell_areas[cell_dof] += area_values.JxW(q);
      AssertThrow(calderon_dual_cell_areas[cell_dof] > 0.0,
                  ExcMessage("Non-positive CalderonDual cell area."));
    }
  AssertThrow(vertex_cell == calderon_dh.end(), ExcInternalError());

  std::vector<std::vector<types::global_dof_index>> cells_at_vertex(
    calderon_dh.n_dofs());
  for (types::global_dof_index cell_dof = 0;
       cell_dof < calderon_dual_cell_vertices.size();
       ++cell_dof)
    for (const auto vertex : calderon_dual_cell_vertices[cell_dof])
      cells_at_vertex[vertex].push_back(cell_dof);

  std::vector<unsigned int> dof_component(
    dh.n_dofs(), numbers::invalid_unsigned_int);
  std::vector<double> component_areas;
  std::vector<unsigned int> component_cell_counts;
  for (types::global_dof_index seed = 0; seed < dh.n_dofs(); ++seed)
    if (dof_component[seed] == numbers::invalid_unsigned_int)
      {
        const unsigned int component = component_areas.size();
        component_areas.push_back(0.0);
        component_cell_counts.push_back(0);
        std::vector<types::global_dof_index> pending(1, seed);
        dof_component[seed] = component;
        while (!pending.empty())
          {
            const auto cell_dof = pending.back();
            pending.pop_back();
            component_areas[component] +=
              calderon_dual_cell_areas[cell_dof];
            ++component_cell_counts[component];
            for (const auto vertex :
                 calderon_dual_cell_vertices[cell_dof])
              for (const auto neighbor : cells_at_vertex[vertex])
                if (dof_component[neighbor] ==
                    numbers::invalid_unsigned_int)
                  {
                    dof_component[neighbor] = component;
                    pending.push_back(neighbor);
                  }
          }
      }

  std::vector<unsigned int> component_order(component_areas.size());
  std::iota(component_order.begin(), component_order.end(), 0);
  std::sort(component_order.begin(),
            component_order.end(),
            [&component_areas](const unsigned int left,
                               const unsigned int right) {
              return component_areas[left] < component_areas[right];
            });
  std::vector<unsigned int> remapped_component(component_areas.size());
  calderon_dual_component_areas.resize(component_areas.size());
  for (unsigned int component = 0; component < component_order.size();
       ++component)
    {
      remapped_component[component_order[component]] = component;
      calderon_dual_component_areas[component] =
        component_areas[component_order[component]];
    }

  for (const auto &cell : dh.active_cell_iterators())
    {
      cell->get_dof_indices(primary_indices);
      AssertDimension(primary_indices.size(), 1);
      calderon_dual_cell_components[cell->active_cell_index()] =
        remapped_component[dof_component[primary_indices[0]]];
    }

  pcout << "CalderonDual connected boundary components: "
        << component_order.size() << std::endl;
  for (unsigned int component = 0; component < component_order.size();
       ++component)
    pcout << "  component " << component << ": "
          << component_cell_counts[component_order[component]]
          << " cells, area " << calderon_dual_component_areas[component]
          << (component_order.size() == 2 ?
                (component == 0 ? " (sphere)" : " (cube)") :
                "")
          << std::endl;

  calderon_sparsity_pattern.reinit(calderon_cpu_set, mpi_communicator);
  for (const auto i : calderon_cpu_set)
    for (types::global_dof_index j = 0; j < calderon_dh.n_dofs(); ++j)
      calderon_sparsity_pattern.add(i, j);
  calderon_sparsity_pattern.compress();
  calderon_hypersingular_matrix.reinit(calderon_sparsity_pattern);
  calderon_mass_matrix.reinit(calderon_sparsity_pattern);

  pcout << "CalderonDual auxiliary FEQ(1) space: "
        << calderon_dh.n_dofs() << " algebraic vertices, "
        << merged_vertices << " duplicate CAD vertices constrained"
        << std::endl;
}


template <>
const Quadrature<2> &
BEMProblem<3>::get_singular_quadrature(const unsigned int index) const
{
  Assert(index < fe->dofs_per_cell, ExcIndexRange(0, fe->dofs_per_cell, index));



  static std::vector<Quadrature<2>> quadratures;
  {
    if (quadratures.size() == 0)
      for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
        {
          quadratures.push_back(QSplit<2>(QDuffy(singular_quadrature_order, 1.),
                                          fe->get_unit_support_points()[i]));
        }
  }

  return quadratures[index];
}

template <>
const Quadrature<1> &
BEMProblem<2>::get_singular_quadrature(const unsigned int index) const
{
  Assert(index < fe->dofs_per_cell, ExcIndexRange(0, fe->dofs_per_cell, index));

  static std::vector<Quadrature<1>> quadratures;
  if (quadratures.size() == 0)
    for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
      {
        quadratures.push_back(QTelles<1>(singular_quadrature_order,
                                         fe->get_unit_support_points()[i]));
      }
  return quadratures[index];
}


template <>
const Quadrature<2>
BEMProblem<3>::get_quasi_singular_quadrature(const typename DoFHandler<2,3>::active_cell_iterator &cell,
                                              const Mapping<2,3> &mapping,
                                              const Point<3> &singularity,
                                              const Point<2> &ref_projection,
                                              const unsigned int exponent,
                                              const unsigned int quadrature_order_override,
                                              const double normalized_distance) const
{
  Assert(exponent == 1 || exponent == 2, ExcNotImplemented());
  const unsigned int quadrature_order = quadrature_order_override > 0 ?
    quadrature_order_override :
    (exponent == 2 ? quasi_singular_neumann_quadrature_order :
                     quasi_singular_dirichlet_quadrature_order);
  if (near_singular_subdivision &&
      normalized_distance < near_singular_subdivision_ratio)
    {
      const double safe_distance = std::max(normalized_distance, 1e-12);
      const unsigned int levels = std::min(
        near_singular_subdivision_max_levels,
        std::max(1u,
                 static_cast<unsigned int>(std::ceil(
                   std::log2(0.5 / safe_distance)))));
      return nested_quasi_singular_quadrature(ref_projection,
                                              quadrature_order,
                                              levels);
    }
  return telles_quadrature(
        cell,
        mapping,
        singularity,
        ref_projection,
        quadrature_order,
        exponent);
}

template <>
const Quadrature<1>
BEMProblem<2>::get_quasi_singular_quadrature(const typename DoFHandler<1,2>::active_cell_iterator &cell,
                                              const Mapping<1,2> &mapping,
                                              const Point<2> &singularity,
                                              const Point<1> &ref_projection,
                                              const unsigned int exponent,
                                              const unsigned int quadrature_order_override,
                                              const double /*normalized_distance*/) const
{
  Assert(exponent == 1 || exponent == 2, ExcNotImplemented());
  const unsigned int quadrature_order = quadrature_order_override > 0 ?
    quadrature_order_override :
    (exponent == 2 ? quasi_singular_neumann_quadrature_order :
                     quasi_singular_dirichlet_quadrature_order);
  return telles_quadrature(
        cell,
        mapping,
        singularity,
        ref_projection,
        quadrature_order,
        exponent);
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
                      "None|ILU|AMG|Jacobi|BlockJacobi|Calderon|"
                      "CalderonDual"));

  prm.declare_entry("Calderon stabilization",
                    "1.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Calderon dual bubble scaling",
                    "1.0",
                    Patterns::Double(0.0));
  prm.declare_entry("Galerkin component RHS diagnostics",
                    "false",
                    Patterns::Bool());

  // The energy norm costs a full pass over the panel pairs, comparable to one
  // matrix assembly, so it is off unless a convergence study asks for it.
  prm.declare_entry("Compute energy norm error", "false", Patterns::Bool());

  // The per-component, per-material and radial error breakdowns. They were
  // written to find where the error concentrated on this geometry; once that
  // is settled they are about thirty lines of output per run, so they are off
  // unless asked for.
  prm.declare_entry("Detailed error diagnostics", "false", Patterns::Bool());

  // How the surface gradient is recovered once the trace and the flux are
  // known. "projection" takes the L2 projection of the gradient reconstructed
  // from the discrete fields; "hypersingular" evaluates the hypersingular
  // boundary integral equation at the support points, which for discontinuous
  // elements are interior to their cell and so avoid the corner ambiguity of
  // the free term that a continuous space would face at vertices.
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
  prm.declare_entry("Galerkin fused right-hand side",
                    "true",
                    Patterns::Bool());

  // Evaluate the near-field quadrature rules from the cell geometry,
  // reconstructed exactly from the mapping at its Gauss-Lobatto nodes, rather
  // than building FEValues objects for every leaf of the subdivision. The
  // reconstruction is checked against FEValues on every cell.
  prm.declare_entry("Galerkin light geometry",
                    "true",
                    Patterns::Bool());

  prm.declare_entry("Solution method",
                    "Direct",
                    Patterns::Selection("Direct|FMA"));

  prm.declare_entry("Linear solver",
                    "GMRES",
                    Patterns::Selection("GMRES|MUMPS|CG"));

  prm.declare_entry(
    "MUMPS ordering",
    "QAMD",
    Patterns::Selection("automatic|AMD|AMF|SCOTCH|PORD|METIS|QAMD"));

  prm.declare_entry("Boundary integral formulation",
                    "collocation",
                    Patterns::Selection("collocation|galerkin"));

  prm.declare_entry(
    "Galerkin Neumann formulation",
    "single_layer",
    Patterns::Selection("single_layer|adjoint_double_layer"));
  prm.declare_entry(
    "Galerkin adjoint continuity",
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
                    "false", Patterns::Bool());
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

  prm.declare_entry("Kernel type", "laplace", Patterns::Selection("laplace|screened"));

  // manufactured_flux builds the Galerkin right-hand side as <V lambda, phi_i>
  // from the exact flux, through the single-layer kernel. It applies only
  // when the exact solution is known, so it is a convergence-study tool, but
  // it is the right-hand side the Galerkin equations actually require, and it
  // avoids the double layer's 1/r^2 kernel entirely.
  prm.declare_entry("Dirichlet RHS treatment",
                    "nodal",
                    Patterns::Selection(
                      "nodal|exact_regularized|manufactured_flux"));

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
    prm.declare_entry("Galerkin quadrature order",
                      "6",
                      Patterns::Integer(2));
    prm.declare_entry(
      "Galerkin singular quadrature",
      "sauter_schwab",
      Patterns::Selection("sauter_schwab|iterated_duffy"));
    prm.declare_entry("Galerkin singular quadrature order",
                      "6",
                      Patterns::Integer(2));
    prm.declare_entry("Galerkin Duffy outer subdivisions",
                      "4",
                      Patterns::Integer(1));
    prm.declare_entry("Galerkin near singular distance ratio",
                      "0.0",
                      Patterns::Double(0.0));
    prm.declare_entry("Galerkin near singular maximum depth",
                      "0",
                      Patterns::Integer(0));
    prm.declare_entry("Galerkin near singular quadrature order",
                      "4",
                      Patterns::Integer(2));
    prm.declare_entry("Quasi singular quadrature order", "5", Patterns::Integer());
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
  prm.enter_subsection("Solver");
  solver_control.parse_parameters(prm);
  prm.leave_subsection();

  preconditioner_type = prm.get("Preconditioner");
  calderon_stabilization = prm.get_double("Calderon stabilization");
  calderon_dual_bubble_scaling =
    prm.get_double("Calderon dual bubble scaling");
  galerkin_component_rhs_diagnostics =
    prm.get_bool("Galerkin component RHS diagnostics");
  compute_energy_norm = prm.get_bool("Compute energy norm error");
  detailed_error_diagnostics = prm.get_bool("Detailed error diagnostics");
  gradient_recovery          = prm.get("Gradient recovery");
  galerkin_far_field_tolerance =
    prm.get_double("Galerkin far field tolerance");
  galerkin_fused_rhs      = prm.get_bool("Galerkin fused right-hand side");
  galerkin_light_geometry = prm.get_bool("Galerkin light geometry");

  solution_method = prm.get("Solution method");

  linear_solver = prm.get("Linear solver");

  mumps_ordering = prm.get("MUMPS ordering");

  boundary_integral_formulation =
    prm.get("Boundary integral formulation");
  galerkin_neumann_formulation =
    prm.get("Galerkin Neumann formulation");
  galerkin_adjoint_continuity =
    prm.get("Galerkin adjoint continuity");
  galerkin_adjoint_component_weighting =
    prm.get_bool("Galerkin adjoint component weighting");
  galerkin_adjoint_primary_material_id =
    prm.get_integer("Galerkin adjoint primary material id");
  galerkin_adjoint_primary_material_ids.clear();
  const std::string primary_material_ids_entry =
    prm.get("Galerkin adjoint primary material ids");
  if (Utilities::trim(primary_material_ids_entry).empty())
    galerkin_adjoint_primary_material_ids.insert(
      static_cast<types::material_id>(
        galerkin_adjoint_primary_material_id));
  else
    for (const auto &material_id :
         Utilities::split_string_list(primary_material_ids_entry))
      galerkin_adjoint_primary_material_ids.insert(
        static_cast<types::material_id>(
          Utilities::string_to_int(material_id)));
  galerkin_adjoint_primary_eta_scale =
    prm.get_double("Galerkin adjoint primary eta scale");
  galerkin_adjoint_secondary_eta_scale =
    prm.get_double("Galerkin adjoint secondary eta scale");
  galerkin_adjoint_local_eta_scaling =
    prm.get_bool("Galerkin adjoint local eta scaling");
  galerkin_adjoint_dual_residual_stabilization =
    prm.get_bool("Galerkin adjoint dual residual stabilization");
  galerkin_adjoint_dual_residual_lift =
    prm.get("Galerkin adjoint dual residual lift");
  galerkin_adjoint_block_completion =
    prm.get_bool("Galerkin adjoint block completion");
  galerkin_adjoint_hybrid_system =
    prm.get_bool("Galerkin adjoint hybrid system");
  galerkin_hybrid_exact_trace_diagnostics =
    prm.get_bool("Galerkin hybrid exact trace diagnostics");
  galerkin_analytical_hypersingular_rhs =
    prm.get_bool("Galerkin analytical hypersingular RHS");
  galerkin_hybrid_conforming_complement =
    prm.get_bool("Galerkin hybrid conforming complement");
  galerkin_hybrid_complement_mass_regularization =
    prm.get_double("Galerkin hybrid complement mass regularization");
  galerkin_hybrid_local_mass_regularization =
    prm.get_double("Galerkin hybrid local mass regularization");
  galerkin_hybrid_local_fluctuation_regularization =
    prm.get_double("Galerkin hybrid local fluctuation regularization");
  galerkin_hybrid_local_jump_regularization =
    prm.get_double("Galerkin hybrid local jump regularization");
  galerkin_hybrid_local_jump_h_scaling =
    prm.get_bool("Galerkin hybrid local jump h scaling");
  galerkin_hybrid_local_mass_material_id =
    prm.get_integer("Galerkin hybrid local mass material id");
  galerkin_hybrid_local_mass_radius_min =
    prm.get_double("Galerkin hybrid local mass radius min");
  galerkin_hybrid_local_mass_radius_max =
    prm.get_double("Galerkin hybrid local mass radius max");
  galerkin_hybrid_first_kind_polishing_steps =
    prm.get_integer("Galerkin hybrid first-kind polishing steps");
  galerkin_hybrid_first_kind_polishing_relaxation =
    prm.get_double("Galerkin hybrid first-kind polishing relaxation");

  kernel_type = prm.get("Kernel type");

  dirichlet_rhs_treatment = prm.get("Dirichlet RHS treatment");

  hypersingular_correction_steps =
    prm.get_integer("Hypersingular correction steps");
  hypersingular_correction_relaxation =
    prm.get_double("Hypersingular correction relaxation");
  hypersingular_correction_max_relative_update =
    prm.get_double("Hypersingular correction maximum relative update");
  enforce_laplace_linear_patch = prm.get_bool("Enforce Laplace linear patch");

  screened_kappa = prm.get_double("Screened Kappa");

  prm.enter_subsection("Quadrature rules");
  {
    quadrature = std::shared_ptr<Quadrature<dim - 1>>(
      new QuadratureSelector<dim - 1>(prm.get("Quadrature type"),
                                      prm.get_integer("Quadrature order")));
    quadrature_order          = prm.get_integer("Quadrature order");
    singular_quadrature_order = prm.get_integer("Singular quadrature order");
    galerkin_quadrature_order =
      prm.get_integer("Galerkin quadrature order");
    galerkin_singular_quadrature =
      prm.get("Galerkin singular quadrature");
    galerkin_singular_quadrature_order =
      prm.get_integer("Galerkin singular quadrature order");
    galerkin_duffy_outer_subdivisions =
      prm.get_integer("Galerkin Duffy outer subdivisions");
    galerkin_near_singular_distance_ratio =
      prm.get_double("Galerkin near singular distance ratio");
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
    near_singular_subdivision =
      prm.get_bool("Near singular subdivision");
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
      AssertThrow(solution_method == "Direct",
                  ExcMessage("Galerkin assembly requires Solution method = "
                             "Direct."));
      // The screened kernel exp(-kappa r)/(4 pi r) enters the single-layer
      // formulation, its right-hand sides, the energy functional and the
      // gradient recovery. The second-kind and Calderon paths use the weak
      // hypersingular operator, whose screened form is not implemented.
      AssertThrow(kernel_type == "laplace" ||
                    (galerkin_neumann_formulation == "single_layer" &&
                     preconditioner_type != "Calderon" &&
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
      if (galerkin_neumann_formulation == "adjoint_double_layer")
        {
          AssertThrow(linear_solver == "MUMPS",
                      ExcMessage("The experimental adjoint-double-layer "
                                 "formulation currently requires Linear "
                                 "solver = MUMPS."));
          AssertThrow(preconditioner_type == "None",
                      ExcMessage("The adjoint-double-layer formulation is "
                                 "a second-kind direct solve and requires "
                                 "Preconditioner = None."));
        }
      if (preconditioner_type == "Calderon" ||
          preconditioner_type == "CalderonDual")
        {
          AssertThrow(linear_solver == "GMRES",
                      ExcMessage("The Calderon operator preconditioner "
                                 "requires Linear solver = GMRES."));
        }
      if (linear_solver == "CG")
        AssertThrow(galerkin_neumann_formulation == "single_layer",
                    ExcMessage("The conjugate gradient path solves the "
                               "symmetric single-layer system and requires "
                               "Galerkin Neumann formulation = single_layer."));
    }
  else
    {
      AssertThrow(galerkin_neumann_formulation == "single_layer",
                  ExcMessage("Galerkin Neumann formulation applies only when "
                             "Boundary integral formulation = galerkin."));
      AssertThrow(linear_solver != "CG",
                  ExcMessage("The collocation operator is not symmetric; the "
                             "conjugate gradient path requires Boundary "
                             "integral formulation = galerkin."));
    }

  mapping_type       = prm.get("Mapping Type");
  mapping_degree     = prm.get_integer("Mapping Q Degree");
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
      if (galerkin_adjoint_block_completion)
        AssertThrow(galerkin_neumann_formulation ==
                        "adjoint_double_layer" &&
                      scalar_fe_type == "FEDGQ",
                    ExcMessage("Galerkin adjoint block completion requires "
                               "the adjoint_double_layer formulation and "
                               "Scalar FE = FEDGQ."));
      if (galerkin_adjoint_hybrid_system)
        AssertThrow(galerkin_neumann_formulation ==
                        "adjoint_double_layer" &&
                      scalar_fe_type == "FEDGQ" &&
                      scalar_fe_order > 0,
                    ExcMessage("Galerkin adjoint hybrid system requires "
                               "the adjoint_double_layer formulation and "
                               "Scalar FE = FEDGQ of positive order."));
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
  have_dirichlet_bc = false;


  Vector<double> non_partitioned_dirichlet_nodes(dh.n_dofs());
  Vector<double> non_partitioned_neumann_nodes(dh.n_dofs());



  cell_it cell = dh.begin_active(), endc = dh.end();


  vector_shift(non_partitioned_neumann_nodes, 1.);
  std::vector<types::global_dof_index> dofs(fe->dofs_per_cell);
  std::vector<types::global_dof_index> gradient_dofs(
    gradient_fe->dofs_per_cell);
  unsigned int helper_dirichlet = 0;
  for (; cell != endc; ++cell)
    {
      if (cell->subdomain_id() == this_mpi_process)
        {
          bool dirichlet = false;
          for (auto dummy : comp_dom.dirichlet_boundary_ids)
            {
              if (dummy == cell->material_id())
                {
                  cell->get_dof_indices(dofs);
                  for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                    {
                      non_partitioned_dirichlet_nodes(dofs[i]) = 1;
                      non_partitioned_neumann_nodes(dofs[i])   = 0;
                      // pcout<<dofs[i]<<"  cellMatId "<<cell->material_id()<<"
                      // surfNodes: "<<dirichlet_nodes(dofs[i])<<"  otherNodes:
                      // "<<neumann_nodes(dofs[i])<<std::endl;
                    }
                  dirichlet        = true;
                  helper_dirichlet = 1.;
                  break;
                }
            }
          if (!dirichlet)
            {
              cell->get_dof_indices(dofs);
              // for(unsigned int i=0; i<fe->dofs_per_cell; ++i)
              // {
              //   non_partitioned_neumann_nodes(dofs[i]) = 1;
              //   non_partitioned_dirichlet_nodes(dofs[i]) = 0;
              // }
            }

          // if (cell->material_id() == comp_dom.dirichlet_sur_ID1 ||
          //     cell->material_id() == comp_dom.dirichlet_sur_ID2 ||
          //     cell->material_id() == comp_dom.dirichlet_sur_ID3)
          //   {
          //     // This is a free surface node.
          //     cell->get_dof_indices(dofs);
          //     for (unsigned int i=0; i<fe->dofs_per_cell; ++i)
          //       {
          //         non_partitioned_dirichlet_nodes(dofs[i]) = 1;
          //         non_partitioned_neumann_nodes(dofs[i]) = 0;
          //         //pcout<<dofs[i]<<"  cellMatId "<<cell->material_id()<<"
          //         surfNodes: "<<dirichlet_nodes(dofs[i])<<"  otherNodes:
          //         "<<neumann_nodes(dofs[i])<<std::endl;
          //       }
          //   }
          // else
          //   {
          //     for (unsigned int i=0; i<fe->dofs_per_cell; ++i)
          //       {
          //         cell->get_dof_indices(dofs);
          //         //pcout<<dofs[i]<<"  cellMatId "<<cell->material_id()<<"
          //         surfNodes: "<<dirichlet_nodes(dofs[i])<<"  otherNodes:
          //         "<<neumann_nodes(dofs[i])<<std::endl;
          //       }
          //
          //   }
        }
    }

  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    if (this_cpu_set.is_element(i))
      {
        dirichlet_nodes(i) = non_partitioned_dirichlet_nodes(i);
        neumann_nodes(i)   = non_partitioned_neumann_nodes(i);
      }
  // dirichlet_nodes.add(non_partitioned_dirichlet_nodes, true);// =
  // non_partitioned_dirichlet_nodes;
  // neumann_nodes.add(non_partitioned_neumann_nodes, true);// =
  // non_partitioned_neumann_nodes;
  unsigned int helper_dirichlet_2;
  // std::cout<<this_mpi_process<<" , "<<helper_dirichlet<<std::endl;
  MPI_Allreduce(&helper_dirichlet,
                &helper_dirichlet_2,
                1,
                MPI_UNSIGNED,
                MPI_MAX,
                mpi_communicator);
  // std::cout<<this_mpi_process<<" , "<<helper_dirichlet<<" ,
  // "<<helper_dirichlet_2<<std::endl;
  if (helper_dirichlet_2 > 0)
    have_dirichlet_bc = true;
  // std::cout<<this_mpi_process<<" , "<<have_dirichlet_bc<<std::endl;
  // for (unsigned int i=0; i<dh.n_dofs(); ++i)
  //    if (this_mpi_process == 1)
  //       pcout<<i<<" "<<dirichlet_nodes(i)<<" "<<neumann_nodes(i)<<std::endl;
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

  typename DoFHandler<dim - 1, dim>::active_cell_iterator cell =
                                                            dh.begin_active(),
                                                          endc = dh.end();
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  edge_set.clear();
  edge_set.set_size(dh.n_dofs());

  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
          {
            cell->face(f)->get_dof_indices(face_dofs);
            for (unsigned int k = 0; k < face_dofs.size(); ++k)
              edge_set.add_index(face_dofs[k]);
          }
    }
  edge_set.compress();

  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    double_nodes_set[i].insert(i);
  for (auto i : edge_set) //(types::global_dof_index i=0; i<dh.n_dofs(); ++i)
    {
      for (auto j : edge_set)
        {
          if (support_points[i].distance(support_points[j]) < tol)
            {
              double_nodes_set[i].insert(j);
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
BEMProblem<dim>::assemble_galerkin_system()
{
  AssertThrow(boundary_integral_formulation == "galerkin",
              ExcInternalError());
  if constexpr (dim == 3)
    {
      pcout << "Assembling Galerkin boundary operators with order "
            << galerkin_quadrature_order << ", singular order "
            << galerkin_singular_quadrature_order << " using "
            << galerkin_singular_quadrature;
      if (galerkin_singular_quadrature == "iterated_duffy")
        pcout << " with " << galerkin_duffy_outer_subdivisions
              << " outer subdivisions";
      pcout << std::endl;
      dirichlet_matrix = 0.0;

      AffineConstraints<double> hanging_node_constraints;
      DoFTools::make_hanging_node_constraints(dh,
                                              hanging_node_constraints);
      hanging_node_constraints.close();

      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_far_field_tolerance(galerkin_far_field_tolerance);
      const bool light = integrator.set_light_geometry(galerkin_light_geometry);
      galerkin_rhs_from_fused_assembly =
        galerkin_fused_rhs && galerkin_fused_dirichlet_function != nullptr;
      if (galerkin_rhs_from_fused_assembly)
        {
          exact_regularized_dirichlet_rhs = 0.0;
          integrator.assemble_single_layer_and_double_layer_rhs(
            dh,
            hanging_node_constraints,
            dirichlet_matrix,
            exact_regularized_dirichlet_rhs,
            galerkin_quadrature_order,
            *galerkin_fused_dirichlet_function,
            this_mpi_process,
            dirichlet_rhs_treatment != "nodal");
        }
      else
        integrator.assemble_single_layer(dh,
                                         hanging_node_constraints,
                                         dirichlet_matrix,
                                         galerkin_quadrature_order,
                                         this_mpi_process);
      dirichlet_matrix.compress(VectorOperation::add);
      if (light)
        pcout << "Near-field rules evaluated from the reconstructed cell "
                 "geometry; largest deviation from FEValues "
              << Utilities::MPI::max(integrator.light_geometry_deviation(),
                                     mpi_communicator)
              << "; element evaluated as "
              << (integrator.tensor_shape_functions() ?
                    "a tensor Lagrange basis" :
                    "a general finite element")
              << std::endl;
      if (galerkin_far_field_tolerance > 0.0)
        {
          // Every rank takes part in every sum, whether or not it used the
          // order: a sum called on some ranks only desynchronises the
          // collective calls that follow.
          pcout << "Far-field quadrature orders used:";
          const auto &counts = integrator.far_field_order_counts();
          for (unsigned int o = 0; o < counts.size(); ++o)
            {
              const unsigned long long total =
                Utilities::MPI::sum(counts[o], mpi_communicator);
              if (total > 0)
                pcout << ' ' << o << '=' << total;
            }
          pcout << std::endl;
        }
      {
        const auto &profile = integrator.rhs_profile();
        pcout << (galerkin_rhs_from_fused_assembly ?
                    "Galerkin V and RHS" :
                    "Galerkin V")
              << " time on the slowest process: far pairs "
              << Utilities::MPI::max(profile.far_seconds, mpi_communicator)
              << " s, near and singular pairs "
              << Utilities::MPI::max(profile.near_seconds, mpi_communicator)
              << " s, of which evaluating the Dirichlet datum "
              << Utilities::MPI::max(profile.function_seconds,
                                     mpi_communicator)
              << " s" << std::endl
              << "  rule construction for uncached pairs "
              << Utilities::MPI::max(profile.setup_seconds, mpi_communicator)
              << " s (" << Utilities::MPI::sum(double(profile.uncached_pairs),
                                               mpi_communicator)
              << " pairs), kernel loops "
              << Utilities::MPI::max(profile.kernel_seconds, mpi_communicator)
              << " s" << std::endl;
        if (light)
          pcout << "  direct path: "
                << Utilities::MPI::sum(double(profile.singular_leaves),
                                       mpi_communicator)
                << " singular leaves ("
                << Utilities::MPI::sum(double(profile.singular_points),
                                       mpi_communicator)
                << " points) "
                << Utilities::MPI::max(profile.singular_seconds,
                                       mpi_communicator)
                << " s, "
                << Utilities::MPI::sum(double(profile.regular_leaves),
                                       mpi_communicator)
                << " regular leaves "
                << Utilities::MPI::max(profile.regular_seconds,
                                       mpi_communicator)
                << " s" << std::endl;
      }
      pcout << "Galerkin V near-singular adaptive pairs/leaves: "
            << Utilities::MPI::sum(integrator.n_near_singular_pairs(),
                                   mpi_communicator)
            << '/'
            << Utilities::MPI::sum(integrator.n_near_singular_leaves(),
                                   mpi_communicator)
            << std::endl;
      if (galerkin_neumann_formulation == "adjoint_double_layer")
        {
          calderon_hypersingular_matrix = 0.0;
          calderon_mass_matrix          = 0.0;
          adjoint_double_layer_matrix   = 0.0;
          adjoint_single_layer_matrix   = 0.0;
          GalerkinPanelPairIntegrator<3> auxiliary_integrator(
            *mapping,
            *calderon_fe,
            singular_strategy,
            galerkin_singular_quadrature_order,
            galerkin_duffy_outer_subdivisions,
            galerkin_near_singular_distance_ratio,
            galerkin_near_singular_maximum_depth,
            galerkin_near_singular_quadrature_order);
          auxiliary_integrator.set_screening(galerkin_screening());
	          const types::material_id hybrid_primary_material =
	            (galerkin_adjoint_hybrid_system &&
	             !galerkin_hybrid_conforming_complement &&
	             galerkin_adjoint_primary_material_ids.size() == 1) ?
	              *galerkin_adjoint_primary_material_ids.begin() :
	              numbers::invalid_material_id;
          pcout << "Assembling auxiliary weak hypersingular operator W"
                << std::endl;
          auxiliary_integrator.assemble_weak_hypersingular(
            calderon_dh,
            calderon_constraints,
            calderon_hypersingular_matrix,
            galerkin_quadrature_order,
            this_mpi_process,
            hybrid_primary_material,
            numbers::invalid_material_id);
          pcout << "Done assembling auxiliary W; assembling mass matrix M"
                << std::endl;
          auxiliary_integrator.assemble_mass_matrix(
            calderon_dh,
            calderon_constraints,
            calderon_mass_matrix,
            galerkin_quadrature_order,
            this_mpi_process,
            hybrid_primary_material);
          pcout << "Done assembling auxiliary M; assembling adjoint "
                   "double-layer operator K'"
                << std::endl;
          auxiliary_integrator.assemble_adjoint_double_layer(
            calderon_dh,
            calderon_constraints,
            adjoint_double_layer_matrix,
            galerkin_quadrature_order,
            this_mpi_process,
            hybrid_primary_material,
            hybrid_primary_material);
          pcout << "Done assembling auxiliary K'; assembling single-layer "
                   "operator V"
                << std::endl;
          auxiliary_integrator.assemble_single_layer(
            calderon_dh,
            calderon_constraints,
            adjoint_single_layer_matrix,
            galerkin_quadrature_order,
            this_mpi_process,
            hybrid_primary_material,
            hybrid_primary_material);
          pcout << "Done assembling auxiliary V" << std::endl;
          calderon_hypersingular_matrix.compress(VectorOperation::add);
          calderon_mass_matrix.compress(VectorOperation::add);
          adjoint_double_layer_matrix.compress(VectorOperation::add);
          adjoint_single_layer_matrix.compress(VectorOperation::add);

          TrilinosWrappers::MPI::Vector constant_trace(calderon_cpu_set,
                                                        mpi_communicator);
          TrilinosWrappers::MPI::Vector constant_image(calderon_cpu_set,
                                                        mpi_communicator);
          for (const auto i : calderon_cpu_set)
            if (!calderon_constraints.is_constrained(i))
              constant_trace[i] = 1.0;
          constant_trace.compress(VectorOperation::insert);
          calderon_hypersingular_matrix.vmult(constant_image,
                                              constant_trace);
          pcout << "Assembled auxiliary W, M, K', and V for the adjoint "
                   "second-kind formulation"
                << std::endl
                << "Auxiliary W constant-patch residual: "
                << constant_image.l2_norm() << std::endl
                << "Auxiliary near-singular adaptive pairs/leaves: "
                << Utilities::MPI::sum(
                     auxiliary_integrator.n_near_singular_pairs(),
                     mpi_communicator)
                << '/'
                << Utilities::MPI::sum(
                     auxiliary_integrator.n_near_singular_leaves(),
                     mpi_communicator)
                << std::endl;
        }
      if (preconditioner_type == "Calderon")
        {
          calderon_hypersingular_matrix = 0.0;
          calderon_mass_matrix          = 0.0;
          integrator.assemble_weak_hypersingular(
            dh,
            hanging_node_constraints,
            calderon_hypersingular_matrix,
            galerkin_quadrature_order,
            this_mpi_process);
          integrator.assemble_mass_matrix(dh,
                                          hanging_node_constraints,
                                          calderon_mass_matrix,
                                          galerkin_quadrature_order,
                                          this_mpi_process);
          calderon_hypersingular_matrix.compress(VectorOperation::add);
          calderon_mass_matrix.compress(VectorOperation::add);
          TrilinosWrappers::MPI::Vector constant_trace(this_cpu_set,
                                                        mpi_communicator);
          TrilinosWrappers::MPI::Vector constant_image(this_cpu_set,
                                                        mpi_communicator);
          for (const auto i : this_cpu_set)
            constant_trace[i] = 1.0;
          constant_trace.compress(VectorOperation::insert);
          calderon_hypersingular_matrix.vmult(constant_image,
                                              constant_trace);
          pcout << "Assembled weak hypersingular W and coupling mass M for "
                   "Calderon operator preconditioning"
                << std::endl;
          pcout << "Weak hypersingular constant-patch residual: "
                << constant_image.l2_norm() << std::endl;
        }
      else if (preconditioner_type == "CalderonDual")
        {
          calderon_hypersingular_matrix = 0.0;
          calderon_mass_matrix          = 0.0;
          GalerkinPanelPairIntegrator<3> auxiliary_integrator(
            *mapping,
            *calderon_fe,
            singular_strategy,
            galerkin_singular_quadrature_order,
            galerkin_duffy_outer_subdivisions);
          auxiliary_integrator.set_screening(galerkin_screening());
          auxiliary_integrator.assemble_weak_hypersingular(
            calderon_dh,
            calderon_constraints,
            calderon_hypersingular_matrix,
            galerkin_quadrature_order,
            this_mpi_process);
          auxiliary_integrator.assemble_mass_matrix(
            calderon_dh,
            calderon_constraints,
            calderon_mass_matrix,
            galerkin_quadrature_order,
            this_mpi_process);
          calderon_hypersingular_matrix.compress(VectorOperation::add);
          calderon_mass_matrix.compress(VectorOperation::add);

          TrilinosWrappers::MPI::Vector constant_trace(calderon_cpu_set,
                                                        mpi_communicator);
          TrilinosWrappers::MPI::Vector constant_image(calderon_cpu_set,
                                                        mpi_communicator);
          for (const auto i : calderon_cpu_set)
            if (!calderon_constraints.is_constrained(i))
              constant_trace[i] = 1.0;
          constant_trace.compress(VectorOperation::insert);
          calderon_hypersingular_matrix.vmult(constant_image,
                                              constant_trace);
          pcout << "Assembled auxiliary weak hypersingular W and mass M for "
                   "P0/P1 CalderonDual"
                << std::endl;
          pcout << "Auxiliary W constant-patch residual: "
                << constant_image.l2_norm() << std::endl;
        }
      pcout << (galerkin_rhs_from_fused_assembly ?
                  "Done assembling Galerkin V together with the regularised "
                  "Dirichlet RHS" :
                  "Done assembling Galerkin V; the analytical Dirichlet RHS "
                  "is assembled separately")
            << std::endl;
    }
  else
    AssertThrow(false,
                ExcMessage("Galerkin assembly currently supports 3D only."));
}


template <int dim>
void
BEMProblem<dim>::assemble_exact_galerkin_dirichlet_rhs(
  const Function<dim> &dirichlet_function,
  const Function<dim> *exact_gradient)
{
  AssertThrow(boundary_integral_formulation == "galerkin",
              ExcInternalError());
  if constexpr (dim == 3)
    {
      // The fused assembly has already added the double-layer part.
      const bool from_fused = galerkin_rhs_from_fused_assembly;
      galerkin_rhs_from_fused_assembly = false;
      if (!from_fused)
        exact_regularized_dirichlet_rhs = 0.0;
      AffineConstraints<double> hanging_node_constraints;
      DoFTools::make_hanging_node_constraints(dh,
                                              hanging_node_constraints);
      hanging_node_constraints.close();

      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_light_geometry(galerkin_light_geometry);
      const QGauss<2> mass_quadrature(galerkin_quadrature_order);
      FEValues<2, 3> mass_values(*mapping,
                                 *fe,
                                 mass_quadrature,
                                 update_values | update_quadrature_points |
                                   update_JxW_values);
      if (!from_fused)
      {
      integrator.assemble_double_layer_rhs(dh,
                                           hanging_node_constraints,
                                           exact_regularized_dirichlet_rhs,
                                           galerkin_quadrature_order,
                                           dirichlet_function,
                                           this_mpi_process,
                                           dirichlet_rhs_treatment !=
                                             "nodal");
      {
        const auto &profile = integrator.rhs_profile();
        pcout << "Galerkin RHS time on the slowest process: far pairs "
              << Utilities::MPI::max(profile.far_seconds, mpi_communicator)
              << " s, near and singular pairs "
              << Utilities::MPI::max(profile.near_seconds, mpi_communicator)
              << " s, of which evaluating the Dirichlet datum "
              << Utilities::MPI::max(profile.function_seconds,
                                     mpi_communicator)
              << " s (" << Utilities::MPI::sum(double(profile.function_calls),
                                               mpi_communicator)
              << " evaluations)" << std::endl
              << "  rule construction for uncached pairs "
              << Utilities::MPI::max(profile.setup_seconds, mpi_communicator)
              << " s (" << Utilities::MPI::sum(double(profile.uncached_pairs),
                                               mpi_communicator)
              << " pairs), kernel loops "
              << Utilities::MPI::max(profile.kernel_seconds, mpi_communicator)
              << " s" << std::endl;
      }
      pcout << "Galerkin RHS near-singular adaptive pairs/leaves: "
            << Utilities::MPI::sum(integrator.n_near_singular_pairs(),
                                   mpi_communicator)
            << '/'
            << Utilities::MPI::sum(integrator.n_near_singular_leaves(),
                                   mpi_communicator)
            << std::endl;
      }

      if (dirichlet_rhs_treatment == "nodal")
        {
          std::vector<types::global_dof_index> test_indices(
            fe->dofs_per_cell);
          for (const auto &test_cell : dh.active_cell_iterators())
            {
              if (test_cell->subdomain_id() != this_mpi_process)
                continue;

              Vector<double> local_rhs(fe->dofs_per_cell);
              mass_values.reinit(test_cell);
              for (unsigned int q = 0; q < mass_quadrature.size(); ++q)
                for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                  local_rhs[i] +=
                    0.5 * mass_values.shape_value(i, q) *
                    dirichlet_function.value(
                      mass_values.quadrature_point(q)) *
                    mass_values.JxW(q);

              local_rhs *= -1.0;
              test_cell->get_dof_indices(test_indices);
              hanging_node_constraints.distribute_local_to_global(
                local_rhs, test_indices, exact_regularized_dirichlet_rhs);
            }
        }

      exact_regularized_dirichlet_rhs.compress(VectorOperation::add);
      {
        // Sixteen digits, so that two assemblies can be compared to rounding.
        const auto old_precision = pcout.get_stream().precision();
        pcout << std::setprecision(16)
              << "Galerkin assembly check: |b| "
              << exact_regularized_dirichlet_rhs.l2_norm() << ", |V|_F "
              << dirichlet_matrix.frobenius_norm()
              << std::setprecision(old_precision) << std::endl;
      }

      // The manufactured right-hand side is assembled whenever the exact
      // gradient is available, so the two constructions can be compared in
      // the same run: their difference is precisely the inconsistency that
      // breaks Galerkin orthogonality, measured rather than inferred.
      const bool compare_rhs_constructions =
        dirichlet_rhs_treatment == "manufactured_flux" ||
        std::getenv("PIBEM_COMPARE_RHS") != nullptr;
      if (exact_gradient != nullptr && compare_rhs_constructions)
        {
          TrilinosWrappers::MPI::Vector manufactured_rhs(this_cpu_set,
                                                         mpi_communicator);
          manufactured_rhs = 0.0;
          integrator.assemble_single_layer_flux_rhs(dh,
                                                    hanging_node_constraints,
                                                    manufactured_rhs,
                                                    galerkin_quadrature_order,
                                                    *exact_gradient,
                                                    this_mpi_process);
          manufactured_rhs.compress(VectorOperation::add);
          // The regularized assembly carries an overall sign, so compare the
          // two on the same footing before reporting their distance.
          TrilinosWrappers::MPI::Vector difference(this_cpu_set,
                                                   mpi_communicator);
          difference = manufactured_rhs;
          difference.add(1.0, exact_regularized_dirichlet_rhs);
          const double reference = manufactured_rhs.l2_norm();
          pcout << "Dirichlet RHS comparison: |b_manufactured| " << reference
                << ", |b_manufactured + b_regularized| " << difference.l2_norm()
                << ", relative "
                << (reference > 0.0 ? difference.l2_norm() / reference : 0.0)
                << std::endl;
          if (dirichlet_rhs_treatment == "manufactured_flux")
            {
              exact_regularized_dirichlet_rhs = manufactured_rhs;
              exact_regularized_dirichlet_rhs *= -1.0;
              pcout << "Using the manufactured single-layer Dirichlet RHS"
                    << std::endl;
            }
        }
      else
        AssertThrow(dirichlet_rhs_treatment != "manufactured_flux" ||
                      exact_gradient != nullptr,
                    ExcMessage("The manufactured Dirichlet RHS needs the "
                               "exact gradient."));

      if (galerkin_neumann_formulation == "adjoint_double_layer")
        {
          TrilinosWrappers::MPI::Vector auxiliary_trace(calderon_cpu_set,
                                                         mpi_communicator);
          std::vector<Point<dim>> auxiliary_support_points(
            calderon_dh.n_dofs());
          DoFTools::map_dofs_to_support_points<dim - 1, dim>(
            *mapping, calderon_dh, auxiliary_support_points);
          for (const auto i : calderon_cpu_set)
            if (!calderon_constraints.is_constrained(i))
              auxiliary_trace[i] =
                dirichlet_function.value(auxiliary_support_points[i]);
          auxiliary_trace.compress(VectorOperation::insert);

          calderon_hypersingular_matrix.vmult(
            adjoint_double_layer_rhs, auxiliary_trace);
          adjoint_first_kind_rhs = 0.0;
          GalerkinPanelPairIntegrator<3> auxiliary_rhs_integrator(
            *mapping,
            *calderon_fe,
            singular_strategy,
            galerkin_singular_quadrature_order,
            galerkin_duffy_outer_subdivisions,
            galerkin_near_singular_distance_ratio,
            galerkin_near_singular_maximum_depth,
            galerkin_near_singular_quadrature_order);
          auxiliary_rhs_integrator.set_screening(galerkin_screening());
          auxiliary_rhs_integrator.assemble_double_layer_rhs(
            calderon_dh,
            calderon_constraints,
            adjoint_first_kind_rhs,
            galerkin_quadrature_order,
            dirichlet_function,
            this_mpi_process,
	            true,
	            nullptr,
	            numbers::invalid_unsigned_int,
	            (galerkin_adjoint_hybrid_system &&
	             !galerkin_hybrid_conforming_complement &&
	             galerkin_adjoint_primary_material_ids.size() == 1) ?
	              *galerkin_adjoint_primary_material_ids.begin() :
	              numbers::invalid_material_id);
          adjoint_first_kind_rhs.compress(VectorOperation::add);
          for (const auto i : calderon_cpu_set)
            if (calderon_constraints.is_constrained(i))
              {
                adjoint_double_layer_rhs[i] = 0.0;
                adjoint_first_kind_rhs[i]   = 0.0;
              }
          adjoint_double_layer_rhs.compress(VectorOperation::insert);
          adjoint_first_kind_rhs.compress(VectorOperation::insert);
          pcout << "Assembled adjoint second-kind RHS W phi, L2 norm "
                << adjoint_double_layer_rhs.l2_norm() << std::endl
                << "Assembled auxiliary first-kind RHS, L2 norm "
                << adjoint_first_kind_rhs.l2_norm()
                << ", near-singular adaptive pairs/leaves "
                << Utilities::MPI::sum(
                     auxiliary_rhs_integrator.n_near_singular_pairs(),
                     mpi_communicator)
                << '/'
                << Utilities::MPI::sum(
                     auxiliary_rhs_integrator.n_near_singular_leaves(),
                     mpi_communicator)
                << std::endl;
        }

      if (galerkin_component_rhs_diagnostics &&
          dirichlet_rhs_treatment == "exact_regularized")
        {
          AssertThrow(preconditioner_type == "CalderonDual",
                      ExcMessage("Galerkin component RHS diagnostics require "
                                 "the CalderonDual topology."));
          AssertThrow(calderon_dual_component_areas.size() == 2,
                      ExcMessage("The sphere/cube RHS diagnostic requires "
                                 "exactly two connected boundary components."));

          std::vector<TrilinosWrappers::MPI::Vector> component_rhs(2);
          for (unsigned int source = 0; source < 2; ++source)
            {
              component_rhs[source].reinit(this_cpu_set,
                                            mpi_communicator);
              integrator.assemble_double_layer_rhs(
                dh,
                hanging_node_constraints,
                component_rhs[source],
                galerkin_quadrature_order,
                dirichlet_function,
                this_mpi_process,
                true,
                &calderon_dual_cell_components,
                source);
              component_rhs[source].compress(VectorOperation::add);
            }

          TrilinosWrappers::MPI::Vector reconstructed_rhs(this_cpu_set,
                                                           mpi_communicator);
          reconstructed_rhs = component_rhs[0];
          reconstructed_rhs += component_rhs[1];
          TrilinosWrappers::MPI::Vector reconstruction_error(
            reconstructed_rhs);
          reconstruction_error.add(-1.0,
                                   exact_regularized_dirichlet_rhs);

          const double sphere_norm = component_rhs[0].l2_norm();
          const double cube_norm   = component_rhs[1].l2_norm();
          const double full_norm = exact_regularized_dirichlet_rhs.l2_norm();
          const double component_cosine =
            sphere_norm > 0.0 && cube_norm > 0.0 ?
              (component_rhs[0] * component_rhs[1]) /
                (sphere_norm * cube_norm) :
              0.0;
          const double cancellation_ratio =
            sphere_norm + cube_norm > 0.0 ?
              full_norm / (sphere_norm + cube_norm) :
              0.0;

          pcout << "Regularized Galerkin RHS source decomposition:" << std::endl
                << "  sphere source: L2 " << sphere_norm << ", Linf "
                << component_rhs[0].linfty_norm() << std::endl
                << "  cube source:   L2 " << cube_norm << ", Linf "
                << component_rhs[1].linfty_norm() << std::endl
                << "  source cosine: " << component_cosine << std::endl
                << "  cancellation ratio ||b||/(||b_s||+||b_c||): "
                << cancellation_ratio << std::endl
                << "  reconstruction relative error: "
                << (full_norm > 0.0 ?
                      reconstruction_error.l2_norm() / full_norm :
                      reconstruction_error.l2_norm())
                << std::endl;

          std::array<std::array<double, 2>, 2> block_norms{};
          std::array<double, 2> row_sum_norms{};
          std::array<double, 2> row_cosines{};
          std::vector<types::global_dof_index> cell_indices(
            fe->dofs_per_cell);
          for (unsigned int target = 0; target < 2; ++target)
            {
              std::array<double, 2> local_block_squares{};
              double local_row_sum_square = 0.0;
              double local_dot            = 0.0;
              for (const auto &cell : dh.active_cell_iterators())
                {
                  if (cell->subdomain_id() != this_mpi_process ||
                      calderon_dual_cell_components
                          [cell->active_cell_index()] != target)
                    continue;
                  cell->get_dof_indices(cell_indices);
                  AssertDimension(cell_indices.size(), 1);
                  const auto dof = cell_indices[0];
                  const double from_sphere = component_rhs[0][dof];
                  const double from_cube   = component_rhs[1][dof];
                  local_block_squares[0] += from_sphere * from_sphere;
                  local_block_squares[1] += from_cube * from_cube;
                  local_row_sum_square +=
                    (from_sphere + from_cube) *
                    (from_sphere + from_cube);
                  local_dot += from_sphere * from_cube;
                }
              for (unsigned int source = 0; source < 2; ++source)
                block_norms[target][source] = std::sqrt(
                  Utilities::MPI::sum(local_block_squares[source],
                                      mpi_communicator));
              row_sum_norms[target] = std::sqrt(
                Utilities::MPI::sum(local_row_sum_square,
                                    mpi_communicator));
              const double row_dot =
                Utilities::MPI::sum(local_dot, mpi_communicator);
              row_cosines[target] =
                block_norms[target][0] > 0.0 &&
                    block_norms[target][1] > 0.0 ?
                  row_dot /
                    (block_norms[target][0] * block_norms[target][1]) :
                  0.0;
            }

          pcout << "Regularized RHS test/source block norms:" << std::endl;
          for (unsigned int target = 0; target < 2; ++target)
            {
              const double denominator = block_norms[target][0] +
                                         block_norms[target][1];
              pcout << "  test " << (target == 0 ? "sphere" : "cube")
                    << ": from sphere " << block_norms[target][0]
                    << ", from cube " << block_norms[target][1]
                    << ", sum " << row_sum_norms[target]
                    << ", cosine " << row_cosines[target]
                    << ", cancellation "
                    << (denominator > 0.0 ?
                          row_sum_norms[target] / denominator :
                          0.0)
                    << std::endl;
            }
        }

      pcout << "Assembled "
            << (dirichlet_rhs_treatment == "exact_regularized" ?
                  "regularized Galerkin Dirichlet RHS from phi(y)-phi(x)" :
                  "legacy Galerkin Dirichlet RHS from K phi and 0.5 M phi")
            << std::endl;
    }
  else
    AssertThrow(false,
                ExcMessage("Galerkin assembly currently supports 3D only."));
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

  unsigned int adaptive_singular_interactions       = 0;
  unsigned int adaptive_quasi_neumann_interactions  = 0;
  unsigned int adaptive_quasi_dirichlet_interactions = 0;
  unsigned int adaptive_unconverged_singular_neumann = 0;
  unsigned int adaptive_unconverged_singular_dirichlet = 0;
  unsigned int adaptive_unconverged_quasi_neumann   = 0;
  unsigned int adaptive_unconverged_quasi_dirichlet = 0;
  unsigned int adaptive_largest_order_used          = 0;



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

  const unsigned int n_q_points = fe_v.n_quadrature_points;

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
  Vector<double> local_neumann_matrix_row_i(fe->dofs_per_cell);
  Vector<double> local_dirichlet_matrix_row_i(fe->dofs_per_cell);

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


  cell_it cell = dh.begin_active(), endc = dh.end();

  Point<dim> D;
  double     s;

  for (cell = dh.begin_active(); cell != endc; ++cell)
  {
    fe_v.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    const std::vector<Point<dim>> &q_points    = fe_v.get_quadrature_points();
    const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();

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
    for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i) // these must now be the locally owned dofs. the rest should stay the same
    {
      if (this_cpu_set.is_element(i))
      {
        local_neumann_matrix_row_i   = 0;
        local_dirichlet_matrix_row_i = 0;

        Point<dim> singularity = support_points[i];
        Point<dim-1> ref_projection(0.0,0.0);
        bool         is_singular    = false;
        unsigned int singular_index = numbers::invalid_unsigned_int;

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
          Point<dim> singularity = support_points[i];
          const double dist_to_center =
            (singularity - cell->center()).norm();
          const double h = cell->diameter();
          if (dist_to_center <=
              (quasi_singular_distance_ratio + 1.0) * h)
          {
            QuasiSingularKernelIntegral<dim> qski(cell, *fe, *mapping, singularity);
            ref_projection = qski.get_closest_reference_point();
            const double dist_to_cell = qski.min_distance;
            
            distance_ratio = dist_to_cell / h;
            if(distance_ratio < quasi_singular_distance_ratio)
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
          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<1, dim> R = q_points[q] - support_points[i];
            if (kernel_type == "laplace")
              LaplaceKernel::kernels(R, D, s);
            else if (kernel_type == "screened")
              ScreenedKernel::kernels(R, D, s, screened_kappa);
            else
              AssertThrow(false, ExcMessage("Unknown kernel type: " + kernel_type));
            AssertThrow(std::isfinite(s) && is_finite_tensor(D),
                        ExcMessage("Non-finite screened kernel in regular "
                                   "assembly."));
            // if(support_points[i][0]==0.25&&support_points[i][1]==0.25)
            //   pcout<<"D "<<D<<" s "<<s<<" , ";
            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
            {
              local_neumann_matrix_row_i(j) +=
                ((D * normals[q]) * fe_v.shape_value(j, q) *
                  fe_v.JxW(q));
              local_dirichlet_matrix_row_i(j) +=
                (s * fe_v.shape_value(j, q) * fe_v.JxW(q));
            }
          }
        }
        else if(is_quasi_singular == true)
        {
          const auto integrate_quasi_singular =
            [&](const unsigned int exponent,
                const unsigned int order) -> Vector<double>
          {
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
              update_values | update_normal_vectors |
                update_quadrature_points | update_JxW_values);
            transformed_fe_values.reinit(cell);

            Vector<double> result(fe->dofs_per_cell);
            const auto &transformed_points =
              transformed_fe_values.get_quadrature_points();
            const auto &transformed_normals =
              transformed_fe_values.get_normal_vectors();
            for (unsigned int q = 0; q < transformed_quadrature.size(); ++q)
              {
                const Tensor<1, dim> R =
                  transformed_points[q] - support_points[i];
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
                  result[j] +=
                    (exponent == 2 ? D * transformed_normals[q] : s) *
                    transformed_fe_values.shape_value(j, q) *
                    transformed_fe_values.JxW(q);
              }
            return result;
          };

          const auto adapt_quasi_singular =
            [&](const unsigned int exponent,
                const unsigned int initial_order) -> Vector<double>
          {
            Vector<double> result =
              integrate_quasi_singular(exponent, initial_order);
            QuadratureConvergenceState convergence_state;
            unsigned int used_order = initial_order;
            bool converged = !adaptive_quadrature;
            while (adaptive_quadrature &&
                   used_order < adaptive_quadrature_max_order)
              {
                const unsigned int next_order = next_adaptive_order(
                  used_order,
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
            adaptive_largest_order_used =
              std::max(adaptive_largest_order_used, used_order);
            if (adaptive_quadrature && !converged)
              {
                if (exponent == 2)
                  ++adaptive_unconverged_quasi_neumann;
                else
                  ++adaptive_unconverged_quasi_dirichlet;
              }
            return result;
          };

          local_neumann_matrix_row_i += adapt_quasi_singular(
            2, quasi_singular_neumann_quadrature_order);
          ++adaptive_quasi_neumann_interactions;
          local_dirichlet_matrix_row_i += adapt_quasi_singular(
            1, quasi_singular_dirichlet_quadrature_order);
          ++adaptive_quasi_dirichlet_interactions;
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
            [&](const unsigned int order) -> SingularRows
          {
            Quadrature<dim - 1> singular_quadrature;
            if constexpr (dim == 3)
              singular_quadrature =
                QSplit<2>(QDuffy(order, 1.0),
                          fe->get_unit_support_points()[singular_index]);
            else
              singular_quadrature =
                QTelles<1>(order,
                           fe->get_unit_support_points()[singular_index]);

            FEValues<dim - 1, dim> singular_fe_values(
              *mapping,
              *fe,
              singular_quadrature,
              update_values | update_normal_vectors |
                update_quadrature_points | update_JxW_values);
            singular_fe_values.reinit(cell);

            SingularRows result(fe->dofs_per_cell);
            const auto &singular_normals =
              singular_fe_values.get_normal_vectors();
            const auto &singular_points =
              singular_fe_values.get_quadrature_points();
            for (unsigned int q = 0; q < singular_quadrature.size(); ++q)
              {
                const Tensor<1, dim> R =
                  singular_points[q] - support_points[i];
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
                    const double shape_j =
                      singular_fe_values.shape_value(j, q);
                    const double weight = singular_fe_values.JxW(q);
                    result.neumann[j] +=
                      (D * singular_normals[q]) * shape_j * weight;
                    result.dirichlet[j] += s * shape_j * weight;
                  }
              }
            return result;
          };

          SingularRows singular_rows =
            integrate_singular(singular_quadrature_order);
          unsigned int used_order = singular_quadrature_order;
          bool neumann_converged   = !adaptive_quadrature;
          bool dirichlet_converged = !adaptive_quadrature;
          QuadratureConvergenceState neumann_convergence_state;
          QuadratureConvergenceState dirichlet_convergence_state;
          while (adaptive_quadrature &&
                 used_order < adaptive_quadrature_max_order)
            {
              const unsigned int next_order = next_adaptive_order(
                used_order,
                adaptive_quadrature_max_order,
                adaptive_quadrature_order_growth);
              SingularRows refined = integrate_singular(next_order);
              neumann_converged = quadrature_vectors_converged(
                singular_rows.neumann,
                refined.neumann,
                adaptive_quadrature_relative_tolerance,
                adaptive_quadrature_absolute_tolerance,
                neumann_convergence_state);
              dirichlet_converged = quadrature_vectors_converged(
                singular_rows.dirichlet,
                refined.dirichlet,
                adaptive_quadrature_relative_tolerance,
                adaptive_quadrature_absolute_tolerance,
                dirichlet_convergence_state);
              singular_rows = std::move(refined);
              used_order = next_order;
              if (neumann_converged && dirichlet_converged)
                break;
            }
          adaptive_largest_order_used =
            std::max(adaptive_largest_order_used, used_order);
          if (adaptive_quadrature && !neumann_converged)
            ++adaptive_unconverged_singular_neumann;
          if (adaptive_quadrature && !dirichlet_converged)
            ++adaptive_unconverged_singular_dirichlet;
          ++adaptive_singular_interactions;
          local_neumann_matrix_row_i += singular_rows.neumann;
          local_dirichlet_matrix_row_i += singular_rows.dirichlet;
        }// end else

        // Finally, we need to add
        // the contributions of the
        // current cell to the
        // global matrix.
        for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
        {
          neumann_matrix.add(i,
                             local_dof_indices[j],
                             local_neumann_matrix_row_i(j));
          dirichlet_matrix.add(i,
                               local_dof_indices[j],
                               local_dirichlet_matrix_row_i(j));
        }
      } // end if (this_cpu_set.is_element(i))
    } // end loop on dofs
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
      const unsigned int total_singular = Utilities::MPI::sum(
        adaptive_singular_interactions, mpi_communicator);
      const unsigned int total_quasi_neumann = Utilities::MPI::sum(
        adaptive_quasi_neumann_interactions, mpi_communicator);
      const unsigned int total_quasi_dirichlet = Utilities::MPI::sum(
        adaptive_quasi_dirichlet_interactions, mpi_communicator);
      const unsigned int total_unconverged_singular_neumann =
        Utilities::MPI::sum(adaptive_unconverged_singular_neumann,
                            mpi_communicator);
      const unsigned int total_unconverged_singular_dirichlet =
        Utilities::MPI::sum(adaptive_unconverged_singular_dirichlet,
                            mpi_communicator);
      const unsigned int total_unconverged_quasi_neumann =
        Utilities::MPI::sum(adaptive_unconverged_quasi_neumann,
                            mpi_communicator);
      const unsigned int total_unconverged_quasi_dirichlet =
        Utilities::MPI::sum(adaptive_unconverged_quasi_dirichlet,
                            mpi_communicator);
      const unsigned int largest_order = Utilities::MPI::max(
        adaptive_largest_order_used, mpi_communicator);
      pcout << "Adaptive quadrature interactions: singular "
            << total_singular << ", quasi Neumann "
            << total_quasi_neumann << ", quasi Dirichlet "
            << total_quasi_dirichlet << std::endl
            << "Adaptive quadrature largest order: " << largest_order
            << ", unconverged singular-N/singular-D/quasi-N/quasi-D: "
            << total_unconverged_singular_neumann << '/'
            << total_unconverged_singular_dirichlet << '/'
            << total_unconverged_quasi_neumann << '/'
            << total_unconverged_quasi_dirichlet
            << std::endl;
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
void
BEMProblem<dim>::assemble_exact_regularized_dirichlet_rhs(
  const Function<dim> &dirichlet_function)
{
  AssertThrow(solution_method == "Direct" && kernel_type == "laplace",
              ExcMessage("Exact regularized Dirichlet RHS is implemented only "
                         "for the direct Laplace formulation."));

  pcout << "Assembling exact regularized Dirichlet RHS" << std::endl;
  exact_regularized_dirichlet_rhs = 0;

  FEValues<dim - 1, dim> regular_values(
    *mapping,
    *fe,
    *quadrature,
    update_normal_vectors | update_quadrature_points | update_JxW_values);
  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);

  unsigned int singular_interactions       = 0;
  unsigned int quasi_interactions          = 0;
  unsigned int unconverged_singular        = 0;
  unsigned int unconverged_quasi           = 0;
  unsigned int largest_order_used          = 0;

  for (auto cell = dh.begin_active(); cell != dh.end(); ++cell)
    {
      regular_values.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      const auto &regular_points = regular_values.get_quadrature_points();
      const auto &regular_normals = regular_values.get_normal_vectors();

      for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
        if (this_cpu_set.is_element(i))
          {
            const Point<dim> &singularity = support_points[i];
            const double source_value = dirichlet_function.value(singularity);
            bool is_singular = false;
            unsigned int singular_index = numbers::invalid_unsigned_int;
            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
              if (double_nodes_set[i].count(local_dof_indices[j]) > 0)
                {
                  is_singular    = true;
                  singular_index = j;
                  break;
                }

            bool is_quasi_singular = false;
            double distance_ratio = std::numeric_limits<double>::infinity();
            Point<dim - 1> ref_projection;
            if (!is_singular)
              {
                const double h = cell->diameter();
                if (singularity.distance(cell->center()) <=
                    (quasi_singular_distance_ratio + 1.0) * h)
                  {
                    QuasiSingularKernelIntegral<dim> closest_point(
                      cell, *fe, *mapping, singularity);
                    ref_projection =
                      closest_point.get_closest_reference_point();
                    distance_ratio = closest_point.min_distance / h;
                    is_quasi_singular =
                      distance_ratio < quasi_singular_distance_ratio;
                  }
              }

            const auto integrate_special =
              [&](const Quadrature<dim - 1> &integration_quadrature) -> double
            {
              FEValues<dim - 1, dim> values(
                *mapping,
                *fe,
                integration_quadrature,
                update_normal_vectors | update_quadrature_points |
                  update_JxW_values);
              values.reinit(cell);
              const auto &points  = values.get_quadrature_points();
              const auto &normals = values.get_normal_vectors();
              double integral = 0.0;
              for (unsigned int q = 0; q < integration_quadrature.size(); ++q)
                {
                  Tensor<1, dim> double_layer_kernel;
                  double single_layer_kernel;
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
                    double single_layer_kernel;
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
                  [&](const unsigned int order) -> Quadrature<dim - 1>
                {
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
                          order,
                          fe->get_unit_support_points()[singular_index]);
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
                bool converged = !adaptive_quadrature;
                QuadratureConvergenceState convergence_state;
                while (adaptive_quadrature &&
                       used_order < adaptive_quadrature_max_order)
                  {
                    const unsigned int next_order = next_adaptive_order(
                      used_order,
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
                    used_order   = next_order;
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
  if( comp_dom.input_grid_name == "../grids/sphere_box_flipped" ||
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

          // initialize geometric alpha = 2*pi then subtracts the angle between every couple of normals 
          double geom_alpha = 0.5;
          if (unique_normals.size() > 1)
            {
              Tensor<1, dim> average_normal;
              for (unsigned int k = 0; k < unique_normals.size(); ++k)
                average_normal += unique_normals[k] * (1.0 / unique_normals.size());
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
                              ExcMessage("Cannot order free-coefficient normals: "
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
                        clamp_to_unit_interval(
                          angular_origin * unique_projected_normals[k]));
                      if (angle < 0.0)
                        angle += 2.0 * numbers::PI;
                      angular_order.emplace_back(angle, k);
                    }

                  std::sort(angular_order.begin(), angular_order.end());
                  for (const auto &entry : angular_order)
                    unique_ordered_normals.push_back(unique_normals[entry.second]);
                }
              else if (dim == 2)
                unique_ordered_normals = unique_normals;
              unique_tangents.resize(unique_ordered_normals.size() + 1);
              if (dim == 3)
                unique_tangents[0] = cross_product_3d(
                  unique_ordered_normals[0],
                  unique_ordered_normals[unique_ordered_normals.size() - 1]);
              if (dim == 2)
                unique_tangents[0] = cross_product_2d(unique_ordered_normals[0]);
              for (unsigned int k = 1; k < unique_ordered_normals.size(); ++k)
                if (dim == 3)
                  unique_tangents[k] =
                    cross_product_3d(unique_ordered_normals[k],
                                     unique_ordered_normals[k - 1]);
                else if (dim == 2)
                  unique_tangents[k] = cross_product_2d(unique_ordered_normals[k]);

              unique_tangents[unique_ordered_normals.size()] = unique_tangents[0];
              // cout<<i<<"  -->Unique tangents: "<<unique_tangents.size()<<endl;
              for (unsigned int k = 0; k < unique_tangents.size(); ++k)
                {
                  const double tangent_norm = unique_tangents[k].norm();
                  AssertThrow(tangent_norm > 1e-14,
                              ExcMessage("Cannot compute free-coefficient tangent: "
                                         "two consecutive normals are parallel."));
                  unique_tangents[k] /= tangent_norm;
                  // cout<<unique_tangents[k]<<endl;
                }

              geom_alpha = 2 * numbers::PI;
              for (unsigned int k = 1; k < unique_ordered_normals.size(); ++k)
                geom_alpha -= acos(clamp_to_unit_interval(
                  unique_ordered_normals[k] * unique_ordered_normals[k - 1]));
              geom_alpha -=
                acos(clamp_to_unit_interval(
                  unique_ordered_normals[unique_ordered_normals.size() - 1] *
                  unique_ordered_normals[0]));
              // then geom_alpha is normalized to get a value between 0 and 1    
              geom_alpha /= 4 * numbers::PI;
            } 
          
          // for concave doimains we need the external angle
          if( comp_dom.input_grid_name == "../grids/sphere_box_flipped" ||
              comp_dom.input_grid_name == "../grids/sphere_box_flipped_double_nodes" ||
              comp_dom.input_grid_name == "../grids/sphere_box_flipped_new")
          {
              if(dof_in_material_1[i])
                geom_alpha = 1 - geom_alpha;
          }

          AssertThrow(std::isfinite(geom_alpha),
                      ExcMessage("Non-finite geometric free coefficient."));
          
          hyp_alpha(i) = geom_alpha;

          // just in case we need to check the code
          //const Point<dim> refinement_center(0, 0, -1);
          //const double distance_from_center = refinement_center.distance(support_points[i]);         
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
                      1. / 4. / numbers::PI *
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
                  C_matrix -= 1 / 2 / numbers::PI *
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
    pcout << "Galerkin jump term uses the consistent 0.5 mass matrix"
          << endl;
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

  pcout << "done computing free coefficients for hypersingular BIE" << std::endl;
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

      fma.generate_multipole_expansions(ones, zeros);
      fma.multipole_matr_vect_products(ones, zeros, alpha, dummy);
    }

//   alpha.print(pcout);
//   for (unsigned int i=0; i<alpha.size(); ++i)
//      {
//        cout<<std::setprecision(15)<<alpha(i) << "  (" << support_points[i] << ")  "<<endl;
//      }
}


// Compute the free coefficients for the Screened Poisson problem
template <int dim>
void
BEMProblem<dim>::compute_alpha(const double kappa)
{
  // normalizers to the size of the mesh
  // Bounding box del dominio
  BoundingBox<dim> bbox = GridTools::compute_bounding_box(comp_dom.tria);
  auto [p_min, p_max] = bbox.get_boundary_points(); // Point<dim>, Point<dim>

  // Diametro della mesh (distanza massima tra due vertici)
  //double diam = GridTools::diameter(comp_dom.tria);  

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
  DoFTools::make_hanging_node_constraints(dh, c_hn);
  c_hn.close();
  
  // Define static variables for function values and normal derivative
  static TrilinosWrappers::MPI::Vector function_coeff, normal_derivative_coeff; 
  static TrilinosWrappers::MPI::Vector tmp1, tmp2;
  
  if (function_coeff.size() != dh.n_dofs() || normal_derivative_coeff.size() != dh.n_dofs())
  {
    // Allocate vectors
    function_coeff.reinit(this_cpu_set, mpi_communicator);
    normal_derivative_coeff.reinit(this_cpu_set, mpi_communicator);
    tmp1.reinit(this_cpu_set, mpi_communicator);
    tmp2.reinit(this_cpu_set, mpi_communicator);
  }

  function_coeff = 0;
  normal_derivative_coeff = 0;
  tmp1 = 0;
  tmp2 = 0;

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
  cell_it cell = dh.begin_active(), endc = dh.end();
  const unsigned int                   dofs_per_cell = fe->dofs_per_cell;
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
        Tensor<1,dim> local_normal;
        for (unsigned int ii = 0; ii < dim; ++ii)
	        local_normal[ii] = 0;        
	        
        for (unsigned int d = 0; d < dim; ++d)
        {
          // Map indices between subdivided and original systems (to obtain vec_index)
          types::global_dof_index dummy =
            sub_wise_to_original[local_dof_indices[j]];
                  
          types::global_dof_index vec_index =
            vec_original_to_sub_wise
              [gradient_dh.n_dofs() / dim * d +
               dummy]; 

          // Consistency check
          Assert(
            vector_this_cpu_set.is_element(vec_index),
            ExcMessage(
              "vector cpu set and cpu set are inconsistent"));
          
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
                      ExcMessage("Non-finite screened alpha Neumann product."));
          AssertThrow(std::isfinite(tmp2(ii)),
                      ExcMessage("Non-finite screened alpha Dirichlet product."));
        }

    // alpha = (-tmp1 - tmp2) / function_coeff
    alpha.reinit(this_cpu_set, mpi_communicator);
    alpha  = tmp2;
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

    fma.generate_multipole_expansions(ones, zeros);
    fma.multipole_matr_vect_products(ones, zeros, alpha, dum);
  }
  
  // fill hanging node values by interpolating from master dofs
  c_hn.distribute(alpha);

// alpha.print(pcout);
// for (unsigned int i=0; i<alpha.size(); ++i)
//   {
//     pcout << i << "-> 	alpha(i): " << std::setprecision(10) << alpha(i) <<endl;
//   }
}


template <int dim>
void
BEMProblem<dim>::vmult(TrilinosWrappers::MPI::Vector       &dst,
                       const TrilinosWrappers::MPI::Vector &src) const
{
  serv_phi = src;
  if (!have_dirichlet_bc)
    {
      vector_shift(serv_phi, -serv_phi.l2_norm());
    }
  serv_dphi_dn = src;



  TrilinosWrappers::MPI::Vector matrVectProdN;
  TrilinosWrappers::MPI::Vector matrVectProdD;

  matrVectProdN.reinit(this_cpu_set, mpi_communicator);
  matrVectProdD.reinit(this_cpu_set, mpi_communicator);

  dst = 0;


  serv_phi.scale(neumann_nodes);
  serv_dphi_dn.scale(dirichlet_nodes);

  if (solution_method == "Direct")
    {
      apply_single_layer(dst, serv_dphi_dn);
      dst *= -1;
      if (boundary_integral_formulation == "collocation")
        neumann_matrix.vmult_add(dst, serv_phi);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_multipole_expansions(serv_phi, serv_dphi_dn);
      fma.multipole_matr_vect_products(serv_phi,
                                       serv_dphi_dn,
                                       matrVectProdN,
                                       matrVectProdD);
      dst += matrVectProdD;
      dst *= -1;
      dst += matrVectProdN;
    }

  if (boundary_integral_formulation == "collocation")
    {
      TrilinosWrappers::MPI::Vector jump_term(this_cpu_set, mpi_communicator);
      apply_jump_term(jump_term, serv_phi);
      dst += jump_term;
    }

  // std::cout<<"*** "<<serv_phi(0)<<" or "<<serv_dphi_dn(0)<<"   src:
  // "<<src(0)<<"  dst: "<<dst(0)<<std::endl;
  // in fully neumann bc case, we have to rescale the vector to have a zero mean
  // one
  if (!have_dirichlet_bc)
    vector_shift(dst, -dst.l2_norm());
  dst.compress(VectorOperation::add);
}


template <int dim>
void
BEMProblem<dim>::apply_single_layer(
  TrilinosWrappers::MPI::Vector       &dst,
  const TrilinosWrappers::MPI::Vector &src) const
{
  dirichlet_matrix.vmult(dst, src);
  if (laplace_patch_correction_initialized)
    for (unsigned int d = 0; d < dim; ++d)
      dst.add(laplace_patch_normal_components[d] * src,
              laplace_patch_correction_coefficients[d]);
}


template <int dim>
void
BEMProblem<dim>::apply_jump_term(
  TrilinosWrappers::MPI::Vector       &dst,
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

  serv_phi     = src;
  serv_dphi_dn = src;

  static TrilinosWrappers::MPI::Vector matrVectProdN;
  static TrilinosWrappers::MPI::Vector matrVectProdD;


  matrVectProdN.reinit(this_cpu_set, mpi_communicator);
  matrVectProdD.reinit(this_cpu_set, mpi_communicator);


  serv_phi.scale(dirichlet_nodes);
  serv_dphi_dn.scale(neumann_nodes);

  if (solution_method == "Direct")
    {
      neumann_matrix.vmult(dst, serv_phi);
      TrilinosWrappers::MPI::Vector jump_term(this_cpu_set,
                                               mpi_communicator);
      apply_jump_term(jump_term, serv_phi);
      dst += jump_term;
      dst *= -1;
      TrilinosWrappers::MPI::Vector single_layer_term(this_cpu_set,
                                                       mpi_communicator);
      apply_single_layer(single_layer_term, serv_dphi_dn);
      dst += single_layer_term;
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_multipole_expansions(serv_phi, serv_dphi_dn);
      fma.multipole_matr_vect_products(serv_phi,
                                       serv_dphi_dn,
                                       matrVectProdN,
                                       matrVectProdD);
      dst += matrVectProdN;
      TrilinosWrappers::MPI::Vector jump_term(this_cpu_set,
                                               mpi_communicator);
      apply_jump_term(jump_term, serv_phi);
      dst += jump_term;
      dst *= -1;
      dst += matrVectProdD;
    }
}





template <int dim>
void
BEMProblem<dim>::assemble_explicit_system(
  TrilinosWrappers::SparseMatrix &system_matrix) const
{
  AssertThrow(solution_method == "Direct",
              ExcMessage("MUMPS requires explicitly assembled operators."));
  AssertThrow(have_dirichlet_bc,
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
              for (unsigned int d = 0;
                   d < localized_patch_normals.size();
                   ++d)
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
BEMProblem<dim>::solve_galerkin_adjoint_double_layer(
  TrilinosWrappers::MPI::Vector &dphi_dn,
  const Function<dim>           *dirichlet_gradient)
{
#ifdef DEAL_II_WITH_MUMPS
  AssertThrow(dim == 3, ExcNotImplemented());
  AssertThrow(linear_solver == "MUMPS", ExcInternalError());

  TrilinosWrappers::MPI::Vector constant_trace(calderon_cpu_set,
                                                mpi_communicator);
  TrilinosWrappers::MPI::Vector integral_functional(calderon_cpu_set,
                                                     mpi_communicator);
  for (const auto i : calderon_cpu_set)
    if (!calderon_constraints.is_constrained(i))
      constant_trace[i] = 1.0;
  constant_trace.compress(VectorOperation::insert);
  calderon_mass_matrix.vmult(integral_functional, constant_trace);
  const int ordering_code =
    mumps_ordering == "AMD"    ? 0 :
    mumps_ordering == "AMF"    ? 2 :
    mumps_ordering == "SCOTCH" ? 3 :
    mumps_ordering == "PORD"   ? 4 :
    mumps_ordering == "METIS"  ? 5 :
    mumps_ordering == "QAMD"   ? 6 : 7;
  const std::vector<Point<dim - 1>> &primary_support_points =
    fe->get_unit_support_points();
  AssertDimension(primary_support_points.size(), fe->dofs_per_cell);
  const auto is_primary_adjoint_material =
    [this](const types::material_id material_id) {
      return galerkin_adjoint_primary_material_ids.find(material_id) !=
             galerkin_adjoint_primary_material_ids.end();
    };
  const auto dof_touches_primary_adjoint_material =
    [this](const types::global_dof_index dof) {
      for (const auto material_id : calderon_dof_material_ids[dof])
        if (galerkin_adjoint_primary_material_ids.find(material_id) !=
            galerkin_adjoint_primary_material_ids.end())
          return true;
      return false;
    };
  std::string primary_material_ids_label;
  for (const auto material_id : galerkin_adjoint_primary_material_ids)
    {
      if (!primary_material_ids_label.empty())
        primary_material_ids_label += ",";
      primary_material_ids_label += Utilities::int_to_string(material_id);
    }
  std::vector<types::global_dof_index> primary_indices(
    fe->dofs_per_cell);
  std::vector<types::global_dof_index> auxiliary_indices(
    calderon_fe->dofs_per_cell);
  const auto interpolate_to_primary =
    [&](const TrilinosWrappers::MPI::Vector &auxiliary_solution,
        TrilinosWrappers::MPI::Vector       &primary_solution) {
      Vector<double> localized_auxiliary_solution(auxiliary_solution);
      calderon_constraints.distribute(localized_auxiliary_solution);
      primary_solution = 0.0;
      auto auxiliary_cell = calderon_dh.begin_active();
      for (const auto &primary_cell : dh.active_cell_iterators())
        {
          AssertThrow(auxiliary_cell != calderon_dh.end(),
                      ExcInternalError());
          if (primary_cell->subdomain_id() == this_mpi_process)
            {
              primary_cell->get_dof_indices(primary_indices);
              auxiliary_cell->get_dof_indices(auxiliary_indices);
              for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                {
                  double value = 0.0;
                  for (unsigned int k = 0;
                       k < calderon_fe->dofs_per_cell;
                       ++k)
                    value +=
                      localized_auxiliary_solution[auxiliary_indices[k]] *
                      calderon_fe->shape_value(k,
                                               primary_support_points[j]);
                  primary_solution[primary_indices[j]] = value;
                }
            }
          ++auxiliary_cell;
        }
      AssertThrow(auxiliary_cell == calderon_dh.end(),
                  ExcInternalError());
      primary_solution.compress(VectorOperation::insert);
    };

  dphi_dn = 0.0;
  TrilinosWrappers::MPI::Vector primary_candidate(this_cpu_set,
                                                   mpi_communicator);
  if (!galerkin_adjoint_hybrid_system)
    {
      const double inverse_boundary_diameter =
        1.0 / std::max(GridTools::diameter(comp_dom.tria),
                       std::numeric_limits<double>::min());
      const auto eta_for_row = [&](const types::global_dof_index i) {
        double eta_scale = galerkin_adjoint_primary_eta_scale;
        if (galerkin_adjoint_component_weighting &&
            !dof_touches_primary_adjoint_material(i))
          eta_scale = galerkin_adjoint_secondary_eta_scale;
        return eta_scale * calderon_stabilization *
               inverse_boundary_diameter;
      };
      pcout << "Solving the consistent combined formulation "
               "(0.5 M - K' + eta V) q = W phi + eta (K + 0.5 I) phi"
            << std::endl
            << "  eta*diameter: primary material "
            << primary_material_ids_label << " -> "
            << galerkin_adjoint_primary_eta_scale * calderon_stabilization
            << ", other materials -> "
            << (galerkin_adjoint_component_weighting ?
                  galerkin_adjoint_secondary_eta_scale :
                  galerkin_adjoint_primary_eta_scale) *
                 calderon_stabilization
            << std::endl;

      TrilinosWrappers::SparseMatrix second_kind_matrix;
      second_kind_matrix.reinit(calderon_sparsity_pattern);
      TrilinosWrappers::MPI::Vector rhs(adjoint_double_layer_rhs);
      const double uniform_eta = galerkin_adjoint_primary_eta_scale *
                                 calderon_stabilization *
                                 inverse_boundary_diameter;
      if (!galerkin_adjoint_component_weighting)
        rhs.add(-uniform_eta, adjoint_first_kind_rhs);
      for (const auto i : calderon_cpu_set)
        {
          if (calderon_constraints.is_constrained(i))
            {
              second_kind_matrix.set(i, i, 1.0);
              const auto *entries =
                calderon_constraints.get_constraint_entries(i);
              if (entries != nullptr)
                for (const auto &entry : *entries)
                  second_kind_matrix.set(i, entry.first, -entry.second);
              rhs[i] = calderon_constraints.get_inhomogeneity(i);
              continue;
            }

          const double eta = galerkin_adjoint_component_weighting ?
                               eta_for_row(i) : uniform_eta;
          if (galerkin_adjoint_component_weighting)
            rhs[i] -= eta * adjoint_first_kind_rhs[i];
          for (types::global_dof_index j = 0; j < calderon_dh.n_dofs(); ++j)
            {
              const double value = 0.5 * calderon_mass_matrix(i, j) -
                                   adjoint_double_layer_matrix(i, j) +
                                   eta * adjoint_single_layer_matrix(i, j);
              if (value != 0.0)
                second_kind_matrix.set(i, j, value);
            }
        }
      second_kind_matrix.compress(VectorOperation::insert);
      rhs.compress(VectorOperation::insert);

      SparseDirectMUMPS direct_solver(
        SparseDirectMUMPS::AdditionalData(false, false, false, false),
        mpi_communicator);
      direct_solver.get_icntl()[6] = ordering_code;
      direct_solver.initialize(second_kind_matrix);
      TrilinosWrappers::MPI::Vector auxiliary_solution(calderon_cpu_set,
                                                        mpi_communicator);
      direct_solver.vmult(auxiliary_solution, rhs);

      TrilinosWrappers::MPI::Vector algebraic_residual(rhs);
      second_kind_matrix.vmult(algebraic_residual, auxiliary_solution);
      algebraic_residual -= rhs;
      interpolate_to_primary(auxiliary_solution, primary_candidate);
      TrilinosWrappers::MPI::Vector first_kind_residual(this_cpu_set,
                                                         mpi_communicator);
      dirichlet_matrix.vmult(first_kind_residual, primary_candidate);
      first_kind_residual += exact_regularized_dirichlet_rhs;
      const double relative_first_kind_residual =
        first_kind_residual.l2_norm() /
        std::max(exact_regularized_dirichlet_rhs.l2_norm(), 1e-30);
      pcout << "Combined algebraic residual: "
            << algebraic_residual.l2_norm() /
                 std::max(rhs.l2_norm(), 1e-30)
            << std::endl
            << "Combined first-kind relative residual: "
            << relative_first_kind_residual << std::endl
            << "Computed total flux: "
            << integral_functional * auxiliary_solution << std::endl;
    }

  if (galerkin_adjoint_hybrid_system)
    {
      if constexpr (dim == 3)
        {
          if (galerkin_hybrid_conforming_complement)
            {
              const auto invalid_index = numbers::invalid_dof_index;
              std::vector<types::global_dof_index> auxiliary_to_hybrid(
                calderon_dh.n_dofs(), invalid_index);
              std::vector<types::global_dof_index> auxiliary_residual_to_hybrid(
                calderon_dh.n_dofs(), invalid_index);

              std::vector<types::global_dof_index> auxiliary_dofs;
              std::vector<types::global_dof_index> primary_auxiliary_dofs;
              std::vector<types::global_dof_index> complement_auxiliary_dofs;
              auxiliary_dofs.reserve(calderon_dh.n_dofs());
              primary_auxiliary_dofs.reserve(calderon_dh.n_dofs());
              complement_auxiliary_dofs.reserve(calderon_dh.n_dofs());

              types::global_dof_index n_hybrid_dofs = 0;
              for (types::global_dof_index i = 0; i < calderon_dh.n_dofs();
                   ++i)
                if (!calderon_constraints.is_constrained(i) &&
                    !calderon_dof_material_ids[i].empty())
                  {
                    auxiliary_to_hybrid[i] = n_hybrid_dofs++;
                    auxiliary_dofs.push_back(i);
                    if (dof_touches_primary_adjoint_material(i))
                      {
                        primary_auxiliary_dofs.push_back(i);
                        if (galerkin_adjoint_dual_residual_stabilization)
                          auxiliary_residual_to_hybrid[i] = n_hybrid_dofs++;
                      }
                    else
                      complement_auxiliary_dofs.push_back(i);
                  }

              std::vector<Point<dim>> primary_component_vertices;
              unsigned int primary_material_cells   = 0;
              unsigned int secondary_material_cells = 0;
              for (const auto &cell : calderon_dh.active_cell_iterators())
                if (is_primary_adjoint_material(cell->material_id()))
                  {
                    ++primary_material_cells;
                    for (unsigned int vertex = 0;
                         vertex < GeometryInfo<dim - 1>::vertices_per_cell;
                         ++vertex)
                      primary_component_vertices.push_back(
                        cell->vertex(vertex));
                  }
                else
                  ++secondary_material_cells;
              AssertThrow(primary_material_cells > 0 &&
                            secondary_material_cells > 0,
                          ExcMessage("The conforming hybrid adjoint system "
                                     "requires both primary and complementary "
                                     "materials."));
              double primary_component_diameter = 0.0;
              for (const auto &left : primary_component_vertices)
                for (const auto &right : primary_component_vertices)
                  primary_component_diameter =
                    std::max(primary_component_diameter,
                             left.distance(right));
              AssertThrow(primary_component_diameter >
                            std::numeric_limits<double>::min(),
                          ExcMessage("Cannot determine the diameter of the "
                                     "primary hybrid component."));

              types::global_dof_index n_local_hybrid_dofs = 0;
              for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size();
                   ++i)
                if (auxiliary_to_hybrid[i] != invalid_index &&
                    calderon_cpu_set.is_element(i))
                  {
                    ++n_local_hybrid_dofs;
                    if (auxiliary_residual_to_hybrid[i] != invalid_index)
                      ++n_local_hybrid_dofs;
                  }
              AssertThrow(n_local_hybrid_dofs > 0,
                          ExcMessage("The conforming hybrid MUMPS solve "
                                     "requires every MPI rank to own at least "
                                     "one hybrid row. Use fewer MPI "
                                     "processes for this mesh."));

              const auto hybrid_counts =
                Utilities::MPI::partial_and_total_sum(n_local_hybrid_dofs,
                                                      mpi_communicator);
              const types::global_dof_index hybrid_offset =
                hybrid_counts.first;
              n_hybrid_dofs = hybrid_counts.second;
              std::vector<types::global_dof_index> local_auxiliary_to_hybrid(
                auxiliary_to_hybrid.size(), 0);
              std::vector<types::global_dof_index>
                local_auxiliary_residual_to_hybrid(
                  auxiliary_residual_to_hybrid.size(), 0);
              types::global_dof_index next_hybrid_dof = hybrid_offset;
              for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size();
                   ++i)
                if (auxiliary_to_hybrid[i] != invalid_index &&
                    calderon_cpu_set.is_element(i))
                  {
                    local_auxiliary_to_hybrid[i] = ++next_hybrid_dof;
                    if (auxiliary_residual_to_hybrid[i] != invalid_index)
                      local_auxiliary_residual_to_hybrid[i] =
                        ++next_hybrid_dof;
                  }

              std::vector<types::global_dof_index> global_auxiliary_to_hybrid(
                local_auxiliary_to_hybrid.size());
              std::vector<types::global_dof_index>
                global_auxiliary_residual_to_hybrid(
                  local_auxiliary_residual_to_hybrid.size());
              Utilities::MPI::sum(local_auxiliary_to_hybrid,
                                  mpi_communicator,
                                  global_auxiliary_to_hybrid);
              Utilities::MPI::sum(local_auxiliary_residual_to_hybrid,
                                  mpi_communicator,
                                  global_auxiliary_residual_to_hybrid);
              for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size();
                   ++i)
                auxiliary_to_hybrid[i] =
                  global_auxiliary_to_hybrid[i] == 0 ?
                    invalid_index :
                    global_auxiliary_to_hybrid[i] - 1;
              for (types::global_dof_index i = 0;
                   i < auxiliary_residual_to_hybrid.size();
                   ++i)
                auxiliary_residual_to_hybrid[i] =
                  global_auxiliary_residual_to_hybrid[i] == 0 ?
                    invalid_index :
                    global_auxiliary_residual_to_hybrid[i] - 1;

              IndexSet hybrid_cpu_set(n_hybrid_dofs);
              hybrid_cpu_set.add_range(hybrid_offset,
                                       hybrid_offset +
                                         n_local_hybrid_dofs);
              hybrid_cpu_set.compress();
              AssertThrow(Utilities::MPI::sum(hybrid_cpu_set.n_elements(),
                                              mpi_communicator) ==
                            n_hybrid_dofs,
                          ExcMessage("Conforming hybrid DoF ownership is not "
                                     "one-to-one."));

              TrilinosWrappers::SparsityPattern hybrid_sparsity;
              hybrid_sparsity.reinit(hybrid_cpu_set, mpi_communicator);
              for (const auto i : hybrid_cpu_set)
                for (types::global_dof_index j = 0; j < n_hybrid_dofs; ++j)
                  hybrid_sparsity.add(i, j);
              hybrid_sparsity.compress();

              TrilinosWrappers::SparseMatrix hybrid_matrix;
              hybrid_matrix.reinit(hybrid_sparsity);
              TrilinosWrappers::MPI::Vector hybrid_rhs(hybrid_cpu_set,
                                                       mpi_communicator);

              const double primary_eta = galerkin_adjoint_primary_eta_scale *
                                         calderon_stabilization *
                                         (1.0 / primary_component_diameter);
              const auto dual_lift_entry =
                [&](const types::global_dof_index row,
                    const types::global_dof_index column) {
                  if (galerkin_adjoint_dual_residual_lift == "mass")
                    return calderon_mass_matrix(row, column);
                  if (galerkin_adjoint_dual_residual_lift == "single_layer")
                    return adjoint_single_layer_matrix(row, column);
                  AssertThrow(galerkin_adjoint_dual_residual_lift ==
                                "hypersingular",
                              ExcInternalError());
                  return calderon_hypersingular_matrix(row, column);
                };

              for (const auto source_row : auxiliary_dofs)
                {
                  const auto row = auxiliary_to_hybrid[source_row];
                  if (!hybrid_cpu_set.is_element(row))
                    continue;
                  const bool primary_row =
                    dof_touches_primary_adjoint_material(source_row);
                  if (primary_row)
                    {
                      hybrid_rhs[row] = adjoint_double_layer_rhs[source_row];
                      for (const auto source_column : auxiliary_dofs)
                        {
                          double value =
                            0.5 * calderon_mass_matrix(source_row,
                                                       source_column) -
                            adjoint_double_layer_matrix(source_row,
                                                        source_column);
                          if (!galerkin_adjoint_dual_residual_stabilization)
                            value += primary_eta *
                                     adjoint_single_layer_matrix(source_row,
                                                                 source_column);
                          if (value != 0.0)
                            hybrid_matrix.add(
                              row,
                              auxiliary_to_hybrid[source_column],
                              value);
                        }
                      if (galerkin_adjoint_dual_residual_stabilization)
                        for (const auto source_column : primary_auxiliary_dofs)
                          {
                            const double value =
                              primary_eta *
                              dual_lift_entry(source_row, source_column);
                            if (value != 0.0)
                              hybrid_matrix.add(
                                row,
                                auxiliary_residual_to_hybrid[source_column],
                                value);
                          }
                    }
                  else
                    {
                      hybrid_rhs[row] = adjoint_first_kind_rhs[source_row];
                      for (const auto source_column : auxiliary_dofs)
                        {
                          const double value =
                            -adjoint_single_layer_matrix(source_row,
                                                         source_column);
                          if (value != 0.0)
                            hybrid_matrix.add(
                              row,
                              auxiliary_to_hybrid[source_column],
                              value);
                        }
                    }
                }

              if (galerkin_adjoint_dual_residual_stabilization)
                for (const auto source_row : primary_auxiliary_dofs)
                  {
                    const auto row = auxiliary_residual_to_hybrid[source_row];
                    if (!hybrid_cpu_set.is_element(row))
                      continue;
                    hybrid_rhs[row] = adjoint_first_kind_rhs[source_row];
                    for (const auto source_column : auxiliary_dofs)
                      {
                        const double value =
                          -adjoint_single_layer_matrix(source_row,
                                                       source_column);
                        if (value != 0.0)
                          hybrid_matrix.add(
                            row,
                            auxiliary_to_hybrid[source_column],
                            value);
                      }
                    for (const auto source_column : primary_auxiliary_dofs)
                      {
                        const double value =
                          calderon_mass_matrix(source_row, source_column);
                        if (value != 0.0)
                          hybrid_matrix.add(
                            row,
                            auxiliary_residual_to_hybrid[source_column],
                            value);
                      }
                  }

              hybrid_matrix.compress(VectorOperation::add);
              hybrid_rhs.compress(VectorOperation::insert);

              pcout << (galerkin_adjoint_dual_residual_stabilization ?
                          "Dual-residual conforming hybrid system: " :
                          "Conforming hybrid system: ")
                    << primary_auxiliary_dofs.size()
                    << " continuous primary-material dofs + "
                    << complement_auxiliary_dofs.size()
                    << " continuous complementary dofs";
              if (galerkin_adjoint_dual_residual_stabilization)
                pcout << " + " << primary_auxiliary_dofs.size()
                      << " dual residual dofs"
                      << " (" << galerkin_adjoint_dual_residual_lift
                      << " lift)";
              pcout << " = " << n_hybrid_dofs << std::endl
                    << "Hybrid primary-component diameter: "
                    << primary_component_diameter << ", eta: " << primary_eta
                    << std::endl;

              if (galerkin_hybrid_exact_trace_diagnostics &&
                  dirichlet_gradient != nullptr)
                {
                  AssertDimension(dirichlet_gradient->n_components, dim);

                  Vector<double> raw_exact_trace(calderon_dh.n_dofs());
                  const auto &auxiliary_support_points =
                    calderon_fe->get_unit_support_points();
                  const Quadrature<dim - 1> support_quadrature(
                    auxiliary_support_points);
                  FEValues<dim - 1, dim> support_values(
                    *mapping,
                    *calderon_fe,
                    support_quadrature,
                    update_quadrature_points | update_normal_vectors);
                  std::vector<types::global_dof_index> support_indices(
                    calderon_fe->dofs_per_cell);
                  Vector<double> gradient_value(dim);
                  for (const auto &cell : calderon_dh.active_cell_iterators())
                    {
                      support_values.reinit(cell);
                      cell->get_dof_indices(support_indices);
                      for (unsigned int j = 0;
                           j < calderon_fe->dofs_per_cell;
                           ++j)
                        {
                          const auto dof = support_indices[j];
                          dirichlet_gradient->vector_value(
                            support_values.quadrature_point(j),
                            gradient_value);
                          Tensor<1, dim> normal =
                            support_values.normal_vector(j);
                          const double normal_norm = normal.norm();
                          AssertThrow(normal_norm > 1e-14,
                                      ExcMessage("Cannot evaluate an exact "
                                                 "trace with a zero normal."));
                          normal /= normal_norm;
                          raw_exact_trace[dof] = 0.0;
                          for (unsigned int d = 0; d < dim; ++d)
                            raw_exact_trace[dof] +=
                              gradient_value[d] * normal[d];
                        }
                    }
                  Vector<double> constrained_exact_trace(raw_exact_trace);
                  calderon_constraints.distribute(constrained_exact_trace);
                  TrilinosWrappers::MPI::Vector parallel_raw_exact_trace(
                    calderon_cpu_set, mpi_communicator);
                  TrilinosWrappers::MPI::Vector exact_trace(
                    calderon_cpu_set, mpi_communicator);
                  for (const auto i : calderon_cpu_set)
                    {
                      parallel_raw_exact_trace[i] = raw_exact_trace[i];
                      exact_trace[i] = constrained_exact_trace[i];
                    }
                  parallel_raw_exact_trace.compress(VectorOperation::insert);
                  exact_trace.compress(VectorOperation::insert);

                  TrilinosWrappers::MPI::Vector trace_constraint_change(
                    parallel_raw_exact_trace);
                  trace_constraint_change -= exact_trace;
                  TrilinosWrappers::MPI::Vector exact_single_layer(
                    calderon_cpu_set, mpi_communicator);
                  TrilinosWrappers::MPI::Vector exact_mass(
                    calderon_cpu_set, mpi_communicator);
                  TrilinosWrappers::MPI::Vector exact_adjoint_double_layer(
                    calderon_cpu_set, mpi_communicator);
                  adjoint_single_layer_matrix.vmult(exact_single_layer,
                                                    exact_trace);
                  calderon_mass_matrix.vmult(exact_mass, exact_trace);
                  adjoint_double_layer_matrix.vmult(
                    exact_adjoint_double_layer, exact_trace);

                  TrilinosWrappers::MPI::Vector first_kind_defect(
                    exact_single_layer);
                  first_kind_defect *= -1.0;
                  first_kind_defect -= adjoint_first_kind_rhs;
                  TrilinosWrappers::MPI::Vector second_kind_core_defect(
                    exact_mass);
                  second_kind_core_defect *= 0.5;
                  second_kind_core_defect -= exact_adjoint_double_layer;
                  second_kind_core_defect -= adjoint_double_layer_rhs;

                  const auto block_l2_norm =
                    [&](const TrilinosWrappers::MPI::Vector &vector,
                        const bool primary_block) {
                      double local_square = 0.0;
                      for (const auto dof : calderon_cpu_set)
                        if (!calderon_constraints.is_constrained(dof) &&
                            dof_touches_primary_adjoint_material(dof) ==
                              primary_block)
                          local_square += vector[dof] * vector[dof];
                      return std::sqrt(Utilities::MPI::sum(local_square,
                                                            mpi_communicator));
                    };

                  const double primary_rhs_l2 =
                    block_l2_norm(adjoint_double_layer_rhs, true);
                  const double primary_first_kind_rhs_l2 =
                    block_l2_norm(adjoint_first_kind_rhs, true);
                  const double complement_first_kind_rhs_l2 =
                    block_l2_norm(adjoint_first_kind_rhs, false);
                  std::set<types::material_id> complement_material_ids;
                  for (const auto &material_ids : calderon_dof_material_ids)
                    for (const auto material_id : material_ids)
                      if (!is_primary_adjoint_material(material_id))
                        complement_material_ids.insert(material_id);
                  const auto material_l2_norm =
                    [&](const TrilinosWrappers::MPI::Vector &vector,
                        const types::material_id material_id) {
                      double local_square = 0.0;
                      for (const auto dof : calderon_cpu_set)
                        if (!calderon_constraints.is_constrained(dof) &&
                            calderon_dof_material_ids[dof].find(material_id) !=
                              calderon_dof_material_ids[dof].end())
                          local_square += vector[dof] * vector[dof];
                      return std::sqrt(Utilities::MPI::sum(local_square,
                                                            mpi_communicator));
                    };
                  pcout << "Hybrid exact-trace block diagnostics "
                           "(q=grad(phi).n):"
                        << std::endl
                        << "  trace change imposed by auxiliary continuity: "
                        << "primary "
                        << block_l2_norm(trace_constraint_change, true)
                        << ", complement "
                        << block_l2_norm(trace_constraint_change, false)
                        << std::endl
                        << "  primary second-kind core (0.5 M-K') residual: "
                        << block_l2_norm(second_kind_core_defect, true)
                        << ", relative "
                        << block_l2_norm(second_kind_core_defect, true) /
                             std::max(primary_rhs_l2, 1e-30)
                        << std::endl
                        << "  primary first-kind/dual-row residual (-Vq-rhs): "
                        << block_l2_norm(first_kind_defect, true)
                        << ", relative "
                        << block_l2_norm(first_kind_defect, true) /
                             std::max(primary_first_kind_rhs_l2, 1e-30)
                        << std::endl
                        << "  complement first-kind residual (-Vq-rhs): "
                        << block_l2_norm(first_kind_defect, false)
                        << ", relative "
                        << block_l2_norm(first_kind_defect, false) /
                             std::max(complement_first_kind_rhs_l2, 1e-30)
                        << std::endl;
                  for (const auto material_id : complement_material_ids)
                    {
                      const double material_rhs_l2 =
                        material_l2_norm(adjoint_first_kind_rhs, material_id);
                      pcout << "    material " << material_id
                            << " first-kind residual: "
                            << material_l2_norm(first_kind_defect, material_id)
                            << ", relative "
                            << material_l2_norm(first_kind_defect, material_id) /
                                 std::max(material_rhs_l2, 1e-30)
                            << std::endl;
                    }
                }

              SparseDirectMUMPS hybrid_direct_solver(
                SparseDirectMUMPS::AdditionalData(false, false, false, false),
                mpi_communicator);
              hybrid_direct_solver.get_icntl()[6] = ordering_code;
              hybrid_direct_solver.initialize(hybrid_matrix);
              TrilinosWrappers::MPI::Vector hybrid_solution(hybrid_cpu_set,
                                                            mpi_communicator);
              hybrid_direct_solver.vmult(hybrid_solution, hybrid_rhs);

              TrilinosWrappers::MPI::Vector hybrid_residual(hybrid_rhs);
              hybrid_matrix.vmult(hybrid_residual, hybrid_solution);
              hybrid_residual -= hybrid_rhs;

              TrilinosWrappers::MPI::Vector auxiliary_solution(
                calderon_cpu_set, mpi_communicator);
              const Vector<double> localized_hybrid_solution(hybrid_solution);
              for (const auto source_dof : auxiliary_dofs)
                if (calderon_cpu_set.is_element(source_dof))
                  auxiliary_solution[source_dof] =
                    localized_hybrid_solution[
                      auxiliary_to_hybrid[source_dof]];
              auxiliary_solution.compress(VectorOperation::insert);
              interpolate_to_primary(auxiliary_solution, primary_candidate);

              TrilinosWrappers::MPI::Vector hybrid_first_kind_residual(
                this_cpu_set, mpi_communicator);
              dirichlet_matrix.vmult(hybrid_first_kind_residual,
                                     primary_candidate);
              hybrid_first_kind_residual += exact_regularized_dirichlet_rhs;

              const double hybrid_rhs_norm      = hybrid_rhs.l2_norm();
              const double hybrid_solution_norm = hybrid_solution.l2_norm();
              const double hybrid_residual_norm = hybrid_residual.l2_norm();
              const double hybrid_relative_residual =
                hybrid_residual_norm / std::max(hybrid_rhs_norm, 1e-30);
              pcout << "Conforming hybrid RHS L2 norm: " << hybrid_rhs_norm
                    << ", solution L2 norm: " << hybrid_solution_norm
                    << std::endl
                    << "Conforming hybrid algebraic residual: "
                    << hybrid_relative_residual << std::endl
                    << "Conforming hybrid full first-kind relative residual: "
                    << hybrid_first_kind_residual.l2_norm() /
                         std::max(exact_regularized_dirichlet_rhs.l2_norm(),
                                  1e-30)
                    << std::endl;
              AssertThrow(std::isfinite(hybrid_relative_residual) &&
                            hybrid_relative_residual < 1e-8,
                          ExcMessage("The conforming hybrid MUMPS solve did "
                                     "not satisfy the assembled algebraic "
                                     "system."));
            }
          else
            {
      const auto invalid_index = numbers::invalid_dof_index;
      std::vector<types::global_dof_index> auxiliary_to_hybrid(
        calderon_dh.n_dofs(), invalid_index);
      std::vector<types::global_dof_index> auxiliary_residual_to_hybrid(
        calderon_dh.n_dofs(), invalid_index);
      std::vector<types::global_dof_index> primary_to_hybrid(
        dh.n_dofs(), invalid_index);

      types::global_dof_index n_hybrid_dofs = 0;
      for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
        if (!calderon_constraints.is_constrained(i) &&
            dof_touches_primary_adjoint_material(i))
          {
            auxiliary_to_hybrid[i] = n_hybrid_dofs++;
            if (galerkin_adjoint_dual_residual_stabilization)
              auxiliary_residual_to_hybrid[i] = n_hybrid_dofs++;
          }

      std::vector<types::global_dof_index> cell_dofs(fe->dofs_per_cell);
      unsigned int primary_material_cells   = 0;
      unsigned int secondary_material_cells = 0;
      std::vector<Point<dim>> primary_component_vertices;
      for (const auto &cell : dh.active_cell_iterators())
        {
          cell->get_dof_indices(cell_dofs);
          if (is_primary_adjoint_material(cell->material_id()))
            {
              ++primary_material_cells;
              for (unsigned int vertex = 0;
                   vertex < GeometryInfo<dim - 1>::vertices_per_cell;
                   ++vertex)
                primary_component_vertices.push_back(cell->vertex(vertex));
            }
          else
            {
              ++secondary_material_cells;
              for (const auto dof : cell_dofs)
                if (primary_to_hybrid[dof] == invalid_index)
                  primary_to_hybrid[dof] = n_hybrid_dofs++;
            }
        }
      AssertThrow(primary_material_cells > 0 &&
                    secondary_material_cells > 0,
                  ExcMessage("The hybrid adjoint system requires both the "
                             "selected primary material and at least one "
                             "other material."));
      double primary_component_diameter = 0.0;
      for (const auto &left : primary_component_vertices)
        for (const auto &right : primary_component_vertices)
          primary_component_diameter =
            std::max(primary_component_diameter, left.distance(right));
      AssertThrow(primary_component_diameter >
                    std::numeric_limits<double>::min(),
                  ExcMessage("Cannot determine the diameter of the primary "
                             "hybrid component."));

      std::vector<types::global_dof_index> local_auxiliary_to_hybrid(
        auxiliary_to_hybrid.size(), 0);
      std::vector<types::global_dof_index> local_auxiliary_residual_to_hybrid(
        auxiliary_residual_to_hybrid.size(), 0);
      std::vector<types::global_dof_index> local_primary_to_hybrid(
        primary_to_hybrid.size(), 0);
      types::global_dof_index n_local_hybrid_dofs = 0;
      for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size(); ++i)
        if (auxiliary_to_hybrid[i] != invalid_index &&
            calderon_cpu_set.is_element(i))
          {
            ++n_local_hybrid_dofs;
            if (auxiliary_residual_to_hybrid[i] != invalid_index)
              ++n_local_hybrid_dofs;
          }
      for (types::global_dof_index i = 0; i < primary_to_hybrid.size(); ++i)
        if (primary_to_hybrid[i] != invalid_index &&
            this_cpu_set.is_element(i))
          ++n_local_hybrid_dofs;
      AssertThrow(n_local_hybrid_dofs > 0,
                  ExcMessage("The hybrid MUMPS solve requires every MPI rank "
                             "to own at least one hybrid row. Use fewer MPI "
                             "processes for this mesh."));

      const auto hybrid_counts =
        Utilities::MPI::partial_and_total_sum(n_local_hybrid_dofs,
                                              mpi_communicator);
      const types::global_dof_index hybrid_offset = hybrid_counts.first;
      n_hybrid_dofs                              = hybrid_counts.second;
      types::global_dof_index next_hybrid_dof    = hybrid_offset;
      for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size(); ++i)
        if (auxiliary_to_hybrid[i] != invalid_index &&
            calderon_cpu_set.is_element(i))
          {
            local_auxiliary_to_hybrid[i] = ++next_hybrid_dof;
            if (auxiliary_residual_to_hybrid[i] != invalid_index)
              local_auxiliary_residual_to_hybrid[i] = ++next_hybrid_dof;
          }
      for (types::global_dof_index i = 0; i < primary_to_hybrid.size(); ++i)
        if (primary_to_hybrid[i] != invalid_index &&
            this_cpu_set.is_element(i))
          local_primary_to_hybrid[i] = ++next_hybrid_dof;

      std::vector<types::global_dof_index> global_auxiliary_to_hybrid(
        local_auxiliary_to_hybrid.size());
      std::vector<types::global_dof_index> global_auxiliary_residual_to_hybrid(
        local_auxiliary_residual_to_hybrid.size());
      std::vector<types::global_dof_index> global_primary_to_hybrid(
        local_primary_to_hybrid.size());
      Utilities::MPI::sum(local_auxiliary_to_hybrid,
                          mpi_communicator,
                          global_auxiliary_to_hybrid);
      Utilities::MPI::sum(local_auxiliary_residual_to_hybrid,
                          mpi_communicator,
                          global_auxiliary_residual_to_hybrid);
      Utilities::MPI::sum(local_primary_to_hybrid,
                          mpi_communicator,
                          global_primary_to_hybrid);
      for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size(); ++i)
        auxiliary_to_hybrid[i] =
          global_auxiliary_to_hybrid[i] == 0 ?
            invalid_index :
            global_auxiliary_to_hybrid[i] - 1;
      for (types::global_dof_index i = 0;
           i < auxiliary_residual_to_hybrid.size();
           ++i)
        auxiliary_residual_to_hybrid[i] =
          global_auxiliary_residual_to_hybrid[i] == 0 ?
            invalid_index :
            global_auxiliary_residual_to_hybrid[i] - 1;
      for (types::global_dof_index i = 0; i < primary_to_hybrid.size(); ++i)
        primary_to_hybrid[i] =
          global_primary_to_hybrid[i] == 0 ?
            invalid_index :
            global_primary_to_hybrid[i] - 1;

      IndexSet hybrid_cpu_set(n_hybrid_dofs);
      hybrid_cpu_set.add_range(hybrid_offset,
                               hybrid_offset + n_local_hybrid_dofs);
      hybrid_cpu_set.compress();
      AssertThrow(Utilities::MPI::sum(hybrid_cpu_set.n_elements(),
                                      mpi_communicator) == n_hybrid_dofs,
                  ExcMessage("Hybrid DoF ownership is not one-to-one."));

      TrilinosWrappers::SparsityPattern hybrid_sparsity;
      hybrid_sparsity.reinit(hybrid_cpu_set, mpi_communicator);
      for (const auto i : hybrid_cpu_set)
        for (types::global_dof_index j = 0; j < n_hybrid_dofs; ++j)
          hybrid_sparsity.add(i, j);
      hybrid_sparsity.compress();

      TrilinosWrappers::SparseMatrix hybrid_matrix;
      hybrid_matrix.reinit(hybrid_sparsity);
      TrilinosWrappers::MPI::Vector hybrid_rhs(hybrid_cpu_set,
                                                mpi_communicator);

      std::vector<types::global_dof_index> sphere_auxiliary_dofs;
      std::vector<types::global_dof_index> cube_primary_dofs;
      sphere_auxiliary_dofs.reserve(calderon_dh.n_dofs());
      cube_primary_dofs.reserve(dh.n_dofs());
      for (types::global_dof_index i = 0; i < auxiliary_to_hybrid.size(); ++i)
        if (auxiliary_to_hybrid[i] != invalid_index)
          sphere_auxiliary_dofs.push_back(i);
      for (types::global_dof_index i = 0; i < primary_to_hybrid.size(); ++i)
        if (primary_to_hybrid[i] != invalid_index)
          cube_primary_dofs.push_back(i);

      const double primary_eta = galerkin_adjoint_primary_eta_scale *
                                 calderon_stabilization *
                                 (1.0 / primary_component_diameter);
      std::vector<double> auxiliary_eta(calderon_dh.n_dofs(), primary_eta);
      const auto dual_lift_entry =
        [&](const types::global_dof_index row,
            const types::global_dof_index column) {
          if (galerkin_adjoint_dual_residual_lift == "mass")
            return calderon_mass_matrix(row, column);
          if (galerkin_adjoint_dual_residual_lift == "single_layer")
            return adjoint_single_layer_matrix(row, column);
          AssertThrow(galerkin_adjoint_dual_residual_lift == "hypersingular",
                      ExcInternalError());
          return calderon_hypersingular_matrix(row, column);
        };
      if (galerkin_adjoint_local_eta_scaling)
        {
          std::vector<double> local_support_length_sum(calderon_dh.n_dofs(),
                                                       0.0);
          std::vector<double> local_support_count(calderon_dh.n_dofs(), 0.0);
          std::vector<types::global_dof_index> local_auxiliary_indices(
            calderon_fe->dofs_per_cell);
          for (const auto &cell : calderon_dh.active_cell_iterators())
            {
              if (!is_primary_adjoint_material(cell->material_id()) ||
                  cell->subdomain_id() != this_mpi_process)
                continue;
              cell->get_dof_indices(local_auxiliary_indices);
              const double cell_length =
                std::max(cell->diameter(), 1e-14);
              for (const auto dof : local_auxiliary_indices)
                {
                  local_support_length_sum[dof] += cell_length;
                  local_support_count[dof] += 1.0;
                }
            }
          std::vector<double> support_length_sum(calderon_dh.n_dofs(), 0.0);
          std::vector<double> support_count(calderon_dh.n_dofs(), 0.0);
          Utilities::MPI::sum(local_support_length_sum,
                              mpi_communicator,
                              support_length_sum);
          Utilities::MPI::sum(local_support_count,
                              mpi_communicator,
                              support_count);

          double mean_length = 0.0;
          unsigned int n_measured_dofs = 0;
          for (const auto dof : sphere_auxiliary_dofs)
            if (support_count[dof] > 0.0)
              {
                mean_length += support_length_sum[dof] / support_count[dof];
                ++n_measured_dofs;
              }
          mean_length =
            Utilities::MPI::sum(mean_length, mpi_communicator);
          n_measured_dofs =
            Utilities::MPI::sum(n_measured_dofs, mpi_communicator);
          mean_length /= std::max(1U, n_measured_dofs);

          double min_eta = std::numeric_limits<double>::max();
          double max_eta = 0.0;
          for (const auto dof : sphere_auxiliary_dofs)
            if (support_count[dof] > 0.0)
              {
                const double support_length =
                  support_length_sum[dof] / support_count[dof];
                const double bounded_ratio =
                  std::max(0.25,
                           std::min(4.0, mean_length / support_length));
                auxiliary_eta[dof] = primary_eta * bounded_ratio;
                min_eta = std::min(min_eta, auxiliary_eta[dof]);
                max_eta = std::max(max_eta, auxiliary_eta[dof]);
              }

          pcout << "Hybrid local eta scaling enabled: mean support length "
                << mean_length << ", eta range [" << min_eta << ", "
                << max_eta << "]" << std::endl;
        }
      for (const auto source_row : sphere_auxiliary_dofs)
        {
          const auto row = auxiliary_to_hybrid[source_row];
          if (!hybrid_cpu_set.is_element(row))
            continue;
          const double row_eta = auxiliary_eta[source_row];
          hybrid_rhs[row] = adjoint_double_layer_rhs[source_row];
          for (const auto source_column : sphere_auxiliary_dofs)
            {
              const double value =
                0.5 * calderon_mass_matrix(source_row, source_column) -
                adjoint_double_layer_matrix(source_row, source_column) +
                (galerkin_adjoint_dual_residual_stabilization ?
                   0.0 :
                   row_eta *
                     adjoint_single_layer_matrix(source_row, source_column));
              if (value != 0.0)
                hybrid_matrix.add(row,
                                  auxiliary_to_hybrid[source_column],
                                  value);
              if (galerkin_adjoint_dual_residual_stabilization)
                {
                  const double dual_value =
                    row_eta * dual_lift_entry(source_row, source_column);
                  if (dual_value != 0.0)
                    hybrid_matrix.add(
                      row,
                      auxiliary_residual_to_hybrid[source_column],
                      dual_value);
                }
            }
        }
      if (galerkin_adjoint_dual_residual_stabilization)
        {
          for (const auto source_row : sphere_auxiliary_dofs)
            {
              const auto row = auxiliary_residual_to_hybrid[source_row];
              if (!hybrid_cpu_set.is_element(row))
                continue;
              hybrid_rhs[row] = adjoint_first_kind_rhs[source_row];
              for (const auto source_column : sphere_auxiliary_dofs)
                {
                  const double first_kind_value =
                    -adjoint_single_layer_matrix(source_row, source_column);
                  if (first_kind_value != 0.0)
                    hybrid_matrix.add(row,
                                      auxiliary_to_hybrid[source_column],
                                      first_kind_value);
                  const double mass_value =
                    calderon_mass_matrix(source_row, source_column);
                  if (mass_value != 0.0)
                    hybrid_matrix.add(
                      row,
                      auxiliary_residual_to_hybrid[source_column],
                      mass_value);
                }
            }
        }
      for (const auto source_row : cube_primary_dofs)
        {
          const auto row = primary_to_hybrid[source_row];
          if (!hybrid_cpu_set.is_element(row))
            continue;
          hybrid_rhs[row] = -exact_regularized_dirichlet_rhs[source_row];
          for (const auto source_column : cube_primary_dofs)
            {
              const double value =
                dirichlet_matrix(source_row, source_column);
              if (value != 0.0)
                hybrid_matrix.add(row,
                                  primary_to_hybrid[source_column],
                                  value);
            }
        }
      if (galerkin_hybrid_complement_mass_regularization > 0.0 ||
          galerkin_hybrid_local_mass_regularization > 0.0 ||
          galerkin_hybrid_local_fluctuation_regularization > 0.0)
        {
          const QGauss<dim - 1> complement_mass_quadrature(
            std::max(2U, galerkin_quadrature_order));
          FEValues<dim - 1, dim> complement_mass_values(
            *mapping,
            *fe,
            complement_mass_quadrature,
            update_values | update_JxW_values);
          FullMatrix<double> local_mass(fe->dofs_per_cell,
                                        fe->dofs_per_cell);
          Vector<double> local_shape_integrals(fe->dofs_per_cell);
          pcout << "Adding complement mass regularization to the hybrid "
                   "first-kind block: "
                << galerkin_hybrid_complement_mass_regularization;
          if (galerkin_hybrid_local_mass_regularization > 0.0 ||
              galerkin_hybrid_local_fluctuation_regularization > 0.0)
            pcout << " plus local " << galerkin_hybrid_local_mass_regularization
                  << " and local fluctuation "
                  << galerkin_hybrid_local_fluctuation_regularization
                  << " on material "
                  << galerkin_hybrid_local_mass_material_id
                  << " for radius in ["
                  << galerkin_hybrid_local_mass_radius_min << ", "
                  << galerkin_hybrid_local_mass_radius_max << "]";
          pcout << std::endl;
          for (const auto &cell : dh.active_cell_iterators())
            {
              if (is_primary_adjoint_material(cell->material_id()) ||
                  cell->subdomain_id() != this_mpi_process)
                continue;
              double cell_regularization =
                galerkin_hybrid_complement_mass_regularization;
              double cell_fluctuation_regularization = 0.0;
              if ((galerkin_hybrid_local_mass_regularization > 0.0 ||
                   galerkin_hybrid_local_fluctuation_regularization > 0.0) &&
                  static_cast<unsigned int>(cell->material_id()) ==
                    galerkin_hybrid_local_mass_material_id)
                {
                  const auto center = cell->center();
                  double radial_distance = 0.0;
                  if constexpr (dim >= 2)
                    radial_distance =
                      std::sqrt(center[0] * center[0] +
                                center[1] * center[1]);
                  if (radial_distance >=
                        galerkin_hybrid_local_mass_radius_min &&
                      radial_distance <
                        galerkin_hybrid_local_mass_radius_max)
                    {
                      cell_regularization +=
                        galerkin_hybrid_local_mass_regularization;
                      cell_fluctuation_regularization =
                        galerkin_hybrid_local_fluctuation_regularization;
                    }
                }
              if (cell_regularization == 0.0 &&
                  cell_fluctuation_regularization == 0.0)
                continue;
              cell->get_dof_indices(cell_dofs);
              local_mass = 0.0;
              local_shape_integrals = 0.0;
              double local_area = 0.0;
              complement_mass_values.reinit(cell);
              for (unsigned int q = 0;
                   q < complement_mass_quadrature.size();
                   ++q)
                {
                  const double jxw = complement_mass_values.JxW(q);
                  local_area += jxw;
                  for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                    {
                      local_shape_integrals[i] +=
                        complement_mass_values.shape_value(i, q) * jxw;
                      for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                        local_mass(i, j) +=
                          complement_mass_values.shape_value(i, q) *
                          complement_mass_values.shape_value(j, q) * jxw;
                    }
                }
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                {
                  const auto row = primary_to_hybrid[cell_dofs[i]];
                  if (!hybrid_cpu_set.is_element(row))
                    continue;
                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                    {
                      double value = cell_regularization * local_mass(i, j);
                      if (cell_fluctuation_regularization > 0.0)
                        value += cell_fluctuation_regularization *
                                 (local_mass(i, j) -
                                  local_shape_integrals[i] *
                                    local_shape_integrals[j] /
                                    local_area);
                      if (value != 0.0)
                        hybrid_matrix.add(row,
                                          primary_to_hybrid[cell_dofs[j]],
                                          value);
                    }
                }
            }
        }

      if (galerkin_hybrid_local_jump_regularization > 0.0)
        {
          const QGauss<dim - 2> jump_quadrature(
            std::max(2U, galerkin_quadrature_order));
          FEFaceValues<dim - 1, dim> jump_values(
            *mapping,
            *fe,
            jump_quadrature,
            update_values | update_quadrature_points | update_JxW_values);
          FEFaceValues<dim - 1, dim> neighbor_jump_values(
            *mapping,
            *fe,
            jump_quadrature,
            update_values | update_quadrature_points);
          std::vector<types::global_dof_index> neighbor_dofs(
            fe->dofs_per_cell);
          pcout << "Adding local jump regularization to the hybrid "
                   "first-kind block: "
                << galerkin_hybrid_local_jump_regularization
                << (galerkin_hybrid_local_jump_h_scaling ?
                      " with p^2/h scaling" :
                      " without h scaling")
                << " on material "
                << galerkin_hybrid_local_mass_material_id
                << " for radius in ["
                << galerkin_hybrid_local_mass_radius_min << ", "
                << galerkin_hybrid_local_mass_radius_max << "]"
                << std::endl;
          for (const auto &cell : dh.active_cell_iterators())
            {
              if (static_cast<unsigned int>(cell->material_id()) !=
                  galerkin_hybrid_local_mass_material_id)
                continue;
              for (unsigned int face_no = 0;
                   face_no < GeometryInfo<dim - 1>::faces_per_cell;
                   ++face_no)
                {
                  if (cell->face(face_no)->at_boundary())
                    continue;
                  const auto neighbor = cell->neighbor(face_no);
                  if (!neighbor->is_active() ||
                      static_cast<unsigned int>(neighbor->material_id()) !=
                        galerkin_hybrid_local_mass_material_id ||
                      cell->active_cell_index() >
                        neighbor->active_cell_index())
                    continue;

                  const auto face_center = cell->face(face_no)->center();
                  double radial_distance = 0.0;
                  if constexpr (dim >= 2)
                    radial_distance =
                      std::sqrt(face_center[0] * face_center[0] +
                                face_center[1] * face_center[1]);
                  if (radial_distance <
                        galerkin_hybrid_local_mass_radius_min ||
                      radial_distance >=
                        galerkin_hybrid_local_mass_radius_max)
                    continue;

                  const unsigned int neighbor_face_no =
                    cell->neighbor_of_neighbor(face_no);
                  cell->get_dof_indices(cell_dofs);
                  neighbor->get_dof_indices(neighbor_dofs);
                  jump_values.reinit(cell, face_no);
                  neighbor_jump_values.reinit(neighbor, neighbor_face_no);
                  double face_measure = 0.0;
                  for (unsigned int q = 0; q < jump_quadrature.size(); ++q)
                    face_measure += jump_values.JxW(q);
                  const double penalty_scale =
                    galerkin_hybrid_local_jump_h_scaling ?
                      (std::max(1U, fe->degree) *
                       std::max(1U, fe->degree) /
                       std::max(face_measure, 1e-30)) :
                      1.0;
                  std::vector<unsigned int> neighbor_quadrature_index(
                    jump_quadrature.size(), numbers::invalid_unsigned_int);
                  for (unsigned int q = 0; q < jump_quadrature.size(); ++q)
                    {
                      double       best_distance =
                        std::numeric_limits<double>::max();
                      unsigned int best_index = numbers::invalid_unsigned_int;
                      for (unsigned int neighbor_q = 0;
                           neighbor_q < jump_quadrature.size();
                           ++neighbor_q)
                        {
                          const double distance =
                            jump_values.quadrature_point(q).distance(
                              neighbor_jump_values.quadrature_point(
                                neighbor_q));
                          if (distance < best_distance)
                            {
                              best_distance = distance;
                              best_index    = neighbor_q;
                            }
                        }
                      AssertThrow(best_distance < 1e-10,
                                  ExcMessage("Could not match DG jump "
                                             "quadrature points on a shared "
                                             "face."));
                      neighbor_quadrature_index[q] = best_index;
                    }
                  for (unsigned int q = 0; q < jump_quadrature.size(); ++q)
                    {
                      const unsigned int neighbor_q =
                        neighbor_quadrature_index[q];
                      const double weight =
                        galerkin_hybrid_local_jump_regularization *
                        penalty_scale *
                        jump_values.JxW(q);
                      for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                        {
                          const auto row_cell =
                            primary_to_hybrid[cell_dofs[i]];
                          const auto row_neighbor =
                            primary_to_hybrid[neighbor_dofs[i]];
                          const bool row_cell_owned =
                            hybrid_cpu_set.is_element(row_cell);
                          const bool row_neighbor_owned =
                            hybrid_cpu_set.is_element(row_neighbor);
                          for (unsigned int j = 0;
                               j < fe->dofs_per_cell;
                               ++j)
                            {
                              const double cell_i =
                                jump_values.shape_value(i, q);
                              const double cell_j =
                                jump_values.shape_value(j, q);
                              const double neighbor_i =
                                neighbor_jump_values.shape_value(i,
                                                                 neighbor_q);
                              const double neighbor_j =
                                neighbor_jump_values.shape_value(j,
                                                                 neighbor_q);
                              if (row_cell_owned)
                                {
                                  hybrid_matrix.add(
                                    row_cell,
                                    primary_to_hybrid[cell_dofs[j]],
                                    weight * cell_i * cell_j);
                                  hybrid_matrix.add(
                                    row_cell,
                                    primary_to_hybrid[neighbor_dofs[j]],
                                    -weight * cell_i * neighbor_j);
                                }
                              if (row_neighbor_owned)
                                {
                                  hybrid_matrix.add(
                                    row_neighbor,
                                    primary_to_hybrid[cell_dofs[j]],
                                    -weight * neighbor_i * cell_j);
                                  hybrid_matrix.add(
                                    row_neighbor,
                                    primary_to_hybrid[neighbor_dofs[j]],
                                    weight * neighbor_i * neighbor_j);
                                }
                            }
                        }
                    }
                }
            }
        }

      AssertDimension(fe->dofs_per_cell, calderon_fe->dofs_per_cell);
      const auto &auxiliary_support_points =
        calderon_fe->get_unit_support_points();
      AssertDimension(primary_support_points.size(),
                      auxiliary_support_points.size());
      std::vector<unsigned int> auxiliary_to_primary_local(
        auxiliary_support_points.size(), numbers::invalid_unsigned_int);
      for (unsigned int i = 0; i < auxiliary_support_points.size(); ++i)
        for (unsigned int j = 0; j < primary_support_points.size(); ++j)
          if (auxiliary_support_points[i].distance(
                primary_support_points[j]) < 1e-14)
            {
              AssertThrow(auxiliary_to_primary_local[i] ==
                            numbers::invalid_unsigned_int,
                          ExcMessage("Ambiguous local FEQ/FEDGQ support-point "
                                     "permutation."));
              auxiliary_to_primary_local[i] = j;
            }
      for (const auto local_index : auxiliary_to_primary_local)
        AssertThrow(local_index != numbers::invalid_unsigned_int,
                    ExcMessage("The hybrid system requires matching local "
                               "FEQ and FEDGQ polynomial bases."));

      using CellIterator =
        typename DoFHandler<dim - 1, dim>::active_cell_iterator;
      std::vector<CellIterator> primary_cells;
      std::vector<std::vector<types::global_dof_index>>
        auxiliary_cell_dofs;
      primary_cells.reserve(comp_dom.tria.n_active_cells());
      auxiliary_cell_dofs.reserve(comp_dom.tria.n_active_cells());
      auto auxiliary_cell = calderon_dh.begin_active();
      for (const auto &primary_cell : dh.active_cell_iterators())
        {
          AssertThrow(auxiliary_cell != calderon_dh.end(),
                      ExcInternalError());
          primary_cells.push_back(primary_cell);
          auxiliary_cell_dofs.emplace_back(calderon_fe->dofs_per_cell);
          auxiliary_cell->get_dof_indices(auxiliary_cell_dofs.back());
          ++auxiliary_cell;
        }
      AssertThrow(auxiliary_cell == calderon_dh.end(),
                  ExcInternalError());

      std::vector<std::vector<std::pair<types::global_dof_index, double>>>
        sphere_expansion(calderon_dh.n_dofs());
      std::vector<std::vector<std::pair<types::global_dof_index, double>>>
        sphere_residual_expansion(calderon_dh.n_dofs());
      for (types::global_dof_index i = 0; i < calderon_dh.n_dofs(); ++i)
        if (calderon_constraints.is_constrained(i))
          {
            const auto *entries =
              calderon_constraints.get_constraint_entries(i);
            if (entries != nullptr)
              for (const auto &entry : *entries)
                {
                  if (auxiliary_to_hybrid[entry.first] != invalid_index)
                    sphere_expansion[i].emplace_back(
                      auxiliary_to_hybrid[entry.first], entry.second);
                  if (auxiliary_residual_to_hybrid[entry.first] !=
                      invalid_index)
                    sphere_residual_expansion[i].emplace_back(
                      auxiliary_residual_to_hybrid[entry.first],
                      entry.second);
                }
          }
        else
          {
            if (auxiliary_to_hybrid[i] != invalid_index)
              sphere_expansion[i].emplace_back(auxiliary_to_hybrid[i], 1.0);
            if (auxiliary_residual_to_hybrid[i] != invalid_index)
              sphere_residual_expansion[i].emplace_back(
                auxiliary_residual_to_hybrid[i],
                1.0);
          }

      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> cross_integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      cross_integrator.set_screening(galerkin_screening());

      std::vector<types::global_dof_index> cube_indices(fe->dofs_per_cell);
      pcout << "Assembling hybrid FEQ-sphere/FEDGQ-complement cross blocks"
            << std::endl;
      for (unsigned int sphere_cell_index = 0;
           sphere_cell_index < primary_cells.size();
           ++sphere_cell_index)
        {
          const auto &sphere_cell = primary_cells[sphere_cell_index];
          if (!is_primary_adjoint_material(sphere_cell->material_id()))
            continue;
          const auto &sphere_indices =
            auxiliary_cell_dofs[sphere_cell_index];
          bool owns_sphere_row = false;
          for (const auto dof : sphere_indices)
            {
              for (const auto &target : sphere_expansion[dof])
                owns_sphere_row = owns_sphere_row ||
                                  hybrid_cpu_set.is_element(target.first);
              for (const auto &target : sphere_residual_expansion[dof])
                owns_sphere_row = owns_sphere_row ||
                                  hybrid_cpu_set.is_element(target.first);
            }

          for (unsigned int cube_cell_index = 0;
               cube_cell_index < primary_cells.size();
               ++cube_cell_index)
            {
              const auto &cube_cell = primary_cells[cube_cell_index];
              if (is_primary_adjoint_material(cube_cell->material_id()))
                continue;
              cube_cell->get_dof_indices(cube_indices);
              bool owns_cube_row = false;
              for (const auto dof : cube_indices)
                owns_cube_row = owns_cube_row ||
                                hybrid_cpu_set.is_element(
                                  primary_to_hybrid[dof]);
              if (!owns_sphere_row && !owns_cube_row)
                continue;

              const auto local_cross =
                cross_integrator.integrate_cross_layers(
                  sphere_cell, cube_cell, galerkin_quadrature_order);

              if (owns_sphere_row)
                for (unsigned int i = 0; i < sphere_indices.size(); ++i)
                  for (const auto &row_target :
                       sphere_expansion[sphere_indices[i]])
                    {
                      if (!hybrid_cpu_set.is_element(row_target.first))
                        continue;
	                      for (unsigned int j = 0; j < cube_indices.size(); ++j)
	                        {
	                          const unsigned int primary_i =
	                            auxiliary_to_primary_local[i];
	                          const double row_eta =
	                            auxiliary_eta[sphere_indices[i]];
	                          const double value =
	                            row_target.second *
	                            ((galerkin_adjoint_dual_residual_stabilization ?
	                                0.0 :
	                                row_eta *
	                                  local_cross.single_layer(primary_i, j)) -
	                             local_cross.adjoint_double_layer(primary_i, j));
	                          if (value != 0.0)
	                            hybrid_matrix.add(
	                              row_target.first,
	                              primary_to_hybrid[cube_indices[j]],
	                              value);
	                        }
	                    }
	              if (galerkin_adjoint_dual_residual_stabilization)
	                for (unsigned int i = 0; i < sphere_indices.size(); ++i)
	                  for (const auto &row_target :
	                       sphere_residual_expansion[sphere_indices[i]])
	                    {
	                      if (!hybrid_cpu_set.is_element(row_target.first))
	                        continue;
	                      for (unsigned int j = 0; j < cube_indices.size(); ++j)
	                        {
	                          const unsigned int primary_i =
	                            auxiliary_to_primary_local[i];
	                          const double value =
	                            -row_target.second *
	                            local_cross.single_layer(primary_i, j);
	                          if (value != 0.0)
	                            hybrid_matrix.add(
	                              row_target.first,
	                              primary_to_hybrid[cube_indices[j]],
	                              value);
	                        }
	                    }

              if (owns_cube_row)
                for (unsigned int j = 0; j < cube_indices.size(); ++j)
                  {
                    const auto row = primary_to_hybrid[cube_indices[j]];
                    if (!hybrid_cpu_set.is_element(row))
                      continue;
                    for (unsigned int i = 0; i < sphere_indices.size(); ++i)
                      for (const auto &column_target :
                           sphere_expansion[sphere_indices[i]])
                        {
                          const unsigned int primary_i =
                            auxiliary_to_primary_local[i];
                          const double value =
                            column_target.second *
                            local_cross.single_layer(primary_i, j);
                          if (value != 0.0)
                            hybrid_matrix.add(row,
                                              column_target.first,
                                              value);
                        }
                  }
            }
        }
      hybrid_matrix.compress(VectorOperation::add);
      hybrid_rhs.compress(VectorOperation::insert);

      pcout << (galerkin_adjoint_dual_residual_stabilization ?
                  "Dual-residual hybrid system: " :
                  "Hybrid system: ")
            << sphere_auxiliary_dofs.size()
            << " continuous primary-material dofs + "
            << cube_primary_dofs.size()
            << " discontinuous complementary dofs";
      if (galerkin_adjoint_dual_residual_stabilization)
        pcout << " + " << sphere_auxiliary_dofs.size()
              << " dual residual dofs"
              << " (" << galerkin_adjoint_dual_residual_lift << " lift)";
      pcout << " = " << n_hybrid_dofs
            << std::endl
            << "Hybrid primary-component diameter: "
            << primary_component_diameter << ", eta: " << primary_eta
            << std::endl
            << "Hybrid cross near-singular adaptive pairs/leaves: "
            << Utilities::MPI::sum(
                 cross_integrator.n_near_singular_pairs(),
                 mpi_communicator)
            << '/'
            << Utilities::MPI::sum(
                 cross_integrator.n_near_singular_leaves(),
                 mpi_communicator)
            << std::endl;

      SparseDirectMUMPS hybrid_direct_solver(
        SparseDirectMUMPS::AdditionalData(false, false, false, false),
        mpi_communicator);
      hybrid_direct_solver.get_icntl()[6] = ordering_code;
      hybrid_direct_solver.initialize(hybrid_matrix);
      TrilinosWrappers::MPI::Vector hybrid_solution(hybrid_cpu_set,
                                                     mpi_communicator);
      hybrid_direct_solver.vmult(hybrid_solution, hybrid_rhs);

      TrilinosWrappers::MPI::Vector hybrid_residual(hybrid_rhs);
      hybrid_matrix.vmult(hybrid_residual, hybrid_solution);
      hybrid_residual -= hybrid_rhs;

      if (dirichlet_gradient != nullptr)
        {
          AssertDimension(dirichlet_gradient->n_components, dim);
          Vector<double> exact_auxiliary(calderon_dh.n_dofs());
          {
            const Quadrature<dim - 1> auxiliary_support_quadrature(
              auxiliary_support_points);
            FEValues<dim - 1, dim> auxiliary_fe_values(
              *mapping,
              *calderon_fe,
              auxiliary_support_quadrature,
              update_quadrature_points | update_normal_vectors);
            Vector<double> gradient_value(dim);
            for (const auto &cell : calderon_dh.active_cell_iterators())
              {
                if (!is_primary_adjoint_material(cell->material_id()))
                  continue;
                auxiliary_fe_values.reinit(cell);
                cell->get_dof_indices(auxiliary_indices);
                for (unsigned int j = 0; j < calderon_fe->dofs_per_cell; ++j)
                  {
                    dirichlet_gradient->vector_value(
                      auxiliary_fe_values.quadrature_point(j),
                      gradient_value);
                    Tensor<1, dim> normal =
                      auxiliary_fe_values.normal_vector(j);
                    const double normal_norm = normal.norm();
                    AssertThrow(normal_norm > 1e-14,
                                ExcMessage("Cannot evaluate the exact normal "
                                           "derivative with a zero normal."));
                    normal /= normal_norm;
                    double value = 0.0;
                    for (unsigned int d = 0; d < dim; ++d)
                      value += gradient_value[d] * normal[d];
                    exact_auxiliary[auxiliary_indices[j]] = value;
                  }
              }
            calderon_constraints.distribute(exact_auxiliary);
          }

          Vector<double> exact_primary(dh.n_dofs());
          {
            const QGauss<dim - 1> projection_quadrature(
              std::max(8U, 2U * (2U * fe->degree + 3U)));
            FEValues<dim - 1, dim> primary_fe_values(
              *mapping,
              *fe,
              projection_quadrature,
              update_values | update_quadrature_points |
                update_normal_vectors | update_JxW_values);
            std::vector<Vector<double>> gradient_values(
              projection_quadrature.size(), Vector<double>(dim));
            for (const auto &cell : dh.active_cell_iterators())
              {
                if (is_primary_adjoint_material(cell->material_id()))
                  continue;
                primary_fe_values.reinit(cell);
                dirichlet_gradient->vector_value_list(
                  primary_fe_values.get_quadrature_points(),
                  gradient_values);
                FullMatrix<double> local_mass(fe->dofs_per_cell,
                                              fe->dofs_per_cell);
                Vector<double> local_rhs(fe->dofs_per_cell);
                for (unsigned int q = 0; q < projection_quadrature.size(); ++q)
                  {
                    Tensor<1, dim> normal =
                      primary_fe_values.normal_vector(q);
                    const double normal_norm = normal.norm();
                    AssertThrow(normal_norm > 1e-14,
                                ExcMessage("Cannot evaluate the exact normal "
                                           "derivative with a zero normal."));
                    normal /= normal_norm;
                    double normal_trace = 0.0;
                    for (unsigned int d = 0; d < dim; ++d)
                      normal_trace += gradient_values[q][d] * normal[d];
                    for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                      {
                        local_rhs[i] += primary_fe_values.shape_value(i, q) *
                                        normal_trace *
                                        primary_fe_values.JxW(q);
                        for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                          local_mass(i, j) +=
                            primary_fe_values.shape_value(i, q) *
                            primary_fe_values.shape_value(j, q) *
                            primary_fe_values.JxW(q);
                      }
                  }
                FullMatrix<double> inverse_mass(local_mass);
                inverse_mass.gauss_jordan();
                Vector<double> local_projection(fe->dofs_per_cell);
                inverse_mass.vmult(local_projection, local_rhs);
                cell->get_dof_indices(primary_indices);
                for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                  exact_primary[primary_indices[j]] = local_projection[j];
              }
          }

          TrilinosWrappers::MPI::Vector exact_hybrid_solution(
            hybrid_cpu_set, mpi_communicator);
          for (const auto source_dof : sphere_auxiliary_dofs)
            {
              const auto hybrid_dof = auxiliary_to_hybrid[source_dof];
              if (hybrid_cpu_set.is_element(hybrid_dof))
                exact_hybrid_solution[hybrid_dof] =
                  exact_auxiliary[source_dof];
            }
          for (const auto source_dof : cube_primary_dofs)
            {
              const auto hybrid_dof = primary_to_hybrid[source_dof];
              if (hybrid_cpu_set.is_element(hybrid_dof))
                exact_hybrid_solution[hybrid_dof] = exact_primary[source_dof];
            }
          exact_hybrid_solution.compress(VectorOperation::insert);

          TrilinosWrappers::MPI::Vector exact_primary_candidate(
            this_cpu_set, mpi_communicator);
          exact_primary_candidate = 0.0;
          for (unsigned int cell_index = 0;
               cell_index < primary_cells.size();
               ++cell_index)
            {
              const auto &primary_cell = primary_cells[cell_index];
              if (primary_cell->subdomain_id() != this_mpi_process)
                continue;
              primary_cell->get_dof_indices(primary_indices);
              if (is_primary_adjoint_material(primary_cell->material_id()))
                {
                  const auto &local_auxiliary_indices =
                    auxiliary_cell_dofs[cell_index];
                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                    {
                      double value = 0.0;
                      for (unsigned int k = 0;
                           k < calderon_fe->dofs_per_cell;
                           ++k)
                        value += exact_auxiliary[local_auxiliary_indices[k]] *
                                 calderon_fe->shape_value(
                                   k, primary_support_points[j]);
                      exact_primary_candidate[primary_indices[j]] = value;
                    }
                }
              else
                for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                  exact_primary_candidate[primary_indices[j]] =
                    exact_primary[primary_indices[j]];
            }
	          exact_primary_candidate.compress(VectorOperation::insert);

	          if (galerkin_adjoint_dual_residual_stabilization)
	            {
	              TrilinosWrappers::MPI::Vector mass_defect(hybrid_rhs);
	              hybrid_matrix.vmult(mass_defect, exact_hybrid_solution);
	              mass_defect -= hybrid_rhs;
	              TrilinosWrappers::MPI::Vector lambda_rhs(calderon_cpu_set,
	                                                       mpi_communicator);
	              for (const auto source_dof : sphere_auxiliary_dofs)
	                {
	                  const auto residual_dof =
	                    auxiliary_residual_to_hybrid[source_dof];
	                  if (hybrid_cpu_set.is_element(residual_dof))
	                    lambda_rhs[source_dof] = -mass_defect[residual_dof];
	                }
	              lambda_rhs.compress(VectorOperation::insert);
	              SparseDirectMUMPS mass_direct_solver(
	                SparseDirectMUMPS::AdditionalData(false,
	                                                   false,
	                                                   false,
	                                                   false),
	                mpi_communicator);
	              mass_direct_solver.get_icntl()[6] = ordering_code;
	              mass_direct_solver.initialize(calderon_mass_matrix);
	              TrilinosWrappers::MPI::Vector exact_lambda(calderon_cpu_set,
	                                                         mpi_communicator);
	              mass_direct_solver.vmult(exact_lambda, lambda_rhs);
	              for (const auto source_dof : sphere_auxiliary_dofs)
	                {
	                  const auto residual_dof =
	                    auxiliary_residual_to_hybrid[source_dof];
	                  if (hybrid_cpu_set.is_element(residual_dof))
	                    exact_hybrid_solution[residual_dof] =
	                      exact_lambda[source_dof];
	                }
	              exact_hybrid_solution.compress(VectorOperation::insert);
	            }

	          TrilinosWrappers::MPI::Vector exact_full_first_kind_residual(
            this_cpu_set, mpi_communicator);
          dirichlet_matrix.vmult(exact_full_first_kind_residual,
                                 exact_primary_candidate);
          exact_full_first_kind_residual += exact_regularized_dirichlet_rhs;
          double local_full_metrics[4] = {};
          for (const auto &cell : dh.active_cell_iterators())
            {
              if (cell->subdomain_id() != this_mpi_process)
                continue;
              const unsigned int component =
                is_primary_adjoint_material(cell->material_id()) ? 0 : 1;
              cell->get_dof_indices(primary_indices);
              for (const auto dof : primary_indices)
                {
                  local_full_metrics[2 * component] +=
                    exact_full_first_kind_residual[dof] *
                    exact_full_first_kind_residual[dof];
                  local_full_metrics[2 * component + 1] +=
                    exact_regularized_dirichlet_rhs[dof] *
                    exact_regularized_dirichlet_rhs[dof];
                }
            }
          double global_full_metrics[4] = {};
          Utilities::MPI::sum(local_full_metrics,
                              mpi_communicator,
                              global_full_metrics);

          TrilinosWrappers::MPI::Vector exact_hybrid_residual(hybrid_rhs);
          hybrid_matrix.vmult(exact_hybrid_residual,
                              exact_hybrid_solution);
          exact_hybrid_residual -= hybrid_rhs;
          TrilinosWrappers::MPI::Vector hybrid_solution_error(
            hybrid_solution);
          hybrid_solution_error -= exact_hybrid_solution;

          double local_metrics[8] = {};
          for (const auto source_dof : sphere_auxiliary_dofs)
            {
              const auto hybrid_dof = auxiliary_to_hybrid[source_dof];
              if (!hybrid_cpu_set.is_element(hybrid_dof))
                continue;
              local_metrics[0] += exact_hybrid_residual[hybrid_dof] *
                                  exact_hybrid_residual[hybrid_dof];
              local_metrics[1] += hybrid_rhs[hybrid_dof] *
                                  hybrid_rhs[hybrid_dof];
              local_metrics[4] += hybrid_solution_error[hybrid_dof] *
                                  hybrid_solution_error[hybrid_dof];
              local_metrics[5] += exact_hybrid_solution[hybrid_dof] *
                                  exact_hybrid_solution[hybrid_dof];
            }
	          for (const auto source_dof : cube_primary_dofs)
	            {
	              const auto hybrid_dof = primary_to_hybrid[source_dof];
	              if (!hybrid_cpu_set.is_element(hybrid_dof))
	                continue;
              local_metrics[2] += exact_hybrid_residual[hybrid_dof] *
                                  exact_hybrid_residual[hybrid_dof];
              local_metrics[3] += hybrid_rhs[hybrid_dof] *
                                  hybrid_rhs[hybrid_dof];
              local_metrics[6] += hybrid_solution_error[hybrid_dof] *
                                  hybrid_solution_error[hybrid_dof];
	              local_metrics[7] += exact_hybrid_solution[hybrid_dof] *
	                                  exact_hybrid_solution[hybrid_dof];
	            }
	          double local_dual_metrics[4] = {};
	          if (galerkin_adjoint_dual_residual_stabilization)
	            for (const auto source_dof : sphere_auxiliary_dofs)
	              {
	                const auto hybrid_dof =
	                  auxiliary_residual_to_hybrid[source_dof];
	                if (!hybrid_cpu_set.is_element(hybrid_dof))
	                  continue;
	                local_dual_metrics[0] += exact_hybrid_residual[hybrid_dof] *
	                                         exact_hybrid_residual[hybrid_dof];
	                local_dual_metrics[1] += hybrid_rhs[hybrid_dof] *
	                                         hybrid_rhs[hybrid_dof];
	                local_dual_metrics[2] += hybrid_solution_error[hybrid_dof] *
	                                         hybrid_solution_error[hybrid_dof];
	                local_dual_metrics[3] += exact_hybrid_solution[hybrid_dof] *
	                                         exact_hybrid_solution[hybrid_dof];
	              }
	          double global_metrics[8] = {};
	          Utilities::MPI::sum(local_metrics,
	                              mpi_communicator,
	                              global_metrics);
	          double global_dual_metrics[4] = {};
	          Utilities::MPI::sum(local_dual_metrics,
	                              mpi_communicator,
	                              global_dual_metrics);
          const double exact_hybrid_relative_residual =
            exact_hybrid_residual.l2_norm() /
            std::max(hybrid_rhs.l2_norm(), 1e-30);
          const double hybrid_solution_relative_error =
            hybrid_solution_error.l2_norm() /
            std::max(exact_hybrid_solution.l2_norm(), 1e-30);
          pcout << "Exact projected hybrid residual: total "
                << exact_hybrid_relative_residual
                << ", sphere rows "
                << std::sqrt(global_metrics[0]) /
                     std::max(std::sqrt(global_metrics[1]), 1e-30)
                << ", cube rows "
                << std::sqrt(global_metrics[2]) /
                     std::max(std::sqrt(global_metrics[3]), 1e-30)
                << std::endl
                << "Exact projected full first-kind residual with hybrid "
                   "trace: total "
                << exact_full_first_kind_residual.l2_norm() /
                     std::max(exact_regularized_dirichlet_rhs.l2_norm(),
                              1e-30)
                << ", sphere rows "
                << std::sqrt(global_full_metrics[0]) /
                     std::max(std::sqrt(global_full_metrics[1]), 1e-30)
                << ", cube rows "
                << std::sqrt(global_full_metrics[2]) /
                     std::max(std::sqrt(global_full_metrics[3]), 1e-30)
                << std::endl
                << "Hybrid algebraic solution error vs exact projection: total "
                << hybrid_solution_relative_error
                << ", sphere unknowns "
                << std::sqrt(global_metrics[4]) /
                     std::max(std::sqrt(global_metrics[5]), 1e-30)
	                << ", cube unknowns "
	                << std::sqrt(global_metrics[6]) /
	                     std::max(std::sqrt(global_metrics[7]), 1e-30)
	                << std::endl;
	          if (galerkin_adjoint_dual_residual_stabilization)
	            {
	              const double lambda_error_norm =
	                std::sqrt(global_dual_metrics[2]);
	              const double exact_lambda_norm =
	                std::sqrt(global_dual_metrics[3]);
	              pcout << "Dual residual rows: exact projected residual "
	                    << std::sqrt(global_dual_metrics[0]) /
	                         std::max(std::sqrt(global_dual_metrics[1]),
	                                  1e-30);
	              if (exact_lambda_norm > 1e-20)
	                pcout << ", lambda relative error "
	                      << lambda_error_norm / exact_lambda_norm;
	              else
	                pcout << ", lambda absolute error "
	                      << lambda_error_norm
	                      << " (exact lambda norm " << exact_lambda_norm
	                      << ")";
	              pcout << std::endl;
	            }
	        }

      const Vector<double> localized_hybrid_solution(hybrid_solution);
      Vector<double> localized_auxiliary_solution(calderon_dh.n_dofs());
      for (const auto source_dof : sphere_auxiliary_dofs)
        localized_auxiliary_solution[source_dof] =
          localized_hybrid_solution[auxiliary_to_hybrid[source_dof]];
      calderon_constraints.distribute(localized_auxiliary_solution);

      primary_candidate = 0.0;
      for (unsigned int cell_index = 0;
           cell_index < primary_cells.size();
           ++cell_index)
        {
          const auto &primary_cell = primary_cells[cell_index];
          if (primary_cell->subdomain_id() != this_mpi_process)
            continue;
          primary_cell->get_dof_indices(primary_indices);
          if (is_primary_adjoint_material(primary_cell->material_id()))
            {
              const auto &local_auxiliary_indices =
                auxiliary_cell_dofs[cell_index];
              for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                {
                  double value = 0.0;
                  for (unsigned int k = 0;
                       k < calderon_fe->dofs_per_cell;
                       ++k)
                    value += localized_auxiliary_solution[
                               local_auxiliary_indices[k]] *
                             calderon_fe->shape_value(
                               k, primary_support_points[j]);
                  primary_candidate[primary_indices[j]] = value;
                }
            }
          else
            for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
              primary_candidate[primary_indices[j]] =
                localized_hybrid_solution[
                  primary_to_hybrid[primary_indices[j]]];
        }
      primary_candidate.compress(VectorOperation::insert);

      TrilinosWrappers::MPI::Vector hybrid_first_kind_residual(
        this_cpu_set, mpi_communicator);
      dirichlet_matrix.vmult(hybrid_first_kind_residual,
                             primary_candidate);
      hybrid_first_kind_residual += exact_regularized_dirichlet_rhs;
      if (galerkin_hybrid_first_kind_polishing_steps > 0)
        {
          pcout << "Polishing hybrid solution with "
                << galerkin_hybrid_first_kind_polishing_steps
                << " damped first-kind correction step(s), relaxation "
                << galerkin_hybrid_first_kind_polishing_relaxation
                << std::endl
                << "  initial first-kind relative residual: "
                << hybrid_first_kind_residual.l2_norm() /
                     std::max(exact_regularized_dirichlet_rhs.l2_norm(),
                              1e-30)
                << std::endl;
          SparseDirectMUMPS polishing_direct_solver(
            SparseDirectMUMPS::AdditionalData(false, false, false, false),
            mpi_communicator);
          polishing_direct_solver.get_icntl()[6] = ordering_code;
          polishing_direct_solver.initialize(dirichlet_matrix);
          for (unsigned int step = 0;
               step < galerkin_hybrid_first_kind_polishing_steps;
               ++step)
            {
              TrilinosWrappers::MPI::Vector correction_rhs(
                hybrid_first_kind_residual);
              correction_rhs *= -1.0;
              TrilinosWrappers::MPI::Vector correction(this_cpu_set,
                                                       mpi_communicator);
              polishing_direct_solver.vmult(correction, correction_rhs);
              primary_candidate.add(
                galerkin_hybrid_first_kind_polishing_relaxation, correction);
              primary_candidate.compress(VectorOperation::add);
              dirichlet_matrix.vmult(hybrid_first_kind_residual,
                                     primary_candidate);
              hybrid_first_kind_residual += exact_regularized_dirichlet_rhs;
              pcout << "  polishing step " << step + 1
                    << " first-kind relative residual: "
                    << hybrid_first_kind_residual.l2_norm() /
                         std::max(exact_regularized_dirichlet_rhs.l2_norm(),
                                  1e-30)
                    << std::endl;
            }
        }
      const double hybrid_rhs_norm       = hybrid_rhs.l2_norm();
      const double hybrid_solution_norm  = hybrid_solution.l2_norm();
      const double hybrid_residual_norm  = hybrid_residual.l2_norm();
      const double hybrid_relative_residual =
        hybrid_residual_norm / std::max(hybrid_rhs_norm, 1e-30);
      pcout << "Hybrid RHS L2 norm: " << hybrid_rhs_norm
            << ", solution L2 norm: " << hybrid_solution_norm
            << std::endl
            << "Hybrid algebraic residual: "
            << hybrid_relative_residual
            << std::endl
            << "Hybrid full first-kind relative residual: "
            << hybrid_first_kind_residual.l2_norm() /
                 std::max(exact_regularized_dirichlet_rhs.l2_norm(),
                          1e-30)
            << std::endl;
      AssertThrow(std::isfinite(hybrid_relative_residual) &&
                    hybrid_relative_residual < 1e-8,
                  ExcMessage("The hybrid MUMPS solve did not satisfy the "
                             "assembled algebraic system."));
        }
        }
      else
        AssertThrow(false,
                    ExcMessage("The hybrid adjoint system is available in "
                               "3D only."));
    }
  else if (galerkin_adjoint_block_completion)
    {
      std::vector<bool> primary_material_dof(dh.n_dofs(), false);
      std::vector<types::global_dof_index> cell_dofs(fe->dofs_per_cell);
      unsigned int primary_material_cells   = 0;
      unsigned int secondary_material_cells = 0;
      for (const auto &cell : dh.active_cell_iterators())
        {
          cell->get_dof_indices(cell_dofs);
	          const bool is_primary_material =
	            is_primary_adjoint_material(cell->material_id());
          if (is_primary_material)
            ++primary_material_cells;
          else
            ++secondary_material_cells;
          for (const auto dof : cell_dofs)
            primary_material_dof[dof] = is_primary_material;
        }
      AssertThrow(primary_material_cells > 0 &&
                    secondary_material_cells > 0,
                  ExcMessage("Adjoint block completion requires both the "
                             "selected primary material and at least one "
                             "other material."));

      const Vector<double> localized_primary_candidate(primary_candidate);
      TrilinosWrappers::SparseMatrix block_matrix;
      block_matrix.copy_from(dirichlet_matrix);
      TrilinosWrappers::MPI::Vector block_rhs(this_cpu_set,
                                               mpi_communicator);
      block_rhs = exact_regularized_dirichlet_rhs;
      block_rhs *= -1.0;
      for (const auto i : this_cpu_set)
        if (primary_material_dof[i])
          {
            block_matrix.clear_row(i, 1.0);
            block_rhs[i] = localized_primary_candidate[i];
          }
      block_rhs.compress(VectorOperation::insert);

      pcout << "Completing the flux with a primary-material K' block and "
               "a complementary V-DG block"
            << std::endl;
      SparseDirectMUMPS block_direct_solver(
        SparseDirectMUMPS::AdditionalData(false, false, false, false),
        mpi_communicator);
      block_direct_solver.get_icntl()[6] = ordering_code;
      block_direct_solver.initialize(block_matrix);
      TrilinosWrappers::MPI::Vector block_solution(this_cpu_set,
                                                    mpi_communicator);
      block_direct_solver.vmult(block_solution, block_rhs);

      TrilinosWrappers::MPI::Vector block_residual(block_rhs);
      block_matrix.vmult(block_residual, block_solution);
      block_residual -= block_rhs;
      TrilinosWrappers::MPI::Vector completed_first_kind_residual(
        this_cpu_set, mpi_communicator);
      dirichlet_matrix.vmult(completed_first_kind_residual,
                             block_solution);
      completed_first_kind_residual += exact_regularized_dirichlet_rhs;
      pcout << "Block completion algebraic residual: "
            << block_residual.l2_norm() /
                 std::max(block_rhs.l2_norm(), 1e-30)
            << std::endl
            << "Completed first-kind relative residual: "
            << completed_first_kind_residual.l2_norm() /
                 std::max(exact_regularized_dirichlet_rhs.l2_norm(),
                          1e-30)
            << std::endl;
      primary_candidate = block_solution;
    }
  dphi_dn = primary_candidate;

  sol = dphi_dn;
#else
  (void)dphi_dn;
  AssertThrow(false,
              ExcMessage("The adjoint second-kind formulation requires a "
                         "deal.II build with MUMPS."));
#endif
}


template <int dim>
void
BEMProblem<dim>::solve_system(TrilinosWrappers::MPI::Vector       &phi,
                              TrilinosWrappers::MPI::Vector       &dphi_dn,
                              const TrilinosWrappers::MPI::Vector &tmp_rhs,
                              const Function<dim>                 *dirichlet_gradient)
{
  Teuchos::TimeMonitor                       LocalTimer(*LacSolveTime);
  system_rhs = 0;
  sol        = 0;
  alpha      = 0;
  if (boundary_integral_formulation == "galerkin" &&
      galerkin_neumann_formulation == "adjoint_double_layer")
    {
      compute_rhs(system_rhs, tmp_rhs);
      compute_constraints(constr_cpu_set, constraints, tmp_rhs);
      ConstrainedOperator<TrilinosWrappers::MPI::Vector, BEMProblem<dim>> cc(
        *this, constraints, constr_cpu_set, mpi_communicator);
      cc.distribute_rhs(system_rhs);
      system_rhs.compress(VectorOperation::insert);
      solve_galerkin_adjoint_double_layer(dphi_dn, dirichlet_gradient);
      return;
    }
  
  
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
              laplace_patch_correction_coefficients[d].reinit(
                this_cpu_set, mpi_communicator);
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

              dirichlet_matrix.vmult(
                residuals[d], laplace_patch_normal_components[d]);
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
              normal_gram(d, e) =
                laplace_patch_normal_components[d] *
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
          pcout << "Applied rank-" << dim
                << " Laplace linear patch correction" << std::endl;
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
      hypersingular_correction_steps > 0 ||
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
            dirichlet_matrix, this_cpu_set, mpi_communicator, constraints);
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
          const CellBlockPreconditioner block_preconditioner(dirichlet_matrix,
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
      else
        {
          AssertThrow(preconditioner_type == "None",
                      ExcMessage("The conjugate gradient path supports "
                                 "Preconditioner = None, Jacobi or "
                                 "BlockJacobi; the other preconditioners are "
                                 "not symmetric."));
          pcout << "Iterative solver: CG without preconditioning"
                << std::endl;
          PreconditionIdentity identity_preconditioner;
          solver.solve(negated_operator,
                       sol,
                       negated_rhs,
                       identity_preconditioner);
        }
      if (constraints.n_constraints() > 0)
        constraints.distribute(sol);
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
      const int ordering_code =
        mumps_ordering == "AMD"    ? 0 :
        mumps_ordering == "AMF"    ? 2 :
        mumps_ordering == "SCOTCH" ? 3 :
        mumps_ordering == "PORD"   ? 4 :
        mumps_ordering == "METIS"  ? 5 :
        mumps_ordering == "QAMD"   ? 6 : 7;
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
          const double boundary_diameter =
            GridTools::diameter(comp_dom.tria);
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
          solver.solve(cc,
                       sol,
                       system_rhs,
                       calderon_preconditioner);
#else
          AssertThrow(false,
                      ExcMessage("The exact mass inverses in the Calderon "
                                 "preconditioner require MUMPS."));
#endif
        }
      else if (preconditioner_type == "CalderonDual")
        {
          const double boundary_diameter =
            GridTools::diameter(comp_dom.tria);
          const double stabilization =
            calderon_stabilization /
            std::max(boundary_diameter, std::numeric_limits<double>::min());
          pcout << "CalderonDual preconditioner: D^-1 [P^T (W + gamma M) P "
                   "+ Q^T B_b Q] D^-1, gamma = "
                << stabilization << ", bubble scaling = "
                << calderon_dual_bubble_scaling << std::endl;
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
          solver.solve(cc,
                       sol,
                       system_rhs,
                       calderon_dual_preconditioner);
        }
      else
        {
          if (preconditioner_type == "None")
            {
              PreconditionIdentity identity_preconditioner;
              solver.solve(cc,
                           sol,
                           system_rhs,
                           identity_preconditioner);
            }
          else
            {
              assemble_preconditioner();
              solver.solve(cc, sol, system_rhs, preconditioner);
            }
        }
      pcout << "GMRES iterations: " << solver_control.last_step()
            << ", final residual: " << solver_control.last_value()
            << std::endl;
    }
  else
    {
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
        solver_control,
        SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(100));
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      TrilinosWrappers::PreconditionILU &fma_preconditioner =
        fma.FMA_preconditioner(alpha, constraints);
      solver.solve(cc, sol, system_rhs, fma_preconditioner);
      pcout << "GMRES iterations: " << solver_control.last_step()
            << ", final residual: " << solver_control.last_value()
            << std::endl;
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

  // for (unsigned int i = 0; i < sol.size(); i++)
  //   if (this_cpu_set.is_element(i))
  //      pcout<<std::setprecision(20)<<sol(i)<<std::endl;



  ///////////////////////////////////
  /*
    std::vector<Point<dim> > support_points(dh.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( mapping, dh,
    support_points); pcout<<"**solution "<<std::endl; for (unsigned int i = 0; i
    < alpha.size(); i++) if (this_cpu_set.is_element(i)) pcout<<i<<"
    ("<<this_mpi_process<<")
    "<<support_points[i](0)+support_points[i](1)+support_points[i](2)<<"
    "<<sol(i)<<std::endl;

     pcout<<"SOLUTION "<<std::endl;
     for (unsigned int i = 0; i < alpha.size(); i++)
         if (this_cpu_set.is_element(i))
            pcout<<i<<" ("<<this_mpi_process<<")  "<<sol(i)<<std::endl;
  */
  //////////////////////////////////

  for (types::global_dof_index i = 0; i < dirichlet_nodes.size(); i++)
    {
      if (this_cpu_set.is_element(i))
        {
          if (dirichlet_nodes(i) == 0)
            {
              phi(i) = sol(i);
            }
          else
            {
              dphi_dn(i) = sol(i);
            }
        }
    }
  phi(this_cpu_set.nth_index_in_set(0)) = phi(this_cpu_set.nth_index_in_set(0));
  dphi_dn(this_cpu_set.nth_index_in_set(0)) =
    dphi_dn(this_cpu_set.nth_index_in_set(0));
  phi.compress(VectorOperation::insert);
  dphi_dn.compress(VectorOperation::insert);

  // if (!have_dirichlet_bc)
  //   vector_shift(phi,-phi.l2_norm());

  // for (unsigned int i=0;i<dh.n_dofs();++i)
  // std::cout<<i<<" "<<tmp_rhs(i)<<" "<<dphi_dn(i)<<" "<<phi(i)<<"
  // "<<dirichlet_nodes(i)<<std::endl;

  // pcout<<"sol "<<std::endl;
  // for (unsigned int i = 0; i < sol.size(); i++)
  //    {
  // pcout<<i<<" "<<sol(i)<<" ";
  // std::set<unsigned int> doubles = double_nodes_set[i];
  // for (std::set<unsigned int>::iterator it = doubles.begin() ; it !=
  // doubles.end(); it++ )
  //    pcout<<*it<<"("<<dirichlet_nodes(*it)<<") ";
  // pcout<<"phi "<<phi(i)<<"  dphi_dn "<<dphi_dn(i);
  // pcout<<std::endl;
  //    }
}



// This method performs a Bem resolution,
// either in a direct or multipole method
template <int dim>
void
BEMProblem<dim>::solve(TrilinosWrappers::MPI::Vector       &phi,
                       TrilinosWrappers::MPI::Vector       &dphi_dn,
                       const TrilinosWrappers::MPI::Vector &tmp_rhs,
                       const Function<dim>                 *dirichlet_function,
                       const Function<dim>                 *dirichlet_gradient)
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
      AssertThrow(globally_dirichlet == dh.n_dofs(),
                  ExcMessage("The current Galerkin formulation supports pure "
                             "Dirichlet boundary conditions only."));
    }

  if (dirichlet_rhs_treatment == "exact_regularized")
    {
      AssertThrow(dirichlet_function != nullptr,
                  ExcMessage("Exact regularized Dirichlet RHS requires a "
                             "boundary function."));
      // The Galerkin assembly regularises the screened double layer too,
      // through the Laplace part of its kernel; the collocation one does not.
      AssertThrow(solution_method == "Direct" &&
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

  if (solution_method == "Direct")
    {
      if (boundary_integral_formulation == "galerkin")
        galerkin_fused_dirichlet_function = dirichlet_function;
      assemble_system();
      galerkin_fused_dirichlet_function = nullptr;
      if (boundary_integral_formulation == "galerkin")
        assemble_exact_galerkin_dirichlet_rhs(*dirichlet_function,
                                              dirichlet_gradient);
      else if (dirichlet_rhs_treatment == "exact_regularized")
        assemble_exact_regularized_dirichlet_rhs(*dirichlet_function);
      // neumann_matrix.print(std::cout);
      // dirichlet_matrix.print(std::cout);
    }
  else
    {
      AssertThrow(dim == 3, ExcMessage("FMA only works in 3D"));

      fma.generate_octree_blocking();
      // fma.compute_m2l_flags();
      fma.direct_integrals();
      fma.multipole_integrals();
    }

  if (galerkin_analytical_hypersingular_rhs)
    {
      AssertThrow(boundary_integral_formulation == "galerkin" &&
                    galerkin_neumann_formulation == "adjoint_double_layer" &&
                    dirichlet_gradient != nullptr,
                  ExcMessage("Analytical W RHS requires Galerkin adjoint "
                             "formulation and the boundary gradient."));
      if constexpr (dim == 3)
        {
          pcout << "Assembling W phi from analytical boundary gradient"
                << std::endl;
          const Vector<double> previous_rhs(adjoint_double_layer_rhs);
          adjoint_double_layer_rhs = 0.0;
          GalerkinPanelPairIntegrator<3> integrator(
            *mapping, *calderon_fe,
            galerkin_singular_quadrature == "iterated_duffy" ?
              GalerkinSingularQuadrature::iterated_duffy :
              GalerkinSingularQuadrature::sauter_schwab,
            galerkin_singular_quadrature_order,
            galerkin_duffy_outer_subdivisions,
            galerkin_near_singular_distance_ratio,
            galerkin_near_singular_maximum_depth,
            galerkin_near_singular_quadrature_order);
          integrator.set_screening(galerkin_screening());
          std::vector<types::global_dof_index> indices(
            calderon_fe->dofs_per_cell);
          for (const auto &test : calderon_dh.active_cell_iterators())
            {
	              if (test->subdomain_id() != this_mpi_process ||
	                  (galerkin_adjoint_hybrid_system &&
	                   galerkin_adjoint_primary_material_ids.find(
	                     test->material_id()) ==
	                     galerkin_adjoint_primary_material_ids.end()))
	                continue;
              Vector<double> local_rhs(calderon_fe->dofs_per_cell);
              for (const auto &trial : calderon_dh.active_cell_iterators())
                local_rhs += integrator.integrate_hypersingular_rhs(
                  test, trial, galerkin_quadrature_order,
                  *dirichlet_gradient);
              test->get_dof_indices(indices);
              calderon_constraints.distribute_local_to_global(
                local_rhs, indices, adjoint_double_layer_rhs);
            }
          adjoint_double_layer_rhs.compress(VectorOperation::add);
          Vector<double> difference(adjoint_double_layer_rhs);
          difference -= previous_rhs;
          pcout << "Analytical W RHS L2: "
                << adjoint_double_layer_rhs.l2_norm()
                << "; change from interpolated trace: "
                << difference.l2_norm() << std::endl;
        }
    }
  solve_system(phi, dphi_dn, tmp_rhs, dirichlet_gradient);

  if (hypersingular_correction_steps > 0)
    {
      AssertThrow(kernel_type == "laplace" && dim == 3,
                  ExcMessage("Hypersingular trace correction is currently "
                             "available only for three-dimensional Laplace."));

      unsigned int locally_dirichlet = 0;
      for (const auto i : this_cpu_set)
        if (dirichlet_nodes[i] > 0.5)
          ++locally_dirichlet;
      const unsigned int globally_dirichlet =
        Utilities::MPI::sum(locally_dirichlet, mpi_communicator);
      AssertThrow(globally_dirichlet == dh.n_dofs(),
                  ExcMessage("Hypersingular trace correction currently "
                             "requires pure Dirichlet boundary conditions."));

      for (unsigned int step = 0; step < hypersingular_correction_steps;
           ++step)
        {
          const double previous_trace_norm = dphi_dn.l2_norm();
          compute_gradients_hypersingular(phi, dphi_dn);
          TrilinosWrappers::MPI::Vector corrected_dphi_dn(dphi_dn);
          TrilinosWrappers::MPI::Vector correction(dphi_dn);
          correction = 0;

          for (const auto i : this_cpu_set)
            {
              const types::global_dof_index original_i =
                sub_wise_to_original[i];
              double hypersingular_trace = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                {
                  const types::global_dof_index vector_i =
                    vec_original_to_sub_wise[d * dh.n_dofs() + original_i];
                  hypersingular_trace += vector_normals_solution[vector_i] *
                                         vector_gradients_solution[vector_i];
                }
              correction[i] = hypersingular_trace - dphi_dn[i];
              corrected_dphi_dn[i] +=
                hypersingular_correction_relaxation * correction[i];
            }
          correction.compress(VectorOperation::insert);
          corrected_dphi_dn.compress(VectorOperation::insert);
          const double relative_update =
            hypersingular_correction_relaxation * correction.l2_norm() /
            std::max(previous_trace_norm,
                     std::numeric_limits<double>::min());
          if (!std::isfinite(relative_update) ||
              relative_update >
                hypersingular_correction_max_relative_update)
            {
              pcout << "Hypersingular trace correction step " << step + 1
                    << " rejected: relative update " << relative_update
                    << " exceeds "
                    << hypersingular_correction_max_relative_update
                    << std::endl;
              break;
            }
          constraints.distribute(corrected_dphi_dn);
          corrected_dphi_dn.compress(VectorOperation::insert);
          dphi_dn = corrected_dphi_dn;

          pcout << "Hypersingular trace correction step " << step + 1
                << " accepted: relative update " << relative_update
                << std::endl;
        }
    }
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
  Vector<double> localized_surface_gradients(vector_surface_gradients_solution);
  Vector<double> localized_normals(vector_normals_solution);
  Vector<double> localized_dirichlet_nodes(dirichlet_nodes);
  Vector<double> loc_tmp_rhs(tmp_rhs.size());
  loc_tmp_rhs = tmp_rhs;

  // we start clearing the constraint matrix
  c.clear();

  // here we prepare the constraint matrix so as to account for the presence
  // hanging nodes

  AffineConstraints<double> c_hn;
  DoFTools::make_hanging_node_constraints(dh, c_hn);
  c_hn.close();

  std::vector<types::subdomain_id> dofs_domain_association(dh.n_dofs());

  DoFTools::get_subdomain_association(dh, dofs_domain_association);
  // here we prepare the constraint matrix so as to account for the presence of
  // double and triple dofs

  // we start looping on the dofs
  for (types::global_dof_index i = 0; i < tmp_rhs.size(); i++)
    {
      // if (this_cpu_set.is_element(i))
      // {
      // in the next line we compute the "first" among the set of double nodes:
      // this node is the first dirichlet node in the set, and if no dirichlet
      // node is there, we get the first neumann node

      std::set<types::global_dof_index> doubles        = double_nodes_set[i];
      types::global_dof_index           firstOfDoubles = *doubles.begin();
      for (std::set<types::global_dof_index>::iterator it = doubles.begin();
           it != doubles.end();
           it++)
        {
          // if(this_cpu_set.is_element(*it))
          if (localized_dirichlet_nodes(*it) == 1)
            {
              firstOfDoubles = *it;
              break;
            }
        }

      // for each set of double nodes, we will perform the correction only once,
      // and precisely when the current node is the first of the set
      if (i == firstOfDoubles)
        {
          // the vector entry corresponding to the first node of the set does
          // not need modification, thus we erase ti form the set
          doubles.erase(i);

          // if the current (first) node is a dirichlet node, for all its
          // neumann doubles we will impose that the potential is equal to that
          // of the first node: this means that in the matrix vector product we
          // will put the potential value of the double node
          if (localized_dirichlet_nodes(i) == 1)
            {
              for (std::set<types::global_dof_index>::iterator it =
                     doubles.begin();
                   it != doubles.end();
                   it++)
                {
                  // if(this_cpu_set.is_element(*it))
                  {
                    if (localized_dirichlet_nodes(*it) == 1)
                      {
                        // this is the dirichlet-dirichlet case on flat edges:
                        // here we impose that dphi_dn on the two (or more)
                        // sides is equal.
                        double normal_alignment = 0;

                        // types::global_dof_index owner_el_1 =
                        // DoFTools::count_dofs_with_subdomain_association (dh,
                        // dofs_domain_association[i]); types::global_dof_index
                        // owner_el_2 =
                        // DoFTools::count_dofs_with_subdomain_association (dh,
                        // dofs_domain_association[*it]);

                        for (unsigned int idim = 0; idim < dim; ++idim)
                          {
                            types::global_dof_index dummy_1 =
                              sub_wise_to_original[i];
                            types::global_dof_index dummy_2 =
                              sub_wise_to_original[*it];
                            types::global_dof_index index1 = vec_original_to_sub_wise
                              [gradient_dh.n_dofs() / dim * idim +
                               dummy_1]; // vector_start_per_process[dofs_domain_association[i]]
                                         // + idim * owner_el_1 + (i -
                                         // start_per_process[dofs_domain_association[i]]);
                                         // //gradient_dh.n_dofs()/dim*idim+i;//vector_start_per_process[this_mpi_process]
                                         // + (i -
                                         // start_per_process[this_mpi_process])
                                         // * dim + idim; //i*dim+idim
                            types::global_dof_index index2 = vec_original_to_sub_wise
                              [gradient_dh.n_dofs() / dim * idim +
                               dummy_2]; // vector_start_per_process[dofs_domain_association[*it]]
                                         // + idim * owner_el_2 + ((*it) -
                                         // start_per_process[dofs_domain_association[*it]]);//gradient_dh.n_dofs()/dim*idim+(*it);
                                         // //vector_start_per_process[this_mpi_process]
                                         // + ((*it) -
                                         // start_per_process[this_mpi_process])
                                         // * dim + idim;//(*it)*dim+idim
                            normal_alignment += localized_normals[index1] *
                                                localized_normals[index2];
                          }
                        if (std::abs(1.0 - normal_alignment) < 1e-4)
                          {
                            c.add_line(*it);
                            c.add_entry(*it, i, 1);
                          }
                        // this is the dirichlet-dirichlet case on sharp edges:
                        // both normal gradients can be computed from surface
                        // gradients of phi and assingned as BC
                        else if (continuos_gradient)
                          {
                            c.add_line(*it);
                            double norm_i_norm_it = 0;
                            double surf_it_norm_i = 0;
                            double surf_i_norm_it = 0;

                            // types::global_dof_index owner_el_1 =
                            // DoFTools::count_dofs_with_subdomain_association
                            // (dh, dofs_domain_association[i]);
                            // types::global_dof_index owner_el_2 =
                            // DoFTools::count_dofs_with_subdomain_association
                            // (dh, dofs_domain_association[*it]);

                            // We no longer have a std::vector of Point<dim> so
                            // we need to perform the scalar product
                            for (unsigned int idim = 0; idim < dim; ++idim)
                              {
                                types::global_dof_index dummy_1 =
                                  sub_wise_to_original[i];
                                types::global_dof_index dummy_2 =
                                  sub_wise_to_original[*it];

                                types::global_dof_index index1 =
                                  vec_original_to_sub_wise
                                    [gradient_dh.n_dofs() / dim * idim +
                                     dummy_1]; // vector_start_per_process[dofs_domain_association[i]]
                                               // + idim * owner_el_1 + (i -
                                               // start_per_process[dofs_domain_association[i]]);//gradient_dh.n_dofs()/dim*idim+i;//vector_start_per_process[this_mpi_process]
                                               // + (i -
                                               // start_per_process[this_mpi_process])
                                               // * dim + idim;
                                types::global_dof_index index2 =
                                  vec_original_to_sub_wise
                                    [gradient_dh.n_dofs() / dim * idim +
                                     dummy_2]; // vector_start_per_process[dofs_domain_association[*it]]
                                               // + idim * owner_el_2 + ((*it) -
                                               // start_per_process[dofs_domain_association[*it]]);//gradient_dh.n_dofs()/dim*idim+(*it);//vector_start_per_process[this_mpi_process]
                                               // + ((*it) -
                                               // start_per_process[this_mpi_process])
                                               // * dim + idim;
                                norm_i_norm_it += localized_normals[index1] *
                                                  localized_normals[index2];
                                surf_it_norm_i +=
                                  localized_surface_gradients[index2] *
                                  localized_normals[index1];
                                surf_i_norm_it +=
                                  localized_surface_gradients[index1] *
                                  localized_normals[index2];
                              }
                            double this_normal_gradient =
                              (1.0 / (1.0 - norm_i_norm_it * norm_i_norm_it)) *
                              (surf_it_norm_i +
                               (surf_i_norm_it) * (norm_i_norm_it));
                            double other_normal_gradient =
                              (1.0 / (1.0 - norm_i_norm_it * norm_i_norm_it)) *
                              (surf_i_norm_it +
                               (surf_it_norm_i) * (norm_i_norm_it));
                            // std::cout<<"i="<<i<<" j="<<*it<<std::endl;
                            // std::cout<<"ni=("<<node_normals[i]<<")
                            // nj=("<<node_normals[*it]<<")"<<std::endl;
                            // std::cout<<"grad_s_phi_i=("<<node_surface_gradients[i]<<")
                            // grad_s_phi_j=("<<node_surface_gradients[*it]<<")"<<std::endl;
                            // std::cout<<"dphi_dn_i="<<this_normal_gradient<<"
                            // dphi_dn_j="<<other_normal_gradient<<std::endl;
                            // Point<3> this_full_gradient =
                            // node_normals[i]*this_normal_gradient +
                            // node_surface_gradients[i]; Point<3>
                            // other_full_gradient =
                            // node_normals[*it]*other_normal_gradient +
                            // node_surface_gradients[*it];
                            // std::cout<<"grad_phi_i=("<<this_full_gradient<<")
                            // grad_phi_j=("<<other_full_gradient<<")"<<std::endl;
                            c.add_line(i);
                            c.set_inhomogeneity(i, this_normal_gradient);
                            c.add_line(*it);
                            c.set_inhomogeneity(*it, other_normal_gradient);
                          }
                      }
                    else
                      {
                        c.add_line(*it);
                        c.set_inhomogeneity(*it, loc_tmp_rhs(i));
                        // dst(*it) = phi(*it)/alpha(*it);
                      }
                  }
                }
            }

          // if the current (first) node is a neumann node, for all its doubles
          // we will impose that the potential is equal to that of the first
          // node: this means that in the matrix vector product we will put the
          // difference between the potential at the fist node in the doubles
          // set, and the current double node
          if (localized_dirichlet_nodes(i) == 0)
            {
              for (std::set<types::global_dof_index>::iterator it =
                     doubles.begin();
                   it != doubles.end();
                   it++)
                {
                  c.add_line(*it);
                  c.add_entry(*it, i, 1);
                  // dst(*it) = phi(*it)/alpha(*it)-phi(i)/alpha(i);
                }
            }
        }
      // else if(firstOfDoubles == *doubles.begin())
      // {
      //   for(std::set<types::global_dof_index>::iterator it = doubles.begin()
      //   ; it != doubles.end(); it++ )
      //     if(*it!=firstOfDoubles)
      //       {
      //         c.add_line(*it);
      //         c.add_entry(*it,firstOfDoubles,1);
      //       }
      // }
      // }
    }

  c.merge(c_hn);
  c.close();

  c_cpu_set.clear();
  c_cpu_set.set_size(this_cpu_set.size());
  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    {
      if (this_cpu_set.is_element(i))
        {
          c_cpu_set.add_index(i);
          if (c.is_constrained(i))
            {
              const std::vector<std::pair<types::global_dof_index, double>>
                *entries = c.get_constraint_entries(i);
              for (types::global_dof_index j = 0; j < entries->size(); ++j)
                c_cpu_set.add_index((*entries)[j].first);
            }
        }
    }
  c_cpu_set.compress();

  /*
  pcout<<"CONSTAINT MATRIX CHECK "<<std::endl;
    for (types::global_dof_index i=0; i<dh.n_dofs(); ++i)
        {
        std::set <types::global_dof_index> duplicates = double_nodes_set[i];
        if (duplicates.size()>1)
           {
           pcout<<"Proc: "<<this_mpi_process<<" i= "<<i<<"
  ("<<localized_dirichlet_nodes(i)<<") duplicates: "; for
  (std::set<types::global_dof_index>::iterator pos = duplicates.begin(); pos
  !=duplicates.end(); pos++) pcout<<" "<<*pos; pcout<<std::endl;
           }
        }

    for(unsigned int i=0; i<dh.n_dofs(); ++i)
      if( (constraints.is_constrained(i)) )
        {pcout<<"Proc: "<<this_mpi_process<<" i= "<<i<<" (";
    const std::vector< std::pair < types::global_dof_index, double > >
      * entries = constraints.get_constraint_entries (i);
          pcout<<entries->size()<<")  Entries:";
    for(unsigned int j=0; j< entries->size(); ++j)
       pcout<<" "<<(*entries)[j].first<<" ("<<(*entries)[j].second<<") ";
         pcout<<" Inomogeneities:
  "<<constraints.get_inhomogeneity(i)<<std::endl;
        }
  */
}

template <int dim>
void
BEMProblem<dim>::assemble_preconditioner()
{
  if (is_preconditioner_initialized == false)
    {
      // pcout<<"Initialising preconditioner"<<std::endl;
      for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
        if (this_cpu_set.is_element(i))
          {
            // types::global_dof_index start_helper, end_helper;
            // if(i>preconditioner_band/2)
            //   start_helper = i-preconditioner_band/2;
            // else
            //   start_helper = (types::global_dof_index) 0;
            // if(i+preconditioner_band/2 < dh.n_dofs())
            //   end_helper = i+preconditioner_band/2;
            // else
            //   end_helper = dh.n_dofs();
            //   for(types::global_dof_index j=start_helper; j<end_helper; ++j)
            // pcout<<start_helper<<"
            // "<<std::min((types::global_dof_index)(i+preconditioner_band/2),(types::global_dof_index)dh.n_dofs())<<std::endl;
            types::global_dof_index start_helper =
              ((i) > preconditioner_band / 2) ? (i - preconditioner_band / 2) :
                                                ((types::global_dof_index)0);
            for (types::global_dof_index j = start_helper;
                 j < std::min((types::global_dof_index)(
                                i + preconditioner_band / 2),
                              (types::global_dof_index)dh.n_dofs());
                 ++j)
              preconditioner_sparsity_pattern.add(i, j);
          }
      preconditioner_sparsity_pattern.compress();
      band_system.reinit(preconditioner_sparsity_pattern);
      is_preconditioner_initialized = true;
    }
  else
    band_system = 0;


  for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i)
    {
      if (this_cpu_set.is_element(i))
        {
          if (constraints.is_constrained(i))
            band_system.add(i, i, 1);
          // types::global_dof_index start_helper, end_helper;
          // if(i>preconditioner_band/2)
          //   start_helper = i-preconditioner_band/2;
          // else
          //   start_helper = (types::global_dof_index) 0;
          // if(i+preconditioner_band/2 < dh.n_dofs())
          //   end_helper = i+preconditioner_band/2;
          // else
          //   end_helper = dh.n_dofs();
          // for(types::global_dof_index j=start_helper; j<end_helper; ++j)
          types::global_dof_index start_helper =
            ((i) > preconditioner_band / 2) ? (i - preconditioner_band / 2) :
                                              ((types::global_dof_index)0);

          for (types::global_dof_index j = start_helper;
               j <
               std::min((types::global_dof_index)i + preconditioner_band / 2,
                        (types::global_dof_index)dh.n_dofs());
               ++j)
            {
              if (constraints.is_constrained(i) == false)
                {
                  if (dirichlet_nodes(i) == 0)
                    {
                      // Nodo di Dirichlet
                      band_system.add(i, j, neumann_matrix(i, j));

                      if (boundary_integral_formulation == "galerkin")
                        band_system.add(i, j, 0.5 * jump_matrix(i, j));
                      else if (i == j)
                        band_system.add(i, j, alpha(i));
                    }
                  else
                    band_system.add(i, j, -dirichlet_matrix(i, j));
                }
            }
        }
    }



  preconditioner.initialize(band_system);

  /*
  band_system.vmult(sol,alpha);
  pcout<<"**solution "<<std::endl;
   for (unsigned int i = 0; i < alpha.size(); i++)
       if (this_cpu_set.is_element(i))
          pcout<<i<<" ("<<this_mpi_process<<")  "<<sol(i)<<"
  "<<sol(i)<<std::endl;
  */
}



template <int dim>
double
BEMProblem<dim>::compute_energy_norm_error(
  const TrilinosWrappers::MPI::Vector &dphi_dn,
  const Function<dim>                 &exact_gradient) const
{
  AssertThrow(boundary_integral_formulation == "galerkin",
              ExcMessage("The energy norm is defined here for the Galerkin "
                         "single-layer formulation."));
  if constexpr (dim == 3)
    {
      const GalerkinSingularQuadrature singular_strategy =
        galerkin_singular_quadrature == "iterated_duffy" ?
          GalerkinSingularQuadrature::iterated_duffy :
          GalerkinSingularQuadrature::sauter_schwab;
      GalerkinPanelPairIntegrator<3> integrator(
        *mapping,
        *fe,
        singular_strategy,
        galerkin_singular_quadrature_order,
        galerkin_duffy_outer_subdivisions,
        galerkin_near_singular_distance_ratio,
        galerkin_near_singular_maximum_depth,
        galerkin_near_singular_quadrature_order);
      integrator.set_screening(galerkin_screening());
      integrator.set_light_geometry(galerkin_light_geometry);
      const double kappa = galerkin_screening();

      // Per-cell quadrature data, computed once. The overwhelming majority of
      // pairs are far apart and regular, and for those the double integral is
      // a plain tensor sum over this cached data; only the singular and
      // near-singular pairs need the panel-pair machinery. This mirrors how
      // the operator assembly is organised, and is the difference between the
      // energy norm costing about one assembly and costing many.
      const QGauss<dim - 1>  quadrature(galerkin_quadrature_order);
      FEValues<dim - 1, dim> values(*mapping,
                                    *fe,
                                    quadrature,
                                    update_values | update_normal_vectors |
                                      update_quadrature_points |
                                      update_JxW_values);
      const Vector<double>                 localized(dphi_dn);
      std::vector<types::global_dof_index> indices(fe->dofs_per_cell);

      std::vector<cell_it>                      cells;
      std::vector<std::vector<double>>          coefficients;
      std::vector<std::vector<Point<dim>>>      points;
      std::vector<std::vector<double>>          jxw;
      std::vector<std::vector<double>>          error_values;
      std::vector<std::vector<double>>          discrete_values;
      Vector<double>                            gradient(dim);
      for (const auto &cell : dh.active_cell_iterators())
        {
          values.reinit(cell);
          cell->get_dof_indices(indices);
          std::vector<double> local(fe->dofs_per_cell);
          for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
            local[i] = localized[indices[i]];

          std::vector<double>     cell_error(quadrature.size());
          std::vector<double>     cell_discrete(quadrature.size());
          std::vector<double>     cell_jxw(quadrature.size());
          std::vector<Point<dim>> cell_points(quadrature.size());
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              cell_points[q] = values.quadrature_point(q);
              cell_jxw[q]    = values.JxW(q);
              exact_gradient.vector_value(cell_points[q], gradient);
              Tensor<1, dim> normal = values.normal_vector(q);
              const double   norm   = normal.norm();
              AssertThrow(norm > 1e-14,
                          ExcMessage("Cannot evaluate the exact flux with a "
                                     "zero normal."));
              normal /= norm;
              double exact = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                exact += gradient[d] * normal[d];
              double discrete = 0.0;
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                discrete += local[i] * values.shape_value(i, q);
              cell_error[q]    = exact - discrete;
              cell_discrete[q] = discrete;
            }

          cells.push_back(cell);
          coefficients.push_back(std::move(local));
          points.push_back(std::move(cell_points));
          jxw.push_back(std::move(cell_jxw));
          error_values.push_back(std::move(cell_error));
          discrete_values.push_back(std::move(cell_discrete));
        }

      // The form is symmetric, so only the upper triangle is visited and the
      // off-diagonal contributions are doubled. Summed over ranks this covers
      // every ordered pair exactly once. The field is a parameter so that the
      // same machinery can be exercised on a known quantity: with a zero
      // exact flux the result must be (V lambda_h, lambda_h), which the
      // assembled matrix gives independently.
      const unsigned int n_q = quadrature.size();
      // The three classes of pair are accumulated separately: a negative
      // total has to come from one of them, and knowing which one is the
      // difference between guessing and knowing.
      const auto energy_of =
        [&](const Function<dim>                    &field,
            const std::vector<std::vector<double>> &values_at_q,
            double                                 &diagonal,
            double                                 &far,
            double                                 &near) {
          double local_diagonal = 0.0;
          double local_far      = 0.0;
          double local_near     = 0.0;
          for (std::size_t i = 0; i < cells.size(); ++i)
            {
              if (cells[i]->subdomain_id() != this_mpi_process)
                continue;
              local_diagonal += integrator.integrate_error_energy(
                cells[i], cells[i], galerkin_quadrature_order, field,
                coefficients[i], coefficients[i]);
              for (std::size_t j = i + 1; j < cells.size(); ++j)
                {
                  if (integrator.is_far_regular_pair(cells[i], cells[j]))
                    {
                      double pair_sum = 0.0;
                      for (unsigned int qx = 0; qx < n_q; ++qx)
                        {
                          const double left =
                            values_at_q[i][qx] * jxw[i][qx];
                          if (left == 0.0)
                            continue;
                          for (unsigned int qy = 0; qy < n_q; ++qy)
                            {
                              const double distance =
                                points[i][qx].distance(points[j][qy]);
                              pair_sum +=
                                left * values_at_q[j][qy] * jxw[j][qy] *
                                (kappa == 0.0 ? 1.0 :
                                                std::exp(-kappa * distance)) /
                                (4.0 * numbers::PI * distance);
                            }
                        }
                      local_far += 2.0 * pair_sum;
                    }
                  else
                    local_near += 2.0 * integrator.integrate_error_energy(
                      cells[i], cells[j], galerkin_quadrature_order, field,
                      coefficients[i], coefficients[j]);
                }
            }
          diagonal = Utilities::MPI::sum(local_diagonal, mpi_communicator);
          far      = Utilities::MPI::sum(local_far, mpi_communicator);
          near     = Utilities::MPI::sum(local_near, mpi_communicator);
          return diagonal + far + near;
        };

      double       diagonal_part = 0.0;
      double       far_part      = 0.0;
      double       near_part     = 0.0;
      const double total         = energy_of(
        exact_gradient, error_values, diagonal_part, far_part, near_part);
      if (std::getenv("PIBEM_ENERGY_SELFTEST") != nullptr)
        pcout << "Energy breakdown: same-panel " << diagonal_part
              << ", far regular " << far_part << ", near and singular "
              << near_part << ", total " << total << std::endl;

      if (std::getenv("PIBEM_ENERGY_SELFTEST") != nullptr)
        {
          Functions::ZeroFunction<dim> zero(dim);
          std::vector<std::vector<double>> minus_discrete(discrete_values);
          for (auto &cell_values : minus_discrete)
            for (double &v : cell_values)
              v = -v;
          double       zero_diagonal = 0.0;
          double       zero_far      = 0.0;
          double       zero_near     = 0.0;
          const double by_quadrature = energy_of(
            zero, minus_discrete, zero_diagonal, zero_far, zero_near);
          TrilinosWrappers::MPI::Vector image(this_cpu_set, mpi_communicator);
          dirichlet_matrix.vmult(image, dphi_dn);
          const double by_matrix = image * dphi_dn;
          pcout << "Energy self-test: (V lambda_h, lambda_h) by quadrature "
                << by_quadrature << ", by matrix " << by_matrix
                << ", relative difference "
                << std::abs(by_quadrature - by_matrix) /
                     std::max(std::abs(by_matrix), 1e-30)
                << std::endl
                << "Energy of the error, raw sum: " << total << std::endl;
        }

      AssertThrow(total > -1e-10 * std::max(1.0, std::abs(total)),
                  ExcMessage("The energy of the flux error came out negative, "
                             "which the single-layer operator forbids."));
      return std::sqrt(std::max(total, 0.0));
    }
  else
    {
      (void)dphi_dn;
      (void)exact_gradient;
      AssertThrow(false, ExcMessage("Energy norm supports 3D only."));
      return 0.0;
    }
}


template <int dim>
void
BEMProblem<dim>::compute_gradients(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn)
{
  Teuchos::TimeMonitor LocalTimer(*GradientTime);

  // We need the solution to be stored on a parallel vector with ghost elements.
  // We let Trilinos take care of it.

  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi = glob_phi;
  TrilinosWrappers::MPI::Vector dphi_dn(ghosted_set);
  dphi_dn = glob_dphi_dn;



  // We reinit the gradient solution
  vector_gradients_solution.reinit(vector_this_cpu_set, mpi_communicator);

  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;


  // The matrix and rhs of our problem. We must decide if compute the mass
  // matrix just once and for all or not.
  TrilinosWrappers::SparseMatrix vector_gradients_matrix;
  TrilinosWrappers::MPI::Vector  vector_gradients_rhs(vector_this_cpu_set,
                                                     mpi_communicator);
  vector_gradients_matrix.reinit(vector_sparsity_pattern);


  // The vector FEValues to used in the assemblage
  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  // The scalar FEValues to interpolate the known value of phi
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_gradients |
                                update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  const unsigned int vector_n_q_points    = vector_fe_v.n_quadrature_points;
  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;
  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);


  std::vector<Tensor<1, dim>> phi_surf_grads(vector_n_q_points);
  std::vector<double>         phi_norm_grads(vector_n_q_points);
  std::vector<Vector<double>> q_vector_normals_solution(vector_n_q_points,
                                                        Vector<double>(dim));

  FullMatrix<double> local_gradients_matrix(vector_dofs_per_cell,
                                            vector_dofs_per_cell);
  Vector<double>     local_gradients_rhs(vector_dofs_per_cell);



  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  Quadrature<dim - 1>    dummy_quadrature(fe->get_unit_support_points());
  FEValues<dim - 1, dim> dummy_fe_v(*mapping,
                                    *fe,
                                    dummy_quadrature,
                                    update_values | update_gradients |
                                      update_normal_vectors |
                                      update_quadrature_points);

  const unsigned int                   dofs_per_cell = fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const unsigned int          n_q_points = dummy_fe_v.n_quadrature_points;
  std::vector<Tensor<1, dim>> dummy_phi_surf_grads(n_q_points);

  cell_it vector_cell = gradient_dh.begin_active();

  cell_it cell = dh.begin_active(), endc = dh.end();


  for (; cell != endc; ++cell, ++vector_cell)
    {
      Assert(cell->index() == vector_cell->index(), ExcInternalError());
      Assert(cell->subdomain_id() == vector_cell->subdomain_id(),
             ExcInternalError());

      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_v.reinit(cell);
          vector_fe_v.reinit(vector_cell);
          local_gradients_matrix = 0;
          local_gradients_rhs    = 0;
          const std::vector<Tensor<1, dim>> &vector_node_normals =
            vector_fe_v.get_normal_vectors();
          fe_v.get_function_gradients(phi, phi_surf_grads);
          fe_v.get_function_values(dphi_dn, phi_norm_grads);
          unsigned int comp_i, comp_j;



          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            {
              Tensor<1, dim> node_normal_grad_dir;
              for (unsigned int i = 0; i < dim; ++i)
                node_normal_grad_dir[i] = q_vector_normals_solution[q][i];
              Tensor<1, dim> gradient =
                vector_node_normals[q] * phi_norm_grads[q] + phi_surf_grads[q];
              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  comp_i = gradient_fe->system_to_component_index(i).first;
                  for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
                    {
                      comp_j = gradient_fe->system_to_component_index(j).first;
                      if (comp_i == comp_j)
                        {
                          local_gradients_matrix(i, j) +=
                            vector_fe_v.shape_value(i, q) *
                            vector_fe_v.shape_value(j, q) * vector_fe_v.JxW(q);
                        }
                    }
                  local_gradients_rhs(i) += (vector_fe_v.shape_value(i, q)) *
                                            gradient[comp_i] *
                                            vector_fe_v.JxW(q);
                }
            }
          vector_cell->get_dof_indices(vector_local_dof_indices);

          vector_constraints.distribute_local_to_global(
            local_gradients_matrix,
            local_gradients_rhs,
            vector_local_dof_indices,
            vector_gradients_matrix,
            vector_gradients_rhs);
        }
    }

  // At this point we can compress anything and solve via GMRES.
  vector_gradients_matrix.compress(VectorOperation::add);
  vector_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));

  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_gradients_matrix);
  solver.solve(vector_gradients_matrix,
               vector_gradients_solution,
               vector_gradients_rhs,
               mass_prec);

  vector_constraints.distribute(vector_gradients_solution);
}

template <int dim>
void
BEMProblem<dim>::compute_surface_gradients(
  const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  Teuchos::TimeMonitor          LocalTimer(*SurfaceGradientTime);
  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi = tmp_rhs;

  vector_surface_gradients_solution.reinit(vector_this_cpu_set,
                                           mpi_communicator);


  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;


  TrilinosWrappers::SparseMatrix vector_surface_gradients_matrix;
  TrilinosWrappers::MPI::Vector  vector_surface_gradients_rhs(
    vector_this_cpu_set, mpi_communicator);


  vector_surface_gradients_matrix.reinit(vector_sparsity_pattern);



  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_gradients |
                                update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  const unsigned int vector_n_q_points    = vector_fe_v.n_quadrature_points;
  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;
  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);

  std::vector<Tensor<1, dim>> phi_surf_grads(vector_n_q_points);
  std::vector<double>         phi_norm_grads(vector_n_q_points);
  std::vector<Vector<double>> q_vector_normals_solution(vector_n_q_points,
                                                        Vector<double>(dim));

  FullMatrix<double> local_gradients_matrix(vector_dofs_per_cell,
                                            vector_dofs_per_cell);
  Vector<double>     local_gradients_rhs(vector_dofs_per_cell);



  std::vector<Point<dim>> support_points(dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*mapping,
                                                     dh,
                                                     support_points);
  std::vector<types::global_dof_index> face_dofs(fe->dofs_per_face);

  Quadrature<dim - 1>    dummy_quadrature(fe->get_unit_support_points());
  FEValues<dim - 1, dim> dummy_fe_v(*mapping,
                                    *fe,
                                    dummy_quadrature,
                                    update_values | update_gradients |
                                      update_normal_vectors |
                                      update_quadrature_points);

  const unsigned int                   dofs_per_cell = fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const unsigned int          n_q_points = dummy_fe_v.n_quadrature_points;
  std::vector<Tensor<1, dim>> dummy_phi_surf_grads(n_q_points);

  cell_it vector_cell = gradient_dh.begin_active();

  cell_it cell = dh.begin_active(), endc = dh.end();


  for (; cell != endc; ++cell, ++vector_cell)
    {
      Assert(cell->index() == vector_cell->index(), ExcInternalError());
      Assert(cell->subdomain_id() == vector_cell->subdomain_id(),
             ExcInternalError());

      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_v.reinit(cell);
          vector_fe_v.reinit(vector_cell);
          local_gradients_matrix = 0;
          local_gradients_rhs    = 0;
          fe_v.get_function_gradients(phi, phi_surf_grads);
          unsigned int comp_i, comp_j;



          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            {
              Tensor<1, dim> gradient = phi_surf_grads[q];
              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  comp_i = gradient_fe->system_to_component_index(i).first;
                  for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
                    {
                      comp_j = gradient_fe->system_to_component_index(j).first;
                      if (comp_i == comp_j)
                        {
                          local_gradients_matrix(i, j) +=
                            vector_fe_v.shape_value(i, q) *
                            vector_fe_v.shape_value(j, q) * vector_fe_v.JxW(q);
                        }
                    }
                  local_gradients_rhs(i) += (vector_fe_v.shape_value(i, q)) *
                                            gradient[comp_i] *
                                            vector_fe_v.JxW(q);
                }
            }
          vector_cell->get_dof_indices(vector_local_dof_indices);

          vector_constraints.distribute_local_to_global(
            local_gradients_matrix,
            local_gradients_rhs,
            vector_local_dof_indices,
            vector_surface_gradients_matrix,
            vector_surface_gradients_rhs);
        }
    }

  vector_surface_gradients_matrix.compress(VectorOperation::add);
  vector_surface_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));

  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_surface_gradients_matrix);

  solver.solve(vector_surface_gradients_matrix,
               vector_surface_gradients_solution,
               vector_surface_gradients_rhs,
               mass_prec);

  vector_constraints.distribute(vector_surface_gradients_solution);
}


template <int dim>
void
BEMProblem<dim>::compute_gradients_weak(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<dim>                 *analytic_phi,
  const bool                           separate_normal)
{
  Teuchos::TimeMonitor LocalTimer(*GradientTime);
  pcout << "Computing gradients in weak form"
        << (analytic_phi != nullptr ? " from the analytical trace" : "")
        << std::endl;

  // The surface gradient of the trace is recovered without ever
  // differentiating the discrete trace. Integrating by parts on each cell,
  //
  //     \int_K v . grad_S phi = - \int_K phi div_S v + \int_dK phi (v . m),
  //
  // moves the derivative onto the test function, which is a polynomial and is
  // differentiated exactly, and leaves phi undifferentiated. Differentiating
  // an interpolant of degree p costs an order; this does not. The identity is
  // exact cell by cell, so no neighbour is consulted and hanging nodes need no
  // special treatment. When the trace is prescribed data, as it is for a pure
  // Dirichlet problem, phi can then be taken from the analytical function
  // rather than from its interpolant, and the tangential part of the gradient
  // becomes as accurate as the quadrature.
  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi = glob_phi;
  TrilinosWrappers::MPI::Vector dphi_dn(ghosted_set);
  dphi_dn = glob_dphi_dn;

  vector_gradients_solution.reinit(vector_this_cpu_set, mpi_communicator);

  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;

  TrilinosWrappers::SparseMatrix vector_gradients_matrix;
  TrilinosWrappers::MPI::Vector  vector_gradients_rhs(vector_this_cpu_set,
                                                     mpi_communicator);
  vector_gradients_matrix.reinit(vector_sparsity_pattern);

  FEValues<dim - 1, dim> vector_fe_v(*mapping,
                                     *gradient_fe,
                                     *quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
  FEValues<dim - 1, dim> fe_v(*mapping,
                              *fe,
                              *quadrature,
                              update_values | update_quadrature_points |
                                update_JxW_values);

  // Faces of a surface cell are curves in space. The co-normal m is the unit
  // vector tangent to the surface, orthogonal to the face and pointing out of
  // the cell; it is built here from the surface normal and the face tangent
  // rather than taken from the library, and its orientation is fixed by
  // testing it against the direction from the cell centre.
  const QGauss<dim - 2>      face_quadrature(quadrature->size());
  FEFaceValues<dim - 1, dim> vector_fe_face(*mapping,
                                            *gradient_fe,
                                            face_quadrature,
                                            update_values |
                                              update_quadrature_points |
                                              update_JxW_values);
  FEFaceValues<dim - 1, dim> fe_face(*mapping,
                                     *fe,
                                     face_quadrature,
                                     update_values |
                                       update_quadrature_points |
                                       update_JxW_values);
  FEFaceValues<dim - 1, dim> normal_face(*mapping,
                                         *fe,
                                         face_quadrature,
                                         update_normal_vectors |
                                           update_quadrature_points);

  const unsigned int vector_n_q_points    = vector_fe_v.n_quadrature_points;
  const unsigned int vector_dofs_per_cell = gradient_fe->dofs_per_cell;
  std::vector<types::global_dof_index> vector_local_dof_indices(
    vector_dofs_per_cell);

  std::vector<double> flux_values(vector_n_q_points);
  std::vector<double> trace_values(vector_n_q_points);
  std::vector<double> face_trace_values(face_quadrature.size());

  // Integrating by parts on a curved surface leaves a curvature term,
  //     \int_K phi (div_S n)(v . n),
  // which vanishes only where the surface is flat. It is evaluated from the
  // projected normal field, treated like any other finite element field.
  const Vector<double>            localized_normals(vector_normals_solution);
  const FEValuesExtractors::Vector all_components(0);
  std::vector<double>              total_curvature(vector_n_q_points);

  FullMatrix<double> local_matrix(vector_dofs_per_cell, vector_dofs_per_cell);
  Vector<double>     local_rhs(vector_dofs_per_cell);

  // Total curvature div_S n of the mapped surface, from its geometry rather
  // than from a projected normal field. For every quadrature point the exact
  // normal is also evaluated at four nearby reference points, and
  //     div_S n = sum_{a,b} G^{-1}_{ab} (dn/dxi_a . a_b),
  // with a_b the columns of the mapping Jacobian and G = A^T A the metric.
  // Central differences with this step are accurate to about 1e-10.
  const double         fd_step = 1e-5;
  std::vector<Point<dim - 1>> fd_points;
  for (unsigned int q = 0; q < vector_n_q_points; ++q)
    {
      const Point<dim - 1> x = quadrature->point(q);
      fd_points.push_back(x);
      for (unsigned int a = 0; a < dim - 1; ++a)
        {
          Point<dim - 1> plus = x, minus = x;
          plus[a] += fd_step;
          minus[a] -= fd_step;
          fd_points.push_back(plus);
          fd_points.push_back(minus);
        }
    }
  const unsigned int     fd_stride = 1 + 2 * (dim - 1);
  const Quadrature<dim - 1> fd_quadrature(fd_points);
  FEValues<dim - 1, dim> fd_values(*mapping,
                                   *fe,
                                   fd_quadrature,
                                   update_normal_vectors | update_jacobians);
  const auto geometric_curvature = [&](const cell_it &c,
                                       std::vector<double> &kappa) {
    fd_values.reinit(c);
    for (unsigned int q = 0; q < vector_n_q_points; ++q)
      {
        const unsigned int base = q * fd_stride;
        const auto        &jac  = fd_values.jacobian(base);
        Tensor<2, dim - 1> metric;
        for (unsigned int a = 0; a < dim - 1; ++a)
          for (unsigned int c2 = 0; c2 < dim - 1; ++c2)
            for (unsigned int d = 0; d < dim; ++d)
              metric[a][c2] += jac[d][a] * jac[d][c2];
        const Tensor<2, dim - 1> inverse_metric = invert(metric);
        double                   div            = 0.0;
        for (unsigned int a = 0; a < dim - 1; ++a)
          {
            Tensor<1, dim> n_plus = fd_values.normal_vector(base + 1 + 2 * a);
            Tensor<1, dim> n_minus =
              fd_values.normal_vector(base + 2 + 2 * a);
            n_plus /= n_plus.norm();
            n_minus /= n_minus.norm();
            const Tensor<1, dim> dn = (n_plus - n_minus) / (2.0 * fd_step);
            for (unsigned int c2 = 0; c2 < dim - 1; ++c2)
              {
                double dn_dot_a = 0.0;
                for (unsigned int d = 0; d < dim; ++d)
                  dn_dot_a += dn[d] * jac[d][c2];
                div += inverse_metric[a][c2] * dn_dot_a;
              }
          }
        kappa[q] = div;
      }
  };
  double curvature_check_defect = 0.0;
  const bool check_curvature =
    std::getenv("PIBEM_WEAK_GRADIENT_CHECK") != nullptr;

  cell_it vector_cell = gradient_dh.begin_active();
  cell_it cell = dh.begin_active(), endc = dh.end();

  for (; cell != endc; ++cell, ++vector_cell)
    {
      if (cell->subdomain_id() != this_mpi_process)
        continue;

      fe_v.reinit(cell);
      vector_fe_v.reinit(vector_cell);
      local_matrix = 0;
      local_rhs    = 0;

      const std::vector<Tensor<1, dim>> &normals =
        vector_fe_v.get_normal_vectors();
      fe_v.get_function_values(dphi_dn, flux_values);
      if (separate_normal)
        geometric_curvature(cell, total_curvature);
      else
        vector_fe_v[all_components].get_function_divergences(
          localized_normals, total_curvature);
      if (analytic_phi != nullptr)
        for (unsigned int q = 0; q < vector_n_q_points; ++q)
          trace_values[q] = analytic_phi->value(fe_v.quadrature_point(q));
      else
        fe_v.get_function_values(phi, trace_values);

      for (unsigned int q = 0; q < vector_n_q_points; ++q)
        for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
          {
            const unsigned int ci =
              gradient_fe->system_to_component_index(i).first;
            for (unsigned int j = 0; j < vector_dofs_per_cell; ++j)
              if (ci == gradient_fe->system_to_component_index(j).first)
                local_matrix(i, j) += vector_fe_v.shape_value(i, q) *
                                      vector_fe_v.shape_value(j, q) *
                                      vector_fe_v.JxW(q);
            if (separate_normal)
              // Only the tangential gradient grad_S phi is projected here,
              // curvature term included, so the projected field is exactly
              // tangential and its projection error has nothing directed
              // along the normal to pass on. The flux is added afterwards.
              local_rhs(i) += (-trace_values[q] *
                                 vector_fe_v.shape_grad(i, q)[ci] +
                               trace_values[q] * total_curvature[q] *
                                 vector_fe_v.shape_value(i, q) *
                                 normals[q][ci]) *
                              vector_fe_v.JxW(q);
            else
              // normal part from the flux, tangential part by parts, plus
              // the curvature term the integration by parts leaves behind
              local_rhs(i) += (vector_fe_v.shape_value(i, q) *
                                 flux_values[q] * normals[q][ci] -
                               trace_values[q] *
                                 vector_fe_v.shape_grad(i, q)[ci] +
                               trace_values[q] * total_curvature[q] *
                                 vector_fe_v.shape_value(i, q) *
                                 normals[q][ci]) *
                              vector_fe_v.JxW(q);
          }

      for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell; ++f)
        {
          vector_fe_face.reinit(vector_cell, f);
          fe_face.reinit(cell, f);
          normal_face.reinit(cell, f);
          if (analytic_phi != nullptr)
            for (unsigned int q = 0; q < face_quadrature.size(); ++q)
              face_trace_values[q] =
                analytic_phi->value(fe_face.quadrature_point(q));
          else
            fe_face.get_function_values(phi, face_trace_values);

          const Point<dim> face_centre = cell->face(f)->center();
          const Tensor<1, dim> outward = face_centre - cell->center();
          const Point<dim>     tail    = cell->face(f)->vertex(0);
          const Point<dim>     head    = cell->face(f)->vertex(1);
          Tensor<1, dim>       tangent = head - tail;
          const double         length  = tangent.norm();
          if (length < 1e-14)
            continue;
          tangent /= length;

          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            {
              // On a codimension-one mesh the face "normal" reported by the
              // library is already the co-normal: tangent to the surface,
              // orthogonal to the face. Only its orientation is fixed here.
              Tensor<1, dim> co_normal = normal_face.normal_vector(q);
              const double   m_length  = co_normal.norm();
              if (m_length < 1e-14)
                continue;
              co_normal /= m_length;
              if (co_normal * outward < 0.0)
                co_normal *= -1.0;

              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  const unsigned int ci =
                    gradient_fe->system_to_component_index(i).first;
                  local_rhs(i) += face_trace_values[q] *
                                  vector_fe_face.shape_value(i, q) *
                                  co_normal[ci] * vector_fe_face.JxW(q);
                }
            }
        }

      if (check_curvature && separate_normal)
        {
          // With phi = 1 the identity reads  -V_i + B_i + C_i = 0, with V the
          // area integral of div_S v, B the boundary integral of v.m and C
          // the curvature term. Its defect measures the curvature itself.
          std::vector<double> balance(vector_dofs_per_cell, 0.0);
          double              scale = 0.0;
          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
              {
                const unsigned int ci =
                  gradient_fe->system_to_component_index(i).first;
                const double v = vector_fe_v.shape_grad(i, q)[ci] *
                                 vector_fe_v.JxW(q);
                balance[i] -= v;
                balance[i] += total_curvature[q] *
                              vector_fe_v.shape_value(i, q) * normals[q][ci] *
                              vector_fe_v.JxW(q);
                scale = std::max(scale, std::abs(v));
              }
          for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell;
               ++f)
            {
              vector_fe_face.reinit(vector_cell, f);
              normal_face.reinit(cell, f);
              const Tensor<1, dim> outward =
                cell->face(f)->center() - cell->center();
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  Tensor<1, dim> m = normal_face.normal_vector(q);
                  if (m.norm() < 1e-14)
                    continue;
                  m /= m.norm();
                  if (m * outward < 0.0)
                    m *= -1.0;
                  for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                    {
                      const unsigned int ci =
                        gradient_fe->system_to_component_index(i).first;
                      balance[i] += vector_fe_face.shape_value(i, q) * m[ci] *
                                    vector_fe_face.JxW(q);
                    }
                }
            }
          for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
            if (scale > 0.0)
              curvature_check_defect =
                std::max(curvature_check_defect, std::abs(balance[i]) / scale);
        }

      vector_cell->get_dof_indices(vector_local_dof_indices);
      vector_constraints.distribute_local_to_global(local_matrix,
                                                    local_rhs,
                                                    vector_local_dof_indices,
                                                    vector_gradients_matrix,
                                                    vector_gradients_rhs);
    }

  // Does the assembled pair of terms satisfy the surface divergence theorem?
  // For a flat cell the boundary integral of v.m and the area integral of
  // div_S v must agree exactly; on a curved cell they differ by the curvature
  // term. Measuring both tells a coding error from a missing term.
  if (std::getenv("PIBEM_WEAK_GRADIENT_CHECK") != nullptr)
    {
      double flat_defect = 0.0, curved_defect = 0.0;
      for (cell_it c = dh.begin_active(), vc = gradient_dh.begin_active();
           c != endc;
           ++c, ++vc)
        {
          if (c->subdomain_id() != this_mpi_process)
            continue;
          vector_fe_v.reinit(vc);
          const std::vector<Tensor<1, dim>> &nn =
            vector_fe_v.get_normal_vectors();
          bool flat = true;
          for (unsigned int q = 1; q < vector_n_q_points; ++q)
            if ((nn[q] - nn[0]).norm() > 1e-8)
              flat = false;
          std::vector<double> volume(vector_dofs_per_cell, 0.0);
          std::vector<double> boundary(vector_dofs_per_cell, 0.0);
          double scale = 0.0;
          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
              {
                const unsigned int ci =
                  gradient_fe->system_to_component_index(i).first;
                volume[i] +=
                  vector_fe_v.shape_grad(i, q)[ci] * vector_fe_v.JxW(q);
                scale = std::max(scale,
                                 std::abs(vector_fe_v.shape_grad(i, q)[ci]) *
                                   vector_fe_v.JxW(q));
              }
          for (unsigned int f = 0;
               f < GeometryInfo<dim - 1>::faces_per_cell;
               ++f)
            {
              vector_fe_face.reinit(vc, f);
              normal_face.reinit(c, f);
              const Tensor<1, dim> outward = c->face(f)->center() - c->center();
              Tensor<1, dim> tangent = c->face(f)->vertex(1) -
                                       c->face(f)->vertex(0);
              if (tangent.norm() < 1e-14)
                continue;
              tangent /= tangent.norm();
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  Tensor<1, dim> m = normal_face.normal_vector(q);
                  if (m.norm() < 1e-14)
                    continue;
                  m /= m.norm();
                  if (m * outward < 0.0)
                    m *= -1.0;
                  for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                    {
                      const unsigned int ci =
                        gradient_fe->system_to_component_index(i).first;
                      boundary[i] += vector_fe_face.shape_value(i, q) * m[ci] *
                                     vector_fe_face.JxW(q);
                    }
                }
            }
          double worst = 0.0;
          for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
            worst = std::max(worst, std::abs(boundary[i] - volume[i]));
          if (scale > 0.0)
            (flat ? flat_defect : curved_defect) =
              std::max(flat ? flat_defect : curved_defect, worst / scale);
        }
      pcout << "Weak gradient divergence-theorem defect: flat cells "
            << Utilities::MPI::max(flat_defect, mpi_communicator)
            << ", curved cells "
            << Utilities::MPI::max(curved_defect, mpi_communicator)
            << std::endl;
    }

  vector_gradients_matrix.compress(VectorOperation::add);
  vector_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));
  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_gradients_matrix);
  solver.solve(vector_gradients_matrix,
               vector_gradients_solution,
               vector_gradients_rhs,
               mass_prec);
  vector_constraints.distribute(vector_gradients_solution);

  gradient_normal_from_flux = separate_normal;
  if (separate_normal)
    {
      if (check_curvature)
        pcout << "Weak gradient divergence-theorem defect with geometric "
                 "curvature: "
              << Utilities::MPI::max(curvature_check_defect, mpi_communicator)
              << std::endl;

      // Assemble the full gradient as the L2 projection, at the quadrature
      // points, of  P g_t + lambda_h n,  with n the exact normal of the
      // mapped surface there. Projecting rather than interpolating at the
      // nodes gives the normal component the best approximation of lambda n
      // the space allows, which is what the projection-based recovery gets
      // right and the nodal replacement did not.
      TrilinosWrappers::MPI::Vector tangential_field(vector_gradients_solution);
      const Vector<double>          tangential(tangential_field);
      TrilinosWrappers::MPI::Vector full_rhs(vector_this_cpu_set,
                                             mpi_communicator);
      std::vector<Tensor<1, dim>>   tangential_values(vector_n_q_points);
      std::vector<types::global_dof_index> indices(vector_dofs_per_cell);
      Vector<double> cell_rhs(vector_dofs_per_cell);
      cell_it c = dh.begin_active(), vc = gradient_dh.begin_active();
      for (; c != endc; ++c, ++vc)
        {
          if (c->subdomain_id() != this_mpi_process)
            continue;
          fe_v.reinit(c);
          vector_fe_v.reinit(vc);
          fe_v.get_function_values(dphi_dn, flux_values);
          vector_fe_v[all_components].get_function_values(tangential,
                                                          tangential_values);
          const std::vector<Tensor<1, dim>> &normals_q =
            vector_fe_v.get_normal_vectors();
          cell_rhs = 0;
          for (unsigned int q = 0; q < vector_n_q_points; ++q)
            {
              Tensor<1, dim> normal = normals_q[q];
              normal /= normal.norm();
              const Tensor<1, dim> &g = tangential_values[q];
              const Tensor<1, dim>  full =
                g - (g * normal) * normal + flux_values[q] * normal;
              for (unsigned int i = 0; i < vector_dofs_per_cell; ++i)
                {
                  const unsigned int ci =
                    gradient_fe->system_to_component_index(i).first;
                  cell_rhs(i) += vector_fe_v.shape_value(i, q) * full[ci] *
                                 vector_fe_v.JxW(q);
                }
            }
          vc->get_dof_indices(indices);
          vector_constraints.distribute_local_to_global(cell_rhs,
                                                        indices,
                                                        full_rhs);
        }
      full_rhs.compress(VectorOperation::add);
      solver.solve(vector_gradients_matrix,
                   vector_gradients_solution,
                   full_rhs,
                   mass_prec);
      vector_constraints.distribute(vector_gradients_solution);
      return;
    }

  // The component of the gradient along the normal is the flux, by
  // definition, and the flux is what the Galerkin system solved for. Imposing
  // it removes the only place where a discrete field still had to be
  // differentiated: the curvature term, which touches the normal component
  // alone. What the weak form is left to supply is the tangential part.
  {
    const Vector<double> recovered(vector_gradients_solution);
    const Vector<double> nodal_normals(vector_normals_solution);
    const Vector<double> nodal_flux(glob_dphi_dn);
    for (types::global_dof_index scalar_i = 0; scalar_i < dh.n_dofs();
         ++scalar_i)
      {
        const types::global_dof_index original_i =
          sub_wise_to_original[scalar_i];
        std::array<types::global_dof_index, dim> index;
        Tensor<1, dim>                          value, normal;
        bool owned = true;
        for (unsigned int d = 0; d < dim; ++d)
          {
            index[d] =
              vec_original_to_sub_wise[d * dh.n_dofs() + original_i];
            if (!vector_this_cpu_set.is_element(index[d]))
              owned = false;
            value[d]  = recovered(index[d]);
            normal[d] = nodal_normals(index[d]);
          }
        if (!owned)
          continue;
        const double length = normal.norm();
        if (length < 1e-14)
          continue;
        normal /= length;
        const Tensor<1, dim> corrected =
          value - (value * normal) * normal + nodal_flux[scalar_i] * normal;
        for (unsigned int d = 0; d < dim; ++d)
          vector_gradients_solution(index[d]) = corrected[d];
      }
    vector_gradients_solution.compress(VectorOperation::insert);
  }
}

template <int dim>
void
BEMProblem<dim>::compute_gradients_hypersingular_weak(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<dim>                 &analytic_phi,
  const Function<dim>                 &trace_gradient,
  const bool                           discrete_trace)
{
  if constexpr (dim != 3)
    {
      (void)glob_phi;
      (void)glob_dphi_dn;
      (void)analytic_phi;
      (void)trace_gradient;
      (void)discrete_trace;
      AssertThrow(false,
                  ExcMessage("The Galerkin form of the gradient "
                             "representation is implemented for surfaces "
                             "in three dimensions."));
    }
  else
    {
      Teuchos::TimeMonitor LocalTimer(*GradientTime);
      pcout << "Computing gradients from the Galerkin form of the gradient "
               "representation, "
            << (discrete_trace ? "discrete" : "analytical") << " trace"
            << std::endl;

      // On the boundary, the average of the interior and exterior traces of
      // the representation is half the gradient. With the mesh normal
      // n = sigma n_out and the flux lambda = grad u . n,
      //
      //   tangential:  grad_S phi = 2 sigma grad_S (V lambda - K phi)
      //                           = grad_S (phi + 2 sigma (V lambda - R phi)),
      //   normal:      lambda     = 2 sigma (K' lambda + W phi),
      //
      // where R phi = int dG/dn_y (phi(y) - phi(x)) is the regularised
      // double layer, K phi = R phi - sigma phi / 2 at smooth points. The
      // first line is tested with the vector space and integrated by parts
      // on each cell; the second with the scalar space, W in Maue's form.
      // Both are exact for the exact data, and with the Galerkin flux the
      // tangential part differs from grad_S phi by the gradient of the
      // first-kind residual V lambda_h - R phi.

      // The densities are needed on every cell, not only on the owned ones.
      const Vector<double> phi_full(glob_phi);
      const Vector<double> flux_full(glob_dphi_dn);
      const bool           self_check =
        std::getenv("PIBEM_HYPERSINGULAR_WEAK_CHECK") != nullptr;
      if (self_check)
        pcout << "  self-check: the exact flux replaces the Galerkin flux"
              << std::endl;

      // Orientation of the mesh normals: int_Gamma n_out . x = 3 |Omega| > 0
      // for a bounded domain. Every process holds all cells.
      double position_flux = 0.0;
      {
        FEValues<dim - 1, dim> values(*mapping,
                                      *fe,
                                      QGauss<dim - 1>(4),
                                      update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
        for (const auto &cell : dh.active_cell_iterators())
          {
            values.reinit(cell);
            for (unsigned int q = 0; q < values.n_quadrature_points; ++q)
              position_flux +=
                (values.normal_vector(q) * values.quadrature_point(q)) *
                values.JxW(q);
          }
      }
      const double sigma = position_flux > 0.0 ? 1.0 : -1.0;

      SurfacePotentialEvaluator::Densities densities;
      if (self_check)
        densities.flux_gradient = &trace_gradient;
      else
        densities.flux_coefficients = &flux_full;
      if (discrete_trace)
        densities.trace_coefficients = &phi_full;
      else
        {
          densities.trace_function = &analytic_phi;
          densities.trace_gradient = &trace_gradient;
        }
      const char  *duffy_env = std::getenv("PIBEM_HW_DUFFY_ORDER");
      const char  *near_env  = std::getenv("PIBEM_HW_NEAR_RATIO");
      const unsigned int duffy_order =
        duffy_env != nullptr ? std::atoi(duffy_env) : 12;
      const double near_ratio =
        near_env != nullptr ? std::atof(near_env) : 0.5;

      const auto start = std::chrono::steady_clock::now();

      // <K' lambda, psi_j> as a Galerkin double integral. Evaluated at the
      // quadrature points of the test cell, K' lambda is log-singular at the
      // edges of the box and varies on the scale of the gap near a contact,
      // and a tensor rule on the test cell resolves neither. The panel-pair
      // machinery handles both: Sauter-Schwab for touching pairs, subdivision
      // for close ones. Far pairs use cached per-cell data and a factorised
      // sum, as in the assembly of the matrix.
      std::vector<Vector<double>> adjoint_moments(
        dh.get_triangulation().n_active_cells());
      if (!discrete_trace)
        {
          const GalerkinSingularQuadrature singular_strategy =
            galerkin_singular_quadrature == "iterated_duffy" ?
              GalerkinSingularQuadrature::iterated_duffy :
              GalerkinSingularQuadrature::sauter_schwab;
          const char *ratio_env = std::getenv("PIBEM_HW_PAIR_RATIO");
          const char *depth_env = std::getenv("PIBEM_HW_PAIR_DEPTH");
          const char *order_env = std::getenv("PIBEM_HW_PAIR_ORDER");
          GalerkinPanelPairIntegrator<3> integrator(
            *mapping,
            *fe,
            singular_strategy,
            galerkin_singular_quadrature_order,
            galerkin_duffy_outer_subdivisions,
            ratio_env != nullptr ? std::atof(ratio_env) :
                                   galerkin_near_singular_distance_ratio,
            // The kernel of K' is as sharp as the gap near a contact; the
            // depth used for V is not enough to resolve it.
            depth_env != nullptr ?
              std::atoi(depth_env) :
              std::max(10U, galerkin_near_singular_maximum_depth),
            order_env != nullptr ? std::atoi(order_env) :
                                   galerkin_near_singular_quadrature_order);
          integrator.set_screening(galerkin_screening());
          integrator.set_light_geometry(galerkin_light_geometry);
          const double kappa = galerkin_screening();

          const QGauss<dim - 1>  pair_rule(galerkin_quadrature_order);
          FEValues<dim - 1, dim> pair_values(*mapping,
                                             *fe,
                                             pair_rule,
                                             update_values |
                                               update_normal_vectors |
                                               update_quadrature_points |
                                               update_JxW_values);
          const unsigned int n_pair_points = pair_rule.size();
          const unsigned int n_cells = dh.get_triangulation().n_active_cells();
          std::vector<std::vector<Point<dim>>>     cached_points(n_cells);
          std::vector<std::vector<Tensor<1, dim>>> cached_normals(n_cells);
          std::vector<std::vector<double>>         cached_weighted_flux(n_cells);
          std::vector<std::vector<double>>         cached_JxW(n_cells);
          std::vector<FullMatrix<double>>          cached_shapes(n_cells);
          std::vector<types::global_dof_index>     dofs(fe->dofs_per_cell);
          Vector<double>                           exact(dim);
          for (const auto &cell : dh.active_cell_iterators())
            {
              const unsigned int c = cell->active_cell_index();
              pair_values.reinit(cell);
              cell->get_dof_indices(dofs);
              cached_points[c]  = pair_values.get_quadrature_points();
              cached_normals[c] = pair_values.get_normal_vectors();
              cached_weighted_flux[c].resize(n_pair_points);
              cached_JxW[c].resize(n_pair_points);
              cached_shapes[c].reinit(fe->dofs_per_cell, n_pair_points);
              for (unsigned int q = 0; q < n_pair_points; ++q)
                {
                  double flux = 0.0;
                  if (self_check)
                    {
                      trace_gradient.vector_value(cached_points[c][q], exact);
                      for (unsigned int d = 0; d < dim; ++d)
                        flux += exact[d] * cached_normals[c][q][d];
                    }
                  else
                    for (unsigned int k = 0; k < fe->dofs_per_cell; ++k)
                      flux += flux_full(dofs[k]) * pair_values.shape_value(k, q);
                  cached_weighted_flux[c][q] = flux * pair_values.JxW(q);
                  cached_JxW[c][q]           = pair_values.JxW(q);
                  for (unsigned int k = 0; k < fe->dofs_per_cell; ++k)
                    cached_shapes[c](k, q) = pair_values.shape_value(k, q);
                }
            }

          std::vector<double> potential(n_pair_points);
          for (const auto &test_cell : dh.active_cell_iterators())
            {
              if (test_cell->subdomain_id() != this_mpi_process)
                continue;
              const unsigned int t = test_cell->active_cell_index();
              Vector<double>     moments(fe->dofs_per_cell);
              std::fill(potential.begin(), potential.end(), 0.0);
              for (const auto &trial_cell : dh.active_cell_iterators())
                {
                  const unsigned int r = trial_cell->active_cell_index();
                  if (integrator.is_far_regular_pair(test_cell, trial_cell))
                    {
                      for (unsigned int q = 0; q < n_pair_points; ++q)
                        {
                          double sum = 0.0;
                          for (unsigned int k = 0; k < n_pair_points; ++k)
                            {
                              const Tensor<1, dim> displacement =
                                cached_points[r][k] - cached_points[t][q];
                              const double distance = displacement.norm();
                              sum += (displacement * cached_normals[t][q]) *
                                     (kappa == 0.0 ?
                                        1.0 :
                                        (1.0 + kappa * distance) *
                                          std::exp(-kappa * distance)) /
                                     (4.0 * numbers::PI * distance *
                                      distance * distance) *
                                     cached_weighted_flux[r][k];
                            }
                          potential[q] += sum;
                        }
                    }
                  else if (self_check)
                    moments.add(1.0,
                                integrator.integrate_adjoint_double_layer_flux(
                                  test_cell,
                                  trial_cell,
                                  galerkin_quadrature_order,
                                  trace_gradient));
                  else
                    {
                      const FullMatrix<double> block =
                        integrator.integrate_adjoint_double_layer(
                          test_cell, trial_cell, galerkin_quadrature_order);
                      trial_cell->get_dof_indices(dofs);
                      for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                        for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                          moments(i) += block(i, j) * flux_full(dofs[j]);
                    }
                }
              for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
                for (unsigned int q = 0; q < n_pair_points; ++q)
                  moments(i) +=
                    cached_shapes[t](i, q) * potential[q] * cached_JxW[t][q];
              adjoint_moments[t] = moments;
            }
        }

      const double pair_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      start)
          .count();
      const SurfacePotentialEvaluator evaluator(
        *mapping,
        dh,
        mapping_type == "FE" ? vector_fe_order : mapping_degree,
        densities,
        duffy_order,
        near_ratio,
        16,
        galerkin_screening());

      // Quadrature for the test integrals. The potentials are smooth on
      // each cell, so a rule a few orders above the polynomial degree
      // suffices; each point costs a sweep over all cells.
      const char        *points_env = std::getenv("PIBEM_HW_POINTS");
      const unsigned int n_points =
        points_env != nullptr ? std::atoi(points_env) : fe->degree + 4;
      const QGauss<dim - 1>    area_rule(n_points);
      const QGauss<1>          edge_rule(n_points);
      FEValues<dim - 1, dim>   vector_values(*mapping,
                                           *gradient_fe,
                                           area_rule,
                                           update_values | update_gradients |
                                             update_normal_vectors |
                                             update_quadrature_points |
                                             update_JxW_values);
      FEValues<dim - 1, dim>   scalar_values(*mapping,
                                           *fe,
                                           area_rule,
                                           update_values | update_gradients |
                                             update_quadrature_points |
                                             update_JxW_values);

      // Edge points are given directly in the reference square, face by
      // face, so no face orientation enters. Face f lies at xi_a = 0 or 1
      // with a = 0 for f = 0, 1 and a = 1 for f = 2, 3.
      std::vector<std::unique_ptr<FEValues<dim - 1, dim>>> edge_vector_values;
      std::vector<std::unique_ptr<FEValues<dim - 1, dim>>> edge_scalar_values;
      for (unsigned int f = 0; f < 4; ++f)
        {
          std::vector<Point<dim - 1>> points;
          std::vector<double>         weights;
          for (unsigned int q = 0; q < edge_rule.size(); ++q)
            {
              const double s = edge_rule.point(q)[0];
              points.push_back(f == 0 ? Point<dim - 1>(0, s) :
                               f == 1 ? Point<dim - 1>(1, s) :
                               f == 2 ? Point<dim - 1>(s, 0) :
                                        Point<dim - 1>(s, 1));
              weights.push_back(edge_rule.weight(q));
            }
          const Quadrature<dim - 1> rule(points, weights);
          edge_vector_values.push_back(
            std::make_unique<FEValues<dim - 1, dim>>(*mapping,
                                                     *gradient_fe,
                                                     rule,
                                                     update_values));
          edge_scalar_values.push_back(
            std::make_unique<FEValues<dim - 1, dim>>(
              *mapping,
              *fe,
              rule,
              update_values | update_quadrature_points |
                update_normal_vectors | update_jacobians));
        }

      const unsigned int nv = gradient_fe->dofs_per_cell;
      const unsigned int ns = fe->dofs_per_cell;
      const double kappa_square = galerkin_screening() * galerkin_screening();
      std::vector<unsigned int> component(nv);
      for (unsigned int i = 0; i < nv; ++i)
        component[i] = gradient_fe->system_to_component_index(i).first;

      FullMatrix<double> mass_v(nv, nv), inverse_v(nv, nv);
      FullMatrix<double> mass_s(ns, ns), inverse_s(ns, ns);
      Vector<double>     rhs_t(nv), coef_t(nv), rhs_full(nv), coef_full(nv);
      Vector<double>     rhs_check(nv), check_scale(nv);
      Vector<double>     rhs_n(ns), coef_n(ns), rhs_best(ns), coef_best(ns);
      std::vector<double> trace_q(area_rule.size());
      std::vector<double> flux_q(area_rule.size());
      std::vector<double> exact_flux_q(area_rule.size());
      std::vector<double> edge_trace(edge_rule.size());
      std::vector<types::global_dof_index> vector_indices(nv);
      Vector<double>                       gradient_value(dim);

      double residual_square = 0.0, measure = 0.0, residual_max = 0.0;
      double edge_residual_max = 0.0, divergence_defect = 0.0;
      double error_star = 0.0, error_galerkin = 0.0, error_best = 0.0;
      double norm_exact = 0.0;
      double n_targets  = 0.0;
      Point<dim> worst_location, worst_edge_location;
      std::array<double, 4> star_to_best_by_material{{0, 0, 0, 0}};
      std::array<double, 3> tangential_by_region{{0, 0, 0}};
      const bool split_by_region = std::getenv("PIBEM_HW_DEBUG") != nullptr;
      Point<dim> box_lower, box_upper;
      for (unsigned int d = 0; d < dim; ++d)
        {
          box_lower[d] = std::numeric_limits<double>::max();
          box_upper[d] = -std::numeric_limits<double>::max();
        }
      for (const auto &v : dh.get_triangulation().get_vertices())
        for (unsigned int d = 0; d < dim; ++d)
          {
            box_lower[d] = std::min(box_lower[d], v[d]);
            box_upper[d] = std::max(box_upper[d], v[d]);
          }

      vector_gradients_solution.reinit(vector_this_cpu_set,
                                       mpi_communicator);

      auto vector_cell = gradient_dh.begin_active();
      for (auto cell = dh.begin_active(); cell != dh.end();
           ++cell, ++vector_cell)
        {
          if (cell->subdomain_id() != this_mpi_process)
            continue;
          vector_values.reinit(vector_cell);
          scalar_values.reinit(cell);
          mass_v      = 0;
          mass_s      = 0;
          rhs_t       = 0;
          rhs_n       = 0;
          rhs_best    = 0;
          rhs_check   = 0;
          check_scale = 0;
          if (discrete_trace)
            scalar_values.get_function_values(phi_full, trace_q);
          scalar_values.get_function_values(flux_full, flux_q);

          for (unsigned int q = 0; q < area_rule.size(); ++q)
            {
              const Point<dim>     &x      = vector_values.quadrature_point(q);
              const Tensor<1, dim> &normal = vector_values.normal_vector(q);
              const double          JxW    = vector_values.JxW(q);
              const double trace =
                discrete_trace ? trace_q[q] : analytic_phi.value(x);
              const auto potentials = evaluator.evaluate(x, normal, trace);
              n_targets += 1.0;
              const double residual =
                potentials.single_layer - potentials.double_layer;
              const double w = trace + 2.0 * sigma * residual;
              const double curvature =
                evaluator.geometry(cell, area_rule.point(q)).curvature;
              residual_square += residual * residual * JxW;
              measure += JxW;
              if (std::abs(residual) > residual_max)
                {
                  residual_max   = std::abs(residual);
                  worst_location = x;
                }

              for (unsigned int i = 0; i < nv; ++i)
                {
                  const unsigned int ci = component[i];
                  const double       v  = vector_values.shape_value(i, q);
                  const double div = vector_values.shape_grad(i, q)[ci];
                  rhs_t(i) +=
                    (-w * div + curvature * w * v * normal[ci]) * JxW;
                  rhs_check(i) +=
                    (-div + curvature * v * normal[ci]) * JxW;
                  check_scale(i) += std::abs(div) * JxW;
                  for (unsigned int j = 0; j < nv; ++j)
                    if (component[j] == ci)
                      mass_v(i, j) +=
                        v * vector_values.shape_value(j, q) * JxW;
                }

              trace_gradient.vector_value(x, gradient_value);
              double exact_flux = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                exact_flux += gradient_value[d] * normal[d];
              exact_flux_q[q] = exact_flux;

              for (unsigned int j = 0; j < ns; ++j)
                {
                  const double psi = scalar_values.shape_value(j, q);
                  // The screened hypersingular operator adds
                  // kappa^2 int psi n_x . V(n phi) to Maue's form.
                  if (!discrete_trace)
                    rhs_n(j) += 2.0 * sigma *
                                (cross_product_3d(normal,
                                                  scalar_values.shape_grad(j,
                                                                           q)) *
                                   potentials.curl_single_layer +
                                 kappa_square * psi *
                                   (normal * potentials.normal_single_layer)) *
                                JxW;
                  rhs_best(j) += psi * exact_flux * JxW;
                  for (unsigned int k = 0; k < ns; ++k)
                    mass_s(j, k) +=
                      psi * scalar_values.shape_value(k, q) * JxW;
                }
            }

          for (unsigned int f = 0; f < 4; ++f)
            {
              FEValues<dim - 1, dim> &ev = *edge_vector_values[f];
              FEValues<dim - 1, dim> &es = *edge_scalar_values[f];
              ev.reinit(vector_cell);
              es.reinit(cell);
              if (discrete_trace)
                es.get_function_values(phi_full, edge_trace);
              const unsigned int along   = f < 2 ? 1 : 0;
              const unsigned int across  = 1 - along;
              const double       outward = (f % 2 == 0) ? -1.0 : 1.0;
              for (unsigned int q = 0; q < edge_rule.size(); ++q)
                {
                  const Point<dim>     &x      = es.quadrature_point(q);
                  const Tensor<1, dim> &normal = es.normal_vector(q);
                  const auto           &jac    = es.jacobian(q);
                  Tensor<1, dim>        t, o;
                  for (unsigned int d = 0; d < dim; ++d)
                    {
                      t[d] = jac[d][along];
                      o[d] = outward * jac[d][across];
                    }
                  const double ds = t.norm() * edge_rule.weight(q);
                  t /= t.norm();
                  // Co-normal: tangent to the surface, orthogonal to the
                  // edge, pointing out of the cell.
                  Tensor<1, dim> m = o - (o * t) * t - (o * normal) * normal;
                  m /= m.norm();

                  const double trace =
                    discrete_trace ? edge_trace[q] : analytic_phi.value(x);
                  const auto potentials = evaluator.evaluate(x, normal, trace);
                  n_targets += 1.0;
                  const double residual =
                    potentials.single_layer - potentials.double_layer;
                  const double w = trace + 2.0 * sigma * residual;
                  if (std::abs(residual) > edge_residual_max)
                    {
                      edge_residual_max   = std::abs(residual);
                      worst_edge_location = x;
                    }

                  for (unsigned int i = 0; i < nv; ++i)
                    {
                      const double vm = ev.shape_value(i, q) * m[component[i]];
                      rhs_t(i) += w * vm * ds;
                      rhs_check(i) += vm * ds;
                    }
                  if (!discrete_trace)
                    {
                      // -oint psi (n x m) . V(n x grad phi): the edge term
                      // of Maue's form, which cancels between neighbours
                      // only for continuous test functions.
                      const double edge_density =
                        cross_product_3d(normal, m) *
                        potentials.curl_single_layer;
                      for (unsigned int j = 0; j < ns; ++j)
                        rhs_n(j) += 2.0 * sigma *
                                    (-es.shape_value(j, q) * edge_density) *
                                    ds;
                    }
                }
            }

          // With w = 1 the tangential right-hand side must vanish: the
          // divergence theorem on the cell, curvature and co-normals
          // included.
          if (!discrete_trace)
            rhs_n.add(2.0 * sigma,
                      adjoint_moments[cell->active_cell_index()]);

          // Normalised by the largest term of the cell: a test function
          // along the normal of a flat cell has no divergence at all, and
          // its own scale would only measure round-off.
          {
            const double cell_scale = check_scale.linfty_norm();
            for (unsigned int i = 0; i < nv; ++i)
              if (cell_scale > 0.0)
                divergence_defect =
                  std::max(divergence_defect,
                           std::abs(rhs_check(i)) / cell_scale);
          }

          inverse_v.invert(mass_v);
          inverse_s.invert(mass_s);
          inverse_v.vmult(coef_t, rhs_t);
          if (discrete_trace)
            cell->get_dof_values(flux_full, coef_n);
          else
            inverse_s.vmult(coef_n, rhs_n);
          inverse_s.vmult(coef_best, rhs_best);

          // Full field: L2 projection of P g_t + lambda* n, with n the exact
          // normal of the mapped surface at each quadrature point.
          rhs_full = 0;
          for (unsigned int q = 0; q < area_rule.size(); ++q)
            {
              const Tensor<1, dim> &normal = vector_values.normal_vector(q);
              const double          JxW    = vector_values.JxW(q);
              Tensor<1, dim>        g;
              for (unsigned int k = 0; k < nv; ++k)
                g[component[k]] += coef_t(k) * vector_values.shape_value(k, q);
              double lambda_star = 0.0, lambda_best = 0.0;
              for (unsigned int j = 0; j < ns; ++j)
                {
                  lambda_star += coef_n(j) * scalar_values.shape_value(j, q);
                  lambda_best += coef_best(j) * scalar_values.shape_value(j, q);
                }
              const Tensor<1, dim> full =
                g - (g * normal) * normal + lambda_star * normal;
              for (unsigned int i = 0; i < nv; ++i)
                rhs_full(i) += vector_values.shape_value(i, q) *
                               full[component[i]] * JxW;

              const double exact = exact_flux_q[q];
              error_star += (lambda_star - exact) * (lambda_star - exact) * JxW;
              {
                const unsigned int material =
                  std::min<unsigned int>(cell->material_id(), 3);
                star_to_best_by_material[material] +=
                  (lambda_star - lambda_best) * (lambda_star - lambda_best) *
                  JxW;
              }
              error_galerkin +=
                (flux_q[q] - exact) * (flux_q[q] - exact) * JxW;
              error_best += (lambda_best - exact) * (lambda_best - exact) * JxW;
              norm_exact += exact * exact * JxW;
            }
          inverse_v.vmult(coef_full, rhs_full);

          // Tangential error of the recovered field, split between the
          // region within 0.35 of the vertical axis below the origin - the
          // contact core of the sphere-in-box benchmark - and the rest.
          if (split_by_region)
            for (unsigned int q = 0; q < area_rule.size(); ++q)
              {
                const Point<dim>     &x      = vector_values.quadrature_point(q);
                const Tensor<1, dim> &normal = vector_values.normal_vector(q);
                Tensor<1, dim>        g;
                for (unsigned int k = 0; k < nv; ++k)
                  g[component[k]] +=
                    coef_full(k) * vector_values.shape_value(k, q);
                trace_gradient.vector_value(x, gradient_value);
                Tensor<1, dim> e;
                for (unsigned int d = 0; d < dim; ++d)
                  e[d] = g[d] - gradient_value[d];
                e -= (e * normal) * normal;
                const bool core =
                  std::sqrt(x[0] * x[0] + x[1] * x[1]) < 0.35 && x[2] < -0.5;
                // Within 0.5 of an edge of the box: two coordinates close to
                // the planes of the box faces.
                unsigned int close_planes = 0;
                for (unsigned int d = 0; d < dim; ++d)
                  if (std::abs(x[d] - box_lower[d]) < 0.5 ||
                      std::abs(x[d] - box_upper[d]) < 0.5)
                    ++close_planes;
                const unsigned int region = core               ? 0 :
                                            close_planes >= 2 ? 1 :
                                                                2;
                tangential_by_region[region] +=
                  (e * e) * vector_values.JxW(q);
              }
          vector_cell->get_dof_indices(vector_indices);
          for (unsigned int i = 0; i < nv; ++i)
            vector_gradients_solution(vector_indices[i]) = coef_full(i);
        }
      vector_gradients_solution.compress(VectorOperation::insert);

      const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      start)
          .count();
      const auto   counts   = evaluator.counts();
      const double far      = Utilities::MPI::sum(double(counts[0]),
                                             mpi_communicator);
      const double leaves   = Utilities::MPI::sum(double(counts[1]),
                                                mpi_communicator);
      const double singular = Utilities::MPI::sum(double(counts[2]),
                                                  mpi_communicator);
      const double targets = Utilities::MPI::sum(n_targets, mpi_communicator);
      const double rms =
        std::sqrt(Utilities::MPI::sum(residual_square, mpi_communicator) /
                  Utilities::MPI::sum(measure, mpi_communicator));
      const double exact_norm =
        std::sqrt(Utilities::MPI::sum(norm_exact, mpi_communicator));
      const double e_star =
        std::sqrt(Utilities::MPI::sum(error_star, mpi_communicator));
      const double e_galerkin =
        std::sqrt(Utilities::MPI::sum(error_galerkin, mpi_communicator));
      const double e_best =
        std::sqrt(Utilities::MPI::sum(error_best, mpi_communicator));
      const double max_residual =
        Utilities::MPI::max(residual_max, mpi_communicator);
      const double max_edge_residual =
        Utilities::MPI::max(edge_residual_max, mpi_communicator);
      const double defect =
        Utilities::MPI::max(divergence_defect, mpi_communicator);
      const double deviation =
        Utilities::MPI::max(evaluator.geometry_deviation(), mpi_communicator);
      const double max_seconds =
        Utilities::MPI::max(seconds, mpi_communicator);
      const double max_pair_seconds =
        Utilities::MPI::max(pair_seconds, mpi_communicator);
      const double sources = Utilities::MPI::sum(
        double(evaluator.source_evaluations()), mpi_communicator);

      pcout << "  orientation sigma " << sigma << ", reconstructed geometry "
            << "deviation " << deviation << ", divergence identity defect "
            << defect << std::endl
            << "  " << targets << " target points (" << n_points
            << " per direction); panels far " << far << ", near leaves "
            << leaves << ", near or singular " << singular << ", "
            << sources << " source points off the far rules; "
            << max_seconds << " s, of which " << max_pair_seconds
            << " s for <K' lambda, psi>" << std::endl
            << "  first-kind residual V lambda - R phi at the targets: rms "
            << rms << ", max interior " << max_residual << ", max on edges "
            << max_edge_residual << std::endl;
      if (std::getenv("PIBEM_HW_DEBUG") != nullptr)
        std::cerr << "  rank " << this_mpi_process << ": worst interior residual "
                  << residual_max << " at " << worst_location
                  << ", worst edge residual " << edge_residual_max << " at "
                  << worst_edge_location << std::endl;
      if (split_by_region)
        pcout << "  tangential error of the recovered field: contact core "
              << std::sqrt(Utilities::MPI::sum(tangential_by_region[0],
                                               mpi_communicator))
              << ", near box edges "
              << std::sqrt(Utilities::MPI::sum(tangential_by_region[1],
                                               mpi_communicator))
              << ", elsewhere "
              << std::sqrt(Utilities::MPI::sum(tangential_by_region[2],
                                               mpi_communicator))
              << std::endl;
      if (!discrete_trace && std::getenv("PIBEM_HW_DEBUG") != nullptr)
        {
          pcout << "  |lambda* - Q_h lambda| / |lambda| by material:";
          for (unsigned int m = 1; m < 4; ++m)
            pcout << " " << m << ": "
                  << std::sqrt(Utilities::MPI::sum(star_to_best_by_material[m],
                                                   mpi_communicator)) /
                       exact_norm;
          pcout << std::endl;
        }
      if (!discrete_trace)
        pcout << "  normal part, L2 error relative to |lambda|: "
                 "representation "
              << e_star / exact_norm << ", Galerkin flux "
              << e_galerkin / exact_norm << ", best approximation "
              << e_best / exact_norm << std::endl;
    }
}


template <int dim>
void
BEMProblem<dim>::compute_gradients_hypersingular(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn)
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  pcout << "Computing gradients with hypersingular integrals" << std::endl;
  std::array<double, 4> hypersingular_free_term_report{
    {std::numeric_limits<double>::max(), 0.0, 0.0, 0.0}};

  TrilinosWrappers::MPI::Vector vector_hyp_gradients_solution(
    vector_this_cpu_set, mpi_communicator);
  TrilinosWrappers::MPI::Vector vector_b_free_coeff(vector_this_cpu_set,
                                                    mpi_communicator);

  Vector<double> phi_local(glob_phi);
  Vector<double> dphi_dn_local(glob_dphi_dn);

  // Tensor<1, dim> node_gradient;

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

  const unsigned int n_q_points = fe_v.n_quadrature_points;

  std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
  pcout << fe->dofs_per_cell << " " << std::endl;

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


  cell_it cell = dh.begin_active(), endc = dh.end();

  Tensor<1, dim> D;
  // Tensor<1, dim> R;
  Tensor<2, dim> H;
  double         s;

  for (cell = dh.begin_active(); cell != endc; ++cell)
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);

      const std::vector<Point<dim>> &q_points    = fe_v.get_quadrature_points();
      const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();

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
      // one is the singular index:
      for (types::global_dof_index i = 0; i < dh.n_dofs(); ++i) // these must now be the locally owned dofs. the rest should stay the same
        {
          Tensor<1, dim> integral;
          Tensor<1, dim> b_integral;
          if (this_cpu_set.is_element(i))
            {
              Point<dim> singularity = support_points[i];
              Point<dim-1> ref_projection(0.0,0.0);
              bool         is_singular    = false;
              unsigned int singular_index = numbers::invalid_unsigned_int;

              // Whether a cell needs the singular treatment is a question
              // about geometry, not about degrees of freedom. With a
              // continuous space the collocation point at a vertex is the
              // same dof in every cell that touches it, so matching dof
              // indices finds them all; a discontinuous space gives each cell
              // its own dof there, and a hanging node puts the point in the
              // middle of a coarse cell's edge where that cell has no dof at
              // all. In every one of those cases the point still lies on the
              // cell and the kernel is still singular, so the test is: does
              // the point sit on this cell?
              for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                if (double_nodes_set[i].count(local_dof_indices[j]) > 0 ||
                    support_points[local_dof_indices[j]].distance(
                      support_points[i]) < 1e-10)
                  {
                    singular_index = j;
                    is_singular    = true;
                    break;
                  }
                
              bool is_quasi_singular = false;
              if (is_singular == false)
              {
                Point<dim> singularity = support_points[i];
                double dist_to_center = (singularity - cell->center()).norm();
                double h = cell->diameter();
                if(dist_to_center - 0.5*h < 0.001)
                {
                  QuasiSingularKernelIntegral<dim> qski(cell, *fe, *mapping, singularity);
                  ref_projection = qski.get_closest_reference_point();
                  double dist_to_cell = qski.min_distance;
                  
                  double distance_ratio = dist_to_cell / cell->diameter();
                  if(distance_ratio < 10)
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
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  const Tensor<1, dim> R = q_points[q] - support_points[i];
                  if (kernel_type == "laplace")
                    LaplaceKernel::kernels(R, H, D, s);
                  else if (kernel_type == "screened")
                    ScreenedKernel::kernels(R, H, D, s, screened_kappa);
                  else
                    AssertThrow(false, ExcMessage("Unknown kernel type: " + kernel_type));
                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                  {
                    integral += -phi_local(local_dof_indices[j]) *
                                  (H * normals[q]) *
                                  fe_v.shape_value(j, q) * fe_v.JxW(q) +
                                dphi_dn_local(local_dof_indices[j]) * D *
                                  fe_v.shape_value(j, q) * fe_v.JxW(q);
                    b_integral += -1.0 * (H * normals[q]) *
                                  fe_v.shape_value(j, q) * fe_v.JxW(q);
                  }
                }
              }
              else if(is_quasi_singular == true)
              {
                //pcout << cell << " \t s: " << singularity << "\t \t --> the quadrature is quasi singular" << std::endl;
                const Quadrature<dim - 1> quasi_singular_quadrature =
                    get_quasi_singular_quadrature(cell, *mapping, singularity, ref_projection, 2);
                Assert(quasi_singular_quadrature.size() > 0, ExcMessage("empty quasi-singular quadrature!"));

                FEValues<dim - 1, dim> fe_v_quasi_singular(
                    *mapping,
                    *fe,
                    quasi_singular_quadrature,
                    update_jacobians | update_values | update_normal_vectors |
                      update_quadrature_points | update_JxW_values);

                fe_v_quasi_singular.reinit(cell);

                const std::vector<Tensor<1, dim>> &quasi_singular_normals =
                    fe_v_quasi_singular.get_normal_vectors();
                const std::vector<Point<dim>> &quasi_singular_q_points =
                    fe_v_quasi_singular.get_quadrature_points();

                for (unsigned int q = 0; q < quasi_singular_quadrature.size(); ++q)
                {
                  const Tensor<1, dim> R = quasi_singular_q_points[q] - support_points[i];
                  if (kernel_type == "laplace")
                    LaplaceKernel::kernels(R, H, D, s);
                  else if (kernel_type == "screened")
                    ScreenedKernel::kernels(R, H, D, s, screened_kappa);
                  else
                    AssertThrow(false, ExcMessage("Unknown kernel type: " + kernel_type));

                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                  {
                    integral += -phi_local(local_dof_indices[j]) *
                                  (H * quasi_singular_normals[q]) *
                                  fe_v_quasi_singular.shape_value(j, q) * fe_v_quasi_singular.JxW(q) +
                                dphi_dn_local(local_dof_indices[j]) * D *
                                  fe_v_quasi_singular.shape_value(j, q) * fe_v_quasi_singular.JxW(q);
                    b_integral += -1.0 * (H * quasi_singular_normals[q]) *
                                  fe_v_quasi_singular.shape_value(j, q) * fe_v_quasi_singular.JxW(q);
                  }
                }
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

                  Assert(
                    (*fe).has_support_points(),
                    ExcMessage(
                      "The FE selected has no support points. This is not supported."));
                  const Point<dim - 1> P =
                    (*fe).unit_support_point(singular_index);
                  // pcout<<"P: "<<P<<std::endl;

                  SingularKernelIntegral<dim> sing_kernel_integrator(cell,
                                                                     *fe,
                                                                     *mapping,
                                                                     P);
                  std::vector<Tensor<1, dim>> Vk_integrals =
                    sing_kernel_integrator.evaluate_VkNj_integrals();
                  std::vector<Tensor<1, dim>> Wk_integrals =
                    sing_kernel_integrator.evaluate_WkNj_integrals();
                  Tensor<1, dim> singular_cell_contribution_hyp;
                  Tensor<1, dim> singular_cell_contribution_str;
                  // Tensor<1,dim> b_singular_cell_contribution_hyp;
                  for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
                    {
                      // const Tensor<1, dim> R =
                      // support_points[local_dof_indices[j]] -
                      // support_points[i]; pcout<<"* "<<cell<<"  "<<R<<"
                      // "<<support_points[local_dof_indices[j]]<<std::endl;

                      singular_cell_contribution_hyp +=
                        -phi_local(local_dof_indices[j]) * Vk_integrals[j];
                      singular_cell_contribution_str +=
                        dphi_dn_local(local_dof_indices[j]) * Wk_integrals[j];
                      // this was an attempt to compute b_i in an alternative,
                      // numerical way, as alpha. Couldn't get it to work
                      // b_singular_cell_contribution_hyp+= -Vk_integrals[j];

                      // pcout<<"*** "<<cell<<"
                      // "<<dphi_dn_local(local_dof_indices[j])<<"
                      // "<<Wk_integrals[j]<<std::endl; pcout<<"j "<<j<<"
                      // "<<cell<<"  "<<phi_local(local_dof_indices[j])<<"
                      // "<<Vk_integrals[j]<<std::endl;
                    }
                  // pcout<<cell<<"   "<<singular_cell_contribution_str<<"
                  // "<<singular_cell_contribution_hyp<<std::endl;
                  // integral_3+=singular_cell_contribution_str+singular_cell_contribution_hyp;
                  // pcout<<"Qmark Hyp: "<<integral_3<<std::endl;
                  integral += singular_cell_contribution_str +
                              singular_cell_contribution_hyp;
                  // b_integral+= b_singular_cell_contribution_hyp;
                }

              const unsigned int scalar_dh_index = sub_wise_to_original[i];
              for (unsigned int d = 0; d < dim; ++d)
                {
                  const types::global_dof_index vector_dh_index =
                    vec_original_to_sub_wise[scalar_dh_index +
                                             d * dh.n_dofs()];
                  vector_hyp_gradients_solution(vector_dh_index) += integral[d];
                }
              // vector_b_free_coeff(vector_dh_index_x_component)+=b_integral[0];
              // vector_b_free_coeff(vector_dh_index_y_component)+=b_integral[1];
              // vector_b_free_coeff(vector_dh_index_z_component)+=b_integral[2];
            }
        }
    }
  vector_hyp_gradients_solution.compress(VectorOperation::add);
  vector_b_free_coeff.compress(VectorOperation::add);


  //  for (types::global_dof_index i = 0; i < dh.n_dofs();
  //       ++i) // these must now be the locally owned dofs. the rest should
  //            // stay the same
  //    {
  //      if (this_cpu_set.is_element(i))
  //        { pcout<<i<<"->    Support point: "<<support_points[i]<<std::endl;
  //          pcout << free_term_b_all[i][0] << " "
  //                << free_term_b_all[i][1] << " "
  //                << free_term_b_all[i][2] << std::endl;
  //
  //        }
  //    }



  // we now have all the ingredients for the computation of the gradients
  // through the hypersingual BIE. For each DOF, such a BIE is a vector
  // equation, of which we already have the right hand side (assembled with the
  // integrals involving phi and dphi_dn). In the right hand side we also have
  // the potential multiplied by the second --- Mantic --- free coefficient As
  // for the left hand side, the gradient of phi (our unknown) multiplies tensor
  // C. So, for each line we must invert C to obtain our gradient
  for (types::global_dof_index i = 0; i < dh.n_dofs();
       ++i) // these must now be the locally owned dofs. the rest should
            // stay the same
    {
      if (this_cpu_set.is_element(i))
        {
          // these will be useful
          unsigned int scalar_dh_index = sub_wise_to_original[i];
          std::vector<types::global_dof_index> vector_dh_indices(dim);
          for (unsigned int d = 0; d < dim; ++d)
            vector_dh_indices[d] =
              vec_original_to_sub_wise[scalar_dh_index + d * dh.n_dofs()];
          // cout<<this_mpi_process<<"  Scalar index original
          // "<<scalar_dh_index<<"   Sub-wise correspondent "<<i<<"  Vector x
          // index original "<< scalar_dh_index+ 0+dh.n_dofs()<<"   Sub-wise
          // correspondent "<<vector_dh_index_x_component<<"  Vector y index
          // original "<< scalar_dh_index+ 1+dh.n_dofs()<<"   Sub-wise
          // correspondent "<<vector_dh_index_y_component<<"  Vector z index
          // original "<< scalar_dh_index+ 2+dh.n_dofs()<<"   Sub-wise
          // correspondent "<<vector_dh_index_z_component<<endl;
          // cout<<this_mpi_process<<"  Range:
          // ("<<vector_hyp_gradients_solution.local_range().first<<","<<vector_hyp_gradients_solution.local_range().second<<")
          // "<<vector_dh_index_x_component<<" "<<vector_dh_index_y_component<<"
          // "<<vector_dh_index_z_component<<endl; pcout<<i<<"->    Support
          // point: "<<support_points[i]<<std::endl; pcout<<"b(mantic):
          // "<<free_term_b_all[i]<<std::endl; pcout<<"b(alt):
          // "<<vector_b_free_coeff(i)<<" "
          //                  <<vector_b_free_coeff(i+dh.n_dofs())<<" "
          //                  <<vector_b_free_coeff(i+2*dh.n_dofs())<<std::endl;
          // pcout<<"phi: "<<phi_local[i]<<std::endl;
          //  we must first reassemble the free coefficient vector b
          Tensor<1, dim> b;
          for (unsigned int d = 0; d < dim; ++d)
            b[d] = b_i[d][i];

          // we also need to reassemble free coefficient tensor C_ij
          FullMatrix<double> C(dim, dim);
          FullMatrix<double> Cinv(dim, dim);
          for (unsigned int di = 0; di < dim; ++di)
            for (unsigned int dj = 0; dj < dim; ++dj)
              C(di, dj) = C_ij[di * dim + dj](i);

          // we will also need the inverse of C
          Tensor<2, dim> CC;
          // pcout<<"C: "<<std::endl;
          // C.print_formatted(std::cout, 7, true,10,"0");
          if (std::getenv("PIBEM_HYP_GRADIENT_CHECK") != nullptr)
            {
              const double det =
                C(0, 0) * (C(1, 1) * C(2, 2) - C(1, 2) * C(2, 1)) -
                C(0, 1) * (C(1, 0) * C(2, 2) - C(1, 2) * C(2, 0)) +
                C(0, 2) * (C(1, 0) * C(2, 1) - C(1, 1) * C(2, 0));
              double norm = 0.0;
              for (unsigned int di = 0; di < dim; ++di)
                for (unsigned int dj = 0; dj < dim; ++dj)
                  norm += C(di, dj) * C(di, dj);
              hypersingular_free_term_report[0] =
                std::min(hypersingular_free_term_report[0], std::abs(det));
              hypersingular_free_term_report[1] =
                std::max(hypersingular_free_term_report[1], std::abs(det));
              hypersingular_free_term_report[2] =
                std::max(hypersingular_free_term_report[2],
                         std::sqrt(norm));
              hypersingular_free_term_report[3] =
                std::max(hypersingular_free_term_report[3], b.norm());
            }
          Cinv.invert(C);
          Cinv.copy_to(CC);
          // pcout<<"Cinv: "<<std::endl;
          // Cinv.print(std::cout, 5, 5);

          // now let's assemble the right hand side of the hypersingular BIE
          Tensor<1, dim> rhs;
          for (unsigned int d = 0; d < dim; ++d)
            rhs[d] = vector_hyp_gradients_solution(vector_dh_indices[d]);

          rhs += -b * phi_local[i];
          // pcout<<rhs<<std::endl;

          // and finally multiply C^-1 by the rhs to obtain the gradient
          Tensor<1, dim> hyp_gradient;
          hyp_gradient = CC * rhs;
          for (unsigned int d = 0; d < dim; ++d)
            vector_hyp_gradients_solution(vector_dh_indices[d]) =
              hyp_gradient[d];
          // pcout<<"Hyp. Rhs:"<<rhs<<std::endl;
          // pcout<<"Hyp. Gradient:"<<hyp_gradient<<std::endl;
        }
    }

  vector_hyp_gradients_solution.compress(VectorOperation::insert);
  vector_gradients_solution = vector_hyp_gradients_solution;
  if (std::getenv("PIBEM_HYP_GRADIENT_CHECK") != nullptr)
    pcout << "Hypersingular free term: |det C| in ["
          << Utilities::MPI::min(hypersingular_free_term_report[0],
                                 mpi_communicator)
          << ", "
          << Utilities::MPI::max(hypersingular_free_term_report[1],
                                 mpi_communicator)
          << "], max |C| "
          << Utilities::MPI::max(hypersingular_free_term_report[2],
                                 mpi_communicator)
          << ", max |b| "
          << Utilities::MPI::max(hypersingular_free_term_report[3],
                                 mpi_communicator)
          << std::endl;
  pcout << "done computing gradients with hypersingular integrals" << std::endl;
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


  cell_it vector_cell = gradient_dh.begin_active(),
          vector_endc = gradient_dh.end();


  for (; vector_cell != vector_endc; ++vector_cell)
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

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));
  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_normals_matrix);


  solver.solve(vector_normals_matrix,
               vector_normals_solution,
               vector_normals_rhs,
               mass_prec);

  vector_constraints.distribute(vector_normals_solution);

  for (const auto i : this_cpu_set)
    {
      const types::global_dof_index original_index = sub_wise_to_original[i];
      std::vector<types::global_dof_index> component_indices(dim);
      double normal_norm_square = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
        {
          component_indices[d] =
            vec_original_to_sub_wise[d * dh.n_dofs() + original_index];
          Assert(vector_this_cpu_set.is_element(component_indices[d]),
                 ExcInternalError());
          const double component = vector_normals_solution[component_indices[d]];
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

template <int dim>
void
BEMProblem<dim>::adaptive_refinement(
  const TrilinosWrappers::MPI::Vector &error_vector)
{
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
