// The program starts with including a bunch
// of include files that we will use in the
// various parts of the program. Most of them
// have been discussed in previous tutorials
// already:



#include "../include/boundary_conditions.h"
#include "mapping_hanging_constraints.h"

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/filtered_iterator.h>

#include "../include/vector_tools_integrate_difference.h"

template <int dim, int spacedim>
class FilteredDataOut : public DataOut<dim, spacedim>
{
public:
  FilteredDataOut(const unsigned int subdomain_id)
    : subdomain_id(subdomain_id)
  {}

  virtual typename DataOut<dim, spacedim>::cell_iterator
  first_cell()
  {
    typename DataOut<dim, spacedim>::active_cell_iterator cell =
      this->dofs->begin_active();
    while ((cell != this->dofs->end()) &&
           (cell->subdomain_id() != subdomain_id))
      {
        ++cell;
      }
    return cell;
  }

  virtual typename DataOut<dim, spacedim>::cell_iterator
  next_cell(const typename DataOut<dim, spacedim>::cell_iterator &old_cell)
  {
    if (old_cell != this->dofs->end())
      {
        const IteratorFilters::SubdomainEqualTo predicate(subdomain_id);
        return ++(
          FilteredIterator<
            typename DataOut<dim, spacedim>::active_cell_iterator>(predicate,
                                                                   old_cell));
      }
    else
      {
        return old_cell;
      }
  }

private:
  const unsigned int subdomain_id;
};

#include "Teuchos_TimeMonitor.hpp"

#include <algorithm>
#include <map>

using Teuchos::RCP;
using Teuchos::Time;
using Teuchos::TimeMonitor;

RCP<Time> PrepareTime = Teuchos::TimeMonitor::getNewTimer("PrepareBEMVectors");
RCP<Time> ErrorsTime  = Teuchos::TimeMonitor::getNewTimer("Errors");
RCP<Time> OutputTimer = Teuchos::TimeMonitor::getNewTimer("Output");
namespace
{
  template <typename VEC>
  void
  vector_shift(VEC &in_vec, double a_scalar)
  {
    for (auto i : in_vec.locally_owned_elements())
      in_vec[i] += a_scalar;
  }
} // namespace
template <int dim>
void
BoundaryConditions<dim>::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Output file name", "result", Patterns::Anything());

  std::vector<std::string> bcond_names = {"Potential",
                                          "Wind function",
                                          "Robin coefficients"};
  std::vector<std::string> defaults_2d = {"x+y", "1; 1", "1; -1; 0"};
  std::vector<std::string> defaults_3d = {"x+y+z", "1; 1; 1", "0; 0; 0"};

  for (unsigned int d = 0; d < 2; ++d)
    {
      for (unsigned int comp = 0; comp < MAX_COMPONENTS; ++comp)
        {
          for (unsigned int cond = 0; cond < bcond_names.size(); ++cond)
            {
              for (unsigned int slot = 0; slot < MAX_CONDITION_SLOTS; ++slot)
                {
                  std::string label = bcond_names[cond] + " " +
                                      Utilities::int_to_string(comp) + " " +
                                      Utilities::int_to_string(slot) + " " +
                                      Utilities::int_to_string(d + 2) + "d";

                  prm.enter_subsection(label);

                  if (!cond)
                    {
                      // potentials are scalars
                      if (!d)
                        {
                          Functions::ParsedFunction<2>::declare_parameters(prm);
                          prm.set("Function expression", defaults_2d[cond]);
                        }
                      else
                        {
                          Functions::ParsedFunction<3>::declare_parameters(prm);
                          prm.set("Function expression", defaults_3d[cond]);
                        }
                    }
                  else
                    {
                      // other are vectors: the wind has one entry per
                      // direction, the Robin coefficients a; b; g three in
                      // every dimension
                      const unsigned int n_entries =
                        cond == BoundaryConditionType::robin ? 3 : d + 2;
                      if (!d)
                        {
                          Functions::ParsedFunction<2>::declare_parameters(
                            prm, n_entries);
                          prm.set("Function expression", defaults_2d[cond]);
                        }
                      else
                        {
                          Functions::ParsedFunction<3>::declare_parameters(
                            prm, n_entries);
                          prm.set("Function expression", defaults_3d[cond]);
                        }
                    }
                  prm.leave_subsection();
                }
            }
        }
    }

  // The subsections of the earlier versions, without component and slot,
  // for the first slot of the first component. Empty unless a file sets
  // them; a file that does takes them instead of Potential 0 0 and
  // Wind function 0 0.
  for (unsigned int d = 2; d <= 3; ++d)
    for (const std::string name : {"Potential", "Wind function"})
      {
        prm.enter_subsection(name + " " + Utilities::int_to_string(d) + "d");
        if (d == 2)
          Functions::ParsedFunction<2>::declare_parameters(
            prm, name == "Potential" ? 1 : 2);
        else
          Functions::ParsedFunction<3>::declare_parameters(
            prm, name == "Potential" ? 1 : 3);
        prm.set("Function expression", "");
        prm.leave_subsection();
      }
}

template <int dim>
void
BoundaryConditions<dim>::parse_parameters(ParameterHandler &prm)
{
  output_file_name = prm.get("Output file name");

  std::vector<std::string> bcond_names = {"Potential",
                                          "Wind function",
                                          "Robin coefficients"};

  for (unsigned int comp = 0; comp < n_components; ++comp)
    {
      for (unsigned int cond = 0; cond < bcond_names.size(); ++cond)
        {
          for (unsigned int slot = 0; slot < MAX_CONDITION_SLOTS; ++slot)
            {
              std::string label = bcond_names[cond] + " " +
                                  Utilities::int_to_string(comp) + " " +
                                  Utilities::int_to_string(slot) + " " +
                                  Utilities::int_to_string(dim) + "d";

              auto pos = comp * MAX_CONDITION_SLOTS + slot;

              prm.enter_subsection(label);
              switch (cond)
                {
                  case BoundaryConditionType::dirichlet:
                    potentials[pos].reset(
                      new Functions::ParsedFunction<dim>(1));
                    potentials[pos]->parse_parameters(prm);
                    break;
                  case BoundaryConditionType::neumann:
                    winds[pos].reset(new Functions::ParsedFunction<dim>(dim));
                    winds[pos]->parse_parameters(prm);
                    break;
                  case BoundaryConditionType::robin:
                    // a; b; g of a phi + b dphi_dn = g, in 2D as in 3D
                    robin_coeffs[pos].reset(
                      new Functions::ParsedFunction<dim>(3));
                    robin_coeffs[pos]->parse_parameters(prm);
                    break;
                  case BoundaryConditionType::invalid:
                  default:
                    break;
                }
              prm.leave_subsection();
            }
        }
    }

  for (const std::string name : {"Potential", "Wind function"})
    {
      prm.enter_subsection(name + " " + Utilities::int_to_string(dim) + "d");
      if (!prm.get("Function expression").empty())
        {
          auto &function = name == "Potential" ? potentials[0] : winds[0];
          function.reset(
            new Functions::ParsedFunction<dim>(name == "Potential" ? 1 : dim));
          function->parse_parameters(prm);
        }
      prm.leave_subsection();
    }
}

template <int dim>
void
BoundaryConditions<dim>::solve_problem(bool reset_matrix)
{
  for (unsigned int i = 0; i < MAX_CONDITION_SLOTS; ++i)
    {
      get_potential(current_component, i).set_time(0);
      get_wind(current_component, i).set_time(0);
      get_robin_coeffs(current_component, i).set_time(0);
    }

  const types::global_dof_index    n_dofs = bem.dh.n_dofs();
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);
  DoFTools::get_subdomain_association(bem.dh, dofs_domain_association);
  this_cpu_set.clear();
  this_cpu_set = bem.this_cpu_set;
  this_cpu_set.compress();

  get_phi().reinit(this_cpu_set, mpi_communicator);
  get_dphi_dn().reinit(this_cpu_set, mpi_communicator);
  tmp_rhs.reinit(this_cpu_set, mpi_communicator);
  bem.robin_matrix_diagonal.reinit(this_cpu_set, mpi_communicator);
  bem.robin_rhs.reinit(this_cpu_set, mpi_communicator);

  pcout << "Computing normal vector" << std::endl;
  if (reset_matrix)
    {
      bem.compute_normals();
    }
  prepare_bem_vectors(tmp_rhs);
  prepare_robin_datastructs(bem.robin_matrix_diagonal, bem.robin_rhs);

  set_cellwise_data(current_component, 0);
  bem.solve(get_phi(),
            get_dphi_dn(),
            tmp_rhs,
            cellwise_potential[0].get(),
            cellwise_wind[0].get(),
            reset_matrix);

  if (!bem.can_determine_phi)
    {
      pcout << "Computing phi shift" << std::endl;
      std::vector<Point<dim>> support_points(n_dofs);
      DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                         bem.dh,
                                                         support_points);
      double shift = 0.0;
      if (this_mpi_process == 0)
        {
          shift = get_potential(current_component, 0)
                    .value(support_points[*bem.this_cpu_set.begin()]) -
                  get_phi()(*bem.this_cpu_set.begin());
        }
      MPI_Bcast(&shift, 1, MPI_DOUBLE, 0, mpi_communicator);
      vector_shift(get_phi(), shift);

      pcout << "Phi shift of " << shift << std::endl;
    }
}

template <int dim>
void
BoundaryConditions<dim>::set_cellwise_data(unsigned int component,
                                           unsigned int part)
{
  // The analytical data: the Galerkin formulation integrates them on the
  // cells, with the function of the slot of each.
  std::vector<const Function<dim> *> potential_pieces, wind_pieces;
  for (unsigned int slot = 0; slot < MAX_CONDITION_SLOTS; ++slot)
    {
      potential_pieces.push_back(&get_potential(component, slot));
      wind_pieces.push_back(&get_wind(component, slot));
    }
  cellwise_potential[part] = std::make_unique<CellwiseFunction<dim>>(
    potential_pieces, comp_dom.manifold2bcondition_slot_map);
  cellwise_wind[part] = std::make_unique<CellwiseFunction<dim>>(
    wind_pieces, comp_dom.manifold2bcondition_slot_map);
}

template <int dim>
void
BoundaryConditions<dim>::solve_complex_problem(bool reset_matrix)
{
  for (unsigned int i = 0; i < MAX_CONDITION_SLOTS; ++i)
    {
      // real parts - current component
      get_potential(current_component, i).set_time(0);
      get_wind(current_component, i).set_time(0);
      get_robin_coeffs(current_component, i).set_time(0);
      // imaginary parts - next component
      get_potential(current_component + 1, i).set_time(0);
      get_wind(current_component + 1, i).set_time(0);
      get_robin_coeffs(current_component + 1, i).set_time(0);
    }

  const types::global_dof_index    n_dofs = bem.dh.n_dofs();
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);
  DoFTools::get_subdomain_association(bem.dh, dofs_domain_association);
  this_cpu_set.clear();
  this_cpu_set = bem.this_cpu_set;
  this_cpu_set.compress();

  // real parts - current component
  get_phi().reinit(this_cpu_set, mpi_communicator);
  get_dphi_dn().reinit(this_cpu_set, mpi_communicator);
  tmp_rhs.reinit(this_cpu_set, mpi_communicator);
  bem.robin_matrix_diagonal.reinit(this_cpu_set, mpi_communicator);
  bem.robin_rhs.reinit(this_cpu_set, mpi_communicator);
  // imaginary parts - next component
  get_phi(current_component + 1).reinit(this_cpu_set, mpi_communicator);
  get_dphi_dn(current_component + 1).reinit(this_cpu_set, mpi_communicator);
  TrilinosWrappers::MPI::Vector tmp_rhs_imag;
  tmp_rhs_imag.reinit(this_cpu_set, mpi_communicator);
  bem.robin_matrix_diagonal_imag.reinit(this_cpu_set, mpi_communicator);
  bem.robin_rhs_imag.reinit(this_cpu_set, mpi_communicator);

  if (reset_matrix)
    {
      pcout << "Computing normal vector" << std::endl;
      bem.compute_normals();
    }

  // TODO: these calls waste the retrieval of the support points
  // real parts - current component
  prepare_bem_vectors(tmp_rhs);
  // imaginary parts - next component
  set_current_phi_component(current_component + 1);
  prepare_bem_vectors(tmp_rhs_imag);
  set_current_phi_component(current_component - 1);

  prepare_robin_datastructs(bem.robin_matrix_diagonal,
                            bem.robin_matrix_diagonal_imag,
                            bem.robin_rhs,
                            bem.robin_rhs_imag);

  set_cellwise_data(current_component, 0);
  set_cellwise_data(current_component + 1, 1);
  bem.solve(get_phi(),
            get_phi(current_component + 1),
            get_dphi_dn(),
            get_dphi_dn(current_component + 1),
            tmp_rhs,
            tmp_rhs_imag,
            {{cellwise_potential[0].get(), cellwise_potential[1].get()}},
            {{cellwise_wind[0].get(), cellwise_wind[1].get()}},
            reset_matrix);

  if (!bem.can_determine_phi)
    {
      pcout << "Computing phi shift of real part" << std::endl;
      // TODO: it seems a bit wasteful to retrieve all n_dofs support pts
      std::vector<Point<dim>> support_points(n_dofs);
      DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                         bem.dh,
                                                         support_points);
      double shift = 0.0;
      if (this_mpi_process == 0)
        {
          shift = get_potential(current_component, 0)
                    .value(support_points[*bem.this_cpu_set.begin()]) -
                  get_phi()(*bem.this_cpu_set.begin());
        }
      MPI_Bcast(&shift, 1, MPI_DOUBLE, 0, mpi_communicator);
      vector_shift(get_phi(), shift);

      pcout << "Phi shift of real part : " << shift << std::endl;
      pcout << "Computing phi shift of imaginary part" << std::endl;
      shift = 0.0;
      if (this_mpi_process == 0)
        {
          shift = get_potential(current_component + 1, 0)
                    .value(support_points[*bem.this_cpu_set.begin()]) -
                  get_phi(current_component + 1)(*bem.this_cpu_set.begin());
        }
      MPI_Bcast(&shift, 1, MPI_DOUBLE, 0, mpi_communicator);
      vector_shift(get_phi(current_component + 1), shift);

      pcout << "Phi shift of imaginary part : " << shift << std::endl;
    }
}

template <int dim>
void
BoundaryConditions<dim>::prepare_bem_vectors(TrilinosWrappers::MPI::Vector &rhs)
{
  Teuchos::TimeMonitor LocalTimer(*PrepareTime);

  rhs           = 0;
  get_phi()     = 0;
  get_dphi_dn() = 0;

  const types::global_dof_index n_dofs = bem.dh.n_dofs();
  std::vector<Point<dim>>       support_points(n_dofs);
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.dh,
                                                     support_points);

  std::vector<Point<dim>> vec_support_points(bem.gradient_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.gradient_dh,
                                                     vec_support_points);


  const unsigned int                   dofs_per_cell = bem.fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  FEValues<dim - 1, dim>               fe_v(*bem.mapping,
                              *bem.fe,
                              *bem.quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);

  // pcout << "local_dof_indices elements: " << local_dof_indices.size()
  //       << std::endl;
  // pcout << "phi elements: " << get_phi().size() << std::endl;
  // pcout << "dphi_dn elements: " << get_phi().size() << std::endl;
  // pcout << "support_points elements: " << support_points.size() << std::endl;
  // pcout << "vec_support_points elements: " << vec_support_points.size()

  Vector<double> coeffs(3);
  for (const auto &cell : bem.dh.active_cell_iterators())
    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int j = 0; j < bem.fe->dofs_per_cell; ++j)
        {
          if (this_cpu_set.is_element(local_dof_indices[j]))
            {
              auto slot =
                comp_dom.manifold2bcondition_slot_map[cell->manifold_id()];

              bool dirichlet = bem.dirichlet_nodes(local_dof_indices[j]) == 1;
              if (dirichlet)
                {
                  rhs(local_dof_indices[j]) =
                    get_potential(current_component, slot)
                      .value(support_points[local_dof_indices[j]]);
                  get_phi()(local_dof_indices[j]) = rhs(local_dof_indices[j]);
                }
              else
                {
                  bool neumann = bem.neumann_nodes(local_dof_indices[j]) == 1;
                  if (neumann)
                    {
                      Vector<double> imposed_pot_grad(dim);
                      get_wind(current_component, slot)
                        .vector_value(support_points[local_dof_indices[j]],
                                      imposed_pot_grad);

                      double tmp_dphi_dn = 0;
                      double normy       = 0;

                      for (unsigned int d = 0; d < dim; ++d)
                        {
                          types::global_dof_index dummy =
                            bem.sub_wise_to_original[local_dof_indices[j]];
                          types::global_dof_index vec_index =
                            bem.vec_original_to_sub_wise
                              [bem.gradient_dh.n_dofs() / dim * d + dummy];

                          Assert(
                            bem.vector_this_cpu_set.is_element(vec_index),
                            ExcMessage(
                              "vector cpu set and cpu set are inconsistent"));

                          tmp_dphi_dn += imposed_pot_grad[d] *
                                         bem.vector_normals_solution[vec_index];
                          normy += bem.vector_normals_solution[vec_index] *
                                   bem.vector_normals_solution[vec_index];
                        }

                      rhs(local_dof_indices[j])           = tmp_dphi_dn;
                      get_dphi_dn()(local_dof_indices[j]) = tmp_dphi_dn;
                    }
                  else
                    {
                      rhs(local_dof_indices[j])           = 0;
                      get_phi()(local_dof_indices[j])     = 0;
                      get_dphi_dn()(local_dof_indices[j]) = 0;
                    }
                }
            }
        }
    }
}

template <int dim>
void
BoundaryConditions<dim>::prepare_robin_datastructs(
  TrilinosWrappers::MPI::Vector &robin_matrix_diagonal,
  TrilinosWrappers::MPI::Vector &robin_rhs)
{
  Teuchos::TimeMonitor LocalTimer(*PrepareTime);

  const types::global_dof_index n_dofs = bem.dh.n_dofs();
  std::vector<Point<dim>>       support_points(n_dofs);
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.dh,
                                                     support_points);

  const unsigned int                   dofs_per_cell = bem.fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Vector<double>                    coeffs(3);
  std::set<types::global_dof_index> processed;
  for (const auto &cell : bem.dh.active_cell_iterators())
    {
      if (comp_dom.manifold2bcondition_map[cell->manifold_id()] ==
          BoundaryConditionType::robin)
        {
          auto slot =
            comp_dom.manifold2bcondition_slot_map[cell->manifold_id()];
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              if (this_cpu_set.is_element(local_dof_indices[j]) &&
                  !processed.count(local_dof_indices[j]))
                {
                  // evaluate robin coefficients
                  // coeffs(0) * phi + coeffs(1) * dphi_dn = coeffs(2)
                  get_robin_coeffs(current_component, slot)
                    .vector_value(support_points[local_dof_indices[j]], coeffs);
                  robin_matrix_diagonal(local_dof_indices[j]) =
                    coeffs(0) / coeffs(1);
                  robin_rhs(local_dof_indices[j]) = coeffs(2) / coeffs(1);

                  processed.insert(local_dof_indices[j]);
                }
            }
        }
    }
}

template <int dim>
void
BoundaryConditions<dim>::prepare_robin_datastructs(
  TrilinosWrappers::MPI::Vector &robin_matrix_diagonal,
  TrilinosWrappers::MPI::Vector &robin_matrix_diagonal_imag,
  TrilinosWrappers::MPI::Vector &robin_rhs,
  TrilinosWrappers::MPI::Vector &robin_rhs_imag)
{
  Teuchos::TimeMonitor LocalTimer(*PrepareTime);

  const types::global_dof_index n_dofs = bem.dh.n_dofs();
  std::vector<Point<dim>>       support_points(n_dofs);
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.dh,
                                                     support_points);

  const unsigned int                   dofs_per_cell = bem.fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  Vector<double>                    coeffs(3);
  Vector<double>                    coeffs_imag(3);
  std::set<types::global_dof_index> processed;
  for (const auto &cell : bem.dh.active_cell_iterators())
    {
      if (comp_dom.manifold2bcondition_map[cell->manifold_id()] ==
          BoundaryConditionType::robin)
        {
          auto slot =
            comp_dom.manifold2bcondition_slot_map[cell->manifold_id()];
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              if (this_cpu_set.is_element(local_dof_indices[j]) &&
                  !processed.count(local_dof_indices[j]))
                {
                  // evaluate robin coefficients
                  // coeffs(0) * phi + coeffs(1) * dphi_dn = coeffs(2)
                  get_robin_coeffs(current_component, slot)
                    .vector_value(support_points[local_dof_indices[j]], coeffs);
                  get_robin_coeffs(current_component + 1, slot)
                    .vector_value(support_points[local_dof_indices[j]],
                                  coeffs_imag);
                  std::complex<double> c0(coeffs(0), coeffs_imag(0));
                  std::complex<double> c1(coeffs(1), coeffs_imag(1));
                  std::complex<double> c2(coeffs(2), coeffs_imag(2));

                  std::complex<double> diag = c0 / c1;
                  std::complex<double> rhs  = c2 / c1;

                  robin_matrix_diagonal(local_dof_indices[j]) = std::real(diag);
                  robin_matrix_diagonal_imag(local_dof_indices[j]) =
                    std::imag(diag);
                  robin_rhs(local_dof_indices[j])      = std::real(rhs);
                  robin_rhs_imag(local_dof_indices[j]) = std::imag(rhs);

                  processed.insert(local_dof_indices[j]);
                }
            }
        }
    }
}

template <int dim>
void
BoundaryConditions<dim>::compute_errors(bool complex, bool current_is_real)
{
  AssertThrow(
    complex || current_is_real,
    ExcMessage(
      "Called compute errors for a real problem, but specified that the current component is imaginary"));

  // A real problem with the data of the first slot: the errors below, of the
  // collocation and of the Galerkin solutions. Complex problems, Robin parts
  // and the other slots: the errors of the boundary conditions by material.
  bool by_material = complex || bem.robin_nodes.linfty_norm() > 0;
  for (const auto &pair : comp_dom.manifold2bcondition_slot_map)
    by_material = by_material || pair.second != 0;
  if (by_material)
    {
      compute_errors_by_material(complex, current_is_real);
      return;
    }

  Teuchos::TimeMonitor LocalTimer(*ErrorsTime);

  TrilinosWrappers::MPI::Vector        &phi       = get_phi();
  TrilinosWrappers::MPI::Vector        &dphi_dn   = get_dphi_dn();
  const Functions::ParsedFunction<dim> &potential = get_potential(current_component, 0);
  const Functions::ParsedFunction<dim> &wind      = get_wind(current_component, 0);

  // The mixed solver has its own block operator and row residuals. The
  // Dirichlet-only operator below is not its operator (and may be empty).
  bool mixed_galerkin_problem = false;
  if (bem.boundary_integral_formulation == "galerkin")
    {
      mixed_galerkin_problem = std::getenv("PIBEM_FORCE_MIXED") != nullptr;
      unsigned int locally_dirichlet = 0;
      for (const auto i : bem.this_cpu_set)
        if (bem.dirichlet_nodes[i] > 0.5)
          ++locally_dirichlet;
      mixed_galerkin_problem = mixed_galerkin_problem ||
        Utilities::MPI::sum(locally_dirichlet, mpi_communicator) != bem.dh.n_dofs();
    }

  // We still need to communicate our results to compute the errors.
  bem.gradient_normal_from_flux = false;
  if (bem.gradient_recovery == "hypersingular")
    bem.compute_gradients_hypersingular(phi, dphi_dn);
  else if (bem.gradient_recovery == "weak")
    // The trace is prescribed data here, so the weak form can use it
    // directly instead of its interpolant. The tangential gradient is
    // projected alone, with the geometric curvature term, and the full field
    // is then the L2 projection of that part plus lambda_h n.
    bem.compute_gradients_weak(phi, dphi_dn, &potential, true);
  else if (bem.gradient_recovery == "weak_joint")
    // Earlier variant, kept for comparison: one projection of
    // lambda n + grad_S phi, normal component replaced at the nodes. Worse
    // at every mesh tried: the normal part is interpolated rather than
    // projected, and the projection error of lambda n leaks into the
    // tangential components.
    bem.compute_gradients_weak(phi, dphi_dn, &potential, false);
  else if (bem.gradient_recovery == "weak_discrete")
    bem.compute_gradients_weak(phi, dphi_dn);
  else if (bem.gradient_recovery == "hypersingular_weak")
    // Galerkin form of the gradient representation, analytical trace.
    bem.compute_gradients_hypersingular_weak(
      phi, dphi_dn, potential, wind, false);
  else if (bem.gradient_recovery == "hypersingular_weak_discrete")
    // Same, with the discrete trace in the tangential part: the case of a
    // trace that is computed rather than given.
    bem.compute_gradients_hypersingular_weak(
      phi, dphi_dn, potential, wind, true);
  else
    bem.compute_gradients(phi, dphi_dn);
  Vector<double> localized_gradient_solution(
    bem.get_vector_gradients_solution()); // vector_gradients_solution
  Vector<double> localized_phi(phi);
  Vector<double> localized_dphi_dn(dphi_dn);
  Vector<double> localised_normals(bem.vector_normals_solution);
  std::vector<Point<dim>> support_points(bem.dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.dh,
                                                     support_points);

  std::vector<Vector<double>> exact_gradient_values(bem.dh.n_dofs(),
                                                     Vector<double>(dim));
  wind.vector_value_list(support_points, exact_gradient_values);
  Vector<double> exact_dphi_dn(bem.dh.n_dofs());
  for (types::global_dof_index scalar_i = 0;
       scalar_i < bem.dh.n_dofs();
       ++scalar_i)
    {
      const types::global_dof_index original_i =
        bem.sub_wise_to_original[scalar_i];
      for (unsigned int d = 0; d < dim; ++d)
        exact_dphi_dn[scalar_i] +=
          localised_normals
            [bem.vec_original_to_sub_wise[d * bem.dh.n_dofs() + original_i]] *
          exact_gradient_values[scalar_i][d];
    }

  const bool has_cellwise_l2_projection =
    bem.fe->get_name().find("FE_DGQ") == 0;
  // For continuous elements the L2 projection couples neighbouring cells and
  // is computed globally, with the hanging-node constraints.
  const bool has_global_l2_projection = bem.fe->get_name().find("FE_Q<") == 0;
  Vector<double> l2_projected_exact_dphi_dn(bem.dh.n_dofs());
  if (has_global_l2_projection)
    {
      AffineConstraints<double> hanging_node_constraints;
      if (bem.exact_geometry)
        ExactGeometry::make_mapping_hanging_node_constraints(
          *bem.mapping, bem.dh, hanging_node_constraints);
      else
        DoFTools::make_hanging_node_constraints(bem.dh, hanging_node_constraints);
      hanging_node_constraints.close();
      DynamicSparsityPattern dsp(bem.dh.n_dofs());
      DoFTools::make_sparsity_pattern(bem.dh,
                                      dsp,
                                      hanging_node_constraints,
                                      false);
      SparsityPattern sparsity;
      sparsity.copy_from(dsp);
      SparseMatrix<double> mass(sparsity);
      Vector<double>       projection_rhs(bem.dh.n_dofs());

      const QGauss<dim - 1> projection_quadrature(
        std::max(8U, 2U * (2U * bem.fe->degree + 3U)));
      FEValues<dim - 1, dim> projection_values(
        *bem.mapping,
        *bem.fe,
        projection_quadrature,
        update_values | update_quadrature_points |
          update_normal_vectors | update_JxW_values);
      std::vector<Vector<double>> exact_gradient_at_quadrature(
        projection_quadrature.size(), Vector<double>(dim));
      std::vector<types::global_dof_index> cell_indices(
        bem.fe->dofs_per_cell);
      FullMatrix<double> local_mass(bem.fe->dofs_per_cell,
                                    bem.fe->dofs_per_cell);
      Vector<double>     local_rhs(bem.fe->dofs_per_cell);
      for (const auto &cell : bem.dh.active_cell_iterators())
        {
          projection_values.reinit(cell);
          wind.vector_value_list(projection_values.get_quadrature_points(),
                                 exact_gradient_at_quadrature);
          local_mass = 0.0;
          local_rhs  = 0.0;
          for (unsigned int q = 0; q < projection_quadrature.size(); ++q)
            {
              double normal_trace = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                normal_trace += exact_gradient_at_quadrature[q][d] *
                                projection_values.normal_vector(q)[d];
              for (unsigned int i = 0; i < bem.fe->dofs_per_cell; ++i)
                {
                  local_rhs[i] += projection_values.shape_value(i, q) *
                                  normal_trace * projection_values.JxW(q);
                  for (unsigned int j = 0; j < bem.fe->dofs_per_cell; ++j)
                    local_mass(i, j) +=
                      projection_values.shape_value(i, q) *
                      projection_values.shape_value(j, q) *
                      projection_values.JxW(q);
                }
            }
          cell->get_dof_indices(cell_indices);
          hanging_node_constraints.distribute_local_to_global(
            local_mass, local_rhs, cell_indices, mass, projection_rhs);
        }
      SolverControl projection_control(10 * bem.dh.n_dofs(),
                                       1e-14 * projection_rhs.l2_norm());
      SolverCG<Vector<double>>          projection_solver(projection_control);
      PreconditionSSOR<SparseMatrix<double>> projection_preconditioner;
      projection_preconditioner.initialize(mass, 1.2);
      projection_solver.solve(mass,
                              l2_projected_exact_dphi_dn,
                              projection_rhs,
                              projection_preconditioner);
      hanging_node_constraints.distribute(l2_projected_exact_dphi_dn);
    }
  const bool has_l2_projection =
    has_cellwise_l2_projection || has_global_l2_projection;
  if (has_cellwise_l2_projection)
    {
      const QGauss<dim - 1> projection_quadrature(
        std::max(8U, 2U * (2U * bem.fe->degree + 3U)));
      FEValues<dim - 1, dim> projection_values(
        *bem.mapping,
        *bem.fe,
        projection_quadrature,
        update_values | update_quadrature_points |
          update_normal_vectors | update_JxW_values);
      std::vector<Vector<double>> exact_gradient_at_quadrature(
        projection_quadrature.size(), Vector<double>(dim));
      std::vector<types::global_dof_index> cell_indices(
        bem.fe->dofs_per_cell);
      for (const auto &cell : bem.dh.active_cell_iterators())
        {
          projection_values.reinit(cell);
          wind.vector_value_list(projection_values.get_quadrature_points(),
                                 exact_gradient_at_quadrature);
          FullMatrix<double> local_mass(bem.fe->dofs_per_cell,
                                        bem.fe->dofs_per_cell);
          Vector<double> local_rhs(bem.fe->dofs_per_cell);
          for (unsigned int q = 0; q < projection_quadrature.size(); ++q)
            {
              double normal_trace = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                normal_trace += exact_gradient_at_quadrature[q][d] *
                                projection_values.normal_vector(q)[d];
              for (unsigned int i = 0; i < bem.fe->dofs_per_cell; ++i)
                {
                  local_rhs[i] += projection_values.shape_value(i, q) *
                                  normal_trace * projection_values.JxW(q);
                  for (unsigned int j = 0; j < bem.fe->dofs_per_cell; ++j)
                    local_mass(i, j) +=
                      projection_values.shape_value(i, q) *
                      projection_values.shape_value(j, q) *
                      projection_values.JxW(q);
                }
            }
          FullMatrix<double> inverse_mass(local_mass);
          inverse_mass.gauss_jordan();
          Vector<double> local_projection(bem.fe->dofs_per_cell);
          inverse_mass.vmult(local_projection, local_rhs);
          cell->get_dof_indices(cell_indices);
          for (unsigned int i = 0; i < bem.fe->dofs_per_cell; ++i)
            l2_projected_exact_dphi_dn[cell_indices[i]] =
              local_projection[i];
        }
    }

  TrilinosWrappers::MPI::Vector parallel_exact_dphi_dn(bem.this_cpu_set,
                                                        mpi_communicator);
  const bool exact_residual_uses_l2_projection =
    bem.boundary_integral_formulation == "galerkin" &&
    has_l2_projection;
  if (exact_residual_uses_l2_projection)
    for (const auto i : bem.this_cpu_set)
      parallel_exact_dphi_dn[i] = l2_projected_exact_dphi_dn[i];
  else
    for (const auto i : bem.this_cpu_set)
      parallel_exact_dphi_dn[i] = exact_dphi_dn[i];
  parallel_exact_dphi_dn.compress(VectorOperation::insert);

  TrilinosWrappers::MPI::Vector exact_residual(bem.this_cpu_set,
                                                mpi_communicator);
  ConstrainedOperator<TrilinosWrappers::MPI::Vector, BEMProblem<dim>>
    constrained_operator(bem,
                         bem.constraints,
                         bem.constr_cpu_set,
                         mpi_communicator);
  if (!mixed_galerkin_problem)
    {
      constrained_operator.vmult(exact_residual, parallel_exact_dphi_dn);
      exact_residual.add(-1.0, bem.system_rhs);
    }
  const double exact_residual_l2   = exact_residual.l2_norm();
  const double exact_residual_linf = exact_residual.linfty_norm();
  const double rhs_l2              = bem.system_rhs.l2_norm();
  const double relative_exact_residual =
    rhs_l2 > 0.0 ? exact_residual_l2 / rhs_l2 : exact_residual_l2;
  const bool sphere_box_grid =
    comp_dom.input_grid_name.find("sphere_box") != std::string::npos;

  std::array<double, 2> component_exact_residual_l2{};
  std::array<double, 2> component_rhs_l2{};
  if (bem.boundary_integral_formulation == "galerkin" &&
      (sphere_box_grid ||
       (bem.galerkin_component_rhs_diagnostics &&
        bem.calderon_dual_component_areas.size() == 2)))
    {
      std::array<double, 2> local_residual_squares{};
      std::array<double, 2> local_rhs_squares{};
      std::vector<types::global_dof_index> cell_indices(
        bem.fe->dofs_per_cell);
      for (const auto &cell : bem.dh.active_cell_iterators())
        {
          if (cell->subdomain_id() != this_mpi_process)
            continue;
          const unsigned int component = sphere_box_grid ?
            (cell->material_id() == 1 ? 0 : 1) :
            bem.calderon_dual_cell_components[cell->active_cell_index()];
          AssertIndexRange(component, 2);
          cell->get_dof_indices(cell_indices);
          for (const auto dof : cell_indices)
            {
              local_residual_squares[component] +=
                exact_residual[dof] * exact_residual[dof];
              local_rhs_squares[component] +=
                bem.system_rhs[dof] * bem.system_rhs[dof];
            }
        }
      for (unsigned int component = 0; component < 2; ++component)
        {
          component_exact_residual_l2[component] = std::sqrt(
            Utilities::MPI::sum(local_residual_squares[component],
                                mpi_communicator));
          component_rhs_l2[component] = std::sqrt(
            Utilities::MPI::sum(local_rhs_squares[component],
                                mpi_communicator));
        }
    }

  std::array<double, dim> coordinate_patch_relative_residuals{};
  if (bem.kernel_type == "laplace" && bem.solution_method == "Direct" &&
      bem.boundary_integral_formulation == "collocation")
    for (unsigned int d = 0; d < dim; ++d)
      {
        TrilinosWrappers::MPI::Vector coordinate(bem.this_cpu_set,
                                                  mpi_communicator);
        TrilinosWrappers::MPI::Vector normal_component(bem.this_cpu_set,
                                                       mpi_communicator);
        for (const auto i : bem.this_cpu_set)
          {
            coordinate[i] = support_points[i][d];
            const types::global_dof_index original_i =
              bem.sub_wise_to_original[i];
            normal_component[i] =
              localised_normals[bem.vec_original_to_sub_wise
                                  [d * bem.dh.n_dofs() + original_i]];
          }
        coordinate.compress(VectorOperation::insert);
        normal_component.compress(VectorOperation::insert);

        TrilinosWrappers::MPI::Vector patch_residual(bem.this_cpu_set,
                                                      mpi_communicator);
        TrilinosWrappers::MPI::Vector patch_rhs(bem.this_cpu_set,
                                                 mpi_communicator);
        TrilinosWrappers::MPI::Vector jump_coordinate(coordinate);
        bem.apply_single_layer(patch_residual, normal_component);
        bem.neumann_matrix.vmult(patch_rhs, coordinate);
        bem.apply_jump_term(jump_coordinate, coordinate);
        patch_rhs += jump_coordinate;
        patch_residual.add(-1.0, patch_rhs);
        const double patch_rhs_norm = patch_rhs.l2_norm();
        coordinate_patch_relative_residuals[d] =
          patch_rhs_norm > 0.0 ? patch_residual.l2_norm() / patch_rhs_norm :
                                 patch_residual.l2_norm();
      }

  // The two energy figures are computed by every rank, because both reduce
  // across the communicator: the matrix-vector product and the dot product
  // below, and the MPI sum inside compute_energy_norm_error. Running them
  // inside the rank-0 block that follows would leave rank 0 waiting in a
  // collective the other ranks never enter, and MPI would pair that call
  // with whatever collective the others reach next - returning a number
  // built from mismatched reductions rather than hanging outright.
  double algebraic_energy_square = 0.0;
  // The mixed Galerkin problem frees the single-layer matrix of the
  // Dirichlet path.
  const bool has_algebraic_energy_distance = !mixed_galerkin_problem &&
    exact_residual_uses_l2_projection && bem.dirichlet_matrix.m() > 0;
  if (has_algebraic_energy_distance)
    {
      // The energy distance between the Galerkin solution and the best L2
      // approximation in the same space. Unlike the energy norm of the error
      // it needs no new quadrature: V is already assembled, so this is one
      // matrix-vector product, exact to the accuracy of the operator itself,
      // and non-negative because V is positive definite. It does not measure
      // the distance to the exact flux; it measures how close the method
      // gets to the best its own space allows.
      TrilinosWrappers::MPI::Vector difference(bem.this_cpu_set,
                                               mpi_communicator);
      difference = parallel_exact_dphi_dn;
      difference.add(-1.0, dphi_dn);
      TrilinosWrappers::MPI::Vector image(bem.this_cpu_set, mpi_communicator);
      bem.dirichlet_matrix.vmult(image, difference);
      algebraic_energy_square = image * difference;
    }
  const double energy_norm_error =
    bem.compute_energy_norm ? bem.compute_energy_norm_error(dphi_dn, wind) :
                              0.0;

  // Galerkin orthogonality check. The discrete solution is the V-orthogonal
  // projection of the exact flux, so the best L2 projection, the discrete
  // solution and the exact flux form a right triangle in the energy inner
  // product. The three sides are measured independently here - two of them
  // above - so agreement is a genuine test of the energy machinery rather
  // than an identity it satisfies by construction.
  if (bem.compute_energy_norm && exact_residual_uses_l2_projection &&
      std::getenv("PIBEM_ENERGY_SELFTEST") != nullptr)
    {
      const double projection_energy =
        bem.compute_energy_norm_error(parallel_exact_dphi_dn, wind);

      // Isolate whether the right-hand side is the one V induces. The
      // Galerkin equations make lambda_h the V-orthogonal projection only if
      // b(mu) equals <V lambda, mu>. The cross term <V lambda, lambda_I> is
      // recovered from three energies, each measured independently, and
      // compared against the assembled b - so a mismatch indicts the
      // right-hand side rather than the operator.
      {
        TrilinosWrappers::MPI::Vector zero(bem.this_cpu_set, mpi_communicator);
        zero = 0.0;
        const double exact_energy = bem.compute_energy_norm_error(zero, wind);
        TrilinosWrappers::MPI::Vector projection_image(bem.this_cpu_set,
                                                       mpi_communicator);
        bem.dirichlet_matrix.vmult(projection_image, parallel_exact_dphi_dn);
        const double projection_square =
          projection_image * parallel_exact_dphi_dn;
        const double cross_from_energies =
          0.5 * (exact_energy * exact_energy + projection_square -
                 projection_energy * projection_energy);
        const double cross_from_rhs = bem.system_rhs * parallel_exact_dphi_dn;

        // The sharpest form of the same test. Taking the zero function as the
        // comparison member of the space, orthogonality reads
        //     |lambda|^2 = |lambda - lambda_h|^2 + |lambda_h|^2,
        // and because the energy functional is exactly quadratic the common
        // term <V lambda, lambda> cancels out of the defect, so the quadrature
        // error in it does not enter. Printed at full precision because the
        // defect lives many digits below the terms themselves.
        TrilinosWrappers::MPI::Vector discrete_image(bem.this_cpu_set,
                                                     mpi_communicator);
        bem.dirichlet_matrix.vmult(discrete_image, dphi_dn);
        const double discrete_square = discrete_image * dphi_dn;
        const double defect = exact_energy * exact_energy -
                              energy_norm_error * energy_norm_error -
                              discrete_square;
        const std::streamsize saved = pcout.get_stream().precision(17);
        pcout << "Orthogonality defect: |lambda|^2 "
              << exact_energy * exact_energy << ", |lambda-lambda_h|^2 "
              << energy_norm_error * energy_norm_error << ", |lambda_h|^2 "
              << discrete_square << ", defect " << defect << std::endl;
        pcout.get_stream().precision(saved);
        pcout << "RHS consistency: <V lambda, lambda_I> from energies "
              << cross_from_energies << ", from the assembled b "
              << cross_from_rhs << ", relative difference "
              << std::abs(cross_from_energies - cross_from_rhs) /
                   std::max(std::abs(cross_from_rhs), 1e-30)
              << std::endl
              << "  |lambda|_V " << exact_energy << ", |lambda_I|_V "
              << std::sqrt(std::max(projection_square, 0.0)) << std::endl;
      }
      const double predicted = std::sqrt(energy_norm_error * energy_norm_error +
                                         std::max(algebraic_energy_square, 0.0));
      pcout << "Galerkin orthogonality: |lambda - lambda_I|_V measured "
            << projection_energy << ", predicted from the other two sides "
            << predicted << ", relative difference "
            << std::abs(projection_energy - predicted) /
                 std::max(projection_energy, 1e-30)
            << std::endl;
    }

  // Vector<double> localised_alpha(bem.alpha);
  // We let only the first processor do the error computations
  if (this_mpi_process == 0)
    {
      pcout << "computing boundary errors" << std::endl;
      // for(auto i : localised_alpha.locally_owned_elements())
      //   localised_alpha[i] -= 0.5;

      Vector<double> grad_difference_per_cell(comp_dom.tria.n_active_cells());
      Vector<double> difference_per_cell(comp_dom.tria.n_active_cells());

      // removing this because now solution is translated after BEM solution
      // is given back by bem_problem in solve_problem
      /*
            if (false)//(!have_dirichlet_bc)
              {
                std::vector<double> exact_sol(bem.dh.n_dofs());
                Vector<double> exact_sol_deal(bem.dh.n_dofs());
                potential.value_list(support_points,exact_sol);
                for (auto i : exact_sol_deal.locally_owned_elements())
                  exact_sol_deal[bem.original_to_sub_wise[i]] =
         exact_sol[bem.original_to_sub_wise[i]]; auto exact_mean =
         VectorTools::compute_mean_value(*bem.mapping,bem.dh,QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
         exact_sol_deal, 0); exact_sol_deal.add(-exact_mean); auto my_mean =
         VectorTools::compute_mean_value(*bem.mapping,bem.dh,QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
         localized_phi, 0); localized_phi.add(-my_mean);
                localized_phi.sadd(1.,-1.,exact_sol_deal);
                std::cout<<"Extracting the mean value from phi
         solution"<<std::endl; VectorTools::integrate_difference (*bem.mapping,
         bem.dh, localized_phi, ZeroFunction<dim, double> (1),
                                                   difference_per_cell,
                                                   QGauss<(dim-1)>(2*(2*bem.fe->degree+1)),
                                                   VectorTools::L2_norm);

                phi_max_error = localized_phi.linfty_norm();


              }
            else
              {
      */
      VectorTools::integrate_difference(*bem.mapping,
                                        bem.dh,
                                        localized_phi,
                                        potential,
                                        difference_per_cell,
                                        QGauss<(dim - 1)>(
                                          2 * (2 * bem.fe->degree + 1)),
                                        VectorTools::L2_norm);

      //        }
      VectorTools::integrate_difference(*bem.mapping,
                                        bem.gradient_dh,
                                        localized_gradient_solution,
                                        wind,
                                        grad_difference_per_cell,
                                        QGauss<(dim - 1)>(
                                          2 * (2 * bem.fe->degree + 1)),
                                        VectorTools::L2_norm);
      const double grad_L2_error = grad_difference_per_cell.l2_norm();

      const double L2_error = difference_per_cell.l2_norm();


      Vector<double> vector_gradients_node_error(bem.gradient_dh.n_dofs());
      for (types::global_dof_index scalar_i = 0;
           scalar_i < bem.dh.n_dofs();
           ++scalar_i)
        {
          const types::global_dof_index original_i =
            bem.sub_wise_to_original[scalar_i];
          for (types::global_dof_index d = 0; d < dim; ++d)
            vector_gradients_node_error(
              bem.vec_original_to_sub_wise[d * bem.dh.n_dofs() +
                                           original_i]) =
              exact_gradient_values[scalar_i](d);
        }
      vector_gradients_node_error *= -1.0;
      vector_gradients_node_error.add(1., localized_gradient_solution);

      Vector<double>      phi_node_error(bem.dh.n_dofs());
      std::vector<double> phi_nodes_errs(bem.dh.n_dofs());
      potential.value_list(support_points, phi_nodes_errs);
      for (types::global_dof_index scalar_i = 0;
           scalar_i < bem.dh.n_dofs();
           ++scalar_i)
        phi_node_error[scalar_i] = phi_nodes_errs[scalar_i];

      phi_node_error *= -1.0;
      phi_node_error.add(1., localized_phi);
      const double phi_node_Linf_error = phi_node_error.linfty_norm();

      Vector<double>              dphi_dn_node_error(bem.dh.n_dofs());
      dphi_dn_node_error = exact_dphi_dn;
      dphi_dn_node_error *= -1.0;
      dphi_dn_node_error.add(1., localized_dphi_dn);

      std::array<double, 2> component_nodal_linf{};
      if (sphere_box_grid)
        {
          std::vector<types::global_dof_index> cell_indices(
            bem.fe->dofs_per_cell);
          for (const auto &cell : bem.dh.active_cell_iterators())
            {
              const unsigned int component =
                cell->material_id() == 1 ? 0 : 1;
              cell->get_dof_indices(cell_indices);
              for (const auto dof : cell_indices)
                component_nodal_linf[component] =
                  std::max(component_nodal_linf[component],
                           std::abs(dphi_dn_node_error[dof]));
            }
        }

      Vector<double> recovered_dphi_dn(bem.dh.n_dofs());
      for (types::global_dof_index original_i = 0;
           original_i < bem.dh.n_dofs();
           ++original_i)
        {
          const types::global_dof_index scalar_i =
            bem.original_to_sub_wise[original_i];
          for (unsigned int d = 0; d < dim; ++d)
            {
              const types::global_dof_index vector_i =
                bem.vec_original_to_sub_wise[d * bem.dh.n_dofs() + original_i];
              recovered_dphi_dn[scalar_i] +=
                localised_normals[vector_i] *
                localized_gradient_solution[vector_i];
            }
        }
      Vector<double> recovered_dphi_dn_error(recovered_dphi_dn);
      recovered_dphi_dn_error.add(-1.0, exact_dphi_dn);

      // Separate the BEM error from the error introduced by interpolating the
      // analytical normal trace in the scalar finite element space.
      const QGauss<dim - 1> dphi_dn_error_quadrature(
        2 * (2 * bem.fe->degree + 1));
      FEValues<dim - 1, dim> dphi_dn_error_fe_values(
        *bem.mapping,
        *bem.fe,
        dphi_dn_error_quadrature,
        update_values | update_quadrature_points | update_normal_vectors |
          update_JxW_values);
      std::vector<double> numerical_dphi_dn(
        dphi_dn_error_quadrature.size());
      std::vector<double> interpolated_exact_dphi_dn(
        dphi_dn_error_quadrature.size());
      std::vector<double> projected_exact_dphi_dn(
        dphi_dn_error_quadrature.size());
      std::vector<Vector<double>> exact_gradient_at_quadrature(
        dphi_dn_error_quadrature.size(), Vector<double>(dim));
      Vector<double> true_dphi_dn_difference_per_cell(
        comp_dom.tria.n_active_cells());
      Vector<double> interpolation_difference_per_cell(
        comp_dom.tria.n_active_cells());
      Vector<double> projection_difference_per_cell(
        comp_dom.tria.n_active_cells());
      double true_dphi_dn_error_squared          = 0.0;
      double dphi_dn_interpolation_error_squared = 0.0;
      double projection_error_squared            = 0.0;
      double exact_dphi_dn_norm_squared          = 0.0;
      std::array<double, 2> component_true_error_squared{};
      std::array<double, 2> component_projection_error_squared{};
      std::array<double, 2> component_exact_norm_squared{};
      std::array<double, 2> component_true_linf{};
      std::array<Point<dim>, 2> component_true_linf_point{};
      std::map<unsigned int, double> material_true_error_squared;
      std::map<unsigned int, double> material_projection_error_squared;
      std::map<unsigned int, double> material_exact_norm_squared;
      std::map<unsigned int, double> material_true_linf;
      std::map<unsigned int, Point<dim>> material_true_linf_point;
      std::array<std::array<double, 3>, 2> gap_true_error_squared{};
      std::array<std::array<double, 3>, 2> gap_projection_error_squared{};
      std::array<std::array<double, 3>, 2> gap_exact_norm_squared{};
      std::array<double, 3> cube_bottom_true_error_squared{};
      std::array<double, 3> cube_bottom_projection_error_squared{};
      std::array<double, 3> cube_bottom_exact_norm_squared{};
      std::array<double, 3> cube_bottom_signed_error{};
      std::array<double, 3> cube_bottom_area{};
      std::array<std::array<double, 5>, 3> cube_bottom_radial_error_squared{};
      std::array<std::array<double, 5>, 3> cube_bottom_radial_exact_norm_squared{};
      std::array<std::array<double, 5>, 3> cube_bottom_radial_signed_error{};
      std::array<std::array<double, 5>, 3> cube_bottom_radial_area{};
      std::array<double, 4> cube_bottom_edge_error_squared{};
      std::array<double, 4> cube_bottom_edge_exact_norm_squared{};
      std::array<double, 4> cube_bottom_edge_signed_error{};
      std::array<double, 4> cube_bottom_edge_area{};
      std::array<double, 3> cube_bottom_alignment_error_squared{};
      std::array<double, 3> cube_bottom_alignment_exact_norm_squared{};
      std::array<double, 3> cube_bottom_alignment_signed_error{};
      std::array<double, 3> cube_bottom_alignment_area{};
      struct CellErrorInfo
      {
        double contribution = 0.0;
        unsigned int cell_index = numbers::invalid_unsigned_int;
        unsigned int material_id = numbers::invalid_unsigned_int;
        Point<dim> center;
        double radial_distance = 0.0;
        double edge_distance = 0.0;
      };
      std::vector<CellErrorInfo> top_true_error_cells;
      constexpr double gap_core_radius    = 0.35;
      constexpr double gap_annulus_radius = 1.00;
      constexpr double gap_z_cut          = -0.90;
      constexpr double cube_half_extent   = 6.5;
      const auto gap_region = [&](const Point<dim> &point) {
        if constexpr (dim == 3)
          {
            const double radial_distance =
              std::sqrt(point[0] * point[0] + point[1] * point[1]);
            if (point[2] >= gap_z_cut)
              return 2U;
            if (radial_distance < gap_core_radius)
              return 0U;
            if (radial_distance < gap_annulus_radius)
              return 1U;
            return 2U;
          }
        else
          return 2U;
      };
      const auto radial_bin = [](const double radial_distance) {
        if (radial_distance < 0.15)
          return 0U;
        if (radial_distance < 0.35)
          return 1U;
        if (radial_distance < 0.70)
          return 2U;
        if (radial_distance < 1.00)
          return 3U;
        return 4U;
      };
      const auto edge_bin = [](const double edge_distance) {
        if (edge_distance < 0.25)
          return 0U;
        if (edge_distance < 0.75)
          return 1U;
        if (edge_distance < 1.50)
          return 2U;
        return 3U;
      };
      const auto alignment_bin = [](const Point<dim> &point) {
        if constexpr (dim == 3)
          {
            const double ax = std::abs(point[0]);
            const double ay = std::abs(point[1]);
            const double largest = std::max(ax, ay);
            if (largest < 1e-14)
              return 1U;
            const double balance = std::min(ax, ay) / largest;
            if (balance < 0.25)
              return 0U; // Near the coordinate axes.
            if (balance > 0.75)
              return 2U; // Near the diagonals x=+/-y.
            return 1U;
          }
        else
          return 1U;
      };

      for (const auto &cell : bem.dh.active_cell_iterators())
        {
          dphi_dn_error_fe_values.reinit(cell);
          dphi_dn_error_fe_values.get_function_values(localized_dphi_dn,
                                                       numerical_dphi_dn);
          dphi_dn_error_fe_values.get_function_values(
            exact_dphi_dn, interpolated_exact_dphi_dn);
          if (has_l2_projection)
            dphi_dn_error_fe_values.get_function_values(
              l2_projected_exact_dphi_dn, projected_exact_dphi_dn);
          wind.vector_value_list(
            dphi_dn_error_fe_values.get_quadrature_points(),
            exact_gradient_at_quadrature);

          double cell_true_error_squared          = 0.0;
          double cell_interpolation_error_squared = 0.0;
          double cell_projection_error_squared    = 0.0;
	          const unsigned int boundary_component =
	            cell->material_id() == 1 ? 0 : 1;
	          const unsigned int material_id =
	            static_cast<unsigned int>(cell->material_id());
	          for (unsigned int q = 0; q < dphi_dn_error_quadrature.size(); ++q)
	            {
              double exact_normal_trace = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                exact_normal_trace +=
                  exact_gradient_at_quadrature[q][d] *
                  dphi_dn_error_fe_values.normal_vector(q)[d];
              const double true_error =
                numerical_dphi_dn[q] - exact_normal_trace;
              const double interpolation_error =
                interpolated_exact_dphi_dn[q] - exact_normal_trace;
              const double weight = dphi_dn_error_fe_values.JxW(q);

              cell_true_error_squared += true_error * true_error * weight;
              cell_interpolation_error_squared +=
                interpolation_error * interpolation_error * weight;
              exact_dphi_dn_norm_squared +=
                exact_normal_trace * exact_normal_trace * weight;
              if (sphere_box_grid)
                {
                  const Point<dim> quadrature_point =
                    dphi_dn_error_fe_values.quadrature_point(q);
                  const unsigned int region = gap_region(quadrature_point);
	                  component_true_error_squared[boundary_component] +=
	                    true_error * true_error * weight;
	                  component_exact_norm_squared[boundary_component] +=
	                    exact_normal_trace * exact_normal_trace * weight;
	                  material_true_error_squared[material_id] +=
	                    true_error * true_error * weight;
	                  material_exact_norm_squared[material_id] +=
	                    exact_normal_trace * exact_normal_trace * weight;
	                  gap_true_error_squared[boundary_component][region] +=
	                    true_error * true_error * weight;
	                  gap_exact_norm_squared[boundary_component][region] +=
	                    exact_normal_trace * exact_normal_trace * weight;
	                  if (material_id == 3)
	                    {
	                      cube_bottom_true_error_squared[region] +=
	                        true_error * true_error * weight;
	                      cube_bottom_exact_norm_squared[region] +=
	                        exact_normal_trace * exact_normal_trace * weight;
	                      cube_bottom_signed_error[region] +=
	                        true_error * weight;
	                      cube_bottom_area[region] += weight;
	                      const double radial_distance =
	                        std::sqrt(quadrature_point[0] *
	                                    quadrature_point[0] +
	                                  quadrature_point[1] *
	                                    quadrature_point[1]);
	                      const unsigned int radius_bin =
	                        radial_bin(radial_distance);
	                      cube_bottom_radial_error_squared[region]
	                                                      [radius_bin] +=
	                        true_error * true_error * weight;
	                      cube_bottom_radial_exact_norm_squared[region]
	                                                          [radius_bin] +=
	                        exact_normal_trace * exact_normal_trace * weight;
	                      cube_bottom_radial_signed_error[region]
	                                                     [radius_bin] +=
	                        true_error * weight;
	                      cube_bottom_radial_area[region][radius_bin] +=
	                        weight;
	                      const double edge_distance =
	                        cube_half_extent -
	                        std::max(std::abs(quadrature_point[0]),
	                                 std::abs(quadrature_point[1]));
	                      const unsigned int boundary_bin =
	                        edge_bin(edge_distance);
	                      cube_bottom_edge_error_squared[boundary_bin] +=
	                        true_error * true_error * weight;
	                      cube_bottom_edge_exact_norm_squared
	                        [boundary_bin] +=
	                        exact_normal_trace * exact_normal_trace * weight;
	                      cube_bottom_edge_signed_error[boundary_bin] +=
	                        true_error * weight;
	                      cube_bottom_edge_area[boundary_bin] += weight;
	                      const unsigned int axis_diagonal_bin =
	                        alignment_bin(quadrature_point);
	                      cube_bottom_alignment_error_squared
	                        [axis_diagonal_bin] +=
	                        true_error * true_error * weight;
	                      cube_bottom_alignment_exact_norm_squared
	                        [axis_diagonal_bin] +=
	                        exact_normal_trace * exact_normal_trace * weight;
	                      cube_bottom_alignment_signed_error
	                        [axis_diagonal_bin] += true_error * weight;
	                      cube_bottom_alignment_area[axis_diagonal_bin] +=
	                        weight;
	                    }
	                  if (std::abs(true_error) >
	                      component_true_linf[boundary_component])
                    {
                      component_true_linf[boundary_component] =
                        std::abs(true_error);
                      component_true_linf_point[boundary_component] =
	                        dphi_dn_error_fe_values.quadrature_point(q);
	                    }
	                  if (std::abs(true_error) >
	                      material_true_linf[material_id])
	                    {
	                      material_true_linf[material_id] =
	                        std::abs(true_error);
	                      material_true_linf_point[material_id] =
	                        dphi_dn_error_fe_values.quadrature_point(q);
	                    }
	                }
              if (has_l2_projection)
                {
                  const double projection_error =
                    projected_exact_dphi_dn[q] - exact_normal_trace;
                  cell_projection_error_squared +=
                    projection_error * projection_error * weight;
                  if (sphere_box_grid)
                    {
                      const Point<dim> quadrature_point =
                        dphi_dn_error_fe_values.quadrature_point(q);
                      const unsigned int region =
                        gap_region(quadrature_point);
	                      component_projection_error_squared
	                        [boundary_component] +=
	                          projection_error * projection_error * weight;
	                      material_projection_error_squared[material_id] +=
	                        projection_error * projection_error * weight;
	                      gap_projection_error_squared[boundary_component]
	                                                  [region] +=
	                        projection_error * projection_error * weight;
	                      if (material_id == 3)
	                        cube_bottom_projection_error_squared[region] +=
	                          projection_error * projection_error * weight;
	                    }
	                }
            }

          const unsigned int cell_index = cell->active_cell_index();
          true_dphi_dn_difference_per_cell[cell_index] =
            std::sqrt(cell_true_error_squared);
          interpolation_difference_per_cell[cell_index] =
            std::sqrt(cell_interpolation_error_squared);
          projection_difference_per_cell[cell_index] =
            std::sqrt(cell_projection_error_squared);
          true_dphi_dn_error_squared += cell_true_error_squared;
          dphi_dn_interpolation_error_squared +=
            cell_interpolation_error_squared;
          projection_error_squared += cell_projection_error_squared;
          if (sphere_box_grid)
            {
              CellErrorInfo info;
              info.contribution = std::sqrt(cell_true_error_squared);
              info.cell_index   = cell_index;
              info.material_id  = static_cast<unsigned int>(cell->material_id());
              info.center       = cell->center();
              if constexpr (dim == 3)
                {
                  info.radial_distance =
                    std::sqrt(info.center[0] * info.center[0] +
                              info.center[1] * info.center[1]);
                  info.edge_distance =
                    cube_half_extent -
                    std::max(std::abs(info.center[0]),
                             std::abs(info.center[1]));
                }
              top_true_error_cells.push_back(info);
            }
        }
      if (sphere_box_grid)
        {
          std::sort(top_true_error_cells.begin(),
                    top_true_error_cells.end(),
                    [](const CellErrorInfo &a, const CellErrorInfo &b) {
                      return a.contribution > b.contribution;
                    });
          if (top_true_error_cells.size() > 8)
            top_true_error_cells.resize(8);
        }

      const double true_dphi_dn_L2_error =
        std::sqrt(true_dphi_dn_error_squared);
      const double dphi_dn_interpolation_L2_error =
        std::sqrt(dphi_dn_interpolation_error_squared);
      const double projection_L2_error =
        std::sqrt(projection_error_squared);
      const double exact_dphi_dn_L2_norm =
        std::sqrt(exact_dphi_dn_norm_squared);
      const double relative_true_dphi_dn_L2_error =
        exact_dphi_dn_L2_norm > 0.0 ?
          true_dphi_dn_L2_error / exact_dphi_dn_L2_norm :
          true_dphi_dn_L2_error;

      types::global_dof_index max_dphi_error_index = 0;
      for (types::global_dof_index i = 1; i < bem.dh.n_dofs(); ++i)
        if (std::abs(dphi_dn_node_error[i]) >
            std::abs(dphi_dn_node_error[max_dphi_error_index]))
          max_dphi_error_index = i;
      const double max_dphi_exact =
        localized_dphi_dn[max_dphi_error_index] -
        dphi_dn_node_error[max_dphi_error_index];

      // dphi_dn_node_error.print(std::cout);
      Vector<double> difference_per_cell_2(comp_dom.tria.n_active_cells());
      VectorTools::integrate_difference(
        *bem.mapping,
        bem.dh,
        dphi_dn_node_error,
        dealii::Functions::ZeroFunction<dim, double>(1),
        difference_per_cell_2,
        QGauss<(dim - 1)>(2 * (2 * bem.fe->degree + 1)),
        VectorTools::L2_norm);
      const double dphi_dn_L2_error = difference_per_cell_2.l2_norm();
      Vector<double> recovered_difference_per_cell(
        comp_dom.tria.n_active_cells());
      VectorTools::integrate_difference(
        *bem.mapping,
        bem.dh,
        recovered_dphi_dn_error,
        dealii::Functions::ZeroFunction<dim, double>(1),
        recovered_difference_per_cell,
        QGauss<(dim - 1)>(2 * (2 * bem.fe->degree + 1)),
        VectorTools::L2_norm);
      const double recovered_dphi_dn_L2_error =
        recovered_difference_per_cell.l2_norm();

      const double grad_phi_max_error =
        vector_gradients_node_error.linfty_norm();
      types::global_dof_index gradient_max_node = 0;
      unsigned int gradient_max_component = 0;
      double gradient_max_value = -1.;
      for (types::global_dof_index scalar_i = 0; scalar_i < bem.dh.n_dofs(); ++scalar_i)
        for (unsigned int d = 0; d < dim; ++d)
          {
            const auto original_i = bem.sub_wise_to_original[scalar_i];
            const auto vector_i = bem.vec_original_to_sub_wise[
              d * bem.dh.n_dofs() + original_i];
            if (std::abs(vector_gradients_node_error[vector_i]) > gradient_max_value)
              {
                gradient_max_value = std::abs(vector_gradients_node_error[vector_i]);
                gradient_max_node = scalar_i;
                gradient_max_component = d;
              }
          }
      const types::global_dof_index n_active_cells =
        comp_dom.tria.n_active_cells();
      const types::global_dof_index n_dofs = bem.dh.n_dofs();

      unsigned int max_phi_cell_index = 0;
      unsigned int max_dphi_cell_index = 0;
      unsigned int max_true_dphi_cell_index = 0;
      for (unsigned int cell_index = 1;
           cell_index < comp_dom.tria.n_active_cells();
           ++cell_index)
        {
          if (difference_per_cell[cell_index] >
              difference_per_cell[max_phi_cell_index])
            max_phi_cell_index = cell_index;
          if (difference_per_cell_2[cell_index] >
              difference_per_cell_2[max_dphi_cell_index])
            max_dphi_cell_index = cell_index;
          if (true_dphi_dn_difference_per_cell[cell_index] >
              true_dphi_dn_difference_per_cell[max_true_dphi_cell_index])
            max_true_dphi_cell_index = cell_index;
        }

      Point<dim>   max_phi_cell_center;
      Point<dim>   max_dphi_cell_center;
      Point<dim>   max_true_dphi_cell_center;
      unsigned int max_phi_material_id  = numbers::invalid_unsigned_int;
      unsigned int max_dphi_material_id = numbers::invalid_unsigned_int;
      unsigned int max_true_dphi_material_id =
        numbers::invalid_unsigned_int;
      for (const auto &cell : comp_dom.tria.active_cell_iterators())
        {
          if (cell->active_cell_index() == max_phi_cell_index)
            {
              max_phi_cell_center = cell->center();
              max_phi_material_id =
                static_cast<unsigned int>(cell->material_id());
            }
          if (cell->active_cell_index() == max_dphi_cell_index)
            {
              max_dphi_cell_center = cell->center();
              max_dphi_material_id =
                static_cast<unsigned int>(cell->material_id());
            }
          if (cell->active_cell_index() == max_true_dphi_cell_index)
            {
              max_true_dphi_cell_center = cell->center();
              max_true_dphi_material_id =
                static_cast<unsigned int>(cell->material_id());
            }
        }

      pcout << "   Number of active cells:       " << n_active_cells
            << std::endl
            << "   Number of degrees of freedom: " << n_dofs << std::endl;

      pcout << "Phi Nodes error L_inf norm: " << phi_node_Linf_error
            << std::endl;
      pcout << "Phi Cells maximum L2 contribution: "
            << difference_per_cell.linfty_norm() << std::endl;
      pcout << "Phi Cells maximum at cell " << max_phi_cell_index
            << ", material " << max_phi_material_id << ", center "
            << max_phi_cell_center << std::endl;
      pcout << "Phi Cells error L_2 norm: " << L2_error << std::endl;
      pcout << "dPhidN Nodes error L_inf norm: "
            << dphi_dn_node_error.linfty_norm() << std::endl;
      pcout << "dPhidN Nodes error L_2 norm: " << dphi_dn_L2_error << std::endl;
      pcout << "dPhidN true quadrature L_2 norm: "
            << true_dphi_dn_L2_error << std::endl;
      pcout << "dPhidN true relative L_2 error: "
            << relative_true_dphi_dn_L2_error << std::endl;
      pcout << "Exact nodal-normal dPhidN interpolation L_2 norm: "
            << dphi_dn_interpolation_L2_error << std::endl;
      if (has_algebraic_energy_distance)
        pcout << "dPhidN algebraic energy distance to best projection: "
              << std::sqrt(std::max(algebraic_energy_square, 0.0))
              << " (raw " << algebraic_energy_square << ")" << std::endl;
      if (bem.compute_energy_norm)
        pcout << "dPhidN energy norm error: " << energy_norm_error
              << std::endl;
      if (has_l2_projection)
        {
          const std::string projection_name =
            has_cellwise_l2_projection ? "Best cellwise FE_DGQ" :
                                         "Best global FE_Q";
          pcout << projection_name << " dPhidN projection L_2 norm: "
                << projection_L2_error << std::endl;
          pcout << projection_name << " dPhidN relative projection error: "
                << (exact_dphi_dn_L2_norm > 0.0 ?
                      projection_L2_error / exact_dphi_dn_L2_norm :
                      projection_L2_error)
                << std::endl;
          pcout << (mixed_galerkin_problem ?
                      "All-surface flux error / best full-space projection: " :
                      "Galerkin error / best-projection error: ")
                << (projection_L2_error > 0.0 ?
                      true_dphi_dn_L2_error / projection_L2_error :
                      true_dphi_dn_L2_error)
                << std::endl;
        }
      if (sphere_box_grid && bem.detailed_error_diagnostics)
        {
          pcout << "dPhidN errors separated by boundary component:"
                << std::endl;
          for (unsigned int component = 0; component < 2; ++component)
            {
              const double component_error =
                std::sqrt(component_true_error_squared[component]);
              const double component_exact_norm =
                std::sqrt(component_exact_norm_squared[component]);
              pcout << "  " << (component == 0 ? "sphere" : "cube")
                    << ": L2 " << component_error
                    << ", relative L2 "
                    << (component_exact_norm > 0.0 ?
                          component_error / component_exact_norm :
                          component_error)
                    << ", quadrature Linf "
                    << component_true_linf[component]
                    << ", nodal Linf "
                    << component_nodal_linf[component]
                    << ", max point "
                    << component_true_linf_point[component];
              if (has_l2_projection)
                pcout << ", best projection L2 "
                      << std::sqrt(
                           component_projection_error_squared[component]);
              pcout << std::endl;
            }
          pcout << "dPhidN true L2 split by gap zone "
                << "(core: r<" << gap_core_radius
                << ", annulus: r<" << gap_annulus_radius
                << ", z<" << gap_z_cut << "):" << std::endl;
          for (unsigned int component = 0; component < 2; ++component)
            for (unsigned int region = 0; region < 3; ++region)
              {
                const double region_error =
                  std::sqrt(gap_true_error_squared[component][region]);
                const double region_exact_norm =
                  std::sqrt(gap_exact_norm_squared[component][region]);
                pcout << "  " << (component == 0 ? "sphere" : "cube")
                      << ' '
                      << (region == 0 ? "gap-core" :
                          region == 1 ? "gap-annulus" :
                                        "rest")
                      << ": L2 " << region_error
                      << ", relative L2 "
                      << (region_exact_norm > 0.0 ?
                            region_error / region_exact_norm :
                            region_error);
                if (has_l2_projection)
                  pcout << ", best projection L2 "
                        << std::sqrt(
                             gap_projection_error_squared[component][region]);
	                pcout << std::endl;
	              }
	          pcout << "dPhidN true L2 split by material id:" << std::endl;
	          for (const auto &entry : material_true_error_squared)
	            {
	              const unsigned int material_id = entry.first;
	              const double material_error = std::sqrt(entry.second);
	              const double material_exact_norm =
	                std::sqrt(material_exact_norm_squared[material_id]);
	              pcout << "  material " << material_id
	                    << ": L2 " << material_error
	                    << ", relative L2 "
	                    << (material_exact_norm > 0.0 ?
	                          material_error / material_exact_norm :
	                          material_error)
	                    << ", quadrature Linf "
	                    << material_true_linf[material_id]
	                    << ", max point "
	                    << material_true_linf_point[material_id];
	              if (has_l2_projection)
	                pcout << ", best projection L2 "
	                      << std::sqrt(
	                           material_projection_error_squared[material_id]);
	              pcout << std::endl;
	            }
	          pcout << "dPhidN true L2 split on cube bottom face "
	                << "(material 3):" << std::endl;
	          for (unsigned int region = 0; region < 3; ++region)
	            {
	              const double region_error =
	                std::sqrt(cube_bottom_true_error_squared[region]);
	              const double region_exact_norm =
	                std::sqrt(cube_bottom_exact_norm_squared[region]);
	              pcout << "  "
	                    << (region == 0 ? "gap-core" :
	                        region == 1 ? "gap-annulus" :
	                                      "rest")
	                    << ": L2 " << region_error
	                    << ", relative L2 "
	                    << (region_exact_norm > 0.0 ?
	                          region_error / region_exact_norm :
	                          region_error);
	              if (has_l2_projection)
	                pcout << ", best projection L2 "
	                      << std::sqrt(
	                           cube_bottom_projection_error_squared[region]);
	              pcout << ", signed mean error "
	                    << (cube_bottom_area[region] > 0.0 ?
	                          cube_bottom_signed_error[region] /
	                            cube_bottom_area[region] :
	                          0.0);
	              pcout << std::endl;
	            }
	          pcout << "dPhidN true L2 radial split on cube bottom face:"
	                << std::endl;
	          const char *region_names[3] = {"gap-core",
	                                         "gap-annulus",
	                                         "rest"};
	          const char *radial_names[5] = {"r<0.15",
	                                         "0.15<=r<0.35",
	                                         "0.35<=r<0.70",
	                                         "0.70<=r<1.00",
	                                         "r>=1.00"};
	          for (unsigned int region = 0; region < 3; ++region)
	            for (unsigned int bin = 0; bin < 5; ++bin)
	              if (cube_bottom_radial_area[region][bin] > 0.0)
	                {
	                  const double bin_error =
	                    std::sqrt(
	                      cube_bottom_radial_error_squared[region][bin]);
	                  const double bin_exact_norm =
	                    std::sqrt(
	                      cube_bottom_radial_exact_norm_squared[region][bin]);
	                  pcout << "  " << region_names[region] << ", "
	                        << radial_names[bin] << ": L2 " << bin_error
	                        << ", relative L2 "
	                        << (bin_exact_norm > 0.0 ?
	                              bin_error / bin_exact_norm :
	                              bin_error)
	                        << ", signed mean error "
	                        << cube_bottom_radial_signed_error[region][bin] /
	                             cube_bottom_radial_area[region][bin]
	                        << std::endl;
	                }
	          pcout << "dPhidN true L2 split by distance from cube-bottom "
	                   "outer edge:"
	                << std::endl;
	          const char *edge_names[4] = {"edge<0.25",
	                                       "0.25<=edge<0.75",
	                                       "0.75<=edge<1.50",
	                                       "edge>=1.50"};
	          for (unsigned int bin = 0; bin < 4; ++bin)
	            if (cube_bottom_edge_area[bin] > 0.0)
	              {
	                const double bin_error =
	                  std::sqrt(cube_bottom_edge_error_squared[bin]);
	                const double bin_exact_norm =
	                  std::sqrt(cube_bottom_edge_exact_norm_squared[bin]);
	                pcout << "  " << edge_names[bin] << ": L2 "
	                      << bin_error << ", relative L2 "
	                      << (bin_exact_norm > 0.0 ?
	                            bin_error / bin_exact_norm :
	                            bin_error)
	                      << ", signed mean error "
	                      << cube_bottom_edge_signed_error[bin] /
	                           cube_bottom_edge_area[bin]
	                      << std::endl;
	              }
	          pcout << "dPhidN true L2 split by cube-bottom xy alignment:"
	                << std::endl;
	          const char *alignment_names[3] = {"near axes",
	                                            "off-axis",
	                                            "near diagonals"};
	          for (unsigned int bin = 0; bin < 3; ++bin)
	            if (cube_bottom_alignment_area[bin] > 0.0)
	              {
	                const double bin_error =
	                  std::sqrt(cube_bottom_alignment_error_squared[bin]);
	                const double bin_exact_norm =
	                  std::sqrt(
	                    cube_bottom_alignment_exact_norm_squared[bin]);
	                pcout << "  " << alignment_names[bin] << ": L2 "
	                      << bin_error << ", relative L2 "
	                      << (bin_exact_norm > 0.0 ?
	                            bin_error / bin_exact_norm :
	                            bin_error)
	                      << ", signed mean error "
	                      << cube_bottom_alignment_signed_error[bin] /
	                           cube_bottom_alignment_area[bin]
	                      << std::endl;
	              }
	          pcout << "Top true dPhidN error cells:" << std::endl;
	          for (const auto &info : top_true_error_cells)
	            pcout << "  cell " << info.cell_index << ", material "
	                  << info.material_id << ", L2 contribution "
	                  << info.contribution << ", center " << info.center
	                  << ", r " << info.radial_distance
	                  << ", edge distance " << info.edge_distance
	                  << std::endl;
	        }
      pcout << "dPhidN true maximum cell L2 contribution: "
            << true_dphi_dn_difference_per_cell.linfty_norm() << ", at cell "
            << max_true_dphi_cell_index << ", material "
            << max_true_dphi_material_id << ", center "
            << max_true_dphi_cell_center << std::endl;
      pcout << "dPhidN Cells maximum L2 contribution: "
            << difference_per_cell_2.linfty_norm() << ", at cell "
            << max_dphi_cell_index << ", material " << max_dphi_material_id
            << ", center " << max_dphi_cell_center << std::endl;
      if (!mixed_galerkin_problem)
        {
          pcout << "Exact dPhidN discrete residual L_inf norm: "
                << exact_residual_linf << std::endl;
          pcout << "Exact dPhidN discrete residual L_2 norm: "
                << exact_residual_l2 << std::endl;
          pcout << "Exact dPhidN relative discrete residual: "
                << relative_exact_residual << std::endl;
          if (exact_residual_uses_l2_projection)
            pcout << "  (using the L2 projection of grad(phi).n)"
                  << std::endl;
        }
      else
        pcout << "Dirichlet-only residual diagnostics: not applicable to the "
                 "mixed problem; use the flux/potential row residuals above"
              << std::endl;
      if (!mixed_galerkin_problem &&
          bem.boundary_integral_formulation == "galerkin" &&
          bem.detailed_error_diagnostics &&
          (sphere_box_grid ||
           (bem.galerkin_component_rhs_diagnostics &&
            bem.calderon_dual_component_areas.size() == 2)))
        for (unsigned int component = 0; component < 2; ++component)
          pcout << "Exact residual on "
                << (component == 0 ? "sphere" : "cube") << ": L2 "
                << component_exact_residual_l2[component]
                << ", relative to component RHS "
                << (component_rhs_l2[component] > 0.0 ?
                      component_exact_residual_l2[component] /
                        component_rhs_l2[component] :
                      component_exact_residual_l2[component])
                << std::endl;
      if (bem.boundary_integral_formulation == "collocation")
        {
          pcout << "Laplace coordinate patch relative residuals:";
          for (unsigned int d = 0; d < dim; ++d)
            pcout << ' ' << coordinate_patch_relative_residuals[d];
          pcout << std::endl;
        }
      pcout << "Recovered dPhidN error L_inf norm: "
            << recovered_dphi_dn_error.linfty_norm() << std::endl;
      pcout << "Recovered dPhidN error L_2 norm: "
            << recovered_dphi_dn_L2_error << std::endl;
      pcout << "dPhidN maximum error at dof " << max_dphi_error_index
            << ", point " << support_points[max_dphi_error_index]
            << ", numerical " << localized_dphi_dn[max_dphi_error_index]
            << ", exact " << max_dphi_exact
            << ", duplicates "
            << bem.double_nodes_set[max_dphi_error_index].size() << std::endl;
      // Split the gradient error along the surface normal and tangent to it,
      // in the same integral norm as the totals reported below. An earlier
      // version of this summed squares over nodes, which does not shrink as a
      // mesh is refined even when the pointwise error does - the node count
      // grows at the same rate - and so made a converging quantity look flat.
      {
        const QGauss<dim - 1> split_quadrature(2 * (2 * bem.fe->degree + 1));
        FEValues<dim - 1, dim> split_values(*bem.mapping,
                                            *bem.gradient_fe,
                                            split_quadrature,
                                            update_values |
                                              update_normal_vectors |
                                              update_quadrature_points |
                                              update_JxW_values);
        const FEValuesExtractors::Vector all(0);
        std::vector<Tensor<1, dim>> recovered(split_quadrature.size());
        Vector<double>              exact(dim);
        double normal_square = 0.0, tangential_square = 0.0;
        double normal_max = 0.0, tangential_max = 0.0;
        // The same split on each part of the boundary, by material id, and
        // where the normal error is largest.
        std::map<types::material_id, std::array<double, 2>> material_square;
        Point<dim> normal_max_point;
        // This block runs on rank 0 only, so it must not reduce across the
        // communicator: the triangulation is shared, so rank 0 simply visits
        // every cell itself.
        for (const auto &cell : bem.gradient_dh.active_cell_iterators())
          {
            split_values.reinit(cell);
            split_values[all].get_function_values(localized_gradient_solution,
                                                  recovered);
            for (unsigned int q = 0; q < split_quadrature.size(); ++q)
              {
                wind.vector_value(split_values.quadrature_point(q), exact);
                Tensor<1, dim> error = recovered[q];
                for (unsigned int d = 0; d < dim; ++d)
                  error[d] -= exact(d);
                Tensor<1, dim> normal = split_values.normal_vector(q);
                const double   length = normal.norm();
                if (length < 1e-14)
                  continue;
                normal /= length;
                const double         along  = error * normal;
                const Tensor<1, dim> across = error - along * normal;
                normal_square += along * along * split_values.JxW(q);
                tangential_square += across.norm_square() * split_values.JxW(q);
                material_square[cell->material_id()][0] +=
                  along * along * split_values.JxW(q);
                material_square[cell->material_id()][1] +=
                  across.norm_square() * split_values.JxW(q);
                if (std::abs(along) > normal_max)
                  normal_max_point = split_values.quadrature_point(q);
                normal_max = std::max(normal_max, std::abs(along));
                tangential_max = std::max(tangential_max, across.norm());
              }
          }
        pcout << "Phi Gradient error split (integral): normal L_2 "
              << std::sqrt(normal_square) << ", L_inf " << normal_max
              << "; tangential L_2 " << std::sqrt(tangential_square)
              << ", L_inf " << tangential_max << std::endl;
        pcout << "Phi Gradient error split by material:";
        for (const auto &entry : material_square)
          pcout << " " << static_cast<unsigned int>(entry.first)
                << ": normal L_2 " << std::sqrt(entry.second[0])
                << ", tangential L_2 " << std::sqrt(entry.second[1]) << ";";
        pcout << " largest normal error at " << normal_max_point << std::endl;
      }

      // The gradient with the flux as its normal component: the projected
      // field with its normal component replaced, point by point, by
      // lambda_h, the normal being that of the mapped surface. Projecting
      // lambda n on the vector space represents the curved normal field by
      // polynomials of the degree of the space, an error that does not
      // decrease where the mesh is not refined; the tangential part keeps
      // the projection, in which the variation of the discrete normal within
      // a cell cancels between the two parts of the gradient.
      if (bem.gradient_normal_from_flux)
        {
          const QGauss<dim - 1> split_quadrature(2 * (2 * bem.fe->degree + 1));
          FEValues<dim - 1, dim> vector_values(*bem.mapping,
                                               *bem.gradient_fe,
                                               split_quadrature,
                                               update_values |
                                                 update_normal_vectors |
                                                 update_quadrature_points |
                                                 update_JxW_values);
          FEValues<dim - 1, dim> flux_fe_values(*bem.mapping,
                                                *bem.fe,
                                                split_quadrature,
                                                update_values);
          const FEValuesExtractors::Vector all(0);
          std::vector<Tensor<1, dim>> projected(split_quadrature.size());
          std::vector<double>         flux(split_quadrature.size());
          Vector<double>              exact(dim);
          double normal_square = 0.0, tangential_square = 0.0;
          std::map<types::material_id, std::array<double, 2>> material_square;
          auto vector_cell = bem.gradient_dh.begin_active();
          for (const auto &cell : bem.dh.active_cell_iterators())
            {
              vector_values.reinit(vector_cell);
              flux_fe_values.reinit(cell);
              vector_values[all].get_function_values(
                localized_gradient_solution, projected);
              flux_fe_values.get_function_values(localized_dphi_dn, flux);
              for (unsigned int q = 0; q < split_quadrature.size(); ++q)
                {
                  wind.vector_value(vector_values.quadrature_point(q), exact);
                  Tensor<1, dim> normal = vector_values.normal_vector(q);
                  normal /= normal.norm();
                  Tensor<1, dim> error = projected[q] -
                                         (projected[q] * normal) * normal +
                                         flux[q] * normal;
                  for (unsigned int d = 0; d < dim; ++d)
                    error[d] -= exact(d);
                  const double along  = error * normal;
                  const double across = (error - along * normal).norm_square();
                  normal_square += along * along * vector_values.JxW(q);
                  tangential_square += across * vector_values.JxW(q);
                  material_square[cell->material_id()][0] +=
                    along * along * vector_values.JxW(q);
                  material_square[cell->material_id()][1] +=
                    across * vector_values.JxW(q);
                }
              ++vector_cell;
            }
          pcout << "Phi Gradient with the flux as normal component: L_2 "
                << std::sqrt(normal_square + tangential_square)
                << "; normal L_2 " << std::sqrt(normal_square)
                << ", tangential L_2 " << std::sqrt(tangential_square)
                << "; by material:";
          for (const auto &entry : material_square)
            pcout << " " << static_cast<unsigned int>(entry.first)
                  << ": normal L_2 " << std::sqrt(entry.second[0])
                  << ", tangential L_2 " << std::sqrt(entry.second[1]) << ";";
          pcout << std::endl;
        }
      pcout << "Phi Nodes Gradient error L_inf norm: " << grad_phi_max_error
            << std::endl;
      pcout << "Phi gradient maximum nodal component at point "
            << support_points[gradient_max_node] << ", component "
            << gradient_max_component << std::endl;
      pcout << "Phi Cells Gradient  error L_2 norm: " << grad_L2_error
            << std::endl;
      // pcout<<"alpha Nodes error L_inf norm:
      // "<<localised_alpha.linfty_norm()<<std::endl; pcout<<"alpha Nodes error
      // L_2 norm: "<<localised_alpha.l2_norm()<<std::endl;

      std::string filename_vector = "vector_error.vtu";
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
      DataOut<dim - 1, dim> dataout_vector;
      dataout_vector.attach_dof_handler(bem.gradient_dh);
      dataout_vector.add_data_vector(
        vector_gradients_node_error,
        std::vector<std::string>(dim, "phi_gradient_error"),
        DataOut<dim - 1, dim>::type_dof_data,
        data_component_interpretation);

      dataout_vector.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_vector(filename_vector.c_str());

      dataout_vector.write_vtu(file_vector);

      std::string           filename_scalar = "scalar_error.vtu";
      DataOut<dim - 1, dim> dataout_scalar;
      dataout_scalar.attach_dof_handler(bem.dh);
      dataout_scalar.add_data_vector(phi_node_error,
                                     std::vector<std::string>(1, "phi_error"),
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.add_data_vector(dphi_dn_node_error,
                                     std::vector<std::string>(1,
                                                              "dphi_dn_error"),
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_scalar(filename_scalar.c_str());
      dataout_scalar.write_vtu(file_scalar);

      // A short account of the run, after the detail. Everything above is
      // either optional or a breakdown of these few numbers.
      pcout << "----- summary -----" << std::endl;
      pcout << "  mesh              " << n_active_cells << " cells, "
            << n_dofs << " dofs, " << bem.fe->get_name() << std::endl;
      if (mixed_galerkin_problem)
        pcout << "  scope             all-surface postprocessing; unknown "
                 "flux on D and prescribed Neumann data on N. Mixed solve "
                 "errors and row residuals are printed above"
              << std::endl;
      pcout << "  dPhidN L2 error   " << true_dphi_dn_L2_error
            << "  (relative " << relative_true_dphi_dn_L2_error << ")"
            << std::endl;
      if (has_l2_projection)
        pcout << "  best L2 projection " << projection_L2_error
              << (mixed_galerkin_problem ? "  (all-surface / best " :
                                           "  (Galerkin / best ")
              << (projection_L2_error > 0.0 ?
                    true_dphi_dn_L2_error / projection_L2_error :
                    0.0)
              << ")" << std::endl;
      if (bem.compute_energy_norm)
        {
          pcout << "  energy norm error " << energy_norm_error;
          if (has_algebraic_energy_distance)
            pcout << "  (to best projection "
                  << std::sqrt(std::max(algebraic_energy_square, 0.0)) << ")";
          pcout << std::endl;
        }
      if (!mixed_galerkin_problem)
        pcout << "  discrete residual " << relative_exact_residual
              << " relative" << std::endl;
      pcout << "-------------------" << std::endl;
    }
}

template <int dim>
void
BoundaryConditions<dim>::compute_errors_by_material(bool complex,
                                                    bool current_is_real)
{
  Teuchos::TimeMonitor LocalTimer(*ErrorsTime);

  // We still need to communicate our results to compute the errors.
  bem.compute_gradients(get_phi(), get_dphi_dn());
  Vector<double> localized_phi(get_phi());
  Vector<double> localized_dphi_dn(get_dphi_dn());
  Vector<double> localized_phi_other;
  Vector<double> localized_dphi_dn_other;
  if (complex)
    {
      if (current_is_real)
        {
          localized_phi_other     = get_phi(current_component + 1);
          localized_dphi_dn_other = get_dphi_dn(current_component + 1);
        }
      else
        {
          localized_phi_other     = get_phi(current_component - 1);
          localized_dphi_dn_other = get_dphi_dn(current_component - 1);
        }
    }
  Vector<double> localized_gradient_solution(
    bem.get_vector_gradients_solution());
  Vector<double> localized_surface_gradient_solution(
    bem.get_vector_surface_gradients_solution());
  Vector<double> localised_normals(bem.vector_normals_solution);

  Vector<double> localized_dirichlet_flags(bem.dirichlet_flags);
  Vector<double> localized_neumann_flags(bem.neumann_flags);

  // We let only the first processor do the error computations
  if (this_mpi_process == 0)
    {
      pcout << "computing errors on P0" << std::endl;

      Vector<double> phi_diff_cell(comp_dom.tria.n_active_cells());
      Vector<double> phi_diff_node(bem.dh.n_dofs());

      Vector<double> dphi_dn_diff_cell(comp_dom.tria.n_active_cells());
      Vector<double> dphi_dn_diff_node(bem.dh.n_dofs());

      Vector<double> gradphi_diff_cell(comp_dom.tria.n_active_cells());
      Vector<double> gradphi_diff_node(bem.gradient_dh.n_dofs());

      Vector<double> robin_diff_cell(comp_dom.tria.n_active_cells());
      Vector<double> robin_diff_node(bem.dh.n_dofs());
      Vector<double> coeffs(3);
      Vector<double> coeffs_other(3);

      std::vector<double>         phi_refval_node(bem.dh.n_dofs());
      std::vector<Vector<double>> gradphi_refval_node(bem.dh.n_dofs(),
                                                      Vector<double>(dim));

      std::vector<Point<dim>> support_points(bem.dh.n_dofs());
      DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                         bem.dh,
                                                         support_points);

      std::vector<types::global_dof_index> dofs(bem.fe->dofs_per_cell);
      std::vector<types::global_dof_index> dofs_gradient(
        bem.gradient_fe->dofs_per_cell);

      // map material (an indirection step to boundary) to the actual expression
      std::map<types::material_id, const Function<dim, double> *>
        bcond_functions;

      // first, build the map for the potential itself
      for (const auto &pair : comp_dom.manifold2bcondition_map)
        {
          if (pair.second == BoundaryConditionType::dirichlet)
            {
              auto slot = comp_dom.manifold2bcondition_slot_map[pair.first];
              bcond_functions[pair.first] =
                &get_potential(current_component, slot);
            }
        }

      integrate_difference_based_on_material_id(*bem.mapping,
                                                bem.dh,
                                                localized_phi,
                                                bcond_functions,
                                                phi_diff_cell,
                                                QGauss<(dim - 1)>(
                                                  2 * (2 * bem.fe->degree + 1)),
                                                VectorTools::L2_norm);

      bcond_functions.clear();
      // now, build the map for the gradient
      for (const auto &pair : comp_dom.manifold2bcondition_map)
        {
          if (pair.second == BoundaryConditionType::neumann)
            {
              auto slot = comp_dom.manifold2bcondition_slot_map[pair.first];
              bcond_functions[pair.first] = &get_wind(current_component, slot);
            }
        }

      integrate_difference_based_on_material_id(*bem.mapping,
                                                bem.gradient_dh,
                                                localized_gradient_solution,
                                                bcond_functions,
                                                gradphi_diff_cell,
                                                QGauss<(dim - 1)>(
                                                  2 * (2 * bem.fe->degree + 1)),
                                                VectorTools::L2_norm);

      // collect the error on each dof evaluating the difference between the
      // actual solution and that imposed by the b.condition
      // consider that the same dof might be confronted with multiple functions;
      // sum their errors

      // process dirichlet nodes, store |phi - imposed_phi|
      std::vector<bool> covered(bem.dh.n_dofs());
      for (unsigned int slot = 0; slot < MAX_CONDITION_SLOTS; ++slot)
        {
          get_potential(current_component, slot)
            .value_list(support_points, phi_refval_node);

          std::fill(covered.begin(), covered.end(), false);
          for (const auto &cell : bem.dh.active_cell_iterators())
            {
              if (cell->is_locally_owned() &&
                  comp_dom.manifold2bcondition_map[cell->manifold_id()] ==
                    BoundaryConditionType::dirichlet &&
                  comp_dom.manifold2bcondition_slot_map[cell->manifold_id()] ==
                    slot)
                {
                  cell->get_dof_indices(dofs);
                  for (auto dof : dofs)
                    {
                      if (!covered[dof])
                        {
                          covered[dof] = true;

                          phi_diff_node[dof] +=
                            std::abs(phi_refval_node[dof] - localized_phi[dof]);
                        }
                    }
                }
            }
        }

      // process neumann nodes, store |dphi_dn - imposed_dphi_dn|
      // also, do |grad_phi - imposed_grad_phi|?
      for (unsigned int slot = 0; slot < MAX_CONDITION_SLOTS; ++slot)
        {
          get_wind(current_component, slot)
            .vector_value_list(support_points, gradphi_refval_node);

          std::fill(covered.begin(), covered.end(), false);
          for (const auto &cell : bem.dh.active_cell_iterators())
            {
              if (cell->is_locally_owned() &&
                  comp_dom.manifold2bcondition_map[cell->manifold_id()] ==
                    BoundaryConditionType::neumann &&
                  comp_dom.manifold2bcondition_slot_map[cell->manifold_id()] ==
                    slot)
                {
                  cell->get_dof_indices(dofs);
                  for (auto dof : dofs)
                    {
                      if (!covered[dof])
                        {
                          covered[dof] = true;

                          // compute dphi_dn from the vector_gradient_solution?
                          Tensor<1, dim> tmp_surface_gradphi, tmp_norm_gradphi,
                            tmp_gradphi, tmp_boundary_gradphi, tmp_norm;
                          double tmp_dphi_dn = 0;
                          for (unsigned int d = 0; d < dim; ++d)
                            {
                              tmp_boundary_gradphi[d] =
                                gradphi_refval_node[dof][d];

                              auto vec_dof = d * bem.dh.n_dofs() + dof;

                              tmp_norm[d] = localised_normals
                                [bem.vec_original_to_sub_wise[vec_dof]];

                              tmp_surface_gradphi[d] =
                                localized_surface_gradient_solution
                                  [bem.vec_original_to_sub_wise[vec_dof]];
                              tmp_norm_gradphi[d] =
                                localized_dphi_dn[dof] *
                                localised_normals
                                  [bem.vec_original_to_sub_wise[vec_dof]];
                              tmp_gradphi[d] =
                                tmp_surface_gradphi[d] + tmp_norm_gradphi[d];

                              // uses gradient_dh
                              gradphi_diff_node
                                [bem.vec_original_to_sub_wise[vec_dof]] +=
                                std::abs(
                                  gradphi_refval_node
                                    [bem.original_to_sub_wise[dof]][d] -
                                  localized_gradient_solution
                                    [bem.vec_original_to_sub_wise[vec_dof]]);

                              tmp_dphi_dn +=
                                localised_normals
                                  [bem.vec_original_to_sub_wise[vec_dof]] *
                                gradphi_refval_node
                                  [bem.original_to_sub_wise[dof]][d];
                            }

                          // if ((tmp_boundary_gradphi - tmp_gradphi).norm() >
                          //     1e-3)
                          //   {
                          //     pcout << "dof " << dof << std::endl;
                          //     pcout << "\tcoords " << support_points[dof]
                          //           << std::endl;
                          //     pcout << "\tnormal " << tmp_norm << std::endl;
                          //     pcout << "\tdphi_dn (boundary) " << tmp_dphi_dn
                          //           << std::endl;
                          //     pcout << "\tdphi_dn (solution) "
                          //           << localized_dphi_dn[dof] << std::endl;
                          //     pcout << "\tgradphi (boundary) "
                          //           << gradphi_refval_node[dof] << std::endl;
                          //     pcout << "\tgradphi (solution) " << tmp_gradphi
                          //           << std::endl;
                          //     pcout << "\tgradphi_norm " << tmp_norm_gradphi
                          //           << std::endl;
                          //     pcout << "\tgradphi_surface "
                          //           << tmp_surface_gradphi << std::endl;
                          //   }

                          dphi_dn_diff_node[bem.original_to_sub_wise[dof]] +=
                            std::abs(
                              tmp_dphi_dn -
                              localized_dphi_dn[bem.original_to_sub_wise[dof]]);
                        }
                    }
                }
            }
        }

      // process robin nodes, store |coeffs(0) * phi + coeffs(1) * dphi_dn -
      // coeffs(2)|
      for (unsigned int slot = 0; slot < MAX_CONDITION_SLOTS; ++slot)
        {
          std::fill(covered.begin(), covered.end(), false);
          for (const auto &cell : bem.dh.active_cell_iterators())
            {
              if (cell->is_locally_owned() &&
                  comp_dom.manifold2bcondition_map[cell->manifold_id()] ==
                    BoundaryConditionType::robin &&
                  comp_dom.manifold2bcondition_slot_map[cell->manifold_id()] ==
                    slot)
                {
                  cell->get_dof_indices(dofs);
                  for (auto dof : dofs)
                    {
                      if (!covered[dof])
                        {
                          get_robin_coeffs(current_component, slot)
                            .vector_value(support_points[dof], coeffs);
                          if (complex)
                            {
                              if (current_is_real)
                                {
                                  get_robin_coeffs(current_component + 1, slot)
                                    .vector_value(support_points[dof],
                                                  coeffs_other);
                                }
                              else
                                {
                                  get_robin_coeffs(current_component - 1, slot)
                                    .vector_value(support_points[dof],
                                                  coeffs_other);
                                }
                            }
                          else
                            {
                              coeffs_other = 0;
                            }
                          // assemble the complex values considering the current
                          // component's role
                          std::complex<double> c0(coeffs(0), coeffs_other(0));
                          std::complex<double> c1(coeffs(1), coeffs_other(1));
                          std::complex<double> c2(coeffs(2), coeffs_other(2));

                          double phi_other = 0, dphi_dn_other = 0;
                          if (complex)
                            {
                              phi_other     = localized_phi_other(dof);
                              dphi_dn_other = localized_dphi_dn_other(dof);
                            }
                          std::complex<double> ph(localized_phi(dof),
                                                  phi_other);
                          std::complex<double> dph_dn(localized_dphi_dn(dof),
                                                      dphi_dn_other);
                          if (!current_is_real)
                            {
                              double tmp = c0.real();
                              c0.real(c0.imag());
                              c0.imag(tmp);

                              tmp = c1.real();
                              c1.real(c1.imag());
                              c1.imag(tmp);

                              tmp = c2.real();
                              c2.real(c2.imag());
                              c2.imag(tmp);

                              tmp = ph.real();
                              ph.real(ph.imag());
                              ph.imag(tmp);

                              tmp = dph_dn.real();
                              dph_dn.real(dph_dn.imag());
                              dph_dn.imag(tmp);
                            }

                          auto residual = c0 * ph + c1 * dph_dn;
                          if (current_is_real)
                            {
                              robin_diff_node[dof] +=
                                std::abs(std::real(residual - c2));
                            }
                          else
                            {
                              robin_diff_node[dof] +=
                                std::abs(std::imag(residual - c2));
                            }
                        }
                    }
                }
            }
        }

      VectorTools::integrate_difference(
        *bem.mapping,
        bem.dh,
        dphi_dn_diff_node,
        dealii::Functions::ZeroFunction<dim, double>(1),
        dphi_dn_diff_cell,
        QGauss<(dim - 1)>(2 * (2 * bem.fe->degree + 1)),
        VectorTools::L2_norm);

      VectorTools::integrate_difference(
        *bem.mapping,
        bem.dh,
        robin_diff_node,
        dealii::Functions::ZeroFunction<dim, double>(1),
        robin_diff_cell,
        QGauss<(dim - 1)>(2 * (2 * bem.fe->degree + 1)),
        VectorTools::L2_norm);

      pcout << "   Number of active cells:       "
            << comp_dom.tria.n_active_cells() << std::endl;
      pcout << "   Number of degrees of freedom: " << bem.dh.n_dofs()
            << std::endl;

      // TODO: phi_max_error should probably use the phi_diff_node vector
      pcout << "Phi Nodes error L_inf norm: " << phi_diff_node.linfty_norm()
            << std::endl;
      pcout << "Phi Cells error L_2 norm: " << phi_diff_cell.l2_norm()
            << std::endl;

      pcout << "dPhidN Nodes error L_inf norm: "
            << dphi_dn_diff_node.linfty_norm() << std::endl;
      pcout << "dPhidN Cells error L_2 norm: " << dphi_dn_diff_cell.l2_norm()
            << std::endl;

      pcout << "Phi Nodes Gradient error L_inf norm: "
            << gradphi_diff_node.linfty_norm() << std::endl;
      pcout << "Phi Cells Gradient error L_2 norm: "
            << gradphi_diff_cell.l2_norm() << std::endl;

      pcout << "Robin Nodes Residual error L_inf norm: "
            << robin_diff_node.linfty_norm() << std::endl;
      pcout << "Robin Nodes Residual error L_2 norm: "
            << robin_diff_cell.l2_norm() << std::endl;

      std::string filename_vector =
        "error" +
        (current_component ? "_" + Utilities::int_to_string(current_component) :
                             "") +
        "_vector.vtu";
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
      DataOut<dim - 1, dim> dataout_vector;
      dataout_vector.attach_dof_handler(bem.gradient_dh);
      dataout_vector.add_data_vector(
        gradphi_diff_node,
        std::vector<std::string>(dim, "phi_gradient_error"),
        DataOut<dim - 1, dim>::type_dof_data,
        data_component_interpretation);

      dataout_vector.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_vector(filename_vector.c_str());

      dataout_vector.write_vtu(file_vector);

      std::string filename_scalar =
        "error" +
        (current_component ? "_" + Utilities::int_to_string(current_component) :
                             "") +
        "_scalar.vtu";
      DataOut<dim - 1, dim> dataout_scalar;
      dataout_scalar.attach_dof_handler(bem.dh);
      dataout_scalar.add_data_vector(phi_diff_node,
                                     std::vector<std::string>(1, "phi_error"),
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.add_data_vector(dphi_dn_diff_node,
                                     std::vector<std::string>(1,
                                                              "dphi_dn_error"),
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.add_data_vector(robin_diff_node,
                                     std::vector<std::string>(1,
                                                              "robin_c_error"),
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_scalar(filename_scalar.c_str());
      dataout_scalar.write_vtu(file_scalar);
    }
}

template <int dim>
void
BoundaryConditions<dim>::output_results(const std::string filename)
{
  Teuchos::TimeMonitor LocalTimer(*OutputTimer);

  // Even for the output we need to serialize the code and then perform the
  // output only on the first processor.
  const Vector<double> localized_phi(get_phi());
  const Vector<double> localized_dphi_dn(get_dphi_dn());
  const Vector<double> localized_alpha(bem.alpha);
  const Vector<double> localized_gradients(bem.get_vector_gradients_solution());
  const Vector<double> localized_surf_gradients(
    bem.get_vector_surface_gradients_solution());
  const Vector<double> localized_normals(bem.vector_normals_solution);

  if (this_mpi_process == 0)
    {
      std::string filename_scalar, filename_vector;
      filename_scalar = filename + "_scalar_results" + ".vtu";
      filename_vector = filename + "_vector_results" + ".vtu";

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);

      DataOut<dim - 1, dim> dataout_scalar;
      DataOut<dim - 1, dim> dataout_vector;

      dataout_scalar.attach_dof_handler(bem.dh);
      dataout_vector.attach_dof_handler(bem.gradient_dh);



      dataout_scalar.add_data_vector(localized_phi,
                                     "phi",
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.add_data_vector(localized_dphi_dn,
                                     "dphi_dn",
                                     DataOut<dim - 1, dim>::type_dof_data);
      dataout_scalar.add_data_vector(localized_alpha,
                                     "alpha",
                                     DataOut<dim - 1, dim>::type_dof_data);

      dataout_vector.add_data_vector(localized_gradients,
                                     std::vector<std::string>(dim,
                                                              "phi_gradient"),
                                     DataOut<dim - 1, dim>::type_dof_data,
                                     data_component_interpretation);
      if (localized_surf_gradients.size() == bem.gradient_dh.n_dofs())
        dataout_vector.add_data_vector(
          localized_surf_gradients,
          std::vector<std::string>(dim, "phi_surf_gradient"),
          DataOut<dim - 1, dim>::type_dof_data,
          data_component_interpretation);
      if (localized_normals.size() == bem.gradient_dh.n_dofs())
        dataout_vector.add_data_vector(
          localized_normals,
          std::vector<std::string>(dim, "normals_at_nodes"),
          DataOut<dim - 1, dim>::type_dof_data,
          data_component_interpretation);


      dataout_scalar.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_scalar(filename_scalar.c_str());

      dataout_scalar.write_vtu(file_scalar);

      dataout_vector.build_patches(*bem.mapping,
                                   bem.mapping_degree,
                                   DataOut<dim - 1, dim>::curved_inner_cells);

      std::ofstream file_vector(filename_vector.c_str());

      dataout_vector.write_vtu(file_vector);
    }
}

template class BoundaryConditions<2>;
template class BoundaryConditions<3>;
