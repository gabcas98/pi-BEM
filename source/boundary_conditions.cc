
// The program starts with including a bunch
// of include files that we will use in the
// various parts of the program. Most of them
// have been discussed in previous tutorials
// already:

#include "../include/boundary_conditions.h"

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/filtered_iterator.h>

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

  prm.enter_subsection("Wind function 2d");
  {
    Functions::ParsedFunction<2>::declare_parameters(prm, 2);
    prm.set("Function expression", "1; 1");
  }
  prm.leave_subsection();

  prm.enter_subsection("Wind function 3d");
  {
    Functions::ParsedFunction<3>::declare_parameters(prm, 3);
    prm.set("Function expression", "1; 1; 1");
  }
  prm.leave_subsection();


  prm.enter_subsection("Potential 2d");
  {
    Functions::ParsedFunction<2>::declare_parameters(prm);
    prm.set("Function expression", "x+y");
  }
  prm.leave_subsection();

  prm.enter_subsection("Potential 3d");
  {
    Functions::ParsedFunction<3>::declare_parameters(prm);
    prm.set("Function expression", "x+y+z");
  }
  prm.leave_subsection();
}

template <int dim>
void
BoundaryConditions<dim>::parse_parameters(ParameterHandler &prm)
{
  output_file_name = prm.get("Output file name");


  prm.enter_subsection(std::string("Wind function ") +
                       Utilities::int_to_string(dim) + std::string("d"));
  {
    wind.parse_parameters(prm);
  }
  prm.leave_subsection();

  prm.enter_subsection(std::string("Potential ") +
                       Utilities::int_to_string(dim) + std::string("d"));
  {
    potential.parse_parameters(prm);
  }
  prm.leave_subsection();
}



/*template <int dim>
double* BoundaryConditions<dim>::initial_conditions() {

  initial_wave_shape.set_time(initial_time);
  initial_wave_potential.set_time(initial_time);
  wind.set_time(initial_time);

  Vector<double> instantWindValue(dim);
  Point<dim> zero(0,0,0);
  wind.vector_value(zero,instantWindValue);
  bem.pcout<<std::endl<<"Simulation time= "<<initial_time<<"   Vinf= ";
  instantWindValue.print(cout,4,false,true);
  bem.pcout<<std::endl;

  dofs_number = bem.dh.n_dofs()+bem.gradient_dh.n_dofs();

  phi.reinit(bem.dh.n_dofs());
  dphi_dn.reinit(bem.dh.n_dofs());
  tmp_rhs.reinit(bem.dh.n_dofs());

  DXDt_and_DphiDt_vector.resize(n_dofs());

  std::vector<Point<dim> > support_points(bem.dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( bem.mapping, bem.dh,
support_points); std::vector<Point<dim> >
gradient_support_points(bem.gradient_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( bem.mapping,
bem.gradient_dh, gradient_support_points);


  unsigned int j = dim-1;
  for(unsigned int i=j; i<bem.gradient_dh.n_dofs(); i=i+dim)
      {
      DXDt_and_DphiDt_vector[i] =
initial_wave_shape.value(gradient_support_points[i]);
      //bem.pcout<<DXDt_and_DphiDt_vector[i]<<std::endl;
      }

   for (unsigned int i=0; i<bem.dh.n_dofs(); i++)
      {
      DXDt_and_DphiDt_vector[i+bem.gradient_dh.n_dofs()] =
initial_wave_potential.value(gradient_support_points[i]);
      //bem.pcout<<DXDt_and_DphiDt_vector[i+bem.gradient_dh.n_dofs()]<<std::endl;
      }

   max_y_coor_value = 0;
   for (unsigned int i=0; i < bem.dh.n_dofs(); i++)
       {
       //for printout
       //bem.pcout<<"Node "<<i<< "["<<support_points[i]<<"] "<<std::endl;
       max_y_coor_value =
std::max(max_y_coor_value,std::abs(support_points[i](1)));
       }

  std::string filename = ( output_file_name + "_" +
         Utilities::int_to_string(0) +
         ".vtk" );

  output_results(filename);

  return &DXDt_and_DphiDt_vector[0];
} //*/



template <int dim>
void
BoundaryConditions<dim>::solve_problem()
{
  potential.set_time(0);
  wind.set_time(0);

  const types::global_dof_index    n_dofs = bem.dh.n_dofs();
  std::vector<types::subdomain_id> dofs_domain_association(n_dofs);
  DoFTools::get_subdomain_association(bem.dh, dofs_domain_association);
  this_cpu_set.clear();
  this_cpu_set = bem.this_cpu_set;
  this_cpu_set.compress();



  phi.reinit(this_cpu_set, mpi_communicator);
  dphi_dn.reinit(this_cpu_set, mpi_communicator);
  tmp_rhs.reinit(this_cpu_set, mpi_communicator);
  pcout << "Computing normal vector" << std::endl;
  bem.compute_normals();
  prepare_bem_vectors();


  bem.solve(phi, dphi_dn, tmp_rhs, &potential, &wind);
  have_dirichlet_bc = bem.have_dirichlet_bc;
  if (!have_dirichlet_bc)
    {
      std::vector<Point<dim>> support_points(n_dofs);
      DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                         bem.dh,
                                                         support_points);
      double shift = 0.0;
      if (this_mpi_process == 0)
        shift = potential.value(support_points[*bem.this_cpu_set.begin()]) -
                phi(*bem.this_cpu_set.begin());
      MPI_Bcast(&shift, 1, MPI_DOUBLE, 0, mpi_communicator);
      vector_shift(phi, shift);
    }

  // bem.compute_gradients(phi, dphi_dn);
}

template <int dim>
const TrilinosWrappers::MPI::Vector &
BoundaryConditions<dim>::get_phi()
{
  return phi;
}

template <int dim>
const TrilinosWrappers::MPI::Vector &
BoundaryConditions<dim>::get_dphi_dn()
{
  return dphi_dn;
}



template <int dim>
void
BoundaryConditions<dim>::prepare_bem_vectors()
{
  Teuchos::TimeMonitor LocalTimer(*PrepareTime);
  // bem.compute_normals();
  const types::global_dof_index n_dofs = bem.dh.n_dofs();

  phi.reinit(this_cpu_set, mpi_communicator);
  dphi_dn.reinit(this_cpu_set, mpi_communicator);
  // tmp_rhs.reinit(this_cpu_set,mpi_communicator);


  std::vector<Point<dim>> support_points(n_dofs);
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.dh,
                                                     support_points);

  std::vector<Point<dim>> vec_support_points(bem.gradient_dh.n_dofs());
  DoFTools::map_dofs_to_support_points<dim - 1, dim>(*bem.mapping,
                                                     bem.gradient_dh,
                                                     vec_support_points);

  cell_it cell = bem.dh.begin_active(), endc = bem.dh.end();

  const unsigned int                   dofs_per_cell = bem.fe->dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  FEValues<dim - 1, dim>               fe_v(*bem.mapping,
                              *bem.fe,
                              *bem.quadrature,
                              update_values | update_normal_vectors |
                                update_quadrature_points | update_JxW_values);


  for (cell = bem.dh.begin_active(); cell != endc; ++cell)

    {
      fe_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      // const std::vector<Point<dim> > &node_normals =
      // fe_v.get_cell_normal_vectors();////provv
      for (unsigned int j = 0; j < bem.fe->dofs_per_cell; ++j)
        if (this_cpu_set.is_element(local_dof_indices[j]))
          {
            // bem.pcout<<cell<<" "<<cell->material_id()<<"
            // ("<<node_normals[0]<<")
            // "<<-normals_sys_solution(local_dof_indices[j])<<std::endl;
            bool dirichlet = false;
            bool neumann   = false;
            for (auto dbound : comp_dom.dirichlet_boundary_ids)
              if (cell->material_id() == dbound)
                {
                  dirichlet = true;
                  break;
                }
            if (dirichlet)
              {
                // tmp_rhs(local_dof_indices[j]) = node_coors[j](0);
                phi(local_dof_indices[j]) =
                  potential.value(support_points[local_dof_indices[j]]);
                tmp_rhs(local_dof_indices[j]) =
                  potential.value(support_points[local_dof_indices[j]]);
                // bem.pcout<<"internalElse "<<local_dof_indices[j]<<" norm
                // ("<<node_normals[j]<<")  "<<" pos ("<<node_coors[j]<<")
                // "<<node_coors[j](0)<<std::endl;
              }
            else
              {
                for (auto nbound : comp_dom.neumann_boundary_ids)
                  if (cell->material_id() == nbound)
                    {
                      neumann = true;
                      break;
                    }

                if (neumann)
                  {
                    // tmp_rhs(local_dof_indices[j]) =
                    // normals_sys_solution(local_dof_indices[j]);
                    // dphi_dn(local_dof_indices[j]) =
                    // normals_sys_solution(local_dof_indices[j]);
                    Vector<double> imposed_pot_grad(dim);
                    wind.vector_value(support_points[local_dof_indices[j]],
                                      imposed_pot_grad);
                    // Point<dim> imposed_potential_gradient;
                    double tmp_dphi_dn = 0;
                    double normy       = 0;
                    // double tol = 1e-1;
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        types::global_dof_index dummy =
                          bem.sub_wise_to_original[local_dof_indices[j]];
                        types::global_dof_index vec_index =
                          bem.vec_original_to_sub_wise
                            [bem.gradient_dh.n_dofs() / dim * d +
                             dummy]; // bem.vector_start_per_process[this_mpi_process]
                                     // + d*bem.this_cpu_set.n_elements() +
                                     // local_dof_indices[j]-bem.start_per_process[this_mpi_process];//bem.gradient_dh.n_dofs()/dim*d+local_dof_indices[j];//bem.vector_start_per_process[this_mpi_process]+((local_dof_indices[j]-bem.start_per_process[this_mpi_process])*dim+d);
                                     // //local_dof_indices[j]*dim+d;
                        // std::cout<<this_mpi_process<<"
                        // "<<support_points[local_dof_indices[j]]<<"
                        // "<<vec_support_points[vec_index]<<std::endl;
                        Assert(
                          bem.vector_this_cpu_set.is_element(vec_index),
                          ExcMessage(
                            "vector cpu set and cpu set are inconsistent"));
                        // Assert(support_points[local_dof_indices[j]]==vec_support_points[vec_index],
                        // ExcMessage("the support points of dh and gradient_dh
                        // are different"));
                        tmp_dphi_dn += imposed_pot_grad[d] *
                                       bem.vector_normals_solution[vec_index];
                        normy += bem.vector_normals_solution[vec_index] *
                                 bem.vector_normals_solution[vec_index];
                      }
                    // Assert(std::fabs(normy-1.)<tol, ExcMessage("you are using
                    // wrongly the normal vector"));
                    tmp_rhs(local_dof_indices[j]) = tmp_dphi_dn;
                    dphi_dn(local_dof_indices[j]) = tmp_dphi_dn;
                  }
                else
                  {
                    tmp_rhs(local_dof_indices[j]) = 0;
                    dphi_dn(local_dof_indices[j]) = 0;
                  }
              }
          }
    }
}

template <int dim>
void
BoundaryConditions<dim>::compute_errors()
{
  Teuchos::TimeMonitor LocalTimer(*ErrorsTime);

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
    bem.vector_gradients_solution); // vector_gradients_solution
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
      DoFTools::make_hanging_node_constraints(bem.dh,
                                              hanging_node_constraints);
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
  constrained_operator.vmult(exact_residual, parallel_exact_dphi_dn);
  exact_residual.add(-1.0, bem.system_rhs);
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
  if (exact_residual_uses_l2_projection)
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

      // Norms of the exact solution, the scale of the errors above: a
      // solution such as a screened plane wave may be of size 1e35 at one
      // point of the boundary, and its absolute errors of size 1e32 are then
      // small.
      double exact_phi_L2_norm = 0., exact_gradient_L2_norm = 0.;
      {
        Vector<float>  per_cell(comp_dom.tria.n_active_cells());
        Vector<double> zero(localized_phi.size());
        VectorTools::integrate_difference(*bem.mapping,
                                          bem.dh,
                                          zero,
                                          potential,
                                          per_cell,
                                          QGauss<(dim - 1)>(
                                            2 * (2 * bem.fe->degree + 1)),
                                          VectorTools::L2_norm);
        exact_phi_L2_norm = per_cell.l2_norm();
        Vector<double> zero_gradient(localized_gradient_solution.size());
        VectorTools::integrate_difference(*bem.mapping,
                                          bem.gradient_dh,
                                          zero_gradient,
                                          wind,
                                          per_cell,
                                          QGauss<(dim - 1)>(
                                            2 * (2 * bem.fe->degree + 1)),
                                          VectorTools::L2_norm);
        exact_gradient_L2_norm = per_cell.l2_norm();
      }
      const auto relative = [](const double error, const double norm) {
        return norm > 0. ? error / norm : error;
      };


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

      double exact_phi_max = 0.;
      for (const double v : phi_nodes_errs)
        exact_phi_max = std::max(exact_phi_max, std::abs(v));
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
      if (exact_residual_uses_l2_projection)
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
          pcout << "Galerkin error / best-projection error: "
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
      pcout << "Exact dPhidN discrete residual L_inf norm: "
            << exact_residual_linf << std::endl;
      pcout << "Exact dPhidN discrete residual L_2 norm: "
            << exact_residual_l2 << std::endl;
      pcout << "Exact dPhidN relative discrete residual: "
            << relative_exact_residual << std::endl;
      if (exact_residual_uses_l2_projection)
        pcout << "  (using the cellwise L2 projection of grad(phi).n)"
              << std::endl;
      if (bem.boundary_integral_formulation == "galerkin" &&
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
      pcout << "Phi Cells Gradient  error L_2 norm: " << grad_L2_error
            << std::endl;

      // The same errors relative to the size of the exact solution: L2
      // errors over the L2 norm of the exact field, nodal maxima over its
      // largest nodal value.
      double exact_gradient_max = 0.;
      for (const auto &g : exact_gradient_values)
        exact_gradient_max = std::max(exact_gradient_max, g.l2_norm());
      pcout << "Relative errors (to the exact solution): phi L_2 "
            << relative(L2_error, exact_phi_L2_norm) << ", phi nodes L_inf "
            << relative(phi_node_Linf_error, exact_phi_max)
            << "; dPhidN L_2 " << relative_true_dphi_dn_L2_error
            << ", dPhidN nodes L_inf "
            << relative(dphi_dn_node_error.linfty_norm(),
                        exact_dphi_dn.linfty_norm())
            << "; gradient L_2 "
            << relative(grad_L2_error, exact_gradient_L2_norm)
            << ", gradient nodes L_inf "
            << relative(grad_phi_max_error, exact_gradient_max) << std::endl;
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
      // log10 of |error| / |exact| at each node, the exact value floored at
      // 1e-12 times its largest one, below which roundoff decides.
      Vector<double> gradient_relative_error(bem.gradient_dh.n_dofs());
      for (types::global_dof_index scalar_i = 0; scalar_i < bem.dh.n_dofs();
           ++scalar_i)
        {
          const types::global_dof_index original_i =
            bem.sub_wise_to_original[scalar_i];
          double error_square = 0.;
          for (unsigned int d = 0; d < dim; ++d)
            error_square += std::pow(
              vector_gradients_node_error(
                bem.vec_original_to_sub_wise[d * bem.dh.n_dofs() +
                                             original_i]),
              2);
          const double scale =
            std::max(exact_gradient_values[scalar_i].l2_norm(),
                     1e-12 * exact_gradient_max);
          for (unsigned int d = 0; d < dim; ++d)
            gradient_relative_error(
              bem.vec_original_to_sub_wise[d * bem.dh.n_dofs() +
                                           original_i]) =
              std::log10(std::max(std::sqrt(error_square), 1e-300) /
                         std::max(scale, 1e-300));
        }
      dataout_vector.add_data_vector(
        gradient_relative_error,
        std::vector<std::string>(dim, "phi_gradient_log10_relative_error"),
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
      // Errors on a logarithmic scale, readable when the solution spans many
      // orders of magnitude: log10 of |error| / |exact| at each node (the
      // number of correct digits, with the exact value floored at 1e-12
      // times its largest one, below which roundoff decides), log10 of
      // |error| / max |exact| (where the relative global error comes from),
      // and log10 |exact| (where the solution matters).
      const auto log10_ratio = [](const double a, const double b) {
        return std::log10(std::max(a, 1e-300) / std::max(b, 1e-300));
      };
      const double dphi_max = exact_dphi_dn.linfty_norm();
      std::vector<Vector<double>> log_fields(6, Vector<double>(bem.dh.n_dofs()));
      for (types::global_dof_index i = 0; i < bem.dh.n_dofs(); ++i)
        {
          const double phi_exact  = std::abs(phi_nodes_errs[i]);
          const double dphi_exact = std::abs(exact_dphi_dn(i));
          log_fields[0](i) =
            log10_ratio(std::abs(phi_node_error(i)),
                        std::max(phi_exact, 1e-12 * exact_phi_max));
          log_fields[1](i) =
            log10_ratio(std::abs(phi_node_error(i)), exact_phi_max);
          log_fields[2](i) = log10_ratio(phi_exact, 1.);
          log_fields[3](i) =
            log10_ratio(std::abs(dphi_dn_node_error(i)),
                        std::max(dphi_exact, 1e-12 * dphi_max));
          log_fields[4](i) =
            log10_ratio(std::abs(dphi_dn_node_error(i)), dphi_max);
          log_fields[5](i) = log10_ratio(dphi_exact, 1.);
        }
      const std::vector<std::string> log_names = {"phi_log10_relative_error",
                                                  "phi_log10_global_error",
                                                  "phi_log10_exact",
                                                  "dphi_dn_log10_relative_error",
                                                  "dphi_dn_log10_global_error",
                                                  "dphi_dn_log10_exact"};
      for (unsigned int k = 0; k < log_fields.size(); ++k)
        dataout_scalar.add_data_vector(log_fields[k],
                                       std::vector<std::string>(1,
                                                                log_names[k]),
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
      pcout << "  dPhidN L2 error   " << true_dphi_dn_L2_error
            << "  (relative " << relative_true_dphi_dn_L2_error << ")"
            << std::endl;
      pcout << "  phi L2 error      " << L2_error << "  (relative "
            << relative(L2_error, exact_phi_L2_norm) << ")" << std::endl;
      pcout << "  gradient L2 error " << grad_L2_error << "  (relative "
            << relative(grad_L2_error, exact_gradient_L2_norm) << ")"
            << std::endl;
      if (has_l2_projection)
        pcout << "  best L2 projection " << projection_L2_error
              << "  (Galerkin / best "
              << (projection_L2_error > 0.0 ?
                    true_dphi_dn_L2_error / projection_L2_error :
                    0.0)
              << ")" << std::endl;
      if (bem.compute_energy_norm)
        pcout << "  energy norm error " << energy_norm_error
              << "  (to best projection "
              << std::sqrt(std::max(algebraic_energy_square, 0.0)) << ")"
              << std::endl;
      pcout << "  discrete residual " << relative_exact_residual
            << " relative" << std::endl;
      pcout << "-------------------" << std::endl;
    }
}

template <int dim>
void
BoundaryConditions<dim>::output_results(const std::string filename)
{
  Teuchos::TimeMonitor LocalTimer(*OutputTimer);

  // Even for the output we need to serialize the code and then perform the
  // output only on the first processor.
  const Vector<double> localized_phi(phi);
  const Vector<double> localized_dphi_dn(dphi_dn);
  const Vector<double> localized_alpha(bem.alpha);
  const Vector<double> localized_gradients(bem.vector_gradients_solution);
  const Vector<double> localized_surf_gradients(
    bem.vector_surface_gradients_solution);
  const Vector<double> localized_normals(bem.vector_normals_solution);
  // localized_dphi_dn.print(std::cout);
  // localized_phi.print(std::cout);
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
