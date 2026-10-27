// ---------------------------------------------------------------------
//
// Copyright (C) 2026 by the pi-BEM authors.
//
// This file is part of the pi-BEM library.
//
// The pi-BEM is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License version 2.1 as published by the Free Software Foundation.
// The full text of the license can be found in the file LICENSE at
// the top level of the pi-BEM distribution.
//
// ---------------------------------------------------------------------

// The recoveries of the gradient of BEMProblem (Gradient recovery): the L2
// projection of the surface gradient with the flux as normal component, the
// weak form with the analytical trace, the hypersingular integral equation
// of the collocation formulation and the Galerkin form of the gradient
// representation; and the surface gradients of the constraints.

#include "../include/bem_problem.h"
#include "../include/cellwise_function.h"
#include "../include/galerkin_calderon_fmm_operator.h"
#include "../include/galerkin_panel_pair_integrator.h"
#include "../include/laplace_kernel.h"
#include "../include/screened_kernel.h"
#include "../include/quasi_singular_kernel_integral.h"
#include "../include/singular_kernel_integral.h"
#include "../include/surface_potential_evaluator.h"
#include "bem_problem_helpers.h"
#include "mapping_hanging_constraints.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

using namespace std;

template <int dim>
void
BEMProblem<dim>::compute_gradients(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn)
{
  Teuchos::TimeMonitor LocalTimer(*GradientTime);

  // We need the solution to be stored on a parallel vector with ghost
  // elements. We let Trilinos take care of it.
  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi = glob_phi;
  TrilinosWrappers::MPI::Vector dphi_dn(ghosted_set);
  dphi_dn = glob_dphi_dn;

  // We reinit the gradient solution
  get_vector_gradients_solution().reinit(vector_this_cpu_set, mpi_communicator);

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
                {
                  node_normal_grad_dir[i] = q_vector_normals_solution[q][i];
                }
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

  // At this point we can compress everything and solve via GMRES.
  vector_gradients_matrix.compress(VectorOperation::add);
  vector_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));

  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_gradients_matrix);
  solver.solve(vector_gradients_matrix,
               get_vector_gradients_solution(),
               vector_gradients_rhs,
               mass_prec);

  vector_constraints.distribute(get_vector_gradients_solution());
}

template <int dim>
void
BEMProblem<dim>::compute_surface_gradients(
  const TrilinosWrappers::MPI::Vector &tmp_rhs)
{
  Teuchos::TimeMonitor          LocalTimer(*SurfaceGradientTime);
  TrilinosWrappers::MPI::Vector phi(ghosted_set);
  phi = tmp_rhs;

  get_vector_surface_gradients_solution().reinit(vector_this_cpu_set,
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
               get_vector_surface_gradients_solution(),
               vector_surface_gradients_rhs,
               mass_prec);

  vector_constraints.distribute(get_vector_surface_gradients_solution());
}

template <int dim>
void
BEMProblem<dim>::compute_gradients_weak(
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<dim>                 &analytic_phi)
{
  Teuchos::TimeMonitor LocalTimer(*GradientTime);
  pcout << "Computing gradients in weak form from the analytical trace"
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
  TrilinosWrappers::MPI::Vector dphi_dn(ghosted_set);
  dphi_dn = glob_dphi_dn;

  get_vector_gradients_solution().reinit(vector_this_cpu_set, mpi_communicator);

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
  // which vanishes only where the surface is flat.
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
      geometric_curvature(cell, total_curvature);
      for (unsigned int q = 0; q < vector_n_q_points; ++q)
        trace_values[q] =
          on_cell(analytic_phi, cell).value(fe_v.quadrature_point(q));

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
          }

      for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell; ++f)
        {
          vector_fe_face.reinit(vector_cell, f);
          fe_face.reinit(cell, f);
          normal_face.reinit(cell, f);
          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            face_trace_values[q] =
              on_cell(analytic_phi, cell).value(fe_face.quadrature_point(q));

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

      vector_cell->get_dof_indices(vector_local_dof_indices);
      vector_constraints.distribute_local_to_global(local_matrix,
                                                    local_rhs,
                                                    vector_local_dof_indices,
                                                    vector_gradients_matrix,
                                                    vector_gradients_rhs);
    }

  vector_gradients_matrix.compress(VectorOperation::add);
  vector_gradients_rhs.compress(VectorOperation::add);

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(
    solver_control,
    SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData(1000));
  TrilinosWrappers::PreconditionAMG mass_prec;
  mass_prec.initialize(vector_gradients_matrix);
  solver.solve(vector_gradients_matrix,
               get_vector_gradients_solution(),
               vector_gradients_rhs,
               mass_prec);
  vector_constraints.distribute(get_vector_gradients_solution());

  gradient_normal_from_flux = true;
    {
      // Assemble the full gradient as the L2 projection, at the quadrature
      // points, of  P g_t + lambda_h n,  with n the exact normal of the
      // mapped surface there. Projecting rather than interpolating at the
      // nodes gives the normal component the best approximation of lambda n
      // the space allows, which is what the projection-based recovery gets
      // right and the nodal replacement did not.
      TrilinosWrappers::MPI::Vector tangential_field(get_vector_gradients_solution());
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
                   get_vector_gradients_solution(),
                   full_rhs,
                   mass_prec);
      vector_constraints.distribute(get_vector_gradients_solution());
    }
}

template <int dim>
void
BEMProblem<dim>::compute_gradients_hypersingular_weak(
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<dim>                 &analytic_phi,
  const Function<dim>                 &trace_gradient)
{
  if constexpr (dim != 3)
    {
      (void)glob_dphi_dn;
      (void)analytic_phi;
      (void)trace_gradient;
      AssertThrow(false,
                  ExcMessage("The Galerkin form of the gradient "
                             "representation is implemented for surfaces "
                             "in three dimensions."));
    }
  else
    {
      Teuchos::TimeMonitor LocalTimer(*GradientTime);
      pcout << "Computing gradients from the Galerkin form of the gradient "
               "representation, analytical trace"
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
      const Vector<double> flux_full(glob_dphi_dn);

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
      densities.flux_coefficients = &flux_full;
      densities.trace_function    = &analytic_phi;
      densities.trace_gradient    = &trace_gradient;
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
      integrator.set_folded_singular_order(
        galerkin_folded_singular_quadrature_order);
      integrator.set_material_singular_order(
        galerkin_singular_quadrature_material_ids,
        galerkin_material_singular_quadrature_order);
      integrator.set_skewed_singular_order(
        galerkin_skewed_singular_quadrature_order,
        galerkin_skewed_minimum_angle);
      integrator.set_kinked_singular_order(
        galerkin_kinked_singular_quadrature_order,
        galerkin_kinked_minimum_angle);
      integrator.set_kinked_distance_ratio(
        galerkin_kinked_distance_ratio);
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
      // The degree of the reconstruction is that of the mapping: of the
      // polynomial mapping, of the interpolation of the exact geometry, or
      // of the mesh mapping a projected mapping starts from.
      unsigned int geometry_degree =
        mapping_type == "FE" ? vector_fe_order : mapping_degree;
      if (const auto *polynomial =
            dynamic_cast<const MappingQ<dim - 1, dim> *>(mapping.get()))
        geometry_degree = polynomial->get_degree();
      const SurfacePotentialEvaluator evaluator(
        *mapping,
        dh,
        geometry_degree,
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
      std::vector<double> flux_q(area_rule.size());
      std::vector<double> exact_flux_q(area_rule.size());
      std::vector<types::global_dof_index> vector_indices(nv);
      Vector<double>                       gradient_value(dim);

      double residual_square = 0.0, measure = 0.0, residual_max = 0.0;
      double edge_residual_max = 0.0, divergence_defect = 0.0;
      double error_star = 0.0, error_galerkin = 0.0, error_best = 0.0;
      double norm_exact = 0.0;
      double n_targets  = 0.0;
      Point<dim> worst_location, worst_edge_location;

      get_vector_gradients_solution().reinit(vector_this_cpu_set,
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
          scalar_values.get_function_values(flux_full, flux_q);

          for (unsigned int q = 0; q < area_rule.size(); ++q)
            {
              const Point<dim>     &x      = vector_values.quadrature_point(q);
              const Tensor<1, dim> &normal = vector_values.normal_vector(q);
              const double          JxW    = vector_values.JxW(q);
              const double trace = on_cell(analytic_phi, cell).value(x);
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

              on_cell(trace_gradient, cell).vector_value(x, gradient_value);
              double exact_flux = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                exact_flux += gradient_value[d] * normal[d];
              exact_flux_q[q] = exact_flux;

              for (unsigned int j = 0; j < ns; ++j)
                {
                  const double psi = scalar_values.shape_value(j, q);
                  // The screened hypersingular operator adds
                  // kappa^2 int psi n_x . V(n phi) to Maue's form.
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

                  const double trace = on_cell(analytic_phi, cell).value(x);
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
              error_galerkin +=
                (flux_q[q] - exact) * (flux_q[q] - exact) * JxW;
              error_best += (lambda_best - exact) * (lambda_best - exact) * JxW;
              norm_exact += exact * exact * JxW;
            }
          inverse_v.vmult(coef_full, rhs_full);

          vector_cell->get_dof_indices(vector_indices);
          for (unsigned int i = 0; i < nv; ++i)
            get_vector_gradients_solution()(vector_indices[i]) = coef_full(i);
        }
      get_vector_gradients_solution().compress(VectorOperation::insert);

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

      AssertThrow(deviation < 1e-8,
                  ExcMessage("The geometry reconstructed for the surface "
                             "potentials does not reproduce the mapping."));
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
      pcout << "  normal part, L2 error relative to |lambda|: "
                 "representation "
              << e_star / exact_norm << ", Galerkin flux "
              << e_galerkin / exact_norm << ", best approximation "
              << e_best / exact_norm << std::endl;
    }
}

// The singular hypersingular kernels are implemented only for surface cells
// in 3D. Reject the 2D call before entering the three-component algorithm.
template <>
void
BEMProblem<2>::compute_gradients_hypersingular(
  const TrilinosWrappers::MPI::Vector &,
  const TrilinosWrappers::MPI::Vector &)
{
  AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void
BEMProblem<dim>::compute_gradients_hypersingular(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn)
{
  Teuchos::TimeMonitor LocalTimer(*AssembleTime);
  pcout << "Computing gradients with hypersingular integrals" << std::endl;

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
  get_vector_gradients_solution() = vector_hyp_gradients_solution;
  pcout << "done computing gradients with hypersingular integrals" << std::endl;
}


template void
BEMProblem<2>::compute_gradients(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn);
template void
BEMProblem<3>::compute_gradients(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn);
template void
BEMProblem<2>::compute_surface_gradients(
  const TrilinosWrappers::MPI::Vector &tmp_rhs);
template void
BEMProblem<3>::compute_surface_gradients(
  const TrilinosWrappers::MPI::Vector &tmp_rhs);
template void
BEMProblem<2>::compute_gradients_weak(
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<2>                 &analytic_phi);
template void
BEMProblem<3>::compute_gradients_weak(
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<3>                 &analytic_phi);
template void
BEMProblem<2>::compute_gradients_hypersingular_weak(
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<2>                 &analytic_phi,
  const Function<2>                 &trace_gradient);
template void
BEMProblem<3>::compute_gradients_hypersingular_weak(
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn,
  const Function<3>                 &analytic_phi,
  const Function<3>                 &trace_gradient);
template void
BEMProblem<3>::compute_gradients_hypersingular(
  const TrilinosWrappers::MPI::Vector &glob_phi,
  const TrilinosWrappers::MPI::Vector &glob_dphi_dn);
