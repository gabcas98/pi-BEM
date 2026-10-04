// ---------------------------------------------------------------------
//
// Copyright (C) 2014 - 2020 by the pi-BEM authors.
//
// This file is part of the pi-BEM library.
//
// The pi-BEM is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License version 2.1 as published by the Free Software Foundation.
// The full text of the license can be found in the file LICENSE at
// the top level of the pi-BEM distribution.
//
// Authors: Nicola Giuliani, Andrea Mola, Luca Heltai

#ifndef bem_problem_h
#define bem_problem_h
// @sect3{Include files}

// The program starts with including a bunch
// of include files that we will use in the
// various parts of the program.


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

// #include <deal.II/lac/petsc_vector.h>
// #include <deal.II/lac/petsc_parallel_vector.h>
// #include <deal.II/lac/petsc_parallel_sparse_matrix.h>
// #include <deal.II/lac/petsc_solver.h>
// #include <deal.II/lac/petsc_precondition.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q1_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>



// And here are a few C++ standard header
// files that we will need:

#include <mpi.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>

#include "../include/ass_leg_function.h"
#include "../include/bem_fma.h"
#include "../include/galerkin_fmm_operator.h"
#include "../include/cad_projectors.h"
#include "../include/computational_domain.h"
#include "../include/constrained_matrix.h"
#include "../include/local_expansion.h"
#include "../include/multipole_expansion.h"
#include "../include/octree_block.h"
#include "../include/projected_mapping.h"
#include "../include/projected_quadrature.h"

using namespace dealii;

// using namespace TrilinosWrappers;
// using namespace TrilinosWrappers::MPI;

/**
 * - BEMProblem. This class is the core of the BEM simulation
 *   - it receives the variables vector filled in with the proper boundary
 * condition;
 *   - it creates the codimension 1 functional space setting up the BEM;
 *   - it solves the system using a preconditioned parallel GMRES solver;
 *   - it eventually interacts with the FMM accelerator.
 */
template <int dim>
class BEMProblem : public ParameterAcceptor
{
public:
  typedef typename DoFHandler<dim - 1, dim>::active_cell_iterator cell_it;

  BEMProblem(ComputationalDomain<dim> &comp_dom,
             const MPI_Comm            comm = MPI_COMM_WORLD);

  void
  solve(TrilinosWrappers::MPI::Vector       &phi,
        TrilinosWrappers::MPI::Vector       &dphi_dn,
        const TrilinosWrappers::MPI::Vector &tmp_rhs,
        const Function<dim>                 *dirichlet_function = nullptr,
        const Function<dim>                 *dirichlet_gradient = nullptr);

  /// This function takes care of the proper initialization of all the elements
  /// needed by the bem problem class. Since we need to sum elements associated
  /// with scalar and vectorial Finite Element spaces we have chosen to renumber
  /// the dofs and force the the IndexSet for the parallel partitioning to be
  /// consistent. Without this enforcing we are getting in trouble with ghost
  /// elements. We set up the two TrilinosSparsityPattern to be used in our
  /// computations (assemble system and compute_normals-gradients).
  void
  reinit();

  const Quadrature<dim - 1> &
  get_singular_quadrature(const unsigned int index) const;

  const Quadrature<dim - 1>
  get_quasi_singular_quadrature(const typename DoFHandler<dim-1,dim>::active_cell_iterator &cell,
                                const Mapping<dim-1,dim> &mapping,
                                const Point<dim> &singularity,
                                const Point<dim-1> &ref_projection,
                                const unsigned int exponent,
                                const unsigned int quadrature_order_override = 0,
                                const double normalized_distance =
                                  std::numeric_limits<double>::infinity()) const;

  /// This function compute a very specific case, a double node that has a
  /// dirichlet-dirichlet condition. In this case there is a constraint for
  /// the normal derivative since we want a conitnuos velocity thus a conitnuos
  /// total gradient. We have solved this problem using an analytical expression
  /// for these constraints. Since we need to know all the double nodes set we
  /// have kept this function serial. We stress that it needs to be called only
  /// once.
  void
  compute_constraints(IndexSet                            &c_cpu_set,
                      AffineConstraints<double>           &constraints,
                      const TrilinosWrappers::MPI::Vector &tmp_rhs);

  //  private:

  /// We declare the parameters needed by the class. We made good use of the
  /// deal.ii SwissArmyKnife library. The parameters will be read from a file if
  /// it is existent or a file will be created. The class need a controller for
  /// the GMRES solver, quadrature rules, resolution strategy (direct or fma).
  virtual void
  declare_parameters(ParameterHandler &prm) override;

  /// We declare the parameters needed by the class. We made good use of the
  /// deal.ii SwissArmyKnife library.
  virtual void
  parse_parameters(ParameterHandler &prm) override;

  /// This function computes the free coefficients appearing  in the
  /// hypersingular BIE.
  void
  compute_hypersingular_free_coeffs();

  /// This function computes the fraction of solid angles seen by our domain. We
  /// use the Double Layer Operator (through the Neumann matrix) to determine
  /// it.
  void
  compute_alpha();
  
  /// This function computes the free coefficients for the Screened Poisson 
  /// problem. The computation involves evaluating exp(-kappa/sqrt(3) * (x+y+z))
  /// and its normal derivative on the degrees of freedom.
  void
  compute_alpha(const double kappa);

  /// This function assembles the full distributed matrices needed by the direct
  /// method. We compute both the Double Layer Operator (Neumann matrix) and
  /// Single Layer Operator (Dirichlet matrix). Then we have to use dirichlet
  /// and neumann vector to assemble properly the system matrix and its right
  /// hand side.
  void
  assemble_system();

  void
  assemble_galerkin_system();

  /// With Solution method = FMA, the Galerkin single-layer operator is
  /// applied by GalerkinFMMOperator instead of the dense matrix, which is
  /// then never allocated. This sets it up: geometry, octree, near field of
  /// the operator and of the right-hand sides that the solve will need.
  void
  setup_galerkin_fmm(const Function<dim> *dirichlet_function,
                     const Function<dim> *exact_gradient);

  void
  initialize_calderon_dual_space();


  /// The next three methods are
  /// needed by the GMRES solver:
  /// the first provides result of
  /// the product of the system
  /// matrix (a combination of Neumann
  /// and Dirichlet matrices) by the
  /// vector src. The result is stored
  /// in the vector dst.
  void
  vmult(TrilinosWrappers::MPI::Vector       &dst,
        const TrilinosWrappers::MPI::Vector &src) const;

  void
  apply_single_layer(TrilinosWrappers::MPI::Vector       &dst,
                     const TrilinosWrappers::MPI::Vector &src) const;

  void
  apply_jump_term(TrilinosWrappers::MPI::Vector       &dst,
                  const TrilinosWrappers::MPI::Vector &src) const;

  /// The second method computes the
  /// right hand side vector of the
  /// system.

  void
  compute_rhs(TrilinosWrappers::MPI::Vector       &dst,
              const TrilinosWrappers::MPI::Vector &src) const;

  void
  assemble_explicit_system(
    TrilinosWrappers::SparseMatrix &system_matrix) const;

  void
  assemble_exact_regularized_dirichlet_rhs(
    const Function<dim> &dirichlet_function);

  void
  assemble_exact_galerkin_dirichlet_rhs(
    const Function<dim> &dirichlet_function,
    const Function<dim> *exact_gradient = nullptr);

  void
  solve_galerkin_adjoint_double_layer(
    TrilinosWrappers::MPI::Vector &dphi_dn,
    const Function<dim>           *dirichlet_gradient = nullptr);

  /// The third method computes the
  /// product between the solution vector
  /// and the (fully populated) sytstem
  /// matrix.

  /// This function assembles in parallel the band preconditioner to be used in
  /// the direct resolution method.
  void
  assemble_preconditioner();

  /// This is the function that guides the execution of the BEM problem.
  /// Depending on the resolution stategy we go whether for the direct or fma
  /// strategy.
  void
  solve_system(TrilinosWrappers::MPI::Vector       &phi,
               TrilinosWrappers::MPI::Vector       &dphi_dn,
               const TrilinosWrappers::MPI::Vector &tmp_rhs,
               const Function<dim>                 *dirichlet_gradient =
                 nullptr);


  void
  output_results(const std::string);

  /// We have parallelised the computation of the surface gradients. We need a
  /// solution vector that has also ghost cells. for this reason we made use of
  /// a ghosted IndexSet that we have computed in the reinit function. After
  /// this we simply make use of deal.ii and its TrilinosWrappers to built and
  /// solve a mass matrix system.
  void
  compute_surface_gradients(const TrilinosWrappers::MPI::Vector &tmp_rhs);

  /// We have parallelised the computation of gradients. We need a
  /// solution vector that has also ghost cells. for this reason we made use of
  /// a ghosted IndexSet that we have computed in the reinit function. After
  /// this we simply make use of deal.ii and its TrilinosWrappers to built and
  /// solve a mass matrix system. We want the gradients to be continuos so we
  /// need  to make good use of both surface gradients and the normal
  /// derivative.
  /** Energy norm of the flux error, sqrt of the double integral of the
   * single-layer kernel against e(x) e(y) with e = lambda - lambda_h. This is
   * the norm in which the Galerkin method for the first-kind equation is
   * quasi-optimal; the L2 norm is not, so on a graded mesh the ratio of the L2
   * error to the best L2 projection is not a diagnostic of the method.
   *
   * Costs one pass over all panel pairs, so it is gated by a parameter.
   */
  double
  compute_energy_norm_error(const TrilinosWrappers::MPI::Vector &dphi_dn,
                            const Function<dim> &exact_gradient) const;

  void
  compute_gradients(const TrilinosWrappers::MPI::Vector &phi,
                    const TrilinosWrappers::MPI::Vector &dphi_dn);


  /// We compute the potential gradients also in an alternative way.
  /// Here we make use of the hypersingular integrals computed with the
  /// SingularKernelIntegral class
  void
  compute_gradients_hypersingular(const TrilinosWrappers::MPI::Vector &phi,
                                  const TrilinosWrappers::MPI::Vector &dphi_dn);

  /// Recover the gradient in weak form: the tangential part comes from
  /// integrating by parts on each cell, so the trace is never differentiated
  /// and may be taken from the analytical Dirichlet datum when one is given.
  void
  compute_gradients_weak(const TrilinosWrappers::MPI::Vector &phi,
                         const TrilinosWrappers::MPI::Vector &dphi_dn,
                         const Function<dim> *analytic_phi = nullptr,
                         const bool           separate_normal = true);

  /// Recover the gradient from the Galerkin form of its boundary integral
  /// representation, the route that leads to the hypersingular equation. On
  /// the boundary the representation gives
  ///     grad u = grad_S (2 sigma (V lambda - K phi)) + 2 sigma (K' lambda + W phi) n,
  /// with sigma the orientation of the mesh normals relative to the outward
  /// normal of the domain. The tangential part is tested with the vector
  /// space and integrated by parts on each cell, edges included, so neither
  /// the trace nor the potentials are differentiated; W is taken in Maue's
  /// form, with the edge terms that discontinuous test functions leave. Only
  /// the tangential gradient of the trace enters, through n x grad phi; the
  /// normal component of trace_gradient is used for diagnostics alone. With
  /// discrete_trace the tangential part uses the discrete trace instead of
  /// the analytical one and the normal part is the Galerkin flux.
  void
  compute_gradients_hypersingular_weak(
    const TrilinosWrappers::MPI::Vector &phi,
    const TrilinosWrappers::MPI::Vector &dphi_dn,
    const Function<dim>                 &analytic_phi,
    const Function<dim>                 &trace_gradient,
    const bool                           discrete_trace);

  /// We have parallelised the computation of the L2 projection of the normal
  /// vector. We need a solution vector that has also ghost cells. for this
  /// reason we made use of a ghosted IndexSet that we have computed in the
  /// reinit function. After this we simply make use of deal.ii and its
  /// TrilinosWrappers to built and solve a mass matrix system. In this function
  /// we don't need any vector with ghost cells.
  void
  compute_normals();

  /// this method is needed to
  /// separate Dirichlet dofs from
  /// Neumann nodes.

  void
  compute_dirichlet_and_neumann_dofs_vectors();


  /// in the imported mesh, the nodes on the
  /// domain edges are doubled: this routine
  /// creates a std::vector of std::set which
  /// allows to relate each node to their
  /// double(s). Since the geometry is shared among
  /// all processors we can let every processors to compute_normals
  /// the overall double nodes set.

  void
  compute_double_nodes_set();

  void
  compute_reordering_vectors();

  void
  adaptive_refinement(const TrilinosWrappers::MPI::Vector &error_vector);



  ConditionalOStream        pcout;
  ComputationalDomain<dim> &comp_dom;

  std::string                                  scalar_fe_type, vector_fe_type;
  unsigned int                                 scalar_fe_order, vector_fe_order;
  std::unique_ptr<FiniteElement<dim - 1, dim>> fe;
  std::unique_ptr<FiniteElement<dim - 1, dim>> gradient_fe;
  std::unique_ptr<FiniteElement<dim - 1, dim>> calderon_fe;
  DoFHandler<dim - 1, dim>                     dh;
  DoFHandler<dim - 1, dim>                     gradient_dh;
  DoFHandler<dim - 1, dim>                     calderon_dh;

  // FE_Q<dim-1,dim>                   fe;
  // FESystem<dim-1,dim>      gradient_fe;

  double refinement_threshold, coarsening_threshold;

  /// An Eulerian Mapping is created to deal
  /// with the free surface and boat mesh
  /// deformation

  Vector<double>                         map_vector;
  std::shared_ptr<Mapping<dim - 1, dim>> mapping;
  unsigned int                           mapping_degree;

  /// Integrate over the exact geometry: the mapping is an
  /// ExactGeometry::ProjectedMapping of degree mapping_degree, whose
  /// projectors come from the CAD surfaces of the domain or from the
  /// analytic manifolds attached to the triangulation.
  bool         exact_geometry = false;
  std::string  exact_geometry_source;
  /// Degree of the polynomial interpolation of the exact geometry, a
  /// MappingQCache filled by ExactGeometry::interpolate_exact_geometry, or
  /// zero for the lowest degree that meets the tolerance; zero degree and
  /// zero tolerance project every point.
  unsigned int exact_geometry_degree    = 0;
  double       exact_geometry_tolerance = 1e-10;
  bool         interpolate_exact_geometry = false;
  std::shared_ptr<ExactGeometry::ProjectedMapping<dim - 1, dim>>
    exact_mapping;
  std::shared_ptr<MappingQCache<dim - 1, dim>> exact_mapping_cache;
  /// The interpolation of the previous reinit(), kept alive until the next
  /// one: objects built in reinit(), such as the FMA, observe the mapping
  /// and are rebound to the new one only after it is made.
  std::shared_ptr<MappingQCache<dim - 1, dim>> retired_mapping_cache;
  std::vector<std::unique_ptr<ExactGeometry::Projector<dim - 1, dim>>>
    exact_projectors;
  std::unique_ptr<ExactGeometry::ProjectorCollection<dim - 1, dim>>
    exact_projector_collection;

  /// The mapping onto the exact geometry, with the projectors.
  std::shared_ptr<Mapping<dim - 1, dim>>
  make_exact_geometry_mapping();
  Vector<double>                         map_points;


  /// these are the std::vectors of std::sets
  /// containing informations on multiple
  /// nodes on the edges: one vector is
  /// created for the points associated with
  /// the degrees of freedom of the potential
  /// function, and one is created for the
  /// points associated with the degrees of
  /// freedom of its gradient (a vector field)

  std::vector<std::set<types::global_dof_index>> double_nodes_set;
  std::vector<std::set<types::global_dof_index>> gradient_double_nodes_set;



  std::shared_ptr<Quadrature<dim - 1>> quadrature;
  unsigned int                         quadrature_order;

  /// the number of standard quadrature points
  /// and singular kernel quadrature to be
  /// used
  unsigned int singular_quadrature_order;
  unsigned int galerkin_quadrature_order;
  unsigned int galerkin_singular_quadrature_order;
  unsigned int galerkin_duffy_outer_subdivisions;
  double       galerkin_near_singular_distance_ratio;
  unsigned int galerkin_near_singular_maximum_depth;
  unsigned int galerkin_near_singular_quadrature_order;
  std::string  galerkin_singular_quadrature;
  unsigned int quasi_singular_neumann_quadrature_order;
  unsigned int quasi_singular_dirichlet_quadrature_order;
  bool         adaptive_quadrature;
  double       adaptive_quadrature_relative_tolerance;
  double       adaptive_quadrature_absolute_tolerance;
  unsigned int adaptive_quadrature_max_order;
  double       adaptive_quadrature_order_growth;
  double       quasi_singular_distance_ratio;
  bool         near_singular_subdivision;
  double       near_singular_subdivision_ratio;
  unsigned int near_singular_subdivision_max_levels;

  unsigned int hypersingular_correction_steps;
  double       hypersingular_correction_relaxation;
  double       hypersingular_correction_max_relative_update;


  TrilinosWrappers::SparsityPattern full_sparsity_pattern;
  TrilinosWrappers::SparsityPattern calderon_sparsity_pattern;
  TrilinosWrappers::SparseMatrix    neumann_matrix;
  TrilinosWrappers::SparseMatrix    dirichlet_matrix;
  TrilinosWrappers::SparseMatrix    jump_matrix;
  TrilinosWrappers::SparseMatrix    calderon_hypersingular_matrix;
  TrilinosWrappers::SparseMatrix    calderon_mass_matrix;
  TrilinosWrappers::SparseMatrix    adjoint_double_layer_matrix;
  TrilinosWrappers::SparseMatrix    adjoint_single_layer_matrix;

  std::vector<TrilinosWrappers::MPI::Vector>
    laplace_patch_normal_components;
  std::vector<TrilinosWrappers::MPI::Vector>
    laplace_patch_correction_coefficients;
  bool laplace_patch_correction_initialized;

  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::MPI::Vector exact_regularized_dirichlet_rhs;
  TrilinosWrappers::MPI::Vector adjoint_double_layer_rhs;
  TrilinosWrappers::MPI::Vector adjoint_first_kind_rhs;

  /// solution and alpha vectors
  TrilinosWrappers::MPI::Vector sol;
  TrilinosWrappers::MPI::Vector alpha;
  /// an alternatively computed alpha vector (obtained with geometric
  /// computations)
  TrilinosWrappers::MPI::Vector hyp_alpha;
  /// a set of distributed vectors which contain all the entries of the
  /// C_ij tensor appearing in the hypersingular BIE
  std::vector<TrilinosWrappers::MPI::Vector> C_ij;
  /// a set of distributed vectors which contain all the entries of the
  /// b_i vector appearing in the hypersingular BIE
  std::vector<TrilinosWrappers::MPI::Vector> b_i;

  mutable TrilinosWrappers::MPI::Vector serv_phi;
  mutable TrilinosWrappers::MPI::Vector serv_dphi_dn;
  TrilinosWrappers::MPI::Vector         serv_tmp_rhs;

  AffineConstraints<double> constraints;

  std::string preconditioner_type;

  double calderon_stabilization;
  double calderon_dual_bubble_scaling;
  bool   galerkin_component_rhs_diagnostics;

  /// compute the energy norm of the flux error after the solve; costs one
  /// extra pass over all panel pairs, so it is off by default
  bool   compute_energy_norm;
  bool        detailed_error_diagnostics;
  std::string gradient_recovery;
  double galerkin_far_field_tolerance;
  /// Assemble the single layer and the double-layer right-hand side in one
  /// pass over the panel pairs, sharing their quadrature.
  bool   galerkin_fused_rhs;
  /// Evaluate the near-field rules from the reconstructed cell geometry
  /// instead of FEValues.
  bool   galerkin_light_geometry;
  /// Set by solve() while assemble_system() runs, so that the Galerkin
  /// assembly can build the right-hand side together with the matrix; and
  /// whether it did.
  const Function<dim> *galerkin_fused_dirichlet_function = nullptr;
  bool                 galerkin_rhs_from_fused_assembly  = false;

  IndexSet                  calderon_cpu_set;
  AffineConstraints<double> calderon_constraints;
  std::vector<std::vector<types::global_dof_index>>
    calderon_dual_cell_vertices;
  std::vector<unsigned int> calderon_dual_vertex_valence;
  std::vector<double>       calderon_dual_cell_areas;
  std::vector<unsigned int> calderon_dual_cell_components;
  std::vector<double>       calderon_dual_component_areas;
  std::vector<std::set<types::material_id>> calderon_dof_material_ids;

  std::string mapping_type;

  std::string solution_method;

  std::string linear_solver;

  std::string mumps_ordering;

  std::string boundary_integral_formulation;

  std::string galerkin_neumann_formulation;

  std::string galerkin_adjoint_continuity;

  bool         galerkin_adjoint_component_weighting;
  unsigned int galerkin_adjoint_primary_material_id;
  std::set<types::material_id> galerkin_adjoint_primary_material_ids;
  double       galerkin_adjoint_primary_eta_scale;
  double       galerkin_adjoint_secondary_eta_scale;
  bool         galerkin_adjoint_local_eta_scaling;
  bool         galerkin_adjoint_dual_residual_stabilization;
  std::string  galerkin_adjoint_dual_residual_lift;
  bool         galerkin_adjoint_block_completion;
  bool         galerkin_adjoint_hybrid_system;
  bool         galerkin_hybrid_exact_trace_diagnostics;
  bool         galerkin_analytical_hypersingular_rhs;
  bool         galerkin_hybrid_conforming_complement;
  double       galerkin_hybrid_complement_mass_regularization;
  double       galerkin_hybrid_local_mass_regularization;
  double       galerkin_hybrid_local_fluctuation_regularization;
  double       galerkin_hybrid_local_jump_regularization;
  bool         galerkin_hybrid_local_jump_h_scaling;
  unsigned int galerkin_hybrid_local_mass_material_id;
  double       galerkin_hybrid_local_mass_radius_min;
  double       galerkin_hybrid_local_mass_radius_max;
  unsigned int galerkin_hybrid_first_kind_polishing_steps;
  double       galerkin_hybrid_first_kind_polishing_relaxation;

  std::string kernel_type;

  std::string dirichlet_rhs_treatment;

  double screened_kappa;

  /// Screening parameter passed to the Galerkin panel-pair integrators:
  /// kappa for the screened kernel, zero for the Laplace kernel.
  double
  galerkin_screening() const
  {
    return kernel_type == "screened" ? screened_kappa : 0.0;
  }

  SolverControl solver_control;

  bool   enforce_laplace_linear_patch;

  // TODO AMG preconditioner
  TrilinosWrappers::PreconditionILU preconditioner;

  TrilinosWrappers::SparsityPattern preconditioner_sparsity_pattern;

  TrilinosWrappers::SparseMatrix band_system;

  types::global_dof_index preconditioner_band;

  bool is_preconditioner_initialized;

  bool continuos_gradient;

  MPI_Comm mpi_communicator;

  unsigned int n_mpi_processes;

  unsigned int this_mpi_process;

  /// the following vector is needed to
  /// treat Dirichlet nodes.
  /// Each component
  /// is null if it corresponds
  /// to a Dirichlet node, and zero if
  /// it corresponds to a Neumann node.
  TrilinosWrappers::MPI::Vector dirichlet_nodes;
  /// The vector has instead null
  /// entries for Dirichlet nodes, and ones
  /// for Neumann nodes
  TrilinosWrappers::MPI::Vector neumann_nodes;



  /// The IndexSet for the problem without considering any ghost element for the
  /// scalar FE
  IndexSet this_cpu_set;
  /// The IndexSet for the problem considering every ghost element for the
  /// scalar FE
  IndexSet ghosted_set;
  /// The IndexSet for the problem without considering any ghost element for the
  /// vector FE
  IndexSet vector_this_cpu_set;

  IndexSet constr_cpu_set;

  IndexSet edge_set;

  TrilinosWrappers::MPI::Vector vector_gradients_solution;
  /// Whether the normal component of the recovered gradient is meant to be
  /// the flux (Gradient recovery = weak). The gradient at a point is then
  /// also evaluated as the projected field with its normal component
  /// replaced by lambda_h, the normal being that of the mapped surface
  /// there: the representation of the curved normal field by the vector
  /// space no longer enters it.
  bool gradient_normal_from_flux = false;

  TrilinosWrappers::MPI::Vector vector_surface_gradients_solution;

  TrilinosWrappers::MPI::Vector vector_normals_solution;

  std::vector<types::global_dof_index> start_per_process;

  std::vector<types::global_dof_index> vector_start_per_process;

  TrilinosWrappers::SparsityPattern vector_sparsity_pattern;

  AffineConstraints<double> vector_constraints;

  std::vector<types::global_dof_index> original_to_sub_wise;

  std::vector<types::global_dof_index> sub_wise_to_original;

  std::vector<types::global_dof_index> vec_original_to_sub_wise;

  std::vector<types::global_dof_index> vec_sub_wise_to_original;

  bool have_dirichlet_bc;

  BEMFMA<dim> fma;

  /// The Galerkin single-layer operator through the FMM; set up only for
  /// the Galerkin formulation with Solution method = FMA.
  std::unique_ptr<GalerkinFMMOperator> galerkin_fmm;

  /// The matrix from which the Galerkin preconditioners take their entries:
  /// the dense single layer, or its near field with the FMM.
  const TrilinosWrappers::SparseMatrix &
  galerkin_preconditioner_matrix() const
  {
    return galerkin_fmm ? galerkin_fmm->near_field_matrix() : dirichlet_matrix;
  }
};

#endif
