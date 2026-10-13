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

#ifndef mixed_calderon_preconditioner_h
#define mixed_calderon_preconditioner_h

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <functional>
#include <vector>

using namespace dealii;

/**
 * Operator preconditioning of the mixed Galerkin problem by the Calderon
 * identities, block by block: the flux rows (V) by M_D^{-1} W M_D^{-1},
 * the potential rows (W) by M_N^{-1} V M_N^{-1}, with the consistent masses
 * of the two parts factorized once (lumped ones need three times the
 * steps), since V M^{-1} W = 1/4 M - K M^{-1} K up to the discretization.
 *
 * W vanishes on the constants of the Dirichlet part, one for each of its
 * pieces whose flux unknowns no cell connects (the cells on the two sides
 * of a sharp edge have separate unknowns), which rank-one terms restore,
 * scaled to V^{-1} on each: the preconditioned operator then takes the
 * value 1 on them, at the top of the rest of its spectrum. Without them it
 * would be singular on all but one, and GMRES would stop on its residual
 * with the true one large.
 *
 * The vectors are those of all indices of the mixed problem (BEMProblem::
 * solve_galerkin_mixed()): flux unknowns, free potentials (the masters of
 * the potentials tied to them) and the rest, which the preconditioner
 * leaves unchanged. Each application costs one call of the products, a
 * product with V and one with W, and four solves with the masses.
 */
class MixedCalderonPreconditioner
{
public:
  enum Kind : unsigned char
  {
    other,
    flux,
    free_potential,
    tied_potential
  };

  /// v = V v_src and w = W w_src, whole vectors of all indices.
  using Products = std::function<void(const Vector<double> &v_src,
                                      const Vector<double> &w_src,
                                      Vector<double>       &v,
                                      Vector<double>       &w)>;

  /**
   * The kind of every index, the master of every free or tied potential,
   * the piece of the Dirichlet part of every flux unknown (numbered from
   * zero), the mass matrix of all indices, V of a vector (for the rank-one
   * terms) and the products of an application; with lumped, the lumped
   * masses.
   */
  MixedCalderonPreconditioner(
    const std::vector<Kind>                                   &kind,
    const std::vector<types::global_dof_index>                &master,
    const std::vector<unsigned int>                           &piece,
    const SparseMatrix<double>                                &mass,
    const std::function<Vector<double>(const Vector<double> &)> &single_layer,
    Products                                                   products,
    const bool                                                 lumped = false);

  /// y = P r.
  void
  vmult(Vector<double> &y, const Vector<double> &r) const;

  unsigned int
  n_pieces() const
  {
    return alpha.size();
  }

  /// The number of applications and their time, and that of the mass
  /// solves within it.
  unsigned int
  n_applications() const
  {
    return applications;
  }

  double
  seconds() const
  {
    return total_seconds;
  }

  double
  mass_seconds() const
  {
    return solve_seconds;
  }

private:
  /// r on the flux unknowns times the inverse mass of the Dirichlet part.
  Vector<double>
  inverse_flux_mass(const Vector<double> &r) const;

  /// r on the masters times the inverse mass of the Neumann part; the
  /// result also on their tied copies.
  Vector<double>
  inverse_master_mass(const Vector<double> &r) const;

  const types::global_dof_index        n;
  const std::vector<Kind>              kind;
  const std::vector<types::global_dof_index> master;
  const std::vector<unsigned int>      piece;
  const Products                       products;
  const bool                           lumped;
  /// The scale of the rank-one term of each piece, 1 / (1^T V 1).
  std::vector<double> alpha;
  /// The lumped masses of all indices, and summed onto the masters.
  Vector<double> lumped_mass, master_mass;
  std::vector<types::global_dof_index> flux_index, master_index;
  types::global_dof_index              n_flux = 0, n_masters = 0;
  SparsityPattern                      flux_pattern, master_pattern;
  SparseMatrix<double>                 flux_mass, master_mass_matrix;
  mutable SparseDirectUMFPACK          flux_solver, master_solver;

  mutable unsigned int applications  = 0;
  mutable double       total_seconds = 0, solve_seconds = 0;
};

#endif
