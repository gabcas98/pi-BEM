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

#include "../include/mixed_calderon_preconditioner.h"

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <chrono>

namespace
{
  double
  seconds_since(const std::chrono::steady_clock::time_point &start)
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
      .count();
  }
} // namespace



MixedCalderonPreconditioner::MixedCalderonPreconditioner(
  const std::vector<Kind>                                     &kind,
  const std::vector<types::global_dof_index>                  &master,
  const std::vector<unsigned int>                             &piece,
  const SparseMatrix<double>                                  &mass,
  const std::function<Vector<double>(const Vector<double> &)> &single_layer,
  Products                                                     products,
  const bool                                                   lumped)
  : n(kind.size())
  , kind(kind)
  , master(master)
  , piece(piece)
  , products(std::move(products))
  , lumped(lumped)
  , flux_index(n, numbers::invalid_dof_index)
  , master_index(n, numbers::invalid_dof_index)
{
  Vector<double> ones(n);
  ones = 1.0;
  lumped_mass.reinit(n);
  mass.vmult(lumped_mass, ones);
  master_mass.reinit(n);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == free_potential || kind[e] == tied_potential)
      master_mass[master[e]] += lumped_mass[e];

  // The rank-one terms: 1 / (1^T V 1) on each piece.
  unsigned int n_pieces = 0;
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == flux)
      n_pieces = std::max(n_pieces, piece[e] + 1);
  alpha.resize(n_pieces);
  for (unsigned int p = 0; p < n_pieces; ++p)
    {
      Vector<double> piece_ones(n);
      for (types::global_dof_index e = 0; e < n; ++e)
        if (kind[e] == flux && piece[e] == p)
          piece_ones[e] = 1.0;
      const Vector<double> v_ones   = single_layer(piece_ones);
      double               constant = 0;
      for (types::global_dof_index e = 0; e < n; ++e)
        if (kind[e] == flux && piece[e] == p)
          constant += v_ones[e];
      alpha[p] = 1.0 / constant;
    }

  // The consistent masses of the flux unknowns and of the free potentials
  // (their tied copies folded into them), factorized once.
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == flux)
      flux_index[e] = n_flux++;
    else if (kind[e] == free_potential)
      master_index[e] = n_masters++;
  if (lumped)
    return;
  const auto master_of = [&](const types::global_dof_index e) {
    return kind[e] == free_potential || kind[e] == tied_potential ?
             master_index[master[e]] :
             numbers::invalid_dof_index;
  };
  DynamicSparsityPattern d_flux(n_flux, n_flux), d_master(n_masters, n_masters);
  for (auto entry = mass.begin(); entry != mass.end(); ++entry)
    {
      const auto e = entry->row(), f = entry->column();
      if (flux_index[e] != numbers::invalid_dof_index &&
          flux_index[f] != numbers::invalid_dof_index)
        d_flux.add(flux_index[e], flux_index[f]);
      if (master_of(e) != numbers::invalid_dof_index &&
          master_of(f) != numbers::invalid_dof_index)
        d_master.add(master_of(e), master_of(f));
    }
  flux_pattern.copy_from(d_flux);
  master_pattern.copy_from(d_master);
  flux_mass.reinit(flux_pattern);
  master_mass_matrix.reinit(master_pattern);
  for (auto entry = mass.begin(); entry != mass.end(); ++entry)
    {
      const auto e = entry->row(), f = entry->column();
      if (flux_index[e] != numbers::invalid_dof_index &&
          flux_index[f] != numbers::invalid_dof_index)
        flux_mass.add(flux_index[e], flux_index[f], entry->value());
      if (master_of(e) != numbers::invalid_dof_index &&
          master_of(f) != numbers::invalid_dof_index)
        master_mass_matrix.add(master_of(e), master_of(f), entry->value());
    }
  // Either part may be empty (PIBEM_FORCE_MIXED without a Neumann part);
  // UMFPACK does not take an empty matrix.
  if (n_flux > 0)
    flux_solver.initialize(flux_mass);
  if (n_masters > 0)
    master_solver.initialize(master_mass_matrix);
}



Vector<double>
MixedCalderonPreconditioner::inverse_flux_mass(const Vector<double> &r) const
{
  Vector<double> out(n);
  if (lumped)
    {
      for (types::global_dof_index e = 0; e < n; ++e)
        if (kind[e] == flux)
          out[e] = r[e] / lumped_mass[e];
      return out;
    }
  if (n_flux == 0)
    return out;
  Vector<double> reduced(n_flux);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == flux)
      reduced[flux_index[e]] = r[e];
  flux_solver.solve(reduced);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == flux)
      out[e] = reduced[flux_index[e]];
  return out;
}



Vector<double>
MixedCalderonPreconditioner::inverse_master_mass(const Vector<double> &r) const
{
  Vector<double> out(n);
  if (lumped)
    {
      for (types::global_dof_index e = 0; e < n; ++e)
        if (kind[e] == free_potential || kind[e] == tied_potential)
          out[e] = r[master[e]] / master_mass[master[e]];
      return out;
    }
  if (n_masters == 0)
    return out;
  Vector<double> reduced(n_masters);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == free_potential)
      reduced[master_index[e]] = r[e];
  master_solver.solve(reduced);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == free_potential || kind[e] == tied_potential)
      out[e] = reduced[master_index[master[e]]];
  return out;
}



void
MixedCalderonPreconditioner::vmult(Vector<double>       &y,
                                   const Vector<double> &r) const
{
  const auto start = std::chrono::steady_clock::now();
  ++applications;
  y = r;
  std::vector<double> sum_d(alpha.size(), 0.);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == flux)
      sum_d[piece[e]] += r[e];
  auto                 solve_start = std::chrono::steady_clock::now();
  const Vector<double> x_d         = inverse_flux_mass(r);
  const Vector<double> x_n         = inverse_master_mass(r);
  solve_seconds += seconds_since(solve_start);
  Vector<double> w(n), v(n);
  products(x_n, x_d, v, w);
  Vector<double> z_n(n);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == free_potential || kind[e] == tied_potential)
      z_n[master[e]] += v[e];
  solve_start              = std::chrono::steady_clock::now();
  const Vector<double> z_d = inverse_flux_mass(w);
  const Vector<double> zz  = inverse_master_mass(z_n);
  solve_seconds += seconds_since(solve_start);
  for (types::global_dof_index e = 0; e < n; ++e)
    if (kind[e] == flux)
      y[e] = z_d[e] + alpha[piece[e]] * sum_d[piece[e]];
    else if (kind[e] == free_potential)
      y[e] = zz[e];
  total_seconds += seconds_since(start);
}
