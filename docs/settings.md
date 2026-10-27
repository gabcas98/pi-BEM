# Settings of the mixed Galerkin solver

The defaults of the code are conservative. Every option that depends on the
geometry (raised rules at sharp edges, refinement by curvature) or trades
accuracy for time (relaxed products, a coarser preconditioner) is off unless
set, so an existing parameter file keeps its results. A CAD geometry with
sharp edges needs the settings below. They are those of the level 3 run of
the NACA wing with the plane wave u = x + y + z (p = 2, exact parametric
geometry, FMM, 8 processes): 90 s, flux L2 error 6.6e-5 against 3.7e-5 for
the best approximation. The measurements behind each value are in
`docs/naca_accuracy.md`.

## Recommended settings for a CAD geometry with sharp edges

### Geometry and datum (`BEMProblem<3>`)

```text
set Exact geometry = true
set Exact geometry source = cad
set Exact geometry map = parametric
set Mixed Galerkin Dirichlet datum = projection
set Galerkin far field tolerance = 1e-10
```

The L2-projected datum (the default) keeps the flux error within about the
local best approximation. The nodal interpolant (`interpolation`) leaves a
smooth error that the flux receives one order lower.
A far-field tolerance of 1e-10 changes no printed digit and halves the far
field on fine meshes. 1e-8 changes results by about 1e-5, and 1e-6 is wrong.

### Quadrature (`BEMProblem<3>/Quadrature rules`)

```text
set Galerkin singular quadrature order = 7
set Galerkin singular quadrature order for kinked pairs = 10
set Galerkin kinked pair angle = 60
set Galerkin singular quadrature order for folded pairs = 14
set Galerkin near singular distance ratio for kinked pairs = 1.0
```

- The base singular order stays 7. On the finest quadratic mesh, order 6
  costs 0.9% of the flux and order 5 costs 8%, although coarse meshes show
  no difference.
- *Kinked pairs*: these are touching pairs whose vertex normals make more
  than the given angle, such as the edges of the tips. Choose the angle
  between the angles of touching cells in the smooth surfaces (0–45 degrees
  on the wing, nose included) and those across the edges (75–105 degrees).
  Order 10 removes the peaks of the flux at the corners of the leading edge
  with the tips, at no measurable cost.
- *Folded pairs*: these are touching pairs with nearly opposite normals, as
  at the trailing edge. Order 10 leaves 2e-3 there at level 3, while order 14
  leaves 1e-4.
- *Kinked distance ratio*: non-touching pairs with normals beyond the kinked
  angle become near-singular up to this ratio of the larger diameter.
  Examples are the upper and lower cells across the thin aft section, where
  the double layer of one surface on the other is of order one. With 1.0 the
  aft error falls from 2.75e-3 to 4.5e-4. Pairs between the base ratio (0.5)
  and this one take a single tensor rule of matching order (at most 12),
  not a subdivision.

### Mesh (`ComputationalDomain<3>`)

```text
set Curvature refinement cycles = 1
set Curvature refinement angle = 10
```

These cut, in one direction only, the cells across which the surface turns
by more than the angle. At the nose of the wing that direction is the
chord, so the new lines run along the span. Flux at level 3: 2.41e-4 ->
6.6e-5 with 4.5% more cells.

### Solver (`BEMProblem<3>`)

```text
set Mixed Galerkin preconditioner = calderon
set Mixed Galerkin solver reduction = 1e-9
set Mixed Galerkin relaxed products below = 1e-5
```

`Mixed Galerkin relaxed products below` is inexact GMRES. Once the
preconditioned residual falls below this fraction of the preconditioned
right-hand side, the products use cheaper expansions. Take the fraction as
about the reduction divided by the relative error of the relaxed products.
`PIBEM_MIXED_DIAGNOSTICS` prints that error: 1.3–1.7e-4 for order 4 and
opening 0.6. The flux then changes in the fourth digit.

### FMM (`BEMProblem<3>` and `BEMFMA<3>/FMA Params`)

```text
set Solution method = FMA                                  # BEMProblem<3>

set FMA Truncation Order = 8                               # FMA Params
set Galerkin opening criterion = 0.4
set Galerkin maximum points per leaf = 64
set Galerkin preconditioner truncation order = 2
set Galerkin preconditioner opening criterion = 0.95
set Galerkin preconditioner maximum points per leaf = 32
set Galerkin relaxed truncation order = 4
set Galerkin relaxed opening criterion = 0.6
```

The preconditioner only needs to be the same operator at every step, not an
accurate one, so its expansions can be coarse: order 2 with opening 0.95 and
32 points per leaf takes the same number of steps as order 4 with opening
0.6, at half the cost. The relaxed products, however, enter the solution and
need their own expansions. When `Galerkin relaxed truncation order` is zero
they use those of the preconditioner, so a coarse preconditioner without
them changes the solution.

### Regression cases

`RUN_REGRESSION/run_regression.py` exercises these settings:

- `naca_mixed_exact_p2_fmm`: kinked order 10 and kinked ratio 1.0;
- `naca_mixed_curvature_p2_fmm`: curvature refinement;
- `naca_mixed_tips_dirichlet_p1_fmm`: a Dirichlet part in several pieces,
  relaxed products, and a coarse preconditioner with separate relaxed
  expansions.
- `robin_real_fma`, `robin_complex_direct`, `robin_complex_fma`, `robin_2d`:
  the Robin and complex problems of upstream pi-BEM on the sphere of its
  samples (a Robin datum that is not zero with the multipole method) and on
  a decagon in 2D.
- `robin_galerkin_fma`: a Robin part in the mixed Galerkin problem, multipole
  method.
- `slots_mixed_fma`, `slots_dirichlet_dg`, `slots_dirichlet_fma`: two slots,
  each exact on its manifold and wrong on the other, in the Galerkin data.

## Environment switches

Switches that are only for diagnostics and comparisons are set by
environment variables, not parameters. Unless a value is given, any value
enables them.

### Mixed problem

| Variable | Effect |
|---|---|
| `PIBEM_MIXED_DIAGNOSTICS` | Runs the checks that cost products: \|W 1\|, the residual of the exact solution by kind of row, and the error of the relaxed products against the exact ones. |
| `PIBEM_RESIDUAL_DUMP=file` | For every flux and free potential unknown, writes its support point, whether it is a flux, the residual of the exact solution and the right-hand side. |
| `PIBEM_MIXED_ERROR_DUMP=file` | One line per cell: its center, whether it is a Dirichlet cell, whether it touches a double node, its squared errors and norm, and the squared error of the best approximation. |
| `PIBEM_FORCE_MIXED` | Uses the mixed solver also when the whole boundary is Dirichlet, to compare with the Dirichlet solver. |
| `PIBEM_CALDERON_LUMPED` | Uses lumped masses in the Calderon preconditioner (about three times the steps). |
| `PIBEM_KEEP_MESHES` | Writes the mesh of every adaptive cycle as `mesh_cycle_<i>.inp`. |

### Near field and FMM

| Variable | Effect |
|---|---|
| `PIBEM_NEAR_PROFILE` | Prints the time of the near field by stage and the parts of the integrator, summed over the processes. |
| `PIBEM_NEAR_MEMORY` | Prints the memory of the process and the rule cache every 2000 near pairs. |
| `PIBEM_NO_CONGRUENCE` | Integrates every near pair instead of reusing the block of a congruent pair. |
| `PIBEM_NO_PAIR_CACHE` | Turns off the cache of the singular rules of full panels. |
| `PIBEM_UNFUSED_ROWS` | Uses three separate FMM evaluations for the rows of the mixed problem instead of the fused pass. |
| `PIBEM_CALDERON_DIPOLE_KT` | Computes K' from three dipole sums. |

### Older diagnostics

| Variable | Effect |
|---|---|
| `PIBEM_COMPARE_RHS` | Assembles the manufactured right-hand side next to the consistent one in the Dirichlet Galerkin problem. |
| `PIBEM_ENERGY_SELFTEST` | Checks the energy-norm error against the best L2 projection (right triangle). |
| `PIBEM_WEAK_GRADIENT_CHECK` | Checks the surface divergence theorem in `compute_gradients_weak()`. |
| `PIBEM_HYPERSINGULAR_WEAK_CHECK` | Replaces the Galerkin flux with the exact one in `compute_gradients_hypersingular_weak()`. |
| `PIBEM_HW_DEBUG` | Splits the errors of `compute_gradients_hypersingular_weak()` by region. |
| `PIBEM_HW_DUFFY_ORDER`, `PIBEM_HW_NEAR_RATIO`, `PIBEM_HW_PAIR_RATIO`, `PIBEM_HW_PAIR_DEPTH`, `PIBEM_HW_PAIR_ORDER`, `PIBEM_HW_POINTS` | Set the rules of `compute_gradients_hypersingular_weak()`. Defaults: Duffy order 12, near ratio 0.5, pair ratio, depth and order from the near-singular parameters (depth at least 10), and degree + 4 points. |
| `PIBEM_HYP_GRADIENT_CHECK` | Reports the free-term matrices of `compute_gradients_hypersingular()`. |
