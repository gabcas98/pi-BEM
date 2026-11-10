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
- *Adapted rules* (`Galerkin singular quadrature adaptation = true`, off by
  default): the Sauter-Schwab rules follow the shape of each singular pair.
  After the Duffy transformations `x - y = xi D(eta)`, with `D` affine in
  the angular variables `eta` on affine cells. On a skewed cell, or across
  a sharp edge next to one, `|D|` falls to a fraction of its size at a
  point of the box of `eta` or just outside, and the kernels have a peak
  there that the Gauss points resolve slowly. Where the smallest `|D|` is
  below `Galerkin singular quadrature adaptation ratio` (0.3) times the
  largest, the angular variables take the iterated sinh transformation of
  Johnston, Johnston and Elliott, centred where `|D|`, minimised over the
  inner variables, is smallest, on the scale of the width of its peak. The
  order of the variables matters: on a pair with a common vertex across a
  sharp edge next to a skewed cell `|D|` is small along a ridge, and the
  best order is a thousand times more accurate than the worst; the integral
  of `1/|D|^2` over the box, with `n` and `2n` points and no kernel or map,
  chooses it. The number of points does not change, and the rules of
  well-shaped pairs are the same to the last bit. On the thin rhombi (20
  degrees) of the tips at the trailing edge of the wing, with seven points
  per variable, the double layer of the pair across the edge of the tip
  goes from an error of 1.5e-3 to 6.2e-6, that of the pair with a common
  vertex across it from 6.9e-4 to 1.0e-5, the single layer of two rhombi
  with a common edge from 1.8e-5 to 5.5e-7; with the ratio 0.5 a rhombus
  with itself from 1.9e-4 to 7.6e-7. With the multipole order that the mesh
  asks for (12 at level 3), the flux error at the corners of the trailing
  edge and the tips falls at level 3 from 7.6e-5 to 3.9e-6 of the largest
  flux for the point source, and no longer grows with the mesh; a higher
  order for the skewed or the kinked pairs changes nothing more.

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

`FMA Truncation Order`, `Galerkin opening criterion` and `Galerkin maximum
points per leaf` also set the collocation FMM (`CollocationFMMOperator`),
despite their names; the other entries belong to the mixed Galerkin problem.

The order grows with the mesh unless `FMA truncation order from the mesh =
false`: `FMA Truncation Order` is then the smallest order, and the solver
takes `ceil(2 (p+1) log10(D/h) - 2)` where the mesh needs more, for
elements of degree `p`, the diagonal `D` of the bounding box of the surface
and the mean size `h` of its cells (the square root of the area per cell).
The log prints both orders (`FMM truncation order 12: the mesh asks for
12 ...`). The products of the multipole method carry an error that
decreases by about a digit every two orders and does not decrease with the
mesh, while the discretisation error of a smooth solution decreases as
`(h/D)^(p+1)`: at `p = 2` a fixed order stops the convergence. On the
sphere of the campaign of the report (section 12) the rule gives the orders
that were needed there: at `p = 2` order 8 with 1536 cells, 10 with 6144
and 12 with 24576, for which order 8 left the flux 2.5 to 9.5 times its
best approximation; at `p = 1` order 8 up to 98304 cells. On the NACA
wing at level 3, `p = 2`, it asks for 12. Although the global flux error
there is far above the floor of order 8, order 8 leaves the error of the
cells along the edges of the tips growing with the mesh (x3 per level):
with order 12 their error falls from 1.4e-3 to 3.6e-5 of the largest flux
(linear solution); 16 changes no more. The solve takes about 1.4 times as
long.
The collocation FMM takes its near field from the singular and quasi-singular
rules of the matrices, so it reproduces `Solution method = Direct` up to the
truncation: order 10 matches six digits with the Laplace kernel. The screened
kernel converges more slowly in the order; on the box of
`screened_p2_bal0_weak.prm` (kappa 0.5) order 10 gives four digits and order
18 six. `Octree Params` and `Granularity for TBB simple for cycles` belong to
the FMA of upstream, which LaplaceFMM has replaced; they are still accepted
and are ignored.

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
- `naca_mixed_exact_p2_fmm_order_from_mesh`: the same with `FMA Truncation
  Order = 4`, which the mesh raises to 8, so that its values are those of
  the case above;
- `naca_mixed_exact_p2_fmm_adapted`: the same with the adapted singular
  rules;
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
- `screened_collocation_robin`, `screened_collocation_robin_fma`: the
  screened kernel in the collocation problem with Dirichlet and Robin parts,
  with the matrices and with the multipole method.
- `screened_collocation_jump_p2_fma`: its jump coefficient, the row sums of
  the double layer of the Laplace kernel, at `p = 2`, `kappa = 0.5`, 6146
  unknowns, where the geometric coefficient gave three times the flux error.
- `helmholtz_collocation_robin`, `helmholtz_galerkin_robin`,
  `helmholtz_collocation_robin_fma`, `helmholtz_galerkin_robin_fma`: the
  Helmholtz kernel, the plane wave with Dirichlet and impedance parts
  against the exact solution, collocation and mixed Galerkin problem, with
  dense matrices and with the multipole method of the Helmholtz kernel.
- `screened_mixed_robin`, `screened_mixed_robin_fma`: the screened kernel in
  the mixed Galerkin problem, with Dirichlet and Robin parts, dense and
  multipole, Calderon preconditioner.
- `slots_mixed_fma`, `slots_dirichlet_dg`, `slots_dirichlet_fma`: two slots,
  each exact on its manifold and wrong on the other, in the Galerkin data.
- `components_mixed_dense`, `components_dirichlet_dg`: two components, the
  second with the operators of the first.
- `slots_collocation`, `complex_collocation_errors`: the errors against the
  exact solution with two slots in the collocation problem, and for the two
  components of a complex problem (`Errors = exact solution`).
- `complex_galerkin_dense`, `complex_galerkin_fma`,
  `complex_galerkin_dirichlet`: the complex sample in the Galerkin
  formulation, real and imaginary part in one mixed system coupled on the
  Robin part, and with Dirichlet conditions everywhere.

`ctest` runs sixteen of these cases on 2 processes (`tests/solver_*.cc`,
with the helpers in `tests/solver_tests.h`), so that the CI covers the
solver: Dirichlet with the Laplace and the screened kernel, the mixed
problem with Robin, components and slots and with the screened kernel, the
complex problem in both formulations, the collocation multipole method with
both kernels, the Helmholtz kernel, and the NACA wing on the exact geometry.
They print the quantities as the solver does, and numdiff compares them. In
Debug, eight of them run on the sphere without its refinement cycle, since
the Debug build is thirty to fifty times slower. The dense mixed problems
with the screened and the Helmholtz kernel, the two slowest in Debug, run
in Release only: in Debug the screened mixed problem with the multipole
method covers the integrator of the screened kernel, which the Helmholtz
kernel shares, and the complex Galerkin problem the coupling of the real
and the imaginary part.

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

### Gradient recovery

| Variable | Effect |
|---|---|
| `PIBEM_HW_DUFFY_ORDER`, `PIBEM_HW_NEAR_RATIO`, `PIBEM_HW_PAIR_RATIO`, `PIBEM_HW_PAIR_DEPTH`, `PIBEM_HW_PAIR_ORDER`, `PIBEM_HW_POINTS` | Set the rules of `compute_gradients_hypersingular_weak()`. Defaults: Duffy order 12, near ratio 0.5, pair ratio, depth and order from the near-singular parameters (depth at least 10), and degree + 4 points. |

The older diagnostic switches (`PIBEM_COMPARE_RHS`, `PIBEM_ENERGY_SELFTEST`,
`PIBEM_WEAK_GRADIENT_CHECK`, `PIBEM_HYPERSINGULAR_WEAK_CHECK`,
`PIBEM_HW_DEBUG`, `PIBEM_HYP_GRADIENT_CHECK`) were removed in the cleanup of
2026-10-27, with the gradient recoveries `weak_joint`, `weak_discrete` and
`hypersingular_weak_discrete` and the hypersingular trace correction. Their
parameters are still read, so that older files run, and a file that selects
one of them stops with a message.
