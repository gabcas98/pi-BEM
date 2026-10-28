# Multiple potentials, complex problems and Robin conditions

## Build and run with deal.II 9.8.0

From the repository root, build both configurations in the supplied container:

```sh
docker run --rm -v "$PWD:/workspace" -w /workspace \
  dealii/dealii:v9.8.0-resolute-r1-arm64 bash -lc '
    cmake -S . -B build-docker/9.8.0 -GNinja \
      -DCMAKE_BUILD_TYPE=DebugRelease -DCMAKE_CXX_FLAGS=-Werror &&
    cmake --build build-docker/9.8.0 -j 4 &&
    cd build-docker/9.8.0 && ctest --output-on-failure
  '
```

Executables are `bem_fma_2d` and `bem_fma_3d` in the build directory;
Debug executables have the `.g` suffix. The first positional argument is the
number of threads, not a parameter filename. Each program reads
`parameters_bem_2.prm` or `parameters_bem_3.prm` in its working directory and
writes `used_parameters_bem_2.prm` or `used_parameters_bem_3.prm`.

The sphere examples use paths relative to `samples/`. Keep each run's output
separate, and copy both the parameter file and its grid directory:

```sh
mkdir -p build-docker/9.8.0/example-complex
cp -R samples/grids build-docker/9.8.0/example-complex/
cp samples/ref_params_sphere_complex_DR_direct.prm \
  build-docker/9.8.0/example-complex/parameters_bem_3.prm
docker run --rm -v "$PWD:/workspace" \
  -w /workspace/build-docker/9.8.0/example-complex \
  dealii/dealii:v9.8.0-resolute-r1-arm64 bash -lc '
    mpirun -np 2 ../bem_fma_3d 2
  '
```

The default container user can run MPI. If running as root, also pass
`-e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1` to Docker.
The samples cover real/complex potentials, DN (Dirichlet/Neumann), DR
(Dirichlet/Robin), NR (Neumann/Robin), and `direct`/`fma` methods.
`Direct` assembles integral matrices and uses an iterative solver; it does not
mean a dense direct factorization. FMA is supported in 3D only.

The results of this fork on the samples differ from those of upstream pi-BEM
in two ways. Real problems with `Direct` assemble the cells near a
collocation point with the quasi-singular rules (`Quasi singular distance
ratio = 2.0`), which changes alpha by 2e-6; with `Quasi singular distance
ratio = 0` the assembly is that of upstream. And the type of a degree of
freedom shared by cells of two types no longer depends on the partition:
upstream classified the dofs of a process only from the cells it owns, so
that on 4 and 8 processes the same problem had different unknowns at the
edges between the parts. A shared dof is now a Dirichlet dof if one of its
cells is Dirichlet, else a Neumann dof if one is Neumann, else a Robin dof.

Two errors of upstream are corrected. With `FMA`, the Robin datum `g/b` of a
real problem entered only the near field of the right-hand side: on the
sphere with `phi + dphi_dn = 2(x+y+z)` the potential was wrong by 0.18
(0.0008 after the correction, as with `Direct`). In 2D the Robin functions
had two entries instead of three.

The projected nodal normals are not normalised, as in upstream: the Neumann
datum is then consistent with the discrete surface. On the DN sample the
potential converges at order 3 with them and at order 2 with normalised
normals (Direct, three refinements: 3.0e-5 against 5.2e-4). The normals of
`Nodal normals = interpolation`, sums over the cells at a node, are
normalised.

## Hosted CI environment

Local validation uses `dealii/dealii:v9.8.0-resolute-r1-arm64`, built from
the frozen recipes in [docker-files PR 90](https://github.com/dealii/docker-files/pull/90).
Until that image is published, hosted CI builds `docker/Dockerfile.ci` with the
exact deal.II 9.8.0 backports package on Ubuntu 26.04. This supports hosted AMD64
runners without depending on a tag that currently exists only locally.
The CI environment uses mold to link the large Debug libraries efficiently.
The formatting job retains clang-format 16.0.6 from its existing image.

To reproduce the hosted environment locally, build it and substitute
`pi-bem-ci:9.8.0` for the image in the commands above:

```sh
docker build -f docker/Dockerfile.ci -t pi-bem-ci:9.8.0 .
```

The deal.II 9.8 update replaces the removed three-argument Trilinos vector
`reinit(source,false,true)` with assignment into the already initialized
ghost layout. The assignment imports both locally owned and ghost values;
it must participate collectively on the vector communicator.

## Components and boundary-function slots

`Potential components` counts real-valued storage components. Each complex
potential uses consecutive components: its real part at `i`, imaginary part at
`i+1`. `Complex components` lists the real indices, with no overlapping pairs:

```text
subsection Driver<3>
  set Potential components = 3
  set Complex components = 0
  set Set Global Refinement = true
end
```

This solves one complex potential in components 0 and 1, followed by a real
potential in component 2. Parameter declarations provide up to 14 storage
components. Every component has two boundary-function slots, numbered 0 and 1.

Under `BoundaryConditions<dim>`, old function subsection names must be replaced
by `Potential <component> <slot> <dim>d`, `Wind function <component> <slot> <dim>d`
and `Robin coefficients <component> <slot> <dim>d`. Potential expressions are
scalar; wind expressions have `dim` entries and supply the normal derivative
through their dot product with the outward normal. Functions used by the solve
are evaluated at time zero. Output uses the configured stem for component 0
and appends `_i` for later components.

The earlier names `Potential <dim>d` and `Wind function <dim>d` are still
read: when a file sets them, they take the place of slot 0 of component 0.
The Galerkin formulation takes on every cell the functions of the slot of
its manifold (`CellwiseFunction`, `include/cellwise_function.h`): the
Dirichlet datum and the Neumann datum of the mixed problem, the load vector
of the Dirichlet problem, dense or with the multipole method, and the errors
of the mixed solver. A point is evaluated with the function of the cell it
lies on, also in the regularised integrals, where the datum at the test point
follows the test cell; a node of the Dirichlet part takes the function of
its Dirichlet cells. With two slots that are each exact on their own
manifold and wrong by 7 on the other, the sphere of the samples gives the
numbers of a single function to the last digit (mixed and Dirichlet
problems, continuous and discontinuous elements, dense and multipole).

A node takes its datum from the slot of a cell of its kind: a Dirichlet
node from a Dirichlet cell, a Neumann node from a Neumann cell, a Robin node
from a Robin cell (`BoundaryConditions::dof_slots()`). Before, it was the
slot of the last of its cells, of whatever kind.

`compute_errors()` gives either the errors against the exact solution or the
errors of the boundary conditions by material of upstream pi-BEM
(`compute_errors_by_material()`), as `BoundaryConditions<dim>/Errors` says:

- `exact solution`: the potential and the wind function are taken as the
  solution on the whole surface, on every cell those of its slot and at
  every node those of the slot above. A complex problem compares each
  component with its own functions, and Robin parts are measured like the
  rest of the surface.
- `boundary conditions`: on each material the residual of its condition,
  which needs only the boundary data (on Dirichlet parts it is zero by
  construction; the Robin residual is the informative number).
- `auto` (default): the exact solution for a real component without a Robin
  part, with any slots, and the boundary conditions otherwise.

The Galerkin multipole method of the Dirichlet problem needs `Mapping Type =
Q`: it evaluates the geometry of the near field directly. The mixed problem
also runs with `Mapping Type = FE`.

With several real components the Galerkin formulation sets the operators up
for the first and reuses them for the others on the same mesh: V, K and W of
the mixed problem (dense or through the multipole method), the single layer
of the Dirichlet problem with dense matrices, whose load vector each
component assembles alone. The operators are freed after the last
component, and by `BEMProblem::reinit()` on a new mesh. The Galerkin
multipole method of the Dirichlet problem is set up again for every
component, since its near field integrates the operator and the datum
together. On the sample sphere with x+y+z and x^2-y^2, each component gives
the numbers of a run of its own (the load vector of the Dirichlet problem to
rounding: it is assembled alone instead of with the single layer), and the
second mixed solve takes 0.2 s instead of 8 s with dense matrices.

The mixed Galerkin problem takes Robin parts too. On their cells the
potential is unknown, as on the Neumann part, and the flux is
`g/b - (a/b) u` with `a/b` and `g/b` at the nodes: the flux `-(a/b) u` of
the unknowns joins that of the Dirichlet part in `V` and `K^T`, and its
identity term `1/2 M` goes to the potential rows. A node of Neumann and Robin
cells is a Neumann node. With Dirichlet on manifold 1 and
`u + du/dn = 2(x+y+z)` on manifold 2 of the sample sphere, p = 1, the
potential converges at order 2 (1.8e-3, 4.6e-4, 1.2e-4 at one to three
refinements; dense matrices and FMM agree to 1e-3 of the error), and the
flux on the Robin part at order 1 as that of the Dirichlet part. A nearly
Dirichlet condition (`a/b = 100`) and Robin conditions on the whole surface
(no Dirichlet part) converge in 82 and 21 GMRES steps. The mixed solver
prints the flux error of the Robin part beside the other two.

Complex problems run in the Galerkin formulation too, with the functions of
the slots of both components. The mixed problem solves the real and the
imaginary part in one GMRES on vectors of both parts, the real one first:
`V`, `K` and `W` act on each part, and the parts are coupled only on the
Robin part, where the flux `g/b - (a/b) u` takes the complex product of
`a/b` (`robin_matrix_diagonal` and `robin_matrix_diagonal_imag`) and `u`.
The diagonal preconditioner divides at each node by the complex diagonal
`w + 1/2 (a/b) m`; the Calderon preconditioner acts on each part alone.
Errors and indicators are printed for each part (`, real part:` and
`, imaginary part:`), and the indicators of adaptive refinement add the two.
With Dirichlet conditions on the whole surface the parts are not coupled:
they are solved as two real problems, the imaginary one with the single
layer of the real one (with the multipole method the Dirichlet problem is
set up again, as for another component). On the sample sphere with
`phi = (x+y+z+2) + i (x^2-y^2)` and `a/b = 1 + 0.5 i` on the Robin part
(`RUN_REGRESSION/base/sphere_complex_galerkin.prm`), p = 1, both parts of the
potential converge at order 2 (real 5.9e-4, 1.5e-4, 3.8e-5; imaginary
1.0e-2, 2.6e-3, 6.4e-4 at one to three refinements) and the fluxes at order
1. The upstream sample (`sphere_complex_robin.prm`,
`a/b = (1+i)/(2(x+y+z+2))`) behaves the same, with dense matrices and FMM
agreeing to 2e-3 of the error. With a real `a/b` the complex solve gives the
numbers of two real components to the last printed digit, and with Dirichlet
conditions everywhere to 16 digits. The coupling costs about 30% more GMRES
steps with the diagonal preconditioner (76, 108, 150 against 60, 86, 116 for
the real problem); the Calderon preconditioner takes 26 steps instead of 108
at two refinements, and with the FMM and relaxed products 10 s instead of
57 s. Although it leaves the coupling out, the Calderon preconditioner needs
only three to four steps more than for the same problem with a real `a/b`
(29, 31, 34 against 25, 28, 30 at one to three refinements, with the FMM),
whatever the mesh.

## Robin conditions and geometry IDs

The 3D Robin function has three entries `a; b; g`, expressing
`a*phi + b*dphi_dn = g`. The solver stores `a/b` and `g/b`, so `b` must be
nonzero at every Robin node. For complex problems, the active component
contains the real parts of all three coefficients and the following component
contains their imaginary parts; the division is complex division.

For example, the condition `(2+3i)*phi + dphi_dn = 4+5i` in slot 0 is:

```text
subsection BoundaryConditions<3>
  subsection Robin coefficients 0 0 3d
    set Function expression = 2; 1; 4
  end
  subsection Robin coefficients 1 0 3d
    set Function expression = 3; 0; 5
  end
end
```

Boundary-type lists select mesh **manifold IDs**. Their names retain the word
"boundary" for compatibility. The three lists must be disjoint, except for
110, the id with which parameter files pad the lists and which no cell has.
The Robin list is empty by default. The slot map is a flat comma-separated
list of `manifold, slot` pairs:

```text
subsection ComputationalDomain<3>
  subsection Boundary Conditions ID Numbers
    set Dirichlet boundary ids = 1
    set Neumann boundary ids =
    set Robin boundary ids = 2
    set Manifold to boundary slot map = 1,0, 2,1
  end
end
```

This uses slot 0 on manifold 1 and slot 1 on manifold 2, for every component.
Supply the corresponding potential/wind/Robin functions for each selected slot.
The Robin function has the three entries `a; b; g` in 2D as well
(`RUN_REGRESSION/base/circle_robin_2d.prm`).

## C++ API migration

All distributed operations must be invoked by every rank in the object's
communicator, with compatible vector ownership. The geometric dimension is
`dim`, and the boundary mesh dimension is `dim-1`.

| Interface | Contract and migration |
| --- | --- |
| `Driver(n_components)`, `BEMProblem(domain,comm,n_components)`, `BoundaryConditions(domain,bem,comm,n_components)` | Counts refer to real storage components; defaults preserve a single real potential. Keep referenced objects alive. |
| `n_phi_components()`, `set_n_phi_components(n)` | Query or grow component storage; shrinking is unsupported. Grow before initialization/parameter parsing. |
| `get_current_phi_component()`, `set_current_phi_component(i)` | Select a zero-based storage component. Selecting through `BoundaryConditions` also selects the bound BEM component. A complex solve requires `i+1<n`. |
| `get_phi([i])`, `get_dphi_dn([i])` | Mutable vector references for the current or explicit component replace direct access to single solution fields. Storage growth can invalidate references. |
| `get_wind(i,slot)`, `get_potential(i,slot)`, `get_robin_coeffs(i,slot)` | Parsed-function access requires initialized parameters, valid component and slot. Const public overloads expose the functions; mutable overloads are protected. |
| `get_phi_components([indices])` | Returns an owned block-vector copy collectively. Explicit indices retain their component block numbers; the no-argument overload selects all. |
| `solve_problem(reset_matrix)`, `solve_complex_problem(reset_matrix)` | Prepare data and solve current component/pair. Default `true` assembles the integral operators. Use `false` only while geometry, mapping, quadrature and discretization remain unchanged. |
| `BEMProblem::solve(...)` | Real overload takes `phi, dphi_dn, rhs, dirichlet_function, dirichlet_gradient, reset_matrix`; complex overload takes `phi, phi_imag, dphi_dn, dphi_dn_imag, rhs, rhs_imag, dirichlet_functions, dirichlet_gradients, reset_matrix`, the functions of the real and of the imaginary part (needed by the Galerkin formulation). Boundary vectors and Robin data must already be prepared. |
| `prepare_robin_datastructs(...)` | Real overload fills diagonal/RHS; complex overload fills real and imaginary diagonal/RHS from consecutive component functions. |
| `vmult(...)`, `compute_rhs(...)`, `solve_system(...)` | Additional overloads operate on separate real/imaginary vectors. They require previously initialized operators and current boundary data. |
| `assemble_system()` | The direct collocation assembly of real and complex problems, with the rules of the fork (singular `QSplit` of `QDuffy`, quasi-singular rules, adaptive orders, Laplace or screened kernel); the rows of each process are integrated in parallel with `MultithreadInfo::n_threads()` threads. It replaces upstream's `assemble_system_tbb()` and `assemble_system_singlethread()`, which the complex solve used with the rules of upstream. |
| `assemble_preconditioner_complex()`, `get_preconditioner(complex)` | Set up and select configured ILU/AMG storage. Selection returns a reference owned by the BEM problem. |
| `get_vector_gradients_solution([i])`, `get_vector_surface_gradients_solution([i])` | Mutable component-aware gradient access replaces the old single gradient fields. |
| `get_system_matrix()`, `get_system_matrix_complex()` | Collective dense diagnostic copies, size N or 2N, ordered real then imaginary. Require direct matrices, use quadratic memory per rank and omit constrained-row replacement. |
| `compute_errors(complex,current_is_real)` | Replace the old no-argument call: `(false,true)` for real, `(true,true)` for the real part of a pair, `(true,false)` for its imaginary part. |
| `output_results(stem)` | Writes the selected component after errors/gradients have been computed. |

For an initialized boundary object `bc`, component selection is explicit:

```cpp
bc.set_current_phi_component(0);
bc.solve_complex_problem(true);
bc.compute_errors(true, true);
bc.output_results("potential_real");
bc.set_current_phi_component(1);
bc.compute_errors(true, false);
bc.output_results("potential_imag");
```

The old `bem_problem_access.h` adapter and `time_integrator.h` are removed.
Use the component-aware public accessors; no time-integration replacement is
provided. `ComputationalDomain` now exposes `load_cad_objects()` (returns the
largest CAD tolerance), `refine_and_resize_by_aspect_ratio()`,
`refine_and_resize_by_cad_projections(max_tol)` and
`refine_and_resize_wrapup()` (partition and mesh output). Normal applications
should call `refine_and_resize(level)`, which runs these stages in order.
The option for attaching flat manifolds to IDs without CAD data preserves
import of grids whose flags also distinguish boundary conditions.

## Complex operators and the FMM

`ConstrainedComplexOperator<VEC,MATRIX>` accepts one flat vector ordered
`[real,imaginary]`. `BlockConstrainedComplexOperator<VEC,MATRIX>` accepts a
two-block vector in the same order. Their constructors take a matrix, real and
imaginary constraints, a constraint-relevant IndexSet and a communicator.
Both constraint objects must describe the same constrained rows with unshifted
indices. The matrix provides `this_cpu_set` and a four-vector `vmult`.
`distribute_rhs()` installs the corresponding inhomogeneities.
`BlockPreconditioner<P>` applies a referenced preconditioner independently to
each block through `vmult`/`Tvmult`. Referenced matrices, constraints and
preconditioners must outlive these adapters. The flat operator keeps mutable
scratch vectors and must not be used concurrently from multiple threads.

With `Solution method = FMA`, `CollocationFMMOperator` applies the
double-layer and single-layer matrices; `BEMProblem::vmult()` calls it once
for the real part and once for the imaginary part. The preconditioner of each
block is the ILU (or AMG) of the real system on the near band of the operator,
the Robin columns included. The FMA of upstream (`BEMFMA` with its `_tbb` and
`_omp` backends, `MultipoleExpansion`, `LocalExpansion`) has been removed;
`BEMFMA` now only holds the parameters.

## Geometry and error-integration helpers

`QuasiSingularKernelIntegral<3>` searches a mapped quadrilateral for the closest
point to an external point. Its constructor initializes reference coordinates
`eta` and physical `min_distance`; the search considers the interior, edges and
vertices. The referenced iterator object, mapping and finite element must remain
alive. The 2D constructor does not initialize a projection or distance. This
helper currently provides closest-point geometry, not a public quadrature API.

`integrate_difference_based_on_material_id()` and its hp implementation
`do_integrate_difference_based_on_material_id()` select non-owning exact-function
pointers by **material ID**. They return cellwise differences and put zero on
unmapped/non-owned cells. Function component counts must match the finite
element. Compute the global norm separately with
`VectorTools::compute_global_error()`. These helpers depend on deal.II internal
integration interfaces and require checking when updating deal.II.
