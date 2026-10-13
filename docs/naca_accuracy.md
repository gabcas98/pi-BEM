# Local quadrature and planar mesh quality on the NACA wing

Both changes are opt-in. Ordinary parameter files keep their current behavior.
The settings are independent of the manufactured boundary data and work with
both the dense Galerkin assembly and the Galerkin FMM near-field corrections.

## Quadrature on selected materials

Under `BEMProblem<3>/Quadrature rules`:

```text
set Galerkin singular quadrature order = 7
set Galerkin singular quadrature order for folded pairs = 10
set Galerkin singular quadrature material ids = 1,2
set Galerkin singular quadrature order for selected materials = 12
```

Materials 1 and 2 are the planar tips of this NACA input. Use the appropriate
ids for another mesh. Any touching pair involving a selected material uses at
least the selected order; this includes coincident, edge, vertex and
nonconforming contacts. Pairs outside the selection keep their usual order.
The folded-pair setting remains active and can further raise the order.
An empty material list or order zero disables this promotion.
Order 12 is the tested economical setting below; order 14 provides a useful
higher-order comparison. Neither has yet been validated by a new global L3 solve.

This is particularly useful at orthogonal surface–tip contacts. Their normals
are not nearly opposite, so raising only the folded-pair order misses them.
Regular subpatches created by singular-pair balancing retain their regular
and near-singular quadrature settings. This option therefore does not certify
all integrals to a tolerance; check order convergence on critical pairs.
The higher order can consume substantial memory in quadrature caches.

## Planar mesh smoothing

Under `ComputationalDomain<3>`:

```text
set Planar mesh smoothing material ids = 1,2
set Planar mesh smoothing iterations = 20
```

After refinement, interior vertices of these planar material patches are
optimized in their plane. The objective is the sum of cell qualities, where
each cell quality is its minimum signed corner mean ratio
`2 det(a,b) / (|a|^2 + |b|^2)`. This accounts for both skew and elongation.
Each accepted move must preserve or improve the minimum corner quality of
**every** incident cell. It cannot reverse a cell.

The algorithm fixes boundary vertices, material interfaces and all vertices
of nonconforming faces, including the coarse endpoints. It neither changes
the topology nor adds cells. It validates planarity and convexity before
moving any vertex, and rejects curved patches. A material id must describe
one coplanar patch; curved surfaces need a different optimization method.
An empty list or zero iterations disables smoothing.

The reported global minimum may stay fixed because the actual sharp wing
corner cannot be improved by moving interior vertices. Improving this metric
does not necessarily increase every interior angle: aspect and skew both
enter the corner mean ratio. Stronger improvements around hanging contacts
would require changing mesh topology; this smoother deliberately fixes them.

## Verification

The release tests `galerkin_material_quadrature`,
`galerkin_panel_pair_integrator`, and `planar_mesh_quality` passed.
They check selective promotion at an orthogonal contact, unchanged rules
outside the selection, the existing quadrature invariants, quality improvement,
fixed boundaries and hanging contacts, and rejection of curved patches.

The NACA comparison used FE_Q(2), exact parametric CAD geometry, level 1,
720 cells, 4 MPI processes, one thread each, identical data and FMM settings.
No new global solve at level 2 or 3 was run. Detailed run files are in
`RUN_NACA/accuracy_trial`; these run artifacts are ignored by git.

At the trailing-edge tip region (`x > 0.9`, `|y| > 2.4`, based on cell centers),
maximum absolute flux errors sampled on the Dirichlet cells were:

| Case | Sampled tip flux maximum | Wall time [s] |
|---|---:|---:|
| Baseline, order 7 / folded 10 | 0.00200039 | 22.61 |
| Selected materials at order 14 | 0.000574149 | 95.71 |
| Smoothing only | 0.0234537 | 30.20 |
| Smoothing and selected order 14 | 0.000536686 | 87.58 |
| Smoothing and selected order 12 | 0.000543734 | 47.95 |

The final smoother raised mean tip-cell quality from 0.536392 to 0.545125,
with 51 accepted vertex moves. The minimum stayed 0.276016 at the physical
sharp corner. Smoothing alone made the flux error worse at the old quadrature:
it changes the integration geometry and must be accompanied by accurate
quadrature. With order 14, smoothing reduced the tip maximum a further 6.5%.
Order 12 with smoothing stays within 1.3% of the order-14 result, takes about
half the time, and lowers the sampled tip maximum by a factor of 3.68 versus
the baseline. Peak RSS of the largest process fell from about 3.8 GB at order
14 to 2.3 GB at order 12 (timing includes other concurrent local checks).

The global flux maximum stayed 0.440719 in these coarse runs because it is
located near the source, away from the tips. The relative flux L2 error stayed
approximately 0.0206336. These results establish a local improvement, not
removal of a global L-infinity plateau. The dump maxima are Gauss-point
samples, not certified continuous suprema.

A one-pair diagnostic on the level-3 geometry compared the critical lower
surface panel 6944 with tip panel 208. Relative maximum-entry differences of
K against an order-18 comparison were about 6.8e-4 at the baseline order and
1.3e-6 with selected order 14, an approximately 500-fold reduction. The order
18 value is a convergence comparison rather than an analytic exact integral.
That contact remained fixed by the smoothing constraints.

To reproduce the small solves, from the repository root:

```bash
python3 RUN_NACA/accuracy_trial/prepare.py
RUN_NACA/accuracy_trial/run_small.sh
```

The script runs only level 1 and imposes a 120-second limit per solve.
Keep regional maxima as well as the global maximum when comparing mesh or
quadrature variants.

## Global accuracy follow-up

The first tip changes were active in the user's level-2 run, but the reported
maxima were near the source and the leading edge, outside their target region.
The all-surface nodal flux maximum also includes the projected Neumann datum,
which is prescribed data on material 3 rather than a solved flux unknown.

A new tested configuration is now in `RUN_NACA/parameters_bem_3.prm` and
`RUN_NACA/naca_global_accuracy_p2.prm`. The previous input is preserved as
`RUN_NACA/parameters_bem_3.before_global_accuracy.prm`. The scalar and vector
spaces remain FE_Q(2), with the same manufactured data and FMM settings.
The tested profile uses Mapping Q degree 1 underneath the exact parametric
geometry, rather than the previous degree 2. It combines:

- Base global level 1, instead of 2.
- Anisotropic curvature refinement at the leading edge, 6 cells per circle,
  up to 4 curvature passes.
- Grading of materials 3 and 4 around (0.4,0.3,0), ratio 0.4, exponent 1,
  minimum target diameter 0.08, up to 2 local passes.
- L2 projection of the Dirichlet datum.
- The accurate tip quadrature and planar smoothing described above.

With eight processes, one thread each, the completed solve took about 100 s:

| Indicator | User level-2 run | Tested locally refined profile |
|---|---:|---:|
| Cells | 2880 | 1680 |
| Scalar DoFs | 12291 | 7403 |
| Unknown flux maximum sampled on D | 0.115288 | 0.0277242 |
| All-surface nodal flux maximum | 0.311344 | 0.0362543 |
| Nodal gradient maximum component | 0.409872 | 0.341784 |
| All-surface nodal potential maximum | 0.0139513 | 0.0117141 |

The flux maximum improves substantially with fewer cells. The gradient
maximum improves only modestly and is still dominated by the tangential
part (sampled tangential maximum 0.344098 versus normal maximum 0.0326024).
This profile does not establish removal of a tangential-gradient plateau.
It is one controlled comparison, not a convergence study of the new grading.

Refining near the source alone gave 1374 cells and worsened the maximum at
the leading edge. Resolving that curvature and projecting the Dirichlet datum
were both included in the successful profile. A further experiment reducing
aspect ratios after curvature refinement produced 2724 cells, but its solve
hit the 120-second cap during assembly; it is not the selected configuration.

The updated input was separately parsed and its mesh generated in
`global_accuracy_trial/final_input_check`: all effective parameters matched
the completed solve except `Mesh only`, and the mesh had 1680 cells. No solve
was run in the user's original folder, so existing result files are retained.
Detailed trials, dumps and a CSV comparison are in
`RUN_NACA/global_accuracy_trial` (ignored run artifacts).

### Diagnostic corrections

The boundary summary previously hardcoded `FE_DGQ(p)` even when the actual
space was FE_Q. It now prints `fe->get_name()`. In a mixed problem it identifies
the scope as all-surface postprocessing, including prescribed Neumann data.

The Dirichlet-only residual and algebraic energy distance are not applicable
to the mixed block operator. Previously the empty Dirichlet operator/RHS
produced misleading zeros. Those values are no longer evaluated or printed
for a mixed solve; the actual flux/potential row residuals already printed by
the mixed solver remain available. Full-space projection comparisons are
labelled as all-surface comparisons, rather than Galerkin optimality ratios
for the mixed unknown flux. The gradient report also gives the point and
component of its largest nodal component error.

A six-cell pure Dirichlet smoke run retained its valid residual diagnostics
(relative exact-data residual 0.00114677 and nonzero algebraic energy distance),
while the completed mixed run printed the corrected FE_Q summary and omitted
the invalid zeros.

### Local refinement correction

Local grading left `limit_level_difference_at_vertices` active after its
passes. The later CAD-seam alignment could then propagate anisotropic
refinement levels globally: a mesh-only preview grew beyond 47000 cells and
was stopped by its timeout before any solve. Graded and exponential passes
now restore the previous smoothing flags. They only enable the isotropic
level limiter when the existing refinement history is isotropic: anisotropic
levels count cuts in different directions and are not a physical-size metric.

The new `local_refinement_anisotropic` regression checks that a material-local
refinement preserves unrelated anisotropic materials and restores smoothing
flags, including for exponential grading. It and the three existing quadrature
and planar-quality release tests passed.

## Physical hanging-node constraints on the CAD

The next investigation used the user's later level-2 output, with 4764 cells.
Its largest exported gradient component error was about 0.296706 at
(0.111479,0,0.0486555), on a curved refinement transition. This suggested a
constraint issue rather than another tip-quadrature issue.

CAD refinement can place the fine midpoint at a different physical point from
the midpoint of the coarse mapping's parameter interval. Standard topological
hanging-node weights then equate values at different points. A geometry-only
diagnostic found a single-master separation of 0.00138989 on this mesh. With
representative nodal data u=1/r and the standard constraints, the largest
mismatch between a constrained fine-node value and the coarse trace evaluated
at that same physical point was 0.00389597. This is a diagnostic field, not a
measurement of the stored BEM solution's trace jump.

`ExactGeometry::make_mapping_hanging_node_constraints` now pulls each fine
support point back onto the coarse mapped edge and evaluates the coarse FE
trace there. It uses geometry and basis functions only. It preserves the
standard constraints for points that do not lie on the coarse edge, reporting
these separately rather than forcing a projection across a geometry gap.
The correction is active when `Exact geometry` is enabled, in scalar
constraints, vector constraints, assembly, and the projection diagnostic.
The initial vector constraints used to construct the mapping are rebuilt
after the final mapping is available.

On the user's 4764-cell mesh, the diagnostic corrected 444 scalar constraint
lines, with no off-edge points. The largest representative nodal trace
mismatch fell to 8.88e-16. This verifies the hanging-node values; it does not
establish continuity at every point of differently parameterized edges.
Only the geometry diagnostic was run at this size, not a full BEM solve.

A full solve used the previously completed 1680-cell configuration, unchanged
parameters, FE_Q(2), 7403 scalar DoFs, and eight MPI processes:

| Indicator | Standard constraints | Physical constraints |
|---|---:|---:|
| Unknown flux maximum sampled on D | 0.0277242 | 0.0249198 |
| Potential maximum sampled on N | 0.010185 | 0.00166828 |
| All-surface nodal potential maximum | 0.0117141 | 0.00141267 |
| All-surface nodal flux maximum | 0.0362543 | 0.0362543 |
| Nodal gradient maximum component | 0.341784 | 0.178786 |
| Sampled tangential-gradient maximum | 0.344098 | 0.166326 |
| Elapsed seconds | 100.11 | 102.45 |

The corrected solve's largest nodal gradient error moved near the source,
to (0.391608,0.390625,0.0583022), component y. The unchanged nodal flux maximum
and remaining gradient error show that this does not remove every accuracy
limit. Local refinement near the source or higher approximation degree are
next candidates; no new full-solve convergence claim is made for them here.

The regression `mapping_hanging_constraints` checks scalar and three-component
traces at degrees 1, 2, and 3, preservation of constants, and unchanged weights
on an affine mesh. The release executable builds successfully; this regression
and the four existing quadrature/refinement/mesh-quality tests pass. The
user's current parameter file, including level 2, was left unchanged.
Diagnostic logs, the small solve, and comparison files are under
`RUN_NACA/gradient_trial` (ignored run artifacts).

## Local refinement after the physical-constraint correction

A further small comparison kept the corrected constraints, FE_Q(2), all
quadrature settings, FMM settings, source data, and eight single-threaded MPI
processes. It increased the radial grading limit from two to three passes and
changed the minimum target diameter from 0.08 to 0.075. The focus remained
(0.4,0.3,0), ratio 0.4, exponent 1, on materials 3 and 4. This generated 2040
cells and 9003 scalar DoFs, versus 1680 cells and 7403 DoFs. The smallest
graded-cell diameter fell from 0.0783882 to 0.0410868.

The full solve finished within its 180-second cap:

| Indicator | Corrected baseline | More local refinement |
|---|---:|---:|
| Nodal gradient maximum component | 0.178786 | 0.149360 |
| All-surface gradient L2 | 0.0313873 | 0.0251017 |
| All-surface nodal potential maximum | 0.00141267 | 0.00132127 |
| Potential maximum sampled on N | 0.00166828 | 0.00118199 |
| Unknown flux maximum sampled on D | 0.0249198 | 0.0508321 |
| Unknown flux L2 on D | 0.00349910 | 0.00420905 |
| All-surface nodal flux maximum | 0.0362543 | 0.0648571 |
| All-surface flux L2 | 0.00610541 | 0.00531613 |
| Elapsed seconds | 102.45 | 109.39 |

The effect is mixed: the gradient improves, while the solved flux on D gets
worse in both sampled maximum and L2. The all-surface flux L2 includes
prescribed Neumann data on N and therefore masks this deterioration of the
unknown flux. The actual mixed row residuals remain small (flux 1.02e-10,
potential 3.50e-12), so the linear solver has converged.

The maximum nodal gradient component moves from near the source to the leading
edge at (0.0206491,0.3125,-0.0239488), component x. The new sampled flux maximum
is on the lower surface at (0.391165,0.467431,-0.0583177). Its cell has a hanging
node, as did the baseline's worst flux cell; this association alone does not
prove that the constraint or quadrature caused the deterioration. The
all-surface best FE_Q flux projection L2 improves from 0.00542718 to 0.00378842,
so poor approximation capacity alone does not explain the worse unknown flux.

Mesh-only previews with third-pass floors 0.06, 0.07, 0.075 and 0.08 produced
the same 2040-cell mesh. Floors 0.085, 0.09 and 0.10 produced the original
1680-cell mesh: these parameter changes cross discrete cell-splitting
thresholds rather than continuously changing cell size. Identical mesh
variants were not solved again.

This is a single local refinement step, not a convergence study; nodal and
quadrature maxima sample different points on the two meshes. The profile is
retained for experiments, not adopted as a general accuracy improvement.
The user's parameter file remains unchanged. Logs, parameters, and a
reproducible comparison script are in `RUN_NACA/local_refinement_trial`.

## Non-conforming contacts on curved edges (2026-10-10)

The local-refinement deterioration above was caused by the integrator, not by
the constraints. `GalerkinPanelPairIntegrator::nonconforming_contact` tested
whether the ends of the shorter edge lay on the straight chord of the longer
one, within `1e-10` of its length. On the exact geometry a hanging vertex lies
on the curved edge of the coarse cell, at a distance of about `L^2 kappa/8`
from its chord, so the contact was missed and the half-edge pair received the
vertex rule or the near-singular subdivision. In the 2040-cell mesh above, 209
of the 267 hanging vertices on the wing surfaces were missed (180 on chordwise
edges, 29 where the CAD projection moves the vertex off the straight spanwise
line), and the ten largest flux errors were all in cells touching a missed
contact.

The test now pulls the ends of the shorter edge back onto the mapped edge of
the containing cell (Newton on `map_to_real_cell`, tolerance `1e-9` of the
edge length, after a loose filter on the chord); the reference coordinates
found there bound the contact strips. Flat meshes, whose coarse edges are
straight and whose hanging vertices lie on the CAD off them, keep their
behavior: those pairs do not touch. The regression suite is unchanged and the
cost is not measurable on it.

| Indicator | 1680 cells, before | 2040 cells, before | 1680 cells, after | 2040 cells, after |
|---|---:|---:|---:|---:|
| Unknown flux L2 on D | 0.0034991 | 0.0042090 | 0.00066186 | 0.00062523 |
| Unknown flux sampled max on D | 0.024920 | 0.050832 | 0.014392 | 0.014391 |
| Potential unknown L2 on N | 0.00062139 | 0.00055229 | 0.00021485 | 0.00019091 |

More local refinement now improves the flux. The remaining maximum lies at
the nose, where it equals the local best approximation, and in the cells
behind the coarse nose strips (0.156 in span against 0.078 in the graded
region): a resolution limit along the span, not a quadrature defect.

## Skewed cells (2026-10-10)

`Galerkin singular quadrature order for skewed cells` (with `Galerkin skewed
cell angle`, default 30 degrees) promotes any touching pair with a cell whose
smallest interior angle is below the angle. On the tips only the thin rhombi
near the trailing edge qualify (66 of 1792 tip cells at level 3). On the
campaign meshes (p = 2, parametric exact geometry, Dirichlet datum
interpolated), error at the corners of the trailing edge and the tips:

| Rules | L1 | L2 | L3 | near field at L3 |
|---|---:|---:|---:|---:|
| order 7, folded 10 | 2.5e-4 | 8.9e-4 | 3.7e-3 | 184 s |
| tips (materials 1, 2) at 12 | 2.6e-5 | 3.4e-5 | 6.3e-5 | 649 s |
| cells below 30 degrees at 12 | 2.6e-5 | 3.4e-5 | 6.5e-5 | 190 s |

The geometric criterion gives the result of the material criterion at the
cost of the standard rules.

## Dirichlet datum of the mixed problem

`Mixed Galerkin Dirichlet datum` is now `projection` by default (L2 projection
on the Dirichlet cells; the copies on the Neumann side of the interface take
the value of their twin). The nodal interpolant, still available as
`interpolation`, leaves an error that is smooth where the cells change size;
the map from the datum to the flux turns it into an error of the flux one
order lower. With the projection the L-infinity error of the flux in the
interior and at the junctions stays within 0.8-1.3 times the local best
approximation, and the L2 error converges at the rate of the best
approximation (p = 1, level 4: 0.091% against 0.25%; p = 2, level 3: 0.034%
against 0.086%).

## Memory of raised quadrature orders (2026-10-10)

The direct path caches every singular rule it builds, keyed by pair type,
order and shared vertices, with its points, weights, shape values and
derivatives (`light_rule_cache`). The cache had no bound. A common-edge rule
of order n has 6 n^4 points, in up to 32 configurations; at order 12 and
p = 2 one configuration takes about 60 MB. With
`Galerkin singular quadrature order for skewed cells = 12` the campaign runs
took 0.4-0.75 GB more per process (p = 2, level 1: 1.37 GB against 0.62 GB),
and the two runs of level 4 at p = 1 were stopped by the memory guard.

Rules of orders above `Galerkin singular quadrature order` (folded, material
and skewed pairs) now share a budget of 128 MB, the least recently used
evicted first; rules of the base order are cached as before. No leaf holds a
rule across lookups, and a rebuilt rule is identical. On p = 2, level 1 the
memory falls to 0.72 GB with the same errors; the regression suite passes.

## Adaptive refinement and hanging nodes across the interface (2026-10-10)

`Mixed Galerkin error indicator = jumps | exact` computes, after each mixed
solve, an indicator per cell: the jumps of the conormal derivative of the
flux (Dirichlet cells) and of the potential (Neumann cells) across the edges
to cells of the same part and material, with the extent a_F = |T|/|F| of the
cell across each edge, a_F/2 for H^1 and a_F^3/(2 pi^2) for L^2 (Poincare
constant a/pi), weighted by a = min a_F for the energy norm or normalised by
the norms of the two parts for `Mixed Galerkin indicator norm = L2`. `exact`
uses the exact errors of the manufactured solution (oracle). With `Set Global
Refinement = false`, `Initial refinement level` (Driver) and `Number of
cycles`, `adaptive_refinement()` marks the cells carrying `Mixed Galerkin
refinement fraction` of the sum (Doerfler). On hanging edges the coarse side
is evaluated at the points of the fine edge pulled back onto the mapped
coarse edge (the hanging vertex of the exact geometry is not at the midpoint
of the reference edge). The indicator is written as column 13 of the error
dump. On the uniform meshes it correlates with the exact error cell by cell
(log-correlation 0.95-0.98) and marks nearly the same cells as the oracle.

The first adaptive runs exposed a defect of the mixed problem: hanging
nodes across the interface of the two parts, which uniform meshes do not
have. The standard constraints of the whole surface tie a node of the finer
side to the degrees of freedom of the coarse cell of the other part: the
flux was constrained to potentials, or a potential to fluxes, and the node
in the middle of a coarse Neumann edge (p = 2) next to finer Dirichlet cells
became a free potential unknown on the interface. On such meshes the
residual of the exact solution jumped from 1e-3 to 7e-2 and the flux error
from 0.5% to 6-10%. Now the constraints are built per part: a hanging node
whose coarse cell lies on the other part is free on the Dirichlet side (the
flux is continuous on that part only) and fixed by the datum on the Neumann
side (the trace of the coarse Dirichlet cell, by the weights of the
constraint), and every Neumann degree of freedom on an edge to Dirichlet
cells has its potential fixed by the datum. The regression suite is
unchanged.

## Folded pairs at refined trailing edges (2026-10-11)

When adaptive refinement reached the trailing edge, a refined upper cell faced
a lower cell that `make_edges_conformal` had cut only across the double-node
edge: a folded pair (16.5 degrees) whose cells reach to depths 2:1 from the
edge. The rule of ten points per variable then loses accuracy (at p = 2 the
flux error rose from 0.10 to 0.57%, 80% of it in these cells, local best
1e-5). A reproducer (level 1, graded refinement of the upper side at the
trailing edge) showed that fourteen points remove it and the near-singular
subdivision does not matter. Two fixes, each sufficient:

- the integrator halves the deeper patch of a folded pair parallel to the
  common edge until the depths are within 1.25 (`folded_balance_ratio`);
- after adaptive refinement (`update_triangulation(true)` from the driver)
  the cell across a double-node edge is refined isotropically. The meshes of
  `refine_and_resize()` keep the anisotropic cut (the exponential sphere-box
  mesh of the regression changed otherwise).

On finely refined trailing edges the rule of ten points still limits the
flux error near 5e-4 (largest error 5.3e-3 of the largest flux with ten
points, 4.0e-4 with fourteen, 7.3e-5 with eighteen; local best 6e-6). The
rules of raised orders now keep only points and weights in the cache and
evaluate the shapes per pair, so fourteen points for the folded pairs cost a
near field 46% longer on the uniform level 2 (72 s against 49 s; before the
change 271 s from cache churn); results at ten points are bitwise unchanged.
The adaptive runs at p = 2 use fourteen points (`run_adapt7.sh`).

## Performance of the mixed solver (2026-10-11)

Measured on the wing, exact parametric geometry, p = 2, eight processes.

Near field (PIBEM_NEAR_PROFILE prints its time by pair type and the parts
of the direct path): 82% of it was in coincident and common-edge pairs, and
in these the sums over the pair points of the joint kernel loop read some
600 arrays of all points per pair, from memory. Now the points go in blocks
of 128 that stay in cache, the sums are small matrix products in register
blocks of 3 x 3 (`add_row_products`), W is formed from the reference
derivatives with four tangent weights instead of the surface curls, and the
work arrays of the direct path are kept from pair to pair (allocated anew,
arrays of megabytes cost a page fault per page) and released after the
assembly. Common-edge pair 4.1 -> 1.5 ms, coincident 3.0 -> 1.7 ms; near
field at level 3 234 -> 122 s.

Products: `LaplaceFMM::evaluate_mixed()` computes V t, K^T t, K u and the
three curl sums of W u in one pass (shared tree passes, one direct kernel,
one sum over the processes), when charges and dipoles have the same order
and the multipoles of five sets take less than 256 MB per process: -10% per
product. The M2L translations, half of a product, are the same five sets
either way. Order 7 instead of 8 costs 0.6% in the flux at level 3 (order 6:
11%) and saves 20% of a product.

`Mixed Galerkin solver reduction` (default 1e-12): 1e-9 takes 43 GMRES steps
instead of 54 at level 3 and changes the flux error in the sixth digit.

Adaptive refinement: the exact near-field blocks of pairs of cells that are
not refined are kept from one solve to the next (`NearFieldCache`, pairs
dealt out to the processes by a hash of the levels and indices of their
cells), 90-97% of the pairs per cycle; GMRES starts from the traces of the
previous mesh (each new cell evaluates the polynomial of its ancestor), with
the stopping criterion relative to the preconditioned right-hand side.

| run | before | now |
|---|---:|---:|
| uniform level 3, near field | 234 s | 122 s |
| uniform level 3, total | 417 s | 276 s (252 s with reduction 1e-9) |
| adaptive, 14 cycles (p = 2, folded order 14) | 767 s | 223 s (179 s with 1e-9) |

Errors are unchanged to all printed digits (adaptive) and bitwise in the
near-field blocks up to the order of the sums. The regression suite has an
adaptive case (`naca_mixed_adaptive_p1_fmm`).

## Performance, second round (2026-10-11)

Same setting (wing, exact parametric geometry, p = 2, eight processes,
reduction 1e-9).

M2L (62% of a product at level 3). The batched rotation applied the
reflection D(P) entry by entry with the sums in memory, so that every
product waited for the previous one; now the rotation of a group,
D(Q) = R(theta) Z(-phi), uses dense blocks per degree of the rotation about
y, R(theta) = D(P) Z(theta) D(P), which depends only on the polar angle of
the offset (1793 distinct angles on 10^5 points, kept from `reinit`), in a
register-blocked product (`translate_blocks`, 16 lanes); pairs and sets of
a group fill the lanes together, multipoles and locals are converted once
per box to the scaled real coefficients, and the lanes are gathered and
scattered by 4 x 4 transposes. Kernel 230 -> 111 ns per expansion, M2L of
the five sets 0.93 -> 0.55 s on one core (benchmark `RUN_NACA/m2l`), level
3: 1.27 -> 0.63 s per product.

Near field:
- a cell with itself on half of the eight Sauter-Schwab regions (the other
  four are their images under the exchange of the two points), completed by
  V = A + A^T, K = C + D^T, W = B + B^T; the pair of the two halves of a
  split coincident cell gives its mirror image the same way;
- leaves on patches smaller than the cell use the shapes and derivatives
  kept with the rule and take the result to the shapes of the cell by the
  transform of Q_p under the affine map of the patch, T_x B T_y^T
  (`LightLeaf::transformed`, 98% of the singular leaves);
- the geometry of a rule is evaluated at its distinct points only (n^2 and
  n^3 of the n^4 pair points of a common vertex, n^3 of a common edge) and
  gathered: 7.6e8 -> 3.0e8 points at level 2;
- without a cache the pairs are taken by the processes as they get free,
  from a counter on process 0 (`MPI_Fetch_and_op`), in the order of
  decreasing cost, and assembled afterwards: imbalance 1.31 -> 1.00 on the
  hybrid cores (P and E).

Calderon preconditioner: the rank-one term of the constant of the Dirichlet
part is scaled to V^{-1} instead of 4 V^{-1}, which puts the constant at 1,
inside the spectrum, instead of the isolated value 3.7: one step less.

Krylov recycling was evaluated and not adopted. GMRES converges linearly,
by 0.55-0.65 per step in every adaptive cycle, with only three small
isolated harmonic Ritz values (0.054, 0.10, 0.15); deflating the exact
harmonic Ritz vectors of the same system (`PIBEM_DEFLATION_TEST`,
`include/deflated_gmres.h`) saves 4-6 of about 26 steps, and in two cycles
of six costs steps, an upper bound for vectors transferred between meshes.

| run | morning | now |
|---|---:|---:|
| level 2, near field | 27.3 s | 11.1 s |
| level 2, total | 59.0 s | 34.0 s |
| level 3, near field | 115 s | 42.8 s |
| level 3, product (both rows) | 2.15 s | 1.40 s |
| level 3, total | 252 s | 143 s |
| adaptive, 10 cycles | 90.0 s | 68.4 s |

Errors identical to all printed digits; memory +60 MB per process at level 3
(the blocks kept until the assembly).

## Kinked pairs: the edges of the tips (2026-10-12)

With the plane wave u = x + y + z (level 3, p = 2, exact parametric
geometry) the potential is exact to 2.7e-7 and the flux to 2.7e-4, 75% of
it in the cells along the leading edge, where it is the approximation of
n . (1, 1, 1) around the nose (3.65e-5 per cell against 2.81e-5 for the
best approximation; 1.6e-4 in total). Pointwise, the flux error peaked at
the two corners of the leading edge with the tips, 2.6e-2 against 1.9e-3
for the best approximation in those cells (4% of the squared L2 error):
the singular rules of the pairs across the edge of the tip, at 90 degrees
where the nose bends most.

Under `BEMProblem<3>/Quadrature rules`:

```text
set Galerkin singular quadrature order for kinked pairs = 10
set Galerkin kinked pair angle = 60
```

promote the touching pairs whose vertex normals make more than the given
angle. On the wing at level 3 the angles of touching cells are 0-45 degrees
within the smooth surfaces (the nose included), 75-105 across the edges of
the tips (760 pairs) and 150-165 at the trailing edge (folded pairs), so 60
degrees takes the edges only. Corner cells, maximum flux error: 2.6e-2
(order 7), 1.2e-2 (8), 6.3e-3 (9), 3.5e-3 (10, best approximation 1.9e-3);
the same as order 10 on all pairs touching the tips (materials 1, 2, 11,
12), which also takes the planar pairs within the tips and costs 54 s of
near field more, while the kinked pairs cost nothing measurable. The
largest pointwise error left is at the corners of the trailing edge with
the tips, 3.8e-3, unchanged. Off by default; the regression case
`naca_mixed_exact_p2_fmm` uses order 10.

## Thin regions, trailing edge and tip edges with the plane wave (2026-10-12)

Plane wave u = x + y + z, level 3, p = 2, exact parametric geometry, kinked
order 10. The flux error behind x = 0.6 (best approximation about 3e-8 per
cell) had three sources, told apart one knob at a time (runs in
`RUN_NACA/plane/`):

- far pairs across the thin aft section: upper and lower cells a diameter
  apart get the tensor rule of the multipole sums (order 6), too coarse for
  the double layer of one surface across the other, whose weight is of
  order one within their separation; on one smooth surface it nearly
  vanishes. The new parameter `Galerkin near singular distance ratio for
  kinked pairs` (normals beyond the kinked pair angle) makes those pairs
  near-singular up to the given ratio of the larger diameter: 1.0 takes
  59 000 more pairs at level 3, where the same ratio on all pairs takes
  350 000 for the same accuracy. Peaks of 1e-3 next to the spanwise lines of
  hanging nodes at x = 0.8 and 0.91 came from these pairs, not from the
  non-conforming contacts (order 12 on those changes nothing);
- folded pairs at the trailing edge: order 10 leaves 2e-3 at x > 0.97, order
  14 1e-4 (order 20 the same). On an isolated folded pair (16 degrees,
  `RUN_NACA/diag_tip/folded.cc`) the double layer is accurate to 1.3e-5 at
  order 10 and 1.1e-7 at order 14;
- not the multipole truncation (order 12 and dense matrices give the same at
  level 2), nor GMRES (reduction 1e-12), nor the datum (interpolated), nor
  the close regular pairs (order 10 for their exact blocks, or depth and
  ratio raised, change nothing).

| level 3 | flux L2 | aft max, mid-span | aft max, tip rows | near field |
|---|---:|---:|---:|---:|
| kinked 10 | 2.68e-4 | 2.75e-3 | 3.8e-3 | 40 s |
| + folded 14 | 2.50e-4 | 1.62e-3 | 3.8e-3 | 60 s |
| + kinked ratio 1.0 | 2.41e-4 | 4.5e-4 (1e-4 for abs(y) < 1) | 4.4e-3 | 65 s |

The best approximation on this mesh is 1.6e-4. Left: the rows of the lower
surface along the edges of the tips behind x = 0.6, 2e-3 to 6e-3 at the
nodes on the edge itself (the interior nodes of those cells 5e-4), larger on
the +y tip, where the tip cells are smaller. Insensitive to every rule above
(kinked and tip-material orders 7-12, regular, near-singular, folded); at
level 2 all rules raised together leave 5.8e-4 there against 6.9e-5 at
mid-span. The residual of the exact solution (`PIBEM_RESIDUAL_DUMP`) is not
larger on those rows. Open.

The rules of raised orders now keep the shapes and their derivatives at
their distinct points (n^2 or n^3 of the n^4 pair points of common vertex
and edge rules) and gather them per leaf, on patches through the affine
transform, instead of evaluating them at every pair point.

## Speed with the new rules, inexact GMRES, Dirichlet parts in pieces (2026-10-12)

Kinked pairs near only by the kinked ratio (between the near-singular ratio
and the kinked one) take one tensor rule of the order that matches, at
their separation, the accuracy of the base order at the kinked ratio (the
Gauss convergence factor rho(r) = 2r + sqrt(4r^2 + 1) at r diameters: order
10 at half a diameter, 8 at three quarters, at most 12), instead of the
subdivision that made every subpair a diameter apart: level 3 near field
65 -> 53 s, errors identical.

`Mixed Galerkin relaxed products below` (default 0): inexact GMRES. Once the
preconditioned residual is below the given fraction of the preconditioned
right-hand side, the products of the mixed rows use the expansions of the
preconditioner (order 4, opening 0.6) with the exact near field; the solver
prints their relative error (1.3e-4 at level 2, 1.7e-4 at level 3), and the
fraction should be about the reduction over it (1e-5 for 1e-9). Level 3: 18
of 41 steps relaxed, the relaxed product 0.43 s against 1.44 s, GMRES 85 ->
70 s, total 149 -> 137 s; the flux error changes in the fourth digit
(2.41329e-4 -> 2.41396e-4). The GMRES of `include/inexact_gmres.h` runs it.

Bug: with the tips in the Dirichlet part (Dirichlet ids 1,2,4), the flux was
52% wrong at level 2 (24% with every surface Dirichlet). W vanishes on the
constant of each piece of the Dirichlet part whose flux unknowns no cell
joins (the cells on the two sides of a sharp edge have separate unknowns),
and the Calderon preconditioner restored one constant only: singular on the
others, it let GMRES stop on its residual with the true residual of the flux
rows at 2e-3. Now one rank-one term per piece (0.22% at level 2, true
residual 6e-8). Regression case `naca_mixed_tips_dirichlet_p1_fmm`.

Tip edges (continued). With the tips Dirichlet, the lower surface along their
edges keeps the same error as in the mixed problem: not the Dirichlet-Neumann
interface. Refining towards the edges (`Edge refinement cycles` 1 and 2 at
level 2) raises the error there (2.1e-3 -> 3.9e-3 -> 1.1e-2) while the best
approximation falls tenfold; on that mesh the residual of the exact solution
is 3000 times its median at the hanging nodes of the nose next to the tips
(the D-N interface on the most curved part), and stays so with all rules
raised: the constraints there, not the quadrature. Along the aft tip edges
the residual stays small on every mesh: their error is pollution from such
defects and from the approximation, not a local rule. Open.

## The leading-edge band and refinement by curvature (2026-10-13)

With the plane wave the flux error lay in a band along the leading edge: 98%
of its square in the cells with x < 0.03, at the level of the best
approximation, since the flux n . (1, 1, 1) turns by 180 degrees round a
nose of radius 1.6% of the chord and a cell of level 3 there spans 40-70
degrees of it. New options of `ComputationalDomain`:

```text
set Curvature refinement cycles = 1
set Curvature refinement angle = 10
```

cut, after the global and the edge refinement, the cells across which the
surface turns by more than the angle in a reference direction, in that
direction only (the turning of an edge from the sagitta of its midpoint on
the manifold, 4 atan(2s/L)); at the nose the new lines run along the span,
so no hanging node lands on the Dirichlet-Neumann interface, and the tips
are made conforming along their edges (`make_edges_conformal()`).

| level 3, plane wave | cells | flux L2 | best approximation | LE band share | max |
|---|---:|---:|---:|---:|---:|
| none | 11520 | 2.41e-4 | 1.6e-4 | 98% | 4.4e-3 |
| 20 degrees, 1 cycle | 11782 | 1.27e-4 | 8.1e-5 | 95% | 2.3e-2 |
| 20 degrees, 2 cycles | 12044 | 1.26e-4 | 7.7e-5 | 90% | 8.8e-3 |
| 10 degrees, 1 cycle | 12042 | 6.6e-5 | 3.7e-5 | 84% | 2.4e-2 |

At 20 degrees only the first row of the nose is cut (a second cycle cuts it
again); 10 degrees takes the second row as well. The maximum moves to the
two cells of the lower surface at the corners of the nose with the tips,
2.4e-2 against 5e-4 for their best approximation, 30% of the squared error
(5.5e-5 without them). There the residuals of both the computed solution
(6e-8) and the exact one are small: a nearly singular mode of the discrete
system at that corner, which appears with the hanging nodes the conformity
leaves on the tips next to it (14 cells, none without the refinement), as
with `Edge refinement cycles`. Open. Regression case
`naca_mixed_curvature_p2_fmm`.

## Speed of the whole run (2026-10-13)

Level 3 with the settings of `RUN_NACA/parameters_bem_3.prm` (kinked pairs,
folded 14, relaxed products, curvature refinement), timeline from the
output: 162 s, of which 32 s before the mixed solve, all of it repeated on
every process. Now:

- the global refinement on the CAD surfaces (18.8 s) went to
  `OpenCASCADE::NormalToMeshProjectionManifold`, which loads the intersector
  of the shape (its face classifiers) for every new point and projects both
  ends of every line anew: `CachedNormalToMeshProjectionManifold`, the same
  algorithm and the same points, loads it once and keeps the normals of the
  vertices: 3.5 s;
- the estimate of the degree of the interpolated exact geometry (9 s) took
  every 64th cell, scattered, so that every projection started cold: now
  every 8th cell of a contiguous part per process: 1 s, the same degree;
- the right-hand side takes one product of the rows of the mixed problem
  instead of seven separate ones; |W 1|, the residual of the exact solution
  and the error of the relaxed products only with
  `PIBEM_MIXED_DIAGNOSTICS` (or `PIBEM_RESIDUAL_DUMP`);
- the Calderon preconditioner applies V and W in one evaluation of four
  sets (`vmult_preconditioner_pair`): one direct kernel and two sums over
  the processes instead of four;
- near pairs whose quadrature points are translates of those of a pair
  integrated before (rounded to 1e-9 of the size of the cell, with their
  normals and materials) take its block: 31% of the pairs at level 3 (the
  parametrization of the CAD surfaces is not linear along the span, so
  most translated cells differ inside); `PIBEM_NO_CONGRUENCE` integrates
  all.

| level 3 | before | now |
|---|---:|---:|
| before the mixed solve | 31.8 s | 8.3 s |
| near field | 57 s | 37 s |
| right-hand side and checks | 5.9 s | 3.2 s |
| GMRES (42 steps) | 59 s | 60 s |
| total | 162 s | 115 s |

Same flux and potential errors to all digits. Measured on a cool machine:
after a few runs in a row the laptop throttles by up to 30%.

## A cheaper Calderon preconditioner (2026-10-13)

An application of the preconditioner at level 2 took 66 ms: 88% the
products with V and W through the expansions of the preconditioner, 12% the
four solves with the consistent masses. The preconditioner needs only to be
the same operator at every step, not an accurate one; but the relaxed
products of GMRES used the same expansions, so that a coarser preconditioner
changed the solution. They now have their own (`Galerkin relaxed truncation
order` and `Galerkin relaxed opening criterion`, zero for those of the
preconditioner), and `Galerkin preconditioner maximum points per leaf`
gives the preconditioner its own leaves. Level 2, relaxed products at order
4 and opening 0.6:

| preconditioner (order, opening, points per leaf) | GMRES steps | 43 applications | V or W |
|---|---:|---:|---:|
| 4, 0.6, 64 | 40 | 2.89 s | 29.9 ms |
| 2, 0.8, 64 | 40 | 2.02 s | 19.0 ms |
| 2, 0.8, 16 | 40 | 2.63 s | 25.6 ms |
| 3, 0.8, 32 | 40 | 2.43 s | 23.3 ms |
| 2, 0.9, 32 | 40 | 1.98 s | 18.0 ms |
| 2, 0.95, 32 | 40 | 1.62 s | 14.9 ms |
| 1, 0.9, 32 | 41 | 1.68 s | 14.8 ms |

The flux error stays 2.40665e-4 throughout. Level 3 with order 2, opening
0.95 and 32 points per leaf (now in `RUN_NACA/parameters_bem_3.prm`): V or
W 0.154 -> 0.065 s, GMRES 60 -> 47 s, total 115 -> 90 s, 42 steps as before;
the mass solves are now 28% of the preconditioner. What remains of its
products is mostly the two sums over the processes.

The recommended settings and the environment switches are collected in
`docs/settings.md`.
