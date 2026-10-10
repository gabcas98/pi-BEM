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
