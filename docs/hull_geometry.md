# Surface integrals on two ship hulls (2026-10-13)

Two CAD models were tested: the Wigley hull and the DTMB 5415. Both were
supplied as zip files in `RUN_WIGLEY` and `RUN_DTMB` (run folders, not
tracked), each with `Color_i.iges` surfaces, `Curve_j.iges` curves and a
coarse UCD mesh. Both are wetted hulls, open at the waterline z = 0. The
tests compare surface integrals over the mesh with those over the exact
CAD geometry.

## Input

The meshes needed four fixes, which are applied to copies;
`cad/` keeps the files as supplied.

- The line elements carry material 100 + j for `Curve_j.iges`, but
  `ComputationalDomain` puts the curve projectors on manifold ids 10 + j.
  Without renumbering, the keel, waterline and stems are refined as
  straight lines.
- The DTMB header declared 88 cells, but the file has 93. The last five are
  the transom (material 3, `Color_3.iges`) and its edges, which deal.II
  ignored.
- The DTMB transom cell was oriented against the hull.
- All DTMB normals were reversed so that they point out of the body, as on
  the NACA wing and the Wigley.

`tools/cut_cad` writes `cad_wet/`:

- It keeps the part of every `Color_i.iges` below z = 0, so that the
  waterline is a boundary of the faces.
- It rebuilds the given curves as the waterline edges of the cut faces,
  joined into one B-spline. The curve of the mesh lines and the boundary of
  the faces are then the same curve.
- It prints the wetted area and the volume of the cut faces by
  OpenCASCADE. These are the references below.

The commands are `cut_cad cad cad_wet 0 1=1 2=2` for the Wigley and
`cut_cad cad cad_wet 0 3=1 4=2 7=3` for the DTMB. The DTMB call adds
`Curve_7`, the transom waterline, which the supplied files lacked; its mesh
lines were refined as a straight chord, 3.2e-3 off the transom.

## The benchmark

`benchmarks/exact_geometry_surfaces domain.prm out.csv 4` computes, at
levels 0-4:

- the area of each material;
- the volume, int z n_z;
- the waterplane area, -int n_z;
- the longitudinal centre of buoyancy.

The lid of the opening at z = 0 adds nothing to the volume and centre
integrals, and its area is the waterplane area by Green's formula on the
waterline curves.

The geometries compared are:

- the mesh with Q1 and Q2 maps (Q2 through the CAD manifolds);
- the closest point of the CAD;
- its parametrization;
- the interpolated parametrization that the solver uses.

## Results

The Wigley CAD has one face per side. It is wall-sided above the waterline
and has C1 knot lines at z = 0 and at z = -T/2. It is a fit of the formula
y = B/2 (1 - (2x/L)^2)(1 - (z/T)^2): its waterline is off by up to 2.8e-4,
its volume by +4.7e-4 and its waterplane by +6.8e-4.

Relative errors of the area against OpenCASCADE:

| level (cells) | mesh Q1 | mesh Q2 | parametric | interpolated |
|---|---:|---:|---:|---:|
| 0 (26) | 3.0e-2 | 2.7e-3 | 1.5e-5 | 1.3e-7 |
| 2 (416) | 2.4e-3 | 7.9e-6 | 2.7e-6 | 2.4e-9 |
| 4 (6656) | 1.5e-4 | 2.8e-8 | 1.0e-8 | 1.5e-9 |

The volume of the interpolated map is within 2e-10 from level 3 on. The
exact maps stop at about 1e-8 because of the C1 knots inside the cells: 16
and 24 Gauss points differ by that much.

The DTMB CAD has three faces per side (bow with sonar dome, a narrow
transition, main hull) and two transom patches. It has C0 and C2 knots, and
a gap of 0.1 mm between the transom and the hull edge curves.

| level 4 (7424 cells) | area | volume | waterplane |
|---|---:|---:|---:|
| mesh Q2 | -1.8e-4 | -2.5e-5 | -2.6e-6 |
| parametric | -4.2e-6 | -5.9e-6 | -2.9e-6 |
| interpolated | -2.6e-6 | -2.2e-6 | -1.7e-7 |

At level 3 the parametric map gives 8.5e-6 on the area, against 7.3e-4 for
the Q2 mesh. Levels 0-2 stay at 1e-2 to 1e-3:

- the coarse mesh is too coarse for the sonar dome, and the closest point
  folds there;
- the projection stays at 3e-4 even at level 4.

## Cells across faces of a CAD model

`OpenCASCADE::ShapeProjector` used to map a cell onto one face only: the
face of its centre. DTMB cells that straddle the junctions of the faces had
their points sent to the surface of the neighbouring face, extrapolated or
clamped at its edge. The exact maps converged at first order: level 4 area
-8.5e-4, worse than the Q2 mesh.

Now a cell of a shape of several faces counts as multi-face when it is
within a diameter of a trimming curve of its face and the foot of one of
its nodes lies off that face. "Off the face" means the classifier says
outside, or x - S is not normal to the surface, as at an untrimmed edge of
the parameters, where Newton's method stops.

A multi-face cell is mapped as follows:

- Each point takes the closest of the faces near the cell that contains
  its foot.
- Under the parametric map, the cell is the projection of the transfinite
  interpolation of its edges. An edge whose feet lie on one face is the
  segment between the parameters of its ends on that face, as for the cell
  of that face on its other side. Other edges are projected.
- The images of the edges are kept per parameter on the cell, which takes
  the cost on the DTMB from 3.4 to 1.4 times that of the old, wrong map.

Two variants failed:

- Counting a cell as multi-face only when another face holds the off-face
  foot lost cells, because of gaps between the faces.
- Leaving cells at a free edge on the parametric map lost 3e-4 at the bow.

Shapes of one face are unchanged (Wigley, NACA wing, regression suite).

## Kinks of the parametrization

A B-spline face with an interior knot of multiplicity at least its degree
has a parametrization that is only continuous there. Where the surface is
nevertheless smooth, a map linear in the parameters has a kink inside the
cells that cross the knot line, and Gauss rules integrate it only to about
1e-4. An example is a NURBS circle made of rational arcs.

`CADPatch::kinks` holds these knots. A knot counts only if the normal is
continuous across it at 21 points; a knot line along which the surface has
an edge is left out, because every map has a kink there. Under the
parametric map, a cell that crosses a kink takes the projection of the
transfinite map of its edges, like a cell on several faces, provided its
nodes are within reach of the closest point. An edge is never a parametric
segment across a kink.

In the test, the split cylinder now uses a mesh whose cells straddle both
the junctions and the knots of the arcs. All maps are at round-off, with
Jacobians checked against central differences.

All C0 knots of the DTMB faces are true edges of the surface, so they stay
on the parametric map. Sending those cells to the projection lost 2.5e-4
at level 3, on the concave side of the edges.

Cells at a free edge with no other face close, such as the keel, take the
projection only if their nodes are within reach of the closest point
(G - H positive definite). Otherwise they keep the parametric map, which
does not fold them.

## Coarse levels and the sonar dome

At levels 0-2 the keel cells under the sonar dome have edges about 10 cm
long, whose chords lie up to 4 cm inside the bulb. The projection sends
such a cell to the part of the surface nearest its chords, a strip along
the keel, and loses up to 70% of its area. The parametric map fares no
better at the free edge. The remedy is in the mesh: four cycles of
refinement by curvature at 30 degrees (`RUN_DTMB/domain_curvature4.prm`).

| parametric, area | uniform | curvature, 4 cycles |
|---|---:|---:|
| level 0 | -3.4e-2 | -1.4e-3 |
| level 1 | -7.1e-3 | -2.5e-4 |
| level 2 | -6.8e-3 | +4.0e-5 |
| level 3 | -8.5e-6 | +4.7e-5 |

The anisotropic refinement leaves hanging nodes and long cells, where the
images of neighbouring cells overlap. The exact maps then stop at about
+5e-5, and the Q2 mesh drifts to +1.7e-3 at level 3. Refinement by
curvature rescues the coarse levels but is no mesh to converge on. A
conforming coarse mesh that resolves the dome is needed.

## The interpolated geometry

The solver interpolates the exact map with one polynomial degree per mesh.
On the DTMB at level 3, the number of cells whose surface element deviates
by more than 1e-3 falls with the degree, but the largest deviation stays
about 0.5. Those cells are crossed by edges of the surface or by face
junctions, many of them along the keel.

| degree | cells above 1e-3 (of 1856) | largest deviation |
|---:|---:|---:|
| 4 | 746 | 0.20 |
| 6 | 321 | 0.48 |
| 8 | 93 | 0.55 |

Interpolating the projection instead is far worse: deviations up to 72,
where it folds cells. The integrals stay good (level 4 area -2.6e-6),
since the bad cells are few. For a solve, the remedies are a mesh aligned
with the faces and edges of the CAD, or the per-point exact map
(interpolation degree 0, tolerance 0).

## Open

- A conforming coarse DTMB mesh that resolves the sonar dome and follows
  the faces and edges of the CAD.
- The 0.1 mm gap between the transom and the hull in the DTMB CAD.
- Both hulls are open at the waterline. A boundary element solve needs a
  closed surface: a double body mirrored in z = 0, or a lid.
