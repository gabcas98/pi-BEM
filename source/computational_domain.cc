

#include "../include/computational_domain.h"
#include "../include/cached_normal_projection_manifold.h"

#include <atomic>
#include "planar_mesh_quality.h"

#include <deal.II/grid/grid_tools.h>

using namespace std;

// @sect4{ComputationalDomain::ComputationalDomain and
// ComputationalDomain::read_parameters}
// The constructor initializes the
// variuous object in much the same
// way as done in the finite element
// programs such as step-4 or
// step-6. The only new ingredient
// here is the ParsedFunction object,
// which needs, at construction time,
// the specification of the number of
// components.
//
// For the exact solution the number
// of vector components is one, and
// no action is required since one is
// the default value for a
// ParsedFunction object. The wind,
// however, requires dim components
// to be specified. Notice that when
// declaring entries in a parameter
// file for the expression of the
// Functions::ParsedFunction, we need
// to specify the number of
// components explicitly, since the
// function
// Functions::ParsedFunction::declare_parameters
// is static, and has no knowledge of
// the number of components.
template <int dim>
ComputationalDomain<dim>::ComputationalDomain(MPI_Comm comm)
  : mpi_communicator(comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , used_spherical_manifold(false)
  , pcout(std::cout)
{
  // Only output on first processor.
  pcout.set_condition(this_mpi_process == 0);
}

template <int dim>
ComputationalDomain<dim>::~ComputationalDomain()
{
  for (unsigned int i = 0; i < cad_surfaces.size(); ++i)
    {
      tria.reset_manifold(1 + i);
    }

  for (unsigned int i = 0; i < cad_curves.size(); ++i)
    {
      tria.reset_manifold(11 + i);
    }

  tria.reset_manifold(0);
}



template <int dim>
void
ComputationalDomain<dim>::declare_parameters(ParameterHandler &prm)
{
  if (dim == 3)
    prm.declare_entry("Input grid name",
                      "../grids/coarse_cube_double_nodes",
                      Patterns::Anything());
  else
    prm.declare_entry("Input grid name",
                      "../grids/circle",
                      Patterns::Anything());


  prm.declare_entry("Input grid format", "inp", Patterns::Anything());

  prm.declare_entry("Input path to CAD files", "./", Patterns::Anything());

  prm.declare_entry("Number of cycles", "2", Patterns::Integer());

  prm.declare_entry("Max aspect ratio", "3.5", Patterns::Double());

  prm.declare_entry("Use iges surfaces and curves", "false", Patterns::Bool());

  prm.declare_entry("Cad tolerance to projectors tolerance ratio",
                    "100",
                    Patterns::Double());

  prm.declare_entry("Surface curvature adaptive refinement",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Cells per circle", "12", Patterns::Double());

  prm.declare_entry("Maximum number of curvature adaptive refinement cycles",
                    "5",
                    Patterns::Integer());

  // The curvature refinement above compares the diameter of a cell with the
  // radius of curvature, and refines isotropically: a cell elongated along a
  // line of high curvature, as along the leading edge of a wing, is divided
  // along the line too. With this option a cell is cut only across the
  // directions in which the normal of the CAD surface turns by more than
  // 2 pi / (Cells per circle) between its opposite faces, as the aspect
  // ratio reduction cuts it.
  prm.declare_entry("Anisotropic curvature refinement",
                    "false",
                    Patterns::Bool());

  // A refinement concentrated on a curved line leaves cells much longer
  // than their neighbours across it, and the Galerkin solution deteriorates
  // where the size jumps. With a positive ratio, after the curvature
  // refinement every cell longer than the ratio times a neighbour across a
  // face, both measured across that face, is cut in that direction, until
  // no cell is. Only neighbours of the same material count: across an edge
  // between two surfaces the lengths are measured in different directions
  // (along the span on one side, through the thickness on the other).
  prm.declare_entry("Curvature refinement grading ratio",
                    "0",
                    Patterns::Double(0.));

  // The anisotropic curvature refinement makes the cells across a curved
  // line thin; with this option they are then cut along their longer
  // direction until no aspect ratio exceeds Max aspect ratio, as the input
  // grid is before the curvature refinement.
  prm.declare_entry("Aspect ratio reduction after curvature refinement",
                    "false",
                    Patterns::Bool());

  // Before the global refinement, the coarse side of every hanging node is
  // cut across the edge that holds it, cycle after cycle, until the mesh has
  // no hanging node; the global refinements keep it so.
  prm.declare_entry("Remove hanging nodes", "false", Patterns::Bool());

  // After the global refinement, the cells with a vertex on an edge of the
  // surface, a boundary face of the triangulation where two CAD surfaces
  // meet, are refined, this many times: a geometric grading towards the
  // edges, one ring of cells per size, that leaves the rest of the mesh as
  // it is.
  prm.declare_entry("Edge refinement cycles", "0", Patterns::Integer(0));
  // After the global and the edge refinement, the cells across which the
  // surface turns by more than the angle (degrees) in a direction are cut
  // in that direction, this many times: as the leading edge of a wing,
  // where the trace of the flux follows the normal round the nose. The
  // turning along a direction is that of the edges of the cell along it,
  // from the distance of their midpoints on the geometry to their chords.
  prm.declare_entry("Curvature refinement cycles", "0", Patterns::Integer(0));
  prm.declare_entry("Curvature refinement angle",
                    "20",
                    Patterns::Double(0., 180.));
  prm.declare_entry("Planar mesh smoothing material ids", "",
                    Patterns::List(Patterns::Integer(0)));
  prm.declare_entry("Planar mesh smoothing iterations", "0", Patterns::Integer(0));

  prm.declare_entry(
    "Number of global refinement to be executed before local refinement cycle",
    "0",
    Patterns::Integer());

  // Uniform refinement keeps whatever resolution imbalance the input grid
  // carries. These two entries refine a chosen set of material ids further,
  // so that surfaces meshed coarsely in the input can be brought up to the
  // resolution the solution actually needs.
  prm.declare_entry("Extra refinement material ids",
                    "",
                    Patterns::List(Patterns::Integer(0)));

  prm.declare_entry("Extra refinement cycles", "0", Patterns::Integer(0));

  // A second set, refined after the first with its own number of cycles, so
  // that two surfaces can be brought to different resolutions: for instance
  // the sides of a box and a body facing it.
  prm.declare_entry("Additional refinement material ids",
                    "",
                    Patterns::List(Patterns::Integer(0)));

  prm.declare_entry("Additional refinement cycles",
                    "0",
                    Patterns::Integer(0));

  // Graded refinement drives the cell size by distance from a focus point
  // rather than refining uniformly, so that the far field, where the flux is
  // negligible, is not refined along with the region that needs it.
  prm.declare_entry("Graded refinement material ids",
                    "",
                    Patterns::List(Patterns::Integer(0)));

  prm.declare_entry("Graded refinement focus",
                    "0,0,0",
                    Patterns::List(Patterns::Double(), dim, dim));

  prm.declare_entry("Graded refinement size ratio",
                    "0.5",
                    Patterns::Double(0.0));

  prm.declare_entry("Graded refinement minimum size",
                    "0.1",
                    Patterns::Double(0.0));

  // Equidistributing the L2 error of a flux decaying like d^-3 wants the cell
  // size to grow like d^((p+4)/(p+2)), i.e. 5/3 for piecewise linears. The
  // default 1.0 is the plain distance-proportional law, which over-resolves
  // the far field and makes the cell count grow like log of the domain size.
  prm.declare_entry("Graded refinement growth exponent",
                    "1.0",
                    Patterns::Double(0.0));

  prm.declare_entry("Graded refinement maximum cycles",
                    "0",
                    Patterns::Integer(0));

  // Exponential refinement, for a solution that varies like e^{-k.x}, such
  // as a plane wave of the screened equation: the cell diameter is driven
  // towards minimum_size e^{g k.(x - x*)}, x* the point of the mesh where the
  // solution is largest. The L2 error of piecewise polynomials of degree p
  // on a surface is equidistributed by g = 1/(p+2): each cell then carries
  // the same share, h^{p+2} e^{-k.x} constant.
  prm.declare_entry("Exponential refinement decay vector",
                    "",
                    Patterns::List(Patterns::Double(), 0, dim));

  prm.declare_entry("Exponential refinement minimum size",
                    "0.1",
                    Patterns::Double(0.0));

  prm.declare_entry("Exponential refinement grading",
                    "0.3333333333333333",
                    Patterns::Double(0.0));

  prm.declare_entry("Exponential refinement maximum cycles",
                    "0",
                    Patterns::Integer(0));

  prm.enter_subsection("Boundary Conditions ID Numbers");
  {
    prm.declare_entry("Dirichlet boundary ids",
                      "1,110,110",
                      Patterns::List(Patterns::Integer(0)));
    prm.declare_entry("Neumann boundary ids",
                      "0,110,110",
                      Patterns::List(Patterns::Integer(0)));
  }
  prm.leave_subsection();

  prm.declare_entry("Use a spheroid", "false", Patterns::Bool());

  prm.declare_entry("Axis x dimension", "2.", Patterns::Double());

  prm.declare_entry("Axis y dimension", "3.", Patterns::Double());

  prm.declare_entry("Axis z dimension", "4.", Patterns::Double());
}

template <int dim>
void
ComputationalDomain<dim>::parse_parameters(ParameterHandler &prm)
{
  input_grid_name            = prm.get("Input grid name");
  input_grid_format          = prm.get("Input grid format");
  input_cad_path             = prm.get("Input path to CAD files");
  n_cycles                   = prm.get_integer("Number of cycles");
  max_element_aspect_ratio   = prm.get_double("Max aspect ratio");
  use_cad_surface_and_curves = prm.get_bool("Use iges surfaces and curves");
  surface_curvature_refinement =
    prm.get_bool("Surface curvature adaptive refinement");
  cells_per_circle       = prm.get_double("Cells per circle");
  pre_global_refinements = prm.get_integer(
    "Number of global refinement to be executed before local refinement cycle");
  extra_refinement_cycles = prm.get_integer("Extra refinement cycles");
  extra_refinement_material_ids.clear();
  for (const auto &id :
       Utilities::split_string_list(prm.get("Extra refinement material ids")))
    if (!id.empty())
      extra_refinement_material_ids.insert(
        static_cast<types::material_id>(Utilities::string_to_int(id)));

  additional_refinement_cycles = prm.get_integer("Additional refinement cycles");
  additional_refinement_material_ids.clear();
  for (const auto &id : Utilities::split_string_list(
         prm.get("Additional refinement material ids")))
    if (!id.empty())
      additional_refinement_material_ids.insert(
        static_cast<types::material_id>(Utilities::string_to_int(id)));

  graded_refinement_material_ids.clear();
  for (const auto &id :
       Utilities::split_string_list(prm.get("Graded refinement material ids")))
    if (!id.empty())
      graded_refinement_material_ids.insert(
        static_cast<types::material_id>(Utilities::string_to_int(id)));
  {
    const std::vector<std::string> focus =
      Utilities::split_string_list(prm.get("Graded refinement focus"));
    AssertThrow(focus.size() == dim,
                ExcMessage("Graded refinement focus needs one coordinate per "
                           "space dimension."));
    for (unsigned int d = 0; d < dim; ++d)
      graded_refinement_focus[d] = Utilities::string_to_double(focus[d]);
  }
  graded_refinement_size_ratio = prm.get_double("Graded refinement size ratio");
  graded_refinement_minimum_size =
    prm.get_double("Graded refinement minimum size");
  graded_refinement_growth_exponent =
    prm.get_double("Graded refinement growth exponent");
  graded_refinement_maximum_cycles =
    prm.get_integer("Graded refinement maximum cycles");
  exponential_refinement_decay_vector.clear();
  for (const auto &k : Utilities::split_string_list(
         prm.get("Exponential refinement decay vector")))
    if (!k.empty())
      exponential_refinement_decay_vector.push_back(
        Utilities::string_to_double(k));
  AssertThrow(exponential_refinement_decay_vector.empty() ||
                exponential_refinement_decay_vector.size() == dim,
              ExcMessage("Exponential refinement decay vector needs one "
                         "component per space dimension, or none."));
  exponential_refinement_minimum_size =
    prm.get_double("Exponential refinement minimum size");
  exponential_refinement_grading =
    prm.get_double("Exponential refinement grading");
  exponential_refinement_maximum_cycles =
    prm.get_integer("Exponential refinement maximum cycles");
  max_curvature_ref_cycles =
    prm.get_integer("Maximum number of curvature adaptive refinement cycles");
  anisotropic_curvature_refinement =
    prm.get_bool("Anisotropic curvature refinement");
  curvature_refinement_grading =
    prm.get_double("Curvature refinement grading ratio");
  aspect_ratio_after_curvature_refinement =
    prm.get_bool("Aspect ratio reduction after curvature refinement");
  remove_hanging_nodes = prm.get_bool("Remove hanging nodes");
  edge_refinement_cycles = prm.get_integer("Edge refinement cycles");
  curvature_refinement_cycles =
    prm.get_integer("Curvature refinement cycles");
  curvature_refinement_angle = prm.get_double("Curvature refinement angle");
  planar_smoothing_iterations = prm.get_integer("Planar mesh smoothing iterations");
  planar_smoothing_material_ids.clear();
  for (const auto &id : Utilities::split_string_list(
         prm.get("Planar mesh smoothing material ids")))
    if (!id.empty())
      planar_smoothing_material_ids.insert(
        static_cast<types::material_id>(Utilities::string_to_int(id)));
  cad_to_projectors_tolerance_ratio =
    prm.get_double("Cad tolerance to projectors tolerance ratio");


  spheroid_bool   = prm.get_bool("Use a spheroid");
  spheroid_x_axis = prm.get_double("Axis x dimension");
  spheroid_y_axis = prm.get_double("Axis y dimension");
  spheroid_z_axis = prm.get_double("Axis z dimension");

  prm.enter_subsection("Boundary Conditions ID Numbers");
  {
    std::vector<std::string> dirichlet_string_list =
      Utilities::split_string_list(prm.get("Dirichlet boundary ids"));
    dirichlet_boundary_ids.resize(dirichlet_string_list.size());
    for (unsigned int i = 0; i < dirichlet_string_list.size(); ++i)
      {
        std::istringstream reader(dirichlet_string_list[i]);
        reader >> dirichlet_boundary_ids[i];
      }

    std::vector<std::string> neumann_string_list =
      Utilities::split_string_list(prm.get("Neumann boundary ids"));
    neumann_boundary_ids.resize(neumann_string_list.size());
    for (unsigned int i = 0; i < neumann_string_list.size(); ++i)
      {
        std::istringstream reader(neumann_string_list[i]);
        reader >> neumann_boundary_ids[i];
      }

    // dirichlet_sur_ID1 = prm.get_integer("Dirichlet Surface 1 ID");
    // dirichlet_sur_ID2 = prm.get_integer("Dirichlet Surface 2 ID");
    // dirichlet_sur_ID3 = prm.get_integer("Dirichlet Surface 3 ID");
    // neumann_sur_ID1 = prm.get_integer("Neumann Surface 1 ID");
    // neumann_sur_ID2 = prm.get_integer("Neumann Surface 2 ID");
    // neumann_sur_ID3 = prm.get_integer("Neumann Surface 3 ID");
  }
  prm.leave_subsection();
}


// @sect4{ComputationalDomain::read_domain}

// A boundary element method
// triangulation is basically the
// same as a (dim-1) dimensional
// triangulation, with the difference
// that the vertices belong to a
// (dim) dimensional space.
//
// Some of the mesh formats supported
// in deal.II use by default three
// dimensional points to describe
// meshes. These are the formats
// which are compatible with the
// boundary element method
// capabilities of deal.II. In
// particular we can use either UCD
// or GMSH formats. In both cases, we
// have to be particularly careful
// with the orientation of the mesh,
// because, unlike in the standard
// finite element case, no reordering
// or compatibility check is
// performed here.  All meshes are
// considered as oriented, because
// they are embedded in a higher
// dimensional space. (See the
// documentation of the GridIn and of
// the Triangulation for further
// details on orientation of cells in
// a triangulation.) In our case, the
// normals to the mesh are external
// to both the circle in 2d or the
// sphere in 3d.
//
// The other detail that is required
// for appropriate refinement of the
// boundary element mesh, is an
// accurate description of the
// manifold that the mesh is
// approximating. We already saw this
// several times for the boundary of
// standard finite element meshes
// (for example in step-5 and
// step-6), and here the principle
// and usage is the same, except that
// the HyperBallBoundary class takes
// an additional template parameter
// that specifies the embedding space
// dimension. The function object
// still has to be static to live at
// least as long as the triangulation
// object to which it is attached.

template <int dim>
void
ComputationalDomain<dim>::read_domain()
{
  std::ifstream in;
  in.open(input_grid_name + "." + input_grid_format);
  GridIn<dim - 1, dim> gi;
  gi.attach_triangulation(tria);
  if (input_grid_format == "vtk")
    gi.read_vtk(in);
  else if (input_grid_format == "msh")
    gi.read_msh(in);
  else if (input_grid_format == "inp")
    gi.read_ucd(in, true);
  else
    Assert(false, ExcNotImplemented());

  // GridTools::copy_material_to_manifold_id(tria);

  if (input_grid_name == "../grids/coarse_sphere" ||
      input_grid_name == "../grids/coarse_sphere_double_nodes" ||
      input_grid_name == "../grids/circle")
    {
      manifold = new SphericalManifold<dim - 1, dim>;
      tria.set_all_manifold_ids(0);
      tria.set_manifold(0, *manifold);
      used_spherical_manifold = true;
    }
}

template <>
void
ComputationalDomain<2>::create_initial_mesh()
{
  AssertThrow(true, ExcMessage("Create initial mesh only works in 3D"));
}

template <int dim>
void
ComputationalDomain<dim>::create_initial_mesh()
{
  std::vector<Point<dim>>        vertices;
  std::vector<CellData<dim - 1>> cells;
  SubCellData                    subcelldata;

  vertices.resize(34);

  vertices[0](0)  = 0;
  vertices[0](1)  = 0;
  vertices[0](2)  = 1;
  vertices[1](0)  = 0.57735;
  vertices[1](1)  = 0.57735;
  vertices[1](2)  = 0.57735;
  vertices[2](0)  = -0.57735;
  vertices[2](1)  = 0.57735;
  vertices[2](2)  = 0.57735;
  vertices[3](0)  = -0.57735;
  vertices[3](1)  = -0.57735;
  vertices[3](2)  = 0.57735;
  vertices[4](0)  = 0.57735;
  vertices[4](1)  = -0.57735;
  vertices[4](2)  = 0.57735;
  vertices[5](0)  = 0.70711;
  vertices[5](1)  = 0;
  vertices[5](2)  = 0.70711;
  vertices[6](0)  = 0;
  vertices[6](1)  = 0.70711;
  vertices[6](2)  = 0.70711;
  vertices[7](0)  = -0.70711;
  vertices[7](1)  = 0;
  vertices[7](2)  = 0.70711;
  vertices[8](0)  = 0;
  vertices[8](1)  = -0.70711;
  vertices[8](2)  = 0.70711;
  vertices[9](0)  = 1;
  vertices[9](1)  = 0;
  vertices[9](2)  = 0;
  vertices[10](0) = 0;
  vertices[10](1) = 1;
  vertices[10](2) = 0;
  vertices[11](0) = -1;
  vertices[11](1) = 0;
  vertices[11](2) = 0;
  vertices[12](0) = 0;
  vertices[12](1) = -1;
  vertices[12](2) = 0;
  vertices[13](0) = 0.70711;
  vertices[13](1) = 0.70711;
  vertices[13](2) = 0;
  vertices[14](0) = -0.70711;
  vertices[14](1) = 0.70711;
  vertices[14](2) = 0;
  vertices[15](0) = -0.70711;
  vertices[15](1) = -0.70711;
  vertices[15](2) = 0;
  vertices[16](0) = 0.70711;
  vertices[16](1) = -0.70711;
  vertices[16](2) = 0;
  vertices[17](0) = 0;
  vertices[17](1) = 0;
  vertices[17](2) = -1;
  vertices[18](0) = 0.57735;
  vertices[18](1) = 0.57735;
  vertices[18](2) = -0.57735;
  vertices[19](0) = -0.57735;
  vertices[19](1) = 0.57735;
  vertices[19](2) = -0.57735;
  vertices[20](0) = -0.57735;
  vertices[20](1) = -0.57735;
  vertices[20](2) = -0.57735;
  vertices[21](0) = 0.57735;
  vertices[21](1) = -0.57735;
  vertices[21](2) = -0.57735;
  vertices[22](0) = 0.70711;
  vertices[22](1) = 0;
  vertices[22](2) = -0.70711;
  vertices[23](0) = 0;
  vertices[23](1) = 0.70711;
  vertices[23](2) = -0.70711;
  vertices[24](0) = -0.70711;
  vertices[24](1) = 0;
  vertices[24](2) = -0.70711;
  vertices[25](0) = 0;
  vertices[25](1) = -0.70711;
  vertices[25](2) = -0.70711;
  vertices[26](0) = 1;
  vertices[26](1) = 0;
  vertices[26](2) = 0;
  vertices[27](0) = 0;
  vertices[27](1) = 1;
  vertices[27](2) = 0;
  vertices[28](0) = -1;
  vertices[28](1) = 0;
  vertices[28](2) = 0;
  vertices[29](0) = 0;
  vertices[29](1) = -1;
  vertices[29](2) = 0;
  vertices[30](0) = 0.70711;
  vertices[30](1) = 0.70711;
  vertices[30](2) = 0;
  vertices[31](0) = -0.70711;
  vertices[31](1) = 0.70711;
  vertices[31](2) = 0;
  vertices[32](0) = -0.70711;
  vertices[32](1) = -0.70711;
  vertices[32](2) = 0;
  vertices[33](0) = 0.70711;
  vertices[33](1) = -0.70711;
  vertices[33](2) = 0;


  cells.resize(24);



  cells[0].vertices[0]  = 0;
  cells[0].vertices[1]  = 5;
  cells[0].vertices[2]  = 1;
  cells[0].vertices[3]  = 6;
  cells[1].vertices[0]  = 7;
  cells[1].vertices[1]  = 0;
  cells[1].vertices[2]  = 6;
  cells[1].vertices[3]  = 2;
  cells[2].vertices[0]  = 7;
  cells[2].vertices[1]  = 3;
  cells[2].vertices[2]  = 8;
  cells[2].vertices[3]  = 0;
  cells[3].vertices[0]  = 4;
  cells[3].vertices[1]  = 5;
  cells[3].vertices[2]  = 0;
  cells[3].vertices[3]  = 8;
  cells[4].vertices[0]  = 9;
  cells[4].vertices[1]  = 13;
  cells[4].vertices[2]  = 1;
  cells[4].vertices[3]  = 5;
  cells[5].vertices[0]  = 13;
  cells[5].vertices[1]  = 10;
  cells[5].vertices[2]  = 6;
  cells[5].vertices[3]  = 1;
  cells[6].vertices[0]  = 10;
  cells[6].vertices[1]  = 14;
  cells[6].vertices[2]  = 2;
  cells[6].vertices[3]  = 6;
  cells[7].vertices[0]  = 14;
  cells[7].vertices[1]  = 11;
  cells[7].vertices[2]  = 7;
  cells[7].vertices[3]  = 2;
  cells[8].vertices[0]  = 11;
  cells[8].vertices[1]  = 15;
  cells[8].vertices[2]  = 3;
  cells[8].vertices[3]  = 7;
  cells[9].vertices[0]  = 15;
  cells[9].vertices[1]  = 12;
  cells[9].vertices[2]  = 8;
  cells[9].vertices[3]  = 3;
  cells[10].vertices[0] = 12;
  cells[10].vertices[1] = 16;
  cells[10].vertices[2] = 4;
  cells[10].vertices[3] = 8;
  cells[11].vertices[0] = 16;
  cells[11].vertices[1] = 9;
  cells[11].vertices[2] = 5;
  cells[11].vertices[3] = 4;

  cells[12].vertices[0] = 29;
  cells[12].vertices[1] = 32;
  cells[12].vertices[2] = 20;
  cells[12].vertices[3] = 25;
  cells[13].vertices[0] = 33;
  cells[13].vertices[1] = 29;
  cells[13].vertices[2] = 25;
  cells[13].vertices[3] = 21;
  cells[14].vertices[0] = 32;
  cells[14].vertices[1] = 28;
  cells[14].vertices[2] = 24;
  cells[14].vertices[3] = 20;
  cells[15].vertices[0] = 28;
  cells[15].vertices[1] = 31;
  cells[15].vertices[2] = 19;
  cells[15].vertices[3] = 24;
  cells[16].vertices[0] = 31;
  cells[16].vertices[1] = 27;
  cells[16].vertices[2] = 23;
  cells[16].vertices[3] = 19;
  cells[17].vertices[0] = 27;
  cells[17].vertices[1] = 30;
  cells[17].vertices[2] = 18;
  cells[17].vertices[3] = 23;
  cells[18].vertices[0] = 30;
  cells[18].vertices[1] = 26;
  cells[18].vertices[2] = 22;
  cells[18].vertices[3] = 18;
  cells[19].vertices[0] = 26;
  cells[19].vertices[1] = 33;
  cells[19].vertices[2] = 21;
  cells[19].vertices[3] = 22;
  cells[20].vertices[0] = 25;
  cells[20].vertices[1] = 20;
  cells[20].vertices[2] = 24;
  cells[20].vertices[3] = 17;
  cells[21].vertices[0] = 22;
  cells[21].vertices[1] = 21;
  cells[21].vertices[2] = 25;
  cells[21].vertices[3] = 17;
  cells[22].vertices[0] = 24;
  cells[22].vertices[1] = 19;
  cells[22].vertices[2] = 23;
  cells[22].vertices[3] = 17;
  cells[23].vertices[0] = 23;
  cells[23].vertices[1] = 18;
  cells[23].vertices[2] = 22;
  cells[23].vertices[3] = 17;


  cells[0].material_id  = 1;
  cells[1].material_id  = 1;
  cells[2].material_id  = 1;
  cells[3].material_id  = 1;
  cells[4].material_id  = 1;
  cells[5].material_id  = 1;
  cells[6].material_id  = 1;
  cells[7].material_id  = 1;
  cells[8].material_id  = 1;
  cells[9].material_id  = 1;
  cells[10].material_id = 1;
  cells[11].material_id = 1;
  cells[12].material_id = 0;
  cells[13].material_id = 0;
  cells[14].material_id = 0;
  cells[15].material_id = 0;
  cells[16].material_id = 0;
  cells[17].material_id = 0;
  cells[18].material_id = 0;
  cells[19].material_id = 0;
  cells[20].material_id = 0;
  cells[21].material_id = 0;
  cells[22].material_id = 0;
  cells[23].material_id = 0;


  GridTools::delete_unused_vertices(vertices, cells, subcelldata);
  GridTools::consistently_order_cells(cells);
  tria.create_triangulation(vertices, cells, subcelldata);

  static const Point<dim>                      center = Point<dim>();
  static const SphericalManifold<dim - 1, dim> manifold(center);
  tria.set_manifold(1, manifold);
  tria.set_manifold(0, manifold);
}

namespace
{
  std::atomic<unsigned int> checked_arclength_corrections{0};
}


CheckedArclengthProjectionLineManifold::CheckedArclengthProjectionLineManifold(
  const TopoDS_Shape &shape,
  const double        tolerance)
  : OpenCASCADE::ArclengthProjectionLineManifold<2, 3>(shape, tolerance)
  , shape(shape)
  , tolerance(tolerance)
{}


std::unique_ptr<Manifold<2, 3>>
CheckedArclengthProjectionLineManifold::clone() const
{
  return std::make_unique<CheckedArclengthProjectionLineManifold>(shape,
                                                                  tolerance);
}


unsigned int
CheckedArclengthProjectionLineManifold::n_corrected()
{
  return checked_arclength_corrections;
}


Point<3>
CheckedArclengthProjectionLineManifold::checked(
  const Point<3>                  &candidate,
  const ArrayView<const Point<3>> &surrounding_points,
  const ArrayView<const double>   &weights) const
{
  Point<3> average;
  for (unsigned int i = 0; i < surrounding_points.size(); ++i)
    average += weights[i] * surrounding_points[i];
  double scale = 0;
  for (unsigned int i = 0; i < surrounding_points.size(); ++i)
    for (unsigned int j = i + 1; j < surrounding_points.size(); ++j)
      scale =
        std::max(scale, surrounding_points[i].distance(surrounding_points[j]));
  // The closest point of the curve to the average is near the right
  // point, and the average lies within a sagitta of the curve; a candidate
  // on the other branch is far from it.
  const Point<3> closest = OpenCASCADE::closest_point(shape, average, tolerance);
  const double   sagitta = closest.distance(average);
  if (candidate.distance(closest) <= 0.05 * scale + 2. * sagitta)
    return candidate;
  ++checked_arclength_corrections;
  return closest;
}


Point<3>
CheckedArclengthProjectionLineManifold::get_intermediate_point(
  const Point<3> &p1,
  const Point<3> &p2,
  const double    w) const
{
  const std::array<Point<3>, 2> points  = {{p1, p2}};
  const std::array<double, 2>   weights = {{1. - w, w}};
  return checked(
    OpenCASCADE::ArclengthProjectionLineManifold<2, 3>::get_intermediate_point(
      p1, p2, w),
    make_array_view(points),
    make_array_view(weights));
}


Point<3>
CheckedArclengthProjectionLineManifold::get_new_point(
  const ArrayView<const Point<3>> &surrounding_points,
  const ArrayView<const double>   &weights) const
{
  return checked(
    OpenCASCADE::ArclengthProjectionLineManifold<2, 3>::get_new_point(
      surrounding_points, weights),
    surrounding_points,
    weights);
}


void
CheckedArclengthProjectionLineManifold::get_new_points(
  const ArrayView<const Point<3>> &surrounding_points,
  const Table<2, double>          &weights,
  ArrayView<Point<3>>              new_points) const
{
  OpenCASCADE::ArclengthProjectionLineManifold<2, 3>::get_new_points(
    surrounding_points, weights, new_points);
  std::vector<double> row(surrounding_points.size());
  for (unsigned int k = 0; k < new_points.size(); ++k)
    {
      for (unsigned int i = 0; i < row.size(); ++i)
        row[i] = weights(k, i);
      new_points[k] =
        checked(new_points[k], surrounding_points, make_array_view(row));
    }
}


// @sect4{ComputationalDomain::refine_and_resize}

// This function globally refines the
// mesh, distributes degrees of
// freedom, and resizes matrices and
// vectors.

template <>
void
ComputationalDomain<2>::refine_and_resize(const unsigned int refinement_level)
{
  pcout << "Refining and resizing mesh as required" << std::endl;
  tria.refine_global(refinement_level);
  pcout << "We have a tria of " << tria.n_active_cells() << " cells."
        << std::endl;
  GridTools::partition_triangulation(n_mpi_processes, tria);
  std::string   filename0 = ("meshResult.inp");
  std::ofstream logfile0(filename0.c_str());
  GridOut       grid_out0;
  grid_out0.write_ucd(tria, logfile0);
  pcout << "...done refining and resizing mesh" << std::endl;
}


template <int dim>
void
ComputationalDomain<dim>::refine_and_resize(const unsigned int refinement_level)
{
  pcout << "Refining and resizing mesh as required" << std::endl;


  double max_tol = 0;
  if (use_cad_surface_and_curves)
    {
      pcout << "Color Files" << endl;
      unsigned int ii    = 1;
      bool         go_on = true;
      while (go_on == true)
        {
          std::string color_filename = (input_cad_path + "Color_" +
                                        Utilities::int_to_string(ii) + ".iges");
          ifstream    f(color_filename);
          if (f.good())
            {
              pcout << ii << "-th file exists" << endl;
              TopoDS_Shape surface =
                OpenCASCADE::read_IGES(color_filename, 1e-3);
              cad_surfaces.push_back(surface);
            }
          else
            go_on = false;
          ii++;
        }

      pcout << "Edge Files" << endl;
      ii    = 1;
      go_on = true;
      while (go_on == true)
        {
          std::string edge_filename = (input_cad_path + "Curve_" +
                                       Utilities::int_to_string(ii) + ".iges");
          ifstream    f(edge_filename);
          if (f.good())
            {
              pcout << ii << "-th file exists" << endl;
              TopoDS_Shape curve = OpenCASCADE::read_IGES(edge_filename, 1e-3);
              cad_curves.push_back(curve);
            }
          else
            go_on = false;
          ii++;
        }


      for (unsigned int i = 0; i < cad_surfaces.size(); ++i)
        {
          pcout << i << endl;
          max_tol =
            fmax(max_tol, OpenCASCADE::get_shape_tolerance(cad_surfaces[i]));
          pcout << max_tol << endl;
        }
      for (unsigned int i = 0; i < cad_curves.size(); ++i)
        {
          pcout << i + cad_surfaces.size() << endl;
          max_tol =
            fmax(max_tol, OpenCASCADE::get_shape_tolerance(cad_curves[i]));
          pcout << max_tol << endl;
        }

      const double tolerance = cad_to_projectors_tolerance_ratio * max_tol;



      pcout << "Used tolerance is: " << tolerance << endl;
      for (unsigned int i = 0; i < cad_surfaces.size(); ++i)
        {
          normal_to_mesh_projectors.push_back(
            std::make_shared<CachedNormalToMeshProjectionManifold>(
              cad_surfaces[i], tolerance));
        }
      // static OpenCASCADE::DirectionalProjectionBoundary<2,3>
      //        directional_projector_lat(cad_surfaces[0],
      //        Point<3>(0.0,1.0,0.0), tolerance);
      // static OpenCASCADE::NormalProjectionBoundary<2,3>
      //        normal_projector_lat(cad_surfaces[0], tolerance);

      for (unsigned int i = 0; i < cad_curves.size(); ++i)
        {
          line_projectors.push_back(
            std::make_shared<CheckedArclengthProjectionLineManifold>(
              cad_curves[i], tolerance));
        }
      /////////////////////////////////////////
      // these lines are placed to fix a problem that deal developers have
      // created imposing that for each cell/face with a manifold_id,
      // a manifold must be prescribed.

      // in addition now if I set the manifold_id of a cell, its faces
      // won't inherit that automatically.so we need to fix that mess too
      Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
      Triangulation<2, 3>::active_cell_iterator endc = tria.end();
      std::set<unsigned int> detected_manifold_ids;
      for (; cell != endc; ++cell)
      {
        if (cell->manifold_id() != numbers::flat_manifold_id)
        {
          detected_manifold_ids.insert(cell->manifold_id());
          for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
            if (cell->face(f)->manifold_id() == numbers::flat_manifold_id)
              cell->face(f)->set_manifold_id(cell->manifold_id());
        }
        for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
          if (cell->face(f)->manifold_id() != numbers::flat_manifold_id)
            detected_manifold_ids.insert(cell->face(f)->manifold_id());
      }

      FlatManifold<2, 3> flat_manifold_default;
      for (std::set<unsigned int>::iterator it = detected_manifold_ids.begin();
           it != detected_manifold_ids.end();
           it++)
      {
        pcout << "Manifold_id detected: " << *it << std::endl;
        tria.set_manifold(*it, flat_manifold_default);
      }
      ////////////////////////////////////////
      
      for (unsigned int i = 0; i < cad_surfaces.size(); ++i)
        {
          tria.set_manifold(1 + i, *normal_to_mesh_projectors[i]);
        }

      for (unsigned int i = 0; i < cad_curves.size(); ++i)
        {
          tria.set_manifold(11 + i, *line_projectors[i]);
        }
    }


  unsigned int refinedCellCounter = 1;
  unsigned int cycles_counter     = 0;
  // we repeat the aspect ratio refinement cycle until no cell has been
  // flagged for refinement, or until we reach a maximum of 10 cycles
  while ((refinedCellCounter) && (cycles_counter < 10))
    {
      // the refined cells counter is zeroed at the start of each cycle
      refinedCellCounter = 0;
      // we loop on the all the triangulation active cells
      Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
      Triangulation<2, 3>::active_cell_iterator endc = tria.end();
      for (; cell != endc; ++cell)
        {
          // the following lines determine if the cell is more elongated
          // in its 0 or 1 direction
          unsigned int max_extent_dim = 0;
          unsigned int min_extent_dim = 1;
          if (cell->extent_in_direction(0) < cell->extent_in_direction(1))
            {
              max_extent_dim = 1;
              min_extent_dim = 0;
            }
          // we compute the extent of the cell in its maximum and minimum
          // elongation direction respectively
          double min_extent = cell->extent_in_direction(min_extent_dim);
          double max_extent = cell->extent_in_direction(max_extent_dim);
          // if the aspect ratio exceeds the prescribed maximum value, the cell
          // is refined
          if (max_extent > max_element_aspect_ratio * min_extent)
            {
              cell->set_refine_flag(
                RefinementCase<2>::cut_axis(max_extent_dim));
              refinedCellCounter++;
            }
        }
      // the number of cells refined in this cycle is reported before
      // proceeding with the next one
      pcout << "Aspect Ratio Reduction Cycle: " << cycles_counter << " ("
            << refinedCellCounter << ")" << endl;
      tria.execute_coarsening_and_refinement();

      // the following commented lines are here for debug puroposes: if
      // something fails during the aspect ratio reduction cycles, they
      // should be uncommented, so that a mesh file per cycle can be
      // produced to document the evolution of the mesh through the
      // refinements. If the make_edges_conformal() function
      // is suspect in creating some error, the lines can also be
      // moved after the make_edges_conformal() function is called

      // std::string filename = ( "meshIntermediateResult_" +
      //         Utilities::int_to_string(int(round(cycles_counter))) +
      //         ".inp" );
      // std::ofstream logfile(filename.c_str());
      // GridOut grid_out;
      // grid_out.write_ucd(tria, logfile);

      make_edges_conformal();
      cycles_counter++;
    }

  // the following refinement cycle is based upon the original CAD
  // geometry curvature. For this reason it can be activated not only
  // when the user requires it with the surface_curvature_refinement
  // option in the input file. Of course, this is possible only if
  // also the use_cad_surface_and_curves flag is set to thrue through
  // the input file. Only in this way in fact, CAD surfaces and curves
  // are prescribed for the triangulation refinements on some of
  // its manifold ids.
  if (use_cad_surface_and_curves && surface_curvature_refinement)
    {
      const double tolerance = cad_to_projectors_tolerance_ratio * max_tol;
      refinedCellCounter     = 1;
      cycles_counter         = 0;
      // the refinement procedure is recursively repeated until no more cells
      // are flagged for refinement, or until the user specified maximum number
      // of curvature refinement cycles is reached
      while ((refinedCellCounter) &&
             (cycles_counter < max_curvature_ref_cycles))
        {
          // the refined cells counter is zeroed at the start of each cycle
          refinedCellCounter = 0;
          // we loop on the all the triangulation active cells
          Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
          Triangulation<2, 3>::active_cell_iterator endc = tria.end();
          for (; cell != endc; ++cell)
            {
              if (anisotropic_curvature_refinement)
                {
                  // The angle between the normals of the CAD surface at the
                  // centres of the opposite faces of the cell in each of its
                  // directions; the cell is cut across those in which it
                  // exceeds a cells_per_circle fraction of the circle.
                  if (int(cell->material_id()) - 1 >= (int)cad_surfaces.size())
                    continue;
                  const TopoDS_Shape &shape =
                    cad_surfaces[int(cell->material_id()) - 1];
                  bool cut[2] = {false, false};
                  for (unsigned int a = 0; a < 2; ++a)
                    {
                      const auto lower =
                        OpenCASCADE::closest_point_and_differential_forms(
                          shape, cell->face(2 * a)->center(), tolerance);
                      const auto upper =
                        OpenCASCADE::closest_point_and_differential_forms(
                          shape, cell->face(2 * a + 1)->center(), tolerance);
                      const double cosine =
                        std::get<1>(lower) * std::get<1>(upper) /
                        (std::get<1>(lower).norm() * std::get<1>(upper).norm());
                      const double angle =
                        std::acos(std::max(-1., std::min(1., cosine)));
                      cut[a] = (angle > 2 * dealii::numbers::PI /
                                          cells_per_circle) &&
                               (cell->extent_in_direction(a) > 10 * tolerance);
                    }
                  if (cut[0] && cut[1])
                    cell->set_refine_flag();
                  else if (cut[0] || cut[1])
                    cell->set_refine_flag(
                      RefinementCase<2>::cut_axis(cut[0] ? 0 : 1));
                  if (cut[0] || cut[1])
                    refinedCellCounter++;
                  continue;
                }
              // In the following lines, we try to come up with an estimation
              // of the cell normal. It is obtained from the average of the
              // normal to the 4 triangles in which the cell can be split using
              // the vertices and the center. The commented lines can be used
              // for checks in case something goes wrong.

              // cout<<"center: "<<cell->center()<<endl;
              // cout<<"v0: "<<cell->vertex(0)<<endl;
              // cout<<"v1: "<<cell->vertex(1)<<endl;
              // cout<<"v2: "<<cell->vertex(2)<<endl;
              // cout<<"v3: "<<cell->vertex(3)<<endl;
              Point<3> t0 = cell->vertex(0) + (-1.0) * cell->center();
              Point<3> t1 = cell->vertex(1) + (-1.0) * cell->center();
              Point<3> t2 = cell->vertex(2) + (-1.0) * cell->center();
              Point<3> t3 = cell->vertex(3) + (-1.0) * cell->center();
              // cout<<"t0: "<<t0<<endl;
              // cout<<"t1: "<<t1<<endl;
              // cout<<"t2: "<<t2<<endl;
              // cout<<"t3: "<<t3<<endl;

              Point<3> nn0(t0(1) * t1(2) - t0(2) * t1(1),
                           t0(2) * t1(0) - t0(0) * t1(2),
                           t0(0) * t1(1) - t0(1) * t1(0));
              nn0 /= nn0.norm();
              Point<3> nn1(t1(1) * t3(2) - t1(2) * t3(1),
                           t1(2) * t3(0) - t1(0) * t3(2),
                           t1(0) * t3(1) - t1(1) * t3(0));
              nn1 /= nn1.norm();
              Point<3> nn2(t3(1) * t2(2) - t3(2) * t2(1),
                           t3(2) * t2(0) - t3(0) * t2(2),
                           t3(0) * t2(1) - t3(1) * t2(0));
              nn2 /= nn2.norm();
              Point<3> nn3(t2(1) * t0(2) - t2(2) * t0(1),
                           t2(2) * t0(0) - t2(0) * t0(2),
                           t2(0) * t0(1) - t2(1) * t0(0));
              nn3 /= nn3.norm();
              Point<3> n = (nn0 + nn1 + nn2 + nn3) / 4.0;
              n /= n.norm();
              // cout<<cell<<endl;
              // cout<<nn0<<endl;
              // cout<<nn1<<endl;
              // cout<<nn2<<endl;
              // cout<<nn3<<endl;
              // cout<<n<<endl;
              // cout<<cell<<"  material id: "<<int(cell->material_id())<<endl;

              // once the cell normal has beed created, we want to use it as the
              // direction of the projection onto the CAD surface
              // first though, let's check that we are using a CAD surface for
              // the refinement of the manifold_id associated with the present
              // cell
              double cell_size;
              if (int(cell->material_id()) - 1 < (int)cad_surfaces.size())
                {
                  // if so, the cad_surface associated with the present
                  // manifold_id is identified...
                  TopoDS_Shape neededShape =
                    cad_surfaces[int(cell->material_id()) - 1];
                  // ...and used to set up a line intersection to project the
                  // cell center on the CAD surface along the direction
                  // specified by the previously computed cell normal
                  Point<3> projection = OpenCASCADE::line_intersection(
                    neededShape, cell->center(), n, tolerance);
                  // in correspondence with the projected point, we ask all the
                  // surface differential forms
                  auto tup = OpenCASCADE::closest_point_and_differential_forms(
                    neededShape, projection, tolerance);
                  // among the differential point, we select the maximum
                  // absolute curvature
                  double max_abs_curv =
                    fmax(fabs(std::get<2>(tup)), fabs(std::get<3>(tup)));
                  // this commented line is just for debug purposes
                  // cout<<"Point: "<<std::get<0>(tup)<<"  Kmin:
                  // "<<std_cxx11::get<2>(tup)<<"  Kmax:
                  // "<<std_cxx11::get<3>(tup)<<endl; the minimum curvature
                  // radius is computed from the maximum absolute curvatur
                  double curvature_radius = 1.0 / fmax(max_abs_curv, tolerance);
                  // the target cell size is selected so that it corresponds to
                  // a cells_per_circle fraction of the circumference
                  // corresponding to the minimum curvature radius
                  cell_size = 2 * dealii::numbers::PI / cells_per_circle *
                              curvature_radius;
                }
              else
                {
                  // if the cell manifold_id is not associated to a CAD surface,
                  // the target cell_size is set to and extremely high value, so
                  // that the cell is never refined
                  cell_size =
                    2 * dealii::numbers::PI / cells_per_circle / tolerance;
                }

              // the following line si for debug puropses and should be
              // uncommented if something is not working with the refinement
              // cout<<"Cell Diam: "<<cell->diameter()<<"  Target Cell Size:
              // "<<cell_size<<endl;


              // if the cell diameter is higher than the target cell size, the
              // refinement flag is set (unless the cell is already very small
              // ---which for us means 10xtolerance)
              if ((cell->diameter() > cell_size) &&
                  (cell->diameter() > 10 * tolerance))
                {
                  cell->set_refine_flag();
                  refinedCellCounter++;
                }
            }
          // the number of cells to be refined in this cycle is reported, the
          // refinement is carried out and the make_edges_conformal function is
          // called to check no edge presents non comformities
          pcout << "Curvature Based Local Refinement Cycle: " << cycles_counter
                << " (" << refinedCellCounter << ")" << endl;
          tria.execute_coarsening_and_refinement();
          make_edges_conformal();
          cycles_counter++;

          // std::string filename = ( "DTMB_II_meshResult_max_curv" +
          //                         Utilities::int_to_string(int(round(cycles_counter)))
          //                         +
          //                         ".vtk" );
          // std::ofstream logfile(filename.c_str());
          // GridOut grid_out;
          // grid_out.write_vtk(tria, logfile);
          // std::string stl_filename = ( "DTMB_II_meshResult_max_curv" +
          //                         Utilities::int_to_string(int(round(cycles_counter)))
          //                         +
          //                         ".stl" );
          // SaveSTL(tria,stl_filename);
        }

      if (curvature_refinement_grading > 0)
        {
          // Two faces of cells of one material are neighbours when they lie
          // on one line and overlap; a cell is cut across a face when it is
          // longer across it than the ratio times the neighbour across the
          // matching face. The coarse
          // mesh has a few hundred cells, so the pairs are simply all
          // compared.
          for (unsigned int pass = 0; pass < 20; ++pass)
            {
              std::vector<Triangulation<2, 3>::active_cell_iterator> cells;
              for (const auto &cell : tria.active_cell_iterators())
                cells.push_back(cell);
              unsigned int n_cut = 0;
              for (const auto &cell : cells)
                {
                  bool cut[2] = {false, false};
                  for (unsigned int f = 0; f < 4; ++f)
                    {
                      const Point<3>     a      = cell->face(f)->vertex(0);
                      const Point<3>     b      = cell->face(f)->vertex(1);
                      const Tensor<1, 3> ab     = b - a;
                      const double       length = ab.norm();
                      const double tolerance_f = 1e-6 * length;
                      for (const auto &other : cells)
                        {
                          if (other == cell ||
                              other->material_id() != cell->material_id())
                            continue;
                          for (unsigned int g = 0; g < 4; ++g)
                            {
                              const Point<3> c = other->face(g)->vertex(0);
                              const Point<3> d = other->face(g)->vertex(1);
                              // Collinear: both ends of the other face on
                              // the line of this one.
                              const auto off_line = [&](const Point<3> &p) {
                                const Tensor<1, 3> ap = p - a;
                                return (ap - (ap * ab) / (length * length) * ab)
                                         .norm() > tolerance_f;
                              };
                              if (off_line(c) || off_line(d))
                                continue;
                              // Overlapping with a positive length.
                              const double tc = (c - a) * ab / (length * length);
                              const double td = (d - a) * ab / (length * length);
                              const double overlap =
                                std::min(1., std::max(tc, td)) -
                                std::max(0., std::min(tc, td));
                              if (overlap <= 1e-6)
                                continue;
                              if (cell->extent_in_direction(f / 2) >
                                  curvature_refinement_grading *
                                    other->extent_in_direction(g / 2) *
                                    (1 + 1e-6))
                                cut[f / 2] = true;
                            }
                        }
                    }
                  if (cut[0] && cut[1])
                    cell->set_refine_flag();
                  else if (cut[0] || cut[1])
                    cell->set_refine_flag(
                      RefinementCase<2>::cut_axis(cut[0] ? 0 : 1));
                  if (cut[0] || cut[1])
                    ++n_cut;
                }
              pcout << "Curvature refinement grading pass " << pass << " ("
                    << n_cut << ")" << endl;
              if (n_cut == 0)
                break;
              tria.execute_coarsening_and_refinement();
              make_edges_conformal();
            }
        }

      if (aspect_ratio_after_curvature_refinement)
        for (unsigned int cycle = 0; cycle < 10; ++cycle)
          {
            // As the aspect ratio reduction of the input grid above.
            unsigned int n_cut = 0;
            for (const auto &cell : tria.active_cell_iterators())
              {
                const unsigned int longer =
                  cell->extent_in_direction(0) < cell->extent_in_direction(1) ?
                    1 :
                    0;
                if (cell->extent_in_direction(longer) >
                    max_element_aspect_ratio *
                      cell->extent_in_direction(1 - longer))
                  {
                    cell->set_refine_flag(RefinementCase<2>::cut_axis(longer));
                    ++n_cut;
                  }
              }
            pcout << "Aspect ratio reduction after the curvature refinement, "
                     "cycle "
                  << cycle << " (" << n_cut << ")" << endl;
            if (n_cut == 0)
              break;
            tria.execute_coarsening_and_refinement();
            make_edges_conformal();
          }
    }
  //*/

  // A face with children is the edge of a hanging node: its cell is cut
  // across that face (faces 0 and 1 run along the second direction, 2 and 3
  // along the first), which moves the hanging nodes to the opposite face,
  // until they reach cells that are as fine.
  if (remove_hanging_nodes)
    for (unsigned int cycle = 0; cycle < 50; ++cycle)
      {
        unsigned int n_cut = 0;
        for (const auto &cell : tria.active_cell_iterators())
          {
            bool cut[2] = {false, false};
            for (const unsigned int f : cell->face_indices())
              if (!cell->at_boundary(f) && cell->face(f)->has_children())
                cut[f < 2 ? 1 : 0] = true;
            if (cut[0] || cut[1])
              {
                cell->set_refine_flag(
                  cut[0] && cut[1] ?
                    RefinementCase<2>(RefinementCase<2>::isotropic_refinement) :
                    RefinementCase<2>::cut_axis(cut[0] ? 0 : 1));
                ++n_cut;
              }
          }
        pcout << "Removing hanging nodes, cycle " << cycle << " (" << n_cut
              << ")" << endl;
        if (n_cut == 0)
          break;
        tria.execute_coarsening_and_refinement();
        make_edges_conformal();
      }

  if( input_grid_name == "../grids/sphere_box_flipped" ||
      input_grid_name == "../grids/sphere_box_flipped_double_nodes" ||
      input_grid_name == "../grids/sphere_box_flipped_new")
  {
    // These grids describe a unit sphere and planar box faces.  Analytical
    // manifolds make high-order mappings exact here and avoid repeated IGES
    // projections inside the adaptive quadrature.
    static const SphericalManifold<dim - 1, dim> sphere_manifold{Point<dim>()};
    static const FlatManifold<dim - 1, dim>      flat_manifold;
    tria.set_manifold(1, sphere_manifold);
    tria.set_manifold(2, flat_manifold);
    tria.set_manifold(3, flat_manifold);
    pcout << "Using analytical sphere-box manifolds" << std::endl;

    if (pre_global_refinements > 0)
    {
      pcout << "Applying " << pre_global_refinements
            << " global refinement cycle(s) before local gap refinement"
            << std::endl;
      tria.refine_global(pre_global_refinements);
      make_edges_conformal();
      pcout << "Cells after global refinement: "
            << tria.n_active_cells() << std::endl;
    }

    //////////  
    // if you want to refine around a point
    // first do one pass of global refinement
    //tria.refine_global(1);
    
    // refine only bottom face of the box around (0,0,-1.003)
    const Point<dim> refinement_center(0, 0, -1);
    for (unsigned int step = 0; step < (refinement_level > 0 ? 4 : 0); ++step)
    {
      Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
      Triangulation<2, 3>::active_cell_iterator endc = tria.end();
      for (; cell != endc; ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
        {
          const double distance_from_center = refinement_center.distance(cell->vertex(v));
          if((int(cell->material_id()) == 3) && (distance_from_center < 0.2))
            cell->set_refine_flag();
        }
      }
      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
      make_edges_conformal();
    }
    
    // refine everything around (0,0,-1.003)
    for (unsigned int step = 0; step < refinement_level; ++step)
    {
      Triangulation<2, 3>::active_cell_iterator cell = tria.begin_active();
      Triangulation<2, 3>::active_cell_iterator endc = tria.end();
      for (; cell != endc; ++cell)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell; ++v)
        {
          const double distance_from_center = refinement_center.distance(cell->vertex(v));
          if (distance_from_center < 0.1)
          {
            cell->set_refine_flag();
            break;
          }
        }
      }
      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
      make_edges_conformal();
    }

    //////////
  }
  else
  {
    tria.refine_global(refinement_level);
  }

  for (unsigned int cycle = 0; cycle < edge_refinement_cycles; ++cycle)
    {
      std::vector<std::pair<Point<dim>, Point<dim>>> edges;
      for (const auto &cell : tria.active_cell_iterators())
        for (const unsigned int f : cell->face_indices())
          if (cell->at_boundary(f))
            edges.emplace_back(cell->face(f)->vertex(0),
                               cell->face(f)->vertex(1));
      const auto distance_to_edges = [&](const Point<dim> &x) {
        double d = std::numeric_limits<double>::max();
        for (const auto &[a, b] : edges)
          {
            const Tensor<1, dim> ab = b - a;
            const double t = std::clamp((x - a) * ab / ab.norm_square(), 0., 1.);
            d = std::min(d, x.distance(a + t * ab));
          }
        return d;
      };
      unsigned int flagged = 0;
      for (const auto &cell : tria.active_cell_iterators())
        for (const unsigned int v : cell->vertex_indices())
          if (distance_to_edges(cell->vertex(v)) < 1e-10 * cell->diameter())
            {
              cell->set_refine_flag();
              ++flagged;
              break;
            }
      pcout << "Refinement towards the edges of the surface, cycle " << cycle
            << " (" << flagged << " of " << tria.n_active_cells()
            << " cells)" << std::endl;
      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
      make_edges_conformal();
    }
  
  if constexpr (dim == 3)
  for (unsigned int cycle = 0; cycle < curvature_refinement_cycles; ++cycle)
    {
      // The turning of an edge from the midpoint on its manifold: an arc of
      // chord L and sagitta s turns by 4 atan(2 s / L).
      const auto turning = [](const auto &line) {
        const Point<dim> a = line->vertex(0), b = line->vertex(1);
        const double     L = a.distance(b);
        if (L == 0.)
          return 0.;
        const double sagitta = line->center(true).distance(Point<dim>(0.5 * (a + b)));
        return 4. * std::atan(2. * sagitta / L);
      };
      const double limit    = curvature_refinement_angle * M_PI / 180.;
      unsigned int flagged  = 0;
      for (const auto &cell : tria.active_cell_iterators())
        {
          // Faces 2 and 3 run along the first reference direction, 0 and 1
          // along the second.
          const double along_x = std::max(turning(cell->face(2)),
                                          turning(cell->face(3)));
          const double along_y = std::max(turning(cell->face(0)),
                                          turning(cell->face(1)));
          const bool cut_x = along_x > limit, cut_y = along_y > limit;
          if (cut_x || cut_y)
            {
              cell->set_refine_flag(
                cut_x && cut_y ?
                  RefinementCase<2>(RefinementCase<2>::isotropic_refinement) :
                  RefinementCase<2>::cut_axis(cut_x ? 0 : 1));
              ++flagged;
            }
        }
      pcout << "Refinement by the curvature of the surface, cycle " << cycle
            << " (" << flagged << " of " << tria.n_active_cells()
            << " cells)" << std::endl;
      if (flagged == 0)
        break;
      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
      make_edges_conformal();
    }

  //else just do the global refinement
  //tria.refine_global(refinement_level);

  // Bring the selected material ids up to a finer resolution than the global
  // level. Conformity is deliberately not restored afterwards:
  // make_edges_conformal() propagates refinement across the double-node seams
  // and would drag the remaining surfaces along, which is exactly what this
  // option exists to avoid. Refining a set of materials that covers whole
  // connected components keeps the mesh conforming anyway, and where it does
  // not, the discontinuous trace space carries no hanging-node constraints and
  // the panel-pair integrator handles non-conforming edges.
  const auto refine_materials =
    [&](const std::set<types::material_id> &material_ids,
        const unsigned int                  cycles,
        const std::string                  &name) {
      if (cycles == 0 || material_ids.empty())
        return;
      for (unsigned int step = 0; step < cycles; ++step)
        {
          for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
            if (material_ids.count(cell->material_id()) > 0)
              cell->set_refine_flag();
          tria.prepare_coarsening_and_refinement();
          tria.execute_coarsening_and_refinement();
        }

      std::map<types::material_id, unsigned int> cells_per_material;
      for (const auto &cell : tria.active_cell_iterators())
        ++cells_per_material[cell->material_id()];
      pcout << name << " refinement: " << cycles
            << " cycle(s) on material ids";
      for (const auto id : material_ids)
        pcout << " " << static_cast<unsigned int>(id);
      pcout << "; cells per material:";
      for (const auto &entry : cells_per_material)
        pcout << " " << static_cast<unsigned int>(entry.first) << "="
              << entry.second;
      pcout << std::endl;
    };
  refine_materials(extra_refinement_material_ids,
                   extra_refinement_cycles,
                   "Extra");
  refine_materials(additional_refinement_material_ids,
                   additional_refinement_cycles,
                   "Additional");

  // Graded refinement. The flux on a face looking at a nearby body varies on a
  // scale set by the distance from the point of closest approach, so a target
  // size proportional to that distance equidistributes the approximation error
  // and leaves the far field, where the flux is flat and small, coarse.
  //
  // The level difference is limited to one: the panel-pair integrator handles a
  // non-conforming edge by pairing a coarse cell with the children of its
  // neighbour, which is only well defined across a single level.
  if (graded_refinement_maximum_cycles > 0 &&
      !graded_refinement_material_ids.empty())
    {
      // Limit levels only during these local refinement passes. Leaving
      // this flag enabled makes a later seam-conformity pass equalize the
      // anisotropic refinement levels of unrelated surface directions,
      // which can turn a local NACA refinement into a global cascade.
      const auto previous_smoothing = tria.get_mesh_smoothing();
      bool isotropic_history = true;
      for (const auto &cell : tria.cell_iterators())
        if (cell->has_children() &&
            cell->refinement_case() != RefinementCase<dim - 1>::isotropic_refinement)
          isotropic_history = false;
      if (isotropic_history)
        tria.set_mesh_smoothing(
          Triangulation<dim - 1, dim>::limit_level_difference_at_vertices);
      // Anisotropic levels count cuts in different directions, not a
      // uniform physical size. Equalizing them would refine unselected
      // materials and erase the locality of the requested grading.

      unsigned int cycle = 0;
      for (; cycle < graded_refinement_maximum_cycles; ++cycle)
        {
          unsigned int flagged = 0;
          for (const auto &cell : tria.active_cell_iterators())
            if (graded_refinement_material_ids.count(cell->material_id()) > 0)
              {
                const double distance =
                  graded_refinement_focus.distance(cell->center());
                const double target = std::max(
                  graded_refinement_minimum_size,
                  graded_refinement_size_ratio *
                    std::pow(distance, graded_refinement_growth_exponent));
                if (cell->diameter() > target)
                  {
                    cell->set_refine_flag();
                    ++flagged;
                  }
              }
          if (flagged == 0)
            break;
          tria.prepare_coarsening_and_refinement();
          tria.execute_coarsening_and_refinement();
        }

      double smallest = std::numeric_limits<double>::max();
      double largest  = 0.0;
      std::map<types::material_id, unsigned int> graded_cells;
      for (const auto &cell : tria.active_cell_iterators())
        {
          ++graded_cells[cell->material_id()];
          if (graded_refinement_material_ids.count(cell->material_id()) > 0)
            {
              smallest = std::min(smallest, cell->diameter());
              largest  = std::max(largest, cell->diameter());
            }
        }
      pcout << "Graded refinement: " << cycle << " cycle(s), ratio "
            << graded_refinement_size_ratio << ", exponent "
            << graded_refinement_growth_exponent << ", floor "
            << graded_refinement_minimum_size << ", focus "
            << graded_refinement_focus << std::endl
            << "  graded cell size from " << smallest << " to " << largest
            << "; cells per material:";
      for (const auto &entry : graded_cells)
        pcout << " " << static_cast<unsigned int>(entry.first) << "="
              << entry.second;
      pcout << std::endl;
      tria.set_mesh_smoothing(previous_smoothing);
    }

  // Exponential refinement, with the level difference limited to one as for
  // the graded refinement.
  if (exponential_refinement_maximum_cycles > 0 &&
      !exponential_refinement_decay_vector.empty())
    {
      // Limit levels only during these local refinement passes. Leaving
      // this flag enabled makes a later seam-conformity pass equalize the
      // anisotropic refinement levels of unrelated surface directions,
      // which can turn a local NACA refinement into a global cascade.
      const auto previous_smoothing = tria.get_mesh_smoothing();
      bool isotropic_history = true;
      for (const auto &cell : tria.cell_iterators())
        if (cell->has_children() &&
            cell->refinement_case() != RefinementCase<dim - 1>::isotropic_refinement)
          isotropic_history = false;
      if (isotropic_history)
        tria.set_mesh_smoothing(
          Triangulation<dim - 1, dim>::limit_level_difference_at_vertices);
      // Anisotropic levels count cuts in different directions, not a
      // uniform physical size. Equalizing them would refine unselected
      // materials and erase the locality of the requested grading.
      Tensor<1, dim> k;
      for (unsigned int d = 0; d < dim; ++d)
        k[d] = exponential_refinement_decay_vector[d];
      // k.x*: the smallest k.x over the vertices, where e^{-k.x} is largest.
      double k_x_star = std::numeric_limits<double>::max();
      for (const auto &v : tria.get_vertices())
        k_x_star = std::min(k_x_star, k * v);
      const auto target = [&](const auto &cell) {
        double k_x = std::numeric_limits<double>::max();
        for (const unsigned int v : cell->vertex_indices())
          k_x = std::min(k_x, k * cell->vertex(v));
        return exponential_refinement_minimum_size *
               std::exp(exponential_refinement_grading * (k_x - k_x_star));
      };

      unsigned int cycle = 0;
      for (; cycle < exponential_refinement_maximum_cycles; ++cycle)
        {
          unsigned int flagged = 0;
          for (const auto &cell : tria.active_cell_iterators())
            if (cell->diameter() > target(cell))
              {
                cell->set_refine_flag();
                ++flagged;
              }
          if (flagged == 0)
            break;
          tria.prepare_coarsening_and_refinement();
          tria.execute_coarsening_and_refinement();
        }

      double smallest = std::numeric_limits<double>::max();
      std::map<types::material_id, unsigned int> cells_per_material;
      for (const auto &cell : tria.active_cell_iterators())
        {
          ++cells_per_material[cell->material_id()];
          smallest = std::min(smallest, cell->diameter());
        }
      pcout << "Exponential refinement: " << cycle << " cycle(s), decay "
            << k << ", minimum size "
            << exponential_refinement_minimum_size << ", grading "
            << exponential_refinement_grading << "; smallest cell "
            << smallest << "; cells per material:";
      for (const auto &entry : cells_per_material)
        pcout << " " << static_cast<unsigned int>(entry.first) << "="
              << entry.second;
      pcout << std::endl;
      tria.set_mesh_smoothing(previous_smoothing);
    }

  if (planar_smoothing_iterations > 0 && !planar_smoothing_material_ids.empty())
    {
      const auto quality = PlanarMeshQuality::smooth(
        tria, planar_smoothing_material_ids, planar_smoothing_iterations);
      pcout << "Planar mesh quality: " << quality.accepted_moves
            << " accepted vertex moves; minimum corner mean ratio "
            << quality.minimum_before << " -> " << quality.minimum_after
            << ", mean cell quality " << quality.mean_before
            << " -> " << quality.mean_after << std::endl;
    }

  if (CheckedArclengthProjectionLineManifold::n_corrected() > 0)
    pcout << "CAD curves: "
          << CheckedArclengthProjectionLineManifold::n_corrected()
          << " new points of the arclength projection on the wrong branch, "
             "replaced by the closest point of the curve"
          << std::endl;
  pcout << "We have a tria of " << tria.n_active_cells() << " cells."
        << std::endl;
  GridTools::partition_triangulation(n_mpi_processes, tria);
  std::string   filename0 = ("meshResult.inp");
  std::ofstream logfile0(filename0.c_str());
  GridOut       grid_out0;
  grid_out0.write_ucd(tria, logfile0);


  pcout << "...done refining and resizing mesh" << std::endl;
}


template <int dim>
void
ComputationalDomain<dim>::conditional_refine_and_resize(
  const unsigned int refinement_level)
{
  pcout << "Conditionally refining and resizing mesh as required" << std::endl;

  const Point<dim> center(0, 0, -1);
  compute_double_vertex_cache();
  make_edges_conformal();

  for (unsigned int step = 0; step < refinement_level; ++step)
    {
      auto cell = tria.begin_active();
      auto endc = tria.end();
      for (; cell != endc; ++cell)
        {
          for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell;
               ++v)
            {
              const double distance_from_center =
                center.distance(cell->vertex(v));
              if (std::fabs(distance_from_center) < 0.1)
                {
                  cell->set_refine_flag();
                  break;
                }
            }
        }
      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
      // compute_double_vertex_cache();
      make_edges_conformal();
    }
  update_triangulation();
}

template <int dim>
void
ComputationalDomain<dim>::update_triangulation(const bool isotropic_conformity)
{
  // compute_double_vertex_cache();
  // After adaptive refinement (isotropic_conformity) the cell across a
  // double-node edge from a refined cell is refined isotropically, not only
  // across the edge: at a thin trailing edge the two cells of a folded pair
  // then keep the same depth.
  make_edges_conformal(true, isotropic_conformity);
  // tria.execute_coarsening_and_refinement ();

  pcout << "We have a tria of " << tria.n_active_cells() << " cells."
        << std::endl;
  GridTools::partition_triangulation(n_mpi_processes, tria);
  std::string   filename0 = ("meshResult.inp");
  std::ofstream logfile0(filename0.c_str());
  GridOut       grid_out0;
  grid_out0.write_ucd(tria, logfile0);

  std::ostringstream filename;
  filename << "mesh.vtu";
  std::ofstream output(filename.str().c_str());

  FE_Q<dim - 1, dim> fe_dummy(1);

  DoFHandler<dim - 1, dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_dummy);
  DataOut<dim - 1, dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<unsigned int> partition_int(tria.n_active_cells());
  GridTools::get_subdomain_association(tria, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning,
                           "partitioning",
                           DataOut<dim - 1, dim>::type_cell_data);
  data_out.build_patches();
  data_out.write_vtu(output);


  pcout << "...done refining and resizing mesh" << std::endl;
}

template <>
void
ComputationalDomain<2>::make_edges_conformal(
  const bool with_double_nodes,
  const bool isotropic_ref_on_opposite_side)
{
  if (with_double_nodes || !with_double_nodes ||
      isotropic_ref_on_opposite_side || !isotropic_ref_on_opposite_side)
    AssertThrow(true, ExcMessage("Make  edges conformal only works in 3D"));
}

template <int dim>
void
ComputationalDomain<dim>::make_edges_conformal(
  const bool with_double_nodes,
  const bool isotropic_ref_on_opposite_side)
{
  if (with_double_nodes == false)
    {
      auto cell = tria.begin_active();
      auto endc = tria.end();

      for (cell = tria.begin_active(); cell != endc; ++cell)
        {
          for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
            if (
              cell->face(f)
                ->at_boundary()) // material_id()!=numbers::invalid_material_id)//dovrei
                                 // essere su un buondary
              {
                // TriaIterator<CellAccessor<dim-1,dim> > cell_neigh =
                // cell->neighbor(f);
                if (cell->neighbor_is_coarser(f))
                  {
                    TriaIterator<CellAccessor<dim - 1, dim>> cell_neigh =
                      cell->neighbor(f);
                    cell_neigh->set_refine_flag(
                      RefinementCase<dim - 1>::isotropic_refinement);
                    // std::cout<<"mammina..."<<std::endl;
                  }
              }
        }
      tria.prepare_coarsening_and_refinement();
      tria.execute_coarsening_and_refinement();
    }
  else
    {
      pcout << "Restoring mesh conformity on edges..." << std::endl;
      pcout << "cells before : " << tria.n_active_cells() << std::endl;
      // pcout<<"dofs before: "<<dhh.n_dofs()<<std::endl;
      bool to_restore = true;

      while (to_restore)
        {
          compute_double_vertex_cache();
          types::global_dof_index n_vertex     = tria.n_vertices();
          auto                    all_vertices = tria.get_vertices();

          double tol = 1e-7;

          to_restore = false;
          pcout << to_restore << std::endl;
          for (types::global_dof_index i = 0; i < n_vertex; ++i)
            {
              if (vertex_on_boundary[i] == true &&
                  double_vertex_vector[i].size() == 1)
                {
                  std::vector<Point<dim>> nodes(
                    GeometryInfo<dim - 1>::vertices_per_face);
                  for (types::global_dof_index kk = 0;
                       kk < vert_to_elems[i].size();
                       ++kk) // ogni faccia ha due estremi
                    {
                      auto cell =
                        vert_to_elems[i][kk]; // mi riconduco alla cella con il
                                              // nodo non conforme
                      for (unsigned int f = 0;
                           f < GeometryInfo<dim - 1>::faces_per_cell;
                           ++f)
                        {
                          if (cell->face(f)
                                ->at_boundary()) // ritrovo la faccia con l'edge
                                                 // non doppio
                            {
                              // std::cout<<cell->face(f)->vertex(1)<<"
                              // "<<cell->face(f)->vertex(0)<<std::endl;
                              if (all_vertices[i].distance(
                                    cell->face(f)->vertex(0)) < tol)
                                nodes[kk] = cell->face(f)->vertex(1);
                              else if (all_vertices[i].distance(
                                         cell->face(f)->vertex(1)) < tol)
                                nodes[kk] = cell->face(f)->vertex(0);
                            }
                        }
                      // std::cout<<std::endl;
                    }
                  // std::cout<<nodes[0]<<"    "<<ref_points[i]<<"
                  // "<<nodes[1]<<std::endl;
                  // we can now compute the center of the parent cell face
                  Point<3> parent_face_center = 0.5 * (nodes[0] + nodes[1]);
                  for (auto jt = edge_cells.begin(); jt != edge_cells.end();
                       ++jt)
                    for (unsigned int d = 0;
                         d < GeometryInfo<2>::faces_per_cell;
                         ++d)
                      if ((*jt)->face(d)->at_boundary())
                        {
                          // cout<<parent_face_center.distance((*jt)->face(d)->center())<<"
                          // "<<tol<<endl;
                          if (parent_face_center.distance(
                                ((*jt)->face(d)->vertex(0) +
                                 (*jt)->face(d)->vertex(1)) /
                                2) < tol)
                            {
                              if (isotropic_ref_on_opposite_side)
                                {
                                  (*jt)->set_refine_flag();
                                  to_restore = true;
                                }
                              // otherwise, use anisotropic refinement to make
                              // edge mesh conformal
                              else
                                {
                                  if ((d == 0) || (d == 1))
                                    (*jt)->set_refine_flag(
                                      RefinementCase<2>::cut_axis(1));
                                  else
                                    (*jt)->set_refine_flag(
                                      RefinementCase<2>::cut_axis(0));
                                  to_restore = true;
                                }
                            }
                        }
                }
            }

          if (to_restore)
            {
              tria.prepare_coarsening_and_refinement();
              tria.execute_coarsening_and_refinement();
              pcout << "found non conformity, new cell number : "
                    << tria.n_active_cells() << std::endl;
            }
          // pcout<<"pippo"<<std::endl;
        }
      pcout << "cells after : " << tria.n_active_cells() << std::endl;
      pcout << "...Done restoring mesh conformity" << std::endl;
    }
}

template <int dim>
void
ComputationalDomain<dim>::compute_double_vertex_cache()
{
  pcout << "Computing infos for the double_vertex" << std::endl;
  double toll = 1e-7;
  double_vertex_vector.clear();
  types::global_dof_index n_vertex = tria.n_vertices();
  double_vertex_vector.resize(n_vertex);
  vertex_on_boundary.resize(n_vertex);
  std::fill(vertex_on_boundary.begin(), vertex_on_boundary.end(), false);

  auto all_vertices = tria.get_vertices();

  for (types::global_dof_index i = 0; i < n_vertex; ++i)
    {
      for (types::global_dof_index j = 0; j < n_vertex; ++j)
        {
          if (all_vertices[i].distance(all_vertices[j]) <= toll)
            {
              double_vertex_vector[i].push_back(j);
            }
        }
    }



  auto cell = tria.begin_active();
  auto endc = tria.end();
  vert_to_elems.clear();
  edge_cells.clear();

  for (cell = tria.begin_active(); cell != endc; ++cell)
    {
      std::vector<Point<dim>> cell_vertices(
        GeometryInfo<dim - 1>::vertices_per_cell);

      for (unsigned int v = 0; v < GeometryInfo<dim - 1>::vertices_per_cell;
           ++v)
        {
          vert_to_elems[cell->vertex_index(v)].push_back(cell);
          cell_vertices[v] = cell->vertex(v);
        }


      if (cell->at_boundary())
        {
          edge_cells.insert(cell);

          for (unsigned int f = 0; f < GeometryInfo<dim - 1>::faces_per_cell;
               ++f)
            {
              if (cell->face(f)->at_boundary())
                {
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim - 1>::vertices_per_cell;
                       ++v)
                    {
                      if (cell->face(f)->vertex(0) == cell_vertices[v])
                        vertex_on_boundary[cell->vertex_index(v)] = true;
                      else if (cell->face(f)->vertex(1) == cell_vertices[v])
                        vertex_on_boundary[cell->vertex_index(v)] = true;
                    }
                }
            }
        }
    }
  pcout << "done double_vertex cache" << std::endl;
}



template class ComputationalDomain<2>;
template class ComputationalDomain<3>;
