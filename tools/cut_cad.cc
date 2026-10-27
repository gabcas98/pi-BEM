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

// Cut the CAD model of a hull at its waterline, for ComputationalDomain and
// the exact geometry: the part of every Color_i.iges of the input folder
// below the plane z = z0 goes to Color_i.iges of the output folder, so that
// the waterline is a boundary of the faces, which the exact maps follow
// (OpenCASCADE::ShapeProjector maps the cells along the trimming curves of
// the faces onto them). The curves Curve_j.iges are copied, except those
// given as j=i, which become the waterline of the cut Color_i: its edges in
// the plane, joined into one B-spline curve, so that the curve of the mesh
// lines and the boundary of the faces are the same curve.
//
// Usage: cut_cad input/ output/ z0 [j=i ...]
//
// z0 is in the units of the mesh (the files are read in millimetres and
// ComputationalDomain scales them by 1e-3). For each cut face the program
// prints the area by OpenCASCADE, and for each curve the largest distance
// of its points below the plane from the cut faces.

#include <deal.II/base/exceptions.h>

#include <deal.II/opencascade/utilities.h>

#include <BRepAdaptor_Curve.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepGProp.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <GProp_GProps.hxx>
#include <GeomConvert_CompCurveToBSplineCurve.hxx>
#include <Geom_BSplineCurve.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <ShapeAnalysis_FreeBounds.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_HSequenceOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Wire.hxx>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;

namespace
{
  // The files are in millimetres once read; the mesh in metres.
  constexpr double unit = 1e3;

  std::string
  color_file(const std::string &folder, const unsigned int i)
  {
    return folder + "/Color_" + std::to_string(i) + ".iges";
  }

  std::string
  curve_file(const std::string &folder, const unsigned int j)
  {
    return folder + "/Curve_" + std::to_string(j) + ".iges";
  }

  // The edges of a shape that lie in the plane z = z0.
  Handle(TopTools_HSequenceOfShape)
    edges_in_plane(const TopoDS_Shape &shape, const double z0)
  {
    Handle(TopTools_HSequenceOfShape) edges = new TopTools_HSequenceOfShape;
    for (TopExp_Explorer e(shape, TopAbs_EDGE); e.More(); e.Next())
      {
        const TopoDS_Edge edge = TopoDS::Edge(e.Current());
        BRepAdaptor_Curve curve(edge);
        double            deviation = 0;
        for (unsigned int k = 0; k <= 10; ++k)
          {
            const double t = curve.FirstParameter() +
                             k * (curve.LastParameter() -
                                  curve.FirstParameter()) /
                               10;
            deviation = std::max(deviation, std::abs(curve.Value(t).Z() - z0));
          }
        if (deviation < 1e-6 * unit)
          edges->Append(edge);
      }
    return edges;
  }

  // The edges joined, in the order of the wire they form, into one B-spline
  // curve: an IGES composite curve does not keep the orientation of its
  // reversed edges.
  TopoDS_Edge
  joined_curve(const Handle(TopTools_HSequenceOfShape) & edges)
  {
    Handle(TopTools_HSequenceOfShape) wires = new TopTools_HSequenceOfShape;
    Handle(TopTools_HSequenceOfShape) input = edges;
    ShapeAnalysis_FreeBounds::ConnectEdgesToWires(input,
                                                  1e-6 * unit,
                                                  Standard_False,
                                                  wires);
    AssertThrow(wires->Length() == 1,
                ExcMessage("The waterline of a face set is not one curve: " +
                           std::to_string(wires->Length()) + " pieces."));
    GeomConvert_CompCurveToBSplineCurve joined;
    for (BRepTools_WireExplorer e(TopoDS::Wire(wires->Value(1))); e.More();
         e.Next())
      {
        double                   first, last;
        const Handle(Geom_Curve) curve =
          BRep_Tool::Curve(e.Current(), first, last);
        Handle(Geom_TrimmedCurve) piece =
          new Geom_TrimmedCurve(curve, first, last);
        if (e.Current().Orientation() == TopAbs_REVERSED)
          piece->Reverse();
        AssertThrow(joined.Add(piece, 1e-6 * unit, Standard_True),
                    ExcMessage("Could not join the edges of a waterline."));
      }
    return BRepBuilderAPI_MakeEdge(joined.BSplineCurve()).Edge();
  }

  // The largest distance of the points of a curve below the plane from a
  // shape.
  double
  distance_below(const TopoDS_Shape &curve_shape,
                 const TopoDS_Shape &shape,
                 const double        z0)
  {
    double distance = 0;
    for (TopExp_Explorer e(curve_shape, TopAbs_EDGE); e.More(); e.Next())
      {
        BRepAdaptor_Curve curve(TopoDS::Edge(e.Current()));
        for (unsigned int k = 0; k <= 50; ++k)
          {
            const gp_Pnt p =
              curve.Value(curve.FirstParameter() +
                          k * (curve.LastParameter() - curve.FirstParameter()) /
                            50);
            if (p.Z() > z0 + 1e-9 * unit)
              continue;
            BRepExtrema_DistShapeShape d(BRepBuilderAPI_MakeVertex(p).Vertex(),
                                         shape);
            distance = std::max(distance, d.Value());
          }
      }
    return distance / unit;
  }
} // namespace


int
main(int argc, char **argv)
{
  AssertThrow(argc > 3,
              ExcMessage("Usage: cut_cad input/ output/ z0 [j=i ...]"));
  const std::string input = argv[1], output = argv[2];
  const double      z0 = std::atof(argv[3]) * unit;
  std::map<unsigned int, unsigned int> waterlines;
  for (int a = 4; a < argc; ++a)
    {
      const std::string arg = argv[a];
      const auto        eq  = arg.find('=');
      AssertThrow(eq != std::string::npos, ExcMessage("Expected j=i: " + arg));
      waterlines[std::stoi(arg.substr(0, eq))] = std::stoi(arg.substr(eq + 1));
    }
  std::filesystem::create_directories(output);
  std::cout << std::setprecision(12);

  // The faces below the plane.
  const TopoDS_Shape below =
    BRepPrimAPI_MakeBox(gp_Pnt(-1e4 * unit, -1e4 * unit, -1e4 * unit),
                        gp_Pnt(1e4 * unit, 1e4 * unit, z0))
      .Shape();
  std::vector<TopoDS_Shape> cut;
  TopoDS_Compound           all_faces;
  BRep_Builder              builder;
  builder.MakeCompound(all_faces);
  for (unsigned int i = 1; std::filesystem::exists(color_file(input, i)); ++i)
    {
      const TopoDS_Shape shape = OpenCASCADE::read_IGES(color_file(input, i), 1.);
      BRepAlgoAPI_Common common(shape, below);
      AssertThrow(common.IsDone() && !common.HasErrors(),
                  ExcMessage("The cut of Color_" + std::to_string(i) +
                             " failed."));
      cut.push_back(common.Shape());
      builder.Add(all_faces, cut.back());
      unsigned int n_faces = 0;
      for (TopExp_Explorer e(cut.back(), TopAbs_FACE); e.More(); e.Next())
        ++n_faces;
      GProp_GProps before, after;
      BRepGProp::SurfaceProperties(shape, before, 1e-12);
      BRepGProp::SurfaceProperties(cut.back(), after, 1e-12);
      // The volume int (z - z0) n_z of the faces, to which a lid in the
      // plane adds nothing, each face oriented out of the body (the faces
      // of a file need not be oriented alike): the volume of the hull below
      // its waterline.
      double volume = 0;
      for (TopExp_Explorer e(cut.back(), TopAbs_FACE); e.More(); e.Next())
        {
          GProp_GProps face_volume;
          BRepGProp::VolumePropertiesGK(e.Current(),
                                        face_volume,
                                        gp_Pln(gp_Pnt(0, 0, z0),
                                               gp_Dir(0, 0, 1)),
                                        1e-12,
                                        Standard_False,
                                        Standard_True);
          volume += std::abs(face_volume.Mass());
        }
      std::cout << "Color_" << i << ": " << n_faces << " faces below z0, area "
                << after.Mass() / (unit * unit) << " (whole "
                << before.Mass() / (unit * unit) << "), volume "
                << volume / (unit * unit * unit) << std::endl;
      OpenCASCADE::write_IGES(cut.back(), color_file(output, i));
    }

  for (unsigned int j = 1; std::filesystem::exists(curve_file(input, j)) ||
                           waterlines.count(j) > 0;
       ++j)
    {
      TopoDS_Shape curve;
      if (const auto w = waterlines.find(j); w != waterlines.end())
        {
          AssertThrow(w->second >= 1 && w->second <= cut.size(),
                      ExcMessage("No Color_" + std::to_string(w->second)));
          curve = joined_curve(edges_in_plane(cut[w->second - 1], z0));
          OpenCASCADE::write_IGES(curve, curve_file(output, j));
          std::cout << "Curve_" << j << ": waterline of Color_" << w->second;
        }
      else
        {
          curve = OpenCASCADE::read_IGES(curve_file(input, j), 1.);
          std::filesystem::copy_file(
            curve_file(input, j),
            curve_file(output, j),
            std::filesystem::copy_options::overwrite_existing);
          std::cout << "Curve_" << j << ": copied";
        }
      std::cout << ", largest distance below z0 from the faces "
                << distance_below(curve, all_faces, z0) << std::endl;
    }
}
