// The whole solver on the cases of the regression suite (RUN_REGRESSION):
// the parameter file of a case is written from its base file and settings
// as run_regression.py does, the program runs in the folder of the test,
// with links to the meshes and CAD files of the suite, and the quantities it
// prints go to the output. The suite runs on 8 processes and compares more
// digits; these tests run its smaller cases on 2 processes.

#ifndef solver_tests_h
#define solver_tests_h

#include <deal.II/base/parameter_acceptor.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "driver.h"
#include "tests.h"

namespace solver_tests
{
  using Settings = std::vector<std::pair<std::string, std::string>>;

  const std::string suite = SOURCE_DIR "/../RUN_REGRESSION";

  // The line of the solver output that holds a quantity, its value as the
  // first group: the patterns of QUANTITIES in run_regression.py.
  const std::map<std::string, std::string> quantities = {
    {"rhs_norm", R"(Galerkin assembly check: \|b\| (\S+),)"},
    {"flux", R"(dPhidN true relative L_2 error: (\S+))"},
    {"gradient", R"(Phi Cells Gradient  error L_2 norm: (\S+))"},
    {"best_ratio", R"(Galerkin error / best-projection error: (\S+))"},
    {"iterations", R"((?:CG|GMRES) iterations: (\d+))"},
    {"mixed_flux", R"(flux on the Dirichlet part, relative L2 error (\S+);)"},
    {"mixed_potential",
     R"(potential on the Neumann part, relative L2 error (\S+))"},
    {"mixed_robin_flux", R"(flux on the Robin part, relative L2 error (\S+))"},
    {"mixed_flux_real",
     R"(real part: flux on the Dirichlet part, relative L2 error (\S+);)"},
    {"mixed_potential_real",
     R"(real part: flux on the Dirichlet part, relative L2 error \S+; )"
     R"(potential on the Neumann part, relative L2 error (\S+))"},
    {"phi_nodes", R"(Phi Nodes error L_inf norm: (\S+))"},
    {"steps", R"(Solve terminated at step (\d+))"}};

  // Set every occurrence of a parameter; add it to BEMProblem<3>, or to
  // BoundaryConditions<3> for Errors, if the file does not have it.
  inline std::string
  set_value(const std::string &text,
            const std::string &key,
            const std::string &value)
  {
    const std::string escaped_key =
      std::regex_replace(key, std::regex(R"([.^$|()\[\]{}*+?\\])"), R"(\$&)");
    const std::regex pattern(R"((\n\s*set )" + escaped_key + R"(\s*=)[^\n]*)");
    if (std::distance(std::sregex_iterator(text.begin(), text.end(), pattern),
                      std::sregex_iterator()) == 0)
      {
        const std::string section =
          key == "Errors" ? "BoundaryConditions" : "BEMProblem";
        return text + "\nsubsection " + section + "<3>\n  set " + key + " = " +
               value + "\nend\n";
      }
    const std::string escaped_value =
      std::regex_replace(value, std::regex(R"(\$)"), "$$$$");
    return std::regex_replace(text, pattern, "$1 " + escaped_value);
  }

  // The settings of a case that runs in Debug as well: there the sphere
  // skips its refinement cycle (96 cells instead of 384), since the Debug
  // build is thirty to fifty times slower.
  inline Settings
  unrefined_in_debug(Settings settings)
  {
    if (running_in_debug_mode())
      settings.emplace_back("Number of cycles", "0");
    return settings;
  }

  // Run the case and write the last value of each quantity to deallog. On
  // a failure the output of the solver goes to std::cerr.
  inline void
  run(const std::string              &base,
      const Settings                 &settings,
      const std::vector<std::string> &names)
  {
    namespace fs        = std::filesystem;
    const bool root     = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0;
    const auto add_link = [](const std::string &target, const fs::path &link) {
      if (!fs::exists(fs::symlink_status(link)))
        fs::create_symlink(target, link);
    };
    if (root)
      {
        mpi_initlog();
        for (const std::string name : {"sphere", "naca", "Color_1.iges"})
          add_link(suite + "/" + name, name);
        // The sphere in a box is read as ../grids/sphere_box_flipped_new,
        // a name the mesh code checks.
        add_link(SOURCE_DIR "/../grids", "../grids");

        std::ifstream      in(suite + "/base/" + base);
        std::ostringstream text;
        text << in.rdbuf();
        std::string prm = text.str();
        for (const auto &[key, value] : settings)
          prm = set_value(prm, key, value);
        std::ofstream("parameters_bem_3.prm") << prm;
      }
    MPI_Barrier(MPI_COMM_WORLD);

    std::ostringstream log;
    std::streambuf    *console = std::cout.rdbuf(log.rdbuf());
    // The solvers log their steps to deallog.
    const unsigned int depth = deallog.depth_file(0);
    try
      {
        Driver<3> driver;
        ParameterAcceptor::initialize("parameters_bem_3.prm",
                                      "used_parameters_bem_3.prm");
        driver.run();
      }
    catch (...)
      {
        std::cout.rdbuf(console);
        std::cerr << log.str() << std::endl;
        throw;
      }
    deallog.depth_file(depth);
    std::cout.rdbuf(console);

    if (!root)
      return;
    const std::string output = log.str();
    for (const std::string &name : names)
      {
        const std::regex pattern(quantities.at(name));
        std::string      value = "missing";
        for (auto match =
               std::sregex_iterator(output.begin(), output.end(), pattern);
             match != std::sregex_iterator();
             ++match)
          value = (*match)[1];
        deallog << name << ": " << value << std::endl;
      }
  }
} // namespace solver_tests

#endif
