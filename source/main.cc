

#include "driver.h"

#include <dlfcn.h>

#include <iostream>
#include <streambuf>

namespace
{
  /// Limit the threads of the BLAS that is loaded, whichever it is, found at
  /// run time. A threaded BLAS in every MPI process oversubscribes the
  /// cores: on eight processes a LAPACK inverse of an 81 x 81 block took
  /// 20 ms instead of 0.1.
  void
  limit_blas_threads(const int n_threads)
  {
    using Setter = void (*)(int);
    for (const char *name : {"openblas_set_num_threads", "MKL_Set_Num_Threads"})
      if (const auto set = reinterpret_cast<Setter>(dlsym(RTLD_DEFAULT, name)))
        set(n_threads);
  }

  /// Swallows everything written to it.
  class NullBuffer : public std::streambuf
  {
  public:
    int
    overflow(int c) override
    {
      return c;
    }
  };
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      unsigned int threads;
      if (argc == 1)
        threads = numbers::invalid_unsigned_int;
      else
        threads = atoi(argv[1]);
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, threads);
      // The threads of the BLAS as those of the program: the given number,
      // or one per process when there are several.
      if (threads != numbers::invalid_unsigned_int)
        limit_blas_threads(threads);
      else if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        limit_blas_threads(1);

      // Libraries we call write progress to std::cout from every rank, so
      // the same line appeared once per process. The application's own
      // output already goes through a rank-filtered stream, so silencing
      // std::cout elsewhere loses nothing. std::cerr is left alone: a
      // failure on any rank must still be visible. The buffer is static so
      // that it outlives the stream it is attached to.
      static NullBuffer null_buffer;
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
        std::cout.rdbuf(&null_buffer);

      std::string pname =
        "parameters_bem_" + std::to_string(DEAL_II_DIMENSION) + ".prm";
      std::string pname2 =
        "used_parameters_bem_" + std::to_string(DEAL_II_DIMENSION) + ".prm";

      Driver<DEAL_II_DIMENSION> driver;
      dealii::ParameterAcceptor::initialize(pname, pname2);

      driver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
