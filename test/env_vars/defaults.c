#include <stdlib.h>

#include "environment_types.h"
#include "environment.h"

#ifdef _MPI
#include <mpi.h>
#endif


int main(int argc, char **argv) {
#if defined(_MPI)
  PMPI_Init(&argc, &argv);
#else
  (void) argc;
  (void) argv;
#endif
  environment_t environment;
  environment = vftr_read_environment();
  vftr_environment_assert(stderr, environment);
  vftr_print_environment(stdout, environment);
  vftr_environment_free(&environment);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
