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
  // The basic usage of the environment module:
  putenv("VFTR_OFF=yes");
  environment_t environment;
  environment = vftr_read_environment();
  // TODO: vftr_assert_environment();
  vftr_print_env(stdout, environment);
  vftr_environment_free(&environment);

  // Check each of the different possible data types(except regular expression).
  fprintf(stdout, "***************************\n");
  putenv("VFTR_OFF=no");
  putenv("VFTR_SAMPLING=YES");
  putenv("VFTR_REGIONS_PRECISE=0");
  putenv("VFTR_MPI_LOG=on");
  putenv("VFTR_OUT_DIRECTORY=\"foo/bar\"");
  putenv("VFTR_BUFSIZE=1234");
  putenv("VFTR_SAMPLETIME=12.34");

  environment = vftr_read_environment();
  // TODO: vftr_assert_environment();
  vftr_print_env(stdout, environment);
  vftr_environment_free(&environment);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
