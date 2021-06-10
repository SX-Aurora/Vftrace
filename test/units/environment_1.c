#include "vftr_environment.h"
#include "vftr_setup.h"
#ifdef _MPI
#include <mpi.h>
#endif


int main (int argc, char **argv) {

#if defined(_MPI)
  PMPI_Init(&argc, &argv);
  vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);
#else 
  vftr_mpirank = 0;
  vftr_mpisize = 1;
#endif

  // The basic usage of the environment module:
  putenv ("VFTR_OFF=yes");
  vftr_read_environment();
  vftr_assert_environment();
  vftr_print_environment(stdout);
  vftr_free_environment();

  // Check each of the different possible data types (except regular expression).
  fprintf (stdout, "***************************\n");
  putenv ("VFTR_OFF=no");
  putenv ("VFTR_SAMPLING=YES");
  putenv ("VFTR_REGIONS_PRECISE=0");
  putenv ("VFTR_MPI_LOG=on");
  putenv ("VFTR_OUT_DIRECTORY=\"foo/bar\"");
  putenv ("VFTR_BUFSIZE=1234");
  putenv ("VFTR_SAMPLETIME=12.34");
  
  vftr_read_environment ();
  vftr_assert_environment ();
  vftr_print_environment (stdout);
  vftr_free_environment ();

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
