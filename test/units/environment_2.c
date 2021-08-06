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

  // Check if the environment advisor works
  putenv ("VFTR_OF=yes"); // Should be VFTR_OFF
  putenv ("VFTR_TRUNCATE=yes"); // Should be VFTR_PROF_TRUNCATE

  vftr_read_environment ();
  vftr_assert_environment ();
  vftr_free_environment ();

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
